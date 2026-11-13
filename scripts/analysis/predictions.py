"""Read current ESkill predictions from the real action forward; GT is loaded afterwards."""
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from openpi.models.grounding import VisualSupervision, load_vae
from openpi.policies.eskill_dataset import EpisodeDataset
from openpi.policies.policy_config import create_trained_policy
from openpi.training.checkpoints import check_resources
from openpi.transforms import role_canvases
from scripts.analysis.attention_viewer import AttentionCapture, FrameView, default_layers, load_config
from scripts.analysis.language_attention import LanguageCapture


class VisualReadout:
    """Load only a checkpoint's saved head and the matching frozen VAE, never a random head."""
    def __init__(self, checkpoint, config, device, vae_path=None):
        self.module, self.status = None, 'VISUAL HEAD NOT SAVED'
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
        prefix = 'visual_supervision.head.'
        weights = {key[len(prefix):]: value for key, value in saved['model'].items() if key.startswith(prefix)}
        if not weights:
            return
        path = vae_path or config.vae_path
        if not path or not Path(path).exists():
            self.status = 'VAE UNAVAILABLE; supply --vae-path'
            return
        check_resources(saved, replace(config, vae_path=str(path)), fields=('vae_path',),
                        purpose='decode saved Grounding predictions')
        self.module = VisualSupervision(config.model, load_vae(path)).to(device).eval()
        self.module.head.load_state_dict(weights, strict=True)
        self.status = 'SAVED HEAD / MATCHED FROZEN VAE'

    @torch.no_grad()
    def decode(self, prediction, targets, presence, precision):
        if self.module is None:
            return None, None
        visual = self.module
        device = prediction['grounding'].device
        present = torch.as_tensor(presence, device=device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == 'bfloat16'):
            predicted = visual.predict_latents(prediction['grounding'][0, present]).flatten(0, 1)
        images = torch.as_tensor(targets['role_images'][presence], device=device).flatten(0, 1)
        reference = visual.encode_targets(images)
        with torch.autocast(device.type, enabled=False):
            z = torch.cat((predicted.float(), reference), 0)
            raw = z / visual.vae.config.scaling_factor + (getattr(visual.vae.config, 'shift_factor', None) or 0.)
            decoded = visual.vae.decode(raw).sample
        rgb = ((decoded.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        prediction_rgb = np.zeros((2, 2, *rgb.shape[1:]), np.uint8)
        reference_rgb = np.zeros_like(prediction_rgb)
        n = int(presence.sum()) * 2
        prediction_rgb[presence] = rgb[:n].reshape(-1, 2, *rgb.shape[1:])
        reference_rgb[presence] = rgb[n:].reshape(-1, 2, *rgb.shape[1:])
        return prediction_rgb, reference_rgb


@dataclass
class PredictionFrame:
    frame: int
    action_row: int
    attention: FrameView
    prediction: dict
    targets: dict
    presence: np.ndarray
    metadata: dict


def collect(checkpoint, episode, *, device='cuda', layers=None, stride=1, seed=0, steps=10,
            tokenizer_path=None, data_index=None, vae_path=None, inference_precision=None, max_frames=None,
            check_action_parity=False, object_word=None, target_word=None):
    config = load_config(checkpoint)
    if config.model_type != 'eskill' or config.platform != 'libero':
        raise ValueError('Prediction panel requires a current LIBERO ESkill checkpoint')
    m = config.model
    index = data_index or config.data_index
    options = dict(action_horizon=m.action_horizon, cameras=config.cameras, episode_ids={episode},
                   include_state=False)
    observations = EpisodeDataset(index, auxiliary=False, **options)
    if not len(observations):
        raise ValueError(f'Episode {episode} is not in {index}')
    labels = EpisodeDataset(index, auxiliary=True, skill_names=m.skill_names, motion_queries=m.motion_queries,
                            motion_stride=m.motion_stride, motion_min_horizon_frames=m.motion_min_horizon_frames,
                            image_size=m.image_size, active_skill_ids=m.active_skill_ids, **options)
    policy = create_trained_policy(checkpoint, device=device, tokenizer_path=tokenizer_path,
                                   inference_precision=inference_precision)
    visual = VisualReadout(checkpoint, config, device, vae_path)
    model = policy.model
    layers = default_layers(m.depth) if layers is None else layers
    if not layers or not all(0 <= layer < m.depth for layer in layers):
        raise ValueError('Attention layers are outside the action expert')
    task = observations.metadata[0]['instructions'][0]
    language = LanguageCapture(policy.transforms.tokenizer.tokenizer, task,
                               object_word=object_word, target_word=target_word)
    capture = AttentionCapture(model, layers, language=language)
    frames = {}
    parity = None
    if check_action_parity:
        first = next(observations[i] for i in range(len(observations))
                     if observations[i]['metadata']['frame'] % stride == 0)
        generator = torch.Generator().manual_seed(seed + first['metadata']['frame'])
        noise = torch.randn(m.action_horizon, m.action_dim, generator=generator).numpy()
        expected = policy.infer(first['inputs'], noise=noise, steps=steps)['actions']
    with capture.active():
        for i in range(len(observations)):
            current = observations[i]
            frame = int(current['metadata']['frame'])
            if frame % stride or frame in frames:
                continue
            generator = torch.Generator().manual_seed(seed + frame)
            noise = torch.randn(m.action_horizon, m.action_dim, generator=generator).numpy()
            result = policy.infer(current['inputs'], noise=noise, steps=steps)
            if check_action_parity and not frames:
                parity = float(np.max(np.abs(result['actions'] - expected)))
                if not np.array_equal(result['actions'], expected):
                    raise RuntimeError(f'Attention capture changed the action output: max error {parity}')
            raw = capture.prediction
            with torch.no_grad(), torch.autocast(torch.device(device).type, dtype=torch.bfloat16,
                                                enabled=policy.precision == 'bfloat16'):
                trace, gripper = model.read_motion(raw)
            presence = capture.inputs['arm_presence'][0].cpu().numpy()
            prediction = dict(skill_prob=raw['skill_logits'][0].float().softmax(-1).cpu().numpy(),
                              trace=trace[0].cpu().numpy(), gripper_prob=gripper[0].sigmoid().cpu().numpy(),
                              actions=result['actions'])
            # Future images, Event boundaries and labels are queried only after action prediction.
            truth = labels[i]
            targets = dict(truth['targets'])
            targets.update(role_canvases(targets.pop('role_sources'), m.canvas_size,
                                        future_bbox_expand_px=m.future_bbox_expand_px))
            prediction['role_rgb'], prediction['role_roundtrip'] = visual.decode(raw, targets, presence, policy.precision)
            targets['role_rgb'] = ((targets.pop('role_images').transpose(0, 1, 3, 4, 2) + 1) * 127.5).round().astype(np.uint8)
            targets.pop('role_weights')
            frames[frame] = PredictionFrame(frame, current['metadata']['action_row'], capture.frame_view(),
                                            prediction, targets, presence, truth['metadata'])
            print(f'prediction frame {frame}: route={result["current_skills"].tolist()}', flush=True)
            if max_frames is not None and len(frames) >= max_frames:
                break
    metadata = dict(checkpoint=str(checkpoint), episode=episode, data_index=str(index), precision=policy.precision,
                    layers=layers, fm_steps=steps, seed=seed, visual_readout=visual.status,
                    task=task, technical=config.smoke,
                    tokenizer_path=str(tokenizer_path or config.tokenizer_path), vae_path=str(vae_path or config.vae_path),
                    language_attention=dict(layers=layers, layout=language.layout,
                        method='VLM instruction-word queries to image keys; FP32 Q/K reconstruction; mean selected layers/heads/phrase tokens'),
                    action_parity_max_abs_error=parity,
                    step=torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True).get('step'))
    return frames, config, metadata
