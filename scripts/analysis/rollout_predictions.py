"""Record actual policy calls and align their readouts to the LIBERO control timeline."""
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from openpi.models.pi0_config import ESkillConfig
from openpi.transforms import resize_image
from scripts.analysis.attention_viewer import AttentionCapture, FrameView, default_layers
from scripts.analysis.language_attention import LanguageCapture
from scripts.analysis.predictions import PredictionFrame, VisualReadout


class RecordedPolicy:
    """Observe the action forward once per real request; no additional observation probes."""
    def __init__(self, policy, checkpoint, config, directory, *, vae_path=None, layers=None,
                 object_word=None, target_word=None, check_action_parity=False):
        if config.model_type != 'eskill' or config.platform != 'libero':
            raise ValueError('Prediction recording requires a current LIBERO ESkill policy')
        self.policy, self.config = policy, config
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.layers = default_layers(config.model.depth) if layers is None else layers
        if not self.layers or not all(0 <= i < config.model.depth for i in self.layers):
            raise ValueError('Attention layers are outside the model')
        self.words = dict(object_word=object_word, target_word=target_word)
        self.check_action_parity, self.index = check_action_parity, 0
        device = torch.device(policy.device)
        # Loading a trained readout must not advance the policy's sampling RNG.
        devices = [device.index or 0] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            self.visual = VisualReadout(checkpoint, config, policy.device, vae_path)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
        self.metadata = dict(checkpoint=str(checkpoint), step=saved.get('step'), model=asdict(config.model),
            cameras=list(config.cameras), precision=policy.precision, technical=config.smoke,
            visual_readout=self.visual.status, layers=self.layers, kind='actual_rollout_inference',
            gt_usage='Unavailable in live rollout; never supplied to the policy')
        (self.directory / 'recording.json').write_text(json.dumps(self.metadata, indent=2) + '\n')

    @torch.no_grad()
    def infer(self, observation, *, noise=None, steps=10):
        policy, m = self.policy, self.config.model
        # Same device, shape and generator as the native sampler. Share this one noise
        # draw with the optional first-call parity check, without a second RNG advance.
        if noise is None:
            noise = torch.randn((1, m.action_horizon, m.action_dim), device=policy.device)
        parity = None
        if self.check_action_parity and self.index == 0:
            expected = policy.infer(observation, noise=noise, steps=steps)['actions']
        language = LanguageCapture(policy.transforms.tokenizer.tokenizer, observation['prompt'], **self.words)
        capture = AttentionCapture(policy.model, self.layers, language=language)
        with capture.active():
            result = policy.infer(observation, noise=noise, steps=steps)
        if self.check_action_parity and self.index == 0:
            parity = float(np.max(np.abs(expected - result['actions'])))
            np.testing.assert_array_equal(expected, result['actions'])
        raw = capture.prediction
        with torch.autocast(torch.device(policy.device).type, dtype=torch.bfloat16,
                            enabled=policy.precision == 'bfloat16'):
            trace, gripper = policy.model.read_motion(raw)
        view = capture.frame_view()
        presence = capture.inputs['arm_presence'][0].cpu().numpy()
        role_rgb, _ = self.visual.decode(raw, None, presence, policy.precision)
        arrays = dict(input_images=np.array(view.images), image_attention=np.array(view.attention),
            view_mass=np.array(view.view_mass), key_mass=np.array(list(view.key_mass.values())),
            motion_attention=view.motion_attention, arm_presence=presence,
            pred_skill_prob=raw['skill_logits'][0].float().softmax(-1).cpu().numpy(),
            pred_trace=trace[0].cpu().numpy(), pred_gripper_prob=gripper[0].sigmoid().cpu().numpy(),
            pred_actions=result['actions'])
        if role_rgb is not None:
            arrays['pred_role_rgb'] = role_rgb
        for name in ('maps', 'view_mass', 'valid', 'probabilities'):
            arrays['language_' + name] = view.language_attention[name]
        identity = f'{self.index:06d}'
        np.savez_compressed(self.directory / (identity + '.npz'), **arrays)
        info = dict(request_id=identity, task=observation['prompt'], fm_steps=steps,
                    action_parity_max_abs_error=parity, language_layout=language.layout)
        (self.directory / (identity + '.json')).write_text(json.dumps(info, indent=2) + '\n')
        self.index += 1
        print(f'prediction request {identity}: saved actual inference; parity={parity}', flush=True)
        return dict(result, diagnostic_id=identity)


def load_rollout(record, directory=None):
    """Join by recorded request IDs and control steps, never by an assumed frame modulus."""
    path = Path(record)
    episode = json.loads(path.read_text())
    directory = Path(directory or episode['prediction_recording']['directory'])
    metadata = json.loads((directory / 'recording.json').read_text())
    config = SimpleNamespace(model=ESkillConfig(**metadata['model']), cameras=tuple(metadata['cameras']))
    metadata.update(episode=path.stem, task=episode['instruction'], control_fps=episode['control_fps'],
                    replan_steps=episode['replan_steps'], rollout_record=str(path), error=episode['error'],
                    outcome=episode['success'], purpose='rollout_visualization')
    frames = {}
    with np.load(path.with_suffix('.npz')) as execution:
        images = execution['diagnostic_images'].copy()
        steps = len(execution['actions'])
        if len(images) != steps + 1 or images.ndim != 4:
            raise ValueError('Expected every pre-control RGB plus the final observation')
        requests = episode['inference_requests']
        sources = [r['step'] for r in requests]
        if not sources or sources[0] != 0 or any(b <= a for a, b in zip(sources, sources[1:])):
            raise ValueError('Rollout must have increasing actual inference steps starting at zero')
        for i, request in enumerate(requests):
            source, identity = request['step'], request['diagnostic_id']
            info = json.loads((directory / (identity + '.json')).read_text())
            if info['task'] != metadata['task']:
                raise ValueError('Recorded instruction differs from the rollout')
            with np.load(directory / (identity + '.npz')) as data:
                prediction = {name: data['pred_' + name].copy() for name in
                              ('skill_prob', 'trace', 'gripper_prob', 'actions')}
                prediction.update(role_rgb=data['pred_role_rgb'].copy() if 'pred_role_rgb' in data else None,
                                  role_roundtrip=None)
                np.testing.assert_array_equal(prediction['actions'], execution[f'prediction_{i:04d}'])
                np.testing.assert_array_equal(data['input_images'][0], resize_image(images[source], config.model.image_size))
                language = {name: data['language_' + name].copy() for name in ('maps', 'view_mass', 'valid', 'probabilities')}
                language['layout'] = info['language_layout']
                view = FrameView(list(data['input_images'].copy()), list(data['image_attention'].copy()),
                    dict(zip(('image', 'text', 'motion', 'action'), data['key_mass'].tolist())),
                    data['view_mass'].tolist(), data['motion_attention'].copy(), language)
                # No future GT exists in a new rollout. Invalid placeholders are display-only.
                targets = dict(trace=np.zeros((2, config.model.motion_queries, 2)),
                    trace_valid=np.zeros((2, config.model.motion_queries), bool),
                    skill_labels=np.full(2, -1), skill_valid=np.zeros(2, bool), role_valid=np.zeros((2, 2), bool))
                frames[source] = PredictionFrame(source, source, view, prediction, targets,
                                                 data['arm_presence'].copy(), info)
        # Verify that the recorded actions actually executed in each chunk.
        for step, action in enumerate(execution['actions']):
            i = int(np.searchsorted(sources, step, side='right') - 1)
            if int(execution['chunk_ids'][step]) != i:
                raise ValueError('Actual execution chunk does not match its inference request')
            np.testing.assert_array_equal(action, frames[sources[i]].prediction['actions'][step - sources[i]])
    return frames, config, metadata, images
