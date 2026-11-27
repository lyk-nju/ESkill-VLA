"""Construct an online policy from a checkpoint and its saved training transforms."""
import torch

from openpi.policies.libero_policy import LiberoOutputs
from openpi.policies.policy import Policy
from openpi.policies.robotwin_policy import RobotwinOutputs
from openpi.training.config import TrainConfig, create_model
from openpi.training.checkpoints import check_resources, checkpoint_config, load_weights
from openpi.training.data_loader import input_adapter
from openpi.transforms import ModelTransforms, Normalize, TokenizePrompt


def load_train_config(saved, *, device="cpu", tokenizer_path=None):
    settings = checkpoint_config(saved)
    # Complete historical snapshots need the legacy horizon; omitted fields follow
    # TrainConfig.from_dict defaults, just as a minimally specified config file does.
    if settings.get('model_type') == 'eskill' and 'model' in settings:
        settings['model'] = {'motion_min_horizon_frames': 0, **settings['model']}
    settings["device"] = device
    if tokenizer_path is not None:
        settings["tokenizer_path"] = tokenizer_path
    return TrainConfig.from_dict(settings)


def create_trained_policy(checkpoint, *, device="cpu", tokenizer_path=None, inference_precision=None):
    # Trusted local training output. Pretrained third-party initialization uses safetensors instead.
    # Map full checkpoints: inference does not need to read their large optimizer tensors.
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    config = load_train_config(saved, device=device, tokenizer_path=tokenizer_path)
    if config.training_stage == 'motion_planner':
        raise ValueError('Motion Planner checkpoints predict 2D traces. Use create_motion_planner_policy.')
    if config.training_stage == 'stage1':
        raise ValueError('Stage 1 learns visual priors; initialize Stage 2 before policy evaluation')
    if config.training_stage in ('grounding', 'visual_primitive'):
        raise ValueError('Grounding-only checkpoints use create_current_grounding_policy, not action rollout')
    adapter_only = saved.get("weights_kind") == "lora"
    if adapter_only:
        check_resources(saved, config, fields=("pretrained_path",), purpose="load LoRA policy")
    model = create_model(config, training=False, initialize=adapter_only)
    precision = config.precision if inference_precision is None else inference_precision
    if precision == "float32":
        model.float()
    weights = {key: value for key, value in saved["model"].items()
               if not key.startswith("visual_supervision.")}
    load_weights(model, weights, adapter_only=adapter_only)
    outputs = {"libero": LiberoOutputs, "robotwin": RobotwinOutputs}[config.platform]
    transforms = ModelTransforms(config.model, saved["norm_stats"],
                                 TokenizePrompt(config.tokenizer_path, config.model.max_token_len))
    return Policy(model, input_adapter(config), outputs(Normalize(saved["norm_stats"])), transforms,
                  device=device, precision=precision, cameras=config.cameras)
