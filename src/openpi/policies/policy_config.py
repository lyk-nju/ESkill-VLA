"""Construct an online policy from a checkpoint and its saved training transforms."""
import torch

from openpi.models.pi0_config import ESkillConfig
from openpi.policies.libero_policy import LiberoInputs, LiberoOutputs, LiberoSFTInputs
from openpi.policies.policy import Policy
from openpi.policies.robotwin_policy import RobotwinInputs, RobotwinOutputs
from openpi.training.config import TrainConfig, create_model
from openpi.training.checkpoints import load_weights
from openpi.transforms import ModelTransforms, Normalize, TokenizePrompt


def create_trained_policy(checkpoint, *, device="cpu", tokenizer_path=None):
    # Trusted local training output. Pretrained third-party initialization uses safetensors instead.
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    settings = dict(saved["config"])
    settings["model"] = ESkillConfig(**settings["model"])
    settings["device"] = device
    if tokenizer_path is not None:
        settings["tokenizer_path"] = tokenizer_path
    config = TrainConfig(**settings)
    adapter_only = saved.get("weights_kind") == "lora"
    model = create_model(config, training=False, initialize=adapter_only)
    weights = {key: value for key, value in saved["model"].items()
               if not key.startswith("visual_supervision.")}
    load_weights(model, weights, adapter_only=adapter_only)
    normalize = Normalize(saved["norm_stats"])
    inputs, outputs = {"libero": (LiberoInputs, LiberoOutputs),
                       "robotwin": (RobotwinInputs, RobotwinOutputs)}[config.platform]
    if config.data_format == "libero_sft":
        inputs = LiberoSFTInputs
    input_adapter = (LiberoInputs(use_arm_state=config.model.use_arm_state)
                     if inputs is LiberoInputs else inputs())
    transforms = ModelTransforms(config.model, saved["norm_stats"],
                                 TokenizePrompt(config.tokenizer_path, config.model.max_token_len))
    return Policy(model, input_adapter, outputs(normalize), transforms, device=device)
