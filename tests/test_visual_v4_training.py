"""Normal recipe, compact checkpoint, policy and optimizer resume contracts for Stage V."""
from dataclasses import replace
from pathlib import Path
import json

import numpy as np
import pytest
import sentencepiece
import torch
from safetensors.torch import save_file

from openpi.models.pi0 import Pi0
from openpi.models.pi0_config import ESkillConfig
from openpi.policies.current_grounding import create_current_grounding_policy
from openpi.training.config import TrainConfig, WandbConfig
from scripts.train import train
from test_current_grounding_training import saved_dataset


def test_visual_ablations_match():
    common = None
    for i in range(4):
        cfg = TrainConfig.load(f'examples/libero/train_visual_v4_e{i}.json').to_dict()
        assert cfg.pop('loss_weights') == dict(gac=float(i in (1, 3)), current_align=.1 if i in (2, 3) else 0.,
                                               subject=1.)
        assert cfg.pop('lora_rank') == (8 if i else 0)
        assert cfg['model'].pop('learned_alpha') == bool(i)
        cfg.pop('output_dir')
        cfg.pop('run_name')
        if common is None:
            common = cfg
        assert common == cfg


@pytest.mark.parametrize('experiment', [0, 3])
def test_compact_checkpoint_policy_and_resume(tmp_path, experiment):
    original = saved_dataset(tmp_path)
    corpus = tmp_path/'corpus'
    corpus.write_text('put the milk in the basket\nput the chocolate pudding in the basket\n')
    sentencepiece.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(tmp_path/'sp'),
                                            vocab_size=32, hard_vocab_limit=False, minloglevel=2)
    sizes = ESkillConfig.tiny(vocab_size=128, grounding_subject=True, current_grounding='visual_v4',
                              learned_alpha=bool(experiment))
    base = Pi0(sizes)
    base_path = tmp_path/'base.safetensors'
    save_file({k: v.clone() for k, v in base.state_dict().items()}, str(base_path))
    stats = tmp_path/'stats.json'
    stats.write_text('{}')
    cfg = TrainConfig(original.config.data_index, str(tmp_path/'sp.model'), str(tmp_path/'run'), str(stats),
                      model=sizes, pretrained_path=str(base_path), device='cpu', precision='float32',
                      training_stage='visual_v4', validation_fraction=0, num_steps=1, batch_size=2,
                      lora_rank=2 if experiment else 0, loss_weights=dict(gac=float(bool(experiment)),
                      current_align=.1 if experiment else 0., subject=1.), num_workers=0,
                      wandb_info=WandbConfig(mode='disabled'))
    checkpoint = train(cfg)
    first = torch.load(checkpoint, weights_only=False)
    assert first['weights_kind'] == 'visual_adapter'
    assert not set(first['model']).intersection(base.state_dict())
    rgb = np.full((32, 32, 3), 150, np.uint8)
    task = 'put the milk in the basket'
    policy = create_current_grounding_policy(checkpoint)
    output = policy.infer([rgb], task)
    assert output['current_map'].shape == output['subject_map'].shape == (1, 16)
    assert output['current_map'].sum().item() == pytest.approx(1.)
    resumed = train(replace(cfg, num_steps=2), resume=checkpoint)
    direct = train(replace(cfg, num_steps=2, output_dir=str(tmp_path/'direct')))
    a, b = [torch.load(p, weights_only=False) for p in (resumed, direct)]
    for key in a['model']:
        torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
    if experiment:
        norms = json.loads((Path(cfg.output_dir)/'metrics.jsonl').read_text().splitlines()[0])['unweighted_q_lora_grad_norm']
        assert norms['current_align'] > 0 and norms['gac'] > 0 and norms['subject'] == 0
