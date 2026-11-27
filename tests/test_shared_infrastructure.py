from dataclasses import replace

import pytest
import torch

from openpi.models.lora import LoRALinear
from openpi.models import research
from openpi.models.pi0_config import ESkillConfig, Pi0Config
from openpi.training.config import TrainConfig, create_model


@pytest.mark.parametrize('mode,layers', [('frozen', 'all'), ('full', [1]), ('full', 'all'), ('lora', [0])])
def test_language_update_policy_is_independent_of_other_modules(mode, layers):
    cfg = TrainConfig('', '', '', '', model_type='pi05', model=Pi0Config.tiny(vocab_size=80),
                      smoke=True, device='cpu', vlm_finetune=dict(mode=mode, layers=layers,
                      target_modules=['v_proj'], lora_rank=2, lora_alpha=3, freeze_image_projector=False))
    assert TrainConfig.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()
    assert cfg.effective_lora_rank == (2 if mode == 'lora' else 0)
    model = create_model(cfg)
    vlm = model.paligemma_with_expert.paligemma
    assert not any(p.requires_grad for p in vlm.vision_tower.parameters())
    assert all(p.requires_grad for p in vlm.multi_modal_projector.parameters())
    assert model.action_in_proj.weight.requires_grad
    for i, layer in enumerate(vlm.language_model.layers):
        enabled = layers == 'all' or i in layers
        if mode == 'lora' and enabled:
            assert isinstance(layer.self_attn.v_proj, LoRALinear)
            assert {n for n, p in layer.named_parameters() if p.requires_grad} == {
                'self_attn.v_proj.lora_A', 'self_attn.v_proj.lora_B'}
        else:
            assert all(p.requires_grad == (mode == 'full' and enabled) for p in layer.parameters())
    assert vlm.language_model.norm.weight.requires_grad == (mode == 'full' and layers == 'all')


def test_fm_sample_validity_is_independent_of_padding():
    cfg = TrainConfig('', '', '', '', model_type='pi05', model=Pi0Config.tiny(vocab_size=80, mask_action_loss=False),
                      smoke=True, device='cpu')
    model = create_model(cfg)
    values = torch.zeros(2, 8, 32)
    values[1] = 100
    model.velocity = lambda noisy, t, cond: torch.zeros_like(noisy)
    loss = model.flow_loss(values, values.bool(), None, noise=torch.ones_like(values), time=torch.ones(2),
                           sample_valid=torch.tensor([True, False]))
    assert loss.item() == 1.

def test_lora_gradient_diagnostic_follows_configured_projections():
    from openpi.training.metrics import visual_lora_gradient_norms

    cfg = TrainConfig('', '', '', '', model_type='pi05', model=Pi0Config.tiny(vocab_size=80),
                      smoke=True, device='cpu', vlm_finetune=dict(mode='lora', layers=[1], target_modules=['v_proj'],
                                                                lora_rank=2, lora_alpha=3))
    model = create_model(cfg)
    layer = model.paligemma_with_expert.paligemma.language_model.layers[1].self_attn.v_proj
    norms = visual_lora_gradient_norms(model, {'readout': layer.lora_A.sum()})
    assert norms['readout'] == pytest.approx(layer.lora_A.numel() ** .5)


@pytest.mark.skipif('motion_planner' not in research.GROUNDING, reason='Requires the planning stage')
@pytest.mark.parametrize('mode,head,kl', [('full', False, .1), ('full', False, 0.), ('frozen', True, .1),
                                         ('frozen', True, 0.), ('frozen', False, 0.)])
def test_planner_current_head_backbone_and_kl_are_independent(mode, head, kl):
    cfg = TrainConfig('', '', '', '', model=ESkillConfig.tiny(current_grounding='motion_planner',
                      planner_version='v2_joint', train_current=head, vocab_size=80), smoke=True, device='cpu',
                      vlm_finetune=dict(mode=mode, layers=[1]), loss_weights=dict(current_align=kl, skill=.02, trace=.25))
    model = create_model(cfg)
    assert all(p.requires_grad == head for p in model.grounding.parameters())
    language = model.paligemma_with_expert.paligemma.language_model
    assert all(p.requires_grad == (mode == 'full') for p in language.layers[1].parameters())
    assert not any(p.requires_grad for p in language.layers[0].parameters())
    if kl:
        inputs = dict(images=torch.randn(2, 1, 3, 32, 32), image_mask=torch.ones(2, 1).bool(),
                      tokens=torch.tensor([[2, 5, 7], [6, 4, 3]]), token_mask=torch.ones(2, 3).bool())
        # KL against a single-patch target; verify the actual gradient destination.
        loss = -model.predict_current(inputs)[:, 0].log().mean()
        loss.backward()
        for module, active in ((model.grounding, head), (language, mode == 'full')):
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()) == active
    if not head and mode == 'frozen':
        with pytest.raises(ValueError, match='current_align'):
            replace(cfg, loss_weights=dict(current_align=.1, skill=.02, trace=.25))


def test_wrist_pairing_rate_excludes_missing_source_frames():
    from openpi.training.metrics import wrist_sample_statistics

    observations = [('ordinary', 'paired'), ('text_swap', 'random_dropout'), ('ordinary', 'missing'),
                    ('copy_and_text_swap', 'copy_unverified')]
    record = wrist_sample_statistics(dict(wrist_group=g, wrist_status=s) for g, s in observations)
    assert record['wrist_pairing'] == {'with_copy': dict(paired=0, available=1, rate=0.),
                                       'without_copy': dict(paired=1, available=2, rate=.5)}
    assert wrist_sample_statistics([])['wrist_pairing']['with_copy']['rate'] is None
