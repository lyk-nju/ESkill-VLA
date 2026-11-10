"""Masked semantic routes, fixed-time path/gripper errors and global counts."""
from datetime import timedelta
from types import SimpleNamespace

import pytest
torch = pytest.importorskip('torch')
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F

from openpi.models.pi0_config import ESkillConfig
from openpi.models.pi0_eskill import Pi0ESkill
from openpi.training.metrics import reduce_diagnostics


def example():
    cfg = ESkillConfig.tiny(image_size=100, motion_queries=2, motion_stride=5)
    inputs = dict(arm_presence=torch.tensor([[True, False]]))
    prediction = dict(skill_ids=torch.tensor([[0, 1]]),
                      gripper_logits=torch.tensor([[[2., 2.], [999., 999.]]]))
    targets = dict(skill_labels=torch.tensor([[0, 0]]), skill_valid=torch.ones(1, 2, dtype=torch.bool),
                   trace=torch.zeros(1, 2, 2, 2), trace_valid=torch.ones(1, 2, 2, dtype=torch.bool),
                   gripper=torch.tensor([[[1., 0.], [1., 0.]]]),
                   gripper_valid=torch.ones(1, 2, 2, dtype=torch.bool),
                   terminal_padded_mask=torch.zeros(1, 2, 2, dtype=torch.bool))
    targets['trace'][:, 1] = float('nan')
    targets['gripper'][:, 1] = float('nan')
    trace = torch.tensor([[[[.3, .4], [0., .2]], [[999., 999.], [999., 999.]]]], requires_grad=True)
    return cfg, inputs, prediction, targets, trace


def diagnostic_pairs(cfg, inputs, prediction, targets, trace):
    """Use the same effective masks and detached summary path as the model forward."""
    presence = inputs['arm_presence']
    skill_valid = targets['skill_valid'] & presence
    valid = targets['trace_valid'] & presence[:, :, None] & ~targets['terminal_padded_mask']
    grip_valid = targets['gripper_valid'] & presence[:, :, None] & ~targets['terminal_padded_mask']
    gripper = prediction['gripper_logits']
    error = F.binary_cross_entropy_with_logits(gripper, torch.where(grip_valid, targets['gripper'], 0),
                                             reduction='none')
    return Pi0ESkill.statistics(SimpleNamespace(config=cfg), prediction, trace, gripper,
                               inputs, targets, skill_valid, valid, grip_valid, error)


def test_diagnostics_mask_absent_arms_and_report_shared_query_errors():
    values = example()
    pairs = diagnostic_pairs(*values)
    assert all(not value.requires_grad for value in pairs.values())
    result = reduce_diagnostics(pairs)
    assert result['current_skill_count'] == 1 and result['current_skill_accuracy'] == 1
    assert result['current_skill_pred_pick'] == 1 and result['current_skill_pred_place'] == 0
    assert result['motion_l2_px'] == pytest.approx(35.)
    assert result['motion_offset_5f_l2_px'] == pytest.approx(50.)
    assert result['motion_offset_10f_l2_px'] == pytest.approx(20.)
    assert result['motion_count'] == 2
    assert result['gripper_accuracy'] == .5 and result['gripper_accuracy_count'] == 2
    assert result['gripper_transition_accuracy'] == 0
    assert result['gripper_transition_accuracy_count'] == 1
    assert result['skill_pick_accuracy'] == 1 and result['skill_pick_accuracy_count'] == 1
    assert result['skill_pick_motion_l2_px'] == pytest.approx(35.)
    assert result['skill_pick_motion_l2_px_count'] == 2
    assert result['skill_pick_gripper_accuracy'] == .5
    assert result['skill_place_accuracy_count'] == 0
    assert not any('future_skill' in key for key in result)


def test_terminal_padding_and_missing_labels_do_not_fabricate_summaries():
    cfg, inputs, prediction, targets, trace = example()
    targets['terminal_padded_mask'][0, 0, 1] = True
    result = reduce_diagnostics(diagnostic_pairs(cfg, inputs, prediction, targets, trace))
    assert result['motion_l2_px'] == pytest.approx(50.) and result['motion_count'] == 1
    assert result['gripper_accuracy'] == 1 and result['gripper_accuracy_count'] == 1
    assert result['gripper_transition_accuracy_count'] == 0
    for key in ('skill_valid', 'trace_valid', 'gripper_valid'):
        targets[key].zero_()
    empty = reduce_diagnostics(diagnostic_pairs(cfg, inputs, prediction, targets, trace))
    assert empty['current_skill_count'] == empty['motion_count'] == empty['gripper_accuracy_count'] == 0
    assert empty['current_skill_accuracy'] == empty['motion_l2_px'] == empty['gripper_accuracy'] == 0
    # Predicted routes still exist when CE labels are missing; route frequency uses presence.
    assert empty['current_skill_pred_pick'] == 1


def diagnostic_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        for empty in (False, True):
            count = 0 if empty else (1 if rank == 0 else 9)
            correct = 1. if rank == 0 and not empty else 0.
            stats = dict(current_skill_accuracy=torch.tensor([correct, count]),
                         motion_l2_px=torch.tensor([9. if not empty else 0., count]),
                         gripper_accuracy=torch.tensor([correct, count]),
                         gripper_transition_accuracy=torch.tensor([0., 0.]))
            result = reduce_diagnostics(stats)
            assert result['current_skill_count'] == (0 if empty else 10)
            assert result['current_skill_accuracy'] == pytest.approx(0 if empty else .1)
            assert result['motion_l2_px'] == pytest.approx(0 if empty else 1.8)
            assert result['gripper_accuracy'] == pytest.approx(0 if empty else .1)
            assert result['gripper_accuracy_count'] == (0 if empty else 10)
            assert result['gripper_transition_accuracy_count'] == 0
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason='Requires Gloo')
def test_distributed_diagnostics_use_global_counts_even_with_empty_ranks(tmp_path):
    mp.spawn(diagnostic_worker, args=((tmp_path/'rendezvous').as_uri(),), nprocs=2, join=True)
