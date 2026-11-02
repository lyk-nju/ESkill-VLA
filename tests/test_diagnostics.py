"""Masked skill frequencies, pixel Motion error and global diagnostic counts."""
from datetime import timedelta

import pytest
torch = pytest.importorskip('torch')
import torch.distributed as dist
import torch.multiprocessing as mp

from openpi.models.diagnostics import prediction_statistics
from openpi.models.pi0_config import ESkillConfig
from openpi.training.metrics import reduce_diagnostics


def example():
    cfg = ESkillConfig.tiny(image_size=100, motion_queries=2, skill_names=('Pick', 'Place'))
    inputs = dict(arm_presence=torch.tensor([[True, False]]))
    prediction = dict(skill_logits=torch.tensor([[[3., 0.], [0., 3.]]]),
                      future_skill_logits=torch.tensor([[[[3., 0.], [0., 3.]], [[0., 3.], [3., 0.]]]]))
    targets = dict(skill_labels=torch.tensor([[0, 0]]), skill_valid=torch.ones(1, 2, dtype=torch.bool),
                   future_skill_labels=torch.tensor([[[0, 0], [1, 0]]]),
                   future_skill_valid=torch.ones(1, 2, 2, dtype=torch.bool),
                   trace=torch.zeros(1, 2, 2, 2), trace_valid=torch.ones(1, 2, 2, dtype=torch.bool),
                   future_observed=torch.tensor([[[True, False], [True, True]]]),
                   future_padded=torch.tensor([[[False, True], [False, False]]]))
    targets['trace'][:, 1] = float('nan')
    trace = torch.tensor([[[[.3, .4], [0., .2]], [[999., 999.], [999., 999.]]]], requires_grad=True)
    return cfg, inputs, prediction, targets, trace


def test_diagnostics_mask_absent_arms_and_split_real_and_padded_points():
    cfg, inputs, prediction, targets, trace = example()
    pairs = prediction_statistics(prediction, trace, inputs, targets, cfg)
    assert all(not value.requires_grad for value in pairs.values())
    result = reduce_diagnostics(pairs)
    assert result['current_skill_count'] == 1
    assert result['current_skill_accuracy'] == 1
    assert result['current_skill_pred_pick'] == 1 and result['current_skill_pred_place'] == 0
    assert result['future_skill_count'] == 2 and result['future_skill_accuracy'] == .5
    assert result['future_skill_pred_pick'] == result['future_skill_pred_place'] == .5
    assert result['motion_l2_uv'] == pytest.approx(.35)
    assert result['motion_l2_px'] == pytest.approx(35.)
    assert result['motion_observed_l2_px'] == pytest.approx(50.)
    assert result['motion_padded_l2_px'] == pytest.approx(20.)
    assert result['motion_offset_4_l2_px'] == pytest.approx(50.)
    assert result['motion_offset_8_l2_px'] == pytest.approx(20.)
    assert result['motion_count'] == 2
    assert result['motion_observed_count'] == result['motion_padded_count'] == 1
    assert result['motion_unclassified_count'] == 0


def test_empty_labels_and_unknown_point_source_are_reported_without_fabrication():
    cfg, inputs, prediction, targets, trace = example()
    targets['skill_valid'].zero_()
    targets['future_skill_valid'].zero_()
    del targets['future_observed'], targets['future_padded']
    result = reduce_diagnostics(prediction_statistics(prediction, trace, inputs, targets, cfg))
    assert result['current_skill_count'] == result['future_skill_count'] == 0
    assert result['current_skill_accuracy'] == result['future_skill_accuracy'] == 0
    assert result['motion_observed_count'] == result['motion_padded_count'] == 0
    assert result['motion_unclassified_count'] == 2


def diagnostic_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        for empty in (False, True):
            count = 0 if empty else (1 if rank == 0 else 9)
            stats = dict(current_skill_accuracy=torch.tensor([1. if rank == 0 and not empty else 0., count]),
                         motion_l2_px=torch.tensor([9. if not empty else 0., count]),
                         motion_observed_l2_px=torch.tensor([9. if rank == 1 and not empty else 0.,
                                                            count if rank == 1 else 0]))
            result = reduce_diagnostics(stats)
            assert result['current_skill_count'] == (0 if empty else 10)
            assert result['current_skill_accuracy'] == pytest.approx(0 if empty else .1)
            assert result['motion_l2_px'] == pytest.approx(0 if empty else 1.8)
            assert result['motion_observed_l2_px'] == pytest.approx(0 if empty else 1.)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason='Requires Gloo')
def test_distributed_diagnostics_use_global_counts_even_with_empty_ranks(tmp_path):
    mp.spawn(diagnostic_worker, args=((tmp_path/'rendezvous').as_uri(),), nprocs=2, join=True)
