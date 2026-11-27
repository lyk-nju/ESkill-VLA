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


class MotionProbe(torch.nn.Module):
    """Small analytic gradients, with a large auxiliary-only head deliberately excluded."""
    def __init__(self):
        super().__init__()
        self.motion = torch.nn.Module()
        self.motion.core = torch.nn.Parameter(torch.tensor([.1, .2]))
        self.motion.trace_head = torch.nn.Linear(2, 1)
        self.motion.gripper_head = torch.nn.Linear(2, 1)

    def forward(self, fm_coefficient, aux_coefficient):
        p = self.motion.core
        heads = sum(v.sum() for m in (self.motion.trace_head, self.motion.gripper_head) for v in m.parameters())
        return dict(action=(p * fm_coefficient).sum(), trace=(p * aux_coefficient).sum() + 1000 * heads,
                    skill=heads * 0, current=heads * 0, future_object=heads * 0, gripper=heads * 0)


@pytest.mark.parametrize('fm,aux,norm_fm,norm_aux,cosine', [
    ([3., 4.], [0., 5.], 5., 5., .8),
    ([1., 0.], [-2., 0.], 1., 2., -1.),
    ([1., 0.], [0., 1.], 1., 1., 0.),
    ([0., 0.], [0., 0.], 0., 0., None),
    ([1., 0.], [0., 0.], 1., 0., None),
    ([0., 0.], [1., 0.], 0., 1., None),
])
def test_motion_gradient_metrics_do_not_modify_backward_or_rng(fm, aux, norm_fm, norm_aux, cosine):
    from openpi.training.metrics import motion_gradient_balance
    model = MotionProbe()
    losses = model(torch.tensor(fm), torch.tensor(aux))
    rng = torch.get_rng_state().clone()
    result = motion_gradient_balance(model, losses)
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(p.grad is None for p in model.parameters())
    assert result['fm_norm'] == pytest.approx(norm_fm)
    assert result['aux_norm'] == pytest.approx(norm_aux)
    assert result['cosine'] == pytest.approx(cosine) if cosine is not None else result['cosine'] is None
    ratio = norm_fm / norm_aux if norm_aux else None
    assert result['fm_aux_ratio'] == pytest.approx(ratio) if ratio is not None else result['fm_aux_ratio'] is None
    sum(losses.values()).backward()
    torch.testing.assert_close(model.motion.core.grad, torch.tensor(fm) + torch.tensor(aux))


@pytest.mark.parametrize('fm_weight,trace_weight', [(2., .25), (1., 0.), (0., .5)])
def test_motion_gradient_metrics_use_actual_loss_weights(fm_weight, trace_weight):
    from openpi.training.metrics import motion_gradient_balance
    model = MotionProbe()
    losses = model(torch.tensor([3., 4.]), torch.tensor([0., 5.]))
    weights = dict(action=fm_weight, skill=.02, current=.01, future_object=.01,
                   trace=trace_weight, gripper=.05)
    result = motion_gradient_balance(model, losses, weights)
    assert result['fm_norm'] == pytest.approx(5 * fm_weight)
    assert result['aux_norm'] == pytest.approx(5 * trace_weight)
    assert result['cosine'] == (pytest.approx(.8) if fm_weight and trace_weight else None)
    assert result['fm_aux_ratio'] == (pytest.approx(fm_weight / trace_weight) if trace_weight else None)
    assert all(p.grad is None for p in model.parameters())
    sum(weights[key] * loss for key, loss in losses.items()).backward()
    torch.testing.assert_close(model.motion.core.grad,
                               fm_weight * torch.tensor([3., 4.]) + trace_weight * torch.tensor([0., 5.]))


def motion_gradient_worker(rank, rendezvous):
    from openpi.training.metrics import motion_gradient_balance
    from torch.nn.parallel import DistributedDataParallel
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = MotionProbe()
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        wrapped = DistributedDataParallel(model, broadcast_buffers=False)
        weights = dict(action=2., skill=.02, current=.01, future_object=.01, trace=.25, gripper=.05)
        for empty_aux in (False, True):
            model.zero_grad(set_to_none=True)
            fm = torch.tensor([1., 0.] if rank == 0 else [-1., 2.])
            aux = torch.tensor([0., 0.] if empty_aux or rank else [0., 2.])
            losses = wrapped(fm, aux)
            result = motion_gradient_balance(model, losses, weights)
            assert result['fm_norm'] == pytest.approx(2.)  # Norm of averaged vectors, not averaged norms.
            assert result['aux_norm'] == pytest.approx(0. if empty_aux else .25)
            assert result['cosine'] == (None if empty_aux else pytest.approx(1.))
            assert all(p.grad is None for p in model.parameters())
            sum(weights[key] * loss for key, loss in losses.items()).backward()
            torch.testing.assert_close(model.motion.core.grad, torch.tensor([0., 2. if empty_aux else 2.25]))
    finally:
        dist.destroy_process_group()


def test_distributed_motion_gradients_average_vectors_before_norms(tmp_path):
    mp.spawn(motion_gradient_worker, args=((tmp_path/'motion_rendezvous').as_uri(),), nprocs=2, join=True)
