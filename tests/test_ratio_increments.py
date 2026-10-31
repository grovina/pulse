"""The ratio of two baseline-subtracted areas.

Vollmer 2008 reports integrated incremental GIP, not the ratio of the two
window means. The fasting level is different on the two plates only if the
run-in differs, and it has to cancel either way.
"""

from __future__ import annotations

import torch

from pulse.cohort_loss import _score_arm_trajs
from pulse.knowledge.cohort_statistics import ALL_COHORT_STATISTICS, SEALED_COHORT_STATISTICS
from pulse.knowledge.cohorts.incretins import VOLLMER_2008_GIP
from pulse.types import EMBEDDING_DIM, SERIES_INDEX, STATE_DIM

_GIP = SERIES_INDEX["gip"]
_T = 300


def _arm(baseline: float, rise: float) -> torch.Tensor:
    traj = torch.zeros(1, _T, STATE_DIM)
    traj[:, :60, _GIP] = baseline
    traj[:, 60:, _GIP] = baseline + rise
    return traj


def test_each_arm_subtracts_its_own_baseline() -> None:
    names = {s.name for s in (*ALL_COHORT_STATISTICS, *SEALED_COHORT_STATISTICS)}
    assert VOLLMER_2008_GIP.name in names
    # Rises of 2 and 5.72 are a ratio of 2.86 whatever the two baselines are.
    # A ratio of the raw means would be (4 + 5.72) / (10 + 2).
    loss, pred, z = _score_arm_trajs(
        VOLLMER_2008_GIP,
        [_arm(10.0, 2.0), _arm(4.0, 5.72)],
        torch.zeros(1, EMBEDDING_DIM),
        None,
    )
    assert abs(pred - 2.86) < 1e-4
    assert abs(z) < 1e-3
    assert float(loss.detach()) < 1e-6


def test_the_board_scores_the_ratio_as_one_residual() -> None:
    from pulse.diagnostics.identifiability import _cohort_residual
    from pulse.model import ModularPhysiologyNetwork
    from pulse.types import NORM_CENTER

    torch.manual_seed(0)
    model = ModularPhysiologyNetwork()
    emb = torch.zeros(1, EMBEDDING_DIM, requires_grad=True)
    raw, _shaped = _cohort_residual(
        model, VOLLMER_2008_GIP, emb, torch.tensor(NORM_CENTER),
    )
    assert torch.isfinite(raw).all()
    raw.sum().backward()
    fat = model.appetite.gip_kinetics.log_gip_fat_weight
    assert fat.grad is not None and float(fat.grad) != 0.0


def test_the_breakfast_reaches_the_fat_weight() -> None:
    from pulse.cohort_loss import cohort_statistic_loss_one_spec
    from pulse.model import ModularPhysiologyNetwork
    from pulse.types import NORM_CENTER

    torch.manual_seed(0)
    model = ModularPhysiologyNetwork()
    loss, pred, _z = cohort_statistic_loss_one_spec(
        model, [torch.zeros(EMBEDDING_DIM)], VOLLMER_2008_GIP, torch.tensor(NORM_CENTER),
    )
    assert pred > 1.0
    model.zero_grad()
    loss.backward()
    fat = model.appetite.gip_kinetics.log_gip_fat_weight
    assert fat.grad is not None and float(fat.grad) != 0.0


def test_the_published_ratio_is_the_ratio_of_the_mean_areas() -> None:
    glucose = torch.cat([_arm(0.0, 1.0), _arm(0.0, 3.0)], dim=0)
    breakfast = torch.cat([_arm(0.0, 1.0), _arm(0.0, 9.0)], dim=0)
    _loss, pred, _z = _score_arm_trajs(
        VOLLMER_2008_GIP,
        [glucose, breakfast],
        torch.zeros(2, EMBEDDING_DIM),
        2,
    )
    assert abs(pred - 2.5) < 1e-4


def test_the_area_carries_a_gradient() -> None:
    rise = torch.tensor(5.72, requires_grad=True)
    breakfast = _arm(4.0, 1.0)
    breakfast[:, 60:, _GIP] = 4.0 + rise
    loss, _, _ = _score_arm_trajs(
        VOLLMER_2008_GIP,
        [_arm(10.0, 2.0), breakfast],
        torch.zeros(1, EMBEDDING_DIM),
        None,
    )
    loss.backward()
    assert rise.grad is not None and float(rise.grad) != 0.0
