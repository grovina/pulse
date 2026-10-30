"""Cryer 1987: a fasting glucose with a stated SD, and baseline epinephrine.

The SD is the paper's, marked as a standard deviation. The spread's sigma is
the sampling error of that SD. Baseline epinephrine is a level, so it reaches
the resting parameter. The rise near 80 mg/dL does not, and is not filed.
"""

from __future__ import annotations

import torch

from pulse.cohort_loss import _score_arm_trajs, cohort_statistic_loss_one_spec
from pulse.knowledge.cohort_statistics import ALL_COHORT_STATISTICS, SEALED_COHORT_STATISTICS
from pulse.knowledge.cohorts.counterregulation import (
    CRYER_1987_EPINEPHRINE_BASELINE,
    CRYER_1987_FASTING_GLUCOSE,
    CRYER_1987_FASTING_GLUCOSE_SPREAD,
)
from pulse.model import ModularPhysiologyNetwork
from pulse.types import EMBEDDING_DIM, NORM_CENTER, SERIES_INDEX, STATE_DIM

_GLUCOSE = SERIES_INDEX["glucose"]
_EPI = SERIES_INDEX["epinephrine"]


def _flat(level_by_person: list[float], column: int) -> torch.Tensor:
    traj = torch.zeros(len(level_by_person), 30, STATE_DIM)
    for i, level in enumerate(level_by_person):
        traj[i, :, column] = level
    return traj


def test_the_sd_is_seven_and_the_mean_is_ninety_one() -> None:
    names = {s.name for s in (*ALL_COHORT_STATISTICS, *SEALED_COHORT_STATISTICS)}
    assert CRYER_1987_FASTING_GLUCOSE.name in names
    assert CRYER_1987_FASTING_GLUCOSE_SPREAD.name in names
    assert CRYER_1987_EPINEPHRINE_BASELINE.name in names

    glucose = _flat([84.0, 91.0, 98.0], _GLUCOSE)
    codes = torch.zeros(3, EMBEDDING_DIM)
    _loss, mean, mean_z = _score_arm_trajs(CRYER_1987_FASTING_GLUCOSE, [glucose], codes, None)
    _loss, spread, spread_z = _score_arm_trajs(
        CRYER_1987_FASTING_GLUCOSE_SPREAD, [glucose], codes, None,
    )
    assert abs(mean - 91.0) < 1e-4
    assert abs(mean_z) < 1e-3
    assert abs(spread - 7.0) < 1e-4
    assert abs(spread_z) < 1e-3


def test_resting_splanchnic_output_reaches_glucose_uptake() -> None:
    from pulse.knowledge.cohorts.glucose_handling import WAHREN_RESTING_SPLANCHNIC_GLUCOSE

    torch.manual_seed(0)
    model = ModularPhysiologyNetwork()
    loss, pred, _z = cohort_statistic_loss_one_spec(
        model,
        [torch.zeros(EMBEDDING_DIM)],
        WAHREN_RESTING_SPLANCHNIC_GLUCOSE,
        torch.tensor(NORM_CENTER),
    )
    assert pred > 0.0
    model.zero_grad()
    loss.backward()
    uptake = model.metabolic.log_k_ii
    assert uptake.grad is not None and float(uptake.grad.detach()) != 0.0


def test_baseline_epinephrine_reaches_its_level() -> None:
    torch.manual_seed(0)
    model = ModularPhysiologyNetwork()
    loss, _pred, _z = cohort_statistic_loss_one_spec(
        model,
        [torch.zeros(EMBEDDING_DIM)],
        CRYER_1987_EPINEPHRINE_BASELINE,
        torch.tensor(NORM_CENTER),
    )
    model.zero_grad()
    loss.backward()
    level = model.stress.adrenal.log_epi_b
    assert level.grad is not None and float(level.grad.detach()) != 0.0
