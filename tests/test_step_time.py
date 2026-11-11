"""The step-time tool times the same sampled step down both paths."""

from __future__ import annotations

import pytest
import torch

from pulse.diagnostics import step_time as tool
from pulse.knowledge.cohort_types import CohortArmSpec, CohortStatisticSpec, StatisticKind, StatisticWindow
from pulse.model import ModularPhysiologyNetwork


def _spec(name: str, minutes: int) -> CohortStatisticSpec:
    arm = CohortArmSpec(label="a", duration_min=minutes, start_hour=8.0, meals=((5.0, 40.0, 10.0, 10.0),))
    return CohortStatisticSpec(
        name=name, source="t", description="", arms=(arm,), marker_id="glucose",
        kind=StatisticKind.MEAN_IN_WINDOW, window=StatisticWindow(5, minutes), target=100.0, sigma=5.0,
    )


def test_the_arm_cap_keeps_only_short_items_and_both_paths_are_timed(monkeypatch) -> None:
    monkeypatch.setattr(tool, "ALL_COHORT_STATISTICS", (_spec("short", 20), _spec("long", 400)))
    monkeypatch.setattr(tool, "PHYSIOLOGY_RULES", ())
    torch.manual_seed(0)
    result = tool.step_time(
        ModularPhysiologyNetwork(metabolic_hidden=8, appetite_hidden=8, stress_hidden=8,
                                 cardiovascular_hidden=8, thermoreg_hidden=8, respiratory_hidden=8),
        batch_size=2, n_draws=2, max_arm_minutes=60, n_steps=1, fed_day_minutes=10,
    )
    assert (result.pool_items, result.pool_rules) == (1, 0)
    assert result.serial_s > 0.0 and result.batched_s > 0.0
    assert result.n_backwards == 1.0 and result.groups == [[(4, 20)]]
    assert "speed-up" in tool.render(result)
