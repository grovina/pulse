"""Shared types and cold-model trajectory helper for textbook scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ...types import MARKER_INDEX
from ..full_body import PatientParams, simulate_full_body

# Same-minute grid as simulate_full_body: state row + gut outputs (4 columns).
TrajectoryProvider = Callable[[np.random.Generator], tuple[np.ndarray, np.ndarray]]

ScenarioFn = Callable[[np.random.Generator, TrajectoryProvider | None], "ScenarioResult"]


@dataclass
class ScenarioCheck:
    name: str
    description: str
    passed: bool
    value: float
    threshold: float

    # Iter 97 (review 5.9): the thresholds are hard-edged and sit 3-15% from
    # the teacher's own value, so a pass rate moves in steps of 1/n_checks on
    # hairline misses (iter 96's 0.95 -> 0.8625 was cortisol ratio 1.291 vs
    # 1.3 and bhb -0.006 vs 0). The binary rate is kept -- the thresholds are
    # NOT moved -- and a signed, scale-free margin is reported next to it.
    # The check's direction is not stored, so it is inferred from the verdict:
    # a check that passed with value > threshold (or failed with value <=
    # threshold) is an "above" check; otherwise "below".
    @property
    def direction(self) -> str:
        if self.value == self.threshold:
            return "above"
        above = self.value > self.threshold
        return "above" if self.passed == above else "below"

    @property
    def signed_margin(self) -> float:
        """> 0 = passed by this much (raw units), < 0 = missed by this much."""
        d = float(self.value) - float(self.threshold)
        return d if self.direction == "above" else -d

    @property
    def relative_margin(self) -> float:
        """signed_margin over |threshold|: -0.03 is a 3% miss. A zero threshold
        (bhb >= 0, ACTH-before-cortisol timing) has no scale of its own, so the
        margin is taken in raw units, floored at 1 unit."""
        thr = abs(float(self.threshold))
        scale = thr if thr > 1e-9 else max(abs(float(self.value)), 1.0)
        return self.signed_margin / scale

    def soft_score(self, softness: float = 0.10) -> float:
        """sigmoid(relative_margin / softness): 0.5 at the edge, ~1 well inside."""
        import math
        x = self.relative_margin / max(softness, 1e-6)
        return 1.0 / (1.0 + math.exp(-max(min(x, 60.0), -60.0)))

    def is_hairline(self, tolerance: float = 0.05) -> bool:
        return abs(self.relative_margin) < tolerance


@dataclass
class ScenarioResult:
    name: str
    source: str
    description: str
    checks: list[ScenarioCheck]
    pass_rate: float


def cold_model_trajectory(
    params: PatientParams,
    meals: list[tuple[float, float, float, float]],
    duration_min: int,
    start_hour: float,
    sleep_wake: np.ndarray | None = None,
    activity: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if sleep_wake is None:
        sleep_wake = np.ones(duration_min, dtype=np.float32)
    if activity is None:
        activity = np.full(duration_min, 0.05, dtype=np.float32)
    traj, _ = simulate_full_body(
        params, meals, sleep_wake, activity,
        duration_min, start_hour, noise_scale=0.001, rng=rng,
    )
    return traj


# Re-export for scenarios that reference marker layout
__all__ = [
    "ScenarioCheck",
    "ScenarioResult",
    "ScenarioFn",
    "cold_model_trajectory",
    "MARKER_INDEX",
]
