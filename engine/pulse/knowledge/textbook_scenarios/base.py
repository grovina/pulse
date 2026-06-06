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
