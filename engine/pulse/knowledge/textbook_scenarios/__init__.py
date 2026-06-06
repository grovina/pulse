"""
Textbook physiology scenarios for model validation.

Domain modules each define SCENARIO_RUNNERS; this package merges them.
Add scenarios by extending a domain file or adding a new module and listing
it in ALL_SCENARIO_RUNNERS below.
"""

from __future__ import annotations

import numpy as np

from .base import (
    ScenarioCheck,
    ScenarioResult,
    ScenarioFn,
    TrajectoryProvider,
    cold_model_trajectory,
)
from . import metabolic, exercise, sleep_circadian, flow_stories

ALL_SCENARIO_RUNNERS: list[ScenarioFn] = [
    *metabolic.SCENARIO_RUNNERS,
    *exercise.SCENARIO_RUNNERS,
    *sleep_circadian.SCENARIO_RUNNERS,
    *flow_stories.SCENARIO_RUNNERS,
]


def run_all_scenarios(
    trajectory_fn: TrajectoryProvider | None = None,
    trajectory_fn_by_name: dict[str, TrajectoryProvider] | None = None,
    seed: int = 42,
) -> list[ScenarioResult]:
    """Run all merged textbook scenarios.

    By default each scenario uses its cold-model trajectory (typically ignores
    ``trajectory_fn``). Pass ``trajectory_fn`` to supply one provider for every
    runner that accepts it (today only the dietary-carbohydrate flow story uses it;
    others ignore the argument). Prefer ``trajectory_fn_by_name`` to attach a
    provider to specific scenario functions without affecting the rest: keys are
    scenario callables' ``__name__`` (e.g. ``\"dietary_carbohydrate_meal_flow_scenario\"``).

    Only one of ``trajectory_fn`` or ``trajectory_fn_by_name`` may be non-None.
    """
    if trajectory_fn is not None and trajectory_fn_by_name is not None:
        raise ValueError("Pass at most one of trajectory_fn or trajectory_fn_by_name")
    rng = np.random.default_rng(seed)
    results: list[ScenarioResult] = []
    for fn in ALL_SCENARIO_RUNNERS:
        tfn: TrajectoryProvider | None = None
        if trajectory_fn is not None:
            tfn = trajectory_fn
        elif trajectory_fn_by_name is not None:
            tfn = trajectory_fn_by_name.get(fn.__name__)
        results.append(fn(rng, tfn))
    return results


__all__ = [
    "ScenarioCheck",
    "ScenarioResult",
    "ScenarioFn",
    "TrajectoryProvider",
    "cold_model_trajectory",
    "ALL_SCENARIO_RUNNERS",
    "run_all_scenarios",
]
