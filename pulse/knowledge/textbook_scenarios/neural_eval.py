"""Run textbook qualitative scenarios on a trained checkpoint (neural trajectories)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .base import ScenarioResult
from .exercise import acute_z2_bout_neural_on_model, exercise_bout_neural_on_model
from .flow_stories import dietary_carbohydrate_meal_flow_scenario
from .metabolic import (
    meal_dose_response_neural_on_model,
    ogtt_neural_on_model,
    overnight_fast_neural_on_model,
)
from .neural_rollout import make_dietary_carbohydrate_neural_trajectory_fn
from .sleep_circadian import (
    cortisol_awakening_neural_on_model,
    sleep_wake_transition_neural_on_model,
)
from ...model import ModularPhysiologyNetwork
from ...types import EMBEDDING_DIM


def _scenario_to_dict(r: ScenarioResult) -> dict[str, Any]:
    return {
        "name": r.name,
        "source": r.source,
        "description": r.description,
        "pass_rate": r.pass_rate,
        "checks": [
            {
                "name": c.name,
                "description": c.description,
                "passed": c.passed,
                "value": float(c.value),
                "threshold": float(c.threshold),
            }
            for c in r.checks
        ],
    }


def run_textbook_scenarios_on_model(
    model: ModularPhysiologyNetwork,
    *,
    rng: np.random.Generator | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Six neural textbook rollouts (OGTT, fast, meal dose, exercise, sleep, CAR) + dietary flow story."""
    if rng is None:
        rng = np.random.default_rng(42)
    dev = torch.device(device)
    emb = torch.zeros(EMBEDDING_DIM, dtype=torch.float32, device=dev)

    results: list[ScenarioResult] = [
        ogtt_neural_on_model(model, rng, device=device),
        overnight_fast_neural_on_model(model, rng, device=device),
        meal_dose_response_neural_on_model(model, rng, device=device),
        exercise_bout_neural_on_model(model, rng, device=device),
        acute_z2_bout_neural_on_model(model, rng, device=device),
        sleep_wake_transition_neural_on_model(model, rng, device=device),
        cortisol_awakening_neural_on_model(model, rng, device=device),
        dietary_carbohydrate_meal_flow_scenario(
            rng,
            trajectory_fn=make_dietary_carbohydrate_neural_trajectory_fn(
                model, emb, device=dev,
            ),
        ),
    ]

    mean_pass = float(np.mean([r.pass_rate for r in results])) if results else 0.0
    return {
        "textbook_mean_pass_rate": mean_pass,
        "textbook_scenarios": [_scenario_to_dict(r) for r in results],
    }
