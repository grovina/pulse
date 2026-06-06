"""Roll out the modular network for flow-story protocols (torch)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from ..full_body import PatientParams, simulate_full_body
from .flow_story_protocol import (
    DIETARY_CARB_FLOW_DURATION_MIN,
    DIETARY_CARB_FLOW_MEALS,
    DIETARY_CARB_FLOW_START_HOUR,
)
from ...model import ModularPhysiologyNetwork, integrate
from ...modules.gut import MealEvent
from ...types import GUT_OUTPUT_DIM, STATE_DIM

DIETARY_CARB_FLOW_MEAL_EVENTS: list[MealEvent] = [
    MealEvent(
        time=DIETARY_CARB_FLOW_MEALS[0][0],
        carbs=DIETARY_CARB_FLOW_MEALS[0][1],
        fats=DIETARY_CARB_FLOW_MEALS[0][2],
        proteins=DIETARY_CARB_FLOW_MEALS[0][3],
    ),
]


def _cold_absorption_for_dietary_carb_flow(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Teacher trajectory and gut profile for the dietary-carbohydrate flow protocol."""
    params = PatientParams()
    duration_min = DIETARY_CARB_FLOW_DURATION_MIN
    sleep_wake = np.ones(duration_min, dtype=np.float32)
    activity = np.full(duration_min, 0.05, dtype=np.float32)
    prng = np.random.default_rng(rng.integers(0, 2**32))
    traj, absorption = simulate_full_body(
        params,
        DIETARY_CARB_FLOW_MEALS,
        sleep_wake,
        activity,
        duration_min,
        DIETARY_CARB_FLOW_START_HOUR,
        noise_scale=0.001,
        rng=prng,
    )
    return traj, absorption


def integrate_dietary_carbohydrate_flow_with_teacher_gut(
    model: ModularPhysiologyNetwork,
    embedding: torch.Tensor,
    rng: np.random.Generator,
    *,
    device: torch.device | None = None,
    initial_state: torch.Tensor | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the neural model for the flow-story protocol using cold-model gut outputs.

    Matches meal timing and per-minute glucose/lipid/amino appearance + nutrient flag
    from ``simulate_full_body`` so flow-story checks that depend on Ra stay well-defined.
    Initial state defaults to the teacher's t=0 state for the same RNG stream.
    """
    if device is None:
        device = next(model.parameters()).device
    traj_np, absorption_np = _cold_absorption_for_dietary_carb_flow(rng)
    if absorption_np.shape != (DIETARY_CARB_FLOW_DURATION_MIN, GUT_OUTPUT_DIM):
        raise ValueError(
            f"Expected absorption ({DIETARY_CARB_FLOW_DURATION_MIN}, {GUT_OUTPUT_DIM}), "
            f"got {absorption_np.shape}",
        )
    gut_t = torch.tensor(absorption_np, dtype=torch.float32, device=device)
    if initial_state is None:
        state0 = torch.tensor(traj_np[0], dtype=torch.float32, device=device)
    else:
        state0 = initial_state.to(device=device, dtype=torch.float32)
        if state0.shape[0] != STATE_DIM:
            raise ValueError(f"initial_state must have length {STATE_DIM}")

    duration_min = DIETARY_CARB_FLOW_DURATION_MIN
    sleep_wake = torch.ones(duration_min, dtype=torch.float32, device=device)
    activity = torch.full((duration_min,), 0.05, dtype=torch.float32, device=device)
    start_time_minutes = DIETARY_CARB_FLOW_START_HOUR * 60.0

    emb = embedding.to(device=device, dtype=torch.float32)
    if emb.shape[0] != model.embedding_dim:
        raise ValueError(
            f"embedding dim {emb.shape[0]} != model.embedding_dim {model.embedding_dim}",
        )

    model.eval()
    with torch.no_grad():
        pred = integrate(
            model,
            state0,
            emb,
            duration_min,
            dt=1.0,
            start_time_minutes=start_time_minutes,
            meals=DIETARY_CARB_FLOW_MEAL_EVENTS,
            sleep_wake=sleep_wake,
            activity=activity,
            gut_outputs=gut_t,
        )
    return pred.detach().cpu().numpy(), absorption_np


def make_dietary_carbohydrate_neural_trajectory_fn(
    model: ModularPhysiologyNetwork,
    embedding: torch.Tensor,
    *,
    device: torch.device | None = None,
    initial_state: torch.Tensor | None = None,
) -> Callable[[np.random.Generator], tuple[np.ndarray, np.ndarray]]:
    """Build ``trajectory_fn`` for ``dietary_carbohydrate_meal_flow_scenario``."""

    def _fn(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        return integrate_dietary_carbohydrate_flow_with_teacher_gut(
            model,
            embedding,
            rng,
            device=device,
            initial_state=initial_state,
        )

    return _fn
