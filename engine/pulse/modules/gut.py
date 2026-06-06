"""
Gut & Absorption module.

Learned absorption kernel that maps meals to nutrient appearance in the
bloodstream. Not an ODE — processes meal history into time-varying signals.
Multiple meals superimpose linearly (physically valid for normal eating).

Two batched APIs share one underlying kernel:
- ``forward(t, meals, embedding)``: single time-point. Vectorizes across
  the active meal list, so the kernel runs once on ``(M_active, ...)``
  instead of M_active sequential calls.
- ``forward_window(times, meals, embedding)``: every time-point in a
  window at once. Vectorizes across both T and M (one kernel call on
  ``(T*M_active, ...)`` then masked + summed). Replaces the per-step
  Python loop the trajectory signal used for gut-loss supervision.

Both paths are bit-equivalent to the previous looped implementation
modulo float-reduction order (≤1e-5 differences in practice).
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from .base import GutModuleBase
from ..types import GUT_OUTPUT_DIM


@dataclass
class MealEvent:
    time: float  # minutes from simulation start
    carbs: float
    fats: float
    proteins: float


# Maximum age of a meal that still contributes to absorption (minutes).
# Beyond this the kernel output is treated as zero.
MEAL_ACTIVE_WINDOW_MIN: float = 480.0


# Per-channel typical-excursion scale for gut appearance outputs (mg/min for
# the macro channels, dimensionless for nutrient_flag). Used by every signal
# that supervises ``forward_window`` so MSE on the gut kernel has comparable
# magnitude across channels and to typical state MSE in mg/dL after dividing
# by NORM_SCALE. Numbers were chosen from cold-model peaks across the meal
# distribution we train on (typical mixed-meal peaks: glucose ~2 mg/min,
# lipid ~0.3, amino ~0.5; nutrient_flag is binary). The previous scale
# inherited from blood-state (mg/dL) units flattened gut MSE by ~225-1000x
# per channel — see iter 12 → iter 13 handoff.
GUT_OUTPUT_SCALE: tuple[float, float, float, float] = (2.0, 0.3, 0.5, 1.0)


def _active_meal_tensors(
    t_minutes: float,
    meals: list[MealEvent],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return (macros[M_active, 3], dt[M_active]) for meals active at ``t_minutes``,
    or ``None`` if no meal is active."""
    rows = []
    dts = []
    for meal in meals:
        dt = t_minutes - meal.time
        if 0.0 <= dt <= MEAL_ACTIVE_WINDOW_MIN:
            rows.append((meal.carbs, meal.fats, meal.proteins))
            dts.append(dt)
    if not rows:
        return None
    macros = torch.tensor(rows, dtype=torch.float32, device=device)
    dt_tensor = torch.tensor(dts, dtype=torch.float32, device=device)
    return macros, dt_tensor


class GutModule(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.kernel = GutModuleBase(embedding_dim=embedding_dim, hidden_dim=hidden_dim)
        self.register_buffer(
            "_zero_output", torch.zeros(GUT_OUTPUT_DIM, dtype=torch.float32),
        )

    def forward(
        self,
        t_minutes: float,
        meals: list[MealEvent],
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Total nutrient appearance at one time-point from all active meals.

        Returns ``[glucose_appearance, lipid_appearance, amino_appearance, nutrient_flag]``.
        Vectorizes across active meals: one kernel call on ``(M_active, ...)``.
        """
        if not meals:
            return self._zero_output.clone()
        active = _active_meal_tensors(t_minutes, meals, self._zero_output.device)
        if active is None:
            return self._zero_output.clone()
        macros, dt = active
        emb_batch = embedding.unsqueeze(0).expand(macros.shape[0], -1)
        appearance = self.kernel.forward_single_meal(macros, dt, emb_batch)
        return appearance.sum(dim=0)

    def forward_window(
        self,
        times: torch.Tensor,
        meals: list[MealEvent],
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        """All time-points in a window at once.

        Shape-polymorphic on the embedding's leading dim:

        - ``embedding[EMB]``  → ``[T, GUT_OUTPUT_DIM]``
        - ``embedding[B, EMB]`` → ``[B, T, GUT_OUTPUT_DIM]``

        ``times`` is a 1-D tensor of absolute minute-of-day values, length
        T (shared across batch members — gut depends on (t, meals, emb),
        not on state, so per-batch times aren't useful here). Output is
        bit-equivalent (modulo float-reduction order) to stacking
        ``forward(t, meals, embedding)`` for each ``t`` and each batch row.

        Vectorizes across B, T, and the meal list, so the kernel runs once
        on ``(B·T·M, ...)`` with masking — one kernel call per window
        regardless of batch size.
        """
        unbatched = embedding.dim() == 1
        if unbatched:
            embedding = embedding.unsqueeze(0)
        B = int(embedding.shape[0])
        T = int(times.shape[0])
        device = times.device

        if not meals:
            out = self._zero_output.to(device).expand(B, T, -1).clone()
            return out.squeeze(0) if unbatched else out

        macros = torch.tensor(
            [(m.carbs, m.fats, m.proteins) for m in meals],
            dtype=torch.float32, device=device,
        )  # [M, 3]
        meal_times = torch.tensor(
            [m.time for m in meals], dtype=torch.float32, device=device,
        )  # [M]

        dt = times.unsqueeze(1) - meal_times.unsqueeze(0)  # [T, M]
        mask = (dt >= 0.0) & (dt <= MEAL_ACTIVE_WINDOW_MIN)  # [T, M]

        M = int(macros.shape[0])
        # One kernel call on (B*T*M, ...) — broadcast B and T across the
        # meal-and-embedding axes, then reshape back at the end.
        dt_flat = dt.unsqueeze(0).expand(B, -1, -1).reshape(-1)
        macros_flat = (
            macros.unsqueeze(0).unsqueeze(0)
            .expand(B, T, -1, -1)
            .reshape(B * T * M, 3)
        )
        emb_flat = (
            embedding.unsqueeze(1).unsqueeze(2)
            .expand(-1, T, M, -1)
            .reshape(B * T * M, -1)
        )

        appearance = self.kernel.forward_single_meal(macros_flat, dt_flat, emb_flat)
        appearance = appearance.reshape(B, T, M, GUT_OUTPUT_DIM)
        appearance = appearance * mask.unsqueeze(0).unsqueeze(-1).to(appearance.dtype)
        out = appearance.sum(dim=2)  # [B, T, GUT_OUTPUT_DIM]
        return out.squeeze(0) if unbatched else out
