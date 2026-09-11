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
  window at once. Vectorizes across both T and M. Replaces the per-step
  Python loop the trajectory signal used for gut-loss supervision.

Iter 97: the kernel is ``f_bio(emb) · density(t; emb)`` with an analytic,
normalized, zero-at-origin basis (see ``base.GutModuleBase``), so the
learned MLP runs once per patient and time enters only through the basis.
The appearance channels are additive across meals; ``nutrient_flag`` is
``1 − exp(−unabsorbed_mass / 10 g)`` and combines across meals as
``1 − Π(1 − flag_m)`` (which is the same statement about the summed
unabsorbed mass), so it stays in [0, 1) for any number of meals.
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
#
# Iter 97: a NUMERICAL NO-OP, not a cliff. Every component of the kernel's basis
# bank carries < 1 % of its mass beyond this age by construction
# (``GutModuleBase._KERNEL_BASIS``; asserted in tests/test_gut_module.py), so the
# mask only saves compute on long-dead meals. Through iter 96 the kernel ended in
# a hard step here (0.49 → 0 at 479 → 481 min for a 60 g meal), which for a 19:00
# dinner put a −8 mg/dL/h glucose cliff at 03:00 — inside the scored pre-dawn
# window. Raising this constant is safe; lowering it below ~480 is not.
#
# 720, not 480: the teacher's mass-conserving kernels (iter 97) run to 8/rate,
# and its slow carbohydrate component still holds ~1.5 % of a meal at 480 min
# (15 % of a 60 g load appears after 240 min). The window — which is ALSO the
# meal lookback used by the trajectory signal and the benchmark — must cover
# what the teacher absorbs, or the student sees a fasted input on a fed state.
MEAL_ACTIVE_WINDOW_MIN: float = 720.0


# Per-channel typical-excursion scale for gut appearance outputs (mg/dL/min in
# the 70 kg reference space for glucose; appearance-units/min for lipid/amino;
# dimensionless for nutrient_flag). Training supervises the three appearance
# channels; the flag is a different quantity in teacher vs student and is not
# distilled. The fourth scale remains so diagnostics that plot the flag have a
# number. Peaks from the cold-model meal distribution we train on: glucose ~2,
# lipid ~0.3, amino ~0.5; flag is in [0, 1).
GUT_OUTPUT_SCALE: tuple[float, float, float, float] = (2.0, 0.3, 0.5, 1.0)

# Appearance units per gram of macro (teacher convention), re-exported so the
# metabolic module can recover grams for its carbon budget without reaching
# into the kernel class.
APPEARANCE_UNITS_PER_G: tuple[float, float, float] = GutModuleBase.APPEARANCE_UNITS_PER_G


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


def combine_meals(per_meal: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Sum the appearance channels over the meal axis (dim −2) and combine the
    flags as ``1 − Π(1 − flag_m)``. ``per_meal[..., M, 4]`` → ``[..., 4]``."""
    if mask is not None:
        per_meal = per_meal * mask.unsqueeze(-1).to(per_meal.dtype)
    appearance = per_meal[..., :3].sum(dim=-2)
    flag = 1.0 - torch.prod(1.0 - per_meal[..., 3], dim=-1)
    return torch.cat([appearance, flag.unsqueeze(-1)], dim=-1)


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
        per_meal = self.kernel.forward_single_meal(macros, dt, emb_batch)  # [M, 4]
        return combine_meals(per_meal)

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

        ``times`` is a 1-D tensor in the SAME frame as ``meal.time`` — i.e.
        window-offset minutes (0-based from the window start), NOT absolute
        minute-of-day. Absorption is a function of ``dt = times - meal.time``
        (see below), and meal times are 0-based offsets matching the teacher's
        ``simulate_full_body`` frame, the benchmark dataset, and every
        scenario/protocol generator. Passing an absolute-minute-of-day clock
        here shifts every meal's absorption curve by the window start and
        crushes post-meal amplitude — this was the iter-87 frame bug. Length
        T, shared across batch members (gut depends on (t, meals, emb), not on
        state, so per-batch times aren't useful here). Output equals combining
        ``forward(t, meals, embedding)`` over ``t`` for each batch row (modulo
        float-reduction order).

        The learned part of the kernel runs ONCE per batch row; the analytic
        basis runs once per (t, meal); the two meet in one einsum.
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
        mask = ((dt >= 0.0) & (dt <= MEAL_ACTIVE_WINDOW_MIN)).to(torch.float32)  # [T, M]

        weights, f_bio = self.kernel.mixture(embedding)      # [B, 3, K], [B, 3]
        density = self.kernel.basis_density(dt)              # [T, M, K]
        survival = self.kernel.basis_survival(dt)            # [T, M, K]

        # appearance[b,t,m,j] = macros[m,j] · f_bio[b,j] · Σ_k w[b,j,k] · basis_k(dt[t,m])
        appearance = torch.einsum("bjk,tmk,mj,bj->btmj", weights, density, macros, f_bio)
        unabsorbed = torch.einsum("bjk,tmk,mj->btm", weights, survival, macros)
        m = mask.unsqueeze(0)
        appearance = (appearance * m.unsqueeze(-1)).sum(dim=2)               # [B, T, 3]
        flag = 1.0 - torch.exp(-(unabsorbed * m).sum(dim=2) / self.kernel.FLAG_GATE_SCALE_G)
        out = torch.cat([appearance, flag.unsqueeze(-1)], dim=-1)            # [B, T, 4]
        return out.squeeze(0) if unbatched else out
