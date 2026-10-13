"""
Gut appearance.

What reaches the blood is the stomach's outflow (``pulse.gastric``) times
the bioavailable fraction. The fraction is per person and per macro; the
time course is the calorie rate, shared with the teacher and with duodenal
delivery. A larger meal lasts longer. It does not leave faster.

``forward`` is one minute. ``forward_window`` is the whole window, stepped
from the start so a late query still sees what was swallowed earlier.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from .base import GutModuleBase
from ..gastric import FLAG_GATE_SCALE_G, EMPTY_KCAL_PER_MIN, simulate, swallow_from_meals
from ..types import GUT_OUTPUT_DIM


@dataclass
class MealEvent:
    time: float  # minutes from simulation start
    carbs: float
    fats: float
    proteins: float


# How far back a protocol keeps a meal. 720 min at 2.13 kcal/min is about
# 1500 kcal, so an ordinary meal has left and a very large one has not been
# dropped while it is still emptying. The trajectory signal and the benchmark
# use the same lookback.
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
        kcal_per_min: torch.Tensor | float = EMPTY_KCAL_PER_MIN,
    ) -> torch.Tensor:
        """Total nutrient appearance at one time-point from all active meals.

        Returns ``[glucose_appearance, lipid_appearance, amino_appearance, nutrient_flag]``.
        Vectorizes across active meals: one kernel call on ``(M_active, ...)``.
        """
        if not meals:
            return self._zero_output.clone()
        times = torch.tensor([float(t_minutes)], dtype=torch.float32, device=self._zero_output.device)
        return self.forward_window(times, meals, embedding, kcal_per_min=kcal_per_min).squeeze(0)

    def forward_window(
        self,
        times: torch.Tensor,
        meals: list[MealEvent],
        embedding: torch.Tensor,
        kcal_per_min: torch.Tensor | float = EMPTY_KCAL_PER_MIN,
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

        # The queried times may be a single late minute. The compartment has
        # to be stepped from the start of the window to reach it.
        last = int(times.detach().amax().item())
        horizon = max(last + 1, T)
        swallow = swallow_from_meals(horizon, meals, device)
        left, filled = simulate(swallow, kcal_per_min)
        idx = times.round().to(torch.long).clamp(0, horizon - 1)
        grams = left.index_select(0, idx)                              # [T, 3] carb, protein, fat
        present = filled.index_select(0, idx).sum(dim=-1)              # [T]
        _, f_bio = self.kernel.mixture(embedding)                       # [B, 3] appearance units per gram
        # Channel order is glucose, lipid, amino — carbohydrate, fat, protein.
        ordered = torch.stack([grams[:, 0], grams[:, 2], grams[:, 1]], dim=-1)
        appearance = ordered.unsqueeze(0) * f_bio.unsqueeze(1)
        flag = 1.0 - torch.exp(-present / FLAG_GATE_SCALE_G)
        out = torch.cat([appearance, flag.view(1, T, 1).expand(B, T, 1)], dim=-1)
        return out.squeeze(0) if unbatched else out
