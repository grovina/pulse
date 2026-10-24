"""One gastric compartment, shared by the teacher and the student.

Brener, Hendrix and McHugh, Gastroenterology 1983;85:76-82. Once glucose had
filled the stomach, it left at 2.13 kcal/min for every concentration they
tried. Carbohydrate, protein and fat that are in the stomach together share
that rate in proportion to the calories still there, so a fattier meal
delivers its carbohydrate more slowly. What leaves over the whole bout is
what entered.
"""

from __future__ import annotations

import math

import torch

from .correction import correction_weights

EMPTY_KCAL_PER_MIN = 2.13
KCAL_PER_G = (4.0, 4.0, 9.0)  # carbohydrate, protein, fat
FLAG_GATE_SCALE_G = 10.0


def _allocate_kcal(
    available: torch.Tensor, budget: torch.Tensor, weight: torch.Tensor,
) -> torch.Tensor:
    """Calories from each macro, proportional to ``available * weight``, capped by the pool.

    A pool has one part per macro, so this many passes always finish: each pass
    locks any part the share would empty, and the next pass gives the leftover
    calories to what remains. Equal weights are the calorie proportion, and
    there the cap does not bind.
    """
    remaining = available
    left = budget
    taken = torch.zeros_like(available)
    for _ in range(available.shape[-1]):
        weighted = remaining * weight
        total = weighted.sum(dim=-1, keepdim=True)
        proposal = left.unsqueeze(-1) * weighted / total.clamp(min=1e-12)
        proposal = torch.where(total > 0, proposal, torch.zeros_like(proposal))
        take = torch.minimum(proposal, remaining)
        taken = taken + take
        remaining = remaining - take
        left = left - take.sum(dim=-1)
    return taken


def outflow_grams(
    pool: torch.Tensor,
    kcal_per_min: torch.Tensor | float,
    correction: torch.Tensor | None = None,
) -> torch.Tensor:
    """Grams leaving this minute, same layout as ``pool`` (carb, protein, fat).

    The calorie rate is shared in proportion to the calories still in the pool.
    ``correction`` is a per-macro shift of that proportion. The teacher omits
    it. Zero leaves the proportion, and the calories that leave are still the
    rate, capped by what each macro has.
    """
    kcal_g = pool.new_tensor(KCAL_PER_G)
    available = (pool * kcal_g).clamp(min=0)
    kcal = available.sum(-1)
    rate = torch.as_tensor(kcal_per_min, dtype=pool.dtype, device=pool.device)
    budget = torch.minimum(rate.expand_as(kcal), kcal)
    weight = correction_weights(correction, n=pool.shape[-1], like=pool)
    return _allocate_kcal(available, budget, weight) / kcal_g


def simulate(
    swallow: torch.Tensor,
    kcal_per_min: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Step a compartment minute by minute.

    ``swallow`` is grams added at each minute, shape ``[T, 3]`` or ``[B, T, 3]``
    in carbohydrate, protein, fat order. Returns ``(outflow, filled)`` with
    ``filled`` the pool after that minute's swallow and before it emptied, so
    a flag can see that food is present.
    """
    batched = swallow.dim() == 3
    if not batched:
        swallow = swallow.unsqueeze(0)
    steps = int(swallow.shape[1])
    pool = swallow.new_zeros(swallow.shape[0], 3)
    outs = []
    filled = []
    for t in range(steps):
        pool = pool + swallow[:, t]
        filled.append(pool)
        left = outflow_grams(pool, kcal_per_min)
        outs.append(left)
        pool = pool - left
    out = torch.stack(outs, dim=1)
    full = torch.stack(filled, dim=1)
    if not batched:
        out = out.squeeze(0)
        full = full.squeeze(0)
    return out, full


def swallow_from_meals(n_steps: int, meals: list, device: torch.device) -> torch.Tensor:
    """Grams swallowed at each window-offset minute: carbohydrate, protein, fat."""
    out = torch.zeros(n_steps, 3, dtype=torch.float32, device=device)
    for meal in meals:
        # The first whole minute at or after the meal, matching the teacher's
        # integer clock (a meal at 60.2 is absent at minute 60).
        minute = int(math.ceil(float(meal.time) - 1e-9))
        if 0 <= minute < n_steps:
            out[minute, 0] = out[minute, 0] + float(meal.carbs)
            out[minute, 1] = out[minute, 1] + float(meal.proteins)
            out[minute, 2] = out[minute, 2] + float(meal.fats)
    return out


def step_grams(
    carb: float, protein: float, fat: float, kcal_per_min: float = EMPTY_KCAL_PER_MIN,
) -> tuple[float, float, float]:
    """The teacher's one minute. Returns grams of carb, protein and fat leaving."""
    kcal = KCAL_PER_G[0] * carb + KCAL_PER_G[1] * protein + KCAL_PER_G[2] * fat
    if kcal <= 0.0:
        return 0.0, 0.0, 0.0
    frac = min(kcal_per_min, kcal) / kcal
    return carb * frac, protein * frac, fat * frac
