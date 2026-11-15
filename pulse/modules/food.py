"""What a meal is made of, read into how the gut delivers it.

A meal reaches the model as minute, carbohydrate, fat and protein. A paper may say
more (``knowledge.cohort_types.MealDetails``): fibre, sugars, fat by type, the protein's
source, liquid or solid. ``meal_features`` turns that into a fixed vector, and
``MealReader`` turns the vector into two per-meal modifiers, each per macronutrient:

* **speed**, how fast the meal leaves the stomach. The stomach empties ``R`` kcal/min
  in proportion to the calories in it (``pulse.gastric``): a part holding ``kcal_i`` of
  ``K`` leaves ``kcal_i · min(1, R/K)``. A meal's speed ``s_i`` scales its own parts'
  share, ``kcal_i · min(1, R·s_i/K)``, so a liquid or whey meal empties faster and a
  casein or fibre-rich one slower, and everything still leaves. At ``s = 1`` this is the
  shared law exactly. The stomach's outflow is also the duodenum's delivery, so speed
  moves both.
* **share**, how much of what leaves is absorbed: a shift of the logit of the person's
  bioavailable fraction (``GutModuleBase.bioavailable_fraction``), so the fraction stays
  in (0, 1) and appearance never passes one gram's worth per gram eaten (A9). Share
  moves the blood only; the duodenum still sees the grams.

Both are bounded: speed in [1/3, 3] and the logit shift in [-ln 3, ln 3] (the odds of
absorption between a third and three times), so training cannot make a meal vanish or
pour out at once. The last layer starts at zero, so a fresh reader changes nothing.

``fat_source`` and ``food`` are free text and are not read. A meal that states only
those, or nothing, has no features (``None``) and is delivered exactly as without a
reader.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..types import FAT_TYPES, MEAL_FORMS, PROTEIN_SOURCES

# The feature vector, in order. Shares are dimensionless in [0, 1]: fibre over all
# carbohydrate (available plus fibre), sugars over available carbohydrate, each fat type
# over the fat. Every numeric and categorical field has a mask entry, 1 when the paper
# states it, so "not stated" is not the same as zero.
_MASKED = ("fibre", "sugars", *FAT_TYPES, "protein_source", "form")
FEATURE_NAMES: tuple[str, ...] = (
    "fibre_share", "sugars_share", *(f"{t}_share" for t in FAT_TYPES),
    *(f"protein_{s}" for s in PROTEIN_SOURCES), *(f"form_{f}" for f in MEAL_FORMS),
    *(f"stated_{k}" for k in _MASKED),
)
FEATURE_DIM = len(FEATURE_NAMES)

# Stomach order, as ``pulse.gastric``: carbohydrate, protein, fat.
MACROS = ("carb", "protein", "fat")
LOG_BOUND = math.log(3.0)


def _share(part: float, whole: float) -> float:
    return min(max(part / whole, 0.0), 1.0) if whole > 0.0 else 0.0


def meal_features(meal) -> list[float] | None:
    """``FEATURE_DIM`` numbers for one meal (a ``MealEvent``), or ``None`` when its
    ``details`` state nothing the reader reads."""
    d = getattr(meal, "details", None)
    if d is None:
        return None
    by_type = dict(d.fat_g_by_type)
    stated = (
        d.fibre_g is not None, d.sugars_g is not None, *(t in by_type for t in FAT_TYPES),
        d.protein_source is not None, d.form is not None,
    )
    if not any(stated):
        return None
    carbs, fats = float(meal.carbs), float(meal.fats)
    fibre = float(d.fibre_g) if d.fibre_g is not None else 0.0
    values = [
        _share(fibre, carbs + fibre),
        _share(float(d.sugars_g), carbs) if d.sugars_g is not None else 0.0,
        *(_share(float(by_type[t]), fats) if t in by_type else 0.0 for t in FAT_TYPES),
        *(1.0 if d.protein_source == s else 0.0 for s in PROTEIN_SOURCES),
        *(1.0 if d.form == f else 0.0 for f in MEAL_FORMS),
        *(1.0 if s else 0.0 for s in stated),
    ]
    assert len(values) == FEATURE_DIM
    return values


def meal_feature_table(
    meal_lists: list[list], device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """``(features[M, K, FEATURE_DIM], stated[M, K])`` for ``M`` meal lists padded to the
    longest ``K``, or ``None`` when no meal of any list has features."""
    rows = [[meal_features(m) for m in meals] for meals in meal_lists]
    if not any(f is not None for row in rows for f in row):
        return None
    k = max(len(row) for row in rows)
    feats = torch.zeros(len(rows), k, FEATURE_DIM, dtype=torch.float32)
    stated = torch.zeros(len(rows), k, dtype=torch.bool)
    for i, row in enumerate(rows):
        for j, f in enumerate(row):
            if f is not None:
                feats[i, j] = torch.tensor(f, dtype=torch.float32)
                stated[i, j] = True
    return feats.to(device), stated.to(device)


class MealReader(nn.Module):
    """Features of a meal -> ``(speed, logit_shift)``, each ``[..., 3]`` in stomach order.

    ``FEATURE_DIM -> hidden -> 6`` with a tanh between; 262 parameters at ``hidden=8``.
    The last layer is zero, so a fresh reader returns speed 1 and shift 0 exactly.
    """

    def __init__(self, hidden: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(FEATURE_DIM, hidden), nn.Tanh(), nn.Linear(hidden, 2 * len(MACROS)))
        with torch.no_grad():
            self.net[-1].weight.zero_()
            self.net[-1].bias.zero_()

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = LOG_BOUND * torch.tanh(self.net(features))
        n = len(MACROS)
        return torch.exp(raw[..., :n]), raw[..., n:]

    def modifiers(
        self, features: torch.Tensor, stated: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``modifiers`` of every meal, with speed 1 and shift 0 exactly where ``stated``
        is False (a meal with no features, or a padding slot)."""
        speed, shift = self(features)
        on = stated.unsqueeze(-1)
        return torch.where(on, speed, torch.ones_like(speed)), torch.where(on, shift, torch.zeros_like(shift))


__all__ = [
    "FEATURE_DIM",
    "FEATURE_NAMES",
    "LOG_BOUND",
    "MACROS",
    "MealReader",
    "meal_feature_table",
    "meal_features",
]
