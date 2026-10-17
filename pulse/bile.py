"""How much of the fat that leaves the stomach is absorbed.

Borgström, Barrowman and Lindström, New Comprehensive Biochemistry
1985, chapter 14: in the absence of bile, 60–70% of a normal fat load
is still absorbed. The floor is the midpoint of that range. The
intestinal content at rest is bile present — the marker's typical,
1 mmol — and at that content and above, the fat that leaves the stomach
is absorbed in full. Below it the fraction falls, in a straight line,
to the floor at an empty intestine. It never exceeds 1: appearance is
already the whole outflow.
"""

from __future__ import annotations

import torch

BILE_FAT_FLOOR = 0.65
INT_PRESENT_MMOL = 1.0


def fat_bile_factor(intestinal_bile: float) -> float:
    x = min(max(float(intestinal_bile), 0.0) / INT_PRESENT_MMOL, 1.0)
    return BILE_FAT_FLOOR + (1.0 - BILE_FAT_FLOOR) * x


def fat_bile_factor_t(intestinal_bile: torch.Tensor) -> torch.Tensor:
    x = (intestinal_bile.clamp(min=0.0) / INT_PRESENT_MMOL).clamp(max=1.0)
    return BILE_FAT_FLOOR + (1.0 - BILE_FAT_FLOOR) * x
