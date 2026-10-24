"""Measured physiological numbers, learned from the value they were measured at.

PLAN section 2: a number a physiologist could measure is a learnable population
parameter, initialised at the current best value, with nothing pulling it back.
A hub declares each of its numbers with the study whose protocol identifies it,
and the module that owns the law holds them as ``PublishedNumbers``. The
identifiability report names an open direction with that study, so the map of
holes is also the reading list.

The teacher keeps the published values as plain constants: it is the generator
the student is checked against, not a model the evidence moves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn


@dataclass(frozen=True)
class Published:
    """One number: today's best value and the study that would move it.

    A fraction lives in (0, 1) and is learned in logit space; any other number
    is positive and is learned in log space, so no step leaves its domain.
    """

    value: float
    study: str
    fraction: bool = False

    def __post_init__(self) -> None:
        if self.fraction and not 0.0 < self.value < 1.0:
            raise ValueError(f"a fraction must lie in (0, 1), got {self.value}")
        if not self.fraction and self.value <= 0.0:
            raise ValueError(f"a positive number must exceed 0, got {self.value}")

    @property
    def raw(self) -> float:
        p = self.value
        return math.log(p / (1.0 - p)) if self.fraction else math.log(p)


def initial_values(numbers: Mapping[str, Published]) -> dict[str, torch.Tensor]:
    """The numbers at their published values, as the student starts them."""
    return {name: torch.tensor(float(p.value)) for name, p in numbers.items()}


class PublishedNumbers(nn.Module):
    """The learnable numbers of one law. ``forward()`` returns them by name."""

    def __init__(self, numbers: Mapping[str, Published]) -> None:
        super().__init__()
        self._names = tuple(numbers)
        self._fractions = frozenset(n for n, p in numbers.items() if p.fraction)
        for name, p in numbers.items():
            self.register_parameter(self._raw_name(name), nn.Parameter(torch.tensor(p.raw)))
        self.identified_by = {self._raw_name(n): p.study for n, p in numbers.items()}

    def _raw_name(self, name: str) -> str:
        return f"logit_{name}" if name in self._fractions else f"log_{name}"

    def forward(self) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for name in self._names:
            raw = getattr(self, self._raw_name(name))
            out[name] = torch.sigmoid(raw) if name in self._fractions else raw.exp()
        return out
