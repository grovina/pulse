"""
Core abstractions for training signals.

A ``TrainingSignal`` is a self-contained gradient source. The trainer calls
``compute(...)`` once per epoch with a shared ``SignalContext`` (epoch, rng,
optimizer, device, etc.). The signal performs whatever rollouts it needs,
applies its loss(es), calls ``backward`` and ``optimizer.step`` as it sees fit
(per-window, per-epoch, per-spec — the signal decides), and returns a
``SignalResult`` for logging.

This is deliberately thin: signals are not pure functions of state. They own
their internal cadence so each knowledge source can use the gradient pattern
that matches the loss (e.g. trajectory distillation steps per window,
cohort-statistic loss steps once per epoch).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn


@dataclass
class WeightSchedule:
    """Per-epoch weight resolver with optional warmup boundary.

    ``base`` is the steady-state weight. Before ``enable_at_epoch`` the
    resolved weight is 0 (used by the two-phase schedule where coupling /
    verifier / cohort losses are silenced during pure distillation).
    """

    base: float
    enable_at_epoch: int = 0

    def at(self, epoch: int) -> float:
        if self.base <= 0.0:
            return 0.0
        return self.base if epoch >= self.enable_at_epoch else 0.0


@dataclass
class SignalContext:
    """Shared per-epoch state passed to every signal.

    The ``rng`` is shared across signals so the order of consumption is
    deterministic across the epoch (preserves reproducibility under refactor).
    """

    epoch: int
    total_epochs: int
    rng: np.random.Generator
    device: torch.device
    optimizer: torch.optim.Optimizer
    params: list[torch.nn.Parameter]
    grad_clip: float


@dataclass
class SignalResult:
    """Aggregated, detached metrics for one signal in one epoch."""

    loss_sum: float = 0.0
    n_units: int = 0
    sub_metrics: dict[str, float] = field(default_factory=dict)

    @property
    def avg_loss(self) -> float:
        if self.n_units == 0:
            return 0.0
        return self.loss_sum / float(self.n_units)


class TrainingSignal(ABC):
    """Abstract base for all training-loss producers."""

    name: str
    source: str  # human-readable origin (cold-model, literature, etc.)
    category: str  # taxonomy: trajectory | cohort_statistic | mechanism | scenario

    @abstractmethod
    def weight_at(self, epoch: int) -> float:
        """Resolved scalar weight applied to this signal's loss this epoch."""
        ...

    @abstractmethod
    def compute(
        self,
        model: nn.Module,
        embeddings: nn.Embedding,
        ctx: SignalContext,
    ) -> SignalResult:
        """Run the signal: rollouts, losses, backward, optimizer step(s).

        Implementations decide their own batching cadence. Return aggregated
        metrics for logging.
        """
        ...
