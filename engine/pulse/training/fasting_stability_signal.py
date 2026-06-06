"""
Fasting stability signal — penalizes glucose drift from the resting state.

Iter 26 diagnostic: the model drifts +57 mg/dL over 4h with no meals at
the zero embedding. Root cause: the neural metabolic module has no explicit
setpoint mechanism — unlike the knowledge model's
``dG = -(Sg + X)*(G - Gb) + Ra``, learned production and consumption
heads can settle at a resting rate that is production > clearance.

This signal fills that gap by running a short fasting rollout from
NORM_CENTER at the zero ("default") embedding and penalising the mean
squared drift of glucose from its starting value. It gives the model a
direct gradient to keep glucose near its resting setpoint when no food
is arriving — the exact condition the benchmark probe tests.

Design notes:
- Zero embedding only: the benchmark textbook scenarios all query at the
  zero embedding, so the fix must land there. Patient embeddings have their
  own resting points that the trajectory signal supervises.
- Enabled from epoch 0 (no warmup): glucose homeostasis is a prerequisite
  for correct meal-response dynamics, not an additive loss that should wait
  until Phase 2.
- window_min=120: 2h is long enough to produce meaningful drift gradient
  at the observed 14 mg/dL/h rate, and short enough to be cheap (no gut
  precompute needed — meals=[] so the gut kernel is never called).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from ..model import integrate
from ..types import EMBEDDING_DIM, MARKER_INDEX, NORM_CENTER, NORM_SCALE
from .safe_step import safe_step
from .signals import SignalContext, SignalResult, TrainingSignal, WeightSchedule


@dataclass
class FastingStabilitySignal(TrainingSignal):
    """Penalizes glucose drift during a no-meal rollout from the resting state.

    One forward pass per epoch: integrate for ``window_min`` steps from
    ``NORM_CENTER`` with no meals at the zero embedding, then MSE the
    glucose deviation from its initial value.
    """

    window_min: int = 120
    weight: WeightSchedule = field(default_factory=lambda: WeightSchedule(0.0))
    start_time_minutes: float = 480.0  # 8 am — typical post-wake resting state

    name: str = "fasting_stability"
    source: str = "physiology — resting glucose homeostasis"
    category: str = "mechanism"

    def weight_at(self, epoch: int) -> float:
        return self.weight.at(epoch)

    def compute(
        self,
        model: nn.Module,
        embeddings: nn.Embedding,
        ctx: SignalContext,
    ) -> SignalResult:
        w = self.weight_at(ctx.epoch)
        if w <= 0:
            return SignalResult()

        device = ctx.device
        embedding = torch.zeros(EMBEDDING_DIM, device=device)
        initial = torch.tensor(NORM_CENTER, dtype=torch.float32, device=device)
        glc_idx = MARKER_INDEX["glucose"]
        glc_scale = float(NORM_SCALE[glc_idx])

        pred = integrate(
            model, initial, embedding, self.window_min,
            dt=1.0, start_time_minutes=self.start_time_minutes, meals=[],
        )

        drift = ((pred[:, glc_idx] - initial[glc_idx]) / glc_scale).pow(2).mean()

        safe_step(
            w * drift,
            ctx,
            signal=self.name,
            extra={
                "raw_loss": float(drift.detach().item()),
                "weight": float(w),
            },
        )

        return SignalResult(
            loss_sum=float(drift.detach().item()),
            n_units=1,
            sub_metrics={"glucose_drift_norm": float(drift.detach().item())},
        )
