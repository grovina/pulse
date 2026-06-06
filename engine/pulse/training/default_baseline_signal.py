"""
Default-baseline signal — pulls the trained model's zero-embedding output
toward population-typical baselines for selected markers.

Iter 36 calibration investigation
(apps/pulse/docs/iter36-calibration-investigation.md) showed that
hr_mape ≈ 0.24 across iters 32-35 was a constant +15 bpm bias on every
prediction: the trained model's "default patient" (zero embedding,
fasting initial state) settled at ~88 bpm vs the bench eval mean of
~63 bpm. Calibration removed ~10 bpm but plateaued — the loss landscape
through 12h of rate-of-change integration is shallow on the HR-baseline
axis, and adding architectural shortcuts (iter 36) corrupted the
coupling pathway.

This signal is the simplest direct fix: at every epoch, run a fasted
no-meal rollout at zero embedding and penalise deviation of the
mean-over-window output from ``NORM_CENTER`` for the requested markers.
This does not add capacity; it only constrains the trained
default-patient's resting equilibrium toward the population norm so
calibration starts from a saner basin.

Default scope is HR only (the marker that motivated the investigation).
NORM_CENTER[hr] = 70 bpm; bench eval mean is ~63 bpm; the resulting
floor on hr_mape is ~7 / 63 ≈ 0.11 — under the 0.15 gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
import torch.nn as nn

from ..model import integrate, precompute_gut_outputs
from ..types import EMBEDDING_DIM, MARKER_INDEX, NORM_CENTER, NORM_SCALE
from .safe_step import safe_step
from .signals import SignalContext, SignalResult, TrainingSignal, WeightSchedule

# Default rollout window: 4h fasted morning, the regime the bench eval
# mostly samples. Long enough that any coupling-driven drift away from
# population norm has time to manifest; short enough that one rollout
# per epoch stays cheap.
_DURATION_MIN = 240
_START_HOUR = 7.0
# Average over the second half so we don't penalise the cold-start
# transient at t=0; the model gets a window to settle into its trained
# equilibrium.
_AVG_START = 60
_AVG_END = 240


@dataclass
class DefaultBaselineSignal(TrainingSignal):
    """Constrain zero-embedding fasted mean output toward NORM_CENTER.

    For each marker in ``markers``, run a single fasted rollout at zero
    embedding, mean the output over the configured window, and apply
    Gaussian discrepancy in normalized units against ``NORM_CENTER``.
    """

    weight: WeightSchedule = field(default_factory=lambda: WeightSchedule(0.0))
    markers: Sequence[str] = field(default_factory=lambda: ("hr",))

    name: str = "default_baseline"
    source: str = "Iter 36 calibration investigation: zero-embedding default-patient anchor"
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
        start_min = _START_HOUR * 60.0

        # Single fasted, no-meal rollout. ``precompute_gut_outputs`` for the
        # empty meal list is essentially free but keeps the integrate API
        # consistent with how every other signal calls it.
        gut = precompute_gut_outputs(
            model, embedding, _DURATION_MIN,
            dt=1.0, start_time_minutes=start_min, meals=[],
        )
        pred = integrate(
            model, initial, embedding, _DURATION_MIN,
            dt=1.0, start_time_minutes=start_min, meals=[],
            gut_outputs=gut,
        )

        loss_terms: list[torch.Tensor] = []
        sub_metrics: dict[str, float] = {}
        for marker in self.markers:
            idx = MARKER_INDEX.get(marker)
            if idx is None:
                continue
            target = NORM_CENTER[idx]
            scale = float(NORM_SCALE[idx])
            mean_pred = pred[_AVG_START:_AVG_END, idx].mean()
            residual = (mean_pred - target) / scale
            loss_terms.append(residual.pow(2))
            sub_metrics[f"{marker}_mean"] = float(mean_pred.detach().item())
            sub_metrics[f"{marker}_residual"] = float((mean_pred - target).detach().item())

        if not loss_terms:
            return SignalResult()

        # Mean across markers so the contribution of this signal scales
        # with `weight` rather than with how many markers we picked.
        loss = torch.stack(loss_terms).mean()

        safe_step(
            w * loss,
            ctx,
            signal=self.name,
            extra={
                "raw_loss": float(loss.detach().item()),
                "weight": float(w),
                **sub_metrics,
            },
        )

        return SignalResult(
            loss_sum=float(loss.detach().item()),
            n_units=1,
            sub_metrics=sub_metrics,
        )
