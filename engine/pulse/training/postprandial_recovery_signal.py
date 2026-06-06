"""
Postprandial recovery signal — penalizes residual glucose elevation
~7h after a standardized meal.

Iter 32 hypothesis: across iters 27-31 the bench check
``glucose_returns_near_baseline`` (dietary_carbohydrate_meal_flow) has
been failing with residuals 60-70 mg/dL above baseline at minutes
420-475, while the closest training signal (fasting-stability) trains
a *no-meal* rollout in a disjoint dynamical regime. The model has
never received gradient on post-meal long-horizon recovery —
exactly the failure mode the bench probes.

This signal mirrors the bench's flow-story protocol exactly:
- start at 08:00 (start_time_minutes=480) from NORM_CENTER, zero embedding
- single mixed meal at minute 30: 50g carbs / 12g fats / 18g proteins
- duration 480 min
- baseline = mean glucose over minutes [0, 25)
- recovery = mean glucose over minutes [420, 475)
- loss = ((recovery - baseline) / glc_scale) ** 2

Because the signal *replaces* fasting-stability in iter 32 (per the
iter32-review.md plan), the no-meal resting-equilibrium path is no
longer trained — but `cohort-statistic-weight`, the trajectory MSE
on long zero-embedding rollouts (`n-default-patients`), and the
fasting-resilience checks in `overnight_fast` continue to constrain
the resting setpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from ..model import integrate, precompute_gut_outputs
from ..modules.gut import MealEvent
from ..types import EMBEDDING_DIM, MARKER_INDEX, NORM_CENTER, NORM_SCALE
from .safe_step import safe_step
from .signals import SignalContext, SignalResult, TrainingSignal, WeightSchedule

# Match flow_story_protocol.py exactly so training and bench measure the same thing.
_DURATION_MIN = 480
_START_HOUR = 8.0
_MEAL = MealEvent(time=30.0, carbs=50.0, fats=12.0, proteins=18.0)
_BASELINE_END = 25
_RECOVERY_START = 420
_RECOVERY_END = 475


@dataclass
class PostprandialRecoverySignal(TrainingSignal):
    """Penalises residual glucose elevation at +7h after a standard meal.

    One forward pass per epoch: precompute gut outputs at the zero embedding,
    integrate from ``NORM_CENTER`` for 480 min, MSE the difference between
    mean glucose in the recovery window and the pre-meal baseline.
    """

    weight: WeightSchedule = field(default_factory=lambda: WeightSchedule(0.0))

    name: str = "postprandial_recovery"
    source: str = "Pulse flow-story dietary-carbohydrate scenario (recovery window)"
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
        start_min = _START_HOUR * 60.0

        # precompute_gut_outputs is the canonical path even for unbatched
        # rollouts — keeps the gut kernel call out of the per-step loop.
        gut = precompute_gut_outputs(
            model, embedding, _DURATION_MIN,
            dt=1.0, start_time_minutes=start_min, meals=[_MEAL],
        )
        pred = integrate(
            model, initial, embedding, _DURATION_MIN,
            dt=1.0, start_time_minutes=start_min, meals=[_MEAL],
            gut_outputs=gut,
        )

        baseline = pred[:_BASELINE_END, glc_idx].mean()
        recovery = pred[_RECOVERY_START:_RECOVERY_END, glc_idx].mean()
        residual_norm = (recovery - baseline) / glc_scale
        loss = residual_norm.pow(2)

        safe_step(
            w * loss,
            ctx,
            signal=self.name,
            extra={
                "raw_loss": float(loss.detach().item()),
                "weight": float(w),
                "baseline_mg_dl": float(baseline.detach().item()),
                "recovery_mg_dl": float(recovery.detach().item()),
                "residual_mg_dl": float((recovery - baseline).detach().item()),
            },
        )

        return SignalResult(
            loss_sum=float(loss.detach().item()),
            n_units=1,
            sub_metrics={
                "residual_mg_dl": float((recovery - baseline).detach().item()),
                "baseline_mg_dl": float(baseline.detach().item()),
                "recovery_mg_dl": float(recovery.detach().item()),
            },
        )
