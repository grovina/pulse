"""
Dose-response training signal — multi-marker amplitude axis.

Per epoch: sample K patient embeddings, run the dose-response protocol for
each, compute per-marker loss (slope-vs-literature for glucose, ranking
hinge for insulin / GLP-1 — see ``pulse.dose_response`` for the rationale)
and aggregate. One backward + step.

This is a sibling of ``CohortStatisticSignal`` but with its own dedicated
weight and a differentiable soft-peak per marker.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import torch
from torch import nn

from ..dose_response import (
    DoseResponseProtocol,
    cold_initial_state,
    dose_response_epoch_loss,
)
from .embedding_sampler import select_supervised_embeddings
from .safe_step import accumulate_grad
from .signals import SignalContext, SignalResult, TrainingSignal, WeightSchedule


def perturb_dose_response_protocol(
    protocol: DoseResponseProtocol, rng: np.random.Generator,
    *, dose_frac: float = 0.20, time_min: int = 30, start_hour: float = 1.0,
) -> DoseResponseProtocol:
    """A perturbed copy of the dose-response protocol (review 4.10).

    One dose scale for the whole ladder (the ranking / slope targets need the
    doses to stay ordered), a meal-offset shift that keeps the pre-meal
    baseline and the post window inside the rollout, and a start-hour shift.
    """
    scale = 1.0 + float(rng.uniform(-dose_frac, dose_frac))
    doses = tuple(float(d) * scale for d in protocol.carb_doses_g)
    lo = int(protocol.pre_window)
    hi = int(protocol.duration_min - protocol.post_window)
    offset = int(protocol.meal_offset_min + rng.integers(-time_min, time_min + 1))
    offset = int(min(max(offset, lo), hi))
    sh = (float(protocol.start_hour) + float(rng.uniform(-start_hour, start_hour))) % 24.0
    return replace(protocol, carb_doses_g=doses, meal_offset_min=offset, start_hour=sh)


@dataclass
class DoseResponseSignal(TrainingSignal):
    """Dedicated per-epoch supervision of the carb→peak-marker dose response.

    Always supervises the zero ("default") embedding alongside randomly sampled
    patient embeddings — this is the embedding the benchmark uses for textbook
    scenarios, so training distribution must include it explicitly to get the
    dose-response gradient onto the model parameters that actually serve it.
    """

    n_patients: int = 0
    sample_patients: int = 4
    include_default_embedding: bool = True
    weight: WeightSchedule = field(default_factory=lambda: WeightSchedule(0.0))
    protocol: DoseResponseProtocol = field(default_factory=DoseResponseProtocol)

    # Iter 97 (review 4.10): the protocol IS the textbook meal_dose_response
    # scenario (30/90 g at 08:30). Perturb it per compute — doses +/-20 %, meal
    # time +/-30 min, start hour +/-1 h — and keep the fixed protocol as one
    # sample in ``1/perturb_fixed_prob``. The peak targets are per-gram lines
    # through the origin, so a perturbed dose carries its own target.
    perturb_protocols: bool = False
    perturb_fixed_prob: float = 0.25

    name: str = "dose_response"
    source: str = "Wolever (1991, 1996) — glycemic response to carb dose"
    category: str = "dose_response"

    def weight_at(self, epoch: int) -> float:
        return self.weight.at(epoch)

    def protocol_for_step(self, rng: np.random.Generator) -> DoseResponseProtocol:
        """The (possibly perturbed) protocol for one compute call."""
        if not self.perturb_protocols or rng.random() < self.perturb_fixed_prob:
            return self.protocol
        return perturb_dose_response_protocol(self.protocol, rng)

    def compute(
        self,
        model: nn.Module,
        embeddings: nn.Embedding,
        ctx: SignalContext,
    ) -> SignalResult:
        w = self.weight_at(ctx.epoch)
        if w <= 0:
            return SignalResult()

        emb_list = select_supervised_embeddings(
            embeddings=embeddings,
            n_patients=self.n_patients,
            sample_patients=self.sample_patients,
            rng=ctx.rng,
            device=ctx.device,
            include_default=self.include_default_embedding,
        )
        if not emb_list:
            return SignalResult()

        protocol = self.protocol_for_step(ctx.rng)
        initial = cold_initial_state(protocol, rng=ctx.rng, device=ctx.device)
        loss, diagnostics = dose_response_epoch_loss(
            model=model,
            embeddings_to_supervise=emb_list,
            protocol=protocol,
            initial_state=initial,
            device=ctx.device,
        )
        diagnostics = dict(diagnostics)
        diagnostics["perturbed"] = 0.0 if protocol is self.protocol else 1.0
        # Flatten diagnostics into the safe_step extra dict (all floats).
        extra: dict[str, float] = {
            "raw_loss": float(loss.detach().item()),
            "weight": float(w),
            "target_slope": float(protocol.target_slope),
            "n_emb": float(len(emb_list)),
        }
        extra.update(diagnostics)
        accumulate_grad(w * loss, ctx, signal=self.name, extra=extra)
        return SignalResult(
            loss_sum=float(loss.detach().item()),
            n_units=1,
            sub_metrics=diagnostics,
        )
