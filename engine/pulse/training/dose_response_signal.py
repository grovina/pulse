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

from dataclasses import dataclass, field

import torch
from torch import nn

from ..dose_response import (
    DoseResponseProtocol,
    cold_initial_state,
    dose_response_epoch_loss,
)
from .embedding_sampler import select_supervised_embeddings
from .safe_step import safe_step
from .signals import SignalContext, SignalResult, TrainingSignal, WeightSchedule


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

    name: str = "dose_response"
    source: str = "Wolever (1991, 1996) — glycemic response to carb dose"
    category: str = "dose_response"

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

        initial = cold_initial_state(self.protocol, rng=ctx.rng, device=ctx.device)
        loss, diagnostics = dose_response_epoch_loss(
            model=model,
            embeddings_to_supervise=emb_list,
            protocol=self.protocol,
            initial_state=initial,
            device=ctx.device,
        )
        # Flatten diagnostics into the safe_step extra dict (all floats).
        extra: dict[str, float] = {
            "raw_loss": float(loss.detach().item()),
            "weight": float(w),
            "target_slope": float(self.protocol.target_slope),
            "n_emb": float(len(emb_list)),
        }
        extra.update(diagnostics)
        safe_step(w * loss, ctx, signal=self.name, extra=extra)
        return SignalResult(
            loss_sum=float(loss.detach().item()),
            n_units=1,
            sub_metrics=diagnostics,
        )
