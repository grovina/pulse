"""
Embedding-prior signal — shape the patient code so the eval prior is well-specified.

Iter 91. Calibration finds a new person by SOLVING for their embedding from their observations.
That is an inverse problem, and it is underdetermined: a calibration window supplies ~5 markers
x ~10 check-ins ~= 50 numbers, against (before this iter) 64 unknowns. Measured consequence --
the recovered embedding is essentially ORTHOGONAL to the truth (cos ~= 0) while still explaining
the data, and fasting-glucose recovery is worse than simply predicting the population mean.

For an underdetermined inverse problem the correct treatment is a PRIOR. iter 91 re-enables the
calibration prior (benchmark.py, PRIOR_WEIGHT 0 -> 1.0; measured: Gb skill -1.47 -> +0.21) and
halves the number of unknowns (EMBEDDING_DIM 64 -> 32, types.py).

But that prior is a DIAGONAL GAUSSIAN N(prior_mean, prior_std^2) fitted POST-HOC to the trained
embedding table -- and nothing during training ever shaped that table. It is whatever cloud the
reconstruction loss happened to leave behind: measured on iter-90, ||prior_mean|| = 0.26 (a
well-formed code would be centred at 0) with per-dimension scales differing 1.7x. So the eval
prior is an approximation of an arbitrary cloud, when it could be exact.

This signal closes that gap: penalise ||emb||^2 so the learned codes are centred and isotropic.
Then N(0, sigma^2 I) is not a post-hoc fit -- it is the distribution the embeddings were trained
to have, and the calibration prior becomes correct by construction rather than by luck.

It also directly counteracts the failure mode the norm clamp was bolted on to contain (iter 81):
calibration walking the embedding far off the trained manifold, where the ODE detonates. A code
trained to be compact leaves less room to wander.

Deliberately weak. This is a regulariser, not an objective: it must shape the cloud without
flattening the per-patient information the setpoint and meal-response signals put there. If it
is too strong the embeddings collapse toward zero and every patient looks like the population
mean -- which would show up immediately as a rise in setpoint_supervision's per-marker MAE.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .safe_step import accumulate_grad
from .signals import SignalContext, SignalResult, TrainingSignal, WeightSchedule


@dataclass
class EmbeddingPriorSignal(TrainingSignal):
    """L2 toward the origin on the learned patient codes."""

    weight: WeightSchedule = field(default_factory=lambda: WeightSchedule(0.0))

    name: str = "embedding_prior"
    source: str = "Iter 91: calibration is an underdetermined inverse problem; give it a prior"
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

        emb = embeddings.weight                       # [N, EMB] — every patient code
        loss = emb.pow(2).sum(dim=-1).mean()          # mean squared norm

        with torch.no_grad():
            norms = emb.norm(dim=-1)
            sub = {
                "emb_norm_mean": float(norms.mean()),
                "emb_norm_max": float(norms.max()),
                # The eval prior is a diagonal Gaussian; it is only exact if the cloud is
                # centred. Watch this go toward 0.
                "emb_centre_norm": float(emb.mean(dim=0).norm()),
                # ...and isotropic. Watch this go toward 1.
                "emb_std_ratio": float(
                    emb.std(dim=0).max() / emb.std(dim=0).min().clamp_min(1e-6)
                ),
            }

        accumulate_grad(
            w * loss, ctx, signal=self.name,
            extra={"raw_loss": float(loss.detach().item()), "weight": float(w), **sub},
        )
        return SignalResult(loss_sum=float(loss.detach().item()), n_units=1, sub_metrics=sub)
