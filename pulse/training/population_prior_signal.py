"""Prior on the population constants that used to be pins.

A pin cannot be moved by a later measurement. A prior can: the gradient is
zero at the mode and restoring away from it, so a direction the loss does
not identify stays put.

The widths are the ones the teacher already draws, written next to the
constants in ``modules/metabolic.py``. Nothing here is a new sigma.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from ..modules.metabolic import (
    _GN_CENTER,
    _GN_LOG_SIGMA,
    _K_FFA,
    _K_FFA_LOG_SIGMA,
    _K_II,
    _K_II_LOG_SIGMA,
)
from .safe_step import accumulate_grad
from .signals import SignalContext, SignalResult, TrainingSignal, WeightSchedule


def _lognormal_nll(log_x: torch.Tensor, log_median: float, log_sigma: float) -> torch.Tensor:
    return (log_x - log_median).pow(2) / (2.0 * log_sigma * log_sigma)


def population_prior_nll(met: nn.Module) -> torch.Tensor:
    """Negative log density of ``k_ii``, ``k_ffa`` and glucagon basal.

    ``f_gng`` is not here. Landau's fasting trajectory is the slice that
    holds it, and a prior on top of that slice would pull the fraction back
    off the measurement.
    """
    k_term = _lognormal_nll(met.log_k_ii, math.log(_K_II), _K_II_LOG_SIGMA)
    ffa_term = _lognormal_nll(met.log_k_ffa, math.log(_K_FFA), _K_FFA_LOG_SIGMA)
    gn_term = _lognormal_nll(met.log_gn_b, math.log(_GN_CENTER), _GN_LOG_SIGMA)
    return k_term + ffa_term + gn_term


@dataclass
class PopulationPriorSignal(TrainingSignal):
    """Holds the population constants that still have no slice at their citation mode."""

    weight: WeightSchedule = field(default_factory=lambda: WeightSchedule(1.0))

    name: str = "population_prior"
    source: str = "M1.5: priors on k_ii, k_ffa and glucagon basal until their slices exist"
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
        nll = population_prior_nll(model.metabolic)
        met = model.metabolic
        with torch.no_grad():
            sub = {
                "k_ii": float(met.k_ii()),
                "f_gng": float(met.f_gng_value()),
                "k_ffa": float(met.k_ffa_value()),
                "gn_b": float(met.log_gn_b.exp()),
                "nll": float(nll),
            }
        accumulate_grad(w * nll, ctx, signal=self.name, extra={"weight": float(w), **sub})
        return SignalResult(loss_sum=float(nll.detach()), n_units=1, sub_metrics=sub)
