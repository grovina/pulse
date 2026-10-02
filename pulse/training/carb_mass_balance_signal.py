"""
Carbohydrate mass-balance signal — cross-module conservation (iter 74).

Every other signal supervises markers *locally* (a trajectory shape, a
dose-response amplitude, a kernel curve). None enforces a conservation law
that *spans* modules: the gut emits a carbohydrate load, and the metabolic
glycogen pools store some of it — but nothing ties the two together, so the
model can refill glycogen from nowhere or deplete it while being fed and pay
no penalty. The gut-boundary budget (carb in ⇒ glucose appearance) is already
pinned by ``gut_dose_sweep``'s AUC term; this signal closes the *downstream*
half: carb in ⇒ glycogen storage.

It is deliberately formulated as **conservation inequalities**, not target
deltas, so it cannot fight real physiology — it only fires on a frank
violation of mass balance:

  * **Storage ceiling** — over a fed window you cannot store more glycogen
    than the carbohydrate mass you ingested:
        Δ(liver_glycogen + muscle_glycogen) ≤ dose_g
    (absorbed ≤ ingested and stored ≤ absorbed, so stored ≤ ingested — a
    hard physical truth independent of any rate constant).
  * **Direction floor** — eating carbohydrate from a fasted state refills
    glycogen; the pools should not net-*deplete* across the fed window:
        Δ(liver_glycogen + muscle_glycogen) ≥ 0

Both bounds are zero-gradient when satisfied, so a model whose glycogen
dynamics are already physical sees nothing. The pools are otherwise among the
most gradient-starved states in the model (cohort window-means only), so when
the bounds *do* bind they supply a rare absolute-mass gradient in the
physically-correct direction.

Off by default (``weight=0``): glycogen dynamics are slow (liver τ ≈ 1 day),
the storage ceiling rarely binds over a few-hour window, and the signal is new
— enable it deliberately via ``--carb-mass-balance-weight``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from ..model import integrate, precompute_gut_outputs
from ..modules.gut import MealEvent
from ..types import MARKER_INDEX, NORM_CENTER
from .embedding_sampler import select_supervised_embeddings
from .safe_step import accumulate_grad
from .signals import SignalContext, SignalResult, TrainingSignal, WeightSchedule

_LIVER = MARKER_INDEX["liver_glycogen"]
_MUSCLE = MARKER_INDEX["muscle_glycogen"]


@dataclass
class CarbMassBalanceSignal(TrainingSignal):
    """Cross-module carb→glycogen conservation, as mass-balance inequalities.

    For each carb dose, run a fasted→fed rollout, measure the net change in
    total glycogen (liver + muscle) across the window, and hinge-penalise the
    two physical violations: storing more than was ingested, or net-depleting
    while fed. Supervises the zero embedding (textbook scenarios query it) plus
    a few sampled patients.
    """

    weight: WeightSchedule = field(default_factory=lambda: WeightSchedule(0.0))
    carb_doses_g: tuple[float, ...] = (30.0, 60.0, 90.0)
    fats_g: float = 5.0
    proteins_g: float = 10.0
    window_min: int = 240
    meal_time_min: float = 30.0
    start_hour: float = 8.0
    n_patients: int = 0
    sample_patients: int = 2
    include_default_embedding: bool = True

    name: str = "carb_mass_balance"
    source: str = "physiology — carbohydrate mass conservation (gut → glycogen)"
    category: str = "mechanism"

    def weight_at(self, epoch: int) -> float:
        return self.weight.at(epoch)

    def _glycogen_deltas(
        self, model: nn.Module, embeddings: torch.Tensor, device: torch.device,
    ) -> torch.Tensor:
        """Net Δ(liver+muscle glycogen) across the fed window, in grams, for every
        (embedding, dose) pair — row ``e·D + d`` — from ONE batched rollout (each row
        carries its own meal).
        """
        doses = list(self.carb_doses_g)
        n_rows = int(embeddings.shape[0]) * len(doses)
        initial = torch.tensor(NORM_CENTER, dtype=torch.float32, device=device).expand(n_rows, -1)
        meals = [
            [MealEvent(time=self.meal_time_min, carbs=float(dose_g),
                       fats=float(self.fats_g), proteins=float(self.proteins_g))]
            for _ in range(int(embeddings.shape[0])) for dose_g in doses
        ]
        pred = integrate(
            model, initial, embeddings.repeat_interleave(len(doses), dim=0), self.window_min,
            dt=1.0, start_time_minutes=self.start_hour * 60.0, meals=meals,
        )
        total = pred[..., _LIVER] + pred[..., _MUSCLE]
        # Pre-meal baseline (before the meal lands) vs the last hour of the
        # window (storage has had the whole window to accumulate).
        pre_end = max(1, int(self.meal_time_min))
        return total[:, -60:].mean(dim=1) - total[:, :pre_end].mean(dim=1)

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

        device = ctx.device
        delta = self._glycogen_deltas(model, torch.stack(emb_list, dim=0), device)
        dose = torch.tensor(
            [float(d) for _ in emb_list for d in self.carb_doses_g],
            dtype=delta.dtype, device=device,
        )
        # Normalise the violation by the dose so a 90 g and a 30 g meal contribute
        # comparable gradient when equally violated.
        norm = dose.clamp(min=1.0)
        over_loss = (torch.relu(delta - dose) / norm).pow(2).mean()   # stored > eaten
        under_loss = (torch.relu(-delta) / norm).pow(2).mean()        # depleted while fed
        loss = over_loss + under_loss
        deltas = delta.detach().tolist()

        n_pairs = len(emb_list) * len(self.carb_doses_g)
        accumulate_grad(
            w * loss,
            ctx,
            signal=self.name,
            extra={
                "raw_loss": float(loss.detach().item()),
                "weight": float(w),
                "over_store": float(over_loss.detach().item()),
                "depletion": float(under_loss.detach().item()),
                "mean_delta_g": float(np.mean(deltas)) if deltas else 0.0,
                "n_dose_emb_pairs": float(n_pairs),
            },
        )

        return SignalResult(
            loss_sum=float(loss.detach().item()),
            n_units=1,
            sub_metrics={
                "over_store": float(over_loss.detach().item()),
                "depletion": float(under_loss.detach().item()),
                "mean_delta_g": float(np.mean(deltas)) if deltas else 0.0,
                "n_dose_emb_pairs": float(n_pairs),
            },
        )
