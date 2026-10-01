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

from ..modules.gut import MealEvent
from ..rollouts import RolloutRequest, rollout_many
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
    ) -> list[list[tuple[torch.Tensor, torch.Tensor]]]:
        """Net Δ(liver+muscle glycogen) across the fed window for every
        (embedding, dose): ``out[b][d] = (delta, pre)`` in grams — the post-minus-
        pre change and the pre-meal baseline. Every pair is one row of ONE batched
        rollout (``rollouts.rollout_many``)."""
        B = int(embeddings.shape[0])
        initial = torch.tensor(NORM_CENTER, dtype=torch.float32, device=device)
        requests = [
            RolloutRequest(
                duration_min=self.window_min,
                start_minutes=self.start_hour * 60.0,
                meals=[MealEvent(
                    time=self.meal_time_min, carbs=float(dose_g),
                    fats=float(self.fats_g), proteins=float(self.proteins_g),
                )],
                embeddings=embeddings,
                states=initial.unsqueeze(0).expand(B, -1),
            )
            for dose_g in self.carb_doses_g
        ]
        trajs = rollout_many(model, requests)  # per dose: [B, T, STATE]
        # Pre-meal baseline (before the meal lands) vs the last hour of the
        # window (storage has had the whole window to accumulate).
        pre_end = max(1, int(self.meal_time_min))
        out: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
        for b in range(B):
            row = []
            for traj in trajs:
                total = traj[b, :, _LIVER] + traj[b, :, _MUSCLE]
                pre = total[:pre_end].mean()
                row.append((total[-60:].mean() - pre, pre))
            out.append(row)
        return out

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
        over_terms: list[torch.Tensor] = []
        under_terms: list[torch.Tensor] = []
        deltas: list[float] = []
        pairs = self._glycogen_deltas(model, torch.stack(emb_list, dim=0), device)
        for row in pairs:
            for dose, (delta, _pre) in zip(self.carb_doses_g, row):
                # Normalise the violation by the dose so a 90 g and a 30 g
                # meal contribute comparable gradient when equally violated.
                norm = max(float(dose), 1.0)
                over = torch.relu(delta - float(dose)) / norm   # stored > eaten
                under = torch.relu(-delta) / norm               # depleted while fed
                over_terms.append(over.pow(2))
                under_terms.append(under.pow(2))
                deltas.append(float(delta.detach().item()))

        over_loss = torch.stack(over_terms).mean()
        under_loss = torch.stack(under_terms).mean()
        loss = over_loss + under_loss

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
