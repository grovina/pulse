"""
Gut dose-sweep distillation signal.

The gut module is a stateless absorption kernel: given (carbs, fats, proteins,
time-since-meal, embedding), it outputs a 4-dim appearance vector. Its
correctness is independent of any downstream ODE — and so is its training.

Iter 11 diagnostics showed that at the zero ("default") embedding the gut
kernel had drifted into a pathological regime — outputs *decreased* with
increasing carb dose, and even produced large appearance for a 0 g carb
"meal" (driven entirely by fats/proteins/embedding bias). Every other
training signal nominally supervised the gut, but only indirectly:

  * trajectory MSE supervised the integrated state, not the kernel
  * the per-patient gut MSE only saw whatever doses the cold-model meal plans
    happened to sample (a narrow window centered on ≈ 60 g carbs)
  * dose-response saw the integrated glucose curve, which the model could
    "satisfy" via baseline drift instead of through the gut

This signal closes that gap: it directly supervises ``model.gut.forward_window``
against the analytical cold-model absorption profile across an explicit
dose sweep, at the zero embedding (and a few sampled patients). One
vectorized kernel call per (embedding, dose) pair — cheap, focused,
gradient lands exactly on the gut pipeline (kernel + gut embedding projection).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import numpy as np
import torch
import torch.nn as nn

from ..knowledge.full_body import PatientParams, compute_absorption_profile
from ..model import ModularPhysiologyNetwork
from ..modules.gut import GUT_OUTPUT_SCALE, MealEvent
from ..types import GUT_OUTPUT_DIM
from .embedding_sampler import select_supervised_embeddings
from .safe_step import safe_step
from .signals import SignalContext, SignalResult, TrainingSignal, WeightSchedule


@dataclass(frozen=True)
class GutDoseSweepProtocol:
    """Defines the (dose × time) grid the gut kernel is supervised on.

    Spans 0 g (which exercises the "no carbs ⇒ no glucose appearance"
    constraint) up through 120 g (well above the largest training meal),
    giving the kernel a wide carb axis to learn the dose response over.
    Fats and proteins are held constant at the same ratios as the
    dose-response protocol so the cohort/dose-response signals see the
    same meal shape during their own rollouts.

    ``rank_margin`` (iter 15) controls the dose-monotonicity constraint:
    the predicted AUC gap between any pair of doses ``(d_i < d_j)`` must
    be at least ``rank_margin × (cold_target_AUC(d_j) − cold_target_AUC(d_i))``
    on every channel where the cold target itself ranks the doses (i.e.
    where the target AUC strictly increases with dose). Iter 13/14 showed
    that pure per-element MSE admits low-gradient minima where the kernel
    is dose-inverted (peak at 30 g, decreasing after); the ranking term
    makes those minima infeasible by construction, regardless of the
    per-channel ``abs_scale``.
    """

    carb_doses_g: tuple[float, ...] = (0.0, 15.0, 30.0, 45.0, 60.0, 90.0, 120.0)
    fats_g: float = 5.0
    proteins_g: float = 10.0
    post_window_min: int = 240
    rank_margin: float = 0.3


def _cold_target_for_dose(
    dose_g: float,
    fats_g: float,
    proteins_g: float,
    n_steps: int,
    params: PatientParams,
) -> np.ndarray:
    """Analytical cold-model absorption profile for one (dose, fats, prot) meal
    over ``n_steps`` minutes, meal at t=0.

    Returns ``(n_steps, GUT_OUTPUT_DIM)`` matching ``GutModule.forward_window``.
    """
    out = np.zeros((n_steps, GUT_OUTPUT_DIM), dtype=np.float32)
    meals = [(0.0, float(dose_g), float(fats_g), float(proteins_g))]
    for t in range(n_steps):
        out[t] = compute_absorption_profile(float(t), meals, params)
    return out


@dataclass
class GutDoseSweepSignal(TrainingSignal):
    """Direct distillation of the gut kernel against the cold-model dose curve.

    Pre-computes cold-model absorption profiles for every dose in the
    protocol once at construction (numpy / closed-form, fast). Per epoch:
    for each (embedding, dose) pair, run ``model.gut.forward_window``
    against the matching cold target and MSE-step the gut kernel +
    embedding projection.

    Always supervises the zero embedding (the benchmark uses it) plus a
    random subset of patient embeddings so the kernel learns the *shape*
    of the dose response across the embedding manifold, not just at zero.
    """

    n_patients: int = 0
    sample_patients: int = 4
    include_default_embedding: bool = True
    weight: WeightSchedule = field(default_factory=lambda: WeightSchedule(0.0))
    protocol: GutDoseSweepProtocol = field(default_factory=GutDoseSweepProtocol)
    ranking_weight: float = 1.0
    # AUC-matching term weight (iter 18). Per-element MSE on a [B, T, C]
    # window is dominated by mass averaging — most time-steps and channels
    # have ~0 cold-target output, so a uniform multiplicative over-amp on
    # the active region gets diluted by the inactive region. The AUC term
    # collapses time and channel into a single scalar per (dose, batch)
    # before squaring, so a 50% over-amp at any dose contributes loss
    # proportional to that dose's target AUC squared — undiluted.
    # Iter 16/17 showed the kernel ending at uniform 1.5× cold-target
    # amplitude despite gut_dose_sweep raw_loss=0.05; the per-element MSE
    # was simply too dilute to push amplitude back. The AUC term targets
    # exactly that pathology.
    auc_weight: float = 1.0

    name: str = "gut_dose_sweep"
    source: str = "cold_model gut kernel (compute_absorption_profile)"
    category: str = "gut"

    def __post_init__(self) -> None:
        params = PatientParams()
        targets = np.stack([
            _cold_target_for_dose(
                d, self.protocol.fats_g, self.protocol.proteins_g,
                self.protocol.post_window_min, params,
            )
            for d in self.protocol.carb_doses_g
        ])  # [D, T, GUT_OUTPUT_DIM]
        self._targets = torch.tensor(targets, dtype=torch.float32)

        # Per-channel scales — single source of truth in modules.gut so this
        # signal and TrajectoryRolloutSignal supervise the kernel on the same
        # error magnitudes.
        self._abs_scale = torch.tensor(GUT_OUTPUT_SCALE, dtype=torch.float32)

        # Pairwise cold-target AUC gaps and the dose-ordering mask. For each
        # ordered pair (i, j) and channel c, we want
        #   pred_AUC(d_j, c) − pred_AUC(d_i, c) ≥ rank_margin × tgt_gap[i, j, c]
        # whenever ``tgt_gap > 0``. Channels whose targets don't actually
        # rank the swept dose (lipid/amino under fixed fats/proteins) are
        # masked out — the constraint applies only where the cold model
        # itself says "more dose ⇒ more AUC".
        tgt_aucs = self._targets.sum(dim=1)                              # [D, C]
        tgt_gap = tgt_aucs.unsqueeze(0) - tgt_aucs.unsqueeze(1)          # [D, D, C], gap[i,j] = j − i
        self._rank_target_gap = tgt_gap
        self._rank_mask = (tgt_gap > 0).to(torch.float32)

        # Cached cold-target AUC and its normalization scale, used by the
        # AUC-matching term. The scale is the typical "AUC magnitude"
        # under a kernel-shaped profile: peak ≈ abs_scale, time-integral
        # ≈ peak · T / 4 (triangular-ish profile across the window).
        # Dividing the squared error by this scale keeps the per-channel
        # contributions comparable across channels with different units.
        self._auc_targets = tgt_aucs                                     # [D, C]
        self._auc_scale = self._abs_scale * float(self.protocol.post_window_min) / 4.0  # [C]

    def weight_at(self, epoch: int) -> float:
        return self.weight.at(epoch)

    def _losses(
        self,
        per_dose_pred: list[torch.Tensor],
        targets: torch.Tensor,
        abs_scale: torch.Tensor,
        T: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute (mse, rank, auc, n_inversions_at_zero_emb) from a list
        of per-dose predictions ``[D] of [B, T, C]``. Pure tensor function:
        no autograd side-effects, no optimizer interaction. Used by both
        ``compute`` and the unit tests so the loss math has a single
        in-code definition.

        Three loss components, each targeting a different aspect of the
        kernel's output:

        * ``mse``  — per-element shape supervision, mean over (B, T, C).
          Pulls the kernel toward the cold target's per-time-step profile.
        * ``rank`` — pairwise dose-ordering hinge, only meaningful when
          targets themselves rank doses on a channel.
        * ``auc``  — total integrated output per (dose, batch, channel)
          matching the cold target's integrated output. Targets total
          magnitude directly; cannot be diluted by time-averaging because
          the integral collapses the time dimension before squaring.

        ``n_inversions_at_zero_emb`` counts pairs (i, j) where the pred AUC
        on the ranked channel at the *first* batch element (the zero
        embedding under ``include_default_embedding``) violates the cold
        target's dose ordering.
        """
        device = abs_scale.device
        per_dose_mse = []
        for di, pred in enumerate(per_dose_pred):
            tgt = targets[di]
            per_dose_mse.append(((pred - tgt) / abs_scale).pow(2).mean())
        mse_loss = torch.stack(per_dose_mse).mean()

        pred_stack = torch.stack(per_dose_pred, dim=0)               # [D, B, T, C]
        pred_aucs = pred_stack.sum(dim=2)                            # [D, B, C]
        pred_gap = pred_aucs.unsqueeze(0) - pred_aucs.unsqueeze(1)   # [D, D, B, C]
        rank_target_gap = self._rank_target_gap.to(device)
        rank_mask = self._rank_mask.to(device)
        margin_required = self.protocol.rank_margin * rank_target_gap
        norm = abs_scale * float(T)
        violation = torch.relu(margin_required.unsqueeze(2) - pred_gap) / norm
        violation = violation * rank_mask.unsqueeze(2)
        B = int(pred_aucs.shape[1])
        n_active = rank_mask.sum() * float(B)
        if float(n_active) > 0:
            rank_loss = violation.sum() / n_active
        else:
            rank_loss = torch.zeros((), device=device)

        # AUC-matching term. ``pred_aucs`` is [D, B, C]; ``self._auc_targets``
        # is [D, C] (broadcasts across batch). Normalize by per-channel AUC
        # scale so high-throughput channels (glucose, lipid) and the binary
        # nutrient_flag channel contribute on comparable footing. Mean over
        # all (D, B, C) entries — but unlike the per-element MSE this is a
        # mean over D·B·C = 7·4·4 ≈ 112 already-collapsed scalars, not over
        # 7·4·240·4 ≈ 27000 mostly-zero per-time-step errors, so a uniform
        # multiplicative over-amp lands a real per-dose penalty.
        auc_targets = self._auc_targets.to(device).unsqueeze(1)      # [D, 1, C]
        auc_scale = self._auc_scale.to(device)                       # [C]
        auc_loss = ((pred_aucs - auc_targets) / auc_scale).pow(2).mean()

        with torch.no_grad():
            zero_aucs = pred_aucs[:, 0, 0]
            # An inversion is "the lower-dose embedding (i) shows more AUC than
            # the higher-dose one (j)" on a pair where the cold target itself
            # ranks j > i (i.e. ``rank_mask[i, j] > 0``).
            inv = (zero_aucs.unsqueeze(1) > zero_aucs.unsqueeze(0)) & (rank_mask[..., 0] > 0)
            n_inv_zero = inv.sum().to(torch.float32)

        return mse_loss, rank_loss, auc_loss, n_inv_zero

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
        targets = self._targets.to(device)        # [D, T, 4]
        abs_scale = self._abs_scale.to(device)
        T = self.protocol.post_window_min
        times = torch.arange(T, dtype=torch.float32, device=device)
        net = cast(ModularPhysiologyNetwork, model)

        # Batch all embeddings into one kernel call per dose: B = len(emb_list).
        # forward_window vectorizes across (B, T, M=1), so the inner loop is
        # per-dose only — D kernel calls instead of D × B.
        embs = torch.stack(emb_list, dim=0)  # [B, EMB]
        emb_gut = net.embedding_projections["gut"](embs)  # [B, EMB_GUT]
        B = int(embs.shape[0])

        per_dose_pred: list[torch.Tensor] = []
        for dose in self.protocol.carb_doses_g:
            meal = MealEvent(
                time=0.0, carbs=float(dose),
                fats=float(self.protocol.fats_g),
                proteins=float(self.protocol.proteins_g),
            )
            pred = net.gut.forward_window(times, [meal], emb_gut)  # [B, T, 4]
            per_dose_pred.append(pred)

        mse_loss, rank_loss, auc_loss, n_inv_zero = self._losses(
            per_dose_pred, targets, abs_scale, T,
        )
        loss = (
            mse_loss
            + self.ranking_weight * rank_loss
            + self.auc_weight * auc_loss
        )

        n_pairs = B * len(self.protocol.carb_doses_g)
        n_inv_zero_f = float(n_inv_zero) if self.include_default_embedding else 0.0

        safe_step(
            w * loss,
            ctx,
            signal=self.name,
            extra={
                "raw_loss": float(loss.detach().item()),
                "weight": float(w),
                "mse": float(mse_loss.detach().item()),
                "rank": float(rank_loss.detach().item()),
                "auc": float(auc_loss.detach().item()),
                "n_dose_emb_pairs": float(n_pairs),
                "n_inversions_zero_emb": n_inv_zero_f,
            },
        )

        return SignalResult(
            loss_sum=float(loss.detach().item()),
            n_units=1,
            sub_metrics={
                "n_dose_emb_pairs": float(n_pairs),
                "mse": float(mse_loss.detach().item()),
                "rank": float(rank_loss.detach().item()),
                "auc": float(auc_loss.detach().item()),
                "n_inversions_zero_emb": n_inv_zero_f,
            },
        )
