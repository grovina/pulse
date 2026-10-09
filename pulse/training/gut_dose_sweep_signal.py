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
dose sweep. The zero embedding is the median person. A sampled row is scored
against that patient's own teacher patient. One vectorized kernel call per
(embedding, dose) pair — cheap, focused, gradient lands exactly on the gut
pipeline (kernel + gut embedding projection).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import numpy as np
import torch
import torch.nn as nn

from ..knowledge.full_body import PatientParams, compute_absorption_profile
from ..model import ModularPhysiologyNetwork
from ..modules.base import GutModuleBase
from ..modules.gut import GUT_OUTPUT_SCALE, MEAL_ACTIVE_WINDOW_MIN, MealEvent
from ..types import GUT_OUTPUT_DIM
from .embedding_sampler import select_supervised_rows
from .safe_step import accumulate_grad
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
    # Iter 97 (teacher hand-off): the teacher kernel is mass-conserving and active
    # to 8/rate (~667 min at the default slow rate); over 240 min it delivers only
    # 85 % of a 60 g dose (AUC 391 of 462 mg/dL-min). The sweep covers the whole
    # window the student kernel can express (MEAL_ACTIVE_WINDOW_MIN = 720,
    # >99 % of the mass), so the AUC target is the ingested mass and not a
    # truncation of it.
    post_window_min: int = int(MEAL_ACTIVE_WINDOW_MIN)
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

    The zero embedding is the median person and is scored against
    ``PatientParams()``. A sampled row is scored against that patient's own
    teacher ``PatientParams`` (PLAN B4). A6 had stopped sampling because every
    row was scored against the median person's absorption kernel, so
    ``sample_patients = 4`` pulled each patient's absorption onto the median
    from epoch 0. The trajectory record's ``patient_params`` is the ground
    truth; a sampled id with no entry raises rather than falling back to the
    median target.
    """

    n_patients: int = 0
    # 0: zero embedding only, against PatientParams(). Above 0, each sampled
    # row is scored against patient_params[pid].
    sample_patients: int = 0
    # pid -> the teacher patient that row was simulated from. Empty is correct
    # when nothing is sampled. A sampled pid absent from this map is an error.
    patient_params: dict[int, PatientParams] = field(default_factory=dict)
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
        # The zero row's profile. Each recorded teacher patient gets its own;
        # body mass scales mg/dL per gram, so two patients do not share a curve.
        self._targets = self._appearance_targets(PatientParams())
        # Built on first use. A run that does not sample pays nothing, and a
        # sampled patient is built once.
        self._patient_targets: dict[int, torch.Tensor] = {}

        # Per-channel scales — single source of truth in modules.gut so this
        # signal and TrajectoryRolloutSignal supervise the kernel on the same
        # error magnitudes. Appearance only: the teacher's nutrient_flag is a
        # binary "appearance > 0.01" square wave, the student's is survival of
        # unabsorbed mass. Matching their AUCs is a category error (iter 98:
        # appearance already matched cold to 2 %, the logged gut_sweep ≈ 32
        # was eight aux-steps of flag AUC).
        self._abs_scale = torch.tensor(
            GUT_OUTPUT_SCALE[:GutModuleBase.N_APPEARANCE], dtype=torch.float32,
        )

        # Median-person AUC, kept for callers that read the zero-row target.
        # The loss itself ranks and matches AUC from the targets it is given,
        # which are per row once patients are sampled. The scale is the typical
        # "AUC magnitude" under a kernel-shaped profile: peak ≈ abs_scale,
        # time-integral ≈ peak · T / 4. Dividing by it keeps channels with
        # different units on comparable footing.
        self._auc_targets = self._targets.sum(dim=1)                     # [D, C]
        self._auc_scale = self._abs_scale * float(self.protocol.post_window_min) / 4.0  # [C]

    def _appearance_targets(self, params: PatientParams) -> torch.Tensor:
        """Cold appearance profiles for one patient, ``[D, T, N_APPEARANCE]``."""
        targets = np.stack([
            _cold_target_for_dose(
                d, self.protocol.fats_g, self.protocol.proteins_g,
                self.protocol.post_window_min, params,
            )[..., :GutModuleBase.N_APPEARANCE]
            for d in self.protocol.carb_doses_g
        ])
        return torch.tensor(targets, dtype=torch.float32)

    def _targets_for(self, pid: int | None) -> torch.Tensor:
        """Appearance target for one supervised row. ``None`` is the zero embedding."""
        if pid is None:
            return self._targets
        params = self.patient_params.get(pid)
        if params is None:
            raise RuntimeError(
                f"gut dose sweep sampled patient {pid} with no patient_params; "
                "refusing to score that row against the median person"
            )
        cached = self._patient_targets.get(pid)
        if cached is None:
            cached = self._appearance_targets(params)
            self._patient_targets[pid] = cached
        return cached

    def _targets_for_rows(self, pids: list[int | None]) -> torch.Tensor:
        """Stack per-row profiles on a batch axis: ``[D, B, T, C]``."""
        return torch.stack([self._targets_for(pid) for pid in pids], dim=1)

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

        ``targets`` is ``[D, T, C]`` (one profile, broadcast across the batch)
        or ``[D, B, T, C]`` (each row its own patient). Rank hinges and the
        AUC term are taken from that tensor, not from the median cache.

        ``n_inversions_at_zero_emb`` counts pairs (i, j) where the pred AUC
        on the ranked channel at the *last* batch element (the zero embedding
        under ``include_default_embedding``: ``select_supervised_rows``
        appends it after the sampled rows, so with ``sample_patients > 0`` the
        first element is a sampled patient) violates that row's cold target
        dose ordering.
        """
        device = abs_scale.device
        per_row = targets.dim() == 4
        per_dose_mse = []
        for di, pred in enumerate(per_dose_pred):
            tgt = targets[di]
            per_dose_mse.append(((pred - tgt) / abs_scale).pow(2).mean())
        mse_loss = torch.stack(per_dose_mse).mean()

        pred_stack = torch.stack(per_dose_pred, dim=0)               # [D, B, T, C]
        pred_aucs = pred_stack.sum(dim=2)                            # [D, B, C]
        pred_gap = pred_aucs.unsqueeze(0) - pred_aucs.unsqueeze(1)   # [D, D, B, C]
        if per_row:
            tgt_aucs = targets.sum(dim=2)                            # [D, B, C]
            tgt_gap = tgt_aucs.unsqueeze(0) - tgt_aucs.unsqueeze(1)  # [D, D, B, C]
            rank_mask = (tgt_gap > 0).to(torch.float32)
            zero_mask = rank_mask[:, :, -1, 0]
            auc_targets = tgt_aucs
            n_active = rank_mask.sum()
        else:
            tgt_aucs = targets.sum(dim=1)                            # [D, C]
            tgt_gap = tgt_aucs.unsqueeze(0) - tgt_aucs.unsqueeze(1)  # [D, D, C]
            rank_mask = (tgt_gap > 0).to(torch.float32)
            zero_mask = rank_mask[..., 0]
            tgt_gap = tgt_gap.unsqueeze(2)                           # [D, D, 1, C]
            rank_mask = rank_mask.unsqueeze(2)
            auc_targets = tgt_aucs.unsqueeze(1)                      # [D, 1, C]
            n_active = rank_mask.sum() * float(pred_aucs.shape[1])
        margin_required = self.protocol.rank_margin * tgt_gap
        norm = abs_scale * float(T)
        violation = torch.relu(margin_required - pred_gap) / norm
        violation = violation * rank_mask
        if float(n_active) > 0:
            rank_loss = violation.sum() / n_active
        else:
            rank_loss = torch.zeros((), device=device)

        # AUC-matching term. ``pred_aucs`` is [D, B, C]; the target is that
        # shape too when each row has its own patient, and [D, 1, C] when one
        # profile is broadcast. Normalize by per-channel AUC scale so glucose
        # / lipid / amino contribute on comparable footing. Mean over D·B·C
        # already-collapsed scalars, not over D·B·T·C mostly-zero per-time-step
        # errors, so a uniform multiplicative over-amp lands a real per-dose
        # penalty.
        auc_scale = self._auc_scale.to(device)                       # [C]
        auc_loss = ((pred_aucs - auc_targets) / auc_scale).pow(2).mean()

        with torch.no_grad():
            zero_aucs = pred_aucs[:, -1, 0]
            # An inversion is "the lower-dose embedding (i) shows more AUC than
            # the higher-dose one (j)" on a pair where that row's cold target
            # itself ranks j > i.
            inv = (zero_aucs.unsqueeze(1) > zero_aucs.unsqueeze(0)) & (zero_mask > 0)
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

        rows = select_supervised_rows(
            embeddings=embeddings,
            n_patients=self.n_patients,
            sample_patients=self.sample_patients,
            rng=ctx.rng,
            device=ctx.device,
            include_default=self.include_default_embedding,
        )
        if not rows:
            return SignalResult()

        device = ctx.device
        targets = self._targets_for_rows([pid for pid, _emb in rows]).to(device)  # [D, B, T, C]
        abs_scale = self._abs_scale.to(device)
        T = self.protocol.post_window_min
        times = torch.arange(T, dtype=torch.float32, device=device)
        net = cast(ModularPhysiologyNetwork, model)

        # Batch all embeddings into one kernel call per dose: B = len(rows).
        # forward_window vectorizes across (B, T, M=1), so the inner loop is
        # per-dose only — D kernel calls instead of D × B.
        embs = torch.stack([emb for _pid, emb in rows], dim=0)  # [B, EMB]
        emb_gut = net.embedding_projections["gut"](embs)  # [B, EMB_GUT]
        B = int(embs.shape[0])

        per_dose_pred: list[torch.Tensor] = []
        for dose in self.protocol.carb_doses_g:
            meal = MealEvent(
                time=0.0, carbs=float(dose),
                fats=float(self.protocol.fats_g),
                proteins=float(self.protocol.proteins_g),
            )
            pred = net.gut.forward_window(times, [meal], emb_gut)
            per_dose_pred.append(pred[..., :GutModuleBase.N_APPEARANCE])

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

        accumulate_grad(
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
