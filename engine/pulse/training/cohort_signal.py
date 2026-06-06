"""
Cohort statistic signal — quantitative literature effect-size matching.

Per-epoch: select a small set of embeddings (sampled patient embeddings plus
the zero "default" embedding the textbook benchmark queries), run each cohort
spec's arms, extract a scalar statistic per spec, penalize Gaussian-z²
discrepancy versus the literature target. One backward + step at the end.

Always supervising the zero embedding alongside sampled patient embeddings is
critical: every textbook scenario ends up calling the model at zero, so any
counter-regulatory physiology that lives only in patient embeddings is invisible
at evaluation time. See ``training.embedding_sampler``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from ..cohort_loss import (
    InitialStateFn,
    cohort_statistic_loss_one_spec,
    norm_center_initial_state,
)
from ..knowledge.cohort_types import CohortStatisticSpec, InitMode
from ..knowledge.full_body import PatientParams
from ..knowledge.textbook_scenarios.base import cold_model_trajectory
from ..types import NORM_CENTER
from .embedding_sampler import select_supervised_embeddings
from .safe_step import NaNTrainingAbort, _grad_stats
from .signals import SignalContext, SignalResult, TrainingSignal, WeightSchedule


@dataclass
class CohortStatisticSignal(TrainingSignal):
    """Population-level supervision against literature effect sizes.

    Each spec declares its own ``init_mode`` (``cold`` or
    ``norm_center``) — see ``CohortStatisticSpec``. Cold initial
    states are pre-computed once at construction (they depend only on
    the spec definition, ``PatientParams()`` defaults, and
    ``cold_init_seed``). Norm-center initial states are spec-agnostic
    and built once per device. ``use_cold_initial_state=False`` keeps
    the legacy global override that forces every spec to use
    ``NORM_CENTER``, ignoring per-spec settings — useful for ablations
    that need a single anchor.
    """

    specs: list[CohortStatisticSpec] = field(default_factory=list)
    n_patients: int = 0
    sample_patients: int = 16
    include_default_embedding: bool = True
    weight: WeightSchedule = field(default_factory=lambda: WeightSchedule(0.0))
    use_cold_initial_state: bool = True
    cold_init_seed: int = 0
    # Iter 74: violation-proportional reweighting between specs, mirroring
    # PhysiologyRulesSignal.adaptive (iter 67). With ~31 specs sharing a single
    # 0.15 signal weight, a default-weighted spec sees only ~0.15/Σweight ≈
    # 2.6e-3 of pull, uniform whether it's badly missed or already matched.
    # Until iter 74 the only counter to this was hand-tuned per-spec weight
    # bumps (12×/3×). Adaptive reweights specs at aggregation time so the
    # per-spec coefficient tracks the spec's EMA discrepancy, preserving
    # Σ(spec.weight) so the signal-level pull against other signals is
    # unchanged — only the inter-spec distribution moves. The violation
    # measure is the per-spec Gaussian-z² loss (≥0, larger = worse miss).
    adaptive: bool = False
    ema_alpha: float = 0.1
    # Floor on the per-spec EMA (relative to the population max) so a matched
    # spec keeps a residual pull and can recover if later drift re-violates it
    # — without it, a once-satisfied spec would zero-weight irreversibly.
    ema_floor_frac: float = 0.02

    name: str = "cohort_statistic"
    source: str = "literature_effect_sizes"
    category: str = "cohort_statistic"

    def __post_init__(self) -> None:
        self._cold_init_np: dict[str, np.ndarray] = {}
        if self.use_cold_initial_state:
            prng = np.random.default_rng(self.cold_init_seed)
            for spec in self.specs:
                if spec.init_mode is not InitMode.COLD:
                    continue
                arm = spec.arms[0]
                spec_rng = np.random.default_rng(prng.integers(0, 2**32))
                traj = cold_model_trajectory(
                    PatientParams(),
                    list(arm.meals),
                    arm.duration_min,
                    arm.start_hour,
                    rng=spec_rng,
                )
                self._cold_init_np[spec.name] = traj[0]
        self._cold_init_tensor: dict[tuple[str, str], torch.Tensor] = {}
        self._norm_center_np = np.array(NORM_CENTER, dtype=np.float32)
        self._norm_center_tensor: dict[str, torch.Tensor] = {}
        # Iter 74: per-spec discrepancy EMA + fixed weight budget for adaptive
        # reweighting. Initialised lazily on the first epoch the specs produce a
        # loss so a cold start doesn't zero-weight everything before any data
        # has flowed. Mirrors PhysiologyRulesSignal.
        self._violation_ema: dict[str, float] = {}
        self._base_weight_sum: float = sum(spec.weight for spec in self.specs)

    def _update_violation_ema(self, loss_by_spec: dict[str, float]) -> None:
        """Refresh the per-spec discrepancy EMA from this epoch's losses."""
        for sname, loss in loss_by_spec.items():
            prev = self._violation_ema.get(sname)
            self._violation_ema[sname] = (
                loss if prev is None
                else (1.0 - self.ema_alpha) * prev + self.ema_alpha * loss
            )

    def _adaptive_weights_from_ema(self) -> dict[str, float]:
        """Adaptive per-spec weights from the current EMA.

        Preserves ``sum(spec.weight)`` so the signal's overall pull on the
        optimizer stays put — only the inter-spec distribution changes. Specs
        absent from the EMA (no loss yet) fall back to their base weight in the
        per-spec lookup below.
        """
        if not self._violation_ema:
            return {}
        max_ema = max(self._violation_ema.values()) or 1.0
        floor = self.ema_floor_frac * max_ema
        floored = {k: max(v, floor) for k, v in self._violation_ema.items()}
        ema_sum = sum(floored.values()) or 1.0
        budget = self._base_weight_sum or 1.0
        return {k: budget * v / ema_sum for k, v in floored.items()}

    def weight_at(self, epoch: int) -> float:
        return self.weight.at(epoch)

    def _build_initial_state_fn(
        self, rng: np.random.Generator, device: torch.device | str,
    ) -> InitialStateFn:
        if not self.use_cold_initial_state:
            return norm_center_initial_state(device)

        device_key = str(device)

        def norm_center() -> torch.Tensor:
            cached = self._norm_center_tensor.get(device_key)
            if cached is not None:
                return cached
            tensor = torch.tensor(self._norm_center_np, dtype=torch.float32, device=device)
            self._norm_center_tensor[device_key] = tensor
            return tensor

        def fn(spec: CohortStatisticSpec) -> torch.Tensor:
            if spec.init_mode is InitMode.NORM_CENTER:
                return norm_center()
            key = (spec.name, device_key)
            cached = self._cold_init_tensor.get(key)
            if cached is not None:
                return cached
            arr = self._cold_init_np[spec.name]
            tensor = torch.tensor(arr, dtype=torch.float32, device=device)
            self._cold_init_tensor[key] = tensor
            return tensor

        return fn

    def compute(
        self,
        model: nn.Module,
        embeddings: nn.Embedding,
        ctx: SignalContext,
    ) -> SignalResult:
        w = self.weight_at(ctx.epoch)
        if w <= 0 or not self.specs:
            return SignalResult()

        # Iter 68 r6: sample embedding IDs once per epoch (stable across specs),
        # but redo the lookup fresh inside the per-spec loop. r5's single
        # ``select_supervised_embeddings`` call produced non-leaf tensors from
        # ``embeddings(pid_t)`` that were shared across all 31 specs' forward
        # passes; the first spec's ``weighted.backward()`` freed that lookup
        # op's saved tensors, and the second spec's backward then tried to
        # traverse the same op a second time and crashed with
        # "Trying to backward through the graph a second time". Per-spec
        # re-lookup gives each spec its own autograd subgraph.
        sample_emb_list = select_supervised_embeddings(
            embeddings=embeddings,
            n_patients=self.n_patients,
            sample_patients=self.sample_patients,
            rng=ctx.rng,
            device=ctx.device,
            include_default=self.include_default_embedding,
        )
        if not sample_emb_list:
            return SignalResult()
        del sample_emb_list  # only used to early-out; re-sampled per spec below

        sampled_pids: np.ndarray | None = None
        if self.n_patients > 0 and self.sample_patients > 0:
            k = min(self.sample_patients, self.n_patients)
            sampled_pids = ctx.rng.choice(self.n_patients, size=k, replace=False)

        def build_emb_list() -> list[torch.Tensor]:
            out: list[torch.Tensor] = []
            if sampled_pids is not None:
                for pid in sampled_pids:
                    pid_t = torch.tensor(int(pid), dtype=torch.long, device=ctx.device)
                    out.append(embeddings(pid_t))
            if self.include_default_embedding:
                out.append(torch.zeros(embeddings.embedding_dim, device=ctx.device))
            return out

        init_fn = self._build_initial_state_fn(ctx.rng, ctx.device)

        # Iter 68 r5: per-spec gradient accumulation. The iter-67 saga's
        # final remaining OOM was here — the old cohort_statistic_epoch_loss
        # accumulated all 31 specs' autograd graphs in memory before one
        # total.backward(), peaking at ~18 GB RSS and leaving no headroom
        # for physiology_rules + final backward later in the epoch.
        #
        # Switch to per-spec forward → backward → drop graph → next; PyTorch
        # accumulates gradients into ``param.grad`` so the math is identical
        # to the old composite backward by linearity of differentiation.
        # One optimizer.step at the end. Peak RSS during cohort drops from
        # ~18 GB to ~600 MB. NaN-abort + grad clip inlined to mirror
        # safe_step's semantics (which we can't call per-spec — it
        # would optimizer.step 31 times per epoch, changing the dynamics).
        # Iter 74: in adaptive mode, supply the current EMA-derived per-spec
        # weights. The first epoch the signal fires has no EMA yet — override
        # is empty, every spec uses its base weight, and the EMA is initialised
        # from this epoch's losses. Subsequent epochs use the updated EMA.
        override = self._adaptive_weights_from_ema() if self.adaptive else None
        total_weight = sum(
            (override.get(spec.name, spec.weight) if override else spec.weight)
            for spec in self.specs
        ) or 1e-8
        ctx.optimizer.zero_grad()
        raw_weighted_sum = 0.0
        z_by_spec: dict[str, float] = {}
        loss_by_spec: dict[str, float] = {}
        for spec in self.specs:
            emb_list = build_emb_list()
            n_emb = len(emb_list)
            loss_t, _pred, z = cohort_statistic_loss_one_spec(
                model, emb_list, spec, init_fn(spec),
            )
            z_by_spec[spec.name] = z
            sw = override.get(spec.name, spec.weight) if override else spec.weight
            loss_by_spec[spec.name] = float(loss_t.detach().item())
            weighted = (w * sw / total_weight) * loss_t
            if not torch.isfinite(weighted):
                ctx.optimizer.zero_grad()
                raise NaNTrainingAbort(
                    signal=self.name,
                    epoch=ctx.epoch,
                    cause="loss",
                    loss_value=float(weighted.detach().item()),
                    extra={
                        "spec": spec.name,
                        "weight": float(w),
                        "n_emb": float(n_emb),
                        **{f"z_{n}": float(zv) for n, zv in z_by_spec.items()},
                    },
                )
            weighted.backward()
            raw_weighted_sum += loss_by_spec[spec.name] * sw
            del loss_t, weighted, _pred, emb_list  # release graph leaves promptly

        raw_avg = raw_weighted_sum / total_weight

        all_finite, gstats = _grad_stats(ctx.params)
        if not all_finite:
            ctx.optimizer.zero_grad()
            merged: dict[str, float] = dict(gstats)
            merged.update({f"z_{n}": float(zv) for n, zv in z_by_spec.items()})
            raise NaNTrainingAbort(
                signal=self.name,
                epoch=ctx.epoch,
                cause="grad",
                loss_value=raw_avg,
                extra=merged,
            )
        nn.utils.clip_grad_norm_(ctx.params, max_norm=ctx.grad_clip)
        ctx.optimizer.step()
        ctx.optimizer.zero_grad()

        if self.adaptive:
            self._update_violation_ema(loss_by_spec)

        sub_metrics: dict[str, float] = {f"z_{name}": z for name, z in z_by_spec.items()}
        if self.adaptive:
            for spec in self.specs:
                sw = override.get(spec.name, spec.weight) if override else spec.weight
                sub_metrics[f"w_{spec.name}"] = float(sw)
        return SignalResult(
            loss_sum=raw_avg,
            n_units=1,
            sub_metrics=sub_metrics,
        )
