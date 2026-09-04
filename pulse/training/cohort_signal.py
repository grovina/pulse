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

from dataclasses import dataclass, field, replace

import numpy as np
import torch
from torch import nn

from ..cohort_loss import (
    InitialStateFn,
    cohort_statistic_loss_group,
    norm_center_initial_state,
)
from ..knowledge.cohort_types import CohortArmSpec, CohortStatisticSpec, InitMode
from ..knowledge.full_body import PatientParams
from ..types import NORM_CENTER
from .adaptive_weights import adaptive_multipliers
from .arm_init import cold_initial_state_for_arm
from .embedding_sampler import select_supervised_embeddings
from .safe_step import finalize_aux_accumulation, grad_snapshot
from .signals import SignalContext, SignalResult, TrainingSignal, WeightSchedule


# Iter 97 (review 4.10): per-epoch protocol perturbation for the long arms.
# The sleep cohort arm IS the gate episode benchmark-cohort-sleep-48h-adequate,
# so training on it verbatim makes that score a memory test. Meal grams move
# +/-20 %, meal times +/-30 min and the start hour +/-1 h — but a meal is never
# allowed to cross a statistic-window boundary (that would change WHAT the
# statistic measures, not just the protocol it is measured on), and only arms of
# at least a day are perturbed (the 300-min OGTT/breakfast groups have windows
# keyed to the minute of the meal). The fixed protocol stays one sample among
# the perturbed ones.
PERTURB_DOSE_FRAC = 0.20
PERTURB_TIME_MIN = 30.0
PERTURB_START_HOUR = 1.0
PERTURB_MIN_DURATION = 1440


def _windows_of(spec: CohortStatisticSpec, arm_idx: int) -> tuple[int, int]:
    w = spec.per_arm_windows[arm_idx] if spec.per_arm_windows is not None else spec.window
    return int(w.start_min), int(w.end_min)


def _same_side(t: float, t_new: float, bounds: list[int]) -> bool:
    return all((t < b) == (t_new < b) for b in bounds)


def perturb_group_arms(
    specs: list[CohortStatisticSpec],
    rng: np.random.Generator,
    *,
    dose_frac: float = PERTURB_DOSE_FRAC,
    time_min: float = PERTURB_TIME_MIN,
    start_hour: float = PERTURB_START_HOUR,
) -> tuple[CohortArmSpec, ...] | None:
    """A perturbed copy of the group's arms, or ``None`` if the group is not
    eligible (any arm shorter than a day). Meals equal across arms (a shared
    schedule) get the SAME perturbation so a delta statistic still compares
    like with like. A meal whose shifted time would cross any spec window
    boundary keeps its original time (grams still move)."""
    arms = specs[0].arms
    if any(a.duration_min < PERTURB_MIN_DURATION for a in arms):
        return None
    d_start = float(rng.uniform(-start_hour, start_hour))
    per_meal: dict[tuple[float, float, float, float], tuple[float, float, float, float]] = {}
    out: list[CohortArmSpec] = []
    for arm_idx, arm in enumerate(arms):
        bounds: list[int] = []
        for sp in specs:
            lo, hi = _windows_of(sp, arm_idx)
            bounds.extend((lo, hi))
        new_meals = []
        for m in arm.meals:
            key = tuple(float(x) for x in m)
            if key not in per_meal:
                t, c, f, pr = key
                scale = 1.0 + float(rng.uniform(-dose_frac, dose_frac))
                t_new = t + float(rng.uniform(-time_min, time_min))
                t_new = min(max(t_new, 0.0), float(arm.duration_min - 1))
                per_meal[key] = (t_new, c * scale, f * scale, pr * scale)
            t_new, c2, f2, p2 = per_meal[key]
            t0 = key[0]
            if not _same_side(t0, t_new, bounds):
                t_new = t0
            new_meals.append((t_new, c2, f2, p2))
        out.append(replace(
            arm,
            meals=tuple(new_meals),
            start_hour=(float(arm.start_hour) + d_start) % 24.0,
        ))
    return tuple(out)


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
    # Iter 97 (review 4.3): the adaptive factor MULTIPLIES ``spec.weight`` and
    # no spec may take more than this share of the budget (one spec took 98.5 %
    # on the iter-95 CCK case). See ``training.adaptive_weights``.
    adaptive_cap_share: float = 0.25
    # Iter 97 (review 4.10): perturb the long arms' protocols per epoch; the
    # fixed protocol is drawn with ``perturb_fixed_prob``.
    perturb_protocols: bool = False
    perturb_fixed_prob: float = 0.25
    # Iter 97 (review 4.9): how many arm-protocol groups to score per compute
    # call (0 = all). With the interleaved aux cadence each call is one of
    # many per epoch, so a slice of the specs per call covers them all.
    groups_per_step: int = 0

    name: str = "cohort_statistic"
    source: str = "literature_effect_sizes"
    category: str = "cohort_statistic"

    def __post_init__(self) -> None:
        # Iter 97 (review 4.5 / 4.9): the cold init comes from the shared arm
        # frame (declared pre-fast honoured; awake / rest when undeclared).
        self._cold_init_np: dict[str, np.ndarray] = {}
        if self.use_cold_initial_state:
            prng = np.random.default_rng(self.cold_init_seed)
            for spec in self.specs:
                if spec.init_mode is not InitMode.COLD:
                    continue
                arm = spec.arms[0]
                spec_rng = np.random.default_rng(prng.integers(0, 2**32))
                self._cold_init_np[spec.name] = cold_initial_state_for_arm(
                    arm, PatientParams(), rng=spec_rng,
                )
        self._cold_init_tensor: dict[tuple[str, str], torch.Tensor] = {}
        self._norm_center_np = np.array(NORM_CENTER, dtype=np.float32)
        self._norm_center_tensor: dict[str, torch.Tensor] = {}
        # Iter 74: per-spec discrepancy EMA + fixed weight budget for adaptive
        # reweighting. Initialised lazily on the first epoch the specs produce a
        # loss so a cold start doesn't zero-weight everything before any data
        # has flowed. Mirrors PhysiologyRulesSignal.
        self._violation_ema: dict[str, float] = {}
        self._base_weight_sum: float = sum(spec.weight for spec in self.specs)
        self._base_weight_by_spec: dict[str, float] = {spec.name: float(spec.weight) for spec in self.specs}
        self._group_cursor: int = 0

    def _update_violation_ema(self, loss_by_spec: dict[str, float]) -> None:
        """Refresh the per-spec discrepancy EMA from this epoch's losses."""
        for sname, loss in loss_by_spec.items():
            prev = self._violation_ema.get(sname)
            self._violation_ema[sname] = (
                loss if prev is None
                else (1.0 - self.ema_alpha) * prev + self.ema_alpha * loss
            )

    def _adaptive_weights_from_ema(self) -> dict[str, float]:
        """Adaptive per-spec weights from the current EMA (of the z^2 loss —
        already dimensionless): ``spec.weight * multiplier`` preserving
        ``sum(spec.weight)``, share capped (``training.adaptive_weights``).
        Specs absent from the EMA keep their base weight.
        """
        if not self._violation_ema:
            return {}
        return adaptive_multipliers(
            self._violation_ema, self._base_weight_by_spec,
            cap_share=self.adaptive_cap_share, floor_frac=self.ema_floor_frac,
        )

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
                    # Iter 97 (review 4.3): DETACHED. A population statistic says
                    # nothing about which patient is which; letting it move the
                    # embedding table pulled every sampled patient toward the
                    # literature mean and entangled the codes calibration relies on.
                    out.append(embeddings(pid_t).detach())
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
        # iter 79: CROSS-SPEC PROTOCOL BATCHING. The iter-68 loop rolled each
        # spec's arm(s) one-at-a-time, but many specs share an identical arm
        # protocol and differ only in marker/window/target (e.g. four
        # 2880-step sleep specs, five 1440-step extended-fast specs, four
        # 300-step OGTT specs). Group specs by their ``arms`` tuple (frozen +
        # hashable, so value-equality IS the batchability condition) and roll
        # each distinct protocol ONCE over the group's stacked per-spec
        # cold-init states — collapsing the long shared rollouts. Singleton
        # protocols fall through the same path with S=1 (no behavior change).
        #
        # GRADIENT-IDENTICAL to the per-spec path: the group's per-spec losses
        # share one rollout graph; summing the weighted losses and doing ONE
        # backward per group accumulates into .grad exactly as per-spec
        # backward did (linearity — the same argument the iter-68 block above
        # relies on). Memory bound preserved: one group's graph is live at a
        # time (largest group ≈ 5 specs × B, vs the OOM-causing all-31 graph).
        #
        # ``emb_list`` rebuilt fresh per group (not per spec): within a group
        # there is one backward, so the shared embeddings(pid) lookup is
        # traversed once — the r5 shared-graph crash only bites across
        # independent backwards, which now happen per group, not per spec.
        # ``snapshot`` backs the isolation policy (non-finite spec dropped).
        snapshot = grad_snapshot(ctx)

        groups: dict[tuple, list[CohortStatisticSpec]] = {}
        for spec in self.specs:
            groups.setdefault(spec.arms, []).append(spec)
        group_list = list(groups.values())
        # Iter 97 (review 4.9 / 4.1): a slice of the groups per call, round-robin,
        # so every spec is visited across the interleaved aux steps of an epoch.
        if self.groups_per_step > 0 and self.groups_per_step < len(group_list):
            k = int(self.groups_per_step)
            start = self._group_cursor % len(group_list)
            idxs = [(start + i) % len(group_list) for i in range(k)]
            self._group_cursor = (start + k) % len(group_list)
            group_list = [group_list[i] for i in idxs]
            total_weight = sum(
                (override.get(spec.name, spec.weight) if override else spec.weight)
                for g in group_list for spec in g
            ) or 1e-8

        raw_weighted_sum = 0.0
        z_by_spec: dict[str, float] = {}
        loss_by_spec: dict[str, float] = {}
        n_perturbed = 0
        for group_specs in group_list:
            emb_list = build_emb_list()
            init_states = [init_fn(spec) for spec in group_specs]
            arms_override = None
            if self.perturb_protocols and ctx.rng.random() >= self.perturb_fixed_prob:
                arms_override = perturb_group_arms(group_specs, ctx.rng)
                if arms_override is not None:
                    n_perturbed += 1
            results = cohort_statistic_loss_group(
                model, emb_list, group_specs, init_states, arms_override=arms_override,
            )
            group_loss = None
            for spec in group_specs:
                loss_t, _pred, z = results[spec.name]
                z_by_spec[spec.name] = z
                sw = override.get(spec.name, spec.weight) if override else spec.weight
                loss_by_spec[spec.name] = float(loss_t.detach().item())
                contrib = (w * sw / total_weight) * loss_t
                if not torch.isfinite(contrib):
                    print(
                        f"[SKIP-NONFINITE] signal={self.name} epoch={ctx.epoch} "
                        f"cause=loss spec={spec.name} (spec dropped this epoch)",
                        flush=True,
                    )
                    continue
                group_loss = contrib if group_loss is None else group_loss + contrib
                raw_weighted_sum += loss_by_spec[spec.name] * sw
            if group_loss is not None:
                group_loss.backward()
            del results, group_loss, emb_list  # release graph leaves promptly

        raw_avg = raw_weighted_sum / total_weight

        finalize_aux_accumulation(ctx, snapshot, signal=self.name)

        if self.adaptive:
            self._update_violation_ema(loss_by_spec)

        sub_metrics: dict[str, float] = {f"z_{name}": z for name, z in z_by_spec.items()}
        sub_metrics["n_groups"] = float(len(group_list))
        sub_metrics["n_perturbed_groups"] = float(n_perturbed)
        if self.adaptive:
            for spec in self.specs:
                sw = override.get(spec.name, spec.weight) if override else spec.weight
                sub_metrics[f"w_{spec.name}"] = float(sw)
        return SignalResult(
            loss_sum=raw_avg,
            n_units=1,
            sub_metrics=sub_metrics,
        )
