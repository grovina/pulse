"""
Rollout evidence engine — protocol → simulate → observation operator.

One TrainingSignal scores the evidence families that are operators on a
simulated protocol: cohort statistics, physiology-rule hinges, carb
dose-response, and per-patient meal glucose. Map, rate, and structure
signals stay their own engines.

Construct one instance per family (separate weights, same class). A
population statistic detaches embeddings; a hinge or meal-response does
not. See ``training.embedding_sampler``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import torch
from torch import nn

from ..cohort_loss import (
    InitialStateFn,
    _rollout_arm_batched,
    cohort_statistic_loss_group,
    norm_center_initial_state,
)
from ..dose_response import (
    DoseResponseProtocol,
    MarkerDoseTarget,
    cold_initial_state,
    dose_response_epoch_loss,
)
from ..knowledge.cohort_types import CohortArmSpec, CohortStatisticSpec, InitMode
from ..knowledge.evidence import (
    Authority,
    EvidenceItem,
    Observation,
    OperatorKind,
    from_cohort,
    from_rule,
    teacher_meal_glucose_item,
)
from ..knowledge.full_body import (
    PatientParams,
    STANDARD_MEAL_CARBS_G,
    STANDARD_MEAL_FATS_G,
    STANDARD_MEAL_PROTEINS_G,
)
from ..knowledge.physiology_rules import PhysiologyRule
from ..model import integrate, precompute_gut_outputs
from ..modules import metabolic as _met
from ..modules.gut import MealEvent
from ..physiology_rules_loss import rule_context_for_arm
from ..types import MARKER_INDEX, NORM_CENTER, NORM_SCALE
from .adaptive_weights import adaptive_multipliers
from .arm_init import cold_initial_state_for_arm
from .embedding_sampler import select_supervised_embeddings
from .safe_step import accumulate_grad, finalize_aux_accumulation, grad_snapshot
from .signals import SignalContext, SignalResult, TrainingSignal, WeightSchedule

_GLUCOSE = MARKER_INDEX["glucose"]
_MEAL_DURATION_MIN = 360
_MEAL_TIME_MIN = 120
_MEAL_START_HOUR = 8.0
_SIGMA_PEAK_MG_DL = 11.0
_SIGMA_TPEAK_MIN = 20.0
_SIGMA_FASTING_MG_DL = 8.0
_FASTING_WEIGHT = 1.0
_TPEAK_WEIGHT = 0.3
_SOFTARGMAX_BETA = 3.0

_FAMILY_NAMES = {
    "cohort": "cohort_statistic",
    "hinge": "physiology_rules",
    "dose": "dose_response",
    "meal": "meal_response",
}


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


def perturb_dose_response_protocol(
    protocol: DoseResponseProtocol, rng: np.random.Generator,
    *, dose_frac: float = 0.20, time_min: int = 30, start_hour: float = 1.0,
) -> DoseResponseProtocol:
    """A perturbed copy of the dose-response protocol (review 4.10)."""
    scale = 1.0 + float(rng.uniform(-dose_frac, dose_frac))
    doses = tuple(float(d) * scale for d in protocol.carb_doses_g)
    lo = int(protocol.pre_window)
    hi = int(protocol.duration_min - protocol.post_window)
    offset = int(protocol.meal_offset_min + rng.integers(-time_min, time_min + 1))
    offset = int(min(max(offset, lo), hi))
    sh = (float(protocol.start_hour) + float(rng.uniform(-start_hour, start_hour))) % 24.0
    return replace(protocol, carb_doses_g=doses, meal_offset_min=offset, start_hour=sh)


_DOSE_KIND = {
    "slope": OperatorKind.DOSE_SLOPE,
    "peak": OperatorKind.DOSE_PEAK,
    "rank": OperatorKind.DOSE_RANK,
}


def _dose_item(target: MarkerDoseTarget) -> EvidenceItem:
    t = target
    return EvidenceItem(
        name=f"dose_{t.marker}_{t.mode}",
        source="Wolever (1991, 1996); Brand-Miller (2003)",
        description=f"Carb dose-response {t.mode} on {t.marker}",
        authority=Authority.LITERATURE,
        arms=(),
        observations=(
            Observation(
                marker_id=t.marker,
                kind=_DOSE_KIND[t.mode],
                target=t.target_slope,
                sigma=t.sigma_slope,
            ),
        ),
        weight=t.weight,
    )


@dataclass
class RolloutEvidenceSignal(TrainingSignal):
    """One engine for evidence that is an operator on a simulated protocol.

    Pass exactly one family: ``specs`` (cohort), ``rules`` (hinges),
    ``protocol`` (dose-response), or ``targets`` (per-patient meal glucose).
    """

    specs: list[CohortStatisticSpec] = field(default_factory=list)
    rules: list[PhysiologyRule] = field(default_factory=list)
    protocol: DoseResponseProtocol | None = None
    targets: dict[int, dict[str, float]] = field(default_factory=dict)
    family: str = ""
    n_patients: int = 0
    sample_patients: int = 4
    include_default_embedding: bool = True
    weight: WeightSchedule = field(default_factory=lambda: WeightSchedule(0.0))
    use_cold_initial_state: bool = True
    cold_init_seed: int = 0
    adaptive: bool = False
    ema_alpha: float = 0.1
    ema_floor_frac: float = 0.02
    adaptive_cap_share: float = 0.25
    perturb_protocols: bool = False
    perturb_fixed_prob: float = 0.25
    groups_per_step: int = 0
    arms_per_step: int = 0

    name: str = "rollout_evidence"
    source: str = "evidence"
    category: str = "rollout"

    def __post_init__(self) -> None:
        if self.family:
            families = [self.family]
        else:
            families = []
            if self.specs:
                families.append("cohort")
            if self.rules:
                families.append("hinge")
            if self.targets:
                families.append("meal")
            if self.protocol is not None:
                families.append("dose")
            if len(families) > 1:
                raise ValueError(f"one evidence family per signal, got {families}")
            if not families:
                self.protocol = DoseResponseProtocol()
                families = ["dose"]
        self._family = families[0]
        if self.name == "rollout_evidence":
            self.name = _FAMILY_NAMES[self._family]
        self.items: tuple[EvidenceItem, ...] = ()
        self._cold_init_np: dict[str, np.ndarray] = {}
        self._cold_init_tensor: dict[tuple[str, str], torch.Tensor] = {}
        self._norm_center_np = np.array(NORM_CENTER, dtype=np.float32)
        self._norm_center_tensor: dict[str, torch.Tensor] = {}
        self._violation_ema: dict[str, float] = {}
        self._group_cursor: int = 0
        self._arm_cursor: int = 0
        if self._family == "cohort":
            self.items = tuple(from_cohort(s) for s in self.specs)
            self.source = "literature_effect_sizes"
            self.category = "cohort_statistic"
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
            self._base_weight_sum = sum(spec.weight for spec in self.specs)
            self._base_weight_by_spec = {spec.name: float(spec.weight) for spec in self.specs}
        elif self._family == "hinge":
            self.items = tuple(from_rule(r) for r in self.rules)
            self.source = "literature_plausibility_constraints"
            self.category = "mechanism"
            prng = np.random.default_rng(self.cold_init_seed)
            for rule in self.rules:
                if rule.init_mode is not InitMode.COLD:
                    continue
                for arm in rule.arms:
                    if arm.label in self._cold_init_np:
                        continue
                    rule_rng = np.random.default_rng(prng.integers(0, 2**32))
                    self._cold_init_np[arm.label] = cold_initial_state_for_arm(
                        arm, PatientParams(), rng=rule_rng,
                    )
            self._scale_by_rule = {rule.name: float(rule.scale) for rule in self.rules}
            self._base_weight_by_rule = {rule.name: float(rule.weight) for rule in self.rules}
            self._base_weight_sum = sum(rule.weight for rule in self.rules)
        elif self._family == "dose":
            assert self.protocol is not None
            self.items = tuple(_dose_item(t) for t in self.protocol.effective_targets())
            self.source = "Wolever (1991, 1996) — glycemic response to carb dose"
            self.category = "dose_response"
        else:
            self.items = (teacher_meal_glucose_item(),)
            self.source = "simulate_full_body per-patient 75 g meal (wearable glucose)"
            self.category = "mechanism"

    def _update_violation_ema(self, loss_by_spec: dict[str, float]) -> None:
        """Refresh the per-spec discrepancy EMA from this epoch's losses."""
        for sname, loss in loss_by_spec.items():
            prev = self._violation_ema.get(sname)
            self._violation_ema[sname] = (
                loss if prev is None
                else (1.0 - self.ema_alpha) * prev + self.ema_alpha * loss
            )

    def _adaptive_weights_from_ema(self) -> dict[str, float]:
        if not self._violation_ema:
            return {}
        base = (
            self._base_weight_by_rule if self._family == "hinge"
            else self._base_weight_by_spec
        )
        return adaptive_multipliers(
            self._violation_ema, base,
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
        if self._family == "cohort":
            return self._compute_cohort(model, embeddings, ctx)
        if self._family == "hinge":
            return self._compute_hinge(model, embeddings, ctx)
        if self._family == "dose":
            return self._compute_dose(model, embeddings, ctx)
        return self._compute_meal(model, embeddings, ctx)

    def protocol_for_step(self, rng: np.random.Generator) -> DoseResponseProtocol:
        assert self.protocol is not None
        if not self.perturb_protocols or rng.random() < self.perturb_fixed_prob:
            return self.protocol
        return perturb_dose_response_protocol(self.protocol, rng)

    def _compute_cohort(
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
                input_dropout=float(ctx.input_dropout), rng=ctx.rng,
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

    def _hinge_initial_state_fn(self, device: torch.device | str):
        device_key = str(device)

        def norm_center() -> torch.Tensor:
            cached = self._norm_center_tensor.get(device_key)
            if cached is not None:
                return cached
            tensor = torch.tensor(self._norm_center_np, dtype=torch.float32, device=device)
            self._norm_center_tensor[device_key] = tensor
            return tensor

        def fn(rule: PhysiologyRule, arm) -> torch.Tensor:
            if rule.init_mode is InitMode.NORM_CENTER:
                return norm_center()
            key = (arm.label, device_key)
            cached = self._cold_init_tensor.get(key)
            if cached is not None:
                return cached
            arr = self._cold_init_np[arm.label]
            tensor = torch.tensor(arr, dtype=torch.float32, device=device)
            self._cold_init_tensor[key] = tensor
            return tensor

        return fn

    def _update_rule_ema(self, diags: dict[str, dict[str, float]]) -> None:
        for rname, d in diags.items():
            scale = self._scale_by_rule.get(rname, 1.0) or 1.0
            v = float(d.get("violation_mean", 0.0)) / scale
            prev = self._violation_ema.get(rname)
            self._violation_ema[rname] = (
                v if prev is None else (1.0 - self.ema_alpha) * prev + self.ema_alpha * v
            )

    def _compute_hinge(
        self,
        model: nn.Module,
        embeddings: nn.Embedding,
        ctx: SignalContext,
    ) -> SignalResult:
        w = self.weight_at(ctx.epoch)
        if w <= 0 or not self.rules:
            return SignalResult()

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
        del sample_emb_list

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

        override = self._adaptive_weights_from_ema() if self.adaptive else None
        init_fn = self._hinge_initial_state_fn(ctx.device)

        weight_sum = 0.0
        for rule in self.rules:
            rw = rule.weight
            if override is not None and rule.name in override:
                rw = float(override[rule.name])
            weight_sum += rw
        weight_sum = weight_sum or 1e-8

        snapshot = grad_snapshot(ctx)
        B = len(build_emb_list())

        def applied_weight(rule: PhysiologyRule) -> float:
            rw = rule.weight
            if override is not None and rule.name in override:
                rw = float(override[rule.name])
            return rw

        groups: dict[tuple[str, InitMode], list[tuple[PhysiologyRule, object]]] = {}
        for rule in self.rules:
            for arm in rule.arms:
                groups.setdefault((arm.label, rule.init_mode), []).append((rule, arm))
        group_items = list(groups.items())
        if self.arms_per_step > 0 and self.arms_per_step < len(group_items):
            k = int(self.arms_per_step)
            start = self._arm_cursor % len(group_items)
            group_items = [group_items[(start + i) % len(group_items)] for i in range(k)]
            self._arm_cursor = (start + k) % len(groups)

        n_units = {rule.name: len(rule.arms) * B for rule in self.rules}
        visited = {rule.name: 0 for rule in self.rules}
        sq_sum: dict[str, float] = {rule.name: 0.0 for rule in self.rules}
        v_sum: dict[str, float] = {rule.name: 0.0 for rule in self.rules}
        sat_count: dict[str, float] = {rule.name: 0.0 for rule in self.rules}

        for (label, _init_mode), members in group_items:
            rule_repr, arm_repr = members[0]
            init_state = init_fn(rule_repr, arm_repr)
            emb_list = build_emb_list()
            embs = torch.stack(emb_list, dim=0)
            traj = _rollout_arm_batched(
                model, embs, arm_repr, init_state,
                input_dropout=float(ctx.input_dropout), rng=ctx.rng,
            )
            rule_ctx = rule_context_for_arm(arm_repr)

            arm_loss = None
            for rule, _arm in members:
                rw = applied_weight(rule)
                n_R = n_units[rule.name]
                vs = torch.stack(
                    [rule.predicate(traj[b], rule_ctx) for b in range(B)], dim=0,
                )
                contrib = (w * rw / weight_sum) * (vs / rule.scale).pow(2).sum() / n_R
                if not torch.isfinite(contrib):
                    print(
                        f"[SKIP-NONFINITE] signal={self.name} epoch={ctx.epoch} "
                        f"cause=loss rule={rule.name} arm={label} "
                        f"(contribution dropped this epoch)",
                        flush=True,
                    )
                    continue
                arm_loss = contrib if arm_loss is None else arm_loss + contrib
                vs_d = vs.detach()
                sq_sum[rule.name] += float((vs_d / rule.scale).pow(2).sum().item())
                v_sum[rule.name] += float(vs_d.sum().item())
                sat_count[rule.name] += float((vs_d <= 0).float().sum().item())
                visited[rule.name] += B

            if arm_loss is not None:
                arm_loss.backward()
            del traj, embs, emb_list, arm_loss

        raw_weighted_sum = 0.0
        diags: dict[str, dict[str, float]] = {}
        for rule in self.rules:
            n_V = visited[rule.name]
            if n_V == 0:
                continue
            loss_R = sq_sum[rule.name] / n_V
            rw = applied_weight(rule)
            raw_weighted_sum += loss_R * rw
            diags[rule.name] = {
                "violation_mean": v_sum[rule.name] / n_V,
                "satisfied_fraction": sat_count[rule.name] / n_V,
                "applied_weight": float(rw),
            }

        raw_avg = raw_weighted_sum / weight_sum
        finalize_aux_accumulation(ctx, snapshot, signal=self.name)
        if self.adaptive:
            self._update_rule_ema(diags)

        sub_metrics: dict[str, float] = {"n_arm_groups": float(len(group_items))}
        for rname, d in diags.items():
            sub_metrics[f"viol_{rname}"] = d["violation_mean"]
            sub_metrics[f"sat_{rname}"] = d["satisfied_fraction"]
            sub_metrics[f"w_{rname}"] = d["applied_weight"]
        return SignalResult(loss_sum=raw_avg, n_units=1, sub_metrics=sub_metrics)

    def _compute_dose(
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

    def _compute_meal(
        self,
        model: nn.Module,
        embeddings: nn.Embedding,
        ctx: SignalContext,
    ) -> SignalResult:
        w = self.weight_at(ctx.epoch)
        if w <= 0 or not self.targets:
            return SignalResult()

        device = ctx.device
        available = sorted(self.targets)
        if not available:
            return SignalResult()
        k = min(int(self.sample_patients), len(available))
        pids = [int(p) for p in ctx.rng.choice(available, size=k, replace=False)]

        pid_t = torch.tensor(pids, dtype=torch.long, device=device)
        emb = embeddings(pid_t)
        P = emb.shape[0]

        meals = [MealEvent(
            time=float(_MEAL_TIME_MIN),
            carbs=STANDARD_MEAL_CARBS_G,
            fats=STANDARD_MEAL_FATS_G,
            proteins=STANDARD_MEAL_PROTEINS_G,
        )]
        initial = torch.tensor(NORM_CENTER, dtype=torch.float32, device=device)
        initial = initial.unsqueeze(0).repeat(P, 1)
        e_met = model.embedding_projections["metabolic"](emb)
        b_emb = _met._GLUCOSE_BASELINE_MAX_Z * torch.tanh(
            model.metabolic.glucose_baseline_net(e_met).squeeze(-1)
        )
        gb_raw = NORM_CENTER[_GLUCOSE] + float(NORM_SCALE[_GLUCOSE]) * b_emb
        initial = initial.clone()
        initial[:, _GLUCOSE] = gb_raw

        gut = precompute_gut_outputs(
            model, emb, _MEAL_DURATION_MIN, dt=1.0,
            start_time_minutes=_MEAL_START_HOUR * 60.0, meals=meals,
        )
        traj = integrate(
            model, initial, emb, _MEAL_DURATION_MIN, dt=1.0,
            start_time_minutes=_MEAL_START_HOUR * 60.0, meals=meals, gut_outputs=gut,
        )

        glucose = traj[..., _GLUCOSE]
        pre = glucose[:, _MEAL_TIME_MIN - 1]
        post = glucose[:, _MEAL_TIME_MIN:]
        wts = torch.softmax(_SOFTARGMAX_BETA * post, dim=-1)
        peak = (wts * post).sum(dim=-1)
        idx = torch.arange(post.shape[-1], device=device, dtype=post.dtype)
        tpeak = (wts * idx).sum(dim=-1)
        pred_rise = peak - pre

        tgt_rise = torch.tensor(
            [self.targets[p]["glucose_peak_rise"] for p in pids],
            dtype=torch.float32, device=device,
        )
        tgt_tpeak = torch.tensor(
            [self.targets[p]["glucose_time_to_peak_min"] for p in pids],
            dtype=torch.float32, device=device,
        )
        tgt_fasting = torch.tensor(
            [self.targets[p]["glucose_fasting"] for p in pids],
            dtype=torch.float32, device=device,
        )
        loss_fasting = ((pre - tgt_fasting) / _SIGMA_FASTING_MG_DL).pow(2).mean()
        loss_peak = ((pred_rise - tgt_rise) / _SIGMA_PEAK_MG_DL).pow(2).mean()
        loss_tpeak = ((tpeak - tgt_tpeak) / _SIGMA_TPEAK_MIN).pow(2).mean()
        loss = _FASTING_WEIGHT * loss_fasting + loss_peak + _TPEAK_WEIGHT * loss_tpeak

        sub = {
            "n_patients": float(P),
            "peak_rise_pred_mean": float(pred_rise.detach().mean()),
            "peak_rise_target_mean": float(tgt_rise.mean()),
            "peak_rise_mae": float((pred_rise - tgt_rise).abs().detach().mean()),
            "peak_rise_pred_sd": float(pred_rise.detach().std()) if P > 1 else 0.0,
            "peak_rise_target_sd": float(tgt_rise.std()) if P > 1 else 0.0,
            "tpeak_mae": float((tpeak - tgt_tpeak).abs().detach().mean()),
            "fasting_mae": float((pre - tgt_fasting).abs().detach().mean()),
        }
        accumulate_grad(
            w * loss, ctx, signal=self.name,
            extra={"raw_loss": float(loss.detach().item()), "weight": float(w), **sub},
        )
        return SignalResult(loss_sum=float(loss.detach().item()), n_units=1, sub_metrics=sub)
