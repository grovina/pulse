"""
Physiology-rules signal — per-trajectory inequality / relational
supervision against literature-derived plausibility constraints.

Companion to ``CohortStatisticSignal``. Cohort statistics average
across N synthetic patients, then score the population mean — leaves
escape hatches where some patients have strong dynamics and others
flat. Physiology rules score every trajectory individually with
hinge losses, so the per-rollout structure has nowhere to hide.

Operationalizes the PRD's *Calibrated supervision* principle:
encode every plausibility constraint the literature supports
(directions, signs, timing windows, relations), hinge-shaped so
gradient stops at satisfaction.

Per epoch: select supervised embeddings (sampled patient embeddings
plus the zero "default" embedding the textbook benchmark queries),
roll out each rule's arm, score the predicate, hinge-loss the
violation. One backward + step at the end (cadence matches the
cohort signal).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from ..knowledge.cohort_types import InitMode
from ..knowledge.full_body import PatientParams
from ..knowledge.physiology_rules import PhysiologyRule
from ..knowledge.textbook_scenarios.base import cold_model_trajectory
from ..cohort_loss import _rollout_arm_batched
from ..physiology_rules_loss import (
    physiology_rules_epoch_loss,  # noqa: F401 — kept for non-training callers / tests
    rule_context_for_arm,
)
from ..types import NORM_CENTER
from .embedding_sampler import select_supervised_embeddings
from .safe_step import finalize_aux_accumulation, grad_snapshot
from .signals import SignalContext, SignalResult, TrainingSignal, WeightSchedule


@dataclass
class PhysiologyRulesSignal(TrainingSignal):
    """Per-trajectory hinge supervision against literature-derived rules.

    ``rules`` is the list of ``PhysiologyRule`` to evaluate each epoch.
    Each rule's ``init_mode`` decides whether the rollout starts from a
    cold-model trajectory or NORM_CENTER (mirrors CohortStatisticSpec).

    Adaptive weighting (iter 67, ``adaptive=True``)
    ------------------------------------------------
    With ~60 rules at a single signal weight of 0.05, each rule sees
    a ~8e-4 pull — uniformly small whether it's badly violated or
    perfectly satisfied. Adaptive weighting reweights rules at
    aggregation time so the per-rule coefficient is proportional to the
    rule's current EMA-violation: violated rules pull harder, satisfied
    rules fade. The total ``rule.weight``-sum (and thus the signal-level
    weight) is preserved across reweightings — adaptive only rebalances
    *between* rules, never against other signals.

    The EMA tracks ``violation_mean`` with smoothing ``ema_alpha``
    (default 0.1 ⇒ ~10-epoch half-life), initialised on the first epoch
    that produces diagnostics so we don't start from a zero baseline
    that would zero-weight every rule before any data has flowed.
    """

    rules: list[PhysiologyRule] = field(default_factory=list)
    n_patients: int = 0
    sample_patients: int = 4
    include_default_embedding: bool = True
    weight: WeightSchedule = field(default_factory=lambda: WeightSchedule(0.0))
    cold_init_seed: int = 0
    # Iter 67: violation-proportional reweighting toggle.
    adaptive: bool = False
    ema_alpha: float = 0.1
    # Floor on the per-rule EMA (relative to the population max) so
    # satisfied rules retain a small residual pull. Without a floor the
    # adaptive scheme can drive a rule to weight=0, which is irreversible
    # — once a rule is satisfied it stops generating gradient and cannot
    # recover if drift later re-violates it.
    ema_floor_frac: float = 0.02

    name: str = "physiology_rules"
    source: str = "literature_plausibility_constraints"
    category: str = "mechanism"

    def __post_init__(self) -> None:
        # Pre-compute cold-model initial states per (rule, arm). Each
        # arm has its own protocol so the cold trajectory differs per
        # arm; keying by arm.label keeps this independent of how many
        # rules share an arm.
        self._cold_init_np: dict[str, np.ndarray] = {}
        prng = np.random.default_rng(self.cold_init_seed)
        for rule in self.rules:
            if rule.init_mode is not InitMode.COLD:
                continue
            for arm in rule.arms:
                if arm.label in self._cold_init_np:
                    continue
                rule_rng = np.random.default_rng(prng.integers(0, 2**32))
                traj = cold_model_trajectory(
                    PatientParams(),
                    list(arm.meals),
                    arm.duration_min,
                    arm.start_hour,
                    rng=rule_rng,
                )
                self._cold_init_np[arm.label] = traj[0]
        self._cold_init_tensor: dict[tuple[str, str], torch.Tensor] = {}
        self._norm_center_np = np.array(NORM_CENTER, dtype=np.float32)
        self._norm_center_tensor: dict[str, torch.Tensor] = {}
        # Iter 67: per-rule violation EMA + total-weight sum to renormalise
        # adaptive weights against the rule list's fixed budget. Initialised
        # lazily on the first epoch the rules produce diagnostics so a cold
        # start doesn't zero-weight everything.
        self._violation_ema: dict[str, float] = {}
        self._base_weight_sum: float = sum(rule.weight for rule in self.rules)

    def _update_violation_ema(self, diags: dict[str, dict[str, float]]) -> None:
        """Refresh the per-rule violation EMA from this epoch's diagnostics."""
        for rname, d in diags.items():
            v = float(d.get("violation_mean", 0.0))
            prev = self._violation_ema.get(rname)
            self._violation_ema[rname] = (
                v if prev is None else (1.0 - self.ema_alpha) * prev + self.ema_alpha * v
            )

    def _adaptive_weights_from_ema(self) -> dict[str, float]:
        """Adaptive per-rule weights from the current EMA.

        Preserves ``sum(rule.weight)`` so the signal's overall pull on the
        optimizer stays put — only the inter-rule distribution changes.
        Rules absent from the EMA (no diagnostics yet) keep their base
        weight via the override-fallback path in
        ``physiology_rules_epoch_loss``.
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

    def _initial_state_fn(self, device: torch.device | str):
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

    def compute(
        self,
        model: nn.Module,
        embeddings: nn.Embedding,
        ctx: SignalContext,
    ) -> SignalResult:
        w = self.weight_at(ctx.epoch)
        if w <= 0 or not self.rules:
            return SignalResult()

        # Iter 68 r7: per-rule gradient accumulation. r6 crashed phase 2 ep 50
        # with SIGKILL during physiology_rules — same anti-pattern that crashed
        # cohort_statistic in r4: physiology_rules_epoch_loss summed all 60
        # rules' losses into one composite tensor before a single backward, so
        # 60 autograd graphs lived in memory simultaneously → OOM. Mirror
        # cohort_signal's r6 fix: sample embedding ids once, rebuild emb_list
        # fresh per rule (avoids the r5 shared-graph crash), per-rule
        # weighted.backward() drops each graph promptly. PyTorch's additive
        # .grad semantics make this identical to the old composite backward
        # by linearity of differentiation; one optimizer.step at the end.
        # NaN-abort + grad clip inlined to mirror safe_step's semantics
        # (per-rule safe_step would optimizer.step 60 times per epoch and
        # change the dynamics). physiology_rules_epoch_loss is kept for
        # diagnostics / non-training callers.
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
        del sample_emb_list  # rebuild fresh per rule below

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

        # Iter 67: in adaptive mode, supply the current EMA-derived per-rule
        # weights. The first epoch the signal fires has no EMA yet — pass
        # override=None, run with base weights, and initialise the EMA from
        # the returned diags. Subsequent epochs use the updated EMA.
        override = self._adaptive_weights_from_ema() if self.adaptive else None
        init_fn = self._initial_state_fn(ctx.device)

        weight_sum = 0.0
        for rule in self.rules:
            rw = rule.weight
            if override is not None and rule.name in override:
                rw = float(override[rule.name])
            weight_sum += rw
        weight_sum = weight_sum or 1e-8

        # iter 79: ARM-MAJOR rollout dedup. The ~60 rules reference only ~10
        # distinct arm protocols, but the iter-68 rule-major loop re-rolled each
        # arm once PER RULE (~115 integrate() calls/epoch). Here we roll each
        # distinct (arm.label, init_mode) ONCE and score every rule that uses it
        # against the shared trajectory (~10 rollouts/epoch). This is
        # GRADIENT-IDENTICAL to the per-rule path: a rule's loss is the mean
        # over (arms × patients) of squared violations, and mean/sum are linear,
        # so each arm contributes a separable partial loss; one backward per
        # arm-group accumulates into .grad exactly as one backward per rule did
        # (PyTorch additive-grad — the same linearity the iter-68 comment
        # relies on). Peak memory is LOWER: one arm rollout graph is live at a
        # time (vs a multi-arm rule's whole set before).
        #
        # ``emb_list`` is still rebuilt fresh per arm-group — sharing the
        # embeddings(pid) lookup tensors across independent backwards is the r5
        # shared-graph crash (the first backward frees the emb→weight subgraph).
        # ``snapshot`` (pre-first-backward) backs the isolation policy: a
        # non-finite contribution is rolled back + dropped, never aborts.
        snapshot = grad_snapshot(ctx)

        # Number of supervised embeddings this epoch (constant across rules);
        # a rule's reported loss averages over (its arms × B) violations.
        B = len(build_emb_list())

        def applied_weight(rule: PhysiologyRule) -> float:
            rw = rule.weight
            if override is not None and rule.name in override:
                rw = float(override[rule.name])
            return rw

        # (arm.label, init_mode) -> [(rule, arm)]. Rules sharing a key share
        # protocol AND init state by construction (cold init is keyed by
        # arm.label; norm_center is arm-independent), so one rollout serves all.
        groups: dict[tuple[str, InitMode], list[tuple[PhysiologyRule, object]]] = {}
        for rule in self.rules:
            for arm in rule.arms:
                groups.setdefault((arm.label, rule.init_mode), []).append((rule, arm))

        # Per-rule detached accumulators — reconstruct the exact per-rule
        # reported loss + diagnostics from the arm-grouped passes.
        n_units = {rule.name: len(rule.arms) * B for rule in self.rules}
        sq_sum: dict[str, float] = {rule.name: 0.0 for rule in self.rules}
        v_sum: dict[str, float] = {rule.name: 0.0 for rule in self.rules}
        sat_count: dict[str, float] = {rule.name: 0.0 for rule in self.rules}

        for (label, _init_mode), members in groups.items():
            rule_repr, arm_repr = members[0]
            init_state = init_fn(rule_repr, arm_repr)
            emb_list = build_emb_list()
            embs = torch.stack(emb_list, dim=0)  # [B, EMB]
            traj = _rollout_arm_batched(model, embs, arm_repr, init_state)  # [B, T, STATE]
            rule_ctx = rule_context_for_arm(arm_repr)

            arm_loss = None
            for rule, _arm in members:
                rw = applied_weight(rule)
                n_R = n_units[rule.name]
                vs = torch.stack(
                    [rule.predicate(traj[b], rule_ctx) for b in range(B)], dim=0,
                )  # [B]
                # Partial contribution of THIS arm to the rule's signal-level
                # loss: (w·rw/weight_sum) · (Σ_b (v/scale)²) / n_R. Summing
                # across the rule's arms reconstructs (w·rw/weight_sum)·loss_R.
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

            if arm_loss is not None:
                arm_loss.backward()
            del traj, embs, emb_list, arm_loss

        # Reconstruct per-rule reported loss + diagnostics from the accumulators.
        raw_weighted_sum = 0.0
        diags: dict[str, dict[str, float]] = {}
        for rule in self.rules:
            n_R = n_units[rule.name] or 1
            loss_R = sq_sum[rule.name] / n_R
            rw = applied_weight(rule)
            raw_weighted_sum += loss_R * rw
            diags[rule.name] = {
                "violation_mean": v_sum[rule.name] / n_R,
                "satisfied_fraction": sat_count[rule.name] / n_R,
                "applied_weight": float(rw),
            }

        raw_avg = raw_weighted_sum / weight_sum

        finalize_aux_accumulation(ctx, snapshot, signal=self.name)

        if self.adaptive:
            self._update_violation_ema(diags)

        sub_metrics: dict[str, float] = {}
        for rname, d in diags.items():
            sub_metrics[f"viol_{rname}"] = d["violation_mean"]
            sub_metrics[f"sat_{rname}"] = d["satisfied_fraction"]
            sub_metrics[f"w_{rname}"] = d["applied_weight"]
        return SignalResult(
            loss_sum=raw_avg,
            n_units=1,
            sub_metrics=sub_metrics,
        )
