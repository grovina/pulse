"""
Differentiable physiology-rule losses — per-trajectory hinge supervision.

Companion to ``cohort_loss.py``. Cohort statistics average across N
sampled patients before scoring; physiology rules score every
trajectory and hinge-loss the violation. The PRD's *Calibrated
supervision* principle in concrete form: encode every plausibility
constraint we have, hinge-shaped so the loss vanishes past
satisfaction (no enforcement beyond what we know).

Loss form per rule (per embedding):

    loss = (predicate(traj, ctx) / rule.scale) ** 2

The predicate already returns a non-negative violation in marker-
native units (the rule author applies ReLU inside the predicate).
Squared scale matches CohortStatisticSpec's Gaussian-z² so weights
compose with the rest of the training pipeline without rescaling.

Returns aggregate loss + per-rule violation diagnostics for logging.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .cohort_loss import _rollout_arm_batched
from .knowledge.physiology_rules import PhysiologyRule, RuleContext
from .types import MARKER_INDEX, NORM_CENTER

__all__ = [
    "physiology_rule_loss_one_rule",
    "physiology_rules_epoch_loss",
    "rule_context_for_arm",
]


def rule_context_for_arm(arm) -> RuleContext:
    """Build the per-arm context used by predicates. ``dt=1.0`` in the
    rollout pipeline pins ``step_min=1.0``; the marker index is the
    canonical one from ``types.MARKER_INDEX``."""
    return RuleContext(
        marker_index=dict(MARKER_INDEX),
        step_min=1.0,
        arm=arm,
    )


def physiology_rule_loss_one_rule(
    model: nn.Module,
    embeddings_to_supervise: list[torch.Tensor],
    rule: PhysiologyRule,
    initial_state_for_arm,
) -> tuple[torch.Tensor, float, float]:
    """Mean squared-hinge loss across (arms × embeddings).

    For multi-arm rules, evaluates the predicate under each arm's
    rollout and averages. Single-arm rules are the degenerate case
    of one arm. Embeddings are stacked into one batched forward pass
    per arm. Returns ``(loss_tensor, violation_mean_detached,
    satisfied_fraction_detached)`` where the satisfied fraction is
    over (arms × embeddings) — every (arm, embedding) pair must
    satisfy to count.

    ``initial_state_for_arm`` is a callable that returns the
    integration starting state for a given ``CohortArmSpec`` (signal
    layer pre-builds these so the cold-model trajectory cost is paid
    once per arm, not per epoch).
    """
    if not embeddings_to_supervise:
        device = next(model.parameters()).device
        zero = torch.tensor(0.0, device=device)
        return zero, 0.0, 1.0

    embs = torch.stack(embeddings_to_supervise, dim=0)  # [B, EMB]

    # Per-(arm, embedding) violations; concatenated across arms then
    # averaged. Mean (rather than max) so partial satisfaction reduces
    # but doesn't eliminate gradient — keeps a model that satisfies
    # one arm but not others under tension.
    all_v: list[torch.Tensor] = []
    for arm in rule.arms:
        init_state = initial_state_for_arm(arm)
        traj = _rollout_arm_batched(model, embs, arm, init_state)  # [B, T, STATE]
        ctx = rule_context_for_arm(arm)
        for b in range(int(embs.shape[0])):
            all_v.append(rule.predicate(traj[b], ctx))
    v = torch.stack(all_v, dim=0)  # [arms × B]

    loss = (v / rule.scale).pow(2).mean()
    v_detached = v.detach()
    return (
        loss,
        float(v_detached.mean().item()),
        float((v_detached <= 0).float().mean().item()),
    )


def physiology_rules_epoch_loss(
    model: nn.Module,
    embeddings_to_supervise: list[torch.Tensor],
    rules: list[PhysiologyRule],
    device: torch.device | str,
    initial_state_fn=None,
    rule_weight_override: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, dict[str, float]]]:
    """Weighted average loss across all rules and the supplied embeddings.

    ``initial_state_fn`` returns the integration starting state for a
    given (rule, arm). Defaults to a NORM_CENTER tensor when unset;
    pass a cold-model factory to mirror benchmark / cohort conditions.

    ``rule_weight_override`` optionally substitutes the per-rule
    coefficient at aggregation time (iter 67 adaptive weighting). The
    keys are ``rule.name``; missing rules fall back to ``rule.weight``.
    The substituted weights replace the per-rule coefficient on both
    numerator and denominator of the weighted-mean — total normalised
    pull on the supervision objective is preserved across reweightings.

    Returns ``(loss, per_rule_diagnostics)`` where each rule's diag is
    ``{"violation_mean": float, "satisfied_fraction": float,
       "applied_weight": float}``.
    """
    if not rules or not embeddings_to_supervise:
        return torch.tensor(0.0, device=device), {}
    if initial_state_fn is None:
        norm = torch.tensor(NORM_CENTER, dtype=torch.float32, device=device)
        initial_state_fn = lambda _rule, _arm: norm
    total = torch.tensor(0.0, device=device)
    weight_sum = 0.0
    diags: dict[str, dict[str, float]] = {}
    for rule in rules:
        # Curry the (rule, arm) → state factory into an arm-only one
        # for physiology_rule_loss_one_rule.
        rule_init_for_arm = lambda arm, _rule=rule: initial_state_fn(_rule, arm)
        loss, v_mean, satisfied = physiology_rule_loss_one_rule(
            model, embeddings_to_supervise, rule, rule_init_for_arm,
        )
        w = rule.weight
        if rule_weight_override is not None and rule.name in rule_weight_override:
            w = float(rule_weight_override[rule.name])
        total = total + w * loss
        weight_sum += w
        diags[rule.name] = {
            "violation_mean": v_mean,
            "satisfied_fraction": satisfied,
            "applied_weight": float(w),
        }
    return total / max(weight_sum, 1e-8), diags
