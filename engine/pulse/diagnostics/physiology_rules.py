"""
Per-rule physiology-rule diagnostics: where each plausibility
constraint stands at a checkpoint, and which modules its gradient
reaches.

Mirrors ``cohort_ablation`` for the new physiology-rule surface
(see ``knowledge/physiology_rules.py``). For each rule we report:

* ``violation_mean`` — the predicate's output averaged across the
  sampled patient embeddings + zero embedding. ``violation_mean == 0``
  means the rule is already satisfied at the checkpoint; positive
  values are the violation amount in marker-native units.
* ``satisfied_fraction`` — fraction of supervised embeddings that
  satisfy the rule.
* ``grad_norm_<module>`` — L2 grad norm landed on each physiology
  module by isolating *just this rule* and back-propping at the
  checkpoint state. Catches dead rules (gradient muted by the model's
  current parameterization) before iter cost is sunk on training
  against them.

Use this to:

* prove that newly-added rules land non-trivial gradients on the
  module owning the marker — avoids the iter-39 vitality-signal
  failure mode where a loss term has zero practical effect.
* surface which markers are downstream of fragile pathways:
  rules whose grad onto Metabolic is order(1e-6) at iter-38 but
  order(1e-2) on a hand-crafted seeded baseline reveal the
  representational reach problem the iter-41 NADIR landmarks
  hit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, cast

import numpy as np
import torch
import torch.nn as nn

from ..knowledge.cohort_types import InitMode
from ..knowledge.full_body import PatientParams
from ..knowledge.physiology_rules import PHYSIOLOGY_RULES, PhysiologyRule
from ..knowledge.textbook_scenarios.base import cold_model_trajectory
from ..physiology_rules_loss import physiology_rule_loss_one_rule
from ..training.embedding_sampler import select_supervised_embeddings
from ..types import EMBEDDING_DIM, NORM_CENTER
from .probe import load_model_from_checkpoint


_MODULE_NAMES = (
    "gut",
    "metabolic",
    "appetite",
    "stress",
    "cardiovascular",
    "thermoreg",
    "respiratory",
)


@dataclass(frozen=True)
class PhysiologyRuleRow:
    name: str
    source: str
    description: str
    arm_labels: tuple[str, ...]
    scale: float
    violation_mean: float
    satisfied_fraction: float
    grad_norms: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PhysiologyRulesReport:
    checkpoint: str
    n_sample_patients: int
    rows: tuple[PhysiologyRuleRow, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "n_sample_patients": self.n_sample_patients,
            "rows": [asdict(r) for r in self.rows],
        }


def _module_param_groups(model: nn.Module) -> dict[str, list[nn.Parameter]]:
    groups: dict[str, list[nn.Parameter]] = {n: [] for n in _MODULE_NAMES}
    for name, p in model.named_parameters():
        for m in _MODULE_NAMES:
            if name.startswith(f"{m}.") or f"embedding_projections.{m}" in name:
                groups[m].append(p)
                break
    return groups


def _grad_norm(params: Iterable[nn.Parameter]) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum().item())
    return float(np.sqrt(total))


def _zero_grads(model: nn.Module) -> None:
    for p in model.parameters():
        p.grad = None


def _initial_state_for_arm(
    rule: PhysiologyRule,
    arm,
    *,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    if rule.init_mode is InitMode.NORM_CENTER:
        return torch.tensor(NORM_CENTER, dtype=torch.float32, device=device)
    rng = np.random.default_rng(seed)
    traj = cold_model_trajectory(
        PatientParams(),
        list(arm.meals),
        arm.duration_min,
        arm.start_hour,
        rng=rng,
    )
    return torch.tensor(traj[0], dtype=torch.float32, device=device)


def physiology_rules_diagnostic(
    model: nn.Module,
    *,
    n_patients: int = 20,
    sample_patients: int = 4,
    seed: int = 0,
    rules: list[PhysiologyRule] | None = None,
) -> tuple[PhysiologyRuleRow, ...]:
    """Per-rule violation + per-module grad-norm at the checkpoint."""
    model.train()
    embeddings = nn.Embedding(n_patients, EMBEDDING_DIM)
    nn.init.zeros_(embeddings.weight)
    rng = np.random.default_rng(seed)
    device = torch.device("cpu")
    emb_list = [
        e.detach().clone().requires_grad_(False)
        for e in select_supervised_embeddings(
            embeddings=embeddings,
            n_patients=n_patients,
            sample_patients=sample_patients,
            rng=rng,
            device=device,
            include_default=True,
        )
    ]
    groups = _module_param_groups(model)
    rule_iter = list(rules) if rules is not None else list(PHYSIOLOGY_RULES)

    rows: list[PhysiologyRuleRow] = []
    for i, rule in enumerate(rule_iter):
        init_state_factory = (
            lambda arm, _rule=rule, _i=i:
                _initial_state_for_arm(_rule, arm, seed=seed + _i, device=device)
        )
        _zero_grads(model)
        loss, v_mean, satisfied = physiology_rule_loss_one_rule(
            cast(nn.Module, model), emb_list, rule, init_state_factory,
        )
        # Only backprop when there's non-zero gradient to find — saves a
        # full backward pass on rules already satisfied at the checkpoint.
        grad_norms = {m: 0.0 for m in _MODULE_NAMES}
        if loss.requires_grad and float(loss.detach().item()) > 0.0:
            loss.backward()
            grad_norms = {m: _grad_norm(params) for m, params in groups.items()}
        rows.append(PhysiologyRuleRow(
            name=rule.name,
            source=rule.source,
            description=rule.description,
            arm_labels=tuple(a.label for a in rule.arms),
            scale=float(rule.scale),
            violation_mean=v_mean,
            satisfied_fraction=satisfied,
            grad_norms=grad_norms,
        ))

    return tuple(rows)


def physiology_rules_for_checkpoint(
    checkpoint: str | Path,
    **kwargs: Any,
) -> PhysiologyRulesReport:
    model, _ = load_model_from_checkpoint(checkpoint)
    rows = physiology_rules_diagnostic(model, **kwargs)
    return PhysiologyRulesReport(
        checkpoint=str(checkpoint),
        n_sample_patients=int(kwargs.get("sample_patients", 4)),
        rows=rows,
    )


def render(report: PhysiologyRulesReport) -> str:
    lines: list[str] = []
    lines.append(f"physiology-rules: {report.checkpoint}")
    lines.append(f"  n_sample_patients={report.n_sample_patients}, n_rules={len(report.rows)}")
    header = (
        f"  {'rule':<42} {'arms':>2} "
        f"{'violation':>10} {'sat%':>5} "
        f"{'||g_gut||':>10} {'||g_met||':>10} {'||g_oth||':>10}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for r in report.rows:
        gut = r.grad_norms.get("gut", 0.0)
        met = r.grad_norms.get("metabolic", 0.0)
        other = sum(
            v for k, v in r.grad_norms.items()
            if k not in ("gut", "metabolic")
        )
        lines.append(
            f"  {r.name[:42]:<42} {len(r.arm_labels):>2} "
            f"{r.violation_mean:>10.3f} {r.satisfied_fraction*100:>4.0f}% "
            f"{gut:>10.3e} {met:>10.3e} {other:>10.3e}"
        )
    return "\n".join(lines)
