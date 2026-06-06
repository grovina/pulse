"""
Per-spec cohort ablation: where each literature finding stands at a checkpoint.

The cohort statistic signal averages a Gaussian-z² loss across many
specs. That averaging makes it invisible whether one spec is moving
the model or whether the bulk loss is being dragged down by two or
three already-saturated specs while the rest stagnate. The PRD's
literature-absorption story is exactly the opposite: each spec
should land a *targeted* gradient on the right module slice.

For a checkpoint, this module reports two per-spec measurements:

* ``z_residual``  — Gaussian z computed on the current model's
  prediction vs the literature target. ``|z| < 1`` means the spec
  is already satisfied within one σ; ``|z| > 3`` means the cohort
  loss is heavily dominated by this spec.
* ``grad_norm_<module>`` — L2 grad norm landed on each physiological
  module (gut, metabolic, appetite, stress, cardiovascular,
  thermoreg, respiratory) by isolating *just this spec* in the
  cohort signal and back-propping at the checkpoint state. This
  reveals which physiology pillar each finding actually pulls on
  — the missing piece for designing future iterations: if "OGTT
  glucose peak" lands no gradient on ``metabolic``, then the model
  has converged into a local minimum where this spec's gradient
  is structurally muted, and we need a structural intervention
  rather than another weight bump.

Use this to:

* prove that newly-added specs land non-trivial gradients (avoids
  silent dead specs);
* identify which specs are constraining the metabolic vs gut
  modules independently of trajectory MSE;
* measure dispersion of pulls across the modular network — a
  healthy ablation report has *some* spec pulling on every
  physiology pillar that owns a marker the spec touches.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, cast

import numpy as np
import torch
import torch.nn as nn

from ..cohort_loss import cohort_statistic_loss_one_spec
from ..knowledge import ALL_COHORT_STATISTICS
from ..knowledge.cohort_types import CohortStatisticSpec, InitMode
from ..knowledge.full_body import PatientParams
from ..knowledge.textbook_scenarios.base import cold_model_trajectory
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
class CohortAblationRow:
    """One spec's measurements at the checkpoint."""

    name: str
    source: str
    marker_id: str
    predicted: float
    target: float
    sigma: float
    z_residual: float
    grad_norms: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class CohortAblationReport:
    checkpoint: str
    n_sample_patients: int
    rows: tuple[CohortAblationRow, ...]

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


def _initial_state_for_spec(
    spec: CohortStatisticSpec,
    *,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    if spec.init_mode is InitMode.NORM_CENTER:
        return torch.tensor(NORM_CENTER, dtype=torch.float32, device=device)
    arm = spec.arms[0]
    rng = np.random.default_rng(seed)
    traj = cold_model_trajectory(
        PatientParams(),
        list(arm.meals),
        arm.duration_min,
        arm.start_hour,
        rng=rng,
    )
    return torch.tensor(traj[0], dtype=torch.float32, device=device)


def cohort_ablation(
    model: nn.Module,
    *,
    n_patients: int = 20,
    sample_patients: int = 4,
    seed: int = 0,
    specs: list[CohortStatisticSpec] | None = None,
) -> tuple[CohortAblationRow, ...]:
    """Per-spec z-residual + per-module grad-norm at the checkpoint."""
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
    spec_iter = list(specs) if specs is not None else list(ALL_COHORT_STATISTICS)

    rows: list[CohortAblationRow] = []
    for i, spec in enumerate(spec_iter):
        init_state = _initial_state_for_spec(spec, seed=seed + i, device=device)
        _zero_grads(model)
        loss, pred, z = cohort_statistic_loss_one_spec(
            cast(nn.Module, model), emb_list, spec, init_state,
        )
        loss.backward()
        grad_norms = {m: _grad_norm(params) for m, params in groups.items()}
        rows.append(CohortAblationRow(
            name=spec.name,
            source=spec.source,
            marker_id=spec.marker_id,
            predicted=pred,
            target=float(spec.target),
            sigma=float(spec.sigma),
            z_residual=z,
            grad_norms=grad_norms,
        ))

    return tuple(rows)


def cohort_ablation_for_checkpoint(
    checkpoint: str | Path,
    **kwargs: Any,
) -> CohortAblationReport:
    model, _ = load_model_from_checkpoint(checkpoint)
    rows = cohort_ablation(model, **kwargs)
    return CohortAblationReport(
        checkpoint=str(checkpoint),
        n_sample_patients=int(kwargs.get("sample_patients", 4)),
        rows=rows,
    )


def render(report: CohortAblationReport) -> str:
    lines: list[str] = []
    lines.append(f"cohort-ablation: {report.checkpoint}")
    lines.append(f"  n_sample_patients={report.n_sample_patients}, n_specs={len(report.rows)}")
    header = (
        f"  {'spec':<40} {'marker':<14} {'pred':>9} {'target':>9} "
        f"{'z':>6} {'||g_gut||':>10} {'||g_met||':>10} {'||g_oth||':>10}"
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
            f"  {r.name[:40]:<40} {r.marker_id:<14} "
            f"{r.predicted:>9.3f} {r.target:>9.3f} {r.z_residual:>6.2f} "
            f"{gut:>10.3e} {met:>10.3e} {other:>10.3e}"
        )
    return "\n".join(lines)
