"""
Identifiability report: which population scalars the literature constrains.

The cohort statistics are the only supervision a population scalar (``log_k_ii``,
``log_k_cort``, ...) ever sees. This diagnostic linearizes them at a checkpoint:
for the median person (zero embedding) it takes each spec's signed residual
``r_s = (pred_s - target_s) / sigma_s`` and its Jacobian ``J[s, p] = d r_s / d theta_p``
with respect to every scalar population parameter. The Gauss-Newton Fisher
information is ``F = J^T J``.

* Eigenvalues of ``F`` near zero are *flat directions*: combinations of parameters
  that no cohort statistic moves, so the literature cannot identify them and the
  optimizer is free to drift along them.
* ``per_param_fisher_diagonal`` is how strongly the specs constrain each parameter
  taken alone; a parameter with a zero diagonal is untouched by every spec.

The residual is the raw ``(pred - target) / sigma``, not the shaped residual the
loss uses: the shaped residual is zero inside a band / on the allowed side of a
one-sided target, which would report those parameters as unidentified when the
statistic does depend on them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch

from ..cohort_loss import (
    _arm_window,
    _arm_statistic_batched,
    _rollout_arm_batched,
    norm_center_initial_state,
)
from ..knowledge import ALL_COHORT_STATISTICS
from ..knowledge.cohort_types import CohortStatisticSpec, StatisticKind
from ..model import ModularPhysiologyNetwork
from ..types import EMBEDDING_DIM, MARKER_INDEX
from .probe import load_model_from_checkpoint

__all__ = [
    "identifiability_for_checkpoint",
    "identifiability_report",
    "population_scalar_params",
    "render",
]

_EIGENVALUE_RELATIVE_FLOOR = 1e-6
_MIN_FLAT_DIRECTIONS = 5
_TOP_PARAMS_PER_DIRECTION = 3
_CONDITION_EPS = 1e-30


def population_scalar_params(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Every single-element ``nn.Parameter`` keyed by dotted name, sorted by name."""
    return {
        name: p for name, p in sorted(model.named_parameters()) if p.numel() == 1
    }


def _spec_residual(
    model: ModularPhysiologyNetwork,
    spec: CohortStatisticSpec,
    embedding: torch.Tensor,
    initial_state: torch.Tensor,
) -> torch.Tensor:
    """Differentiable ``(pred - target) / sigma`` for one member."""
    marker = MARKER_INDEX[spec.marker_id]
    per_arm: list[torch.Tensor] = []
    for arm_idx, arm in enumerate(spec.arms):
        traj = _rollout_arm_batched(model, embedding, arm, initial_state)
        per_arm.append(_arm_statistic_batched(
            traj, marker, _arm_window(spec, arm_idx), spec.kind, spec.softargmax_beta,
        ))
    if spec.kind in (StatisticKind.DELTA_MEANS, StatisticKind.DELTA_PEAKS):
        pred = per_arm[1] - per_arm[0]
    else:
        pred = per_arm[0]
    return (pred[0] - float(spec.target)) / float(spec.sigma)


def identifiability_report(
    model: ModularPhysiologyNetwork,
    specs: Sequence[CohortStatisticSpec] = ALL_COHORT_STATISTICS,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Gauss-Newton Fisher spectrum of the cohort residuals over the scalar parameters."""
    model.to(device)
    params = population_scalar_params(model)
    names = list(params)
    tensors = list(params.values())
    embedding = torch.zeros(1, EMBEDDING_DIM, device=device)
    state_fn = norm_center_initial_state(device)

    rows: list[torch.Tensor] = []
    for spec in specs:
        r = _spec_residual(model, spec, embedding, state_fn(spec))
        grads = torch.autograd.grad(r, tensors, retain_graph=True, allow_unused=True)
        rows.append(torch.stack([
            torch.zeros((), device=device) if g is None else g.detach().reshape(())
            for g in grads
        ]))
    jac = torch.stack(rows).to(torch.float64)  # [n_specs, n_params]

    fisher = jac.T @ jac
    eigvals, eigvecs = torch.linalg.eigh(fisher)  # ascending
    eig = eigvals.tolist()
    max_eig = max(eig[-1], 0.0)
    condition = max_eig / max(eig[0], _CONDITION_EPS)

    threshold = _EIGENVALUE_RELATIVE_FLOOR * max_eig
    flat_idx = [i for i, e in enumerate(eig) if e < threshold]
    if not flat_idx:
        flat_idx = list(range(min(_MIN_FLAT_DIRECTIONS, len(eig))))
    flat_directions = []
    for i in flat_idx:
        vec = eigvecs[:, i]
        top = torch.argsort(vec.abs(), descending=True)[:_TOP_PARAMS_PER_DIRECTION]
        flat_directions.append({
            "eigenvalue": eig[i],
            "top_params": [(names[j], float(vec[j])) for j in top.tolist()],
        })

    return {
        "param_names": names,
        "n_specs": len(specs),
        "eigenvalues": eig,
        "condition_number": condition,
        "flat_directions": flat_directions,
        "per_param_fisher_diagonal": {n: float(fisher[i, i]) for i, n in enumerate(names)},
        "max_abs_jacobian": float(jac.abs().max()) if jac.numel() else 0.0,
    }


def identifiability_for_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
    **kwargs: Any,
) -> dict[str, Any]:
    model, _ = load_model_from_checkpoint(checkpoint_path)
    report = identifiability_report(model, device=device, **kwargs)
    report["checkpoint"] = str(checkpoint_path)
    return report


def render(report: dict[str, Any]) -> str:
    eig = report["eigenvalues"]
    lines: list[str] = []
    title = report.get("checkpoint", "<model>")
    lines.append(f"identifiability: {title}")
    lines.append(
        f"  n_specs={report['n_specs']}, n_params={len(report['param_names'])}, "
        f"condition_number={report['condition_number']:.3e}",
    )
    lines.append("  Fisher eigenvalues (ascending, log10; <= -30 shown as exact zero):")
    row: list[str] = []
    for i, e in enumerate(eig):
        row.append("   zero" if e <= 1e-30 else f"{torch.log10(torch.tensor(e)).item():>7.2f}")
        if len(row) == 10 or i == len(eig) - 1:
            lines.append("    " + " ".join(row))
            row = []

    lines.append(f"  flat directions ({len(report['flat_directions'])}):")
    for d in report["flat_directions"]:
        params = ", ".join(f"{n} ({w:+.2f})" for n, w in d["top_params"])
        lines.append(f"    lambda={d['eigenvalue']:>10.3e}  {params}")

    diag = sorted(report["per_param_fisher_diagonal"].items(), key=lambda kv: kv[1])
    lines.append("  per-parameter Fisher diagonal (weakest 10):")
    for name, v in diag[:10]:
        lines.append(f"    {name:<44} {v:>10.3e}")
    return "\n".join(lines)
