"""
Identifiability report: which directions of the model the literature constrains.

PLAN Phase 0's map of holes. Every training literature item — cohort statistics
and hinge rules; a sealed item is not training evidence and is left out — is
linearised at a checkpoint for the median person (zero embedding), from the
initial state training starts it in. A cohort contributes one residual
``(pred - target) / sigma``; a rule one per arm, ``violation / scale``. With
``J[i, p] = d r_i / d theta_p`` the Gauss-Newton Fisher information is
``F = J^T J``.

Two parameter sets are reported.

* **Population scalars**, every single-element ``nn.Parameter``. An eigenvalue
  of ``F`` below a relative floor is an *open direction*: a combination of
  parameters no item moves, along which the optimiser is free to drift. Each is
  named by its leading parameters and, for each, the study whose protocol would
  close it (``PhysiologyModule.identified_by``) — the reading list. A parameter
  with weight on the spectrum names the items that hold it.
* **The person axis**, the embedding coordinates. How many directions of the
  person code the literature moves at all; the rest are set by individual data
  alone.

**Conflicts.** Two items both outside one sigma whose descent directions on the
population scalars oppose: to first order, a step that improves one worsens the
other, so the two cannot both be met by moving these parameters. PLAN section 3:
a real conflict is a signal to change structure, not to widen a sigma.

The residual is the raw ``(pred - target) / sigma``, not the shaped residual the
loss uses: the shaped residual is zero inside a band or on the allowed side of a
one-sided target, which would report those parameters as unidentified when the
statistic does depend on them. A rule's hinge is zero once satisfied, and a
satisfied rule truly holds nothing locally.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from ..cohort_loss import (
    _arm_statistic_batched, _arm_window, _excess_area, _rollout_arm_batched, per_person_statistic,
    shaped_residual,
)
from ..knowledge import ALL_COHORT_STATISTICS
from ..knowledge.cohort_types import (
    PER_PERSON_KINDS, CohortArmSpec, CohortStatisticSpec, InitMode, StatisticKind, combine_arms,
)
from ..knowledge.physiology_rules import PHYSIOLOGY_RULES, PhysiologyRule
from ..model import ModularPhysiologyNetwork
from ..physiology_rules_loss import rule_context_for_arm
from ..training.arm_init import student_initial_states
from ..types import EMBEDDING_DIM, NORM_CENTER, SERIES_INDEX, is_observable
from .probe import load_model_from_checkpoint

__all__ = [
    "InitialState",
    "identifiability_for_checkpoint",
    "identifiability_report",
    "population_scalar_params",
    "render",
    "studies_by_parameter",
    "training_initial_states",
]

#: ``(init_mode, arm, item_index) -> [STATE_DIM]`` — where an item's arm starts.
InitialState = Callable[[InitMode, CohortArmSpec, int], torch.Tensor]

_EIGENVALUE_RELATIVE_FLOOR = 1e-6
_TOP_PARAMS_PER_DIRECTION = 3
_TOP_ITEMS_PER_PARAM = 3
_CONDITION_EPS = 1e-30
# Descent directions closer to anti-parallel than this are a conflict.
_CONFLICT_COSINE = -0.5


def population_scalar_params(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Every single-element ``nn.Parameter`` keyed by dotted name, sorted by name."""
    return {
        name: p for name, p in sorted(model.named_parameters()) if p.numel() == 1
    }


def studies_by_parameter(model: torch.nn.Module) -> dict[str, str]:
    """``identified_by`` of every module, under the parameter's dotted name."""
    out: dict[str, str] = {}
    for prefix, module in model.named_modules():
        for local, study in getattr(module, "identified_by", {}).items():
            out[f"{prefix}.{local}" if prefix else local] = study
    return out


def training_initial_states(
    model: torch.nn.Module, device: torch.device | str = "cpu", seed: int = 0,
) -> InitialState:
    """The start rows training uses: marker typicals, or the student's own spin-up.

    The spin-up is the median person (the zero code), run once per
    ``(prefast_hours, start_hour)`` and reused. ``seed`` is unused: the median
    person is not a draw.
    """
    del seed
    cache: dict[tuple[float, float], torch.Tensor] = {}
    zero = torch.zeros(1, EMBEDDING_DIM, device=device)

    def initial(init_mode: InitMode, arm: CohortArmSpec, index: int) -> torch.Tensor:
        del index
        if init_mode is InitMode.NORM_CENTER:
            return torch.tensor(NORM_CENTER, dtype=torch.float32, device=device)
        key = (float(arm.prefast_hours), float(arm.start_hour))
        if key not in cache:
            with torch.no_grad():
                cache[key] = student_initial_states(model, zero, arm)[0].detach()
        return cache[key]

    return initial


def _cohort_residual(
    model: ModularPhysiologyNetwork,
    spec: CohortStatisticSpec,
    embedding: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable ``[(pred - target) / sigma]`` for one member, and the same
    residual after the spec's shape (detached)."""
    series = SERIES_INDEX[spec.marker_id]
    observe = is_observable(spec.marker_id)
    if spec.kind is StatisticKind.RATIO_INCREMENTS:
        # One person: the ratio of the two areas is the ratio of the means.
        assert spec.baseline is not None
        areas = []
        for arm_idx, arm in enumerate(spec.arms):
            traj = _rollout_arm_batched(model, embedding, arm, initial_state, observe=observe)
            areas.append(_excess_area(
                traj, series, _arm_window(spec, arm_idx), spec.baseline,
            ))
        pred = combine_arms(spec.kind, areas)
    elif spec.kind in PER_PERSON_KINDS:
        traj = _rollout_arm_batched(model, embedding, spec.arms[0], initial_state, observe=observe)
        pred = per_person_statistic(traj[:, :, series], spec)
    else:
        per_arm: list[torch.Tensor] = []
        for arm_idx, arm in enumerate(spec.arms):
            traj = _rollout_arm_batched(model, embedding, arm, initial_state, observe=observe)
            per_arm.append(_arm_statistic_batched(
                traj, series, _arm_window(spec, arm_idx), spec.kind, spec.softargmax_beta,
            ))
        pred = combine_arms(spec.kind, per_arm)
    raw = ((pred[0] - float(spec.target)) / float(spec.sigma)).reshape(1)
    shaped = (shaped_residual(pred[0].detach(), spec) / float(spec.sigma)).reshape(1)
    return raw, shaped


def _rule_residual(
    model: ModularPhysiologyNetwork,
    rule: PhysiologyRule,
    embedding: torch.Tensor,
    initial_state: Callable[[CohortArmSpec], torch.Tensor],
) -> torch.Tensor:
    """Differentiable ``[violation / scale]`` per arm for one member."""
    out = []
    for arm in rule.arms:
        traj = _rollout_arm_batched(model, embedding, arm, initial_state(arm))
        out.append(rule.predicate(traj[0], rule_context_for_arm(arm)) / float(rule.scale))
    return torch.stack(out)


def _jacobian_rows(
    residual: torch.Tensor, tensors: list[torch.Tensor],
) -> list[list[torch.Tensor]]:
    rows = []
    for k in range(int(residual.shape[0])):
        grads = torch.autograd.grad(residual[k], tensors, retain_graph=True, allow_unused=True)
        rows.append([
            torch.zeros_like(t) if g is None else g.detach() for g, t in zip(grads, tensors)
        ])
    return rows


def _conflicts(
    items: list[str], residuals: torch.Tensor, jac: torch.Tensor,
) -> list[dict[str, Any]]:
    """Item pairs outside one sigma whose descent directions oppose.

    ``residuals`` are shaped (zero where an item is met), ``jac`` is of the raw
    residual; outside the allowed region the two differ by a constant, so the
    item's descent direction is ``-shaped · J``.
    """
    by_item: dict[str, list[int]] = {}
    for row, name in enumerate(items):
        by_item.setdefault(name, []).append(row)
    names, z, descent = [], [], []
    for name, rows in by_item.items():
        r = residuals[rows]
        miss = float(r.abs().max())
        g = (r.unsqueeze(-1) * jac[rows]).sum(dim=0)
        if miss <= 1.0 or float(g.norm()) == 0.0:
            continue
        names.append(name)
        z.append(miss)
        descent.append(-g / g.norm())
    out: list[dict[str, Any]] = []
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            cosine = float(descent[a] @ descent[b])
            if cosine < _CONFLICT_COSINE:
                out.append({
                    "kind": "gradient", "items": [names[a], names[b]],
                    "abs_z": [z[a], z[b]], "cosine": cosine,
                })
    return sorted(out, key=lambda c: c["cosine"])


def identifiability_report(
    model: ModularPhysiologyNetwork,
    specs: Sequence[CohortStatisticSpec] = ALL_COHORT_STATISTICS,
    rules: Sequence[PhysiologyRule] = PHYSIOLOGY_RULES,
    device: torch.device | str = "cpu",
    initial_state: InitialState | None = None,
) -> dict[str, Any]:
    """Gauss-Newton Fisher spectrum of the literature residuals."""
    model.to(device)
    init = initial_state if initial_state is not None else training_initial_states(model, device)
    params = population_scalar_params(model)
    names = list(params)
    tensors = list(params.values())
    embedding = torch.zeros(1, EMBEDDING_DIM, device=device, requires_grad=True)

    item_of_row: list[str] = []
    residuals: list[torch.Tensor] = []
    scalar_rows: list[torch.Tensor] = []
    person_rows: list[torch.Tensor] = []

    def collect(name: str, raw: torch.Tensor, shaped: torch.Tensor) -> None:
        for row in _jacobian_rows(raw, [*tensors, embedding]):
            item_of_row.append(name)
            scalar_rows.append(torch.stack([g.reshape(()) for g in row[:-1]]))
            person_rows.append(row[-1].reshape(-1))
        residuals.append(shaped.detach())

    for i, spec in enumerate(specs):
        # Spread and correlation are properties of a population, not a residual
        # of the median person.
        if spec.kind in (StatisticKind.SPREAD, StatisticKind.CORRELATION):
            continue
        state = init(spec.init_mode, spec.arms[0], i)
        collect(spec.name, *_cohort_residual(model, spec, embedding, state))
    for j, rule in enumerate(rules):
        index = len(specs) + j
        r = _rule_residual(
            model, rule, embedding, lambda arm, _r=rule, _i=index: init(_r.init_mode, arm, _i),
        )
        collect(rule.name, r, r)

    jac = torch.stack(scalar_rows).to(torch.float64)        # [n_rows, n_params]
    person = torch.stack(person_rows).to(torch.float64)     # [n_rows, EMBEDDING_DIM]
    r_all = torch.cat(residuals).to(torch.float64)

    fisher = jac.T @ jac
    eigvals, eigvecs = torch.linalg.eigh(fisher)  # ascending
    eig = eigvals.tolist()
    max_eig = max(eig[-1], 0.0)
    condition = max_eig / max(eig[0], _CONDITION_EPS)
    threshold = _EIGENVALUE_RELATIVE_FLOOR * max_eig

    studies = studies_by_parameter(model)

    def named(j: int) -> dict[str, Any]:
        return {"param": names[j], "study": studies.get(names[j])}

    open_directions = []
    for i, e in enumerate(eig):
        if e >= threshold:
            break
        vec = eigvecs[:, i]
        top = torch.argsort(vec.abs(), descending=True)[:_TOP_PARAMS_PER_DIRECTION]
        open_directions.append({
            "eigenvalue": e,
            "top_params": [{**named(j), "weight": float(vec[j])} for j in top.tolist()],
        })

    diagonal = torch.diagonal(fisher)
    coverage: dict[str, dict[str, Any]] = {}
    for j, name in enumerate(names):
        col = jac[:, j].abs()
        held = float(diagonal[j]) >= threshold
        top_rows = torch.argsort(col, descending=True)[:_TOP_ITEMS_PER_PARAM].tolist() if held else []
        coverage[name] = {
            "fisher": float(diagonal[j]),
            "held_by": list(dict.fromkeys(item_of_row[k] for k in top_rows if float(col[k]) > 0.0)),
            "study": studies.get(name),
        }

    person_eig = torch.linalg.eigvalsh(person.T @ person).tolist()
    person_floor = _EIGENVALUE_RELATIVE_FLOOR * max(max(person_eig), 0.0)

    return {
        "param_names": names,
        "n_items": len(specs) + len(rules),
        "n_residuals": int(jac.shape[0]),
        "eigenvalues": eig,
        "condition_number": condition,
        "n_open": len(open_directions),
        "open_directions": open_directions,
        "coverage": coverage,
        "person_axis": {
            "dim": EMBEDDING_DIM,
            "n_constrained": sum(1 for e in person_eig if e >= person_floor and e > 0.0),
            "eigenvalues": person_eig,
        },
        "conflicts": _conflicts(item_of_row, r_all, jac),
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
        f"  n_items={report['n_items']}, n_residuals={report['n_residuals']}, "
        f"n_params={len(report['param_names'])}, condition_number={report['condition_number']:.3e}",
    )
    lines.append("  Fisher eigenvalues (ascending, log10; <= -30 shown as exact zero):")
    row: list[str] = []
    for i, e in enumerate(eig):
        row.append("   zero" if e <= 1e-30 else f"{torch.log10(torch.tensor(e)).item():>7.2f}")
        if len(row) == 10 or i == len(eig) - 1:
            lines.append("    " + " ".join(row))
            row = []

    lines.append(f"  open directions ({report['n_open']}):")
    for d in report["open_directions"]:
        params = ", ".join(
            f"{p['param']} ({p['weight']:+.2f}; {p['study'] or 'no study filed'})"
            for p in d["top_params"]
        )
        lines.append(f"    lambda={d['eigenvalue']:>10.3e}  {params}")

    weakest = sorted(report["coverage"].items(), key=lambda kv: kv[1]["fisher"])[:10]
    lines.append("  weakest parameters (Fisher diagonal):")
    for name, c in weakest:
        held = ", ".join(c["held_by"]) or (c["study"] or "no study filed")
        lines.append(f"    {name:<44} {c['fisher']:>10.3e}  {held}")

    person = report["person_axis"]
    lines.append(f"  person axis: {person['n_constrained']} of {person['dim']} directions constrained")
    lines.append(f"  conflicts ({len(report['conflicts'])}):")
    for c in report["conflicts"]:
        a, b = c["items"]
        lines.append(f"    {a} <-> {b}  cos={c['cosine']:+.2f}  |z|={c['abs_z'][0]:.1f}/{c['abs_z'][1]:.1f}")
    return "\n".join(lines)
