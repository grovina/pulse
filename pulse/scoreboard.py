"""
The scoreboard (PLAN section 4): the continuous board every checkpoint is scored on.

There is no gate. A run that regresses on one line tells us something; it is not
rejected. Three lines:

* **Fit to held-out literature** — scored the way training scores an item,
  reported as the median miss in units of each item's own sigma and the shares
  within one and two sigma, overall and by protocol family. Two held-out lines
  are kept apart: ``sealed`` is the items of sealed *papers*, which no part of
  the model has seen; ``held_out_items`` is the items held out from papers that
  also train (``sealing.HELD_OUT_ITEMS``), which share a paper, an arm or a
  starting constant with training and so flatter the model. Points of one curve
  and items of one paper are not independent, so each line reports the number
  of papers and a 90% interval on its median from a bootstrap that resamples
  papers. The training corpus is scored the same way and logged beside them.
  Given ``baseline=True`` every line is also scored for the untrained student,
  so a checkpoint is read against where it started.
* **Individual prediction skill** — calibrate-then-predict on held-out people,
  against the population-median person and the person's own running mean
  (``benchmark.individual_skill``); attached when a ruler was run.
* **How open the model still is** — the open directions of the identifiability
  report, each named with the study that would close it.

Beside the lines the board keeps what PLAN Phase 4 asks to be visible at every
checkpoint: a residual for every item, the conflicts (two items whose targets
disagree beyond their sigmas; two items the model cannot both meet), and, given
the previous board, the breaks (an item that moved from inside one sigma to well
outside two).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

from .cohort_loss import cohort_statistic_loss_group
from .diagnostics.identifiability import (
    InitialState, identifiability_report, training_initial_states,
)
from .knowledge.cohort_statistics import ALL_COHORT_STATISTICS, SEALED_COHORT_STATISTICS
from .knowledge.cohort_types import CohortStatisticSpec, TargetShape
from .knowledge.evidence import OPERATOR_OF_STATISTIC, OperatorKind
from .knowledge.physiology_rules import PHYSIOLOGY_RULES, SEALED_PHYSIOLOGY_RULES, PhysiologyRule
from .knowledge.protocol_family import protocol_family
from .knowledge.sealing import HELD_OUT_ITEMS, SHARES_WITH_TRAINING, paper_is_sealed
from .model import ModularPhysiologyNetwork
from .physiology_rules_loss import physiology_rule_loss_one_rule
from .types import EMBEDDING_DIM

__all__ = [
    "ItemResidual",
    "attach_individual_skill",
    "breaks",
    "fit_summary",
    "population_members",
    "render",
    "score_cohorts",
    "score_rules",
    "scoreboard",
    "scoreboard_for_checkpoint",
    "scoreboard_object",
    "scoreboard_path",
    "target_conflicts",
]

# A break: inside this many sigma on the previous board, beyond the second now.
_INSIDE = 1.0
_WELL_OUTSIDE = 2.0


@dataclass(frozen=True)
class ItemResidual:
    """One item at one checkpoint.

    ``z`` is in units of the item's own sigma: the shaped residual of the
    population mean over ``sigma`` for a cohort (zero inside a band or on the
    allowed side), the root-mean-square ``violation / scale`` over arms and
    members for a rule. ``predicted`` / ``target`` are native units; a rule's
    target is no violation.
    """

    name: str
    source: str
    paper: str
    family: str
    operator: str
    marker_id: str
    sealed: bool
    predicted: float
    target: float
    sigma: float
    z: float


def population_members(
    n_draws: int, seed: int, *, device: torch.device | str = "cpu",
) -> tuple[list[torch.Tensor], int]:
    """The people a population item is scored on, laid out as training lays them.

    ``n_draws`` fresh codes from the factor prior, then the median person —
    the zero code — last. Returns the members and how many lead rows are
    draws (``cohort_loss.score_batch_statistic``'s ``n_population``). The
    zero code is not a draw.
    """
    from .training.embedding_sampler import draw_people

    rng = np.random.default_rng(seed)
    codes = draw_people(max(int(n_draws), 0), rng, device)
    members = [codes[i].detach().clone() for i in range(int(codes.shape[0]))]
    members.append(torch.zeros(EMBEDDING_DIM, device=device))
    return members, len(members) - 1


def score_cohorts(
    model: nn.Module,
    specs: Sequence[CohortStatisticSpec],
    members: list[torch.Tensor],
    n_population: int,
    initial_state: InitialState,
) -> list[ItemResidual]:
    """Every spec on ``members``, one rollout per shared arm protocol."""
    index = {spec.name: i for i, spec in enumerate(specs)}
    groups: dict[tuple, list[CohortStatisticSpec]] = {}
    for spec in specs:
        groups.setdefault(spec.arms, []).append(spec)
    scored: dict[str, tuple[float, float]] = {}
    with torch.no_grad():
        for group in groups.values():
            states = [initial_state(s.init_mode, s.arms[0], index[s.name]) for s in group]
            out = cohort_statistic_loss_group(model, members, group, states, n_population=n_population)
            scored.update({name: (pred, z) for name, (_loss, pred, z) in out.items()})
    return [
        ItemResidual(
            name=spec.name, source=spec.source, paper=spec.paper,
            family=protocol_family(spec.arms).value,
            operator=OPERATOR_OF_STATISTIC[spec.kind].value, marker_id=spec.marker_id,
            sealed=spec.sealed, predicted=scored[spec.name][0], target=float(spec.target),
            sigma=float(spec.sigma), z=scored[spec.name][1],
        )
        for spec in specs
    ]


def score_rules(
    model: nn.Module,
    rules: Sequence[PhysiologyRule],
    members: list[torch.Tensor],
    initial_state: InitialState,
    first_index: int = 0,
) -> list[ItemResidual]:
    """Every rule on every member (a rule is a claim about each trajectory)."""
    out: list[ItemResidual] = []
    with torch.no_grad():
        for j, rule in enumerate(rules):
            loss, violation, _satisfied = physiology_rule_loss_one_rule(
                model, members, rule,
                lambda arm, _r=rule, _i=first_index + j: initial_state(_r.init_mode, arm, _i),
            )
            out.append(ItemResidual(
                name=rule.name, source=rule.source, paper=rule.paper,
                family=protocol_family(rule.arms).value,
                operator=OperatorKind.HINGE.value, marker_id="", sealed=rule.sealed,
                predicted=violation, target=0.0, sigma=float(rule.scale),
                z=math.sqrt(max(float(loss), 0.0)),
            ))
    return out


_BOOTSTRAP_DRAWS = 1000
_BOOTSTRAP_SEED = 0
_INTERVAL_PERCENTILES = (5.0, 95.0)


def _median_interval(by_paper: dict[str, np.ndarray]) -> list[float] | None:
    """90% interval on the pooled median |z| from resampling whole papers.

    Points of one curve and items of one paper move together, so the paper is
    the unit that is drawn; the seed is fixed so a board reproduces. One paper
    gives no spread to resample, so no interval.
    """
    if len(by_paper) < 2:
        return None
    papers = list(by_paper.values())
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    medians = np.empty(_BOOTSTRAP_DRAWS)
    for b in range(_BOOTSTRAP_DRAWS):
        picked = rng.integers(0, len(papers), len(papers))
        medians[b] = np.median(np.concatenate([papers[k] for k in picked]))
    lo, hi = np.percentile(medians, _INTERVAL_PERCENTILES)
    return [float(lo), float(hi)]


def _summary(rows: Sequence[ItemResidual]) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0, "n_papers": 0, "median_abs_z": None, "median_abs_z_90": None,
            "within_1sigma": None, "within_2sigma": None,
        }
    a = [abs(r.z) for r in rows]
    by_paper: dict[str, list[float]] = {}
    for r in rows:
        by_paper.setdefault(r.paper, []).append(abs(r.z))
    return {
        "n": len(rows),
        "n_papers": len(by_paper),
        "median_abs_z": float(median(a)),
        "median_abs_z_90": _median_interval({p: np.asarray(v) for p, v in by_paper.items()}),
        "within_1sigma": sum(1 for x in a if x <= 1.0) / len(a),
        "within_2sigma": sum(1 for x in a if x <= 2.0) / len(a),
    }


def fit_summary(rows: Sequence[ItemResidual]) -> dict[str, Any]:
    """Median miss, its paper-level interval and the shares within 1 and 2 sigma,
    overall and by family."""
    by_family: dict[str, list[ItemResidual]] = {}
    for r in rows:
        by_family.setdefault(r.family, []).append(r)
    return {**_summary(rows), "by_family": {f: _summary(rs) for f, rs in sorted(by_family.items())}}


def _lines(rows: Sequence[ItemResidual]) -> dict[str, dict[str, Any]]:
    """The three literature lines. A held-out row is from a sealed paper or is
    a held-out item of a paper that also trains."""
    sealed = [r for r in rows if r.sealed and paper_is_sealed(r.paper)]
    held_out = [r for r in rows if r.sealed and not paper_is_sealed(r.paper)]
    training = [r for r in rows if not r.sealed]
    return {
        "sealed": fit_summary(sealed),
        "held_out_items": fit_summary(held_out),
        "training": fit_summary(training),
    }


def _held_out_detail(rows: Sequence[ItemResidual]) -> list[dict[str, Any]]:
    """For each held-out item on the board, what it shares with training."""
    out = []
    for name in HELD_OUT_ITEMS:
        points = [r for r in rows if r.name == name or r.name.startswith(f"{name}__t")]
        if not points:
            continue
        shares = []
        for overlap in SHARES_WITH_TRAINING.get(name, ()):
            entry: dict[str, Any] = {"kind": overlap.kind, "with": list(overlap.with_)}
            if overlap.kind == "paper":
                entry["n_training_rows"] = sum(
                    1 for r in rows if not r.sealed and r.paper == overlap.with_[0]
                )
            shares.append(entry)
        out.append({
            "item": name, "paper": points[0].paper, "n_points": len(points),
            "median_abs_z": float(median(abs(r.z) for r in points)), "shares_with_training": shares,
        })
    return out


def _allowed(spec: CohortStatisticSpec) -> tuple[float, float]:
    """The values an item accepts, widened by its own sigma."""
    t, s = float(spec.target), float(spec.sigma)
    if spec.shape is TargetShape.AT_MOST:
        return -math.inf, t + s
    if spec.shape is TargetShape.AT_LEAST:
        return t - s, math.inf
    half = float(spec.band_halfwidth) if spec.shape is TargetShape.BAND else 0.0
    return t - half - s, t + half + s


def target_conflicts(specs: Sequence[CohortStatisticSpec]) -> list[dict[str, Any]]:
    """Items on one protocol, marker, operator and window whose targets disagree
    beyond their sigmas (PLAN section 3). Recorded, never averaged."""
    groups: dict[tuple, list[CohortStatisticSpec]] = {}
    for spec in specs:
        key = (spec.arms, spec.marker_id, spec.kind, spec.window, spec.per_arm_windows)
        groups.setdefault(key, []).append(spec)
    out: list[dict[str, Any]] = []
    for group in groups.values():
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                lo_a, hi_a = _allowed(a)
                lo_b, hi_b = _allowed(b)
                gap = max(lo_a - hi_b, lo_b - hi_a)
                if gap > 0.0:
                    out.append({
                        "kind": "targets", "items": [a.name, b.name],
                        "gap_sigma": gap / math.hypot(a.sigma, b.sigma),
                    })
    return out


def breaks(previous: Sequence[dict[str, Any]], rows: Sequence[ItemResidual]) -> list[dict[str, Any]]:
    """Items that were within one sigma on ``previous`` and are beyond two now."""
    before = {p["name"]: float(p["z"]) for p in previous}
    return [
        {"name": r.name, "z_before": before[r.name], "z_now": r.z}
        for r in rows
        if r.name in before and abs(before[r.name]) <= _INSIDE and abs(r.z) > _WELL_OUTSIDE
    ]


def _fingerprint(specs: Sequence[CohortStatisticSpec], rules: Sequence[PhysiologyRule]) -> str:
    """Which items a board scored, and against what — two boards compare only if equal."""
    lines = sorted(
        [f"{s.name}|{s.paper}|{s.marker_id}|{s.kind.value}|{s.target!r}|{s.sigma!r}" for s in specs]
        + [f"{r.name}|{r.paper}|{r.scale!r}" for r in rules]
    )
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:16]


def _reference_model(model: ModularPhysiologyNetwork, seed: int) -> ModularPhysiologyNetwork:
    """The untrained student: ``model``'s architecture, every published number at
    its published value, fresh network weights drawn from ``seed``."""
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return type(model)(**model.constructor_kwargs)


def _score_all(
    model: nn.Module,
    specs: Sequence[CohortStatisticSpec],
    rules: Sequence[PhysiologyRule],
    members: list[torch.Tensor],
    n_population: int,
    init: InitialState,
) -> list[ItemResidual]:
    return [
        *score_cohorts(model, specs, members, n_population, init),
        *score_rules(model, rules, members, init, first_index=len(specs)),
    ]


def scoreboard(
    model: ModularPhysiologyNetwork,
    *,
    n_draws: int = 4,
    seed: int = 0,
    previous: dict[str, Any] | None = None,
    identifiability: bool = True,
    baseline: bool = False,
    device: torch.device | str = "cpu",
    sealed_specs: Sequence[CohortStatisticSpec] = SEALED_COHORT_STATISTICS,
    sealed_rules: Sequence[PhysiologyRule] = SEALED_PHYSIOLOGY_RULES,
    training_specs: Sequence[CohortStatisticSpec] = ALL_COHORT_STATISTICS,
    training_rules: Sequence[PhysiologyRule] = PHYSIOLOGY_RULES,
) -> dict[str, Any]:
    """Score ``model`` on the board. ``previous`` is an earlier board, for breaks.

    ``baseline`` also scores the untrained student (``_reference_model``) on the
    same items and people, which doubles the cost.
    """
    model.to(device)
    was_training = model.training
    model.eval()
    init = training_initial_states(model, device, seed)
    members, n_population = population_members(n_draws, seed, device=device)

    specs = [*sealed_specs, *training_specs]
    rules = [*sealed_rules, *training_rules]
    rows = _score_all(model, specs, rules, members, n_population, init)

    reference = None
    if baseline:
        start = _reference_model(model, seed).to(device).eval()
        start_rows = _score_all(
            start, specs, rules, members, n_population, training_initial_states(start, device, seed),
        )
        reference = {
            "literature_fit": _lines(start_rows),
            "items": [{"name": r.name, "z": r.z} for r in start_rows],
        }

    openness = None
    conflicts = target_conflicts(specs)
    if identifiability:
        report = identifiability_report(
            model, specs=training_specs, rules=training_rules, device=device, initial_state=init,
        )
        openness = {
            "n_open": report["n_open"],
            "n_params": len(report["param_names"]),
            "open_directions": report["open_directions"],
            "person_axis": {
                k: v for k, v in report["person_axis"].items() if k != "eigenvalues"
            },
        }
        conflicts.extend(report["conflicts"])
    model.train(was_training)

    return {
        "evidence": {
            "sealed": _fingerprint(sealed_specs, sealed_rules),
            "training": _fingerprint(training_specs, training_rules),
        },
        "n_population": n_population,
        "literature_fit": _lines(rows),
        "held_out_items": _held_out_detail(rows),
        "baseline": reference,
        "individual_skill": None,
        "openness": openness,
        "conflicts": conflicts,
        "breaks": breaks(previous["items"], rows) if previous else [],
        "items": [asdict(r) for r in rows],
    }


def attach_individual_skill(board: dict[str, Any], skill: dict[str, Any]) -> dict[str, Any]:
    """The second line, from a ruler run (``benchmark.individual_skill``)."""
    return {**board, "individual_skill": skill}


def scoreboard_path(checkpoint_path: str | Path) -> Path:
    """Where a checkpoint's board lives: beside it, ``<stem>.scoreboard.json``."""
    p = Path(checkpoint_path)
    return p.with_name(f"{p.stem}.scoreboard.json")


def scoreboard_object(checkpoint_object: str) -> str:
    """The board's object name beside a checkpoint object, whatever the local file was called."""
    name = scoreboard_path(checkpoint_object).name
    parent = checkpoint_object.rsplit("/", 1)[0] if "/" in checkpoint_object else ""
    return f"{parent}/{name}" if parent else name


def scoreboard_for_checkpoint(
    checkpoint_path: str | Path,
    previous_path: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    from .diagnostics.probe import load_model_from_checkpoint

    model, _ckpt = load_model_from_checkpoint(checkpoint_path)
    previous = json.loads(Path(previous_path).read_text()) if previous_path else None
    board = scoreboard(model, previous=previous, **kwargs)
    board["checkpoint"] = str(checkpoint_path)
    return board


def _share_text(share: dict[str, Any]) -> str:
    if share["kind"] == "paper":
        return f"paper {share['with'][0]} with {share['n_training_rows']} training rows"
    if share["kind"] == "arm":
        return "an arm with " + ", ".join(share["with"])
    return "its target with the initial " + ", ".join(share["with"])


def _fit_line(label: str, s: dict[str, Any], start: dict[str, Any] | None = None) -> str:
    if not s["n"]:
        return f"    {label:<14} n=0"
    interval = s["median_abs_z_90"]
    spread = f" [{interval[0]:.2f}, {interval[1]:.2f}]" if interval else " [no interval]"
    line = (
        f"    {label:<14} n={s['n']:<4} papers={s['n_papers']:<3} "
        f"median|z|={s['median_abs_z']:.2f}{spread}  "
        f"within 1σ={s['within_1sigma']:.0%}  within 2σ={s['within_2sigma']:.0%}"
    )
    if start is not None and start["n"]:
        line += f"   (untrained: median|z|={start['median_abs_z']:.2f})"
    return line


_LINE_TITLES = {
    "sealed": "sealed papers",
    "held_out_items": "held-out items of papers that also train",
    "training": "training",
}


def render(board: dict[str, Any]) -> str:
    lines = [f"scoreboard: {board.get('checkpoint', '<model>')}"]
    lines.append(f"  evidence sealed={board['evidence']['sealed']} training={board['evidence']['training']}")
    reference = (board.get("baseline") or {}).get("literature_fit")
    for split, title in _LINE_TITLES.items():
        fit = board["literature_fit"][split]
        start = reference[split] if reference else None
        lines.append(f"  literature fit ({split}): {title}; interval is 90%, resampling papers")
        lines.append(_fit_line("all", fit, start))
        for family, s in fit["by_family"].items():
            lines.append(_fit_line(family, s, start["by_family"].get(family) if start else None))
    for item in board.get("held_out_items", []):
        shares = "; ".join(_share_text(s) for s in item["shares_with_training"]) or "nothing recorded"
        lines.append(
            f"    held out: {item['item']} ({item['paper']}, {item['n_points']} point"
            f"{'' if item['n_points'] == 1 else 's'}, median|z|={item['median_abs_z']:.2f}) shares {shares}"
        )

    skill = board.get("individual_skill")
    lines.append("  individual skill:" + ("" if skill else " (no ruler run)"))
    for source, by_marker in (skill or {}).get("by_source", {}).items():
        for marker, s in sorted(by_marker.items()):
            lines.append(
                f"    {source:<14} {marker:<8} vs median person {s['vs_median_person']:+.3f}  "
                f"vs running mean {s['vs_running_mean']:+.3f}  (n={s['n_episodes']})"
            )

    openness = board.get("openness")
    if openness is None:
        lines.append("  open directions: (not computed)")
    else:
        lines.append(f"  open directions: {openness['n_open']} of {openness['n_params']}")
        for d in openness["open_directions"]:
            lead = d["top_params"][0]
            lines.append(f"    {lead['param']:<44} {lead['study'] or 'no study filed'}")
        person = openness["person_axis"]
        lines.append(f"  person axis: {person['n_constrained']} of {person['dim']} constrained")

    lines.append(f"  conflicts ({len(board['conflicts'])}):")
    for c in board["conflicts"]:
        a, b = c["items"]
        detail = f"gap {c['gap_sigma']:.1f}σ" if c["kind"] == "targets" else f"cos {c['cosine']:+.2f}"
        lines.append(f"    [{c['kind']}] {a} <-> {b}  {detail}")
    lines.append(f"  breaks ({len(board['breaks'])}):")
    for b in board["breaks"]:
        lines.append(f"    {b['name']}  z {b['z_before']:+.2f} -> {b['z_now']:+.2f}")
    return "\n".join(lines)
