"""The gate: ``python -m pulse.admit <file>``.

Admits one paper's file into ``corpus/``. For a finding file it checks, in order,
and says which check failed:

1. the file parses (strictly; every problem is reported together), and its
   protocols, units, place in the paper and uncertainty are each well formed;
2. it is filed once: named for its paper, no other file or Python registry holds
   the paper, no other slug carries the same DOI / PMID / PMCID, and it is not
   also a refusal;
3. every protocol runs on the **untrained** student;
4. a gradient from each finding reaches at least one parameter;
5. no finding silently contradicts an item already admitted (the scoreboard's
   ``target_conflicts``, over the likelihood's own standard error); a conflict
   the file names in ``acknowledged_conflicts`` is recorded, not silent.

It also says whether the paper is sealed (held out of training). Exit 0 when
admitted, 1 otherwise.

The gate uses only the untrained student, never a checkpoint, and prints nothing
about how well any model fits a finding: not a residual, not a prediction, not a
loss. Nobody should be able to drop a paper because it scores badly. Check 4
differentiates the statistic against a far-away target, so the number it looks at
is whether a gradient exists, not where the model stands.

``python -m pulse.admit --refusals`` counts the refusal pile by kind and by the
missing piece: the specification for the next change to the model.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .knowledge.corpus import (
    CHECKS,
    REFUSAL_KINDS,
    CorpusError,
    Finding,
    Problem,
    corpus_root,
    count_refusals,
    read_finding,
    read_queue_item,
    read_refusal,
)

__all__ = ["Admission", "Check", "admit", "format_admission", "main"]

_PROBE_DRAWS = 4
_PROBE_SEED = 0
_FAR = 1.0e4  # sigmas: the probe target sits this far from the statistic

_LABELS = {
    "parse": "the file parses",
    "protocol": "protocols are well formed",
    "units": "units are listed and values plausible",
    "place": "each number has a place in the paper",
    "uncertainty": "each uncertainty is of a stated kind",
    "filing": "the paper is filed once",
    "runs": "each protocol runs on the untrained student",
    "gradient": "a gradient from each finding reaches a parameter",
    "conflicts": "no finding silently contradicts an admitted item",
}


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    reasons: tuple[str, ...] = ()
    skipped: bool = False

    @property
    def label(self) -> str:
        return _LABELS[self.name]


@dataclass(frozen=True)
class Admission:
    kind: str  # "finding" | "refusal"
    paper: str
    checks: tuple[Check, ...]
    sealed: bool | None = None
    sealed_note: str = ""
    items: int = 0

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    @property
    def failed(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if not c.ok and not c.skipped)


# --------------------------------------------------------------------------------
# Filing
# --------------------------------------------------------------------------------

def _raw_records(root: Path) -> list[tuple[Path, dict]]:
    out = []
    for sub in ("findings", "refusals", "queue"):
        d = root / sub
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if isinstance(data, dict):
                out.append((p, data))
    return out


def _ids_of(record: Any) -> set[str]:
    """Identifiers of a raw file: in its ``citation`` (findings, refusals) or at its top level (queue)."""
    out: set[str] = set()
    for citation in (record.get("citation"), record) if isinstance(record, dict) else ():
        if not isinstance(citation, dict):
            continue
        if isinstance(citation.get("doi"), str):
            out.add(f"doi:{citation['doi'].lower()}")
        if isinstance(citation.get("pmid"), str):
            out.add(f"pmid:{citation['pmid']}")
        if isinstance(citation.get("pmcid"), str):
            out.add(f"pmcid:{citation['pmcid'].upper()}")
    return out


def _python_papers() -> set[str]:
    from .knowledge.cohort_statistics import _PYTHON_AUTHORED
    from .knowledge.physiology_rules import _AUTHORED_RULES

    return {s.paper for s in _PYTHON_AUTHORED} | {r.paper for r in _AUTHORED_RULES}


def _filing_problems(path: Path, paper: str, identifiers: set[str], root: Path) -> list[str]:
    out: list[str] = []
    me = path.resolve()
    for other, data in _raw_records(root):
        if other.resolve() == me:
            continue
        where = f"{other.parent.name}/{other.name}"
        if data.get("paper") == paper:
            hint = "; filing a finding or a refusal replaces the queue file: delete it" if other.parent.name == "queue" else ""
            out.append(f"{paper} is already filed as {where}{hint}")
        elif identifiers & _ids_of(data):
            shared = sorted(identifiers & _ids_of(data))[0]
            out.append(f"the paper is already filed under another slug, {data.get('paper')} ({shared})")
    if paper in _python_papers():
        out.append(f"{paper} is already in a Python registry; a paper has one source")
    return out


def _sealed(paper: str) -> tuple[bool, str]:
    from .knowledge.sealing import GRANDFATHERED_PAPERS, paper_is_sealed

    if paper in GRANDFATHERED_PAPERS:
        return False, "grandfathered: the warm start was fitted to it, so it trains"
    if paper_is_sealed(paper):
        return True, "drawn by the hash of its slug: held out of training, scored only on the board"
    return False, "not drawn: trains"


# --------------------------------------------------------------------------------
# Running the untrained student
# --------------------------------------------------------------------------------

class _Probe:
    """Rolls findings on the untrained student and asks only whether gradients exist."""

    def __init__(self, device: str, fed_day_minutes: int | None) -> None:
        import numpy as np
        import torch

        from .model import ModularPhysiologyNetwork

        torch.manual_seed(_PROBE_SEED)
        self.torch = torch
        self.np = np
        self.device = device
        from .training.factors import FED_DAY_MINUTES

        self.fed = FED_DAY_MINUTES if fed_day_minutes is None else fed_day_minutes

        self.model = ModularPhysiologyNetwork().to(device).eval()
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self._people: dict = {}
        self._starts: dict = {}
        self._arms: dict = {}

    def _codes(self, population):
        """The same people for every arm of one population, as training draws them."""
        from .training.embedding_sampler import draw_people

        if population not in self._people:
            rng = self.np.random.default_rng(_PROBE_SEED)
            self._people[population] = draw_people(_PROBE_DRAWS, rng, self.device, population)
        return self._people[population]

    def _state(self, spec, arm, codes):
        """Where an arm starts. The spin-up through a fed day depends on the clock, the
        pre-fast and the people, not on the protocol, so arms that agree share it."""
        from .training.arm_init import student_initial_states
        from .types import NORM_CENTER

        key = (spec.init_mode.value, arm.start_hour, arm.prefast_hours, arm.population)
        if key not in self._starts:
            if spec.init_mode.value == "norm_center":
                basal = self.torch.tensor(NORM_CENTER, dtype=self.torch.float32, device=self.device)
                state = basal.unsqueeze(0).expand(int(codes.shape[0]), -1).contiguous()
            else:
                state = student_initial_states(self.model, codes, arm, fed_day_minutes=self.fed)
            self._starts[key] = state
        return self._starts[key]

    def roll(self, spec):
        """``(codes, trajs)`` of ``spec``'s arms. An arm is rolled once however many
        findings read it; the graph is kept so each finding can be differentiated."""
        from .cohort_loss import _rollout_arm_states

        codes, trajs = [], []
        for arm in spec.arms:
            c = self._codes(arm.population)
            key = (arm, spec.init_mode)
            if key not in self._arms:
                state = self._state(spec, arm, c)
                self._arms[key] = _rollout_arm_states(self.model, c, arm, state, observe=True)
            codes.append(c)
            trajs.append(self._arms[key])
        return codes, trajs

    def finite(self, trajs) -> bool:
        return all(bool(self.torch.isfinite(t).all()) for t in trajs)

    def reaches_parameter(self, spec) -> tuple[int, bool]:
        """``(parameters reached, all gradients finite)`` for ``spec``'s statistic."""
        from .cohort_loss import _score_arm_trajs
        from .knowledge.cohort_statistics import expand_curve_spec

        codes, trajs = self.roll(spec)
        total = None
        for s in expand_curve_spec(spec):
            far = replace(s, target=s.target + _FAR * s.sigma)
            z2, _pred, _z = _score_arm_trajs(far, trajs, codes[0], _PROBE_DRAWS)
            total = z2 if total is None else total + z2
        assert total is not None
        grads = self.torch.autograd.grad(total, self.params, retain_graph=True, allow_unused=True)
        reached = 0
        finite = True
        for g in grads:
            if g is None:
                continue
            if not bool(self.torch.isfinite(g).all()):
                finite = False
            elif bool((g != 0).any()):
                reached += 1
        return reached, finite


def _run_checks(finding: Finding, device: str, fed_day_minutes: int | None) -> tuple[Check, Check]:
    probe = _Probe(device, fed_day_minutes)
    run_problems: list[str] = []
    grad_problems: list[str] = []
    ordered = sorted(finding.items, key=lambda i: str([a.label for a in i.spec.arms]))
    for item in ordered:
        spec = item.spec
        try:
            _codes, trajs = probe.roll(spec)
        except Exception as err:  # a protocol the model cannot run, whatever the reason
            labels = ", ".join(a.label for a in spec.arms)
            run_problems.append(f"{item.id}: protocol {labels} does not run: {type(err).__name__}: {err}")
            continue
        if not probe.finite(trajs):
            labels = ", ".join(a.label for a in spec.arms)
            run_problems.append(f"{item.id}: protocol {labels} gives a non-finite trajectory")
            continue
        try:
            reached, finite = probe.reaches_parameter(spec)
        except Exception as err:
            grad_problems.append(f"{item.id}: the statistic cannot be differentiated: {type(err).__name__}: {err}")
            continue
        if not finite:
            grad_problems.append(f"{item.id}: the gradient is not finite")
        elif reached == 0:
            grad_problems.append(
                f"{item.id}: no gradient from this finding reaches any parameter "
                f"(the model cannot move this number)",
            )
    return (
        Check("runs", not run_problems, tuple(run_problems)),
        Check("gradient", not grad_problems, tuple(grad_problems)),
    )


# --------------------------------------------------------------------------------
# Conflicts
# --------------------------------------------------------------------------------

def _comparable(spec):
    """``spec`` as the conflict check compares it: arm labels dropped (two papers
    that run one protocol name it differently) and sigma the standard error the
    likelihood actually uses, so a wide between-person SD does not hide a gap."""
    from .knowledge.cohort_types import StatisticKind

    sigma = spec.sigma
    if spec.n_arm is not None and spec.kind not in (
        StatisticKind.SPREAD, StatisticKind.CORRELATION, StatisticKind.RATIO_INCREMENTS,
    ):
        sigma = spec.sigma / (spec.n_arm ** 0.5)
    arms = tuple(replace(a, label="") for a in spec.arms)
    return replace(spec, arms=arms, sigma=sigma)


def _conflict_check(finding: Finding) -> Check:
    from .knowledge.cohort_statistics import (
        ALL_COHORT_STATISTICS, SEALED_COHORT_STATISTICS, expand_curve_spec,
    )
    from .scoreboard import target_conflicts

    admitted = [
        s for s in (*ALL_COHORT_STATISTICS, *SEALED_COHORT_STATISTICS) if s.paper != finding.paper
    ]
    new: list = []
    owner: dict[str, str] = {}
    for item in finding.items:
        for s in expand_curve_spec(item.spec):
            new.append(s)
            owner[s.name] = item.id
    acknowledged = {item.id: set(item.acknowledged_conflicts) for item in finding.items}
    pairs = target_conflicts([_comparable(s) for s in (*admitted, *new)])
    reasons: list[str] = []
    used: dict[str, set[str]] = {k: set() for k in acknowledged}
    for pair in pairs:
        a, b = pair["items"]
        if (a in owner) == (b in owner):
            continue  # between admitted items, or within this file: not ours to judge here
        mine, other = (a, b) if a in owner else (b, a)
        item_id = owner[mine]
        base = other.partition("__t")[0]
        if base in acknowledged[item_id]:
            used[item_id].add(base)
            continue
        reasons.append(
            f"{item_id} contradicts {base} on one protocol, marker and window "
            f"({pair['gap_sigma']:.1f} combined standard errors apart); "
            f"if the conflict is real, name it in acknowledged_conflicts",
        )
    for item_id, names in acknowledged.items():
        for nm in sorted(names - used[item_id]):
            reasons.append(f"{item_id} acknowledges a conflict with {nm}, but they do not conflict")
    return Check("conflicts", not reasons, tuple(sorted(set(reasons))))


# --------------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------------

def _group(problems: list[Problem]) -> list[Check]:
    return [
        Check(name, not any(p.check == name for p in problems),
              tuple(p.message for p in problems if p.check == name))
        for name in CHECKS
    ]


def _skip(names: tuple[str, ...]) -> list[Check]:
    return [Check(n, False, ("not run: an earlier check failed",), skipped=True) for n in names]


def admit(
    path: Path, *, kind: str | None = None, root: Path | None = None, device: str = "cpu",
    fed_day_minutes: int | None = None,
) -> Admission:
    """Run the gate on ``path``. ``kind`` is ``"finding"`` or ``"refusal"``; when
    omitted it is read from the directory the file sits in."""
    path = Path(path)
    root = root or corpus_root()
    kind = kind or {"findings": "finding", "refusals": "refusal", "queue": "queue"}.get(path.parent.name)
    if kind not in ("finding", "refusal", "queue"):
        return Admission("finding", "", (Check(
            "parse", False,
            (f"cannot tell whether {path} is a finding, a refusal or a queue item: put it in "
             f"findings/, refusals/ or queue/, or pass --as finding|refusal|queue",),
        ),))
    later = ("filing", "runs", "gradient", "conflicts") if kind == "finding" else ("filing",)
    paper = ""
    reader = {"finding": read_finding, "refusal": read_refusal, "queue": read_queue_item}[kind]
    try:
        parsed = reader(path)
    except CorpusError as err:
        paper = err.paper
        checks = _group(err.problems) if kind == "finding" else [
            Check("parse", False, tuple(p.message for p in err.problems)),
        ]
        sealed, note = _sealed(paper) if paper else (None, "")
        return Admission(kind, paper, (*checks, *_skip(later)), sealed, note)
    except OSError as err:
        return Admission(kind, "", (Check("parse", False, (str(err),)), *_skip(later)))
    paper = parsed.paper
    base_checks = _group([]) if kind == "finding" else [Check("parse", True)]
    sealed, note = _sealed(paper)
    filing = _filing_problems(path, paper, set(parsed.identifiers() if kind == "queue" else parsed.citation.identifiers()), root)
    filing_check = Check("filing", not filing, tuple(filing))
    if kind in ("refusal", "queue"):
        return Admission(kind, paper, (*base_checks, filing_check), sealed if kind == "queue" else None,
                         note if kind == "queue" else "")
    assert isinstance(parsed, Finding)
    if filing:
        return Admission(
            kind, paper, (*base_checks, filing_check, *_skip(("runs", "gradient", "conflicts"))),
            sealed, note, len(parsed.items),
        )
    runs, gradient = _run_checks(parsed, device, fed_day_minutes)
    conflicts = _conflict_check(parsed)
    return Admission(
        kind, paper, (*base_checks, filing_check, runs, gradient, conflicts), sealed, note,
        len(parsed.items),
    )


def format_admission(result: Admission) -> str:
    head = f"paper {result.paper or 'unknown'} ({result.kind}"
    head += f", {result.items} findings)" if result.kind == "finding" and result.items else ")"
    lines = [head]
    for c in result.checks:
        mark = "skip" if c.skipped else ("ok  " if c.ok else "FAIL")
        lines.append(f"  {mark}  {c.label}")
        if not c.ok and not c.skipped:
            lines.extend(f"        - {r}" for r in c.reasons)
    if result.sealed is not None:
        lines.append(f"sealed: {'yes' if result.sealed else 'no'} ({result.sealed_note})")
    lines.append("admitted" if result.ok else "not admitted")
    return "\n".join(lines)


def format_refusals(root: Path) -> str:
    from .knowledge.corpus import load_corpus

    corpus = load_corpus(root)
    counts = count_refusals(corpus.refusals)
    if not counts:
        return "no refusals"
    lines = [f"{len(corpus.refusals)} refusals"]
    for kind in REFUSAL_KINDS:
        if kind in counts:
            lines.append(f"  {kind}: {sum(counts[kind].values())}")
            lines.extend(f"    {n:3d}  {name}" for name, n in counts[kind].items())
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pulse.admit",
        description="Admit one paper's file into corpus/. Exit 0 when admitted.",
    )
    parser.add_argument("path", type=Path, nargs="?", help="corpus/{findings,refusals,queue}/<paper>.json")
    parser.add_argument("--as", dest="kind", choices=("finding", "refusal", "queue"),
                        help="what the file is, when its directory does not say")
    parser.add_argument("--refusals", action="store_true", help="count the refusal pile by kind and name")
    parser.add_argument("--piles", action="store_true", help="count the papers in queue, findings and refusals")
    args = parser.parse_args(argv)
    if args.piles:
        from .knowledge.corpus import count_piles

        print("\n".join(f"{k}: {n}" for k, n in count_piles(corpus_root()).items()))
        return 0
    if args.refusals:
        print(format_refusals(corpus_root()))
        return 0
    if args.path is None:
        parser.error("a file is required")
    result = admit(args.path, kind=args.kind)
    print(format_admission(result))
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
