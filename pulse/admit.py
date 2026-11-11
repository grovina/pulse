"""The gate: ``python -m pulse.admit <file>``.

Admits one paper's file into ``corpus/``. For a finding file it checks, in order,
and says which check failed:

1. the file parses (strictly; every problem is reported together), and its
   protocols, units, place in the paper and uncertainty are each well formed;
2. it is filed once: named for its paper, no other file or Python registry holds
   the paper, no other slug carries the same DOI / PMID / PMCID, and it is not
   also a refusal;
3. every protocol runs on the **untrained** student;
4. each finding's predicted statistic moves when the parameters are nudged, the
   protocol rolled from the same start state as training rolls it;
5. no finding silently contradicts an item already admitted (the scoreboard's
   ``target_conflicts``, over the likelihood's own standard error); a conflict
   the file names in ``acknowledged_conflicts`` is recorded, not silent.

It also says whether the paper is sealed (held out of training). Exit 0 when
admitted, 1 otherwise.

The gate uses only the untrained student, never a checkpoint, and prints nothing
about how well any model fits a finding: not a residual, not a prediction, not a
loss. Nobody should be able to drop a paper because it scores badly. Check 4
compares the statistic before and after a random nudge of every parameter, so
the number it looks at is whether the statistic can move, not where the model
stands. It runs no backward pass: the protocols of a file are rolled together
three times, which takes seconds where a gradient per finding took minutes.

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
_DIRECTIONS = 2      # random nudges tried; a finding passes if any one moves it
_NUDGE = 1e-2        # relative size of a nudge (see corpus/README.md and test_admit for the measurement)
_NUDGE_FLOOR = 0.1   # added to a tensor's rms so zero-initialised corrections are nudged too
_MOVED = 1e-6        # relative move below which a prediction counts as not moving

_LABELS = {
    "parse": "the file parses",
    "protocol": "protocols are well formed",
    "units": "units are listed and values plausible",
    "place": "each number has a place in the paper",
    "uncertainty": "each uncertainty is of a stated kind",
    "filing": "the paper is filed once",
    "runs": "each protocol runs on the untrained student",
    "gradient": "each finding moves when the parameters move",
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
    # (protocol, keys) for each protocol with values the paper does not state
    assumed: tuple[tuple[str, tuple[str, ...]], ...] = ()

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
    """Rolls a file's protocols on the untrained student and asks, finding by finding,
    whether its prediction can move when a parameter moves. No backward pass."""

    def __init__(self, device: str, fed_day_minutes: int | None) -> None:
        import numpy as np
        import torch

        from .model import ModularPhysiologyNetwork
        from .training.factors import FED_DAY_MINUTES

        torch.manual_seed(_PROBE_SEED)
        self.torch = torch
        self.np = np
        self.device = device
        self.fed = FED_DAY_MINUTES if fed_day_minutes is None else fed_day_minutes
        self.model = ModularPhysiologyNetwork().to(device).eval()
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self._people: dict = {}
        self.keys: list = []
        self.codes: dict = {}
        self.states: dict = {}

    def _codes(self, population):
        """The same people for every arm of one population, as training draws them."""
        from .training.embedding_sampler import draw_people

        if population not in self._people:
            rng = self.np.random.default_rng(_PROBE_SEED)
            self._people[population] = draw_people(_PROBE_DRAWS, rng, self.device, population)
        return self._people[population]

    def prepare(self, specs) -> None:
        """The file's distinct arms, their people, and their start states from ONE batched
        spin-up. The start states are detached and fixed, as in training: only the protocol
        that follows depends on the weights."""
        from .knowledge.cohort_types import InitMode
        from .training.arm_init import student_initial_states_for_rows
        from .types import NORM_CENTER

        keys: list = []
        for spec in specs:
            for arm in spec.arms:
                if (arm, spec.init_mode) not in keys:
                    keys.append((arm, spec.init_mode))
        self.keys = keys
        self.codes = {k: self._codes(k[0].population) for k in keys}
        cold = [k for k in keys if k[1] is InitMode.COLD]
        starts = student_initial_states_for_rows(
            self.model, [(self.codes[k], k[0]) for k in cold], fed_day_minutes=self.fed,
        )
        self.states = dict(zip(cold, starts))
        basal = self.torch.tensor(NORM_CENTER, dtype=self.torch.float32, device=self.device)
        for k in keys:
            if k not in self.states:
                self.states[k] = basal.unsqueeze(0).expand(_PROBE_DRAWS, -1).contiguous()

    def roll_all(self, keys=None) -> dict:
        """Every arm (or ``keys``) rolled together as rows of one rollout, without gradient."""
        from .cohort_loss import rollout_arms

        keys = self.keys if keys is None else keys
        with self.torch.no_grad():
            trajs = rollout_arms(
                self.model, [(k[0], self.codes[k], self.states[k]) for k in keys], observe=True,
            )
        return dict(zip(keys, trajs))

    @staticmethod
    def finite(trajs) -> bool:
        import torch

        return all(bool(torch.isfinite(t).all()) for t in trajs)

    def predictions(self, spec, rolled: dict) -> list[float]:
        """The statistic of ``spec`` (one number, or one per curve point) on ``rolled``,
        scored by the function training scores it with."""
        from .cohort_loss import _score_arm_trajs
        from .knowledge.cohort_statistics import expand_curve_spec

        keys = [(arm, spec.init_mode) for arm in spec.arms]
        trajs = [rolled[k] for k in keys]
        embeddings = self.codes[keys[0]]
        out = []
        for s in expand_curve_spec(spec):
            _loss, pred, _z = _score_arm_trajs(s, trajs, embeddings, _PROBE_DRAWS)
            out.append(float(pred))
        return out

    def _direction(self, k: int):
        g = self.torch.Generator().manual_seed(_PROBE_SEED + 1 + k)
        return [
            self.torch.randn(p.shape, generator=g) * (float(p.detach().pow(2).mean().sqrt()) + _NUDGE_FLOOR)
            for p in self.params
        ]

    def moves(self, specs: dict, base: dict) -> dict[str, float]:
        """``{id: largest relative move of the prediction}`` over ``_DIRECTIONS`` random
        nudges of every parameter, the protocols rolled from the same start states.
        Relative to the larger of the prediction and the finding's sigma."""
        best = {i: 0.0 for i in specs}
        base_pred = {i: self.predictions(s, base) for i, s in specs.items()}
        for k in range(_DIRECTIONS):
            direction = self._direction(k)
            saved = [p.detach().clone() for p in self.params]
            try:
                with self.torch.no_grad():
                    for p, d in zip(self.params, direction):
                        p.add_(_NUDGE * d.to(p.device))
                nudged = self.roll_all()
                for i, s in specs.items():
                    scale = max(abs(self.np.array(base_pred[i])).max(), float(s.sigma))
                    now = self.predictions(s, nudged)
                    move = max(abs(a - b) for a, b in zip(now, base_pred[i])) / scale
                    best[i] = max(best[i], move if move == move else float("inf"))
            finally:
                with self.torch.no_grad():
                    for p, v in zip(self.params, saved):
                        p.copy_(v)
        return best


def _on_one_thread(fn, *args):
    """``fn(*args)`` with torch on one intra-op thread, restored afterwards. A minute of
    the model is many small operations, so threads have nothing to split; left at the
    default (every core the host shows) several gates on one small machine spend their
    time waiting on each other."""
    import torch

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return fn(*args)
    finally:
        torch.set_num_threads(threads)


def _run_checks(finding: Finding, device: str, fed_day_minutes: int | None) -> tuple[Check, Check]:
    probe = _Probe(device, fed_day_minutes)
    run_problems: list[str] = []
    grad_problems: list[str] = []
    specs = {item.id: item.spec for item in finding.items}
    try:
        probe.prepare(list(specs.values()))
        base = probe.roll_all()
    except Exception:  # find which protocol the model cannot run, one arm at a time
        base = {}
        for k in probe.keys or []:
            try:
                base.update(probe.roll_all([k]))
            except Exception as err:
                run_problems.append(f"protocol {k[0].label} does not run: {type(err).__name__}: {err}")
        if not probe.keys:
            run_problems.append("the protocols could not be prepared")
    for k, traj in base.items():
        if not probe.finite([traj]):
            run_problems.append(f"protocol {k[0].label} gives a non-finite trajectory")
    if run_problems:
        return Check("runs", False, tuple(run_problems)), Check("gradient", False, ("not run: the protocols do not run",), skipped=True)
    try:
        moved = probe.moves(specs, base)
    except Exception as err:
        return Check("runs", True), Check("gradient", False, (f"the probe failed: {type(err).__name__}: {err}",))
    for item_id, move in moved.items():
        if move != move or move == float("inf"):
            grad_problems.append(f"{item_id}: the prediction is not finite when a parameter moves")
        elif move <= _MOVED:
            grad_problems.append(
                f"{item_id}: its prediction does not move when the parameters do "
                f"(the model cannot move this number; does it read only the start state?)",
            )
    return Check("runs", True), Check("gradient", not grad_problems, tuple(grad_problems))


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
    assumed = tuple((name, keys) for name, keys in parsed.assumed.items() if keys)
    if filing:
        return Admission(
            kind, paper, (*base_checks, filing_check, *_skip(("runs", "gradient", "conflicts"))),
            sealed, note, len(parsed.items), assumed,
        )
    runs, gradient = _on_one_thread(_run_checks, parsed, device, fed_day_minutes)
    conflicts = _conflict_check(parsed)
    return Admission(
        kind, paper, (*base_checks, filing_check, runs, gradient, conflicts), sealed, note,
        len(parsed.items), assumed,
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
    if result.assumed:
        lines.append("assumed by the reader, not stated by the paper:")
        lines.extend(f"  {name}: {', '.join(keys)}" for name, keys in result.assumed)
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
