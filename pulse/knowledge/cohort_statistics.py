"""
Registry of all quantitative cohort statistic specs.

Each domain file under ``knowledge.cohorts`` defines ``COHORT_STATISTICS``;
this module merges them in training order. Add a new slice by creating or
extending a file there and appending its list below.

Two transforms run over the merged list before it is used:

* ``expand_curves`` compiles every ``kind=CURVE`` spec (a whole published
  trajectory) into one ``MEAN_IN_WINDOW`` spec per timepoint, so the batched
  scorer handles a curve with no special-casing (PLAN Phase 4).
* findings filed as files (``corpus/findings/``, ``corpus.py``) join the authored
  list, so they pass through the same seal. A paper may live in the Python
  registries or in a file, never both.
* the sealed split holds back held-out items (PLAN Phase 0): ``sealing`` decides
  from each item's ``paper``, the training registry ``ALL_COHORT_STATISTICS`` is
  the unsealed items only, and the scoreboard reads ``SEALED_COHORT_STATISTICS``.
  Authors name the paper and never seal.
"""

from __future__ import annotations

from dataclasses import replace

from .cohort_types import (
    CohortStatisticSpec,
    StatisticKind,
    StatisticWindow,
)
from .corpus import load_file_specs
from .sealing import apply_seal
from .cohorts import (
    breadth_floor, cardiovascular, counterregulation, curves, glucose_handling, hepatobiliary,
    incretins,
    ivgtt, nutrition, renal, sleep,
)


def expand_curve_spec(spec: CohortStatisticSpec) -> list[CohortStatisticSpec]:
    """A CURVE spec as one point-statistic per timepoint.

    Each point is a ``MEAN_IN_WINDOW`` over its own one-minute window. The
    per-point sigma is inflated by ``sqrt(n_points)`` so a smooth digitized
    curve, whose neighbouring points are highly correlated, counts as roughly
    one independent shape constraint rather than ``n`` independent means — the
    down-weight PLAN Phase 4 calls for. A non-CURVE spec is returned unchanged.
    """
    if spec.kind is not StatisticKind.CURVE:
        return [spec]
    assert spec.curve is not None  # guaranteed by __post_init__
    n = len(spec.curve)
    inflate = float(n) ** 0.5
    return [
        replace(
            spec,
            name=f"{spec.name}__t{cp.t_min}",
            kind=StatisticKind.MEAN_IN_WINDOW,
            window=StatisticWindow(cp.t_min, cp.t_min + 1),
            target=cp.target,
            sigma=cp.sigma * inflate,
            n_arm=1,
            curve=None,
            per_arm_windows=None,
        )
        for cp in spec.curve
    ]


def expand_curves(specs: list[CohortStatisticSpec]) -> list[CohortStatisticSpec]:
    out: list[CohortStatisticSpec] = []
    for spec in specs:
        out.extend(expand_curve_spec(spec))
    return out


_PYTHON_AUTHORED: list[CohortStatisticSpec] = [
    *nutrition.COHORT_STATISTICS,
    *sleep.COHORT_STATISTICS,
    *glucose_handling.COHORT_STATISTICS,
    *cardiovascular.COHORT_STATISTICS,
    *breadth_floor.COHORT_STATISTICS,
    *hepatobiliary.COHORT_STATISTICS,
    *incretins.COHORT_STATISTICS,
    *counterregulation.COHORT_STATISTICS,
    *ivgtt.COHORT_STATISTICS,
    *curves.COHORT_STATISTICS,
    *renal.COHORT_STATISTICS,
]


def join_file_findings(
    python_authored: list[CohortStatisticSpec],
    file_specs: list[CohortStatisticSpec],
    rule_papers: frozenset[str] = frozenset(),
) -> list[CohortStatisticSpec]:
    """The authored list with the corpus files' specs appended.

    A paper that is a file may not also name an item in a Python registry (cohort
    statistics or physiology rules): one source per paper, or a number could be
    in training twice under two sigmas.
    """
    both = {s.paper for s in file_specs} & ({s.paper for s in python_authored} | set(rule_papers))
    if both:
        raise ValueError(
            f"papers filed both as corpus files and in a Python registry: {', '.join(sorted(both))}",
        )
    names = [s.name for s in [*python_authored, *file_specs]]
    clash = sorted({n for n in names if names.count(n) > 1})
    if clash:
        raise ValueError(f"item names used twice: {', '.join(clash)}")
    return [*python_authored, *file_specs]


def _rule_papers() -> frozenset[str]:
    from .physiology_rules import _AUTHORED_RULES

    return frozenset(r.paper for r in _AUTHORED_RULES)


_AUTHORED: list[CohortStatisticSpec] = join_file_findings(
    _PYTHON_AUTHORED, load_file_specs(), _rule_papers(),
)

# The seal is applied to the authored items, then curves become point specs, so a
# sealed curve is sealed on every point (``replace`` preserves it).
_ALL_COHORT_STATISTICS: list[CohortStatisticSpec] = expand_curves([apply_seal(s) for s in _AUTHORED])

ALL_COHORT_STATISTICS: list[CohortStatisticSpec] = [
    s for s in _ALL_COHORT_STATISTICS if not s.sealed
]
SEALED_COHORT_STATISTICS: list[CohortStatisticSpec] = [
    s for s in _ALL_COHORT_STATISTICS if s.sealed
]

__all__ = [
    "ALL_COHORT_STATISTICS",
    "SEALED_COHORT_STATISTICS",
    "expand_curve_spec",
    "expand_curves",
]
