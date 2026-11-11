"""`STATE.md` is a snapshot of this commit, so the counts in it are testable.

The file is rewritten in the same commit as the change it describes. A count
table is the part that goes stale without anyone noticing, so these
assertions read the table and check it against the code.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pulse.knowledge.cohort_statistics as registry
from pulse.correction import Correction
from pulse.knowledge.cohort_statistics import ALL_COHORT_STATISTICS, SEALED_COHORT_STATISTICS
from pulse.knowledge.corpus import count_piles
from pulse.knowledge.coupling_priors import ALL_COUPLING_PRIORS
from pulse.knowledge.physiology_rules import (
    _AUTHORED_RULES,
    PHYSIOLOGY_RULES,
    SEALED_PHYSIOLOGY_RULES,
)
from pulse.knowledge.queue import QUEUE, Status
from pulse.knowledge.sealing import HELD_OUT_ITEMS, paper_is_sealed
from pulse.model import ModularPhysiologyNetwork
from pulse.published import PublishedNumbers
from pulse.types import MARKERS, MODULE_MARKER_IDS, OBSERVABLE_DIM

ROOT = Path(__file__).resolve().parents[1]
STATE = (ROOT / "STATE.md").read_text(encoding="utf-8")


def _row(metric: str) -> str:
    rows = [line for line in STATE.splitlines() if line.startswith(f"| {metric} |")]
    assert len(rows) == 1, f"{metric!r} names {len(rows)} rows in STATE.md"
    return rows[0]


def _counts(metric: str) -> list[int]:
    """Every integer on ``STATE.md``'s row for this metric, in order."""
    return [int(n.replace(",", "")) for n in re.findall(r"\b\d[\d,]*\b", _row(metric))]


def test_the_model_counts_match_the_code():
    assert _counts("Modules")[0] == len(MODULE_MARKER_IDS)
    for module, markers in MODULE_MARKER_IDS.items():
        assert re.search(rf"\b{module} {len(markers)}\b", _row("Modules")), module
    assert _counts("States") == [len(MARKERS)]
    assert _counts("Observables") == [OBSERVABLE_DIM]

    model = ModularPhysiologyNetwork()
    assert _counts("Parameters") == [sum(p.numel() for p in model.parameters())]
    tables = [m for m in model.modules() if isinstance(m, PublishedNumbers)]
    assert _counts("Published numbers") == [sum(len(t()) for t in tables), len(tables)]
    assert _counts("Corrections") == [sum(isinstance(m, Correction) for m in model.modules())]


def test_the_corpus_counts_match_the_registries():
    authored = registry._AUTHORED
    assert _counts("Cohort specs") == [
        len(authored),
        sum(1 for s in authored if s.curve),
        len(ALL_COHORT_STATISTICS),
        len(SEALED_COHORT_STATISTICS),
    ]
    assert _counts("Rules") == [len(PHYSIOLOGY_RULES), len(SEALED_PHYSIOLOGY_RULES)]
    assert _counts("Coupling priors") == [len(ALL_COUPLING_PRIORS)]

    papers = {s.paper for s in authored} | {r.paper for r in _AUTHORED_RULES}
    assert _counts("Papers") == [len(papers), sum(paper_is_sealed(p) for p in papers)]
    assert _counts("Held-out items") == [len(HELD_OUT_ITEMS)]
    piles = count_piles()
    assert _counts("Corpus files") == [piles["findings"], piles["refusals"], piles["queue"]]


def test_the_queue_counts_match_the_queue():
    by_status = Counter(item.status for item in QUEUE)
    assert _counts("Queue") == [
        len(QUEUE),
        by_status[Status.INTEGRATED],
        by_status[Status.BLOCKED],
        by_status[Status.BACKLOG],
        by_status[Status.READY],
    ]
