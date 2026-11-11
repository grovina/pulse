"""Sealing is by paper and the registry decides: no author seals an item."""

from __future__ import annotations

import hashlib
import importlib
import re
from pathlib import Path

import pytest

import pulse.knowledge.cohort_statistics as registry
from pulse.knowledge.cohort_statistics import ALL_COHORT_STATISTICS, SEALED_COHORT_STATISTICS
from pulse.knowledge.cohort_types import CohortArmSpec, CohortStatisticSpec, StatisticKind, StatisticWindow
from pulse.knowledge.evidence import ALL_EVIDENCE, SEALED_EVIDENCE
from pulse.knowledge.physiology_rules import (
    PHYSIOLOGY_RULES, SEALED_PHYSIOLOGY_RULES, _AUTHORED_RULES,
)
from pulse.knowledge.sealing import (
    GRANDFATHERED_PAPERS, HELD_OUT_ITEMS, SHARES_WITH_TRAINING, apply_seal, paper_is_sealed,
)

_SLUG = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")
_KNOWLEDGE = Path(__file__).resolve().parents[1] / "pulse" / "knowledge"
_ARM = CohortArmSpec(label="a", duration_min=60, start_hour=8.0, meals=())


def _digest(names: tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def _spec(name: str, paper: str, **kw) -> CohortStatisticSpec:
    return CohortStatisticSpec(
        name=name, source="t", paper=paper, description="", arms=(_ARM,), marker_id="glucose",
        kind=StatisticKind.MEAN_IN_WINDOW, window=StatisticWindow(30, 60), target=95.0, sigma=5.0,
        **kw,
    )


def _base(name: str) -> str:
    return name.partition("__t")[0]


def _authored() -> list[CohortStatisticSpec]:
    return registry._AUTHORED


# --- every item names its paper ---------------------------------------------

def test_every_cohort_spec_and_rule_names_its_paper_as_a_slug():
    for item in [*_authored(), *_AUTHORED_RULES]:
        assert _SLUG.fullmatch(item.paper), f"{item.name}: {item.paper!r}"


def test_every_evidence_item_from_the_literature_carries_its_paper():
    literature = [it for it in ALL_EVIDENCE if it.rule is not None or it.paper is not None]
    assert len(literature) == len(ALL_COHORT_STATISTICS) + len(PHYSIOLOGY_RULES)
    assert all(it.paper for it in literature)
    assert all(it.paper for it in SEALED_EVIDENCE)


def test_the_registry_refuses_an_item_with_no_paper():
    with pytest.raises(ValueError, match="no paper"):
        apply_seal(_spec("x", ""))


def test_a_slug_is_lowercase_ascii_with_hyphens():
    for bad in ("Schulla-2003", "schulla_2003", "schulla 2003", "", "schulla-"):
        with pytest.raises(ValueError):
            paper_is_sealed(bad)


# --- no author-side seal ----------------------------------------------------

def test_no_author_seals_an_item():
    assert not any(s.sealed for s in _authored())
    assert not any(r.sealed for r in _AUTHORED_RULES)
    for path in [*(_KNOWLEDGE / "cohorts").glob("*.py"), _KNOWLEDGE / "physiology_rules.py"]:
        text = path.read_text(encoding="utf-8")
        assert "sealed_at_random" not in text, path.name
        assert not re.search(r"^\s+sealed=", text, flags=re.M), path.name


def test_the_registry_refuses_an_author_who_seals():
    with pytest.raises(ValueError, match="registry's decision"):
        apply_seal(_spec("x", "schulla-2003", sealed=True))


# --- the decision -----------------------------------------------------------

def test_the_seal_is_one_decision_per_paper_so_all_its_points_agree():
    by_paper: dict[str, set[bool]] = {}
    for s in [*ALL_COHORT_STATISTICS, *SEALED_COHORT_STATISTICS]:
        by_paper.setdefault(s.paper, set()).add(s.sealed and _base(s.name) not in HELD_OUT_ITEMS)
    assert all(len(v) == 1 for v in by_paper.values())


def test_the_registry_seals_exactly_the_sealed_papers_and_the_legacy_items():
    for s in [*ALL_COHORT_STATISTICS, *SEALED_COHORT_STATISTICS]:
        assert s.sealed == (paper_is_sealed(s.paper) or _base(s.name) in HELD_OUT_ITEMS), s.name
    assert {s.name for s in ALL_COHORT_STATISTICS}.isdisjoint(s.name for s in SEALED_COHORT_STATISTICS)


def test_a_paper_the_draw_seals_seals_every_item_that_names_it():
    paper = next(p for p in (f"paper-{i}" for i in range(100)) if paper_is_sealed(p))
    assert apply_seal(_spec("one", paper)).sealed and apply_seal(_spec("two", paper)).sealed
    assert not apply_seal(_spec("three", "wahren-1971")).sealed


def test_no_sealed_paper_has_an_item_in_a_training_registry():
    trained = {s.paper for s in ALL_COHORT_STATISTICS} | {r.paper for r in PHYSIOLOGY_RULES}
    papers = trained | {s.paper for s in SEALED_COHORT_STATISTICS}
    papers |= {r.paper for r in SEALED_PHYSIOLOGY_RULES}
    assert not [p for p in papers if paper_is_sealed(p) and p in trained]


def test_every_paper_with_a_training_item_is_grandfathered():
    trained = {s.paper for s in ALL_COHORT_STATISTICS} | {r.paper for r in PHYSIOLOGY_RULES}
    assert trained <= set(GRANDFATHERED_PAPERS)
    assert not any(paper_is_sealed(p) for p in GRANDFATHERED_PAPERS)


# --- the frozen lists -------------------------------------------------------

def test_the_grandfathered_papers_are_frozen():
    assert list(GRANDFATHERED_PAPERS) == sorted(set(GRANDFATHERED_PAPERS))
    assert _digest(GRANDFATHERED_PAPERS) == (
        "23cb3bc33423d621053c9d8510e7dd03c07416f8e18d9799926b6340f670c5aa"
    )


def test_the_items_held_out_before_sealing_went_by_paper_are_frozen_and_still_held_out():
    assert list(HELD_OUT_ITEMS) == sorted(set(HELD_OUT_ITEMS))
    assert _digest(HELD_OUT_ITEMS) == (
        "efb8003af48820c5d347c45e09b24ba4efa822860b3d602559771b4186139718"
    )
    held = {_base(s.name) for s in SEALED_COHORT_STATISTICS}
    assert set(HELD_OUT_ITEMS) <= held
    assert set(HELD_OUT_ITEMS) <= {s.name for s in _authored()}


def test_no_rule_is_sealed_yet():
    assert SEALED_PHYSIOLOGY_RULES == []
    assert len(PHYSIOLOGY_RULES) == len(_AUTHORED_RULES)


# --- what the held-out items share with training ----------------------------

def _training_by_base() -> dict[str, CohortStatisticSpec]:
    return {_base(s.name): s for s in ALL_COHORT_STATISTICS}


def _published(path: str) -> float:
    *module, table, key = path.split(".")
    return getattr(importlib.import_module(".".join(module)), table)[key].value


def test_every_held_out_item_records_what_it_shares_and_each_claim_is_true():
    assert set(SHARES_WITH_TRAINING) == set(HELD_OUT_ITEMS)
    authored = {s.name: s for s in _authored()}
    training = _training_by_base()
    for name, overlaps in SHARES_WITH_TRAINING.items():
        item = authored[name]
        for overlap in overlaps:
            if overlap.kind == "paper":
                assert overlap.with_ == (item.paper,)
                assert any(s.paper == item.paper for s in training.values()), name
            elif overlap.kind == "arm":
                for other in overlap.with_:
                    assert set(item.arms) & set(training[other].arms), (name, other)
            else:
                assert overlap.kind == "init_constant"
                targets = {item.target} | {p.target for p in item.curve or ()}
                for path in overlap.with_:
                    assert _published(path) in targets, (name, path)


def test_a_held_out_item_with_a_training_paper_says_so():
    training_papers = {s.paper for s in ALL_COHORT_STATISTICS}
    authored = {s.name: s for s in _authored()}
    for name, overlaps in SHARES_WITH_TRAINING.items():
        says_paper = any(o.kind == "paper" for o in overlaps)
        assert says_paper == (authored[name].paper in training_papers), name
