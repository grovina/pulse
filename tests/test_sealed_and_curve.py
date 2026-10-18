"""PLAN Phase 0 / Phase 4: the sealed held-out mechanism and the CURVE operator.

These are structural claims — they hold for every registry state, no checkpoint
needed. The sealed split must never leak a held-out item into training, and a
CURVE spec must compile to one point-statistic per timepoint with the
correlation down-weight applied.
"""

from __future__ import annotations

import math

import pytest

from pulse.knowledge.cohort_statistics import (
    ALL_COHORT_STATISTICS,
    SEALED_COHORT_STATISTICS,
    expand_curve_spec,
    expand_curves,
)
from pulse.knowledge.cohort_types import (
    CohortArmSpec,
    CohortStatisticSpec,
    CurvePoint,
    StatisticKind,
    StatisticWindow,
)
from pulse.knowledge.evidence import (
    ALL_EVIDENCE,
    SEALED_EVIDENCE,
    all_evidence,
    sealed_evidence,
)
from pulse.knowledge.physiology_rules import (
    PHYSIOLOGY_RULES,
    SEALED_PHYSIOLOGY_RULES,
    _ALL_PHYSIOLOGY_RULES,
)


_ARM = CohortArmSpec(
    label="ogtt", duration_min=180, start_hour=8.0, meals=((0.0, 75.0, 0.0, 0.0),),
)


def _spec(name: str, *, sealed: bool = False, **kw) -> CohortStatisticSpec:
    base = dict(
        name=name, source="test", description="test", arms=(_ARM,),
        marker_id="glucose", kind=StatisticKind.MEAN_IN_WINDOW,
        window=StatisticWindow(30, 60), target=120.0, sigma=10.0, sealed=sealed,
    )
    base.update(kw)
    return CohortStatisticSpec(**base)


# --- the sealed held-out mechanism -----------------------------------------

def test_training_registries_hold_no_sealed_item():
    assert all(not s.sealed for s in ALL_COHORT_STATISTICS)
    assert all(not r.sealed for r in PHYSIOLOGY_RULES)
    # all_evidence() is built from the training registries, so it is clean too.
    # (teacher tape / distill / dose items are not cohorts or rules — never sealed.)
    assert ALL_EVIDENCE == all_evidence()


def test_sealed_evidence_is_exactly_the_sealed_items():
    assert SEALED_EVIDENCE == sealed_evidence()
    assert len(SEALED_EVIDENCE) == len(SEALED_COHORT_STATISTICS) + len(SEALED_PHYSIOLOGY_RULES)
    # Every physiology rule is accounted for by exactly one of the two lists.
    assert len(_ALL_PHYSIOLOGY_RULES) == len(PHYSIOLOGY_RULES) + len(SEALED_PHYSIOLOGY_RULES)


def test_split_filters_a_constructed_sealed_spec():
    pool = [_spec("keep", sealed=False), _spec("hold_out", sealed=True)]
    train = [s for s in pool if not s.sealed]
    held = [s for s in pool if s.sealed]
    assert [s.name for s in train] == ["keep"]
    assert [s.name for s in held] == ["hold_out"]


# --- the CURVE operator -----------------------------------------------------

def test_curve_compiles_to_one_point_statistic_per_timepoint():
    curve = (CurvePoint(0, 90.0, 2.0), CurvePoint(30, 150.0, 5.0), CurvePoint(120, 100.0, 3.0))
    spec = _spec(
        "glucose_ogtt_curve", kind=StatisticKind.CURVE,
        window=StatisticWindow(0, 180), target=0.0, sigma=1.0, curve=curve,
    )
    pts = expand_curve_spec(spec)
    assert len(pts) == len(curve)
    inflate = math.sqrt(len(curve))
    for cp, pt in zip(curve, pts):
        assert pt.kind is StatisticKind.MEAN_IN_WINDOW
        assert pt.curve is None
        assert pt.n_arm == 1
        assert (pt.window.start_min, pt.window.end_min) == (cp.t_min, cp.t_min + 1)
        assert pt.target == pytest.approx(cp.target)
        # The per-point sigma is the paper's SEM inflated by sqrt(n_points).
        assert pt.sigma == pytest.approx(cp.sigma * inflate)
        assert pt.marker_id == spec.marker_id


def test_curve_sealed_flag_is_preserved_on_every_point():
    curve = (CurvePoint(0, 90.0, 2.0), CurvePoint(60, 120.0, 3.0))
    spec = _spec(
        "sealed_curve", sealed=True, kind=StatisticKind.CURVE,
        window=StatisticWindow(0, 180), target=0.0, sigma=1.0, curve=curve,
    )
    pts = expand_curve_spec(spec)
    assert all(p.sealed for p in pts)
    # So a sealed curve lands entirely in the sealed registry, never in training.
    train = [s for s in expand_curves([spec]) if not s.sealed]
    assert train == []


def test_non_curve_spec_passes_through_expansion_unchanged():
    spec = _spec("plain")
    assert expand_curve_spec(spec) == [spec]


# --- validation -------------------------------------------------------------

def test_curve_kind_requires_a_non_empty_curve():
    with pytest.raises(ValueError, match="needs a non-empty curve"):
        _spec("empty", kind=StatisticKind.CURVE, window=StatisticWindow(0, 180),
              target=0.0, sigma=1.0, curve=())


def test_curve_point_outside_arm_is_rejected():
    with pytest.raises(ValueError, match="outside"):
        _spec("oob", kind=StatisticKind.CURVE, window=StatisticWindow(0, 180),
              target=0.0, sigma=1.0, curve=(CurvePoint(200, 1.0, 1.0),))


def test_curve_on_a_non_curve_kind_is_rejected():
    with pytest.raises(ValueError, match="only for kind=curve"):
        _spec("misuse", curve=(CurvePoint(0, 1.0, 1.0),))
