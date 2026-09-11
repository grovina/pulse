"""Sparse evidence: literature owns hormones; the teacher tape is vitals."""

from __future__ import annotations

import numpy as np

from pulse.knowledge.cohort_statistics import ALL_COHORT_STATISTICS
from pulse.knowledge.evidence import (
    ALL_EVIDENCE,
    Authority,
    OperatorKind,
    TEACHER_DISTILL_LONG_ONLY,
    TEACHER_DISTILL_MARKERS,
    TEACHER_TAPE_MARKERS,
    from_cohort,
    literature_markers,
    teacher_literature_collisions,
)
from pulse.knowledge.full_body import FullBody
from pulse.knowledge.physiology_rules import PHYSIOLOGY_RULES
from pulse.types import MARKER_INDEX as MI, STATE_DIM
from pulse.training.cold_model_distillation_signal import ColdModelDistillationSignal


def test_cohort_specs_are_literature_evidence() -> None:
    spec = ALL_COHORT_STATISTICS[0]
    item = from_cohort(spec)
    assert item.authority is Authority.LITERATURE
    assert item.name == spec.name
    assert item.observations[0].marker_id == spec.marker_id
    assert item.observations[0].kind in set(OperatorKind)


def test_registry_covers_cohorts_and_rules() -> None:
    names = {item.name for item in ALL_EVIDENCE}
    assert {s.name for s in ALL_COHORT_STATISTICS} <= names
    assert {r.name for r in PHYSIOLOGY_RULES} <= names
    assert "teacher_wearable_tape" in names
    assert "teacher_internal_rate" in names
    assert "teacher_internal_long_level" in names
    assert "teacher_meal_glucose" in names
    assert "dose_glucose_peak" in names
    assert "dose_insulin_peak" in names
    assert "dose_glp1_rank" in names


def test_training_mix_is_the_generator_not_views() -> None:
    from pulse.knowledge import ALL_CONTRIBUTIONS
    assert [c.name for c in ALL_CONTRIBUTIONS] == ["full_body"]


def test_distill_does_not_copy_literature_markers() -> None:
    assert teacher_literature_collisions() == ()
    lit = literature_markers()
    assert not (set(TEACHER_DISTILL_MARKERS) & lit)
    assert "glucagon" in lit and "glucagon" not in TEACHER_DISTILL_MARKERS
    assert "glucose" in lit and "glucose" in TEACHER_TAPE_MARKERS
    by_name = {item.name: item for item in ALL_EVIDENCE}
    rate = by_name["teacher_internal_rate"]
    long = by_name["teacher_internal_long_level"]
    assert rate.authority is Authority.TEACHER
    assert {o.marker_id for o in rate.observations} == (
        set(TEACHER_DISTILL_MARKERS) - set(TEACHER_DISTILL_LONG_ONLY)
    )
    assert {o.marker_id for o in long.observations} == set(TEACHER_DISTILL_LONG_ONLY)


def test_full_body_training_tape_is_wearable_vitals() -> None:
    ep = FullBody(n_days=1).generate_episodes(1, np.random.default_rng(0))[0]
    assert ep.trajectory.shape == (1440, STATE_DIM)
    for m in TEACHER_TAPE_MARKERS:
        assert not np.isnan(ep.trajectory[:, MI[m]]).any(), m
    for m, idx in MI.items():
        if m not in TEACHER_TAPE_MARKERS:
            assert np.isnan(ep.trajectory[:, idx]).all(), m
    assert ep.setpoints is not None
    assert ep.meal_response is not None


def test_long_only_markers_skip_rate_and_short_level() -> None:
    sig = ColdModelDistillationSignal.__new__(ColdModelDistillationSignal)
    sig._long_only = frozenset(TEACHER_DISTILL_LONG_ONLY)
    assert not sig._score_rate("mitochondrial_capacity")
    assert sig._score_rate("crh")
    assert not sig._score_level("mitochondrial_capacity", 60, 60, True)
    assert sig._score_level("mitochondrial_capacity", 200, 60, True)
    assert sig._score_level("mitochondrial_capacity", 60, 60, False)
