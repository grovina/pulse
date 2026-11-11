"""Authored CURVE evidence: the cortisol circadian curve trains, the OGTT curve is held out."""

from __future__ import annotations

import torch

from pulse.cohort_loss import cohort_statistic_loss_one_spec, norm_center_initial_state
from pulse.knowledge.cohort_statistics import (
    ALL_COHORT_STATISTICS,
    SEALED_COHORT_STATISTICS,
)
from pulse.knowledge.cohort_types import StatisticKind
from pulse.knowledge.cohorts.curves import CORTISOL_CIRCADIAN_CURVE, OGTT_GLUCOSE_CURVE
from pulse.knowledge.evidence import SEALED_EVIDENCE
from pulse.knowledge.sealing import HELD_OUT_ITEMS
from pulse.model import ModularPhysiologyNetwork
from pulse.types import EMBEDDING_DIM


def _points(registry, curve_spec):
    return [s for s in registry if s.name.startswith(curve_spec.name + "__t")]


def test_unsealed_cortisol_curve_expands_into_training_points():
    assert not CORTISOL_CIRCADIAN_CURVE.sealed
    pts = _points(ALL_COHORT_STATISTICS, CORTISOL_CIRCADIAN_CURVE)
    assert len(pts) == len(CORTISOL_CIRCADIAN_CURVE.curve)
    for p in pts:
        assert p.kind is StatisticKind.MEAN_IN_WINDOW
        assert p.marker_id == "cortisol"
        assert p.n_arm == 1
    assert _points(SEALED_COHORT_STATISTICS, CORTISOL_CIRCADIAN_CURVE) == []


def test_held_out_ogtt_curve_is_only_in_the_sealed_registry():
    assert OGTT_GLUCOSE_CURVE.name in HELD_OUT_ITEMS
    assert not any(s.name.startswith(OGTT_GLUCOSE_CURVE.name) for s in ALL_COHORT_STATISTICS)
    held = _points(SEALED_COHORT_STATISTICS, OGTT_GLUCOSE_CURVE)
    assert len(held) == len(OGTT_GLUCOSE_CURVE.curve)
    assert all(s.sealed and s.marker_id == "glucose" for s in held)


def test_sealed_evidence_contains_the_ogtt_points():
    assert SEALED_EVIDENCE
    item_names = {it.name for it in SEALED_EVIDENCE}
    held = _points(SEALED_COHORT_STATISTICS, OGTT_GLUCOSE_CURVE)
    assert held
    assert {s.name for s in held} <= item_names


def test_cortisol_curve_point_carries_gradient_to_the_model():
    torch.manual_seed(0)
    model = ModularPhysiologyNetwork()
    # The acrophase point: a t=0 point reads the initial state, which no model
    # parameter has touched yet, so only later points carry gradient.
    spec = next(
        s for s in _points(ALL_COHORT_STATISTICS, CORTISOL_CIRCADIAN_CURVE)
        if s.window.start_min == 480
    )
    state = norm_center_initial_state()(spec)
    loss, _pred, _z = cohort_statistic_loss_one_spec(
        model, [torch.zeros(EMBEDDING_DIM)], spec, state,
    )
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert any(float(g.abs().sum()) > 0.0 for g in grads)
