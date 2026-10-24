"""The identifiability report: the Fisher spectrum of the literature residuals."""

from __future__ import annotations

import pytest
import torch

from pulse.diagnostics.identifiability import (
    _conflicts,
    identifiability_report,
    population_scalar_params,
    render,
    studies_by_parameter,
)
from pulse.knowledge import ALL_COHORT_STATISTICS
from pulse.knowledge.physiology_rules import PHYSIOLOGY_RULES
from pulse.model import ModularPhysiologyNetwork
from pulse.types import EMBEDDING_DIM


@pytest.fixture(scope="module")
def model_and_report():
    model = ModularPhysiologyNetwork()
    report = identifiability_report(
        model, specs=ALL_COHORT_STATISTICS[:6], rules=PHYSIOLOGY_RULES[:2],
    )
    return model, report


def test_spectrum_shape_and_ordering(model_and_report):
    _, report = model_and_report
    eig = report["eigenvalues"]
    assert len(eig) == len(report["param_names"])
    assert report["n_items"] == 8
    assert report["n_residuals"] >= 8
    assert eig == sorted(eig)
    assert min(eig) >= -1e-6
    assert report["condition_number"] >= 1.0


def test_open_directions_are_below_the_floor(model_and_report):
    model, report = model_and_report
    real = {n for n, _ in model.named_parameters()}
    assert set(report["coverage"]) == set(report["param_names"]) == set(population_scalar_params(model))
    assert report["n_open"] == len(report["open_directions"])
    # Six cohorts and two rules cannot pin dozens of scalars.
    assert report["n_open"] >= len(report["param_names"]) - report["n_residuals"]
    for d in report["open_directions"]:
        assert d["top_params"]
        assert all(p["param"] in real for p in d["top_params"])


def test_a_held_parameter_names_the_items_that_hold_it(model_and_report):
    _, report = model_and_report
    held = [c for c in report["coverage"].values() if c["held_by"]]
    assert held
    items = {s.name for s in ALL_COHORT_STATISTICS[:6]} | {r.name for r in PHYSIOLOGY_RULES[:2]}
    assert all(set(c["held_by"]) <= items for c in held)


def test_the_person_axis_is_bounded_by_the_code(model_and_report):
    _, report = model_and_report
    person = report["person_axis"]
    assert person["dim"] == EMBEDDING_DIM
    assert 0 < person["n_constrained"] <= min(EMBEDDING_DIM, report["n_residuals"])


def test_every_study_names_a_real_parameter():
    model = ModularPhysiologyNetwork()
    real = {n for n, _ in model.named_parameters()}
    assert set(studies_by_parameter(model)) <= real


def test_opposed_items_outside_a_sigma_conflict():
    jac = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    # a wants the first parameter down, b wants it up; c is met.
    r = torch.tensor([2.0, -3.0, 0.0], dtype=torch.float64)
    out = _conflicts(["a", "b", "c"], r, jac)
    assert [c["items"] for c in out] == [["a", "b"]]
    assert out[0]["cosine"] == pytest.approx(-1.0)
    # Inside a sigma there is nothing to trade.
    assert _conflicts(["a", "b"], torch.tensor([0.5, -3.0], dtype=torch.float64), jac[:2]) == []


def test_gradient_flows(model_and_report):
    _, report = model_and_report
    assert report["max_abs_jacobian"] > 0.0


def test_render(model_and_report):
    _, report = model_and_report
    text = render(report)
    assert "open directions" in text
    assert "person axis" in text
    assert "condition_number" in text
