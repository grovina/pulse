"""The identifiability report: Fisher spectrum of cohort residuals over scalar params."""

from __future__ import annotations

import pytest

from pulse.diagnostics.identifiability import (
    identifiability_report,
    population_scalar_params,
    render,
)
from pulse.knowledge import ALL_COHORT_STATISTICS
from pulse.model import ModularPhysiologyNetwork


@pytest.fixture(scope="module")
def model_and_report():
    model = ModularPhysiologyNetwork()
    report = identifiability_report(model, specs=ALL_COHORT_STATISTICS[:6])
    return model, report


def test_spectrum_shape_and_ordering(model_and_report):
    _, report = model_and_report
    eig = report["eigenvalues"]
    assert len(eig) == len(report["param_names"])
    assert report["n_specs"] == 6
    assert eig == sorted(eig)
    assert min(eig) >= -1e-6
    assert report["condition_number"] >= 1.0


def test_flat_directions_and_diagonal(model_and_report):
    model, report = model_and_report
    assert report["flat_directions"]
    real = {n for n, _ in model.named_parameters()}
    assert set(report["per_param_fisher_diagonal"]) == set(report["param_names"])
    assert set(report["per_param_fisher_diagonal"]) <= real
    assert set(report["param_names"]) == set(population_scalar_params(model))
    for d in report["flat_directions"]:
        assert d["top_params"]
        assert all(n in real for n, _ in d["top_params"])


def test_gradient_flows(model_and_report):
    _, report = model_and_report
    assert report["max_abs_jacobian"] > 0.0


def test_render(model_and_report):
    _, report = model_and_report
    text = render(report)
    assert "flat directions" in text
    assert "condition_number" in text
