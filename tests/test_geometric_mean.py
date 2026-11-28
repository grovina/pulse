"""Geometric-mean statistics: exp(mean ln x) over people, scored on the log scale."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest
import torch

from pulse.cohort_loss import per_person_statistic, score_batch_statistic, score_geometric
from pulse.knowledge.cohort_types import (
    GEOMETRIC_KINDS, CohortArmSpec, CohortStatisticSpec, StatisticKind, StatisticWindow, log_gap,
)
from pulse.knowledge.corpus import CorpusError, parse_finding, read_finding
from pulse.knowledge.corpus_uncertainty import Uncertainty, UncertaintyError, _t_quantile, to_sigma

K = StatisticKind
FINDING = Path(__file__).parent / "fixtures" / "corpus" / "findings" / "synthetic-test-2099.json"


def _spec(kind=K.GEOMETRIC_MEAN, target=100.0, sigma=0.5, n_arm=10, **kw):
    arm = CohortArmSpec(label="a", duration_min=60, start_hour=8.0, meals=())
    times = kw.pop("times", None)
    window = StatisticWindow(times[0], times[-1] + 1) if times else StatisticWindow(0, 60)
    return CohortStatisticSpec(
        name="g", source="t", description="", paper="x-2000", arms=(arm,), marker_id="insulin",
        kind=kind, window=window, target=target, sigma=sigma, n_arm=n_arm,
        sample_times=tuple(times) if times else None, **kw,
    )


def _data() -> dict:
    return copy.deepcopy(json.loads(FINDING.read_text(encoding="utf-8")))


def _item(data, fid):
    return next(f for f in data["findings"] if f["id"] == fid)


def test_a_lognormal_cohort_has_its_geometric_mean_not_its_arithmetic_one():
    g = torch.Generator().manual_seed(0)
    gm, s = 60.0, 0.6
    x = gm * torch.exp(s * torch.randn(20000, generator=g, dtype=torch.float64))
    series = x.unsqueeze(1).repeat(1, 60)  # each person flat at their own level
    spec = _spec(target=gm, sigma=s)
    per_person = per_person_statistic(series, spec)
    _, pred, z = score_geometric(per_person, spec)
    assert pred == pytest.approx(gm, rel=0.02)
    assert float(per_person.mean()) > 1.1 * gm  # the arithmetic mean sits well above
    assert abs(z) < 0.05


def test_the_geometric_mean_is_exp_of_the_mean_log():
    vals = torch.tensor([10.0, 40.0, 90.0], dtype=torch.float64)
    _, pred, _ = score_geometric(vals, _spec())
    assert pred == pytest.approx(math.exp(sum(math.log(v) for v in (10, 40, 90)) / 3))


def test_z_is_on_the_log_scale():
    spec = _spec(target=50.0, sigma=0.5, n_arm=4)
    vals = torch.full((4,), 100.0, dtype=torch.float64)
    loss, pred, z = score_geometric(vals, spec)
    assert pred == pytest.approx(100.0)
    assert z == pytest.approx(math.log(2.0) / 0.5)
    assert log_gap(100.0, spec) == pytest.approx(math.log(2.0))
    assert float(loss) == pytest.approx(math.log(2.0) ** 2 / (0.5 / 2) ** 2)


def test_the_loss_is_the_mean_loss_on_ln_values():
    g = torch.Generator().manual_seed(3)
    vals = (80.0 * torch.exp(0.4 * torch.randn(8, generator=g))).requires_grad_(True)
    spec = _spec(target=50.0, sigma=0.5, n_arm=8)
    loss, _, _ = score_geometric(vals, spec)
    sem = 0.5 / math.sqrt(8)
    logs = vals.log()
    resid = logs.mean() - math.log(50.0)
    want = torch.relu(resid**2 - logs.var(correction=1) / 8) / sem**2
    assert float(loss) == pytest.approx(float(want), rel=1e-6)
    loss.backward()
    assert vals.grad is not None and torch.isfinite(vals.grad).all()


def test_a_non_positive_prediction_is_finite_and_far():
    vals = torch.tensor([0.0, -1.0, 5.0], dtype=torch.float64)
    loss, pred, z = score_geometric(vals, _spec())
    assert math.isfinite(float(loss)) and pred > 0 and z < 0


def test_a_geometric_target_has_to_be_positive_and_a_point():
    with pytest.raises(ValueError, match="positive"):
        _spec(target=-1.0)
    with pytest.raises(ValueError, match="point"):
        from pulse.knowledge.cohort_types import TargetShape
        _spec(shape=TargetShape.AT_MOST)


def test_a_geometric_area_is_the_trapezoid_of_each_person():
    times = (0, 30, 59)
    series = torch.full((3, 60), 20.0, dtype=torch.float64)
    spec = _spec(K.GEOMETRIC_AREA, target=20.0, times=times)
    assert per_person_statistic(series, spec).tolist() == pytest.approx([20.0] * 3)
    assert K.GEOMETRIC_AREA in GEOMETRIC_KINDS


def test_the_ci_becomes_a_log_scale_sigma():
    u = Uncertainty("ci", 12, low=0.0, high=math.log(50.0 / 32.0), level=95)
    sigma, n_arm = to_sigma(K.GEOMETRIC_MEAN, 40.0, u)
    se_log = math.log(50.0 / 32.0) / (2 * _t_quantile(95, 11))
    assert n_arm == 12 and sigma == pytest.approx(se_log * math.sqrt(12))
    with pytest.raises(UncertaintyError):
        to_sigma(K.GEOMETRIC_MEAN, 40.0, Uncertainty("sd", 12, value=5.0))


def test_an_sd_of_the_logs_is_sigma_itself():
    for kind in (K.GEOMETRIC_MEAN, K.GEOMETRIC_AREA):
        sigma, n_arm = to_sigma(kind, 40.0, Uncertainty("sd_log", 12, value=0.6))
        assert sigma == 0.6 and n_arm == 12
    with pytest.raises(UncertaintyError):
        to_sigma(K.GEOMETRIC_MEAN, 40.0, Uncertainty("sd_log", 12, value=0.0))


def test_z_with_an_sd_log_is_on_the_log_scale():
    spec = _spec(target=100.0, sigma=0.6, n_arm=10)
    _, pred, z = score_geometric(torch.full((10,), 150.0, dtype=torch.float64), spec)
    assert pred == pytest.approx(150.0)
    assert z == pytest.approx((math.log(150.0) - math.log(100.0)) / 0.6)


def test_the_loader_takes_an_sd_log_as_sigma():
    data = _data()
    _item(data, "ogtt_insulin_geometric_mean")["uncertainty"] = {"kind": "sd_log", "value": 0.7, "n": 12}
    f = parse_finding(data)
    spec = next(s for s in f.specs if s.name.endswith("ogtt_insulin_geometric_mean"))
    assert spec.sigma == pytest.approx(0.7) and spec.n_arm == 12


def test_the_gate_rejects_sd_log_on_an_arithmetic_mean_and_a_non_positive_one():
    data = _data()
    _item(data, "ogtt_insulin_geometric_mean")["uncertainty"] = {"kind": "sd_log", "value": 0.0, "n": 12}
    assert any("positive" in p for p in _problems(data))
    data = _data()
    _item(data, "fasting_glucose_mean")["uncertainty"] = {"kind": "sd_log", "value": 0.5, "n": 12}
    assert any("takes an uncertainty of kind" in p for p in _problems(data))


def test_the_loader_reads_a_geometric_mean_and_its_ci_on_the_log_scale():
    f = read_finding(FINDING)
    spec = next(s for s in f.specs if s.name.endswith("ogtt_insulin_geometric_mean"))
    assert spec.kind is K.GEOMETRIC_MEAN and spec.n_arm == 12
    assert 0 < spec.target < 40.0  # pmol/L in the model's uU/mL
    se_log = math.log(50.0 / 32.0) / (2 * _t_quantile(95, 11))
    assert spec.sigma == pytest.approx(se_log * math.sqrt(12))
    area = next(s for s in f.specs if s.name.endswith("ogtt_insulin_geometric_auc"))
    assert area.kind is K.GEOMETRIC_AREA and area.sample_times == (0, 30, 60, 90, 119)


def _problems(data):
    with pytest.raises(CorpusError) as err:
        parse_finding(data)
    return err.value.reasons


def test_the_gate_rejects_a_ci_that_does_not_contain_the_geometric_mean():
    data = _data()
    _item(data, "ogtt_insulin_geometric_mean")["value"] = 60.0
    assert any("outside its ci" in p for p in _problems(data))


def test_the_gate_rejects_non_positive_values_and_ends():
    data = _data()
    _item(data, "ogtt_insulin_geometric_mean")["uncertainty"]["low"] = 0.0
    assert any("positive" in p for p in _problems(data))
    data = _data()
    _item(data, "ogtt_insulin_geometric_mean")["value"] = -5.0
    assert any("positive" in p or "outside its ci" in p for p in _problems(data))


def test_a_geometric_mean_takes_only_a_ci():
    data = _data()
    _item(data, "ogtt_insulin_geometric_mean")["uncertainty"] = {"kind": "sd", "value": 10.0, "n": 12}
    assert any("takes an uncertainty of kind" in p for p in _problems(data))


def test_arithmetic_means_score_as_before():
    spec = CohortStatisticSpec(
        name="m", source="t", description="", paper="x-2000",
        arms=(CohortArmSpec(label="a", duration_min=60, start_hour=8.0, meals=()),),
        marker_id="insulin", kind=K.MEAN_IN_WINDOW, window=StatisticWindow(0, 60), target=50.0, sigma=20.0, n_arm=4,
    )
    loss, pred, z = score_batch_statistic(torch.full((4,), 100.0, dtype=torch.float64), spec)
    assert pred == pytest.approx(100.0) and z == pytest.approx(2.5)
    assert float(loss) == pytest.approx(2500.0 / (20.0 / 2) ** 2)
