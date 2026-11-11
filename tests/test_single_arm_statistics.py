"""Change from baseline, area under the curve and incremental area: one arm against itself.

The arithmetic is checked on trajectories small enough to integrate by hand, the loader on
the fixture paper, and the loss on a short rollout: the per-spec path and the batched step
must give the same number, and the number must be the one worked out from the trajectory.
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from pulse.admit import admit, format_admission
from pulse.cohort_loss import (
    _rollout_arm_batched, cohort_statistic_loss_one_spec, per_person_statistic,
)
from pulse.knowledge.cohort_types import (
    BelowBaseline, CohortArmSpec, CohortStatisticSpec, StatisticKind, StatisticWindow,
    statistic_cuts,
)
from pulse.knowledge.corpus import CorpusError, parse_finding, read_finding
from pulse.model import ModularPhysiologyNetwork
from pulse.scoreboard import target_conflicts
from pulse.training import SignalContext
from pulse.training.evidence_step import EvidenceTerm, accumulate_evidence_step, item_nll
from pulse.types import NORM_CENTER, SERIES_INDEX

FINDING = Path(__file__).parent / "fixtures" / "corpus" / "findings" / "synthetic-test-2099.json"
MG_DL_PER_MMOL = 18.016
TIMES = (0, 30, 60, 90, 120)


# --- the arithmetic, on trajectories integrated by hand -------------------------------

def _hand_spec(kind, *, times=TIMES, below=None, baseline=None, window=None):
    arm = CohortArmSpec(label="a", duration_min=121, start_hour=8.0, meals=())
    if window is None:
        window = StatisticWindow(times[0], times[-1] + 1)
    return CohortStatisticSpec(
        name="hand", source="t", description="", arms=(arm,), marker_id="glucose", kind=kind,
        window=window, target=1.0, sigma=1.0, baseline=baseline,
        sample_times=None if kind is StatisticKind.CHANGE_FROM_BASELINE else tuple(times),
        below_baseline=below,
    )


def _series(rows: list[list[float]], times=TIMES, off_sample=0.0) -> torch.Tensor:
    """``[B, 121]`` trajectories that hold ``rows`` at ``times`` and ``off_sample`` at every
    other minute, so a statistic that reads a minute it should not shows up."""
    out = torch.full((len(rows), 121), off_sample, dtype=torch.float64)
    for b, values in enumerate(rows):
        for t, v in zip(times, values):
            out[b, t] = v
    return out


def test_area_under_curve_is_the_trapezoid_over_the_listed_times():
    # 30 * [(100+160)/2 + (160+140)/2 + (140+90)/2 + (90+100)/2] = 14700 over 120 min
    got = per_person_statistic(_series([[100, 160, 140, 90, 100]]), _hand_spec(StatisticKind.AREA_UNDER_CURVE))
    assert float(got[0]) == pytest.approx(14700.0 / 120.0)


def test_the_area_reads_the_listed_minutes_and_no_others():
    row = [[100, 160, 140, 90, 100]]
    spec = _hand_spec(StatisticKind.AREA_UNDER_CURVE)
    quiet = per_person_statistic(_series(row), spec)
    spiked = _series(row, off_sample=-500.0)
    spiked[0, 15] = 1000.0  # a spike between two sampling times
    spiked[0, 45] = -1000.0
    assert float(per_person_statistic(spiked, spec)[0]) == pytest.approx(float(quiet[0]))
    # A dense integral of the same curve is a different number: the spike would move it.
    assert float(spiked[0].mean()) != pytest.approx(float(quiet[0]))


def test_unevenly_spaced_samples_are_weighted_by_the_gap():
    # samples at 0, 10, 120: 10 * (100+200)/2 + 110 * (200+100)/2 = 18000 over 120
    times = (0, 10, 120)
    got = per_person_statistic(_series([[100, 200, 100]], times), _hand_spec(StatisticKind.AREA_UNDER_CURVE, times=times))
    assert float(got[0]) == pytest.approx(18000.0 / 120.0)


def test_incremental_area_subtracted_counts_what_is_under_baseline_negative():
    # baseline 100 (first sample): d = 0, 60, 40, -10, 0
    # 30 * [30 + 50 + 15 - 5] = 2700 over 120
    spec = _hand_spec(StatisticKind.INCREMENTAL_AREA, below=BelowBaseline.SUBTRACTED)
    got = per_person_statistic(_series([[100, 160, 140, 90, 100]]), spec)
    assert float(got[0]) == pytest.approx(2700.0 / 120.0)


def test_incremental_area_ignored_cuts_a_crossing_segment_and_drops_one_wholly_below():
    spec = _hand_spec(StatisticKind.INCREMENTAL_AREA, below=BelowBaseline.IGNORED)
    # Person 0: d = 0, 60, 40, -10, 0. The segment 40 -> -10 crosses baseline at 4/5 of its
    # length: 30 * 40 * 0.8 / 2 = 480. With 900 and 1500 before it, and the last segment
    # (-10 -> 0, wholly below) counting nothing: 2880 over 120.
    # Person 1: d = 0, -20, -40, 20, 10. Two segments wholly below count nothing. The
    # segment -40 -> 20 crosses at 2/3 of its length: 30 * 20 * (1/3) / 2 = 100; the last
    # is 30 * 15 = 450: 550 over 120.
    got = per_person_statistic(
        _series([[100, 160, 140, 90, 100], [100, 80, 60, 120, 110]]), spec,
    )
    assert got.tolist() == pytest.approx([2880.0 / 120.0, 550.0 / 120.0])
    net = per_person_statistic(
        _series([[100, 160, 140, 90, 100], [100, 80, 60, 120, 110]]),
        _hand_spec(StatisticKind.INCREMENTAL_AREA, below=BelowBaseline.SUBTRACTED),
    )
    assert net.tolist() == pytest.approx([2700.0 / 120.0, -1050.0 / 120.0])


def test_an_unstated_incremental_area_is_held_to_the_nearer_of_the_two_readings():
    rows = _series([[100, 160, 140, 90, 100], [100, 80, 60, 120, 110]])
    net = [2700.0 / 120.0, -1050.0 / 120.0]     # mean 6.875
    above = [2880.0 / 120.0, 550.0 / 120.0]     # mean 14.29
    spec = _hand_spec(StatisticKind.INCREMENTAL_AREA, below=BelowBaseline.UNSTATED)
    # The paper's value lies under both readings: the model is held to the net one.
    assert per_person_statistic(rows, replace(spec, target=5.0)).tolist() == pytest.approx(net)
    # Over both: to the one that ignores what is under baseline.
    assert per_person_statistic(rows, replace(spec, target=20.0)).tolist() == pytest.approx(above)
    # Between them: the people's mean is the paper's value, whichever of the two it printed.
    got = per_person_statistic(rows, replace(spec, target=10.0))
    assert float(got.mean()) == pytest.approx(10.0)
    assert all(lo <= x <= hi for lo, x, hi in zip(net, got.tolist(), above))


def test_an_unstated_area_between_the_readings_pulls_on_nothing():
    rows = _series([[100, 160, 140, 90, 100], [100, 80, 60, 120, 110]]).requires_grad_(True)
    spec = _hand_spec(StatisticKind.INCREMENTAL_AREA, below=BelowBaseline.UNSTATED)
    inside = (per_person_statistic(rows, replace(spec, target=10.0)).mean() - 10.0) ** 2
    (grad,) = torch.autograd.grad(inside, rows)
    assert float(grad.abs().max()) == pytest.approx(0.0, abs=1e-9)
    # Outside them it pulls exactly as the nearer reading would.
    outside = (per_person_statistic(rows, replace(spec, target=5.0)).mean() - 5.0) ** 2
    (grad,) = torch.autograd.grad(outside, rows)
    net_spec = _hand_spec(StatisticKind.INCREMENTAL_AREA, below=BelowBaseline.SUBTRACTED)
    (want,) = torch.autograd.grad((per_person_statistic(rows, net_spec).mean() - 5.0) ** 2, rows)
    assert float(grad.abs().max()) > 0 and torch.allclose(grad, want)


def test_an_area_that_never_goes_under_baseline_is_one_number_under_all_three_words():
    rows = _series([[100, 160, 140, 120, 100], [100, 130, 150, 120, 110]])
    subtracted, ignored, unstated = (
        per_person_statistic(rows, _hand_spec(StatisticKind.INCREMENTAL_AREA, below=b)).tolist()
        for b in (BelowBaseline.SUBTRACTED, BelowBaseline.IGNORED, BelowBaseline.UNSTATED)
    )
    assert subtracted == pytest.approx(ignored) and unstated == pytest.approx(ignored)


def test_incremental_area_takes_its_baseline_from_a_window_when_there_is_one():
    # Minutes 0 and 1 hold 100 and 120, so the baseline is 110 and not the first sample's
    # 100: d = -10, 50, 30, -20, -10.
    rows = _series([[100, 160, 140, 90, 100]])
    rows[0, 1] = 120.0
    window = StatisticWindow(0, 2)
    ignored = per_person_statistic(rows, _hand_spec(
        StatisticKind.INCREMENTAL_AREA, below=BelowBaseline.IGNORED, baseline=window))
    net = per_person_statistic(rows, _hand_spec(
        StatisticKind.INCREMENTAL_AREA, below=BelowBaseline.SUBTRACTED, baseline=window))
    # ignored: crossing up 30 * 50 * (5/6) / 2 = 625; 1200; crossing down 30 * 30 * (3/5) / 2 = 270; 0
    assert float(ignored[0]) == pytest.approx(2095.0 / 120.0)
    # subtracted: 600 + 1200 + 150 - 450 = 1500
    assert float(net[0]) == pytest.approx(1500.0 / 120.0)


def test_the_ignored_area_has_a_finite_gradient_through_every_branch():
    rows = _series([[100, 160, 140, 90, 100], [100, 80, 60, 120, 110]]).requires_grad_(True)
    per_person_statistic(
        rows, _hand_spec(StatisticKind.INCREMENTAL_AREA, below=BelowBaseline.IGNORED),
    ).sum().backward()
    assert torch.isfinite(rows.grad).all()
    assert float(rows.grad.abs().sum()) > 0.0
    # Only the sampling minutes carry weight.
    off = torch.ones(121, dtype=torch.bool)
    off[list(TIMES)] = False
    assert float(rows.grad[:, off].abs().sum()) == 0.0


def test_change_from_baseline_is_the_window_mean_minus_the_baseline_mean():
    # value = a + c t, so the mean over [60, 90) is a + 74.5 c and over [0, 30) a + 14.5 c
    t = torch.arange(121, dtype=torch.float64)
    a = torch.tensor([[100.0], [80.0]])
    c = torch.tensor([[0.5], [-0.25]])
    spec = _hand_spec(
        StatisticKind.CHANGE_FROM_BASELINE, window=StatisticWindow(60, 90), baseline=StatisticWindow(0, 30),
    )
    got = per_person_statistic(a + c * t, spec)
    assert got.tolist() == pytest.approx([30.0, -15.0])


def test_a_sample_time_past_the_series_is_an_error_not_a_clipped_read():
    with pytest.raises(ValueError, match="past the arm"):
        per_person_statistic(_series([[1, 2, 3, 4, 5]])[:, :100], _hand_spec(StatisticKind.AREA_UNDER_CURVE))


def test_the_spec_refuses_what_the_loss_could_not_read():
    base = dict(
        name="s", source="t", description="", marker_id="glucose", target=1.0, sigma=1.0,
        arms=(CohortArmSpec(label="a", duration_min=121, start_hour=8.0, meals=()),),
    )
    auc = StatisticKind.AREA_UNDER_CURVE
    window = StatisticWindow(0, 121)
    with pytest.raises(ValueError, match="at least two"):
        CohortStatisticSpec(kind=auc, window=StatisticWindow(0, 1), sample_times=(0,), **base)
    with pytest.raises(ValueError, match="strictly increasing"):
        CohortStatisticSpec(kind=auc, window=window, sample_times=(0, 60, 60, 120), **base)
    with pytest.raises(ValueError, match="outside the arm"):
        CohortStatisticSpec(kind=auc, window=StatisticWindow(0, 122), sample_times=(0, 121), **base)
    with pytest.raises(ValueError, match="window must be"):
        CohortStatisticSpec(kind=auc, window=StatisticWindow(0, 60), sample_times=(0, 120), **base)
    with pytest.raises(ValueError, match="needs below_baseline"):
        CohortStatisticSpec(kind=StatisticKind.INCREMENTAL_AREA, window=window, sample_times=(0, 120), **base)
    with pytest.raises(ValueError, match="only for kind=incremental_area"):
        CohortStatisticSpec(
            kind=auc, window=window, sample_times=(0, 120), below_baseline=BelowBaseline.IGNORED, **base,
        )
    with pytest.raises(ValueError, match="needs a baseline window"):
        CohortStatisticSpec(kind=StatisticKind.CHANGE_FROM_BASELINE, window=window, **base)
    with pytest.raises(ValueError, match="sample_times is only for"):
        CohortStatisticSpec(kind=StatisticKind.MEAN_IN_WINDOW, window=window, sample_times=(0, 5), **base)


def test_the_cuts_a_meal_must_not_cross_include_the_sample_minutes_and_the_baseline():
    spec = _hand_spec(StatisticKind.INCREMENTAL_AREA, below=BelowBaseline.IGNORED, baseline=StatisticWindow(0, 2))
    cuts = set(statistic_cuts(spec))
    assert {0, 1, 2, 30, 31, 60, 61, 90, 91, 120, 121} <= cuts
    change = _hand_spec(
        StatisticKind.CHANGE_FROM_BASELINE, window=StatisticWindow(60, 90), baseline=StatisticWindow(0, 30),
    )
    assert set(statistic_cuts(change)) == {0, 30, 60, 90}


# --- the loader ----------------------------------------------------------------------

def _data() -> dict:
    return copy.deepcopy(json.loads(FINDING.read_text(encoding="utf-8")))


def _finding(data: dict, fid: str) -> dict:
    return next(f for f in data["findings"] if f["id"] == fid)


def _spec(fid: str, data: dict | None = None):
    finding = parse_finding(data or _data())
    return next(i.spec for i in finding.items if i.id == fid)


def _problems(data: dict) -> list[str]:
    with pytest.raises(CorpusError) as err:
        parse_finding(data)
    return err.value.reasons


def test_the_three_statistics_arrive_in_the_markers_unit():
    # the fixture's ogtt arm has minutes 0..119, so its five samples span 119 minutes
    auc = _spec("ogtt_glucose_auc")
    assert auc.kind is StatisticKind.AREA_UNDER_CURVE and auc.sample_times == (0, 30, 60, 90, 119)
    assert auc.window == StatisticWindow(0, 120) and auc.baseline is None and auc.below_baseline is None
    assert auc.target == pytest.approx(900.0 / 119.0 * MG_DL_PER_MMOL)
    assert auc.sigma == pytest.approx(100.0 / 119.0 * MG_DL_PER_MMOL) and auc.n_arm == 12

    iauc = _spec("ogtt_glucose_iauc")
    assert iauc.below_baseline is BelowBaseline.IGNORED and iauc.baseline is None
    assert iauc.target == pytest.approx(210.0 / 119.0 * MG_DL_PER_MMOL)
    # an SEM of 25 is a between-person SD of 25 * sqrt(12), scaled like the value
    assert iauc.sigma == pytest.approx(25.0 * 12 ** 0.5 / 119.0 * MG_DL_PER_MMOL) and iauc.n_arm == 12

    change = _spec("ogtt_glucose_second_hour_rise")
    assert change.window == StatisticWindow(60, 120) and change.baseline == StatisticWindow(0, 1)
    assert change.sample_times is None
    assert change.target == pytest.approx(1.4 * MG_DL_PER_MMOL)
    assert change.sigma == pytest.approx(0.2 * 12 ** 0.5 * MG_DL_PER_MMOL) and change.n_arm == 12


def test_hours_in_the_time_unit_are_sixty_minutes():
    data = _data()
    _finding(data, "ogtt_glucose_auc").update({"value": 15.0, "time_unit": "h"})
    _finding(data, "ogtt_glucose_auc")["uncertainty"]["value"] = 100.0 / 60.0
    spec = _spec("ogtt_glucose_auc", data)
    assert spec.target == pytest.approx(900.0 / 119.0 * MG_DL_PER_MMOL)
    assert spec.sigma == pytest.approx(100.0 / 119.0 * MG_DL_PER_MMOL)


def test_a_confidence_interval_is_scaled_like_the_value():
    data = _data()
    _finding(data, "ogtt_glucose_auc")["uncertainty"] = {"kind": "ci", "low": 840.0, "high": 960.0, "level": 95, "n": 12}
    # sd = (120 / (2 * 2.201)) * sqrt(12); the same factor as the value
    want = 120.0 / (2 * 2.201) * 12 ** 0.5 / 119.0 * MG_DL_PER_MMOL
    assert _spec("ogtt_glucose_auc", data).sigma == pytest.approx(want)


def test_an_area_in_the_wrong_unit_shows_up_as_an_absurd_mean():
    data = _data()
    _finding(data, "ogtt_glucose_auc")["unit"] = "mg/dL"   # 900 mg/dL x min is 7.6 mg/dL
    assert any("outside [40, 400]" in p for p in _problems(data))
    data = _data()
    _finding(data, "ogtt_glucose_iauc")["value"] = 90000.0   # an incremental mean wider than the whole range
    assert any("wider than its whole range" in p for p in _problems(data))


@pytest.mark.parametrize("fid, key", [
    ("ogtt_glucose_second_hour_rise", "window"),
    ("ogtt_glucose_second_hour_rise", "baseline_window"),
    ("ogtt_glucose_second_hour_rise", "value"),
    ("ogtt_glucose_second_hour_rise", "uncertainty"),
    ("ogtt_glucose_auc", "times"),
    ("ogtt_glucose_auc", "time_unit"),
    ("ogtt_glucose_auc", "value"),
    ("ogtt_glucose_auc", "uncertainty"),
    ("ogtt_glucose_iauc", "times"),
    ("ogtt_glucose_iauc", "time_unit"),
    ("ogtt_glucose_iauc", "below_baseline"),
    ("ogtt_glucose_iauc", "value"),
    ("ogtt_glucose_iauc", "uncertainty"),
])
def test_each_required_key_is_required(fid, key):
    data = _data()
    del _finding(data, fid)[key]
    assert any(f"finding {fid} is missing" in p and key in p for p in _problems(data))


@pytest.mark.parametrize("fid, key, value", [
    ("ogtt_glucose_auc", "window", {"start_min": 0, "end_min": 120}),
    ("ogtt_glucose_auc", "baseline_window", {"start_min": 0, "end_min": 1}),
    ("ogtt_glucose_auc", "below_baseline", "ignored"),
    ("ogtt_glucose_second_hour_rise", "times", [0, 60]),
    ("ogtt_glucose_second_hour_rise", "time_unit", "min"),
    ("ogtt_glucose_second_hour_rise", "per_arm_windows", []),
    ("ogtt_glucose_iauc", "window", {"start_min": 0, "end_min": 120}),
])
def test_a_key_of_another_statistic_is_refused(fid, key, value):
    data = _data()
    _finding(data, fid)[key] = value
    assert any(f"finding {fid} has unknown fields" in p and key in p for p in _problems(data))


@pytest.mark.parametrize("times, message", [
    ([0, 60, 30, 90], "not strictly increasing"),
    ([0, 30, 30, 90], "not strictly increasing"),
    ([90, 60, 30, 0], "not strictly increasing"),
    ([0, 30, 60, 120], "outside the arm's minutes 0..119"),
    ([-5, 30, 60], "outside the arm's minutes 0..119"),
    ([30], "at least two"),
    ([], "at least two"),
    ("0, 30, 60", "at least two"),
    ([0, 30.5, 60], "whole minutes"),
    ([0, True, 60], "whole minutes"),
    ([0, "30", 60], "whole minutes"),
])
@pytest.mark.parametrize("fid", ["ogtt_glucose_auc", "ogtt_glucose_iauc"])
def test_the_sampling_times_are_checked_against_the_arm(fid, times, message):
    data = _data()
    _finding(data, fid)["times"] = times
    assert any(message in p for p in _problems(data))


@pytest.mark.parametrize("bad", ["ignore", "Ignored", "positive", "", None, True, 1])
def test_below_baseline_must_be_one_of_the_three_words(bad):
    data = _data()
    _finding(data, "ogtt_glucose_iauc")["below_baseline"] = bad
    assert any("below_baseline" in p for p in _problems(data))


@pytest.mark.parametrize("bad", ["s", "hours", "minutes", "Min", "", None, 60])
@pytest.mark.parametrize("fid", ["ogtt_glucose_auc", "ogtt_glucose_iauc"])
def test_the_time_unit_is_minutes_or_hours(fid, bad):
    data = _data()
    _finding(data, fid)["time_unit"] = bad
    assert any("time_unit" in p for p in _problems(data))


def test_an_ignored_incremental_area_cannot_be_negative_but_a_net_one_can():
    data = _data()
    _finding(data, "ogtt_glucose_iauc")["value"] = -3.0
    assert any("cannot be negative" in p for p in _problems(data))
    _finding(data, "ogtt_glucose_iauc")["below_baseline"] = "subtracted"
    spec = _spec("ogtt_glucose_iauc", data)
    assert spec.below_baseline is BelowBaseline.SUBTRACTED and spec.target == pytest.approx(-3.0 / 119.0 * MG_DL_PER_MMOL)


def test_an_unstated_incremental_area_loads_and_may_be_negative():
    data = _data()
    _finding(data, "ogtt_glucose_iauc")["below_baseline"] = "unstated"
    assert _spec("ogtt_glucose_iauc", data).below_baseline is BelowBaseline.UNSTATED
    _finding(data, "ogtt_glucose_iauc")["value"] = -3.0
    assert _spec("ogtt_glucose_iauc", data).target < 0


def test_a_baseline_window_outside_the_arm_or_empty_is_refused():
    data = _data()
    _finding(data, "ogtt_glucose_second_hour_rise")["baseline_window"] = {"start_min": 100, "end_min": 125}
    assert any("baseline_window" in p and "duration_min" in p for p in _problems(data))
    data = _data()
    _finding(data, "ogtt_glucose_second_hour_rise")["baseline_window"] = {"start_min": 10, "end_min": 10}
    assert any("baseline_window" in p for p in _problems(data))


def test_an_incremental_area_may_take_a_baseline_window():
    data = _data()
    _finding(data, "ogtt_glucose_iauc")["baseline_window"] = {"start_min": 0, "end_min": 5}
    assert _spec("ogtt_glucose_iauc", data).baseline == StatisticWindow(0, 5)
    _finding(data, "ogtt_glucose_iauc")["baseline_window"] = {"start_min": 0, "end_min": 500}
    assert any("baseline_window" in p for p in _problems(data))


def test_these_are_single_arm_statistics():
    data = _data()
    _finding(data, "ogtt_glucose_auc")["protocols"] = ["fasting", "ogtt"]
    assert any("runs 1 protocol(s), got 2" in p for p in _problems(data))


def _temp(data: dict, fid: str, **update):
    f = _finding(data, fid)
    f.update(marker="temp", **update)
    return f


def test_a_temperature_difference_converts_by_the_factor_alone_and_a_level_by_factor_and_offset():
    data = _data()
    # +1.8 degF is +1.0 degC, whatever the offset between the scales
    _temp(data, "ogtt_glucose_second_hour_rise", value=1.8, unit="degF",
          uncertainty={"kind": "sem", "value": 0.18, "n": 12})
    change = _spec("ogtt_glucose_second_hour_rise", data)
    assert change.target == pytest.approx(1.0)
    assert change.sigma == pytest.approx(0.1 * 12 ** 0.5)

    # an incremental area is a difference: 1.8 degF x 119 min over 119 min is 1.0 degC
    _temp(data, "ogtt_glucose_iauc", value=1.8 * 119, unit="degF", uncertainty={"kind": "sem", "value": 1.8 * 119 / 10, "n": 12})
    iauc = _spec("ogtt_glucose_iauc", data)
    assert iauc.target == pytest.approx(1.0)
    assert iauc.sigma == pytest.approx(0.1 * 12 ** 0.5)

    # an area under a level is a mean level: 98.6 degF for 119 min is 37 degC
    _temp(data, "ogtt_glucose_auc", value=98.6 * 119, unit="degF", uncertainty={"kind": "sd", "value": 1.8 * 119, "n": 12})
    auc = _spec("ogtt_glucose_auc", data)
    assert auc.target == pytest.approx(37.0)
    assert auc.sigma == pytest.approx(1.0)


def test_an_area_in_a_unit_the_loader_does_not_know_is_refused():
    data = _data()
    _finding(data, "ogtt_glucose_auc")["unit"] = "mg*min/dL"
    assert any("unit" in p for p in _problems(data))


def test_conflicts_compare_areas_only_over_the_same_samples_baseline_and_rule():
    f = read_finding(FINDING)
    iauc = next(s for s in f.specs if s.name.endswith("ogtt_glucose_iauc"))
    rival = replace(iauc, name="other", target=iauc.target + 50.0, sigma=0.1)
    tight = replace(iauc, sigma=0.1)
    assert [c["items"] for c in target_conflicts([tight, rival])] == [[tight.name, "other"]]
    for different in (
        replace(rival, below_baseline=BelowBaseline.SUBTRACTED),
        replace(rival, sample_times=(0, 60, 90, 119), window=iauc.window),
        replace(rival, baseline=StatisticWindow(0, 5)),
    ):
        assert target_conflicts([tight, different]) == []


# --- the loss ------------------------------------------------------------------------

MEAL = ((5.0, 60.0, 20.0, 20.0),)


def _model() -> ModularPhysiologyNetwork:
    torch.manual_seed(3)
    model = ModularPhysiologyNetwork(
        metabolic_hidden=12, appetite_hidden=8, stress_hidden=8,
        cardiovascular_hidden=12, thermoreg_hidden=8, respiratory_hidden=8,
        gut_hidden=8, hepatobiliary_hidden=8,
    )
    g = torch.Generator().manual_seed(4)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    return model


def _loss_spec(name, minutes, kind, **kw):
    arm = CohortArmSpec(label=name, duration_min=minutes, start_hour=8.0, meals=MEAL)
    times = kw.pop("times", None)
    window = StatisticWindow(times[0], times[-1] + 1) if times else kw.pop("window")
    return CohortStatisticSpec(
        name=name, source="t", description="", arms=(arm,), marker_id=kw.pop("marker", "glucose"),
        kind=kind, window=window, target=kw.pop("target", 5.0), sigma=kw.pop("sigma", 4.0),
        sample_times=tuple(times) if times else None, **kw,
    )


def _specs():
    return [
        _loss_spec("auc40", 40, StatisticKind.AREA_UNDER_CURVE, times=(0, 10, 25, 39), target=100.0),
        _loss_spec("iauc30", 30, StatisticKind.INCREMENTAL_AREA, times=(0, 10, 20, 29), target=4.0,
                   below_baseline=BelowBaseline.IGNORED),
        _loss_spec("net30", 30, StatisticKind.INCREMENTAL_AREA, times=(5, 15, 29), target=-1.0,
                   below_baseline=BelowBaseline.SUBTRACTED, baseline=StatisticWindow(0, 5)),
        _loss_spec("change40", 40, StatisticKind.CHANGE_FROM_BASELINE, window=StatisticWindow(20, 40),
                   baseline=StatisticWindow(0, 5), target=3.0),
        _loss_spec("plain40", 40, StatisticKind.MEAN_IN_WINDOW, window=StatisticWindow(10, 40), target=100.0),
    ]


def _ctx(model) -> SignalContext:
    params = list(model.parameters())
    return SignalContext(
        epoch=0, total_epochs=1, rng=np.random.default_rng(0), device=torch.device("cpu"),
        optimizer=torch.optim.SGD(params, lr=1e-3), params=params, grad_clip=10.0, aux_signal_clip=0.0,
    )


def test_the_loss_scores_the_number_worked_out_from_the_rolled_trajectory():
    model = _model()
    spec = _specs()[0]
    embs = [torch.zeros(32), torch.randn(32, generator=torch.Generator().manual_seed(1))]
    state = torch.tensor(NORM_CENTER, dtype=torch.float32)
    loss, pred, z = cohort_statistic_loss_one_spec(model, embs, spec, state, n_population=None)

    traj = _rollout_arm_batched(model, torch.stack(embs), spec.arms[0], state).detach().double()
    y = traj[:, list(spec.sample_times), SERIES_INDEX["glucose"]].numpy()
    by_hand = np.mean([np.trapezoid(row, spec.sample_times) / 39.0 for row in y])
    assert pred == pytest.approx(float(by_hand), rel=1e-5)
    assert z == pytest.approx((by_hand - 100.0) / 4.0, rel=1e-4)
    assert float(loss.detach()) >= 0.0


def test_the_batched_step_gives_each_statistic_the_nll_the_per_spec_path_gives():
    """One step rolls the 30- and 40-minute items together; the 30-minute rows are held after
    their last minute, and a statistic that read a minute past it would disagree here."""
    model = _model()
    specs = _specs()
    fed = 30
    rng = np.random.default_rng(11)
    want = [
        float(item_nll(model, s, n_draws=3, rng=rng, device="cpu", fed_day_minutes=fed).detach())
        for s in specs
    ]
    terms = [EvidenceTerm(s, 1.0, 3) for s in specs]
    report = accumulate_evidence_step(
        model, _ctx(model), terms, rng=np.random.default_rng(11), device="cpu", fed_day_minutes=fed,
    )
    assert report.nll == pytest.approx(want, rel=1e-4)
    assert {h for _, h in report.group_sizes} == {40}      # mixed horizons rolled as one group
    assert all(v > 0.0 for v in want)


def test_a_statistic_moves_when_its_sample_minute_moves_and_not_when_a_gap_minute_does():
    """The area's sensitivity to the model sits at the listed minutes only."""
    model = _model()
    spec = _specs()[0]
    embs = torch.randn(2, 32, generator=torch.Generator().manual_seed(2))
    state = torch.tensor(NORM_CENTER, dtype=torch.float32)
    traj = _rollout_arm_batched(model, embs, spec.arms[0], state).detach().double().requires_grad_(True)
    per_person_statistic(traj[:, :, SERIES_INDEX["glucose"]], spec).sum().backward()
    g = traj.grad[:, :, SERIES_INDEX["glucose"]]
    off = [t for t in range(40) if t not in spec.sample_times]
    assert float(g[:, off].abs().sum()) == 0.0
    assert float(g[:, list(spec.sample_times)].abs().sum()) > 0.0


# --- the gate ------------------------------------------------------------------------

def test_a_file_with_the_three_statistics_passes_the_gate_on_the_untrained_model(tmp_path):
    data = _data()
    keep = ("ogtt_glucose_second_hour_rise", "ogtt_glucose_auc", "ogtt_glucose_iauc")
    data["findings"] = [f for f in data["findings"] if f["id"] in keep]
    data["protocols"] = [p for p in data["protocols"] if p["name"] == "ogtt"]
    (tmp_path / "findings").mkdir()
    (tmp_path / "refusals").mkdir()
    path = tmp_path / "findings" / "synthetic-test-2099.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    result = admit(path, root=tmp_path)
    assert result.ok, format_admission(result)
    assert result.items == 3
    assert [c.name for c in result.checks if c.name == "gradient"] == ["gradient"]


# --- the diagnostics that read a spec by kind ---------------------------------------

def test_the_median_persons_residual_and_the_cold_model_read_the_new_kinds():
    from pulse.diagnostics.cold_calibration import cold_calibration
    from pulse.diagnostics.identifiability import _cohort_residual
    from pulse.types import EMBEDDING_DIM

    model = _model()
    state = torch.tensor(NORM_CENTER, dtype=torch.float32)
    zero = torch.zeros(1, EMBEDDING_DIM)
    for spec in _specs()[:4]:
        raw, shaped = _cohort_residual(model, spec, zero, state)
        # the same number the loss scores for one person
        loss, pred, z = cohort_statistic_loss_one_spec(model, [zero[0]], spec, state)
        assert float(raw) == pytest.approx(z, rel=1e-4) and float(shaped) == pytest.approx(z, rel=1e-4)

    rows = cold_calibration(specs=_specs()[:4]).rows
    assert [r.name for r in rows] == ["auc40", "iauc30", "net30", "change40"]
    assert all(np.isfinite(r.cold_predicted) for r in rows)
