"""Uncertainty kinds become the sigma and n_arm the paper supports."""

from __future__ import annotations

import math
from statistics import NormalDist

import pytest

from pulse.knowledge.cohort_types import (
    TWO_ARM_KINDS, CohortArmSpec, CohortStatisticSpec, StatisticKind, StatisticWindow,
)
from pulse.knowledge.corpus_uncertainty import (
    ALLOWED_KINDS, CORRELATION_SIGMA_FLOOR, N_CAP, Uncertainty, UncertaintyError, curve_point_sigma, to_sigma,
)
from pulse.training.evidence_step import _sigma

K = StatisticKind


def _sem_used(kind, target, u, draws=8):
    """``(sem the training likelihood applies, sigma, n_arm)`` for this uncertainty."""
    sigma, n_arm = to_sigma(kind, target, u)
    n_arms = 2 if kind in TWO_ARM_KINDS else 1
    arms = tuple(
        CohortArmSpec(label=f"a{i}", duration_min=10, start_hour=8.0, meals=()) for i in range(n_arms)
    )
    spec = CohortStatisticSpec(
        name="x", source="x", description="", paper="x-2000", arms=arms, marker_id="glucose",
        kind=kind, window=StatisticWindow(2, 10), target=target, sigma=sigma, n_arm=n_arm,
        baseline=StatisticWindow(0, 1) if kind is K.RATIO_INCREMENTS else None,
        partner="insulin_resistance" if kind is K.CORRELATION else None,
    )
    return _sigma(spec, draws), sigma, n_arm


def test_a_between_person_sd_is_divided_by_the_papers_n():
    se, sigma, n_arm = _sem_used(K.MEAN_IN_WINDOW, 100.0, Uncertainty("sd", 10, value=25.0))
    assert (sigma, n_arm) == (25.0, 10)
    assert se == pytest.approx(25.0 / math.sqrt(10))


def test_a_standard_error_is_not_divided_again_by_the_draws():
    # The trap: sigma = SEM with n_arm=None would be divided by sqrt(draws). Here the
    # likelihood's own standard error is the paper's, whatever the draw count.
    for draws in (2, 8, 64):
        se, _, n_arm = _sem_used(K.MEAN_IN_WINDOW, 100.0, Uncertainty("sem", 9, value=3.0), draws)
        assert n_arm is not None
        assert se == pytest.approx(3.0)


def test_repeated_tests_of_the_same_people_do_not_tighten_the_mean():
    """Twelve people tested three times, SEM 0.06 over the 36 tests: the spread is
    0.06 * 6, and the mean is held to what twelve people support."""
    once = Uncertainty("sem", 12, value=0.06)
    repeated = Uncertainty("sem", 12, value=0.06, observations=36)
    se_once, _, _ = _sem_used(K.MEAN_IN_WINDOW, 5.2, once)
    se, sigma, n_arm = _sem_used(K.MEAN_IN_WINDOW, 5.2, repeated)
    assert sigma == pytest.approx(0.36) and n_arm == 12
    assert se == pytest.approx(0.36 / math.sqrt(12))
    assert se == pytest.approx(se_once * math.sqrt(3))
    assert curve_point_sigma(repeated) == pytest.approx(0.06 * math.sqrt(3))
    # A confidence interval over the 36 tests uses 35 degrees of freedom, then the same rule.
    ci = Uncertainty("ci", 12, low=0.0, high=0.24, level=95, observations=36)
    _, sigma_ci, n_ci = _sem_used(K.MEAN_IN_WINDOW, 5.2, ci)
    assert n_ci == 12 and sigma_ci == pytest.approx(0.24 / (2 * 2.030) * 6, rel=2e-3)


@pytest.mark.parametrize("u, fragment", [
    (Uncertainty("sem", 12, value=0.06, observations=12), "must exceed n"),
    (Uncertainty("sem", 12, value=0.06, observations=6), "must exceed n"),
    (Uncertainty("sd", 12, value=0.3, observations=36), "not a sd"),
    (Uncertainty("range", 12, low=0.0, high=1.0, observations=36), "not a range"),
])
def test_observations_is_only_for_more_measurements_than_people(u, fragment):
    assert any(fragment in p for p in u.problems())


def test_a_large_study_is_not_trusted_past_the_cap():
    se, sigma, n_arm = _sem_used(K.MEAN_IN_WINDOW, 100.0, Uncertainty("sem", 100, value=1.0))
    assert n_arm == N_CAP
    assert sigma == pytest.approx(10.0)            # sd = sem * sqrt(n)
    assert se == pytest.approx(10.0 / math.sqrt(N_CAP))  # 2.0, not the paper's 1.0
    se_sd, _, _ = _sem_used(K.MEAN_IN_WINDOW, 100.0, Uncertainty("sd", 400, value=20.0))
    assert se_sd == pytest.approx(20.0 / math.sqrt(N_CAP))


def test_a_confidence_interval_gives_a_standard_error_through_the_t_quantile():
    # n = 12, df = 11: t = 2.201
    u = Uncertainty("ci", 12, low=0.0, high=1.2, level=95)
    se, sigma, n_arm = _sem_used(K.MEAN_IN_WINDOW, 1.0, u)
    assert se == pytest.approx(1.2 / (2 * 2.201), rel=1e-3)
    assert n_arm == 12
    assert sigma == pytest.approx(1.2 / (2 * 2.201) * math.sqrt(12), rel=1e-3)


def test_t_quantiles_beyond_the_table_follow_the_normal_expansion():
    u = Uncertainty("ci", 101, low=0.0, high=2.0, level=95)
    sigma, _ = to_sigma(K.MEAN_IN_WINDOW, 1.0, u)
    t_100 = 1.984  # df = 100
    assert sigma / math.sqrt(101) == pytest.approx(2.0 / (2 * t_100), rel=2e-3)


def test_ci_levels_change_the_quantile():
    s95, _ = to_sigma(K.MEAN_IN_WINDOW, 1.0, Uncertainty("ci", 20, low=0.0, high=1.0, level=95))
    s99, _ = to_sigma(K.MEAN_IN_WINDOW, 1.0, Uncertainty("ci", 20, low=0.0, high=1.0, level=99))
    assert s99 < s95


def test_a_range_gives_an_sd_through_wans_formula():
    n = 12
    xi = 2 * NormalDist().inv_cdf((n - 0.375) / (n + 0.25))
    sigma, n_arm = to_sigma(K.PEAK_VALUE, 20.0, Uncertainty("range", n, low=10.0, high=30.0))
    assert sigma == pytest.approx(20.0 / xi)
    assert n_arm == n


def test_spread_from_n_alone_is_the_sampling_error_of_an_sd():
    sigma, n_arm = to_sigma(K.SPREAD, 10.0, Uncertainty("n_only", 9))
    assert sigma == pytest.approx(10.0 / math.sqrt(2 * 8))
    assert n_arm == 1
    capped, _ = to_sigma(K.SPREAD, 10.0, Uncertainty("n_only", 1000))
    assert capped == pytest.approx(10.0 / math.sqrt(2 * (N_CAP - 1)))


def test_spread_from_its_own_standard_error():
    sigma, _ = to_sigma(K.SPREAD, 10.0, Uncertainty("sem", 10, value=1.5))
    assert sigma == pytest.approx(1.5)


def test_correlation_from_n_alone_and_its_floor():
    sigma, n_arm = to_sigma(K.CORRELATION, 0.4, Uncertainty("n_only", 12))
    assert sigma == pytest.approx((1 - 0.16) / math.sqrt(10))
    assert n_arm == 1
    tight, _ = to_sigma(K.CORRELATION, 0.99, Uncertainty("n_only", 25))
    assert tight == CORRELATION_SIGMA_FLOOR
    with pytest.raises(UncertaintyError):
        to_sigma(K.CORRELATION, 1.0, Uncertainty("n_only", 25))
    with pytest.raises(UncertaintyError):
        to_sigma(K.CORRELATION, 0.4, Uncertainty("n_only", 2))


def test_ratio_of_increments_takes_the_reported_standard_error_as_is():
    se, sigma, n_arm = _sem_used(K.RATIO_INCREMENTS, 1.6, Uncertainty("sem", 10, value=0.2))
    assert (sigma, n_arm) == (0.2, 1)
    assert se == 0.2
    wide, _ = to_sigma(K.RATIO_INCREMENTS, 1.6, Uncertainty("sem", 100, value=0.2))
    assert wide == pytest.approx(0.2 * math.sqrt(100 / N_CAP))


def test_a_curve_point_sigma_is_its_standard_error():
    assert curve_point_sigma(Uncertainty("sem", 12, value=6.0)) == 6.0
    assert curve_point_sigma(Uncertainty("sd", 16, value=8.0)) == pytest.approx(2.0)


def test_every_scalar_kind_gets_an_n_arm():
    for kind in K:
        if kind is K.CURVE:
            continue
        allowed = sorted(ALLOWED_KINDS[kind])[0]
        u = {
            "sd": Uncertainty("sd", 10, value=2.0), "sem": Uncertainty("sem", 10, value=2.0),
            "ci": Uncertainty("ci", 10, low=0.0, high=2.0, level=95),
            "range": Uncertainty("range", 10, low=0.0, high=2.0), "n_only": Uncertainty("n_only", 10),
        }[allowed]
        _, n_arm = to_sigma(kind, 0.5, u)
        assert n_arm is not None and n_arm >= 1, kind


@pytest.mark.parametrize("kind, u", [
    (K.RATIO_MEANS, Uncertainty("sd", 10, value=1.0)),
    (K.RATIO_INCREMENTS, Uncertainty("range", 10, low=0.0, high=1.0)),
    (K.SPREAD, Uncertainty("sd", 10, value=1.0)),
    (K.CORRELATION, Uncertainty("range", 10, low=0.0, high=1.0)),
    (K.MEAN_IN_WINDOW, Uncertainty("n_only", 10)),
    (K.MEAN_IN_WINDOW, Uncertainty("sd", 1, value=1.0)),
    (K.MEAN_IN_WINDOW, Uncertainty("sd", 10, value=-1.0)),
    (K.MEAN_IN_WINDOW, Uncertainty("ci", 10, low=0.0, high=1.0, level=80)),
    (K.MEAN_IN_WINDOW, Uncertainty("ci", 10, low=1.0, high=0.0, level=95)),
    (K.MEAN_IN_WINDOW, Uncertainty("variance", 10, value=1.0)),
])
def test_an_uncertainty_the_statistic_cannot_use_is_refused(kind, u):
    with pytest.raises(UncertaintyError):
        to_sigma(kind, 0.5, u)


def test_a_curve_is_scored_by_point_not_by_to_sigma():
    with pytest.raises(UncertaintyError):
        to_sigma(K.CURVE, 1.0, Uncertainty("sem", 10, value=1.0))
