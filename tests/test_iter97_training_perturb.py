"""Iter 97 (training): teach-to-test protocols are perturbed per step (review 4.10)."""

from __future__ import annotations

import os

import numpy as np

import pulse

REPO = os.path.dirname(os.path.dirname(os.path.abspath(pulse.__file__)))
assert pulse.__file__.startswith(REPO), pulse.__file__

from pulse.dose_response import DoseResponseProtocol  # noqa: E402
from pulse.training import DoseResponseSignal, PostprandialRecoverySignal, WeightSchedule  # noqa: E402
from pulse.training.dose_response_signal import perturb_dose_response_protocol  # noqa: E402
from pulse.training.postprandial_recovery_signal import (  # noqa: E402
    _BASELINE_END, _MEAL, _RECOVERY_START, _START_HOUR,
)


def test_dose_response_perturbation_stays_inside_the_stated_ranges() -> None:
    base = DoseResponseProtocol()
    rng = np.random.default_rng(0)
    for _ in range(100):
        p = perturb_dose_response_protocol(base, rng)
        for d, d0 in zip(p.carb_doses_g, base.carb_doses_g):
            assert 0.8 * d0 - 1e-9 <= d <= 1.2 * d0 + 1e-9
        assert p.carb_doses_g == tuple(sorted(p.carb_doses_g))     # ladder still ordered
        assert abs(p.meal_offset_min - base.meal_offset_min) <= 30
        assert p.meal_offset_min >= p.pre_window
        assert p.meal_offset_min + p.post_window <= p.duration_min
        assert abs(((p.start_hour - base.start_hour + 12) % 24) - 12) <= 1.0 + 1e-9


def test_dose_response_keeps_the_fixed_protocol_as_one_sample() -> None:
    sig = DoseResponseSignal(weight=WeightSchedule(0.4), perturb_protocols=True, perturb_fixed_prob=0.25)
    rng = np.random.default_rng(1)
    n_fixed = sum(1 for _ in range(400) if sig.protocol_for_step(rng) is sig.protocol)
    assert 60 < n_fixed < 140
    off = DoseResponseSignal(weight=WeightSchedule(0.4), perturb_protocols=False)
    assert all(off.protocol_for_step(rng) is off.protocol for _ in range(10))


def test_postprandial_recovery_perturbation_keeps_baseline_before_the_meal() -> None:
    sig = PostprandialRecoverySignal(weight=WeightSchedule(0.1), perturb_protocols=True, perturb_fixed_prob=0.0)
    rng = np.random.default_rng(2)
    for _ in range(100):
        meal, start_hour, perturbed = sig.protocol_for_step(rng)
        assert perturbed
        assert 0.8 * _MEAL.carbs - 1e-9 <= meal.carbs <= 1.2 * _MEAL.carbs + 1e-9
        assert abs(meal.time - _MEAL.time) <= 30.0 and meal.time >= 10.0
        assert meal.time + 240 <= _RECOVERY_START
        assert abs(((start_hour - _START_HOUR + 12) % 24) - 12) <= 1.0 + 1e-9
    fixed = PostprandialRecoverySignal(weight=WeightSchedule(0.1), perturb_protocols=True, perturb_fixed_prob=1.0)
    assert fixed.protocol_for_step(rng) == (_MEAL, _START_HOUR, False)
    assert _BASELINE_END < _MEAL.time
