"""Iter 97 (training): teach-to-test protocols are perturbed per step (review 4.10)."""

from __future__ import annotations

import os

import numpy as np

import pulse

REPO = os.path.dirname(os.path.dirname(os.path.abspath(pulse.__file__)))
assert pulse.__file__.startswith(REPO), pulse.__file__

from pulse.dose_response import DoseResponseProtocol  # noqa: E402
from pulse.training import RolloutEvidenceSignal, WeightSchedule, perturb_dose_response_protocol  # noqa: E402


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
    sig = RolloutEvidenceSignal(weight=WeightSchedule(0.4), perturb_protocols=True, perturb_fixed_prob=0.25)
    rng = np.random.default_rng(1)
    n_fixed = sum(1 for _ in range(400) if sig.protocol_for_step(rng) is sig.protocol)
    assert 60 < n_fixed < 140
    off = RolloutEvidenceSignal(weight=WeightSchedule(0.4), perturb_protocols=False)
    assert all(off.protocol_for_step(rng) is off.protocol for _ in range(10))
