"""Per-statistic-kind unit tests for cohort_loss.

We exercise the differentiable extractors on hand-crafted trajectories so the
expected value for each kind is unambiguous, then check the spec validation
(arm-count, sigma>0).
"""

from __future__ import annotations

import unittest

import torch

from pulse.cohort_loss import _arm_statistic_batched
from pulse.knowledge.cohort_types import (
    CohortArmSpec,
    CohortStatisticSpec,
    StatisticKind,
    StatisticWindow,
)
from pulse.types import MARKER_INDEX


def _const_traj(n_steps: int, marker_value: float, marker_id: str = "glucose") -> torch.Tensor:
    """[1, T, STATE_DIM] trajectory with one marker held at marker_value."""
    state_dim = len(MARKER_INDEX)
    traj = torch.zeros(1, n_steps, state_dim)
    traj[0, :, MARKER_INDEX[marker_id]] = marker_value
    return traj


def _ramp_traj(n_steps: int, peak_at: int, peak_value: float, marker_id: str = "glucose") -> torch.Tensor:
    """[1, T, STATE_DIM] linear ramp up to peak_at, then linear ramp down."""
    state_dim = len(MARKER_INDEX)
    traj = torch.zeros(1, n_steps, state_dim)
    mi = MARKER_INDEX[marker_id]
    for t in range(n_steps):
        if t <= peak_at:
            traj[0, t, mi] = peak_value * (t / max(peak_at, 1))
        else:
            traj[0, t, mi] = peak_value * max(0.0, 1.0 - (t - peak_at) / max(n_steps - peak_at - 1, 1))
    return traj


class TestStatisticExtractors(unittest.TestCase):
    def test_mean_in_window(self) -> None:
        traj = _const_traj(100, 5.0)
        v = _arm_statistic_batched(
            traj, MARKER_INDEX["glucose"], StatisticWindow(10, 50),
            StatisticKind.MEAN_IN_WINDOW, 0.05,
        )
        self.assertEqual(v.shape, (1,))
        self.assertAlmostEqual(float(v[0]), 5.0, places=5)

    def test_peak_value(self) -> None:
        traj = _ramp_traj(100, peak_at=40, peak_value=12.0)
        v = _arm_statistic_batched(
            traj, MARKER_INDEX["glucose"], StatisticWindow(0, 100),
            StatisticKind.PEAK_VALUE, 0.05,
        )
        self.assertAlmostEqual(float(v[0]), 12.0, places=4)

    def test_time_to_peak_softargmax(self) -> None:
        # Sharp peak at index 30 (relative to window) — soft argmax should be close.
        traj = _ramp_traj(120, peak_at=50, peak_value=80.0)
        v = _arm_statistic_batched(
            traj, MARKER_INDEX["glucose"], StatisticWindow(20, 100),
            StatisticKind.TIME_TO_PEAK, 0.5,
        )
        # Window starts at 20, peak originally at 50 → relative index 30. Tolerate softness.
        self.assertGreater(float(v[0]), 25.0)
        self.assertLess(float(v[0]), 35.0)


class TestCounterRegulatorySpecs(unittest.TestCase):
    """The Phase-3 specs encode counter-regulatory physiology that the
    benchmark currently fails. Cheap structural checks: each spec is well-
    formed, references a real marker, has a sensible window, and lives in
    ``ALL_COHORT_STATISTICS`` so the training signal will pick it up.
    """

    def test_counter_regulatory_specs_registered(self) -> None:
        from pulse.knowledge.cohort_statistics import ALL_COHORT_STATISTICS
        names = {s.name for s in ALL_COHORT_STATISTICS}
        for required in (
            "extended_fast_ffa_overnight",
            "extended_fast_insulin_basal",
            "ogtt_glucagon_suppression",
            "mixed_meal_glucagon_suppression",
            "meal_ffa_suppression",
        ):
            self.assertIn(required, names, f"missing cohort spec: {required}")

    def test_counter_regulatory_specs_have_valid_markers_and_windows(self) -> None:
        from pulse.knowledge.cohort_statistics import ALL_COHORT_STATISTICS
        new_names = {
            "extended_fast_ffa_overnight",
            "extended_fast_insulin_basal",
            "ogtt_glucagon_suppression",
            "mixed_meal_glucagon_suppression",
            "meal_ffa_suppression",
        }
        for spec in ALL_COHORT_STATISTICS:
            if spec.name not in new_names:
                continue
            self.assertIn(spec.marker_id, MARKER_INDEX, f"{spec.name}: unknown marker")
            self.assertGreater(spec.sigma, 0.0)
            for arm in spec.arms:
                self.assertGreater(spec.window.end_min, spec.window.start_min)
                self.assertLessEqual(spec.window.end_min, arm.duration_min)


class TestSpecValidation(unittest.TestCase):
    def test_delta_means_requires_two_arms(self) -> None:
        with self.assertRaises(ValueError):
            CohortStatisticSpec(
                name="x", source="x", description="x",
                arms=(CohortArmSpec(label="a", duration_min=10, start_hour=6.0, meals=()),),
                marker_id="glucose",
                kind=StatisticKind.DELTA_MEANS,
                window=StatisticWindow(0, 5),
                target=0.0, sigma=1.0,
            )

    def test_sigma_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            CohortStatisticSpec(
                name="x", source="x", description="x",
                arms=(CohortArmSpec(label="a", duration_min=10, start_hour=6.0, meals=()),),
                marker_id="glucose",
                kind=StatisticKind.MEAN_IN_WINDOW,
                window=StatisticWindow(0, 5),
                target=0.0, sigma=0.0,
            )


if __name__ == "__main__":
    unittest.main()
