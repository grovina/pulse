"""Smoke tests for the cold-model-calibration diagnostic."""

from __future__ import annotations

import unittest

from pulse.diagnostics import cold_calibration
from pulse.diagnostics.cold_calibration import render
from pulse.knowledge import ALL_COHORT_STATISTICS


class TestColdCalibration(unittest.TestCase):
    def test_one_row_per_spec(self) -> None:
        report = cold_calibration()
        self.assertEqual(report.n_specs, len(ALL_COHORT_STATISTICS))
        self.assertEqual(len(report.rows), len(ALL_COHORT_STATISTICS))

    def test_finite_predictions(self) -> None:
        for r in cold_calibration().rows:
            self.assertTrue(r.cold_predicted == r.cold_predicted, f"NaN for {r.name}")
            self.assertNotEqual(r.sigma, 0.0)

    def test_misleading_threshold_flags_outliers(self) -> None:
        report = cold_calibration(misleading_z=0.0)
        n_flag = sum(1 for r in report.rows if r.misleading)
        self.assertGreater(n_flag, 0, "with threshold=0 every nonzero |z| should flag")

    def test_specs_subset_runs_independently(self) -> None:
        subset = list(ALL_COHORT_STATISTICS[:3])
        report = cold_calibration(specs=subset)
        self.assertEqual(report.n_specs, 3)
        self.assertEqual({r.name for r in report.rows}, {s.name for s in subset})

    def test_render_is_nonempty_table(self) -> None:
        text = render(cold_calibration())
        self.assertIn("cold-calibration:", text)
        self.assertGreater(len(text.splitlines()), len(ALL_COHORT_STATISTICS))


if __name__ == "__main__":
    unittest.main()
