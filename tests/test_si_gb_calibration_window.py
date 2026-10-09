"""A meal reaches insulin sensitivity and a fast reaches fasting glucose.

Both gradients stay mostly the fasting-glucose direction. The separable
signal is the Si component of the meal, not a low cosine between the two.

Companion to the checkpoint recovery probe. The rate laws stay at a fresh
init; the two person heads are given orthogonal embedding reads so two people
can differ in one decode. See ``scripts/si_gb_calibration_window.py``.
"""
from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "si_gb_calibration_window.py"
_spec = importlib.util.spec_from_file_location("si_gb_calibration_window", _SCRIPT)
assert _spec is not None and _spec.loader is not None
_window = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_window)


class TestSiGbCalibrationWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = _window.measure()

    def test_people_differ_in_one_decode(self) -> None:
        report = self.report
        self.assertLess(report["si_pair_decode_leak"], 1e-4)
        self.assertLess(report["gb_pair_decode_leak"], 1e-4)
        self.assertGreater(report["si_ratio"], 3.0)
        self.assertGreater(report["gb_span"], 40.0)

    def test_si_moves_the_meal_and_gb_moves_the_fast(self) -> None:
        report = self.report
        self.assertLess(report["si_pair_fasting_gap"], 1.0)
        self.assertGreater(report["si_pair_increment_gap"], 15.0)
        self.assertGreater(report["gb_pair_fasting_gap"], 40.0)
        _window.check(report)

    def test_a_meal_carries_si_and_both_windows_are_mostly_gb(self) -> None:
        report = self.report
        self.assertGreater(report["cold_si_head_grad_meal"], 1.0)
        self.assertGreater(report["cold_gb_head_grad_fast"], 1.0)
        self.assertLess(report["cold_grad_fast_norm"], 1e-6)
        self.assertLess(report["grad_meal_si"], -5.0)
        self.assertGreater(report["grad_fast_gb"], 20.0)
        self.assertGreater(abs(report["grad_meal_si"]), 10.0 * abs(report["grad_fast_si"]))
        self.assertGreater(abs(report["grad_meal_gb"]), 2.0 * abs(report["grad_meal_si"]))
        self.assertGreater(report["cosine"], 0.8)
        self.assertLess(report["grad_fast_off_axis"], 1e-4)
        self.assertLess(report["grad_meal_off_axis"], 1e-4)
        self.assertTrue(math.isfinite(report["cosine"]))
        for key in ("grad_fast_norm", "grad_meal_norm", "si_head_grad_meal", "gb_head_grad_fast"):
            self.assertGreater(report[key], 0.0, msg=key)
            self.assertTrue(math.isfinite(report[key]), msg=key)


if __name__ == "__main__":
    unittest.main()
