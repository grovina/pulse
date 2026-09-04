"""Iter 97 (review 5.9): textbook checks report soft margins next to the binary rate.

Thresholds are NOT moved. iter 96's textbook 0.95 -> 0.8625 was two hairline
misses (cortisol ratio 1.291 vs 1.3; bhb -0.006 vs 0) plus one real regression
(exercise HR +12.7 vs +15); the report should be able to tell those apart.
"""

from __future__ import annotations

import unittest

from pulse.knowledge.textbook_scenarios.base import ScenarioCheck, ScenarioResult
from pulse.knowledge.textbook_scenarios.neural_eval import _scenario_to_dict


class TestSoftMargins(unittest.TestCase):
    def test_direction_is_inferred_from_the_verdict(self) -> None:
        above_pass = ScenarioCheck("a", "", True, 1.4, 1.3)
        above_miss = ScenarioCheck("b", "", False, 1.291, 1.3)
        below_pass = ScenarioCheck("c", "", True, 3.9, 4.5)     # lactate < 4.5
        below_miss = ScenarioCheck("d", "", False, 4.8, 4.5)
        self.assertEqual(above_pass.direction, "above")
        self.assertEqual(above_miss.direction, "above")
        self.assertEqual(below_pass.direction, "below")
        self.assertEqual(below_miss.direction, "below")
        self.assertGreater(above_pass.signed_margin, 0)
        self.assertLess(above_miss.signed_margin, 0)
        self.assertGreater(below_pass.signed_margin, 0)
        self.assertLess(below_miss.signed_margin, 0)

    def test_hairline_vs_real_miss(self) -> None:
        hairline = ScenarioCheck("cortisol_rises_after_waking", "", False, 1.291, 1.3)
        bhb = ScenarioCheck("bhb", "", False, -0.006, 0.0)
        real = ScenarioCheck("hr_rises_during_exercise", "", False, 12.7, 15.0)
        self.assertTrue(hairline.is_hairline())
        self.assertTrue(bhb.is_hairline())
        self.assertFalse(real.is_hairline())
        self.assertGreater(hairline.soft_score(), 0.4)   # a 0.7% miss is ~0.5
        self.assertLess(real.soft_score(), 0.3)          # a 15% miss is clearly below
        self.assertGreater(ScenarioCheck("x", "", True, 57.9, 20.0).soft_score(), 0.99)

    def test_report_carries_margins_and_keeps_pass_rate(self) -> None:
        checks = [
            ScenarioCheck("cortisol_rises_after_waking", "", False, 1.291, 1.3),
            ScenarioCheck("cortisol_morning_peak", "", True, 18.0, 15.0),
        ]
        r = ScenarioResult("car", "src", "desc", checks, 0.5)
        d = _scenario_to_dict(r)
        self.assertEqual(d["pass_rate"], 0.5)
        self.assertEqual(d["hairline"], ["cortisol_rises_after_waking"])
        self.assertIn("soft_score", d)
        self.assertGreater(d["soft_score"], d["pass_rate"])
        for c in d["checks"]:
            for key in ("direction", "signed_margin", "relative_margin", "soft_score", "hairline"):
                self.assertIn(key, c)
            self.assertIn("threshold", c)  # unchanged


if __name__ == "__main__":
    unittest.main()
