"""Iter 97 (review 5.8): the weak-check verifier on partial-day episodes.

- an empty time mask SKIPS the check instead of scoring against the whole
  window mean (legacy_static never sees an evening, cgm_real never an
  afternoon; a flat trajectory used to collect 0.4-0.5 per check for free);
- the sleep dip is scored on 12-h overnight episodes with an evening reference;
- the meal fences are no longer 15x looser than the textbook check;
- a float32-flat series cannot NaN the coupling category.
"""

from __future__ import annotations

import unittest

import numpy as np

from pulse.knowledge.weak_check_params import MEAL, SLEEP
from pulse.types import MARKER_INDEX, NORM_CENTER, STATE_DIM
from pulse.verifier import _safe_corr, evaluate_weak_checks


def _flat(duration: int) -> np.ndarray:
    return np.tile(np.array(NORM_CENTER, dtype=np.float32), (duration, 1))


def _keys(rep: dict) -> list[str]:
    return [c["key"] for c in rep["checks"]]


class TestVerifierMasks(unittest.TestCase):
    def test_empty_mask_skips_the_check(self) -> None:
        # legacy_static covers 03:34 -> 15:34: morning cortisol present, NO evening.
        rep = evaluate_weak_checks(_flat(720), meals=[], start_hour=3.57)
        self.assertNotIn("circadian_cortisol_morning_peak", _keys(rep))
        self.assertIn("circadian_temp_afternoon_higher", _keys(rep))
        # cgm_real covers 19:00 -> 07:00: no afternoon temp window.
        rep = evaluate_weak_checks(_flat(720), meals=[], start_hour=19.0)
        self.assertNotIn("circadian_temp_afternoon_higher", _keys(rep))
        self.assertIn("circadian_cortisol_morning_peak", _keys(rep))
        # a full day has both
        rep = evaluate_weak_checks(_flat(1440), meals=[], start_hour=6.0)
        self.assertIn("circadian_cortisol_morning_peak", _keys(rep))
        self.assertIn("circadian_temp_afternoon_higher", _keys(rep))

    def test_sleep_dip_scored_on_a_12h_night_with_evening_reference(self) -> None:
        traj = _flat(720)
        hr = MARKER_INDEX["hr"]
        sbp = MARKER_INDEX["sbp"]
        # 19:00 -> 07:00; night 00-05 is minutes 300..600
        traj[:, hr] = 70.0
        traj[300:600, hr] = 58.0
        traj[:, sbp] = 120.0
        traj[300:600, sbp] = 110.0
        rep = evaluate_weak_checks(traj, meals=[], start_hour=19.0)
        by = {c["key"]: c for c in rep["checks"]}
        self.assertIn("sleep_hr_dip", by)
        self.assertEqual(by["sleep_hr_dip"]["details"]["reference"], "evening")
        self.assertTrue(by["sleep_hr_dip"]["passed"])
        self.assertTrue(by["sleep_sbp_dip"]["passed"])
        # and a flat night FAILS the dip rather than being unscored
        rep_flat = evaluate_weak_checks(_flat(720), meals=[], start_hour=19.0)
        by_flat = {c["key"]: c for c in rep_flat["checks"]}
        self.assertIn("sleep_hr_dip", by_flat)
        self.assertFalse(by_flat["sleep_hr_dip"]["passed"])
        # a daytime-only 12 h (07:00 -> 19:00) has no night: nothing to score
        rep_day = evaluate_weak_checks(_flat(720), meals=[], start_hour=7.0)
        self.assertNotIn("sleep_hr_dip", _keys(rep_day))
        # a full day still prefers the daytime reference
        rep_full = evaluate_weak_checks(_flat(1440), meals=[], start_hour=6.0)
        by_full = {c["key"]: c for c in rep_full["checks"]}
        self.assertEqual(by_full["sleep_hr_dip"]["details"]["reference"], "day")
        self.assertEqual(SLEEP.min_trajectory_len, 1440)  # the training surrogate's gate, untouched

    def test_meal_fences_are_not_vacuous(self) -> None:
        # 75 g: textbook OGTT wants > 20 mg/dL; the fence must be in that league.
        self.assertGreaterEqual(MEAL.glucose_target(75.0), 15.0)
        self.assertGreaterEqual(MEAL.insulin_target(75.0), 5.0)
        traj = _flat(400)
        meals = [(60.0, 75.0, 5.0, 10.0)]
        rep = evaluate_weak_checks(traj, meals=meals, start_hour=7.0)
        meal_scores = [c["score"] for c in rep["checks"] if c["category"] == "meal"]
        self.assertTrue(meal_scores)
        self.assertLess(max(meal_scores), 0.3)  # flat = no meal response = clearly failing

    def test_float32_flat_series_does_not_nan_coupling(self) -> None:
        a = np.full(2880, 120.03, dtype=np.float32)
        b = np.full(2880, 70.01, dtype=np.float32)
        self.assertTrue(np.isfinite(_safe_corr(a, b)))
        rep = evaluate_weak_checks(_flat(2880), meals=[], start_hour=6.0)
        self.assertTrue(np.isfinite(rep["overall_score"]))
        self.assertTrue(all(np.isfinite(v) for v in rep["category_scores"].values()))


if __name__ == "__main__":
    unittest.main()
