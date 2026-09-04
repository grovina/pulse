"""Iter 97 (review 5.6): in-process teacher episodes score nothing they were fitted to.

Through iter 96 the `teacher` episodes' eval points included their own
calibration check-ins (6/13 glucose, 6/12 hr, 1/1 sbp samples) -- the
`sbp persistence 0.0 / skill 0.0` line in the iter-96 report was that leak.
"""

from __future__ import annotations

import unittest

from pulse.knowledge.benchmark_extras import (
    all_cohort_benchmark_episodes,
    last_check_in_time,
    leaked_eval_points,
)


class TestNoLeakage(unittest.TestCase):
    def test_every_eval_point_is_after_the_last_check_in(self) -> None:
        eps = all_cohort_benchmark_episodes()
        self.assertGreaterEqual(len(eps), 10)
        for ep in eps:
            cutoff = last_check_in_time(ep)
            self.assertGreater(cutoff, 0, ep.user_id)
            self.assertEqual(leaked_eval_points(ep), [], ep.user_id)
            self.assertGreater(min(p.time for p in ep.eval_measurements), cutoff, ep.user_id)

    def test_scored_windows_still_contain_dynamics(self) -> None:
        by_id = {ep.user_id: ep for ep in all_cohort_benchmark_episodes()}
        meal = by_id["benchmark-cohort-meal-postprandial"]
        cutoff = last_check_in_time(meal)
        # the second meal (t=300) lands inside the scored window
        self.assertTrue(any(cutoff < m.time < max(p.time for p in meal.eval_measurements)
                            for m in meal.meals))
        sleep = by_id["benchmark-cohort-sleep-48h-adequate"]
        self.assertEqual(last_check_in_time(sleep), 1440)
        self.assertTrue(all(p.time > 1440 for p in sleep.eval_measurements))


if __name__ == "__main__":
    unittest.main()
