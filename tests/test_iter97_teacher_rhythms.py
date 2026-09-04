"""Iter 97 (teacher): HPA rhythm, rest = 0, the standalone generators as views."""

import unittest

import numpy as np

from pulse.knowledge.full_body import (
    PatientParams, resolve_derived_params, simulate_full_body, generate_activity,
    generate_sleep_wake, _hpa_drive,
)
from pulse.knowledge.bergman_glucose_insulin import BergmanGlucoseInsulin, VIEW_MARKERS as BG_VIEW
from pulse.knowledge.cortisol_circadian import CortisolCircadian, VIEW_MARKERS as CC_VIEW
from pulse.knowledge.cardiovascular_dynamics import CardiovascularDynamics, VIEW_MARKERS as CV_VIEW
from pulse.types import MARKER_INDEX as MI, STATE_DIM


def _sleep(n_days, bed=23.0, wake=7.0, start_hour=6.0):
    n = n_days * 1440
    sw = np.ones(n, dtype=np.float32)
    for d in range(n_days + 1):
        s = int((bed - start_hour) * 60) + d * 1440
        e = int((wake + 24 - start_hour) * 60) + d * 1440
        sw[max(0, s):min(n, e)] = 0.0
    return sw


class TestHPARhythm(unittest.TestCase):
    def setUp(self):
        self.p = resolve_derived_params(PatientParams())
        n_days = 3
        day = [(8.0, 50, 12, 20), (13.0, 65, 22, 28), (19.0, 75, 28, 35)]
        meals = [(d * 1440 + (h - 6.0) * 60.0, c, f, pr) for d in range(n_days) for h, c, f, pr in day]
        self.sw = _sleep(n_days)
        n = n_days * 1440
        self.traj, _ = simulate_full_body(self.p, meals, self.sw, np.zeros(n, dtype=np.float32), n,
                                          noise_scale=0.0, rng=np.random.default_rng(0))
        self.day = slice(1440, 2880)   # day 2, 06:00 -> 06:00
        self.hours = (6.0 + np.arange(1440) / 60.0) % 24

    def test_cortisol_peak_and_nadir_timing(self):
        c = self.traj[self.day, MI["cortisol"]]
        self.assertTrue(7.0 <= self.hours[np.argmax(c)] <= 9.0, f"peak at {self.hours[np.argmax(c)]:.1f} h")
        self.assertTrue(0.0 <= self.hours[np.argmin(c)] <= 4.0, f"nadir at {self.hours[np.argmin(c)]:.1f} h")
        self.assertTrue(2.5 <= c.min() <= 5.5)
        self.assertTrue(15.0 <= c.max() <= 21.0)
        self.assertGreater(c.max() / c.min(), 3.5)

    def test_cortisol_to_acth_ratio_is_the_same_asleep_and_awake(self):
        c = self.traj[self.day, MI["cortisol"]]; a = self.traj[self.day, MI["acth"]]
        ratio = c / a
        asleep = ratio[self.sw[self.day] < 0.5].mean()
        awake = ratio[self.sw[self.day] > 0.5].mean()
        self.assertLess(abs(asleep / awake - 1.0), 0.15, f"asleep {asleep:.3f} vs awake {awake:.3f}")

    def test_drive_is_quiescent_in_the_evening_and_rises_before_dawn(self):
        p = self.p
        d = lambda h: _hpa_drive(h * 60, p.hpa_rise_start_h, p.hpa_peak_h, p.hpa_fall_tau_h)
        self.assertLess(d(23.0), 0.15)
        self.assertLess(d(1.5), d(4.0))
        self.assertLess(d(4.0), d(6.5))
        self.assertAlmostEqual(d(p.hpa_peak_h), 1.0, places=6)
        self.assertGreater(d(9.0), d(12.0))

    def test_low_cortisol_raises_hrv_two_sided(self):
        """HRV sees cortisol below its reference as well as above it."""
        hrv = self.traj[self.day, MI["hrv"]]; c = self.traj[self.day, MI["cortisol"]]
        night = (self.hours >= 1) & (self.hours <= 4)
        self.assertLess(c[night].mean(), self.p.Cort_b)
        self.assertGreater(hrv[night].mean(), self.p.HRV0 * self.p.sleep_hrv_gain)


class TestRestIsZero(unittest.TestCase):
    def test_generated_activity_is_zero_at_rest(self):
        rng = np.random.default_rng(3)
        act = generate_activity(3, 3 * 1440, 6.0, rng)
        self.assertEqual(float(act.min()), 0.0)
        self.assertLess(np.mean(act > 0), 0.05)   # bouts only


class TestGeneratorsAreViews(unittest.TestCase):
    def _check(self, contribution, view):
        eps = contribution.generate_episodes(1, np.random.default_rng(11))
        self.assertEqual(len(eps), 1)
        ep = eps[0]
        self.assertEqual(ep.trajectory.shape, (ep.duration_min, STATE_DIM))
        for m in view:
            self.assertFalse(np.isnan(ep.trajectory[:, MI[m]]).any(), m)
        others = [i for i in range(STATE_DIM) if i not in {MI[m] for m in view}]
        self.assertTrue(np.isnan(ep.trajectory[:, others]).all())
        self.assertIsNotNone(ep.sleep_wake)
        self.assertIsNotNone(ep.activity)
        return ep

    def test_bergman_is_a_metabolic_view(self):
        ep = self._check(BergmanGlucoseInsulin(), BG_VIEW)
        self.assertEqual(ep.source, "bergman_glucose_insulin")

    def test_cortisol_is_a_stress_appetite_view(self):
        ep = self._check(CortisolCircadian(), CC_VIEW)
        self.assertEqual(ep.source, "cortisol_circadian")

    def test_cardiovascular_is_a_vitals_view(self):
        ep = self._check(CardiovascularDynamics(), CV_VIEW)
        self.assertEqual(ep.source, "cardiovascular_dynamics")

    def test_views_are_bit_identical_to_the_coupled_teacher(self):
        """Same seed -> the view IS the coupled teacher's episode, masked. The old
        standalone generator taught a cortisol nadir of 16.5 against full_body's ~4."""
        from pulse.knowledge.full_body import FullBody
        for cls, view in ((CortisolCircadian, CC_VIEW), (BergmanGlucoseInsulin, BG_VIEW),
                          (CardiovascularDynamics, CV_VIEW)):
            ep = cls().generate_episodes(1, np.random.default_rng(5))[0]
            ref = FullBody(n_days=3).generate_episodes(1, np.random.default_rng(5))[0]
            for m in view:
                np.testing.assert_array_equal(ep.trajectory[:, MI[m]], ref.trajectory[:, MI[m]])
        c = ep.trajectory[1440:2880, MI["cortisol"]] if False else \
            CortisolCircadian().generate_episodes(1, np.random.default_rng(5))[0].trajectory[1440:2880, MI["cortisol"]]
        self.assertLess(c.min(), 7.0)


if __name__ == "__main__":
    unittest.main()
