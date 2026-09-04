"""Iter 97 (teacher): one carbon budget, two ledgers -- the structural properties.

Every test pins a STRUCTURE (a flux appears on both books, a fixed point is the
declared level, a floor is absolute), not a tuned number.
"""

import copy
import unittest

import numpy as np

import pulse.knowledge.full_body as fb
from pulse.knowledge.full_body import (
    PatientParams, glucose_fluxes, resolve_derived_params, randomize_params,
    simulate_full_body, MG_DL_PER_G, BODY_MASS_KG, VG_DL_PER_KG, _meal_absorption,
    _kernel_cutoff_min,
)
from pulse.types import MARKER_INDEX as MI


def _run(params, meals, n, sw=None, act=None, start_hour=6.0):
    sw = np.ones(n) if sw is None else sw
    act = np.zeros(n, dtype=np.float32) if act is None else act
    traj, absp = simulate_full_body(params, meals, sw, act, n, start_hour=start_hour,
                                    noise_scale=0.0, rng=np.random.default_rng(0))
    return traj, absp


def _std_day(n_days, start_hour=6.0):
    day = [(8.0, 50, 12, 20), (13.0, 65, 22, 28), (19.0, 75, 28, 35)]
    return [(d * 1440 + (h - start_hour) * 60.0, c, f, p)
            for d in range(n_days) for h, c, f, p in day if (h - start_hour) >= 0]


class TestOneFluxTwoLedgers(unittest.TestCase):
    def setUp(self):
        self.p = resolve_derived_params(PatientParams())

    def test_glycogenolysis_is_the_same_number_on_both_books(self):
        """The flux leaving the liver pool is the flux reaching blood."""
        p = self.p
        fl = glucose_fluxes(p, 90.0, 6.0, -0.001, 75.0, 8.0, 0.7, 60.0, 400.0, 1.6, 0.0, 0.0)
        credited_to_glucose_g = fl["glyco_flux"] * BODY_MASS_KG / 1000.0
        # With no appearance there is no direct synthesis; the pool loses exactly
        # glycogenolysis minus the diverted gluconeogenic carbon.
        self.assertAlmostEqual(fl["syn_L"], 0.0)
        self.assertAlmostEqual(-fl["dLGly"], credited_to_glucose_g - fl["gng_divert"] * BODY_MASS_KG / 1000.0, places=9)
        # And blood receives glycogenolysis + released GNG = the lagged state, in glucose space.
        self.assertAlmostEqual(fl["egp"], (fl["glyco_flux"] + fl["gng_rel_flux"]) / VG_DL_PER_KG, places=9)

    def test_synthesis_is_zero_at_zero_appearance_and_debited_otherwise(self):
        p = self.p
        base = glucose_fluxes(p, 150.0, 50.0, 0.02, 45.0, 12.0, 0.2, 90.0, 400.0, 0.4, 0.0, 0.0)
        fed = glucose_fluxes(p, 150.0, 50.0, 0.02, 45.0, 12.0, 0.2, 90.0, 400.0, 0.4, 4.0, 0.0)
        self.assertEqual(base["syn_L"], 0.0)
        self.assertGreater(fed["syn_L"], 0.0)
        # dG rises by less than Ra because synthesis is taken out of it.
        self.assertAlmostEqual(fed["dG"] - base["dG"], 4.0 - fed["syn_L"] - fed["syn_M"], places=9)
        # The liver ledger receives the identical grams.
        self.assertAlmostEqual(fed["dLGly"] - base["dLGly"], fed["syn_L"] / MG_DL_PER_G, places=9)

    def test_carbon_closes_over_a_eucaloric_day(self):
        """d(glucose + liver glycogen + muscle glycogen) == booked net flux, < 1 g/day."""
        p = self.p
        n_days = 3
        traj, absp = _run(p, _std_day(n_days), n_days * 1440)
        day = slice(1440, 2880)
        G, I, Gn = traj[day, MI["glucose"]], traj[day, MI["insulin"]], traj[day, MI["glucagon"]]
        Cort, FFA = traj[day, MI["cortisol"]], traj[day, MI["ffa"]]
        LG, MG, Hep = traj[day, MI["liver_glycogen"]], traj[day, MI["muscle_glycogen"]], traj[day, MI["hepatic_output"]]
        Ra = absp[day, 0]
        X = np.zeros_like(I); x = 0.0
        for k in range(len(I)):
            X[k] = x
            x += -p.p2 * x + p.Si * p.p2 * (I[k] - p.Ib)
        acc = {k: 0.0 for k in ("ra", "uptake_ii", "uptake_id", "uptake_ex", "gng_rel_flux", "gng_divert", "brk_M_g")}
        for k in range(len(G)):
            fl = glucose_fluxes(p, G[k], I[k], X[k], Gn[k], Cort[k], FFA[k], LG[k], MG[k], Hep[k], Ra[k], 0.0)
            for kk in acc:
                acc[kk] += fl[kk]
        kg = BODY_MASS_KG / 1000.0
        booked = ((acc["ra"] - acc["uptake_ii"] - acc["uptake_id"] - acc["uptake_ex"]) / MG_DL_PER_G
                  + (acc["gng_rel_flux"] + acc["gng_divert"]) * kg - acc["brk_M_g"])
        pools = (G[-1] - G[0]) / MG_DL_PER_G + (LG[-1] - LG[0]) + (MG[-1] - MG[0])
        self.assertLess(abs(pools - booked), 1.0, f"carbon residual {pools - booked:.3f} g/day")
        # And the day is not glycogen-negative by construction: the pool cycles around its typical.
        self.assertGreater(LG.mean(), 0.75 * p.LGly_b)

    def test_kernel_is_mass_conserving_and_untruncated(self):
        rng = np.random.default_rng(7)
        worst = 0.0
        for _ in range(100):
            p = randomize_params(rng)
            for rate in (p.meal_absorption_fast_rate, p.meal_absorption_slow_rate):
                t = np.arange(0, _kernel_cutoff_min(rate) + 1.0)
                mass = sum(_meal_absorption(float(tt), 0.0, 1.0, rate) for tt in t) / MG_DL_PER_G
                worst = max(worst, abs(1.0 - mass))
        self.assertLess(worst, 0.01, f"kernel loses {worst*100:.2f}% of ingested carbohydrate")


class TestDerivedFixedPoints(unittest.TestCase):
    def test_gb_is_the_fixed_point_and_hep_b_is_derived(self):
        for gb in (70.0, 95.0, 130.0):
            p = PatientParams(); p.Gb = gb; p = resolve_derived_params(p)
            self.assertAlmostEqual(p.Hep_b, p.uptake_ii * gb * VG_DL_PER_KG)
            self.assertAlmostEqual(p.Sg, p.uptake_ii * (1.0 + p.hep_autoreg_m))
            fl = glucose_fluxes(p, gb, p.Ib, 0.0, p.Gnb, p.Cort_b, p.FFA_b, p.LGly_b, p.MGly_b, p.Hep_b, 0.0, 0.0)
            self.assertAlmostEqual(fl["dG"], 0.0, places=9)
            self.assertAlmostEqual(fl["hep_target"], p.Hep_b, places=9)
        p = resolve_derived_params(PatientParams())
        self.assertAlmostEqual(p.Hep_b, 2.0, places=6)   # the declared typical, at the typical Gb

    def test_pools_rest_at_their_declared_typicals(self):
        """No meal, awake, 12 h: every pool with a declared `_b` stays there."""
        p = resolve_derived_params(PatientParams())
        traj, _ = _run(p, [], 720)
        for marker, declared in (("gallbladder_bile", p.GB_b), ("intestinal_bile", p.INT_b),
                                 ("bile_acids", p.BA_b), ("muscle_glycogen", p.MGly_b), ("cck", p.CCK_b)):
            self.assertLess(abs(traj[-1, MI[marker]] / declared - 1.0), 0.03, marker)
        self.assertGreater(traj[60, MI["hepatic_output"]], 0.95 * p.Hep_b)

    def test_muscle_glycogen_refills_to_typical_not_to_a_cap(self):
        p = resolve_derived_params(PatientParams())
        n = 3 * 1440
        act = np.zeros(n, dtype=np.float32); act[120:180] = 0.8       # one hard bout on day 1
        traj, _ = _run(p, _std_day(3), n, act=act)
        mg = traj[:, MI["muscle_glycogen"]]
        self.assertLess(mg[200], p.MGly_b - 20.0)                       # it was spent
        # It comes back toward typical (exponential approach: the deficit drives
        # synthesis, so the last grams are slow) and never overshoots it.
        self.assertGreater(mg[-1], mg[200] + 0.7 * (p.MGly_b - mg[200]))
        self.assertLessEqual(mg.max(), p.MGly_b + 1e-6)                 # and never above typical

    def test_enterohepatic_loop_conserves_mass(self):
        """With synthesis and faecal loss both zeroed the loop's total mass is constant."""
        p = resolve_derived_params(PatientParams())
        p.f_ileal = 1.0; p = resolve_derived_params(p)    # no faecal loss -> derived synthesis 0
        traj, _ = _run(p, [(60.0, 70.0, 25.0, 30.0)], 600)
        total = (traj[:, MI["gallbladder_bile"]] + traj[:, MI["intestinal_bile"]]
                 + traj[:, MI["bile_acids"]] / p.ba_spill_gain)
        self.assertLess(total.max() - total.min(), 1e-3 * total[0])


class TestFastedState(unittest.TestCase):
    def _fast(self, gb, hours=48):
        p = PatientParams(); p.Gb = gb; p = resolve_derived_params(p)
        n = 3 * 1440
        traj, _ = _run(p, _std_day(1), n)
        t0 = int((19 - 6) * 60)
        return traj, t0

    def test_fast_floor_is_absolute_not_proportional(self):
        g48 = {}
        for gb in (70.0, 95.0, 130.0):
            traj, t0 = self._fast(gb)
            g48[gb] = traj[t0 + 48 * 60 - 30: t0 + 48 * 60 + 30, MI["glucose"]].mean()
        for gb, g in g48.items():
            self.assertTrue(60.0 <= g <= 82.0, f"Gb {gb}: 48 h glucose {g:.1f}")
        # The 48 h spread is a small fraction of the fed spread (proportional would be 1.0).
        self.assertLess((g48[130.0] - g48[70.0]) / 60.0, 0.35)

    def test_ghrelin_rises_in_a_fast(self):
        traj, t0 = self._fast(95.0)
        ghr = traj[:, MI["ghrelin"]]
        base = ghr[t0 - 1]
        self.assertGreater(ghr[t0 + 24 * 60] / base, 1.08)
        self.assertGreater(ghr[t0 + 48 * 60] / base, ghr[t0 + 24 * 60] / base)

    def test_sub_basal_insulin_accelerates_glycogenolysis(self):
        p = resolve_derived_params(PatientParams())
        lo = glucose_fluxes(p, 90.0, 4.0, 0.0, p.Gnb, p.Cort_b, p.FFA_b, 80.0, 400.0, 1.8, 0.0, 0.0)
        mid = glucose_fluxes(p, 90.0, p.Ib, 0.0, p.Gnb, p.Cort_b, p.FFA_b, 80.0, 400.0, 1.8, 0.0, 0.0)
        hi = glucose_fluxes(p, 90.0, 40.0, 0.0, p.Gnb, p.Cort_b, p.FFA_b, 80.0, 400.0, 1.8, 0.0, 0.0)
        self.assertGreater(lo["hep_target"], mid["hep_target"])
        self.assertGreater(mid["hep_target"], hi["hep_target"])

    def test_insulin_action_is_never_a_glucose_source(self):
        """Item 3.10: below the setpoint the insulin term cannot ADD glucose."""
        p = resolve_derived_params(PatientParams())
        below = glucose_fluxes(p, 85.0, 30.0, 0.008, p.Gnb, p.Cort_b, p.FFA_b, 90.0, 400.0, 2.0, 0.0, 0.0)
        self.assertGreater(below["uptake_id"], 0.0)


class TestMealAndIncretin(unittest.TestCase):
    def test_incretin_share_of_secretion(self):
        p = resolve_derived_params(PatientParams())
        n, mt = 420, 120
        traj, _ = _run(p, [(float(mt), 75.0, 5.0, 10.0)], n, start_hour=8.0)
        g = traj[:, MI["glucose"]]; glp = traj[:, MI["glp1"]]
        gsir = p.gamma * np.maximum(g - p.h, 0)
        ex = np.maximum(glp - p.GLP1_b, 0)
        inc = 1 + p.incretin_gain * ex / (ex + p.K_incretin)
        share = (gsir * (inc - 1)).sum() / (gsir * inc).sum()
        self.assertTrue(0.5 <= share <= 0.7, f"incretin share {share:.2f}")
        # And basal GLP-1 carries NO incretin effect (the absolute-vs-above-basal fix).
        self.assertAlmostEqual(float(inc[mt - 1]), 1.0, places=6)

    def test_standard_meal_holds_the_literature_excursion(self):
        p = resolve_derived_params(PatientParams())
        n, mt = 420, 120
        traj, _ = _run(p, [(float(mt), 75.0, 5.0, 10.0)], n, start_hour=8.0)
        g = traj[:, MI["glucose"]]; i = traj[:, MI["insulin"]]
        rise = g[mt:].max() - g[mt - 1]
        self.assertTrue(50.0 <= rise <= 70.0, f"75 g rise {rise:.1f}")
        self.assertTrue(40 <= int(np.argmax(g[mt:])) <= 65)
        self.assertTrue(45.0 <= i[mt:].max() <= 70.0)


if __name__ == "__main__":
    unittest.main()
