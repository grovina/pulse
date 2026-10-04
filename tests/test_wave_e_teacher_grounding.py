"""PLAN.md Wave E (teacher grounding), 2026-10-04 — the three measured defects.

Every test here pins a STRUCTURE with the number that exposed the defect, in the
house style: a claim without a number is not a claim.

E1  The gluconeogenic SHARE carries the patient's EGP excess, not glycogenolysis.
    Iter 97 derived ``Hep_b = uptake_ii·Gb·V_G`` but left ``Gng_b`` an absolute
    ~1.0 mg/kg/min, so ``glyco_b = Hep_b − Gng_b`` absorbed the whole excess: on
    190 g carbohydrate/day a Gb-130 patient ended day 4 at 26 g liver glycogen
    and 1.60 mM BHB, 12.5 % of 80 sampled patients held a day-4 mean BHB above
    1 mM while eating, and corr(log Hep_b, log BHB) was +0.83. A fed person with
    prediabetic fasting glucose is not ketotic, and iter 97's own Magnusson-1992
    citation says the type-2 EGP excess is gluconeogenic.

E2  The respiratory activity drive saturates and its constant is its equilibrium.
    ``+ act * 5.0`` outside a ``k_rr = 0.1`` relaxation equilibrated at
    RR0 + 50·act: 63 /min at act 0.9, and 926 minutes above the ``rr`` marker's
    declared max of 40 across six randomized 14-day episodes.

E3  Three HPA parameters were drawn and never read. ``cort_circ_amp``,
    ``k_acth_to_cort`` and ``cort_feedback_acth`` have been dead since iter 91
    replaced cortisol's own circadian with the ACTH cascade (iter 98 rebuilt
    that cascade), and ``synthetic_users`` overrode the first in two profiles,
    which therefore did nothing. Same shape as the A5 deletion guard in
    ``test_no_unsupervised_person_heads.py``, and the same test: a parameter the
    ODE ignores must not be in ``PatientParams`` for a profile to set.
"""

import unittest

import numpy as np

import pulse.knowledge.full_body as fb
from pulse.knowledge.full_body import (
    EGP_TYPICAL_MG_KG_MIN, PatientParams, randomize_params, resolve_derived_params,
    simulate_full_body,
)
from pulse.types import MARKER_INDEX as MI, MARKERS

_START = 6.0
# scripts/iter97_teacher_validate.py STD_DAY: 190 g carbohydrate across three meals.
_STD_DAY = ((8.0, 50, 12, 20), (13.0, 65, 22, 28), (19.0, 75, 28, 35))
_CARB_PER_DAY = sum(m[1] for m in _STD_DAY)
_RR_MARKER_MAX = next(m.max for m in MARKERS if m.id == "rr")


def _fed_days(params: PatientParams, n_days: int) -> np.ndarray:
    """``n_days`` of 190 g carbohydrate, 8 h sleep (23:00–07:00), no activity."""
    n = n_days * 1440
    meals = sorted((d * 1440 + (h - _START) * 60.0, c, f, p)
                   for d in range(n_days) for h, c, f, p in _STD_DAY if h >= _START)
    sw = np.ones(n, dtype=np.float32)
    for d in range(n_days + 1):
        s, e = int((23.0 - _START) * 60) + d * 1440, int((31.0 - _START) * 60) + d * 1440
        sw[max(0, s):min(n, e)] = 0.0
    k = np.ones(20, dtype=np.float32) / 20.0
    sw = np.clip(np.convolve(sw, k, mode="same"), 0.0, 1.0).astype(np.float32)
    traj, _ = simulate_full_body(params, meals, sw, np.zeros(n, dtype=np.float32), n,
                                start_hour=_START, noise_scale=0.0,
                                rng=np.random.default_rng(0))
    return traj


def _at_gb(gb: float) -> PatientParams:
    p = PatientParams()
    p.Gb = float(gb)
    return resolve_derived_params(p)


class TestE1FedKetosisIsNotAReadoutOfFastingGlucose(unittest.TestCase):
    """The table that exposed the defect, pinned flat."""

    # Day-4 mean liver glycogen (g) and BHB (mM) on 190 g carbohydrate/day,
    # MEASURED before -> after. Before, the glycogenolytic share rose 0.32 / 0.44 /
    # 0.50 / 0.57 / 0.59 / 0.63 with Gb and took the pool and the ketones with it.
    #   Gb  70   143 / 0.07  ->  143 / 0.07   (identity: Hep_b < EGP_TYPICAL)
    #   Gb  85   122 / 0.07  ->  122 / 0.07   (identity)
    #   Gb  95    88 / 0.21  ->   88 / 0.21   (identity: Hep_b = EGP_TYPICAL)
    #   Gb 110    49 / 0.81  ->   91 / 0.19
    #   Gb 115    41 / 1.01  ->   92 / 0.18
    #   Gb 130    26 / 1.60  ->   95 / 0.15
    _GB_RANGE = (70.0, 85.0, 95.0, 110.0, 115.0, 130.0)

    def setUp(self) -> None:
        self.day4 = {}
        for gb in self._GB_RANGE:
            traj = _fed_days(_at_gb(gb), 5)
            d4 = slice(3 * 1440, 4 * 1440)
            self.day4[gb] = (float(traj[d4, MI["liver_glycogen"]].mean()),
                             float(traj[d4, MI["bhb"]].mean()))

    def test_no_patient_in_the_clinical_gb_range_is_ketotic_while_eating(self):
        for gb, (_lgly, bhb) in self.day4.items():
            self.assertLess(bhb, 0.30, f"Gb {gb}: day-4 mean BHB {bhb:.3f} mM on "
                                       f"{_CARB_PER_DAY:.0f} g carbohydrate/day")

    def test_the_fed_liver_pool_is_flat_in_fasting_glucose(self):
        """Not "high" -- FLAT. The defect was the Gb dependence, 143 g -> 26 g."""
        lgly = [self.day4[gb][0] for gb in self._GB_RANGE]
        # Before: max/min 143/26 = 5.4. After: 143/88 = 1.6, and the whole spread is
        # the LOW-Gb patient over-filling, which is the opposite failure and benign.
        self.assertLess(max(lgly) / min(lgly), 2.0)
        for gb, (lg, _bhb) in self.day4.items():
            self.assertGreater(lg, 70.0, f"Gb {gb}: day-4 mean liver glycogen {lg:.1f} g")

    def test_ketosis_no_longer_rises_with_the_patients_egp(self):
        """Before, day-4 BHB was monotone in Gb over the whole clinical range."""
        bhb = [self.day4[gb][1] for gb in self._GB_RANGE]
        hi = [self.day4[gb][1] for gb in (110.0, 115.0, 130.0)]
        # Above the typical EGP, more EGP now means LESS fed ketosis, because the
        # extra flux is gluconeogenic and the pool is no longer being spent.
        self.assertEqual(hi, sorted(hi, reverse=True))
        self.assertLess(max(bhb), 4.0 * min(bhb))   # before: 1.60 / 0.07 = 23x

    def test_the_population_stops_producing_ketotic_fed_patients(self):
        """The same 40 sampled patients under both splits, day-4 mean BHB on 190 g
        carbohydrate/day (N = 80 in brackets), before -> after:
            median                      0.215 -> 0.126   [0.353 -> 0.112]
            max                         2.043 -> 1.194   [2.922 -> 1.355]
            share above 1 mM            25.0 % -> 2.5 %  [26.3 % -> 2.5 %]
            corr(Gb, BHB)               +0.733 -> +0.136 [+0.749 -> +0.116]
            corr(log Hep_b, log BHB)    +0.749 -> +0.309 [+0.793 -> +0.274]
            day-4 liver glycogen, min    18.8 -> 34.6 g  [17.9 -> 34.6]
            day-4 liver glycogen, median 75.8 -> 95.1 g  [71.6 -> 100.3]

        The RESIDUAL 2.5 % is a different defect and must not be read as this one:
        the worst patient has Gb 94.4 -- a NORMAL fasting glucose -- and gets to
        1.19 mM because `LGly_b` is sampled over 70-130 g while the fed carbohydrate
        load is fixed, so a large-pool patient sits permanently at ~40 % of its own
        declared pool and `keto_glyc_gain`'s gate (1 + 13·(1 - LGly/LGly_b)) reads
        that as a fast. Fixing it means the ketogenesis gate reading an absolute pool
        level, or `LGly_b` co-varying with intake; it is not the GNG share.
        """
        rng = np.random.default_rng(17)
        bhb, hep, gb = [], [], []
        for _ in range(40):
            p = randomize_params(rng)
            traj = _fed_days(p, 4)
            bhb.append(float(traj[3 * 1440:4 * 1440, MI["bhb"]].mean()))
            hep.append(p.Hep_b)
            gb.append(p.Gb)
        bhb, hep, gb = np.array(bhb), np.array(hep), np.array(gb)
        self.assertLess(float(bhb.max()), 1.6, "a fed patient is frankly ketotic")
        self.assertLess(float((bhb > 1.0).mean()), 0.10)
        self.assertLess(float(np.median(bhb)), 0.20)
        # The point of E1: fed ketosis is no longer a reading of fasting glucose.
        self.assertLess(abs(float(np.corrcoef(gb, bhb)[0, 1])), 0.40)
        self.assertLess(abs(float(np.corrcoef(np.log(hep), np.log(bhb))[0, 1])), 0.50)


class TestE1TheShareIsAFraction(unittest.TestCase):
    def test_the_gluconeogenic_share_is_a_fraction_for_every_sampled_patient(self):
        """f_gng in (0, 0.85] and glycogenolysis >= 15 % of EGP, which is what the
        iter-97 ``min(Gng_b, 0.85*Hep_b)`` clip existed to guarantee. MEASURED over
        400 patients: sampled f_gng 0.375-0.698 (median 0.505), realized Gng_b/Hep_b
        0.376-0.850 (median 0.595), and the 0.85 guard binds for 1 of the 400."""
        rng = np.random.default_rng(3)
        for k in range(200):
            p = randomize_params(rng)
            self.assertTrue(0.0 < p.f_gng <= 0.85, f"patient {k}: f_gng {p.f_gng}")
            share = p.Gng_b / p.Hep_b
            self.assertTrue(0.0 < share <= 0.85 + 1e-12,
                            f"patient {k}: Gng_b/Hep_b {share:.4f}")
            self.assertGreater(p.Hep_b - p.Gng_b, 0.0,
                               f"patient {k}: glycogenolysis {p.Hep_b - p.Gng_b:.4f}")

    def test_resolve_is_idempotent_in_the_new_split(self):
        """``simulate_full_body`` resolves on every call, so a second pass must not
        move Gng_b -- the failure mode would compound the excess term silently."""
        rng = np.random.default_rng(11)
        for _ in range(25):
            p = randomize_params(rng)
            g1, f1 = p.Gng_b, p.f_gng
            resolve_derived_params(resolve_derived_params(p))
            self.assertAlmostEqual(p.Gng_b, g1, places=12)
            self.assertAlmostEqual(p.f_gng, f1, places=12)

    def test_the_split_is_an_identity_at_or_below_the_typical_egp(self):
        """So the default patient, the 24 h-fast reference numbers, the 75 g meal and
        the absolute prolonged-fast floor are iter 97's exactly (verified end to end:
        scripts/iter97_teacher_validate.py changes in ONE line, the Gb-130 fast)."""
        for gb in (70.0, 85.0, 95.0):
            p = _at_gb(gb)
            self.assertLessEqual(p.Hep_b, EGP_TYPICAL_MG_KG_MIN + 1e-12)
            self.assertAlmostEqual(p.Gng_b, p.f_gng * EGP_TYPICAL_MG_KG_MIN, places=12)
        # And above it, glycogenolysis is pinned at the typical liver's own turnover
        # (Magnusson 1992: it is LOWER in type-2 diabetes, not higher), so the excess
        # is entirely gluconeogenic.
        base = _at_gb(95.0)
        for gb in (110.0, 115.0, 130.0):
            p = _at_gb(gb)
            self.assertGreater(p.Hep_b, EGP_TYPICAL_MG_KG_MIN)
            self.assertAlmostEqual(p.Hep_b - p.Gng_b, base.Hep_b - base.Gng_b, places=12)
            self.assertGreater(p.Gng_b, base.Gng_b)

    def test_the_default_patient_is_landaus_post_absorptive_share(self):
        """Landau 1996: 47 % of EGP at 14 h of fasting. The default f_gng 0.5 is the
        1.0/2.0 the absolute default already encoded, so nothing moved at the median."""
        p = resolve_derived_params(PatientParams())
        self.assertAlmostEqual(p.f_gng, 0.5, places=12)
        self.assertAlmostEqual(p.Gng_b, 1.0, places=12)
        self.assertAlmostEqual(p.Hep_b, EGP_TYPICAL_MG_KG_MIN, places=12)


class TestE2RespiratoryRateIsPhysiological(unittest.TestCase):
    def _rr_equilibrium(self, act: float, params: PatientParams | None = None) -> float:
        p = resolve_derived_params(PatientParams() if params is None else params)
        n = 8 * 60
        traj, _ = simulate_full_body(p, [], np.ones(n, dtype=np.float32),
                                     np.full(n, act, dtype=np.float32), n,
                                     start_hour=8.0, noise_scale=0.0,
                                     rng=np.random.default_rng(0))
        return float(traj[-1, MI["rr"]])

    def test_the_equilibrium_spans_a_physiological_range(self):
        """MEASURED before -> after, RR equilibrium at act 0 / 0.3 / 0.5 / 0.7 / 0.9 /
        1.0: 15.0 / 30.7 / 40.9 / 51.5 / 63.1 / 68.4 -> 15.0 / 21.3 / 26.3 / 31.5 /
        35.0 / 36.1 /min. Maximal breathing frequency is ~35-45 in an untrained adult
        (~40-60 trained); act 0.5 is moderate effort, not maximal."""
        rr = {a: self._rr_equilibrium(a) for a in (0.0, 0.3, 0.5, 0.7, 0.9, 1.0)}
        self.assertAlmostEqual(rr[0.0], PatientParams().RR0, places=3)   # resting EXACT
        self.assertTrue(19.0 <= rr[0.3] <= 24.0, f"act 0.3: {rr[0.3]:.1f}")
        self.assertTrue(23.0 <= rr[0.5] <= 30.0, f"act 0.5: {rr[0.5]:.1f}")
        self.assertTrue(30.0 <= rr[0.9] <= 40.0, f"act 0.9: {rr[0.9]:.1f}")
        self.assertTrue(30.0 <= rr[1.0] <= 42.0, f"act 1.0: {rr[1.0]:.1f}")
        # Monotone, and saturating: the last 0.1 of effort buys less than the first.
        vals = [rr[a] for a in (0.0, 0.3, 0.5, 0.7, 0.9, 1.0)]
        self.assertEqual(vals, sorted(vals))
        self.assertLess(rr[1.0] - rr[0.9], rr[0.5] - rr[0.3])

    def test_the_rise_still_clears_the_exercise_rule(self):
        """``rr_rises_with_exercise`` (ACSM) wants >= 5 /min at the 0.7 of the
        ``moderate_exercise_bout`` arm. MEASURED +16.5 /min."""
        self.assertGreater(self._rr_equilibrium(0.7) - self._rr_equilibrium(0.0), 5.0)

    def test_no_episode_minute_exceeds_the_markers_declared_max(self):
        """MEASURED 926 minutes above 40 /min across six randomized 14-day episodes
        before (154 per episode, 6 of 6 affected, peak 60.4); 0 and peak 35.0 after."""
        rng = np.random.default_rng(7)
        over, peak = 0, 0.0
        for _ in range(4):
            prng = np.random.default_rng(rng.integers(0, 2 ** 32))
            p = randomize_params(prng)
            n_days, n = 14, 14 * 1440
            traj, _ = simulate_full_body(
                p, fb.generate_meal_plan(n_days, prng, _START),
                fb.generate_sleep_wake(n_days, n, _START, prng),
                fb.generate_activity(n_days, n, _START, prng), n, _START, rng=prng)
            rr = traj[:, MI["rr"]]
            over += int((rr > _RR_MARKER_MAX).sum())
            peak = max(peak, float(rr.max()))
        self.assertEqual(over, 0, f"peak RR {peak:.1f} against the marker max "
                                  f"{_RR_MARKER_MAX}")

    def test_the_sleep_dip_is_untouched(self):
        """Both drives are now equilibrium offsets INSIDE the relaxation, so the
        sleep shift still lands exactly on ``sleep_rr_drop`` -- which is what keeps the
        ``rr_sleep_dip`` anchor (-2.5 +/- 1.5) where iter 94 left it, teacher -2.84 on
        the default patient both before and after."""
        p = resolve_derived_params(PatientParams())
        n = 8 * 60
        traj, _ = simulate_full_body(p, [], np.zeros(n, dtype=np.float32),
                                     np.zeros(n, dtype=np.float32), n, start_hour=0.0,
                                     noise_scale=0.0, rng=np.random.default_rng(0))
        self.assertAlmostEqual(float(traj[-1, MI["rr"]]),
                               p.RR0 - p.sleep_rr_drop, places=2)


class TestE3NoDeadParameters(unittest.TestCase):
    """Deleted, not marked dead: ``synthetic_users.PROFILES`` sets parameters by
    ``setattr``, so a field that survives as dead lets a profile declare a phenotype
    and get nothing -- which is what ``shift_worker`` and ``anxious_stress`` did with
    ``cort_circ_amp`` for 18 iterations."""

    _DELETED = ("cort_circ_amp", "k_acth_to_cort", "cort_feedback_acth")

    def test_the_dead_hpa_parameters_are_gone_from_patient_params(self):
        ref = PatientParams()
        for name in self._DELETED:
            self.assertFalse(
                hasattr(ref, name),
                f"{name} is back in PatientParams. It was dead from iter 91 (the ACTH "
                f"cascade carries cortisol's rhythm and `crh_fb_amp` carries the "
                f"feedback); if the ODE now reads it, say so here and give it a draw.")

    def test_nothing_in_the_ode_reads_them_by_any_route(self):
        """A sampled patient must not acquire them either -- `setattr` would create
        the attribute silently, which is exactly the trap being closed."""
        rng = np.random.default_rng(23)
        for _ in range(10):
            p = randomize_params(rng)
            for name in self._DELETED:
                self.assertFalse(hasattr(p, name), name)

    def test_sg_is_derived_and_declared_a_diagnostic(self):
        """``Sg`` stays, and the test says why: it is DERIVED (no RNG draw, no stream
        position) and read only by reporting, so it is not the ``cort_circ_amp`` trap.
        If it is ever deleted, the edits are ``scripts/iter97_teacher_validate.py``'s
        derived-parameter print and the iter-97 derivation test."""
        rng = np.random.default_rng(29)
        for _ in range(10):
            p = randomize_params(rng)
            self.assertAlmostEqual(p.Sg, p.uptake_ii * (1.0 + p.hep_autoreg_m), places=12)


if __name__ == "__main__":
    unittest.main()
