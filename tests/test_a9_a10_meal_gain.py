"""PLAN A9/A10 — meal amplitude has ONE per-person gain, and it is bounded by 1.

Through iter 109 a meal's effect on glucose carried three multiplying per-person
gains: the gut kernel's ``f_bio``, ``MetabolicModule``'s ``Ra`` (``log_ra`` +
``ra_baseline_net``, iters 80/88) and ``1/V_G`` via body mass. Glucose data
identifies only their product — measured, scaling Ra and body mass together by
1.2 moves glucose 0.18 mg/dL — so two of the three were free directions that
calibration could wander along without the loss noticing, which is the
Gnb-drift failure mode the plan opens with.

A10 deletes ``Ra``. A9 bounds what is left: ``f_bio = APPEARANCE_UNITS_PER_G ·
σ(raw)``, so the bioavailable fraction cannot exceed 1 and ``absorbed ≤
ingested`` is a property of the functional form (``softplus`` had no ceiling, and
above 1 it minted carbon the person never ate while below 1 it deleted carbon no
ledger booked as malabsorbed). ``V_G`` stays a known anatomical scale.

These tests pin the deletion, the bound as the metabolic module sees it, and that
the cold-start regime survived: the gain is still 0.8, so the appearance that
reaches glucose is the same number it was before the move.
"""

from __future__ import annotations

import statistics
import unittest

import torch

from pulse.model import ModularPhysiologyNetwork, integrate, precompute_gut_outputs
from pulse.modules import metabolic as M
from pulse.modules.base import GutModuleBase, compute_time_features
from pulse.modules.gut import MEAL_ACTIVE_WINDOW_MIN, MealEvent
from pulse.types import EMBEDDING_DIM, MARKER_INDEX as MI, MG_DL_PER_G, NORM_CENTER

_CENTER = torch.tensor(NORM_CENTER)

# A fresh model's 75 g carbohydrate meal peak, mg/dL above the pre-meal level,
# averaged over four seeds. MEASURED on the pre-A9/A10 tree (Ra = 0.8 ×
# f_bio = MG_DL_PER_G): per-seed 24.95 / 24.64 / 22.01 / 28.40, mean 25.00.
# After (f_bio = 0.8 × MG_DL_PER_G, no Ra): 27.38 / 22.07 / 21.91 / 28.47,
# mean 24.96 — 0.2 % apart. The per-seed numbers move ±10 % and the mean does
# not, because deleting two heads shifts the global RNG stream: "seed 0" is a
# different DRAW after the change, not a different model of the same person, so
# the distribution is the only thing a seeded comparison can hold fixed.
_PRE_CHANGE_MEAN_PEAK = 25.00
_PEAK_TOLERANCE = 0.15


def _model(seed: int = 0, hidden: int = 32) -> ModularPhysiologyNetwork:
    torch.manual_seed(seed)
    m = ModularPhysiologyNetwork(
        metabolic_hidden=hidden, appetite_hidden=16, stress_hidden=16, cardiovascular_hidden=16,
        thermoreg_hidden=16, respiratory_hidden=16, gut_hidden=16, hepatobiliary_hidden=16)
    m.eval()
    return m


def _meal_peak(m: ModularPhysiologyNetwork, carbs: float = 75.0, n: int = 120) -> float:
    """Glucose rise above the pre-meal level for a carbohydrate-only meal at +30.

    120 min is enough: the untrained peak lands 53-59 min after the meal (the fresh
    insulin head has no second phase, so it is later than the teacher's 55 but not
    by much), and the measured peaks are identical at n = 120, 180 and 300.
    """
    meals = [MealEvent(time=30.0, carbs=carbs, fats=0.0, proteins=0.0)]
    with torch.no_grad():
        tr = integrate(
            m, _CENTER.clone(), torch.zeros(EMBEDDING_DIM), n,
            start_time_minutes=8 * 60.0, meals=meals,
            sleep_wake=torch.ones(n), activity=torch.zeros(n))
    g = tr[:, MI["glucose"]]
    return float((g - g[30]).max())


class TestTheRaGainIsGone(unittest.TestCase):
    def test_no_ra_parameter_head_method_or_constant_survives(self) -> None:
        m = _model(0)
        met = m.metabolic
        names = {n for n, _ in met.named_parameters()}
        self.assertNotIn("log_ra", names)
        self.assertFalse(any("ra_baseline" in n for n in names), msg=str(names))
        for attr in ("log_ra", "ra_baseline_net", "appearance_gain"):
            self.assertFalse(hasattr(met, attr), msg=attr)
        for const in ("_RA_INIT", "_RA_BASELINE_MAX_Z"):
            self.assertFalse(hasattr(M, const), msg=const)

    def test_the_constants_dict_no_longer_carries_ra(self) -> None:
        """``drives`` used to read ``const["ra"]``; a stale consumer would now
        KeyError rather than silently multiplying by 1."""
        m = _model(1)
        emb = m.embedding_projections["metabolic"](torch.zeros(1, EMBEDDING_DIM))
        with torch.no_grad():
            c = m.metabolic.constants(emb)
        self.assertNotIn("ra", c)

    def test_appearance_in_grams_is_the_gut_density_over_the_population_constant(self) -> None:
        """A10: nothing rescales the kernel's mass on the way in. ``app_g`` is
        exactly ``coupling / MG_DL_PER_G``, so the gut's bioavailable grams and the
        metabolic module's carbon ledger are the same grams."""
        m = _model(2)
        met = m.metabolic
        emb = m.embedding_projections["metabolic"](0.5 * torch.randn(8, EMBEDDING_DIM))
        coupling = torch.zeros(8, M._N_COUPLING)
        coupling[:, 0] = torch.linspace(0.0, 3.0, 8)
        external = torch.zeros(8, 2)
        external[:, 1] = 1.0
        tf = compute_time_features(torch.full((8,), 600.0))
        with torch.no_grad():
            c = met.constants(emb)
            d = met.drives(external, coupling, tf, c)
        torch.testing.assert_close(d["app_g"], coupling[:, 0] / MG_DL_PER_G, atol=1e-7, rtol=1e-6)
        # and the only remaining per-person factor on the way to glucose is V_G
        torch.testing.assert_close(d["app_eff"], d["app_g"] * c["mg_dl_per_g"], atol=1e-7, rtol=1e-6)

    def test_body_mass_is_the_only_per_person_scale_left_in_the_module(self) -> None:
        """The Ra↔mass confound A4 names: with Ra gone, moving body mass changes
        the mg/dL a gram becomes and NOTHING else about appearance, so the two are
        no longer a flat direction in the module."""
        m = _model(3)
        met = m.metabolic
        emb = m.embedding_projections["metabolic"](torch.zeros(1, EMBEDDING_DIM))
        coupling = torch.zeros(1, M._N_COUPLING)
        coupling[:, 0] = 2.0
        external = torch.tensor([[0.0, 1.0]])
        tf = compute_time_features(torch.tensor([600.0]))
        with torch.no_grad():
            c0 = met.constants(emb)
            d0 = met.drives(external, coupling, tf, c0)
            met.body_mass_net[-1].bias.fill_(1.0)
            c1 = met.constants(emb)
            d1 = met.drives(external, coupling, tf, c1)
        self.assertGreater(float(c1["body_mass_kg"]), float(c0["body_mass_kg"]))
        self.assertAlmostEqual(float(d0["app_g"]), float(d1["app_g"]), places=7)
        self.assertLess(float(d1["app_eff"]), float(d0["app_eff"]))


class TestBioavailabilityBoundsTheMealAsTheMetabolicModuleSeesIt(unittest.TestCase):
    def test_absorbed_grams_never_exceed_ingested_grams_through_the_gut_module(self) -> None:
        """A9 in the unit the carbon ledger uses. The gut's glucose channel is
        integrated over a whole window, converted to grams with ``MG_DL_PER_G``
        (the same constant ``drives`` divides by) and compared with the dose.
        Holds for every embedding, including past the ‖e‖ ≤ 8 calibration clamp,
        because ``σ`` is the bound and not a trained one."""
        m = _model(4)
        n = 600
        dose = 90.0
        meals = [MealEvent(time=0.0, carbs=dose, fats=30.0, proteins=40.0)]
        g = torch.Generator().manual_seed(5)
        for scale in (0.0, 1.0, 3.0, 8.0, 30.0):
            emb = scale * torch.randn(EMBEDDING_DIM, generator=g)
            with torch.no_grad():
                gut = precompute_gut_outputs(m, emb, n, meals=meals)
            absorbed_g = float(gut[:, 0].sum()) / MG_DL_PER_G
            self.assertLessEqual(absorbed_g, dose + 1e-2, msg=f"‖e‖~{scale}: {absorbed_g:.2f} g")
            self.assertGreater(absorbed_g, 0.0)

    def test_the_fraction_that_reached_glucose_is_the_kernels_own_fraction(self) -> None:
        """∫appearance = fraction · mass, across the module boundary: the grams the
        metabolic module books equal the kernel's bioavailable fraction times the
        dose. One gain, readable in one place."""
        m = _model(6)
        n = int(MEAL_ACTIVE_WINDOW_MIN) + 20   # past the mask; nothing is added after it
        dose = 60.0
        meals = [MealEvent(time=0.0, carbs=dose, fats=0.0, proteins=0.0)]
        g = torch.Generator().manual_seed(7)
        for scale in (0.0, 2.0, 6.0):
            emb = scale * torch.randn(EMBEDDING_DIM, generator=g)
            with torch.no_grad():
                gut = precompute_gut_outputs(m, emb, n, meals=meals)
                frac = m.gut.kernel.bioavailable_fraction(
                    m.embedding_projections["gut"](emb))
            absorbed_g = float(gut[:, 0].sum()) / MG_DL_PER_G
            # The active-window mask truncates at MEAL_ACTIVE_WINDOW_MIN, where every
            # basis component has < 1 % of its mass left (that bound is itself pinned
            # in tests/test_iter97_student_gut.py), so allow 2 %.
            self.assertAlmostEqual(absorbed_g, dose * float(frac[0]), delta=0.02 * dose,
                                   msg=f"‖e‖~{scale}")


class TestColdStartIsPreserved(unittest.TestCase):
    def test_the_fresh_per_gram_gain_into_glucose_is_unchanged(self) -> None:
        """The exact statement of "preserve the cold-start regime": the product
        that used to be ``Ra_init · f_bio_init`` = 0.8 × 7.722 = 6.178 units/g is
        now ``f_bio_init`` alone. Measured at the zero embedding: 6.194, the 0.3 %
        being the kernel output layer's own 0.01-std weight noise."""
        m = _model(0)
        with torch.no_grad():
            _, f_bio = m.gut.kernel.mixture(
                m.embedding_projections["gut"](torch.zeros(EMBEDDING_DIM)))
        expected = GutModuleBase.F_BIO_INIT_FRACTION * MG_DL_PER_G
        self.assertAlmostEqual(expected, 6.178, places=3)
        self.assertAlmostEqual(float(f_bio[0]), expected, delta=0.03 * expected)

    def test_a_fresh_models_75g_meal_peak_is_within_15_percent_of_the_old_one(self) -> None:
        """The end-to-end check that moving the gain did not move the physiology.
        See ``_PRE_CHANGE_MEAN_PEAK`` for the measured before/after."""
        peaks = [_meal_peak(_model(s)) for s in range(4)]
        mean = statistics.mean(peaks)
        rel = abs(mean - _PRE_CHANGE_MEAN_PEAK) / _PRE_CHANGE_MEAN_PEAK
        self.assertLess(rel, _PEAK_TOLERANCE,
                        msg=f"mean peak {mean:.2f} vs pre-change {_PRE_CHANGE_MEAN_PEAK:.2f} "
                            f"({rel:.1%}); per-seed {[round(p, 2) for p in peaks]}")
        # and it is still a meal response, not a flat line or a runaway
        self.assertGreater(mean, 10.0)
        self.assertLess(mean, 80.0)

    def test_the_meal_response_is_still_linear_in_dose(self) -> None:
        """Deleting a gain must not have broken the one property that made the
        dose-response signal meaningful: appearance is exactly linear in dose, so
        a half dose is a smaller excursion, monotonically."""
        m = _model(0)
        peaks = [_meal_peak(m, carbs=d) for d in (0.0, 25.0, 50.0, 75.0)]
        self.assertEqual(peaks[0], max(0.0, peaks[0]))
        for lo, hi in zip(peaks, peaks[1:]):
            self.assertLess(lo, hi)


if __name__ == "__main__":
    unittest.main()
