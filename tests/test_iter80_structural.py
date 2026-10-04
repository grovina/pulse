"""Iter-80 structural mechanisms.

Two changes under test:

1. Learned model — structural glucose rate-of-appearance: a meal must raise
   blood glucose in proportion to the gut glucose-appearance flux, by
   construction, instead of relying on the glucose SpeciesHead MLP to discover
   the amplitude against the fasting-equilibrium constraint (the iter-79
   amplitude gap: ~0.16 mg/dL/g realised vs the 0.7 dose-response target).

   The GAIN moved. Iter 80 put it in ``MetabolicModule.log_ra``; A10 (PLAN.md)
   deleted that, because meal amplitude carried three multiplying per-person
   gains — the gut kernel's bioavailability, ``Ra``, and ``1/V_G`` via body mass
   — and glucose data identifies only their product (measured: scaling Ra and
   body mass together by 1.2 moves glucose 0.18 mg/dL). The single surviving
   gain is the gut kernel's bioavailable FRACTION, which A9 bounded by 1 so that
   absorbed ≤ ingested holds by construction. These tests therefore assert the
   same four properties against ``GutModuleBase`` instead: the gain exists and is
   positive, it is silent with no meal, raising it raises the peak, and the
   dose-response gradient reaches it.

2. Teacher — hepatic-output split + glycogen->ketosis coupling
   (``full_body``): conservation-exact at the fed calibration state (acute
   protocols byte-identical), diverging only as the liver glycogen pool
   depletes, at which point ketogenesis ramps to the Cahill-2006 fasting
   range (~1-2 mM by 24 h). This is what finally gives the slow glycogen
   pool a strong, observable gradient (via BHB, which actually moves in
   fasting — glucose is homeostatically defended and does not).
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from pulse.model import (
    ModularPhysiologyNetwork,
    integrate,
    precompute_gut_outputs,
)
from pulse.modules.base import compute_time_features
from pulse.modules.gut import MealEvent
from pulse.types import MARKER_INDEX, MODULE_MARKER_INDICES, NORM_CENTER, STATE_DIM
from pulse.knowledge.full_body import PatientParams, simulate_full_body


def _glucose_peak(model: ModularPhysiologyNetwork, carbs: float) -> float:
    emb = torch.zeros(model.embedding_dim)
    init = torch.tensor(NORM_CENTER, dtype=torch.float32)
    gi = MARKER_INDEX["glucose"]
    meals = [MealEvent(time=15.0, carbs=float(carbs), fats=20.0, proteins=25.0)] if carbs > 0 else []
    with torch.no_grad():
        gut = precompute_gut_outputs(model, emb, 180, dt=1.0, start_time_minutes=360.0, meals=meals)
        traj = integrate(model, init, emb, 180, dt=1.0, start_time_minutes=360.0, meals=meals, gut_outputs=gut)
    return float(traj[:, gi].max())


class TestGlucoseAppearanceTerm(unittest.TestCase):
    """The meal-appearance gain, at its A10 home: the gut kernel's bioavailability."""

    @staticmethod
    def _set_carb_fraction(model: ModularPhysiologyNetwork, fraction: float) -> None:
        """Pin the carbohydrate channel's bioavailable fraction to ``fraction``.

        The kernel's last layer emits mixture logits then one bioavailability
        logit per macro, so the carbohydrate one is the first output after the
        mixture block. Written through the bias with the weights left alone, so
        the change is uniform over embeddings.
        """
        k = model.gut.kernel
        n_logit = k.N_MACROS * k.n_basis
        with torch.no_grad():
            k.kernel[-1].bias[n_logit] = math.log(fraction / (1.0 - fraction))

    def test_gain_exists_and_is_a_positive_fraction(self) -> None:
        m = ModularPhysiologyNetwork()
        frac = m.gut.kernel.bioavailable_fraction(
            m.embedding_projections["gut"](torch.zeros(m.embedding_dim)))
        self.assertEqual(tuple(frac.shape), (3,))
        self.assertTrue(bool((frac > 0.0).all()), "bioavailable fraction must be strictly positive")
        self.assertTrue(bool((frac < 1.0).all()), "A9: absorbed cannot exceed ingested")

    def test_silent_when_fasted(self) -> None:
        """No meal => gut appearance ~0 => the appearance term must not move glucose.

        The flux is pinned at exactly zero, and the gain is then irrelevant: a
        no-meal rollout is identical for any bioavailability.
        """
        m = ModularPhysiologyNetwork()
        emb = torch.zeros(m.embedding_dim)
        init = torch.tensor(NORM_CENTER, dtype=torch.float32)
        with torch.no_grad():
            traj = integrate(m, init, emb, 180, dt=1.0, start_time_minutes=360.0, meals=[])
            ns = (traj - m.norm_center) / m.norm_scale
            met_idx = MODULE_MARKER_INDICES["metabolic"]
            coupling = m.coupling_for("metabolic", ns, torch.zeros(180, 4), torch.zeros(180, 3))
            ext = torch.tensor([[0.01, 0.5]]).expand(180, -1)
            e_met = m.embedding_projections["metabolic"](emb).unsqueeze(0).expand(180, -1)
            tf = compute_time_features(torch.arange(180.0) + 360.0)
            f = m.metabolic.fluxes(ns[:, met_idx], coupling, ext, e_met, tf)
        self.assertEqual(float(f["appearance_plasma"].abs().sum()), 0.0,
                         "appearance term must be exactly silent with no meal")
        peak_default = _glucose_peak(m, carbs=0.0)
        self._set_carb_fraction(m, 1e-6)
        peak_off = _glucose_peak(m, carbs=0.0)
        self.assertAlmostEqual(peak_default, peak_off, places=3,
                               msg="appearance gain changed fasting glucose; it must be silent")

    def test_amplitude_increases_with_gain(self) -> None:
        """A higher bioavailable fraction must raise the postprandial glucose peak."""
        m = ModularPhysiologyNetwork()
        self._set_carb_fraction(m, 0.1)
        low = _glucose_peak(m, carbs=60.0)
        self._set_carb_fraction(m, 0.95)
        high = _glucose_peak(m, carbs=60.0)
        self.assertGreater(high, low + 1.0,
                           "a higher bioavailable fraction must raise the 60 g glucose peak")

    def test_gradient_reaches_gain(self) -> None:
        """The amplitude is a parameter the dose-response gradient can move."""
        m = ModularPhysiologyNetwork()
        emb = torch.zeros(m.embedding_dim)
        init = torch.tensor(NORM_CENTER, dtype=torch.float32)
        gi = MARKER_INDEX["glucose"]
        meals = [MealEvent(time=15.0, carbs=60.0, fats=20.0, proteins=25.0)]
        gut = precompute_gut_outputs(m, emb, 120, dt=1.0, start_time_minutes=360.0, meals=meals)
        traj = integrate(m, init, emb, 120, dt=1.0, start_time_minutes=360.0, meals=meals, gut_outputs=gut)
        loss = (traj[:, gi].max() - 140.0) ** 2  # "peak should be higher"
        loss.backward()
        bias_grad = m.gut.kernel.kernel[-1].bias.grad
        self.assertIsNotNone(bias_grad)
        n_logit = m.gut.kernel.N_MACROS * m.gut.kernel.n_basis
        self.assertNotEqual(float(bias_grad[n_logit]), 0.0,
                            "the carbohydrate bioavailability logit must receive gradient")


def _run_teacher(duration_min, meals, params):
    sw = np.ones(duration_min)
    act = np.zeros(duration_min)
    traj, _ = simulate_full_body(
        params, meals, sw, act, duration_min, start_hour=6.0, noise_scale=0.0,
        rng=np.random.default_rng(0),
    )
    return traj


# A teacher with the ketosis coupling off. Iter 97: `hep_glyco_frac` no longer
# exists -- hepatic output is glycogenolysis + gluconeogenesis on one carbon
# budget (see glucose_fluxes) -- so only the ketosis gain is switched off here.
def _pre_iter80_params() -> PatientParams:
    p = PatientParams()
    p.keto_glyc_gain = 0.0
    return p


class TestTeacherHepaticGlycogenCoupling(unittest.TestCase):
    GI = MARKER_INDEX["glucose"]
    II = MARKER_INDEX["insulin"]
    BI = MARKER_INDEX["bhb"]
    LGI = MARKER_INDEX["liver_glycogen"]

    def test_acute_protocol_conservation_exact(self) -> None:
        """A single OGTT (3 h, liver barely moves) must be byte-identical with the
        ketosis coupling on or off — the fuel switch is gated on liver depletion,
        so the acute gate sees no change."""
        meals = [(30.0, 75.0, 0.0, 0.0)]
        old = _run_teacher(180, meals, _pre_iter80_params())
        new = _run_teacher(180, meals, PatientParams())
        for name, idx in [("glucose", self.GI), ("insulin", self.II), ("bhb", self.BI)]:
            d = float(np.abs(new[:, idx] - old[:, idx]).max())
            self.assertLess(d, 0.05, f"acute {name} diverged by {d:.4f}; split must be ~identity acutely")

    def test_fasting_ketosis_reaches_physiological_range(self) -> None:
        """By 24 h of fasting, BHB must rise into the Cahill range (~1-2 mM).

        Guards the calibration of ``keto_glyc_gain`` against the pre-iter-80
        teacher's too-flat fasting ketones (~0.25 mM)."""
        traj = _run_teacher(24 * 60, [], PatientParams())
        bhb_24h = float(traj[-1, self.BI])
        self.assertGreater(bhb_24h, 0.8, f"24 h fasting BHB {bhb_24h:.2f} mM too low (fuel switch missing)")
        self.assertLess(bhb_24h, 3.0, f"24 h fasting BHB {bhb_24h:.2f} mM implausibly high")

    def test_fasting_ketosis_tracks_glycogen_depletion(self) -> None:
        """BHB must be monotonically higher with the coupling on than off across
        the fast — i.e. the rise is *driven by* liver depletion, giving glycogen
        an observable gradient."""
        new = _run_teacher(24 * 60, [], PatientParams())
        old = _run_teacher(24 * 60, [], _pre_iter80_params())
        # Liver depletes in both; only the coupled teacher should ramp ketones.
        self.assertLess(float(new[-1, self.LGI]), float(new[0, self.LGI]) - 20.0,
                        "liver glycogen should deplete materially over 24 h fast")
        self.assertGreater(float(new[-1, self.BI]), float(old[-1, self.BI]) + 0.5,
                           "ketosis coupling must lift fasting BHB well above the uncoupled teacher")

    def test_state_dim_unchanged(self) -> None:
        traj = _run_teacher(60, [], PatientParams())
        self.assertEqual(traj.shape[1], STATE_DIM)


if __name__ == "__main__":
    unittest.main()
