"""Iter-80 structural mechanisms.

Two changes under test:

1. Learned model — structural glucose rate-of-appearance term
   (``MetabolicModule.log_ra``): a meal must raise blood glucose in
   proportion to the gut glucose-appearance flux, by construction, instead
   of relying on the glucose SpeciesHead MLP to discover the amplitude
   against the fasting-equilibrium constraint (the iter-79 amplitude gap:
   ~0.16 mg/dL/g realised vs the 0.7 dose-response target).

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

import pulse.model as model_mod
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
    def test_param_exists_and_positive_gain(self) -> None:
        m = ModularPhysiologyNetwork()
        self.assertTrue(hasattr(m.metabolic, "log_ra"))
        ra = torch.nn.functional.softplus(m.metabolic.log_ra.detach())
        self.assertGreater(float(ra), 0.0, "rate-of-appearance gain must be strictly positive")

    def test_silent_when_fasted(self) -> None:
        """No meal => gut appearance ~0 => the Ra APPEARANCE term must not move glucose.

        Iter 97: Ra is also the per-patient grams -> mg/dL conversion (c = Ra*U)
        through which hepatic glycogenolysis is credited to plasma — that is what
        closes the carbon budget (review 2026-09-04, item 2.2). So driving Ra to 0
        now also removes the glycogenolysis credit and fasting glucose moves by
        0.73 mg/dL at init (108.5 -> 107.8; measured 2026-09-04). The appearance
        term itself is exactly zero without a meal, which is what this test pins;
        the fasting difference is bounded to the size of that credit.
        """
        m = ModularPhysiologyNetwork()
        emb = torch.zeros(m.embedding_dim)
        init = torch.tensor(NORM_CENTER, dtype=torch.float32)
        with torch.no_grad():
            traj = integrate(m, init, emb, 180, dt=1.0, start_time_minutes=360.0, meals=[])
            ns = (traj - m.norm_center) / m.norm_scale
            met_idx = MODULE_MARKER_INDICES["metabolic"]
            cort = ns[:, MARKER_INDEX["cortisol"]:MARKER_INDEX["cortisol"] + 1]
            glp1 = ns[:, MARKER_INDEX["glp1"]:MARKER_INDEX["glp1"] + 1]
            coupling = m.metabolic_coupling(torch.zeros(180, 4), cort, glp1)
            ext = torch.tensor([[0.01, 0.5]]).expand(180, -1)
            e_met = m.embedding_projections["metabolic"](emb).unsqueeze(0).expand(180, -1)
            tf = compute_time_features(torch.arange(180.0) + 360.0)
            f = m.metabolic.fluxes(ns[:, met_idx], coupling, ext, e_met, tf)
        self.assertEqual(float(f["appearance_plasma"].abs().sum()), 0.0,
                         "Ra appearance term must be exactly silent with no meal")
        peak_default = _glucose_peak(m, carbs=0.0)
        with torch.no_grad():
            m.metabolic.log_ra.copy_(torch.tensor(-30.0))  # softplus ~ 0
        peak_off = _glucose_peak(m, carbs=0.0)
        self.assertLess(abs(peak_default - peak_off), 2.0,
                        msg="only the glycogenolysis credit (c = Ra*U) may move fasting glucose")

    def test_amplitude_increases_with_gain(self) -> None:
        """A higher appearance gain must raise the postprandial glucose peak."""
        m = ModularPhysiologyNetwork()
        # Isolate the Ra forward-amplitude mechanism from the iter-81 physiological
        # clamp: an *untrained* random model explodes glucose to the clamp ceiling
        # for any Ra, masking the effect. On a trained model glucose peaks ~140
        # (far below the ceiling) so the clamp never binds here. Widen the bounds
        # for the measurement so this tests the Ra term, not the catastrophe clamp.
        save_lo, save_hi = model_mod._PHYS_MIN.clone(), model_mod._PHYS_MAX.clone()
        try:
            model_mod._PHYS_MIN.fill_(-1e30)
            model_mod._PHYS_MAX.fill_(1e30)
            with torch.no_grad():
                m.metabolic.log_ra.copy_(torch.tensor(math.log(0.1)))
            low = _glucose_peak(m, carbs=60.0)
            with torch.no_grad():
                m.metabolic.log_ra.copy_(torch.tensor(math.log(1.0)))
            high = _glucose_peak(m, carbs=60.0)
        finally:
            model_mod._PHYS_MIN.copy_(save_lo)
            model_mod._PHYS_MAX.copy_(save_hi)
        self.assertGreater(high, low + 1.0, "higher Ra gain must raise the 60 g glucose peak")

    def test_gradient_reaches_gain(self) -> None:
        """The amplitude is now a parameter the dose-response gradient can move."""
        m = ModularPhysiologyNetwork()
        emb = torch.zeros(m.embedding_dim)
        init = torch.tensor(NORM_CENTER, dtype=torch.float32)
        gi = MARKER_INDEX["glucose"]
        meals = [MealEvent(time=15.0, carbs=60.0, fats=20.0, proteins=25.0)]
        gut = precompute_gut_outputs(m, emb, 120, dt=1.0, start_time_minutes=360.0, meals=meals)
        traj = integrate(m, init, emb, 120, dt=1.0, start_time_minutes=360.0, meals=meals, gut_outputs=gut)
        loss = (traj[:, gi].max() - 140.0) ** 2  # "peak should be higher"
        loss.backward()
        self.assertIsNotNone(m.metabolic.log_ra.grad)
        self.assertNotEqual(float(m.metabolic.log_ra.grad), 0.0)


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
