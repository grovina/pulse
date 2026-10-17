"""Urinary glucose above the measured renal threshold."""

from __future__ import annotations

import unittest

import torch

import numpy as np

from pulse.knowledge.full_body import (
    PatientParams, glucose_fluxes, resolve_derived_params, simulate_full_body,
)
from pulse.modules.base import compute_time_features
from pulse.modules.metabolic import MetabolicModule
from pulse.renal import (
    EMPA_GFR_ML_PER_MIN, EMPA_RTG_MG_DL, EMPA_UGE_G_PER_DAY, EMPA_UGE_SE_G_PER_DAY,
    GFR_ML_PER_MIN, RTG_MG_DL, urinary_mg_per_min, urinary_mg_per_min_t, urinary_rate,
)
from pulse.types import EMBEDDING_DIM, MARKER_INDEX


class TestTheThreshold(unittest.TestCase):
    def test_nothing_leaves_at_or_below_it(self) -> None:
        self.assertEqual(urinary_mg_per_min(83.0), 0.0)
        self.assertEqual(urinary_mg_per_min(RTG_MG_DL), 0.0)
        self.assertEqual(RTG_MG_DL, 190.0)

    def test_ten_above_the_threshold_is_the_clearance_times_that_excess(self) -> None:
        # 10 mg/dL is 0.10 mg/mL, times 139 mL/min.
        self.assertAlmostEqual(urinary_mg_per_min(200.0), GFR_ML_PER_MIN * 0.10, places=6)

    def test_the_tensor_is_the_scalar(self) -> None:
        g = torch.tensor([80.0, 190.0, 250.0])
        got = urinary_mg_per_min_t(g)
        exp = torch.tensor([urinary_mg_per_min(float(v)) for v in g])
        torch.testing.assert_close(got, exp)


class TestTheLedger(unittest.TestCase):
    def test_fasting_glucose_is_unchanged_and_a_high_glucose_loses_the_excess(self) -> None:
        p = resolve_derived_params(PatientParams())
        fast = glucose_fluxes(
            p, p.Gb, p.Ib, 0.0, p.Gnb, p.Cort_b, p.FFA_b,
            p.LGly_b, p.MGly_b, p.Hep_b, 0.0, 0.0,
        )
        self.assertEqual(fast["urinary"], 0.0)
        self.assertAlmostEqual(fast["dG"], 0.0, places=9)
        high = glucose_fluxes(
            p, 250.0, p.Ib, 0.0, p.Gnb, p.Cort_b, p.FFA_b,
            p.LGly_b, p.MGly_b, p.Hep_b, 0.0, 0.0,
        )
        self.assertAlmostEqual(high["urinary"], urinary_rate(250.0, p.mg_dl_per_g()), places=6)
        self.assertGreater(high["urinary"], 0.4)


class TestEmpagliflozin(unittest.TestCase):
    def test_the_dose_moves_the_threshold_and_the_clearance(self) -> None:
        # 50 mg/dL above 34 is 0.50 mg/mL, times the day-13 clearance.
        at_fasting = urinary_mg_per_min(84.0, 1.0)
        self.assertAlmostEqual(at_fasting, EMPA_GFR_ML_PER_MIN * (84.0 - EMPA_RTG_MG_DL) / 100.0)
        self.assertEqual(urinary_mg_per_min(EMPA_RTG_MG_DL, 1.0), 0.0)
        self.assertEqual(urinary_mg_per_min(84.0, 0.0), 0.0)
        self.assertEqual(urinary_mg_per_min(84.0, 2.0), at_fasting)
        self.assertEqual(urinary_mg_per_min(84.0, -1.0), 0.0)

    def test_the_tensor_is_the_scalar(self) -> None:
        g = torch.tensor([84.0, 34.0, 120.0])
        exposure = torch.tensor([1.0, 1.0, 0.5])
        got = urinary_mg_per_min_t(g, exposure)
        exp = torch.tensor([
            urinary_mg_per_min(float(a), float(b)) for a, b in zip(g, exposure)
        ])
        torch.testing.assert_close(got, exp)

    def test_the_student_subtracts_the_same_rate(self) -> None:
        met = MetabolicModule(EMBEDDING_DIM)
        state = torch.zeros(1, met.n_species)
        center = float(met.typical_val[0])
        scale = float(met.norm_scale_val[0])
        state[0, 0] = (84.0 - center) / scale
        coupling = torch.zeros(1, met.n_coupling)
        emb = torch.zeros(1, EMBEDDING_DIM)
        tf = compute_time_features(torch.tensor([8.0 * 60.0]))
        with torch.no_grad():
            off = met.fluxes(state, coupling, torch.tensor([[0.0, 1.0, 0.0]]), emb, tf)
            on = met.fluxes(state, coupling, torch.tensor([[0.0, 1.0, 1.0]]), emb, tf)
            mg = float(met.constants(emb)["mg_dl_per_g"])
        self.assertEqual(float(off["urinary"]), 0.0)
        self.assertAlmostEqual(float(on["urinary"]), urinary_rate(84.0, mg, 1.0), places=5)

    def test_a_fed_day_does_not_hit_the_free_living_collection(self) -> None:
        # Breakfast, lunch, dinner, asleep from 23:00. The collection in the
        # paper is 42 ± 5 g/day in people eating their own food. This day
        # loses about 104 g, because the clamp regression stays steep at
        # the glucose this model actually holds. The line is not retuned.
        p = resolve_derived_params(PatientParams())
        n = 1440
        start = 6.0
        meals = [((h - start) * 60.0, c, f, pr) for h, c, f, pr in
                 [(8.0, 50, 12, 20), (13.0, 65, 22, 28), (19.0, 75, 28, 35)]]
        sw = np.ones(n)
        sw[1020:] = 0.0
        act = np.zeros(n)
        mg = p.mg_dl_per_g()

        def grams(exposure: float) -> float:
            tr, _ = simulate_full_body(
                p, meals, sw, act, n, start_hour=start, noise_scale=0.0,
                sglt2=np.full(n, exposure),
            )
            g = tr[:, MARKER_INDEX["glucose"]]
            return sum(urinary_rate(float(x), mg, exposure) / mg for x in g)

        self.assertLess(grams(0.0), 1e-6)
        self.assertGreater(grams(1.0), EMPA_UGE_G_PER_DAY + 3.0 * EMPA_UGE_SE_G_PER_DAY)


if __name__ == "__main__":
    unittest.main()
