"""GIP, amino acids on insulin, and GLP-1 on glucagon.

Vollmer et al., Diabetes 2008;57:678: integrated GIP after the 820 kcal
breakfast is 186 ± 17% higher than after 75 g of glucose. Vilsbøll et al.,
Diabetologia 2003;46:183: the two incretins contribute about equally, and
only GLP-1 suppresses glucagon. Salehi et al., JCEM 2008;93:4909, Table:
the exendin contrast is 8.2 pg/mL of glucagon at the oral glucose load.
Westphal et al., AJCN 1990;52:267: 50 g of protein does not move glucose.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from pulse.incretin import (
    GIP_B, GLP1_GN_PG, K_INCRETIN, SALEHI_GLP1_EXCESS, VOLLMER_RATIO, VOLLMER_RATIO_SE,
    amino_insulin_rate, amino_insulin_rate_t, gip_rate, gip_rate_t,
    glp1_glucagon_rate, incretin_factor, incretin_factor_t,
)
from pulse.knowledge.cohorts.nutrition import _WESTPHAL_SIGMA
from pulse.knowledge.full_body import PatientParams, simulate_full_body
from pulse.model import ModularPhysiologyNetwork, integrate
from pulse.modules.gut import MealEvent
from pulse.types import MARKER_INDEX, MODULE_MARKER_IDS, NORM_CENTER


def _run(meals, n, start_hour=8.0):
    return simulate_full_body(
        PatientParams(), meals, np.ones(n), np.zeros(n, dtype=np.float32), n,
        start_hour=start_hour, noise_scale=0.0,
    )[0]


class TestTheFactor(unittest.TestCase):
    def test_basal_is_one_and_equal_excesses_keep_the_old_factor(self) -> None:
        self.assertEqual(incretin_factor(0.0, 0.0), 1.0)
        excess = 16.0
        sat = excess / (excess + K_INCRETIN)
        self.assertAlmostEqual(incretin_factor(excess, excess), 1.0 + 4.0 * sat, places=6)
        got = incretin_factor_t(torch.tensor([0.0, excess]), torch.tensor([0.0, excess]))
        torch.testing.assert_close(
            got, torch.tensor([1.0, 1.0 + 4.0 * sat]), rtol=0, atol=1e-6,
        )

    def test_the_glucagon_hold_is_the_table_and_zero_at_basal(self) -> None:
        k = 0.03
        self.assertEqual(glp1_glucagon_rate(0.0, k), 0.0)
        self.assertAlmostEqual(
            glp1_glucagon_rate(SALEHI_GLP1_EXCESS, k) / k, GLP1_GN_PG, places=5,
        )

    def test_no_protein_adds_nothing(self) -> None:
        self.assertEqual(amino_insulin_rate(0.0), 0.0)
        self.assertEqual(float(amino_insulin_rate_t(torch.tensor(0.0))), 0.0)
        self.assertAlmostEqual(gip_rate(GIP_B, 0.0, 0.0), 0.0, places=6)
        torch.testing.assert_close(
            gip_rate_t(torch.tensor(GIP_B), torch.tensor(0.2), torch.tensor(0.1)),
            torch.tensor(gip_rate(GIP_B, 0.2, 0.1)),
        )


class TestTheBreakfast(unittest.TestCase):
    def test_the_vollmer_ratio_and_the_equal_split(self) -> None:
        p = PatientParams()
        self.assertEqual(p.glp1_meal_gain, 0.7)
        self.assertEqual(p.k_glp1, 0.2)
        ogtt = [(0.0, 75.0, 0.0, 0.0)]
        meal = [(0.0, 90.0, 353.0 / 9.0, 107.0 / 4.0)]
        to, tm = _run(ogtt, 240), _run(meal, 240)

        def iauc(tr):
            y = tr[:, MARKER_INDEX["gip"]]
            return float(np.maximum(y - y[0], 0.0).sum())

        ratio = iauc(tm) / iauc(to)
        self.assertAlmostEqual(ratio, VOLLMER_RATIO, delta=VOLLMER_RATIO_SE)
        gip_ex = np.maximum(tm[:, MARKER_INDEX["gip"]] - GIP_B, 0.0).mean()
        glp_ex = np.maximum(tm[:, MARKER_INDEX["glp1"]] - p.GLP1_b, 0.0).mean()
        self.assertAlmostEqual(gip_ex, glp_ex, places=3)

    def test_protein_moves_insulin_and_not_glucose(self) -> None:
        water = _run([], 300)
        protein = _run([(60.0, 0.0, 0.0, 50.0)], 300)
        d_g = protein[60:300, MARKER_INDEX["glucose"]].mean() - water[60:300, MARKER_INDEX["glucose"]].mean()
        d_i = protein[60:160, MARKER_INDEX["insulin"]].mean() - water[60:160, MARKER_INDEX["insulin"]].mean()
        self.assertLess(abs(d_g), _WESTPHAL_SIGMA)
        self.assertGreater(d_i, 0.5)

    def test_the_student_gip_is_the_same_update(self) -> None:
        self.assertEqual(
            MODULE_MARKER_IDS["appetite"],
            ["ghrelin", "leptin", "glp1", "insulin_slow", "gip"],
        )
        meals = [(0.0, 40.0, 15.0, 10.0)]
        teacher = _run(meals, 90, start_hour=6.0)
        m = ModularPhysiologyNetwork()
        m.eval()
        events = [MealEvent(time=0.0, carbs=40.0, fats=15.0, proteins=10.0)]
        with torch.no_grad():
            student = integrate(
                m, torch.tensor(NORM_CENTER), torch.zeros(m.embedding_dim), 90,
                start_time_minutes=6.0 * 60.0, meals=events,
                sleep_wake=torch.ones(90), activity=torch.zeros(90),
            )
        got = student[1:, MARKER_INDEX["gip"]]
        ref = torch.tensor(teacher[:-1, MARKER_INDEX["gip"]], dtype=got.dtype)
        torch.testing.assert_close(got, ref, atol=1e-3, rtol=1e-3)
