"""The stomach is one compartment in the teacher and the student.

Brener, Hendrix and McHugh, Gastroenterology 1983: glucose solutions left
the stomach at 2.13 kcal/min. Carbohydrate, protein and fat share that
rate. What leaves is what was swallowed.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from pulse.gastric import EMPTY_KCAL_PER_MIN, simulate, step_grams, swallow_from_meals
from pulse.knowledge.full_body import (
    PatientParams, compute_absorption_profile, duodenal_delivery, simulate_full_body,
)
from pulse.model import ModularPhysiologyNetwork, integrate
from pulse.modules.gut import MealEvent
from pulse.types import MARKER_INDEX, NORM_CENTER


class TestTheCompartment(unittest.TestCase):
    def test_a_meal_leaves_in_full_and_fat_slows_the_carbohydrate(self) -> None:
        carb = torch.zeros(400, 3)
        carb[0, 0] = 50.0
        mixed = carb.clone()
        mixed[0, 2] = 30.0
        out_c, _ = simulate(carb, EMPTY_KCAL_PER_MIN)
        out_m, _ = simulate(mixed, EMPTY_KCAL_PER_MIN)
        self.assertAlmostEqual(float(out_c[:, 0].sum()), 50.0, places=3)
        self.assertAlmostEqual(float(out_m[:, 0].sum()), 50.0, places=3)
        self.assertAlmostEqual(float(out_m[:, 2].sum()), 30.0, places=3)
        self.assertLess(float(out_m[0, 0]), 0.5 * float(out_c[0, 0]))
        self.assertGreater(int((out_m[:, 0] > 1e-4).sum()), int((out_c[:, 0] > 1e-4).sum()))

    def test_teacher_appearance_is_that_outflow(self) -> None:
        meals = [(0.0, 50.0, 30.0, 0.0)]
        p = PatientParams()
        d_carb, _d_protein, d_fat = step_grams(50.0, 0.0, 30.0)
        profile = compute_absorption_profile(0.0, meals, p)
        duo = duodenal_delivery(0.0, meals, p)
        self.assertAlmostEqual(profile[0], d_carb * p.mg_dl_per_g(), places=4)
        self.assertAlmostEqual(duo["carb"], d_carb, places=6)
        self.assertAlmostEqual(duo["fat"], d_fat, places=6)
        # 50 g is 200 kcal. With 30 g of fat (270 kcal) it takes longer than
        # the carbohydrate alone, and both are gone inside a few hours.
        alone = sum(compute_absorption_profile(float(t), [(0.0, 50.0, 0.0, 0.0)], p)[0]
                    for t in range(200))
        with_fat = sum(compute_absorption_profile(float(t), meals, p)[0] for t in range(400))
        self.assertAlmostEqual(alone / p.mg_dl_per_g(), 50.0, places=2)
        self.assertAlmostEqual(with_fat / p.mg_dl_per_g(), 50.0, places=2)

    def test_the_student_pool_is_the_same_update(self) -> None:
        m = ModularPhysiologyNetwork()
        m.eval()
        meals = [MealEvent(time=0.0, carbs=40.0, fats=10.0, proteins=15.0)]
        n = 60
        with torch.no_grad():
            traj = integrate(
                m, torch.tensor(NORM_CENTER), torch.zeros(m.embedding_dim), n,
                meals=meals, sleep_wake=torch.ones(n), activity=torch.zeros(n),
            )
            swallow = swallow_from_meals(n, meals, torch.device("cpu"))
            left, filled = simulate(swallow, m.gastric.empty_kcal())
        remain = filled - left
        for name, col in (("gastric_carb", 0), ("gastric_protein", 1), ("gastric_fat", 2)):
            got = traj[1:, MARKER_INDEX[name]]
            torch.testing.assert_close(got, remain[:-1, col], atol=1e-4, rtol=1e-4)

    def test_the_emptying_rate_is_a_parameter_a_meal_can_move(self) -> None:
        m = ModularPhysiologyNetwork()
        m.train()
        meals = [MealEvent(time=10.0, carbs=75.0, fats=0.0, proteins=0.0)]
        n = 40
        traj = integrate(
            m, torch.tensor(NORM_CENTER), torch.zeros(m.embedding_dim), n,
            meals=meals, sleep_wake=torch.ones(n), activity=torch.zeros(n),
        )
        traj[:, MARKER_INDEX["gastric_carb"]].sum().backward()
        grad = m.gastric.log_empty_kcal.grad
        self.assertIsNotNone(grad)
        self.assertGreater(float(grad.abs()), 0.0)

    def test_teacher_trajectory_records_the_pools(self) -> None:
        n = 30
        traj, _ = simulate_full_body(
            PatientParams(), [(0.0, 40.0, 10.0, 15.0)],
            np.ones(n), np.zeros(n), n, noise_scale=0.0, rng=np.random.default_rng(0),
        )
        self.assertAlmostEqual(float(traj[0, MARKER_INDEX["gastric_carb"]]), 40.0 - step_grams(40, 15, 10)[0], places=3)
        self.assertGreater(float(traj[0, MARKER_INDEX["gastric_fat"]]), 9.0)


if __name__ == "__main__":
    unittest.main()
