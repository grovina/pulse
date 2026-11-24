"""A fuller stomach empties faster, and the 75 g drink Brener measured does not move.

The rate is ``EMPTY_KCAL_PER_MIN * clamp(held_kcal / LOAD_REF_KCAL, 1, LOAD_CAP) ** LOAD_EXPONENT``
in the teacher (``step_grams``), the student's pool (``simulate``, ``simulate_by_meal``)
and what ``incretin`` derives from the rate.
"""

from __future__ import annotations

import math
import unittest

import torch

from pulse import incretin
from pulse.gastric import (
    EMPTY_KCAL_PER_MIN, LOAD_CAP, LOAD_EXPONENT, LOAD_REF_KCAL, delivered_kcal,
    load_factor, load_factor_scalar, outflow_grams, simulate, simulate_by_meal, step_grams,
)

_CPU = torch.device("cpu")


def _kcal(grams) -> float:
    return 4.0 * float(grams[0]) + 4.0 * float(grams[1]) + 9.0 * float(grams[2])


def _swallows(*meals: tuple[int, float, float, float], steps: int = 600) -> torch.Tensor:
    swallow = torch.zeros(steps, 3)
    for minute, carb, protein, fat in meals:
        swallow[minute] += torch.tensor([carb, protein, fat])
    return swallow


class TestTheLaw(unittest.TestCase):
    def test_constants(self) -> None:
        self.assertEqual(LOAD_REF_KCAL, 300.0)
        self.assertEqual(LOAD_EXPONENT, 0.33)
        self.assertGreater(LOAD_CAP, 1.0)

    def test_the_75_g_drink_is_unchanged_at_t0(self) -> None:
        drink = torch.tensor([75.0, 0.0, 0.0])
        self.assertAlmostEqual(load_factor_scalar(300.0), 1.0)
        out = outflow_grams(drink, EMPTY_KCAL_PER_MIN)
        self.assertAlmostEqual(_kcal(out), EMPTY_KCAL_PER_MIN, places=5)
        self.assertEqual(step_grams(75.0, 0.0, 0.0), (EMPTY_KCAL_PER_MIN / 4.0, 0.0, 0.0))
        # Anything under 300 kcal is also at the published rate.
        self.assertAlmostEqual(load_factor_scalar(100.0), 1.0)

    def test_a_900_kcal_meal_empties_faster(self) -> None:
        meal = torch.tensor([90.0, 45.0, 40.0])                # 360 + 180 + 360 = 900 kcal
        self.assertAlmostEqual(_kcal(meal), 900.0)
        want = EMPTY_KCAL_PER_MIN * 3.0 ** LOAD_EXPONENT
        self.assertAlmostEqual(_kcal(outflow_grams(meal, EMPTY_KCAL_PER_MIN)), want, places=4)
        self.assertGreater(want, 1.4 * EMPTY_KCAL_PER_MIN)
        c, p, f = step_grams(90.0, 45.0, 40.0)
        self.assertAlmostEqual(4 * c + 4 * p + 9 * f, want, places=4)

    def test_the_cap_binds(self) -> None:
        self.assertAlmostEqual(load_factor_scalar(1e6), LOAD_CAP ** LOAD_EXPONENT)
        self.assertAlmostEqual(float(load_factor(torch.tensor(1e6))), LOAD_CAP ** LOAD_EXPONENT, places=5)

    def test_what_is_swallowed_is_what_leaves(self) -> None:
        out, _ = simulate(_swallows((0, 90.0, 45.0, 40.0), (180, 60.0, 20.0, 20.0)), EMPTY_KCAL_PER_MIN)
        self.assertAlmostEqual(float(out[:, 0].sum()), 150.0, places=2)
        self.assertAlmostEqual(float(out[:, 2].sum()), 60.0, places=2)


class TestTeacherAndStudentAgree(unittest.TestCase):
    def test_the_two_meal_day_matches_minute_by_minute(self) -> None:
        meals = ((0, 90.0, 45.0, 40.0), (150, 60.0, 30.0, 25.0))
        swallow = _swallows(*meals, steps=500)
        out, _ = simulate(swallow, EMPTY_KCAL_PER_MIN)
        pool = [0.0, 0.0, 0.0]
        for t in range(500):
            for minute, c, p, f in meals:
                if minute == t:
                    pool = [pool[0] + c, pool[1] + p, pool[2] + f]
            got = step_grams(*pool)
            torch.testing.assert_close(out[t], torch.tensor(got), atol=1e-4, rtol=1e-4)
            pool = [a - b for a, b in zip(pool, got)]

    def test_the_by_meal_pool_sums_to_the_shared_one(self) -> None:
        a, b = (0, 90.0, 45.0, 40.0), (150, 60.0, 30.0, 25.0)
        shared, _ = simulate(_swallows(a, b), EMPTY_KCAL_PER_MIN)
        split = torch.zeros(1, 2, 600, 3)
        split[0, 0] = _swallows(a)
        split[0, 1] = _swallows(b)
        out, _ = simulate_by_meal(split, EMPTY_KCAL_PER_MIN, torch.ones(1, 2, 3))
        torch.testing.assert_close(out.sum(dim=1)[0], shared, atol=1e-4, rtol=1e-4)

    def test_the_second_meal_is_absorbed_sooner(self) -> None:
        a, b = (0, 90.0, 45.0, 40.0), (150, 60.0, 30.0, 25.0)
        out, _ = simulate(_swallows(a, b), EMPTY_KCAL_PER_MIN)

        # The first meal's leftovers raise the load, so the second meal's calories leave
        # at a higher rate than the constant one it would have had.
        self.assertGreater(float(out[150:].sum()), 0.0)
        const = EMPTY_KCAL_PER_MIN
        left_by_400 = float((out[150:400] * torch.tensor([4.0, 4.0, 9.0])).sum())
        self.assertGreater(left_by_400, 250.0 * const - 1e-3)
        # Same total calories, delivered in fewer minutes than a constant 2.13 would take.
        total = _kcal((150.0, 75.0, 65.0))
        done = int((((out * torch.tensor([4.0, 4.0, 9.0])).sum(-1).cumsum(0)) >= total - 1.0).nonzero()[0])
        self.assertLess(done, total / const)


class TestIncretinFollows(unittest.TestCase):
    def test_the_breakfast_delivers_more_in_four_hours(self) -> None:
        constant = min(1.0, 240.0 * EMPTY_KCAL_PER_MIN / 820.0)
        self.assertGreater(incretin._DELIVERED, constant)
        self.assertAlmostEqual(incretin._DELIVERED, delivered_kcal(820.0, 240) / 820.0)
        # The weight still reproduces Vollmer's ratio for what is delivered.
        stim = incretin._CARB_DELIVERED + incretin.GIP_FAT_WEIGHT * incretin._FAT_DELIVERED
        self.assertAlmostEqual(stim / incretin._OGTT_G, incretin.VOLLMER_RATIO, places=6)

    def test_the_75_g_references_are_the_drink(self) -> None:
        self.assertAlmostEqual(
            incretin.SALEHI_GLP1_EXCESS,
            (incretin._GLP1_MEAL_GAIN / incretin._K_GLP1) * (EMPTY_KCAL_PER_MIN / 4.0)
            * incretin.MG_DL_PER_G, places=6)
        self.assertTrue(math.isfinite(incretin.GIP_GAIN))


if __name__ == "__main__":
    unittest.main()
