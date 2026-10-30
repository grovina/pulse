"""Intestinal bile and the fat that leaves the stomach.

Borgström, Barrowman and Lindström, New Comprehensive Biochemistry 1985,
chapter 14: without bile, 60–70% of a normal fat load is still absorbed.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from pulse.bile import ABSORPTION, BILE_FAT_FLOOR, INT_PRESENT_MMOL, fat_bile_factor, fat_bile_factor_t
from pulse.knowledge.full_body import PatientParams, simulate_full_body
from pulse.model import ModularPhysiologyNetwork, integrate
from pulse.published import initial_values
from pulse.types import EMBEDDING_DIM, MARKER_INDEX as MI, NORM_CENTER, SERIES_INDEX


class TestTheFloor(unittest.TestCase):
    def test_an_empty_intestine_absorbs_the_published_share(self) -> None:
        self.assertEqual(fat_bile_factor(0.0), BILE_FAT_FLOOR)
        self.assertEqual(BILE_FAT_FLOOR, 0.65)

    def test_the_resting_content_absorbs_the_whole_outflow(self) -> None:
        self.assertEqual(fat_bile_factor(INT_PRESENT_MMOL), 1.0)
        self.assertEqual(fat_bile_factor(INT_PRESENT_MMOL * 2.0), 1.0)

    def test_the_tensor_is_the_scalar(self) -> None:
        x = torch.tensor([0.0, 0.4, 1.0, 2.3])
        got = fat_bile_factor_t(x, initial_values(ABSORPTION))
        exp = torch.tensor([fat_bile_factor(float(v)) for v in x])
        torch.testing.assert_close(got, exp)


class TestAMealHasBile(unittest.TestCase):
    def test_a_fatty_meal_absorbs_what_was_swallowed(self) -> None:
        """The intestine does not empty during a meal, so the floor is not
        what a normal plate loses."""
        p = PatientParams()
        n = 360
        fat = 40.0
        meals = [(60.0, 20.0, fat, 15.0)]
        tr, prof = simulate_full_body(
            p, meals, np.ones(n), np.zeros(n), n, start_hour=8, noise_scale=0.0,
        )
        self.assertGreaterEqual(float(tr[:, MI["intestinal_bile"]].min()), INT_PRESENT_MMOL - 0.02)
        absorbed = float(prof[:, 1].sum()) / 3.0
        self.assertAlmostEqual(absorbed, fat, delta=0.02 * fat)


class TestTheFistula(unittest.TestCase):
    def test_draining_the_intestine_reads_the_floor(self) -> None:
        from pulse.knowledge.cohort_statistics import (
            ALL_COHORT_STATISTICS, SEALED_COHORT_STATISTICS,
        )
        names = {s.name for s in (*ALL_COHORT_STATISTICS, *SEALED_COHORT_STATISTICS)}
        self.assertIn("borgstrom_1985_fat_without_bile", names)

        torch.manual_seed(0)
        model = ModularPhysiologyNetwork()
        state = torch.tensor(NORM_CENTER, dtype=torch.float32).unsqueeze(0)
        emb = torch.zeros(1, EMBEDDING_DIM)
        intact = integrate(model, state, emb, 1, observe=True, meals=[])
        drained = integrate(
            model, state, emb, 1, observe=True, meals=[], bile_fistula=torch.ones(1),
        )
        col = SERIES_INDEX["fat_absorption"]
        self.assertAlmostEqual(float(intact[0, 0, col].detach()), 1.0, places=2)
        self.assertAlmostEqual(float(drained[0, 0, col].detach()), BILE_FAT_FLOOR, places=2)


if __name__ == "__main__":
    unittest.main()
