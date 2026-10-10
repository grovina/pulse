"""The epoch log can say what fraction of the step the literature received."""

from __future__ import annotations

import unittest

from pulse.training.step_share import family_of, step_share


class TestLiteratureStepShare(unittest.TestCase):
    def test_families(self) -> None:
        self.assertEqual(family_of("cohort_statistic"), "literature")
        self.assertEqual(family_of("physiology_rules"), "literature")
        self.assertEqual(family_of("dose_response"), "literature")
        self.assertEqual(family_of("trajectory_rollout"), "tape")
        self.assertEqual(family_of("cold_model_distillation"), "distill")
        self.assertEqual(family_of("population_prior"), "other")

    def test_shares_sum_to_one(self) -> None:
        share = step_share(
            aux_grad_norms={
                "cohort_statistic": [1.0, 1.0],
                "physiology_rules": [1.0],
                "dose_response": [1.0],
                "cold_model_distillation": [2.0],
                "embedding_prior": [4.0],
            },
            traj_grad_norms=[3.0, 3.0],
        )
        self.assertEqual(share["literature"]["steps"], 4.0)
        self.assertEqual(share["tape"]["steps"], 2.0)
        self.assertEqual(share["distill"]["steps"], 1.0)
        self.assertEqual(share["other"]["steps"], 1.0)
        self.assertAlmostEqual(share["literature"]["norm_share"], 4.0 / 16.0)
        self.assertAlmostEqual(share["tape"]["norm_share"], 6.0 / 16.0)
        self.assertAlmostEqual(
            sum(share[fam]["norm_share"] for fam in share), 1.0,
        )

    def test_an_empty_epoch_is_zero(self) -> None:
        share = step_share(aux_grad_norms={}, traj_grad_norms=[])
        self.assertTrue(all(share[fam]["norm_share"] == 0.0 for fam in share))
        self.assertTrue(all(share[fam]["steps"] == 0.0 for fam in share))
