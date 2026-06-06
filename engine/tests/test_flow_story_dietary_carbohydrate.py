"""Regression: dietary carbohydrate flow story passes on cold teacher."""

from __future__ import annotations

import unittest

import numpy as np

from pulse.knowledge.textbook_scenarios.flow_stories import (
    dietary_carbohydrate_meal_flow_scenario,
)
from pulse.types import GUT_OUTPUT_DIM, STATE_DIM

try:
    import torch

    from pulse.knowledge.textbook_scenarios.neural_rollout import (
        make_dietary_carbohydrate_neural_trajectory_fn,
    )
    from pulse.model import ModularPhysiologyNetwork

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class TestDietaryCarbohydrateFlowStory(unittest.TestCase):
    def test_cold_model_passes_all_checks(self) -> None:
        rng = np.random.default_rng(0)
        result = dietary_carbohydrate_meal_flow_scenario(rng, trajectory_fn=None)
        self.assertEqual(result.name, "dietary_carbohydrate_meal_flow")
        self.assertGreaterEqual(result.pass_rate, 1.0)
        for c in result.checks:
            self.assertTrue(c.passed, msg=f"{c.name}: {c.description}")


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestDietaryCarbohydrateFlowStoryNeuralTrajectoryFn(unittest.TestCase):
    def test_trajectory_fn_runs_and_matches_teacher_gut(self) -> None:
        rng = np.random.default_rng(1)
        model = ModularPhysiologyNetwork()
        embedding = torch.zeros(model.embedding_dim, dtype=torch.float32)
        traj_fn = make_dietary_carbohydrate_neural_trajectory_fn(model, embedding)
        traj, absorption = traj_fn(rng)
        self.assertEqual(traj.shape[1], STATE_DIM)
        self.assertEqual(absorption.shape[1], GUT_OUTPUT_DIM)
        result = dietary_carbohydrate_meal_flow_scenario(rng, trajectory_fn=traj_fn)
        self.assertIn("neural trajectory", result.source)
        self.assertGreater(result.pass_rate, 0.0)
        self.assertLessEqual(result.pass_rate, 1.0)


if __name__ == "__main__":
    unittest.main()
