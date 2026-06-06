"""run_all_scenarios trajectory_fn_by_name and validation."""

from __future__ import annotations

import unittest

import numpy as np

from pulse.knowledge.textbook_scenarios import run_all_scenarios


class TestRunAllScenariosTrajectoryFn(unittest.TestCase):
    def test_rejects_both_trajectory_sources(self) -> None:
        def dummy(rng: np.random.Generator):
            raise AssertionError("should not be called")

        with self.assertRaises(ValueError):
            run_all_scenarios(
                trajectory_fn=dummy,
                trajectory_fn_by_name={"x": dummy},
                seed=0,
            )

    def test_trajectory_fn_by_name_only_hits_named_scenario(self) -> None:
        calls: list[str] = []

        def tagged_fn(rng: np.random.Generator):
            calls.append("fn")
            from pulse.knowledge.full_body import PatientParams, simulate_full_body

            params = PatientParams()
            from pulse.knowledge.textbook_scenarios.flow_story_protocol import (
                DIETARY_CARB_FLOW_DURATION_MIN,
                DIETARY_CARB_FLOW_MEALS,
                DIETARY_CARB_FLOW_START_HOUR,
            )

            duration = DIETARY_CARB_FLOW_DURATION_MIN
            sw = np.ones(duration, dtype=np.float32)
            act = np.full(duration, 0.05, dtype=np.float32)
            return simulate_full_body(
                params,
                DIETARY_CARB_FLOW_MEALS,
                sw,
                act,
                duration,
                DIETARY_CARB_FLOW_START_HOUR,
                noise_scale=0.001,
                rng=rng,
            )

        results = run_all_scenarios(
            trajectory_fn_by_name={
                "dietary_carbohydrate_meal_flow_scenario": tagged_fn,
            },
            seed=42,
        )
        self.assertEqual(len(calls), 1)
        dietary = next(r for r in results if r.name == "dietary_carbohydrate_meal_flow")
        self.assertGreaterEqual(dietary.pass_rate, 1.0)
        ogtt = next(r for r in results if r.name == "OGTT")
        self.assertGreaterEqual(ogtt.pass_rate, 1.0)


if __name__ == "__main__":
    unittest.main()
