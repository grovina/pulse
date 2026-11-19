"""Insulin secretion driven by absorbed protein and fat, and glucagon by protein.

A mixed meal barely raises glucose, so secretion on glucose above basal leaves
its insulin short. The islet adds ``ins_nut_gain * incretin * (w_p * protein +
w_f * fat)`` leaving the stomach, and ``gn_nut_gain * protein`` glucagon. It
replaces the old protein-only ``amino_ins_gain`` term.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import torch

from pulse.incretin import ISLET, nutrient_glucagon_rate_t, nutrient_insulin_rate_t
from pulse.model import ModularPhysiologyNetwork, integrate
from pulse.modules.gut import MealEvent
from pulse.published import initial_values
from pulse.types import MARKER_INDEX, NORM_CENTER

_INS = MARKER_INDEX["insulin"]
_GN = MARKER_INDEX["glucagon"]
_OLD_CHECKPOINT = Path(os.environ.get("PULSE_OLD_CHECKPOINT", "/nonexistent"))


def _model() -> ModularPhysiologyNetwork:
    torch.manual_seed(0)
    m = ModularPhysiologyNetwork(
        metabolic_hidden=16, appetite_hidden=8, stress_hidden=8,
        cardiovascular_hidden=8, thermoreg_hidden=8, respiratory_hidden=8,
        gut_hidden=8, hepatobiliary_hidden=8,
    )
    m.eval()
    return m


def _run(m, meals, n=120):
    with torch.no_grad():
        return integrate(
            m, torch.tensor(NORM_CENTER), torch.zeros(m.embedding_dim), n,
            start_time_minutes=8.0 * 60.0, meals=meals,
            sleep_wake=torch.ones(n), activity=torch.zeros(n),
        )


def _without_nutrient_term(m):
    with torch.no_grad():
        m.metabolic.islet.log_ins_nut_gain.fill_(-30.0)
        m.metabolic.islet.log_gn_nut_gain.fill_(-30.0)
    return m


class TestTheTerm(unittest.TestCase):
    def test_zero_without_a_meal(self) -> None:
        z, one = torch.tensor(0.0), torch.tensor(1.7)
        self.assertEqual(float(nutrient_insulin_rate_t(z, z, one, initial_values(ISLET))), 0.0)
        self.assertEqual(float(nutrient_glucagon_rate_t(z, initial_values(ISLET))), 0.0)
        a = _run(_model(), [])
        b = _run(_without_nutrient_term(_model()), [])
        torch.testing.assert_close(a, b)

    def test_rises_with_protein(self) -> None:
        islet = initial_values(ISLET)
        one = torch.tensor(1.0)
        r = [float(nutrient_insulin_rate_t(torch.tensor(p), torch.tensor(0.0), one, islet))
             for p in (0.0, 0.1, 0.2)]
        self.assertTrue(r[0] < r[1] < r[2])
        self.assertAlmostEqual(r[1], 2.0 * 2.0 * 0.1, places=5)
        fat = float(nutrient_insulin_rate_t(torch.tensor(0.0), torch.tensor(0.1), one, islet))
        self.assertAlmostEqual(fat, 2.0 * 0.5 * 0.1, places=5)
        # in the model: a protein meal raises insulin and glucagon above the same model without the term
        meal = [MealEvent(time=0.0, carbs=0.0, fats=0.0, proteins=50.0)]
        with_term = _run(_model(), meal)
        without = _run(_without_nutrient_term(_model()), meal)
        self.assertGreater(float((with_term[:, _INS] - without[:, _INS]).max()), 1.0)
        self.assertGreater(float((with_term[:, _GN] - without[:, _GN]).max()), 1.0)

    def test_a_carbohydrate_only_arm_is_unchanged(self) -> None:
        ogtt = [MealEvent(time=0.0, carbs=75.0, fats=0.0, proteins=0.0)]
        torch.testing.assert_close(
            _run(_model(), ogtt)[:, [_INS, _GN]],
            _run(_without_nutrient_term(_model()), ogtt)[:, [_INS, _GN]])


class TestOldCheckpoints(unittest.TestCase):
    def _old_state(self, m):
        state = {k: v.clone() for k, v in m.state_dict().items()
                 if k not in ModularPhysiologyNetwork.ADDED_STATE_KEYS}
        state["metabolic.islet.log_amino_ins_gain"] = torch.tensor(0.05)
        return state

    def test_a_state_without_the_new_keys_and_with_the_retired_one_loads(self) -> None:
        m = _model()
        m2 = _model()
        m2.load_model_state(self._old_state(m), strict=True)
        defaults = initial_values(ISLET)
        for name in ("ins_nut_gain", "ins_nut_w_p", "ins_nut_w_f", "gn_nut_gain"):
            self.assertAlmostEqual(
                float(getattr(m2.metabolic.islet, f"log_{name}").exp()), float(defaults[name]), places=4)
        self.assertFalse(hasattr(m2.metabolic.islet, "log_amino_ins_gain"))

    def test_other_missing_or_extra_keys_still_fail(self) -> None:
        m = _model()
        state = self._old_state(m)
        state["not.a.parameter"] = torch.zeros(1)
        with self.assertRaises(RuntimeError):
            _model().load_model_state(state, strict=True)
        state = self._old_state(m)
        state.pop(next(k for k in state if k.startswith("metabolic.insulin_baseline_net")))
        with self.assertRaises(RuntimeError):
            _model().load_model_state(state, strict=True)

    @unittest.skipUnless(_OLD_CHECKPOINT.exists(), "set PULSE_OLD_CHECKPOINT to an iter-118 checkpoint")
    def test_the_iter118_checkpoint_loads(self) -> None:
        from pulse.diagnostics.probe import load_model_from_checkpoint
        model, _ = load_model_from_checkpoint(_OLD_CHECKPOINT)
        self.assertAlmostEqual(float(model.metabolic.islet.log_ins_nut_gain.exp()), 2.0, places=4)


if __name__ == "__main__":
    unittest.main()
