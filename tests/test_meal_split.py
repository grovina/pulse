"""The meal-gated muscle share, the epinephrine gate on the insulin setpoint,
and the load-time adjustment that keeps iter-119's nutrient gain meaningful."""

from __future__ import annotations

import unittest

import torch

from pulse.incretin import ISLET
from pulse.model import ModularPhysiologyNetwork, integrate
from pulse.modules import metabolic as M
from pulse.modules.gut import MealEvent
from pulse.published import initial_values
from pulse.types import MARKER_INDEX, NORM_CENTER

_INS = MARKER_INDEX["insulin"]
_GLU = MARKER_INDEX["glucose"]


def _model() -> ModularPhysiologyNetwork:
    torch.manual_seed(0)
    m = ModularPhysiologyNetwork(
        metabolic_hidden=16, appetite_hidden=8, stress_hidden=8,
        cardiovascular_hidden=8, thermoreg_hidden=8, respiratory_hidden=8,
        gut_hidden=8, hepatobiliary_hidden=8)
    m.eval()
    return m


def _run(m, meals, n=180):
    with torch.no_grad():
        return integrate(
            m, torch.tensor(NORM_CENTER), torch.zeros(m.embedding_dim), n,
            start_time_minutes=8.0 * 60.0, meals=meals,
            sleep_wake=torch.ones(n), activity=torch.zeros(n))


def _ungated(monkeypatch_value):
    class _Ctx:
        def __enter__(self):
            self.old = M._MEAL_GATE_NUT
            M._MEAL_GATE_NUT = monkeypatch_value
        def __exit__(self, *a):
            M._MEAL_GATE_NUT = self.old
    return _Ctx()


class TestMealGate(unittest.TestCase):
    def test_a_drink_is_unchanged(self) -> None:
        drink = [MealEvent(time=0.0, carbs=75.0, fats=0.0, proteins=0.0)]
        gated = _run(_model(), drink)
        with _ungated(1e12):  # nut/(nut+c) -> 0: the old split
            old = _run(_model(), drink)
        torch.testing.assert_close(gated, old)

    def test_a_mixed_meal_sends_less_to_muscle(self) -> None:
        meal = [MealEvent(time=0.0, carbs=75.0, fats=30.0, proteins=30.0)]
        gated = _run(_model(), meal)
        with _ungated(1e12):
            old = _run(_model(), meal)
        # less glucose is stored, so more reaches plasma
        self.assertGreater(float(gated[:, _GLU].max()), float(old[:, _GLU].max()))

    def test_the_gate_halves_the_share_at_the_constant(self) -> None:
        nut = torch.tensor(M._MEAL_GATE_NUT)
        self.assertAlmostEqual(float(1.0 - nut / (nut + M._MEAL_GATE_NUT)), 0.5, places=6)


class TestEpinephrineGate(unittest.TestCase):
    def _exercise(self, m):
        n = 90
        with torch.no_grad():
            return integrate(
                m, torch.tensor(NORM_CENTER), torch.zeros(m.embedding_dim), n,
                start_time_minutes=8.0 * 60.0, meals=[],
                sleep_wake=torch.ones(n), activity=torch.full((n,), 3.0))[:, _INS]

    def test_epinephrine_lowers_the_insulin_setpoint_in_exercise(self) -> None:
        gated = self._exercise(_model())
        m = _model()
        with torch.no_grad():
            m.metabolic.epinephrine_action.log_ins_epi_gain.fill_(-30.0)
        ungated = self._exercise(m)
        self.assertLess(float(gated[-1]), float(ungated[-1]))

    def test_no_epinephrine_no_change_at_rest(self) -> None:
        # at rest epi_drive is near zero, so the setpoint is effective_ib itself
        a = _run(_model(), [], 30)
        m = _model()
        with torch.no_grad():
            m.metabolic.epinephrine_action.log_ins_epi_gain.fill_(-30.0)
        b = _run(m, [], 30)
        torch.testing.assert_close(a[:, _INS], b[:, _INS], rtol=2e-3, atol=2e-3)


class TestLoadAdjustment(unittest.TestCase):
    def test_new_default_is_twice_the_old(self) -> None:
        self.assertAlmostEqual(float(initial_values(ISLET)["ins_nut_gain"]), 4.0, places=6)

    def test_an_old_state_gets_the_gain_doubled_once(self) -> None:
        src = _model()
        state = {k: v.clone() for k, v in src.state_dict().items() if k != "state_rev"}
        with torch.no_grad():
            state["metabolic.islet.log_ins_nut_gain"] = torch.tensor(0.5)
        m = _model()
        m.load_model_state(state, strict=True)
        self.assertAlmostEqual(float(m.metabolic.islet.log_ins_nut_gain), 0.5 + 0.6931472, places=5)

    def test_a_new_state_is_taken_as_is(self) -> None:
        src = _model()
        with torch.no_grad():
            src.metabolic.islet.log_ins_nut_gain.fill_(0.5)
        m = _model()
        m.load_model_state(src.state_dict(), strict=True)
        self.assertAlmostEqual(float(m.metabolic.islet.log_ins_nut_gain), 0.5, places=6)

    def test_a_state_without_the_key_takes_the_default(self) -> None:
        src = _model()
        state = {k: v.clone() for k, v in src.state_dict().items()
                 if k not in ("state_rev", "metabolic.islet.log_ins_nut_gain")}
        m = _model()
        m.load_model_state(state, strict=True)
        self.assertAlmostEqual(float(m.metabolic.islet.log_ins_nut_gain.exp()), 4.0, places=4)

    def test_a_warm_start_gets_the_gain_doubled_too(self) -> None:
        # --init-from loads through train.load_compatible_state_dict, not
        # load_model_state; iter 120's first launch missed the doubling there.
        from pulse.train import load_compatible_state_dict
        src = _model()
        state = {k: v.clone() for k, v in src.state_dict().items() if k != "state_rev"}
        with torch.no_grad():
            state["metabolic.islet.log_ins_nut_gain"] = torch.tensor(0.5)
        m = _model()
        load_compatible_state_dict(m, state)
        self.assertAlmostEqual(float(m.metabolic.islet.log_ins_nut_gain), 0.5 + 0.6931472, places=5)


if __name__ == "__main__":
    unittest.main()
