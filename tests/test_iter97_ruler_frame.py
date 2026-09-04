"""Iter 97 (review 1.3 / 1.4): the calibration forward map IS the scored forward map.

Through iter 96 the gate's calibration re-integrated every observation window
that started after t=0 from the t=0 state, with meals passed at absolute episode
times while the gut precompute reads ``meal.time`` as a window offset. Measured
on the iter-96 artifact: window 2 of legacy user-01 absorbed its meals at
270/390/510 instead of 90/210/330 and the fitted prediction differed from the
scored one by up to 30 mg/dL; cgm_real mean 5.9 mg/dL (max 15) / 1.6 bpm.

These tests pin the fix: calibration integrates ONCE, continuously, from t=0
in the episode frame, and what it reads at each check-in time equals what the
scorer's full-episode integration produces there.
"""

from __future__ import annotations

import inspect
import unittest

import torch

from pulse.benchmark import (
    MeasurementPoint,
    _calibration_forward,
    _calibration_loss,
    active_meals,
    calibrate_embedding,
)
from pulse.model import ModularPhysiologyNetwork, integrate
from pulse.modules.gut import MEAL_ACTIVE_WINDOW_MIN, MealEvent
from pulse.types import MARKER_INDEX, NORM_CENTER, NORM_SCALE


class TestCalibrationFrame(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = ModularPhysiologyNetwork()
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.init = torch.tensor(NORM_CENTER, dtype=torch.float32)
        self.emb = torch.randn(self.model.embedding_dim) * 0.1
        self.duration = 600
        # Meals at episode minutes 90/210/330 (the legacy protocol), a start
        # time that is NOT 06:00 so an absolute-clock slip would show, and
        # observations that straddle several old 240-min windows.
        self.meals = [
            MealEvent(time=90.0, carbs=60.0, fats=15.0, proteins=20.0),
            MealEvent(time=210.0, carbs=45.0, fats=10.0, proteins=15.0),
            MealEvent(time=330.0, carbs=70.0, fats=20.0, proteins=25.0),
        ]
        self.t0 = 8.5 * 60.0
        self.obs = [
            MeasurementPoint(time=t, marker_id=m, value=v)
            for t, m, v in (
                (30, "glucose", 90.0), (240, "glucose", 110.0), (270, "hr", 68.0),
                (360, "glucose", 120.0), (420, "glucose", 95.0), (480, "hr", 64.0),
            )
        ]
        g = torch.Generator().manual_seed(1)
        self.sw = (torch.rand(self.duration, generator=g) > 0.3).float()
        self.act = torch.rand(self.duration, generator=g) * 0.3

    def _scored(self) -> torch.Tensor:
        with torch.no_grad():
            return integrate(
                model=self.model, initial_state=self.init, embedding=self.emb,
                n_steps=self.duration, dt=1.0, start_time_minutes=self.t0,
                meals=self.meals, sleep_wake=self.sw, activity=self.act,
            )

    def test_calibrated_forward_equals_scored_forward_at_check_ins(self) -> None:
        scored = self._scored()
        with torch.no_grad():
            pred, valid = _calibration_forward(
                self.emb, model=self.model, observations=self.obs,
                initial_state=self.init, meals=self.meals,
                start_time_minutes=self.t0, sleep_wake=self.sw, activity=self.act,
            )
        self.assertEqual(len(valid), len(self.obs))
        self.assertEqual(pred.shape[0], max(o.time for o in self.obs) + 1)
        for o in self.obs:
            idx = MARKER_INDEX[o.marker_id]
            self.assertAlmostEqual(
                float(pred[o.time, idx]), float(scored[o.time, idx]), places=4,
                msg=f"{o.marker_id}@{o.time}: calibration reads a different trajectory than the scorer",
            )

    def test_loss_is_the_scored_residual(self) -> None:
        scored = self._scored()
        norm_scale = torch.tensor(NORM_SCALE, dtype=torch.float32)
        expected = 0.0
        for o in self.obs:
            idx = MARKER_INDEX[o.marker_id]
            expected += ((float(scored[o.time, idx]) - o.value) / float(norm_scale[idx])) ** 2
        expected /= len(self.obs)
        with torch.no_grad():
            loss, n = _calibration_loss(
                self.emb, model=self.model, observations=self.obs,
                initial_state=self.init, meals=self.meals,
                start_time_minutes=self.t0, sleep_wake=self.sw, activity=self.act,
                l2_weight=0.0, norm_scale=norm_scale,
            )
        self.assertEqual(n, len(self.obs))
        self.assertAlmostEqual(float(loss), expected, places=4)

    def test_no_windows_left_in_calibration(self) -> None:
        # The forward map must not take a window size or a per-window meal
        # lookback; those were the frame bug.
        params = inspect.signature(_calibration_forward).parameters
        self.assertNotIn("window_size", params)
        self.assertNotIn("obs_windows", params)
        src = inspect.getsource(_calibration_forward)
        self.assertNotIn("120", src)

    def test_meal_lookback_is_the_gut_active_window(self) -> None:
        meals = [MealEvent(time=t, carbs=50.0, fats=0.0, proteins=0.0)
                 for t in (-MEAL_ACTIVE_WINDOW_MIN - 1.0, -MEAL_ACTIVE_WINDOW_MIN + 1.0, -100.0, 50.0, 700.0)]
        kept = active_meals(meals, 0.0, 600.0)
        self.assertEqual([m.time for m in kept], [-MEAL_ACTIVE_WINDOW_MIN + 1.0, -100.0, 50.0])

    def test_calibrate_embedding_runs_on_the_continuous_map(self) -> None:
        # Smoke: a few Adam steps, embedding moves, window_size accepted but inert.
        res_a = calibrate_embedding(
            model=self.model, observations=self.obs, initial_state=self.init,
            meals=self.meals, duration_min=self.duration, start_time_minutes=self.t0,
            n_steps=3, lr=0.05, l2_weight=0.0, sleep_wake=self.sw, activity=self.act,
            window_size=240,
        )
        res_b = calibrate_embedding(
            model=self.model, observations=self.obs, initial_state=self.init,
            meals=self.meals, duration_min=self.duration, start_time_minutes=self.t0,
            n_steps=3, lr=0.05, l2_weight=0.0, sleep_wake=self.sw, activity=self.act,
            window_size=60,
        )
        self.assertTrue(torch.allclose(res_a.embedding, res_b.embedding))
        self.assertGreater(float(res_a.embedding.norm()), 0.0)


if __name__ == "__main__":
    unittest.main()
