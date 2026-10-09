"""Window state stays on the window, and a glucose reading is interstitial.

Calibration estimates the embedding together with an initial state spun up
from that person's quasi-steady state and a transient glucose appearance.
Only the embedding can be kept. A shifted start or an unlogged meal moves
the window quantities; the person embedding stays near where it started when
the observations are otherwise that person. A hold-out that improves while
the fitted points regress is still rejected.
"""

from __future__ import annotations

import math
import unittest
from unittest import mock

import torch

from pulse import calibration as cal
from pulse.calibration import (
    CalibrationSettings,
    MeasurementPoint,
    calibrate_embedding,
    quasi_steady_state,
)
from pulse.measurement import interstitial_glucose
from pulse.model import ModularPhysiologyNetwork, integrate
from pulse.modules.gut import MealEvent
from pulse.types import MARKER_INDEX, NORM_CENTER, NORM_SCALE


def _model() -> ModularPhysiologyNetwork:
    torch.manual_seed(0)
    model = ModularPhysiologyNetwork()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _wire_glucose(model: ModularPhysiologyNetwork) -> None:
    """A fresh glucose-baseline head is zero, so two embeddings are the same
    person. Open that head so a test can see the embedding move glucose."""
    net = model.metabolic.glucose_baseline_net
    with torch.no_grad():
        net[-1].weight.zero_()
        net[-1].bias.zero_()
        net[-1].weight[0, 0] = 1.5


def _glucose_obs(model, state, embedding, meals, times, t0) -> list[MeasurementPoint]:
    with torch.no_grad():
        traj = integrate(
            model=model, initial_state=state, embedding=embedding,
            n_steps=max(times) + 1, dt=1.0, start_time_minutes=t0, meals=meals,
        )
        glucose = interstitial_glucose(traj[:, MARKER_INDEX["glucose"]])
    return [MeasurementPoint(time=t, marker_id="glucose", value=float(glucose[t])) for t in times]


class TestInterstitialLag(unittest.TestCase):
    def test_eight_minute_time_constant(self) -> None:
        plasma = torch.full((17,), 100.0)
        plasma[1:] = 130.0
        lagged = interstitial_glucose(plasma, tau_min=8.0)
        self.assertAlmostEqual(float(lagged[0]), 100.0, places=5)
        self.assertAlmostEqual(float(lagged[8]), 130.0 - 30.0 * math.exp(-1.0), places=4)
        self.assertLess(float(lagged[8]), float(plasma[8]))
        self.assertTrue(torch.equal(interstitial_glucose(plasma, tau_min=0.0), plasma))

    def test_only_glucose_observations_are_lagged(self) -> None:
        plasma = torch.full((17,), 100.0)
        plasma[1:] = 130.0
        lagged = interstitial_glucose(plasma, tau_min=8.0)
        predicted = torch.tensor(NORM_CENTER, dtype=torch.float32).unsqueeze(0).repeat(17, 1)
        predicted[:, MARKER_INDEX["glucose"]] = plasma
        predicted[:, MARKER_INDEX["hr"]] = 70.0
        scale = torch.tensor(NORM_SCALE, dtype=torch.float32)
        loss_ig, n = cal._data_terms(
            predicted, [MeasurementPoint(8, "glucose", float(lagged[8]))], [], scale, 1.0, 8.0,
        )
        loss_plasma, _ = cal._data_terms(
            predicted, [MeasurementPoint(8, "glucose", float(plasma[8]))], [], scale, 1.0, 8.0,
        )
        loss_hr, _ = cal._data_terms(
            predicted, [MeasurementPoint(8, "hr", 70.0)], [], scale, 1.0, 8.0,
        )
        self.assertEqual(n, 1)
        self.assertAlmostEqual(float(loss_ig), 0.0, places=4)
        self.assertGreater(float(loss_plasma), 0.1)
        self.assertAlmostEqual(float(loss_hr), 0.0, places=5)

    def test_calibration_scores_glucose_through_the_lag(self) -> None:
        model = _model()
        state = torch.tensor(NORM_CENTER, dtype=torch.float32)
        obs = [
            MeasurementPoint(time=0, marker_id="glucose", value=95.0),
            MeasurementPoint(time=30, marker_id="glucose", value=110.0),
        ]
        seen: dict[str, float] = {}
        real = interstitial_glucose

        def spy(plasma, tau_min=8.0, dt=1.0):
            seen["tau"] = float(tau_min)
            return real(plasma, tau_min, dt)

        with mock.patch("pulse.calibration.interstitial_glucose", spy):
            calibrate_embedding(
                model, obs, state, [], 40, start_time_minutes=360.0,
                settings=CalibrationSettings(max_steps=0, patience=0, spinup_minutes=0),
            )
        self.assertEqual(seen["tau"], 8.0)


class TestQuasiSteady(unittest.TestCase):
    def test_spinup_reaches_the_persons_glucose(self) -> None:
        model = _model()
        _wire_glucose(model)
        direction = model.embedding_projections["metabolic"].weight.detach()[0]
        direction = direction / direction.norm().clamp(min=1e-6)
        nominal = torch.tensor(NORM_CENTER, dtype=torch.float32)
        person = 0.8 * direction
        gi = MARKER_INDEX["glucose"]
        qss_person = quasi_steady_state(
            model, person, nominal, spinup_minutes=120, start_time_minutes=360.0,
        )
        qss_zero = quasi_steady_state(
            model, torch.zeros_like(person), nominal, spinup_minutes=120, start_time_minutes=360.0,
        )
        self.assertLess(float(qss_person[gi]), float(qss_zero[gi]) - 2.0)
        self.assertLess(abs(float(qss_zero[gi]) - float(nominal[gi])), 5.0)


class TestWindowStateIsNotThePerson(unittest.TestCase):
    def setUp(self) -> None:
        self.model = _model()
        _wire_glucose(self.model)
        self.person = torch.zeros(self.model.embedding_dim)
        self.nominal = torch.tensor(NORM_CENTER, dtype=torch.float32)
        self.prior_std = torch.full((self.model.embedding_dim,), 0.15)
        self.t0 = 22.0 * 60.0
        self.gi = MARKER_INDEX["glucose"]

    def _fit(self, obs, meals, duration, **setting_kw) -> cal.CalibrationResult:
        settings = CalibrationSettings(max_steps=20, lr=0.05, patience=8, **setting_kw)
        return calibrate_embedding(
            self.model, obs, self.nominal, meals, duration,
            start_time_minutes=self.t0, prior_mean=self.person, prior_std=self.prior_std,
            initial_embedding=self.person, settings=settings,
        )

    def test_shifted_initial_glucose_stays_on_the_window(self) -> None:
        shifted = self.nominal.clone()
        shifted[self.gi] = self.nominal[self.gi] + 40.0
        times = [0, 15, 30, 45, 60]
        obs = _glucose_obs(self.model, shifted, self.person, [], times, self.t0)
        res = self._fit(obs, [], 90)
        window_g = float(res.window_initial_state[self.gi])
        distance = float((res.embedding - self.person).norm())
        self.assertGreater(
            abs(window_g - float(self.nominal[self.gi])), 20.0,
            msg=f"window glucose {window_g:.2f} reason={res.reason} accepted={res.accepted} "
                f"||de||={distance:.3f} val {res.baseline_val_loss:.4f}->{res.val_loss:.4f}",
        )
        self.assertLess(
            distance, 0.25,
            msg=f"embedding moved {distance:.3f} reason={res.reason} window glucose {window_g:.2f}",
        )

    def test_unlogged_meal_stays_on_the_disturbance(self) -> None:
        secret = [MealEvent(time=10.0, carbs=70.0, fats=0.0, proteins=0.0)]
        times = [25, 45, 65, 85, 105]
        obs = _glucose_obs(self.model, self.nominal, self.person, secret, times, self.t0)
        res = self._fit(obs, [], 140)
        distance = float((res.embedding - self.person).norm())
        disturbance = float(res.disturbance.norm())
        self.assertGreater(
            disturbance, 0.3,
            msg=f"disturbance {disturbance:.3f} reason={res.reason} accepted={res.accepted} "
                f"||de||={distance:.3f} val {res.baseline_val_loss:.4f}->{res.val_loss:.4f} "
                f"dev {res.state_deviation_norm:.3f}",
        )
        self.assertLess(
            distance, 0.25,
            msg=f"embedding moved {distance:.3f} disturbance {disturbance:.3f} reason={res.reason}",
        )

    def test_lucky_single_holdout_is_still_rejected(self) -> None:
        times = [0, 20, 40, 60, 80]
        obs = _glucose_obs(self.model, self.nominal, self.person, [], times, self.t0)
        last = obs[-1]
        obs[-1] = MeasurementPoint(time=last.time, marker_id="glucose", value=last.value + 35.0)
        settings = CalibrationSettings(max_steps=12, lr=0.1, patience=6, prior_weight=0.05)
        res = calibrate_embedding(
            self.model, obs, self.nominal, [], 100,
            start_time_minutes=self.t0, prior_mean=self.person, prior_std=self.prior_std,
            initial_embedding=self.person, settings=settings,
        )
        self.assertEqual(res.settings["accept_rel_improvement"], 0.01)
        self.assertEqual(res.settings["accept_max_train_regression"], 0.10)
        self.assertFalse(res.accepted)
        self.assertEqual(
            res.reason, "train_regressed",
            msg=f"train {res.baseline_train_loss:.4f}->{res.train_loss:.4f} "
                f"val {res.baseline_val_loss:.4f}->{res.val_loss:.4f} steps {res.n_steps}",
        )
        self.assertTrue(torch.allclose(res.embedding, self.person))


if __name__ == "__main__":
    unittest.main()
