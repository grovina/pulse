"""Unit tests for differentiable landmark extractors and the per-window loss."""

from __future__ import annotations

import math
import unittest

import torch

from pulse.landmarks import (
    DEFAULT_LANDMARK_SPECS,
    LandmarkDirection,
    LandmarkScales,
    MarkerLandmarkSpec,
    post_meal_landmark_loss,
    post_meal_landmarks,
)
from pulse.modules.gut import MealEvent
from pulse.types import MARKER_INDEX, NORM_CENTER, STATE_DIM


# The landmark loss supervises HR only in production (DEFAULT_LANDMARK_SPECS);
# these metabolic PEAK/NADIR specs are test fixtures so the extractor machinery
# (softargmax, AUC, both directions) gets exercised on more than one marker.
# (They were production specs through iter 33 — see docs/dead-pathways.md for
# why counter-regulatory markers are no longer supervised via landmarks.)
_GLUCOSE_PEAK = MarkerLandmarkSpec(
    "glucose", LandmarkDirection.PEAK,
    LandmarkScales(delta_extremum=30.0, time_to_extremum=30.0,
                   auc=1500.0, softargmax_beta=0.10),
)
_INSULIN_PEAK = MarkerLandmarkSpec(
    "insulin", LandmarkDirection.PEAK,
    LandmarkScales(delta_extremum=10.0, time_to_extremum=30.0,
                   auc=500.0, softargmax_beta=0.20),
)
_GLUCAGON_NADIR = MarkerLandmarkSpec(
    "glucagon", LandmarkDirection.NADIR,
    LandmarkScales(delta_extremum=12.0, time_to_extremum=45.0,
                   auc=600.0, softargmax_beta=0.10),
)
_FFA_NADIR = MarkerLandmarkSpec(
    "ffa", LandmarkDirection.NADIR,
    LandmarkScales(delta_extremum=0.10, time_to_extremum=60.0,
                   auc=10.0, softargmax_beta=10.0),
)
_GHRELIN_NADIR = MarkerLandmarkSpec(
    "ghrelin", LandmarkDirection.NADIR,
    LandmarkScales(delta_extremum=40.0, time_to_extremum=60.0,
                   auc=2000.0, softargmax_beta=0.05),
)
_ALL_LANDMARK_SPECS: tuple[MarkerLandmarkSpec, ...] = (
    _GLUCOSE_PEAK,
    _INSULIN_PEAK,
    _GLUCAGON_NADIR,
    _FFA_NADIR,
    _GHRELIN_NADIR,
    *DEFAULT_LANDMARK_SPECS,
)


GLUCOSE_PEAK_BETA = next(
    s.scales.softargmax_beta for s in _ALL_LANDMARK_SPECS if s.marker_id == "glucose"
)


def _spec(marker: str, direction: LandmarkDirection) -> MarkerLandmarkSpec:
    """Locate a default spec by marker id (for tests that pin a single pair)."""
    return next(
        s for s in _ALL_LANDMARK_SPECS
        if s.marker_id == marker and s.direction is direction
    )


_PEAK_PAIR_SPECS: tuple[MarkerLandmarkSpec, ...] = (
    _spec("glucose", LandmarkDirection.PEAK),
    _spec("insulin", LandmarkDirection.PEAK),
)


def _gaussian_bump(
    n: int, peak_step: int, height: float, width: float, baseline: float = 0.0,
) -> torch.Tensor:
    """A single Gaussian bump on top of ``baseline`` for synthetic landmarks."""
    t = torch.arange(n, dtype=torch.float32)
    return baseline + height * torch.exp(-((t - peak_step) ** 2) / (2.0 * width**2))


def _flat_traj(n: int) -> torch.Tensor:
    """Constant-typical trajectory of shape (n, STATE_DIM) (no NaN)."""
    return torch.tensor(NORM_CENTER, dtype=torch.float32).repeat(n, 1)


class TestPostMealLandmarks(unittest.TestCase):
    def test_recovers_known_peak_and_time_sharp(self) -> None:
        """At high β the soft-peak collapses onto the true (max, argmax)."""
        n = 240
        meal_step = 15
        peak_offset = 60
        series = _gaussian_bump(
            n, peak_step=meal_step + peak_offset, height=60.0, width=10.0, baseline=90.0,
        )
        delta_peak, ttp, auc = post_meal_landmarks(
            series, meal_step=meal_step, pre_window=15, post_window=120,
            softargmax_beta=2.0,  # sharp enough to behave like hard max
            direction=LandmarkDirection.PEAK,
        )
        self.assertAlmostEqual(float(delta_peak.item()), 60.0, delta=0.5)
        self.assertAlmostEqual(float(ttp.item()), float(peak_offset), delta=0.5)
        # AUC ≈ height * sqrt(2π) * width = 60 * 2.5066 * 10 ≈ 1504
        expected_auc = 60.0 * math.sqrt(2.0 * math.pi) * 10.0
        self.assertAlmostEqual(float(auc.item()), expected_auc, delta=expected_auc * 0.05)

    def test_landmarks_track_amplitude_and_timing(self) -> None:
        """At production β the soft estimates are biased but monotonic — what we need."""
        n = 240
        meal_step = 15
        beta = GLUCOSE_PEAK_BETA
        small = _gaussian_bump(n, peak_step=meal_step + 60, height=20.0, width=20.0, baseline=90.0)
        big = _gaussian_bump(n, peak_step=meal_step + 60, height=80.0, width=20.0, baseline=90.0)
        early = _gaussian_bump(n, peak_step=meal_step + 30, height=60.0, width=20.0, baseline=90.0)
        late = _gaussian_bump(n, peak_step=meal_step + 90, height=60.0, width=20.0, baseline=90.0)
        kw = dict(meal_step=meal_step, pre_window=15, post_window=120,
                  softargmax_beta=beta, direction=LandmarkDirection.PEAK)
        d_small, _, a_small = post_meal_landmarks(small, **kw)
        d_big, _, a_big = post_meal_landmarks(big, **kw)
        _, t_early, _ = post_meal_landmarks(early, **kw)
        _, t_late, _ = post_meal_landmarks(late, **kw)
        self.assertGreater(float(d_big.item()), float(d_small.item()))
        self.assertGreater(float(a_big.item()), float(a_small.item()))
        self.assertLess(float(t_early.item()), float(t_late.item()))


class TestPostMealLandmarkLoss(unittest.TestCase):
    def _build_traj(self, n: int, glucose_peak: float, peak_step: int) -> torch.Tensor:
        traj = _flat_traj(n)
        gi = MARKER_INDEX["glucose"]
        ii = MARKER_INDEX["insulin"]
        traj[:, gi] = _gaussian_bump(n, peak_step=peak_step, height=glucose_peak, width=25.0, baseline=90.0)
        # mirror insulin bump (smaller scale)
        traj[:, ii] = _gaussian_bump(n, peak_step=peak_step + 5, height=15.0, width=25.0, baseline=10.0)
        return traj

    def _loss_kwargs(self) -> dict:
        return dict(
            pre_window=15, post_window=120, min_carbs=5.0,
            specs=_PEAK_PAIR_SPECS,
        )

    def test_zero_loss_when_pred_equals_target(self) -> None:
        n = 240
        traj = self._build_traj(n, glucose_peak=60.0, peak_step=60)
        meals = [MealEvent(time=0, carbs=50.0, fats=0.0, proteins=0.0)]
        loss, n_pairs = post_meal_landmark_loss(traj, traj, meals, **self._loss_kwargs())
        self.assertEqual(n_pairs, 0)  # meal at step 0 doesn't have pre_window=15 prior

        meals = [MealEvent(time=20, carbs=50.0, fats=0.0, proteins=0.0)]
        loss, n_pairs = post_meal_landmark_loss(traj, traj, meals, **self._loss_kwargs())
        self.assertEqual(n_pairs, 2)  # glucose + insulin
        self.assertAlmostEqual(float(loss.item()), 0.0, delta=1e-6)

    def test_positive_loss_when_pred_off(self) -> None:
        n = 240
        target = self._build_traj(n, glucose_peak=60.0, peak_step=60)
        pred = self._build_traj(n, glucose_peak=20.0, peak_step=90)  # smaller, later peak
        meals = [MealEvent(time=20, carbs=50.0, fats=0.0, proteins=0.0)]
        loss, n_pairs = post_meal_landmark_loss(pred, target, meals, **self._loss_kwargs())
        self.assertEqual(n_pairs, 2)
        self.assertGreater(float(loss.item()), 0.5)

    def test_skips_low_carb_meals(self) -> None:
        n = 240
        traj = self._build_traj(n, glucose_peak=60.0, peak_step=60)
        meals = [MealEvent(time=20, carbs=2.0, fats=10.0, proteins=10.0)]  # below min_carbs
        _loss, n_pairs = post_meal_landmark_loss(traj, traj, meals, **self._loss_kwargs())
        self.assertEqual(n_pairs, 0)

    def test_skips_meals_outside_window(self) -> None:
        n = 100  # too short for post_window=120
        traj = self._build_traj(n, glucose_peak=60.0, peak_step=40)
        meals = [MealEvent(time=20, carbs=50.0, fats=0.0, proteins=0.0)]
        _loss, n_pairs = post_meal_landmark_loss(traj, traj, meals, **self._loss_kwargs())
        self.assertEqual(n_pairs, 0)

    def test_skips_target_with_nan(self) -> None:
        n = 240
        target = self._build_traj(n, glucose_peak=60.0, peak_step=60)
        pred = target.clone()
        # Inject NaN into target glucose post-window only — insulin should still count.
        gi = MARKER_INDEX["glucose"]
        target[100, gi] = float("nan")
        meals = [MealEvent(time=20, carbs=50.0, fats=0.0, proteins=0.0)]
        _loss, n_pairs = post_meal_landmark_loss(pred, target, meals, **self._loss_kwargs())
        self.assertEqual(n_pairs, 1)  # only insulin

    def test_loss_is_differentiable(self) -> None:
        n = 240
        target = self._build_traj(n, glucose_peak=60.0, peak_step=60)
        pred = self._build_traj(n, glucose_peak=20.0, peak_step=90).requires_grad_(True)
        meals = [MealEvent(time=20, carbs=50.0, fats=0.0, proteins=0.0)]
        loss, n_pairs = post_meal_landmark_loss(pred, target, meals, **self._loss_kwargs())
        self.assertEqual(n_pairs, 2)
        loss.backward()
        self.assertIsNotNone(pred.grad)
        self.assertGreater(float(pred.grad.abs().sum().item()), 0.0)


class TestNadirLandmarks(unittest.TestCase):
    """NADIR direction supervises counter-regulatory suppression: glucagon,
    FFA, ghrelin all drop after a mixed meal in the cold model. The same
    soft-argmax flips into a soft-argmin when the temperature's sign flips,
    so peak/nadir share one extractor."""

    def test_nadir_recovers_known_drop_and_time(self) -> None:
        """Inverted Gaussian (a "drop") with sharp β should recover (-depth, t_min)."""
        n = 240
        meal_step = 15
        nadir_offset = 60
        baseline = 70.0
        depth = 12.0
        # Negative bump: baseline - Gaussian peak.
        series = baseline - _gaussian_bump(
            n, peak_step=meal_step + nadir_offset, height=depth,
            width=10.0, baseline=0.0,
        )
        delta_nadir, ttn, auc = post_meal_landmarks(
            series, meal_step=meal_step, pre_window=15, post_window=120,
            softargmax_beta=2.0, direction=LandmarkDirection.NADIR,
        )
        self.assertAlmostEqual(float(delta_nadir.item()), -depth, delta=0.5)
        self.assertAlmostEqual(float(ttn.item()), float(nadir_offset), delta=0.5)
        # AUC magnitude (area below baseline) ≈ depth · sqrt(2π) · width.
        expected_auc = depth * math.sqrt(2.0 * math.pi) * 10.0
        self.assertAlmostEqual(float(auc.item()), expected_auc, delta=expected_auc * 0.05)

    def test_default_specs_pin_zero_loss_on_flat_counter_reg(self) -> None:
        """Pred/target both flat at NORM_CENTER for counter-regulatory markers
        ⇒ NADIR landmark contributions are zero (Δ=0, AUC=0, TTE identical
        for two flat series)."""
        n = 240
        traj = torch.tensor(NORM_CENTER, dtype=torch.float32).repeat(n, 1)
        meals = [MealEvent(time=20, carbs=50.0, fats=10.0, proteins=10.0)]
        loss, n_pairs = post_meal_landmark_loss(
            traj, traj, meals,
            pre_window=15, post_window=120, min_carbs=5.0,
        )
        # Default specs: glucose+insulin (PEAK) + glucagon+ffa+ghrelin (NADIR) = 5.
        self.assertEqual(n_pairs, len(DEFAULT_LANDMARK_SPECS))
        self.assertAlmostEqual(float(loss.item()), 0.0, delta=1e-6)

    def test_nadir_loss_penalizes_weak_suppression(self) -> None:
        """Predicting flat glucagon when target drops should produce
        positive loss in the glucagon NADIR term."""
        n = 240
        meal_step = 20
        target = torch.tensor(NORM_CENTER, dtype=torch.float32).repeat(n, 1)
        gi = MARKER_INDEX["glucagon"]
        target[:, gi] = NORM_CENTER[gi] - _gaussian_bump(
            n, peak_step=meal_step + 60, height=10.0, width=20.0, baseline=0.0,
        )
        pred = torch.tensor(NORM_CENTER, dtype=torch.float32).repeat(n, 1)  # flat ⇒ no drop
        meals = [MealEvent(time=meal_step, carbs=50.0, fats=10.0, proteins=10.0)]
        glucagon_only = (_spec("glucagon", LandmarkDirection.NADIR),)
        loss, n_pairs = post_meal_landmark_loss(
            pred, target, meals,
            pre_window=15, post_window=120, min_carbs=5.0, specs=glucagon_only,
        )
        self.assertEqual(n_pairs, 1)
        self.assertGreater(float(loss.item()), 0.1)

    def test_nadir_loss_is_differentiable(self) -> None:
        n = 240
        meal_step = 20
        target = torch.tensor(NORM_CENTER, dtype=torch.float32).repeat(n, 1)
        gi = MARKER_INDEX["glucagon"]
        target[:, gi] = NORM_CENTER[gi] - _gaussian_bump(
            n, peak_step=meal_step + 60, height=10.0, width=20.0, baseline=0.0,
        )
        pred = torch.tensor(NORM_CENTER, dtype=torch.float32).repeat(n, 1)
        pred.requires_grad_(True)
        meals = [MealEvent(time=meal_step, carbs=50.0, fats=10.0, proteins=10.0)]
        loss, _ = post_meal_landmark_loss(
            pred, target, meals,
            pre_window=15, post_window=120, min_carbs=5.0,
            specs=(_spec("glucagon", LandmarkDirection.NADIR),),
        )
        loss.backward()
        self.assertIsNotNone(pred.grad)
        self.assertGreater(float(pred.grad.abs().sum().item()), 0.0)


if __name__ == "__main__":
    unittest.main()
