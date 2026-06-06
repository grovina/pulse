"""Regression tests for training window sampling (meal-biased vs uniform)."""

from __future__ import annotations

import unittest

import numpy as np

from pulse.train import TRAIN_WINDOW, _sample_window_start


def _window_has_verifier_friendly_carb_meal(
    win_start: int,
    win_end: int,
    meals: list[tuple[float, float, float, float]],
) -> bool:
    """Match training_verifier_surrogate_loss meal gates: local t in [15, win_len - 121]."""
    win_len = win_end - win_start
    for t, c, _f, _p in meals:
        if float(c) <= 0.0:
            continue
        local = int(round(float(t))) - win_start
        if local >= 15 and local + 120 < win_len:
            return True
    return False


class TestSampleWindowStart(unittest.TestCase):
    def test_full_meal_bias_hits_post_meal_band(self) -> None:
        n_steps = 14 * 1440
        meals = [(600, 50, 10, 20), (1200, 30, 5, 10)]
        rng = np.random.default_rng(42)
        n = 800
        hits = 0
        for _ in range(n):
            ws = _sample_window_start(
                n_steps,
                meals,
                rng,
                window=TRAIN_WINDOW,
                meal_bias_prob=1.0,
            )
            we = min(ws + TRAIN_WINDOW, n_steps)
            if _window_has_verifier_friendly_carb_meal(ws, we, meals):
                hits += 1
        self.assertEqual(hits, n)

    def test_uniform_sampling_rarely_hits_on_sparse_meals(self) -> None:
        n_steps = 14 * 1440
        meals = [(600, 50, 10, 20), (1200, 30, 5, 10)]
        rng = np.random.default_rng(0)
        n = 2000
        hits = 0
        for _ in range(n):
            ws = _sample_window_start(
                n_steps,
                meals,
                rng,
                window=TRAIN_WINDOW,
                meal_bias_prob=0.0,
            )
            we = min(ws + TRAIN_WINDOW, n_steps)
            if _window_has_verifier_friendly_carb_meal(ws, we, meals):
                hits += 1
        self.assertLess(hits, n * 0.08)

    def test_empty_meals_uniform_no_crash(self) -> None:
        rng = np.random.default_rng(1)
        ws = _sample_window_start(
            5000,
            [],
            rng,
            window=TRAIN_WINDOW,
            meal_bias_prob=1.0,
        )
        self.assertGreaterEqual(ws, 0)
        self.assertLess(ws, max(1, 5000 - TRAIN_WINDOW))

    def test_no_carb_meals_falls_back_to_uniform(self) -> None:
        n_steps = 8000
        meals = [(100, 0, 20, 10), (400, 0, 5, 5)]
        rng = np.random.default_rng(2)
        for _ in range(50):
            ws = _sample_window_start(
                n_steps,
                meals,
                rng,
                window=TRAIN_WINDOW,
                meal_bias_prob=1.0,
            )
            self.assertGreaterEqual(ws, 0)
            self.assertLess(ws, max(1, n_steps - TRAIN_WINDOW))


if __name__ == "__main__":
    unittest.main()
