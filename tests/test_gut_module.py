"""Equivalence + correctness tests for the vectorized GutModule paths.

The gut module historically iterated meal-by-meal (and the trajectory
signal iterated time-step-by-time-step). Both loops are now batched.
This test pins down that the batched outputs match the reference
loop-based implementation within float-reduction tolerance — vectorized
gut is the new spine of the gut-loss path, so a regression here would
silently corrupt every future training iter.
"""

from __future__ import annotations

import unittest

import torch

from pulse.modules.gut import GutModule, MEAL_ACTIVE_WINDOW_MIN, MealEvent


def _reference_forward_window(
    gut: GutModule,
    times: torch.Tensor,
    meals: list[MealEvent],
    embedding: torch.Tensor,
) -> torch.Tensor:
    """Reference implementation: per-step calls into the vectorized
    ``forward`` (which itself remains the per-time-point public API)."""
    return torch.stack([gut(float(t.item()), meals, embedding) for t in times])


def _per_meal_forward(
    gut: GutModule,
    t: float,
    meals: list[MealEvent],
    embedding: torch.Tensor,
) -> torch.Tensor:
    """Reference: kernel called once per active meal, then summed.
    Mirrors the pre-vectorization ``forward`` body."""
    total = torch.zeros(4)
    for meal in meals:
        dt = t - meal.time
        if dt < 0 or dt > MEAL_ACTIVE_WINDOW_MIN:
            continue
        macros = torch.tensor(
            [meal.carbs, meal.fats, meal.proteins], dtype=torch.float32,
        ).unsqueeze(0)
        dt_t = torch.tensor(dt, dtype=torch.float32).unsqueeze(0)
        emb_b = embedding.unsqueeze(0)
        appearance = gut.kernel.forward_single_meal(macros, dt_t, emb_b).squeeze(0)
        total = total + appearance
    return total


class TestGutForwardEquivalence(unittest.TestCase):
    """``forward(t, meals, emb)`` (vectorized over meals) should match
    the per-meal loop bit-for-bit modulo float-reduction order."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.gut = GutModule(embedding_dim=8, hidden_dim=16)
        self.gut.eval()
        self.embedding = torch.randn(8)
        self.meals = [
            MealEvent(time=20.0, carbs=50.0, fats=10.0, proteins=20.0),
            MealEvent(time=120.0, carbs=30.0, fats=5.0, proteins=10.0),
            MealEvent(time=300.0, carbs=80.0, fats=20.0, proteins=30.0),
        ]

    def test_matches_per_meal_loop_at_active_time(self) -> None:
        """t=200 → meals at 20 and 120 are active (180/80 min ago); meal at 300 inactive."""
        out = self.gut(200.0, self.meals, self.embedding)
        ref = _per_meal_forward(self.gut, 200.0, self.meals, self.embedding)
        torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-5)

    def test_no_active_meals_returns_zero(self) -> None:
        """t=10 is before any meal → zero output."""
        out = self.gut(10.0, self.meals, self.embedding)
        ref = _per_meal_forward(self.gut, 10.0, self.meals, self.embedding)
        torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-5)
        self.assertEqual(float(out.abs().sum()), 0.0)

    def test_empty_meal_list(self) -> None:
        out = self.gut(200.0, [], self.embedding)
        self.assertEqual(float(out.abs().sum()), 0.0)

    def test_meal_just_outside_active_window(self) -> None:
        """A meal at age = MEAL_ACTIVE_WINDOW_MIN + 1 must not contribute."""
        meals = [MealEvent(time=0.0, carbs=50.0, fats=10.0, proteins=20.0)]
        t = MEAL_ACTIVE_WINDOW_MIN + 1.0
        out = self.gut(t, meals, self.embedding)
        ref = _per_meal_forward(self.gut, t, meals, self.embedding)
        torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-5)
        self.assertEqual(float(out.abs().sum()), 0.0)


class TestGutForwardWindowEquivalence(unittest.TestCase):
    """``forward_window`` (vectorized over T and meals) should match the
    per-step ``forward`` reference. This is what the trajectory signal's
    gut-loss path uses — the hot refactor target."""

    def setUp(self) -> None:
        torch.manual_seed(1)
        self.gut = GutModule(embedding_dim=8, hidden_dim=16)
        self.gut.eval()
        self.embedding = torch.randn(8)
        self.meals = [
            MealEvent(time=30.0, carbs=60.0, fats=10.0, proteins=20.0),
            MealEvent(time=180.0, carbs=40.0, fats=5.0, proteins=12.0),
        ]

    def test_window_matches_per_step_loop(self) -> None:
        """Cover pre-meal, active, and post-active regions in one window."""
        times = torch.arange(0, 240, dtype=torch.float32)
        out = self.gut.forward_window(times, self.meals, self.embedding)
        ref = _reference_forward_window(self.gut, times, self.meals, self.embedding)
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    def test_window_handles_zero_active_meals(self) -> None:
        """All times before any meal → zero output."""
        times = torch.arange(0, 20, dtype=torch.float32)
        out = self.gut.forward_window(times, self.meals, self.embedding)
        self.assertEqual(out.shape, (20, 4))
        self.assertEqual(float(out.detach().abs().sum()), 0.0)

    def test_window_empty_meal_list(self) -> None:
        times = torch.arange(0, 60, dtype=torch.float32)
        out = self.gut.forward_window(times, [], self.embedding)
        self.assertEqual(out.shape, (60, 4))
        self.assertEqual(float(out.abs().sum()), 0.0)

    def test_window_gradient_flows_to_kernel_and_embedding(self) -> None:
        """Vectorized path must remain differentiable end-to-end."""
        times = torch.arange(0, 240, dtype=torch.float32)
        emb = self.embedding.clone().requires_grad_(True)
        out = self.gut.forward_window(times, self.meals, emb)
        loss = out.pow(2).sum()
        loss.backward()
        self.assertIsNotNone(emb.grad)
        self.assertGreater(float(emb.grad.abs().sum()), 0.0)
        kernel_grad_total = sum(
            float(p.grad.abs().sum()) if p.grad is not None else 0.0
            for p in self.gut.kernel.parameters()
        )
        self.assertGreater(kernel_grad_total, 0.0)


class TestGutKernelStructuralProperties(unittest.TestCase):
    """The factorized kernel guarantees two physical properties by
    construction (analytically, not via SGD). These tests pin them down
    so any future architectural change has to preserve them or fail
    loudly.

    1. ``appearance(macros=0) == 0``  — no food, no nutrient appearance.
    2. ``appearance(α·macros) == α·appearance(macros)`` — dose-linearity
       (which also implies dose-monotonicity).

    nutrient_flag must also vanish at zero macros (gated by analytic
    presence factor), but it is non-linear in dose by design.
    """

    def setUp(self) -> None:
        torch.manual_seed(7)
        self.gut = GutModule(embedding_dim=8, hidden_dim=16)
        self.gut.eval()

    def test_zero_macros_gives_zero_output(self) -> None:
        """At every (t, embedding), zero macros must yield zero appearance
        AND zero nutrient_flag. This kills the chronic ~40 mg-min/min
        baseline offset that bled glucose_mape across iters 13-15."""
        emb = torch.randn(8)
        for t in [0.0, 30.0, 90.0, 240.0, 480.0]:
            macros = torch.zeros(1, 3)
            dt = torch.tensor([t])
            out = self.gut.kernel.forward_single_meal(macros, dt, emb.unsqueeze(0)).squeeze(0)
            self.assertEqual(float(out.detach().abs().sum()), 0.0,
                msg=f"non-zero output at zero macros, t={t}: {out}")

    def test_appearance_is_linear_in_dose(self) -> None:
        """Doubling carbs/fats/proteins must exactly double the appearance
        rate channels (channels 0-2). This kills the saturation that
        capped iter 15's response at ~130 AUC for any dose ≥30 g."""
        emb = torch.randn(8)
        macros = torch.tensor([[60.0, 12.0, 20.0]])
        dt = torch.tensor([45.0])
        out_1x = self.gut.kernel.forward_single_meal(macros, dt, emb.unsqueeze(0)).squeeze(0)
        out_2x = self.gut.kernel.forward_single_meal(2 * macros, dt, emb.unsqueeze(0)).squeeze(0)
        # First three channels: appearance rates — must scale linearly.
        torch.testing.assert_close(out_2x[:3], 2 * out_1x[:3], atol=1e-6, rtol=1e-5)

    def test_appearance_scales_with_arbitrary_dose_factor(self) -> None:
        """Stronger version: across a sweep of factors, appearance is
        exactly proportional. Catches any sneaky non-linearity that
        might pass the 1×/2× check."""
        emb = torch.randn(8)
        base = torch.tensor([[50.0, 10.0, 15.0]])
        dt = torch.tensor([30.0])
        out_base = self.gut.kernel.forward_single_meal(base, dt, emb.unsqueeze(0)).squeeze(0)[:3]
        for alpha in [0.1, 0.5, 1.5, 3.0, 10.0]:
            scaled = alpha * base
            out_scaled = self.gut.kernel.forward_single_meal(
                scaled, dt, emb.unsqueeze(0)
            ).squeeze(0)[:3]
            torch.testing.assert_close(
                out_scaled, alpha * out_base, atol=1e-5, rtol=1e-5,
                msg=f"non-linear at α={alpha}",
            )

    def test_nutrient_flag_vanishes_at_zero_macros(self) -> None:
        """The flag presence-gate ``1 − exp(−total_macro / scale)`` must
        force nutrient_flag to exactly zero when no food is present,
        regardless of (t, embedding) and learned weights."""
        emb = torch.randn(8)
        for t in [0.0, 60.0, 240.0]:
            macros = torch.zeros(1, 3)
            dt = torch.tensor([t])
            out = self.gut.kernel.forward_single_meal(macros, dt, emb.unsqueeze(0)).squeeze(0)
            self.assertEqual(float(out[3].detach()), 0.0)

    def test_nutrient_flag_saturates_with_increasing_dose(self) -> None:
        """nutrient_flag is binary fed/fasted and must rise toward 1 as
        food increases (not strict monotone, since the sigmoid-on-(t,emb)
        factor can be small, but the presence gate is monotone)."""
        emb = torch.randn(8)
        dt = torch.tensor([45.0])
        flags = []
        for total in [0.0, 5.0, 20.0, 60.0, 200.0]:
            macros = torch.tensor([[total / 3, total / 3, total / 3]])
            out = self.gut.kernel.forward_single_meal(macros, dt, emb.unsqueeze(0)).squeeze(0)
            flags.append(float(out[3].detach()))
        for i in range(len(flags) - 1):
            self.assertLessEqual(
                flags[i], flags[i + 1] + 1e-6,
                msg=f"flag not monotone at index {i}: {flags}",
            )

    def test_dose_sweep_via_forward_window_is_exactly_proportional(self) -> None:
        """End-to-end dose sweep through the public forward_window API
        (the path gut_dose_sweep_signal exercises) must produce AUCs
        that are exactly proportional to dose. This is the structural
        guarantee that retires the dose-ranking penalty as a corrective
        device — it now serves only as defense in depth."""
        emb = torch.randn(8)
        times = torch.arange(0, 300, dtype=torch.float32)
        meals_a = [MealEvent(time=0.0, carbs=30.0, fats=6.0, proteins=10.0)]
        meals_b = [MealEvent(time=0.0, carbs=120.0, fats=24.0, proteins=40.0)]
        out_a = self.gut.forward_window(times, meals_a, emb)
        out_b = self.gut.forward_window(times, meals_b, emb)
        # 4× macros → 4× appearance rates at every time.
        torch.testing.assert_close(
            out_b[..., :3], 4 * out_a[..., :3], atol=1e-4, rtol=1e-4,
        )


class TestGutForwardWindowSpeedup(unittest.TestCase):
    """Sanity check: ``forward_window`` is meaningfully faster than the
    per-step loop. Not a strict performance gate (CI machines vary), but
    if it ever regresses below 2x we want to know."""

    def test_window_faster_than_per_step_loop(self) -> None:
        import time

        torch.manual_seed(2)
        gut = GutModule(embedding_dim=16, hidden_dim=24)
        gut.eval()
        emb = torch.randn(16)
        meals = [
            MealEvent(
                time=float(i * 60),
                carbs=50.0 + i * 5,
                fats=10.0,
                proteins=15.0,
            )
            for i in range(8)
        ]
        times = torch.arange(0, 240, dtype=torch.float32)

        # Warm-up.
        gut.forward_window(times, meals, emb)
        _reference_forward_window(gut, times, meals, emb)

        with torch.no_grad():
            t0 = time.perf_counter()
            for _ in range(5):
                gut.forward_window(times, meals, emb)
            t_window = time.perf_counter() - t0

            t0 = time.perf_counter()
            for _ in range(5):
                _reference_forward_window(gut, times, meals, emb)
            t_loop = time.perf_counter() - t0

        # Loop must be slower than window. Headroom of 1.5x to avoid
        # flakiness on contended CI; observed locally is typically 5-15x.
        self.assertGreater(t_loop, 1.5 * t_window)


if __name__ == "__main__":
    unittest.main()
