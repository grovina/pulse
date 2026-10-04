"""Per-statistic-kind unit tests for cohort_loss.

We exercise the differentiable extractors on hand-crafted trajectories so the
expected value for each kind is unambiguous, then check the spec validation
(arm-count, sigma>0), and (PLAN A7) the debiased batch-mean loss.
"""

from __future__ import annotations

import unittest
import warnings
from itertools import product

import torch

from pulse.cohort_loss import (
    _arm_statistic_batched,
    _debiased_sq_residual,
    score_batch_statistic,
)
from pulse.knowledge.cohort_types import (
    CohortArmSpec,
    CohortStatisticSpec,
    StatisticKind,
    StatisticWindow,
    TargetShape,
)
from pulse.types import MARKER_INDEX


def _const_traj(n_steps: int, marker_value: float, marker_id: str = "glucose") -> torch.Tensor:
    """[1, T, STATE_DIM] trajectory with one marker held at marker_value."""
    state_dim = len(MARKER_INDEX)
    traj = torch.zeros(1, n_steps, state_dim)
    traj[0, :, MARKER_INDEX[marker_id]] = marker_value
    return traj


def _ramp_traj(n_steps: int, peak_at: int, peak_value: float, marker_id: str = "glucose") -> torch.Tensor:
    """[1, T, STATE_DIM] linear ramp up to peak_at, then linear ramp down."""
    state_dim = len(MARKER_INDEX)
    traj = torch.zeros(1, n_steps, state_dim)
    mi = MARKER_INDEX[marker_id]
    for t in range(n_steps):
        if t <= peak_at:
            traj[0, t, mi] = peak_value * (t / max(peak_at, 1))
        else:
            traj[0, t, mi] = peak_value * max(0.0, 1.0 - (t - peak_at) / max(n_steps - peak_at - 1, 1))
    return traj


class TestStatisticExtractors(unittest.TestCase):
    def test_mean_in_window(self) -> None:
        traj = _const_traj(100, 5.0)
        v = _arm_statistic_batched(
            traj, MARKER_INDEX["glucose"], StatisticWindow(10, 50),
            StatisticKind.MEAN_IN_WINDOW, 0.05,
        )
        self.assertEqual(v.shape, (1,))
        self.assertAlmostEqual(float(v[0]), 5.0, places=5)

    def test_peak_value(self) -> None:
        traj = _ramp_traj(100, peak_at=40, peak_value=12.0)
        v = _arm_statistic_batched(
            traj, MARKER_INDEX["glucose"], StatisticWindow(0, 100),
            StatisticKind.PEAK_VALUE, 0.05,
        )
        self.assertAlmostEqual(float(v[0]), 12.0, places=4)

    def test_time_to_peak_softargmax(self) -> None:
        # Sharp peak at index 30 (relative to window) — soft argmax should be close.
        traj = _ramp_traj(120, peak_at=50, peak_value=80.0)
        v = _arm_statistic_batched(
            traj, MARKER_INDEX["glucose"], StatisticWindow(20, 100),
            StatisticKind.TIME_TO_PEAK, 0.5,
        )
        # Window starts at 20, peak originally at 50 → relative index 30. Tolerate softness.
        self.assertGreater(float(v[0]), 25.0)
        self.assertLess(float(v[0]), 35.0)


class TestCounterRegulatorySpecs(unittest.TestCase):
    """The Phase-3 specs encode counter-regulatory physiology that the
    benchmark currently fails. Cheap structural checks: each spec is well-
    formed, references a real marker, has a sensible window, and lives in
    ``ALL_COHORT_STATISTICS`` so the training signal will pick it up.
    """

    def test_counter_regulatory_specs_registered(self) -> None:
        from pulse.knowledge.cohort_statistics import ALL_COHORT_STATISTICS
        names = {s.name for s in ALL_COHORT_STATISTICS}
        for required in (
            "extended_fast_ffa_overnight",
            "extended_fast_insulin_basal",
            "ogtt_glucagon_suppression",
            "mixed_meal_glucagon_suppression",
            "meal_ffa_suppression",
        ):
            self.assertIn(required, names, f"missing cohort spec: {required}")

    def test_counter_regulatory_specs_have_valid_markers_and_windows(self) -> None:
        from pulse.knowledge.cohort_statistics import ALL_COHORT_STATISTICS
        new_names = {
            "extended_fast_ffa_overnight",
            "extended_fast_insulin_basal",
            "ogtt_glucagon_suppression",
            "mixed_meal_glucagon_suppression",
            "meal_ffa_suppression",
        }
        for spec in ALL_COHORT_STATISTICS:
            if spec.name not in new_names:
                continue
            self.assertIn(spec.marker_id, MARKER_INDEX, f"{spec.name}: unknown marker")
            self.assertGreater(spec.sigma, 0.0)
            for arm in spec.arms:
                self.assertGreater(spec.window.end_min, spec.window.start_min)
                self.assertLessEqual(spec.window.end_min, arm.duration_min)


class TestSpecValidation(unittest.TestCase):
    def test_delta_means_requires_two_arms(self) -> None:
        with self.assertRaises(ValueError):
            CohortStatisticSpec(
                name="x", source="x", description="x",
                arms=(CohortArmSpec(label="a", duration_min=10, start_hour=6.0, meals=()),),
                marker_id="glucose",
                kind=StatisticKind.DELTA_MEANS,
                window=StatisticWindow(0, 5),
                target=0.0, sigma=1.0,
            )

    def test_sigma_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            CohortStatisticSpec(
                name="x", source="x", description="x",
                arms=(CohortArmSpec(label="a", duration_min=10, start_hour=6.0, meals=()),),
                marker_id="glucose",
                kind=StatisticKind.MEAN_IN_WINDOW,
                window=StatisticWindow(0, 5),
                target=0.0, sigma=0.0,
            )


def _point_spec(**kw: object) -> CohortStatisticSpec:
    base: dict[str, object] = dict(
        name="t", source="t", description="t",
        arms=(CohortArmSpec(label="a", duration_min=120, start_hour=8.0, meals=()),),
        marker_id="glucose", kind=StatisticKind.MEAN_IN_WINDOW,
        window=StatisticWindow(0, 60), target=100.0, sigma=2.0, n_arm=1,
    )
    base.update(kw)
    return CohortStatisticSpec(**base)  # type: ignore[arg-type]


# A discrete population, (value, probability), so the expectation over iid batches is an
# EXACT finite sum rather than a simulation. mu = -0.1, between-person variance v = 1.29.
_POP = ((-1.0, 0.5), (0.0, 0.3), (2.0, 0.2))
_POP_MEAN = sum(x * p for x, p in _POP)
_POP_VAR = sum(x * x * p for x, p in _POP) - _POP_MEAN ** 2


def _expect_over_batches(n_draws: int, fn, fixed: tuple[float, ...] = ()) -> float:
    """E[fn(batch)] over iid draws from ``_POP``, with ``fixed`` members appended last."""
    total = 0.0
    for combo in product(range(len(_POP)), repeat=n_draws):
        p = 1.0
        for i in combo:
            p *= _POP[i][1]
        batch = torch.tensor([_POP[i][0] for i in combo] + list(fixed), dtype=torch.float64)
        total += p * float(fn(batch))
    return total


class TestDebiasedBatchStatistic(unittest.TestCase):
    """PLAN A7: the batch-mean loss must not reward between-person shrinkage.

    Iter 97 scored ``((mean - target) / sem)^2``, whose expectation is
    ``bias^2 + v/B`` in sem units: the ``v/B`` part falls when members are made
    more alike. ``score_batch_statistic`` now subtracts ``s^2/B`` and floors at 0.
    """

    def test_estimate_is_unbiased_for_squared_bias_and_the_old_form_is_not(self) -> None:
        spec = _point_spec(target=0.7)  # bias = mu - target = -0.8, bias^2 = 0.64
        bias_sq = (_POP_MEAN - 0.7) ** 2
        self.assertAlmostEqual(bias_sq, 0.64, places=12)
        self.assertAlmostEqual(_POP_VAR, 1.29, places=12)
        for B in (2, 3, 4):
            old = _expect_over_batches(B, lambda x: _debiased_sq_residual(x, spec)[1] ** 2)
            new = _expect_over_batches(B, lambda x: _debiased_sq_residual(x, spec)[2])
            # E[resid^2] = bias^2 + v/B   (the shrinkage term iter 97 left in) ...
            self.assertAlmostEqual(old, bias_sq + _POP_VAR / B, places=10, msg=f"B={B}")
            # ... and E[resid^2 - s^2/B] = bias^2 exactly.
            self.assertAlmostEqual(new, bias_sq, places=10, msg=f"B={B}")

    def test_estimate_is_the_mean_residual_product_over_distinct_pairs(self) -> None:
        spec = _point_spec(target=3.0)
        g = torch.Generator().manual_seed(0)
        for B in (2, 3, 5, 8):
            x = torch.randn(B, generator=g, dtype=torch.float64) * 4.0 + 1.0
            a = x - spec.target
            pair_mean = sum(
                float(a[i] * a[j]) for i in range(B) for j in range(B) if i != j
            ) / (B * (B - 1))
            self.assertAlmostEqual(float(_debiased_sq_residual(x, spec)[2]), pair_mean, places=10)
        # B = 2: the product of the two residuals, positive only on the same side of the target.
        est = _debiased_sq_residual(torch.tensor([103.0, 91.0]), _point_spec())[2]
        self.assertAlmostEqual(float(est), (103.0 - 100.0) * (91.0 - 100.0), places=4)

    def test_batch_mean_off_by_its_own_standard_error_is_not_charged(self) -> None:
        # Zero true bias is a property of the POPULATION; any one batch's mean is off by
        # sampling noise. Here members sit at -20/+20/-10/+10 around a mean of 108, so
        # resid = 8 against s/sqrt(B) = 9.13: the batch cannot be told from an on-target
        # population. sem = sigma = 2 (a published standard error, n_arm = 1).
        #   iter-97 batch-mean form: (8 / 2)^2 = 16.0     debiased: 0.0
        pred = torch.tensor([88.0, 128.0, 98.0, 118.0])
        spec = _point_spec(target=100.0, sigma=2.0, n_arm=1)
        old = float(((pred.mean() - spec.target) / spec.sigma) ** 2)
        loss, mean, z = score_batch_statistic(pred, spec)
        self.assertAlmostEqual(old, 16.0, places=5)
        self.assertEqual(float(loss), 0.0)
        self.assertAlmostEqual(mean, 108.0, places=4)
        self.assertAlmostEqual(z, 4.0, places=5)  # z stays the plain residual over sigma
        self.assertLess(float(_debiased_sq_residual(pred, spec)[2]), 0.0)  # what the floor catches

    def test_floor_keeps_the_loss_non_negative_when_the_mean_is_on_target(self) -> None:
        # Mean exactly on target, members disagree: the unfloored estimate is -s^2/B
        # (s^2 = 140, B = 6 -> -23.33); the loss must be 0, never negative.
        pred = torch.tensor([85.0, 115.0, 90.0, 110.0, 95.0, 105.0], requires_grad=True)
        spec = _point_spec(target=100.0, sigma=10.0, n_arm=None)
        mean, resid, est = _debiased_sq_residual(pred, spec)
        self.assertAlmostEqual(float(est.detach()), -140.0 / 6.0, places=4)
        loss, _, _ = score_batch_statistic(pred, spec)
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        assert pred.grad is not None
        self.assertEqual(float(pred.grad.abs().sum()), 0.0)  # "indistinguishable" -> no gradient

    def test_loss_rises_with_bias_at_fixed_spread(self) -> None:
        spread = torch.tensor([-20.0, 20.0, -10.0, 10.0])  # mean 0, s^2/B = 1000/3/4 = 83.33
        spec = _point_spec(target=100.0, sigma=2.0, n_arm=1)
        dead_zone = (1000.0 / 3.0 / 4.0) ** 0.5  # 9.13: |shift| below this is within the noise

        def loss_at(shift: float) -> float:
            return float(score_batch_statistic(100.0 + shift + spread, spec)[0])

        shifts = [0.0, 3.0, 6.0, 9.0, 10.0, 12.0, 16.0, 24.0, 40.0]
        losses = [loss_at(s) for s in shifts]
        for lo, hi in zip(losses, losses[1:]):
            self.assertLessEqual(lo, hi)
        for s, l in zip(shifts, losses):
            if s <= dead_zone:
                self.assertEqual(l, 0.0, f"shift {s} is inside the noise")
            else:
                # (shift^2 - s^2/B) / sem^2: the squared bias, less its sampling noise
                self.assertAlmostEqual(l, (s * s - 1000.0 / 12.0) / 4.0, places=3)
        for lo, hi in zip(losses[4:], losses[5:]):
            self.assertLess(lo, hi)  # strictly increasing once past the dead zone
        for s in shifts:
            self.assertAlmostEqual(loss_at(s), loss_at(-s), places=4)  # sign-symmetric

    def test_live_zone_gradient_is_the_unbiased_estimators(self) -> None:
        # d(resid^2 - s^2/B)/dx_i = 2 resid / B - 2 (x_i - mean) / (B (B - 1)): the second term
        # pushes members apart, cancelling in expectation the shrinkage that resid^2 alone rewards.
        spec = _point_spec(target=100.0, sigma=2.0, n_arm=1)
        x = torch.tensor([88.0, 128.0, 98.0, 118.0], dtype=torch.float64) + 20.0  # resid = 28, live
        x.requires_grad_(True)
        loss, _, _ = score_batch_statistic(x, spec)
        loss.backward()
        B, resid = 4, float(x.mean().detach()) - 100.0
        expected = (2 * resid / B - 2 * (x.detach() - x.detach().mean()) / (B * (B - 1))) / 4.0
        assert x.grad is not None
        self.assertTrue(torch.allclose(x.grad, expected, atol=1e-9), (x.grad, expected))

    def test_floor_keeps_the_documented_share_of_the_shrinkage_term(self) -> None:
        # The docstring's measured caveat: with the population exactly on target, B = 2 Gaussian
        # members, the floored loss still averages (1/pi) v / sem^2 -- 2/pi = 0.64 of the old
        # v / (B sem^2) = 0.5 -- while the unfloored estimate averages 0. Fixed seed; the
        # tolerances are >= 4 standard errors of the 4000-batch mean, so a different torch RNG
        # stream cannot fail this by chance.
        spec = _point_spec(target=0.0, sigma=1.0, n_arm=1)  # v = 1, sem = 1
        g = torch.Generator().manual_seed(0)
        batches = torch.randn(4000, 2, generator=g, dtype=torch.float64)
        old = floored = unfloored = 0.0
        for x in batches:
            _, resid, est = _debiased_sq_residual(x, spec)
            old += float(resid ** 2)
            unfloored += float(est)
            floored += float(score_batch_statistic(x, spec)[0])
        n = float(len(batches))
        self.assertAlmostEqual(old / n, 0.5, delta=0.05)
        self.assertAlmostEqual(unfloored / n, 0.0, delta=0.07)
        self.assertAlmostEqual(floored / n, 1.0 / 3.141592653589793, delta=0.04)
        self.assertAlmostEqual((floored / old), 2.0 / 3.141592653589793, delta=0.08)

    def test_shrinkage_force_that_survives_the_floor_is_the_documented_fraction(self) -> None:
        # The quantity that matters for training is the FORCE, d E[loss] / d(sigma_between): what
        # pulls members together. Un-debiased at B = 2 it is sigma / sem^2 (E[loss] = sigma^2 / 2);
        # the floored debias keeps 2/pi = 0.64 of it at zero bias, 0.56 at one between-person sd of
        # bias and 0.03 at three (docstring; simulated). Members are sigma * eps around the
        # population mean, so d/d(sigma) by autograd is the force. 4000 fixed batches; the
        # tolerance is >= 4 standard errors.
        spec = _point_spec(target=0.0, sigma=1.0, n_arm=1)  # sem = 1, population sd = sigma = 1
        g = torch.Generator().manual_seed(1)
        eps = torch.randn(4000, 2, generator=g, dtype=torch.float64)
        sigma = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        (sum((sigma * e).mean() ** 2 for e in eps)).backward()
        self.assertAlmostEqual(float(sigma.grad) / len(eps), 1.0, delta=0.1)  # the un-debiased force
        # ... and the status quo this replaces: two draws plus the zero embedding held AT the
        # population mean, E[loss] = 2 sigma^2 / 9, force 4/9 -- BELOW the floored 0.64 above.
        sigma = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        (sum(torch.cat([sigma * e, torch.zeros(1, dtype=torch.float64)]).mean() ** 2 for e in eps)).backward()
        self.assertAlmostEqual(float(sigma.grad) / len(eps), 4.0 / 9.0, delta=0.05)
        for bias, expected, tol in ((0.0, 0.64, 0.08), (1.0, 0.56, 0.08), (3.0, 0.03, 0.06)):
            sigma = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
            total = sum(score_batch_statistic(bias + sigma * e, spec)[0] for e in eps)
            total.backward()
            force = float(sigma.grad) / len(eps)
            self.assertAlmostEqual(force, expected, delta=tol, msg=f"bias={bias} sd")

    def test_one_sided_targets_are_not_charged_inside_the_noise(self) -> None:
        at_most = _point_spec(shape=TargetShape.AT_MOST, target=140.0, sigma=10.0, n_arm=None)
        # allowed side, with spread: 0 (as before)
        self.assertEqual(float(score_batch_statistic(torch.tensor([100.0, 130.0]), at_most)[0]), 0.0)
        # forbidden side by more than its own spread explains: charged
        self.assertGreater(float(score_batch_statistic(torch.tensor([150.0, 190.0]), at_most)[0]), 0.0)
        # forbidden side (mean 150, resid 10) but members 100 apart: indistinguishable -> 0
        self.assertEqual(float(score_batch_statistic(torch.tensor([100.0, 200.0]), at_most)[0]), 0.0)
        at_least = _point_spec(shape=TargetShape.AT_LEAST, target=60.0, sigma=10.0, n_arm=None)
        self.assertEqual(float(score_batch_statistic(torch.tensor([30.0, 90.0]), at_least)[0]), 0.0)
        self.assertGreater(float(score_batch_statistic(torch.tensor([40.0, 44.0]), at_least)[0]), 0.0)

    def test_identical_members_reproduce_the_iter97_loss_and_gradient(self) -> None:
        # No spread, no correction: s^2 = 0 and its gradient is 0, so callers whose members
        # coincide (cohort_ablation scores zero-initialised embeddings) see the iter-97 loss exactly.
        from pulse.cohort_loss import shaped_residual

        specs = (
            _point_spec(sigma=7.0, n_arm=None),
            _point_spec(sigma=7.0, n_arm=1),
            _point_spec(sigma=7.0, n_arm=None, shape=TargetShape.AT_MOST, target=140.0),
            _point_spec(sigma=7.0, n_arm=None, shape=TargetShape.BAND, band_halfwidth=5.0),
        )
        for spec in specs:
            for B in (1, 2, 3, 5):
                x = torch.full((B,), 123.4, dtype=torch.float64, requires_grad=True)
                new, _, _ = score_batch_statistic(x, spec)
                (g_new,) = torch.autograd.grad(new, x)
                n = int(spec.n_arm) if spec.n_arm is not None else B
                old = (shaped_residual(x.mean(), spec) / (spec.sigma / n ** 0.5)).pow(2)
                (g_old,) = torch.autograd.grad(old, x)
                self.assertAlmostEqual(float(new.detach()), float(old.detach()), places=10, msg=f"{spec.shape} B={B}")
                self.assertTrue(torch.allclose(g_new, g_old, rtol=1e-10, atol=1e-12), f"{spec.shape} B={B}")

    def test_single_member_batch_cannot_separate_bias_from_spread(self) -> None:
        spec = _point_spec(target=100.0, sigma=2.0, n_arm=1)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # var() of one member warns; it must not be called
            loss, mean, z = score_batch_statistic(torch.tensor([104.0]), spec)
            first_only, _, _ = score_batch_statistic(torch.tensor([104.0, 900.0]), spec, n_population=1)
        self.assertAlmostEqual(float(loss), (4.0 / 2.0) ** 2, places=5)  # the pre-A7 form
        self.assertAlmostEqual(float(first_only), float(loss), places=5)
        self.assertAlmostEqual(mean, 104.0, places=5)
        self.assertAlmostEqual(z, 2.0, places=5)


class TestPopulationMembers(unittest.TestCase):
    """PLAN A7: ``n_population`` scores the mean over the first n members only."""

    def test_mean_s2_and_n_are_taken_over_the_first_n_members(self) -> None:
        spec = _point_spec(target=80.0, sigma=10.0, n_arm=None)
        pred = torch.tensor([90.0, 110.0, 1000.0], requires_grad=True)
        loss, mean, z = score_batch_statistic(pred, spec, n_population=2)
        # members 90, 110: mean 100, resid 20, s^2 = 200 -> 400 - 200/2 = 300; n = 2 -> sem^2 = 50
        self.assertAlmostEqual(float(loss.detach()), 300.0 / 50.0, places=5)
        self.assertAlmostEqual(mean, 100.0, places=5)
        self.assertAlmostEqual(z, 2.0, places=5)
        same = score_batch_statistic(torch.tensor([90.0, 110.0]), spec)
        self.assertAlmostEqual(float(loss.detach()), float(same[0]), places=6)
        loss.backward()
        assert pred.grad is not None
        self.assertEqual(float(pred.grad[2]), 0.0)  # the excluded member is not read at all
        self.assertGreater(float(pred.grad[:2].abs().sum()), 0.0)

    def test_excluded_member_value_is_irrelevant(self) -> None:
        spec = _point_spec(target=80.0, sigma=10.0, n_arm=None)
        a = score_batch_statistic(torch.tensor([90.0, 110.0, -5.0]), spec, n_population=2)
        b = score_batch_statistic(torch.tensor([90.0, 110.0, 1.0e6]), spec, n_population=2)
        self.assertEqual(float(a[0]), float(b[0]))
        self.assertEqual(a[1:], b[1:])

    def test_default_none_and_zero_score_every_member(self) -> None:
        spec = _point_spec(target=80.0, sigma=10.0, n_arm=None)
        pred = torch.tensor([90.0, 110.0, 1000.0])
        everyone = score_batch_statistic(pred, spec)
        for n in (None, 0, 3):
            got = score_batch_statistic(pred, spec, n_population=n)
            self.assertEqual(float(got[0]), float(everyone[0]), f"n_population={n}")
            self.assertEqual(got[1:], everyone[1:])
        # n_arm pins n regardless of how many members are scored
        pinned = _point_spec(target=80.0, sigma=10.0, n_arm=1)
        self.assertAlmostEqual(
            float(score_batch_statistic(pred, pinned, n_population=2)[0]), 300.0 / 100.0, places=5,
        )

    def test_out_of_range_n_population_is_rejected(self) -> None:
        spec = _point_spec()
        pred = torch.tensor([90.0, 110.0, 120.0])
        for bad in (-1, 4):
            with self.assertRaises(ValueError):
                score_batch_statistic(pred, spec, n_population=bad)

    def test_including_a_fixed_member_biases_the_estimate_by_two_bias_d_over_B(self) -> None:
        # Why the zero embedding must leave the mean: with k iid draws plus one FIXED member
        # d away from the population mean (the median person's value; below it for a lognormal
        # outcome), E[resid^2 - s^2/B] is bias^2 + 2 bias d / B, not bias^2. Exact over the
        # discrete population, B = 3.
        spec = _point_spec(target=0.7)
        bias = _POP_MEAN - 0.7
        fixed = -0.6
        d = fixed - _POP_MEAN
        got = _expect_over_batches(2, lambda x: _debiased_sq_residual(x, spec)[2], fixed=(fixed,))
        self.assertAlmostEqual(got, bias ** 2 + 2 * bias * d / 3, places=10)
        # ... and Var(mean) = k v / B^2 but E[s^2 / B] = (k v + d^2) / B^2
        v = _POP_VAR
        old = _expect_over_batches(2, lambda x: _debiased_sq_residual(x, spec)[1] ** 2, fixed=(fixed,))
        self.assertAlmostEqual(old, (bias + d / 3) ** 2 + 2 * v / 9, places=10)
        s2 = _expect_over_batches(2, lambda x: x.var(correction=1), fixed=(fixed,))
        self.assertAlmostEqual(s2 / 3, (2 * v + d * d) / 9, places=10)


if __name__ == "__main__":
    unittest.main()
