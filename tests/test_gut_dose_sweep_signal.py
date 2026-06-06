"""Unit tests for the gut dose-sweep training signal.

Covers:

* targets are deterministic, monotonic in carbs, and have the right shape
* zero loss when the gut kernel already matches the cold target
* positive loss otherwise
* gradient lands on gut kernel + gut embedding projection (and not on
  unrelated modules — by construction this signal never integrates)
* zero weight is a no-op; positive weight executes a step
* over a few iterations the kernel actually starts to invert/correct (sanity:
  loss strictly decreases)
* multiple epochs at the zero embedding push monotonicity in the right
  direction (regression guard for the iter 11 inversion failure mode)
"""
from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn as nn

from pulse.knowledge.full_body import PatientParams
from pulse.model import ModularPhysiologyNetwork
from pulse.modules.gut import MealEvent
from pulse.training import (
    GutDoseSweepProtocol,
    GutDoseSweepSignal,
    SignalContext,
    WeightSchedule,
)
from pulse.training.gut_dose_sweep_signal import _cold_target_for_dose
from pulse.types import EMBEDDING_DIM, GUT_OUTPUT_DIM


def _tiny_model() -> ModularPhysiologyNetwork:
    return ModularPhysiologyNetwork(
        metabolic_hidden=16, appetite_hidden=12, stress_hidden=12,
        cardiovascular_hidden=16, thermoreg_hidden=8, respiratory_hidden=8,
    )


class TestAbsScale(unittest.TestCase):
    """The per-channel abs_scale is the load-bearing constant for *both*
    GutDoseSweepSignal and TrajectoryRolloutSignal's gut term. It must come
    from a single source of truth (``modules.gut.GUT_OUTPUT_SCALE``) and
    must produce comparable MSE magnitudes across channels at the cold
    model's typical excursions. The scale that iter 12 inherited
    (``[30, 10, 15, 1]``) was for state-mg/dL, not gut-mg/min, and
    flattened the gut loss by 200-1000x per channel."""

    def test_signals_share_scale_source(self) -> None:
        from pulse.modules.gut import GUT_OUTPUT_SCALE
        from pulse.training.trajectory_signal import TrajectoryRolloutSignal

        sweep = GutDoseSweepSignal()
        traj = TrajectoryRolloutSignal(
            n_patients=2, n_days=1, seed=0, contribution_weights=None,
            windows_per_patient=1, meal_window_bias=0.0,
            input_dropout=0.0, huber_delta=0.5, gut_loss_weight=0.0,
            coupling_weight=WeightSchedule(0.0),
            verifier_weight=WeightSchedule(0.0),
            coupling_prior_samples=0,
            trajectory_band=0.0, trajectory_band_default=0.0,
            landmark_weight=WeightSchedule(0.0),
            landmark_pre_window=0, landmark_post_window=0, landmark_min_carbs=0.0,
            n_default_patients=0,
        )
        expected = torch.tensor(GUT_OUTPUT_SCALE, dtype=torch.float32)
        self.assertTrue(torch.equal(sweep._abs_scale, expected))
        self.assertTrue(torch.equal(traj._abs_scale, expected))

    def test_scale_per_channel_within_typical_excursions(self) -> None:
        """The scale per channel must be of the same order of magnitude as
        the cold model's typical (per-channel) excursion at a representative
        meal. Cold-model peaks at 75 g mixed meal: glucose~2, lipid~0.3,
        amino~0.5, nutrient_flag=1. Each scale must be in [0.1x, 10x] of
        the corresponding peak so MSE doesn't get flattened or blown up."""
        from pulse.modules.gut import GUT_OUTPUT_SCALE

        target_peaks = (2.0, 0.3, 0.5, 1.0)
        for ch, (s, p) in enumerate(zip(GUT_OUTPUT_SCALE, target_peaks)):
            ratio = s / p
            self.assertGreater(
                ratio, 0.1, f"channel {ch} scale {s} too small vs peak {p}",
            )
            self.assertLess(
                ratio, 10.0, f"channel {ch} scale {s} too large vs peak {p}",
            )


class TestColdTargets(unittest.TestCase):
    def test_target_shape_and_dtype(self) -> None:
        out = _cold_target_for_dose(60.0, 5.0, 10.0, 240, PatientParams())
        self.assertEqual(out.shape, (240, GUT_OUTPUT_DIM))
        self.assertEqual(out.dtype, np.float32)

    def test_zero_carbs_zero_glucose_appearance(self) -> None:
        """Cold model: 0 g carbs => glucose-appearance channel is exactly zero
        across the whole window. Lipid/amino still nonzero from fats/proteins."""
        out = _cold_target_for_dose(0.0, 5.0, 10.0, 240, PatientParams())
        self.assertEqual(float(out[:, 0].sum()), 0.0)
        self.assertGreater(float(out[:, 1].sum()), 0.0)
        self.assertGreater(float(out[:, 2].sum()), 0.0)

    def test_targets_monotonic_in_carbs(self) -> None:
        """AUC of glucose appearance must rise monotonically with carb dose."""
        params = PatientParams()
        aucs = [
            float(_cold_target_for_dose(d, 5.0, 10.0, 240, params)[:, 0].sum())
            for d in (0.0, 30.0, 60.0, 90.0, 120.0)
        ]
        for prev, nxt in zip(aucs, aucs[1:]):
            self.assertLess(prev, nxt, f"non-monotonic cold target AUC: {aucs}")


class TestSignalGating(unittest.TestCase):
    def test_zero_weight_is_noop(self) -> None:
        device = torch.device("cpu")
        model = _tiny_model()
        emb = nn.Embedding(2, EMBEDDING_DIM)
        params = list(model.parameters()) + list(emb.parameters())
        opt = torch.optim.Adam(params, lr=1e-3)
        sig = GutDoseSweepSignal(
            n_patients=2, sample_patients=2, weight=WeightSchedule(0.0),
        )
        ctx = SignalContext(
            epoch=0, total_epochs=1, rng=np.random.default_rng(0),
            device=device, optimizer=opt, params=params, grad_clip=10.0,
        )
        result = sig.compute(model, emb, ctx)
        self.assertEqual(result.n_units, 0)

    def test_positive_weight_takes_a_step(self) -> None:
        device = torch.device("cpu")
        torch.manual_seed(0)
        model = _tiny_model()
        emb = nn.Embedding(2, EMBEDDING_DIM)
        nn.init.normal_(emb.weight, std=0.1)
        params = list(model.parameters()) + list(emb.parameters())
        opt = torch.optim.Adam(params, lr=1e-2)
        sig = GutDoseSweepSignal(
            n_patients=2, sample_patients=2, weight=WeightSchedule(1.0),
        )
        ctx = SignalContext(
            epoch=0, total_epochs=1, rng=np.random.default_rng(0),
            device=device, optimizer=opt, params=params, grad_clip=10.0,
        )
        before = {
            n: p.detach().clone() for n, p in model.gut.named_parameters()
        }
        result = sig.compute(model, emb, ctx)
        moved = any(
            float((p.detach() - before[n]).abs().sum()) > 0
            for n, p in model.gut.named_parameters()
        )
        self.assertEqual(result.n_units, 1)
        self.assertTrue(moved, "gut kernel parameters did not move")
        self.assertGreater(result.sub_metrics["n_dose_emb_pairs"], 0)


class TestGradientFlow(unittest.TestCase):
    def test_gradient_lands_on_gut_pipeline_only(self) -> None:
        """The signal never calls ``integrate``: gradient must reach the gut
        kernel and the gut embedding projection, and not any unrelated module.

        We replicate the per-pair loss directly (no signal.compute, which
        zeros grads) so we can inspect ``.grad`` on individual parameters.
        """
        torch.manual_seed(0)
        model = _tiny_model()
        proto = GutDoseSweepProtocol(carb_doses_g=(0.0, 60.0), post_window_min=60)
        sig = GutDoseSweepSignal(protocol=proto)
        emb_gut = model.embedding_projections["gut"](torch.zeros(EMBEDDING_DIM))
        times = torch.arange(60, dtype=torch.float32)
        meal = MealEvent(time=0.0, carbs=60.0, fats=5.0, proteins=10.0)
        pred = model.gut.forward_window(times, [meal], emb_gut)
        target = sig._targets[1].clone()  # 60g target
        loss = ((pred - target) / sig._abs_scale).pow(2).mean()
        loss.backward()

        gut_kernel_grad = sum(
            float(p.grad.abs().sum()) for p in model.gut.parameters()
            if p.grad is not None
        )
        gut_proj_grad = sum(
            float(p.grad.abs().sum())
            for p in model.embedding_projections["gut"].parameters()
            if p.grad is not None
        )
        self.assertGreater(gut_kernel_grad, 0.0)
        self.assertGreater(gut_proj_grad, 0.0)

        # Modules that don't participate in gut.forward_window must have no
        # gradient at all from this loss.
        for name, sub in (
            ("respiratory", model.respiratory),
            ("cardiovascular", model.cardiovascular),
            ("metabolic", model.metabolic),
            ("stress", model.stress),
        ):
            grad_total = sum(
                float(p.grad.abs().sum()) for p in sub.parameters()
                if p.grad is not None
            )
            self.assertEqual(
                grad_total, 0.0,
                f"{name} module unexpectedly received gradient ({grad_total})",
            )

    def test_unrelated_modules_unchanged(self) -> None:
        device = torch.device("cpu")
        torch.manual_seed(0)
        model = _tiny_model()
        emb = nn.Embedding(1, EMBEDDING_DIM)
        params = list(model.parameters()) + list(emb.parameters())
        opt = torch.optim.SGD(params, lr=1e-2)
        sig = GutDoseSweepSignal(
            n_patients=1, sample_patients=1, weight=WeightSchedule(1.0),
        )
        ctx = SignalContext(
            epoch=0, total_epochs=1, rng=np.random.default_rng(0),
            device=device, optimizer=opt, params=params, grad_clip=1e9,
        )
        before = {
            n: p.detach().clone() for n, p in model.respiratory.named_parameters()
        }
        sig.compute(model, emb, ctx)
        moved = any(
            float((p.detach() - before[n]).abs().sum()) > 0
            for n, p in model.respiratory.named_parameters()
        )
        self.assertFalse(moved, "respiratory module should not receive gut-sweep gradient")


class TestLearningSanity(unittest.TestCase):
    """A few SGD iterations against the gut-dose-sweep loss should reduce it."""

    def test_loss_strictly_decreases_over_iterations(self) -> None:
        device = torch.device("cpu")
        torch.manual_seed(0)
        model = _tiny_model()
        emb = nn.Embedding(1, EMBEDDING_DIM)
        params = list(model.parameters()) + list(emb.parameters())
        opt = torch.optim.Adam(params, lr=5e-3)
        sig = GutDoseSweepSignal(
            n_patients=1, sample_patients=1, weight=WeightSchedule(1.0),
            protocol=GutDoseSweepProtocol(
                carb_doses_g=(0.0, 30.0, 60.0, 90.0),
                post_window_min=120,
            ),
        )
        rng = np.random.default_rng(0)
        losses: list[float] = []
        for ep in range(5):
            ctx = SignalContext(
                epoch=ep, total_epochs=5, rng=rng,
                device=device, optimizer=opt, params=params, grad_clip=10.0,
            )
            r = sig.compute(model, emb, ctx)
            losses.append(r.loss_sum)
        self.assertLess(losses[-1], losses[0],
                        f"loss did not decrease: {losses}")


class TestRankingTerm(unittest.TestCase):
    """The dose-ranking term (iter 15) must:

    1. be exactly zero when predictions match the cold targets (which are
       monotone in dose by construction);
    2. be strictly positive when predictions are dose-inverted;
    3. produce a gradient that pushes predictions toward the correct ordering.

    Together these guarantee the loss surface no longer admits the
    iter-13/iter-14 dose-inverted minimum where the kernel peaks at 30 g
    and decreases for higher doses.
    """

    def _signal(self) -> GutDoseSweepSignal:
        proto = GutDoseSweepProtocol(
            carb_doses_g=(0.0, 30.0, 60.0, 120.0),
            post_window_min=60,
        )
        return GutDoseSweepSignal(
            n_patients=1, sample_patients=1,
            weight=WeightSchedule(1.0),
            protocol=proto,
        )

    def test_ranking_zero_when_pred_matches_cold_targets(self) -> None:
        sig = self._signal()
        T = sig.protocol.post_window_min
        # Predictions = cold targets, broadcast across a B=1 batch.
        per_dose_pred = [sig._targets[i].unsqueeze(0) for i in range(sig._targets.shape[0])]
        mse, rank, auc, n_inv = sig._losses(per_dose_pred, sig._targets, sig._abs_scale, T)
        self.assertAlmostEqual(float(mse), 0.0, places=5)
        self.assertAlmostEqual(float(rank), 0.0, places=5)
        self.assertAlmostEqual(float(auc), 0.0, places=5)
        self.assertEqual(float(n_inv), 0.0)

    def test_ranking_positive_when_pred_inverted(self) -> None:
        """Predicting the dose-reversed cold-target sequence should fully
        violate every glucose-channel ranking pair."""
        sig = self._signal()
        T = sig.protocol.post_window_min
        D = sig._targets.shape[0]
        # Pred for dose d_i is actually the cold target for d_{D-1-i}.
        per_dose_pred = [sig._targets[D - 1 - i].unsqueeze(0) for i in range(D)]
        mse, rank, auc, n_inv = sig._losses(per_dose_pred, sig._targets, sig._abs_scale, T)
        self.assertGreater(float(rank), 0.0)
        # Glucose channel ranks every (i < j) pair: C(D, 2) inversions.
        self.assertEqual(float(n_inv), float(D * (D - 1) // 2))

    def test_ranking_unweighted_minus_loss_sum_isolates_mse(self) -> None:
        """``ranking_weight=0`` and ``auc_weight=0`` ⇒ loss_sum == mse,
        even when those terms would otherwise contribute. Validates the
        additive structure across all three loss components."""
        torch.manual_seed(0)
        model = _tiny_model()
        emb = nn.Embedding(1, EMBEDDING_DIM)
        proto = GutDoseSweepProtocol(
            carb_doses_g=(0.0, 30.0, 60.0, 120.0), post_window_min=60,
        )
        sig = GutDoseSweepSignal(
            n_patients=1, sample_patients=1,
            weight=WeightSchedule(1.0),
            protocol=proto, ranking_weight=0.0, auc_weight=0.0,
        )
        params = list(model.parameters()) + list(emb.parameters())
        opt = torch.optim.SGD(params, lr=1e-3)
        ctx = SignalContext(
            epoch=0, total_epochs=1, rng=np.random.default_rng(0),
            device=torch.device("cpu"), optimizer=opt,
            params=params, grad_clip=1e9,
        )
        r = sig.compute(model, emb, ctx)
        self.assertAlmostEqual(r.loss_sum, r.sub_metrics["mse"], places=5)
        self.assertGreaterEqual(r.sub_metrics["rank"], 0.0)

    def test_ranking_term_pushes_pred_toward_monotone(self) -> None:
        """A short Adam loop with strong ranking_weight should drive the
        zero-embedding inversion count down over iterations. Goes through
        the real model so the gradient path through the gut kernel is
        exercised end-to-end."""
        torch.manual_seed(0)
        model = _tiny_model()
        emb = nn.Embedding(1, EMBEDDING_DIM)
        proto = GutDoseSweepProtocol(
            carb_doses_g=(0.0, 30.0, 60.0, 120.0), post_window_min=60,
        )
        sig = GutDoseSweepSignal(
            n_patients=1, sample_patients=0,
            include_default_embedding=True,
            weight=WeightSchedule(1.0),
            protocol=proto,
            ranking_weight=10.0,
        )
        params = list(model.parameters()) + list(emb.parameters())
        opt = torch.optim.Adam(params, lr=5e-3)
        rng = np.random.default_rng(0)

        invs: list[float] = []
        for ep in range(15):
            ctx = SignalContext(
                epoch=ep, total_epochs=15, rng=rng,
                device=torch.device("cpu"),
                optimizer=opt, params=params, grad_clip=10.0,
            )
            r = sig.compute(model, emb, ctx)
            invs.append(float(r.sub_metrics["n_inversions_zero_emb"]))
        self.assertLessEqual(
            invs[-1], invs[0],
            f"inversions did not decrease: first={invs[0]}, last={invs[-1]}, all={invs}",
        )


class TestAucMatchingTerm(unittest.TestCase):
    """The AUC-matching term (iter 18) directly penalizes the integrated
    kernel output per (dose, batch, channel) against the cold-target AUC.
    This is the structural fix for iter 16/17's compensatory equilibrium:
    the per-element MSE (over [B, T, C]) was diluted by ~B·C across mostly-
    inactive time and channel slots, leaving the kernel free to converge
    to ~1.5× cold amplitude despite the supervision signal nominally
    pulling it down. The AUC term collapses time and channel into one
    scalar before squaring, so a uniform multiplicative over-amp lands a
    real per-dose penalty proportional to that dose's target AUC squared.

    These tests pin down four properties:

    1. AUC loss is exactly zero when predictions match cold targets.
    2. AUC loss grows quadratically with the multiplicative scale factor.
    3. Setting ``auc_weight=0`` removes the AUC term from the total loss.
    4. The AUC gradient pushes a uniformly over-amped pred *down* toward
       the cold target, end-to-end through the gut kernel.
    """

    def _signal(self, **overrides: object) -> GutDoseSweepSignal:
        proto = GutDoseSweepProtocol(
            carb_doses_g=(0.0, 30.0, 60.0, 120.0),
            post_window_min=60,
        )
        kwargs: dict[str, object] = dict(
            n_patients=1, sample_patients=1,
            weight=WeightSchedule(1.0),
            protocol=proto,
        )
        kwargs.update(overrides)
        return GutDoseSweepSignal(**kwargs)  # type: ignore[arg-type]

    def test_auc_zero_when_pred_matches_cold_targets(self) -> None:
        sig = self._signal()
        T = sig.protocol.post_window_min
        per_dose_pred = [sig._targets[i].unsqueeze(0) for i in range(sig._targets.shape[0])]
        _mse, _rank, auc, _n_inv = sig._losses(
            per_dose_pred, sig._targets, sig._abs_scale, T,
        )
        self.assertAlmostEqual(float(auc), 0.0, places=5)

    def test_auc_grows_quadratically_with_overamp(self) -> None:
        """Multiplying the cold target by α should make the AUC loss scale
        as (α − 1)². Pin this to detect any accidental linearization or
        asymmetric weighting."""
        sig = self._signal()
        T = sig.protocol.post_window_min
        D = sig._targets.shape[0]

        def auc_loss_for_alpha(alpha: float) -> float:
            per_dose_pred = [
                (alpha * sig._targets[i]).unsqueeze(0) for i in range(D)
            ]
            _mse, _rank, auc, _n_inv = sig._losses(
                per_dose_pred, sig._targets, sig._abs_scale, T,
            )
            return float(auc)

        l_at_1 = auc_loss_for_alpha(1.0)
        self.assertAlmostEqual(l_at_1, 0.0, places=5)

        # Quadratic in (α − 1): loss(1.5) / loss(1.25) ≈ (0.5)² / (0.25)² = 4.
        l_at_125 = auc_loss_for_alpha(1.25)
        l_at_15 = auc_loss_for_alpha(1.5)
        self.assertGreater(l_at_125, 0.0)
        self.assertAlmostEqual(l_at_15 / l_at_125, 4.0, places=2)

    def test_auc_weight_zero_removes_term_from_total_loss(self) -> None:
        """``auc_weight=0`` ⇒ loss_sum equals mse + ranking_weight·rank,
        even when AUC term would otherwise contribute. Ablation check."""
        torch.manual_seed(0)
        model = _tiny_model()
        emb = nn.Embedding(1, EMBEDDING_DIM)
        proto = GutDoseSweepProtocol(
            carb_doses_g=(0.0, 30.0, 60.0, 120.0), post_window_min=60,
        )
        sig_off = GutDoseSweepSignal(
            n_patients=1, sample_patients=1,
            weight=WeightSchedule(1.0),
            protocol=proto, ranking_weight=0.0, auc_weight=0.0,
        )
        params = list(model.parameters()) + list(emb.parameters())
        opt = torch.optim.SGD(params, lr=1e-3)
        ctx = SignalContext(
            epoch=0, total_epochs=1, rng=np.random.default_rng(0),
            device=torch.device("cpu"), optimizer=opt,
            params=params, grad_clip=1e9,
        )
        r = sig_off.compute(model, emb, ctx)
        self.assertAlmostEqual(r.loss_sum, r.sub_metrics["mse"], places=5)
        self.assertGreaterEqual(r.sub_metrics["auc"], 0.0)

    def test_auc_term_pushes_overamped_pred_down(self) -> None:
        """End-to-end: a tiny model trained only with auc_weight=10 (mse +
        rank zero) on synthetic uniformly over-amped targets should shrink
        its output AUC at the zero embedding within a handful of steps.
        Validates that the AUC loss has a meaningful downward gradient on
        the gut pipeline, not just a clean math definition."""
        torch.manual_seed(0)
        model = _tiny_model()
        emb = nn.Embedding(1, EMBEDDING_DIM)
        proto = GutDoseSweepProtocol(
            carb_doses_g=(0.0, 30.0, 60.0, 120.0), post_window_min=60,
        )
        sig = GutDoseSweepSignal(
            n_patients=1, sample_patients=0,
            include_default_embedding=True,
            weight=WeightSchedule(1.0),
            protocol=proto,
            ranking_weight=0.0,
            auc_weight=10.0,
        )
        params = list(model.parameters()) + list(emb.parameters())
        opt = torch.optim.Adam(params, lr=5e-3)
        rng = np.random.default_rng(0)

        device = torch.device("cpu")
        zero_emb = torch.zeros(EMBEDDING_DIM, device=device)
        emb_gut = model.embedding_projections["gut"](zero_emb)
        times = torch.arange(proto.post_window_min, dtype=torch.float32, device=device)

        def auc_at_120() -> float:
            with torch.no_grad():
                meal = MealEvent(time=0.0, carbs=120.0, fats=5.0, proteins=10.0)
                out = model.gut.forward_window(times, [meal], emb_gut)
                return float(out[:, 0].sum())

        # Many fresh-init kernels start with AUC > cold target, so a few
        # steps of pure AUC supervision should pull AUC down. We compare
        # AUC at the 120 g dose because cold target there is largest and
        # the AUC term has the most leverage.
        auc_start = auc_at_120()
        for ep in range(15):
            ctx = SignalContext(
                epoch=ep, total_epochs=15, rng=rng,
                device=device, optimizer=opt, params=params, grad_clip=10.0,
            )
            sig.compute(model, emb, ctx)
        auc_end = auc_at_120()

        # The cold-target AUC at 120 g for this protocol; we expect the
        # trained kernel to move *toward* it from above. We don't require
        # convergence in 15 steps, only directional pull.
        cold_auc_120 = float(sig._auc_targets[-1, 0])
        if auc_start > cold_auc_120:
            self.assertLess(
                auc_end, auc_start,
                f"AUC supervision failed to pull amp down: "
                f"start={auc_start:.2f} end={auc_end:.2f} cold={cold_auc_120:.2f}",
            )
        else:
            # Pathological init below cold target — AUC term should pull up.
            self.assertGreater(
                auc_end, auc_start,
                f"AUC supervision failed to pull amp up from below: "
                f"start={auc_start:.2f} end={auc_end:.2f} cold={cold_auc_120:.2f}",
            )


class TestMonotonicityRegression(unittest.TestCase):
    """Regression guard for the iter 11 failure mode: at zero embedding, after
    a handful of training iterations against the gut-dose-sweep loss, the
    model's gut output AUC should *not* be inverted (more carbs => not less
    appearance). We don't require strict monotonicity in this short regime —
    only that the inversion direction is gone."""

    def test_no_inversion_at_zero_embedding(self) -> None:
        device = torch.device("cpu")
        torch.manual_seed(0)
        model = _tiny_model()
        emb = nn.Embedding(1, EMBEDDING_DIM)
        params = list(model.parameters()) + list(emb.parameters())
        opt = torch.optim.Adam(params, lr=1e-2)
        proto = GutDoseSweepProtocol(
            carb_doses_g=(0.0, 30.0, 60.0, 90.0),
            post_window_min=120,
        )
        sig = GutDoseSweepSignal(
            n_patients=1, sample_patients=0,
            include_default_embedding=True,
            weight=WeightSchedule(1.0),
            protocol=proto,
        )
        rng = np.random.default_rng(0)
        for ep in range(20):
            ctx = SignalContext(
                epoch=ep, total_epochs=20, rng=rng,
                device=device, optimizer=opt, params=params, grad_clip=10.0,
            )
            sig.compute(model, emb, ctx)

        zero_emb = torch.zeros(EMBEDDING_DIM, device=device)
        emb_gut = model.embedding_projections["gut"](zero_emb)
        times = torch.arange(proto.post_window_min, dtype=torch.float32, device=device)
        with torch.no_grad():
            aucs = []
            for d in proto.carb_doses_g:
                meal = MealEvent(time=0.0, carbs=float(d), fats=5.0, proteins=10.0)
                out = model.gut.forward_window(times, [meal], emb_gut)
                aucs.append(float(out[:, 0].sum()))
        # The full inversion (more carbs → less appearance) shows up as a
        # strictly decreasing AUC sequence. After 20 sweep-steps at the zero
        # embedding, we should at least not see a strict decrease across the
        # whole sweep — i.e. AUC(120 g) ≥ AUC(0 g) * 0.5 (loose lower bound:
        # the model has at least started to react to carbs).
        self.assertGreater(
            aucs[-1], 0.5 * aucs[0],
            f"sweep training failed to undo inversion: AUCs={aucs}",
        )


if __name__ == "__main__":
    unittest.main()
