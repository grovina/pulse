"""Sanity tests for the training-signal framework (Phase 0)."""

from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn as nn

from pulse.knowledge import ALL_COHORT_STATISTICS
from pulse.model import ModularPhysiologyNetwork
from pulse.training import (
    ColdModelDistillationSignal,
    CohortStatisticSignal,
    SignalContext,
    TrajectoryRolloutSignal,
    WeightSchedule,
)
from pulse.types import EMBEDDING_DIM


class TestWeightSchedule(unittest.TestCase):
    def test_zero_base_always_zero(self) -> None:
        s = WeightSchedule(0.0, enable_at_epoch=5)
        self.assertEqual(s.at(0), 0.0)
        self.assertEqual(s.at(10), 0.0)

    def test_phase_boundary_gates(self) -> None:
        s = WeightSchedule(0.5, enable_at_epoch=3)
        self.assertEqual(s.at(0), 0.0)
        self.assertEqual(s.at(2), 0.0)
        self.assertEqual(s.at(3), 0.5)
        self.assertEqual(s.at(99), 0.5)

    def test_no_boundary_is_active(self) -> None:
        s = WeightSchedule(1.0)
        self.assertEqual(s.at(0), 1.0)


class TestTrajectorySignalShape(unittest.TestCase):
    def test_dataset_and_priors_populated(self) -> None:
        sig = TrajectoryRolloutSignal(
            n_patients=2, n_days=2, seed=0,
            contribution_weights={"full_body": 1.0},
            windows_per_patient=1, meal_window_bias=0.5,
            input_dropout=0.3, huber_delta=1.0, gut_loss_weight=0.5,
            coupling_weight=WeightSchedule(0.0),
            verifier_weight=WeightSchedule(0.0),
        )
        self.assertEqual(len(sig.dataset), 2)
        self.assertGreater(len(sig.priors), 0)


class TestCohortStatisticSignalGate(unittest.TestCase):
    def test_zero_weight_returns_empty_result(self) -> None:
        device = torch.device("cpu")
        model = ModularPhysiologyNetwork()
        embeddings = nn.Embedding(2, EMBEDDING_DIM)
        params = list(model.parameters()) + list(embeddings.parameters())
        opt = torch.optim.Adam(params, lr=1e-4)
        sig = CohortStatisticSignal(
            specs=list(ALL_COHORT_STATISTICS),
            n_patients=2,
            sample_patients=2,
            weight=WeightSchedule(0.0),
        )
        ctx = SignalContext(
            epoch=0, total_epochs=1, rng=np.random.default_rng(0),
            device=device, optimizer=opt, params=params, grad_clip=10.0,
        )
        result = sig.compute(model, embeddings, ctx)
        self.assertEqual(result.n_units, 0)
        self.assertEqual(result.loss_sum, 0.0)


class TestColdModelDistillationSignal(unittest.TestCase):
    def test_zero_weight_returns_empty_result(self) -> None:
        model = ModularPhysiologyNetwork()
        embeddings = nn.Embedding(2, EMBEDDING_DIM)
        params = list(model.parameters())
        opt = torch.optim.Adam(params, lr=1e-4)
        sig = ColdModelDistillationSignal(weight=WeightSchedule(0.0))
        ctx = SignalContext(
            epoch=0, total_epochs=1, rng=np.random.default_rng(0),
            device=torch.device("cpu"), optimizer=opt, params=params, grad_clip=10.0,
        )
        result = sig.compute(model, embeddings, ctx)
        self.assertEqual(result.n_units, 0)
        self.assertEqual(result.loss_sum, 0.0)

    def test_references_built_and_loss_finite(self) -> None:
        model = ModularPhysiologyNetwork()
        embeddings = nn.Embedding(2, EMBEDDING_DIM)
        params = list(model.parameters())
        opt = torch.optim.Adam(params, lr=1e-4)
        sig = ColdModelDistillationSignal(
            weight=WeightSchedule(0.2), protocols_per_epoch=3,
            calib_steps=3, calib_warm_steps=2,
        )
        # Pool built at construction; each protocol carries a cold reference
        # trajectory aligned to its duration. The synthetic pool has 9 named
        # protocols (see _build_protocol_pool).
        self.assertEqual(len(sig._protocols), 9)
        for proto in sig._protocols:
            self.assertEqual(proto.reference.shape[0], proto.duration_min)
            self.assertEqual(proto.reference.shape[1], len(model.norm_center))
        ctx = SignalContext(
            epoch=0, total_epochs=1, rng=np.random.default_rng(0),
            device=torch.device("cpu"), optimizer=opt, params=params, grad_clip=10.0,
        )
        result = sig.compute(model, embeddings, ctx)
        self.assertEqual(result.n_units, 3)
        self.assertTrue(np.isfinite(result.loss_sum))
        # All seven default distill markers report a per-marker term.
        for m in ("glucagon", "ffa", "ghrelin", "leptin", "acth", "cortisol", "bhb"):
            self.assertIn(f"dist_{m}", result.sub_metrics)

    def test_rate_matching_mode(self) -> None:
        # iter 50: teacher-forced rate matching on the bench-cohort protocols.
        model = ModularPhysiologyNetwork()
        embeddings = nn.Embedding(2, EMBEDDING_DIM)
        params = list(model.parameters())
        opt = torch.optim.Adam(params, lr=1e-4)
        sig = ColdModelDistillationSignal(
            weight=WeightSchedule(0.3), pool="bench_cohorts", protocols_per_epoch=2,
            mode="rate", calib_steps=2, calib_warm_steps=1,
        )
        self.assertEqual(len(sig._protocols), 2)
        for proto in sig._protocols:
            self.assertEqual(proto.cold_rate.shape[0], proto.duration_min - 1)
            self.assertEqual(proto.cold_absorption.shape[0], proto.duration_min)
            self.assertEqual(proto.cold_absorption.shape[1], 4)
        ctx = SignalContext(
            epoch=0, total_epochs=1, rng=np.random.default_rng(0),
            device=torch.device("cpu"), optimizer=opt, params=params, grad_clip=10.0,
        )
        result = sig.compute(model, embeddings, ctx)
        self.assertEqual(result.n_units, 2)
        self.assertTrue(np.isfinite(result.loss_sum))
        self.assertEqual(result.sub_metrics.get("mode"), 1.0)

    def test_anchored_mode(self) -> None:
        # iter 75: rate matching + short-horizon free-running level anchor.
        model = ModularPhysiologyNetwork()
        embeddings = nn.Embedding(2, EMBEDDING_DIM)
        params = list(model.parameters())
        opt = torch.optim.Adam(params, lr=1e-4)
        sig = ColdModelDistillationSignal(
            weight=WeightSchedule(0.3), pool="bench_cohorts", protocols_per_epoch=2,
            mode="anchored", anchor_window=30, anchor_samples=4,
            calib_steps=2, calib_warm_steps=1,
        )
        ctx = SignalContext(
            epoch=0, total_epochs=1, rng=np.random.default_rng(0),
            device=torch.device("cpu"), optimizer=opt, params=params, grad_clip=10.0,
        )
        result = sig.compute(model, embeddings, ctx)
        self.assertEqual(result.n_units, 2)
        self.assertTrue(np.isfinite(result.loss_sum))
        self.assertEqual(result.sub_metrics.get("mode"), 2.0)
        self.assertIn("anchor_skips", result.sub_metrics)
        # Every default distill marker still reports a (combined rate+level) term.
        for m in ("glucagon", "ffa", "ghrelin", "leptin", "acth", "cortisol", "bhb"):
            self.assertIn(f"dist_{m}", result.sub_metrics)
        # The level anchor must contribute gradient to the model params (the
        # whole point — rate-only matching is offset-invariant). Backprop and
        # assert at least one param grad is non-zero.
        loss = sig._level_terms(
            model, sig._protocols[0],
            sig._calibrated_embedding(model, sig._protocols[0], 0, ctx),
            torch.device("cpu"),
        )
        self.assertTrue(len(loss) > 0)
        opt.zero_grad()
        torch.stack(list(loss.values())).mean().backward()
        self.assertTrue(any(p.grad is not None and torch.any(p.grad != 0) for p in params))

    def test_bench_cohort_pool(self) -> None:
        model = ModularPhysiologyNetwork()
        embeddings = nn.Embedding(2, EMBEDDING_DIM)
        params = list(model.parameters())
        opt = torch.optim.Adam(params, lr=1e-4)
        sig = ColdModelDistillationSignal(
            weight=WeightSchedule(0.5), pool="bench_cohorts", protocols_per_epoch=2,
            calib_steps=2, calib_warm_steps=1,
        )
        # The two cohort episodes the bench scores unobserved markers on.
        self.assertEqual(len(sig._protocols), 2)
        for proto in sig._protocols:
            self.assertEqual(proto.reference.shape[0], proto.duration_min)
            self.assertTrue(len(proto.obs_points) > 0)
            for t, midx, _v in proto.obs_points:
                self.assertTrue(0 <= t < proto.duration_min)
                self.assertTrue(0 <= midx < len(model.norm_center))
        ctx = SignalContext(
            epoch=0, total_epochs=1, rng=np.random.default_rng(0),
            device=torch.device("cpu"), optimizer=opt, params=params, grad_clip=10.0,
        )
        result = sig.compute(model, embeddings, ctx)
        self.assertEqual(result.n_units, 2)
        self.assertTrue(np.isfinite(result.loss_sum))


if __name__ == "__main__":
    unittest.main()
