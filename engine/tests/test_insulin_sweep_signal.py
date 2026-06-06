"""Unit tests for the insulin sweep training signal.

Covers, mirroring ``test_gut_dose_sweep_signal``:

* cold targets reproduce the Bergman GSIR / clearance shapes
  (monotone increasing in glucose above threshold; monotone
  decreasing in insulin)
* zero weight is a no-op; positive weight steps the metabolic module
* gradient lands on the metabolic module + its embedding projection,
  not on the gut, stress, or vital-sign modules
* a few SGD iterations strictly reduce the loss
"""

from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn as nn

from pulse.knowledge.full_body import PatientParams
from pulse.model import ModularPhysiologyNetwork
from pulse.training import (
    InsulinSweepProtocol,
    InsulinSweepSignal,
    SignalContext,
    WeightSchedule,
)
from pulse.training.insulin_sweep_signal import _cold_metabolic_rates
from pulse.types import EMBEDDING_DIM


def _tiny_model() -> ModularPhysiologyNetwork:
    return ModularPhysiologyNetwork(
        metabolic_hidden=16, appetite_hidden=12, stress_hidden=12,
        cardiovascular_hidden=16, thermoreg_hidden=8, respiratory_hidden=8,
    )


class TestColdTargets(unittest.TestCase):
    def test_dI_monotone_in_glucose_above_threshold(self) -> None:
        """GSIR: dI/dt strictly increases with G once G > h. Cold model uses
        ``params.gamma * max(G - h, 0) * incretin_factor`` as the secretion
        term, so any G > h gives a positive contribution that scales linearly."""
        params = PatientParams()
        di = [
            float(_cold_metabolic_rates(g, params.Ib, params)[1])
            for g in (60.0, 80.0, 95.0, 120.0, 150.0, 200.0, 250.0, 300.0)
        ]
        # First two are below or at threshold h=80 — clearance pulls insulin
        # toward effective_Ib < Ib, so dI is negative.
        self.assertLess(di[0], 0.0)
        # 95 mg/dL: just barely above h=80, slight positive secretion.
        # 200, 250, 300 should be strictly increasing in dI.
        for prev, nxt in zip(di[3:], di[4:]):
            self.assertLess(prev, nxt, f"dI not monotone in G: {di}")

    def test_dI_monotone_decreasing_in_insulin(self) -> None:
        """Clearance: dI/dt strictly decreases with I when GSIR contribution
        is fixed (G held at Gb). At G=Gb the gamma term = 0, so dI is
        purely ``-n * (I - Ib)``."""
        params = PatientParams()
        di = [
            float(_cold_metabolic_rates(params.Gb, i, params)[1])
            for i in (5.0, 10.0, 20.0, 40.0, 80.0)
        ]
        for prev, nxt in zip(di, di[1:]):
            self.assertGreater(prev, nxt, f"dI not monotone-decreasing in I: {di}")

    def test_hep_target_responds_to_insulin(self) -> None:
        """Hepatic glucose output target is suppressed by elevated insulin —
        the cold model has ``-params.ins_hep * max(I - Ib, 0) / (Ib + 5.0)``
        in ``hep_target``. dHep should decrease as I goes from Ib up."""
        params = PatientParams()
        dhep = [
            float(_cold_metabolic_rates(params.Gb, i, params)[6])
            for i in (10.0, 20.0, 40.0, 80.0)
        ]
        for prev, nxt in zip(dhep, dhep[1:]):
            self.assertGreater(prev, nxt, f"dHep not suppressed by I: {dhep}")


class TestSignalGating(unittest.TestCase):
    def test_zero_weight_is_noop(self) -> None:
        device = torch.device("cpu")
        model = _tiny_model()
        emb = nn.Embedding(2, EMBEDDING_DIM)
        params = list(model.parameters()) + list(emb.parameters())
        opt = torch.optim.Adam(params, lr=1e-3)
        sig = InsulinSweepSignal(
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
        sig = InsulinSweepSignal(
            n_patients=2, sample_patients=2, weight=WeightSchedule(1.0),
        )
        ctx = SignalContext(
            epoch=0, total_epochs=1, rng=np.random.default_rng(0),
            device=device, optimizer=opt, params=params, grad_clip=10.0,
        )
        before = {n: p.detach().clone() for n, p in model.metabolic.named_parameters()}
        result = sig.compute(model, emb, ctx)
        moved = any(
            float((p.detach() - before[n]).abs().sum()) > 0
            for n, p in model.metabolic.named_parameters()
        )
        self.assertEqual(result.n_units, 1)
        self.assertTrue(moved, "metabolic module parameters did not move")
        self.assertGreater(result.sub_metrics["n_grid_emb_pairs"], 0)


class TestGradientFlow(unittest.TestCase):
    def test_unrelated_modules_unchanged(self) -> None:
        device = torch.device("cpu")
        torch.manual_seed(0)
        model = _tiny_model()
        emb = nn.Embedding(1, EMBEDDING_DIM)
        params = list(model.parameters()) + list(emb.parameters())
        opt = torch.optim.SGD(params, lr=1e-2)
        sig = InsulinSweepSignal(
            n_patients=1, sample_patients=1, weight=WeightSchedule(1.0),
        )
        ctx = SignalContext(
            epoch=0, total_epochs=1, rng=np.random.default_rng(0),
            device=device, optimizer=opt, params=params, grad_clip=1e9,
        )
        before = {
            mod_name: {n: p.detach().clone() for n, p in mod.named_parameters()}
            for mod_name, mod in (
                ("gut", model.gut),
                ("respiratory", model.respiratory),
                ("cardiovascular", model.cardiovascular),
                ("thermoreg", model.thermoreg),
                ("stress", model.stress),
                ("appetite", model.appetite),
            )
        }
        sig.compute(model, emb, ctx)
        for mod_name, mod in (
            ("gut", model.gut),
            ("respiratory", model.respiratory),
            ("cardiovascular", model.cardiovascular),
            ("thermoreg", model.thermoreg),
            ("stress", model.stress),
            ("appetite", model.appetite),
        ):
            moved = any(
                float((p.detach() - before[mod_name][n]).abs().sum()) > 0
                for n, p in mod.named_parameters()
            )
            self.assertFalse(
                moved,
                f"{mod_name} module unexpectedly moved under insulin-sweep step",
            )

    def test_gradient_lands_on_metabolic_pipeline(self) -> None:
        """Manually replicate a sweep_rates → loss → backward to inspect grads."""
        torch.manual_seed(0)
        model = _tiny_model()
        sig = InsulinSweepSignal(protocol=InsulinSweepProtocol(
            glucose_sweep_mg_dL=(95.0, 200.0),
            insulin_sweep_uU_mL=(10.0, 40.0),
        ))
        emb_full = torch.zeros(1, EMBEDDING_DIM, requires_grad=False)
        g_states = sig._g_states
        g_pred = sig._sweep_rates(model, g_states, emb_full)
        loss = ((g_pred[..., 1] - sig._g_targets[:, 1].unsqueeze(1)) ** 2).mean()
        loss.backward()

        met_grad = sum(
            float(p.grad.abs().sum()) for p in model.metabolic.parameters()
            if p.grad is not None
        )
        proj_grad = sum(
            float(p.grad.abs().sum())
            for p in model.embedding_projections["metabolic"].parameters()
            if p.grad is not None
        )
        self.assertGreater(met_grad, 0.0)
        self.assertGreater(proj_grad, 0.0)

        for name, sub in (
            ("gut", model.gut),
            ("appetite", model.appetite),
            ("cardiovascular", model.cardiovascular),
            ("respiratory", model.respiratory),
        ):
            grad_total = sum(
                float(p.grad.abs().sum()) for p in sub.parameters()
                if p.grad is not None
            )
            self.assertEqual(
                grad_total, 0.0,
                f"{name} module unexpectedly received gradient ({grad_total})",
            )


class TestLearningSanity(unittest.TestCase):
    def test_loss_strictly_decreases_over_iterations(self) -> None:
        device = torch.device("cpu")
        torch.manual_seed(0)
        model = _tiny_model()
        emb = nn.Embedding(1, EMBEDDING_DIM)
        params = list(model.parameters()) + list(emb.parameters())
        opt = torch.optim.Adam(params, lr=5e-3)
        sig = InsulinSweepSignal(
            n_patients=1, sample_patients=1, weight=WeightSchedule(1.0),
        )
        rng = np.random.default_rng(0)
        losses: list[float] = []
        for ep in range(8):
            ctx = SignalContext(
                epoch=ep, total_epochs=8, rng=rng,
                device=device, optimizer=opt, params=params, grad_clip=10.0,
            )
            r = sig.compute(model, emb, ctx)
            losses.append(r.loss_sum)
        self.assertLess(losses[-1], losses[0],
                        f"loss did not decrease: {losses}")


if __name__ == "__main__":
    unittest.main()
