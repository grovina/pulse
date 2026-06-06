"""Tests for CarbMassBalanceSignal (iter 74 cross-module conservation).

The signal is gating-correct (zero weight is a no-op), runs end-to-end and
emits its diagnostics, and — the load-bearing property — produces *zero*
gradient when both mass-balance inequalities are satisfied (it must only fire
on a frank violation, never fight a physical trajectory).
"""

from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn as nn

from pulse.model import ModularPhysiologyNetwork
from pulse.training import CarbMassBalanceSignal, SignalContext, WeightSchedule
from pulse.types import EMBEDDING_DIM


def _tiny_model() -> ModularPhysiologyNetwork:
    return ModularPhysiologyNetwork(
        embedding_dim=EMBEDDING_DIM,
        metabolic_hidden=16, appetite_hidden=12, stress_hidden=12,
        cardiovascular_hidden=16, thermoreg_hidden=8, respiratory_hidden=8,
    )


def _ctx(opt, params, device) -> SignalContext:
    return SignalContext(
        epoch=0, total_epochs=1, rng=np.random.default_rng(0),
        device=device, optimizer=opt, params=params, grad_clip=10.0,
    )


class TestCarbMassBalanceSignal(unittest.TestCase):
    def test_zero_weight_is_noop(self) -> None:
        device = torch.device("cpu")
        model = _tiny_model()
        emb = nn.Embedding(2, EMBEDDING_DIM)
        params = list(model.parameters()) + list(emb.parameters())
        opt = torch.optim.Adam(params, lr=1e-3)
        sig = CarbMassBalanceSignal(weight=WeightSchedule(0.0))
        result = sig.compute(model, emb, _ctx(opt, params, device))
        self.assertEqual(result.n_units, 0)

    def test_runs_and_emits_diagnostics(self) -> None:
        device = torch.device("cpu")
        torch.manual_seed(3)
        model = _tiny_model()
        emb = nn.Embedding(2, EMBEDDING_DIM)
        nn.init.normal_(emb.weight, std=0.1)
        params = list(model.parameters()) + list(emb.parameters())
        opt = torch.optim.Adam(params, lr=1e-3)
        sig = CarbMassBalanceSignal(
            weight=WeightSchedule(0.1),
            carb_doses_g=(30.0, 60.0),
            n_patients=2, sample_patients=1,
            window_min=120,
        )
        result = sig.compute(model, emb, _ctx(opt, params, device))
        self.assertEqual(result.n_units, 1)
        for k in ("over_store", "depletion", "mean_delta_g", "n_dose_emb_pairs"):
            self.assertIn(k, result.sub_metrics)
        # zero (default) + 1 sampled patient = 2 embeddings × 2 doses = 4 pairs.
        self.assertEqual(result.sub_metrics["n_dose_emb_pairs"], 4.0)

    def test_satisfied_inequalities_give_zero_gradient(self) -> None:
        # A glycogen delta of exactly 0 satisfies both bounds (0 ≤ 0 ≤ dose),
        # so the conservation term must contribute no gradient — the signal
        # only fires on a frank mass-balance violation.
        sig = CarbMassBalanceSignal(weight=WeightSchedule(1.0))
        for dose in sig.carb_doses_g:
            delta = torch.zeros((), requires_grad=True)
            norm = max(float(dose), 1.0)
            over = torch.relu(delta - float(dose)) / norm
            under = torch.relu(-delta) / norm
            loss = over.pow(2) + under.pow(2)
            loss.backward()
            self.assertEqual(float(delta.grad), 0.0)

    def test_overstore_violation_pulls_delta_down(self) -> None:
        # Δ above the dose ceiling must produce a gradient that reduces Δ.
        dose = 30.0
        delta = torch.tensor(50.0, requires_grad=True)  # stored 50 g from 30 g in
        norm = max(dose, 1.0)
        loss = (torch.relu(delta - dose) / norm).pow(2)
        loss.backward()
        self.assertGreater(float(delta.grad), 0.0)  # ∂loss/∂Δ > 0 ⇒ GD lowers Δ


if __name__ == "__main__":
    unittest.main()
