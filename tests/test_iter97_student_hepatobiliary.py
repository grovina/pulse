"""Iter 97 — hepatobiliary mass action and the closed enterohepatic loop
(review 2026-09-04, items 3.6 and 2.5)."""

from __future__ import annotations

import math
import unittest

import torch

from pulse.model import ModularPhysiologyNetwork, integrate
from pulse.modules import hepatobiliary as H
from pulse.modules.base import compute_time_features
from pulse.modules.hepatobiliary import DuodenalDeliveryKernel
from pulse.modules.gut import MealEvent
from pulse.types import EMBEDDING_DIM, MARKER_INDEX as MI, NORM_CENTER


def _model(seed=0, perturb=0.5):
    torch.manual_seed(seed)
    m = ModularPhysiologyNetwork(
        metabolic_hidden=16, appetite_hidden=16, stress_hidden=16, cardiovascular_hidden=16,
        thermoreg_hidden=16, respiratory_hidden=16, gut_hidden=16, hepatobiliary_hidden=16)
    with torch.no_grad():
        for p in m.hepatobiliary.parameters():
            p.add_(perturb * torch.randn_like(p))
    m.eval()
    return m


def _inputs(m, batch=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    state = 0.7 * torch.randn(batch, 4, generator=g)
    coupling = torch.rand(batch, 2, generator=g) * 0.3
    external = torch.stack([torch.rand(batch, generator=g), torch.ones(batch)], -1)
    emb = torch.zeros(batch, m.hepatobiliary.embedding_dim)
    tf = compute_time_features(torch.rand(batch, generator=g) * 1440.0)
    return state, coupling, external, emb, tf


class TestLoopConservation(unittest.TestCase):
    def test_pool_ledger_closes_pointwise(self) -> None:
        """d(gallbladder + intestine)/dt == synthesis − faecal loss."""
        m = _model(0)
        hb = m.hepatobiliary
        state, coupling, external, emb, tf = _inputs(m)
        with torch.no_grad():
            f = hb.fluxes(state, coupling, external, emb, tf)
            rates = hb(state, coupling, external, emb, tf)
        lhs = rates[:, H._GB_IDX] + rates[:, H._INT_IDX] + rates[:, H._BA_IDX] / f["spill_gain"]
        torch.testing.assert_close(lhs, f["synthesis"] - f["fecal_loss"], atol=1e-6, rtol=1e-5)

    def test_gallbladder_fills_only_from_hepatic_export(self) -> None:
        m = _model(1)
        hb = m.hepatobiliary
        state, coupling, external, emb, tf = _inputs(m)
        with torch.no_grad():
            f = hb.fluxes(state, coupling, external, emb, tf)
        torch.testing.assert_close(f["to_gallbladder"] + f["to_intestine"], f["export"], atol=1e-7, rtol=1e-5)
        self.assertTrue(bool((f["to_gallbladder"] >= 0).all()))
        self.assertTrue(bool((f["to_gallbladder"] <= f["export"] + 1e-7).all()))

    def test_intestinal_bile_owns_no_parameters(self) -> None:
        m = _model(2)
        for idx in (H._CCK_IDX, H._GB_IDX, H._INT_IDX, H._BA_IDX):
            self.assertEqual(sum(p.numel() for p in m.hepatobiliary.heads[idx].parameters()), 0)
        self.assertEqual(m.hepatobiliary.embedding_dim, 0)

    def test_lowering_canalicular_export_raises_serum_spillover(self) -> None:
        m = _model(0, perturb=0.0)
        hb = m.hepatobiliary
        state, coupling, external, emb, tf = _inputs(m, batch=8)
        with torch.no_grad():
            f_h = hb.fluxes(state, coupling, external, emb, tf)
            hb.log_k_canalicular.fill_(-2.0)
            f_l = hb.fluxes(state, coupling, external, emb, tf)
        self.assertTrue(bool((f_l["extraction"] < f_h["extraction"]).all()))
        self.assertTrue(bool((f_l["spillover"] > f_h["spillover"]).all()))

    def test_twenty_grams_of_fat_has_left_long_before_500_minutes(self) -> None:
        """20 g of fat is 180 kcal, about 84 min at 2.13 kcal/min. Delivery is
        the emptying, so it is present early and gone once the stomach is."""
        k = DuodenalDeliveryKernel()
        meals = [MealEvent(time=0.0, carbs=0.0, fats=20.0, proteins=0.0)]
        with torch.no_grad():
            early = k.forward_window(torch.tensor([10.0]), meals)
            late = k.forward_window(torch.tensor([500.0]), meals)
        self.assertGreater(float(early[0].sum()), 0.0)
        self.assertEqual(float(late[0].sum()), 0.0)


class TestBelowTypicalIsRepresentable(unittest.TestCase):
    def _rollout(self, m, n=1440):
        with torch.no_grad():
            return integrate(m, torch.tensor(NORM_CENTER), torch.zeros(EMBEDDING_DIM), n,
                             start_time_minutes=360, meals=[], sleep_wake=torch.ones(n), activity=torch.zeros(n))

    def test_cck_rests_at_the_typical_while_bile_acids_can_sit_below(self) -> None:
        """CCK's resting level is the population typical. Serum bile acids still
        move with their clearance."""
        m = _model(3, perturb=0.0)
        with torch.no_grad():
            m.hepatobiliary.log_k_ba.fill_(math.log(0.12))
        tr = self._rollout(m)
        self.assertAlmostEqual(float(tr[-1, MI["cck"]]), NORM_CENTER[MI["cck"]], places=2)
        self.assertLess(float(tr[-1, MI["bile_acids"]]), 0.9 * NORM_CENTER[MI["bile_acids"]])

    def test_cck_rests_at_the_typical_while_bile_acids_can_sit_above(self) -> None:
        m = _model(4, perturb=0.0)
        with torch.no_grad():
            m.hepatobiliary.log_k_ba.fill_(math.log(0.008))
        tr = self._rollout(m)
        self.assertAlmostEqual(float(tr[-1, MI["cck"]]), NORM_CENTER[MI["cck"]], places=2)
        self.assertGreater(float(tr[-1, MI["bile_acids"]]), 1.1 * NORM_CENTER[MI["bile_acids"]])

    def test_every_hepatobiliary_parameter_receives_gradient(self) -> None:
        # At the population rates, not a draw. A large canalicular rate sits on
        # the extraction ceiling, and the clamp there has no gradient.
        m = _model(5, perturb=0.0)
        m.train()
        hb = m.hepatobiliary
        state, coupling, external, emb, tf = _inputs(m)
        hb(state, coupling, external, emb, tf).abs().sum().backward()
        dead = [n for n, p in hb.named_parameters() if p.grad is None or float(p.grad.abs().max()) == 0.0]
        self.assertEqual(dead, [])


if __name__ == "__main__":
    unittest.main()
