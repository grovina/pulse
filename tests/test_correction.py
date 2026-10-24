"""Functional forms are defaults. A zero correction leaves the law; a nonzero one
can move it, and the penalty falls on the correction alone.

The teacher keeps the published stomach and the published Hill. It does not
grow a parameter for either.
"""

from __future__ import annotations

import inspect
import unittest

import torch

from pulse.correction import corrected_mass_action, shrinkage_of
from pulse.gastric import EMPTY_KCAL_PER_MIN, KCAL_PER_G, outflow_grams, simulate, step_grams
from pulse.knowledge.full_body import _hill_centred as teacher_hill
from pulse.knowledge.full_body import _stomach_at, glucose_fluxes
from pulse.modules.base import compute_time_features
from pulse.modules.gastric import GastricModule
from pulse.modules.metabolic import (
    _GLUCAGON_IDX, _GLUCOSE_IDX, _HGO_GN_N, MetabolicModule, _hill_centred,
)


def _proportion(pool: torch.Tensor, rate: float) -> torch.Tensor:
    kcal_g = pool.new_tensor(KCAL_PER_G)
    kcal = (pool * kcal_g).sum(-1, keepdim=True)
    frac = torch.where(kcal > 0, torch.minimum(kcal, kcal.new_full(kcal.shape, rate)) / kcal, torch.zeros_like(kcal))
    return pool * frac


def _roll(mod: GastricModule, swallow: torch.Tensor) -> torch.Tensor:
    const = mod.constants(torch.zeros(1))
    raw = torch.zeros(3)
    pools = []
    for t in range(swallow.shape[0]):
        state = (raw - mod._center) / mod._scale
        rate = mod.step(state, torch.zeros(0), const, {"swallow": swallow[t]}, {})
        raw = raw + rate
        pools.append(raw.clone())
    return torch.stack(pools)


def _metabolic_inputs(met: MetabolicModule, glucose: float, glucagon: float):
    emb = torch.zeros(1, met.embedding_dim)
    state = torch.zeros(1, met.n_state)
    state[0, _GLUCOSE_IDX] = (glucose - 95.0) / 30.0
    state[0, _GLUCAGON_IDX] = (glucagon - 70.0) / 20.0
    coupling = torch.zeros(1, met.n_coupling)
    external = torch.zeros(1, len(met.external_inputs))
    external[0, 1] = 1.0
    tf = compute_time_features(torch.tensor([600.0]))
    return state, coupling, external, emb, tf


class TestCorrection(unittest.TestCase):
    def test_zero_correction_is_the_uncorrected_law(self) -> None:
        mod = GastricModule(0)
        self.assertTrue(torch.equal(mod.split_correction.term, torch.zeros(3)))
        pool = torch.tensor([[40.0, 15.0, 10.0], [0.2, 0.0, 20.0]])
        plain = outflow_grams(pool, EMPTY_KCAL_PER_MIN)
        through = outflow_grams(pool, mod.empty_kcal(), mod.split_correction())
        torch.testing.assert_close(through, plain)
        torch.testing.assert_close(plain, _proportion(pool, EMPTY_KCAL_PER_MIN))
        for row in pool:
            got = step_grams(float(row[0]), float(row[1]), float(row[2]), EMPTY_KCAL_PER_MIN)
            torch.testing.assert_close(outflow_grams(row, EMPTY_KCAL_PER_MIN), torch.tensor(got), atol=1e-5, rtol=1e-5)

        swallow = torch.zeros(40, 3)
        swallow[0] = torch.tensor([40.0, 15.0, 10.0])
        with torch.no_grad():
            rolled = _roll(mod, swallow)
            left, filled = simulate(swallow, mod.empty_kcal())
        torch.testing.assert_close(rolled, filled - left, atol=1e-4, rtol=1e-4)

        met = MetabolicModule(8, hidden_dim=8)
        self.assertTrue(torch.equal(met.hill_correction.term, torch.zeros(1)))
        self.assertTrue(torch.equal(met.uptake_correction.term, torch.zeros(1)))
        state, coupling, external, emb, tf = _metabolic_inputs(met, 180.0, 140.0)
        with torch.no_grad():
            fluxes = met.fluxes(state, coupling, external, emb, tf)
            raw = met.raw_state(state)
        gn, g = raw[:, _GLUCAGON_IDX], raw[:, _GLUCOSE_IDX]
        torch.testing.assert_close(fluxes["g_gn"], _hill_centred(gn, fluxes["gn_b"], _HGO_GN_N))
        torch.testing.assert_close(fluxes["uptake_ii"], fluxes["k_ii"] * g)
        self.assertAlmostEqual(float(teacher_hill(140.0, 70.0, 2.0)), float(_hill_centred(torch.tensor(140.0), 70.0, 2.0)), places=5)

    def test_a_nonzero_correction_changes_the_output_and_takes_gradient(self) -> None:
        mod = GastricModule(0)
        swallow = torch.tensor([30.0, 12.0, 10.0])
        const = mod.constants(torch.zeros(1))
        with torch.no_grad():
            base = mod.step(torch.zeros(3), torch.zeros(0), const, {"swallow": swallow}, {})
            mod.split_correction.term.copy_(torch.tensor([0.6, -0.4, 0.2]))
        moved = mod.step(torch.zeros(3), torch.zeros(0), const, {"swallow": swallow}, {})
        self.assertGreater(float((moved - base).abs().sum().detach()), 1e-3)
        kcal_g = torch.tensor(KCAL_PER_G)
        left = (swallow - moved).detach()
        self.assertTrue(torch.all(left >= -1e-5))
        self.assertTrue(torch.all(left <= swallow + 1e-4))
        kcal_out = float((left * kcal_g).sum())
        pool_kcal = float((swallow * kcal_g).sum())
        rate = float(mod.empty_kcal().detach())
        self.assertAlmostEqual(kcal_out, min(rate, pool_kcal), places=4)
        moved.sum().backward()
        self.assertGreater(float(mod.split_correction.term.grad.abs().sum()), 0.0)
        self.assertGreater(float(mod.log_empty_kcal.grad.abs()), 0.0)

        # A shift that would empty the small macro is given back to the others,
        # so the calorie rate still leaves in full.
        with torch.no_grad():
            mod.zero_grad()
            mod.split_correction.term.copy_(torch.tensor([8.0, 0.0, -8.0]))
        tiny = torch.tensor([0.2, 0.0, 20.0])
        grams = outflow_grams(tiny, 2.13, mod.split_correction()).detach()
        self.assertAlmostEqual(float((grams * kcal_g).sum()), 2.13, places=4)
        self.assertLess(float(grams[0]), 0.2 + 1e-4)
        self.assertEqual(float(grams[1]), 0.0)

        met = MetabolicModule(8, hidden_dim=8)
        off = _metabolic_inputs(met, 180.0, 140.0)
        with torch.no_grad():
            before = met.fluxes(*off)
            met.hill_correction.term.fill_(0.5)
            met.uptake_correction.term.fill_(0.4)
            anchored = met.fluxes(*_metabolic_inputs(met, 95.0, 70.0))
            raw = met.raw_state(_metabolic_inputs(met, 95.0, 70.0)[0])
        self.assertAlmostEqual(float(anchored["g_gn"]), 1.0, places=5)
        self.assertAlmostEqual(float(anchored["uptake_ii"]), float(anchored["k_ii"] * raw[0, _GLUCOSE_IDX]), places=5)
        after = met.fluxes(*off)
        self.assertGreater(float((after["g_gn"] - before["g_gn"]).abs().detach()), 1e-3)
        self.assertGreater(float((after["uptake_ii"] - before["uptake_ii"]).abs().detach()), 1e-4)
        after["glucose_rate"].sum().backward()
        self.assertGreater(float(met.hill_correction.term.grad.abs()), 0.0)
        self.assertGreater(float(met.uptake_correction.term.grad.abs()), 0.0)

    def test_shrinkage_pulls_on_the_correction_only(self) -> None:
        mod = GastricModule(0)
        met = MetabolicModule(8, hidden_dim=8)
        self.assertEqual(float(shrinkage_of(mod).detach()), 0.0)
        self.assertEqual(float(shrinkage_of(met).detach()), 0.0)
        with torch.no_grad():
            mod.split_correction.term.copy_(torch.tensor([0.2, -0.1, 0.0]))
            mod.log_empty_kcal.add_(0.4)
            met.uptake_correction.term.fill_(0.3)
            met.log_k_ii.add_(0.5)
        stomach = shrinkage_of(mod)
        uptake = shrinkage_of(met)
        self.assertAlmostEqual(float(stomach.detach()), 0.2 ** 2 + 0.1 ** 2, places=6)
        self.assertAlmostEqual(float(uptake.detach()), 0.3 ** 2, places=6)
        stomach.backward()
        uptake.backward()
        self.assertIsNone(mod.log_empty_kcal.grad)
        self.assertIsNone(met.log_k_ii.grad)
        self.assertGreater(float(mod.split_correction.term.grad.abs().sum()), 0.0)
        self.assertGreater(float(met.uptake_correction.term.grad.abs()), 0.0)
        # Still zero, so the quadratic has nothing to pull.
        self.assertEqual(float(met.hill_correction.term.grad), 0.0)

    def test_the_teacher_path_does_not_grow_a_parameter(self) -> None:
        self.assertNotIn("correction", inspect.signature(step_grams).parameters)
        self.assertNotIn("correction", inspect.signature(_stomach_at).parameters)
        self.assertNotIn("correction", inspect.signature(teacher_hill).parameters)
        self.assertNotIn("correction", inspect.signature(glucose_fluxes).parameters)
        self.assertIsNone(inspect.signature(outflow_grams).parameters["correction"].default)

        mod = GastricModule(0)
        pool = torch.tensor([20.0, 10.0, 8.0])
        teacher = outflow_grams(pool, EMPTY_KCAL_PER_MIN)
        with torch.no_grad():
            mod.split_correction.term.copy_(torch.tensor([1.2, -0.4, 0.3]))
        torch.testing.assert_close(outflow_grams(pool, EMPTY_KCAL_PER_MIN), teacher)
        student = outflow_grams(pool, mod.empty_kcal(), mod.split_correction())
        self.assertGreater(float((student - teacher).abs().sum().detach()), 1e-3)
        self.assertEqual(
            set(dict(mod.named_parameters())),
            {"log_empty_kcal", "split_correction.term"},
        )

        met = MetabolicModule(4, hidden_dim=4)
        names = dict(met.named_parameters())
        self.assertIn("hill_correction.term", names)
        self.assertIn("uptake_correction.term", names)
        self.assertIn("log_k_ii", names)

    def test_the_rate_stays_in_its_domain(self) -> None:
        mod = GastricModule(0)
        with torch.no_grad():
            mod.log_empty_kcal.add_(20.0)
            mod.split_correction.term.fill_(12.0)
        self.assertGreater(float(mod.empty_kcal().detach()), 0.0)
        self.assertTrue(torch.isfinite(mod.empty_kcal().detach()))
        pool = torch.tensor([5.0, 4.0, 3.0])
        grams = outflow_grams(pool, mod.empty_kcal(), mod.split_correction())
        self.assertTrue(torch.isfinite(grams).all())
        self.assertTrue(torch.all(grams >= -1e-5))
        self.assertTrue(torch.all(grams <= pool + 1e-4))

        x = torch.tensor([10.0, 70.0, 400.0])
        hill = _hill_centred(x, 70.0, 2.0, torch.tensor([6.0]))
        self.assertTrue(torch.isfinite(hill).all())
        self.assertTrue(torch.all((hill > 0) & (hill <= 2)))
        self.assertAlmostEqual(float(hill[1]), 1.0, places=6)

        met = MetabolicModule(4, hidden_dim=4)
        with torch.no_grad():
            met.log_k_ii.add_(8.0)
        self.assertGreater(float(met.k_ii().detach()), 0.0)
        glucose = torch.tensor([40.0, 95.0, 250.0])
        rate = met.k_ii()
        uptake = corrected_mass_action(rate, glucose, 95.0, torch.tensor([2.0]))
        self.assertTrue(torch.isfinite(uptake).all())
        self.assertTrue(torch.all(uptake >= 0))
        torch.testing.assert_close(uptake[1], rate * 95.0)
        plain = corrected_mass_action(rate, glucose, 95.0, torch.zeros(1))
        torch.testing.assert_close(plain, rate * glucose)


if __name__ == "__main__":
    unittest.main()
