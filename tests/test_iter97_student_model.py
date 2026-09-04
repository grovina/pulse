"""Iter 97 — model-level contracts: periodic time features (3.1), learned
default inputs conditioned on embedding and time with rest = 0 (1.7), the
``integrate(..., duodenal_outputs=)`` path (4.7), checkpoint round-trip through
``from_checkpoint`` and the legacy loader, and the from-init sanity rollout."""

from __future__ import annotations

import inspect
import io
import math
import unittest

import torch

from pulse.model import (
    ModularPhysiologyNetwork, integrate, precompute_duodenal_outputs, precompute_gut_outputs,
)
from pulse.modules.base import compute_time_features
from pulse.modules.gut import MealEvent
from pulse.types import (
    EMBEDDING_DIM, MARKER_INDEX as MI, NORM_CENTER, PHYSIOLOGICAL_MAX, PHYSIOLOGICAL_MIN,
    TIME_FEATURES_DIM,
)

_MEALS = [MealEvent(120, 60, 20, 25), MealEvent(420, 80, 25, 30), MealEvent(780, 70, 25, 35)]


def _small(seed=0, perturb=0.3):
    torch.manual_seed(seed)
    m = ModularPhysiologyNetwork(
        metabolic_hidden=16, appetite_hidden=16, stress_hidden=16, cardiovascular_hidden=16,
        thermoreg_hidden=16, respiratory_hidden=16, gut_hidden=16, hepatobiliary_hidden=16)
    if perturb:
        with torch.no_grad():
            for p in m.parameters():
                p.add_(perturb * torch.randn_like(p))
    m.eval()
    return m


class TestTimeFeatures(unittest.TestCase):
    def test_four_periodic_features_no_ramp(self) -> None:
        f = compute_time_features(torch.tensor([0.0, 360.0, 720.0]))
        self.assertEqual(f.shape, (3, TIME_FEATURES_DIM))
        self.assertEqual(TIME_FEATURES_DIM, 4)
        torch.testing.assert_close(f[0], torch.tensor([0.0, 1.0, 0.0, 1.0]))
        torch.testing.assert_close(f[1], torch.tensor([1.0, 0.0, 0.0, -1.0]), atol=1e-6, rtol=0)

    def test_continuous_across_midnight(self) -> None:
        a = compute_time_features(torch.tensor(1439.999))
        b = compute_time_features(torch.tensor(0.001))
        torch.testing.assert_close(a, b, atol=1e-4, rtol=0)

    def test_vector_field_has_no_midnight_jump(self) -> None:
        """The reviewer measured −116 bpm/h on HR and −9 C/h on temperature across 00:00."""
        m = _small(0, perturb=1.0)
        typ = torch.tensor(NORM_CENTER).unsqueeze(0)
        emb = torch.randn(1, EMBEDDING_DIM)
        kw = dict(sleep_wake=torch.zeros(1), activity=torch.zeros(1),
                  gut_override=torch.zeros(1, 4), duodenal_override=torch.zeros(1, 2))
        with torch.no_grad():
            before = m(typ, emb, torch.tensor([1439.99]), [], **kw)
            after = m(typ, emb, torch.tensor([0.01]), [], **kw)
        # 0.02 min apart on a continuous field: float32 noise (~1e-3/min ≈ 0.08/h
        # measured), against the old ramp's −116 bpm/h.
        self.assertLess(float((after - before).abs().max()), 1e-2)


class TestDefaultInputs(unittest.TestCase):
    def test_defaults_are_conditioned_on_embedding_and_time(self) -> None:
        m = _small(1, perturb=1.0)
        e = torch.randn(4, EMBEDDING_DIM)
        t1 = compute_time_features(torch.full((4,), 180.0))
        t2 = compute_time_features(torch.full((4,), 900.0))
        with torch.no_grad():
            sw1, act1 = m.default_external_inputs(e, t1)
            sw2, act2 = m.default_external_inputs(e, t2)
            sw3, _ = m.default_external_inputs(torch.randn(4, EMBEDDING_DIM), t1)
        self.assertFalse(torch.allclose(sw1, sw2))
        self.assertFalse(torch.allclose(sw1, sw3))
        self.assertTrue(bool(((sw1 > 0) & (sw1 < 1) & (act1 >= 0) & (act1 < 1)).all()))

    def test_fresh_default_activity_is_near_rest_not_0_1(self) -> None:
        m = _small(2, perturb=0.0)
        with torch.no_grad():
            sw, act = m.default_external_inputs(torch.zeros(1, EMBEDDING_DIM), compute_time_features(torch.tensor([600.0])))
        self.assertAlmostEqual(float(sw), 0.5, places=6)
        self.assertLess(float(act), 0.02)

    def test_rest_is_representable(self) -> None:
        m = _small(3, perturb=0.0)
        with torch.no_grad():
            m.default_inputs_net[-1].bias[1] = -40.0
            _, act = m.default_external_inputs(torch.randn(8, EMBEDDING_DIM), compute_time_features(torch.rand(8) * 1440))
        self.assertLess(float(act.max()), 1e-12)

    def test_provided_sleep_gates_a_defaulted_activity(self) -> None:
        """sleep_wake given as 0 with no activity log ⇒ activity exactly 0."""
        m = _small(4, perturb=1.0)
        typ = torch.tensor(NORM_CENTER).unsqueeze(0).expand(2, -1)
        emb = torch.randn(2, EMBEDDING_DIM)
        kw = dict(gut_override=torch.zeros(2, 4), duodenal_override=torch.zeros(2, 2))
        with torch.no_grad():
            r_default = m(typ, emb, torch.full((2,), 120.0), [], sleep_wake=torch.zeros(2), **kw)
            r_rest = m(typ, emb, torch.full((2,), 120.0), [], sleep_wake=torch.zeros(2), activity=torch.zeros(2), **kw)
        torch.testing.assert_close(r_default, r_rest, atol=1e-6, rtol=1e-6)

    def test_old_scalar_defaults_are_gone(self) -> None:
        m = _small(5, perturb=0.0)
        self.assertFalse(hasattr(m, "default_activity"))
        self.assertFalse(hasattr(m, "default_sleep_wake"))


class TestIntegrateDuodenalOutputs(unittest.TestCase):
    def test_signature(self) -> None:
        params = inspect.signature(integrate).parameters
        self.assertIn("duodenal_outputs", params)
        self.assertIn("gut_outputs", params)
        self.assertIsNone(params["duodenal_outputs"].default)

    def test_precomputed_delivery_reproduces_the_internal_path(self) -> None:
        m = _small(6)
        n = 300
        typ = torch.tensor(NORM_CENTER)
        emb = torch.zeros(EMBEDDING_DIM)
        duo = precompute_duodenal_outputs(m, n, meals=_MEALS[:1])
        self.assertEqual(tuple(duo.shape), (n, 2))
        with torch.no_grad():
            a = integrate(m, typ, emb, n, meals=_MEALS[:1])
            b = integrate(m, typ, emb, n, meals=_MEALS[:1], duodenal_outputs=duo)
            c = integrate(m, typ, emb, n, meals=_MEALS[:1], duodenal_outputs=torch.zeros(n, 2))
        torch.testing.assert_close(a, b)
        self.assertGreater(float((a[:, MI["cck"]] - c[:, MI["cck"]]).abs().max()), 0.1)

    def test_batched_integrate_accepts_shared_duodenal_outputs(self) -> None:
        m = _small(7)
        n = 200
        emb = torch.randn(3, EMBEDDING_DIM)
        typ = torch.tensor(NORM_CENTER).unsqueeze(0).expand(3, -1)
        gut = precompute_gut_outputs(m, emb, n, meals=_MEALS[:1])
        duo = precompute_duodenal_outputs(m, n, meals=_MEALS[:1])
        with torch.no_grad():
            out = integrate(m, typ, emb, n, meals=_MEALS[:1], gut_outputs=gut, duodenal_outputs=duo)
        self.assertEqual(tuple(out.shape), (3, n, len(NORM_CENTER)))


class TestCheckpointRoundTrip(unittest.TestCase):
    def _roundtrip(self, ckpt):
        buf = io.BytesIO(); torch.save(ckpt, buf); buf.seek(0)
        return torch.load(buf, weights_only=False)

    def test_from_checkpoint_with_model_config(self) -> None:
        m = _small(8)
        ckpt = self._roundtrip({"model_state": m.state_dict(), "model_config": m.constructor_kwargs})
        m2 = ModularPhysiologyNetwork.from_checkpoint(ckpt)
        self.assertEqual(m2.constructor_kwargs, m.constructor_kwargs)
        typ = torch.tensor(NORM_CENTER)
        with torch.no_grad():
            a = integrate(m, typ, torch.zeros(EMBEDDING_DIM), 50, meals=_MEALS[:1])
            b = integrate(m2, typ, torch.zeros(EMBEDDING_DIM), 50, meals=_MEALS[:1])
        torch.testing.assert_close(a, b)

    def test_legacy_loader_still_builds_the_new_layout(self) -> None:
        from pulse.diagnostics.probe import load_model_from_checkpoint
        import tempfile, os
        h = 48
        m = ModularPhysiologyNetwork(
            metabolic_hidden=h, appetite_hidden=max(24, h // 2), stress_hidden=max(24, h // 2),
            cardiovascular_hidden=h, thermoreg_hidden=max(16, h // 3), respiratory_hidden=max(16, h // 3))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.pt")
            torch.save({"model_state": m.state_dict(), "hidden_dim": h, "embedding_dim": EMBEDDING_DIM}, path)
            m2, ckpt = load_model_from_checkpoint(path)
            m3 = ModularPhysiologyNetwork.from_checkpoint(ckpt)
        for a, b in zip(m.state_dict().values(), m2.state_dict().values()):
            self.assertTrue(torch.equal(a, b))
        self.assertEqual(m3.constructor_kwargs["metabolic_hidden"], h)


class TestFromInitSanity(unittest.TestCase):
    """hidden 48, zero embedding, 24 h with a 3-meal day and with no meals."""

    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)
        h = 48
        cls.model = ModularPhysiologyNetwork(
            metabolic_hidden=h, appetite_hidden=max(24, h // 2), stress_hidden=max(24, h // 2),
            cardiovascular_hidden=h, thermoreg_hidden=max(16, h // 3), respiratory_hidden=max(16, h // 3))
        cls.model.eval()
        n = 1440
        sw = torch.ones(n)
        for s in range(n):
            t = (360 + s) % 1440
            if t >= 1380 or t < 420:
                sw[s] = 0.0
        typ = torch.tensor(NORM_CENTER)
        with torch.no_grad():
            cls.fed = integrate(cls.model, typ, torch.zeros(EMBEDDING_DIM), n, start_time_minutes=360,
                                meals=_MEALS, sleep_wake=sw, activity=torch.zeros(n))
            cls.fasted = integrate(cls.model, typ, torch.zeros(EMBEDDING_DIM), n, start_time_minutes=360,
                                   meals=[], sleep_wake=sw, activity=torch.zeros(n))

    def _check(self, tr):
        self.assertFalse(bool(torch.isnan(tr).any()))
        lo = torch.tensor(PHYSIOLOGICAL_MIN); hi = torch.tensor(PHYSIOLOGICAL_MAX)
        at_clamp = ((tr <= lo + 1e-6) | (tr >= hi - 1e-6)).any(dim=0)
        # insulin_action's floor IS its fasting value (0); every other marker must be free.
        at_clamp[MI["insulin_action"]] = False
        self.assertFalse(bool(at_clamp.any()), msg=f"at clamp: {[i for i in range(len(at_clamp)) if at_clamp[i]]}")
        g, hr, temp = tr[:, MI["glucose"]], tr[:, MI["hr"]], tr[:, MI["temp"]]
        self.assertTrue(60 <= float(g.min()) and float(g.max()) <= 180, msg=f"glucose {float(g.min())}-{float(g.max())}")
        self.assertTrue(40 <= float(hr.min()) and float(hr.max()) <= 110, msg=f"hr {float(hr.min())}-{float(hr.max())}")
        self.assertTrue(36 <= float(temp.min()) and float(temp.max()) <= 38, msg=f"temp {float(temp.min())}-{float(temp.max())}")
        self.assertTrue(bool((tr[:, MI["sbp"]] > tr[:, MI["dbp"]]).all()))
        self.assertTrue(bool((tr[:, MI["hrv"]] > 0).all()))

    def test_three_meal_day(self) -> None:
        self._check(self.fed)
        self.assertGreater(float(self.fed[:, MI["glucose"]].max()), 105.0)  # meals are visible

    def test_no_meal_day(self) -> None:
        self._check(self.fasted)


if __name__ == "__main__":
    unittest.main()
