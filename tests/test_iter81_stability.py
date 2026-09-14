"""Iter 81: forward-integration physiological clamp + calibration embedding leash.

These guard the two structural robustness fixes for the iter-80 benchmark
regression, which was traced to per-patient calibration walking the embedding
~9x off the trained manifold, where the unbounded forward integration detonated
glucose to ~18,000 mg/dL (catastrophic MAPE, loses to persistence). Iter 81 added
a post-Euler ±20σ box and the calibration embedding leash. Iter 101 deleted the
box after a train that did not explode; the leash is the remaining guard.
"""

from __future__ import annotations

import unittest

import torch

from pulse.benchmark import MeasurementPoint, calibrate_embedding
from pulse.model import ModularPhysiologyNetwork, integrate
from pulse.modules.gut import MealEvent
from pulse.types import (
    MARKER_INDEX, MARKERS, NORM_CENTER, STATE_DIM,
    PHYSIOLOGICAL_MIN, PHYSIOLOGICAL_MAX,
)


class TestPhysiologicalBounds(unittest.TestCase):
    def test_bounds_wellformed(self) -> None:
        self.assertEqual(len(PHYSIOLOGICAL_MIN), STATE_DIM)
        self.assertEqual(len(PHYSIOLOGICAL_MAX), STATE_DIM)
        for marker, lo, hi in zip(MARKERS, PHYSIOLOGICAL_MIN, PHYSIOLOGICAL_MAX):
            self.assertGreater(hi, lo)
            if marker.id == "insulin_action":
                self.assertLess(lo, 0.0)
            elif marker.id == "spo2":
                self.assertEqual(lo, 70.0)
                self.assertEqual(hi, 100.0)
            else:
                self.assertGreaterEqual(lo, 0.0)


class TestIntegratorHasNoPhysiologicalBox(unittest.TestCase):
    """Iter 101: integrate is unclamped Euler. The embedding leash is the
    off-manifold guard; a post-Euler box is not a second law of motion."""

    def test_zero_rate_from_in_range_state_is_identity(self) -> None:
        class _ZeroRate(torch.nn.Module):
            embedding_dim = 8
            def forward(self, state, embedding, t, meals, **kw):
                return torch.zeros_like(state)

        init = torch.tensor(NORM_CENTER, dtype=torch.float32)
        with torch.no_grad():
            traj = integrate(
                _ZeroRate(), init, torch.zeros(8),
                200, dt=1.0, start_time_minutes=360.0, meals=[],
            )
        self.assertTrue(torch.equal(traj, init.expand_as(traj)))

    def test_zero_rate_from_wild_initial_state_is_not_pulled_in(self) -> None:
        class _ZeroRate(torch.nn.Module):
            embedding_dim = 8
            def forward(self, state, embedding, t, meals, **kw):
                return torch.zeros_like(state)

        wild = torch.tensor(NORM_CENTER, dtype=torch.float32) * 100.0
        with torch.no_grad():
            traj = integrate(
                _ZeroRate(), wild, torch.zeros(8),
                10, dt=1.0, start_time_minutes=360.0, meals=[],
            )
        gi = MARKER_INDEX["glucose"]
        self.assertTrue(torch.allclose(traj[:, gi], wild[gi].expand(10)))
        self.assertGreater(float(traj[-1, gi]), PHYSIOLOGICAL_MAX[gi])


class TestGlucoseBaseline(unittest.TestCase):
    """Iter 81: per-patient fasting-glucose setpoint (embedding -> b_emb)."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = ModularPhysiologyNetwork()
        self.model.eval()
        self.init = torch.tensor(NORM_CENTER, dtype=torch.float32)
        self.gi = MARKER_INDEX["glucose"]

    def _fasting_eq(self, emb: torch.Tensor) -> float:
        with torch.no_grad():
            traj = integrate(self.model, self.init, emb, 400, dt=1.0,
                             start_time_minutes=360.0, meals=[])
        return float(traj[-1, self.gi])

    def test_zero_init_offset_is_zero(self) -> None:
        # Final layer zero-init => b_emb = 0 for any embedding at construction,
        # so the fasting setpoint is unchanged (Gb = 95) vs pre-iter-81.
        net = self.model.metabolic.glucose_baseline_net
        for v in (torch.zeros(self.model.embedding_dim), torch.randn(self.model.embedding_dim)):
            self.assertEqual(float(net(self.model.embedding_projections["metabolic"](v))), 0.0)

    def test_offset_shifts_fasting_setpoint_monotonically(self) -> None:
        # Forcing the baseline head to a negative vs positive constant must move
        # the fasting glucose equilibrium down vs up (the embedding now has
        # direct authority over the setpoint the SpeciesHead alone lacked).
        net = self.model.metabolic.glucose_baseline_net
        emb = torch.zeros(self.model.embedding_dim)
        with torch.no_grad():
            net[-1].weight.zero_(); net[-1].bias.fill_(-1.0)   # b_emb < 0 -> lower Gb
        low = self._fasting_eq(emb)
        with torch.no_grad():
            net[-1].bias.fill_(1.0)                            # b_emb > 0 -> higher Gb
        high = self._fasting_eq(emb)
        self.assertLess(low, high - 5.0)

    def test_gradient_reaches_baseline_net(self) -> None:
        # The fasting glucose level must be differentiable w.r.t. the baseline
        # head (so training can learn per-patient setpoints).
        emb = torch.zeros(self.model.embedding_dim, requires_grad=True)
        traj = integrate(self.model, self.init, emb, 60, dt=1.0,
                         start_time_minutes=360.0, meals=[])
        traj[-1, self.gi].backward()
        grads = [p.grad for p in self.model.metabolic.glucose_baseline_net.parameters()
                 if p.grad is not None]
        self.assertTrue(grads and any(float(g.abs().sum()) > 0 for g in grads))


class TestCalibrationLeash(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = ModularPhysiologyNetwork()
        self.model.eval()
        self.init = torch.tensor(NORM_CENTER, dtype=torch.float32)
        # A handful of observations the optimizer will chase hard.
        self.obs = [
            MeasurementPoint(time=120, marker_id="glucose", value=180.0),
            MeasurementPoint(time=240, marker_id="glucose", value=70.0),
            MeasurementPoint(time=120, marker_id="hr", value=110.0),
        ]
        self.meals = [MealEvent(time=60.0, carbs=75.0, fats=20.0, proteins=25.0)]

    def _calibrate(self, max_norm: float) -> torch.Tensor:
        return calibrate_embedding(
            model=self.model, observations=self.obs, initial_state=self.init,
            meals=self.meals, duration_min=300, start_time_minutes=360.0,
            n_steps=60, lr=0.1, l2_weight=0.0, max_norm=max_norm,
        ).embedding

    def test_norm_is_clamped(self) -> None:
        emb = self._calibrate(max_norm=0.5)
        self.assertLessEqual(float(emb.norm()), 0.5 + 1e-5)

    def test_disabled_when_nonpositive(self) -> None:
        # max_norm <= 0 disables the leash; with lr=0.1, 60 steps and no L2 the
        # embedding should grow past any small bound it would otherwise hit.
        emb = self._calibrate(max_norm=0.0)
        self.assertGreater(float(emb.norm()), 0.5)


if __name__ == "__main__":
    unittest.main()
