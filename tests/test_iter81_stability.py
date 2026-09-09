"""Iter 81: forward-integration physiological clamp + calibration embedding leash.

These guard the two structural robustness fixes for the iter-80 benchmark
regression, which was traced to per-patient calibration walking the embedding
~9x off the trained manifold, where the unbounded forward integration detonated
glucose to ~18,000 mg/dL (catastrophic MAPE, loses to persistence). The fixes:

  1. ``integrate`` clamps the state to PHYSIOLOGICAL_MIN/MAX each Euler step —
     a no-op in-distribution, a hard catastrophe bound off-manifold.
  2. ``calibrate_embedding`` projects the embedding back onto ``||emb|| <=
     max_norm`` after each Adam step, keeping personalization on the manifold.
"""

from __future__ import annotations

import unittest

import torch

import pulse.model as model_mod
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


class TestIntegratorClamp(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = ModularPhysiologyNetwork()
        self.model.eval()
        self.init = torch.tensor(NORM_CENTER, dtype=torch.float32)
        self.meals = [MealEvent(time=60.0, carbs=75.0, fats=20.0, proteins=25.0)]
        self.lo = torch.tensor(PHYSIOLOGICAL_MIN, dtype=torch.float32)
        self.hi = torch.tensor(PHYSIOLOGICAL_MAX, dtype=torch.float32)

    def _run(self, emb: torch.Tensor, n_steps: int = 300) -> torch.Tensor:
        with torch.no_grad():
            return integrate(
                self.model, self.init, emb, n_steps, dt=1.0,
                start_time_minutes=360.0, meals=self.meals,
            )

    def test_offmanifold_trajectory_stays_bounded(self) -> None:
        # A wildly off-manifold embedding must not produce a nonphysical /
        # non-finite trajectory: every state stays within the clamp bounds.
        for seed in range(5):
            g = torch.Generator().manual_seed(seed)
            v = torch.randn(self.model.embedding_dim, generator=g)
            emb = v / v.norm() * 30.0          # 30 >> trained ||emb|| ~ 1.2
            traj = self._run(emb)
            self.assertTrue(torch.isfinite(traj).all(), f"non-finite traj seed={seed}")
            # +1e-3 tolerance for the clamp boundary itself.
            self.assertTrue((traj >= self.lo - 1e-3).all(), f"below phys min seed={seed}")
            self.assertTrue((traj <= self.hi + 1e-3).all(), f"above phys max seed={seed}")

    def test_glucose_cannot_detonate(self) -> None:
        # The specific iter-80 failure: off-manifold + meal drove glucose to
        # ~18,000 mg/dL. It must now be bounded by glucose's physiological max.
        gi = MARKER_INDEX["glucose"]
        g = torch.Generator().manual_seed(7)
        v = torch.randn(self.model.embedding_dim, generator=g)
        emb = v / v.norm() * 20.0
        traj = self._run(emb)
        self.assertLessEqual(float(traj[:, gi].max()), PHYSIOLOGICAL_MAX[gi] + 1e-3)

    def test_clamp_is_noop_on_in_range_trajectory(self) -> None:
        # The clamp must not alter a trajectory that stays within the
        # physiological bounds — it is a pure off-manifold safety. Use a
        # zero-rate stub so the state stays exactly at NORM_CENTER (in range,
        # deterministic): the integrated trajectory must be every-step
        # NORM_CENTER, untouched by the clamp. (A real random model diverges to
        # the bounds within tens of steps, so it can't witness this property.)
        class _ZeroRate(torch.nn.Module):
            embedding_dim = self.model.embedding_dim
            def forward(self, state, embedding, t, meals, **kw):
                return torch.zeros_like(state)

        with torch.no_grad():
            traj = integrate(_ZeroRate(), self.init, torch.zeros(self.model.embedding_dim),
                             200, dt=1.0, start_time_minutes=360.0, meals=[])
        expected = self.init.expand_as(traj)
        self.assertTrue(torch.equal(traj, expected))

    def test_clamp_pulls_out_of_range_initial_state_into_bounds(self) -> None:
        # An initial state outside the bounds (or rates driving it out) must be
        # clamped back: zero rates from a 100x-NORM_CENTER start collapse to the
        # physiological ceiling on the first step.
        class _ZeroRate(torch.nn.Module):
            embedding_dim = self.model.embedding_dim
            def forward(self, state, embedding, t, meals, **kw):
                return torch.zeros_like(state)

        wild = self.init * 100.0
        with torch.no_grad():
            traj = integrate(_ZeroRate(), wild, torch.zeros(self.model.embedding_dim),
                             10, dt=1.0, start_time_minutes=360.0, meals=[])
        # Row 0 is the (unclamped) initial state; every subsequent row is bounded.
        self.assertTrue((traj[1:] <= self.hi + 1e-3).all())
        self.assertTrue((traj[1:] >= self.lo - 1e-3).all())


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
