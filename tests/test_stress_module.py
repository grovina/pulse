"""StressModule (iter 64 compact cascade) — structural tests, in the NORMALIZED frame.

The cascade has three state species (cortisol, acth, crh). CRH is retained
as state (so the iter-69 23-D benchmark dataset works without regeneration)
but is mechanically inert — its rate comes from its SpeciesHead only. The
iter-64 compact mechanism, as of iter 96/97, is:

    cortisol_rate +=  α·(ACTH / typical_ACTH)·prod_scale_cortisol
    acth_rate     +=  γ·(1 + diurnal_carrier)·prod_scale_acth
                  −   β·relu((cortisol − typical) / typical)·prod_scale_acth

ITER 97: these tests feed the module what ``model.py`` feeds it — the
NORMALIZED state ``(raw − typical) / NORM_SCALE`` and the 4-dim time features
``[sin θ, cos θ, sin 2θ, cos 2θ]``. Through iter 96 they fed RAW values, which
is exactly why the frame bug the 2026-09-04 review found (item 1.1) passed
every test: ``state[..., idx]`` read as raw made ``relu(ACTH/30)`` really
``relu((ACTH − 30)/540)`` = 0 for all ACTH ≤ 30 and the cortisol feedback
inert until cortisol > 108 µg/dL; ``_beta_raw`` sat at its init after 21 h of
training. A test that hands a module a frame it never sees at runtime tests
nothing.
"""

from __future__ import annotations

import unittest

import torch

from pulse.modules.base import MassActionModule, compute_time_features
from pulse.modules.stress import StressModule, _ACTH_IDX, _CORTISOL_IDX, _CRH_IDX
from pulse.types import (
    EMBEDDING_DIM, MARKER_INDEX, MODULE_MARKER_INDICES, NORM_CENTER, NORM_SCALE,
    TIME_FEATURES_DIM,
)

_STRESS = MODULE_MARKER_INDICES["stress"]
_CENTER = torch.tensor([NORM_CENTER[i] for i in _STRESS])
_SCALE = torch.tensor([NORM_SCALE[i] for i in _STRESS])


def _norm(cortisol: float, acth: float, crh: float = 100.0) -> torch.Tensor:
    """Raw (µg/dL, pg/mL, pg/mL) → the normalized module state ``[1, 3]``."""
    raw = torch.tensor([[cortisol, acth, crh]], dtype=torch.float32)
    return (raw - _CENTER) / _SCALE


def _time(hour: float) -> torch.Tensor:
    return compute_time_features(torch.tensor([hour * 60.0]))


# A phase where the first harmonic is balanced (sin = 0, cos = 1): 00:00.
_BALANCED = _time(0.0)


class TestStressModuleStructure(unittest.TestCase):
    def test_crh_is_internal_marker_in_stress_module(self) -> None:
        self.assertIn("crh", MARKER_INDEX)
        self.assertEqual(len(_STRESS), 3)
        self.assertIn(MARKER_INDEX["cortisol"], _STRESS)
        self.assertIn(MARKER_INDEX["acth"], _STRESS)
        self.assertIn(MARKER_INDEX["crh"], _STRESS)

    def test_module_local_index_order(self) -> None:
        self.assertEqual(_CORTISOL_IDX, 0)
        self.assertEqual(_ACTH_IDX, 1)
        self.assertEqual(_CRH_IDX, 2)

    def test_module_accepts_normalized_state_and_4_time_features(self) -> None:
        torch.manual_seed(0)
        module = StressModule(embedding_dim=EMBEDDING_DIM)
        self.assertEqual(_BALANCED.shape[-1], TIME_FEATURES_DIM)
        rate = module(_norm(12.0, 30.0), torch.zeros(1, 2), torch.zeros(1, 2),
                      torch.zeros(1, EMBEDDING_DIM), _BALANCED)
        self.assertEqual(rate.shape, (1, 3))

    def test_raw_state_round_trips(self) -> None:
        module = StressModule(embedding_dim=EMBEDDING_DIM)
        raw = module.raw_state(_norm(4.4, 8.0, 60.0))
        torch.testing.assert_close(raw, torch.tensor([[4.4, 8.0, 60.0]]))


class TestCascadeMechanism(unittest.TestCase):
    """Drive one stage at a time and confirm the downstream species responds with the
    correct sign, isolating the mechanism (module − mass-action base)."""

    def _module(self) -> StressModule:
        torch.manual_seed(0)
        m = StressModule(embedding_dim=EMBEDDING_DIM)
        for p in (m._alpha_raw, m._gamma_raw, m._beta_raw):
            with torch.no_grad():
                p.fill_(0.0)
        return m

    def _mech(self, module: StressModule, state: torch.Tensor,
              time_features: torch.Tensor = _BALANCED) -> torch.Tensor:
        args = (state, torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1, EMBEDDING_DIM),
                time_features)
        with torch.no_grad():
            return module(*args) - MassActionModule.forward(module, *args)

    def test_acth_drive_is_nonzero_and_proportional_at_30_and_48(self) -> None:
        """The review's case: ACTH = 30 and 48 pg/mL. Through iter 96 the drive at
        30 was exactly zero and at 48 was relu((48−30)/540)·α — 1/48 of the intended
        value. It must be non-zero at both and scale with ACTH."""
        module = self._module()
        d30 = float(self._mech(module, _norm(12.0, 30.0))[0, _CORTISOL_IDX])
        d48 = float(self._mech(module, _norm(12.0, 48.0))[0, _CORTISOL_IDX])
        self.assertGreater(d30, 0.0)
        self.assertGreater(d48, d30)
        self.assertAlmostEqual(d48 / d30, 48.0 / 30.0, places=5)

    def test_acth_above_typical_raises_cortisol_rate(self) -> None:
        module = self._module()
        args = (torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1, EMBEDDING_DIM), _BALANCED)
        with torch.no_grad():
            r_hi = module(_norm(12.0, 45.0), *args)
            r_lo = module(_norm(12.0, 30.0), *args)
        self.assertGreater(float(r_hi[0, _CORTISOL_IDX]), float(r_lo[0, _CORTISOL_IDX]))

    def test_cortisol_drive_is_proportional_below_typical_acth(self) -> None:
        """The ACTH→cortisol map is not flat below typical: ACTH 8 vs 4 pg/mL — the
        overnight range the nadir lives in — must give different drives."""
        module = self._module()
        d8 = float(self._mech(module, _norm(12.0, 8.0))[0, _CORTISOL_IDX])
        d4 = float(self._mech(module, _norm(12.0, 4.0))[0, _CORTISOL_IDX])
        self.assertGreater(d8, d4)
        self.assertGreater(d4, 0.0)

    def test_cortisol_above_typical_lowers_acth_rate_at_physiological_levels(self) -> None:
        """β feedback must fire at cortisol 18 µg/dL (+50 %), not only above 108."""
        module = self._module()
        m18 = float(self._mech(module, _norm(18.0, 30.0))[0, _ACTH_IDX])
        m12 = float(self._mech(module, _norm(12.0, 30.0))[0, _ACTH_IDX])
        self.assertLess(m18, m12)
        beta = float(torch.nn.functional.softplus(module._beta_raw))
        self.assertAlmostEqual(m12 - m18, beta * 0.5 * float(module.prod_scale[_ACTH_IDX]), places=5)

    def test_beta_receives_gradient_in_the_runtime_frame(self) -> None:
        """The iter-96 checkpoint's `_beta_raw` was bit-identical to its init: no
        gradient path. With the raw-frame read, a cortisol excursion above typical must
        reach it."""
        module = self._module()
        state = _norm(20.0, 30.0)
        rate = module(state, torch.zeros(1, 2), torch.zeros(1, 2),
                      torch.zeros(1, EMBEDDING_DIM), _BALANCED)
        rate[0, _ACTH_IDX].backward()
        self.assertIsNotNone(module._beta_raw.grad)
        self.assertNotEqual(float(module._beta_raw.grad), 0.0)

    def test_diurnal_carrier_drives_acth(self) -> None:
        """γ·(1+diurnal_carrier) with carrier = 1 + sin θ at zero phase: 06:00 → 2,
        18:00 → 0 — so morning ACTH mechanism > evening."""
        module = self._module()
        state = _norm(12.0, 30.0)
        m_morning = float(self._mech(module, state, _time(6.0))[0, _ACTH_IDX])
        m_evening = float(self._mech(module, state, _time(18.0))[0, _ACTH_IDX])
        self.assertGreater(m_morning, m_evening)

    def test_mechanism_is_continuous_across_midnight(self) -> None:
        """Iter 97: with the linear ramp gone the time features are periodic, so the
        mechanism (and the base) agree at 23:59.99 and 00:00.01."""
        module = self._module()
        state = _norm(12.0, 30.0)
        args = (state, torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1, EMBEDDING_DIM))
        with torch.no_grad():
            before = module(*args, compute_time_features(torch.tensor([1439.99])))
            after = module(*args, compute_time_features(torch.tensor([0.01])))
        torch.testing.assert_close(before, after, atol=1e-4, rtol=1e-4)

    def test_crh_carries_no_term_in_the_iter64_mechanism(self) -> None:
        module = self._module()
        low = self._mech(module, _norm(12.0, 30.0, 50.0))
        high = self._mech(module, _norm(12.0, 30.0, 200.0))
        for idx, name in ((_CORTISOL_IDX, "cortisol"), (_ACTH_IDX, "acth")):
            self.assertAlmostEqual(float(low[0, idx]), float(high[0, idx]), places=6,
                                   msg=f"iter-64 mechanism routes CRH into {name}")

    def test_cortisol_mechanism_carries_no_independent_circadian(self) -> None:
        """Iter 96: cortisol is driven by ACTH alone — no second circadian."""
        module = self._module()
        state = _norm(12.0, 30.0)
        m_m = self._mech(module, state, _time(6.0))
        m_e = self._mech(module, state, _time(18.0))
        self.assertAlmostEqual(float(m_m[0, _CORTISOL_IDX]), float(m_e[0, _CORTISOL_IDX]),
                               places=6)
        self.assertGreater(float(m_m[0, _ACTH_IDX]), float(m_e[0, _ACTH_IDX]))


if __name__ == "__main__":
    unittest.main()
