"""StressModule — CRH → ACTH → cortisol, in the normalized runtime frame."""

from __future__ import annotations

import unittest

import torch

from pulse.modules.base import compute_time_features
from pulse.modules.stress import StressModule, _ACTH_IDX, _CORTISOL_IDX, _CRH_IDX, _ACTH_PER_CRH
from pulse.types import (
    EMBEDDING_DIM, MARKER_INDEX, MODULE_MARKER_INDICES, NORM_CENTER, NORM_SCALE,
    TIME_FEATURES_DIM,
)

_STRESS = MODULE_MARKER_INDICES["stress"]
_CENTER = torch.tensor([NORM_CENTER[i] for i in _STRESS])
_SCALE = torch.tensor([NORM_SCALE[i] for i in _STRESS])


def _norm(cortisol: float, acth: float, crh: float = 100.0) -> torch.Tensor:
    raw = torch.tensor([[cortisol, acth, crh]], dtype=torch.float32)
    return (raw - _CENTER) / _SCALE


def _time(hour: float) -> torch.Tensor:
    return compute_time_features(torch.tensor([hour * 60.0]))


_BALANCED = _time(0.0)


class TestStressModuleStructure(unittest.TestCase):
    def test_crh_is_internal_marker_in_stress_module(self) -> None:
        self.assertIn("crh", MARKER_INDEX)
        self.assertEqual(len(_STRESS), 3)
        self.assertIn(MARKER_INDEX["crh"], _STRESS)

    def test_module_local_index_order(self) -> None:
        self.assertEqual(_CORTISOL_IDX, 0)
        self.assertEqual(_ACTH_IDX, 1)
        self.assertEqual(_CRH_IDX, 2)

    def test_module_accepts_normalized_state_and_4_time_features(self) -> None:
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
    def _module(self) -> StressModule:
        torch.manual_seed(0)
        return StressModule(embedding_dim=EMBEDDING_DIM)

    def test_higher_crh_raises_acth_rate(self) -> None:
        module = self._module()
        args = (torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1, EMBEDDING_DIM), _BALANCED)
        with torch.no_grad():
            r_hi = module(_norm(12.0, 30.0, 150.0), *args)
            r_lo = module(_norm(12.0, 30.0, 50.0), *args)
        self.assertGreater(float(r_hi[0, _ACTH_IDX]), float(r_lo[0, _ACTH_IDX]))

    def test_acth_raises_cortisol_rate_proportionally(self) -> None:
        module = self._module()
        args = (torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1, EMBEDDING_DIM), _BALANCED)
        with torch.no_grad():
            r30 = module(_norm(12.0, 30.0, 100.0), *args)
            r48 = module(_norm(12.0, 48.0, 100.0), *args)
        self.assertGreater(float(r48[0, _CORTISOL_IDX]), float(r30[0, _CORTISOL_IDX]))

    def test_cortisol_feedback_is_saturating_not_rectified_at_12(self) -> None:
        """High cortisol lowers CRH. Low overnight cortisol does not raise it
        (the nadir is sleep's); a rectifier at typical would leave the
        whole overnight range inert, and two-sided feedback would fight sleep."""
        module = self._module()
        args = (torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1, EMBEDDING_DIM), _BALANCED)
        with torch.no_grad():
            r18 = module(_norm(18.0, 30.0, 100.0), *args)
            r12 = module(_norm(12.0, 30.0, 100.0), *args)
            r6 = module(_norm(6.0, 30.0, 100.0), *args)
        self.assertLess(float(r18[0, _CRH_IDX]), float(r12[0, _CRH_IDX]))
        self.assertAlmostEqual(float(r6[0, _CRH_IDX]), float(r12[0, _CRH_IDX]), places=5)

    def test_feedback_gain_receives_gradient(self) -> None:
        module = self._module()
        rate = module(_norm(20.0, 30.0, 100.0), torch.zeros(1, 2), torch.zeros(1, 2),
                      torch.zeros(1, EMBEDDING_DIM), _BALANCED)
        rate[0, _CRH_IDX].backward()
        self.assertIsNotNone(module._fb_raw.grad)
        self.assertNotEqual(float(module._fb_raw.grad), 0.0)

    def test_diurnal_drive_enters_crh_not_cortisol(self) -> None:
        module = self._module()
        state = _norm(12.0, 30.0, 100.0)
        args = (state, torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1, EMBEDDING_DIM))
        with torch.no_grad():
            morning = module(*args, _time(6.5))
            evening = module(*args, _time(20.0))
        self.assertGreater(float(morning[0, _CRH_IDX]), float(evening[0, _CRH_IDX]))
        self.assertAlmostEqual(float(morning[0, _CORTISOL_IDX]), float(evening[0, _CORTISOL_IDX]),
                               places=6)

    def test_hypoglycaemia_raises_crh(self) -> None:
        module = self._module()
        g_lo = torch.tensor([[(55.0 - 95.0) / 30.0, 0.0]])
        g_hi = torch.tensor([[(95.0 - 95.0) / 30.0, 0.0]])
        ext = torch.zeros(1, 2)
        emb = torch.zeros(1, EMBEDDING_DIM)
        state = _norm(12.0, 30.0, 100.0)
        with torch.no_grad():
            r_lo = module(state, g_lo, ext, emb, _BALANCED)
            r_hi = module(state, g_hi, ext, emb, _BALANCED)
        self.assertGreater(float(r_lo[0, _CRH_IDX]), float(r_hi[0, _CRH_IDX]))

    def test_acth_tracks_crh_at_the_structural_gain(self) -> None:
        module = self._module()
        crh = 80.0
        acth = _ACTH_PER_CRH * crh
        rate = module(_norm(12.0, acth, crh), torch.zeros(1, 2), torch.zeros(1, 2),
                      torch.zeros(1, EMBEDDING_DIM), _BALANCED)
        self.assertAlmostEqual(float(rate[0, _ACTH_IDX]), 0.0, places=4)

    def test_mechanism_is_continuous_across_midnight(self) -> None:
        module = self._module()
        args = (_norm(12.0, 30.0, 100.0), torch.zeros(1, 2), torch.zeros(1, 2),
                torch.zeros(1, EMBEDDING_DIM))
        with torch.no_grad():
            before = module(*args, compute_time_features(torch.tensor([1439.99])))
            after = module(*args, compute_time_features(torch.tensor([0.01])))
        torch.testing.assert_close(before, after, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
