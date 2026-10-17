"""Second-phase insulin is the teacher's delayed potentiator, and the 120-minute anchor is registered.

At a constant glucose the potentiator sits at (G − Gb)+ and secretion is the
memoryless law. When glucose falls, secretion is still the glucose of the
previous half hour. Fasting, with the potentiator at 0, is unchanged.
"""

from __future__ import annotations

import unittest

import torch

from pulse.knowledge.cohort_statistics import ALL_COHORT_STATISTICS
from pulse.knowledge.full_body import PatientParams
from pulse.model import ModularPhysiologyNetwork
from pulse.modules.base import compute_time_features
from pulse.modules.metabolic import (
    _GLUCOSE_IDX, _K_POT, _PHASE2_FRAC, _POT_IDX,
)
from pulse.types import MODULE_MARKER_INDICES


def _metabolic():
    torch.manual_seed(0)
    model = ModularPhysiologyNetwork(
        metabolic_hidden=16, appetite_hidden=8, stress_hidden=8,
        cardiovascular_hidden=8, thermoreg_hidden=8, respiratory_hidden=8,
        gut_hidden=8, hepatobiliary_hidden=8,
    )
    model.eval()
    return model.metabolic


def _fluxes(met, glucose: float, pot: float):
    n = len(MODULE_MARKER_INDICES["metabolic"])
    state = torch.zeros(1, n)
    state[0, _GLUCOSE_IDX] = (glucose - 95.0) / 30.0
    state[0, _POT_IDX] = pot / 30.0
    emb = torch.zeros(1, met.embedding_dim)
    coupling = torch.zeros(1, met.n_coupling)
    external = torch.tensor([[0.0, 1.0, 0.0]])
    tf = compute_time_features(torch.tensor([8.0 * 60.0]))
    with torch.no_grad():
        return met.fluxes(state, coupling, external, emb, tf)


class TestPotentiator(unittest.TestCase):
    def test_constants_are_the_teachers(self) -> None:
        params = PatientParams()
        self.assertEqual(_K_POT, params.k_pot)
        self.assertEqual(_PHASE2_FRAC, params.ins_phase2_frac)

    def test_fasting_secretion_is_zero(self) -> None:
        f = _fluxes(_metabolic(), 95.0, 0.0)
        self.assertEqual(float(f["ins_gsir"]), 0.0)
        self.assertEqual(float(f["pot_rate"]), 0.0)
        self.assertEqual(float(f["phase2_stim"]), 0.0)

    def test_a_new_rise_is_only_the_immediate_share(self) -> None:
        """Pot still at 0, glucose 40 above basal: 70% of the rise, and Pot is filling."""
        f = _fluxes(_metabolic(), 135.0, 0.0)
        self.assertAlmostEqual(float(f["phase2_stim"]), 0.7 * 40.0, places=4)
        self.assertAlmostEqual(float(f["pot_rate"]), _K_POT * 40.0, places=5)
        self.assertGreater(float(f["ins_gsir"]), 0.0)

    def test_equilibrium_is_the_memoryless_law(self) -> None:
        f = _fluxes(_metabolic(), 135.0, 40.0)
        self.assertAlmostEqual(float(f["phase2_stim"]), 40.0, places=4)
        self.assertAlmostEqual(float(f["pot_rate"]), 0.0, places=5)

    def test_a_fall_keeps_the_previous_glucose(self) -> None:
        """Glucose back at basal, Pot still 40: secretion is w·Pot, not zero."""
        f = _fluxes(_metabolic(), 95.0, 40.0)
        self.assertAlmostEqual(float(f["g_above"]), 0.0, places=5)
        self.assertAlmostEqual(float(f["phase2_stim"]), _PHASE2_FRAC * 40.0, places=4)
        self.assertLess(float(f["pot_rate"]), 0.0)
        self.assertGreater(float(f["ins_gsir"]), 0.0)


class TestOgttTailAnchor(unittest.TestCase):
    def test_registered_on_the_ogtt_arm(self) -> None:
        spec = next(s for s in ALL_COHORT_STATISTICS if s.name == "ogtt_75g_insulin_120min")
        self.assertEqual(spec.marker_id, "insulin")
        self.assertEqual(spec.target, 34.6)
        self.assertEqual(spec.sigma, 1.44)
        self.assertEqual(spec.n_arm, 1)
        self.assertEqual(spec.window.start_min, 170)
        self.assertEqual(spec.window.end_min, 190)
        self.assertLessEqual(spec.window.end_min, spec.arms[0].duration_min)
        self.assertEqual(spec.arms[0].meals[0][1], 75.0)
