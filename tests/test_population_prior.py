"""The four former pins are parameters, and the prior holds their mode."""

from __future__ import annotations

import unittest

import torch

from pulse.model import ModularPhysiologyNetwork, integrate
from pulse.training.factors import calibration_prior_nll, project_point_mass
from pulse.modules.metabolic import (
    _F_GNG,
    _GN_CENTER,
    _K_FFA,
    _K_II,
    POPULATION_PRIOR_NAMES,
)
from pulse.training.population_prior_signal import population_prior_nll
from pulse.types import EMBEDDING_DIM, NORM_CENTER, SERIES_INDEX


def _model() -> ModularPhysiologyNetwork:
    torch.manual_seed(0)
    return ModularPhysiologyNetwork(
        metabolic_hidden=16, appetite_hidden=8, stress_hidden=8,
        cardiovascular_hidden=8, thermoreg_hidden=8, respiratory_hidden=8,
        gut_hidden=8, hepatobiliary_hidden=8,
    )


class TestPopulationPrior(unittest.TestCase):
    def test_init_is_the_mode(self) -> None:
        model = _model()
        met = model.metabolic
        self.assertAlmostEqual(float(met.k_ii().detach()), _K_II, places=5)
        self.assertAlmostEqual(float(met.f_gng_value().detach()), _F_GNG, places=5)
        self.assertAlmostEqual(float(met.k_ffa_value().detach()), _K_FFA, places=5)
        emb = torch.zeros(3, EMBEDDING_DIM)
        projected = model.embedding_projections["metabolic"](emb)
        gnb = met.gn_setpoint_raw(projected)
        self.assertTrue(torch.allclose(gnb, torch.full_like(gnb, _GN_CENTER)))

    def test_the_mode_is_a_critical_point_and_a_displacement_is_restoring(self) -> None:
        model = _model()
        nll = population_prior_nll(model.metabolic)
        nll.backward()
        for name in ("log_k_ii", "log_k_ffa", "log_gn_b"):
            grad = getattr(model.metabolic, name).grad
            self.assertIsNotNone(grad, name)
            self.assertAlmostEqual(float(grad), 0.0, places=4, msg=name)
        self.assertIsNone(model.metabolic.logit_f_gng.grad)

        model.zero_grad()
        with torch.no_grad():
            model.metabolic.log_k_ii.add_(0.05)
            model.metabolic.logit_f_gng.add_(0.2)
        population_prior_nll(model.metabolic).backward()
        self.assertGreater(float(model.metabolic.log_k_ii.grad), 0.0)
        self.assertIsNone(model.metabolic.logit_f_gng.grad)

    def test_there_is_no_per_person_copy(self) -> None:
        met = _model().metabolic
        names = [n for n, _ in met.named_parameters()]
        for forbidden in ("k_ii_net", "f_gng_net", "k_ffa_net", "gn_baseline_net"):
            self.assertFalse(any(forbidden in n for n in names))
        self.assertEqual(POPULATION_PRIOR_NAMES, ("k_ii", "k_ffa", "gn_b"))


class TestLandauFraction(unittest.TestCase):
    def test_a_point_mass_coordinate_is_not_divided_by_its_width(self) -> None:
        mean = torch.zeros(4)
        std = torch.tensor([1.0, 1.0, 0.0, 0.0])
        code = torch.tensor([1.0, 0.0, 0.5, -0.25], requires_grad=True)
        nll = calibration_prior_nll(code, mean, std, 1.0)
        self.assertAlmostEqual(float(nll.detach()), 0.5, places=6)
        nll.backward()
        self.assertEqual(float(code.grad[2]), 0.0)
        self.assertEqual(float(code.grad[3]), 0.0)
        with torch.no_grad():
            project_point_mass(code, mean, std)
        self.assertEqual(code[2:].tolist(), [0.0, 0.0])

    def test_the_fasting_fraction_is_half_and_landau_can_move_it(self) -> None:
        from pulse.knowledge.cohort_statistics import (
            ALL_COHORT_STATISTICS, SEALED_COHORT_STATISTICS,
        )
        names = {s.name for s in (*ALL_COHORT_STATISTICS, *SEALED_COHORT_STATISTICS)}
        for hours in (14, 22, 42):
            self.assertIn(f"landau_1996_gng_fraction_{hours}h", names)

        torch.manual_seed(0)
        model = _model()
        state = torch.tensor(NORM_CENTER, dtype=torch.float32).unsqueeze(0)
        emb = torch.zeros(1, EMBEDDING_DIM)
        traj = integrate(model, state, emb, n_steps=2, observe=True, meals=[])
        frac = traj[0, -1, SERIES_INDEX["gng_fraction"]]
        self.assertGreater(float(frac.detach()), 0.3)
        self.assertLess(float(frac.detach()), 0.7)
        frac.backward()
        grad = model.metabolic.logit_f_gng.grad
        self.assertIsNotNone(grad)
        self.assertNotEqual(float(grad.detach()), 0.0)
