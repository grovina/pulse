"""The four former pins are parameters, and the prior holds their mode."""

from __future__ import annotations

import unittest

import torch

from pulse.model import ModularPhysiologyNetwork
from pulse.modules.metabolic import (
    _F_GNG,
    _GN_CENTER,
    _K_FFA,
    _K_II,
    POPULATION_PRIOR_NAMES,
)
from pulse.training.population_prior_signal import population_prior_nll
from pulse.types import EMBEDDING_DIM


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
        for name in ("log_k_ii", "logit_f_gng", "log_k_ffa", "log_gn_b"):
            grad = getattr(model.metabolic, name).grad
            self.assertIsNotNone(grad, name)
            self.assertAlmostEqual(float(grad), 0.0, places=4, msg=name)

        model.zero_grad()
        with torch.no_grad():
            model.metabolic.log_k_ii.add_(0.05)
            model.metabolic.logit_f_gng.add_(0.2)
        population_prior_nll(model.metabolic).backward()
        self.assertGreater(float(model.metabolic.log_k_ii.grad), 0.0)
        self.assertGreater(float(model.metabolic.logit_f_gng.grad), 0.0)

    def test_there_is_no_per_person_copy(self) -> None:
        met = _model().metabolic
        names = [n for n, _ in met.named_parameters()]
        for forbidden in ("k_ii_net", "f_gng_net", "k_ffa_net", "gn_baseline_net"):
            self.assertFalse(any(forbidden in n for n in names))
        self.assertEqual(POPULATION_PRIOR_NAMES, ("k_ii", "f_gng", "k_ffa", "gn_b"))
