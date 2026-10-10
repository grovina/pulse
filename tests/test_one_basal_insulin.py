"""One basal insulin for the whole body.

Appetite used to decode its own Ib from its own projection of the person
code. The metabolic module's decode is the one that sets insulin's fixed
point, and it is the one appetite reads.
"""

from __future__ import annotations

import math
import unittest

import torch

from pulse.knowledge.full_body import PatientParams
from pulse.model import ModularPhysiologyNetwork, integrate
from pulse.types import EMBEDDING_DIM, MARKER_INDEX, NORM_CENTER, STATE_DIM


def _model() -> ModularPhysiologyNetwork:
    torch.manual_seed(0)
    return ModularPhysiologyNetwork(
        metabolic_hidden=16, appetite_hidden=8, stress_hidden=8,
        cardiovascular_hidden=8, thermoreg_hidden=8, respiratory_hidden=8,
        gut_hidden=8, hepatobiliary_hidden=8,
    )


class TestOneBasalInsulin(unittest.TestCase):
    def test_the_appetite_decode_is_gone(self) -> None:
        model = _model()
        self.assertFalse(hasattr(model.appetite, "insulin_baseline_net"))
        owned = [n for n, _ in model.named_parameters() if "insulin_baseline" in n or "body_mass_net" in n]
        self.assertTrue(owned)
        self.assertTrue(all(n.startswith("metabolic.") for n in owned))

    def test_a_fitted_patient_is_the_ib_both_modules_read(self) -> None:
        model = _model()
        patient = PatientParams(Ib=18.0)
        bias = math.atanh(math.log(patient.Ib / 10.0) / 0.9)
        with torch.no_grad():
            last = model.metabolic.insulin_baseline_net[-1]
            last.weight.zero_()
            last.bias.fill_(bias)
        emb = torch.randn(4, EMBEDDING_DIM)
        person = model.person_quantities(emb)
        met = model.embedding_projections["metabolic"](emb)
        met_ib = model.metabolic.insulin_setpoint_raw(met).clamp(min=1e-3)
        torch.testing.assert_close(person["ib"], met_ib)
        torch.testing.assert_close(person["ib"], torch.full_like(person["ib"], patient.Ib))
        const = model.metabolic.constants(met)
        torch.testing.assert_close(const["ib"], person["ib"])
        torch.testing.assert_close(person["body_mass_kg"], model.metabolic.body_mass_kg(met))

        app = model.embedding_projections["appetite"](emb)
        bound = model.appetite.bind_person(model.appetite.constants(app), person)
        torch.testing.assert_close(bound["ib"], person["ib"])

    def test_ghrelin_feels_the_metabolic_basal(self) -> None:
        model = _model()
        state = torch.tensor(NORM_CENTER)
        emb = torch.zeros(EMBEDDING_DIM)
        with torch.no_grad():
            low = model(state, emb, torch.tensor(480.0), [])
            model.metabolic.insulin_baseline_net[-1].bias.fill_(1.0)
        high = model(state, emb, torch.tensor(480.0), [])
        self.assertNotAlmostEqual(
            float(low[MARKER_INDEX["ghrelin"]]),
            float(high[MARKER_INDEX["ghrelin"]].detach()),
            places=4,
        )
        model.zero_grad()
        high[MARKER_INDEX["ghrelin"]].backward()
        grad = model.metabolic.insulin_baseline_net[-1].bias.grad
        self.assertIsNotNone(grad)
        self.assertGreater(float(grad.abs()), 0.0)

    def test_a_rollout_binds_the_shared_basal(self) -> None:
        model = _model()
        traj = integrate(
            model, torch.tensor(NORM_CENTER), torch.zeros(EMBEDDING_DIM), 4, meals=[],
        )
        self.assertEqual(tuple(traj.shape), (4, STATE_DIM))
        self.assertTrue(torch.isfinite(traj).all())

    def test_appetite_without_the_shared_basal_fails_loudly(self) -> None:
        model = _model()
        with self.assertRaises(RuntimeError):
            model.appetite.bind_person(model.appetite.constants(torch.zeros(1, 16)), None)
