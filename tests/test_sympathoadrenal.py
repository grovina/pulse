"""Sympathetic tone and epinephrine: the same law in both models, and the
exercise splanchnic rise that can move the hepatic gain."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from pulse.knowledge.cohort_statistics import ALL_COHORT_STATISTICS
from pulse.knowledge.full_body import PatientParams, glucose_fluxes, simulate_full_body
from pulse.model import ModularPhysiologyNetwork, integrate
from pulse.modules.metabolic import MetabolicModule, _EPI_COUPLING_IDX
from pulse.modules.base import compute_time_features
from pulse.sympathoadrenal import EPI_B, EXERCISE_UPTAKE_K, HEP_EPI_GAIN, LIP_EPI_GAIN
from pulse.types import EMBEDDING_DIM, MARKER_INDEX, NORM_CENTER, NORM_SCALE


def _teacher(activity: np.ndarray) -> np.ndarray:
    dur = len(activity)
    traj, _ = simulate_full_body(
        PatientParams(), [], np.ones(dur), activity, dur,
        start_hour=10.0, noise_scale=0.0,
    )
    return traj


class TestTeacherArm(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dur = 80
        cls.rest = _teacher(np.zeros(dur))
        cls.bout = _teacher(np.array([0.0 if t < 20 else 0.7 for t in range(dur)]))

    def _delta(self, marker: str) -> float:
        i = MARKER_INDEX[marker]
        return float(self.bout[55:70, i].mean() - self.rest[55:70, i].mean())

    def test_rest_is_the_basal_hormone(self) -> None:
        i_epi = MARKER_INDEX["epinephrine"]
        i_sns = MARKER_INDEX["sns_tone"]
        self.assertAlmostEqual(float(self.rest[55:70, i_epi].mean()), EPI_B, places=1)
        self.assertAlmostEqual(float(self.rest[55:70, i_sns].mean()), 0.2, places=2)

    def test_moderate_bout_raises_tone_epinephrine_hgo_and_ffa(self) -> None:
        self.assertAlmostEqual(self._delta("sns_tone"), 0.56, places=2)
        self.assertAlmostEqual(self._delta("epinephrine"), 280.0, places=0)
        self.assertAlmostEqual(self._delta("hepatic_output"), 3.49, delta=0.15)
        self.assertAlmostEqual(self._delta("glucose"), 9.3, delta=0.5)
        self.assertGreater(self._delta("ffa"), 0.15)

    def test_uptake_follows_glucose_and_work(self) -> None:
        p = PatientParams()
        pools = (p.LGly_b, p.MGly_b, p.Hep_b, 0.0)
        rest = glucose_fluxes(
            p, 70.0, p.Ib, 0.0, p.Gnb, p.Cort_b, p.FFA_b, *pools, 0.0,
        )
        low = glucose_fluxes(
            p, 70.0, p.Ib, 0.0, p.Gnb, p.Cort_b, p.FFA_b, *pools, 0.7,
        )
        self.assertEqual(rest["uptake_ex"], 0.0)
        # 70 mg/dL is under 0.8·basal. Working muscle still takes glucose up.
        self.assertAlmostEqual(low["uptake_ex"], EXERCISE_UPTAKE_K * 0.7 * 70.0)

    def test_settled_heart_rate_matches_the_old_activity_gain(self) -> None:
        # act_hr_gain * 0.7 / k_hr once tone has caught activity.
        self.assertAlmostEqual(self._delta("hr"), 58.3, delta=2.0)


class TestStudentArm(unittest.TestCase):
    def test_gains_start_at_the_teacher_constants(self) -> None:
        met = MetabolicModule(EMBEDDING_DIM)
        c = met.constants(torch.zeros(1, EMBEDDING_DIM))
        self.assertAlmostEqual(float(c["hep_epi"].detach()), HEP_EPI_GAIN, places=4)
        self.assertAlmostEqual(float(c["lip_epi"].detach()), LIP_EPI_GAIN, places=4)
        self.assertAlmostEqual(float(c["k_ex"].detach()), EXERCISE_UPTAKE_K, places=4)

    def test_epinephrine_above_basal_raises_glycogenolysis_and_lipolysis(self) -> None:
        met = MetabolicModule(EMBEDDING_DIM)
        state = torch.zeros(1, met.n_species)
        coupling = torch.zeros(1, met.n_coupling)
        external = torch.zeros(1, len(met.external_inputs))
        emb = torch.zeros(1, EMBEDDING_DIM)
        tf = compute_time_features(torch.tensor([10.0 * 60.0]))
        high = coupling.clone()
        high[0, _EPI_COUPLING_IDX] = (300.0 - EPI_B) / NORM_SCALE[MARKER_INDEX["epinephrine"]]
        with torch.no_grad():
            rest = met.fluxes(state, coupling, external, emb, tf)
            bout = met.fluxes(state, high, external, emb, tf)
        self.assertGreater(float(bout["glycogenolysis_plasma"]), float(rest["glycogenolysis_plasma"]) * 2.0)
        self.assertGreater(float(bout["lipolysis"]), float(rest["lipolysis"]))

    def test_the_hepatic_gain_receives_a_gradient_only_above_basal(self) -> None:
        met = MetabolicModule(EMBEDDING_DIM)
        state = torch.zeros(1, met.n_species)
        external = torch.zeros(1, len(met.external_inputs))
        emb = torch.zeros(1, EMBEDDING_DIM)
        tf = compute_time_features(torch.tensor([10.0 * 60.0]))
        rest_c = torch.zeros(1, met.n_coupling)
        rest = met.fluxes(state, rest_c, external, emb, tf)
        rest["glycogenolysis_plasma"].sum().backward()
        self.assertEqual(float(met._hep_epi_raw.grad), 0.0)
        met.zero_grad()
        high = rest_c.clone()
        high[0, _EPI_COUPLING_IDX] = (300.0 - EPI_B) / NORM_SCALE[MARKER_INDEX["epinephrine"]]
        bout = met.fluxes(state, high, external, emb, tf)
        bout["glycogenolysis_plasma"].sum().backward()
        self.assertGreater(float(met._hep_epi_raw.grad.abs()), 0.0)

    def test_a_bout_raises_epinephrine_in_the_network(self) -> None:
        model = ModularPhysiologyNetwork()
        n = 40
        activity = torch.zeros(n)
        activity[10:] = 0.7
        traj = integrate(
            model, torch.tensor(NORM_CENTER, dtype=torch.float32),
            torch.zeros(EMBEDDING_DIM), n, activity=activity,
            start_time_minutes=10.0 * 60.0,
        )
        self.assertEqual(tuple(traj.shape), (n, len(NORM_CENTER)))
        self.assertTrue(torch.isfinite(traj).all())
        epi = MARKER_INDEX["epinephrine"]
        self.assertGreater(float(traj[-1, epi].detach()), 200.0)


class TestWahrenSpec(unittest.TestCase):
    def test_registered(self) -> None:
        spec = next(s for s in ALL_COHORT_STATISTICS if s.name == "wahren_800_splanchnic_glucose")
        self.assertEqual(spec.marker_id, "hepatic_output")
        self.assertAlmostEqual(spec.target, 3.63)
        self.assertAlmostEqual(spec.sigma, 1.29)
        self.assertEqual(spec.n_arm, 1)

    def test_arterial_glucose_registered(self) -> None:
        spec = next(s for s in ALL_COHORT_STATISTICS if s.name == "wahren_800_arterial_glucose")
        self.assertEqual(spec.marker_id, "glucose")
        self.assertAlmostEqual(spec.target, 8.28)
        self.assertAlmostEqual(spec.sigma, 6.32)
        self.assertEqual(spec.n_arm, 1)
