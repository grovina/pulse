"""M2.4. Pulmonary gas exchange is the Wahren table, frequency follows that
CO₂, and fat mass does not book the grams that went into glycogen.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from pulse.expenditure import (
    ACTIVITY_800, ACT_RR_GAIN, KCAL_PER_G_GLYCOGEN, KCAL_PER_KG_FAT, RR_ACT_K,
    VO2_800_ML, VO2_800_SE_ML, VO2_REST_ML, frequency_offset, frequency_offset_t,
    gas_exchange, gas_exchange_t, work_rq,
)
from pulse.bile import fat_bile_factor
from pulse.knowledge.full_body import (
    BODY_MASS_KG, PatientParams, resolve_derived_params, simulate_full_body,
)
from pulse.model import ModularPhysiologyNetwork, integrate
from pulse.modules.base import PhysiologyModule
from pulse.types import EMBEDDING_DIM, MARKER_INDEX as MI, MG_DL_PER_G, NORM_CENTER, STATE_DIM


def _hill(act: float) -> float:
    return ACT_RR_GAIN * act * act / (act * act + RR_ACT_K ** 2)


class TestTheTable(unittest.TestCase):
    def test_rest_and_the_800_load_are_the_published_means(self) -> None:
        rest = gas_exchange(BODY_MASS_KG, 0.0, 0.0, 0.0, 0.0)
        bout = gas_exchange(BODY_MASS_KG, ACTIVITY_800, 0.0, 0.0, 0.0)
        self.assertAlmostEqual(rest.vo2_ml_per_min, VO2_REST_ML, places=6)
        self.assertAlmostEqual(rest.rq, 0.77, places=6)
        self.assertAlmostEqual(bout.vo2_ml_per_min, VO2_800_ML, places=6)
        self.assertAlmostEqual(bout.rq, 0.82, places=6)
        self.assertLess(abs(bout.vo2_ml_per_min - VO2_800_ML), VO2_800_SE_ML)
        self.assertAlmostEqual(work_rq(ACTIVITY_800), 0.82, places=12)

    def test_a_meal_raises_expenditure(self) -> None:
        rest = gas_exchange(BODY_MASS_KG, 0.0, 0.0, 0.0, 0.0)
        fed = gas_exchange(BODY_MASS_KG, 0.0, 0.4, 0.1, 0.2)
        self.assertGreater(fed.kcal_per_min, rest.kcal_per_min)
        self.assertGreater(fed.vco2_ml_per_min, rest.vco2_ml_per_min)

    def test_the_hill_at_the_load_is_the_old_activity_hill(self) -> None:
        gas = gas_exchange(BODY_MASS_KG, ACTIVITY_800, 0.0, 0.0, 0.0)
        self.assertAlmostEqual(
            frequency_offset(gas.vco2_ml_per_min, BODY_MASS_KG),
            _hill(ACTIVITY_800), places=6)
        rest = gas_exchange(BODY_MASS_KG, 0.0, 0.0, 0.0, 0.0)
        self.assertAlmostEqual(frequency_offset(rest.vco2_ml_per_min, BODY_MASS_KG), 0.0, places=8)

    def test_tensor_matches_scalar(self) -> None:
        mass = torch.tensor([70.0, 90.0])
        act = torch.tensor([0.0, 0.7])
        carb = torch.tensor([0.0, 0.3])
        fat = torch.tensor([0.0, 0.1])
        prot = torch.tensor([0.0, 0.2])
        kcal, vo2, vco2, rq = gas_exchange_t(mass, act, carb, fat, prot)
        rise = frequency_offset_t(vco2, mass)
        for i in range(2):
            gas = gas_exchange(float(mass[i]), float(act[i]), float(carb[i]), float(fat[i]), float(prot[i]))
            self.assertAlmostEqual(float(kcal[i]), gas.kcal_per_min, places=4)
            self.assertAlmostEqual(float(vo2[i]), gas.vo2_ml_per_min, places=3)
            self.assertAlmostEqual(float(vco2[i]), gas.vco2_ml_per_min, places=3)
            self.assertAlmostEqual(float(rq[i]), gas.rq, places=5)
            self.assertAlmostEqual(
                float(rise[i]), frequency_offset(gas.vco2_ml_per_min, float(mass[i])), places=4)


class TestTheLedgers(unittest.TestCase):
    def test_fat_changes_by_the_energy_glycogen_did_not_take(self) -> None:
        """Over a fed day the adipose step is the energy residual after the
        glycogen step. The two pools name the same grams."""
        p = resolve_derived_params(PatientParams())
        n = 1440
        meals = [(120.0, 50.0, 12.0, 20.0), (420.0, 65.0, 22.0, 28.0), (780.0, 75.0, 28.0, 35.0)]
        act = np.zeros(n, dtype=np.float32)
        act[600:660] = 0.7
        traj, absp = simulate_full_body(
            p, meals, np.ones(n, dtype=np.float32), act, n,
            start_hour=6.0, noise_scale=0.0, rng=np.random.default_rng(0))
        fat = np.concatenate([[p.Fat_b], traj[:, MI["fat_mass"]]])
        liver = np.concatenate([[p.LGly_b], traj[:, MI["liver_glycogen"]]])
        muscle = np.concatenate([[p.MGly_b], traj[:, MI["muscle_glycogen"]]])
        residual = 0.0
        for t in range(n):
            gas = gas_exchange(
                p.body_mass_kg, float(act[t]),
                float(absp[t, 0]) / p.mg_dl_per_g(),
                float(absp[t, 1]) / 3.0, float(absp[t, 2]) / 3.0)
            kcal_in = 4.0 * float(absp[t, 0]) / p.mg_dl_per_g() + 9.0 * float(absp[t, 1]) / 3.0 + 4.0 * float(absp[t, 2]) / 3.0
            d_gly = (liver[t + 1] - liver[t]) + (muscle[t + 1] - muscle[t])
            expected = (kcal_in - gas.kcal_per_min - KCAL_PER_G_GLYCOGEN * d_gly) / KCAL_PER_KG_FAT
            residual = max(residual, abs(float(fat[t + 1] - fat[t]) - expected))
        self.assertLess(residual, 1e-6, f"fat residual {residual:.3e} kg/min")

    def test_resting_rate_is_the_setpoint(self) -> None:
        p = resolve_derived_params(PatientParams())
        n = 8 * 60
        traj, _ = simulate_full_body(
            p, [], np.ones(n, dtype=np.float32), np.zeros(n, dtype=np.float32), n,
            start_hour=8.0, noise_scale=0.0, rng=np.random.default_rng(0))
        self.assertAlmostEqual(float(traj[-1, MI["rr"]]), p.RR0, places=3)

    def test_the_bout_settles_on_the_hill(self) -> None:
        """Eight hours at activity 0.7. The equilibrium is RR₀ plus the
        Hill at that load, and lactate is not in it."""
        p = resolve_derived_params(PatientParams())
        n = 8 * 60
        act = np.full(n, ACTIVITY_800, dtype=np.float32)
        traj, _ = simulate_full_body(
            p, [], np.ones(n, dtype=np.float32), act, n,
            start_hour=10.0, noise_scale=0.0, rng=np.random.default_rng(0))
        gas = gas_exchange(p.body_mass_kg, ACTIVITY_800, 0.0, 0.0, 0.0)
        target = p.RR0 + frequency_offset(gas.vco2_ml_per_min, p.body_mass_kg)
        self.assertAlmostEqual(float(traj[-1, MI["rr"]]), target, places=2)
        self.assertAlmostEqual(target, p.RR0 + _hill(ACTIVITY_800), places=4)


class TestTheLedgerPublishesTheRise(unittest.TestCase):
    def _model(self, mass_bias: float = 0.0) -> ModularPhysiologyNetwork:
        torch.manual_seed(0)
        model = ModularPhysiologyNetwork()
        if mass_bias:
            with torch.no_grad():
                model.metabolic.body_mass_net[-1].bias.fill_(mass_bias)
        return model

    def _rr(self, model: ModularPhysiologyNetwork, *, glucose: float = 0.0,
            lipid: float = 0.0, activity: float = 0.0, bile: float | None = None) -> torch.Tensor:
        state = torch.tensor(NORM_CENTER, dtype=torch.float32).unsqueeze(0).clone()
        if bile is not None:
            state[0, MI["intestinal_bile"]] = bile
        gut = torch.zeros(1, 4)
        gut[0, 0] = glucose
        gut[0, 1] = lipid
        with torch.no_grad():
            rates = model(
                state, torch.zeros(1, EMBEDDING_DIM), torch.tensor([8.0 * 60.0]), [],
                activity=torch.tensor([activity]), sleep_wake=torch.ones(1),
                gut_override=gut,
            )
        return rates[0, MI["rr"]]

    def test_only_appetite_binds_a_person_quantity(self) -> None:
        model = self._model()
        overrides = [
            name for name, mod in model._modules_by_name.items()
            if type(mod).bind_person is not PhysiologyModule.bind_person
        ]
        self.assertEqual(overrides, ["appetite"])
        const = model.respiratory.constants(torch.zeros(1, model.respiratory.embedding_dim))
        self.assertNotIn("body_mass_kg", const)

    def test_the_lung_does_not_step_without_the_rise(self) -> None:
        rsp = self._model().respiratory
        with self.assertRaises(RuntimeError):
            rsp(
                torch.zeros(1, rsp.n_state), torch.zeros(1, rsp.n_coupling),
                torch.tensor([[0.0, 1.0]]), torch.zeros(1, rsp.embedding_dim),
                torch.zeros(1, 4),
            )

    def test_without_a_meal_mass_cancels_and_with_one_it_does_not(self) -> None:
        ref = self._model()
        heavy_m = self._model(3.0)
        heavy = float(heavy_m.person_quantities(torch.zeros(1, EMBEDDING_DIM))["body_mass_kg"].detach())
        self.assertGreater(heavy, 100.0)
        k = float(ref.respiratory.constants(torch.zeros(1, ref.respiratory.embedding_dim))["k"][0].detach())

        bout_ref = self._rr(ref, activity=ACTIVITY_800)
        bout_heavy = self._rr(heavy_m, activity=ACTIVITY_800)
        self.assertAlmostEqual(float(bout_ref), k * _hill(ACTIVITY_800), places=4)
        self.assertAlmostEqual(float(bout_ref), float(bout_heavy), places=4)

        appearance = 5.0
        meal_ref = self._rr(ref, glucose=appearance)
        meal_heavy = self._rr(heavy_m, glucose=appearance)
        carb_g = appearance / MG_DL_PER_G
        rise_ref = frequency_offset(
            gas_exchange(BODY_MASS_KG, 0.0, carb_g, 0.0, 0.0).vco2_ml_per_min, BODY_MASS_KG)
        rise_heavy = frequency_offset(
            gas_exchange(heavy, 0.0, carb_g, 0.0, 0.0).vco2_ml_per_min, heavy)
        self.assertLess(rise_heavy, rise_ref)
        self.assertAlmostEqual(float(meal_ref), k * rise_ref, places=4)
        self.assertAlmostEqual(
            float(meal_heavy - meal_ref), k * (rise_heavy - rise_ref), places=4)

    def test_an_empty_intestine_lowers_the_rise_in_the_same_minute(self) -> None:
        model = self._model()
        lipid = 3.0
        fat_g = lipid / 3.0
        full = self._rr(model, lipid=lipid, bile=1.0)
        empty = self._rr(model, lipid=lipid, bile=0.0)
        rise_full = frequency_offset(
            gas_exchange(BODY_MASS_KG, 0.0, 0.0, fat_g * fat_bile_factor(1.0), 0.0).vco2_ml_per_min,
            BODY_MASS_KG)
        rise_empty = frequency_offset(
            gas_exchange(BODY_MASS_KG, 0.0, 0.0, fat_g * fat_bile_factor(0.0), 0.0).vco2_ml_per_min,
            BODY_MASS_KG)
        k = float(model.respiratory.constants(torch.zeros(1, model.respiratory.embedding_dim))["k"][0].detach())
        self.assertLess(rise_empty, rise_full)
        self.assertAlmostEqual(float(full), k * rise_full, places=4)
        self.assertAlmostEqual(float(empty), k * rise_empty, places=4)

        state = torch.tensor(NORM_CENTER, dtype=torch.float32).clone()
        state[MI["intestinal_bile"]] = 0.0
        gut = torch.zeros(2, 4)
        gut[0, 1] = lipid
        emb = torch.zeros(EMBEDDING_DIM)
        kw = dict(
            gut_outputs=gut, activity=torch.zeros(2), sleep_wake=torch.ones(2), meals=[],
        )
        with torch.no_grad():
            planned = integrate(model, state, emb, 2, planned=True, **kw)
            pointwise = integrate(model, state, emb, 2, planned=False, **kw)
        torch.testing.assert_close(planned, pointwise)
        self.assertEqual(tuple(planned.shape), (2, STATE_DIM))
        self.assertAlmostEqual(float(planned[1, MI["rr"]]), 15.0 + k * rise_empty, places=3)
