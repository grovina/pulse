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
from pulse.knowledge.full_body import (
    BODY_MASS_KG, PatientParams, resolve_derived_params, simulate_full_body,
)
from pulse.types import MARKER_INDEX as MI


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
