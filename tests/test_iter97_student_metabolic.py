"""Iter 97 — the metabolic module's structural properties (review 2026-09-04,
items 2.2, 2.6, 3.4, 3.10, 3.11 and the dead-parameter finding).

Every test here pins something that holds BY CONSTRUCTION — a random
perturbation of every parameter must not break it.
"""

from __future__ import annotations

import math
import unittest

import torch

from pulse.model import ModularPhysiologyNetwork, integrate, precompute_gut_outputs
from pulse.modules import metabolic as M
from pulse.modules.base import compute_time_features
from pulse.modules.gut import MealEvent
from pulse.types import (
    EMBEDDING_DIM, MARKER_INDEX as MI, MODULE_MARKER_INDICES, NORM_CENTER, NORM_SCALE,
)

_MET = MODULE_MARKER_INDICES["metabolic"]
_CENTER = torch.tensor(NORM_CENTER)
_SCALE = torch.tensor(NORM_SCALE)


def _model(seed: int = 0, hidden: int = 16, perturb: float = 0.3) -> ModularPhysiologyNetwork:
    torch.manual_seed(seed)
    m = ModularPhysiologyNetwork(
        metabolic_hidden=hidden, appetite_hidden=16, stress_hidden=16, cardiovascular_hidden=16,
        thermoreg_hidden=16, respiratory_hidden=16, gut_hidden=16, hepatobiliary_hidden=16)
    if perturb > 0:
        with torch.no_grad():
            for p in m.metabolic.parameters():
                p.add_(perturb * torch.randn_like(p))
    m.eval()
    return m


def _inputs(m, batch=32, seed=0, app=None, act=None):
    """Random normalized module inputs, plus the raw glucose the state implies."""
    g = torch.Generator().manual_seed(seed)
    state = 0.7 * torch.randn(batch, len(_MET), generator=g)
    state[:, M._INSULIN_ACTION_IDX] = torch.rand(batch, generator=g)
    coupling = torch.zeros(batch, M._N_COUPLING)
    coupling[:, 0] = torch.rand(batch, generator=g) * 2.0 if app is None else app
    coupling[:, 4] = 0.5 * torch.randn(batch, generator=g)
    coupling[:, 5] = 0.5 * torch.randn(batch, generator=g)
    external = torch.zeros(batch, 2)
    external[:, 0] = torch.rand(batch, generator=g) if act is None else act
    external[:, 1] = 1.0
    emb = m.embedding_projections["metabolic"](torch.randn(batch, EMBEDDING_DIM, generator=g))
    tf = compute_time_features(torch.rand(batch, generator=g) * 1440.0)
    return state, coupling, external, emb, tf


class TestCarbonBudget(unittest.TestCase):
    """2.2: d(G/c) + dLGly + dMGly = app_g − brk_M − (clearances + exercise − EGP_extra)/c."""

    def test_ledger_closes_pointwise_for_random_states(self) -> None:
        m = _model(0)
        met = m.metabolic
        for seed in range(3):
            state, coupling, external, emb, tf = _inputs(m, seed=seed)
            with torch.no_grad():
                f = met.fluxes(state, coupling, external, emb, tf)
                rates = met(state, coupling, external, emb, tf)
            c = f["c_plasma"]
            lhs = rates[:, M._GLUCOSE_IDX] / c + rates[:, M._LIVER_GLYCOGEN_IDX] + rates[:, M._MUSCLE_GLYCOGEN_IDX]
            uptake = f["clearance_sg"] + f["clearance_ins"] + f["exercise_uptake"] - f["egp_extra"]
            rhs = f["app_g"] - f["brk_muscle"] - uptake / c
            torch.testing.assert_close(lhs, rhs, atol=1e-6, rtol=1e-5)

    def test_ledger_closes_over_a_eucaloric_day(self) -> None:
        """Integrate 24 h with three meals and re-derive the ledger from the trajectory."""
        m = _model(1)
        met = m.metabolic
        meals = [MealEvent(120, 60, 20, 25), MealEvent(420, 80, 25, 30), MealEvent(780, 70, 25, 35)]
        n = 1440
        sw = torch.ones(n); act = torch.zeros(n)
        emb = torch.randn(EMBEDDING_DIM) * 0.5
        typ = _CENTER.clone()
        with torch.no_grad():
            tr = integrate(m, typ, emb, n, start_time_minutes=360, meals=meals, sleep_wake=sw, activity=act)
            gut = precompute_gut_outputs(m, emb, n, meals=meals)
            ns = (tr - _CENTER) / _SCALE
            coupling = torch.cat([gut, ns[:, MI["cortisol"]:MI["cortisol"] + 1], ns[:, MI["glp1"]:MI["glp1"] + 1]], -1)
            external = torch.stack([act, sw], -1)
            e_met = m.embedding_projections["metabolic"](emb).unsqueeze(0).expand(n, -1)
            tf = compute_time_features(torch.tensor([(360 + s) % 1440.0 for s in range(n)]))
            f = met.fluxes(ns[:, _MET], coupling, external, e_met, tf)
        c = float(f["c_plasma"][0])
        d_pool = (float(tr[-1, MI["glucose"]] - tr[0, MI["glucose"]]) / c
                  + float(tr[-1, MI["liver_glycogen"]] - tr[0, MI["liver_glycogen"]])
                  + float(tr[-1, MI["muscle_glycogen"]] - tr[0, MI["muscle_glycogen"]]))
        # Trajectory holds states BEFORE each step, so sum the rates over steps 0..n-2.
        sl = slice(0, n - 1)
        uptake = (f["clearance_sg"] + f["clearance_ins"] + f["exercise_uptake"] - f["egp_extra"])[sl].sum() / c
        rhs = float(f["app_g"][sl].sum() - f["brk_muscle"][sl].sum() - uptake)
        total_in = float(f["app_g"][sl].sum())
        self.assertGreater(total_in, 100.0)   # the day's carbohydrate actually arrived
        self.assertAlmostEqual(d_pool, rhs, delta=1e-3 * total_in)
        # and storage is a FRACTION of what arrived
        stored = float((f["syn_liver"] + f["syn_muscle"])[sl].sum())
        self.assertLess(stored, total_in)
        self.assertGreater(stored, 0.0)

    def test_synthesis_is_zero_without_appearance_and_bounded_by_it(self) -> None:
        m = _model(2)
        met = m.metabolic
        state, coupling, external, emb, tf = _inputs(m, app=0.0)
        with torch.no_grad():
            f = met.fluxes(state, coupling, external, emb, tf)
        self.assertEqual(float(f["syn_liver"].abs().sum()), 0.0)
        self.assertEqual(float(f["syn_muscle"].abs().sum()), 0.0)
        state, coupling, external, emb, tf = _inputs(m, seed=5)
        with torch.no_grad():
            f = met.fluxes(state, coupling, external, emb, tf)
        self.assertTrue(bool((f["syn_liver"] + f["syn_muscle"] <= f["app_g"] + 1e-7).all()))
        self.assertTrue(bool((f["f_plasma"] > 0).all()))


class TestGlycogenGates(unittest.TestCase):
    """2.6: muscle breakdown is zero at rest and liver breakdown is fully insulin-gated."""

    def test_muscle_glycogen_is_not_spent_at_rest(self) -> None:
        m = _model(3, perturb=1.0)
        for act in (0.0, 0.05, M._MUSCLE_ACT_REST):
            state, coupling, external, emb, tf = _inputs(m, act=act)
            with torch.no_grad():
                f = m.metabolic.fluxes(state, coupling, external, emb, tf)
            self.assertEqual(float(f["brk_muscle"].abs().sum()), 0.0, msg=f"activity {act}")
        state, coupling, external, emb, tf = _inputs(m, act=0.6)
        with torch.no_grad():
            f = m.metabolic.fluxes(state, coupling, external, emb, tf)
        self.assertTrue(bool((f["brk_muscle"] > 0).all()))

    def test_liver_breakdown_has_no_ungated_channel(self) -> None:
        """brk_liver == FLUX · drive · avail · 1/(1 + relu(I − Ib)/K): the whole flux is
        behind the insulin gate. At I = Ib the gate is exactly 1; at I = Ib + 10K it is
        exactly 1/11 — for every random state, embedding and head output."""
        m = _model(4, perturb=1.0)
        met = m.metabolic
        state, coupling, external, emb, tf = _inputs(m)
        k = float(torch.nn.functional.softplus(met.log_glyc_ins_supp))

        def _at(excess: float):
            s = state.clone()
            with torch.no_grad():
                ib = met.insulin_setpoint_raw(emb)
                s[:, M._INSULIN_IDX] = (ib + excess - 10.0) / 10.0
                f = met.fluxes(s, coupling, external, emb, tf)
                lgly = 100.0 + 60.0 * s[:, M._LIVER_GLYCOGEN_IDX].clamp(min=-100.0 / 60.0)
                avail = lgly / (lgly + M._LIVER_GLY_K)
                ungated = M._LIVER_GLY_FLUX * f["cons_raw"][:, M._LIVER_GLYCOGEN_IDX] * avail
            return f["brk_liver"], ungated

        brk, ungated = _at(0.0)
        torch.testing.assert_close(brk, ungated, atol=1e-6, rtol=1e-5)
        brk, ungated = _at(10.0 * k)
        torch.testing.assert_close(brk, ungated / 11.0, atol=1e-6, rtol=1e-4)

    def test_no_learned_catabolic_threshold_exists(self) -> None:
        m = _model(5)
        for idx in (M._LIVER_GLYCOGEN_IDX, M._MUSCLE_GLYCOGEN_IDX):
            names = [n for n, _ in m.metabolic.heads[idx].named_parameters()]
            self.assertFalse(any("thresh" in n or "temp" in n for n in names), names)

    def test_resting_36h_fast_preserves_muscle_glycogen(self) -> None:
        """The reviewer's number: 400 -> 184 g at rest. Now unchanged by construction."""
        m = _model(6, perturb=1.0)
        n = 2160
        with torch.no_grad():
            tr = integrate(m, _CENTER.clone(), torch.zeros(EMBEDDING_DIM), n, start_time_minutes=360,
                           meals=[], sleep_wake=torch.ones(n), activity=torch.zeros(n))
        mg = tr[:, MI["muscle_glycogen"]]
        self.assertAlmostEqual(float(mg.min()), 400.0, places=3)


class TestPerPatientGates(unittest.TestCase):
    """3.4: gates fire on (G − Gb)/30 and (I − Ib)/10; the fasting drop is absolute."""

    @staticmethod
    def _set_gb(m, gb: float) -> None:
        z = (gb - 95.0) / 30.0 / M._GLUCOSE_BASELINE_MAX_Z
        with torch.no_grad():
            m.metabolic.glucose_baseline_net[-1].weight.zero_()
            m.metabolic.glucose_baseline_net[-1].bias.fill_(math.atanh(z))

    def test_gate_stimuli_are_deviations_from_the_patients_own_setpoints(self) -> None:
        m = _model(7)
        for gb in (75.0, 120.0):
            self._set_gb(m, gb)
            state, coupling, external, emb, tf = _inputs(m)
            with torch.no_grad():
                f = m.metabolic.fluxes(state, coupling, external, emb, tf)
            # raw_state floors a concentration at 0 (a negative µU/mL is not a state)
            g_raw = (95.0 + 30.0 * state[:, M._GLUCOSE_IDX]).clamp(min=0.0)
            i_raw = (10.0 + 10.0 * state[:, M._INSULIN_IDX]).clamp(min=0.0)
            torch.testing.assert_close(f["glucose_dev"], (g_raw - gb) / 30.0, atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(f["insulin_dev"], (i_raw - f["ib"]) / 10.0, atol=1e-5, rtol=1e-5)
            self.assertTrue(bool((f["ib"] > 0).all()))

    def test_insulin_action_lags_excess_over_the_patients_ib(self) -> None:
        m = _model(8)
        met = m.metabolic
        with torch.no_grad():
            met.insulin_baseline_net[-1].weight.zero_()
            met.insulin_baseline_net[-1].bias.fill_(1.0)  # Ib = 10·exp(0.9·tanh 1) ≈ 19.8
        state, coupling, external, emb, tf = _inputs(m)
        state[:, M._INSULIN_IDX] = 0.5           # insulin 15 µU/mL: above the population 10, below Ib
        state[:, M._INSULIN_ACTION_IDX] = 0.0
        with torch.no_grad():
            f = met.fluxes(state, coupling, external, emb, tf)
        self.assertTrue(bool((f["ib"] > 15.0).all()))
        self.assertEqual(float(f["xa_rate"].abs().sum()), 0.0)  # relu((15 − Ib)/10) = 0

    def test_fasting_drop_is_absolute_and_floored(self) -> None:
        m = _model(9, perturb=0.0)
        met = m.metabolic
        drop = float(torch.nn.functional.softplus(met.log_gb_drop_abs))
        for gb in (75.0, 95.0, 120.0):
            self._set_gb(m, gb)
            state, coupling, external, emb, tf = _inputs(m, batch=1)
            state[:, M._LIVER_GLYCOGEN_IDX] = -100.0 / 60.0   # liver glycogen 0 g: fully depleted
            with torch.no_grad():
                f = met.fluxes(state, coupling, external, emb, tf)
            self.assertAlmostEqual(float(f["gb_fasted"]), max(gb - drop, M._GB_FLOOR_ABS), places=3)

    def test_gb_75_and_gb_120_patients_both_fast_to_physiological_levels(self) -> None:
        """probe2-C pattern: a 16 h fast from typical at rest. Through iter 96 the Gb=75
        patient reached 54 mg/dL and the Gb=120 patient fasted with insulin 23.75 and a
        standing insulin action of 1.4."""
        for gb in (75.0, 120.0):
            m = _model(10, hidden=32, perturb=0.0)
            self._set_gb(m, gb)
            n = 960
            with torch.no_grad():
                tr = integrate(m, _CENTER.clone(), torch.zeros(EMBEDDING_DIM), n, start_time_minutes=480,
                               meals=[], sleep_wake=torch.ones(n), activity=torch.zeros(n))
            g, ins, xa = float(tr[-1, MI["glucose"]]), float(tr[-1, MI["insulin"]]), float(tr[-1, MI["insulin_action"]])
            self.assertGreaterEqual(g, M._GB_FLOOR_ABS - 1.0, msg=f"Gb={gb}: glucose {g}")
            self.assertLessEqual(g, gb + 12.0, msg=f"Gb={gb}: glucose {g}")
            self.assertGreater(ins, 1.0, msg=f"Gb={gb}: insulin {ins}")
            self.assertLess(ins, 25.0, msg=f"Gb={gb}: insulin {ins}")
            # The standing insulin action is exactly the lagged excess over THIS patient's
            # Ib (10 at the zero-init head) — not over a population 10 that the Gb=120
            # patient's insulin would sit permanently above once trained.
            ib = float(m.metabolic.insulin_setpoint_raw(
                m.embedding_projections["metabolic"](torch.zeros(EMBEDDING_DIM))))
            self.assertAlmostEqual(xa, max(ins - ib, 0.0) / 10.0, delta=0.02, msg=f"Gb={gb}")


class TestBergmanClearance(unittest.TestCase):
    """3.10: insulin action is a sink, never a source."""

    def test_insulin_action_lowers_the_glucose_rate_below_the_setpoint(self) -> None:
        m = _model(11, perturb=1.0)
        met = m.metabolic
        state, coupling, external, emb, tf = _inputs(m)
        state[:, M._GLUCOSE_IDX] = (85.0 - 95.0) / 30.0     # G = 85 < Gb ≈ 95 (the review's case)
        s0 = state.clone(); s0[:, M._INSULIN_ACTION_IDX] = 0.0
        s1 = state.clone(); s1[:, M._INSULIN_ACTION_IDX] = 2.0
        with torch.no_grad():
            r0 = met(s0, coupling, external, emb, tf)[:, M._GLUCOSE_IDX]
            r1 = met(s1, coupling, external, emb, tf)[:, M._GLUCOSE_IDX]
            f1 = met.fluxes(s1, coupling, external, emb, tf)
        self.assertTrue(bool((r1 < r0).all()))
        self.assertTrue(bool((f1["clearance_ins"] >= 0).all()))


class TestMitochondrialRole(unittest.TestCase):
    """3.11: mito scales oxidative clearance of FFA / lactate / BHB and nothing else."""

    def test_other_heads_do_not_see_mito(self) -> None:
        m = _model(12, perturb=1.0)
        met = m.metabolic
        state, coupling, external, emb, tf = _inputs(m)
        s1 = state.clone(); s1[:, M._MITO_IDX] += 1.5
        with torch.no_grad():
            p0, c0 = met.species_fluxes(state, coupling, external, emb, tf)
            p1, c1 = met.species_fluxes(s1, coupling, external, emb, tf)
        for i in range(M._N_SPECIES):
            if i == M._MITO_IDX:
                continue
            torch.testing.assert_close(p0[:, i], p1[:, i], atol=0, rtol=0)
            torch.testing.assert_close(c0[:, i], c1[:, i], atol=0, rtol=0)
        self.assertFalse(torch.equal(p0[:, M._MITO_IDX], p1[:, M._MITO_IDX]))

    def test_mito_scales_oxidative_clearance(self) -> None:
        m = _model(13, perturb=0.5)
        met = m.metabolic
        state, coupling, external, emb, tf = _inputs(m)
        state[:, M._FFA_IDX] = 1.0; state[:, M._BHB_IDX] = 1.0; state[:, M._LACTATE_IDX] = 1.0
        s1 = state.clone(); s1[:, M._MITO_IDX] += 1.0   # mito 1.0 -> 1.3
        with torch.no_grad():
            r0 = met(state, coupling, external, emb, tf)
            r1 = met(s1, coupling, external, emb, tf)
        for idx in (M._FFA_IDX, M._BHB_IDX, M._LACTATE_IDX):
            self.assertTrue(bool((r1[:, idx] < r0[:, idx]).all()), msg=f"species {idx}")
        for idx in (M._GLUCOSE_IDX, M._INSULIN_IDX, M._GLUCAGON_IDX, M._HEPATIC_IDX):
            torch.testing.assert_close(r0[:, idx], r1[:, idx], atol=1e-6, rtol=1e-6)


class TestKetogenesisAndHepaticOutput(unittest.TestCase):
    def test_ketogenesis_needs_ffa_above_basal_and_is_insulin_suppressed(self) -> None:
        m = _model(14, perturb=0.5)
        met = m.metabolic
        state, coupling, external, emb, tf = _inputs(m)
        low = state.clone(); low[:, M._FFA_IDX] = -0.5      # FFA 0.4 ≤ basal 0.5
        with torch.no_grad():
            f = met.fluxes(low, coupling, external, emb, tf)
        self.assertEqual(float(f["ketogenesis"].abs().sum()), 0.0)
        hi = state.clone(); hi[:, M._FFA_IDX] = (1.2 - 0.5) / 0.2
        hi[:, M._INSULIN_IDX] = (4.0 - 10.0) / 10.0
        hi_ins = hi.clone(); hi_ins[:, M._INSULIN_IDX] = (60.0 - 10.0) / 10.0
        with torch.no_grad():
            f_fast = met.fluxes(hi, coupling, external, emb, tf)
            f_fed = met.fluxes(hi_ins, coupling, external, emb, tf)
        self.assertTrue(bool((f_fast["ketogenesis"] > 0).all()))
        self.assertTrue(bool((f_fed["ketogenesis"] < f_fast["ketogenesis"]).all()))

    def test_hepatic_output_target_reads_liver_glycogenolysis(self) -> None:
        m = _model(15, perturb=0.5)
        met = m.metabolic
        state, coupling, external, emb, tf = _inputs(m)
        with torch.no_grad():
            f = met.fluxes(state, coupling, external, emb, tf)
        gng = M._TYPICALS[M._HEPATIC_IDX] * f["prod_raw"][:, M._HEPATIC_IDX]
        torch.testing.assert_close(f["hep_target"], M._HEP_UNITS_PER_G_MIN * f["brk_liver"] + gng, atol=1e-5, rtol=1e-5)


class TestNoDeadHeads(unittest.TestCase):
    def test_structural_species_own_no_parameters(self) -> None:
        m = _model(16)
        for idx in (M._GLUCOSE_IDX, M._INSULIN_ACTION_IDX):
            self.assertEqual(sum(p.numel() for p in m.metabolic.heads[idx].parameters()), 0)

    def test_every_metabolic_parameter_receives_gradient(self) -> None:
        m = _model(17, perturb=0.3)
        m.train()
        met = m.metabolic
        state, coupling, external, emb, tf = _inputs(m)
        emb = emb.detach().requires_grad_(True)
        rates = met(state, coupling, external, emb, tf)
        rates.abs().sum().backward()
        dead = [n for n, p in met.named_parameters() if p.grad is None or float(p.grad.abs().max()) == 0.0]
        self.assertEqual(dead, [])


if __name__ == "__main__":
    unittest.main()
