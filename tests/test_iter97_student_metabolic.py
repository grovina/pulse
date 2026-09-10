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
    BODY_MASS_KG, EMBEDDING_DIM, MARKER_INDEX as MI, MG_DL_PER_G, MODULE_MARKER_INDICES,
    NORM_CENTER, NORM_SCALE, VG_DL,
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
    """2.2: d(G/mg) + dLGly + dMGly = app_g − brk_M − (uptake_ii + uptake_id + exercise − gng − syn_id)/mg,
    with mg the patient's 1000/(mass·1.85)."""

    def test_ledger_closes_pointwise_for_random_states(self) -> None:
        m = _model(0)
        met = m.metabolic
        for seed in range(3):
            state, coupling, external, emb, tf = _inputs(m, seed=seed)
            with torch.no_grad():
                f = met.fluxes(state, coupling, external, emb, tf)
                rates = met(state, coupling, external, emb, tf)
            c = f["mg_dl_per_g"]
            lhs = rates[:, M._GLUCOSE_IDX] / c + rates[:, M._LIVER_GLYCOGEN_IDX] + rates[:, M._MUSCLE_GLYCOGEN_IDX]
            uptake = f["uptake_ii"] + f["uptake_id"] + f["exercise_uptake"] - f["gng_plasma"] - f["syn_muscle_id"] * c
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
        sl = slice(0, n - 1)
        mg = f["mg_dl_per_g"][sl]
        d_pool = float((
            (tr[1:, MI["glucose"]] - tr[:-1, MI["glucose"]]) / mg
            + (tr[1:, MI["liver_glycogen"]] - tr[:-1, MI["liver_glycogen"]])
            + (tr[1:, MI["muscle_glycogen"]] - tr[:-1, MI["muscle_glycogen"]])
        ).sum())
        uptake = (
            (f["uptake_ii"] + f["uptake_id"] + f["exercise_uptake"] - f["gng_plasma"])[sl]
            - f["syn_muscle_id"][sl] * mg
        )
        rhs = float((f["app_g"][sl] - f["brk_muscle"][sl] - uptake / mg).sum())
        total_in = float(f["app_g"][sl].sum())
        self.assertGreater(total_in, 50.0)
        self.assertAlmostEqual(d_pool, rhs, delta=1e-3 * total_in)
        stored = float((f["syn_liver"] + f["syn_muscle_oral"])[sl].sum())
        self.assertLess(stored, total_in)
        self.assertGreater(stored, 0.0)

    def test_synthesis_is_zero_without_appearance_and_bounded_by_it(self) -> None:
        m = _model(2)
        met = m.metabolic
        state, coupling, external, emb, tf = _inputs(m, app=0.0)
        with torch.no_grad():
            f = met.fluxes(state, coupling, external, emb, tf)
        self.assertEqual(float(f["syn_liver"].abs().sum()), 0.0)
        self.assertEqual(float(f["syn_muscle_oral"].abs().sum()), 0.0)
        state, coupling, external, emb, tf = _inputs(m, seed=5)
        with torch.no_grad():
            f = met.fluxes(state, coupling, external, emb, tf)
        self.assertTrue(bool((f["syn_liver"] + f["syn_muscle_oral"] <= f["app_g"] + 1e-7).all()))
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
        """glycogenolysis = (1 − f_gng)·EGP_b·(LGly/LGly_b)·g_ins·g_gn·g_G·mod/mod_ref, and the
        insulin gate is the teacher's basal-normalized IC50: exactly 1 at I = Ib, 1/(1+(I/K)^2)-
        shaped above it, → 0 at high insulin."""
        m = _model(4, perturb=1.0)
        met = m.metabolic
        state, coupling, external, emb, tf = _inputs(m)
        with torch.no_grad():
            ib = met.insulin_setpoint_raw(emb)
            k = torch.nn.functional.softplus(met.log_glyc_ins_k)
        for excess in (0.0, 10.0 * float(k)):
            s = state.clone()
            s[:, M._INSULIN_IDX] = (ib + excess - 10.0) / 10.0
            with torch.no_grad():
                f = met.fluxes(s, coupling, external, emb, tf)
            expected_gate = (1 + (ib / k) ** 2) / (1 + ((ib + excess) / k) ** 2)
            torch.testing.assert_close(f["g_ins_glyco"], expected_gate, atol=1e-6, rtol=1e-5)
            expected = ((1 - f["f_gng"]) * f["egp_b"] * (100.0 + 60.0 * s[:, M._LIVER_GLYCOGEN_IDX]).clamp(min=0) / 100.0
                        * f["g_ins_glyco"] * f["g_gn"] * f["g_g"] * f["mod_liver"])
            torch.testing.assert_close(f["glycogenolysis_plasma"], expected, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(f["brk_liver"], f["glycogenolysis_plasma"] / f["mg_dl_per_g"], atol=1e-7, rtol=1e-6)
        self.assertLess(float(f["g_ins_glyco"].max()), 0.02)

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
        self.assertTrue(bool((f["xa_rate"] < 0).all()))  # signed: I < Ib drives X down

    @staticmethod
    def _reference_state(m, gb: float, lgly: float = 100.0):
        """The patient's fasted reference: every species at typical except glucose at Gb,
        insulin at Ib (zero-init head → 10); no appearance; cortisol/GLP-1 basal; awake, rest."""
        emb = m.embedding_projections["metabolic"](torch.zeros(1, EMBEDDING_DIM))
        with torch.no_grad():
            ib = float(m.metabolic.insulin_setpoint_raw(emb))
        state = torch.zeros(1, len(_MET))
        state[0, M._GLUCOSE_IDX] = (gb - 95.0) / 30.0
        state[0, M._INSULIN_IDX] = (ib - 10.0) / 10.0
        with torch.no_grad():
            ffa_b = float(m.metabolic.ffa_setpoint_raw(emb))
            gn_b = float(m.metabolic.gn_setpoint_raw(emb))
        state[0, M._FFA_IDX] = (ffa_b - 0.5) / 0.2
        state[0, M._GLUCAGON_IDX] = (gn_b - 70.0) / 20.0
        state[0, M._LIVER_GLYCOGEN_IDX] = (lgly - 100.0) / 60.0
        coupling = torch.zeros(1, M._N_COUPLING)
        external = torch.tensor([[0.0, 1.0]])
        tf = compute_time_features(torch.tensor([600.0]))
        return state, coupling, external, emb, tf

    def test_gb_is_an_exact_fixed_point_for_every_patient(self) -> None:
        """At the fasted reference dG = EGP_b − k_ii·Gb = 0 exactly, and glycogenolysis is
        exactly (1 − f_gng)·EGP_b — the learned modulations are normalized to 1 there. With
        the pool halved dG < 0: the fasting fall emerges from depletion, not from a moved
        setpoint (there is no Gb_fasted any more)."""
        m = _model(9, perturb=1.0)
        met = m.metabolic
        self.assertFalse(hasattr(met, "log_gb_drop_abs"))
        for gb in (75.0, 95.0, 120.0):
            self._set_gb(m, gb)
            args = self._reference_state(m, gb)
            with torch.no_grad():
                f = met.fluxes(*args)
                rate = met(*args)[0, M._GLUCOSE_IDX]
            self.assertAlmostEqual(float(rate), 0.0, places=5, msg=f"Gb={gb}")
            self.assertAlmostEqual(float(f["glycogenolysis_plasma"]), float((1 - f["f_gng"]) * f["egp_b"]), places=6)
            self.assertAlmostEqual(float(f["gng_plasma"]), float(f["f_gng"] * f["egp_b"]), places=6)
            self.assertAlmostEqual(float(f["egp_b"]), float(f["k_ii"]) * gb, places=6)
            args_half = self._reference_state(m, gb, lgly=50.0)
            with torch.no_grad():
                f_half = met.fluxes(*args_half)
            # First-order in the pool, independent of the learned modulation.
            self.assertAlmostEqual(
                float(f_half["glycogenolysis_plasma"] / f_half["mod_liver"]),
                float(0.5 * (1 - f_half["f_gng"]) * f_half["egp_b"]),
                places=5, msg=f"Gb={gb}",
            )

    def test_basal_insulin_gate_is_gentle_below_gb(self) -> None:
        """2/(1 + (Gb/G)^n) on the basal term: 1 at Gb, ~0.79 at 0.9·Gb (not 0.59), and it
        does not touch the gated peak term."""
        m = _model(9, perturb=0.0)
        met = m.metabolic
        state, coupling, external, emb, tf = self._reference_state(m, 95.0)
        state[0, M._GLUCOSE_IDX] = (0.9 * 95.0 - 95.0) / 30.0
        with torch.no_grad():
            f = met.fluxes(state, coupling, external, emb, tf)
        self.assertAlmostEqual(float(f["ins_basal_gate"]), 2 / (1 + (1 / 0.9) ** 4), places=5)
        self.assertGreater(float(f["ins_basal_gate"]), 0.75)

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
            # the fasting fall is a flux deficit from pool depletion, so glucose sits BELOW
            # Gb after 16 h for every patient, above the absolute GNG floor (~60)
            self.assertGreaterEqual(g, 60.0, msg=f"Gb={gb}: glucose {g}")
            self.assertLessEqual(g, gb + 2.0, msg=f"Gb={gb}: glucose {g}")
            self.assertGreater(ins, 1.0, msg=f"Gb={gb}: insulin {ins}")
            self.assertLess(ins, 25.0, msg=f"Gb={gb}: insulin {ins}")
            # The standing insulin action is exactly the lagged excess over THIS patient's
            # Ib (10 at the zero-init head) — not over a population 10 that the Gb=120
            # patient's insulin would sit permanently above once trained.
            ib = float(m.metabolic.insulin_setpoint_raw(
                m.embedding_projections["metabolic"](torch.zeros(EMBEDDING_DIM))))
            # (tau = 1/p2 = 50 min, so the lag trails a still-drifting insulin by a little)
            self.assertAlmostEqual(xa, (ins - ib) / 10.0, delta=0.08, msg=f"Gb={gb}")


class TestBergmanClearance(unittest.TestCase):
    """3.10: insulin action is a sink, never a source."""

    def test_insulin_action_lowers_the_glucose_rate_below_the_setpoint(self) -> None:
        m = _model(11, perturb=1.0)
        met = m.metabolic
        # fasted (no appearance, so the store split — whose heads also read Xa — is moot)
        state, coupling, external, emb, tf = _inputs(m, app=0.0)
        state[:, M._GLUCOSE_IDX] = (85.0 - 95.0) / 30.0     # G = 85 < Gb ≈ 95 (the review's case)
        s0 = state.clone(); s0[:, M._INSULIN_ACTION_IDX] = 0.0
        s1 = state.clone(); s1[:, M._INSULIN_ACTION_IDX] = 2.0
        with torch.no_grad():
            f0 = met.fluxes(s0, coupling, external, emb, tf)
            f1 = met.fluxes(s1, coupling, external, emb, tf)
        # insulin action enters ONLY as uptake_id = Si·Xa·G ≥ 0 (the hepatic heads also
        # read Xa in x, so the pin is on the non-hepatic balance: appearance − uptakes)
        self.assertEqual(float(f0["uptake_id"].abs().sum()), 0.0)
        self.assertTrue(bool((f1["uptake_id"] > 0).all()))
        non_hep = lambda f: f["appearance_plasma"] - f["uptake_ii"] - f["uptake_id"] - f["exercise_uptake"]
        self.assertTrue(bool((non_hep(f1) < non_hep(f0)).all()))


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
        with torch.no_grad():
            f0 = met.fluxes(state, coupling, external, emb, tf)
            f1 = met.fluxes(s1, coupling, external, emb, tf)
        self.assertFalse(torch.equal(f0["mito_rate"], f1["mito_rate"]))

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
        with torch.no_grad():
            ffa_b = met.ffa_setpoint_raw(emb)
        low = state.clone()
        low[:, M._FFA_IDX] = (0.5 * ffa_b - M._FFA_CENTER) / M._FFA_SCALE
        with torch.no_grad():
            f = met.fluxes(low, coupling, external, emb, tf)
        self.assertEqual(float(f["ketogenesis"].abs().sum()), 0.0)
        hi = state.clone()
        hi[:, M._FFA_IDX] = (2.0 * ffa_b - M._FFA_CENTER) / M._FFA_SCALE
        hi[:, M._INSULIN_IDX] = (4.0 - 10.0) / 10.0
        hi_ins = hi.clone(); hi_ins[:, M._INSULIN_IDX] = (60.0 - 10.0) / 10.0
        with torch.no_grad():
            f_fast = met.fluxes(hi, coupling, external, emb, tf)
            f_fed = met.fluxes(hi_ins, coupling, external, emb, tf)
        self.assertTrue(bool((f_fast["ketogenesis"] > 0).all()))
        self.assertTrue(bool((f_fed["ketogenesis"] < f_fast["ketogenesis"]).all()))

    def test_hepatic_output_target_is_the_two_fluxes_in_mg_per_kg(self) -> None:
        m = _model(15, perturb=0.5)
        met = m.metabolic
        state, coupling, external, emb, tf = _inputs(m)
        with torch.no_grad():
            f = met.fluxes(state, coupling, external, emb, tf)
        expected = (f["glycogenolysis_plasma"] + f["gng_plasma"]) * VG_DL / BODY_MASS_KG
        torch.testing.assert_close(f["hep_target"], expected, atol=1e-5, rtol=1e-5)
        # and the typical patient at rest reads the textbook 2.0 mg/kg/min
        m0 = _model(15, perturb=0.0)
        args = TestPerPatientGates._reference_state(m0, 95.0)
        with torch.no_grad():
            f0 = m0.metabolic.fluxes(*args)
        self.assertAlmostEqual(float(f0["hep_target"]), 2.0, places=2)


class TestNoDeadHeads(unittest.TestCase):
    def test_structural_species_own_no_parameters(self) -> None:
        m = _model(16)
        for idx in (M._GLUCOSE_IDX, M._INSULIN_ACTION_IDX, M._MITO_IDX, M._FAT_MASS_IDX):
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


class TestVolumeAndEnergy(unittest.TestCase):
    def test_a_gram_of_carbohydrate_is_fewer_mg_dl_in_a_heavier_person(self) -> None:
        m = _model(0, perturb=0.0)
        met = m.metabolic
        state, coupling, external, emb, tf = _inputs(m, batch=1, app=2.0)
        with torch.no_grad():
            f70 = met.fluxes(state, coupling, external, emb, tf)
            met.body_mass_net[-1].bias.fill_(1.0)
            f_hi = met.fluxes(state, coupling, external, emb, tf)
        self.assertGreater(float(f_hi["body_mass_kg"]), float(f70["body_mass_kg"]))
        self.assertLess(float(f_hi["mg_dl_per_g"]), float(f70["mg_dl_per_g"]))
        self.assertAlmostEqual(float(f70["app_g"]), float(f_hi["app_g"]), places=5)
        self.assertLess(float(f_hi["app_eff"]), float(f70["app_eff"]))

    def test_fat_mass_falls_on_a_fast(self) -> None:
        m = _model(0, perturb=0.0)
        n = 1440
        with torch.no_grad():
            tr = integrate(
                m, _CENTER, torch.zeros(EMBEDDING_DIM), n,
                start_time_minutes=360, meals=[],
                sleep_wake=torch.ones(n), activity=torch.zeros(n),
            )
        self.assertLess(float(tr[-1, MI["fat_mass"]]), float(tr[0, MI["fat_mass"]]) - 0.05)


class TestDietaryMacrosOnCouplingChannels(unittest.TestCase):
    """Lipid and amino appearance already sit on the metabolic coupling vector.
    They used to affect only the fat-mass calorie residual; the teacher also
    puts them on FFA and glucagon."""

    def test_lipid_appearance_is_an_ffa_source_amino_is_a_glucagon_source(self) -> None:
        m = _model(0, perturb=0.5)
        met = m.metabolic
        state, coupling, external, emb, tf = _inputs(m, app=0.0)
        coupling[:, M._LIPID_COUPLING_IDX] = 0.4
        coupling[:, M._AMINO_COUPLING_IDX] = 0.6
        with torch.no_grad():
            f = met.fluxes(state, coupling, external, emb, tf)
            rate = met(state, coupling, external, emb, tf)
            prod, cons = f["prod_raw"], f["cons_raw"]
            mito, raw = f["mito"], met.raw_state(state)
        torch.testing.assert_close(f["ffa_from_lipid"], M._FFA_FROM_LIPID * coupling[:, M._LIPID_COUPLING_IDX])
        torch.testing.assert_close(f["glucagon_from_amino"], M._GN_FROM_AMINO * coupling[:, M._AMINO_COUPLING_IDX])
        ffa_ma = (prod[:, M._FFA_IDX] * met.prod_scale[M._FFA_IDX]
                  - cons[:, M._FFA_IDX] * met.cons_scale[M._FFA_IDX] * mito * raw[:, M._FFA_IDX])
        gn_ma = (prod[:, M._GLUCAGON_IDX] * met.prod_scale[M._GLUCAGON_IDX]
                 - cons[:, M._GLUCAGON_IDX] * met.cons_scale[M._GLUCAGON_IDX] * raw[:, M._GLUCAGON_IDX])
        torch.testing.assert_close(rate[:, M._FFA_IDX], ffa_ma + f["ffa_from_lipid"], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(rate[:, M._GLUCAGON_IDX], gn_ma + f["glucagon_from_amino"], atol=1e-6, rtol=1e-5)

    def test_carb_appearance_does_not_create_those_terms(self) -> None:
        m = _model(1, perturb=0.5)
        met = m.metabolic
        state, coupling, external, emb, tf = _inputs(m, app=2.0)
        self.assertEqual(float(coupling[:, M._LIPID_COUPLING_IDX].abs().sum()), 0.0)
        self.assertEqual(float(coupling[:, M._AMINO_COUPLING_IDX].abs().sum()), 0.0)
        with torch.no_grad():
            f = met.fluxes(state, coupling, external, emb, tf)
        self.assertEqual(float(f["ffa_from_lipid"].abs().sum()), 0.0)
        self.assertEqual(float(f["glucagon_from_amino"].abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
