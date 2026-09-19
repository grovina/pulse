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
    NORM_CENTER, NORM_SCALE, PHYSIOLOGICAL_MIN, VG_DL,
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

    def test_effective_ib_is_the_teacher_glucose_gated_basal(self) -> None:
        """At G = 0.9·Gb (this patient's Gb), I = Ib: GSIR is 0 and I is restored toward Ib·0.9^5."""
        m = _model(11, perturb=0.3)
        met = m.metabolic
        state, coupling, external, emb, tf = self._reference_state(m, 95.0)
        with torch.no_grad():
            gb = float(met.glucose_setpoint_raw(emb))
            ib = float(met.insulin_setpoint_raw(emb))
        state = state.clone()
        state[0, M._GLUCOSE_IDX] = (0.9 * gb - 95.0) / 30.0
        state[0, M._INSULIN_IDX] = (ib - 10.0) / 10.0
        with torch.no_grad():
            f = met.fluxes(state, coupling, external, emb, tf)
            rate = met(state, coupling, external, emb, tf)[0, M._INSULIN_IDX]
        expected = ib * max((0.9) ** M._FAST_INS_EXP, M._FAST_INS_FLOOR)
        self.assertAlmostEqual(float(f["gb"]), gb, places=5)
        self.assertAlmostEqual(float(f["effective_ib"]), expected, places=5)
        self.assertEqual(float(f["ins_gsir"]), 0.0)
        self.assertLess(float(rate), 0.0)
        self.assertAlmostEqual(float(rate), float(f["ins_rate"]), places=6)
        self.assertAlmostEqual(
            float(f["ins_restoring"]),
            -float(f["k_ins"]) * (ib - expected),
            places=5,
        )

    def test_gsir_is_off_at_gb_and_on_above(self) -> None:
        m = _model(4, perturb=0.0)
        met = m.metabolic
        gb = 95.0
        args = self._reference_state(m, gb)
        with torch.no_grad():
            f0 = met.fluxes(*args)
            rate0 = met(*args)[0, M._INSULIN_IDX]
        self.assertEqual(float(f0["ins_gsir"]), 0.0)
        self.assertAlmostEqual(float(rate0), 0.0, places=5)
        self.assertAlmostEqual(float(f0["effective_ib"]), float(f0["ib"]), places=5)
        state, coupling, external, emb, tf = args
        state = state.clone()
        state[0, M._GLUCOSE_IDX] = (gb + 30.0 - 95.0) / 30.0
        with torch.no_grad():
            f1 = met.fluxes(state, coupling, external, emb, tf)
        self.assertGreater(float(f1["ins_gsir"]), 0.5)
        self.assertAlmostEqual(float(f1["effective_ib"]), float(f1["ib"]), places=5)

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
            self.assertGreater(xa, PHYSIOLOGICAL_MIN[MI["insulin_action"]] + 1.0, msg=f"Gb={gb}")


class TestSignedInsulinAction(unittest.TestCase):
    """Remote insulin is a signed latent, not a concentration.

    Flooring it at 0 in raw_state zeros the tracker whenever X < 0 and walks
    X to PHYSIOLOGICAL_MIN on a fast. Concentrations still floor at 0.
    """

    def test_raw_state_passes_signed_insulin_action_and_floors_concentrations(self) -> None:
        m = _model(0, perturb=0.0)
        met = m.metabolic
        state = torch.zeros(1, len(_MET))
        state[0, M._INSULIN_ACTION_IDX] = -4.0
        state[0, M._INSULIN_IDX] = -2.0          # would be −10 µU/mL
        state[0, M._GLUCOSE_IDX] = -4.0          # would be −25 mg/dL
        raw = met.raw_state(state)
        self.assertAlmostEqual(float(raw[0, M._INSULIN_ACTION_IDX]), -4.0, places=5)
        self.assertEqual(float(raw[0, M._INSULIN_IDX]), 0.0)
        self.assertEqual(float(raw[0, M._GLUCOSE_IDX]), 0.0)

    def test_sub_basal_tracker_restores_when_x_is_too_negative(self) -> None:
        m = _model(8, perturb=0.0)
        met = m.metabolic
        state, coupling, external, emb, tf = _inputs(m, batch=1, seed=0, app=0.0)
        state[0, M._INSULIN_IDX] = (7.0 - 10.0) / 10.0
        state[0, M._INSULIN_ACTION_IDX] = -4.0
        with torch.no_grad():
            f = met.fluxes(state, coupling, external, emb, tf)
        self.assertAlmostEqual(float(f["xa"]), -4.0, places=5)
        self.assertGreater(float(f["insulin_dev"]), -1.0)
        self.assertLess(float(f["insulin_dev"]), 0.0)
        self.assertGreater(float(f["xa_rate"]), 0.0)

    def test_negative_insulin_action_reaches_glucose_as_a_floored_source(self) -> None:
        m = _model(11, perturb=0.0)
        met = m.metabolic
        state, coupling, external, emb, tf = _inputs(m, batch=1, seed=0, app=0.0)
        state[0, M._GLUCOSE_IDX] = 0.0
        state[0, M._INSULIN_ACTION_IDX] = -0.05
        with torch.no_grad():
            f = met.fluxes(state, coupling, external, emb, tf)
        g = 95.0
        self.assertLess(float(f["uptake_id"]), 0.0)
        si = float((M._SI_MIN + M._SI_RANGE * torch.sigmoid(met.log_si)).item())
        self.assertAlmostEqual(float(f["uptake_id"]), si * -0.05 * g, places=6)
        floor = -M._INS_DEP_BASAL_FRAC * float(f["k_ii"])
        state[0, M._INSULIN_ACTION_IDX] = -20.0
        with torch.no_grad():
            f_wall = met.fluxes(state, coupling, external, emb, tf)
        self.assertAlmostEqual(float(f_wall["xa"]), -20.0, places=5)
        self.assertAlmostEqual(float(f_wall["uptake_id"]), floor * g, places=6)

    def test_48h_fast_tracks_insulin_dev_and_does_not_pin_the_wall(self) -> None:
        m = _model(10, hidden=32, perturb=0.0)
        n = 2880
        xa_floor = PHYSIOLOGICAL_MIN[MI["insulin_action"]]
        with torch.no_grad():
            tr = integrate(
                m, _CENTER.clone(), torch.zeros(EMBEDDING_DIM), n,
                start_time_minutes=8 * 60, meals=[],
                sleep_wake=torch.ones(n), activity=torch.zeros(n),
            )
        xa = tr[:, MI["insulin_action"]]
        ins = tr[-1, MI["insulin"]]
        ib = float(m.metabolic.insulin_setpoint_raw(
            m.embedding_projections["metabolic"](torch.zeros(EMBEDDING_DIM))))
        self.assertGreater(float(xa.min()), xa_floor + 1.0)
        self.assertAlmostEqual(float(xa[-1]), float((ins - ib) / 10.0), delta=0.15)

    def test_negative_x_recovers_toward_insulin_dev_instead_of_walking_away(self) -> None:
        """The 99 disease: raw_state floors X at 0, so dX/dt = p2·(I−Ib)/10 with
        no −X term, and a state of −4 never comes back. With the coordinate,
        τ = 50 min brings it back to ~0 in a few hours."""
        m = _model(10, hidden=32, perturb=0.0)
        initial = _CENTER.clone()
        initial[MI["insulin_action"]] = -4.0
        n = 300
        with torch.no_grad():
            tr = integrate(
                m, initial, torch.zeros(EMBEDDING_DIM), n,
                start_time_minutes=8 * 60, meals=[],
                sleep_wake=torch.ones(n), activity=torch.zeros(n),
            )
        xa = tr[:, MI["insulin_action"]]
        self.assertGreater(float(xa[-1]), -1.0)
        self.assertGreater(float(xa.min()), -4.05)


class TestBergmanClearance(unittest.TestCase):
    """3.10: above-basal insulin action is a sink; sub-basal X is a floored source."""

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
    """3.11: mito scales oxidative clearance of lactate, and nothing else."""

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

    def test_activity_above_rest_raises_mito(self) -> None:
        """Mito is a slow training-stimulus species. A rest overnight does not
        prove it is padded; activity above 0.2 does."""
        m = _model(14, perturb=0.0)
        met = m.metabolic
        state, coupling, external, emb, tf = _inputs(m, act=0.0)
        rest = external.clone()
        rest[:, M._ACTIVITY_EXTERNAL_IDX] = 0.0
        bout = external.clone()
        bout[:, M._ACTIVITY_EXTERNAL_IDX] = 0.5
        with torch.no_grad():
            f_rest = met.fluxes(state, coupling, rest, emb, tf)
            f_bout = met.fluxes(state, coupling, bout, emb, tf)
        self.assertTrue(bool((f_bout["mito_rate"] > f_rest["mito_rate"]).all()))

    def test_mito_scales_oxidative_clearance(self) -> None:
        m = _model(13, perturb=0.5)
        met = m.metabolic
        state, coupling, external, emb, tf = _inputs(m)
        state[:, M._FFA_IDX] = 1.0; state[:, M._BHB_IDX] = 1.0; state[:, M._LACTATE_IDX] = 1.0
        s1 = state.clone(); s1[:, M._MITO_IDX] += 1.0   # mito 1.0 -> 1.3
        with torch.no_grad():
            r0 = met(state, coupling, external, emb, tf)
            r1 = met(s1, coupling, external, emb, tf)
        for idx in (M._LACTATE_IDX,):
            self.assertTrue(bool((r1[:, idx] < r0[:, idx]).all()), msg=f"species {idx}")
        torch.testing.assert_close(r0[:, M._BHB_IDX], r1[:, M._BHB_IDX], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(r0[:, M._FFA_IDX], r1[:, M._FFA_IDX], atol=1e-6, rtol=1e-5)
        for idx in (M._GLUCOSE_IDX, M._INSULIN_IDX, M._GLUCAGON_IDX, M._HEPATIC_IDX):
            torch.testing.assert_close(r0[:, idx], r1[:, idx], atol=1e-6, rtol=1e-6)


class TestKetogenesisAndHepaticOutput(unittest.TestCase):
    def test_bhb_basal_is_a_fixed_point(self) -> None:
        """At FFA=FFA_b, I=Ib, LGly=LGly_b, BHB=BHB_b: dBHB=0, even after a random
        perturbation of k_keto — clearance is derived from production."""
        m = _model(14, perturb=1.0)
        gb = 95.0
        args = TestPerPatientGates._reference_state(m, gb)
        with torch.no_grad():
            f = m.metabolic.fluxes(*args)
            rate = m.metabolic(*args)[0, M._BHB_IDX]
        self.assertAlmostEqual(float(rate), 0.0, places=5)
        self.assertAlmostEqual(float(f["bhb_rate"]), 0.0, places=5)
        self.assertGreater(float(f["ketogenesis"]), 0.0)
        self.assertAlmostEqual(float(f["glyco_depletion"]), 0.0, places=5)

    def test_empty_liver_at_basal_substrate_raises_bhb(self) -> None:
        """The 101 miss: production on FFA_b, not on relu(FFA−FFA_b). An empty
        liver at basal FFA/insulin is already a ketogenic source."""
        m = _model(14, perturb=1.0)
        empty = TestPerPatientGates._reference_state(m, 95.0, lgly=0.0)
        with torch.no_grad():
            rate = m.metabolic(*empty)[0, M._BHB_IDX]
        self.assertGreater(float(rate), 0.0)

    def test_zero_ffa_is_zero_ketogenesis(self) -> None:
        m = _model(14, perturb=0.5)
        met = m.metabolic
        state, coupling, external, emb, tf = _inputs(m)
        state = state.clone()
        state[:, M._FFA_IDX] = (0.0 - M._FFA_CENTER) / M._FFA_SCALE
        with torch.no_grad():
            f = met.fluxes(state, coupling, external, emb, tf)
        self.assertEqual(float(f["ketogenesis"].abs().sum()), 0.0)

    def test_empty_liver_multiplies_production_by_one_plus_gain(self) -> None:
        m = _model(14, perturb=0.4)
        full = TestPerPatientGates._reference_state(m, 95.0, lgly=100.0)
        empty = TestPerPatientGates._reference_state(m, 95.0, lgly=0.0)
        with torch.no_grad():
            f_full = m.metabolic.fluxes(*full)
            f_empty = m.metabolic.fluxes(*empty)
        ratio = f_empty["ketogenesis"] / f_full["ketogenesis"].clamp(min=1e-12)
        self.assertAlmostEqual(float(ratio), 1.0 + M._KETO_GLYC_GAIN, places=4)
        self.assertAlmostEqual(float(f_empty["glyco_depletion"]), 1.0, places=5)

    def test_insulin_in_the_denominator_is_a_concentration(self) -> None:
        """Sub-basal insulin raises ketogenesis; high insulin suppresses it.
        Same FFA, full liver — the rectifier on I−Ib cannot do this."""
        m = _model(14, perturb=0.3)
        met = m.metabolic
        state, coupling, external, emb, tf = TestPerPatientGates._reference_state(m, 95.0)
        with torch.no_grad():
            ib = float(met.insulin_setpoint_raw(emb))
        low = state.clone()
        low[:, M._INSULIN_IDX] = (0.4 * ib - 10.0) / 10.0
        high = state.clone()
        high[:, M._INSULIN_IDX] = (4.0 * ib - 10.0) / 10.0
        with torch.no_grad():
            f_b = met.fluxes(state, coupling, external, emb, tf)
            f_low = met.fluxes(low, coupling, external, emb, tf)
            f_high = met.fluxes(high, coupling, external, emb, tf)
        self.assertGreater(float(f_low["ketogenesis"]), float(f_b["ketogenesis"]))
        self.assertLess(float(f_high["ketogenesis"]), float(f_b["ketogenesis"]))

    def test_24h_fast_bhb_rises_above_fed_baseline(self) -> None:
        m = _model(14, hidden=32, perturb=0.0)
        n = 1440
        with torch.no_grad():
            tr = integrate(
                m, _CENTER.clone(), torch.zeros(EMBEDDING_DIM), n,
                start_time_minutes=20.0 * 60.0, meals=[],
                sleep_wake=torch.ones(n), activity=torch.zeros(n),
            )
        self.assertGreater(float(tr[-1, MI["bhb"]]), 0.4)
        self.assertGreater(float(tr[-1, MI["bhb"]]), float(tr[0, MI["bhb"]]) + 0.2)

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
        for idx in (M._GLUCOSE_IDX, M._INSULIN_ACTION_IDX, M._MITO_IDX, M._FAT_MASS_IDX, M._BHB_IDX, M._FFA_IDX):
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


class TestLipolysis(unittest.TestCase):
    def test_ffa_basal_is_a_fixed_point(self) -> None:
        """At I=Ib, FFA=FFA_b: dFFA=0 even after a random perturbation of k_ffa —
        lip_max is derived from clearance."""
        m = _model(18, perturb=1.0)
        args = TestPerPatientGates._reference_state(m, 95.0)
        with torch.no_grad():
            f = m.metabolic.fluxes(*args)
            rate = m.metabolic(*args)[0, M._FFA_IDX]
        self.assertAlmostEqual(float(rate), 0.0, places=5)
        self.assertAlmostEqual(float(f["ffa_rate"]), 0.0, places=5)
        self.assertGreater(float(f["lipolysis"]), 0.0)

    def test_sub_basal_insulin_raises_lipolysis(self) -> None:
        """The 102 miss: I=5.7 should raise FFA, not crash it. Insulin is a
        concentration in the denominator, not a rectifier on I−Ib."""
        m = _model(18, perturb=0.3)
        met = m.metabolic
        state, coupling, external, emb, tf = TestPerPatientGates._reference_state(m, 95.0)
        with torch.no_grad():
            ib = float(met.insulin_setpoint_raw(emb))
        low = state.clone()
        low[:, M._INSULIN_IDX] = (0.4 * ib - 10.0) / 10.0
        high = state.clone()
        high[:, M._INSULIN_IDX] = (4.0 * ib - 10.0) / 10.0
        with torch.no_grad():
            f_b = met.fluxes(state, coupling, external, emb, tf)
            f_low = met.fluxes(low, coupling, external, emb, tf)
            f_high = met.fluxes(high, coupling, external, emb, tf)
            r_low = met(low, coupling, external, emb, tf)[0, M._FFA_IDX]
        self.assertGreater(float(f_low["lipolysis"]), float(f_b["lipolysis"]))
        self.assertLess(float(f_high["lipolysis"]), float(f_b["lipolysis"]))
        self.assertGreater(float(r_low), 0.0)

    def test_excess_ffa_at_basal_insulin_is_cleared(self) -> None:
        m = _model(18, perturb=0.4)
        met = m.metabolic
        state, coupling, external, emb, tf = TestPerPatientGates._reference_state(m, 95.0)
        with torch.no_grad():
            ffa_b = float(met.ffa_setpoint_raw(emb))
        hi = state.clone()
        hi[:, M._FFA_IDX] = (2.0 * ffa_b - 0.5) / 0.2
        with torch.no_grad():
            rate = met(hi, coupling, external, emb, tf)[0, M._FFA_IDX]
        self.assertLess(float(rate), 0.0)

    def test_overnight_ffa_does_not_crash_below_half_basal(self) -> None:
        m = _model(18, hidden=32, perturb=0.0)
        n = 720
        hour = ((20.0 * 60.0 + torch.arange(n)) % 1440.0) / 60.0
        sleep_wake = 1.0 - ((hour >= 23.0) | (hour < 7.0)).float()
        with torch.no_grad():
            tr = integrate(
                m, _CENTER.clone(), torch.zeros(EMBEDDING_DIM), n,
                start_time_minutes=20.0 * 60.0, meals=[],
                sleep_wake=sleep_wake, activity=torch.zeros(n),
            )
            ffa_b = float(m.metabolic.ffa_setpoint_raw(
                m.embedding_projections["metabolic"](torch.zeros(1, EMBEDDING_DIM))))
        self.assertGreater(float(tr[:, MI["ffa"]].min()), 0.5 * ffa_b)


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
            raw = met.raw_state(state)
        torch.testing.assert_close(f["ffa_from_lipid"], M._FFA_FROM_LIPID * coupling[:, M._LIPID_COUPLING_IDX])
        torch.testing.assert_close(f["glucagon_from_amino"], M._GN_FROM_AMINO * coupling[:, M._AMINO_COUPLING_IDX])
        torch.testing.assert_close(rate[:, M._FFA_IDX], f["ffa_rate"], atol=1e-6, rtol=1e-5)
        gn_ma = (prod[:, M._GLUCAGON_IDX] * met.prod_scale[M._GLUCAGON_IDX]
                 - cons[:, M._GLUCAGON_IDX] * met.cons_scale[M._GLUCAGON_IDX] * raw[:, M._GLUCAGON_IDX])
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
