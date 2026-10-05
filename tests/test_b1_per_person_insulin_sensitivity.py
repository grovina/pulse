"""PLAN B1 — per-person insulin sensitivity, and the supervision that identifies it.

Through iter 109 the student held the whole insulin-sensitivity family at ONE
population scalar — peripheral Si, the hepatic gate IC50, the β-cell gain γ and
insulin clearance k_ins — while the teacher draws all four per patient, and it had
no activity term on insulin action at all. The PRD's own example of individual
variation is "exact insulin sensitivity", so this is the one axis the model could
not represent; the only lever it had for a patient who handles a meal badly was
meal amplitude, which moves glucose and insulin for the wrong reason.

Measured here (teacher, N=150 sampled patients, 75 g mixed meal, incremental AUC
over the 240 min after the meal): corr(log Si, glucose iAUC) = −0.48, bottom-vs-top
Si decile 6114 vs 3215 = 1.90× on glucose and 6491 vs 2990 = 2.17× on insulin.
PLAN's headline "34 % of between-person glucose iAUC" is the same thing measured on
TOTAL post-meal AUC (corr −0.58, R² 33.8 %), which also carries the fasting level;
on incremental AUC Si's share is 22.5 % of glucose and 20.7 % of insulin. Either
way it is the largest single axis, and the direction and the decile ratio hold.

What these tests pin:

  * the five reachable spans, and that each contains the teacher's own ±2σ draw
    (measured with ``randomize_params``, not asserted from the declared σ);
  * a zero embedding decodes to the population scalar EXACTLY — the cheapest
    regression test there is, since iter 109's behaviour is the zero code's;
  * strict positivity for 200 random embeddings including the ‖e‖ = 8 calibration
    clamp, which log space gives by construction;
  * the Jensen property (PLAN §1): the mean decoded Si over a symmetric code
    population is strictly ABOVE Si(0), so zero can be the median person while the
    population mean sits above it, with no loss term asking for it;
  * the carve: muscle Si moves peripheral disposal and NOTHING hepatic, the hepatic
    factor moves both hepatic gates and NOT disposal — which is what makes Donga
    2010's "peripheral −29 %, hepatic unchanged" expressible at all;
  * the one that matters: a 75 g meal's glucose iAUC falls monotonically in decoded
    Si, and the student's realized decile ratio is the teacher's (1.77× vs 1.81× on
    the teacher's own pure-Si sweep);
  * activity sensitises insulin action, in the teacher's form and at the teacher's
    place in the lag;
  * ``SetpointSupervisionSignal`` drives every decode onto the teacher's draw, and
    is a no-op without ``param_targets``.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from pulse.knowledge.full_body import FullBody, PatientParams, randomize_params
from pulse.model import ModularPhysiologyNetwork, integrate
from pulse.modules import metabolic as M
from pulse.modules.base import compute_time_features
from pulse.modules.gut import MealEvent
from pulse.training.setpoint_supervision_signal import (
    PERSON_PARAM_CENTERS, SetpointSupervisionSignal, _TEACHER_DEFAULTS, _TEACHER_FIELD,
)
from pulse.training.signals import SignalContext, WeightSchedule
from pulse.types import EMBEDDING_DIM, MARKER_INDEX as MI, NORM_CENTER

_CENTER = torch.tensor(NORM_CENTER)
# The hard clamp on a calibrated embedding: CalibrationSettings.max_norm.
_MAX_NORM = 8.0
# metabolic.py's keys, and the log half-width of each decode.
_FAMILY = ("si", "glyc_ins_k", "gamma", "k_ins", "act_insulin_sens")
_SPANS = M.PERSON_PARAM_LOG_MAX
# Teacher Si deciles over the same 150 draws the module docstring's numbers come from.
_TEACHER_SI_P10, _TEACHER_SI_P90 = 2.158e-04, 7.534e-04
_MEAL_T, _DUR, _WINDOW = 60, 300, 240


def _model(seed: int = 0, hidden: int = 16, perturb: float = 0.0) -> ModularPhysiologyNetwork:
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


def _emb_width(m: ModularPhysiologyNetwork) -> int:
    return int(m.metabolic.insulin_sens_net[0].in_features)


def _at_zero(m: ModularPhysiologyNetwork) -> dict[str, float]:
    """What the zero embedding decodes to — the default person, i.e. iter 109."""
    with torch.no_grad():
        pp = m.metabolic.person_params(torch.zeros(1, _emb_width(m)))
    return {k: float(pp[k]) for k in pp}


def _centre(m: ModularPhysiologyNetwork) -> dict[str, float]:
    """Each decode's population scalar. Equal to ``_at_zero`` only while the heads are
    zero-init, which is exactly why the two are separate helpers: the sweeps below move
    a head's BIAS, after which the zero code is no longer the centre."""
    met, sp = m.metabolic, torch.nn.functional.softplus
    with torch.no_grad():
        return {
            "si": float(M._SI_MIN + M._SI_RANGE * torch.sigmoid(met.log_si)),
            "glyc_ins_k": float(sp(met.log_glyc_ins_k)),
            "gng_ins_k": float(sp(met.log_gng_ins_k)),
            "gamma": float(sp(met.log_gamma)),
            "k_ins": float(sp(met.log_k_ins)),
            "act_insulin_sens": float(sp(met.log_act_ins_sens)),
            "body_mass_kg": 70.0,
        }


def _heads(m: ModularPhysiologyNetwork) -> dict[str, torch.nn.Module]:
    met = m.metabolic
    return {
        "si": met.insulin_sens_net, "glyc_ins_k": met.hepatic_ins_k_net,
        "gamma": met.beta_cell_gain_net, "k_ins": met.insulin_clearance_net,
        "act_insulin_sens": met.act_insulin_sens_net, "body_mass_kg": met.body_mass_net,
    }


def _set_factor(m: ModularPhysiologyNetwork, key: str, factor: float) -> None:
    """Drive one decode to ``centre · factor`` through its head's BIAS.

    ``bias = atanh(log(factor)/L)``: the decode is ``centre·exp(L·tanh(bias))`` at a
    zeroed weight, so this is the inverse. Sweeping the head (not the population
    scalar) is the point — it is the per-person authority under test.
    """
    net = _heads(m)[key]
    with torch.no_grad():
        net[-1].weight.zero_()
        net[-1].bias.fill_(math.atanh(math.log(factor) / _SPANS[key]))


def _odd_head(net: torch.nn.Module, seed: int, std: float = 1.5) -> None:
    """Real authority AND an odd function of the code: both biases zero and tanh odd,
    so a code set closed under negation is exactly symmetric in the head's output and
    the only asymmetry left in the decode is its own convexity (the A1 construction)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for layer in (net[0], net[-1]):
            layer.weight.copy_(std * torch.randn(layer.weight.shape, generator=g))
            layer.bias.zero_()


def _embeddings(m: ModularPhysiologyNetwork, n: int = 200, seed: int = 0) -> torch.Tensor:
    """``n`` module-space codes at the init scale, the soft-norm radius and the hard
    clamp (norm exactly 8), a third each — the three radii a trained model produces."""
    g = torch.Generator().manual_seed(seed)
    raw = torch.randn(n, EMBEDDING_DIM, generator=g)
    unit = raw / raw.norm(dim=-1, keepdim=True)
    radius = torch.empty(n)
    third = n // 3
    radius[:third] = 0.1 * EMBEDDING_DIM ** 0.5
    radius[third:2 * third] = 3.0
    radius[2 * third:] = _MAX_NORM
    return m.embedding_projections["metabolic"](unit * radius.unsqueeze(-1))


def _meal_iauc(m: ModularPhysiologyNetwork, activity: float = 0.0) -> tuple[float, float]:
    """Glucose and insulin incremental AUC over the 240 min after a 75 g mixed meal,
    at the zero embedding (so only the swept head differs between runs). The pre-meal
    sample is the baseline; at a zero code Gb is an exact fixed point, so 60 min of
    settling is enough and the baseline is the patient's own equilibrium."""
    meals = [MealEvent(time=float(_MEAL_T), carbs=75.0, fats=5.0, proteins=10.0)]
    with torch.no_grad():
        tr = integrate(
            m, _CENTER.clone(), torch.zeros(EMBEDDING_DIM), _DUR,
            start_time_minutes=8 * 60.0, meals=meals,
            sleep_wake=torch.ones(_DUR), activity=torch.full((_DUR,), activity))
    out = []
    for marker in ("glucose", "insulin"):
        v = tr[:, MI[marker]]
        out.append(float((v[_MEAL_T:_MEAL_T + _WINDOW] - v[_MEAL_T - 1]).clamp(min=0).sum()))
    return out[0], out[1]


class TestReachableSpans(unittest.TestCase):
    def test_the_spans_are_the_ones_the_comment_states(self) -> None:
        self.assertEqual(M._SI_LOG_MAX, 1.25)
        for key in ("glyc_ins_k", "gamma", "k_ins", "act_insulin_sens"):
            self.assertEqual(_SPANS[key], 0.75)
        self.assertAlmostEqual(math.exp(-1.25), 0.2865, places=4)
        self.assertAlmostEqual(math.exp(1.25), 3.4903, places=4)
        self.assertAlmostEqual(math.exp(-0.75), 0.4724, places=4)
        self.assertAlmostEqual(math.exp(0.75), 2.1170, places=4)

    def test_the_bias_sweep_reaches_the_declared_edges(self) -> None:
        """A saturating tanh is the bound, so a large bias lands exactly on it."""
        m = _model(0)
        pop = _centre(m)
        for key in _FAMILY:
            net = _heads(m)[key]
            for sign in (-1.0, +1.0):
                with torch.no_grad():
                    net[-1].weight.zero_()
                    net[-1].bias.fill_(sign * 40.0)
                    got = float(m.metabolic.person_params(torch.zeros(1, _emb_width(m)))[key])
                # float32: compare as a ratio, the edge itself is e^±L of the centre
                self.assertAlmostEqual(
                    got / (pop[key] * math.exp(sign * _SPANS[key])), 1.0, places=5, msg=key)

    def test_each_span_contains_the_teachers_own_two_sigma_draw(self) -> None:
        """Measured, not asserted from the declared σ: 2,000 ``randomize_params``
        draws, compared as log-ratios to each parameter's default. The teacher CLIPS
        two of them, so the realized spread is what has to fit, and for the hepatic
        gate the whole clip ([12, 50]) fits — 0 draws out of range."""
        draws: dict[str, list[float]] = {k: [] for k in _FAMILY}
        ref = PatientParams()
        for seed in range(2000):
            p = randomize_params(np.random.default_rng(seed))
            for key in _FAMILY:
                field = _TEACHER_FIELD[key]
                draws[key].append(math.log(getattr(p, field) / getattr(ref, field)))
        for key in _FAMILY:
            v = np.array(draws[key])
            lo, hi = np.percentile(v, 2.275), np.percentile(v, 97.725)
            self.assertLess(-_SPANS[key], lo, msg=f"{key}: ±2σ low {lo:+.3f} unreachable")
            self.assertGreater(_SPANS[key], hi, msg=f"{key}: ±2σ high {hi:+.3f} unreachable")
            # Margin, in log units, between ±2σ and the bound. Measured: si 0.25,
            # glyc_ins_k 0.25, gamma 0.14, k_ins 0.15, act_insulin_sens 0.15 — i.e.
            # ≥13 % of headroom everywhere, so the tanh is not saturated on the
            # population it has to fit.
            margin = min(_SPANS[key] + lo, _SPANS[key] - hi)
            self.assertGreater(margin, 0.10, msg=f"{key} margin {margin:.3f}")
        # The hepatic gate's whole clip is reachable, which is stronger than ±2σ.
        hep = np.array(draws["glyc_ins_k"])
        self.assertLessEqual(float(np.abs(hep).max()), _SPANS["glyc_ins_k"])
        # act_insulin_sens is the one exception, and it is the clip's own tail: the
        # teacher clips at 0.1 = −1.10 (3.7σ) and the span floors at 0.142.
        act = np.array(draws["act_insulin_sens"])
        self.assertLess(float((act < -_SPANS["act_insulin_sens"]).mean()), 0.01)

    def test_every_decode_is_strictly_positive_for_two_hundred_embeddings(self) -> None:
        """Log space means positive by construction — including at the ‖e‖ = 8
        calibration clamp, where an additive decode of a rate constant would go
        negative and make insulin action a glucose SOURCE."""
        m = _model(1, perturb=0.5)
        for i, key in enumerate(_FAMILY):
            _odd_head(_heads(m)[key], seed=11 + i, std=4.0)
        emb = _embeddings(m, n=200, seed=3)
        with torch.no_grad():
            pp = m.metabolic.person_params(emb)
        pop = _centre(m)
        for key in (*_FAMILY, "gng_ins_k", "body_mass_kg"):
            v = pp[key]
            self.assertTrue(bool((v > 0).all()), msg=key)
            self.assertTrue(bool(torch.isfinite(v).all()), msg=key)
            if key in _SPANS:
                # tanh saturates, so the span is a HARD bound for any code (float32 slack)
                self.assertLessEqual(
                    float(v.max()), pop[key] * math.exp(_SPANS[key]) * (1 + 1e-5), msg=key)
                self.assertGreaterEqual(
                    float(v.min()), pop[key] * math.exp(-_SPANS[key]) * (1 - 1e-5), msg=key)


class TestZeroIsTheMedianPerson(unittest.TestCase):
    def test_a_zero_embedding_reproduces_the_population_scalars_exactly(self) -> None:
        """The cheapest regression test available: iter 109's behaviour IS the zero
        code's, so if this holds bit-exactly nothing about the default person moved.

        The heads stay zero-init (that is the claim — a ZERO-INIT head is a no-op) while
        every population scalar is moved off its init, so the identity is the decode's
        and not the init values'."""
        for seed in range(3):
            m = _model(seed)
            met = m.metabolic
            with torch.no_grad():
                for name in ("log_si", "log_glyc_ins_k", "log_gng_ins_k", "log_gamma",
                             "log_k_ins", "log_act_ins_sens"):
                    getattr(met, name).add_(0.3 * torch.randn(()))
            with torch.no_grad():
                pp = met.person_params(torch.zeros(4, _emb_width(m)))
                self.assertTrue(torch.equal(
                    pp["si"], (M._SI_MIN + M._SI_RANGE * torch.sigmoid(met.log_si)).expand(4)))
                for key, raw in (("glyc_ins_k", met.log_glyc_ins_k),
                                 ("gng_ins_k", met.log_gng_ins_k),
                                 ("gamma", met.log_gamma), ("k_ins", met.log_k_ins),
                                 ("act_insulin_sens", met.log_act_ins_sens)):
                    self.assertTrue(torch.equal(
                        pp[key], torch.nn.functional.softplus(raw).expand(4)), msg=key)
                self.assertTrue(torch.equal(pp["hep_ins_k_factor"], torch.ones(4)))
                self.assertEqual(float(pp["body_mass_kg"][0]), 70.0)

    def test_the_cold_start_is_the_teachers_median_patient(self) -> None:
        """And the centres themselves are the teacher's defaults, modulo the one unit
        ratio (the student's si is per NORMALIZED insulin unit, so 10× the teacher's)."""
        pop = _at_zero(_model(0))
        ref = PatientParams()
        self.assertAlmostEqual(pop["si"], 10.0 * ref.Si, places=9)
        self.assertAlmostEqual(pop["glyc_ins_k"], ref.glyc_ins_K, places=5)
        self.assertAlmostEqual(pop["gng_ins_k"], ref.gng_ins_K, places=4)
        self.assertAlmostEqual(pop["gamma"], ref.gamma, places=7)
        self.assertAlmostEqual(pop["k_ins"], ref.n, places=7)
        self.assertAlmostEqual(pop["act_insulin_sens"], ref.act_insulin_sens, places=6)
        self.assertAlmostEqual(pop["body_mass_kg"], ref.body_mass_kg, places=5)

    def test_the_mean_decoded_si_is_above_the_zero_code_si(self) -> None:
        """PLAN §1's whole argument, in one assertion. ``d(u) = c·exp(L·tanh u)`` is
        strictly convex about 0, so averaging a code population closed under negation
        gives ``c·cosh(L·tanh u) > c``. Measured at std 1.5, seed 0, 2×8192 codes:
        mean Si 0.006479 against Si(0) = 0.004000, i.e. 1.62×, with the median back at
        0.004121. The teacher's own population mean/median for Si is 1.152
        (= exp(σ²/2) = 1.133 plus sampling), and the student's ratio is set by the
        code spread, so only the SIGN and a floor are pinned here — an additive decode
        would give EXACTLY 1.000 for any zero-mean code, which is the point."""
        for seed, std in ((0, 1.5), (2, 0.8), (3, 3.0)):
            m = _model(seed)
            _odd_head(m.metabolic.insulin_sens_net, seed=seed, std=std)
            d = _emb_width(m)
            e = torch.randn(4096, d, generator=torch.Generator().manual_seed(seed + 100))
            with torch.no_grad():
                si = m.metabolic.person_params(torch.cat([e, -e], dim=0))["si"]
                si0 = float(m.metabolic.person_params(torch.zeros(1, d))["si"])
            mean, median = float(si.mean()), float(si.median())
            self.assertAlmostEqual(median, si0, delta=0.1 * si0, msg=f"seed {seed}")
            self.assertGreater(mean, si0, msg=f"seed {seed}: mean {mean:.6f} vs {si0:.6f}")
            self.assertGreater(mean / median, 1.001, msg=f"seed {seed}")
            # the additive decode this replaces, for contrast
            t = torch.tanh(torch.cat([e, -e]) @ torch.ones(d, 1))
            self.assertAlmostEqual(float((si0 + si0 * t).mean()), si0, delta=1e-6 * si0)


class TestMuscleIsNotLiver(unittest.TestCase):
    """Donga 2010 (clamp, 4 h vs 8 h sleep): whole-body −25 %, peripheral −29 %,
    hepatic essentially unchanged. PLAN §3 records that as the reason the carve is
    exactly here — one 'insulin sensitivity' cannot say it — and C3's sleep debt is
    the consumer. These tests pin that the two are independently movable."""

    @staticmethod
    def _fluxes(m: ModularPhysiologyNetwork, ins: float = 40.0, act: float = 0.0):
        """One fed point, insulin well above basal so both hepatic gates and
        insulin-dependent uptake are live, with ``xa`` held at a fixed value."""
        met = m.metabolic
        state = torch.zeros(1, len(M._TYPICALS))
        state[0, M._INSULIN_IDX] = (ins - 10.0) / 10.0
        state[0, M._GLUCOSE_IDX] = 1.0          # 125 mg/dL
        state[0, M._INSULIN_ACTION_IDX] = 2.0
        coupling = torch.zeros(1, M._N_COUPLING)
        external = torch.tensor([[act, 1.0]])
        emb = m.embedding_projections["metabolic"](torch.zeros(1, EMBEDDING_DIM))
        with torch.no_grad():
            return met.fluxes(state, coupling, external, emb,
                              compute_time_features(torch.tensor([600.0])))

    def test_peripheral_si_moves_disposal_and_leaves_the_hepatic_gates_alone(self) -> None:
        m = _model(0)
        base = self._fluxes(m)
        _set_factor(m, "si", 0.71)      # Donga's peripheral −29 %, i.e. log-ratio −0.342
        low = self._fluxes(m)
        self.assertAlmostEqual(float(low["uptake_id"]) / float(base["uptake_id"]),
                               0.71, places=5)
        for gate in ("g_ins_glyco", "g_ins_gng", "glyc_k", "gng_k"):
            self.assertEqual(float(low[gate]), float(base[gate]), msg=gate)

    def test_the_hepatic_factor_moves_both_gates_and_leaves_disposal_alone(self) -> None:
        m = _model(0)
        base = self._fluxes(m)
        _set_factor(m, "glyc_ins_k", 1.5)
        hep = self._fluxes(m)
        for key in ("glyc_k", "gng_k"):
            self.assertAlmostEqual(float(hep[key]) / float(base[key]), 1.5, places=5, msg=key)
        # A higher IC50 is LESS insulin suppression: hepatic insulin RESISTANCE, which
        # is why the teacher loads glyc_ins_K +0.40 on ir where Si is −0.70.
        self.assertGreater(float(hep["g_ins_glyco"]), float(base["g_ins_glyco"]))
        self.assertGreater(float(hep["glycogenolysis_plasma"]),
                           float(base["glycogenolysis_plasma"]))
        self.assertEqual(float(hep["uptake_id"]), float(base["uptake_id"]))
        self.assertEqual(float(hep["si"]), float(base["si"]))

    def test_the_gng_gate_has_no_head_of_its_own(self) -> None:
        """One liver, one insulin sensitivity. The teacher holds gng_ins_K at 80 for
        every patient, so a second decode here would be a head no ground truth can
        reach — the iter-109 glucagon-basal failure mode."""
        m = _model(0)
        self.assertFalse(hasattr(m.metabolic, "gng_ins_k_net"))
        ref = PatientParams()
        for seed in range(25):
            self.assertEqual(randomize_params(np.random.default_rng(seed)).gng_ins_K,
                             ref.gng_ins_K)
        # and the one factor is shared, so the two gates keep their teacher ratio
        emb = _embeddings(m, n=24, seed=5)
        _odd_head(m.metabolic.hepatic_ins_k_net, seed=9, std=2.0)
        with torch.no_grad():
            pp = m.metabolic.person_params(emb)
        ratio = pp["gng_ins_k"] / pp["glyc_ins_k"]
        torch.testing.assert_close(
            ratio, torch.full_like(ratio, M._GNG_INS_K_INIT / M._GLYC_INS_K_INIT),
            atol=1e-5, rtol=1e-5)


class TestSiDrivesTheMealResponse(unittest.TestCase):
    """The one that matters. Everything above is a property of the decoder; this is
    whether the decoded number reaches the physiology with the teacher's gain."""

    def test_glucose_iauc_falls_monotonically_in_decoded_si(self) -> None:
        m = _model(0)
        pop = _centre(m)["si"]
        got = []
        for factor in (0.35, 0.6, 1.0, 1.7, 2.9):
            _set_factor(m, "si", factor)
            gi, ii = _meal_iauc(m)
            got.append((pop * factor, gi, ii))
        for (si_a, g_a, i_a), (si_b, g_b, i_b) in zip(got, got[1:]):
            self.assertLess(g_b, g_a, msg=f"glucose iAUC rose from Si {si_a:.5f} to {si_b:.5f}")
            self.assertLess(i_b, i_a, msg=f"insulin iAUC rose from Si {si_a:.5f} to {si_b:.5f}")

    def test_the_decile_ratio_is_the_teachers(self) -> None:
        """At the teacher's own Si deciles (2.158e-4 and 7.534e-4 over 150 draws, a
        3.49× spread), the student realizes 4057 vs 2294 = 1.77× on glucose iAUC and
        2649 vs 1558 = 1.70× on insulin (median 3152 / 2095). The teacher's own
        PURE-Si sweep — the same comparison with every other parameter at the median
        person — is 1.81× and 1.89×; its ACROSS-PATIENT decile contrast, where Ib and
        Gb co-vary with Si, is 1.90× and 2.17×. So the student's sensitivity to its
        own Si is the teacher's to within 2 % on glucose, on an untrained model, which
        is the claim: Si now reaches the meal response through mechanism and not
        through meal amplitude."""
        m = _model(0)
        _set_factor(m, "si", 10.0 * _TEACHER_SI_P10 / _centre(m)["si"])
        lo_g, lo_i = _meal_iauc(m)
        _set_factor(m, "si", 10.0 * _TEACHER_SI_P90 / _centre(m)["si"])
        hi_g, hi_i = _meal_iauc(m)
        self.assertAlmostEqual(lo_g / hi_g, 1.77, delta=0.25, msg=f"{lo_g:.0f}/{hi_g:.0f}")
        self.assertAlmostEqual(lo_i / hi_i, 1.70, delta=0.25, msg=f"{lo_i:.0f}/{hi_i:.0f}")


class TestActivitySensitisesInsulinAction(unittest.TestCase):
    def test_the_gain_is_the_teachers_form_and_exactly_one_at_rest(self) -> None:
        m = _model(0, perturb=0.3)
        met = m.metabolic
        emb = m.embedding_projections["metabolic"](torch.zeros(3, EMBEDDING_DIM))
        with torch.no_grad():
            const = met.constants(emb)
            for act in (0.0, 0.35, 1.0):
                external = torch.tensor([[act, 1.0]]).expand(3, 2)
                d = met.drives(external, torch.zeros(3, M._N_COUPLING),
                               compute_time_features(torch.full((3,), 600.0)), const)
                torch.testing.assert_close(
                    d["ins_sens_act_gain"], 1.0 + const["act_ins_sens"] * act,
                    atol=1e-7, rtol=1e-7)
            # a malformed negative activity cannot invert insulin action
            d = met.drives(torch.tensor([[-2.0, 1.0]]).expand(3, 2),
                           torch.zeros(3, M._N_COUPLING),
                           compute_time_features(torch.full((3,), 600.0)), const)
            self.assertTrue(bool((d["ins_sens_act_gain"] == 1.0).all()))

    def test_the_gain_enters_the_remote_insulin_lag_not_the_uptake_site(self) -> None:
        """The teacher's ``dX = −p2·X + Si_eff·p2·(I − Ib)``, divided by Si. Putting it
        here is what makes the student's ``insulin_action`` column the quantity the
        teacher writes on its tape (``X/(Si·10)``) and gives the sensitisation the
        teacher's τ = 1/p2 ramp instead of a step at the start of a bout."""
        m = _model(0)
        met = m.metabolic
        state = torch.zeros(1, len(M._TYPICALS))
        state[0, M._INSULIN_IDX] = 3.0        # 40 µU/mL, well above basal
        state[0, M._INSULIN_ACTION_IDX] = 0.0
        coupling = torch.zeros(1, M._N_COUPLING)
        emb = m.embedding_projections["metabolic"](torch.zeros(1, EMBEDDING_DIM))
        tf = compute_time_features(torch.tensor([600.0]))
        with torch.no_grad():
            rest = met.fluxes(state, coupling, torch.tensor([[0.0, 1.0]]), emb, tf)
            bout = met.fluxes(state, coupling, torch.tensor([[1.0, 1.0]]), emb, tf)
        # the DRIVE on remote insulin carries the gain exactly
        self.assertGreater(float(bout["xa_rate"]), float(rest["xa_rate"]))
        gain = 1.0 + float(rest["act_ins_sens"])
        self.assertAlmostEqual(float(bout["xa_rate"]) / float(rest["xa_rate"]), gain, places=5)
        self.assertAlmostEqual(float(bout["xa_drive"]) / float(rest["xa_drive"]), gain, places=5)
        # ...and the uptake site does not: at a GIVEN xa, insulin-dependent disposal is
        # bit-identical with and without the bout, which is where the teacher puts it.
        state[0, M._INSULIN_ACTION_IDX] = 2.0
        with torch.no_grad():
            rest = met.fluxes(state, coupling, torch.tensor([[0.0, 1.0]]), emb, tf)
            bout = met.fluxes(state, coupling, torch.tensor([[1.0, 1.0]]), emb, tf)
        self.assertGreater(float(rest["uptake_id"]), 0.0)
        self.assertEqual(float(bout["uptake_id"]), float(rest["uptake_id"]))

    def test_a_bout_raises_insulin_dependent_disposal_over_a_rollout(self) -> None:
        """The term is live end to end, isolated from the insulin-INDEPENDENT exercise
        uptake that the same activity also drives: run the identical meal and bout
        twice, once with the sensitivity at the teacher's 0.3 and once switched off
        (the iter-109 model). Measured on a fresh model at activity 0.5, remote insulin
        peaks 1.25× higher with the gain on and glucose iAUC falls from 2749 to 2630
        (−4.3 %); at activity 1.0 it is 2688 vs 2591. Without the term a bout could not
        change how a meal is handled through insulin at all."""
        m = _model(0)
        on_g, _ = _meal_iauc(m, activity=0.5)
        on_xa = self._peak_insulin_action(m, activity=0.5)
        with torch.no_grad():                   # softplus(−20) ≈ 2e-9: sensitisation off
            m.metabolic.log_act_ins_sens.fill_(-20.0)
        off_g, _ = _meal_iauc(m, activity=0.5)
        off_xa = self._peak_insulin_action(m, activity=0.5)
        self.assertLess(on_g, off_g, msg=f"iAUC {on_g:.0f} vs {off_g:.0f}")
        self.assertGreater(on_xa, off_xa * 1.05, msg=f"xa peak {on_xa:.4f} vs {off_xa:.4f}")

    @staticmethod
    def _peak_insulin_action(m: ModularPhysiologyNetwork, activity: float) -> float:
        """``full_body.py`` records ``X/(params.Si·10)`` in the insulin_action column,
        which is the lagged ``(1 + act_insulin_sens·act)·(I − Ib)/10``, so the distilled
        target has the gain in it and the student's own column must too."""
        n = 120
        with torch.no_grad():
            tr = integrate(
                m, _CENTER.clone(), torch.zeros(EMBEDDING_DIM), n,
                start_time_minutes=8 * 60.0,
                meals=[MealEvent(time=5.0, carbs=75.0, fats=5.0, proteins=10.0)],
                sleep_wake=torch.ones(n), activity=torch.full((n,), activity))
        return float(tr[:, MI["insulin_action"]].max())


class TestSupervision(unittest.TestCase):
    """Five new per-person heads without supervision is the disease PLAN is about."""

    _PATIENT_SEEDS = (1, 7, 23, 44)

    def _targets(self) -> dict[int, dict[str, float]]:
        out = {}
        for pid, seed in enumerate(self._PATIENT_SEEDS):
            p = randomize_params(np.random.default_rng(seed))
            out[pid] = {k: float(getattr(p, f)) for k, f in _TEACHER_FIELD.items()}
        return out

    @staticmethod
    def _fit(m, n_patients: int, steps: int = 400, lr: float = 0.03, **kw):
        emb = torch.nn.Embedding(max(n_patients, 1), EMBEDDING_DIM)
        torch.nn.init.normal_(emb.weight, std=0.3)
        params = list(m.parameters()) + list(emb.parameters())
        sig = SetpointSupervisionSignal(weight=WeightSchedule(1.0), **kw)
        opt = torch.optim.Adam(params, lr=lr)
        result = None
        for _ in range(steps):
            opt.zero_grad()
            ctx = SignalContext(
                epoch=0, total_epochs=1, rng=np.random.default_rng(0),
                device=torch.device("cpu"), optimizer=opt, params=params,
                grad_clip=10.0)
            result = sig.compute(m, emb, ctx)
            opt.step()
        return emb, result

    def test_without_param_targets_it_is_a_no_op(self) -> None:
        """Safe before the trainer passes them: no loss, no gradient, no decode moved."""
        m = _model(0)
        before = {n: p.detach().clone() for n, p in m.metabolic.named_parameters()}
        emb, result = self._fit(m, 1, steps=3)
        self.assertEqual(result.loss_sum, 0.0)
        self.assertEqual(result.n_units, 0)
        self.assertNotIn("param_loss", result.sub_metrics)
        for n, p in m.metabolic.named_parameters():
            self.assertTrue(torch.equal(p.detach(), before[n]), msg=n)

    def test_marker_only_targets_do_not_reach_the_family(self) -> None:
        """``targets`` is markers and ``param_targets`` is parameters: supervising Gb
        must not move Si, which is the reason they are two dicts and not one."""
        m = _model(0)
        before = _at_zero(m)
        self._fit(m, 1, steps=40, targets={0: {"glucose": 120.0}})
        after = _at_zero(m)
        for key in _FAMILY:
            self.assertAlmostEqual(after[key], before[key], places=7, msg=key)

    def test_it_fits_the_teachers_own_draws_to_convergence(self) -> None:
        """Four sampled teacher patients, 400 Adam steps. Residuals at convergence are
        ≤0.03 % in log-ratio on every quantity and every patient (measured: worst
        |log(pred/target)| 3.4e-4, i.e. the decode lands on the teacher's number),
        including ``si``, whose target is in the teacher's per-µU/mL frame and whose
        decode is 10× that — the log-ratio-to-default frame absorbs the unit."""
        m = _model(0)
        targets = self._targets()
        emb, result = self._fit(m, len(targets), steps=400, param_targets=targets)
        self.assertLess(result.sub_metrics["param_loss"], 1e-5)
        with torch.no_grad():
            pp = m.metabolic.person_params(m.embedding_projections["metabolic"](emb.weight))
        for pid, want in targets.items():
            for key, raw in want.items():
                ratio = PERSON_PARAM_CENTERS[key] / _TEACHER_DEFAULTS[key]
                got = float(pp[key][pid])
                self.assertAlmostEqual(
                    math.log(got / (raw * ratio)), 0.0, delta=0.01,
                    msg=f"pid {pid} {key}: {got:.6g} vs {raw * ratio:.6g}")

    def test_body_mass_is_supervised_too(self) -> None:
        """It had a head and NO supervision, and it multiplies meal amplitude through
        V_G exactly as the deleted Ra gain did — mass is why the pair was flat."""
        m = _model(0)
        targets = {0: {"body_mass_kg": 96.0}, 1: {"body_mass_kg": 58.0}}
        emb, _ = self._fit(m, len(targets), steps=300, param_targets=targets)
        with torch.no_grad():
            got = m.metabolic.body_mass_kg(
                m.embedding_projections["metabolic"](emb.weight))
        self.assertAlmostEqual(float(got[0]), 96.0, delta=0.5)
        self.assertAlmostEqual(float(got[1]), 58.0, delta=0.5)

    def test_the_frame_is_the_log_ratio_to_each_sides_own_default(self) -> None:
        """The frame's one piece of real content is the unit ratio: every other factor
        cancels, because log(pred/c_student) − log(target/c_teacher) = log(pred/target)
        − log(c_student/c_teacher). ``si`` is the only entry where that is not 1."""
        for key, field in _TEACHER_FIELD.items():
            ratio = PERSON_PARAM_CENTERS[key] / _TEACHER_DEFAULTS[key]
            self.assertAlmostEqual(_TEACHER_DEFAULTS[key],
                                   float(getattr(PatientParams(), field)), places=12)
            self.assertAlmostEqual(ratio, 10.0 if key == "si" else 1.0, places=9)

    def test_a_two_fold_miss_costs_the_same_on_every_parameter(self) -> None:
        """Why log-ratio and not σ-units: the same 2× error has to cost the same
        everywhere, or the loss weights the NARROWEST quantity hardest (a 2× miss is
        1.9 in Si's σ-units and 7.7 in glyc_ins_K's — backwards, since Si is the axis
        that carries the between-person iAUC)."""
        m = _model(0)
        losses = []
        for key in _FAMILY:
            targets = {0: {key: 2.0 * _TEACHER_DEFAULTS[key]}}
            _, result = self._fit(m, 1, steps=1, lr=0.0, param_targets=targets)
            losses.append(result.sub_metrics["param_loss"])
            self.assertAlmostEqual(result.sub_metrics[f"{key}_logmae"], math.log(2.0),
                                   places=5, msg=key)
        for got in losses:
            self.assertAlmostEqual(got, math.log(2.0) ** 2, places=5)

    def test_the_teachers_episode_dict_is_exactly_what_this_signal_expects(self) -> None:
        """The contract with ``knowledge/full_body.py`` and ``training/trajectory_signal.py``:
        ``rec["patient_params"]`` goes into ``param_targets`` unchanged, so a renamed or
        dropped key has to fail HERE rather than silently drop the supervision that five
        per-person heads depend on. One 1-day episode costs 0.2 s."""
        ep = FullBody(n_days=1).generate_episodes(1, np.random.default_rng(0))[0]
        self.assertEqual(set(ep.patient_params), set(_TEACHER_FIELD))
        _, result = self._fit(_model(0), 1, steps=2, param_targets={0: ep.patient_params})
        for key in _TEACHER_FIELD:
            self.assertIn(f"{key}_logmae", result.sub_metrics)

    def test_the_trainers_own_filter_builds_a_NON_EMPTY_dict(self) -> None:
        """The join the test above does not cross: dataset RECORD -> ``param_targets``.

        ``train.py`` builds the dict with
        ``{int(r["patient_id"]): r["patient_params"] for r in dataset
           if not r.get("is_default") and r.get("patient_params")}``.
        Every failure mode of that comprehension is SILENT -- a renamed record key, a
        ``None`` the generator forgot to fill, an ``is_default`` flag that accidentally
        covers everything -- and all of them produce an empty dict, which
        ``SetpointSupervisionSignal`` treats as "no parameter supervision requested" and
        skips without complaint. Five per-person heads then train free, which is the
        iter-109 failure B1/A4 exist to prevent.

        This is also the one code path 749 passing tests do not reach: the bug fixed in
        this same session (``rollout_signal.py`` rebuilding a deleted constant) lived in
        trainer-only code and was found by reading, not by a test. So this crosses the
        join against REAL records from ``generate_trajectory_dataset``, not hand-built
        dicts, and asserts the dict is non-empty rather than merely well-formed.
        """
        from pulse.training.trajectory_signal import generate_trajectory_dataset

        dataset = generate_trajectory_dataset(
            n_patients=2, seed=0, n_days=1,
            weight_by_name={"full_body": 1.0}, n_default_patients=1,
        )
        param_targets = {
            int(r["patient_id"]): r["patient_params"] for r in dataset
            if not r.get("is_default") and r.get("patient_params")
        }
        setpoint_targets = {
            int(r["patient_id"]): r["setpoints"] for r in dataset
            if not r.get("is_default") and r.get("setpoints")
        }
        self.assertEqual(len(param_targets), 2, "the trainer would supervise NO parameters")
        self.assertEqual(len(setpoint_targets), 2, "the trainer would supervise NO setpoints")
        # The default patient is excluded by design (it IS PatientParams(), so the
        # log-ratio target is zero and the zero embedding already decodes there).
        self.assertTrue(any(r.get("is_default") for r in dataset))
        for row in param_targets.values():
            self.assertEqual(set(row), set(_TEACHER_FIELD))
        # A4: and all twelve markers, not the original six.
        for row in setpoint_targets.values():
            self.assertEqual(len(row), 12, sorted(row))
        # The supervision actually lands: every parameter reports its own residual.
        _, result = self._fit(_model(0), 1, steps=2, param_targets={0: param_targets[0]})
        for key in _TEACHER_FIELD:
            self.assertIn(f"{key}_logmae", result.sub_metrics)

    def test_unknown_teacher_keys_are_ignored_rather_than_fatal(self) -> None:
        """The teacher may record a parameter the student has no decode for (B3's
        second-phase potentiator is the next one); that must not kill a run."""
        m = _model(0)
        targets = {0: {"si": 4e-4, "k_second_phase": 0.2}}
        _, result = self._fit(m, 1, steps=2, param_targets=targets)
        self.assertIn("si_logmae", result.sub_metrics)
        self.assertNotIn("k_second_phase_logmae", result.sub_metrics)


class TestNothingIsStructurallyDead(unittest.TestCase):
    def test_every_new_parameter_takes_gradient_from_a_rollout(self) -> None:
        """A new head with no gradient path is exactly what B1 exists to stop. The
        rollout has a meal (so GSIR and the hepatic gates are live) and a bout (so the
        sensitisation is), which is what each of the five needs to be reachable."""
        m = _model(0, perturb=0.2)
        m.train()
        n = 90
        traj = integrate(
            m, _CENTER.clone(), torch.zeros(EMBEDDING_DIM), n, start_time_minutes=8 * 60.0,
            meals=[MealEvent(time=5.0, carbs=75.0, fats=5.0, proteins=10.0)],
            sleep_wake=torch.ones(n), activity=torch.full((n,), 0.5))
        cols = [MI[k] for k in ("glucose", "insulin", "insulin_action", "liver_glycogen")]
        traj[:, cols].pow(2).sum().backward()
        names = ["log_act_ins_sens"] + [
            f"{h}.{i}.{w}" for h in ("insulin_sens_net", "hepatic_ins_k_net",
                                     "beta_cell_gain_net", "insulin_clearance_net",
                                     "act_insulin_sens_net")
            for i in ("0", "2") for w in ("weight", "bias")]
        params = dict(m.metabolic.named_parameters())
        for name in names:
            self.assertIn(name, params)
            p = params[name]
            self.assertIsNotNone(p.grad, msg=name)
            self.assertTrue(bool(torch.isfinite(p.grad).all()), msg=name)
            self.assertGreater(float(p.grad.abs().max()), 0.0, msg=name)

    def test_the_parameter_count_grew_by_the_five_heads_and_one_scalar(self) -> None:
        """+1,326: five zero-init heads (265 each at the default widths — Linear(20,12)
        + Linear(12,1)) plus the activity sensitivity's population scalar. Iter 110
        removed ``log_alpha_gn``, so the module total is 20,813 rather than the
        20,814 counted before that constant. Only the module's own total is
        asserted, because the model's depends on every other module."""
        m = ModularPhysiologyNetwork()
        heads = ("insulin_sens_net", "hepatic_ins_k_net", "beta_cell_gain_net",
                 "insulin_clearance_net", "act_insulin_sens_net")
        for name in heads:
            self.assertEqual(
                sum(p.numel() for p in getattr(m.metabolic, name).parameters()), 265,
                msg=name)
        self.assertEqual(m.metabolic.log_act_ins_sens.numel(), 1)
        self.assertEqual(sum(p.numel() for p in m.metabolic.parameters()), 20813)
        self.assertFalse(hasattr(m.metabolic, "log_alpha_gn"))


if __name__ == "__main__":
    unittest.main()
