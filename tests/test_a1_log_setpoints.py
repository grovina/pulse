"""PLAN A1 — HR, DBP and RR setpoints decode in log space; SpO2 stays additive.

The teacher draws every positive resting vital lognormally (``vary``: val·exp(σ·z); HR0 σ0.22,
SBP0 0.14, DBP0 0.15, RR0 0.15, HRV0 0.4), so ``PatientParams()`` is the population MEDIAN and the
population mean sits above it. Measured over 20 000 ``randomize_params`` draws (seed 0), mean/median
is 1.0232 for HR0, 1.0112 SBP0, 1.0105 DBP0, 1.0093 RR0, 1.0850 HRV0; SpO2_0 is drawn additively and
has none (mean 97.992, median 98.002). Through iter 109 the student decoded HR and DBP as
``center + 3z·tanh(head)`` (iter 89) and RR as ``15 + 2z·tanh(head)``: for a zero-mean embedding
population that gives E[setpoint] = center EXACTLY, so population mean = median, which contradicts
the teacher and cannot be right at both points. ``center·exp(L·tanh(head))`` gets both by Jensen
with no loss term (PLAN.md §1, item A1). These tests pin:

  * a fresh model rests at NORM_CENTER for all six setpoints;
  * SBP > DBP and every setpoint > 0 for every embedding, the ‖e‖ = 8 calibration clamp included
    (iter 97's by-construction invariant, now with HR and DBP log-decoded too);
  * the reachable span of each log-decoded setpoint still covers the clinical range the additive
    bounds documented, and equals the span the modules' comments state;
  * the point of the change — E[HR_sp] over N(0, I) embeddings is strictly ABOVE HR_sp(0);
  * teacher and student agree on the sign of (mean/median − 1) for each of the six markers;
  * the decoded setpoints are what a rest rollout settles on, and the unchanged
    SetpointSupervisionSignal can fit targets the additive bounds could not reach.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch
import torch.nn as nn

from pulse.knowledge.full_body import PatientParams, randomize_params
from pulse.model import ModularPhysiologyNetwork, integrate
from pulse.modules import cardiovascular as C
from pulse.modules import respiratory as R
from pulse.types import EMBEDDING_DIM, MARKER_INDEX as MI, NORM_CENTER, NORM_SCALE

_SIX = ("hr", "hrv", "sbp", "dbp", "rr", "spo2")
_CENTER = {k: float(NORM_CENTER[MI[k]]) for k in _SIX}
# CalibrationSettings.max_norm (pulse/calibration.py): the hard clamp on a calibrated embedding.
_CALIBRATION_CLAMP = 8.0
# Log half-widths L of the log-decoded setpoints.
_LOG_L = {"hr": C._HR_LOG_SP_MAX, "hrv": C._HRV_LOG_SP_MAX, "dbp": C._DBP_LOG_SP_MAX, "rr": R._RR_LOG_SP_MAX}


def _model() -> ModularPhysiologyNetwork:
    return ModularPhysiologyNetwork(
        metabolic_hidden=16, appetite_hidden=16, stress_hidden=16, cardiovascular_hidden=16,
        thermoreg_hidden=16, respiratory_hidden=16, gut_hidden=16, hepatobiliary_hidden=16)


def _six(m: ModularPhysiologyNetwork, e: torch.Tensor) -> dict[str, torch.Tensor]:
    """The six decoded resting setpoints for ``e[..., EMBEDDING_DIM]``, by marker id."""
    cvs = m.cardiovascular.setpoints_raw(m.embedding_projections["cardiovascular"](e))
    rsp = m.respiratory.setpoints_raw(m.embedding_projections["respiratory"](e))
    return {"hr": cvs[..., 0], "hrv": cvs[..., 1], "sbp": cvs[..., 2], "dbp": cvs[..., 3],
            "rr": rsp[..., 0], "spo2": rsp[..., 1]}


def _own_the_heads(m: ModularPhysiologyNetwork, seed: int, spread: float, *, odd: bool) -> None:
    """Overwrite the embedding→setpoint maps of both modules from a LOCAL generator, so the numbers
    quoted in the tests below do not move when another module's init consumes the global RNG.
    Every layer gets unit-norm rows (a unit-variance pre-activation for N(0, I) input); the output
    layer is scaled by ``spread``. ``odd``: all biases zero, which makes the map an odd function of
    the embedding — with an antithetic sample {e, −e} the mean of tanh(head) is then exactly 0."""
    g = torch.Generator().manual_seed(seed)

    def fill(lin: nn.Linear, gain: float) -> None:
        w = torch.randn(lin.weight.shape, generator=g)
        lin.weight.copy_(gain * w / w.norm(dim=1, keepdim=True))
        lin.bias.copy_(torch.zeros_like(lin.bias) if odd else 0.5 * torch.randn(lin.bias.shape, generator=g))

    with torch.no_grad():
        for name in ("cardiovascular", "respiratory"):
            fill(m.embedding_projections[name], 1.0)
            net = getattr(m, name).setpoint_net
            fill(net[0], 1.0)
            fill(net[-1], spread)


def _saturated(sign: float) -> dict[str, torch.Tensor]:
    """Setpoints with every head output driven to tanh = ±1: the supremum / infimum of the decode."""
    m = _model()
    with torch.no_grad():
        for mod in (m.cardiovascular, m.respiratory):
            mod.setpoint_net[-1].weight.zero_()
            mod.setpoint_net[-1].bias.fill_(sign * 20.0)
        return _six(m, torch.zeros(EMBEDDING_DIM))


def _odd_model_and_codes() -> tuple[ModularPhysiologyNetwork, torch.Tensor]:
    """The A1 probe: a head odd in the embedding, spread 0.7 so that sd(log HR_sp) = 0.2167 (the
    teacher's σ for HR0 is 0.22), and 10 000 N(0, I) embeddings together with their negatives."""
    m = _model()
    _own_the_heads(m, seed=0, spread=0.7, odd=True)
    g = torch.Generator().manual_seed(1234)
    half = torch.randn(10_000, EMBEDDING_DIM, generator=g)
    return m, torch.cat([half, -half])


class TestZeroEmbeddingIsTheDefaultPerson(unittest.TestCase):
    def test_zero_embedding_decodes_to_exactly_norm_center_for_all_six_setpoints(self) -> None:
        m = _model()
        with torch.no_grad():
            s = _six(m, torch.zeros(EMBEDDING_DIM))
            z = m.cardiovascular.setpoints_z(m.embedding_projections["cardiovascular"](torch.zeros(EMBEDDING_DIM)))
        for k in _SIX:
            self.assertEqual(float(s[k]), _CENTER[k], msg=k)   # exact: tanh(0) = 0 and exp(0) = 1
        torch.testing.assert_close(z, torch.zeros(4), atol=0.0, rtol=0.0)

    def test_norm_center_is_the_teachers_median_person(self) -> None:
        """What makes zero the median person (PLAN §2): NORM_CENTER is PatientParams()."""
        p = PatientParams()
        self.assertEqual(
            (p.HR0, p.HRV0, p.SBP0, p.DBP0, p.RR0, p.SpO2_0),
            (_CENTER["hr"], _CENTER["hrv"], _CENTER["sbp"], _CENTER["dbp"], _CENTER["rr"], _CENTER["spo2"]))

    def test_every_embedding_decodes_to_norm_center_on_a_fresh_model(self) -> None:
        """The head's output layer is zero-init, so tanh(head) = 0 for ANY embedding, not only zero."""
        m = _model()
        torch.manual_seed(0)
        e = torch.randn(64, EMBEDDING_DIM)
        e = _CALIBRATION_CLAMP * e / e.norm(dim=-1, keepdim=True)
        with torch.no_grad():
            s = _six(m, e)
        for k in _SIX:
            torch.testing.assert_close(s[k], torch.full((64,), _CENTER[k]), atol=0.0, rtol=0.0, msg=k)


class TestOrderingAndPositivityByConstruction(unittest.TestCase):
    """SBP = DBP + PP with PP > 0 (iter 97) must survive HR and DBP going log-space, and every
    setpoint must stay strictly positive for every embedding the calibration can reach."""

    def setUp(self) -> None:
        self.m = _model()
        _own_the_heads(self.m, seed=1, spread=3.0, odd=False)   # saturating and biased: the hard case

    def test_sbp_exceeds_dbp_and_every_setpoint_is_positive_for_200_embeddings(self) -> None:
        g = torch.Generator().manual_seed(7)
        gauss = torch.randn(100, EMBEDDING_DIM, generator=g)
        at_clamp = torch.randn(100, EMBEDDING_DIM, generator=g)
        at_clamp = _CALIBRATION_CLAMP * at_clamp / at_clamp.norm(dim=-1, keepdim=True)
        e = torch.cat([gauss, at_clamp])
        self.assertEqual(e.shape[0], 200)
        self.assertAlmostEqual(float(at_clamp.norm(dim=-1).max()), _CALIBRATION_CLAMP, places=4)
        with torch.no_grad():
            s = _six(self.m, e)
        pp = s["sbp"] - s["dbp"]
        self.assertTrue(bool((s["sbp"] > s["dbp"]).all()))
        self.assertTrue(bool((pp >= 40.0 * math.exp(-C._PP_LOG_SP_MAX) - 1e-3).all()), msg=f"min PP {float(pp.min()):.3f}")
        for k in _SIX:
            self.assertTrue(bool(torch.isfinite(s[k]).all()), msg=k)
            self.assertTrue(bool((s[k] > 0).all()), msg=f"{k} min {float(s[k].min()):.4f}")
        # ...and none of it leaves the span the decode documents.
        for k, L in _LOG_L.items():
            self.assertTrue(bool((s[k] >= _CENTER[k] * math.exp(-L) * (1 - 1e-5)).all()), msg=k)
            self.assertTrue(bool((s[k] <= _CENTER[k] * math.exp(L) * (1 + 1e-5)).all()), msg=k)

    def test_gradient_descent_at_the_clamp_cannot_invert_or_zero_a_setpoint(self) -> None:
        """The iter-97 reviewer's probe, at the 8.0 calibration clamp instead of 3.0, and aimed at
        every positive setpoint: descend the embedding to MINIMIZE the quantity under ‖e‖ ≤ 8."""
        floors = {
            "pp": 40.0 * math.exp(-C._PP_LOG_SP_MAX),
            "hr": _CENTER["hr"] * math.exp(-C._HR_LOG_SP_MAX),
            "dbp": _CENTER["dbp"] * math.exp(-C._DBP_LOG_SP_MAX),
            "rr": _CENTER["rr"] * math.exp(-R._RR_LOG_SP_MAX),
        }
        objectives = {
            "pp": lambda s: s["sbp"] - s["dbp"], "hr": lambda s: s["hr"],
            "dbp": lambda s: s["dbp"], "rr": lambda s: s["rr"],
        }
        for name, objective in objectives.items():
            g = torch.Generator().manual_seed(3)
            e = torch.randn(EMBEDDING_DIM, generator=g).requires_grad_()
            opt = torch.optim.Adam([e], lr=0.1)
            for _ in range(150):
                loss = objective(_six(self.m, e))
                opt.zero_grad()
                loss.backward()
                opt.step()
                with torch.no_grad():
                    n = e.norm()
                    if n > _CALIBRATION_CLAMP:
                        e.mul_(_CALIBRATION_CLAMP / n)
            with torch.no_grad():
                worst = float(objective(_six(self.m, e)))
            self.assertGreater(worst, 0.0, msg=name)
            self.assertGreaterEqual(worst, floors[name] * (1 - 1e-4), msg=f"{name} reached {worst:.4f}")


class TestReachCoversTheDocumentedClinicalRanges(unittest.TestCase):
    def test_log_spans_cover_the_ranges_the_additive_bounds_documented(self) -> None:
        hi, lo = _saturated(+1.0), _saturated(-1.0)
        # (marker, low, high): HR 70±30 (benchmark 49-77), DBP 80±24 (benchmark 61-88) and RR 15±6
        # are the additive spans A1 replaced; HRV (RMSSD ~15-100 in adults) and pulse pressure
        # (20-80) are the iter-97 log spans, unchanged.
        ranges = {"hr": (40.0, 100.0), "dbp": (56.0, 104.0), "rr": (9.0, 21.0), "hrv": (15.0, 100.0)}
        for k, (low, high) in ranges.items():
            self.assertLessEqual(float(lo[k]), low, msg=f"{k} floor {float(lo[k]):.2f}")
            self.assertGreaterEqual(float(hi[k]), high, msg=f"{k} ceiling {float(hi[k]):.2f}")
        self.assertLessEqual(float(lo["sbp"] - lo["dbp"]), 20.0)
        self.assertGreaterEqual(float(hi["sbp"] - hi["dbp"]), 80.0)
        for k, (bench_lo, bench_hi) in {"hr": (49.0, 77.0), "dbp": (61.0, 88.0)}.items():
            self.assertLess(float(lo[k]), bench_lo)
            self.assertGreater(float(hi[k]), bench_hi)

    def test_spans_are_the_ones_the_comments_state(self) -> None:
        hi, lo = _saturated(+1.0), _saturated(-1.0)
        stated = {"hr": (40.0, 122.5), "dbp": (55.8, 114.7), "rr": (8.9, 25.2)}
        for k, (low, high) in stated.items():
            self.assertEqual(round(float(lo[k]), 1), low, msg=k)
            self.assertEqual(round(float(hi[k]), 1), high, msg=k)
            # the form: center·exp(±L), not a re-fit that happens to round to the same numbers
            self.assertAlmostEqual(float(hi[k]), _CENTER[k] * math.exp(_LOG_L[k]), delta=1e-3 * float(hi[k]))
            self.assertAlmostEqual(float(lo[k]), _CENTER[k] * math.exp(-_LOG_L[k]), delta=1e-3 * float(lo[k]))

    def test_spo2_stays_additive_and_under_its_ceiling(self) -> None:
        hi, lo = _saturated(+1.0), _saturated(-1.0)
        self.assertAlmostEqual(float(hi["spo2"]), 98.0 + 1.5, places=4)
        self.assertAlmostEqual(float(lo["spo2"]), 98.0 - 1.5, places=4)
        self.assertLess(float(hi["spo2"]), 100.0)


class TestPopulationMeanSitsAboveTheMedianPerson(unittest.TestCase):
    """The point of A1. Head: odd in the embedding (zero biases), weights from a local generator
    (``_odd_model_and_codes``). The sample is antithetic, so mean(tanh(head)) = 0 exactly and any
    gap between the sample mean and the zero-embedding value is the decoder family's, not
    sampling noise."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.m, cls.e = _odd_model_and_codes()
        with torch.no_grad():
            cls.s = _six(cls.m, cls.e)
            cls.s0 = _six(cls.m, torch.zeros(EMBEDDING_DIM))
            cls.o_cvs = cls.m.cardiovascular.setpoint_net(cls.m.embedding_projections["cardiovascular"](cls.e))
            cls.o_rsp = cls.m.respiratory.setpoint_net(cls.m.embedding_projections["respiratory"](cls.e))

    def test_zero_embedding_is_still_the_default_person_for_this_head(self) -> None:
        for k in _SIX:
            self.assertEqual(float(self.s0[k]), _CENTER[k], msg=k)

    def test_mean_decoded_hr_exceeds_the_zero_embedding_value(self) -> None:
        # Measured on this head (seed 0, spread 0.7, 20 000 antithetic N(0, I) codes):
        #   before A1, additive 70 + 30·tanh(head):  E[HR_sp] = 70.0000 bpm   (gap +0.0000; sd 11.61)
        #   after,     log 70·exp(0.56·tanh(head)):  E[HR_sp] = 71.6559 bpm   (gap +1.6559; sd 15.49)
        # The teacher's own shift over 20 000 randomize_params draws is +1.757 bpm (mean 71.757).
        hr0, hr = float(self.s0["hr"]), self.s["hr"]
        gap = float(hr.mean()) - hr0
        self.assertGreater(float(hr.std()), 8.0)    # the head has authority; a flat head would pass nothing
        self.assertGreater(gap, 0.0)
        # ...and it is the lognormal Jensen shift of the head's own log-spread: c·(exp(σ²/2) − 1)
        sigma2 = float(torch.log(hr / hr0).var())
        self.assertAlmostEqual(gap, hr0 * math.expm1(sigma2 / 2.0), delta=0.1 * gap)

    def test_the_pre_a1_additive_decode_of_the_same_head_has_no_shift(self) -> None:
        """Control: the retired formulas, applied to the very head outputs above."""
        hr_add = _CENTER["hr"] + 3.0 * float(NORM_SCALE[MI["hr"]]) * torch.tanh(self.o_cvs[:, 0])
        dbp_add = _CENTER["dbp"] + 3.0 * float(NORM_SCALE[MI["dbp"]]) * torch.tanh(self.o_cvs[:, 3])
        rr_add = _CENTER["rr"] + 2.0 * float(NORM_SCALE[MI["rr"]]) * torch.tanh(self.o_rsp[:, 0])
        self.assertAlmostEqual(float(hr_add.mean()), 70.0, delta=1e-3)
        self.assertAlmostEqual(float(dbp_add.mean()), 80.0, delta=1e-3)
        self.assertAlmostEqual(float(rr_add.mean()), 15.0, delta=1e-3)

    def test_dbp_rr_and_sbp_means_exceed_their_zero_embedding_values(self) -> None:
        # Same head. After A1: DBP 80.6723 (+0.672), RR 15.2651 (+0.265), SBP 122.3371 (+2.337, the
        # sum of two log-decoded terms). Before A1: DBP 80.0000, RR 15.0000, SBP 121.6648 (PP was
        # already log, DBP was not).
        for k in ("dbp", "rr"):
            z, x = float(self.s0[k]), self.s[k]
            gap = float(x.mean()) - z
            self.assertGreater(gap, 0.0, msg=k)
            sigma2 = float(torch.log(x / z).var())
            self.assertAlmostEqual(gap, z * math.expm1(sigma2 / 2.0), delta=0.1 * gap, msg=k)
        self.assertGreater(float(self.s["sbp"].mean()), float(self.s0["sbp"]) + 0.25)

    def test_spo2_has_no_shift_because_it_is_decoded_additively(self) -> None:
        self.assertAlmostEqual(float(self.s["spo2"].mean()), 98.0, delta=1e-4)


class TestDecoderFamilyMatchesTheTeacher(unittest.TestCase):
    """The principle, as a table: for each marker, the teacher's population has mean/median > 1
    exactly when the student's decode has E[setpoint]/setpoint(0) > 1."""

    def test_sign_of_mean_over_median_agrees_for_each_marker(self) -> None:
        rng = np.random.default_rng(0)
        draws = [randomize_params(rng) for _ in range(4000)]
        attr = {"hr": "HR0", "hrv": "HRV0", "sbp": "SBP0", "dbp": "DBP0", "rr": "RR0", "spo2": "SpO2_0"}
        teacher = {}
        for k, a in attr.items():
            v = np.array([getattr(p, a) for p in draws])
            teacher[k] = float(v.mean() / np.median(v))

        m, e = _odd_model_and_codes()
        with torch.no_grad():
            s = _six(m, e)
            s0 = _six(m, torch.zeros(EMBEDDING_DIM))
        student = {k: float(s[k].mean()) / float(s0[k]) for k in _SIX}

        # 4 000 draws, seed 0: teacher HR 1.0262, HRV 1.0873, SBP 1.0112, DBP 1.0128, RR 1.0080, SpO2 0.9999
        # student (same head): HR 1.0237, HRV 1.0549, SBP 1.0195, DBP 1.0084, RR 1.0177, SpO2 1.0000
        for k in ("hr", "hrv", "sbp", "dbp", "rr"):
            self.assertGreater(teacher[k], 1.0, msg=f"teacher {k}: lognormal draws put the mean above the median")
            self.assertGreater(student[k], 1.0 + 1e-4, msg=f"student {k}: the log decode must do the same")
        # SpO2_0 is drawn additively: symmetric about its median (97.992 vs 98.002 over 20 000 draws).
        self.assertAlmostEqual(teacher["spo2"], 1.0, delta=1e-3)
        self.assertAlmostEqual(student["spo2"], 1.0, delta=1e-5)


class TestDecodedSetpointsDriveTheDynamics(unittest.TestCase):
    def test_rest_rollout_settles_on_log_decoded_setpoints_beyond_the_old_additive_ceilings(self) -> None:
        """Head outputs pinned (zero final weights, set biases) so the target is known without
        trusting the decode: HR_sp 116 bpm and RR_sp 22 /min are above the additive ceilings
        (100 bpm, 21 /min) and DBP_sp 58 mmHg is near the floor. The MLP drivers are zero-init, so a
        rest rollout is pure restoring force and must end ON the decoded setpoints."""
        m = _model()
        with torch.no_grad():
            m.cardiovascular.setpoint_net[-1].weight.zero_()
            m.cardiovascular.setpoint_net[-1].bias.copy_(torch.tensor([1.5, 0.0, 0.0, -1.5]))
            m.respiratory.setpoint_net[-1].weight.zero_()
            m.respiratory.setpoint_net[-1].bias.copy_(torch.tensor([1.0, 0.0]))
        e = torch.zeros(EMBEDDING_DIM)
        with torch.no_grad():
            s = _six(m, e)
        hr_sp = 70.0 * math.exp(C._HR_LOG_SP_MAX * math.tanh(1.5))
        dbp_sp = 80.0 * math.exp(-C._DBP_LOG_SP_MAX * math.tanh(1.5))
        rr_sp = 15.0 * math.exp(R._RR_LOG_SP_MAX * math.tanh(1.0))
        self.assertGreater(hr_sp, 100.0)
        self.assertGreater(rr_sp, 21.0)
        self.assertAlmostEqual(float(s["hr"]), hr_sp, places=3)
        self.assertAlmostEqual(float(s["dbp"]), dbp_sp, places=3)
        self.assertAlmostEqual(float(s["rr"]), rr_sp, places=3)

        n = 240   # 4 h: the slowest restoring time constant is RR / SpO2 at 1/0.08 = 12.5 min
        with torch.no_grad():
            tr = integrate(m, torch.tensor(NORM_CENTER), e, n, start_time_minutes=480, meals=[],
                           sleep_wake=torch.ones(n), activity=torch.zeros(n))
        end = tr[-1]
        self.assertAlmostEqual(float(end[MI["hr"]]), hr_sp, delta=0.5)
        self.assertAlmostEqual(float(end[MI["dbp"]]), dbp_sp, delta=0.5)
        self.assertAlmostEqual(float(end[MI["sbp"]] - end[MI["dbp"]]), 40.0, delta=0.5)
        self.assertAlmostEqual(float(end[MI["rr"]]), rr_sp, delta=0.3)
        self.assertAlmostEqual(float(end[MI["spo2"]]), 98.0, delta=0.3)


class TestSupervisionFrameIsUnchanged(unittest.TestCase):
    """SetpointSupervisionSignal calls ``cvs.setpoints_z``; it needs no change as long as that is
    still ``(raw − NORM_CENTER) / NORM_SCALE`` in [hr, hrv, sbp, dbp] order."""

    def test_setpoints_z_is_the_linear_z_score_of_setpoints_raw(self) -> None:
        m = _model()
        _own_the_heads(m, seed=2, spread=1.5, odd=False)
        torch.manual_seed(0)
        e = m.embedding_projections["cardiovascular"](torch.randn(32, EMBEDDING_DIM))
        with torch.no_grad():
            raw = m.cardiovascular.setpoints_raw(e)
            z = m.cardiovascular.setpoints_z(e)
        idx = [MI["hr"], MI["hrv"], MI["sbp"], MI["dbp"]]
        center = torch.tensor([float(NORM_CENTER[i]) for i in idx])
        scale = torch.tensor([float(NORM_SCALE[i]) for i in idx])
        torch.testing.assert_close(z, (raw - center) / scale, atol=1e-6, rtol=1e-6)

    def test_respiratory_setpoints_z_is_the_same_hand_off(self) -> None:
        """RR is no longer ``MAX_Z·tanh(head)``, so a signal that supervises RR0 / SpO2_0 (PLAN A4)
        must read the z from the module, as it does for cardiovascular."""
        m = _model()
        _own_the_heads(m, seed=2, spread=1.5, odd=False)
        torch.manual_seed(0)
        e = m.embedding_projections["respiratory"](torch.randn(32, EMBEDDING_DIM))
        with torch.no_grad():
            raw = m.respiratory.setpoints_raw(e)
            z = m.respiratory.setpoints_z(e)
        center = torch.tensor([_CENTER["rr"], _CENTER["spo2"]])
        scale = torch.tensor([float(NORM_SCALE[MI["rr"]]), float(NORM_SCALE[MI["spo2"]])])
        torch.testing.assert_close(z, (raw - center) / scale, atol=1e-6, rtol=1e-6)
        fresh = _model()   # zero head ⇒ the median person ⇒ z = 0
        with torch.no_grad():
            z_fresh = fresh.respiratory.setpoints_z(fresh.embedding_projections["respiratory"](torch.zeros(EMBEDDING_DIM)))
        torch.testing.assert_close(z_fresh, torch.zeros(2), atol=0.0, rtol=0.0)

    def test_unchanged_signal_fits_targets_above_the_old_additive_ceilings(self) -> None:
        from pulse.training import SetpointSupervisionSignal, SignalContext, WeightSchedule
        torch.manual_seed(0)
        m = _model()
        emb = nn.Embedding(2, EMBEDDING_DIM)
        nn.init.normal_(emb.weight, std=0.3)
        params = list(m.parameters()) + list(emb.parameters())
        # patient 0 sits above the old HR / DBP ceilings (100 bpm, 104 mmHg), patient 1 at the floors
        targets = {0: {"hr": 112.0, "hrv": 28.0, "sbp": 150.0, "dbp": 106.0},
                   1: {"hr": 46.0, "hrv": 75.0, "sbp": 104.0, "dbp": 58.0}}
        sig = SetpointSupervisionSignal(weight=WeightSchedule(1.0), targets=targets)
        opt = torch.optim.Adam(params, lr=0.03)
        for _ in range(250):   # converged to the targets' second decimal by step 200
            opt.zero_grad()
            ctx = SignalContext(epoch=0, total_epochs=1, rng=np.random.default_rng(0), device=torch.device("cpu"),
                                optimizer=opt, params=params, grad_clip=10.0)
            sig.compute(m, emb, ctx)
            opt.step()
        with torch.no_grad():
            raw = m.cardiovascular.setpoints_raw(m.embedding_projections["cardiovascular"](emb.weight))
        for pid, t in targets.items():
            self.assertAlmostEqual(float(raw[pid, 0]), t["hr"], delta=1.5, msg=f"hr[{pid}]")
            self.assertAlmostEqual(float(raw[pid, 3]), t["dbp"], delta=1.5, msg=f"dbp[{pid}]")
            self.assertAlmostEqual(float(raw[pid, 2]), t["sbp"], delta=2.0, msg=f"sbp[{pid}]")
            self.assertAlmostEqual(float(raw[pid, 1]), t["hrv"], delta=2.0, msg=f"hrv[{pid}]")
        self.assertGreater(float(raw[0, 0]), 100.0)    # unreachable before A1
        self.assertGreater(float(raw[0, 3]), 104.0)


if __name__ == "__main__":
    unittest.main()
