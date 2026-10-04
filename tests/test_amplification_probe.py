"""scripts/amplification_probe.py: the HARNESS is sound (not: the model is good).

PLAN D5 puts a number on the PRD criterion "coupling amplifies information":
calibrate on marker set A, score a disjoint held-out set B against the baseline of
not having calibrated. A number is only worth reading if the thing producing it
cannot flatter itself, so this pins the machinery on small models, with no
checkpoint and no claim about physiology:

* the held-out set is disjoint from the calibrated set, and calibration is never
  shown a held-out marker;
* on a synthetic patient the model generated itself, calibrating on a marker and
  scoring that SAME marker beats the prior mean (if this fails the harness is
  broken, and every held-out number is noise);
* skill is exactly 0 when the calibrated embedding is the prior mean, whether the
  window was rejected or the answer just came back equal -- and negative when the
  embedding is worse, which the report must show rather than hide;
* flat markers (nothing for calibration to move) are counted and left out of the
  headline, the trained prior flows through the checkpoint path, and the real-
  episode branch scores the markers it did not calibrate on.
"""

from __future__ import annotations

import importlib.util
import io
import json
import math
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from pulse.benchmark import BenchmarkEpisode
from pulse.calibration import CalibrationSettings, MeasurementPoint
from pulse.modules.gut import MealEvent
from pulse.types import EMBEDDING_DIM, MARKER_INDEX, NORM_CENTER

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "amplification_probe.py"
_spec = importlib.util.spec_from_file_location("amplification_probe", _SCRIPT)
amp = importlib.util.module_from_spec(_spec)
sys.modules["amplification_probe"] = amp
_spec.loader.exec_module(amp)

SMALL = dict(
    metabolic_hidden=16, appetite_hidden=16, stress_hidden=16, cardiovascular_hidden=16,
    thermoreg_hidden=16, respiratory_hidden=16, gut_hidden=16, hepatobiliary_hidden=16,
)
# 8 check-ins (the hold-out takes the last 2), a meal inside the window, 20 min of "ahead".
PROTO = amp.Protocol(duration_min=60, cal_end_min=40, obs_every_min=5, score_every_min=5, meal_at_min=5.0)


def _model(authority: float = 0.3, seed: int = 0):
    return amp.fresh_model(seed, authority, **SMALL)


def _patient(model, scale: float = 0.5, seed: int = 0, index: int = 0):
    e = amp.draw_embeddings(model, index + 1, seed=seed, emb_sd=scale)[index]
    patient, ok = amp.make_patient(model, index, e, PROTO, "person")
    assert ok
    return patient


def _score_patient(*args, channels: bool = False, **kwargs):
    """``score_patient`` without the channel decomposition (up to 8 extra rollouts a call):
    only TestChannels, and the end-to-end run, pay for it."""
    return amp.score_patient(*args, channels=channels, **kwargs)


@dataclass
class _Fake:
    """The four fields the probe reads from a CalibrationResult."""
    embedding: torch.Tensor
    accepted: bool
    reason: str = "stub"
    n_steps: int = 0


def _stub(embedding, accepted: bool, calls: list):
    def fake(model, observations, initial_state, meals, duration_min, **kw):
        calls.append({"markers": {o.marker_id for o in observations}, "times": {o.time for o in observations},
                      "duration": duration_min, "kw": kw})
        return _Fake(embedding() if callable(embedding) else embedding, accepted)
    return fake


class TestSkillConvention(unittest.TestCase):
    def test_zero_when_nothing_changed(self) -> None:
        self.assertEqual(amp.amplification_skill(2.5, 2.5), 0.0)
        # the gate's form is 0 here too, as long as the floor is not what binds
        self.assertEqual(amp.gate_skill(2.5, 2.5, sigma_obs=1.0), 0.0)

    def test_sign_and_scale(self) -> None:
        self.assertAlmostEqual(amp.amplification_skill(1.0, 4.0), 0.75)
        self.assertLess(amp.amplification_skill(6.0, 4.0), 0.0)          # a real, reportable result
        self.assertAlmostEqual(amp.amplification_skill(6.0, 4.0), -0.5)
        self.assertTrue(math.isnan(amp.amplification_skill(0.0, 0.0)))   # flat baseline: undefined

    def test_gate_form_is_floored_at_sigma_obs_and_the_raw_form_is_not(self) -> None:
        # benchmark.py: 1 - MAE / max(baseline, sigma_obs). With the baseline under the
        # noise floor the gate form hands out credit (+0.95) for an UNCHANGED prediction
        # (0.2 / 4); the raw form, the only one that is exactly 0 for "no change", does not.
        self.assertAlmostEqual(amp.gate_skill(0.2, 0.2, sigma_obs=4.0), 0.95)
        self.assertEqual(amp.amplification_skill(0.2, 0.2), 0.0)

    def test_gate_form_matches_benchmark_py(self) -> None:
        from pulse.benchmark import sigma_obs_for
        for mk in ("glucose", "hr", "sbp", "temp", "insulin"):
            sigma = sigma_obs_for(mk)
            # same expression as _build_per_marker: 1 - mae / max(pers_mae, sigma)
            self.assertAlmostEqual(amp.gate_skill(0.3 * sigma, 2.0 * sigma, sigma), 1.0 - 0.3 / 2.0)
            self.assertAlmostEqual(amp.gate_skill(0.3 * sigma, 0.5 * sigma, sigma), 1.0 - 0.3)

    def test_flat_threshold_is_relative_to_norm_scale(self) -> None:
        self.assertTrue(amp.is_flat(0.0, "sbp"))
        self.assertTrue(amp.is_flat(1e-6, "temp"))
        self.assertFalse(amp.is_flat(0.5, "glucose"))


class TestProtocolAndArgs(unittest.TestCase):
    def test_grids_partition_the_scored_span_and_never_score_t0(self) -> None:
        pr = amp.Protocol()
        spans = pr.span_times()
        grid = pr.score_times()
        self.assertNotIn(0, grid)
        self.assertEqual(sorted(spans[amp.SPAN_WINDOW] + spans[amp.SPAN_AHEAD]), grid)
        self.assertTrue(set(spans[amp.SPAN_WINDOW]).isdisjoint(spans[amp.SPAN_AHEAD]))
        self.assertLessEqual(max(spans[amp.SPAN_WINDOW]), pr.cal_end_min)
        self.assertGreater(min(spans[amp.SPAN_AHEAD]), pr.cal_end_min)
        self.assertLessEqual(max(pr.obs_times()), pr.cal_end_min)
        self.assertLess(max(grid), pr.duration_min)
        sw, act = pr.masks()
        self.assertEqual((sw.shape[0], float(sw.min()), float(act.max())), (pr.duration_min, 1.0, 0.0))
        self.assertEqual(amp.Protocol(meal_at_min=-1).meals(), [])

    def test_a_protocol_the_calibration_cannot_hold_out_of_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            amp.Protocol(duration_min=50, cal_end_min=50)
        with self.assertRaises(ValueError):
            amp.Protocol(cal_end_min=20, obs_every_min=10)       # two check-in times: no hold-out

    def test_set_and_marker_arguments(self) -> None:
        self.assertEqual(amp.parse_sets("glucose,glucose+hr"), [("glucose",), ("glucose", "hr")])
        self.assertEqual(amp.parse_markers("insulin, sbp"), ["insulin", "sbp"])
        with self.assertRaises(Exception):
            amp.parse_sets("glucose+gluclose")


class TestHeldOutIsDisjoint(unittest.TestCase):
    def test_default_held_out_is_disjoint_from_every_default_set(self) -> None:
        for a in amp.DEFAULT_SETS:
            held, dropped = amp.held_out_markers(a, amp.DEFAULT_HELD_OUT)
            self.assertEqual(dropped, [])
            self.assertTrue(set(held).isdisjoint(a))
            self.assertEqual(held, list(amp.DEFAULT_HELD_OUT))

    def test_overlap_is_dropped_and_reported_unknown_is_an_error(self) -> None:
        held, dropped = amp.held_out_markers(("glucose", "hr"), ("hr", "insulin", "glucose", "sbp", "insulin"))
        self.assertEqual(held, ["insulin", "sbp"])
        self.assertEqual(dropped, ["hr", "glucose"])
        with self.assertRaises(ValueError):
            amp.held_out_markers(("glucose",), ("insulim",))

    def test_score_patient_refuses_an_overlapping_set(self) -> None:
        m = _model()
        with self.assertRaises(ValueError):
            _score_patient(m, _patient(m), PROTO, ("glucose",), ("glucose", "insulin"),
                           e_prior=amp.prior_embedding(m), settings=CalibrationSettings(max_steps=1))

    def test_calibration_is_shown_only_the_calibrated_markers(self) -> None:
        m = _model()
        p = _patient(m)
        calls: list = []
        b = ["insulin", "cortisol", "ghrelin", "sbp", "temp"]
        for a in (("glucose",), ("glucose", "hr")):
            with mock.patch.object(amp, "calibrate_embedding", _stub(amp.prior_embedding(m), False, calls)):
                _score_patient(m, p, PROTO, a, b, e_prior=amp.prior_embedding(m),
                               settings=CalibrationSettings(max_steps=1))
        self.assertEqual([c["markers"] for c in calls], [{"glucose"}, {"glucose", "hr"}])
        for c in calls:
            self.assertTrue(c["markers"].isdisjoint(b))
            # only check-ins inside the calibration window, never the scored 'ahead' span
            self.assertLessEqual(max(c["times"]), PROTO.cal_end_min)
            self.assertEqual(c["duration"], PROTO.duration_min)
            self.assertIsNone(c["kw"].get("prior_mean"))      # a fresh model has no trained prior
            # explicit inputs, so the embedding reaches the dynamics only through the projections
            self.assertTrue(torch.equal(c["kw"]["sleep_wake"], torch.ones(PROTO.duration_min)))
            self.assertTrue(torch.equal(c["kw"]["activity"], torch.zeros(PROTO.duration_min)))


class TestHarnessIsNotBroken(unittest.TestCase):
    """Calibrate on a marker, score that same marker: it must beat the prior mean on a
    patient the model itself generated. Real ``calibrate_embedding``, no stubs."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = _model()
        cls.patient = _patient(cls.model, scale=0.5)
        cls.settings = CalibrationSettings(max_steps=25, lr=0.05, patience=8)
        cls.unit = _score_patient(
            cls.model, cls.patient, PROTO, ("glucose",), ("insulin", "ghrelin"),
            e_prior=amp.prior_embedding(cls.model), settings=cls.settings)

    def test_calibration_accepts_and_moves_the_embedding(self) -> None:
        self.assertTrue(self.unit.accepted, self.unit.reason)
        self.assertGreater(self.unit.steps, 0)
        self.assertNotEqual(self.unit.emb_dist_cal, self.unit.emb_dist_prior)

    def test_the_calibrated_marker_beats_the_prior_mean_in_sample_and_overall(self) -> None:
        fit = amp.summarize([self.unit], "glucose", (amp.SPAN_CHECKIN,), "fit pts")
        scored = amp.summarize([self.unit], "glucose", amp.SPANS_SCORED, "calibrated")
        self.assertGreater(fit.mae_prior, 0.0)
        self.assertLess(fit.mae_cal, fit.mae_prior)
        self.assertGreater(fit.skill, amp.CONTROL_MIN_SKILL)       # clearly positive, not just > 0
        self.assertLess(scored.mae_cal, scored.mae_prior)
        self.assertGreater(scored.skill, 0.0)
        self.assertEqual((fit.improved, fit.worse, fit.tied), (1, 0, 0))

    def test_set_result_verdict_is_ok_for_this_control(self) -> None:
        res = amp.build_set_result("synthetic", [self.unit], ["glucose"], ["insulin", "ghrelin"])
        self.assertTrue(res.control_ok, res.control_text)
        self.assertGreater(res.control_checkin, amp.CONTROL_MIN_SKILL)
        self.assertIn("control ok", res.control_text)


class TestZeroAndNegative(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = _model()
        cls.patient = _patient(cls.model, scale=0.5)
        cls.prior = amp.prior_embedding(cls.model)
        cls.a, cls.b = ("glucose",), ["insulin", "ghrelin", "sbp"]

    def _score(self, embedding, accepted: bool):
        with mock.patch.object(amp, "calibrate_embedding", _stub(embedding, accepted, [])):
            return _score_patient(self.model, self.patient, PROTO, self.a, self.b,
                                  e_prior=self.prior, settings=CalibrationSettings(max_steps=1))

    def test_skill_is_exactly_zero_when_the_calibrated_embedding_is_the_prior_mean(self) -> None:
        for accepted in (False, True):    # rejected window, or accepted-but-came-back-equal
            unit = self._score(self.prior.clone(), accepted)
            for mk in (*self.a, *self.b):
                for spans in ((amp.SPAN_WINDOW,), (amp.SPAN_AHEAD,), amp.SPANS_SCORED):
                    r = amp.summarize([unit], mk, spans, "cross")
                    if r.mae_prior > 0:
                        self.assertEqual(r.skill, 0.0, (mk, spans, accepted))
                    self.assertEqual((r.improved, r.worse), (0, 0))
                    self.assertEqual(r.tied, 1)
            res = amp.build_set_result("synthetic", [unit], self.a, self.b)
            self.assertEqual(res.headline["mean"], 0.0)
            self.assertIn("ZERO", amp.render_set(res))

    def test_rejected_means_unchanged_embedding_and_is_counted(self) -> None:
        units = [self._score(self.prior.clone(), False) for _ in range(3)]
        res = amp.build_set_result("synthetic", units, self.a, self.b)
        self.assertEqual(res.accepted, 0)
        self.assertFalse(res.control_ok)
        text = amp.render_set(res)
        self.assertIn("calibration accepted in 0/3 windows (0%)", text)
        self.assertIn("CONTROL UNINFORMATIVE", text)

    def test_the_oracle_embedding_scores_exactly_one_everywhere(self) -> None:
        # The other end of the scale: hand back the patient's TRUE embedding and every marker, A or
        # held-out, is predicted without error. Skill is +1 (not "large"), so +1 is the ceiling.
        unit = self._score(self.patient.embedding.clone(), True)
        res = amp.build_set_result("synthetic", [unit], self.a, self.b)
        live = [r for r in res.rows if not r.flat]
        self.assertTrue(live)
        for r in live:
            self.assertEqual(r.mae_cal, 0.0, r.marker)
            self.assertEqual(r.skill, 1.0, r.marker)
        self.assertEqual(res.headline["mean"], 1.0)

    def test_a_worse_embedding_gives_negative_skill_and_the_report_says_so(self) -> None:
        # Calibrated "answer" on the far side of the prior mean from the truth: the held-out
        # markers must come out WORSE than not calibrating, and the headline must be negative.
        unit = self._score(-3.0 * self.patient.embedding, True)
        res = amp.build_set_result("synthetic", [unit], self.a, self.b)
        live = [r for r in res.rows if r.held_out and not r.flat]
        self.assertTrue(live)
        for r in live:
            self.assertLess(r.skill, 0.0, r.marker)
            self.assertEqual((r.improved, r.worse), (0, 1))
        self.assertLess(res.headline["mean"], 0.0)
        self.assertIn("NEGATIVE", amp.render_set(res))


class TestModuleCut(unittest.TestCase):
    def test_cross_module_means_no_calibrated_marker_shares_the_module(self) -> None:
        self.assertEqual(amp.module_of("glucose"), "metabolic")
        self.assertEqual(amp.module_of("ghrelin"), "appetite")
        self.assertFalse(amp.is_cross_module("insulin", ("glucose",)))       # same module as glucose
        for mk in ("cortisol", "ghrelin", "sbp", "temp"):
            self.assertTrue(amp.is_cross_module(mk, ("glucose",)), mk)
        self.assertFalse(amp.is_cross_module("sbp", ("glucose", "hr")))      # hr is cardiovascular too

    def test_rows_and_headlines_are_split_by_it(self) -> None:
        m = _model()
        p = _patient(m)
        with mock.patch.object(amp, "calibrate_embedding", _stub(0.5 * p.embedding, True, [])):
            u = _score_patient(m, p, PROTO, ("glucose", "hr"), list(amp.DEFAULT_HELD_OUT),
                               e_prior=amp.prior_embedding(m), settings=CalibrationSettings(max_steps=1))
        res = amp.build_set_result("synthetic", [u], ["glucose", "hr"], list(amp.DEFAULT_HELD_OUT))
        roles = {r.marker: r.role for r in res.rows if r.held_out}
        self.assertEqual(roles, {"insulin": "same", "cortisol": "cross", "ghrelin": "cross",
                                 "sbp": "same", "temp": "cross"})
        self.assertEqual(set(res.headline_cross["per_marker"]) | set(res.headline_same["per_marker"]),
                         set(res.headline["per_marker"]))


class TestAccounting(unittest.TestCase):
    """The tallies against a from-scratch recomputation, and the generation/calibration
    forward maps against each other (zero model error is a claim about both)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = _model()
        cls.patient = _patient(cls.model, scale=0.5)
        cls.prior = amp.prior_embedding(cls.model)

    def test_mae_is_the_mean_absolute_error_on_the_scored_grid(self) -> None:
        cal_emb = 0.4 * self.patient.embedding
        with mock.patch.object(amp, "calibrate_embedding", _stub(cal_emb, True, [])):
            u = _score_patient(self.model, self.patient, PROTO, ("glucose",), ["insulin"],
                               e_prior=self.prior, settings=CalibrationSettings(max_steps=1))
        st = self.patient.initial_state
        pred_cal = amp.rollout(self.model, st, cal_emb, PROTO).numpy().astype(np.float64)
        pred_pri = amp.rollout(self.model, st, self.prior, PROTO).numpy().astype(np.float64)
        truth = self.patient.truth.numpy().astype(np.float64)
        grid = PROTO.score_times()
        self.assertEqual(grid[0], PROTO.score_every_min)           # t=0 is the initial state: not scored
        for mk, spans, times in (
            ("insulin", amp.SPANS_SCORED, grid),
            ("insulin", (amp.SPAN_WINDOW,), [t for t in grid if t <= PROTO.cal_end_min]),
            ("insulin", (amp.SPAN_AHEAD,), [t for t in grid if t > PROTO.cal_end_min]),
            ("glucose", (amp.SPAN_CHECKIN,), PROTO.obs_times()),
        ):
            j = MARKER_INDEX[mk]
            c, pr, ini, n = u.mae(mk, spans)
            self.assertEqual(n, len(times))
            self.assertAlmostEqual(c, float(np.abs(pred_cal[times, j] - truth[times, j]).mean()), places=6)
            self.assertAlmostEqual(pr, float(np.abs(pred_pri[times, j] - truth[times, j]).mean()), places=6)
            self.assertAlmostEqual(ini, float(np.abs(float(st[j]) - truth[times, j]).mean()), places=5)

    def test_the_true_embedding_reproduces_its_own_observations_through_the_calibration_forward(self) -> None:
        from pulse.calibration import evaluate_data_loss
        p = self.patient
        obs = [MeasurementPoint(time=t, marker_id=mk, value=float(p.truth[t, MARKER_INDEX[mk]]))
               for mk in ("glucose", "hr") for t in PROTO.obs_times()]
        sw, act = PROTO.masks()
        kw = dict(start_time_minutes=PROTO.start_time_minutes, sleep_wake=sw, activity=act)
        at_truth = evaluate_data_loss(self.model, p.embedding, obs, p.initial_state, PROTO.meals(),
                                      PROTO.duration_min, **kw)
        at_prior = evaluate_data_loss(self.model, self.prior, obs, p.initial_state, PROTO.meals(),
                                      PROTO.duration_min, **kw)
        self.assertLess(at_truth, 1e-10)                  # zero model error: same map, same inputs
        self.assertGreater(at_prior, 1e-6)                # and the data do discriminate patients


class TestChannels(unittest.TestCase):
    """The decomposition reroutes each module's projection; it has to be exact or be silent."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = _model()
        cls.patient = _patient(cls.model, scale=0.5)
        cls.prior = amp.prior_embedding(cls.model)
        cls.cal = 0.4 * cls.patient.embedding
        st = cls.patient.initial_state
        cls.roll = staticmethod(lambda e: amp.rollout(cls.model, st, e, PROTO))

    def test_rerouting_every_projection_reproduces_the_plain_calibrated_rollout(self) -> None:
        plain = self.roll(self.cal)
        names = list(self.model.embedding_projections)
        with amp.seen_by(self.model, {n: self.cal for n in names}):
            every = self.roll(self.prior)
        self.assertLess(float((every - plain).abs().max()), 1e-4)
        # ... and the hooks are gone afterwards
        self.assertTrue(torch.equal(self.roll(self.prior), amp.rollout(self.model, self.patient.initial_state,
                                                                       self.prior, PROTO)))
        self.assertFalse(any(m._forward_pre_hooks for m in self.model.embedding_projections.values()))

    def test_a_module_nothing_reads_cannot_move_any_other_marker(self) -> None:
        from pulse.types import MARKERS, MODULE_COUPLING_CHANNELS
        readers = {ch for chans in MODULE_COUPLING_CHANNELS.values() for ch in chans}
        unread = {m.id for m in MARKERS if m.system.value == "respiratory" and m.id not in readers}
        self.assertTrue({"rr", "spo2"} <= unread)          # the structural premise
        base = self.roll(self.prior)
        with amp.seen_by(self.model, {"respiratory": self.cal}):
            moved = self.roll(self.prior)
        for mk, j in MARKER_INDEX.items():
            diff = float((moved[:, j] - base[:, j]).abs().max())
            if mk in unread:
                self.assertGreater(diff, 1e-6, mk)
            else:
                self.assertLess(diff, 1e-6, mk)        # nothing reads respiratory: not one marker moves

    def test_rejected_window_means_every_channel_is_the_prior(self) -> None:
        with mock.patch.object(amp, "calibrate_embedding", _stub(self.prior.clone(), False, [])):
            u = amp.score_patient(self.model, self.patient, PROTO, ("glucose",), ["insulin", "ghrelin"],
                                  e_prior=self.prior, settings=CalibrationSettings(max_steps=1))
        res = amp.build_set_result("synthetic", [u], ["glucose"], ["insulin", "ghrelin"])
        for c in amp.CHANNELS:
            for mk in ("insulin", "ghrelin"):
                self.assertEqual(res.channel_skill[c][mk], 0.0, (c, mk))
        self.assertEqual(res.hook_err, 0.0)

    def test_channels_are_scored_and_the_full_channel_is_the_table(self) -> None:
        with mock.patch.object(amp, "calibrate_embedding", _stub(self.cal, True, [])):
            u = amp.score_patient(self.model, self.patient, PROTO, ("glucose",), ["insulin", "ghrelin"],
                                  e_prior=self.prior, settings=CalibrationSettings(max_steps=1))
        res = amp.build_set_result("synthetic", [u], ["glucose"], ["insulin", "ghrelin"])
        self.assertLess(res.hook_err, amp.HOOK_TOL)
        held = {r.marker: r for r in res.rows if r.held_out}
        for mk in ("insulin", "ghrelin"):
            self.assertAlmostEqual(res.channel_skill["full"][mk], held[mk].skill, places=9)
            # the partial channels differ from the full one (otherwise they decompose nothing)
            self.assertNotAlmostEqual(res.channel_skill["gut"][mk], res.channel_skill["full"][mk], places=4)
        text = amp.render_set(res)
        self.assertIn("CHANNELS", text)
        self.assertIn("coupling", text)

    def test_an_untrusted_reroute_is_reported_not_used(self) -> None:
        real = amp.channel_trajectories

        def broken(model, roll, e_prior, e_cal, a, b):
            out = real(model, roll, e_prior, e_cal, a, b)
            out["every"] = out["every"] + 5.0          # as if the model read the embedding elsewhere
            return out

        with mock.patch.object(amp, "channel_trajectories", broken), \
                mock.patch.object(amp, "calibrate_embedding", _stub(self.cal, True, [])):
            u = amp.score_patient(self.model, self.patient, PROTO, ("glucose",), ["insulin"],
                                  e_prior=self.prior, settings=CalibrationSettings(max_steps=1))
        res = amp.build_set_result("synthetic", [u], ["glucose"], ["insulin"])
        self.assertGreater(res.hook_err, amp.HOOK_TOL)
        text = amp.render_set(res)
        self.assertIn("untrusted", text)
        self.assertNotIn("not additive", text)


class TestFlatMarkersAndHeadline(unittest.TestCase):
    def test_embedding_blind_markers_are_flat_counted_and_left_out(self) -> None:
        m = _model(authority=0.0)            # a fresh model: sbp/cortisol ignore the embedding entirely
        p = _patient(m, scale=0.1)
        self.assertEqual(float((p.truth[:, MARKER_INDEX["sbp"]] - p.truth[0, MARKER_INDEX["sbp"]]).abs().max()), 0.0)
        with mock.patch.object(amp, "calibrate_embedding", _stub(0.05 * p.embedding, True, [])):
            unit = _score_patient(m, p, PROTO, ("glucose",), ["insulin", "cortisol", "sbp"],
                                  e_prior=amp.prior_embedding(m), settings=CalibrationSettings(max_steps=1))
        res = amp.build_set_result("synthetic", [unit], ["glucose"], ["insulin", "cortisol", "sbp"])
        flat = {r.marker for r in res.rows if r.flat}
        self.assertTrue({"cortisol", "sbp"} <= flat)
        self.assertEqual(res.headline["n_flat"], len(flat - {"glucose"}))
        self.assertNotIn("sbp", res.headline["per_marker"])
        self.assertIn("flat marker(s) excluded", amp.render_set(res))

    def test_headline_is_the_unweighted_mean_of_pooled_per_marker_skills(self) -> None:
        cal = np.array([[1.0, 2.0], [3.0, 2.0]])     # [units, markers]
        pri = np.array([[2.0, 4.0], [4.0, 4.0]])
        h = amp.headline_skill(cal, pri, ["glucose", "insulin"])
        self.assertAlmostEqual(h["per_marker"]["glucose"], 1 - 2.0 / 3.0)
        self.assertAlmostEqual(h["per_marker"]["insulin"], 1 - 2.0 / 4.0)
        self.assertAlmostEqual(h["mean"], 0.5 * ((1 - 2.0 / 3.0) + 0.5))

    def test_verdict_does_not_call_a_noisy_positive_positive(self) -> None:
        self.assertTrue(amp._verdict(0.009, (-0.3, 0.2), 6).startswith("INDISTINGUISHABLE FROM ZERO"))
        self.assertTrue(amp._verdict(0.20, (0.10, 0.30), 8).startswith("POSITIVE"))
        self.assertTrue(amp._verdict(-0.20, (-0.30, -0.10), 8).startswith("NEGATIVE"))
        small = amp._verdict(-0.20, None, 3)
        self.assertTrue(small.startswith("NEGATIVE") and "only 3 patients" in small)
        self.assertTrue(amp._verdict(0.0, None, 1).startswith("ZERO: calibration changed nothing B could feel"))
        self.assertTrue(amp._verdict(0.010, None, 1).startswith("ZERO"))       # inside the dead band
        self.assertTrue(amp._verdict(-0.03, None, 1).startswith("NEGATIVE"))

    def test_bootstrap_interval_brackets_a_constant_effect_and_needs_enough_patients(self) -> None:
        n = amp.MIN_UNITS_FOR_INTERVAL
        cal = np.full((n, 2), 1.0)
        pri = np.full((n, 2), 2.0)
        lo, hi = amp.bootstrap_headline(cal, pri, ["glucose", "insulin"], n_boot=50)
        self.assertAlmostEqual(lo, 0.5)
        self.assertAlmostEqual(hi, 0.5)
        # three patients give ten distinct bootstrap means: no interval rather than a fake one
        self.assertIsNone(amp.bootstrap_headline(cal[:3], pri[:3], ["glucose", "insulin"]))


class TestCheckpointPath(unittest.TestCase):
    def test_trained_prior_is_loaded_and_patients_are_drawn_from_it(self) -> None:
        torch.manual_seed(3)
        src = amp.fresh_model(1, 0.3, **SMALL)
        pm = 0.2 * torch.randn(EMBEDDING_DIM)
        ps = 0.05 + 0.1 * torch.rand(EMBEDDING_DIM)
        table = torch.randn(5, EMBEDDING_DIM)
        blob = {
            "model_state": src.state_dict(), "model_config": src.constructor_kwargs,
            "embedding_dim": EMBEDDING_DIM, "embedding_prior_mean": pm.tolist(),
            "embedding_prior_std": ps.tolist(), "embeddings_state": {"weight": table},
        }
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.pt"
            torch.save(blob, path)
            model, loaded = amp.load_model(str(path))
        self.assertTrue(torch.allclose(amp.prior_embedding(model), pm))
        self.assertFalse(any(p.requires_grad for p in model.parameters()))
        drawn = amp.draw_embeddings(model, 4, seed=11)
        g = torch.Generator().manual_seed(11)
        for e in drawn:
            z = torch.randn(EMBEDDING_DIM, generator=g)
            self.assertTrue(torch.allclose(e, pm + ps * z))
        # patient i does not depend on how many patients were asked for
        self.assertTrue(torch.equal(amp.draw_embeddings(model, 2, seed=11)[1], drawn[1]))
        rows = amp.draw_embeddings(model, 3, seed=11, table=loaded["embeddings_state"]["weight"])
        self.assertTrue(torch.equal(torch.stack(rows), table[:3]))
        # and calibration is handed the trained prior, as the gate hands it
        calls: list = []
        p, _ = amp.make_patient(model, 0, drawn[0], PROTO)
        with mock.patch.object(amp, "calibrate_embedding", _stub(pm, False, calls)):
            _score_patient(model, p, PROTO, ("glucose",), ["insulin"], e_prior=amp.prior_embedding(model),
                           settings=CalibrationSettings(max_steps=1))
        self.assertTrue(torch.allclose(calls[0]["kw"]["prior_mean"], pm))
        self.assertTrue(torch.allclose(calls[0]["kw"]["prior_std"], ps))

    def test_without_a_prior_patients_are_n0_sd_and_the_baseline_is_zeros(self) -> None:
        m = _model()
        e = torch.stack(amp.draw_embeddings(m, 400, seed=0, emb_sd=0.1))
        self.assertAlmostEqual(float(e.std()), 0.1, delta=0.005)
        self.assertTrue(torch.equal(amp.prior_embedding(m), torch.zeros(EMBEDDING_DIM)))


class TestAuthoritySwitch(unittest.TestCase):
    def test_fresh_heads_are_embedding_blind_and_authority_gives_them_a_voice(self) -> None:
        blind, live = _model(0.0), _model(0.3)
        proto = amp.Protocol(duration_min=40, cal_end_min=20, obs_every_min=5, meal_at_min=None)
        e = 0.3 * torch.ones(EMBEDDING_DIM)
        for m, expect_moves in ((blind, False), (live, True)):
            st = torch.tensor(NORM_CENTER)
            a = amp.rollout(m, st, torch.zeros(EMBEDDING_DIM), proto)
            b = amp.rollout(m, st, e, proto)
            moved = float((a - b).abs()[:, MARKER_INDEX["sbp"]].max()) > 1e-4
            self.assertEqual(moved, expect_moves)
        self.assertGreater(amp.give_embedding_authority(_model(0.0), 0.3, 0), 0)


class TestRealEpisodes(unittest.TestCase):
    def _episode(self) -> BenchmarkEpisode:
        init = np.array(NORM_CENTER, dtype=np.float32)
        init[MARKER_INDEX["glucose"]], init[MARKER_INDEX["hr"]] = 90.0, 58.0
        check = [{"time": t, "measurements": {"glucose": 90.0 + 0.3 * t, "hr": 60.0 + 0.1 * t}}
                 for t in (5, 10, 15, 20, 25, 30)]
        ev = [MeasurementPoint(time=t, marker_id=mk, value=v)
              for t in (40, 45, 50) for mk, v in (("glucose", 105.0), ("hr", 63.0))]
        return BenchmarkEpisode(
            user_id="subject-night-01", duration_min=60, initial_state=init,
            meals=[MealEvent(time=5.0, carbs=40.0, fats=5.0, proteins=10.0)],
            calibration_check_ins=check, eval_measurements=ev, start_time_minutes=420.0,
            sleep_wake=np.ones(60, dtype=np.float32), activity=np.zeros(60, dtype=np.float32),
            source="cgm_real")

    def test_calibrate_on_glucose_score_hr(self) -> None:
        m = _model()
        ep = self._episode()
        self.assertEqual(amp.episode_markers([ep]), ["glucose", "hr"])
        calls: list = []
        with mock.patch.object(amp, "calibrate_embedding", _stub(amp.prior_embedding(m), False, calls)):
            u = amp.score_episode(m, ep, ("glucose",), ["hr"], e_prior=amp.prior_embedding(m),
                                  settings=CalibrationSettings(max_steps=1))
        # hr is never fed to calibration; its in-window check-ins and eval points are what gets scored
        self.assertEqual(calls[0]["markers"], {"glucose"})
        self.assertEqual(u.cells[("hr", amp.SPAN_WINDOW)].n, 6)
        self.assertEqual(u.cells[("hr", amp.SPAN_AHEAD)].n, 3)
        self.assertNotIn(("hr", amp.SPAN_CHECKIN), u.cells)
        self.assertEqual(u.cells[("glucose", amp.SPAN_CHECKIN)].n, 6)       # the control, in-sample
        self.assertEqual(u.cells[("glucose", amp.SPAN_AHEAD)].n, 3)
        self.assertEqual(u.subject, "subject")
        res = amp.build_set_result("real", [u], ["glucose"], ["hr"])
        self.assertIn("1 subject(s)", amp.render_set(res))
        # rejected window: unchanged embedding, so skill 0 on hr as everywhere else
        self.assertEqual(res.rows[-1].marker, "hr")
        self.assertEqual(res.rows[-1].skill, 0.0)
        self.assertIn("Out-of-sample (eval points) the calibrated markers score +0.00", res.control_text)
        self.assertNotIn("WORSE", res.control_text)

    def test_out_of_sample_control_says_worse_exactly_when_it_is(self) -> None:
        m = _model()
        ep = self._episode()
        far = 3.0 * torch.ones(EMBEDDING_DIM)             # a calibrated "answer" nowhere near the prior
        with mock.patch.object(amp, "calibrate_embedding", _stub(far, True, [])):
            u = amp.score_episode(m, ep, ("glucose",), ["hr"], e_prior=amp.prior_embedding(m),
                                  settings=CalibrationSettings(max_steps=1), channels=False)
        res = amp.build_set_result("real", [u], ["glucose"], ["hr"])
        oos = [r.skill for r in res.rows if r.role == "calibrated"]
        self.assertEqual(len(oos), 1)
        self.assertEqual("WORSE" in res.control_text, oos[0] < 0)
        self.assertIn("Out-of-sample (eval points)", res.control_text)

    def test_an_episode_without_the_calibrated_marker_is_skipped(self) -> None:
        m = _model()
        ep = self._episode()
        ep.calibration_check_ins = [{"time": 5, "measurements": {"hr": 60.0}}, {"time": 10, "measurements": {"hr": 61.0}}]
        self.assertIsNone(amp.score_episode(m, ep, ("glucose",), ["hr"], e_prior=amp.prior_embedding(m),
                                            settings=CalibrationSettings(max_steps=1)))

    def test_real_calibration_runs_through_the_shared_function(self) -> None:
        m = _model()
        ep = self._episode()
        u = amp.score_episode(m, ep, ("glucose",), ["hr"], e_prior=amp.prior_embedding(m),
                              settings=CalibrationSettings(max_steps=2, patience=0))
        self.assertIn(u.reason, {"accepted", "no_improvement"})
        self.assertFalse(u.diverged)


class TestMainEndToEnd(unittest.TestCase):
    def test_fresh_model_prints_the_full_report_and_the_caveat(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "r.json"
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = amp.main(["--patients", "1", "--steps", "2", "--sets", "glucose", "--duration", "40",
                               "--cal-end", "25", "--obs-every", "5", "--score-every", "5", "--meal-at", "5",
                               "--authority", "0.3", "--json", str(out)])
            text = buf.getvalue()
            data = json.loads(out.read_text())
        self.assertEqual(rc, 0)
        for needle in ("UNTRAINED MODEL", "MAE_cal", "MAE_prior", "skill", "+/-/=", "cross", "same",
                       "calibration accepted in", "HEADLINE", "MACHINERY", "prior-mean embedding",
                       "CHANNELS", "coupling", "SUMMARY"):
            self.assertIn(needle, text)
        self.assertEqual(data["trained"], False)
        self.assertEqual(data["results"][0]["a_markers"], ["glucose"])
        self.assertEqual(data["results"][0]["b_markers"], ["insulin", "cortisol", "ghrelin", "sbp", "temp"])


class TestCliGuards(unittest.TestCase):
    def test_authority_is_for_fresh_models_and_table_needs_a_checkpoint(self) -> None:
        from contextlib import redirect_stderr
        for argv in (["model.pt", "--authority", "0.3"], ["--patients-from", "table"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as cm:
                amp.main(argv)
            self.assertEqual(cm.exception.code, 2, argv)


if __name__ == "__main__":
    unittest.main()
