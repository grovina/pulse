"""Iter 97 (review 5.1 / 5.5 / 5.7 / 5.10 / 5.11): the ruler's arithmetic.

- skill = 1 - MAE / max(persistence_MAE, sigma_obs), physical units, per
  episode first. On the iter-96 report the dynamic BP truth moved 0.13-0.26
  mmHg per episode; the student's absolute error halved and its unfloored
  skill went +0.21 -> -1.82. A zero-persistence cell scored 0.0 = pass.
- headline = equal-source-weight mean of gate-marker MAE / sigma_obs; the
  legacy-dominated overall_weighted_mape is a continuity line.
- per-episode aggregation: teacher_dynamic is n = 8 arms, not 88 points.
- cgm_real is ONE subject and the report says so.
- fallback thresholds identical to the JSON; no-check-in fallback is the prior.
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from pulse import benchmark as bm
from pulse.benchmark import BenchmarkEpisode, MeasurementPoint
from pulse.types import NORM_CENTER


def _episode(user_id: str, source: str, n_eval: int = 4) -> BenchmarkEpisode:
    return BenchmarkEpisode(
        user_id=user_id, duration_min=720,
        initial_state=np.array(NORM_CENTER, dtype=np.float32), meals=[],
        calibration_check_ins=[{"time": 60, "measurements": {"glucose": 90.0, "hr": 60.0}}],
        eval_measurements=[
            MeasurementPoint(time=500 + 30 * k, marker_id=m, value=v)
            for k in range(n_eval) for m, v in (("glucose", 90.0), ("sbp", 120.0))
        ],
        start_time_minutes=360.0, source=source,
    )


def _fake_result(ep: BenchmarkEpisode, glucose_err: float, sbp_err: float,
                 sbp_truth_wiggle: float) -> dict:
    """A per-episode result as ``_evaluate_one_episode`` would return it, with a
    controlled absolute error per marker and a controlled truth movement."""
    points, merr, perr = [], [], []
    seen: dict[str, int] = {}
    for pt in ep.eval_measurements:
        k = seen.get(pt.marker_id, 0)
        seen[pt.marker_id] = k + 1
        truth = pt.value + (sbp_truth_wiggle * (k % 2) if pt.marker_id == "sbp" else 0.0)
        err = glucose_err if pt.marker_id == "glucose" else sbp_err
        pred = truth + err
        baseline = pt.value  # last calibration reading = the flat level
        denom = max(abs(truth), 1e-6)
        points.append({"marker_id": pt.marker_id, "time": pt.time, "truth": truth,
                       "pred": pred, "baseline": baseline})
        merr.append((pt.marker_id, abs(pred - truth) / denom))
        perr.append((pt.marker_id, abs(baseline - truth) / denom))
    return {
        "user_id": ep.user_id, "source": ep.source, "marker_errors": merr,
        "persistence_errors": perr, "eval_points": points, "embedding_norm": 1.0,
        "n_calibration_obs": 2, "verifier_overall": 0.9, "verifier_categories": {"meal": 0.9},
    }


class TestRulerArithmetic(unittest.TestCase):
    def _run(self, episodes, results, thresholds=None):
        by_id = {r["user_id"]: r for r in results}
        with mock.patch.dict(os.environ, {"PULSE_BENCHMARK_PARALLEL": "1"}), \
                mock.patch.object(bm, "_evaluate_one_episode", lambda model, ep: by_id[ep.user_id]):
            return bm.evaluate_model_against_benchmark(object(), episodes, thresholds)

    def test_skill_is_noise_floored_in_physical_units(self) -> None:
        # Truth barely moves (sbp wiggle 0.2 mmHg): persistence MAE 0.1, model
        # MAE 1.5 mmHg. Unfloored ratio says -14; floored by sigma_obs=4 the
        # model is well inside device noise: skill = 1 - 1.5/4 = +0.625.
        eps = [_episode(f"benchmark-dynamic-{k}", "teacher_dynamic") for k in range(3)]
        res = self._run(eps, [_fake_result(e, glucose_err=4.0, sbp_err=1.5, sbp_truth_wiggle=0.2) for e in eps])
        sbp = res["per_marker_by_source"]["teacher_dynamic"]["sbp"]
        self.assertAlmostEqual(sbp["mae"], 1.5, places=6)
        self.assertAlmostEqual(sbp["persistence_mae"], 0.1, places=6)
        self.assertEqual(sbp["sigma_obs"], 4.0)
        self.assertTrue(sbp["skill_floor_active"])
        self.assertAlmostEqual(sbp["skill_vs_persistence"], 1.0 - 1.5 / 4.0, places=6)
        self.assertLess(sbp["skill_vs_persistence_mape"], -10.0)  # the old column, for continuity
        self.assertAlmostEqual(sbp["truth_sd"], 0.1, places=6)
        # glucose: persistence MAE 0 (truth flat) -> floor = sigma 8, error 4 -> +0.5, NOT 0.0.
        glu = res["per_marker_by_source"]["teacher_dynamic"]["glucose"]
        self.assertAlmostEqual(glu["persistence_mae"], 0.0, places=9)
        self.assertAlmostEqual(glu["skill_vs_persistence"], 0.5, places=6)

    def test_skill_uses_persistence_when_truth_moves_more_than_noise(self) -> None:
        eps = [_episode("benchmark-dynamic-0", "teacher_dynamic")]
        res = self._run(eps, [_fake_result(eps[0], glucose_err=4.0, sbp_err=6.0, sbp_truth_wiggle=20.0)])
        sbp = res["per_marker_by_source"]["teacher_dynamic"]["sbp"]
        self.assertAlmostEqual(sbp["persistence_mae"], 10.0, places=6)
        self.assertFalse(sbp["skill_floor_active"])
        self.assertAlmostEqual(sbp["skill_vs_persistence"], 1.0 - 6.0 / 10.0, places=6)

    def test_episode_first_aggregation_and_subject_count(self) -> None:
        eps = [_episode(f"gabriel-night-{k:02d}", "cgm_real", n_eval=2 + 6 * k) for k in range(3)]
        errs = (2.0, 8.0, 14.0)  # episode-first mean = 8; point-pooled would weight the last one
        res = self._run(eps, [_fake_result(e, glucose_err=g, sbp_err=1.0, sbp_truth_wiggle=0.0)
                              for e, g in zip(eps, errs)])
        glu = res["per_marker_by_source"]["cgm_real"]["glucose"]
        self.assertEqual(glu["episodes"], 3)
        self.assertAlmostEqual(glu["mae"], 8.0, places=6)
        self.assertGreater(glu["samples"], 3)
        summ = res["source_summary"]["cgm_real"]
        self.assertEqual(summ["episodes"], 3)
        self.assertEqual(summ["subjects"], 1)
        self.assertEqual(summ["subject_ids"], ["gabriel"])
        self.assertEqual(bm.subject_of("benchmark-dynamic-small-carb"), "benchmark-dynamic-small-carb")

    def test_headline_is_equal_source_weight_and_excludes_legacy(self) -> None:
        legacy = [_episode(f"pulse-benchmark-user-{k:02d}", "legacy_static") for k in range(6)]
        real = [_episode("gabriel-night-01", "cgm_real")]
        dyn = [_episode("benchmark-dynamic-0", "teacher_dynamic")]
        results = (
            [_fake_result(e, glucose_err=0.8, sbp_err=0.4, sbp_truth_wiggle=0.0) for e in legacy]
            + [_fake_result(real[0], glucose_err=16.0, sbp_err=8.0, sbp_truth_wiggle=0.0)]
            + [_fake_result(dyn[0], glucose_err=4.0, sbp_err=2.0, sbp_truth_wiggle=0.0)]
        )
        res = self._run(legacy + real + dyn, results)
        hl = res["headline"]
        self.assertEqual(hl["sources"], ["cgm_real", "teacher_dynamic"])
        # cgm_real: glucose 16/8=2, sbp 8/4=2 -> 2.0 ; teacher_dynamic: 0.5, 0.5 -> 0.5 ; mean 1.25
        self.assertAlmostEqual(hl["by_source"]["cgm_real"]["normalized_mae"], 2.0, places=6)
        self.assertAlmostEqual(hl["by_source"]["teacher_dynamic"]["normalized_mae"], 0.5, places=6)
        self.assertAlmostEqual(hl["normalized_mae"], 1.25, places=6)
        # The continuity line is still there and still legacy-dominated.
        self.assertIn("overall_weighted_mape", res)
        self.assertLess(res["overall_weighted_mape"], res["overall_weighted_mape_by_source"]["cgm_real"])
        # thresholds can name the headline sources explicitly
        res2 = self._run(legacy + real + dyn, results,
                         thresholds={**bm.default_thresholds(), "headline_sources": ["cgm_real"]})
        self.assertEqual(res2["headline"]["sources"], ["cgm_real"])
        self.assertAlmostEqual(res2["headline"]["normalized_mae"], 2.0, places=6)

    def test_fallback_thresholds_match_json(self) -> None:
        path = Path(bm.__file__).with_name("benchmark.thresholds.json")
        loaded = json.loads(path.read_text())
        for key, value in bm._FALLBACK_THRESHOLDS.items():
            self.assertEqual(loaded.get(key), value, key)
        self.assertEqual(bm._FALLBACK_THRESHOLDS["textbook_mean_pass_rate_min"], 0.75)
        self.assertEqual(loaded["headline_sources"], ["cgm_real", "teacher_dynamic", "teacher"])

    def test_no_check_in_fallback_is_the_prior_mean(self) -> None:
        # An episode with no calibration check-ins must integrate the population
        # prior, not a user-id-seeded random vector.
        from pulse.model import ModularPhysiologyNetwork
        torch.manual_seed(0)
        model = ModularPhysiologyNetwork()
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        prior = torch.randn(model.embedding_dim) * 0.1
        model._embedding_prior_mean = prior
        model._embedding_prior_std = torch.full((model.embedding_dim,), 0.15)
        ep = BenchmarkEpisode(
            user_id="nobody", duration_min=30,
            initial_state=np.array(NORM_CENTER, dtype=np.float32), meals=[],
            calibration_check_ins=[],
            eval_measurements=[MeasurementPoint(time=20, marker_id="glucose", value=90.0)],
            start_time_minutes=360.0, source="real",
        )
        captured = {}
        real_integrate = bm.integrate

        def spy(**kw):
            captured["embedding"] = kw["embedding"].detach().clone()
            return real_integrate(**kw)

        with mock.patch.object(bm, "integrate", spy):
            bm._evaluate_one_episode(model, ep)
        self.assertTrue(torch.allclose(captured["embedding"], prior))


if __name__ == "__main__":
    unittest.main()
