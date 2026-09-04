"""Iter 97 (review 5.2): the ruler says what it was, and can be frozen.

The `teacher*` persistence MAPE -- a property of the truth alone -- changed
between the iter-95 and iter-96 reports on the same dataset file and nothing
in either report could say why. Every report now carries a ruler_fingerprint
(git SHA, per-source truth digests, dataset md5, thresholds md5, calibration
settings) and the in-process truth can be frozen to a file and reloaded.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from pulse import benchmark as bm
from pulse.benchmark import (
    BenchmarkEpisode,
    MeasurementPoint,
    episode_truth_digest,
    ruler_fingerprint,
    source_truth_digests,
)
from pulse.knowledge import benchmark_extras as bx
from pulse.types import NORM_CENTER


class TestFingerprint(unittest.TestCase):
    def test_fingerprint_names_the_ruler(self) -> None:
        eps = bx.all_cohort_benchmark_episodes()
        fp = ruler_fingerprint(eps, bm.default_thresholds())
        self.assertRegex(fp["git_sha"], r"^([0-9a-f]{40}(-dirty)?|unknown)$")
        self.assertEqual(set(fp["sources"]), {"teacher", "teacher_dynamic"})
        self.assertEqual(fp["sources"]["teacher_dynamic"]["episodes"], 8)
        self.assertRegex(fp["thresholds_md5"], r"^[0-9a-f]{32}$")
        for key in ("steps", "lr", "prior_weight", "max_norm"):
            self.assertIn(key, fp["calibration"])
        self.assertEqual(fp["sigma_obs"]["glucose"], 8.0)
        # deterministic: the same episodes give the same digests
        self.assertEqual(source_truth_digests(eps), fp["sources"])

    def test_digest_moves_when_truth_moves(self) -> None:
        ep = bx.all_cohort_benchmark_episodes()[0]
        moved = copy.deepcopy(ep)
        moved.eval_measurements[0] = MeasurementPoint(
            time=moved.eval_measurements[0].time,
            marker_id=moved.eval_measurements[0].marker_id,
            value=moved.eval_measurements[0].value + 1.0,
        )
        self.assertNotEqual(episode_truth_digest(ep), episode_truth_digest(moved))
        # ... and is invariant to float32 <-> float64 and JSON round trips
        again = copy.deepcopy(ep)
        again.initial_state = np.array(json.loads(json.dumps(ep.initial_state.tolist())), dtype=np.float32)
        self.assertEqual(episode_truth_digest(ep), episode_truth_digest(again))

    def test_dataset_md5_is_recorded_on_load(self) -> None:
        payload = {"meta": {"iter": 97}, "episodes": [{
            "user_id": "u", "duration_min": 60, "initial_state": list(NORM_CENTER),
            "meals": [], "calibration_check_ins": [],
            "eval_measurements": [{"time": 30, "marker_id": "glucose", "value": 90.0}],
            "start_time_minutes": 360, "source": "cgm_real",
        }]}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "d.json"
            path.write_text(json.dumps(payload))
            eps = bm.load_benchmark_dataset(str(path))
            fp = ruler_fingerprint(eps, {})
            self.assertEqual(fp["dataset_path"], str(path))
            self.assertRegex(fp["dataset_md5"], r"^[0-9a-f]{32}$")
            self.assertEqual(fp["dataset_meta"], {"iter": 97})


class TestFrozenRuler(unittest.TestCase):
    def test_freeze_and_reload_gives_the_same_ruler(self) -> None:
        live = bx.all_cohort_benchmark_episodes()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ruler.frozen.json"
            meta = bx.export_frozen_ruler(path, git_sha="deadbeef")
            self.assertEqual(meta["sources"], source_truth_digests(live))
            reloaded = bx.load_frozen_ruler(path)
            self.assertEqual(len(reloaded), len(live))
            self.assertEqual(source_truth_digests(reloaded), source_truth_digests(live))
            for a, b in zip(live, reloaded):
                self.assertEqual(a.user_id, b.user_id)
                self.assertEqual(a.source, b.source)
                self.assertEqual(len(a.eval_measurements), len(b.eval_measurements))
                self.assertIsNotNone(b.sleep_wake)
                self.assertIsNotNone(b.activity)
            # the env var swaps the live generator for the file, fingerprint says so
            with mock.patch.dict(os.environ, {bx.FROZEN_RULER_ENV: str(path)}):
                via_env = bx.all_cohort_benchmark_episodes()
                self.assertEqual(source_truth_digests(via_env), source_truth_digests(live))
                self.assertEqual(ruler_fingerprint(via_env, {})["frozen_ruler"], str(path))
            # tampering is caught
            raw = json.loads(path.read_text())
            raw["episodes"][0]["eval_measurements"][0]["value"] += 5.0
            path.write_text(json.dumps(raw))
            with self.assertRaises(ValueError):
                bx.load_frozen_ruler(path)


if __name__ == "__main__":
    unittest.main()
