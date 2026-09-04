"""Iter 97 (review 1.7): rest = 0 in the in-process episodes and the real-data ingest.

base.py defines activity 0 = rest, yet the in-process teacher episodes were
generated at 0.05 / 0.1 and the real-data ingest wrote 0.1 for every resting
minute; the teacher reads 0.05 as +4.2 bpm HR over true rest, and cgm_real
nights were scored at 0.1 = +8.3 bpm of teacher-meaning drive.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pulse import benchmark as bm
from pulse.knowledge.benchmark_extras import REST_ACTIVITY, all_cohort_benchmark_episodes
from pulse.types import NORM_CENTER


def _episode_json(activity, **extra) -> dict:
    ep = {
        "user_id": "u", "duration_min": 6, "initial_state": list(NORM_CENTER),
        "meals": [], "calibration_check_ins": [],
        "eval_measurements": [{"time": 3, "marker_id": "glucose", "value": 90.0}],
        "start_time_minutes": 360, "source": "cgm_real",
    }
    if activity is not None:
        ep["activity"] = activity
    ep.update(extra)
    return ep


class TestRestActivity(unittest.TestCase):
    def test_in_process_episodes_rest_at_zero(self) -> None:
        self.assertEqual(REST_ACTIVITY, 0.0)
        for ep in all_cohort_benchmark_episodes():
            self.assertIsNotNone(ep.activity, ep.user_id)
            self.assertEqual(float(np.min(ep.activity)), 0.0, ep.user_id)
            self.assertEqual(float(np.max(ep.activity)), 0.0, ep.user_id)

    def test_loader_treats_missing_activity_as_rest(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "d.json"
            path.write_text(json.dumps({"episodes": [_episode_json(None)]}))
            ep = bm.load_benchmark_dataset(str(path))[0]
            self.assertIsNotNone(ep.activity)
            self.assertEqual(ep.activity.tolist(), [0.0] * 6)

    def test_loader_remaps_legacy_ingest_rest_unless_declared(self) -> None:
        legacy = [0.1, 0.1, 0.4, 0.1, 0.95, 0.1]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "d.json"
            # undeclared: the pre-iter-97 export -> 0.1 is rest
            path.write_text(json.dumps({"episodes": [_episode_json(legacy)]}))
            ep = bm.load_benchmark_dataset(str(path))[0]
            np.testing.assert_allclose(ep.activity, [0.0, 0.0, 0.4, 0.0, 0.95, 0.0], rtol=1e-6)
            # declared: the file says what rest is, nothing is remapped
            path.write_text(json.dumps({"meta": {"activity_rest_level": 0.0},
                                        "episodes": [_episode_json(legacy)]}))
            ep = bm.load_benchmark_dataset(str(path))[0]
            np.testing.assert_allclose(ep.activity, legacy, rtol=1e-6)

    def test_ingest_default_is_rest_and_declares_it(self) -> None:
        path = Path(__file__).resolve().parents[1] / "scripts" / "ingest_real_data.py"
        spec = importlib.util.spec_from_file_location("ingest_real_data", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        self.assertEqual(mod.ACTIVITY_DEFAULT, 0.0)
        self.assertEqual(mod.ACTIVITY_INTENSITY[45], 0.0)  # sleep is rest
        src = path.read_text()
        self.assertIn('"activity_rest_level": ACTIVITY_DEFAULT', src)


if __name__ == "__main__":
    unittest.main()
