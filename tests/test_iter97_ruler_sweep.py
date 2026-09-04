"""Iter 97 (review 1.6): the calibration sweep must pass every swept knob explicitly.

Reassigning ``bm.BENCHMARK_GATE_CALIBRATE_MAX_NORM`` did nothing --
``calibrate_embedding`` bound its default at import -- so the sweep's
``max_norm=1.5`` row re-measured the control.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
import unittest
from pathlib import Path

from pulse import benchmark as bm


def _load_sweep():
    path = Path(__file__).resolve().parents[1] / "scripts" / "iter96_calibration_sweep.py"
    spec = importlib.util.spec_from_file_location("iter96_calibration_sweep", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestSweepPassesKnobs(unittest.TestCase):
    def test_module_constant_reassignment_is_a_no_op(self) -> None:
        # Documents the trap: the default is bound at import.
        saved = bm.BENCHMARK_GATE_CALIBRATE_MAX_NORM
        try:
            bm.BENCHMARK_GATE_CALIBRATE_MAX_NORM = 1.5
            default = inspect.signature(bm.calibrate_embedding).parameters["max_norm"].default
            self.assertEqual(default, saved)
        finally:
            bm.BENCHMARK_GATE_CALIBRATE_MAX_NORM = saved

    def test_sweep_passes_max_norm_explicitly(self) -> None:
        mod = _load_sweep()
        src = inspect.getsource(mod._episode_scores)
        self.assertIn("max_norm=max_norm", src)
        self.assertNotIn("bm.BENCHMARK_GATE_CALIBRATE_MAX_NORM =", src)
        self.assertIn("sigma_obs_for", inspect.getsource(mod.main))


if __name__ == "__main__":
    unittest.main()
