"""Unit tests for inter-marker hinge predicates (iter 74 ratio class).

Pure tensor functions — no model rollout, so these stay fast.
"""

from __future__ import annotations

import unittest

import torch

from pulse.knowledge.physiology_rules import hinge_min_ratio
from pulse.types import MARKER_INDEX, STATE_DIM


def _traj(glucagon: float, insulin: float, t: int = 10) -> torch.Tensor:
    """Constant trajectory with the given glucagon/insulin levels."""
    traj = torch.zeros(t, STATE_DIM)
    traj[:, MARKER_INDEX["glucagon"]] = glucagon
    traj[:, MARKER_INDEX["insulin"]] = insulin
    return traj


class TestHingeMinRatio(unittest.TestCase):
    def test_above_floor_is_zero(self) -> None:
        # glucagon/insulin = 80/4 = 20 ≥ floor 8 → satisfied.
        traj = _traj(glucagon=80.0, insulin=4.0)
        v = hinge_min_ratio(
            traj, MARKER_INDEX["glucagon"], MARKER_INDEX["insulin"],
            window=slice(0, 10), floor_ratio=8.0,
        )
        self.assertAlmostEqual(float(v), 0.0, places=5)

    def test_below_floor_is_positive_and_exact(self) -> None:
        # ratio = 80/20 = 4 < 8 → violation = 8 - 4 = 4.
        traj = _traj(glucagon=80.0, insulin=20.0)
        v = hinge_min_ratio(
            traj, MARKER_INDEX["glucagon"], MARKER_INDEX["insulin"],
            window=slice(0, 10), floor_ratio=8.0,
        )
        self.assertAlmostEqual(float(v), 4.0, places=4)

    def test_denominator_zero_is_finite(self) -> None:
        # insulin driven to 0 — clamp keeps ratio (and gradient) finite.
        traj = _traj(glucagon=80.0, insulin=0.0)
        traj.requires_grad_(True)
        v = hinge_min_ratio(
            traj, MARKER_INDEX["glucagon"], MARKER_INDEX["insulin"],
            window=slice(0, 10), floor_ratio=8.0, eps=1e-3,
        )
        self.assertTrue(torch.isfinite(v))
        # A satisfied-by-huge-ratio case still produces a finite backward.
        v.backward()
        self.assertTrue(torch.isfinite(traj.grad).all())


if __name__ == "__main__":
    unittest.main()
