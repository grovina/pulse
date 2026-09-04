"""Iter 97 (training): the coupling prior used the sign and ignored its own range (review 4.4)."""

from __future__ import annotations

import os

import torch

import pulse

REPO = os.path.dirname(os.path.dirname(os.path.abspath(pulse.__file__)))
assert pulse.__file__.startswith(REPO), pulse.__file__

from pulse.coupling_prior_loss import coupling_band_hinge, normalized_sensitivity  # noqa: E402
from pulse.knowledge.base import CouplingPrior  # noqa: E402
from pulse.types import MARKER_INDEX, NORM_SCALE  # noqa: E402


def test_band_hinge_is_zero_inside_the_declared_range() -> None:
    p = CouplingPrior("cortisol", "hr", +1, (0.05, 0.5))
    for s in (0.05, 0.1, 0.3, 0.5):
        assert float(coupling_band_hinge(torch.tensor(s), p)) == 0.0
    # Too WEAK and too STRONG both cost; a correct sign no longer buys unbounded gain.
    assert float(coupling_band_hinge(torch.tensor(0.01), p)) > 0.0
    assert float(coupling_band_hinge(torch.tensor(5.0), p)) > 0.0
    # Wrong sign costs more than merely weak, and the penalty is linear far out.
    wrong = float(coupling_band_hinge(torch.tensor(-0.3), p))
    weak = float(coupling_band_hinge(torch.tensor(0.0), p))
    assert wrong > weak > 0.0
    far = float(coupling_band_hinge(torch.tensor(50.0), p))
    farther = float(coupling_band_hinge(torch.tensor(100.0), p))
    assert farther - far < (100.0 - 50.0) / (0.5 - 0.05) * 1.01


def test_negative_sign_edge_reads_the_magnitude_of_the_negative_slope() -> None:
    p = CouplingPrior("insulin", "glucose", -1, (0.0001, 0.002))
    assert float(coupling_band_hinge(torch.tensor(-0.001), p)) == 0.0
    assert float(coupling_band_hinge(torch.tensor(+0.001), p)) > 0.0


def test_normalized_sensitivity_uses_the_norm_scale_ratio() -> None:
    raw = torch.tensor(0.05)  # uU/mL/min per mg/dL (Bergman gamma)
    n = normalized_sensitivity(raw, "glucose", "insulin")
    expected = 0.05 * NORM_SCALE[MARKER_INDEX["glucose"]] / NORM_SCALE[MARKER_INDEX["insulin"]]
    assert abs(float(n) - float(expected)) < 1e-6


def test_gradient_pushes_toward_the_band_from_both_sides() -> None:
    p = CouplingPrior("acth", "cortisol", +1, (0.002, 0.02))
    weak = torch.tensor(0.0, requires_grad=True)
    coupling_band_hinge(weak, p).backward()
    assert float(weak.grad) < 0.0        # increase the sensitivity
    strong = torch.tensor(0.2, requires_grad=True)
    coupling_band_hinge(strong, p).backward()
    assert float(strong.grad) > 0.0      # decrease it
