"""Iter 97 (training): trajectory supervision is a fence, not a pointwise copy (review 4.2)."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
import torch.nn as nn

import pulse

REPO = os.path.dirname(os.path.dirname(os.path.abspath(pulse.__file__)))
assert pulse.__file__.startswith(REPO), pulse.__file__

from pulse.model import ModularPhysiologyNetwork  # noqa: E402
from pulse.training import SignalContext, TrajectoryRolloutSignal, WeightSchedule  # noqa: E402
from pulse.training.trajectory_signal import (  # noqa: E402
    DEFAULT_BAND_OBSERVED,
    DEFAULT_BAND_UNOBSERVED,
    SOFT_RANGE_DEADZONE,
    default_band_per_marker,
    parse_band_per_marker,
    shape_marker_loss,
)
from pulse.types import EMBEDDING_DIM, MARKERS, NORM_SCALE  # noqa: E402


def test_default_bands_are_sized_to_teacher_uncertainty() -> None:
    bands = default_band_per_marker()
    assert DEFAULT_BAND_UNOBSERVED >= 0.3
    assert 0.1 <= DEFAULT_BAND_OBSERVED <= 0.2
    for m in ("glucose", "hr", "sbp", "dbp", "temp"):
        assert bands[m] == DEFAULT_BAND_OBSERVED
    for m in ("cortisol", "ghrelin", "leptin", "bhb", "liver_glycogen"):
        assert bands[m] == DEFAULT_BAND_UNOBSERVED
    # The review's band probe: 0.08 on glucose was 2.4 mg/dL, below CGM noise.
    gi = [m.id for m in MARKERS].index("glucose")
    assert bands["glucose"] * NORM_SCALE[gi] >= 4.0


def test_parse_band_map() -> None:
    m = parse_band_per_marker("glucose:0.1;*:0.5;hr:0.2")
    assert m["glucose"] == 0.1 and m["hr"] == 0.2 and m["cortisol"] == 0.5
    assert parse_band_per_marker("") is None and parse_band_per_marker(None) is None
    with pytest.raises(ValueError):
        parse_band_per_marker("glucoze:0.1")


def test_shape_loss_is_zero_when_mean_and_trend_agree_pointwise_not_required() -> None:
    T = 240
    t = torch.linspace(0, 1, T)
    ref = torch.stack([t * 2.0], dim=1)                 # rising trend, mean 1.0
    pred = torch.stack([t * 2.0 + 0.3 * torch.sin(t * 40.0)], dim=1)  # same mean/trend, wobbly
    mask = torch.ones(T, 1, dtype=torch.bool)
    band = torch.tensor([0.3])
    assert float(shape_marker_loss(pred, ref, mask, band)) == pytest.approx(0.0, abs=1e-3)
    # Wrong trend direction costs, even with the right mean.
    pred_flip = torch.stack([2.0 - t * 2.0], dim=1)
    assert float(shape_marker_loss(pred_flip, ref, mask, band)) > 0.5
    # A level offset larger than the band costs.
    assert float(shape_marker_loss(ref + 1.0, ref, mask, band)) > 0.0


def test_soft_range_is_a_dead_zone_not_a_pull() -> None:
    # A 48 h-fast BHB (3.5 mmol/L = 6.8 sigma from typical) must be inside the fence.
    assert SOFT_RANGE_DEADZONE >= 7.0


def _tiny_model() -> ModularPhysiologyNetwork:
    return ModularPhysiologyNetwork(
        metabolic_hidden=8, appetite_hidden=8, stress_hidden=8,
        cardiovascular_hidden=8, thermoreg_hidden=8, respiratory_hidden=8,
    )


def test_signal_runs_with_per_marker_bands_and_shape_markers() -> None:
    sig = TrajectoryRolloutSignal(
        n_patients=1, n_days=1, seed=0, contribution_weights={"full_body": 1.0},
        windows_per_patient=1, meal_window_bias=0.0, input_dropout=0.0, huber_delta=1.0,
        gut_loss_weight=0.0, coupling_weight=WeightSchedule(0.0), verifier_weight=WeightSchedule(0.0),
        trajectory_band_per_marker=default_band_per_marker(), shape_markers=("ghrelin", "cortisol"),
    )
    torch.manual_seed(0)
    model = _tiny_model()
    emb = nn.Embedding(1, EMBEDDING_DIM)
    params = list(model.parameters()) + list(emb.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    ctx = SignalContext(epoch=0, total_epochs=1, rng=np.random.default_rng(0), device=torch.device("cpu"),
                        optimizer=opt, params=params, grad_clip=10.0)
    res = sig.compute(model, emb, ctx)
    assert res.n_units == 1 and np.isfinite(res.loss_sum)
    assert "shape" in res.sub_metrics
