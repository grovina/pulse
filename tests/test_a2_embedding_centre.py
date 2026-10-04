"""Plan A2: the patient table's mean is pinned to zero, by its own strong weight.

The per-row term (spec 0.001) does not do it. The centre is a gauge direction -- every consumer of
a code reads it through an affine map -- so the reconstruction loss never holds it and the cloud sits
wherever it drifted (``||prior_mean|| = 0.26`` on iter 90), while the soft norm, the hard clamp and
the zero-embedding evaluation point are all centred at 0 and the calibration prior at mean(table).

What these pin: the centre term drives ``mean(E)`` to 0 without flattening per-patient information
(the failure mode the signal's own docstring warns about); the per-row term alone does not; the two
weights are independent and both reported; and the flag, the ``train()`` kwarg and the dataclass
agree on the default, so a recipe that forgets to set it still gets the constraint.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch
import torch.nn as nn

from pulse.train import build_arg_parser, train
from pulse.training import (
    EmbeddingPriorSignal,
    SignalContext,
    SignalResult,
    WeightSchedule,
    joint_aux_step,
)
from pulse.training import embedding_prior_signal as eps
from pulse.types import EMBEDDING_DIM

N = 8
ROW_WEIGHT = 0.001           # the spec's per-row weight


def _off_centre(centre_norm: float = 0.5, spread: float = 0.3, seed: int = 0) -> torch.Tensor:
    """N codes whose mean is exactly a vector of norm ``centre_norm``, each ``spread`` (on average)
    from it -- the shape of a table the reconstruction loss left off-centre."""
    g = torch.Generator().manual_seed(seed)
    d = torch.randn(N, EMBEDDING_DIM, generator=g)
    d = d - d.mean(dim=0)
    d = d * (spread / d.norm(dim=-1).mean())
    c = torch.randn(EMBEDDING_DIM, generator=g)
    return c / c.norm() * centre_norm + d


def _table(weights: torch.Tensor | None = None) -> nn.Embedding:
    emb = nn.Embedding(N, EMBEDDING_DIM)
    with torch.no_grad():
        emb.weight.copy_(_off_centre() if weights is None else weights)
    return emb


def _ctx(emb: nn.Embedding, optimizer: torch.optim.Optimizer) -> SignalContext:
    return SignalContext(
        epoch=0, total_epochs=1, rng=np.random.default_rng(0), device=torch.device("cpu"),
        optimizer=optimizer, params=list(emb.parameters()), grad_clip=10.0,
    )


def _shape(w: torch.Tensor) -> tuple[float, float, float]:
    """(||mean||, mean ||e_i - mean||, mean ||e_i||): the centre, the per-patient spread, the norms."""
    m = w.mean(dim=0)
    return float(m.norm()), float((w - m).norm(dim=-1).mean()), float(w.norm(dim=-1).mean())


def _optimise(sig: EmbeddingPriorSignal, emb: nn.Embedding, opt: torch.optim.Optimizer, steps: int) -> None:
    """The signal ALONE, through the trainer's own accumulate -> joint clip + step path."""
    for _ in range(steps):
        ctx = _ctx(emb, opt)
        sig.compute(nn.Identity(), emb, ctx)
        joint_aux_step(ctx)


# --- the centre term pins the mean and leaves the patients alone -------------------------------

def test_centre_term_drives_the_mean_to_zero_without_flattening_patients() -> None:
    emb = _table()
    before = emb.weight.detach().clone()
    c0, spread0, _ = _shape(before)
    sig = EmbeddingPriorSignal(weight=WeightSchedule(ROW_WEIGHT))   # default center_weight
    _optimise(sig, emb, torch.optim.SGD(emb.parameters(), lr=1.0), steps=20)
    after = emb.weight.detach()
    c1, spread1, norm1 = _shape(after)

    assert c1 < 0.02 * c0, f"centre {c0:.3f} -> {c1:.4f}: not pinned"
    # Patients are not flattened: their distance from the centre holds, so the row norms settle at
    # the spread (not at 0) and the pairwise differences -- the per-patient information -- survive.
    assert spread1 > 0.97 * spread0
    assert norm1 > 0.95 * spread0
    d0, d1 = torch.cdist(before, before), torch.cdist(after, after)
    assert float((d1 - d0).abs().max() / d0.max()) < 0.02


def test_centre_term_pins_the_mean_under_adam_too() -> None:
    # The trainer's optimizer. Adam rescales gradients per coordinate, so on the signal alone the
    # fall is slower and not exactly rigid -- the thresholds are the shape of the thing, not its rate.
    emb = _table()
    c0, spread0, _ = _shape(emb.weight.detach())
    sig = EmbeddingPriorSignal(weight=WeightSchedule(ROW_WEIGHT))
    _optimise(sig, emb, torch.optim.Adam(emb.parameters(), lr=0.02), steps=40)
    c1, spread1, norm1 = _shape(emb.weight.detach())
    assert c1 < 0.2 * c0
    assert spread1 > 0.8 * spread0 and norm1 > 0.8 * spread0


def test_per_row_term_alone_does_not_centre_the_table() -> None:
    # Why the centre needs its own term: at the spec's 0.001 the row term's pull on the mean is
    # 2*w/N per step. Same table, same steps as the pin above, centre switched off (the pre-A2 signal).
    emb = _table()
    c0, _, _ = _shape(emb.weight.detach())
    sig = EmbeddingPriorSignal(weight=WeightSchedule(ROW_WEIGHT), center_weight=WeightSchedule(0.0))
    _optimise(sig, emb, torch.optim.SGD(emb.parameters(), lr=1.0), steps=20)
    c1, _, _ = _shape(emb.weight.detach())
    assert c1 > 0.95 * c0


def test_centre_gradient_moves_the_cloud_rigidly() -> None:
    # grad of ||mean||^2 is 2m/N on EVERY row: it shifts the cloud and cannot change e_i - e_j.
    emb = _table()
    m = emb.weight.detach().mean(dim=0)
    sig = EmbeddingPriorSignal(weight=WeightSchedule(0.0), center_weight=WeightSchedule(1.0))
    sig.compute(nn.Identity(), emb, _ctx(emb, torch.optim.SGD(emb.parameters(), lr=1.0)))
    g = emb.weight.grad
    assert torch.allclose(g, (2.0 * m / N).expand_as(g), atol=1e-7)
    assert torch.allclose(g, g[0].expand_as(g))


# --- two independent weights, both reported -----------------------------------------------------

def test_row_term_alone_is_the_pre_a2_signal() -> None:
    emb = _table()
    sig = EmbeddingPriorSignal(weight=WeightSchedule(ROW_WEIGHT), center_weight=WeightSchedule(0.0))
    res = sig.compute(nn.Identity(), emb, _ctx(emb, torch.optim.SGD(emb.parameters(), lr=1.0)))
    w = emb.weight.detach()
    assert torch.allclose(emb.weight.grad, 2.0 * ROW_WEIGHT * w / N, atol=1e-9)
    assert res.n_units == 1
    assert res.loss_sum == pytest.approx(float(w.pow(2).sum(dim=-1).mean()))


def test_signal_runs_on_the_centre_alone_and_is_silent_when_both_are_off() -> None:
    emb = _table()
    only_centre = EmbeddingPriorSignal(weight=WeightSchedule(0.0))
    res = only_centre.compute(nn.Identity(), emb, _ctx(emb, torch.optim.SGD(emb.parameters(), lr=1.0)))
    assert res.n_units == 1 and emb.weight.grad is not None

    emb = _table()
    off = EmbeddingPriorSignal(weight=WeightSchedule(0.0), center_weight=WeightSchedule(0.0))
    res = off.compute(nn.Identity(), emb, _ctx(emb, torch.optim.SGD(emb.parameters(), lr=1.0)))
    assert res == SignalResult() and emb.weight.grad is None


def test_both_weights_and_both_raw_terms_are_logged(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}
    real = eps.accumulate_grad

    def spy(loss, ctx, *, signal, extra=None):
        seen.update(extra or {})
        return real(loss, ctx, signal=signal, extra=extra)

    monkeypatch.setattr(eps, "accumulate_grad", spy)
    emb = _table()
    w = emb.weight.detach()
    sig = EmbeddingPriorSignal(weight=WeightSchedule(0.003), center_weight=WeightSchedule(0.7))
    res = sig.compute(nn.Identity(), emb, _ctx(emb, torch.optim.SGD(emb.parameters(), lr=1.0)))

    assert seen["weight"] == pytest.approx(0.003) and seen["center_weight"] == pytest.approx(0.7)
    assert seen["raw_loss"] == pytest.approx(float(w.pow(2).sum(dim=-1).mean()))
    assert seen["raw_centre_loss"] == pytest.approx(float(w.mean(dim=0).pow(2).sum()))
    c, spread, _ = _shape(w)
    assert res.sub_metrics["emb_centre_norm"] == pytest.approx(c)
    assert res.sub_metrics["emb_spread"] == pytest.approx(spread)
    assert {"emb_norm_mean", "emb_norm_max", "emb_std_ratio"} <= set(res.sub_metrics)


# --- the default is a constraint, in all three places it can be set ---------------------------

def test_center_weight_default_agrees_across_flag_kwarg_and_dataclass() -> None:
    flag = build_arg_parser().parse_args([]).embedding_prior_center_weight
    kwarg = inspect.signature(train).parameters["embedding_prior_center_weight"].default
    field = EmbeddingPriorSignal().center_weight
    assert flag == kwarg == field.base == eps.DEFAULT_CENTER_WEIGHT == 1.0
    assert field.at(0) == 1.0                       # on from epoch 0, no phase gate
    assert eps.DEFAULT_CENTER_WEIGHT >= 1000 * ROW_WEIGHT   # a constraint, not a nudge


def test_flag_overrides_and_zero_disables() -> None:
    parser = build_arg_parser()
    assert parser.parse_args(["--embedding-prior-center-weight", "0.25"]).embedding_prior_center_weight == 0.25
    assert parser.parse_args(["--embedding-prior-center-weight=0"]).embedding_prior_center_weight == 0.0
    assert EmbeddingPriorSignal(center_weight=WeightSchedule(0.0)).center_weight_at(0) == 0.0
