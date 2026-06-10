"""
Tests for iter-78 joint auxiliary-signal accumulation.

Auxiliary signals no longer take a solo ``backward + clip + step``. They
``accumulate_grad`` (or ``finalize_aux_accumulation``) their weighted gradient
without stepping, and the trainer applies one ``joint_aux_step`` over the
combined gradient. The properties under test:

1. ``accumulate_grad`` accumulates gradient WITHOUT stepping; weights compose
   (two signals' gradients sum, rather than each being clipped to the same norm
   and erasing its weight — the iter <78 disease).
2. A single aux signal with a non-finite loss / gradient is rolled back and
   dropped (isolation), the run continues, and the surviving signals' gradient
   is intact.
3. ``joint_aux_step`` strictly aborts on a non-finite *combined* gradient.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pulse.training.safe_step import (
    NaNTrainingAbort,
    accumulate_grad,
    finalize_aux_accumulation,
    grad_snapshot,
    joint_aux_step,
)
from pulse.training.signals import SignalContext


def _make_ctx(params: list[torch.nn.Parameter], *, grad_clip: float = 1e9) -> SignalContext:
    optimizer = torch.optim.SGD(params, lr=0.1)
    return SignalContext(
        epoch=7,
        total_epochs=100,
        rng=np.random.default_rng(0),
        device=torch.device("cpu"),
        optimizer=optimizer,
        params=params,
        grad_clip=grad_clip,
    )


class _NaNGrad(torch.autograd.Function):
    """Finite forward, NaN-injecting backward (mirrors test_safe_step)."""

    @staticmethod
    def forward(ctx, x):  # type: ignore[no-untyped-def]
        return x * 1.0

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore[no-untyped-def]
        return grad_output * float("nan")


def test_accumulate_grad_does_not_step() -> None:
    p = torch.nn.Parameter(torch.tensor([3.0]))
    ctx = _make_ctx([p])
    lv = accumulate_grad((p * p).sum(), ctx, signal="unit/a")
    assert lv == pytest.approx(9.0)
    # No optimizer step taken — the parameter is untouched.
    assert p.item() == pytest.approx(3.0)
    # Gradient accumulated (grad = 2*p = 6.0), and the context is marked.
    assert p.grad is not None
    assert float(p.grad.item()) == pytest.approx(6.0)
    assert ctx.aux_accumulated is True


def test_weights_compose_then_joint_step() -> None:
    p = torch.nn.Parameter(torch.tensor([3.0]))
    ctx = _make_ctx([p])
    # Two signals at different weights. grad = 2*p*(wa+wb), NOT clipped-equal.
    accumulate_grad(0.2 * (p * p).sum(), ctx, signal="unit/a")
    accumulate_grad(0.5 * (p * p).sum(), ctx, signal="unit/b")
    assert float(p.grad.item()) == pytest.approx(6.0 * (0.2 + 0.5))
    joint_aux_step(ctx)
    # SGD lr=0.1: p -= 0.1 * grad
    assert p.item() == pytest.approx(3.0 - 0.1 * 6.0 * 0.7, abs=1e-6)
    # Joint step zeroed grads afterwards.
    assert p.grad is None or float(p.grad.abs().max().item()) == 0.0


def test_accumulate_grad_isolates_nonfinite_signal() -> None:
    p = torch.nn.Parameter(torch.tensor([3.0]))
    ctx = _make_ctx([p])
    accumulate_grad((p * p).sum(), ctx, signal="unit/good")  # grad 6.0
    good_grad = float(p.grad.item())
    # A signal whose backward injects NaN — must be rolled back, not aborted.
    bad = (_NaNGrad.apply(p) ** 2).sum()
    lv = accumulate_grad(bad, ctx, signal="unit/bad")
    assert np.isfinite(lv)  # returns the (finite) loss value, no raise
    # The good signal's gradient survived intact; the bad one's was dropped.
    assert float(p.grad.item()) == pytest.approx(good_grad)
    # Joint step still works on the surviving gradient.
    joint_aux_step(ctx)
    assert p.item() == pytest.approx(3.0 - 0.1 * good_grad, abs=1e-6)


def test_accumulate_grad_nonfinite_loss_skipped() -> None:
    p = torch.nn.Parameter(torch.tensor([3.0]))
    ctx = _make_ctx([p])
    lv = accumulate_grad(torch.tensor(float("nan"), requires_grad=True), ctx, signal="unit/nan")
    assert np.isnan(lv)
    assert p.grad is None  # nothing backed through
    assert ctx.aux_accumulated is False


def test_finalize_rolls_back_nonfinite_accumulation() -> None:
    p = torch.nn.Parameter(torch.tensor([3.0]))
    ctx = _make_ctx([p])
    # Prior aux signal already accumulated a finite gradient.
    accumulate_grad((p * p).sum(), ctx, signal="unit/prior")
    snap = grad_snapshot(ctx)
    # An incremental signal backward-accumulates a NaN on top.
    (_NaNGrad.apply(p) ** 2).sum().backward()
    kept = finalize_aux_accumulation(ctx, snap, signal="unit/incremental")
    assert kept is False
    # Rolled back to the prior signal's gradient (6.0), not NaN.
    assert float(p.grad.item()) == pytest.approx(6.0)


def test_joint_step_aborts_on_nonfinite_combined_grad() -> None:
    p = torch.nn.Parameter(torch.tensor([3.0]))
    ctx = _make_ctx([p])
    p.grad = torch.tensor([float("inf")])
    with pytest.raises(NaNTrainingAbort) as exc:
        joint_aux_step(ctx)
    assert exc.value.cause == "grad"
    assert exc.value.signal == "joint_aux"
