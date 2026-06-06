"""
Smoke tests for the iter-25 NaN abort path.

The behavior under test is exercised only when training is already broken,
so regular signal tests don't cover it. We simulate the three branches:

1. Finite loss + finite gradients → step taken, return loss value, no raise.
2. Non-finite loss → raise immediately (cause="loss"), no parameters touched.
3. Finite loss but non-finite gradient (forced via a custom autograd op) →
   raise after backward (cause="grad"), optimizer state cleared, parameters
   not stepped on.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pulse.training.safe_step import NaNTrainingAbort, safe_step
from pulse.training.signals import SignalContext


def _make_ctx(params: list[torch.nn.Parameter]) -> SignalContext:
    optimizer = torch.optim.SGD(params, lr=0.1)
    return SignalContext(
        epoch=7,
        total_epochs=100,
        rng=np.random.default_rng(0),
        device=torch.device("cpu"),
        optimizer=optimizer,
        params=params,
        grad_clip=10.0,
    )


def test_safe_step_finite_loss_takes_step() -> None:
    p = torch.nn.Parameter(torch.tensor([3.0]))
    ctx = _make_ctx([p])
    loss = (p * p).sum()  # 9.0, grad = 6.0
    returned = safe_step(loss, ctx, signal="unit/finite")
    assert returned == pytest.approx(9.0)
    # SGD lr=0.1, grad=6.0 → param 3.0 - 0.6 = 2.4
    assert p.item() == pytest.approx(2.4, abs=1e-5)
    # zero_grad ran (set_to_none default → grad is None or zero)
    assert p.grad is None or float(p.grad.abs().max().item()) == 0.0


def test_safe_step_nan_loss_aborts_before_backward() -> None:
    p = torch.nn.Parameter(torch.tensor([3.0]))
    ctx = _make_ctx([p])
    loss = torch.tensor(float("nan"), requires_grad=True)

    with pytest.raises(NaNTrainingAbort) as exc:
        safe_step(loss, ctx, signal="unit/nan-loss", extra={"k": 1.0})

    assert exc.value.signal == "unit/nan-loss"
    assert exc.value.epoch == 7
    assert exc.value.cause == "loss"
    assert exc.value.extra == {"k": 1.0}
    # Parameter was not touched.
    assert p.item() == pytest.approx(3.0)
    # No grad accumulated.
    assert p.grad is None


def test_safe_step_inf_loss_aborts_before_backward() -> None:
    p = torch.nn.Parameter(torch.tensor([3.0]))
    ctx = _make_ctx([p])
    loss = torch.tensor(float("inf"), requires_grad=True)

    with pytest.raises(NaNTrainingAbort) as exc:
        safe_step(loss, ctx, signal="unit/inf-loss")

    assert exc.value.cause == "loss"
    assert p.item() == pytest.approx(3.0)


class _NaNGrad(torch.autograd.Function):
    """Forward returns a finite value; backward injects NaN into the gradient."""

    @staticmethod
    def forward(ctx, x):  # type: ignore[no-untyped-def]
        return x * 1.0

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore[no-untyped-def]
        return grad_output * float("nan")


def test_safe_step_nonfinite_grad_aborts_before_step() -> None:
    p = torch.nn.Parameter(torch.tensor([3.0]))
    ctx = _make_ctx([p])
    out = _NaNGrad.apply(p)
    loss = (out * out).sum()  # finite forward
    assert torch.isfinite(loss).item()

    with pytest.raises(NaNTrainingAbort) as exc:
        safe_step(loss, ctx, signal="unit/nan-grad", extra={"k": 2.0})

    assert exc.value.cause == "grad"
    assert exc.value.signal == "unit/nan-grad"
    # extra was preserved AND grad-stats merged in
    assert exc.value.extra["k"] == 2.0
    assert "n_nan_params" in exc.value.extra
    assert exc.value.extra["n_nan_params"] >= 1.0
    # Parameter NOT stepped on (would be NaN otherwise).
    assert p.item() == pytest.approx(3.0)
    # Grads cleared by the abort path (set_to_none default).
    assert p.grad is None or float(p.grad.abs().max().item()) == 0.0
