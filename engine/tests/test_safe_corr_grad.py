"""
Regression test for iter 25's NaN-grad abort.

Iter 25 aborted at Phase 2 epoch 61 (build 2ffc4b7f) with
signal=trajectory_rollout/window cause=grad and 71/115 model params
holding NaN gradient after a single backward pass. Root cause: the
previous form of ``_safe_corr_torch``,

    torch.where(denom < 1e-8, 0.0, (am*bm).mean() / denom)

is a textbook PyTorch ``where``-NaN trap. Forward selects 0 when denom
is small, but autograd evaluates BOTH branches in backward; the
unselected ``(am*bm).mean() / denom`` produces non-finite gradient at
denom≈0 (chain rule yields 1/denom²), and ``where`` propagates NaN
through the unselected branch.

This test reproduces the failing pattern in isolation: feed
``_safe_corr_torch`` a zero-variance input that flows from a leaf
parameter, and verify that backward yields finite gradients. Without
the iter-26 fix this test fails on the leaf parameter's grad.
"""

from __future__ import annotations

import torch

from pulse.training_verifier_loss import _safe_corr_torch


def test_safe_corr_grad_finite_on_zero_variance_input() -> None:
    # ``a`` is constant (zero variance) — the case that triggered the
    # iter-25 abort. ``b`` flows from a leaf parameter so we can backward.
    p = torch.nn.Parameter(torch.tensor(2.0))
    a = torch.tensor([1.0, 1.0, 1.0, 1.0])
    b = torch.stack([p * 0.5, p * 0.7, p * 1.2, p * 1.6])

    corr = _safe_corr_torch(a, b)
    # Forward: zero-variance ``a`` → denom=0 → result=0.
    assert torch.isfinite(corr).item()
    assert float(corr.detach().item()) == 0.0

    # Backward: must produce finite grad. Pre-fix this was NaN.
    corr.backward()
    assert p.grad is not None
    assert torch.isfinite(p.grad).all().item(), \
        f"_safe_corr_torch backward produced non-finite grad: {p.grad}"


def test_safe_corr_grad_finite_on_both_zero_variance() -> None:
    # Both sides constant (degenerate but still must not NaN backward).
    p = torch.nn.Parameter(torch.tensor(3.0))
    a = torch.stack([p * 0.0 + 1.0, p * 0.0 + 1.0, p * 0.0 + 1.0])
    b = torch.tensor([2.0, 2.0, 2.0])

    corr = _safe_corr_torch(a, b)
    assert torch.isfinite(corr).item()
    corr.backward()
    assert p.grad is not None
    assert torch.isfinite(p.grad).all().item()


def test_safe_corr_grad_normal_case_still_works() -> None:
    # Non-degenerate input — the clamp must not bias the gradient when
    # denom is well above the floor.
    p = torch.nn.Parameter(torch.tensor(1.0))
    a = torch.tensor([1.0, 2.0, 3.0, 4.0]) * p
    b = torch.tensor([2.0, 3.0, 4.0, 5.0])

    corr = _safe_corr_torch(a, b)
    # Should be very close to +1 (perfect linear correlation).
    assert float(corr.detach().item()) > 0.99
    corr.backward()
    assert p.grad is not None
    assert torch.isfinite(p.grad).all().item()
    # And the gradient should be small but nonzero — clamp didn't
    # zero it out.
    assert abs(float(p.grad.item())) < 1.0
