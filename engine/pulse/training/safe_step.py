"""
Strict NaN/Inf abort for the training inner loop.

Iter 24 added skip-and-continue NaN guards in every backward site to keep
training going through transient blow-ups. The result was the iter-24 run
silently zeroing the loss for the last 35 epochs of phase 2 and producing a
NaN-filled checkpoint. ``safe_step`` replaces that pattern with a strict
abort: the first non-finite loss OR gradient halts training and raises
``NaNTrainingAbort`` with enough context for the caller to dump diagnostics
and write the most recent good checkpoint to GCS, so iter 26 can attack the
underlying numerical defect rather than re-running blind.

Strict here means: exactly the opposite of "robust to noise". NaN at this
stage is diagnostic information; preserving it is the point.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from .signals import SignalContext


class NaNTrainingAbort(RuntimeError):
    """Raised by ``safe_step`` on the first non-finite loss or gradient.

    The trainer catches this once at the epoch-loop level, dumps the
    ``signal``/``epoch``/``cause``/``loss_value``/``extra`` fields to a
    JSON artifact, uploads the rolling last-good checkpoint, and exits.
    """

    def __init__(
        self,
        *,
        signal: str,
        epoch: int,
        cause: str,
        loss_value: float,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.signal = signal
        self.epoch = epoch
        self.cause = cause  # "loss" or "grad"
        self.loss_value = loss_value
        self.extra: dict[str, Any] = dict(extra or {})
        super().__init__(
            f"NaNTrainingAbort: signal={signal} epoch={epoch} cause={cause} "
            f"loss={loss_value!r}",
        )


def _grad_stats(params: list[torch.nn.Parameter]) -> tuple[bool, dict[str, float]]:
    """Inspect ``param.grad`` for finiteness and summary norms.

    Returns ``(all_finite, stats)``. Stats are summed across all parameters
    that currently have a gradient — finite-only contributions go into the
    L2 norm so a single Inf doesn't drown the rest of the gradient signal.
    """
    n_params = 0
    n_finite_params = 0
    n_nan_params = 0
    n_inf_params = 0
    max_abs = 0.0
    sum_sq = 0.0
    for p in params:
        if p.grad is None:
            continue
        n_params += 1
        g = p.grad.detach()
        if torch.isfinite(g).all():
            n_finite_params += 1
            ma = float(g.abs().max().item())
            if ma > max_abs:
                max_abs = ma
            sum_sq += float(g.pow(2).sum().item())
        else:
            if torch.isnan(g).any():
                n_nan_params += 1
            if torch.isinf(g).any():
                n_inf_params += 1
    grad_norm = math.sqrt(sum_sq) if sum_sq > 0 else 0.0
    return (n_nan_params == 0 and n_inf_params == 0), {
        "n_params_with_grad": float(n_params),
        "n_finite_params": float(n_finite_params),
        "n_nan_params": float(n_nan_params),
        "n_inf_params": float(n_inf_params),
        "max_abs_grad_finite": max_abs,
        "grad_l2_finite_only": grad_norm,
    }


def safe_step(
    loss: torch.Tensor,
    ctx: SignalContext,
    *,
    signal: str,
    extra: dict[str, Any] | None = None,
) -> float:
    """Backward + clip + step + zero_grad with strict NaN/Inf abort.

    Caller passes the *already-weighted* loss (so the gradient magnitude
    matches what the optimizer should actually see). Two abort conditions:

    1. ``loss`` is non-finite — abort with ``cause="loss"`` BEFORE backward.
       No parameters touched.
    2. Any ``param.grad`` is non-finite after backward — abort with
       ``cause="grad"`` AFTER backward but BEFORE step. ``optimizer.zero_grad()``
       is called first so the optimizer state is clean for the abort handler
       (otherwise corrupt grads would persist into any post-abort save).

    On success: returns ``float(loss.detach().item())``.
    """
    if not torch.isfinite(loss):
        raise NaNTrainingAbort(
            signal=signal,
            epoch=ctx.epoch,
            cause="loss",
            loss_value=float(loss.detach().item()),
            extra=extra,
        )
    loss.backward()
    all_finite, gstats = _grad_stats(ctx.params)
    if not all_finite:
        ctx.optimizer.zero_grad()
        merged = dict(extra or {})
        merged.update(gstats)
        raise NaNTrainingAbort(
            signal=signal,
            epoch=ctx.epoch,
            cause="grad",
            loss_value=float(loss.detach().item()),
            extra=merged,
        )
    nn.utils.clip_grad_norm_(ctx.params, max_norm=ctx.grad_clip)
    ctx.optimizer.step()
    ctx.optimizer.zero_grad()
    return float(loss.detach().item())
