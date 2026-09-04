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
    total_norm = nn.utils.clip_grad_norm_(ctx.params, max_norm=ctx.grad_clip)
    ctx.traj_grad_norms.append(float(total_norm))
    ctx.optimizer.step()
    ctx.optimizer.zero_grad()
    return float(loss.detach().item())


def _delta_norm_and_clip(
    ctx: SignalContext,
    snapshot: list[torch.Tensor | None],
    *,
    signal: str,
) -> float:
    """Norm of THIS signal's gradient contribution (grad minus the snapshot
    taken before its backward), recorded per signal; rescaled in place to
    ``ctx.aux_signal_clip`` when that is set (iter 97, review 4.11)."""
    sum_sq = 0.0
    deltas: list[tuple[torch.nn.Parameter, torch.Tensor | None, torch.Tensor]] = []
    for p, g0 in zip(ctx.params, snapshot):
        if p.grad is None:
            continue
        d = p.grad.detach() if g0 is None else p.grad.detach() - g0
        sum_sq += float(d.pow(2).sum().item())
        deltas.append((p, g0, d))
    norm = math.sqrt(sum_sq)
    ctx.record_aux_grad(signal, norm)
    clip = float(ctx.aux_signal_clip)
    if clip > 0.0 and norm > clip:
        f = clip / (norm + 1e-12)
        for p, g0, d in deltas:
            base = torch.zeros_like(p.grad) if g0 is None else g0
            p.grad.copy_(base + d * f)
    return norm


# ---------------------------------------------------------------------------
# Joint auxiliary-signal accumulation (iter 78).
#
# ``safe_step`` is per-signal: backward + clip_grad_norm_(grad_clip) + step. When
# every auxiliary signal calls it, each takes a *solo* clipped step in sequence,
# so (a) a signal's relative weight is erased whenever its grad-norm exceeds the
# clip (30 iters of weight sweeps went byte-identical for exactly this reason),
# and (b) one unstable signal can both fight the others and unilaterally abort
# the run (iter 77 died at epoch 39 when ``postprandial_recovery``'s long rollout
# ran away and its gradient went NaN — before phase 2 ever engaged).
#
# The fix: auxiliary signals ACCUMULATE their (already-weighted) gradient via
# ``accumulate_grad`` / ``finalize_aux_accumulation`` without stepping, and the
# trainer applies ONE ``joint_aux_step`` (clip + step) over the combined
# gradient per epoch. Weights then compose (the joint clip scales the whole, so
# relative magnitudes survive), signals stop fighting sequentially, and per-epoch
# aux drift drops from ~10 clipped steps to one. The trajectory signal keeps its
# own per-window SGD (the main data fit, ~84 steps/epoch) — folding it in would
# cut its steps ~90x and be untrainable.
#
# Isolation policy: a single aux signal whose loss or gradient is non-finite has
# its contribution rolled back and dropped (logged), not aborted — so it can
# neither poison the joint gradient nor kill a run before it produces numbers.
# Strict abort is preserved for the trajectory signal (its safe_step) and, as
# defense-in-depth, for the combined gradient in ``joint_aux_step``.
# ---------------------------------------------------------------------------


def grad_snapshot(ctx: SignalContext) -> list[torch.Tensor | None]:
    """Clone the current ``.grad`` buffers so one aux signal's contribution can
    be rolled back if it turns out non-finite. The model is small (~0.2M params)
    so the clone is cheap relative to a rollout."""
    return [None if p.grad is None else p.grad.detach().clone() for p in ctx.params]


def _rollback_grad(ctx: SignalContext, snapshot: list[torch.Tensor | None]) -> None:
    """Restore ``.grad`` to a prior snapshot, undoing the most recent signal's
    accumulation while preserving every earlier aux signal's contribution."""
    for p, g0 in zip(ctx.params, snapshot):
        if g0 is None:
            p.grad = None
        elif p.grad is None:
            p.grad = g0.clone()
        else:
            p.grad.copy_(g0)


def _log_skip(signal: str, ctx: SignalContext, cause: str, detail: str) -> None:
    print(
        f"[SKIP-NONFINITE] signal={signal} epoch={ctx.epoch} cause={cause} "
        f"{detail} (contribution dropped this epoch)",
        flush=True,
    )


def accumulate_grad(
    loss: torch.Tensor,
    ctx: SignalContext,
    *,
    signal: str,
    extra: dict[str, Any] | None = None,  # noqa: ARG001 — accepted for safe_step parity
) -> float:
    """Backward-accumulate ONE single-tensor aux signal's (already weighted)
    gradient into the shared buffers WITHOUT clipping/stepping/zeroing.

    Per-signal isolation: if ``loss`` or the resulting gradient is non-finite,
    this signal's contribution is rolled back and dropped (logged), and the run
    continues. On success, marks ``ctx.aux_accumulated`` so the trainer applies
    one joint step. Returns the detached loss value (always, for logging).
    """
    lv = float(loss.detach().item())
    if not math.isfinite(lv):
        _log_skip(signal, ctx, "loss", f"loss={lv!r}")
        return lv
    snapshot = grad_snapshot(ctx)
    loss.backward()
    all_finite, gstats = _grad_stats(ctx.params)
    if not all_finite:
        _rollback_grad(ctx, snapshot)
        _log_skip(
            signal, ctx, "grad",
            f"n_nan={gstats['n_nan_params']:.0f} n_inf={gstats['n_inf_params']:.0f}",
        )
        return lv
    _delta_norm_and_clip(ctx, snapshot, signal=signal)
    ctx.aux_accumulated = True
    ctx.aux_steps_by_signal[signal] = ctx.aux_steps_by_signal.get(signal, 0) + 1
    return lv


def finalize_aux_accumulation(
    ctx: SignalContext,
    snapshot: list[torch.Tensor | None],
    *,
    signal: str,
) -> bool:
    """Finalize an aux signal that accumulated gradient via several per-term
    ``backward()`` calls (cohort, physiology_rules — they backward per spec/rule
    and drop each graph to cap memory). ``snapshot`` must have been taken with
    ``grad_snapshot`` before the signal's first backward. If the combined
    gradient is non-finite, roll this signal's whole contribution back and drop
    it (logged); otherwise mark the context accumulated. Returns whether it was
    kept.
    """
    all_finite, gstats = _grad_stats(ctx.params)
    if not all_finite:
        _rollback_grad(ctx, snapshot)
        _log_skip(
            signal, ctx, "grad",
            f"n_nan={gstats['n_nan_params']:.0f} n_inf={gstats['n_inf_params']:.0f}",
        )
        return False
    _delta_norm_and_clip(ctx, snapshot, signal=signal)
    ctx.aux_accumulated = True
    ctx.aux_steps_by_signal[signal] = ctx.aux_steps_by_signal.get(signal, 0) + 1
    return True


def joint_aux_step(ctx: SignalContext, *, signal: str = "joint_aux") -> dict[str, float]:
    """One clip + step + zero over the auxiliary signals' accumulated gradient.

    Defense-in-depth strict abort: individual aux signals already isolate their
    own NaNs, so a non-finite *combined* gradient here is unexpected and treated
    as a real divergence (raises ``NaNTrainingAbort`` for the trainer to dump).
    """
    all_finite, gstats = _grad_stats(ctx.params)
    if not all_finite:
        ctx.optimizer.zero_grad()
        raise NaNTrainingAbort(
            signal=signal,
            epoch=ctx.epoch,
            cause="grad",
            loss_value=float("nan"),
            extra=gstats,
        )
    total_norm = nn.utils.clip_grad_norm_(ctx.params, max_norm=ctx.grad_clip)
    gstats["joint_grad_norm_pre_clip"] = float(total_norm)
    ctx.optimizer.step()
    ctx.optimizer.zero_grad()
    ctx.aux_accumulated = False
    return gstats
