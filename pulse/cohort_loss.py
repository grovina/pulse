"""
Differentiable cohort statistic losses for population-level priors.

Rolls the modular model forward under each arm's protocol per sampled
patient embedding, extracts a scalar statistic per arm (mean, peak,
soft-argmax time), and penalizes the squared z-score versus the literature
target: ``((predicted - target) / sigma) ** 2``.

This is the PRD's cohort / summary-statistic supervision in concrete
form — quantitative effect sizes, not just orderings. Slope-vs-dose
statistics live in their own dedicated signal (``pulse.dose_response``)
because they need a differentiable soft-peak and a focused weight.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .knowledge.cohort_types import (
    CohortArmSpec,
    CohortStatisticSpec,
    StatisticKind,
    StatisticWindow,
)
from .model import ModularPhysiologyNetwork, integrate, precompute_gut_outputs
from .modules.gut import MealEvent
from .types import MARKER_INDEX, NORM_CENTER

__all__ = [
    "cohort_statistic_epoch_loss",
    "cohort_statistic_loss_group",
    "cohort_statistic_loss_one_spec",
    "norm_center_initial_state",
]

InitialStateFn = Callable[[CohortStatisticSpec], torch.Tensor]


def norm_center_initial_state(
    device: torch.device | str = "cpu",
) -> InitialStateFn:
    """Default initial-state factory: spec-agnostic ``NORM_CENTER`` tensor."""
    state = torch.tensor(NORM_CENTER, dtype=torch.float32, device=device)
    return lambda _spec: state


def _meals_from_spec(tuples: tuple[tuple[float, float, float, float], ...]) -> list[MealEvent]:
    return [
        MealEvent(time=float(t), carbs=float(c), fats=float(f), proteins=float(p))
        for t, c, f, p in tuples
    ]


def _optional_series_tensor(
    name: str,
    series: tuple[float, ...] | None,
    n_steps: int,
    device: torch.device,
) -> torch.Tensor | None:
    if series is None:
        return None
    if len(series) != n_steps:
        raise ValueError(f"CohortArmSpec {name} length {len(series)} must equal duration_min {n_steps}")
    return torch.tensor(series, dtype=torch.float32, device=device)


def _validate_window(spec_name: str, w: StatisticWindow, duration_min: int) -> None:
    if w.start_min < 0 or w.end_min > duration_min or w.start_min >= w.end_min:
        raise ValueError(
            f"{spec_name}: window [{w.start_min}, {w.end_min}) invalid for duration {duration_min}",
        )


def _rollout_arm_states(
    model: ModularPhysiologyNetwork,
    embeddings: torch.Tensor,
    arm: CohortArmSpec,
    state_batch: torch.Tensor,
) -> torch.Tensor:
    """Roll one arm forward for ``N`` (embedding, initial-state) pairs.

    ``embeddings`` is ``[N, EMB]`` and ``state_batch`` is ``[N, STATE_DIM]``
    — one starting state per batch row (unlike ``_rollout_arm_batched``,
    which broadcasts a single state across all rows). The arm's protocol
    (meals, sleep/wake, activity, duration) is shared across the batch, as
    ``integrate`` requires. Returns ``[N, T, STATE_DIM]``.

    This is the primitive behind cross-spec batching (iter 79): several
    cohort specs sharing one arm protocol but each with its own cold-init
    state are rolled in a single ``integrate`` call by stacking their states
    here, collapsing N sequential rollouts of the same long protocol into one.
    """
    n_steps = arm.duration_min
    t0 = float(arm.start_hour * 60.0)
    meals = _meals_from_spec(arm.meals)
    device = state_batch.device
    sw = _optional_series_tensor(f"{arm.label}.sleep_wake", arm.sleep_wake, n_steps, device)
    act = _optional_series_tensor(f"{arm.label}.activity", arm.activity, n_steps, device)
    gut = precompute_gut_outputs(
        model, embeddings, n_steps,
        dt=1.0, start_time_minutes=t0, meals=meals,
    )
    # Iter 68 round 3: gradient checkpointing on the cohort rollout. The
    # iter-67 saga turned out to have TWO memory eaters, not one. Round 1+2
    # fixed cold-distill (the visible one in watchdog stack traces); round 3
    # fixes cohort_statistic, which the intra-epoch memprof in r2 caught
    # red-handed: RSS jumped 879 MB → 26 GB inside one cohort_statistic
    # compute() call. With 31 specs × 1-3 arms × 13 embeddings × up to 1440
    # steps unchunked, the held activation graph is ~25 GB. Checkpointing at
    # sqrt(n_steps) segments mirrors the cold-distill fix and drops this by
    # ~sqrt(n_steps)×. Same back-compat guarantees: bench-time integrate has
    # default checkpoint_segments=0 so prediction quality is unaffected.
    ckpt_segs = max(1, int(n_steps**0.5))
    return integrate(
        model, state_batch, embeddings, n_steps,
        dt=1.0, start_time_minutes=t0, meals=meals,
        sleep_wake=sw, activity=act,
        gut_outputs=gut,
        checkpoint_segments=ckpt_segs,
    )


def _rollout_arm_batched(
    model: ModularPhysiologyNetwork,
    embeddings: torch.Tensor,
    arm: CohortArmSpec,
    initial_state: torch.Tensor,
) -> torch.Tensor:
    """Roll one arm forward for ``B`` embeddings sharing one initial state.

    ``embeddings`` is ``[B, EMB]``; the same protocol AND initial state are
    applied to every batch member. Returns ``[B, T, STATE_DIM]``. Thin
    wrapper over ``_rollout_arm_states`` that broadcasts the single state.
    """
    state_b = initial_state.unsqueeze(0).expand(int(embeddings.shape[0]), -1)
    return _rollout_arm_states(model, embeddings, arm, state_b)


def _arm_window(spec: CohortStatisticSpec, arm_idx: int) -> StatisticWindow:
    if spec.per_arm_windows is None:
        return spec.window
    return spec.per_arm_windows[arm_idx]


def _arm_statistic_batched(
    traj: torch.Tensor,
    marker_idx: int,
    window: StatisticWindow,
    kind: StatisticKind,
    softargmax_beta: float,
) -> torch.Tensor:
    """Extract one scalar per batch member: ``[B]``."""
    series = traj[:, window.start_min:window.end_min, marker_idx]  # [B, W]
    if kind in (StatisticKind.MEAN_IN_WINDOW, StatisticKind.DELTA_MEANS):
        return series.mean(dim=1)
    if kind in (StatisticKind.PEAK_VALUE, StatisticKind.DELTA_PEAKS):
        return series.max(dim=1).values
    if kind == StatisticKind.TIME_TO_PEAK:
        n = int(series.shape[1])
        weights = F.softmax(softargmax_beta * series, dim=1)
        idx = torch.arange(n, device=series.device, dtype=series.dtype)
        return (weights * idx).sum(dim=1)
    raise ValueError(f"Unknown statistic kind: {kind}")


def cohort_statistic_loss_one_spec(
    model: nn.Module,
    embeddings_to_supervise: list[torch.Tensor],
    spec: CohortStatisticSpec,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, float, float]:
    """Mean Gaussian-z² discrepancy across the supplied embeddings.

    Embeddings are stacked into one batched forward pass per arm (B = len
    of ``embeddings_to_supervise``), so each arm calls ``integrate`` once
    instead of B times. Returns (loss_tensor, predicted_mean_detached,
    residual_z_detached). The caller decides which embeddings deserve
    supervision (typically a sampled subset of patient embeddings plus the
    zero "default" embedding the benchmark uses) — see
    ``training.embedding_sampler``.
    """
    if not embeddings_to_supervise:
        zero = torch.tensor(0.0, device=initial_state.device)
        return zero, 0.0, 0.0

    embs = torch.stack(embeddings_to_supervise, dim=0)  # [B, EMB]
    mi = MARKER_INDEX[spec.marker_id]

    per_arm: list[torch.Tensor] = []
    for arm_idx, arm in enumerate(spec.arms):
        win = _arm_window(spec, arm_idx)
        _validate_window(spec.name, win, arm.duration_min)
        traj = _rollout_arm_batched(model, embs, arm, initial_state)  # [B, T, STATE]
        per_arm.append(
            _arm_statistic_batched(traj, mi, win, spec.kind, spec.softargmax_beta),
        )  # [B]

    if spec.kind in (StatisticKind.DELTA_MEANS, StatisticKind.DELTA_PEAKS):
        pred = per_arm[1] - per_arm[0]  # [B]
    elif spec.kind in (
        StatisticKind.MEAN_IN_WINDOW,
        StatisticKind.PEAK_VALUE,
        StatisticKind.TIME_TO_PEAK,
    ):
        pred = per_arm[0]  # [B]
    else:
        raise ValueError(f"Unknown statistic kind: {spec.kind}")

    z = (pred - spec.target) / spec.sigma  # [B]
    loss = z.pow(2).mean()
    pred_mean = float(pred.detach().mean().item())
    z_mean = (pred_mean - spec.target) / spec.sigma
    return loss, pred_mean, z_mean


def cohort_statistic_loss_group(
    model: nn.Module,
    embeddings_to_supervise: list[torch.Tensor],
    specs: list[CohortStatisticSpec],
    initial_states: list[torch.Tensor],
) -> dict[str, tuple[torch.Tensor, float, float]]:
    """Batched per-spec loss for specs that share one arm protocol (iter 79).

    Every spec in ``specs`` must have identical ``arms`` (by value) — the
    grouping condition the signal enforces. They may differ freely in
    ``marker_id`` / ``window`` / ``kind`` / ``target`` / ``sigma`` and in
    their ``initial_states`` (cold-init is seeded per spec). For each arm we
    stack ``[S × B]`` rows (spec s's init state repeated across the B
    supervised embeddings) and call ``integrate`` ONCE, instead of one
    rollout per spec — collapsing e.g. the four 2880-step sleep specs or the
    five 1440-step fast specs into a single long rollout each.

    Returns ``{spec.name: (loss_tensor, predicted_mean, residual_z)}`` — the
    same per-spec triple ``cohort_statistic_loss_one_spec`` returns. The loss
    tensors share one autograd graph (the batched rollout), so the caller
    sums the weighted per-spec losses and does ONE backward for the group:
    gradient-identical to per-spec backward by linearity, with the per-spec
    memory bound preserved (one group's graph live at a time).
    """
    if not embeddings_to_supervise or not specs:
        return {}
    arms = specs[0].arms
    assert all(s.arms == arms for s in specs), "group specs must share arms"

    embs = torch.stack(embeddings_to_supervise, dim=0)  # [B, EMB]
    B = int(embs.shape[0])
    S = len(specs)
    # Row r = s*B + b → spec s's init state, embedding b.
    emb_batch = embs.repeat(S, 1)  # [S*B, EMB]
    init_stack = torch.stack(list(initial_states), dim=0)  # [S, STATE]
    state_batch = init_stack.repeat_interleave(B, dim=0)  # [S*B, STATE]

    arm_trajs: list[torch.Tensor] = [
        _rollout_arm_states(model, emb_batch, arm, state_batch)  # [S*B, T, STATE]
        for arm in arms
    ]

    out: dict[str, tuple[torch.Tensor, float, float]] = {}
    for s_idx, spec in enumerate(specs):
        mi = MARKER_INDEX[spec.marker_id]
        rows = slice(s_idx * B, (s_idx + 1) * B)
        per_arm: list[torch.Tensor] = []
        for arm_idx, arm in enumerate(arms):
            win = _arm_window(spec, arm_idx)
            _validate_window(spec.name, win, arm.duration_min)
            per_arm.append(
                _arm_statistic_batched(
                    arm_trajs[arm_idx][rows], mi, win, spec.kind, spec.softargmax_beta,
                ),
            )  # [B]
        if spec.kind in (StatisticKind.DELTA_MEANS, StatisticKind.DELTA_PEAKS):
            pred = per_arm[1] - per_arm[0]
        else:
            pred = per_arm[0]
        z = (pred - spec.target) / spec.sigma
        loss = z.pow(2).mean()
        pred_mean = float(pred.detach().mean().item())
        z_mean = (pred_mean - spec.target) / spec.sigma
        out[spec.name] = (loss, pred_mean, z_mean)
    return out


def cohort_statistic_epoch_loss(
    model: nn.Module,
    embeddings_to_supervise: list[torch.Tensor],
    specs: list[CohortStatisticSpec],
    device: torch.device | str,
    initial_state_fn: InitialStateFn | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Weighted average loss across all specs and the supplied embeddings.

    ``initial_state_fn`` returns the integration starting state for a given
    spec. Defaults to ``norm_center_initial_state`` when not provided; pass a
    cold-model-derived factory to mirror benchmark conditions.

    Returns (loss_tensor, per_spec_z_residuals).
    """
    if not specs or not embeddings_to_supervise:
        return torch.tensor(0.0, device=device), {}
    init_fn = initial_state_fn if initial_state_fn is not None else norm_center_initial_state(device)
    total = torch.tensor(0.0, device=device)
    weight_sum = 0.0
    z_by_spec: dict[str, float] = {}
    for spec in specs:
        loss, _pred, z = cohort_statistic_loss_one_spec(
            model, embeddings_to_supervise, spec, init_fn(spec),
        )
        total = total + spec.weight * loss
        weight_sum += spec.weight
        z_by_spec[spec.name] = z
    return total / max(weight_sum, 1e-8), z_by_spec
