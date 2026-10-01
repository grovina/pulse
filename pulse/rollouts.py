"""
Many protocols, one rollout.

The student's rollout cost is per-step overhead, not batch width: on CPU a
60-row Euler step costs ~1.2x a 3-row one (``docs/training-efficiency.md``).
So a training signal that rolls out k protocols — k cohort arms, k carb
doses, k (embedding, dose) pairs — pays ~k times what it needs to if it calls
``integrate`` once per protocol. ``rollout_many`` stacks them into ONE batched
``integrate``: every row keeps its own clock, meals (its own gut and duodenal
windows), sleep/activity tape and length (``integrate(active_steps=)``), and
rows never interact, so each request's trajectory — and any loss on it — is
what rolling it out alone gives, to float rounding. Summing the requests'
losses into one backward is gradient-identical to a backward per request.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .model import (
    ModularPhysiologyNetwork, integrate, precompute_duodenal_outputs, precompute_gut_outputs,
)
from .modules.gut import MealEvent


@dataclass(frozen=True, eq=False)
class RolloutRequest:
    """One protocol rolled out for ``states.shape[0]`` rows.

    ``sleep_wake`` / ``activity`` are ``[duration_min]`` series shared by the rows,
    or ``None`` for the model's learned default. ``gut_outputs`` /
    ``duodenal_outputs`` (``[duration_min, 4]`` / ``[duration_min, 3]``, shared by
    the rows) replace the student's own gut and duodenal kernels on ``meals`` —
    for a protocol driven by a reference absorption tape.
    """

    duration_min: int
    start_minutes: float
    meals: Sequence[MealEvent]
    embeddings: torch.Tensor  # [N, EMB]
    states: torch.Tensor  # [N, STATE_DIM]
    sleep_wake: torch.Tensor | None = None
    activity: torch.Tensor | None = None
    gut_outputs: torch.Tensor | None = None
    duodenal_outputs: torch.Tensor | None = None


def rollout_many(
    model: ModularPhysiologyNetwork,
    requests: Sequence[RolloutRequest],
    *,
    checkpoint_segments: int = 0,
    isolate_nonfinite: bool = False,
) -> list[torch.Tensor | None]:
    """Roll every request out in one batched ``integrate`` → ``[N_i, T_i, STATE_DIM]`` each.

    A single request is rolled out exactly as a direct ``integrate`` call would.
    ``checkpoint_segments`` is passed through (it applies when the embeddings or
    states require grad — ``integrate``'s rule).

    ``isolate_nonfinite``: a request whose trajectory is not finite comes back as
    ``None`` and the others are re-rolled WITHOUT it. Rolled alone, a diverged
    protocol's graph is simply never backpropagated; sharing a batch, its NaN
    activations would still reach every weight gradient (0 · NaN in the batched
    matmuls' reductions). The re-roll only happens when something diverged.
    """
    out = _rollout_many(model, requests, checkpoint_segments)
    if not isolate_nonfinite:
        return out
    bad = [i for i, t in enumerate(out) if not bool(torch.isfinite(t).all())]
    if not bad:
        return out
    keep = [i for i in range(len(requests)) if i not in bad]
    redo = _rollout_many(model, [requests[i] for i in keep], checkpoint_segments)
    result: list[torch.Tensor | None] = [None] * len(requests)
    for i, t in zip(keep, redo):
        result[i] = t
    return result


def _rollout_many(
    model: ModularPhysiologyNetwork,
    requests: Sequence[RolloutRequest],
    checkpoint_segments: int,
) -> list[torch.Tensor]:
    if not requests:
        return []
    if len(requests) == 1:
        r = requests[0]
        meals = list(r.meals)
        gut = r.gut_outputs
        if gut is None:
            gut = precompute_gut_outputs(model, r.embeddings, r.duration_min, dt=1.0, meals=meals)
        else:
            gut = gut.unsqueeze(0).expand(int(r.states.shape[0]), -1, -1)
        return [integrate(
            model, r.states, r.embeddings, r.duration_min,
            dt=1.0, start_time_minutes=float(r.start_minutes), meals=meals,
            sleep_wake=r.sleep_wake, activity=r.activity,
            gut_outputs=gut,
            duodenal_outputs=r.duodenal_outputs,
            checkpoint_segments=checkpoint_segments,
        )]

    n_steps = max(int(r.duration_min) for r in requests)
    device = requests[0].states.device
    sizes = [int(r.states.shape[0]) for r in requests]

    def _per_row(values: list[float], dtype: torch.dtype) -> torch.Tensor:
        return torch.cat([
            torch.full((n,), v, dtype=dtype, device=device) for v, n in zip(values, sizes)
        ])

    emb = torch.cat([r.embeddings for r in requests], dim=0)
    states = torch.cat([r.states for r in requests], dim=0)
    start = _per_row([float(r.start_minutes) for r in requests], torch.float64)
    active = _per_row([int(r.duration_min) for r in requests], torch.long)
    meal_lists = [list(r.meals) for r in requests]

    def _pad(x: torch.Tensor) -> torch.Tensor:
        """A ``[T_i, k]`` tape to ``[n_steps, k]`` (zeros past the protocol's end)."""
        return x[:n_steps] if x.shape[0] >= n_steps else torch.cat(
            [x, x.new_zeros(n_steps - x.shape[0], x.shape[1])])

    gut = torch.cat([
        precompute_gut_outputs(model, r.embeddings, n_steps, dt=1.0, meals=meals)
        if r.gut_outputs is None else _pad(r.gut_outputs).unsqueeze(0).expand(n, -1, -1)
        for r, meals, n in zip(requests, meal_lists, sizes)
    ], dim=0)
    duo = None
    if any(meal_lists) or any(r.duodenal_outputs is not None for r in requests):
        duo = torch.cat([
            (precompute_duodenal_outputs(model, n_steps, dt=1.0, meals=meals)
             if r.duodenal_outputs is None else _pad(r.duodenal_outputs))
            .unsqueeze(0).expand(n, -1, -1)
            for r, meals, n in zip(requests, meal_lists, sizes)
        ], dim=0)

    def _tape(key: str) -> torch.Tensor | None:
        series = [getattr(r, key) for r in requests]
        if all(x is None for x in series):
            return None
        rows = []
        for x, n in zip(series, sizes):
            if x is None:  # withheld for these rows: NaN = the learned default
                x = torch.full((n_steps,), float("nan"), device=device)
            elif x.shape[0] < n_steps:  # past the protocol's end: hold its last value
                x = torch.cat([x, x[-1:].expand(n_steps - x.shape[0])])
            rows.append(x[:n_steps].unsqueeze(0).expand(n, -1))
        return torch.cat(rows, dim=0)

    traj = integrate(
        model, states, emb, n_steps,
        dt=1.0, start_time_minutes=start,
        sleep_wake=_tape("sleep_wake"), activity=_tape("activity"),
        gut_outputs=gut, duodenal_outputs=duo, active_steps=active,
        checkpoint_segments=checkpoint_segments,
    )
    out: list[torch.Tensor] = []
    row = 0
    for r, n in zip(requests, sizes):
        out.append(traj[row:row + n, :int(r.duration_min)])
        row += n
    return out
