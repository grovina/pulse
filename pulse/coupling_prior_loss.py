"""
Coupling-prior penalties from knowledge contributions.

Uses a one-sided finite difference on instantaneous rates: perturb the source
marker in state, measure the change in the target marker's rate of change, and
soft-penalize disagreement with the declared sign. This is cheap (no extra
rollout) and gradients reach module parameters through model.forward.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .knowledge.base import CouplingPrior, KnowledgeContribution
from .types import MARKER_INDEX, NORM_SCALE


def merge_coupling_priors(
    contributions: list[KnowledgeContribution],
    extra_priors: list[CouplingPrior] | None = None,
) -> list[CouplingPrior]:
    """Union coupling edges from contributions and any standalone prior lists.

    When two sources declare the same edge with the same sign, the magnitude
    ranges are intersected (a tighter consensus). Conflicting signs are
    skipped — the registry should be edited to resolve disagreement rather
    than silently picking one source over another.
    """
    merged: dict[tuple[str, str], CouplingPrior] = {}

    def _ingest(p: CouplingPrior) -> None:
        key = (p.source_marker, p.target_marker)
        if key not in merged:
            merged[key] = p
            return
        o = merged[key]
        if p.sign != o.sign:
            return
        lo = max(o.magnitude_range[0], p.magnitude_range[0])
        hi = min(o.magnitude_range[1], p.magnitude_range[1])
        if lo <= hi:
            merged[key] = CouplingPrior(p.source_marker, p.target_marker, p.sign, (lo, hi))

    for c in contributions:
        for p in c.coupling_priors():
            _ingest(p)
    for p in extra_priors or ():
        _ingest(p)
    return list(merged.values())


def _eps_for_marker(marker_id: str) -> float:
    idx = MARKER_INDEX[marker_id]
    scale = float(NORM_SCALE[idx])
    return max(scale * 0.04, 1e-4)


def normalized_sensitivity(sens_raw: torch.Tensor, source_marker: str, target_marker: str) -> torch.Tensor:
    """``d(rate_target)/d(state_source)`` in NORMALIZED units (iter 97, review 4.4).

    ``sens_raw`` is in target-units/min per source-unit. Multiplying by
    ``NORM_SCALE[source] / NORM_SCALE[target]`` gives "normalized target units
    per minute per normalized source unit", the frame every ``magnitude_range``
    in ``knowledge/coupling_priors`` is read in. Without this a cortisol->hr edge
    (ug/dL -> bpm) and a glucose->insulin edge (mg/dL -> uU/mL) were compared to
    their ranges in unrelated raw units.
    """
    si = MARKER_INDEX[source_marker]
    ti = MARKER_INDEX[target_marker]
    return sens_raw * (float(NORM_SCALE[si]) / float(NORM_SCALE[ti]))


def coupling_band_hinge(sens_norm: torch.Tensor, prior: CouplingPrior) -> torch.Tensor:
    """Two-sided band hinge on the declared magnitude range (iter 97, review 4.4).

    ``s = sens_norm * sign`` must lie in ``[lo, hi]``: zero loss inside, a
    Huber-shaped penalty on the distance outside measured in units of the band
    width. The pre-iter-97 form was ``softplus(-20 * sens * sign)`` — a
    sign-only hinge that ignored ``magnitude_range`` entirely and kept pushing
    every edge STRONGER until the sensitivity was 7-250x above its declared
    range (22 of 32 edges, including cortisol->hr, the coupling iter 96 cut).
    """
    lo, hi = float(prior.magnitude_range[0]), float(prior.magnitude_range[1])
    if hi < lo:
        lo, hi = hi, lo
    width = max(hi - lo, 1e-6)
    s = sens_norm * float(prior.sign)
    excess = (F.relu(lo - s) + F.relu(s - hi)) / width
    # Huber: quadratic inside one band width, linear beyond (a wrong-signed
    # edge is many widths out and must not dominate the window loss).
    return torch.where(excess < 1.0, 0.5 * excess.pow(2), excess - 0.5)


def _band_hinge_batched(
    sens_norm: torch.Tensor, sign: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor,
    width: torch.Tensor,
) -> torch.Tensor:
    """``coupling_band_hinge`` for many priors at once (``[..., P]`` against ``[P]``)."""
    s = sens_norm * sign
    excess = (F.relu(lo - s) + F.relu(s - hi)) / width
    return torch.where(excess < 1.0, 0.5 * excess.pow(2), excess - 0.5)


def _sample_indices(n_steps: int, n_samples: int) -> list[int]:
    indices: list[int] = []
    for k in range(n_samples):
        if n_samples == 1:
            idx = n_steps // 2
        else:
            idx = int((k + 1) * (n_steps - 1) / (n_samples + 1))
        indices.append(max(0, min(n_steps - 1, idx)))
    return sorted(set(indices))


def coupling_prior_loss_on_window(
    model: torch.nn.Module,
    pred_traj: torch.Tensor,
    embedding: torch.Tensor,
    start_time_minutes: float,
    meals: list,
    sleep_wake: torch.Tensor | None,
    activity: torch.Tensor | None,
    priors: list[CouplingPrior],
    n_samples: int = 3,
    *,
    gut_window: torch.Tensor | None = None,
    duodenal_window: torch.Tensor | None = None,
) -> torch.Tensor:
    """Average coupling loss at a few interior timesteps along an integrated trajectory.

    For each sampled step the trajectory's state is evaluated together with one copy
    per prior whose SOURCE marker is nudged by ``eps``; the target's rate difference
    over ``eps`` is the edge's sensitivity, scored by ``coupling_band_hinge`` in the
    normalized frame. All rows go through ONE batched ``model.forward`` — the
    unperturbed rate is shared by every prior at a step instead of recomputed per
    prior (3 samples x 35 edges used to be 210 single-row forwards, about
    the cost of the window's own rollout).

    The sensitivity is taken at the trajectory's operating point, so the rows carry
    that minute's gut appearance and duodenal delivery (``gut_window`` /
    ``duodenal_window``, ``[T, ...]`` on the window-offset clock — what the rollout
    itself saw). The model is nonlinear: its d(rate)/d(state) depends on the meal
    input. Through iter 107 the probe ran on the ABSOLUTE clock (so an in-window meal
    was usually invisible) with zero duodenal drive, i.e. the edges were scored at a
    fasted operating point whatever the window contained. When the windows are not
    given they are computed here from ``meals`` on the window-offset clock.
    """
    n_steps = pred_traj.shape[0]
    if n_steps < 2 or not priors:
        return pred_traj.new_tensor(0.0)
    device = pred_traj.device
    dtype = pred_traj.dtype
    indices = _sample_indices(n_steps, n_samples)
    I, P = len(indices), len(priors)
    rows_per_step = 1 + P
    idx_t = torch.tensor(indices, dtype=torch.long, device=device)

    src = [MARKER_INDEX[p.source_marker] for p in priors]
    tgt = [MARKER_INDEX[p.target_marker] for p in priors]
    eps = [_eps_for_marker(p.source_marker) for p in priors]
    nudge = torch.zeros(P, pred_traj.shape[-1], dtype=dtype, device=device)
    nudge[torch.arange(P, device=device), torch.tensor(src, device=device)] = torch.tensor(
        eps, dtype=dtype, device=device)

    base = pred_traj.index_select(0, idx_t)                                   # [I, S]
    states = torch.cat([base.unsqueeze(1), base.unsqueeze(1) + nudge], dim=1)  # [I, 1+P, S]
    states = states.reshape(I * rows_per_step, -1)

    if gut_window is None or duodenal_window is None:
        from .model import precompute_duodenal_outputs, precompute_gut_outputs
        if gut_window is None:
            gut_window = precompute_gut_outputs(model, embedding, n_steps, meals=meals)
        if duodenal_window is None:
            duodenal_window = precompute_duodenal_outputs(model, n_steps, meals=meals)

    def per_row(x: torch.Tensor) -> torch.Tensor:
        return x.index_select(0, idx_t).repeat_interleave(rows_per_step, dim=0)

    t_rows = torch.tensor(
        [(start_time_minutes + float(i)) % 1440.0 for i in indices], dtype=torch.float32, device=device,
    ).repeat_interleave(rows_per_step)
    rates = model(
        states,
        embedding.reshape(1, -1).expand(states.shape[0], -1),
        t_rows,
        meals,
        sleep_wake=per_row(sleep_wake) if sleep_wake is not None else None,
        activity=per_row(activity) if activity is not None else None,
        gut_override=per_row(gut_window),
        duodenal_override=per_row(duodenal_window),
    ).reshape(I, rows_per_step, -1)

    tgt_t = torch.tensor(tgt, dtype=torch.long, device=device)
    r0 = rates[:, 0, :].index_select(-1, tgt_t)                                     # [I, P]
    r1 = rates[:, 1:, :].gather(-1, tgt_t.view(1, P, 1).expand(I, P, 1)).squeeze(-1)  # [I, P]

    def _vec(xs: list[float]) -> torch.Tensor:
        return torch.tensor(xs, dtype=dtype, device=device)

    ratio = [float(NORM_SCALE[si]) / float(NORM_SCALE[ti]) for si, ti in zip(src, tgt)]
    bounds = [sorted((float(p.magnitude_range[0]), float(p.magnitude_range[1]))) for p in priors]
    sens_n = (r1 - r0) / _vec(eps) * _vec(ratio)
    hinge = _band_hinge_batched(
        sens_n, _vec([float(p.sign) for p in priors]),
        _vec([lo for lo, _ in bounds]), _vec([hi for _, hi in bounds]),
        _vec([max(hi - lo, 1e-6) for lo, hi in bounds]),
    )
    return hinge.mean()
