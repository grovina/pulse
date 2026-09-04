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


def coupling_prior_loss_at_step(
    model: torch.nn.Module,
    state: torch.Tensor,
    embedding: torch.Tensor,
    t_abs: float,
    meals: list,
    sleep_wake_step: torch.Tensor | None,
    activity_step: torch.Tensor | None,
    priors: list[CouplingPrior],
) -> torch.Tensor:
    """Scalar loss averaged over priors (sign alignment on ∂(rate_target)/∂(state_source))."""
    if not priors:
        return state.new_tensor(0.0)

    device = state.device
    total = state.new_tensor(0.0)
    t_tensor = torch.tensor([t_abs % 1440.0], device=device, dtype=torch.float32)

    for prior in priors:
        si = MARKER_INDEX[prior.source_marker]
        ti = MARKER_INDEX[prior.target_marker]
        eps = _eps_for_marker(prior.source_marker)

        s0 = state
        s1 = state.clone()
        s1[si] = s1[si] + eps

        # gut_clock_exempt (iter 90): this is a pointwise finite difference in the
        # SOURCE STATE only. The gut appearance term depends on (meals, t, embedding)
        # and not on state, so it is identical in r0 and r1 and cancels exactly in
        # (r1 - r0). The absolute-vs-window-offset gut clock therefore cannot affect
        # this loss — the one place the frame contract may be waived.
        r0 = model(
            s0.unsqueeze(0),
            embedding.unsqueeze(0),
            t_tensor,
            meals,
            sleep_wake=sleep_wake_step.unsqueeze(0) if sleep_wake_step is not None else None,
            activity=activity_step.unsqueeze(0) if activity_step is not None else None,
            gut_clock_exempt=True,
        ).squeeze(0)
        r1 = model(
            s1.unsqueeze(0),
            embedding.unsqueeze(0),
            t_tensor,
            meals,
            sleep_wake=sleep_wake_step.unsqueeze(0) if sleep_wake_step is not None else None,
            activity=activity_step.unsqueeze(0) if activity_step is not None else None,
            gut_clock_exempt=True,
        ).squeeze(0)

        sens = (r1[ti] - r0[ti]) / eps
        sens_n = normalized_sensitivity(sens, prior.source_marker, prior.target_marker)
        total = total + coupling_band_hinge(sens_n, prior)

    return total / max(len(priors), 1)


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
) -> torch.Tensor:
    """Average coupling loss at a few interior timesteps along an integrated trajectory."""
    n_steps = pred_traj.shape[0]
    if n_steps < 2 or not priors:
        return pred_traj.new_tensor(0.0)

    indices: list[int] = []
    for k in range(n_samples):
        if n_samples == 1:
            idx = n_steps // 2
        else:
            idx = int((k + 1) * (n_steps - 1) / (n_samples + 1))
        idx = max(0, min(n_steps - 1, idx))
        indices.append(idx)
    indices = sorted(set(indices))

    acc = pred_traj.new_tensor(0.0)
    for idx in indices:
        t_abs = start_time_minutes + float(idx)
        sw_s = sleep_wake[idx] if sleep_wake is not None else None
        act_s = activity[idx] if activity is not None else None
        acc = acc + coupling_prior_loss_at_step(
            model,
            pred_traj[idx],
            embedding,
            t_abs,
            meals,
            sw_s,
            act_s,
            priors,
        )
    return acc / max(len(indices), 1)
