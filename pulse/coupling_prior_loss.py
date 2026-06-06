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

        r0 = model(
            s0.unsqueeze(0),
            embedding.unsqueeze(0),
            t_tensor,
            meals,
            sleep_wake=sleep_wake_step.unsqueeze(0) if sleep_wake_step is not None else None,
            activity=activity_step.unsqueeze(0) if activity_step is not None else None,
        ).squeeze(0)
        r1 = model(
            s1.unsqueeze(0),
            embedding.unsqueeze(0),
            t_tensor,
            meals,
            sleep_wake=sleep_wake_step.unsqueeze(0) if sleep_wake_step is not None else None,
            activity=activity_step.unsqueeze(0) if activity_step is not None else None,
        ).squeeze(0)

        sens = (r1[ti] - r0[ti]) / eps
        # Encourage sens * sign > 0 (soft hinge)
        margin = sens * float(prior.sign)
        total = total + F.softplus(-margin * 20.0)

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
