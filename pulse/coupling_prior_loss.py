"""
Coupling-prior penalties from knowledge contributions.

Uses a one-sided finite difference on instantaneous rates: perturb the source
marker in state, measure the change in the target marker's rate of change, and
penalize a sensitivity outside the declared signed magnitude band. This is cheap
(no extra rollout: one batched model.forward per window) and gradients reach
module parameters through it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .knowledge.base import CouplingPrior, KnowledgeContribution
from .model import precompute_duodenal_outputs, precompute_gut_outputs
from .types import DUODENAL_DIM, GUT_OUTPUT_DIM, MARKER_INDEX, NORM_SCALE


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


def _band_hinge(
    sens_norm: torch.Tensor, sign: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor,
) -> torch.Tensor:
    """Elementwise ``coupling_band_hinge`` over tensors of edges."""
    width = (hi - lo).clamp(min=1e-6)
    s = sens_norm * sign
    excess = (F.relu(lo - s) + F.relu(s - hi)) / width
    # Huber: quadratic inside one band width, linear beyond (a wrong-signed
    # edge is many widths out and must not dominate the window loss).
    return torch.where(excess < 1.0, 0.5 * excess.pow(2), excess - 0.5)


def coupling_band_hinge(sens_norm: torch.Tensor, prior: CouplingPrior) -> torch.Tensor:
    """Two-sided band hinge on the declared magnitude range (iter 97, review 4.4).

    ``s = sens_norm * sign`` must lie in ``[lo, hi]``: zero loss inside, a
    Huber-shaped penalty on the distance outside measured in units of the band
    width. The pre-iter-97 form was ``softplus(-20 * sens * sign)`` — a
    sign-only hinge that ignored ``magnitude_range`` entirely and kept pushing
    every edge STRONGER until the sensitivity was 7-250x above its declared
    range (22 of 32 edges, including cortisol->hr, the coupling iter 96 cut).
    """
    lo, hi = sorted((float(prior.magnitude_range[0]), float(prior.magnitude_range[1])))
    return _band_hinge(
        sens_norm, sens_norm.new_tensor(float(prior.sign)),
        sens_norm.new_tensor(lo), sens_norm.new_tensor(hi),
    )


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
    gut_outputs: torch.Tensor | None = None,
    duodenal_outputs: torch.Tensor | None = None,
) -> torch.Tensor:
    """Average coupling loss at a few interior timesteps along an integrated trajectory.

    ``pred_traj`` is the window's ``[T, STATE_DIM]`` rollout; at each sampled step the
    loss is the mean band hinge over ``priors`` of the finite-difference sensitivity
    ``∂(rate_target)/∂(state_source)``. Every (step, prior) difference — the
    unperturbed state and one perturbed copy per prior — is a row of ONE batched
    ``model.forward`` (through iter 99: two batch-1 forwards per prior per step, the
    unperturbed one recomputed for every prior — ~190 calls a window).

    The step's inputs are the window's own: ``gut_outputs`` / ``duodenal_outputs``
    (``[T, 4]`` / ``[T, 3]``, the tapes the rollout integrated) at the sampled step,
    computed from ``meals`` on the window-offset clock when not given. Through iter
    99 the gut was evaluated on the ABSOLUTE clock (the iter-87 frame) and duodenal
    delivery was zero, on the argument that the gut term cancels in the
    difference — true only for edges whose target rate is additive in it, not for
    e.g. glucose, whose plasma share of appearance depends on the glycogen state.
    """
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

    device = pred_traj.device
    dtype = pred_traj.dtype
    K, P, S = len(indices), len(priors), int(pred_traj.shape[-1])
    if gut_outputs is None:
        gut_outputs = (
            precompute_gut_outputs(model, embedding, n_steps, meals=meals) if meals
            else torch.zeros(n_steps, GUT_OUTPUT_DIM, dtype=dtype, device=device)
        )
    if duodenal_outputs is None:
        duodenal_outputs = (
            precompute_duodenal_outputs(model, n_steps, meals=meals)
            if meals and hasattr(model, "duodenal")
            else torch.zeros(n_steps, DUODENAL_DIM, dtype=dtype, device=device)
        )

    src = torch.tensor([MARKER_INDEX[p.source_marker] for p in priors], device=device)
    tgt = torch.tensor([MARKER_INDEX[p.target_marker] for p in priors], device=device)
    eps = torch.tensor([_eps_for_marker(p.source_marker) for p in priors], dtype=dtype, device=device)
    sign = torch.tensor([float(p.sign) for p in priors], dtype=dtype, device=device)
    bounds = torch.tensor(
        [sorted((float(p.magnitude_range[0]), float(p.magnitude_range[1]))) for p in priors],
        dtype=dtype, device=device,
    )
    norm_scale = torch.tensor(NORM_SCALE, dtype=dtype, device=device)

    # Rows: per sampled step, the state itself then one copy per prior with its
    # source marker raised by eps.
    step_idx = torch.tensor(indices, device=device)
    perturb = torch.zeros(P + 1, S, dtype=dtype, device=device)
    perturb[torch.arange(1, P + 1, device=device), src] = eps
    states = (pred_traj[step_idx].unsqueeze(1) + perturb.unsqueeze(0)).reshape(K * (P + 1), S)

    def _rows(x: torch.Tensor | None) -> torch.Tensor | None:
        return None if x is None else x[step_idx].repeat_interleave(P + 1, dim=0)

    t_abs = ((start_time_minutes + step_idx.to(torch.float64)) % 1440.0).to(dtype)
    rates = model(
        states,
        embedding.unsqueeze(0).expand(K * (P + 1), -1),
        t_abs.repeat_interleave(P + 1),
        meals,
        sleep_wake=_rows(sleep_wake),
        activity=_rows(activity),
        gut_override=_rows(gut_outputs),
        duodenal_override=_rows(duodenal_outputs),
    ).reshape(K, P + 1, S)

    cols = torch.arange(P, device=device)
    r1 = rates[:, 1:, :][:, cols, tgt]  # [K, P]: target rate under the prior's perturbation
    r0 = rates[:, 0, :][:, tgt]  # [K, P]: target rate unperturbed
    sens = (r1 - r0) / eps
    sens_n = sens * (norm_scale[src] / norm_scale[tgt])
    return _band_hinge(sens_n, sign, bounds[:, 0], bounds[:, 1]).mean()
