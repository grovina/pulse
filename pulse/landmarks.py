"""
Differentiable landmark extractors for post-meal physiology windows.

A landmark is a scalar summary of a post-meal time series — a soft extremum
(peak or nadir), the time at which it occurs, and the area between the
series and its pre-meal baseline. Together they describe the qualitative
shape of the response: how much it climbs or drops, how fast, how much
area is enclosed. The model can match landmarks without imitating the
cold-model waveform point-by-point, supplying a "match the shape" gradient
that bypasses the trajectory band's dead zone — useful for small but
clinically meaningful counter-regulatory deflections.

Each marker can be supervised in either direction:
- PEAK markers (glucose, insulin, GLP-1) track Δpeak and positive AUC.
- NADIR markers (glucagon, FFA, ghrelin) track Δnadir and negative AUC.

The same softmax-weighted statistic produces both — flip the sign of the
softmax temperature to turn soft argmax into soft argmin.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn.functional as F

from .modules.gut import MealEvent
from .types import MARKER_INDEX


class LandmarkDirection(str, Enum):
    PEAK = "peak"
    NADIR = "nadir"


@dataclass(frozen=True)
class LandmarkScales:
    """Per-marker normalizers; the loss is the mean of (term/scale)²."""

    delta_extremum: float    # raw marker units (Δpeak for PEAK, |Δnadir| for NADIR)
    time_to_extremum: float  # minutes
    auc: float               # value · minutes
    softargmax_beta: float   # softmax temperature in raw marker units


@dataclass(frozen=True)
class MarkerLandmarkSpec:
    """Which marker to supervise, in which direction, with which scales."""

    marker_id: str
    direction: LandmarkDirection
    scales: LandmarkScales


# The landmark loss currently supervises HR only. Earlier iters tried
# metabolic PEAK/NADIR specs too (iter 33: all six; iter 34: HR-only after the
# glucose/insulin PEAK regressors dominated the gradient budget; iter 41: a
# brief NADIR revival that proved inert). Counter-regulatory marker revival is
# now handled by cold-model distillation, not landmarks — see
# docs/dead-pathways.md. Scales are derived from cold-model amplitudes on a
# mixed meal so each (residual / scale) sits roughly in the unit range when off
# by ~1 SD of the typical biological response.
_HR_PEAK = MarkerLandmarkSpec(
    "hr", LandmarkDirection.PEAK,
    LandmarkScales(delta_extremum=10.0, time_to_extremum=30.0,
                   auc=300.0, softargmax_beta=0.15),
)

DEFAULT_LANDMARK_SPECS: tuple[MarkerLandmarkSpec, ...] = (
    _HR_PEAK,
)


def post_meal_landmarks(
    series: torch.Tensor,
    *,
    meal_step: int,
    pre_window: int,
    post_window: int,
    softargmax_beta: float,
    direction: LandmarkDirection,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (delta_extremum, time_to_extremum_steps, signed_auc) for one meal.

    For ``PEAK``: extremum is the soft max above baseline; AUC is positive
    area above baseline. For ``NADIR``: extremum is the soft min below
    baseline (signed Δ is negative for suppression); AUC is the (positive
    magnitude of) area below baseline. Flipping the softmax temperature's
    sign converts soft argmax into soft argmin.

    Caller guarantees ``meal_step - pre_window >= 0`` and
    ``meal_step + post_window <= len(series)``.
    """
    pre = series[meal_step - pre_window : meal_step]
    post = series[meal_step : meal_step + post_window]
    base = pre.mean()
    sign = 1.0 if direction is LandmarkDirection.PEAK else -1.0
    weights = F.softmax(sign * softargmax_beta * post, dim=0)
    extremum = (weights * post).sum()
    idx = torch.arange(post.shape[0], device=post.device, dtype=post.dtype)
    tte = (weights * idx).sum()
    if direction is LandmarkDirection.PEAK:
        auc = F.relu(post - base).sum()
    else:
        auc = F.relu(base - post).sum()
    return extremum - base, tte, auc


def _per_marker_landmark_loss(
    pred_series: torch.Tensor,
    target_series: torch.Tensor,
    *,
    meal_step: int,
    pre_window: int,
    post_window: int,
    spec: MarkerLandmarkSpec,
) -> torch.Tensor | None:
    """Z² landmark loss for one meal & marker, or None if any target sample is NaN."""
    tgt_pre = target_series[meal_step - pre_window : meal_step]
    tgt_post = target_series[meal_step : meal_step + post_window]
    if torch.isnan(tgt_pre).any() or torch.isnan(tgt_post).any():
        return None

    pdp, ptp, pauc = post_meal_landmarks(
        pred_series, meal_step=meal_step,
        pre_window=pre_window, post_window=post_window,
        softargmax_beta=spec.scales.softargmax_beta,
        direction=spec.direction,
    )
    tdp, ttp, tauc = post_meal_landmarks(
        target_series, meal_step=meal_step,
        pre_window=pre_window, post_window=post_window,
        softargmax_beta=spec.scales.softargmax_beta,
        direction=spec.direction,
    )
    return (
        ((pdp - tdp) / spec.scales.delta_extremum).pow(2)
        + ((ptp - ttp) / spec.scales.time_to_extremum).pow(2)
        + ((pauc - tauc) / spec.scales.auc).pow(2)
    )


def post_meal_landmark_loss(
    pred_traj: torch.Tensor,
    target_traj: torch.Tensor,
    meals: list[MealEvent],
    *,
    pre_window: int,
    post_window: int,
    min_carbs: float,
    specs: tuple[MarkerLandmarkSpec, ...] = DEFAULT_LANDMARK_SPECS,
) -> tuple[torch.Tensor, int]:
    """Mean per-meal landmark loss across all configured markers for a window.

    Skips meals that don't sit fully inside ``[pre_window, len - post_window)``
    and meals whose target series contain NaN in the relevant span. Returns
    ``(loss, n_meal_marker_pairs)``; the loss is a zero-tensor on the same
    device when no meal qualifies.
    """
    win_steps = pred_traj.shape[0]
    parts: list[torch.Tensor] = []
    for m in meals:
        meal_step = int(round(float(m.time)))
        if (
            float(m.carbs) < min_carbs
            or meal_step < pre_window
            or meal_step + post_window > win_steps
        ):
            continue
        for spec in specs:
            idx = MARKER_INDEX[spec.marker_id]
            term = _per_marker_landmark_loss(
                pred_traj[:, idx], target_traj[:, idx],
                meal_step=meal_step,
                pre_window=pre_window,
                post_window=post_window,
                spec=spec,
            )
            if term is not None:
                parts.append(term)

    if not parts:
        return pred_traj.new_tensor(0.0), 0
    return torch.stack(parts).mean(), len(parts)
