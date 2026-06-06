"""
Training-time surrogate aligned with verifier.evaluate_weak_checks.

Thresholds, soft scales, and weights come from knowledge.weak_check_params so
gradients match the post-hoc verifier (PRD: priors as traceable signal).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .knowledge.weak_check_params import (
    MEAL,
    COUPLING,
    CIRCADIAN,
    SLEEP,
    SANITY_RANGES,
    sanity_range_scale,
)
from .modules.gut import MealEvent
from .types import MARKER_INDEX


def _safe_corr_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.numel() < 2:
        return a.new_tensor(0.0)
    a = a.float()
    b = b.float()
    am = a - a.mean()
    bm = b - b.mean()
    sa = am.std(unbiased=False)
    sb = bm.std(unbiased=False)
    denom = sa * sb
    # Iter 26: clamp the divisor BEFORE the divide. The previous form
    # was ``torch.where(denom < 1e-8, 0.0, (am*bm).mean() / denom)`` which
    # is a textbook autograd NaN trap: forward selects 0 when denom is
    # tiny, but backward evaluates BOTH branches and the unselected
    # ``(am*bm).mean() / denom`` produces a non-finite gradient at
    # denom≈0 (1/denom² in chain rule), and ``where`` then propagates
    # NaN through the unselected branch into every parameter that fed
    # ``a`` or ``b``. This is what aborted iter 25 at Phase 2 epoch 61
    # (build 2ffc4b7f, abort_diagnostics.json shows
    # signal=trajectory_rollout/window cause=grad n_nan=71/115). Clamping
    # the divisor first keeps both branches finite during backward, and
    # the ``where`` selects the genuine zero forward when the marker
    # series has near-zero variance (correlation undefined).
    denom_safe = denom.clamp(min=1e-8)
    corr = (am * bm).mean() / denom_safe
    return torch.where(denom < 1e-8, a.new_tensor(0.0), corr)


def _mask_mean_torch(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.any():
        return values[mask].mean()
    return values.mean()


def _range_score(values: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    out_lo = F.relu(lo - values)
    out_hi = F.relu(values - hi)
    violation = (out_lo + out_hi).mean()
    scale = sanity_range_scale(lo, hi)
    return torch.exp(-violation / scale)


def training_verifier_surrogate_loss(
    pred_traj: torch.Tensor,
    *,
    meals: list[MealEvent],
    start_hour: float,
    timeline_offset_min: float = 0.0,
) -> torch.Tensor:
    """Mean weighted (1 - score) matching verifier weak checks; 0 if no checks apply."""
    T = pred_traj.shape[0]
    device = pred_traj.device

    weighted_one_minus: list[torch.Tensor] = []
    weights: list[float] = []

    glucose = pred_traj[:, MARKER_INDEX["glucose"]]
    insulin = pred_traj[:, MARKER_INDEX["insulin"]]

    pw = MEAL.pre_window
    ps, pe = MEAL.post_start, MEAL.post_end
    for m in meals:
        t = int(round(float(m.time)))
        carbs = float(m.carbs)
        if carbs <= 0 or t < pw or t + pe >= T:
            continue

        pre_g = glucose[t - pw : t]
        pre_i = insulin[t - pw : t]
        post_g = glucose[t + ps : t + pe]
        post_i = insulin[t + ps : t + pe]
        g_delta = post_g.max() - pre_g.mean()
        i_delta = post_i.max() - pre_i.mean()
        g_target = MEAL.glucose_target(carbs)
        i_target = MEAL.insulin_target(carbs)

        s_g = torch.sigmoid((g_delta - g_target) / MEAL.glucose_soft_scale)
        s_i = torch.sigmoid((i_delta - i_target) / MEAL.insulin_soft_scale)
        weighted_one_minus.append(MEAL.glucose_weight * (1.0 - s_g))
        weights.append(MEAL.glucose_weight)
        weighted_one_minus.append(MEAL.insulin_weight * (1.0 - s_i))
        weights.append(MEAL.insulin_weight)

    hr = pred_traj[:, MARKER_INDEX["hr"]]
    sbp = pred_traj[:, MARKER_INDEX["sbp"]]
    dbp = pred_traj[:, MARKER_INDEX["dbp"]]
    glucagon = pred_traj[:, MARKER_INDEX["glucagon"]]

    hr_sbp_corr = _safe_corr_torch(hr, sbp)
    g_gn_corr = _safe_corr_torch(glucose, glucagon)
    sbp_dbp_margin = (sbp - dbp).min()

    s1 = torch.sigmoid((hr_sbp_corr - COUPLING.hr_sbp_corr_min) / COUPLING.hr_sbp_soft_scale)
    weighted_one_minus.append(COUPLING.hr_sbp_weight * (1.0 - s1))
    weights.append(COUPLING.hr_sbp_weight)

    s2 = torch.sigmoid((-g_gn_corr - COUPLING.glucose_glucagon_inverse_min) / COUPLING.glucose_glucagon_soft_scale)
    weighted_one_minus.append(COUPLING.glucose_glucagon_weight * (1.0 - s2))
    weights.append(COUPLING.glucose_glucagon_weight)

    s3 = torch.sigmoid((sbp_dbp_margin - COUPLING.sbp_dbp_margin_min) / COUPLING.sbp_dbp_soft_scale)
    weighted_one_minus.append(COUPLING.sbp_dbp_weight * (1.0 - s3))
    weights.append(COUPLING.sbp_dbp_weight)

    cortisol = pred_traj[:, MARKER_INDEX["cortisol"]]
    temp = pred_traj[:, MARKER_INDEX["temp"]]

    minutes = torch.arange(T, device=device, dtype=torch.float32)
    abs_hour = ((start_hour * 60.0 + timeline_offset_min + minutes) % 1440.0) / 60.0

    c = CIRCADIAN
    morning_mask = (abs_hour >= c.morning_hour_lo) & (abs_hour <= c.morning_hour_hi)
    evening_mask = (abs_hour >= c.evening_hour_lo) & (abs_hour <= c.evening_hour_hi)
    early_temp_mask = (abs_hour >= c.early_temp_hour_lo) & (abs_hour <= c.early_temp_hour_hi)
    afternoon_temp_mask = (abs_hour >= c.afternoon_temp_hour_lo) & (abs_hour <= c.afternoon_temp_hour_hi)

    morning = _mask_mean_torch(cortisol, morning_mask)
    evening = _mask_mean_torch(cortisol, evening_mask)
    early_temp = _mask_mean_torch(temp, early_temp_mask)
    afternoon_temp = _mask_mean_torch(temp, afternoon_temp_mask)

    s4 = torch.sigmoid((morning - evening - c.cortisol_morning_evening_min) / c.cortisol_soft_scale)
    weighted_one_minus.append(c.cortisol_weight * (1.0 - s4))
    weights.append(c.cortisol_weight)

    s5 = torch.sigmoid((afternoon_temp - early_temp - c.temp_afternoon_early_min) / c.temp_soft_scale)
    weighted_one_minus.append(c.temp_weight * (1.0 - s5))
    weights.append(c.temp_weight)

    slp = SLEEP
    if T >= slp.min_trajectory_len:
        hr_day = _mask_mean_torch(hr, (abs_hour >= slp.daytime_hour_lo) & (abs_hour <= slp.daytime_hour_hi))
        hr_night = _mask_mean_torch(hr, (abs_hour >= slp.nighttime_hour_lo) & (abs_hour <= slp.nighttime_hour_hi))
        sbp_day = _mask_mean_torch(sbp, (abs_hour >= slp.daytime_hour_lo) & (abs_hour <= slp.daytime_hour_hi))
        sbp_night = _mask_mean_torch(sbp, (abs_hour >= slp.nighttime_hour_lo) & (abs_hour <= slp.nighttime_hour_hi))
        hr_margin = hr_day - hr_night
        sbp_margin = sbp_day - sbp_night
        s6 = torch.sigmoid((hr_margin - slp.hr_dip_min) / slp.hr_dip_soft_scale)
        weighted_one_minus.append(slp.hr_dip_weight * (1.0 - s6))
        weights.append(slp.hr_dip_weight)
        s7 = torch.sigmoid((sbp_margin - slp.sbp_dip_min) / slp.sbp_dip_soft_scale)
        weighted_one_minus.append(slp.sbp_dip_weight * (1.0 - s7))
        weights.append(slp.sbp_dip_weight)

    for mid, spec in SANITY_RANGES:
        vals = pred_traj[:, MARKER_INDEX[mid]]
        s = _range_score(vals, spec.lo, spec.hi)
        weighted_one_minus.append(spec.weight * (1.0 - s))
        weights.append(spec.weight)

    if not weights:
        return pred_traj.new_tensor(0.0)

    tw = float(sum(weights))
    return torch.stack(weighted_one_minus).sum() / tw
