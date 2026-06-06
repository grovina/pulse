"""
Quality gates for the modular physiology model.

Verifiers check whether trained model output exhibits expected physiological
properties. They serve as post-training evaluation — asserting that the
model has learned plausible dynamics from its knowledge contributions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

from .types import MARKER_INDEX
from .knowledge.weak_check_params import (
    MEAL,
    COUPLING,
    CIRCADIAN,
    SLEEP,
    SANITY_RANGES,
    sanity_range_scale,
)


@dataclass(frozen=True)
class WeakCheckResult:
    key: str
    label: str
    category: str
    passed: bool
    score: float
    weight: float
    details: dict[str, float | int | str]


def evaluate_weak_checks(
    trajectory: np.ndarray,
    *,
    meals: list[Any] | None = None,
    subjective_reports: list[Any] | None = None,
    start_hour: float = 6.0,
    timeline_offset_min: float = 0.0,
) -> dict[str, Any]:
    """timeline_offset_min: minutes from episode start added to each index (e.g. sliding window start)."""
    meals = meals or []
    subjective_reports = subjective_reports or []
    checks: list[WeakCheckResult] = []

    checks.extend(_meal_response_checks(trajectory, meals))
    checks.extend(_coupling_checks(trajectory))
    checks.extend(_circadian_checks(trajectory, start_hour=start_hour, timeline_offset_min=timeline_offset_min))
    checks.extend(_sleep_checks(trajectory, start_hour=start_hour, timeline_offset_min=timeline_offset_min))
    checks.extend(_sanity_checks(trajectory))

    total_weight = float(sum(c.weight for c in checks))
    weighted_score = float(sum(c.score * c.weight for c in checks) / max(total_weight, 1e-9))
    pass_rate = float(sum(1 for c in checks if c.passed) / max(len(checks), 1))
    category_scores = _aggregate_by_category(checks)

    return {
        "overall_score": weighted_score,
        "pass_rate": pass_rate,
        "n_checks": len(checks),
        "category_scores": category_scores,
        "checks": [
            {
                "key": c.key, "label": c.label, "category": c.category,
                "passed": c.passed, "score": c.score, "weight": c.weight,
                "details": c.details,
            }
            for c in checks
        ],
    }


def _meal_response_checks(trajectory: np.ndarray, meals: list[Any]) -> list[WeakCheckResult]:
    if not meals:
        return []

    glucose = trajectory[:, MARKER_INDEX["glucose"]]
    insulin = trajectory[:, MARKER_INDEX["insulin"]]

    checks: list[WeakCheckResult] = []
    for meal in meals:
        if isinstance(meal, (list, tuple)):
            t = int(round(float(meal[0])))
            carbs = float(meal[1]) if len(meal) > 1 else 0.0
        else:
            t = int(round(float(getattr(meal, "time", 0))))
            carbs = float(getattr(meal, "carbs", 0))
        pw = MEAL.pre_window
        ps, pe = MEAL.post_start, MEAL.post_end
        if carbs <= 0 or t < pw or t + pe >= len(trajectory):
            continue

        pre = slice(t - pw, t)
        post = slice(t + ps, t + pe)

        g_delta = float(np.max(glucose[post]) - np.mean(glucose[pre]))
        i_delta = float(np.max(insulin[post]) - np.mean(insulin[pre]))
        g_target = MEAL.glucose_target(carbs)
        i_target = MEAL.insulin_target(carbs)

        checks.append(_build_threshold_check(
            key=f"meal_glucose_rise_{t}", label="Meal should trigger glucose rise",
            category="meal", metric=g_delta, pass_threshold=g_target,
            soft_scale=MEAL.glucose_soft_scale, weight=MEAL.glucose_weight,
            details={"meal_time": t, "carbs": carbs, "delta": g_delta},
        ))
        checks.append(_build_threshold_check(
            key=f"meal_insulin_rise_{t}", label="Meal should trigger insulin rise",
            category="meal", metric=i_delta, pass_threshold=i_target,
            soft_scale=MEAL.insulin_soft_scale, weight=MEAL.insulin_weight,
            details={"meal_time": t, "carbs": carbs, "delta": i_delta},
        ))

    return checks


def _coupling_checks(trajectory: np.ndarray) -> list[WeakCheckResult]:
    hr = trajectory[:, MARKER_INDEX["hr"]]
    sbp = trajectory[:, MARKER_INDEX["sbp"]]
    dbp = trajectory[:, MARKER_INDEX["dbp"]]
    glucose = trajectory[:, MARKER_INDEX["glucose"]]
    glucagon = trajectory[:, MARKER_INDEX["glucagon"]]

    hr_sbp_corr = _safe_corr(hr, sbp)
    g_gn_corr = _safe_corr(glucose, glucagon)
    sbp_dbp_margin = float(np.min(sbp - dbp))

    return [
        _build_threshold_check(
            key="coupling_hr_sbp_corr", label="HR and SBP should be positively coupled",
            category="coupling", metric=hr_sbp_corr, pass_threshold=COUPLING.hr_sbp_corr_min,
            soft_scale=COUPLING.hr_sbp_soft_scale, weight=COUPLING.hr_sbp_weight,
            details={"corr": hr_sbp_corr},
        ),
        _build_threshold_check(
            key="coupling_glucose_glucagon_inverse",
            label="Glucose and glucagon should be inversely coupled",
            category="coupling", metric=-g_gn_corr, pass_threshold=COUPLING.glucose_glucagon_inverse_min,
            soft_scale=COUPLING.glucose_glucagon_soft_scale, weight=COUPLING.glucose_glucagon_weight,
            details={"corr": g_gn_corr},
        ),
        _build_threshold_check(
            key="coupling_sbp_ge_dbp", label="Systolic should stay above diastolic",
            category="coupling", metric=sbp_dbp_margin, pass_threshold=COUPLING.sbp_dbp_margin_min,
            soft_scale=COUPLING.sbp_dbp_soft_scale, weight=COUPLING.sbp_dbp_weight,
            details={"min_margin": sbp_dbp_margin},
        ),
    ]


def _circadian_checks(
    trajectory: np.ndarray,
    start_hour: float,
    timeline_offset_min: float = 0.0,
) -> list[WeakCheckResult]:
    cortisol = trajectory[:, MARKER_INDEX["cortisol"]]
    temp = trajectory[:, MARKER_INDEX["temp"]]

    minutes = np.arange(len(trajectory), dtype=np.float32)
    abs_hour = ((start_hour * 60.0 + timeline_offset_min + minutes) % 1440.0) / 60.0

    c = CIRCADIAN
    morning = _mask_mean(cortisol, (abs_hour >= c.morning_hour_lo) & (abs_hour <= c.morning_hour_hi))
    evening = _mask_mean(cortisol, (abs_hour >= c.evening_hour_lo) & (abs_hour <= c.evening_hour_hi))
    early_temp = _mask_mean(temp, (abs_hour >= c.early_temp_hour_lo) & (abs_hour <= c.early_temp_hour_hi))
    afternoon_temp = _mask_mean(temp, (abs_hour >= c.afternoon_temp_hour_lo) & (abs_hour <= c.afternoon_temp_hour_hi))

    return [
        _build_threshold_check(
            key="circadian_cortisol_morning_peak",
            label="Cortisol should be higher in morning than evening",
            category="circadian", metric=morning - evening, pass_threshold=c.cortisol_morning_evening_min,
            soft_scale=c.cortisol_soft_scale, weight=c.cortisol_weight,
            details={"morning": morning, "evening": evening},
        ),
        _build_threshold_check(
            key="circadian_temp_afternoon_higher",
            label="Core temp should be higher in afternoon than early morning",
            category="circadian", metric=afternoon_temp - early_temp, pass_threshold=c.temp_afternoon_early_min,
            soft_scale=c.temp_soft_scale, weight=c.temp_weight,
            details={"afternoon": afternoon_temp, "early_morning": early_temp},
        ),
    ]


def _sleep_checks(
    trajectory: np.ndarray,
    start_hour: float,
    timeline_offset_min: float = 0.0,
) -> list[WeakCheckResult]:
    """Sleep physiology: HR/BP should dip during typical sleep hours (0-5 AM)."""
    s = SLEEP
    if len(trajectory) < s.min_trajectory_len:
        return []

    hr = trajectory[:, MARKER_INDEX["hr"]]
    sbp = trajectory[:, MARKER_INDEX["sbp"]]

    minutes = np.arange(len(trajectory), dtype=np.float32)
    abs_hour = ((start_hour * 60.0 + timeline_offset_min + minutes) % 1440.0) / 60.0

    daytime = (abs_hour >= s.daytime_hour_lo) & (abs_hour <= s.daytime_hour_hi)
    nighttime = (abs_hour >= s.nighttime_hour_lo) & (abs_hour <= s.nighttime_hour_hi)

    if not np.any(daytime) or not np.any(nighttime):
        return []

    hr_day = _mask_mean(hr, daytime)
    hr_night = _mask_mean(hr, nighttime)
    sbp_day = _mask_mean(sbp, daytime)
    sbp_night = _mask_mean(sbp, nighttime)

    return [
        _build_threshold_check(
            key="sleep_hr_dip",
            label="HR should be lower during sleep than daytime",
            category="sleep", metric=hr_day - hr_night, pass_threshold=s.hr_dip_min,
            soft_scale=s.hr_dip_soft_scale, weight=s.hr_dip_weight,
            details={"hr_day": hr_day, "hr_night": hr_night},
        ),
        _build_threshold_check(
            key="sleep_sbp_dip",
            label="SBP should dip during sleep",
            category="sleep", metric=sbp_day - sbp_night, pass_threshold=s.sbp_dip_min,
            soft_scale=s.sbp_dip_soft_scale, weight=s.sbp_dip_weight,
            details={"sbp_day": sbp_day, "sbp_night": sbp_night},
        ),
    ]


def _sanity_checks(trajectory: np.ndarray) -> list[WeakCheckResult]:
    out: list[WeakCheckResult] = []
    for mid, spec in SANITY_RANGES:
        vals = trajectory[:, MARKER_INDEX[mid]]
        out.append(_range_check(
            f"sanity_{mid}_range",
            f"{mid.replace('_', ' ')} in plausible range",
            "sanity",
            values=vals, lo=spec.lo, hi=spec.hi, weight=spec.weight,
        ))
    return out


def _range_check(key: str, label: str, category: str, *, values: np.ndarray, lo: float, hi: float, weight: float) -> WeakCheckResult:
    out_lo = np.maximum(lo - values, 0.0)
    out_hi = np.maximum(values - hi, 0.0)
    violation = float(np.mean(out_lo + out_hi))
    score = float(np.exp(-violation / sanity_range_scale(lo, hi)))
    return WeakCheckResult(key=key, label=label, category=category, passed=score >= 0.8, score=score, weight=weight, details={"violation": violation, "lo": lo, "hi": hi})


def _build_threshold_check(*, key: str, label: str, category: str, metric: float, pass_threshold: float, soft_scale: float, weight: float, details: dict) -> WeakCheckResult:
    margin = float(metric - pass_threshold)
    score = float(1.0 / (1.0 + np.exp(-margin / max(soft_scale, 1e-6))))
    return WeakCheckResult(key=key, label=label, category=category, passed=margin >= 0.0, score=score, weight=weight, details={**details, "metric": metric, "threshold": pass_threshold, "margin": margin})


def _aggregate_by_category(checks: list[WeakCheckResult]) -> dict[str, float]:
    bucket: dict[str, list[WeakCheckResult]] = {}
    for check in checks:
        bucket.setdefault(check.category, []).append(check)
    return {
        category: float(sum(c.score * c.weight for c in cs) / max(sum(c.weight for c in cs), 1e-9))
        for category, cs in bucket.items()
    }


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or float(np.std(a)) < 1e-8 or float(np.std(b)) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _mask_mean(values: np.ndarray, mask: np.ndarray) -> float:
    return float(np.mean(values[mask])) if np.any(mask) else float(np.mean(values))
