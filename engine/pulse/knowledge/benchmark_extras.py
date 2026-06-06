"""
Extra benchmark episodes aligned with cohort protocols (traceable validation targets).

Merged in train._run_benchmark so the gate exercises long sleep schedules without
regenerating the large exported check-in dataset (PRD: provenance on disk).

Item-A coverage of unobserved markers
-------------------------------------

The exported real-user bench (``benchmark.dataset.generated.json``) only
emits eval points for the 5 markers users actually self-measure (glucose,
hr, sbp, dbp, temp). The dead-pathway problem (5 markers stuck flat —
glucagon / ffa / ghrelin / leptin / acth — per the iter-39 probe) was
invisible to the gate.

The cohort episodes here are synthetic — ground truth comes from
``simulate_full_body`` (cold-model rollout), so we have access to all
19 markers' time series. We use that to score the dead-pathway markers
and other unobserved-but-load-bearing markers (insulin, cortisol, glp1,
bhb) at multiple time points across each cohort episode. The gate will
report per-marker ``mean_mape`` for any marker that has eval samples;
the ``thresholds.json`` decides which markers actually fail the gate.

For v1 we add eval points but no new gate thresholds — the report is
diagnostic. Iter 41+ can promote markers into the gate once we have
baseline numbers from iter 38.
"""

from __future__ import annotations

import numpy as np

from .cohorts.sleep import (
    SLEEP_COHORT_48H_MEALS,
    SLEEP_COHORT_48H_START_HOUR,
    sleep_two_nights_adequate,
)
from .full_body import PatientParams, simulate_full_body
from ..benchmark import BenchmarkEpisode, MeasurementPoint
from ..modules.gut import MealEvent
from ..types import MARKER_INDEX

_sleep_episodes_cache: list[BenchmarkEpisode] | None = None
_meal_episodes_cache: list[BenchmarkEpisode] | None = None

# Markers we add per-time-point on cohort episodes for unobserved-marker
# visibility. Order chosen to span the 4 modules currently dead in the
# iter-39 probe (metabolic counter-regulation, appetite, stress) plus
# ones needed to interpret post-meal dynamics (insulin, glp1).
_DEAD_PATHWAY_MARKERS: tuple[str, ...] = (
    "insulin", "glucagon", "ffa", "bhb",
    "ghrelin", "leptin", "glp1",
    "cortisol", "acth",
)


def _eval_block(
    traj: np.ndarray,
    times: tuple[int, ...],
    markers: tuple[str, ...],
) -> list[MeasurementPoint]:
    """Build a list of MeasurementPoints from a cold-model trajectory."""
    out: list[MeasurementPoint] = []
    for t in times:
        for m in markers:
            out.append(MeasurementPoint(
                time=t, marker_id=m, value=float(traj[t, MARKER_INDEX[m]]),
            ))
    return out


def cohort_sleep_48h_benchmark_episodes() -> list[BenchmarkEpisode]:
    """Two-day adequate-sleep protocol matching cohort control arm; targets from cold model."""
    global _sleep_episodes_cache
    if _sleep_episodes_cache is not None:
        return _sleep_episodes_cache

    duration_min = 2880
    start_hour = SLEEP_COHORT_48H_START_HOUR
    meals = list(SLEEP_COHORT_48H_MEALS)
    sw = np.array(sleep_two_nights_adequate(duration_min, start_hour), dtype=np.float32)
    activity = np.full(duration_min, 0.05, dtype=np.float32)
    rng = np.random.default_rng(20260412)
    params = PatientParams()
    traj, _abs = simulate_full_body(
        params, meals, sw, activity,
        duration_min, start_hour, noise_scale=0.001, rng=rng,
    )
    initial_state = traj[0].astype(np.float32)
    t0_min = start_hour * 60.0

    meal_events = [
        MealEvent(time=float(t), carbs=float(c), fats=float(f), proteins=float(p))
        for t, c, f, p in meals
    ]

    def pt(t: int, mids: tuple[str, ...]) -> list[MeasurementPoint]:
        return [
            MeasurementPoint(time=t, marker_id=m, value=float(traj[t, MARKER_INDEX[m]]))
            for m in mids
        ]

    calibration_check_ins = []
    for t in (480, 960, 1440, 1560):
        calibration_check_ins.append({
            "time": t,
            "measurements": {
                "glucose": float(traj[t, MARKER_INDEX["glucose"]]),
                "hr": float(traj[t, MARKER_INDEX["hr"]]),
                "sbp": float(traj[t, MARKER_INDEX["sbp"]]),
                "dbp": float(traj[t, MARKER_INDEX["dbp"]]),
                "temp": float(traj[t, MARKER_INDEX["temp"]]),
            },
        })

    # Existing eval block: real-user-style sparse sampling on the 5 measured markers.
    eval_measurements = (
        pt(1560, ("glucose", "hr", "sbp"))
        + pt(1620, ("glucose", "hr"))
        + pt(1680, ("glucose", "temp"))
    )
    # Unobserved-marker eval block — 11 timestamps spanning both protocol days
    # (pre-meal, postprandial peaks, overnight fasting trough, awakening).
    # 11 × 9 = 99 new eval samples per cohort episode.
    eval_measurements = eval_measurements + _eval_block(
        traj,
        times=(360, 720, 840, 1080, 1500, 1560, 1620, 1800, 2160, 2280, 2640),
        markers=_DEAD_PATHWAY_MARKERS,
    )

    ep = BenchmarkEpisode(
        user_id="benchmark-cohort-sleep-48h-adequate",
        duration_min=duration_min,
        initial_state=initial_state,
        meals=meal_events,
        calibration_check_ins=calibration_check_ins,
        eval_measurements=eval_measurements,
        start_time_minutes=t0_min,
        sleep_wake=sw,
        activity=activity,
    )
    _sleep_episodes_cache = [ep]
    return _sleep_episodes_cache


# Standard 75g-carb breakfast at t=60, dinner-equivalent at t=300.
# Window covers overnight-fasting baseline (low glucose, elevated FFA/ghrelin/
# glucagon, morning cortisol peak) → first-meal response → late-afternoon
# fasting drift → second-meal response.
_OGTT_MEALS: tuple[tuple[float, float, float, float], ...] = (
    (60.0, 75.0, 5.0, 10.0),     # OGTT-style breakfast: 75g carbs, low fat/protein
    (300.0, 60.0, 20.0, 25.0),   # mixed dinner: 60g carbs, 20g fat, 25g protein
)


def cohort_meal_postprandial_benchmark_episodes() -> list[BenchmarkEpisode]:
    """8h fed-state cohort: OGTT-style breakfast + mixed dinner.

    Targets the metabolic / appetite / stress modules under explicit
    nutrient stimulation. Unlike the sleep cohort (where dead-pathway
    markers see mostly fasted dynamics), this episode forces postprandial
    insulin / glucagon / glp1 / ghrelin signatures into the eval window.
    """
    global _meal_episodes_cache
    if _meal_episodes_cache is not None:
        return _meal_episodes_cache

    duration_min = 480  # 8 hours (07:00 → 15:00)
    start_hour = 7.0
    meals = list(_OGTT_MEALS)
    sw = np.zeros(duration_min, dtype=np.float32) + 1.0  # awake throughout
    activity = np.full(duration_min, 0.1, dtype=np.float32)  # sedentary baseline
    rng = np.random.default_rng(20260508)
    params = PatientParams()
    traj, _abs = simulate_full_body(
        params, meals, sw, activity,
        duration_min, start_hour, noise_scale=0.001, rng=rng,
    )
    initial_state = traj[0].astype(np.float32)
    t0_min = start_hour * 60.0

    meal_events = [
        MealEvent(time=float(t), carbs=float(c), fats=float(f), proteins=float(p))
        for t, c, f, p in meals
    ]

    # Calibration check-ins: pre-meal baseline + each meal landmark, on the
    # standard 5 measured markers (matches what real users would log).
    calibration_check_ins = []
    for t in (30, 90, 120, 180, 330, 420):
        calibration_check_ins.append({
            "time": t,
            "measurements": {
                "glucose": float(traj[t, MARKER_INDEX["glucose"]]),
                "hr": float(traj[t, MARKER_INDEX["hr"]]),
                "sbp": float(traj[t, MARKER_INDEX["sbp"]]),
                "dbp": float(traj[t, MARKER_INDEX["dbp"]]),
                "temp": float(traj[t, MARKER_INDEX["temp"]]),
            },
        })

    # Eval block — pre-meal, peak (60min), recovery (120min), inter-meal trough,
    # second-meal peak, and end-of-window.
    eval_times = (45, 90, 120, 150, 180, 240, 330, 360, 420, 470)
    eval_measurements = (
        # Standard markers — sparse, matching real-user style
        _eval_block(traj, eval_times, ("glucose", "hr"))
        # Dead-pathway markers — full coverage
        + _eval_block(traj, eval_times, _DEAD_PATHWAY_MARKERS)
    )

    ep = BenchmarkEpisode(
        user_id="benchmark-cohort-meal-postprandial",
        duration_min=duration_min,
        initial_state=initial_state,
        meals=meal_events,
        calibration_check_ins=calibration_check_ins,
        eval_measurements=eval_measurements,
        start_time_minutes=t0_min,
        sleep_wake=sw,
        activity=activity,
    )
    _meal_episodes_cache = [ep]
    return _meal_episodes_cache


def all_cohort_benchmark_episodes() -> list[BenchmarkEpisode]:
    """All cohort episodes injected into the bench gate at runtime."""
    return cohort_sleep_48h_benchmark_episodes() + cohort_meal_postprandial_benchmark_episodes()
