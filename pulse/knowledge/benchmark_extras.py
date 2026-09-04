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
from .full_body import PatientParams, randomize_params, simulate_full_body
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

    # Iter 97 (review 5.6): calibration ends at 1440 and EVERY eval point is
    # strictly after it. Through iter 96 the check-in at 1560 was also an eval
    # point (6/13 glucose, 6/12 hr, 1/1 sbp samples on the `teacher` source were
    # fitted check-ins) -- the `sbp persistence 0.0 / skill 0.0` line in the
    # iter-96 report was that leakage.
    calibration_check_ins = []
    for t in (480, 960, 1440):
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
    # Unobserved-marker eval block — timestamps on day 2 only (after the last
    # check-in): postprandial peaks, overnight fasting trough, awakening.
    # 7 × 9 = 63 eval samples per cohort episode.
    eval_measurements = eval_measurements + _eval_block(
        traj,
        times=(1500, 1560, 1620, 1800, 2160, 2280, 2640),
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
        # Ground truth is the cold model the network also distils from —
        # scoring against it measures distillation fidelity, not physiology.
        source="teacher",
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

    # Calibration check-ins: pre-meal baseline + the FIRST meal's landmarks, on
    # the standard 5 measured markers (matches what real users would log).
    # Iter 97 (review 5.6): calibration ends at 180; the second meal (t=300)
    # and everything scored come strictly after it. Through iter 96 four of the
    # ten eval timestamps were fitted check-ins.
    calibration_check_ins = []
    for t in (30, 90, 120, 180):
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

    # Eval block — inter-meal trough, second-meal rise (t=330), peak (360),
    # recovery (420), end-of-window. All after the last check-in at 180.
    eval_times = (240, 330, 360, 420, 470)
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
        # Cold-model ground truth — circular vs the distillation teacher.
        source="teacher",
    )
    _meal_episodes_cache = [ep]
    return _meal_episodes_cache


_dynamic_episodes_cache: list[BenchmarkEpisode] | None = None

# Iter 94 — episodes whose SCORED window actually contains a meal.
#
# The ruler audit (scripts/iter94_ruler_audit.py, docs/iter94-proposal.md §0.6)
# found that on the 24 exported check-in episodes all 72 meals fall inside the
# CALIBRATION window and not one lands in the eval window, so the scored 3 hours
# are always quiescent: ground-truth within-episode sd is ~1 mg/dL glucose and
# 0.035 °C temp, and carrying the last reading forward beats every gate threshold
# by 11-50x. Every postprandial iteration since that dataset was exported has been
# graded almost entirely on flat windows.
#
# These episodes fix that directly: calibration ends at t=360, the eval window is
# t=390..690, and a meal lands at t=450 — inside it. They deliberately score only
# the five markers a real user measures, so they land in the SAME gate metrics as
# the real episodes rather than in a separate diagnostic bucket. Meal size and
# patient vary so the set is not one protocol repeated.
_DYNAMIC_CAL_END = 360
_DYNAMIC_EVAL_TIMES: tuple[int, ...] = tuple(range(390, 691, 30))
# 8 arms, not 4: per-marker statistics here are per-EPISODE MAPEs (one per episode,
# not one per eval point), so the arm count IS the sample size behind the skill
# threshold. Four was too thin to gate on.
_DYNAMIC_ARMS: tuple[tuple[str, float, tuple[tuple[float, float, float, float], ...]], ...] = (
    # label, start_hour, meals ((t, carbs, fats, proteins), ...)
    ("small-carb", 7.0, ((60.0, 40.0, 10.0, 15.0), (450.0, 40.0, 10.0, 15.0))),
    ("large-carb", 7.0, ((60.0, 60.0, 15.0, 20.0), (450.0, 100.0, 20.0, 25.0))),
    ("high-fat", 8.0, ((60.0, 50.0, 15.0, 20.0), (450.0, 30.0, 55.0, 30.0))),
    ("late-start", 11.0, ((60.0, 55.0, 12.0, 22.0), (450.0, 70.0, 18.0, 28.0))),
    ("protein-rich", 7.5, ((60.0, 45.0, 12.0, 18.0), (450.0, 35.0, 15.0, 55.0))),
    ("pure-glucose", 9.0, ((60.0, 50.0, 12.0, 18.0), (450.0, 75.0, 0.0, 0.0))),
    ("grazing", 7.0, ((60.0, 50.0, 12.0, 18.0), (420.0, 30.0, 8.0, 10.0),
                      (540.0, 30.0, 8.0, 10.0), (660.0, 30.0, 8.0, 10.0))),
    ("early-start", 5.5, ((60.0, 65.0, 18.0, 24.0), (450.0, 55.0, 14.0, 20.0))),
)


def cohort_meal_in_eval_window_episodes() -> list[BenchmarkEpisode]:
    """Teacher episodes that score a meal response instead of a flat window."""
    global _dynamic_episodes_cache
    if _dynamic_episodes_cache is not None:
        return _dynamic_episodes_cache

    duration_min = 720
    measured = ("glucose", "hr", "sbp", "dbp", "temp")
    out: list[BenchmarkEpisode] = []
    for k, (label, start_hour, meals) in enumerate(_DYNAMIC_ARMS):
        rng = np.random.default_rng(940_000 + k)
        # Vary the patient across arms so the set is not one body repeated; seeded,
        # so the ruler is reproducible episode-for-episode.
        params = PatientParams() if k == 0 else randomize_params(
            np.random.default_rng(94_100 + k))
        sw = np.ones(duration_min, dtype=np.float32)          # awake throughout
        activity = np.full(duration_min, 0.05, dtype=np.float32)
        traj, _abs = simulate_full_body(
            params, list(meals), sw, activity,
            duration_min, start_hour, noise_scale=0.001, rng=rng,
        )
        cal = [
            {
                "time": t,
                "measurements": {m: float(traj[t, MARKER_INDEX[m]]) for m in measured},
            }
            for t in range(30, _DYNAMIC_CAL_END + 1, 30)
        ]
        out.append(BenchmarkEpisode(
            user_id=f"benchmark-dynamic-{label}",
            duration_min=duration_min,
            initial_state=traj[0].astype(np.float32),
            meals=[MealEvent(time=float(t), carbs=float(c), fats=float(f),
                             proteins=float(p)) for t, c, f, p in meals],
            calibration_check_ins=cal,
            eval_measurements=_eval_block(traj, _DYNAMIC_EVAL_TIMES, measured),
            start_time_minutes=start_hour * 60.0,
            sleep_wake=sw,
            activity=activity,
            source="teacher_dynamic",
        ))
    _dynamic_episodes_cache = out
    return _dynamic_episodes_cache


def distillation_pool_episodes() -> list[BenchmarkEpisode]:
    """Cohort episodes the distillation signal may TRAIN on.

    Deliberately NOT the same set as `all_cohort_benchmark_episodes`. The
    cold-model distillation's ``pool="bench_cohorts"`` turns these episodes into
    training protocols, so anything in here is trained on as well as scored — the
    circularity the sleep/meal episodes already carry knowingly.

    The iter-94 `teacher_dynamic` episodes exist to measure whether the model can
    predict a meal response it has not been fitted to, so putting them in the
    training pool would answer that question with the answer written on the back.
    They are excluded here and appear only in the gate.
    """
    return (
        cohort_sleep_48h_benchmark_episodes()
        + cohort_meal_postprandial_benchmark_episodes()
    )


def last_check_in_time(ep: BenchmarkEpisode) -> int:
    """Time of the last calibration check-in (-1 if there are none)."""
    times = [int(round(float(c["time"]))) for c in ep.calibration_check_ins
             if isinstance(c, dict) and c.get("time") is not None]
    return max(times) if times else -1


def leaked_eval_points(ep: BenchmarkEpisode) -> list[MeasurementPoint]:
    """Eval points at or before the last calibration check-in.

    Iter 97 (review 5.6): scoring a fitted check-in is not a prediction. The
    in-process episodes are asserted leak-free at build time; the loader does
    not rewrite exported datasets, but the benchmark report can call this on
    any episode.
    """
    cutoff = last_check_in_time(ep)
    return [p for p in ep.eval_measurements if p.time <= cutoff]


def all_cohort_benchmark_episodes() -> list[BenchmarkEpisode]:
    """All cohort episodes injected into the bench gate at runtime."""
    eps = distillation_pool_episodes() + cohort_meal_in_eval_window_episodes()
    for ep in eps:
        leaked = leaked_eval_points(ep)
        if leaked:  # pragma: no cover - construction bug, not a data condition
            raise AssertionError(
                f"{ep.user_id}: {len(leaked)} eval point(s) at or before the last "
                f"check-in (t={last_check_in_time(ep)}) -- see review 5.6")
    return eps
