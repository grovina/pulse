"""
Cohort statistics: the iter-58 physiology breadth floor.

Rationale: `docs/physiology-coverage.md`. Twelve state
variables had zero population-level supervision; the embedding
bottleneck freely reshuffled representational capacity between them
(iter 57: ghrelin -0.40 while glp1 +0.66 in the same run, uncorrelated
with the mechanism under test). An unanchored marker is not merely
unmodelled — it is an active source of attribution noise that makes the
whole model an unreliable reference.

This file gives every previously-unanchored *cohort* marker ONE gentle,
uncontroversial, literature-cited anchor: a wide-sigma DELTA_MEANS (or a
level MEAN_IN_WINDOW) chosen so it cannot be physiologically "wrong" and
is wide enough not to fight existing fits. Sigma starts at roughly half
the marker's NORM_SCALE per the doc's gentle-calibration rule; tightening
is deferred to the depth campaign (iter 59+).

Two markers (acth, mito) are floored elsewhere and are intentionally not
here: `acth→cortisol +1` and `mitochondrial_capacity→ffa +1` are
sign-only coupling priors (`knowledge.coupling_priors`), not cohort
deltas — encoding a level target for a latent driver would be the wrong
primitive.

Arm protocols deliberately mirror the shapes already used by the
sleep / cardiovascular / nutrition slices (same meal schedule, same
sleep mask construction) so the model sees consistent contexts across
the cohort surface.
"""

from __future__ import annotations

import numpy as np

from ..cohort_types import (
    CohortArmSpec,
    CohortStatisticSpec,
    InitMode,
    StatisticKind,
    StatisticWindow,
)

# ---------------------------------------------------------------------------
# Shared arm protocols
# ---------------------------------------------------------------------------
# MEMORY BUDGET (load-bearing): Phase 2 sums every cohort spec's
# rollout autograd graph before one backward, so peak memory scales with
# sum(arm.duration_min) across ALL specs x cohort-sample-patients. iter
# 58's first dispatch (exec x4msq) was SIGKILLed (OOM, 32 Gi) at the
# Phase 1->2 boundary: the original breadth arms added ~40k step-rollouts
# /patient on top of iter-57's load (which already carries one 2x2880
# SLEEP_RESTRICTION pair and barely fit). Every arm here is therefore cut
# to the minimum duration its statistic needs — no 48 h arms, no full-day
# arm where a window-spanning sub-day arm suffices. Do not lengthen these
# without re-checking the OOM headroom.


def _smooth(x: np.ndarray, k: int = 20) -> tuple[float, ...]:
    kernel = np.ones(k, dtype=np.float32) / float(k)
    sm = np.clip(np.convolve(x, kernel, mode="same"), 0.0, 1.0)
    return tuple(float(v) for v in sm)


def _moderate_bout(duration_min: int, lo: int, hi: int, level: float) -> tuple[float, ...]:
    """Rest except a single moderate-intensity bout in [lo, hi)."""
    act = np.zeros(duration_min, dtype=np.float32)
    act[lo:hi] = level
    return _smooth(act, k=10)


# Nocturnal-dip pair: 12 h evening->morning (start 19:00, 720 min).
# awake-throughout vs asleep 23:00–07:00. Window 02:00–05:00 (mid-NREM,
# minutes 420–600). One light dinner so the arms aren't pure-fasted.
# Shared by all four dip anchors so the parasympathetic/circadian
# contrast is isolated from meal/activity confounds.
_DIP_DUR = 720
_DIP_MEALS = ((60.0, 60.0, 15.0, 30.0),)  # ~20:00 dinner
_dip_sleep = np.ones(_DIP_DUR, dtype=np.float32)
_dip_sleep[240:720] = 0.0  # 23:00 (min 240) -> 07:00 (min 720, end)
_AWAKE_DIP = CohortArmSpec(
    label="awake_eve_to_morn", duration_min=_DIP_DUR, start_hour=19.0,
    meals=_DIP_MEALS, sleep_wake=tuple([1.0] * _DIP_DUR),
)
_SLEEP_DIP = CohortArmSpec(
    label="sleep_23to07", duration_min=_DIP_DUR, start_hour=19.0,
    meals=_DIP_MEALS, sleep_wake=_smooth(_dip_sleep),
)
_DIP_ARMS = (_AWAKE_DIP, _SLEEP_DIP)
_DIP_WINDOW = StatisticWindow(start_min=420, end_min=600)

# CAR pair: 9 h (start 23:00, 540 min). Asleep 23:00–07:00 (min 0–480),
# awake 07:00–08:00. Two identical arms + per_arm_windows: pre-wake
# (last hour of sleep) vs 0–45 min post-wake. Fasted (CAR is a
# pre-breakfast phenomenon). 2x540 — replaces the original 2x2880.
_CAR_DUR = 540
_car_sleep = np.ones(_CAR_DUR, dtype=np.float32)
_car_sleep[0:480] = 0.0  # asleep 23:00 -> 07:00 (wake at min 480)
_CAR_ARM = CohortArmSpec(
    label="overnight_sleep_wake", duration_min=_CAR_DUR, start_hour=23.0,
    meals=(), sleep_wake=_smooth(_car_sleep),
)

# Core-temp circadian pair: 17 h (start 14:00, 1020 min) spanning the
# late-afternoon acrophase and the next early-morning nadir. Asleep
# 23:00–07:00 (min 540–1020). Two identical arms + per_arm_windows:
# acrophase 16–18 h (min 120–240) vs nadir 04–06 h (min 840–960).
# 2x1020 — replaces the original 2x2880.
_TEMP_DUR = 1020
_temp_sleep = np.ones(_TEMP_DUR, dtype=np.float32)
_temp_sleep[540:1020] = 0.0  # 23:00 (min 540) -> 07:00 (min 1020, end)
_TEMP_ARM = CohortArmSpec(
    label="afternoon_to_nadir", duration_min=_TEMP_DUR, start_hour=14.0,
    meals=((300.0, 60.0, 15.0, 30.0),),  # ~19:00 dinner
    sleep_wake=_smooth(_temp_sleep),
)

# SpO2 level anchor: short 8 h awake arm (start 08:00, 480 min), one
# breakfast. Window [60, 420] (skip the first-hour settle).
_SPO2_ARM = CohortArmSpec(
    label="awake_day_short", duration_min=480, start_hour=8.0,
    meals=((30.0, 50.0, 10.0, 15.0),), sleep_wake=tuple([1.0] * 480),
)

# Fasted exercise pair (no meals, isolates the activity drive). One
# 2 h moderate bout (60–180 min) at ~0.65 intensity vs rest. Shared by
# the lactate (acute rise) and muscle_glycogen (depletion) anchors.
_EX_REST = CohortArmSpec(
    label="rest_fasted", duration_min=360, start_hour=8.0,
    meals=(), activity=tuple([0.0] * 360),
)
_EX_BOUT = CohortArmSpec(
    label="moderate_bout_fasted", duration_min=360, start_hour=8.0,
    meals=(), activity=_moderate_bout(360, 60, 180, 0.65),
)
_EX_ARMS = (_EX_REST, _EX_BOUT)


# ---------------------------------------------------------------------------
# Cardiovascular / respiratory nocturnal dips (4 anchors on the shared pair)
# ---------------------------------------------------------------------------
# Task Force (1996): vagally-mediated HRV (RMSSD) is markedly higher in
# NREM sleep than in quiet wakefulness. Gentle +12 ms ± 7.5.
HRV_SLEEP_RISE = CohortStatisticSpec(
    name="hrv_sleep_rise",
    source="Task Force of ESC/NASPE (1996) — heart rate variability",
    description="Mid-NREM HRV (RMSSD) +12 ms vs hypothetical-awake same window",
    arms=_DIP_ARMS, marker_id="hrv", kind=StatisticKind.DELTA_MEANS,
    window=_DIP_WINDOW, target=12.0, sigma=7.5,
)

# Dipper pattern (O'Brien 1988; Staessen 1997): healthy nocturnal SBP
# falls ~10–20% (≈ -10–20 mmHg) vs daytime. Gentle -10 mmHg ± 5.
SBP_SLEEP_DIP = CohortStatisticSpec(
    name="sbp_sleep_dip",
    source="O'Brien et al. (1988); Staessen et al. (1997) — nocturnal BP dipping",
    description="Mid-NREM systolic BP -10 mmHg vs hypothetical-awake same window",
    arms=_DIP_ARMS, marker_id="sbp", kind=StatisticKind.DELTA_MEANS,
    window=_DIP_WINDOW, target=-10.0, sigma=5.0,
)

# Dipper pattern: nocturnal DBP falls ~10–15% (≈ -7–12 mmHg). Gentle
# -7 mmHg ± 4.
DBP_SLEEP_DIP = CohortStatisticSpec(
    name="dbp_sleep_dip",
    source="O'Brien et al. (1988); Staessen et al. (1997) — nocturnal BP dipping",
    description="Mid-NREM diastolic BP -7 mmHg vs hypothetical-awake same window",
    arms=_DIP_ARMS, marker_id="dbp", kind=StatisticKind.DELTA_MEANS,
    window=_DIP_WINDOW, target=-7.0, sigma=4.0,
)

# Douglas et al. (1982): respiratory rate falls in NREM sleep vs
# wakefulness. Gentle -2.5 /min ± 1.5.
RR_SLEEP_DIP = CohortStatisticSpec(
    name="rr_sleep_dip",
    source="Douglas et al. (1982) — respiration during sleep",
    description="Mid-NREM respiratory rate -2.5 /min vs hypothetical-awake same window",
    arms=_DIP_ARMS, marker_id="rr", kind=StatisticKind.DELTA_MEANS,
    window=_DIP_WINDOW, target=-2.5, sigma=1.5,
)


# ---------------------------------------------------------------------------
# Respiratory level: SpO2 stays in the normal band
# ---------------------------------------------------------------------------
# Textbook: resting daytime SpO2 in healthy adults sits 95–100%. A pure
# level anchor (MEAN_IN_WINDOW), wide sigma so it pins the band without
# dictating fine dynamics.
SPO2_NORMAL_BAND = CohortStatisticSpec(
    name="spo2_normal_band",
    source="textbook resting pulse oximetry (95–100% in healthy adults)",
    description="Daytime mean SpO2 ≈ 98% (normal band)",
    arms=(_SPO2_ARM,), marker_id="spo2", kind=StatisticKind.MEAN_IN_WINDOW,
    window=StatisticWindow(start_min=60, end_min=420), target=98.0, sigma=1.5,
)


# ---------------------------------------------------------------------------
# HPA: cortisol awakening response (CAR)
# ---------------------------------------------------------------------------
# Pruessner et al. (1997): cortisol rises ~50–75% within 30–45 min of
# morning awakening. With typical ≈12 µg/dL a +50% rise is ≈ +6 µg/dL.
# Two windows on one overnight arm: pre-wake baseline (last hour asleep)
# vs 0–45 min post the 07:00 awakening (minute 480). Gentle +6 ± 4.
CORTISOL_AWAKENING_RESPONSE = CohortStatisticSpec(
    name="cortisol_awakening_response",
    source="Pruessner et al. (1997) — cortisol awakening response",
    description="Cortisol +6 µg/dL in 0–45 min post-wake vs pre-wake baseline",
    arms=(_CAR_ARM, _CAR_ARM),
    marker_id="cortisol", kind=StatisticKind.DELTA_MEANS,
    window=StatisticWindow(start_min=480, end_min=525),  # unused (per_arm)
    per_arm_windows=(
        StatisticWindow(start_min=420, end_min=480),  # pre-wake (last hour asleep)
        StatisticWindow(start_min=480, end_min=525),  # 0–45 min post-wake
    ),
    target=6.0, sigma=4.0,
)


# ---------------------------------------------------------------------------
# Thermoregulation: core-temperature circadian rhythm
# ---------------------------------------------------------------------------
# Core-temp rhythm (Krauchi & Wirz-Justice 1994): nadir ~04–06 h,
# acrophase late afternoon, amplitude ≈0.4–0.5 °C. Two windows on one
# 17 h arm: late-afternoon acrophase (16–18 h) vs next-morning nadir
# (04–06 h). Target -0.5 ± 0.3, wide sigma so it ratifies (not
# flattens) the cold model's existing rhythm — a gentle phase/sign
# floor, not an amplitude correction (that is depth-campaign work).
# Re-validated by cold-calibration after the arm resize (within band).
TEMP_CIRCADIAN = CohortStatisticSpec(
    name="temp_circadian_nadir",
    source="Krauchi & Wirz-Justice (1994) — core body temperature rhythm",
    description="Core temp early-morning nadir -0.4 °C vs late-afternoon acrophase",
    arms=(_TEMP_ARM, _TEMP_ARM),
    marker_id="temp", kind=StatisticKind.DELTA_MEANS,
    window=StatisticWindow(start_min=840, end_min=960),  # unused (per_arm)
    per_arm_windows=(
        StatisticWindow(start_min=120, end_min=240),  # 16–18 h (acrophase)
        StatisticWindow(start_min=840, end_min=960),  # next 04–06 h (nadir)
    ),
    target=-0.5, sigma=0.3,
)


# ---------------------------------------------------------------------------
# Appetite: leptin tracks fed vs fasted energy state
# ---------------------------------------------------------------------------
# Boden et al. (1996); Kolaczynski et al. (1996): a ≈16 h fast lowers
# circulating leptin; conversely the fed state runs higher. Reuses the
# nutrition fed vs extended-fast 1-day arms. arm[1]-arm[0] = fed - fasted.
# +2 ng/mL ± 1.0. weight=12: the iter-57 cohort-ablation showed the
# raw leptin gradient (~0.08 on the appetite module) is ~100x quieter
# than the ghrelin spec sharing that module — pure iter-56-style
# gradient starvation (large residual, near-zero gradient). Per the
# SLEEP_HR_DIP precedent (weight 1.0->3.0 for the same reason), bump
# the per-spec weight + tighten sigma so the anchor actually bites;
# leptin is observed and was a named victim of the embedding-reshuffle
# noise this floor exists to kill, so it must bite, not merely exist.
# NORM_CENTER init (free-living-ish, fed start).
_FED_DAY = (
    (120.0, 50.0, 8.0, 15.0),
    (360.0, 65.0, 12.0, 25.0),
    (720.0, 60.0, 15.0, 30.0),
)
_FASTED_DAY = (
    (120.0, 50.0, 8.0, 15.0),
    (360.0, 65.0, 12.0, 25.0),
)
LEPTIN_FED_FASTED = CohortStatisticSpec(
    name="leptin_fed_vs_fasted",
    source="Boden et al. (1996); Kolaczynski et al. (1996) — leptin and fasting",
    description="Fed-state late-day mean leptin +2 ng/mL vs ≈16 h-fasted",
    arms=(
        CohortArmSpec(label="extended_fast", duration_min=1440, start_hour=6.0, meals=_FASTED_DAY),
        CohortArmSpec(label="normal_eating", duration_min=1440, start_hour=6.0, meals=_FED_DAY),
    ),
    marker_id="leptin", kind=StatisticKind.DELTA_MEANS,
    window=StatisticWindow(start_min=960, end_min=1320),
    target=2.0, sigma=1.0, weight=12.0,
)


# ---------------------------------------------------------------------------
# Exercise: acute lactate rise + muscle-glycogen depletion (shared arms)
# ---------------------------------------------------------------------------
# Brooks (1986): moderate exercise raises blood lactate ≥1–3 mmol/L above
# the ≈1.0 mmol/L resting level. arm[1]-arm[0] = bout - rest. Window spans
# the bout and early recovery. Gentle +1.5 ± 0.6. NORM_CENTER init
# (the fasted-rest cold trajectory is a poor run-in for an exercise arm).
LACTATE_EXERCISE_RISE = CohortStatisticSpec(
    name="lactate_exercise_rise",
    source="Brooks (1986) — lactate metabolism during exercise",
    description="Blood lactate +1.5 mmol/L during a moderate bout vs rest",
    arms=_EX_ARMS, marker_id="lactate", kind=StatisticKind.DELTA_MEANS,
    window=StatisticWindow(start_min=90, end_min=210),
    target=1.5, sigma=0.6, init_mode=InitMode.NORM_CENTER,
)

# Bergstrom & Hultman (1967): exercise depletes muscle glycogen
# substantially while rest preserves it. -25 g ± 30 (wide). weight=3.0
# (SLEEP_HR_DIP precedent value): the iter-57 ablation showed muscle's
# raw gradient is ~0.055 — but unlike leptin this is *structurally*
# expected, not a tuning miss. muscle_glycogen's cons_scale (3.3e-5,
# tau ~ 3 wk by design) genuinely cannot move much on a 1-day bout —
# the same R2 timescale-vs-protocol mismatch iter 55 hit. The floor's
# job here is only to put a *signed* anchor on the otherwise-zero
# activity->muscle pathway (a coupling prior is impossible: `activity`
# is a metabolic external, not a state marker). Real muscle dynamics
# need the multi-week chronic-exercise protocol — explicitly the
# iter-59+ depth campaign (docs/physiology-coverage.md);
# do NOT fight the deliberate slow tau with a large weight here.
MUSCLE_GLYCOGEN_EXERCISE_DEPLETION = CohortStatisticSpec(
    name="muscle_glycogen_exercise_depletion",
    source="Bergstrom & Hultman (1967) — muscle glycogen and exercise",
    description="Muscle glycogen -25 g post moderate bout vs rest",
    arms=_EX_ARMS, marker_id="muscle_glycogen", kind=StatisticKind.DELTA_MEANS,
    window=StatisticWindow(start_min=150, end_min=360),
    target=-25.0, sigma=30.0, weight=3.0, init_mode=InitMode.NORM_CENTER,
)


COHORT_STATISTICS: list[CohortStatisticSpec] = [
    HRV_SLEEP_RISE,
    SBP_SLEEP_DIP,
    DBP_SLEEP_DIP,
    RR_SLEEP_DIP,
    SPO2_NORMAL_BAND,
    CORTISOL_AWAKENING_RESPONSE,
    TEMP_CIRCADIAN,
    LEPTIN_FED_FASTED,
    LACTATE_EXERCISE_RISE,
    MUSCLE_GLYCOGEN_EXERCISE_DEPLETION,
]
