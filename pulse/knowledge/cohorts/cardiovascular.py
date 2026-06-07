"""
Cohort statistics: heart-rate amplitude anchors.

Iter 33's HR landmark gradient was the first signal that ever moved
hr_mape across iters 27-33, but iter 34 (HR-only landmark) showed it
isn't enough on its own — narrowing landmark to HR alone partially
regressed hr_mape vs iter 33 (0.218 → 0.233) because the per-window
landmark term was sample-starved (~26 meal·hr pairs/epoch). These
cohort specs provide quantitative literature anchors that the model
hits at population scale, complementing the per-window landmark
signal with a per-arm amplitude target.

Two anchors:
- Postprandial HR rise: mixed-meal arm vs fasted arm, peak HR Δ.
- Sleep HR dip: sleep arm vs evening (awake) arm, mean HR Δ.

Both reuse arm patterns from existing nutrition / sleep specs to keep
the cohort rollout cost amortized.
"""

from __future__ import annotations

from ..cohort_types import (
    CohortArmSpec,
    CohortStatisticSpec,
    StatisticKind,
    StatisticWindow,
)


# ---------------------------------------------------------------------------
# Postprandial HR rise — mixed meal vs fasted contrast
# ---------------------------------------------------------------------------
# Brunzell, Robertson & Lerner (1971); Kearney et al. (1995); Marfella
# et al. (2000): a mixed meal raises HR ~5–10 bpm above pre-meal baseline
# at peak (typically 30–90 min postprandial), driven by sympathetic
# activation and cardiac output increase to support splanchnic blood
# flow. Encoded as DELTA_PEAKS: peak HR in 30–120 min post-meal window
# of the meal arm minus peak HR of the same window in a fasted arm.
# Target +7 bpm ± 3.
_HR_FASTED: tuple[tuple[float, float, float, float], ...] = ()
_HR_MEAL = ((60.0, 75.0, 15.0, 25.0),)
POSTPRANDIAL_HR_RISE = CohortStatisticSpec(
    name="postprandial_hr_rise",
    source="Brunzell et al. (1971); Kearney et al. (1995); Marfella et al. (2000)",
    description="Mixed meal (75g carb) → HR peak +7 bpm vs fasted arm at 30–120 min post-meal",
    arms=(
        CohortArmSpec(label="fasted", duration_min=300, start_hour=8.0, meals=_HR_FASTED),
        CohortArmSpec(label="meal", duration_min=300, start_hour=8.0, meals=_HR_MEAL),
    ),
    marker_id="hr",
    kind=StatisticKind.DELTA_PEAKS,
    window=StatisticWindow(start_min=90, end_min=180),
    target=7.0,
    sigma=3.0,
)


# ---------------------------------------------------------------------------
# Sleep HR dip — sleep mean vs evening-awake mean
# ---------------------------------------------------------------------------
# Somers et al. (1993); Burgess et al. (1997); Trinder et al. (2001):
# nocturnal HR is ~6–12 bpm lower than evening awake HR in healthy
# adults, driven by parasympathetic dominance during NREM sleep.
# Encoded as DELTA_MEANS: mean HR in mid-sleep window of an
# adequate-sleep arm minus mean HR in the corresponding evening window
# of an awake arm. Target -8 bpm ± 4.
#
# Use a 24h arm starting at 7am: evening window 14:00–17:00 (840–1020
# min) is post-lunch awake; sleep window 02:00–05:00 (1140–1320 min)
# is mid-NREM. The contrast isolates the parasympathetic dip from any
# meal/circadian confounds.
_DAY_AWAKE_START_HOUR = 7.0
_DAY_AWAKE_DURATION = 1440  # 24h in minutes
_DAY_AWAKE_MEALS = (
    (60.0, 50.0, 10.0, 15.0),     # 8am: breakfast
    (420.0, 65.0, 12.0, 25.0),    # 14:00: lunch
    (780.0, 60.0, 15.0, 30.0),    # 20:00: dinner
)


def _adequate_sleep_24h() -> tuple[float, ...]:
    """Awake all day, asleep 23:00–07:00 (16h–24h of arm)."""
    sw = [1.0] * _DAY_AWAKE_DURATION
    bt = int((23.0 - _DAY_AWAKE_START_HOUR) * 60)  # 16:00 of arm = 11pm
    wt = _DAY_AWAKE_DURATION
    for i in range(bt, wt):
        sw[i] = 0.0
    # 20-min smoothing kernel like the 48h sleep helper.
    smoothed = list(sw)
    k = 20
    for i in range(_DAY_AWAKE_DURATION):
        lo, hi = max(0, i - k // 2), min(_DAY_AWAKE_DURATION, i + k // 2)
        smoothed[i] = sum(sw[lo:hi]) / (hi - lo)
    return tuple(smoothed)


def _all_awake_24h() -> tuple[float, ...]:
    return tuple([1.0] * _DAY_AWAKE_DURATION)


SLEEP_HR_DIP = CohortStatisticSpec(
    name="sleep_hr_dip",
    source="Somers et al. (1993); Burgess et al. (1997); Trinder et al. (2001)",
    description="Sleep arm mid-NREM mean HR -8 bpm vs awake-arm mid-afternoon mean",
    arms=(
        CohortArmSpec(
            label="awake_24h",
            duration_min=_DAY_AWAKE_DURATION, start_hour=_DAY_AWAKE_START_HOUR,
            meals=_DAY_AWAKE_MEALS, sleep_wake=_all_awake_24h(),
        ),
        CohortArmSpec(
            label="sleep_23to07",
            duration_min=_DAY_AWAKE_DURATION, start_hour=_DAY_AWAKE_START_HOUR,
            meals=_DAY_AWAKE_MEALS, sleep_wake=_adequate_sleep_24h(),
        ),
    ),
    marker_id="hr",
    kind=StatisticKind.DELTA_MEANS,
    # arm[1] (sleep) - arm[0] (awake), so target is negative for the dip.
    window=StatisticWindow(start_min=1140, end_min=1320),
    # Meaningful dip on arm[1] requires both arms evaluated in the same
    # absolute window; the awake arm's 1140-1320 window is the wee hours
    # but with sleep_wake=1.0 throughout, so it's "hypothetical awake at
    # 02:00-05:00". Imperfect baseline but isolates the parasympathetic
    # contribution to the dip.
    target=-8.0,
    sigma=4.0,
    # Iter 40: per-spec weight bumped 1.0 → 3.0 after Bayesian validation
    # on Gabriel's real overnight CGM+Oura HR (docs/bayesian-and-real-data.md)
    # showed iter-38 over-predicts HR drift by +1.5 bpm/h across 14
    # overnight episodes. The base Somers spec exists but at default
    # weight 1.0 / cohort_statistic_weight 0.15 / 19 specs sharing
    # budget, its effective gradient is ~0.008 — too small to bite.
    weight=3.0,
)


COHORT_STATISTICS: list[CohortStatisticSpec] = [
    POSTPRANDIAL_HR_RISE,
    SLEEP_HR_DIP,
]
