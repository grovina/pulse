"""
Cohort statistics: sleep and circadian manipulation.
"""

from __future__ import annotations

import numpy as np

from ..cohort_types import (
    CohortArmSpec,
    CohortStatisticSpec,
    StatisticKind,
    StatisticWindow,
)


def _smooth_sleep_mask(sw: np.ndarray) -> np.ndarray:
    kernel = np.ones(20, dtype=np.float32) / 20.0
    return np.clip(np.convolve(sw, kernel, mode="same"), 0.0, 1.0).astype(np.float32)


def sleep_two_nights_adequate(duration_min: int, start_hour: float) -> tuple[float, ...]:
    """~11pm–~7am sleep both nights (wake hour 31 = 7am next calendar day)."""
    sw = np.ones(duration_min, dtype=np.float32)
    for day_start in (0, 1440):
        if day_start >= duration_min:
            break
        bt = int((23.0 - start_hour) * 60) + day_start
        wt = int((31.0 - start_hour) * 60) + day_start
        lo, hi = max(0, bt), min(duration_min, wt)
        if lo < hi:
            sw[lo:hi] = 0.0
    sw = _smooth_sleep_mask(sw)
    return tuple(float(x) for x in sw)


def sleep_first_night_restricted_second_adequate(duration_min: int, start_hour: float) -> tuple[float, ...]:
    """First night ~4 h sleep (~11pm–~3am); second night same adequate window."""
    sw = np.ones(duration_min, dtype=np.float32)
    bt0 = int((23.0 - start_hour) * 60)
    wt_short = int((27.0 - start_hour) * 60)
    lo0, hi0 = max(0, bt0), min(duration_min, wt_short)
    if lo0 < hi0:
        sw[lo0:hi0] = 0.0
    day_start = 1440
    if day_start < duration_min:
        bt = int((23.0 - start_hour) * 60) + day_start
        wt = int((31.0 - start_hour) * 60) + day_start
        lo, hi = max(0, bt), min(duration_min, wt)
        if lo < hi:
            sw[lo:hi] = 0.0
    sw = _smooth_sleep_mask(sw)
    return tuple(float(x) for x in sw)


_SLEEP_START_HOUR = 7.0
_MEALS_48H = (
    (60.0, 50.0, 10.0, 15.0),
    (420.0, 65.0, 12.0, 25.0),
    (780.0, 60.0, 15.0, 30.0),
    (1500.0, 50.0, 10.0, 15.0),
    (1860.0, 65.0, 12.0, 25.0),
    (2220.0, 60.0, 15.0, 30.0),
)
_short_sleep_48 = sleep_first_night_restricted_second_adequate(2880, _SLEEP_START_HOUR)
_adequate_sleep_48 = sleep_two_nights_adequate(2880, _SLEEP_START_HOUR)

# Spiegel et al. (1999, 2009); Donga et al. (2010): one night of partial
# sleep restriction increases morning glucose / impairs insulin sensitivity
# such that next-day mean glucose runs ~5–10 mg/dL higher in a pre-lunch
# window. Use Δ = +6 mg/dL ± 4.
SLEEP_RESTRICTION_NEXT_DAY_GLUCOSE = CohortStatisticSpec(
    name="sleep_restriction_next_day_glucose",
    source="Spiegel et al. (1999, 2009); Donga et al. (2010) — sleep loss and glycemia",
    description="One night of restricted sleep → +6 mg/dL mean glucose next morning (pre-lunch window)",
    arms=(
        CohortArmSpec(
            label="adequate_sleep",
            duration_min=2880, start_hour=_SLEEP_START_HOUR,
            meals=_MEALS_48H, sleep_wake=_adequate_sleep_48,
        ),
        CohortArmSpec(
            label="short_first_night",
            duration_min=2880, start_hour=_SLEEP_START_HOUR,
            meals=_MEALS_48H, sleep_wake=_short_sleep_48,
        ),
    ),
    marker_id="glucose",
    kind=StatisticKind.DELTA_MEANS,
    window=StatisticWindow(start_min=1500, end_min=1740),
    target=6.0,
    sigma=4.0,
)

COHORT_STATISTICS: list[CohortStatisticSpec] = [
    SLEEP_RESTRICTION_NEXT_DAY_GLUCOSE,
]

# Benchmark / reproducibility: same protocol as the adequate-sleep arm.
SLEEP_COHORT_48H_START_HOUR = _SLEEP_START_HOUR
SLEEP_COHORT_48H_MEALS: tuple[tuple[float, float, float, float], ...] = _MEALS_48H
