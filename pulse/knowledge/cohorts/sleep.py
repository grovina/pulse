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

def _allowed_sleep(
    duration_min: int, start_hour: float, bed_h: float, wake_h: float,
) -> tuple[float, ...]:
    """Binary sleep on [bed, wake), hours from the start day's midnight."""
    sw = np.ones(duration_min, dtype=np.float32)
    lo = max(0, int(round((bed_h - start_hour) * 60)))
    hi = min(duration_min, int(round((wake_h - start_hour) * 60)))
    if lo < hi:
        sw[lo:hi] = 0.0
    return tuple(float(x) for x in sw)


# Donga et al. JCEM 2010;95:2963, Table 1. One night, n = 9. Sleep allowed
# 23:00–07:30 or 01:00–05:00, both centered at 03:00. Fasted from 22:00.
# Basal glucose the next late morning is 4.8 ± 0.18 mmol/L on both occasions
# (P = 0.32). The table does not publish the paired difference, so sigma is
# the unpaired SE of two arms that each report SEM 0.18 mmol/L.
_DONGA_START_HOUR = 7.0
_DONGA_DURATION_MIN = 1860
_DONGA_DINNER = ((600.0, 40.0, 10.0, 15.0),)
_DONGA_BASAL_SEM_MGDL = 0.18 * 18.0 * float(np.sqrt(2.0))
_normal_night = _allowed_sleep(_DONGA_DURATION_MIN, _DONGA_START_HOUR, 23.0, 31.5)
_restricted_night = _allowed_sleep(_DONGA_DURATION_MIN, _DONGA_START_HOUR, 25.0, 29.0)

DONGA_2010_BASAL_GLUCOSE = CohortStatisticSpec(
    name="donga_2010_basal_glucose",
    source="Donga et al. (2010) J Clin Endocrinol Metab 95:2963, Table 1",
    description=(
        "One night of 4 h versus 8.5 h sleep, same midpoint: basal glucose "
        "the next morning is unchanged"
    ),
    arms=(
        CohortArmSpec(
            label="normal_sleep",
            duration_min=_DONGA_DURATION_MIN, start_hour=_DONGA_START_HOUR,
            meals=_DONGA_DINNER, sleep_wake=_normal_night,
        ),
        CohortArmSpec(
            label="restricted_sleep",
            duration_min=_DONGA_DURATION_MIN, start_hour=_DONGA_START_HOUR,
            meals=_DONGA_DINNER, sleep_wake=_restricted_night,
        ),
    ),
    marker_id="glucose",
    kind=StatisticKind.DELTA_MEANS,
    window=StatisticWindow(start_min=1680, end_min=1740),
    target=0.0,
    sigma=_DONGA_BASAL_SEM_MGDL,
    n_arm=1,
)

COHORT_STATISTICS: list[CohortStatisticSpec] = [
    DONGA_2010_BASAL_GLUCOSE,
]

# Benchmark / reproducibility: same protocol as the adequate-sleep arm.
SLEEP_COHORT_48H_START_HOUR = _SLEEP_START_HOUR
SLEEP_COHORT_48H_MEALS: tuple[tuple[float, float, float, float], ...] = _MEALS_48H
