"""
Cohort statistics: sleep and circadian manipulation.
"""

from __future__ import annotations

import numpy as np

from ...circadian import CLAMP_INSULIN_FOLD
from ...types import MARKER_INDEX, NORM_CENTER
from ..cohort_types import (
    CohortArmSpec,
    CohortStatisticSpec,
    StatisticKind,
    StatisticWindow,
    held,
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
    paper="donga-2010",
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

# The clamp row of the same table: the slice that identifies the three
# sleep-debt gains of ``pulse.circadian``. After the basal hour the clamp held
# insulin at 22.0 mU/L against a basal of 3.1, glucose at the basal 4.8 mmol/L,
# and reported, short night versus normal:
#
#     glucose disposal      32.5 ± 3.6  vs  40.7 ± 5.1 µmol/kg LBM/min
#     endogenous production  4.4 ± 0.3  vs   3.6 ± 0.2
#     NEFA                    68 ± 5    vs    57 ± 4   µmol/L
#
# Each is scored as short over normal. Disposal is per kilogram of lean mass and
# the assay's basal insulin is a third of this model's, so the ratio of the two
# nights is the part of the row that means the same thing here; the gains are
# that ratio. Its SE is the ratio times the two relative SEs in quadrature. The
# insulin plateau is the table's fold over basal, on this model's basal.
_DONGA_CLAMP_ONSET_MIN = 1740                       # 12:00, after the basal hour
_DONGA_CLAMP_MIN = 180
_DONGA_CLAMP_DURATION_MIN = _DONGA_CLAMP_ONSET_MIN + _DONGA_CLAMP_MIN
_DONGA_CLAMP_INSULIN_UU_ML = CLAMP_INSULIN_FOLD * NORM_CENTER[MARKER_INDEX["insulin"]]
_DONGA_CLAMP_GLUCOSE_MG_DL = 4.8 * 18.0
_DONGA_CLAMP_WINDOW = StatisticWindow(
    start_min=_DONGA_CLAMP_DURATION_MIN - 60, end_min=_DONGA_CLAMP_DURATION_MIN,
)


def _clamp_arm(label: str, bed_h: float, wake_h: float) -> CohortArmSpec:
    """The night, the next morning's basal hour, then the clamp."""
    n = _DONGA_CLAMP_DURATION_MIN
    return CohortArmSpec(
        label=label,
        duration_min=n,
        start_hour=_DONGA_START_HOUR,
        meals=_DONGA_DINNER,
        sleep_wake=_allowed_sleep(n, _DONGA_START_HOUR, bed_h, wake_h),
        insulin_clamp=held(_DONGA_CLAMP_INSULIN_UU_ML, n, _DONGA_CLAMP_ONSET_MIN),
        glucose_clamp=held(_DONGA_CLAMP_GLUCOSE_MG_DL, n, _DONGA_CLAMP_ONSET_MIN),
    )


_DONGA_CLAMP_ARMS = (
    _clamp_arm("normal_sleep_clamp", 23.0, 31.5),
    _clamp_arm("restricted_sleep_clamp", 25.0, 29.0),
)


def _donga_clamp_ratio(
    name: str, marker: str, what: str, short: tuple[float, float],
    normal: tuple[float, float], unit: str,
) -> CohortStatisticSpec:
    """Short night over normal; the arms are (normal, restricted)."""
    ratio = short[0] / normal[0]
    sigma = ratio * float(np.hypot(short[1] / short[0], normal[1] / normal[0]))
    return CohortStatisticSpec(
        name=name,
        source="Donga et al. (2010) J Clin Endocrinol Metab 95:2963, Table 1",
        paper="donga-2010",
        description=(
            f"One night of 4 h versus 8.5 h sleep, same midpoint: at the next "
            f"afternoon's hyperinsulinaemic clamp, {what} is "
            f"{short[0]:g} ± {short[1]:g} versus {normal[0]:g} ± {normal[1]:g} {unit}"
        ),
        arms=_DONGA_CLAMP_ARMS,
        marker_id=marker,
        kind=StatisticKind.RATIO_MEANS,
        window=_DONGA_CLAMP_WINDOW,
        target=ratio,
        sigma=sigma,
        n_arm=1,
    )


DONGA_2010_CLAMP_DISPOSAL = _donga_clamp_ratio(
    "donga_2010_clamp_glucose_disposal", "glucose_disposal", "glucose disposal",
    (32.5, 3.6), (40.7, 5.1), "µmol/kg LBM/min",
)
DONGA_2010_CLAMP_PRODUCTION = _donga_clamp_ratio(
    "donga_2010_clamp_endogenous_production", "hepatic_output",
    "endogenous glucose production", (4.4, 0.3), (3.6, 0.2), "µmol/kg LBM/min",
)
DONGA_2010_CLAMP_NEFA = _donga_clamp_ratio(
    "donga_2010_clamp_nefa", "ffa", "NEFA", (68.0, 5.0), (57.0, 4.0), "µmol/L",
)


COHORT_STATISTICS: list[CohortStatisticSpec] = [
    DONGA_2010_BASAL_GLUCOSE,
    DONGA_2010_CLAMP_DISPOSAL,
    DONGA_2010_CLAMP_PRODUCTION,
    DONGA_2010_CLAMP_NEFA,
]

# Benchmark / reproducibility: same protocol as the adequate-sleep arm.
SLEEP_COHORT_48H_START_HOUR = _SLEEP_START_HOUR
SLEEP_COHORT_48H_MEALS: tuple[tuple[float, float, float, float], ...] = _MEALS_48H
