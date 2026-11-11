"""
Cohort statistics: whole published trajectories (``kind=CURVE``).

A curve is one finding that replaces the teacher's dense minute-by-minute tape:
the marker's mean +/- SEM at a handful of timepoints. ``expand_curve_spec``
compiles each into one point statistic per timepoint, with sigma inflated by
sqrt(n_points) because neighbouring points of a smooth curve are strongly
correlated.

All curve values below are DIGITIZED ESTIMATES read off the published figures
and standard reference shapes, not transcribed table entries; the SEMs are
rounded envelopes of the reported scatter.
"""

from __future__ import annotations

import numpy as np

from ...expenditure import (
    RQ_800, RQ_REST, RQ_SE, VO2_800_ML, VO2_800_SE_ML, VO2_REST_ML, VO2_REST_SE_ML,
)
from ..cohort_types import (
    CohortArmSpec,
    CohortStatisticSpec,
    CurvePoint,
    InitMode,
    StatisticKind,
    StatisticWindow,
)
from .glucose_handling import WAHREN_EXERCISE_HGO


# ---------------------------------------------------------------------------
# 24 h plasma cortisol circadian rhythm
# ---------------------------------------------------------------------------
# Weitzman et al. (1971) J Clin Endocrinol Metab 33:14 and the standard
# cosinor description of the human cortisol rhythm: nadir ~3-4 ug/dL around
# midnight-02:00, a pre-waking rise from ~03:00, acrophase ~16 ug/dL near
# 08:00, then a steady decline through the day. Digitized estimates, ug/dL.
# The arm starts at midnight with a normal night (asleep 23:00-07:00) and three
# ordinary meals, from the student's own day so the trough is the clock's.
_CORTISOL_DURATION_MIN = 1440
_CORTISOL_MEALS = (
    (480.0, 60.0, 15.0, 20.0),   # 08:00 breakfast
    (750.0, 75.0, 25.0, 25.0),   # 12:30 lunch
    (1110.0, 80.0, 30.0, 30.0),  # 18:30 dinner
)


def _night_sleep(duration_min: int) -> tuple[float, ...]:
    """Asleep 00:00-07:00 and 23:00-24:00; smoothed over 20 min like sleep.py."""
    sw = np.ones(duration_min, dtype=np.float32)
    sw[: 7 * 60] = 0.0
    sw[23 * 60 :] = 0.0
    kernel = np.ones(20, dtype=np.float32) / 20.0
    sw = np.clip(np.convolve(sw, kernel, mode="same"), 0.0, 1.0)
    return tuple(float(x) for x in sw)


_CORTISOL_CURVE = (
    CurvePoint(0, 3.8, 1.0),      # 00:00
    CurvePoint(120, 3.4, 1.0),    # 02:00 nadir
    CurvePoint(240, 5.5, 1.3),    # 04:00
    CurvePoint(360, 11.5, 1.8),   # 06:00 pre-waking rise
    CurvePoint(480, 16.0, 2.0),   # 08:00 acrophase
    CurvePoint(600, 12.5, 1.6),   # 10:00
    CurvePoint(720, 9.5, 1.4),    # 12:00
    CurvePoint(840, 8.0, 1.2),    # 14:00
    CurvePoint(960, 7.0, 1.2),    # 16:00
    CurvePoint(1080, 5.8, 1.1),   # 18:00
    CurvePoint(1200, 4.8, 1.0),   # 20:00
    CurvePoint(1320, 4.2, 1.0),   # 22:00
)

CORTISOL_CIRCADIAN_CURVE = CohortStatisticSpec(
    name="cortisol_24h_circadian_curve",
    source="Weitzman et al. (1971) J Clin Endocrinol Metab 33:14; standard cosinor cortisol rhythm",
    paper="weitzman-1971",
    description=(
        "24 h plasma cortisol in healthy adults: trough ~3-4 ug/dL near midnight-02:00, "
        "acrophase ~16 ug/dL near 08:00, declining through the day"
    ),
    arms=(
        CohortArmSpec(
            label="cortisol_24h",
            duration_min=_CORTISOL_DURATION_MIN, start_hour=0.0,
            meals=_CORTISOL_MEALS, sleep_wake=_night_sleep(_CORTISOL_DURATION_MIN),
        ),
    ),
    marker_id="cortisol",
    kind=StatisticKind.CURVE,
    window=StatisticWindow(0, _CORTISOL_DURATION_MIN),
    target=0.0,
    sigma=1.0,
    init_mode=InitMode.COLD,
    curve=_CORTISOL_CURVE,
)


# ---------------------------------------------------------------------------
# 75 g OGTT plasma glucose curve
# ---------------------------------------------------------------------------
# DeFronzo et al. (1979) Am J Physiol 237:E214 and the standard 75 g OGTT
# shape in healthy adults: fasting ~90 mg/dL, peak ~155 mg/dL at 30-60 min,
# ~120 mg/dL by 2 h, back toward ~95 mg/dL by 4 h. Digitized estimates, mg/dL.
# It is held out of training as an item (``sealing.HELD_OUT_ITEMS``), not
# because its paper was drawn: DeFronzo 1979 also sources training items.
_OGTT_CURVE = (
    CurvePoint(0, 90.0, 4.0),
    CurvePoint(30, 150.0, 6.0),
    CurvePoint(60, 158.0, 7.0),
    CurvePoint(90, 140.0, 7.0),
    CurvePoint(120, 120.0, 6.0),
    CurvePoint(180, 100.0, 5.0),
    CurvePoint(239, 95.0, 5.0),
)

OGTT_GLUCOSE_CURVE = CohortStatisticSpec(
    name="ogtt_75g_glucose_curve",
    source="DeFronzo et al. (1979) Am J Physiol 237:E214; standard 75 g OGTT",
    paper="defronzo-1979",
    description=(
        "75 g OGTT plasma glucose in healthy adults: fasting ~90, peak ~155 mg/dL at "
        "30-60 min, ~120 by 2 h, back toward ~95 by 4 h"
    ),
    arms=(
        CohortArmSpec(
            label="ogtt_75g_curve",
            duration_min=240, start_hour=8.0, meals=((0.0, 75.0, 0.0, 0.0),),
        ),
    ),
    marker_id="glucose",
    kind=StatisticKind.CURVE,
    window=StatisticWindow(0, 240),
    target=0.0,
    sigma=1.0,
    init_mode=InitMode.COLD,
    curve=_OGTT_CURVE,
)


# ---------------------------------------------------------------------------
# Pulmonary gas exchange, rest and 40 min at 800 kg-m/min
# ---------------------------------------------------------------------------
# Wahren, Felig, Ahlborg and Jorfeldt, J Clin Invest 1971;50:2715, Table I.
# Healthy postabsorptive men, upright bicycle, mean ± SE: O₂ uptake 256 ± 19
# ml/min at rest and 1811 ± 69 after 40 min; respiratory quotient 0.77 ± 0.01
# and 0.82 ± 0.01. Transcribed table entries, not digitized. The arm is the
# splanchnic spec's bout: 20 min seated, then activity 0.7, so 40 min of work
# is minute 60. These identify the four Table I numbers of
# ``pulse.expenditure.GAS``, read through the ``vo2`` and ``rq`` observables.
_WAHREN_ARM = WAHREN_EXERCISE_HGO.arms[1]
_WAHREN_REST_MIN = 10
_WAHREN_40_MIN = 60


def _wahren_curve(name: str, marker: str, what: str, rest: tuple[float, float],
                  work: tuple[float, float]) -> CohortStatisticSpec:
    return CohortStatisticSpec(
        name=name,
        source="Wahren et al. (1971) J Clin Invest 50:2715, Table I",
        paper="wahren-1971",
        description=f"{what} at rest and after 40 min at 800 kg-m/min on an upright bicycle",
        arms=(_WAHREN_ARM,),
        marker_id=marker,
        kind=StatisticKind.CURVE,
        window=StatisticWindow(0, _WAHREN_ARM.duration_min),
        target=0.0,
        sigma=1.0,
        curve=(
            CurvePoint(_WAHREN_REST_MIN, *rest),
            CurvePoint(_WAHREN_40_MIN, *work),
        ),
    )


WAHREN_VO2_CURVE = _wahren_curve(
    "wahren_1971_vo2_curve", "vo2", "Pulmonary O₂ uptake",
    (VO2_REST_ML, VO2_REST_SE_ML), (VO2_800_ML, VO2_800_SE_ML),
)
WAHREN_RQ_CURVE = _wahren_curve(
    "wahren_1971_rq_curve", "rq", "Respiratory quotient",
    (RQ_REST, RQ_SE), (RQ_800, RQ_SE),
)


COHORT_STATISTICS: list[CohortStatisticSpec] = [
    CORTISOL_CIRCADIAN_CURVE,
    OGTT_GLUCOSE_CURVE,
    WAHREN_VO2_CURVE,
    WAHREN_RQ_CURVE,
]
