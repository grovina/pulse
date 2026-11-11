"""
Cohort statistics: the glucose the kidney does not take back.

Diabetes 2017;66:1999–2006 (doi 10.2337/db17-0100), people without diabetes,
n = 15. Urinary glucose was regressed on plasma glucose. The line crosses zero
at 190 ± 2 mg/dL and its slope above that is the creatinine clearance,
139 ± 7 mL/min, so the excretion is ``139 · (G − 190)/100`` mg/min. On
empagliflozin 25 mg/day the day-14 line crosses at 34 mg/dL with the day-13
clearance of 126 ± 10 mL/min. The same paper collects 42 ± 5 g over day 13 in
people eating their own food.

These are the slices that identify the four numbers of ``pulse.renal``. They
read a plasma glucose the model does not reach on its own, so each arm is a
hyperglycaemic clamp that holds one point of the regression while the urine is
collected: the ``urinary_glucose`` observable is the law's own output in that
minute, in the paper's mg/min.

The 24 h collection is filed beside the regression on purpose, not instead of
it. A fed day of this model loses far more than 42 g, because the regression
stays steep at the glucose the model actually holds, and the two cannot both
be met: that is a conflict (PLAN section 3), and it is the signal to change
the law rather than to widen a sigma.
"""

from __future__ import annotations

from ...renal import (
    EMPA_GFR_ML_PER_MIN, EMPA_GFR_SE_ML_PER_MIN, EMPA_RTG_MG_DL, EMPA_RTG_SE_MG_DL,
    EMPA_UGE_G_PER_DAY, EMPA_UGE_SE_G_PER_DAY, GFR_ML_PER_MIN, GFR_SE_ML_PER_MIN,
    RTG_MG_DL, RTG_SE_MG_DL,
)
from ..cohort_types import (
    CohortArmSpec,
    CohortStatisticSpec,
    InitMode,
    StatisticKind,
    StatisticWindow,
    held,
)
from .nutrition import nocturnal_sleep_wake

_PAPER = "diabetes-2017-66-1999"
_SOURCE = "Diabetes 2017;66:1999–2006 — urinary glucose regressed on plasma glucose"
_DURATION_MIN = 180
# The clamp needs time to reach its target; the collection is the plateau.
_WINDOW = StatisticWindow(start_min=90, end_min=_DURATION_MIN)


def _excretion(glucose_mg_dl: float, rtg: float, gfr: float) -> float:
    """The regression's mg/min at this plasma glucose."""
    return gfr * max(glucose_mg_dl - rtg, 0.0) / 100.0


def _excretion_se(glucose_mg_dl: float, rtg: float, gfr: float,
                  rtg_se: float, gfr_se: float) -> float:
    """The line's standard error there, from the two the paper reports."""
    from_slope = gfr_se * max(glucose_mg_dl - rtg, 0.0) / 100.0
    from_intercept = gfr * rtg_se / 100.0
    return float((from_slope ** 2 + from_intercept ** 2) ** 0.5)


def _regression_point(glucose_mg_dl: float, *, on_drug: bool = False) -> CohortStatisticSpec:
    """One point of the published line, held by a clamp."""
    if on_drug:
        rtg, gfr = EMPA_RTG_MG_DL, EMPA_GFR_ML_PER_MIN
        rtg_se, gfr_se = EMPA_RTG_SE_MG_DL, EMPA_GFR_SE_ML_PER_MIN
    else:
        rtg, gfr = RTG_MG_DL, GFR_ML_PER_MIN
        rtg_se, gfr_se = RTG_SE_MG_DL, GFR_SE_ML_PER_MIN
    drug = " on empagliflozin 25 mg/day" if on_drug else ""
    level = int(glucose_mg_dl)
    name = f"renal_glucose_at_{level}" + ("_empagliflozin" if on_drug else "")
    return CohortStatisticSpec(
        name=name,
        source=f"{_SOURCE}{drug}",
        paper=_PAPER,
        description=(
            f"At a plasma glucose of {level} mg/dL{drug}, urinary glucose is "
            f"{_excretion(glucose_mg_dl, rtg, gfr):.1f} mg/min on the published line"
        ),
        arms=(
            CohortArmSpec(
                label=f"{'empa' if on_drug else 'clamp'}_{level}",
                duration_min=_DURATION_MIN, start_hour=8.0, meals=(),
                glucose_clamp=held(glucose_mg_dl, _DURATION_MIN),
                sglt2=held(1.0, _DURATION_MIN) if on_drug else None,
            ),
        ),
        marker_id="urinary_glucose",
        kind=StatisticKind.MEAN_IN_WINDOW,
        window=_WINDOW,
        target=_excretion(glucose_mg_dl, rtg, gfr),
        sigma=_excretion_se(glucose_mg_dl, rtg, gfr, rtg_se, gfr_se),
        init_mode=InitMode.NORM_CENTER,
        n_arm=1,
    )


# The threshold is identified from above: a kidney that leaks below 190 mg/dL
# excretes something at 190, and nothing pushes the other way, which is the
# shape of the measurement. Two points above it fix the slope.
RENAL_AT_THRESHOLD = _regression_point(RTG_MG_DL)
RENAL_AT_250 = _regression_point(250.0)
RENAL_AT_300 = _regression_point(300.0)
# The drug moves both the threshold and the slope, so its line needs a point
# near fasting, where the untreated kidney excretes nothing, and one above.
RENAL_EMPAGLIFLOZIN_AT_84 = _regression_point(84.0, on_drug=True)
RENAL_EMPAGLIFLOZIN_AT_200 = _regression_point(200.0, on_drug=True)

# Day 13 of the same study, free-living. 42 g over 1440 min is a mean of
# 29.2 mg/min, and the 5 g SE carries over the same way. The night is asleep:
# an arm with no sleep series is awake for the whole day.
_UGE_NAME = "empagliflozin_24h_urinary_glucose"
_UGE_MEALS = (
    (120.0, 50.0, 12.0, 20.0),
    (420.0, 65.0, 22.0, 28.0),
    (780.0, 75.0, 28.0, 35.0),
)
_MG_PER_MIN_PER_G_PER_DAY = 1000.0 / 1440.0
EMPAGLIFLOZIN_24H_URINARY_GLUCOSE = CohortStatisticSpec(
    name=_UGE_NAME,
    source=f"{_SOURCE}, day-13 24 h collection",
    paper=_PAPER,
    description=(
        "Empagliflozin 25 mg/day in people without diabetes: a day's urinary glucose "
        "is 42 ± 5 g, eating their own food"
    ),
    arms=(
        CohortArmSpec(
            label="empa_free_living_day",
            duration_min=1440, start_hour=6.0, meals=_UGE_MEALS,
            sleep_wake=nocturnal_sleep_wake(1440, 6.0),
            activity=(0.0,) * 1440,
            sglt2=held(1.0, 1440),
        ),
    ),
    marker_id="urinary_glucose",
    kind=StatisticKind.MEAN_IN_WINDOW,
    window=StatisticWindow(start_min=0, end_min=1440),
    target=EMPA_UGE_G_PER_DAY * _MG_PER_MIN_PER_G_PER_DAY,
    sigma=EMPA_UGE_SE_G_PER_DAY * _MG_PER_MIN_PER_G_PER_DAY,
    n_arm=1,
)


COHORT_STATISTICS: list[CohortStatisticSpec] = [
    RENAL_AT_THRESHOLD,
    RENAL_AT_250,
    RENAL_AT_300,
    RENAL_EMPAGLIFLOZIN_AT_84,
    RENAL_EMPAGLIFLOZIN_AT_200,
    EMPAGLIFLOZIN_24H_URINARY_GLUCOSE,
]
