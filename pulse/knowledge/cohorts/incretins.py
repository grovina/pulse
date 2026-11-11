"""Vollmer's GIP comparison, as the ratio of two baseline-subtracted areas.

Vollmer et al., Diabetes 2008;57:678. Over 240 min, integrated incremental
GIP after an 820 kcal breakfast was 186 ± 17% higher than after 75 g of
glucose. A ratio of window means keeps the fasting level in both arms, so
it is not this comparison. Each arm subtracts its own run-in, and the
statistic is the ratio of those areas.
"""

from __future__ import annotations

from ..cohort_types import (
    CohortArmSpec,
    CohortStatisticSpec,
    InitMode,
    StatisticKind,
    StatisticWindow,
)

# Breakfast: 360 kcal carbohydrate, 353 fat, 107 protein. The glucose arm is
# 75 g. Both start an hour into the arm so the run-in is the baseline the
# increment subtracts, and the area is the 240 min the paper integrated.
_MIN = 300
_MEAL_MIN = 60.0
_NAME = "vollmer_2008_gip_increment_ratio"
_BREAKFAST = (
    _MEAL_MIN,
    360.0 / 4.0,
    353.0 / 9.0,
    107.0 / 4.0,
)

VOLLMER_2008_GIP = CohortStatisticSpec(
    name=_NAME,
    source="Vollmer et al. (2008) Diabetes 57:678",
    paper="vollmer-2008",
    description=(
        "Integrated incremental GIP over 240 min is 186 ± 17% higher after an "
        "820 kcal breakfast than after 75 g of glucose"
    ),
    arms=(
        CohortArmSpec(
            label="glucose_75g",
            duration_min=_MIN,
            start_hour=8.0,
            meals=((_MEAL_MIN, 75.0, 0.0, 0.0),),
        ),
        CohortArmSpec(
            label="breakfast_820kcal",
            duration_min=_MIN,
            start_hour=8.0,
            meals=(_BREAKFAST,),
        ),
    ),
    marker_id="gip",
    kind=StatisticKind.RATIO_INCREMENTS,
    window=StatisticWindow(start_min=60, end_min=_MIN),
    baseline=StatisticWindow(start_min=0, end_min=60),
    target=1.0 + 1.86,
    sigma=0.17,
    n_arm=1,
    init_mode=InitMode.NORM_CENTER,
)

COHORT_STATISTICS: list[CohortStatisticSpec] = [VOLLMER_2008_GIP]
