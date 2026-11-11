"""Cryer 1987, the measurements the stepped clamp was built on.

Schwartz, Clutter, Shah and Cryer, J Clin Invest 1987;79:777. Ten normal
young adults, ages 23–34, after an overnight fast, before the insulin
infusion. Postabsorptive arterialized glucose was 91 ± 7 mg/dL, and that
± is the standard deviation (the paper's other ± are standard errors).
Baseline plasma epinephrine, the level the 90-to-80 step then doubled
from, was 34 ± 7 pg/mL.

The calculated epinephrine threshold of 69 ± 2 mg/dL is a different
claim. It is the glucose at which each subject first exceeded a
confidence band, and the authors say the biologic threshold looks higher.
The concentration reached while glucose was still near 80 sits above this
model's initialized threshold, where the hypoglycaemic drive is identically
zero and carries no gradient. That rise is not filed. Growth hormone,
which the same paper ranks in the hierarchy, has no state.
"""

from __future__ import annotations

from ..cohort_types import (
    CohortArmSpec,
    CohortStatisticSpec,
    Population,
    StatisticKind,
    StatisticWindow,
)

_N = 10
_GLUCOSE_SD = 7.0
# The stated SD and n. The mean's standard error is the usual conversion.
# The spread's standard error is the sampling error of a normal sample SD,
# sd / sqrt(2(n − 1)), which is what a published SD is uncertain by.
_GLUCOSE_SEM = _GLUCOSE_SD / _N ** 0.5
_GLUCOSE_SD_SE = _GLUCOSE_SD / (2.0 * (_N - 1)) ** 0.5

_POPULATION = Population(name="cryer_1987", age=(23.0, 34.0), healthy=True)
_ARM = CohortArmSpec(
    label="cryer_postabsorptive",
    duration_min=30,
    start_hour=8.0,
    meals=(),
    population=_POPULATION,
)
_WINDOW = StatisticWindow(0, 30)

_GLUCOSE_NAME = "cryer_1987_fasting_glucose"
CRYER_1987_FASTING_GLUCOSE = CohortStatisticSpec(
    name=_GLUCOSE_NAME,
    source="Schwartz, Clutter, Shah and Cryer (1987) J Clin Invest 79:777",
    paper="schwartz-1987",
    description="Postabsorptive arterialized glucose before the insulin infusion, 91 ± 7 mg/dL (SD, n = 10)",
    arms=(_ARM,),
    marker_id="glucose",
    kind=StatisticKind.MEAN_IN_WINDOW,
    window=_WINDOW,
    target=91.0,
    sigma=_GLUCOSE_SEM,
    n_arm=1,
)

_SPREAD_NAME = "cryer_1987_fasting_glucose_spread"
CRYER_1987_FASTING_GLUCOSE_SPREAD = CohortStatisticSpec(
    name=_SPREAD_NAME,
    source="Schwartz, Clutter, Shah and Cryer (1987) J Clin Invest 79:777",
    paper="schwartz-1987",
    description="Between-person SD of that postabsorptive glucose, 7 mg/dL in the same ten subjects",
    arms=(_ARM,),
    marker_id="glucose",
    kind=StatisticKind.SPREAD,
    window=_WINDOW,
    target=_GLUCOSE_SD,
    sigma=_GLUCOSE_SD_SE,
)

_EPI_NAME = "cryer_1987_epinephrine_baseline"
CRYER_1987_EPINEPHRINE_BASELINE = CohortStatisticSpec(
    name=_EPI_NAME,
    source="Schwartz, Clutter, Shah and Cryer (1987) J Clin Invest 79:777",
    paper="schwartz-1987",
    description="Baseline plasma epinephrine before the glucose step, 34 ± 7 pg/mL",
    arms=(_ARM,),
    marker_id="epinephrine",
    kind=StatisticKind.MEAN_IN_WINDOW,
    window=_WINDOW,
    target=34.0,
    sigma=7.0,
    n_arm=1,
)

COHORT_STATISTICS: list[CohortStatisticSpec] = [
    CRYER_1987_FASTING_GLUCOSE,
    CRYER_1987_FASTING_GLUCOSE_SPREAD,
    CRYER_1987_EPINEPHRINE_BASELINE,
]
