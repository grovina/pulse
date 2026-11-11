"""
Cohort statistics: the enterohepatic circulation.

Iter 95. Sourced anchors and the design rationale live in
``docs/iter95-biliary-anchors.md``; this file turns the ones that are scalar
statistics over a window into differentiable targets.

WHAT IS ENCODED HERE AND WHAT IS NOT. Distillation against the teacher already
supervises the whole trajectory shape of these four markers. These specs exist for the
quantities where the LITERATURE, not the teacher, is the authority — so that if the
teacher's biliary constants are wrong, something pulls back. That is the iter-93/94
lesson: the teacher is an approximate law, and cohort specs are the fence.

Deliberately NOT encoded:

* The gallbladder ejection fraction (>=35-38 % at 60 min). It is a RATIO of a state to
  its own earlier value, which no ``StatisticKind`` expresses; encoding it as an
  absolute post-meal level would silently also pin the fasting volume, which is a
  different (and more weakly sourced, 10-40 mL) quantity. It is checked instead by
  ``scripts/iter95_biliary_validate.py``.
* "CCK stays elevated 3-5 h." Qualitative in the sources — the model realizes +17 % over
  basal at 3 h, which is the right direction and is not evidence of anything.
* Liddle's CCK spike, both its minute and its height. A mixed liquid meal peaks near
  7 pmol/L within about 10 min and then falls. This stomach empties at Brener's
  2.13 kcal/min for as long as the meal is in it, so CCK rises in a few minutes and
  stays at whatever level that delivery supports. The soft-argmax of that plateau is
  the middle of the window (iter 114 read 63 min against 10). The highest value in
  the same window is the plateau itself, and scoring it at 6.8 would hold that level
  for the whole emptying. The level at 30 min is written as about 3.5, with no
  standard error. Fasting CCK remains. The spike waits on an early emptying.
* Anything about ALP/GGT/ALT/bilirubin. Those markers do not exist yet, and will not
  until there is a driver for them (see the anchors doc).

BILE-ACID TIME-TO-PEAK STILL NEEDS A SHARP SOFT-ARGMAX. ``StatisticKind.TIME_TO_PEAK``
is a softmax-weighted average of times. Beta has to be large next to the marker's range
or the average sits on the window centroid. On a trajectory that really peaks, beta = 10
lands within a few minutes of the bile-acid argmax; the default 0.05 does not.
"""

from __future__ import annotations

from ..cohort_types import (
    CohortArmSpec,
    CohortStatisticSpec,
    InitMode,
    StatisticKind,
    StatisticWindow,
    TargetShape,
    held,
)

# A single mixed meal at t=60, after an hour of fasting run-in so the gallbladder has
# partially refilled — matching the design of the studies these targets come from
# (overnight-fasted subjects given a mixed test meal).
_MIXED_MEAL = ((60.0, 70.0, 25.0, 30.0),)
_ARM = CohortArmSpec(
    label="mixed_meal",
    duration_min=360,
    start_hour=8.0,
    meals=_MIXED_MEAL,
)

# --- CCK ---------------------------------------------------------------------------
# Fasting means cluster 0.8-1.2 pmol/L, so the target is 1.0 and sigma is 0.2.
# The postprandial spike is not a target. See the module docstring.
CCK_FASTING = CohortStatisticSpec(
    name="cck_fasting_basal",
    source="Liddle et al. (JCI 1985); Rehfeld (J Intern Med 2025)",
    paper="liddle-1985",
    description="Fasting plasma CCK in healthy adults",
    arms=(_ARM,),
    marker_id="cck",
    kind=StatisticKind.MEAN_IN_WINDOW,
    window=StatisticWindow(start_min=0, end_min=55),
    target=1.0,
    sigma=0.2,
    # Liddle 1985 reports fasting CCK as mean +/- SEM over ~10 subjects; the
    # individual sd (~0.5) is 2-3x this -- pin n_arm = 1 (review 4.12).
    n_arm=1,
    init_mode=InitMode.NORM_CENTER,
)

# --- Serum bile acids ----------------------------------------------------------------
# Fasting reference interval 4.4-14.1 umol/L, postprandial 4.7-20.2, peak 75-120 min.
# Targets are interval midpoints; sigma covers the interval half-width, so this is a
# deliberately WEAK pull — the interval is wide because the quantity genuinely varies.
BILE_ACIDS_FASTING = CohortStatisticSpec(
    name="bile_acids_fasting",
    source="Reference interval 4.4-14.1 umol/L (ICP reference-range study, 2022)",
    paper="icp-reference-range-2022",
    description="Fasting serum total bile acids",
    arms=(_ARM,),
    marker_id="bile_acids",
    kind=StatisticKind.MEAN_IN_WINDOW,
    window=StatisticWindow(start_min=0, end_min=55),
    target=7.0,
    sigma=2.5,
    init_mode=InitMode.NORM_CENTER,
)

# Serum bile acids peak 75-120 min after the meal: gallbladder emptying, intestinal
# transit, ileal reabsorption and hepatic extraction in series. This is that late end.
# The early CCK clock is not the other pin. See the module docstring.
BILE_ACIDS_TIME_TO_PEAK = CohortStatisticSpec(
    name="bile_acids_time_to_peak",
    source="Fasting and postprandial serum bile acids in normal persons (Digestion 1978)",
    paper="digestion-1978",
    description="Serum total bile acids peak 75-120 min after a meal, long after CCK",
    arms=(_ARM,),
    marker_id="bile_acids",
    kind=StatisticKind.TIME_TO_PEAK,
    window=StatisticWindow(start_min=60, end_min=300),
    target=95.0,
    sigma=20.0,
    softargmax_beta=10.0,  # see the beta note in the module docstring; a real peak, not a plateau
    init_mode=InitMode.NORM_CENTER,
)

BILE_ACIDS_POSTPRANDIAL_PEAK = CohortStatisticSpec(
    name="bile_acids_postprandial_peak",
    source="Postprandial reference interval 4.7-20.2 umol/L (ICP reference-range study, 2022)",
    paper="icp-reference-range-2022",
    description="Peak serum total bile acids after a mixed meal",
    arms=(_ARM,),
    marker_id="bile_acids",
    kind=StatisticKind.PEAK_VALUE,
    window=StatisticWindow(start_min=60, end_min=300),
    target=12.0,
    sigma=4.0,
    init_mode=InitMode.NORM_CENTER,
)

# Borgström 1985: with no bile in the intestine, 60–70% of a normal fat load
# is still absorbed. That is a range with no standard error, so the loss is
# zero inside it and the sigma only scales how far outside counts. The fistula
# drains the intestine; the meal is fat alone so the share is the whole finding.
_FISTULA_MIN = 60
_FISTULA_NAME = "borgstrom_1985_fat_without_bile"
BORGSTROM_FAT_WITHOUT_BILE = CohortStatisticSpec(
    name=_FISTULA_NAME,
    source="Borgström, Barrowman and Lindström (1985) New Comprehensive Biochemistry, ch. 14",
    paper="borgstrom-1985",
    description="With no bile in the intestine, 60–70% of a fat load is still absorbed",
    arms=(
        CohortArmSpec(
            label="bile_fistula_fat_meal",
            duration_min=_FISTULA_MIN,
            start_hour=8.0,
            meals=((0.0, 0.0, 30.0, 0.0),),
            bile_fistula=held(1.0, _FISTULA_MIN),
        ),
    ),
    marker_id="fat_absorption",
    kind=StatisticKind.MEAN_IN_WINDOW,
    window=StatisticWindow(10, 40),
    target=0.65,
    sigma=0.05,
    shape=TargetShape.BAND,
    band_halfwidth=0.05,
    n_arm=1,
    init_mode=InitMode.NORM_CENTER,
)


COHORT_STATISTICS: list[CohortStatisticSpec] = [
    CCK_FASTING,
    BILE_ACIDS_FASTING,
    BILE_ACIDS_TIME_TO_PEAK,
    BILE_ACIDS_POSTPRANDIAL_PEAK,
    BORGSTROM_FAT_WITHOUT_BILE,
]
