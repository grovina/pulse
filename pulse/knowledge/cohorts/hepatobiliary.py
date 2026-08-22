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
* Anything about ALP/GGT/ALT/bilirubin. Those markers do not exist yet, and will not
  until there is a driver for them (see the anchors doc).

SOFT-ARGMAX BETA IS NOT OPTIONAL HERE. ``StatisticKind.TIME_TO_PEAK`` estimates the peak
minute as a softmax-weighted average of times, ``softmax(beta * value)``. The default
beta = 0.05 in ``cohort_types`` was never exercised — these are the first cohort specs to
use TIME_TO_PEAK — and it is far too soft for markers with a small absolute range.
Measured on the iter-95 validation checkpoint:

    marker        true argmax   range    beta=0.05   0.5    1.0    2.0    5.0   10.0
    cck                    12    5.54         57.2  32.4   17.2   13.2   12.4   12.2
    bile_acids             80    2.13        119.2 115.1  109.3   98.8   86.4   82.7

At the default, the CCK spec reported a peak at 57 min for a trajectory that peaks at 12,
and contributed z = 9.4 — 88.9 of the whole registry's 181 total z², i.e. HALF the cohort
objective, spent driving the model against a broken measurement. The bile-acid spec
"passed" while measuring something 39 minutes from its own peak.

Rule of thumb: beta * (marker's postprandial range) should be >= ~20 for the softmax to
concentrate on the peak rather than drift toward the window centroid. Hence beta = 5 for
cck (range ~5.5) and 10 for bile_acids (range ~2.1), both verified above to land within
~3 min of the true argmax.

NOTE for whoever touches ``knowledge/physiology_rules.py``: ``hinge_argmax_in_band`` and
``hinge_a_precedes_b`` (the ACTH->cortisol lead rule) carry the same 0.05 default over
markers whose ranges are ~30 and ~10, so beta*range is 1.5 and 0.5. They are likely
measuring closer to the window centroid than to the peak. NOT changed here — iterations
have been tuned around that behaviour and the impact is unmeasured — but it should be.
"""

from __future__ import annotations

from ..cohort_types import (
    CohortArmSpec,
    CohortStatisticSpec,
    InitMode,
    StatisticKind,
    StatisticWindow,
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
# Fasting 0.8-1.2 pmol/L; peak 6.5-7.1 within ~10 min of a mixed liquid meal.
# sigma is set from the spread of the reported means across studies, not invented:
# fasting means cluster 0.8-1.2 (sigma 0.2), peaks 6.5-7.1 (sigma 0.4).
CCK_FASTING = CohortStatisticSpec(
    name="cck_fasting_basal",
    source="Liddle et al. (JCI 1985); Rehfeld (J Intern Med 2025)",
    description="Fasting plasma CCK in healthy adults",
    arms=(_ARM,),
    marker_id="cck",
    kind=StatisticKind.MEAN_IN_WINDOW,
    window=StatisticWindow(start_min=0, end_min=55),
    target=1.0,
    sigma=0.2,
    init_mode=InitMode.NORM_CENTER,
)

CCK_POSTPRANDIAL_PEAK = CohortStatisticSpec(
    name="cck_postprandial_peak",
    source="Liddle et al. (JCI 1985); Rehfeld (J Intern Med 2025)",
    description="Peak plasma CCK after a mixed meal",
    arms=(_ARM,),
    marker_id="cck",
    kind=StatisticKind.PEAK_VALUE,
    window=StatisticWindow(start_min=60, end_min=180),
    target=6.8,
    sigma=0.4,
    init_mode=InitMode.NORM_CENTER,
)

CCK_TIME_TO_PEAK = CohortStatisticSpec(
    name="cck_time_to_peak",
    source="Liddle et al. (JCI 1985)",
    description="CCK peaks within ~10 min of a mixed meal — the FAST arm of the axis",
    arms=(_ARM,),
    marker_id="cck",
    kind=StatisticKind.TIME_TO_PEAK,
    # Window opens at the meal so the soft-argmax is measured from meal onset.
    window=StatisticWindow(start_min=60, end_min=180),
    target=10.0,
    sigma=5.0,
    softargmax_beta=5.0,   # see the beta note in the module docstring
    init_mode=InitMode.NORM_CENTER,
)

# --- Serum bile acids ----------------------------------------------------------------
# Fasting reference interval 4.4-14.1 umol/L, postprandial 4.7-20.2, peak 75-120 min.
# Targets are interval midpoints; sigma covers the interval half-width, so this is a
# deliberately WEAK pull — the interval is wide because the quantity genuinely varies.
BILE_ACIDS_FASTING = CohortStatisticSpec(
    name="bile_acids_fasting",
    source="Reference interval 4.4-14.1 umol/L (ICP reference-range study, 2022)",
    description="Fasting serum total bile acids",
    arms=(_ARM,),
    marker_id="bile_acids",
    kind=StatisticKind.MEAN_IN_WINDOW,
    window=StatisticWindow(start_min=0, end_min=55),
    target=7.0,
    sigma=2.5,
    init_mode=InitMode.NORM_CENTER,
)

# THE LOAD-BEARING ONE. The gap between the CCK peak (~10 min) and the serum bile-acid
# peak (75-120 min) is gallbladder emptying, intestinal transit, ileal reabsorption and
# hepatic first-pass extraction acting IN SERIES. Pinning both ends means the axis
# cannot satisfy the literature by collapsing the middle into a single lag.
BILE_ACIDS_TIME_TO_PEAK = CohortStatisticSpec(
    name="bile_acids_time_to_peak",
    source="Fasting and postprandial serum bile acids in normal persons (Digestion 1978)",
    description="Serum total bile acids peak 75-120 min after a meal, long after CCK",
    arms=(_ARM,),
    marker_id="bile_acids",
    kind=StatisticKind.TIME_TO_PEAK,
    window=StatisticWindow(start_min=60, end_min=300),
    target=95.0,
    sigma=20.0,
    softargmax_beta=10.0,  # see the beta note in the module docstring
    init_mode=InitMode.NORM_CENTER,
)

BILE_ACIDS_POSTPRANDIAL_PEAK = CohortStatisticSpec(
    name="bile_acids_postprandial_peak",
    source="Postprandial reference interval 4.7-20.2 umol/L (ICP reference-range study, 2022)",
    description="Peak serum total bile acids after a mixed meal",
    arms=(_ARM,),
    marker_id="bile_acids",
    kind=StatisticKind.PEAK_VALUE,
    window=StatisticWindow(start_min=60, end_min=300),
    target=12.0,
    sigma=4.0,
    init_mode=InitMode.NORM_CENTER,
)

COHORT_STATISTICS: list[CohortStatisticSpec] = [
    CCK_FASTING,
    CCK_POSTPRANDIAL_PEAK,
    CCK_TIME_TO_PEAK,
    BILE_ACIDS_FASTING,
    BILE_ACIDS_TIME_TO_PEAK,
    BILE_ACIDS_POSTPRANDIAL_PEAK,
]
