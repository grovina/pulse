"""
Cohort statistics: glucose-stimulated insulin release, postprandial
glucose handling, and hepatic glucose output suppression.

This is the iter 20 batch — six quantitative findings drawn from the
classic glucose-handling literature, targeting the same downstream
physiology that the insulin sweep signal probes directly. Insulin
sweep distills the GSIR / clearance / hepatic-suppression *shapes*
against the cold model; these cohort specs anchor the *amplitudes*
to real human trial outcomes, which the cold model cannot guarantee.

The two should cross-constrain: if the metabolic module satisfies the
sweep but lands at the wrong amplitude, these specs penalize it; if
it satisfies the cohort targets via a wrong-shape function, the sweep
penalizes it.
"""

from __future__ import annotations

from ..cohort_types import (
    TargetShape,
    CohortArmSpec,
    CohortStatisticSpec,
    InitMode,
    StatisticKind,
    StatisticWindow,
)


# Standard 75 g OGTT protocol — water-only run-in, glucose load at t=60min.
# Window minutes are relative to arm start (t=0 corresponds to start_hour).
_OGTT_75G_MEALS = ((60.0, 75.0, 0.0, 0.0),)


# ---------------------------------------------------------------------------
# 75 g OGTT → glucose peak in healthy adults
# ---------------------------------------------------------------------------
# DeFronzo (1979); ADA OGTT criteria; WHO 1999. Healthy adult plasma
# glucose peaks ≈140–170 mg/dL at 30–60 min after 75 g oral glucose,
# returning toward baseline by 120 min. Peak target 150 ± 25.
OGTT_GLUCOSE_PEAK = CohortStatisticSpec(
    name="ogtt_75g_glucose_peak",
    source="DeFronzo (1979); ADA / WHO OGTT criteria",
    description="75 g OGTT → glucose peak ≈ 150 mg/dL at 30–60 min post-load (healthy adults)",
    arms=(
        CohortArmSpec(label="ogtt_75g", duration_min=300, start_hour=8.0, meals=_OGTT_75G_MEALS),
    ),
    marker_id="glucose",
    kind=StatisticKind.PEAK_VALUE,
    window=StatisticWindow(start_min=60, end_min=180),
    target=150.0,
    sigma=25.0,
)


# ---------------------------------------------------------------------------
# 75 g OGTT → glucose at 120 min returns to ≈ 120 mg/dL
# ---------------------------------------------------------------------------
# WHO OGTT criterion: 2-h plasma glucose < 140 mg/dL is normal glucose
# tolerance. Iter 97 (review 4.3): that is a CEILING, not a target — the old
# 120 ± 15 point target penalized a healthy 105 as much as a 135. Encoded as
# AT_MOST 140 with the same sigma on the excess.
OGTT_GLUCOSE_LATE = CohortStatisticSpec(
    name="ogtt_75g_glucose_120min",
    source="WHO (1999); ADA OGTT criteria",
    description="75 g OGTT → mean glucose < 140 mg/dL in the 90–150 min window (normal glucose tolerance)",
    arms=(
        CohortArmSpec(label="ogtt_75g", duration_min=300, start_hour=8.0, meals=_OGTT_75G_MEALS),
    ),
    marker_id="glucose",
    kind=StatisticKind.MEAN_IN_WINDOW,
    window=StatisticWindow(start_min=150, end_min=210),
    target=140.0,
    sigma=15.0,
    shape=TargetShape.AT_MOST,
)


# ---------------------------------------------------------------------------
# 75 g OGTT → insulin peak in healthy adults
# ---------------------------------------------------------------------------
# Polonsky et al. (1988); DeFronzo (1979): peak post-OGTT insulin ≈ 60
# µU/mL at 30–60 min in healthy adults (range 40–100). Sigma 25 reflects
# the wide normal-range envelope.
OGTT_INSULIN_PEAK = CohortStatisticSpec(
    name="ogtt_75g_insulin_peak",
    source="Polonsky et al. (1988); DeFronzo (1979) — β-cell function in vivo",
    description="75 g OGTT → insulin peak ≈ 60 µU/mL at 30–60 min post-load (healthy adults)",
    arms=(
        CohortArmSpec(label="ogtt_75g", duration_min=300, start_hour=8.0, meals=_OGTT_75G_MEALS),
    ),
    marker_id="insulin",
    kind=StatisticKind.PEAK_VALUE,
    window=StatisticWindow(start_min=60, end_min=180),
    target=60.0,
    sigma=25.0,
)


# ---------------------------------------------------------------------------
# 75 g OGTT → insulin mean over 30–180 min
# ---------------------------------------------------------------------------
# Polonsky et al. (1988): integrated insulin over the 0–3 h post-load
# window in normoglycemic adults averages ≈30–40 µU/mL — the AUC analog
# that exposes total β-cell output rather than just the peak instant.
OGTT_INSULIN_MEAN = CohortStatisticSpec(
    name="ogtt_75g_insulin_mean_3h",
    source="Polonsky et al. (1988) — integrated post-OGTT insulin",
    description="75 g OGTT → mean insulin ≈ 35 µU/mL in 30–180 min window (healthy adults)",
    arms=(
        CohortArmSpec(label="ogtt_75g", duration_min=300, start_hour=8.0, meals=_OGTT_75G_MEALS),
    ),
    marker_id="insulin",
    kind=StatisticKind.MEAN_IN_WINDOW,
    window=StatisticWindow(start_min=90, end_min=240),
    target=35.0,
    sigma=15.0,
)


# ---------------------------------------------------------------------------
# 75 g OGTT → insulin at 120 min, the tail
# ---------------------------------------------------------------------------
# PMID 3525269, 247 healthy subjects with normal glucose tolerance: insulin
# at 2 h is 0.24 ± 0.01 nmol/L (SEM). At 6.945 pmol/L per µU/mL that is
# 34.6 ± 1.44 µU/mL. The 3 h mean above can be met by a high peak and a
# crash; this window is the sample itself, 110–130 min after the load
# (the arm's meal is at minute 60). Sigma is the published SEM, so n_arm
# stays 1 and the loss is not tightened by the training batch.
OGTT_INSULIN_120 = CohortStatisticSpec(
    name="ogtt_75g_insulin_120min",
    source="PMID 3525269 — 75 g OGTT, 247 healthy subjects, normal glucose tolerance",
    description="75 g OGTT → insulin ≈ 34.6 µU/mL at 120 min (0.24 nmol/L)",
    arms=(
        CohortArmSpec(label="ogtt_75g", duration_min=300, start_hour=8.0, meals=_OGTT_75G_MEALS),
    ),
    marker_id="insulin",
    kind=StatisticKind.MEAN_IN_WINDOW,
    window=StatisticWindow(start_min=170, end_min=190),
    target=34.6,
    sigma=1.44,
    n_arm=1,
)


# ---------------------------------------------------------------------------
# Mixed meal → hepatic glucose output suppression
# ---------------------------------------------------------------------------
# Rizza et al. (1981); Basu et al. (2000): physiological postprandial
# hyperinsulinemia suppresses endogenous glucose production by ≈60–80%
# within 60–180 min post-meal. From a fasted basal of ≈2.0 mg/kg/min,
# HGO drops by ≈-1.0 mg/kg/min over the post-meal window. We use a 75 g
# carb-rich mixed meal to drive a clean hyperinsulinemic episode, then
# encode the HGO contrast vs a same-duration fasted arm.
_HGO_FASTED: tuple[tuple[float, float, float, float], ...] = ()
_HGO_MEAL = ((60.0, 75.0, 15.0, 25.0),)
MEAL_HGO_SUPPRESSION = CohortStatisticSpec(
    name="meal_hgo_suppression",
    source="Rizza et al. (1981); Basu et al. (2000) — postprandial HGO suppression",
    description="Mixed meal (75 g carb) → HGO ≈ -1.0 mg/kg/min lower vs fasted in 90–180 min post-meal",
    arms=(
        CohortArmSpec(label="fasted", duration_min=300, start_hour=8.0, meals=_HGO_FASTED),
        CohortArmSpec(label="meal", duration_min=300, start_hour=8.0, meals=_HGO_MEAL),
    ),
    marker_id="hepatic_output",
    kind=StatisticKind.DELTA_MEANS,
    window=StatisticWindow(start_min=150, end_min=240),
    target=-1.0,
    sigma=0.5,
)


# ---------------------------------------------------------------------------
# Small (15 g) carb challenge → modest glucose peak
# ---------------------------------------------------------------------------
# Wolever (1991, 1996); standard low-dose carb challenges: 15 g glucose
# raises plasma glucose ≈10–25 mg/dL above basal at the 30–60 min peak
# in healthy adults. Encoded as PEAK_VALUE = 110 ± 12 mg/dL (basal ≈95
# + ~15 mg/dL excursion). Pairs with the 75 g peak at ~150 mg/dL to
# anchor the slope of the dose-response curve at the low end where the
# Wolever-style slope analysis breaks down.
_SMALL_CARB_MEAL = ((60.0, 15.0, 0.0, 0.0),)
SMALL_CARB_GLUCOSE_PEAK = CohortStatisticSpec(
    name="small_carb_glucose_peak",
    source="Wolever (1991, 1996) — low-dose glycemic-response anchor",
    description="15 g glucose challenge → glucose peak ≈ 110 mg/dL at 30–60 min (low-dose anchor)",
    arms=(
        CohortArmSpec(label="small_carb", duration_min=240, start_hour=8.0, meals=_SMALL_CARB_MEAL),
    ),
    marker_id="glucose",
    kind=StatisticKind.PEAK_VALUE,
    window=StatisticWindow(start_min=60, end_min=180),
    target=110.0,
    sigma=12.0,
)


# These specs all begin from a clean fasted state, which is what the
# cold-model trajectory delivers at t=0. The OGTT / mixed-meal paradigm
# is the canonical use case for cold-model anchoring; per-spec init_mode
# is left at the default (cold).
_ = InitMode  # intentional: marker that init_mode is reviewed for this batch.

# ---------------------------------------------------------------------------
# Upright bicycle, 40 min, splanchnic glucose output
# ---------------------------------------------------------------------------
# Wahren, Felig, Ahlborg and Jorfeldt, J Clin Invest 1971;50:2715 (PMID
# 5129319). Healthy postabsorptive men. Splanchnic glucose production is
# estimated hepatic blood flow times the arterial–hepatic venous
# difference. At 800 kg-m/min (130 W; 1 kg-m/min = 0.163 W), n = 7:
# rest 0.82 ± 0.10 mmol/min, 40 min 2.23 ± 0.49. The model's moderate
# bout is activity 0.7. The delta, on the 70 kg reference mass, is
# (2.23 − 0.82) × 180 / 70 = 3.63 mg/kg/min. The SEM of the difference
# is sqrt(0.49² + 0.10²) × 180 / 70 = 1.29, so n_arm stays 1.
_WAHREN_MIN = 80
_WAHREN_REST = tuple(0.0 for _ in range(_WAHREN_MIN))
_WAHREN_BOUT = tuple(0.0 if t < 20 else 0.7 for t in range(_WAHREN_MIN))
WAHREN_EXERCISE_HGO = CohortStatisticSpec(
    name="wahren_800_splanchnic_glucose",
    source="Wahren et al. (1971) J Clin Invest — splanchnic glucose output, 800 kg-m/min",
    description=(
        "40 min of moderate cycling raises splanchnic glucose output "
        "by ≈ 3.63 mg/kg/min above rest (2.23 vs 0.82 mmol/min)"
    ),
    arms=(
        CohortArmSpec(
            label="rest", duration_min=_WAHREN_MIN, start_hour=10.0,
            meals=(), activity=_WAHREN_REST,
        ),
        CohortArmSpec(
            label="cycle_800", duration_min=_WAHREN_MIN, start_hour=10.0,
            meals=(), activity=_WAHREN_BOUT,
        ),
    ),
    marker_id="hepatic_output",
    kind=StatisticKind.DELTA_MEANS,
    window=StatisticWindow(start_min=55, end_min=70),
    target=3.63,
    sigma=1.29,
    n_arm=1,
)


# Same bout, Table II. Arterial glucose, mmol/L, n = 9: rest 4.16 ± 0.12,
# 40 min 4.62 ± 0.33. The rise is 0.46 × 18 = 8.28 mg/dL. The SEM of the
# difference, unpaired, is sqrt(0.12² + 0.33²) × 18 = 6.32. Read with the
# splanchnic spec: production and this rise together are what identify
# exercise uptake.
WAHREN_EXERCISE_GLUCOSE = CohortStatisticSpec(
    name="wahren_800_arterial_glucose",
    source="Wahren et al. (1971) J Clin Invest — arterial glucose, 800 kg-m/min",
    description=(
        "40 min of moderate cycling raises arterial glucose "
        "by ≈ 8.3 mg/dL (4.62 vs 4.16 mmol/L)"
    ),
    arms=WAHREN_EXERCISE_HGO.arms,
    marker_id="glucose",
    kind=StatisticKind.DELTA_MEANS,
    window=StatisticWindow(start_min=55, end_min=70),
    target=8.28,
    sigma=6.32,
    n_arm=1,
)


COHORT_STATISTICS: list[CohortStatisticSpec] = [
    OGTT_GLUCOSE_PEAK,
    OGTT_GLUCOSE_LATE,
    OGTT_INSULIN_PEAK,
    OGTT_INSULIN_MEAN,
    OGTT_INSULIN_120,
    MEAL_HGO_SUPPRESSION,
    SMALL_CARB_GLUCOSE_PEAK,
    WAHREN_EXERCISE_HGO,
    WAHREN_EXERCISE_GLUCOSE,
]
