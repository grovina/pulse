"""
Cohort statistics: meals, fasting, macronutrients, gut hormones.

Each spec encodes a literature effect-size and standard error so the loss
penalizes magnitude (Gaussian z²), not just sign of the contrast.
"""

from __future__ import annotations

from ..cohort_types import (
    CohortArmSpec,
    CohortStatisticSpec,
    StatisticKind,
    StatisticWindow,
)

# ---------------------------------------------------------------------------
# Skipped breakfast → lower morning mean glucose vs control breakfast
# ---------------------------------------------------------------------------
# Betts et al. (2014, Bath Breakfast Project): mean plasma glucose in the
# 8–11 AM window was ≈10–20 mg/dL lower in the fasted arm because no carb
# load arrived in that period. Use −12 mg/dL ± 6 mg/dL as a plausible
# group-mean effect (within typical inter-trial variance).
_BREAKFAST_CTRL_MEALS = (
    (120.0, 55.0, 8.0, 15.0),
    (360.0, 65.0, 12.0, 25.0),
    (720.0, 60.0, 15.0, 30.0),
)
_BREAKFAST_SKIP_MEALS = (
    (360.0, 65.0, 12.0, 25.0),
    (720.0, 60.0, 15.0, 30.0),
)
FASTING_BREAKFAST_GLUCOSE = CohortStatisticSpec(
    name="fasting_breakfast_glucose_morning",
    source="Betts et al. (2014) — Bath Breakfast Project",
    description="Skip breakfast → lower mean glucose in the 8–11 AM window",
    arms=(
        CohortArmSpec(
            label="control_breakfast",
            duration_min=1440, start_hour=6.0, meals=_BREAKFAST_CTRL_MEALS,
        ),
        CohortArmSpec(
            label="skipped_breakfast",
            duration_min=1440, start_hour=6.0, meals=_BREAKFAST_SKIP_MEALS,
        ),
    ),
    marker_id="glucose",
    kind=StatisticKind.DELTA_MEANS,
    window=StatisticWindow(start_min=120, end_min=300),
    target=-12.0,
    sigma=6.0,
)

# ---------------------------------------------------------------------------
# Note: postprandial glucose vs carbohydrate dose (Wolever 1991/1996) lives in
# its own dedicated training signal — see ``pulse.dose_response`` — because the
# cohort signal's hard ``series.max()`` peak extractor gives sparse / noisy
# gradient on a slope statistic, and slope-vs-dose deserves a focused weight
# rather than competing with six other cohort findings.
# ---------------------------------------------------------------------------
# Extended fast → BHB rise overnight
# ---------------------------------------------------------------------------
# Cahill (2006) and Owen (1969, 1983): after ~16 h fast BHB rises from
# ≈0.1 to ≈0.3 mmol/L in healthy adults. Use Δ = +0.20 mmol/L ± 0.10.
_NORMAL_EATING = (
    (120.0, 50.0, 8.0, 15.0),
    (360.0, 65.0, 12.0, 25.0),
    (720.0, 60.0, 15.0, 30.0),
)
_EXTENDED_FAST = (
    (120.0, 50.0, 8.0, 15.0),
    (360.0, 65.0, 12.0, 25.0),
)
EXTENDED_FAST_BHB = CohortStatisticSpec(
    name="extended_fast_bhb_overnight",
    source="Cahill (2006); Owen (1969) — fuel metabolism in fasting",
    description="Skipping dinner (≈16 h fast) → +0.2 mmol/L BHB in late-night/morning window",
    arms=(
        CohortArmSpec(label="normal_eating", duration_min=1440, start_hour=6.0, meals=_NORMAL_EATING),
        CohortArmSpec(label="extended_fast", duration_min=1440, start_hour=6.0, meals=_EXTENDED_FAST),
    ),
    marker_id="bhb",
    kind=StatisticKind.DELTA_MEANS,
    window=StatisticWindow(start_min=960, end_min=1320),
    target=0.20,
    sigma=0.10,
)

# Cahill (2006); Coppack (1989): the liver glycogen pool (~80-100 g) is
# largely depleted by a ≈16 h fast, while muscle glycogen (~400 g) is
# approximately preserved if no exercise is performed.
#
# Iter 56 split the iter-55 lumped `glycogen_pool` into `liver_glycogen`
# (τ ≈ 1 d) and `muscle_glycogen` (τ ≈ 3 wk). The −60 g fast-vs-fed
# delta is supervised on `liver_glycogen` alone — physically reachable
# within the 1-day protocol because liver's τ ≈ 1 d (the iter-55 lumped
# pool's τ ≈ 10 d made it unreachable; probe verified Δ = −0.01 g vs the
# −60 g target). `muscle_glycogen` is left unsupervised this iter (like
# mitochondrial_capacity): it is a *separate* state variable with its
# own SetpointHead, so the liver spec's gradient does not flow into it —
# no companion "muscle preserved" spec is needed (a Δ≈0 target on a
# zero-init SetpointHead would carry no gradient anyway). Muscle
# preservation under a resting fast falls out of its slow τ + absence of
# an exercise drive; its exercise supervision lands in iter 57 with the
# chronic-block protocol. See apps/pulse/docs/multi-timescale-plan.md.
EXTENDED_FAST_GLYCOGEN = CohortStatisticSpec(
    name="extended_fast_liver_glycogen_overnight",
    source="Cahill (2006); Coppack et al. (1989) — hepatic glycogen turnover",
    description="Skipping dinner (≈16 h fast) → −60 g mean liver_glycogen in late-fast window (hepatic depletion)",
    arms=(
        CohortArmSpec(label="normal_eating", duration_min=1440, start_hour=6.0, meals=_NORMAL_EATING),
        CohortArmSpec(label="extended_fast", duration_min=1440, start_hour=6.0, meals=_EXTENDED_FAST),
    ),
    marker_id="liver_glycogen",
    kind=StatisticKind.DELTA_MEANS,
    window=StatisticWindow(start_min=960, end_min=1320),
    target=-60.0,
    sigma=25.0,
)

# ---------------------------------------------------------------------------
# Extended fast → modest morning glucose drop
# ---------------------------------------------------------------------------
# Cahill (2006); Browning (2012): 16–24 h fasting drops glucose ~5–15
# mg/dL in healthy adults. Use Δ = −8 mg/dL ± 4.
EXTENDED_FAST_GLUCOSE = CohortStatisticSpec(
    name="extended_fast_glucose_morning",
    source="Cahill (2006); Browning et al. (2012) — fasting glycemia",
    description="Skipping dinner → −8 mg/dL mean glucose in overnight/morning window",
    arms=(
        CohortArmSpec(label="normal_eating", duration_min=1440, start_hour=6.0, meals=_NORMAL_EATING),
        CohortArmSpec(label="extended_fast", duration_min=1440, start_hour=6.0, meals=_EXTENDED_FAST),
    ),
    marker_id="glucose",
    kind=StatisticKind.DELTA_MEANS,
    window=StatisticWindow(start_min=960, end_min=1320),
    target=-8.0,
    sigma=4.0,
)

# ---------------------------------------------------------------------------
# Extended fast → FFA rise (lipolysis)
# ---------------------------------------------------------------------------
# Cahill (2006); Coppack et al. (1989); Frayn (2002): in healthy adults a
# 16–24 h fast roughly doubles plasma FFA from ≈0.4 to ≈0.8 mmol/L as
# adipose lipolysis is disinhibited. Vs the same-time mean in the
# normally-fed arm (where FFA is suppressed by postprandial insulin),
# Δ ≈ +0.40 mmol/L ± 0.20.
EXTENDED_FAST_FFA = CohortStatisticSpec(
    name="extended_fast_ffa_overnight",
    source="Cahill (2006); Coppack (1989); Frayn (2002) — fasting lipolysis",
    description="Skipping dinner (≈16 h fast) → +0.4 mmol/L FFA in late-night/morning window",
    arms=(
        CohortArmSpec(label="normal_eating", duration_min=1440, start_hour=6.0, meals=_NORMAL_EATING),
        CohortArmSpec(label="extended_fast", duration_min=1440, start_hour=6.0, meals=_EXTENDED_FAST),
    ),
    marker_id="ffa",
    kind=StatisticKind.DELTA_MEANS,
    window=StatisticWindow(start_min=960, end_min=1320),
    target=0.40,
    sigma=0.20,
)

# ---------------------------------------------------------------------------
# Extended fast → low fasting insulin (basal floor)
# ---------------------------------------------------------------------------
# Polonsky et al. (1988); Tabák et al. (2009): basal plasma insulin in
# healthy adults after an overnight (12–16 h) fast sits around 5–10 µU/mL.
# Encode the *level*, not a contrast — single-arm MEAN_IN_WINDOW in the
# late-night/morning window of the extended-fast protocol. Target 7 ± 4.
EXTENDED_FAST_INSULIN = CohortStatisticSpec(
    name="extended_fast_insulin_basal",
    source="Polonsky et al. (1988); Tabák et al. (2009) — basal insulin physiology",
    description="≈16 h fast → mean insulin ≈ 7 µU/mL in late-night/morning window",
    arms=(
        CohortArmSpec(label="extended_fast", duration_min=1440, start_hour=6.0, meals=_EXTENDED_FAST),
    ),
    marker_id="insulin",
    kind=StatisticKind.MEAN_IN_WINDOW,
    window=StatisticWindow(start_min=960, end_min=1320),
    target=7.0,
    sigma=4.0,
)

# ---------------------------------------------------------------------------
# OGTT (75 g glucose) → glucagon suppression
# ---------------------------------------------------------------------------
# Müller et al. (1970); Unger (1971); Müller, Unger (1971): a 75 g oral
# glucose load drops plasma glucagon ≈30–40% from preprandial baseline
# within 60–120 min (≈70 → ≈45 pg/mL). Encoded against a water-only
# control arm; mean glucagon in the 60–180 min window is ≈25 pg/mL lower.
_OGTT_75G = ((60.0, 75.0, 0.0, 0.0),)
_OGTT_WATER: tuple[tuple[float, float, float, float], ...] = ()
OGTT_GLUCAGON_SUPPRESSION = CohortStatisticSpec(
    name="ogtt_glucagon_suppression",
    source="Müller et al. (1970); Unger (1971) — α-cell response to oral glucose",
    description="75 g OGTT → mean glucagon ≈ 25 pg/mL lower vs water in 60–180 min post-load",
    arms=(
        CohortArmSpec(label="water_control", duration_min=300, start_hour=8.0, meals=_OGTT_WATER),
        CohortArmSpec(label="ogtt_75g",     duration_min=300, start_hour=8.0, meals=_OGTT_75G),
    ),
    marker_id="glucagon",
    kind=StatisticKind.DELTA_MEANS,
    window=StatisticWindow(start_min=120, end_min=240),
    target=-25.0,
    sigma=12.0,
)

# ---------------------------------------------------------------------------
# Mixed meal → modest glucagon suppression
# ---------------------------------------------------------------------------
# Mixed meals suppress glucagon less than pure glucose because dietary
# amino acids stimulate α-cells (Unger 1971; Holst 2007). A typical
# mixed meal (50 g carb + protein/fat) drops glucagon ≈10–15 pg/mL in
# the 60–180 min post-meal window vs a fasted control. Δ ≈ −10 ± 8.
_MIXED_MEAL_60G = ((60.0, 60.0, 12.0, 20.0),)
_MIXED_MEAL_NONE: tuple[tuple[float, float, float, float], ...] = ()
MIXED_MEAL_GLUCAGON_SUPPRESSION = CohortStatisticSpec(
    name="mixed_meal_glucagon_suppression",
    source="Unger (1971); Holst (2007) — α-cell response to mixed nutrients",
    description="Mixed meal (60 g carb, protein, fat) → −10 pg/mL glucagon vs fasted in 60–180 min post-meal",
    arms=(
        CohortArmSpec(label="fasted",     duration_min=300, start_hour=8.0, meals=_MIXED_MEAL_NONE),
        CohortArmSpec(label="mixed_meal", duration_min=300, start_hour=8.0, meals=_MIXED_MEAL_60G),
    ),
    marker_id="glucagon",
    kind=StatisticKind.DELTA_MEANS,
    window=StatisticWindow(start_min=120, end_min=240),
    target=-10.0,
    sigma=8.0,
)

# ---------------------------------------------------------------------------
# Mixed meal → FFA suppression (insulin antilipolysis)
# ---------------------------------------------------------------------------
# Coppack et al. (1989); Frayn (2002): a mixed meal raises insulin which
# is the dominant antilipolytic hormone. FFA drops from ≈0.5 mmol/L
# preprandially to ≈0.15–0.25 mmol/L within 60–180 min post-meal (≈50–70%
# suppression). Vs a same-duration fasted arm, Δ ≈ −0.30 mmol/L ± 0.15.
MEAL_FFA_SUPPRESSION = CohortStatisticSpec(
    name="meal_ffa_suppression",
    source="Coppack et al. (1989); Frayn (2002) — postprandial lipid metabolism",
    description="Mixed meal → −0.30 mmol/L FFA vs fasted in 60–180 min post-meal (insulin antilipolysis)",
    arms=(
        CohortArmSpec(label="fasted",     duration_min=300, start_hour=8.0, meals=_MIXED_MEAL_NONE),
        CohortArmSpec(label="mixed_meal", duration_min=300, start_hour=8.0, meals=_MIXED_MEAL_60G),
    ),
    marker_id="ffa",
    kind=StatisticKind.DELTA_MEANS,
    window=StatisticWindow(start_min=120, end_min=240),
    target=-0.30,
    sigma=0.15,
)

# ---------------------------------------------------------------------------
# Large vs small meal → GLP-1 peak
# ---------------------------------------------------------------------------
# Holst (2007); Vilsbøll et al. (2003): peak postprandial GLP-1 is dose-
# dependent on meal carbohydrate + fat. A large mixed meal (≈90 g carb,
# 25 g fat) yields peak ≈25 pmol/L; a small one ≈10 pmol/L. Δ peak ≈ +12 ± 6.
_LARGE_MEAL = ((120.0, 90.0, 25.0, 35.0),)
_SMALL_MEAL = ((120.0, 30.0, 5.0, 10.0),)
LARGE_MEAL_GLP1 = CohortStatisticSpec(
    name="large_meal_glp1_peak",
    source="Holst (2007); Vilsbøll et al. (2003) — physiology of GLP-1",
    description="Large mixed meal → +12 pmol/L higher GLP-1 peak vs small meal (30–120 min post-meal)",
    arms=(
        CohortArmSpec(label="small_meal", duration_min=360, start_hour=6.0, meals=_SMALL_MEAL),
        CohortArmSpec(label="large_meal", duration_min=360, start_hour=6.0, meals=_LARGE_MEAL),
    ),
    marker_id="glp1",
    kind=StatisticKind.DELTA_PEAKS,
    window=StatisticWindow(start_min=150, end_min=240),
    target=12.0,
    sigma=6.0,
    # iter 59: weight 1.0 -> 12.0. iter 58 (the breadth floor) anchored
    # 12 markers and overall_mape stayed flat because the embedding-
    # bottleneck reshuffle migrated entirely into glp1 (+0.477 MAPE
    # while glucagon/insulin/ffa each improved 0.13-0.33) — glp1 was
    # the one major high-variance marker left under-anchored: this was
    # its ONLY cohort spec, at default weight, and the iter-57-ckpt
    # cohort-ablation showed it gradient-starved (g_met 0.30, pred ~0
    # vs target 12, z -2.0) — the exact pattern leptin had before its
    # iter-58 weight=12 fix. Same remedy (apps/pulse/docs/physiology-
    # coverage.md "iter 58 RESULT"): bump weight so the peak anchor
    # bites, AND add the level anchor below (a peak-difference alone
    # does not pin the glp1 *level* the reshuffle moves).
    weight=12.0,
)

# ---------------------------------------------------------------------------
# Fed vs fasted → GLP-1 level (incretin presence)
# ---------------------------------------------------------------------------
# Holst (2007); Vilsbøll et al. (2003): GLP-1 is low in the fasted state
# (~5–10 pmol/L) and sustains an elevated level for 1–3 h after a mixed
# meal. The large/small PEAK spec above constrains the response *slope*
# but not the absolute level; this DELTA_MEANS level anchor is glp1's
# equivalent of the breadth-floor MEAN anchors that pinned the other 12
# markers in iter 58. Gentle +8 ± 6 pmol/L. Mirrors the ghrelin-
# suppression fed/fasted 480-min arm shape, kept self-contained
# (defined before that section in module order; barely moves the
# Phase-2 OOM budget tracked in breadth_floor.py). weight stays DEFAULT
# (1.0): the iter-58-ckpt cohort-ablation pre-flight showed this
# DELTA_MEANS level anchor already carries g_oth≈4.4 at weight 1 (same
# order as the ghrelin spec sharing the appetite module) — a well-posed
# window-mean residual flows broad gradient, unlike the *sparse*
# DELTA_PEAKS gradient of the peak spec. Bumping THIS to 12 would make
# it ~5x louder than ghrelin and swamp the module / over-pull glp1
# (a fresh reshuffle). Only the peak spec was starved and needs the
# weight; this one bites on its own.
_GLP1_FED_ARM = (
    (120.0, 55.0, 12.0, 20.0),
    (360.0, 65.0, 15.0, 25.0),
)
_GLP1_FASTED_ARM: tuple[tuple[float, float, float, float], ...] = ()
MEAL_GLP1_RISE = CohortStatisticSpec(
    name="meal_glp1_rise",
    source="Holst (2007); Vilsbøll et al. (2003) — physiology of GLP-1",
    description="Eating → mean GLP-1 ~8 pmol/L higher than fasting in 1–3 h post-meal window",
    arms=(
        CohortArmSpec(label="fasted", duration_min=480, start_hour=6.0, meals=_GLP1_FASTED_ARM),
        CohortArmSpec(label="fed",    duration_min=480, start_hour=6.0, meals=_GLP1_FED_ARM),
    ),
    marker_id="glp1",
    kind=StatisticKind.DELTA_MEANS,
    window=StatisticWindow(start_min=180, end_min=300),
    target=8.0,
    sigma=6.0,
)

# ---------------------------------------------------------------------------
# Fed vs fasted → ghrelin suppression
# ---------------------------------------------------------------------------
# Cummings et al. (2001): postprandial ghrelin drops ≈30% from preprandial
# baseline within 60–120 min, recovering before next meal. Compared with a
# continuously fasting arm at matched time, mean ghrelin in 1–3 h post-meal
# is ≈30 pg/mL lower. Use Δ = −30 ± 15.
_FED_ARM = (
    (120.0, 55.0, 12.0, 20.0),
    (360.0, 65.0, 15.0, 25.0),
)
_FASTED_ARM: tuple[tuple[float, float, float, float], ...] = ()
MEAL_GHRELIN_SUPPRESSION = CohortStatisticSpec(
    name="meal_ghrelin_suppression",
    source="Cummings et al. (2001) — preprandial rise in ghrelin",
    description="Eating → mean ghrelin ~30 pg/mL lower than fasting in 1–3 h post-meal window",
    arms=(
        CohortArmSpec(label="fasted", duration_min=480, start_hour=6.0, meals=_FASTED_ARM),
        CohortArmSpec(label="fed",    duration_min=480, start_hour=6.0, meals=_FED_ARM),
    ),
    marker_id="ghrelin",
    kind=StatisticKind.DELTA_MEANS,
    window=StatisticWindow(start_min=180, end_min=300),
    target=-30.0,
    sigma=15.0,
)

COHORT_STATISTICS: list[CohortStatisticSpec] = [
    FASTING_BREAKFAST_GLUCOSE,
    EXTENDED_FAST_BHB,
    EXTENDED_FAST_GLYCOGEN,
    EXTENDED_FAST_GLUCOSE,
    EXTENDED_FAST_FFA,
    EXTENDED_FAST_INSULIN,
    LARGE_MEAL_GLP1,
    MEAL_GLP1_RISE,
    MEAL_GHRELIN_SUPPRESSION,
    OGTT_GLUCAGON_SUPPRESSION,
    MIXED_MEAL_GLUCAGON_SUPPRESSION,
    MEAL_FFA_SUPPRESSION,
]
