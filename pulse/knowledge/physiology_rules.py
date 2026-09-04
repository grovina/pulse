"""
Physiology rules — per-trajectory inequality / relational supervision.

SKETCH for iter 42. Not yet wired into ``train.py``; this file exists
to pressure-test the schema before committing to the full wiring.

Operationalizes the PRD's *Calibrated supervision* principle: encode
every plausibility constraint the literature supports, hinge-shaped so
the loss is zero past satisfaction. Sits alongside ``CohortStatisticSpec``
(population-mean targets) and ``CouplingPrior`` (graph-edge sign /
magnitude), but supervises *individual* simulated trajectories against
qualitative or relational claims that don't reduce to a single scalar
moment.

Why a new surface (vs. another cohort spec):

================  =====================================  ===========================
                  CohortStatisticSpec                    PhysiologyRule
================  =====================================  ===========================
evaluated over    mean across N synthetic patients,      every individual trajectory
                  then one statistic
escape hatch      per-patient noise averages out — one   nowhere to hide; every
                  patient flat, another moving, mean OK  rollout must satisfy
loss shape        ``((mean - target) / sigma) ** 2``     ``(violation / scale) ** 2``
                  always penalizes                       zero past satisfaction
target shape      scalar moment (mean, peak, AUC, …)     arbitrary predicate
                                                         (direction, sign, timing,
                                                         relation, range)
================  =====================================  ===========================

The hinge form is structurally aligned with "we shouldn't enforce more
than we know": once the predicate is satisfied, gradient stops. We're
not pushing the marker beyond what the literature says — only ruling
out the regime the literature says shouldn't happen.

Authoring a rule:

1. Cite a source (textbook chapter, RCT, review). Like cohort specs,
   the citation goes in ``source`` and a one-line summary in
   ``description``.
2. Define a single arm protocol (``arm: CohortArmSpec``) that exercises
   the relevant physiological regime. Reuses cohort_types so existing
   protocols compose.
3. Write a predicate ``(trajectory, ctx) -> Tensor[scalar]`` that
   returns the *violation amount* in marker-native units. Always
   non-negative — apply ReLU inside the predicate. Zero means
   satisfied. Helpers below cover the common shapes.
4. Choose ``scale``: violation magnitude that should produce loss=1.
   Looser scale for qualitative claims, tighter for quantitative
   ones. Same role as ``sigma`` in CohortStatisticSpec.

The trap to avoid (cf. iter-39 vitality range floor): predicates must
be *informative on flat data*. A rule like ``post_meal < pre_meal``
with strict inequality is satisfied by any flat trajectory because
``≤`` is satisfied by equality. Use a strict magnitude:
``pre - post > Δ`` with Δ from the literature, so a flat trajectory
incurs the full Δ as violation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from .cohort_types import CohortArmSpec, InitMode


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RuleContext:
    """Metadata available to a predicate at evaluation time.

    The predicate receives the rolled-out trajectory plus this context.
    Marker indexing is by name (predicates are written against marker
    ids, not column positions, so the contract survives marker reorders).
    """

    marker_index: dict[str, int]
    step_min: float  # integration step size in minutes
    arm: CohortArmSpec  # arm being evaluated (carries meal times etc.)

    def col(self, marker: str) -> int:
        return self.marker_index[marker]

    def step(self, t_min: float) -> int:
        """Convert minutes-from-arm-start to trajectory row index."""
        return int(round(t_min / self.step_min))

    def window(self, start_min: float, end_min: float) -> slice:
        return slice(self.step(start_min), self.step(end_min))

    @property
    def meal_times(self) -> tuple[float, ...]:
        return tuple(m[0] for m in self.arm.meals)


# Predicate signature: (trajectory[T, M], ctx) -> Tensor[scalar].
# Returns non-negative violation in marker-native units.
PredicateFn = Callable[[torch.Tensor, RuleContext], torch.Tensor]


@dataclass(frozen=True)
class PhysiologyRule:
    """One literature-derived plausibility constraint.

    A rule is a predicate that must hold across **every** arm in
    ``arms``. Single-arm rules pass a 1-tuple. Multi-arm rules force
    the relationship to hold under multiple protocols (postprandial
    breakfast vs OGTT, fasted vs fed, etc.) — the iter-42 lesson was
    that satisfying a rule on one arm doesn't generalize to the
    bench's different arms, so multi-arm coverage is the route to
    bench-translatable supervision.

    The loss aggregator (in ``physiology_rules_loss``) sums the
    per-arm violations and divides by the arm count: any single arm
    fully violating still produces (full_violation / N) of pull, so
    one-arm satisfaction reduces but doesn't eliminate the gradient.
    """

    name: str
    source: str
    description: str
    arms: tuple[CohortArmSpec, ...]
    predicate: PredicateFn
    scale: float  # violation magnitude that yields loss=1
    weight: float = 1.0
    init_mode: InitMode = InitMode.COLD
    # Iter 97 (review 4.5): a rule the TEACHER violates pulls the student two
    # ways at once (trajectory imitation vs the rule). That is acceptable only
    # when it is deliberate — the rule encodes literature the teacher gets wrong
    # and is meant to correct it. Such rules are flagged here with the reason;
    # ``scripts/rules_teacher_audit.py`` fails on any teacher violation that is
    # NOT flagged, so a contradiction cannot enter the registry unnoticed.
    teacher_correction: bool = False
    teacher_correction_note: str = ""

    def __post_init__(self) -> None:
        if self.scale <= 0:
            raise ValueError(f"{self.name}: scale must be positive (got {self.scale})")
        if not self.arms:
            raise ValueError(f"{self.name}: at least one arm required")
        if self.teacher_correction and not self.teacher_correction_note:
            raise ValueError(f"{self.name}: teacher_correction needs a note saying why")


# ---------------------------------------------------------------------------
# Predicate helpers
# ---------------------------------------------------------------------------
# Common shapes, kept tiny on purpose. Each rule is more readable when
# its predicate uses these directly rather than hiding behavior in a
# DSL. Strictly positive return convention — caller never wraps in ReLU.

def hinge_min_drop(traj: torch.Tensor, col: int, pre: slice, post: slice, min_drop: float) -> torch.Tensor:
    """Violation when ``mean(pre) - mean(post) < min_drop``.

    Use for "marker decreases after event by at least Δ" rules.
    """
    delta = traj[pre, col].mean() - traj[post, col].mean()
    return torch.relu(min_drop - delta)


def hinge_min_rise(traj: torch.Tensor, col: int, pre: slice, post: slice, min_rise: float) -> torch.Tensor:
    """Violation when ``mean(post) - mean(pre) < min_rise``."""
    delta = traj[post, col].mean() - traj[pre, col].mean()
    return torch.relu(min_rise - delta)


def _safe_window_corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pearson correlation of two window series with a scale-free guard.

    Add eps INSIDE the sqrt (not clamp_min on the result): when a marker's
    window is perfectly flat (variance 0), sqrt(0) is finite in the forward but
    its backward is 1/(2·sqrt(0)) = NaN, and clamping the sqrt OUTPUT leaves that
    NaN gradient intact (this aborted iter 83/84, n_nan=111).

    Iter 97 (review 4.6): the eps used to be an ABSOLUTE 1e-6 in squared marker
    units, so it meant nothing for glucose (sum-of-squares ~1e4) and everything
    for temp or bhb (~1e-2). It is now a fraction of the PARTNER's variance: a
    flat series against a moving one yields corr ~ 0 with a gradient bounded
    relative to the partner's scale, whatever the units.
    """
    a = a - a.mean()
    b = b - b.mean()
    aa = (a * a).sum()
    bb = (b * b).sum()
    rel = 1e-3
    tiny = 1e-12
    denom = torch.sqrt((aa + rel * bb + tiny) * (bb + rel * aa + tiny))
    return (a * b).sum() / denom


def hinge_max_correlation(
    traj: torch.Tensor, col_a: int, col_b: int, window: slice, max_corr: float,
) -> torch.Tensor:
    """Violation when correlation in window exceeds ``max_corr``.

    Use for "X and Y should be inversely related" rules with
    ``max_corr`` set negative (e.g. -0.3 means corr must be ≤ -0.3).
    """
    corr = _safe_window_corr(traj[window, col_a], traj[window, col_b])
    return torch.relu(corr - max_corr)


# Iter 96: sharpness for every soft-argmax/argmin here, expressed in units of
# the marker's own value range rather than absolute marker units. ~20 is where
# the weights stop reporting the window centroid and start reporting the peak
# (measured across six markers spanning ranges 0.5 to 32 — see the helper below).
_SOFTARGMAX_BETA_SCALE = 20.0


def _range_normalized_softmax_weights(
    values: torch.Tensor, beta_scale: float, sign: float = 1.0,
) -> torch.Tensor:
    """Softmax weights whose sharpness is measured in units of the marker's own range.

    ITER 96 — WHY THIS EXISTS. Every soft-argmax here used an ABSOLUTE
    ``beta`` of 0.05, i.e. `softmax(0.05 · value)`. Sharpness is only
    meaningful relative to how far the values actually spread: the useful
    quantity is ``beta · range``, and it needs to be ~20 or more before the
    weights concentrate on the peak at all. Measured on a teacher 24 h
    trajectory (scratchpad/beta.py):

        marker      range   true argmax   reported at beta=0.05   beta*range
        cortisol    14.13       508 min          654 min             0.71
        sbp         15.88       513              687                 0.79
        temp         0.51       920              722                 0.03
        acth        31.98       459              570                 1.60
        leptin       0.94       418              714                 0.05

    Every one collapses toward 719.5 — the CENTROID of the 1440-minute
    window. The rules were not measuring a peak; they were measuring the
    middle of the window and calling it a peak. Two of them
    (``cortisol_morning_peak``, ``sbp_morning_surge``) therefore reported
    standing violations of 114 and 87 minutes against trajectories whose
    true peaks sit INSIDE their target bands, at ~0.95 and ~0.73 loss each,
    with a gradient that pushes the whole trajectory toward midday rather
    than moving the peak. ``temp_afternoon_peak``'s real 40-minute error was
    reported as 238.

    Normalising by the (detached) range makes ``beta_scale`` dimensionless
    and self-calibrating, so a marker added later cannot inherit a silently
    broken sharpness the way `bile_acids` did in iter 95 (that one was
    caught only because it happened to eat half the cohort objective).
    The range is detached: it sets the temperature, it is not a thing to
    optimise through.
    """
    spread = (values.max() - values.min()).detach()
    # NEAR-FLAT WINDOWS. Dividing by the raw spread is a trap: a window that
    # barely moves drives `beta` toward infinity, and -- because the weights on
    # a flat input stay UNIFORM rather than saturating -- the gradient scales
    # with it. Measured on a 240-step constant window, a 1e-6 spread floor gave
    # |grad| = 1.2e9 against 1.97e1 for a normal sinusoidal window: eight orders
    # of magnitude, on a term meant to be a soft plausibility hinge. So the
    # floor is RELATIVE to the window's own level. Every marker these rules
    # cover clears it comfortably -- temp has the narrowest real range at 1.4 %
    # of its level, 4.6x the floor -- while a genuinely flat window, where the
    # time of the peak means nothing anyway, gets a bounded beta rather than an
    # explosive one.
    level = values.abs().mean().detach()
    min_spread = torch.clamp(3e-3 * level, min=1e-6)
    beta = beta_scale / torch.maximum(spread, min_spread)
    return torch.softmax(sign * values * beta, dim=0)


def hinge_argmax_in_band(
    traj: torch.Tensor, col: int, window: slice,
    target_min: float, target_max: float, step_min: float,
    softargmax_beta: float = _SOFTARGMAX_BETA_SCALE,
) -> torch.Tensor:
    """Violation when soft-argmax minute over ``window`` falls outside [target_min, target_max].

    ``target_min`` / ``target_max`` are in minutes relative to the
    window start. ``step_min`` is the integration step in minutes
    (passed from RuleContext). ``softargmax_beta`` is a RANGE-RELATIVE
    sharpness (iter 96) — see ``_range_normalized_softmax_weights``.
    """
    values = traj[window, col]
    weights = _range_normalized_softmax_weights(values, softargmax_beta)
    indices = torch.arange(values.shape[0], device=values.device, dtype=values.dtype)
    soft_step = (weights * indices).sum()
    soft_min = soft_step * step_min
    return torch.relu(target_min - soft_min) + torch.relu(soft_min - target_max)


def hinge_a_precedes_b(
    traj: torch.Tensor, col_a: int, col_b: int, window: slice,
    min_lead_min: float, max_lead_min: float, step_min: float,
    softargmax_beta: float = _SOFTARGMAX_BETA_SCALE,
) -> torch.Tensor:
    """Violation when A's soft-argmax does not lead B's by [min_lead, max_lead] minutes.

    Iter 65: proper relational predicate replacing the previous
    "two independent argmax bands" idiom (iter 64 paired
    acth/cortisol via two independent ``hinge_argmax_in_band``
    rules, but the model could satisfy both bands with simultaneous
    peaks because the bands overlapped 06:00–07:30). This computes
    ``lead = argmax_b − argmax_a`` (in minutes) and penalises
    departures from the literature interval — for ACTH→cortisol the
    cascade kinetics imply 15–30 min lead.
    """
    def soft_argmax_minutes(col: int) -> torch.Tensor:
        values = traj[window, col]
        weights = _range_normalized_softmax_weights(values, softargmax_beta)
        indices = torch.arange(values.shape[0], device=values.device, dtype=values.dtype)
        return (weights * indices).sum() * step_min

    lead = soft_argmax_minutes(col_b) - soft_argmax_minutes(col_a)
    return torch.relu(min_lead_min - lead) + torch.relu(lead - max_lead_min)


def _soft_extremum(values: torch.Tensor, beta_scale: float, sign: float) -> torch.Tensor:
    """Range-normalized soft max (``sign=+1``) / soft min (``sign=-1``).

    Iter 97 (review 4.6): the iter-96 fix routed the ARGMAX helpers through
    ``_range_normalized_softmax_weights`` but left the four VALUE helpers on an
    absolute ``beta = 0.05``. Measured with that beta on realistic bumps: a
    190 mg/dL glucose peak read 167 (violation 0 against a 180 ceiling), a
    55 mg/dL dip read 69 (violation 0 against a 65 floor), and a 1.0 °C
    temperature swing read 0.013 °C — the amplitude rule could never fire.
    40 rule arms were scoring the window MEAN and calling it the extremum.
    """
    weights = _range_normalized_softmax_weights(values, beta_scale, sign=sign)
    return (weights * values).sum()


def hinge_max_value(
    traj: torch.Tensor, col: int, window: slice, ceiling: float,
    softmax_beta: float = _SOFTARGMAX_BETA_SCALE,
) -> torch.Tensor:
    """Violation when soft-max over ``window`` exceeds ``ceiling``.

    Use for upper-bound asymptotes: postprandial glucose < 180,
    glp1 peak below clinical-toxicity, etc. The soft-max keeps the
    gradient differentiable across all timesteps — argmax-style
    hard predicates only reach the peak step. ``softmax_beta`` is a
    RANGE-RELATIVE sharpness (iter 97) — see ``_soft_extremum``.
    """
    values = traj[window, col]
    return torch.relu(_soft_extremum(values, softmax_beta, +1.0) - ceiling)


def hinge_min_value(
    traj: torch.Tensor, col: int, window: slice, floor: float,
    softmax_beta: float = _SOFTARGMAX_BETA_SCALE,
) -> torch.Tensor:
    """Violation when soft-min over ``window`` drops below ``floor``.

    Use for lower-bound asymptotes: fasting glucose > 65, etc. Range-relative
    sharpness (iter 97) — see ``_soft_extremum``.
    """
    values = traj[window, col]
    return torch.relu(floor - _soft_extremum(values, softmax_beta, -1.0))


def hinge_min_ratio(
    traj: torch.Tensor, num_col: int, denom_col: int, window: slice,
    floor_ratio: float, eps: float = 1e-3,
) -> torch.Tensor:
    """Violation when ``mean(num)/mean(denom)`` over ``window`` < ``floor_ratio``.

    Iter 74 inter-marker-ratio class. True ratio form, as opposed to the
    two-independent-per-marker-hinge idiom that ``GLUCAGON_INSULIN_RATIO``
    previously used (which only floored glucagon and never actually
    constrained the relationship). A ratio pins the *coupling* between two
    markers directly: the regime is satisfied iff their balance is right,
    not iff each clears an independent absolute threshold — so it stays
    correct under per-patient level shifts that move both markers together.

    ``num``/``denom`` are window-mean levels (not soft-extrema): ratio
    regimes are about sustained balance over the window, not a transient
    peak. ``denom`` is clamped to ``eps`` to keep the gradient finite when
    the denominator marker is driven toward zero (e.g. insulin in deep
    fast).
    """
    num = traj[window, num_col].mean()
    denom = traj[window, denom_col].mean().clamp_min(eps)
    ratio = num / denom
    return torch.relu(floor_ratio - ratio)


def hinge_monotone_decrease(
    traj: torch.Tensor, col: int, window: slice, min_total_drop: float,
) -> torch.Tensor:
    """Violation when ``mean(first quarter) − mean(last quarter) < min_total_drop``.

    Use for "marker monotonically depletes over window" rules
    (liver glycogen in fast, ghrelin in feeding window). Compares
    coarse window-endpoint means so noise inside the window
    doesn't fire the predicate spuriously.
    """
    values = traj[window, col]
    n = values.shape[0]
    quarter = max(1, n // 4)
    first_q = values[:quarter].mean()
    last_q = values[-quarter:].mean()
    drop = first_q - last_q
    return torch.relu(min_total_drop - drop)


def hinge_monotone_increase(
    traj: torch.Tensor, col: int, window: slice, min_total_rise: float,
) -> torch.Tensor:
    """Violation when ``mean(last quarter) − mean(first quarter) < min_total_rise``."""
    values = traj[window, col]
    n = values.shape[0]
    quarter = max(1, n // 4)
    first_q = values[:quarter].mean()
    last_q = values[-quarter:].mean()
    rise = last_q - first_q
    return torch.relu(min_total_rise - rise)


def hinge_min_correlation(
    traj: torch.Tensor, col_a: int, col_b: int, window: slice, min_corr: float,
) -> torch.Tensor:
    """Violation when correlation in window falls below ``min_corr``.

    Companion to ``hinge_max_correlation``. Use for "X and Y should
    be positively related" rules with ``min_corr`` set positive
    (e.g. +0.3 means corr must be ≥ +0.3).
    """
    corr = _safe_window_corr(traj[window, col_a], traj[window, col_b])
    return torch.relu(min_corr - corr)


def hinge_circadian_amplitude(
    traj: torch.Tensor, col: int, window: slice, min_amplitude: float,
    max_amplitude: float | None = None,
    softmax_beta: float = _SOFTARGMAX_BETA_SCALE,
) -> torch.Tensor:
    """Violation when (soft-max − soft-min) over ``window`` leaves the band.

    Use for "marker has a meaningful daily swing" rules — cortisol
    morning-peak vs evening-trough, temp pre-dawn vs late-afternoon,
    HR/BP day vs night, etc.

    Iter 94: ``max_amplitude`` added. With a floor alone this predicate is
    one-sided — every amplitude above ``min_amplitude`` is free — and the
    teacher's core-temperature swing had drifted to 1.04 °C, ~2x literature,
    without ever registering as a violation. Pass a ceiling wherever the
    literature bounds the swing from above as well; leaving it ``None``
    keeps the old floor-only behaviour for rules that genuinely only know
    "there must be a rhythm".
    """
    values = traj[window, col]
    amplitude = _soft_extremum(values, softmax_beta, +1.0) - _soft_extremum(values, softmax_beta, -1.0)
    violation = torch.relu(min_amplitude - amplitude)
    if max_amplitude is not None:
        violation = violation + torch.relu(amplitude - max_amplitude)
    return violation


def hinge_max_drift(
    traj: torch.Tensor, col: int, window: slice, max_drift: float,
    softmax_beta: float = _SOFTARGMAX_BETA_SCALE,
) -> torch.Tensor:
    """Violation when (soft-max − soft-min) over ``window`` exceeds ``max_drift``.

    Use for "marker should be stable" rules — slow chronic
    markers (leptin, mitochondrial capacity), overnight glucose
    drift, postprandial SpO2 stability.
    """
    values = traj[window, col]
    drift = _soft_extremum(values, softmax_beta, +1.0) - _soft_extremum(values, softmax_beta, -1.0)
    return torch.relu(drift - max_drift)


def hinge_argmin_in_band(
    traj: torch.Tensor, col: int, window: slice,
    target_min: float, target_max: float, step_min: float,
    softargmin_beta: float = _SOFTARGMAX_BETA_SCALE,
) -> torch.Tensor:
    """Violation when soft-argmin minute over ``window`` falls outside [target_min, target_max].

    Mirror of ``hinge_argmax_in_band`` for trough timing — cortisol
    evening trough, temp pre-dawn trough, HR sleep nadir.
    """
    values = traj[window, col]
    weights = _range_normalized_softmax_weights(values, softargmin_beta, sign=-1.0)
    indices = torch.arange(values.shape[0], device=values.device, dtype=values.dtype)
    soft_step = (weights * indices).sum()
    soft_min = soft_step * step_min
    return torch.relu(target_min - soft_min) + torch.relu(soft_min - target_max)


# ---------------------------------------------------------------------------
# Protocols (reused across rules)
# ---------------------------------------------------------------------------

# Standard mixed meal at t=60min after a fasted run-in. 75g carb / 25g
# protein / 20g fat — the typical "Western breakfast" used in
# postprandial physiology trials (e.g. Frayn 2003 review protocols).
_MIXED_MEAL_BREAKFAST_ARM = CohortArmSpec(
    label="mixed_meal_breakfast",
    duration_min=300,  # 0..60 fasted, 60..300 postprandial
    start_hour=8.0,
    meals=((60.0, 75.0, 25.0, 20.0),),
)

# Pure-carb load — same shape as a clinical 75g OGTT, but bracketed
# the same way as the breakfast arm for predicate compatibility. The
# pure-carb arm differs from the breakfast arm in macronutrient mix
# (fat/protein delay absorption); using both arms forces the same
# predicate to hold across both kinetic regimes.
_OGTT_75G_ARM = CohortArmSpec(
    label="ogtt_75g",
    duration_min=300,
    start_hour=8.0,
    meals=((60.0, 75.0, 0.0, 0.0),),
)

# Evening mixed meal — same macros as breakfast but at a different
# circadian phase. Catches postprandial dynamics that are
# time-of-day-modulated (insulin sensitivity, cortisol context).
_MIXED_MEAL_DINNER_ARM = CohortArmSpec(
    label="mixed_meal_dinner",
    duration_min=300,
    start_hour=19.0,
    meals=((60.0, 75.0, 25.0, 20.0),),
)

# Overnight fast extension: from 8h fasted (start) to 16h fasted.
# Used for ketogenesis / FFA rise / ghrelin rise rules.
_FAST_8H_TO_16H_ARM = CohortArmSpec(
    label="fast_8h_to_16h",
    duration_min=480,
    start_hour=22.0,  # 22:00 → 06:00, last meal already 8h prior
    meals=(),
)

# Longer fast — from 12h fasted (last meal at 18:00 yesterday) onward.
# Tests that the fasting-rise dynamics aren't pinned to a single
# duration; ketones should keep climbing past 16h.
_FAST_12H_TO_24H_ARM = CohortArmSpec(
    label="fast_12h_to_24h",
    duration_min=720,
    start_hour=6.0,  # 06:00 → 18:00, last meal yesterday at 18:00
    meals=(),
)

# 24h free-living arm with circadian probe (no meals — isolates
# diurnal cortisol pattern from meal effects).
_CIRCADIAN_24H_ARM = CohortArmSpec(
    label="circadian_24h",
    duration_min=24 * 60,
    start_hour=0.0,
    meals=(),
)

# 24h with realistic meals — cortisol's circadian peak should still
# fire even with meal-driven cortisol spikes layered on. Tests
# robustness of the timing rule under fed conditions.
_CIRCADIAN_24H_WITH_MEALS_ARM = CohortArmSpec(
    label="circadian_24h_with_meals",
    duration_min=24 * 60,
    start_hour=0.0,
    meals=(
        (8.0 * 60.0, 60.0, 20.0, 15.0),   # 08:00 breakfast
        (13.0 * 60.0, 70.0, 25.0, 20.0),  # 13:00 lunch
        (19.0 * 60.0, 80.0, 30.0, 25.0),  # 19:00 dinner
    ),
)


# Iter 66: prolonged fast continuation. 24h → 48h fasted. Tests that
# BHB keeps climbing past 24h (continues into ketosis plateau), FFA
# stays elevated, glucose floor holds at ~65 mg/dL, glycogen pools
# stay depleted, glucagon stays elevated. Cahill (1970) prolonged-
# fast paradigm: by 48h essentially all hepatic glycogen is gone,
# ketogenesis is in full swing.
# Iter 97 (review 4.5): ``prefast_hours=24`` — the cold initial state is the
# teacher's 24 h-fasted row (liver glycogen ~42 g, BHB ~1.3, insulin ~3.8), not
# its fed row 0 (100 g). Before this the arm labelled "24-48 h" was really
# "0-24 h", so ``liver_glycogen_depleted_prolonged_fast`` demanded < 30 g of a
# pool the teacher itself only brings to 55 g in that time, and the rule read
# as a 1.91 teacher violation that was entirely the frame.
_FAST_24H_TO_48H_ARM = CohortArmSpec(
    label="fast_24h_to_48h",
    duration_min=24 * 60,  # 24h window starting at 24h fasted
    start_hour=6.0,
    meals=(),
    prefast_hours=24.0,
)


# Iter 66: moderate exercise bout — 30 min rest → 60 min exercise
# (activity=0.7, ~50–60% VO2 max equivalent) → 90 min recovery.
# Tests HR rise, lactate rise (anaerobic component), FFA rise
# (mobilisation), muscle glycogen drop, cortisol acute rise.
# Length 180 min is enough for full kinetic shape; activity tuple
# is per-minute (1-min resolution matches the integrator step).
_MODERATE_EXERCISE_ARM = CohortArmSpec(
    label="moderate_exercise_bout",
    duration_min=180,
    start_hour=10.0,
    meals=(),
    activity=tuple(0.7 if 30 <= t < 90 else 0.0 for t in range(180)),
)


# Iter 66: full 24h with explicit sleep-wake schedule (22:00–06:00
# asleep). With realistic meals layered on. Tests HR/BP drop during
# sleep, HRV rise during sleep, temp trough pre-dawn, melatonin-coupled
# cortisol low at sleep onset.
#
# ITER 97 (review 1.2) — THE LITERAL WAS INVERTED. The convention everywhere
# else in the engine is ``sleep_wake = 0`` asleep / ``1`` awake
# (``model.py`` default_sleep_wake, ``full_body.py`` generate_sleep_wake,
# every cohort arm in ``knowledge/cohorts``). This arm wrote 1.0 for the night
# hours, so the teacher under it was AWAKE from 22:00 to 06:00 and ASLEEP from
# 06:00 to 22:00: SBP was 15 mmHg HIGHER at 02-05 than at 15-18, and every
# sleep/night rule that inherits this arm (15 of them) has been pushing the
# student toward the wrong phase since iter 66. Measured on the teacher with
# the mask corrected: sbp_falls_during_sleep violation 6.44 -> 0 (see
# scripts/rules_teacher_audit.py for the full before/after).
_SLEEP_WAKE_24H_ARM = CohortArmSpec(
    label="sleep_wake_24h",
    duration_min=24 * 60,
    start_hour=0.0,
    meals=(
        (8.0 * 60.0, 60.0, 20.0, 15.0),
        (13.0 * 60.0, 70.0, 25.0, 20.0),
        (19.0 * 60.0, 80.0, 30.0, 25.0),
    ),
    # 0 = asleep (22:00-06:00), 1 = awake — the engine-wide convention.
    sleep_wake=tuple(0.0 if (t < 6 * 60 or t >= 22 * 60) else 1.0 for t in range(24 * 60)),
    # Rest = 0 (review 1.7): the arm says nothing about exercise, so it must not
    # carry any. Explicit so the training frame equals the audit frame (4.9).
    activity=tuple(0.0 for _ in range(24 * 60)),
)


# ---------------------------------------------------------------------------
# Rules — first 5 across the dead-pathway markers
# ---------------------------------------------------------------------------

GLUCAGON_FALLS_POSTPRANDIAL = PhysiologyRule(
    name="glucagon_falls_postprandial",
    source="Unger & Orci (1981); Müller et al. (1970)",
    description=(
        "Plasma glucagon decreases by ≥ 5 pg/mL within 60 min of a "
        "carbohydrate-containing meal (alpha-cell suppression by "
        "elevated glucose / insulin). Required across mixed-meal "
        "and pure-carb (OGTT) and evening-meal arms."
    ),
    arms=(_MIXED_MEAL_BREAKFAST_ARM, _OGTT_75G_ARM, _MIXED_MEAL_DINNER_ARM),
    predicate=lambda traj, ctx: hinge_min_drop(
        traj, ctx.col("glucagon"),
        pre=ctx.window(30.0, 60.0),
        post=ctx.window(75.0, 120.0),
        min_drop=5.0,
    ),
    scale=5.0,
)


FFA_INVERSE_TO_INSULIN_POSTPRANDIAL = PhysiologyRule(
    name="ffa_inverse_to_insulin_postprandial",
    source="Frayn (2003) review; Boden (1996)",
    description=(
        "Postprandial FFA is anti-correlated with insulin "
        "(antilipolysis): correlation ≤ -0.3 over the 0–180 min "
        "post-meal window. Required across mixed-meal, OGTT, and "
        "evening-meal arms (kinetics differ; relationship shouldn't)."
    ),
    arms=(_MIXED_MEAL_BREAKFAST_ARM, _OGTT_75G_ARM, _MIXED_MEAL_DINNER_ARM),
    # Iter 97 (review 4.12): FFA suppression LAGS insulin (Frayn 2003, Fig. 3:
    # insulin peaks ~45-60 min, FFA nadir ~120-180 min post-meal), so the LINEAR
    # correlation over 0-180 min post-meal is modest even when the relation is
    # perfect — the teacher's is -0.2. -0.3 was a shape claim the lag forbids;
    # -0.15 still rules out zero or positive coupling.
    predicate=lambda traj, ctx: hinge_max_correlation(
        traj, ctx.col("ffa"), ctx.col("insulin"),
        window=ctx.window(60.0, 240.0),
        max_corr=-0.15,
    ),
    scale=0.5,  # correlation is unitless; scale=0.5 means full anti-corr violation ≈ loss 1
    # Iter 44: trim FFA's contribution. iter-43 diagnostic showed
    # ||g_met||=19.8 — much larger than the other 4 rules combined —
    # and insulin MAPE regressed 1.08 → 1.23 between iter 38 and 43.
    # The FFA correlation is real and must hold, but it shouldn't
    # outweigh the rest of the metabolic supervision. 0.5 = half-pull.
    weight=0.5,
)


GHRELIN_FALLS_AFTER_MEAL = PhysiologyRule(
    name="ghrelin_falls_after_meal",
    source="Cummings et al. (2001)",
    description=(
        "Plasma ghrelin drops by ≥ 30 pg/mL within 60 min of a meal "
        "(largest single hormonal correlate of pre/post-meal hunger). "
        "Required across mixed-meal and OGTT arms; the postprandial "
        "drop reflects nutrient sensing rather than carb-only effects."
    ),
    arms=(_MIXED_MEAL_BREAKFAST_ARM, _OGTT_75G_ARM, _MIXED_MEAL_DINNER_ARM),
    # Iter 97 (review 4.12): Cummings 2001 puts the postprandial NADIR at -30..
    # -50 % (60-90 min post-meal). This rule's post window is 15-60 min after the
    # meal — the descent, not the nadir — so its mean sits at roughly -20 %,
    # 20 pg/mL at a typical 100. 30 pg/mL was the nadir value scored on the
    # descent.
    predicate=lambda traj, ctx: hinge_min_drop(
        traj, ctx.col("ghrelin"),
        pre=ctx.window(30.0, 60.0),
        post=ctx.window(75.0, 120.0),
        min_drop=20.0,
    ),
    scale=30.0,
)


BHB_RISES_DURING_FAST = PhysiologyRule(
    name="bhb_rises_during_fast",
    source="Cahill (2006); Owen et al. (1969)",
    description=(
        "β-hydroxybutyrate rises by ≥ 0.05 mmol/L over an 8h→16h "
        "fast extension (onset of ketogenesis as hepatic glycogen "
        "depletes). Required across two fast-window arms — the "
        "monotone rise should hold over both 8h→16h and 12h→24h "
        "extensions."
    ),
    arms=(_FAST_8H_TO_16H_ARM, _FAST_12H_TO_24H_ARM),
    predicate=lambda traj, ctx: hinge_min_rise(
        traj, ctx.col("bhb"),
        pre=ctx.window(0.0, 30.0),
        # post window starts at duration-60min so it works for both
        # 480-min and 720-min arms; ctx.arm exposes the active arm.
        post=ctx.window(float(ctx.arm.duration_min) - 60.0, float(ctx.arm.duration_min)),
        min_rise=0.05,
    ),
    scale=0.05,
)


CORTISOL_MORNING_PEAK = PhysiologyRule(
    name="cortisol_morning_peak",
    source="Czeisler et al. (1999); Weitzman et al. (1971)",
    description=(
        "Plasma cortisol peaks between 06:00 and 09:00 in normally-"
        "entrained adults (cortisol awakening response on top of the "
        "circadian rise). Required under both no-meal and "
        "realistic-meal conditions — the morning peak should not be "
        "drowned out by postprandial cortisol spikes."
    ),
    arms=(_CIRCADIAN_24H_ARM, _CIRCADIAN_24H_WITH_MEALS_ARM),
    # Window: 0..24h trajectory, target 06:00–09:00 == minutes 360..540.
    predicate=lambda traj, ctx: hinge_argmax_in_band(
        traj, ctx.col("cortisol"),
        window=ctx.window(0.0, 24 * 60.0),
        target_min=360.0,
        target_max=540.0,
        step_min=ctx.step_min,
    ),
    scale=120.0,  # violation in minutes; 2h off peak ≈ loss 1
)


# Iter 65: replace iter 64's pair of independent argmax-in-band rules
# (acth_precedes_cortisol_morning_peak + cortisol_morning_peak) with a
# proper relational predicate. The iter-64 outcome (acth +0.067 Δ
# vs iter 61 w=1.0, but glp1 +1.33 and ghrelin +0.59 collateral) was
# consistent with the model satisfying both argmax bands at the
# overlap of the two windows (06:00–07:30) — two synchronous peaks
# trivially clear both rules. ``hinge_a_precedes_b`` computes the
# soft-argmax difference and penalises departures from a target lead
# interval, so synchronous peaks are now a violation rather than a
# solution. The cascade kinetics (Veldhuis 1990) put ACTH 15–30 min
# upstream of cortisol; this rule pins that with a 5–60 min lead
# tolerance (loose lower bound for noise headroom).
ACTH_PRECEDES_CORTISOL_RELATIONAL = PhysiologyRule(
    name="acth_precedes_cortisol_relational",
    source="Veldhuis et al. (1990); Weitzman et al. (1971)",
    description=(
        "Soft-argmax of ACTH leads cortisol's by 5–60 minutes within "
        "the 04:00–09:00 morning window — the canonical HPA pulsatile "
        "cascade. Replaces iter-64's pair of independent argmax-band "
        "rules, which the model could trivially satisfy by colocating "
        "both peaks at the band overlap. The relational predicate "
        "makes synchronous peaks an explicit violation."
    ),
    arms=(_CIRCADIAN_24H_ARM, _CIRCADIAN_24H_WITH_MEALS_ARM),
    predicate=lambda traj, ctx: hinge_a_precedes_b(
        traj, ctx.col("acth"), ctx.col("cortisol"),
        window=ctx.window(4 * 60.0, 9 * 60.0),
        min_lead_min=5.0,
        max_lead_min=60.0,
        step_min=ctx.step_min,
    ),
    scale=30.0,  # violation in minutes; 30 min off lead → loss 1
)


# Iter 64: postprandial insulin must rise meaningfully. iter 60's wider
# dim-64 code lets insulin track per-patient noise via embedding
# shortcut directions instead of through the glucose-driven Hill
# response the metabolic module is supposed to express. Min-rise floor
# is conservative — the literature typical first-phase rise is 30-60
# μU/mL — but 10 μU/mL rules out the "flat trajectory" regime, and
# forces the model to couple insulin to the meal-glucose surge at
# training time. The iter-61 spec R4 prescription was exactly this
# (force glucose→insulin Hill response shape via per-trajectory hinge).
INSULIN_RISES_POSTPRANDIAL = PhysiologyRule(
    name="insulin_rises_postprandial",
    source="Polonsky et al. (1988); standard OGTT physiology",
    description=(
        "Plasma insulin rises by ≥ 10 μU/mL within 60 min of a meal "
        "(first-phase secretion + sustained release). Required across "
        "mixed-meal and pure-carb (OGTT) and evening-meal arms — the "
        "rise should not depend on macronutrient mix or time of day."
    ),
    arms=(_MIXED_MEAL_BREAKFAST_ARM, _OGTT_75G_ARM, _MIXED_MEAL_DINNER_ARM),
    predicate=lambda traj, ctx: hinge_min_rise(
        traj, ctx.col("insulin"),
        pre=ctx.window(30.0, 60.0),
        post=ctx.window(75.0, 120.0),
        min_rise=10.0,
    ),
    scale=10.0,
)


# ---------------------------------------------------------------------------
# Iter 65: Move C expansion — additional coverage rules
# ---------------------------------------------------------------------------
# Direct attack on the iter-64 collateral damage (glp1 +1.33, ghrelin
# +0.59 vs iter 61 w=1.0). The wider 64-dim embedding lets these
# markers drift without supervision once the StressModule no longer
# needs them to encode HPA dynamics; tighter rules close the slack.
# Plus broad coverage on previously-unsupervised markers (glucose
# bounds, FFA fast-rise, BHB plateau, glycogen depletion, sbp > dbp).


GLP1_RETURNS_TO_BASELINE_POSTPRANDIAL = PhysiologyRule(
    name="glp1_returns_to_baseline_postprandial",
    source="Holst (2007) review; Drucker (2006)",
    description=(
        "Plasma GLP-1 returns toward baseline (≤ 15 pmol/L, ~1.5x "
        "typical 10 pmol/L) by 3 h post-meal. Anchors the postprandial "
        "tail so the rise-peak-decay shape cannot drift into a "
        "sustained elevation regime."
    ),
    arms=(_MIXED_MEAL_BREAKFAST_ARM, _OGTT_75G_ARM, _MIXED_MEAL_DINNER_ARM),
    predicate=lambda traj, ctx: hinge_max_value(
        traj, ctx.col("glp1"),
        window=ctx.window(240.0, 300.0),  # 3–4 h post-meal start (meal at t=60)
        ceiling=15.0,
    ),
    scale=10.0,
)


GLP1_PEAK_BOUNDED = PhysiologyRule(
    name="glp1_peak_bounded",
    source="Vilsbøll et al. (2003); standard postprandial GLP-1 kinetics",
    description=(
        "Plasma GLP-1 postprandial peak stays below 60 pmol/L — the "
        "upper clinical envelope for a 75g mixed/OGTT load. Caps the "
        "supraphysiological peaks the iter-60 wider embedding allowed."
    ),
    arms=(_MIXED_MEAL_BREAKFAST_ARM, _OGTT_75G_ARM, _MIXED_MEAL_DINNER_ARM),
    predicate=lambda traj, ctx: hinge_max_value(
        traj, ctx.col("glp1"),
        window=ctx.window(60.0, 180.0),  # 0–2 h after meal at t=60
        ceiling=60.0,
    ),
    scale=20.0,
)


GHRELIN_RISES_PRE_MEAL = PhysiologyRule(
    name="ghrelin_rises_pre_meal",
    source="Cummings et al. (2001, 2004)",
    description=(
        "Plasma ghrelin rises ≥ 20 pg/mL during the pre-meal hour as "
        "anticipatory hunger builds (clock-regulated and reinforced by "
        "habitual feeding times). Counterpart to the existing "
        "ghrelin_falls_after_meal — together they pin the meal-locked "
        "rise-fall pattern instead of just the fall."
    ),
    arms=(_MIXED_MEAL_BREAKFAST_ARM, _OGTT_75G_ARM, _MIXED_MEAL_DINNER_ARM),
    predicate=lambda traj, ctx: hinge_min_rise(
        traj, ctx.col("ghrelin"),
        pre=ctx.window(0.0, 20.0),
        post=ctx.window(40.0, 60.0),  # last 20 min before meal at t=60
        min_rise=20.0,
    ),
    scale=20.0,
    # Iter 97 (review 4.5): the teacher's fasting ghrelin is FLAT (100.0 at 12/24/
    # 36/48 h of fasting, review 3.2) so it cannot show the pre-meal rise
    # Cummings 2001 (Diabetes 50:1714) measured: +78 % from the post-meal nadir
    # over the 1-2 h before a habitual meal, i.e. well over 20 pg/mL. The rule
    # is right and the teacher is wrong; keep the pull. Re-audit after the
    # teacher's rectifier fix lands and drop this flag when it passes.
    teacher_correction=True,
    teacher_correction_note=(
        "teacher ghrelin has no anticipatory pre-meal rise (flat at basal while "
        "fasting, review 3.2); Cummings 2001 measures +78% over the pre-meal 1-2 h"
    ),
)


GLUCOSE_POSTPRANDIAL_BOUNDED = PhysiologyRule(
    name="glucose_postprandial_bounded",
    source="ADA criteria (2024); ISO 15197",
    description=(
        "Postprandial glucose stays below 180 mg/dL in non-diabetics "
        "after a 75g mixed-carb meal (or OGTT). Directly addresses the "
        "glucose_mape > 0.2 gate failure that has persisted across "
        "iters 60–64 — the model overshoots the postprandial peak."
    ),
    arms=(_MIXED_MEAL_BREAKFAST_ARM, _OGTT_75G_ARM, _MIXED_MEAL_DINNER_ARM),
    predicate=lambda traj, ctx: hinge_max_value(
        traj, ctx.col("glucose"),
        window=ctx.window(60.0, 240.0),  # 0–3 h after meal
        ceiling=180.0,
    ),
    scale=30.0,
)


GLUCOSE_FASTING_FLOOR = PhysiologyRule(
    name="glucose_fasting_floor",
    source="ADA criteria (2024); Cahill (2006)",
    description=(
        "Plasma glucose stays ≥ 65 mg/dL across a 16h fast — the "
        "physiological floor below which counter-regulation kicks in "
        "hard. Constrains the fasting end of the dynamic range."
    ),
    arms=(_FAST_8H_TO_16H_ARM, _FAST_12H_TO_24H_ARM),
    predicate=lambda traj, ctx: hinge_min_value(
        traj, ctx.col("glucose"),
        window=ctx.window(60.0, float(ctx.arm.duration_min)),
        floor=65.0,
    ),
    scale=15.0,
)


FFA_RISES_DURING_FAST = PhysiologyRule(
    name="ffa_rises_during_fast",
    source="Frayn (2003); Coppack et al. (1992)",
    description=(
        "Plasma FFA rises by ≥ 0.2 mmol/L over a fast extension "
        "(lipolysis kicks in as insulin falls). Complements the "
        "existing ffa_inverse_to_insulin rule by anchoring the "
        "magnitude and direction in a meal-free window."
    ),
    arms=(_FAST_8H_TO_16H_ARM, _FAST_12H_TO_24H_ARM),
    # Iter 97 (review 4.12): 0.2 mmol/L was uncited and the same for an 8 h and a
    # 12 h extension. Klein et al. 1993 (Am J Physiol 265:E801): plasma FFA
    # ~0.5 mmol/L at 12 h -> ~0.9 at 36 h, ~0.017 mmol/L per hour. The floor is
    # half that slope, 0.008/h (8 h -> 0.064, 12 h -> 0.096), so it rules out a
    # flat fast without demanding the population mean of every patient.
    predicate=lambda traj, ctx: hinge_min_rise(
        traj, ctx.col("ffa"),
        pre=ctx.window(0.0, 30.0),
        post=ctx.window(float(ctx.arm.duration_min) - 60.0, float(ctx.arm.duration_min)),
        min_rise=0.008 * float(ctx.arm.duration_min) / 60.0,
    ),
    scale=0.2,
)


BHB_PLATEAU_CAPPED = PhysiologyRule(
    name="bhb_plateau_capped",
    source="Cahill (2006); Owen et al. (1969)",
    description=(
        "β-hydroxybutyrate stays below 5 mmol/L during a 24h fast — "
        "the typical clinical ceiling before diabetic ketoacidosis "
        "regimes. Caps the ketogenesis trajectory so the model "
        "doesn't fit unbounded BHB rises."
    ),
    arms=(_FAST_8H_TO_16H_ARM, _FAST_12H_TO_24H_ARM),
    predicate=lambda traj, ctx: hinge_max_value(
        traj, ctx.col("bhb"),
        window=ctx.window(60.0, float(ctx.arm.duration_min)),
        ceiling=5.0,
    ),
    scale=1.0,
)


INSULIN_GLUCAGON_ANTAGONIST_POSTPRANDIAL = PhysiologyRule(
    name="insulin_glucagon_antagonist_postprandial",
    source="Unger & Orci (1981); Cherrington (1999)",
    description=(
        "Insulin and glucagon are anti-correlated postprandially "
        "(insulin rises with glucose, glucagon falls under alpha-cell "
        "suppression). Correlation ≤ −0.3 over 60–240 min post-meal. "
        "Forces the alpha-beta antagonism mechanistically rather than "
        "leaving it to per-marker fits."
    ),
    arms=(_MIXED_MEAL_BREAKFAST_ARM, _OGTT_75G_ARM, _MIXED_MEAL_DINNER_ARM),
    predicate=lambda traj, ctx: hinge_max_correlation(
        traj, ctx.col("insulin"), ctx.col("glucagon"),
        window=ctx.window(60.0, 240.0),
        max_corr=-0.3,
    ),
    scale=0.5,
)


LIVER_GLYCOGEN_DEPLETES_IN_FAST = PhysiologyRule(
    name="liver_glycogen_depletes_in_fast",
    source="Nilsson & Hultman (1973); Rothman et al. (1991)",
    description=(
        "Liver glycogen depletes by ≥ 30 g over a 12–24h fast (from "
        "typical ~100g, the overnight-depletable pool — Rothman MRS "
        "data). Complements the iter-56 cohort spec by giving the "
        "depletion a per-trajectory hinge rather than only a "
        "window-mean target."
    ),
    arms=(_FAST_12H_TO_24H_ARM,),
    predicate=lambda traj, ctx: hinge_monotone_decrease(
        traj, ctx.col("liver_glycogen"),
        window=ctx.window(0.0, float(ctx.arm.duration_min)),
        min_total_drop=30.0,
    ),
    scale=30.0,
)


SBP_ABOVE_DBP_ALWAYS = PhysiologyRule(
    name="sbp_above_dbp_always",
    source="Standard cardiovascular physiology (universal)",
    description=(
        "Systolic blood pressure exceeds diastolic by ≥ 10 mmHg at "
        "all times across all protocols. A trivial physiological "
        "constraint the model nominally satisfies but can violate at "
        "extreme embedding regions — locks the ordering in."
    ),
    arms=(_MIXED_MEAL_BREAKFAST_ARM, _CIRCADIAN_24H_WITH_MEALS_ARM, _FAST_8H_TO_16H_ARM),
    predicate=lambda traj, ctx: hinge_min_value(
        traj=traj[..., ctx.col("sbp")].unsqueeze(-1) - traj[..., ctx.col("dbp")].unsqueeze(-1),
        col=0,
        window=ctx.window(0.0, float(ctx.arm.duration_min)),
        floor=10.0,
    ),
    scale=20.0,
)


# ---------------------------------------------------------------------------
# Iter 66: Move C full — order-of-magnitude rule expansion (knowledge
# revolution). The compact iter 65 added 11 rules and confirmed the
# framework holds; iter 66 doubles down with 40+ new rules using the
# new helpers (hinge_min_correlation, hinge_circadian_amplitude,
# hinge_argmin_in_band, hinge_max_drift) and the three new cohort arms
# (prolonged-fast 24–48h, moderate exercise, sleep-wake 24h). Every
# rule is literature-grounded with citation and a single-axis hinge —
# no overlap with cohort-statistic specs (those supervise window-mean
# moments; rules supervise per-trajectory shape constraints).
# ---------------------------------------------------------------------------


# --- Metabolic axis ---

GLUCOSE_RISES_WITH_MEAL = PhysiologyRule(
    name="glucose_rises_with_meal",
    source="ADA criteria (2024); standard OGTT physiology",
    description=(
        "Plasma glucose rises ≥ 20 mg/dL within 60 min of a 75g "
        "carb-containing meal — the iter-66 counterpart to "
        "glucose_postprandial_bounded. Pins both ends of the "
        "excursion (rise floor + peak ceiling)."
    ),
    arms=(_MIXED_MEAL_BREAKFAST_ARM, _OGTT_75G_ARM, _MIXED_MEAL_DINNER_ARM),
    predicate=lambda traj, ctx: hinge_min_rise(
        traj, ctx.col("glucose"),
        pre=ctx.window(30.0, 60.0),
        post=ctx.window(75.0, 120.0),
        min_rise=20.0,
    ),
    scale=20.0,
)


GLUCOSE_RETURNS_BELOW_140_POSTPRANDIAL = PhysiologyRule(
    name="glucose_returns_below_140_postprandial",
    source="ADA criteria (2024); 75g OGTT 2-hr glucose criterion",
    description=(
        "Postprandial glucose returns below 140 mg/dL by 150 min "
        "post-load in normoglycaemic adults (the 75g OGTT diagnostic "
        "criterion). Anchors the postprandial tail."
    ),
    arms=(_OGTT_75G_ARM, _MIXED_MEAL_BREAKFAST_ARM),
    predicate=lambda traj, ctx: hinge_max_value(
        traj, ctx.col("glucose"),
        window=ctx.window(180.0, 240.0),  # 2-3 h post meal at t=60
        ceiling=140.0,
    ),
    scale=20.0,
)


GLUCOSE_OVERNIGHT_STABILITY = PhysiologyRule(
    name="glucose_overnight_stability",
    source="Polonsky et al. (1988); standard CGM-derived nocturnal data",
    description=(
        "Overnight (fasted) glucose stays within a 25 mg/dL range "
        "across the 8-16h fast window — the dawn phenomenon and "
        "minor hepatic regulation give some variation but not large "
        "swings. Constrains the basal regime stability."
    ),
    arms=(_FAST_8H_TO_16H_ARM,),
    predicate=lambda traj, ctx: hinge_max_drift(
        traj, ctx.col("glucose"),
        window=ctx.window(0.0, float(ctx.arm.duration_min)),
        max_drift=25.0,
    ),
    scale=15.0,
)


INSULIN_FIRST_PHASE_RISE = PhysiologyRule(
    name="insulin_first_phase_rise",
    source="Cerasi & Luft (1967); Polonsky et al. (1988)",
    description=(
        "First-phase insulin secretion: insulin rises ≥ 5 μU/mL "
        "within 30 minutes of a 75g carb load (the rapid β-cell "
        "exocytosis response). Tighter timing predicate than the "
        "iter-64 insulin_rises rule (which uses 60 min)."
    ),
    arms=(_OGTT_75G_ARM, _MIXED_MEAL_BREAKFAST_ARM),
    predicate=lambda traj, ctx: hinge_min_rise(
        traj, ctx.col("insulin"),
        pre=ctx.window(30.0, 60.0),
        post=ctx.window(75.0, 90.0),
        min_rise=5.0,
    ),
    scale=5.0,
)


INSULIN_RETURNS_TO_BASAL_POSTPRANDIAL = PhysiologyRule(
    name="insulin_returns_to_basal_postprandial",
    source="Polonsky et al. (1988); Vilsbøll et al. (2003)",
    description=(
        "Plasma insulin returns to ≤ 15 μU/mL by 240 min post-load "
        "(~1.5x basal). Anchors the tail of the rise-peak-decay "
        "envelope so post-meal insulin doesn't drift into "
        "hyperinsulinaemic territory."
    ),
    arms=(_MIXED_MEAL_BREAKFAST_ARM, _OGTT_75G_ARM, _MIXED_MEAL_DINNER_ARM),
    predicate=lambda traj, ctx: hinge_max_value(
        traj, ctx.col("insulin"),
        window=ctx.window(240.0, 300.0),
        ceiling=15.0,
    ),
    scale=10.0,
)


INSULIN_FALLS_DURING_FAST = PhysiologyRule(
    name="insulin_falls_during_fast",
    source="Cahill (2006); Owen et al. (1969)",
    description=(
        "Plasma insulin drops to ≤ 8 μU/mL by 12 h fasted "
        "(suppression as β-cells respond to falling glucose & "
        "rising counter-regulatory tone)."
    ),
    arms=(_FAST_12H_TO_24H_ARM, _FAST_24H_TO_48H_ARM),
    # Iter 97 (review 4.12): the ceiling of 8 uU/mL sat at the population MEAN
    # of fasting insulin and was applied to the WHOLE 12-24 h window, whose first
    # hour is the 12 h-fasted basal (Ib = 10 in the teacher). Healthy fasting
    # insulin: mean ~6-8, upper reference ~12 uU/mL (Laakso 1993, Am J
    # Epidemiol 137:959; HOMA reference cohorts). The rule now excludes the
    # hyperinsulinaemic regime rather than half the healthy cohort.
    predicate=lambda traj, ctx: hinge_max_value(
        traj, ctx.col("insulin"),
        window=ctx.window(0.0, float(ctx.arm.duration_min)),
        ceiling=12.0,
    ),
    scale=5.0,
)


GLUCAGON_RISES_DURING_FAST = PhysiologyRule(
    name="glucagon_rises_during_fast",
    source="Unger & Orci (1981); Cherrington (1999)",
    description=(
        "Plasma glucagon rises by ≥ 10 pg/mL over a 16h fast "
        "extension — counter-regulatory α-cell activation as "
        "insulin falls. The mirror of glucagon_falls_postprandial."
    ),
    arms=(_FAST_8H_TO_16H_ARM, _FAST_12H_TO_24H_ARM),
    # Iter 97 (review 4.12): the flat 10 pg/mL was an uncited round number
    # applied to both an 8 h and a 12 h extension. Marliss et al. 1970 (J Clin
    # Invest 49:2256): glucagon 108 -> 158 pg/mL over a 3-day fast, i.e.
    # ~0.7 pg/mL per fasted hour; Aguilar-Parada 1969 (+50 % by 48-72 h) gives
    # ~0.5/h. The floor is 0.5 pg/mL per hour of the arm (8 h -> 4, 12 h -> 6).
    predicate=lambda traj, ctx: hinge_min_rise(
        traj, ctx.col("glucagon"),
        pre=ctx.window(0.0, 30.0),
        post=ctx.window(float(ctx.arm.duration_min) - 60.0, float(ctx.arm.duration_min)),
        min_rise=0.5 * float(ctx.arm.duration_min) / 60.0,
    ),
    scale=10.0,
    # The teacher's fasting glucagon rise is ~0.25 pg/mL/h (2.1 over 8 h, 3.5 over
    # 12 h) — half the Marliss/Aguilar-Parada slope. Deliberate correction; re-audit
    # after the teacher's counter-regulation work lands.
    teacher_correction=True,
    teacher_correction_note=(
        "teacher fasting glucagon rises ~0.25 pg/mL/h vs the cited 0.5-0.7/h "
        "(Marliss 1970; Aguilar-Parada 1969)"
    ),
)


FFA_FALLS_POSTPRANDIAL = PhysiologyRule(
    name="ffa_falls_postprandial",
    source="Frayn (2003); Boden (1996) antilipolysis review",
    description=(
        "Plasma FFA drops ≥ 0.1 mmol/L within 60 min of a "
        "carbohydrate-containing meal (insulin-driven anti-lipolysis "
        "on the adipocyte). The direction counterpart to "
        "ffa_rises_during_fast."
    ),
    arms=(_MIXED_MEAL_BREAKFAST_ARM, _OGTT_75G_ARM),
    predicate=lambda traj, ctx: hinge_min_drop(
        traj, ctx.col("ffa"),
        pre=ctx.window(30.0, 60.0),
        post=ctx.window(90.0, 150.0),
        min_drop=0.1,
    ),
    scale=0.1,
)


BHB_RETURNS_TO_BASELINE_POSTPRANDIAL = PhysiologyRule(
    name="bhb_returns_to_baseline_postprandial",
    source="Cahill (2006); Owen et al. (1969)",
    description=(
        "Plasma β-hydroxybutyrate drops to ≤ 0.2 mmol/L by 240 min "
        "after a mixed meal (insulin suppression of ketogenesis). "
        "Caps any post-meal ketone elevation."
    ),
    arms=(_MIXED_MEAL_BREAKFAST_ARM, _OGTT_75G_ARM),
    predicate=lambda traj, ctx: hinge_max_value(
        traj, ctx.col("bhb"),
        window=ctx.window(180.0, 300.0),
        ceiling=0.2,
    ),
    scale=0.2,
)


HEPATIC_OUTPUT_INVERSE_TO_INSULIN = PhysiologyRule(
    name="hepatic_output_inverse_to_insulin",
    source="Cherrington (1999); DeFronzo et al. (1982)",
    description=(
        "Endogenous hepatic glucose output is anti-correlated with "
        "insulin over the postprandial window (insulin suppresses "
        "gluconeogenesis + glycogenolysis). Corr ≤ -0.3 over "
        "60–240 min post-meal."
    ),
    arms=(_MIXED_MEAL_BREAKFAST_ARM, _OGTT_75G_ARM, _MIXED_MEAL_DINNER_ARM),
    predicate=lambda traj, ctx: hinge_max_correlation(
        traj, ctx.col("hepatic_output"), ctx.col("insulin"),
        window=ctx.window(60.0, 240.0),
        max_corr=-0.3,
    ),
    scale=0.5,
)


# --- Exercise / lactate / muscle axis (uses _MODERATE_EXERCISE_ARM) ---

LACTATE_RISES_WITH_EXERCISE = PhysiologyRule(
    name="lactate_rises_with_exercise",
    source="Brooks (1985); Robergs et al. (2004)",
    description=(
        "Plasma lactate rises ≥ 1 mmol/L during a 60-min "
        "moderate-intensity bout (anaerobic glycolysis component "
        "even at submaximal intensity). Tests the exercise→lactate "
        "coupling the current architecture has no supervision for."
    ),
    arms=(_MODERATE_EXERCISE_ARM,),
    predicate=lambda traj, ctx: hinge_min_rise(
        traj, ctx.col("lactate"),
        pre=ctx.window(0.0, 30.0),  # pre-exercise rest
        post=ctx.window(60.0, 90.0),  # mid-exercise
        min_rise=1.0,
    ),
    scale=1.0,
)


LACTATE_RETURNS_POST_EXERCISE = PhysiologyRule(
    name="lactate_returns_post_exercise",
    source="Brooks (1985); standard recovery physiology",
    description=(
        "Plasma lactate returns toward baseline within 60 min after "
        "exercise stops (drops back below 2 mmol/L during the "
        "60-min recovery window). Caps the post-exercise tail."
    ),
    arms=(_MODERATE_EXERCISE_ARM,),
    predicate=lambda traj, ctx: hinge_max_value(
        traj, ctx.col("lactate"),
        window=ctx.window(150.0, 180.0),  # last 30 min of recovery
        ceiling=2.0,
    ),
    scale=1.0,
)


FFA_RISES_WITH_EXERCISE = PhysiologyRule(
    name="ffa_rises_with_exercise",
    source="Romijn et al. (1993); Coppack et al. (1992)",
    description=(
        "Plasma FFA rises ≥ 0.2 mmol/L during moderate exercise "
        "(catecholamine-driven lipolysis as energy demand outpaces "
        "stored carb)."
    ),
    arms=(_MODERATE_EXERCISE_ARM,),
    # Iter 97 (review 4.12): Romijn et al. 1993 (Am J Physiol 265:E380): at 65 %
    # VO2max plasma FFA rises only modestly in the first hour (from ~0.4 to
    # ~0.5-0.6 mmol/L; the large rise is at 25 % and after 2 h). 0.2 was the
    # 2 h low-intensity value; the 60-min moderate-bout floor is 0.1.
    predicate=lambda traj, ctx: hinge_min_rise(
        traj, ctx.col("ffa"),
        pre=ctx.window(0.0, 30.0),
        post=ctx.window(60.0, 90.0),
        min_rise=0.1,
    ),
    scale=0.2,
)


MUSCLE_GLYCOGEN_FALLS_WITH_EXERCISE = PhysiologyRule(
    name="muscle_glycogen_falls_with_exercise",
    source="Coyle (1991); Hultman & Bergström (1967)",
    description=(
        "Muscle glycogen drops ≥ 10g during a 60-min moderate bout "
        "(direct glycolytic consumption). Tests that the iter-57 "
        "GlycogenFluxHead's activity-gated catabolic term actually "
        "fires under the exercise arm."
    ),
    arms=(_MODERATE_EXERCISE_ARM,),
    predicate=lambda traj, ctx: hinge_min_drop(
        traj, ctx.col("muscle_glycogen"),
        pre=ctx.window(0.0, 30.0),
        post=ctx.window(60.0, 90.0),
        min_drop=10.0,
    ),
    scale=10.0,
)


# --- Appetite axis (ghrelin / leptin / glp1) ---

GHRELIN_EVENING_PEAK = PhysiologyRule(
    name="ghrelin_evening_peak",
    source="Cummings et al. (2001, 2004); Drazen et al. (2006)",
    description=(
        "Plasma ghrelin shows a circadian peak in the evening "
        "(approx 18:00–22:00) — a learned, meal-anticipatory peak "
        "before habitual dinner timing."
    ),
    arms=(_CIRCADIAN_24H_WITH_MEALS_ARM, _SLEEP_WAKE_24H_ARM),
    # ITER 96 — RE-WINDOWED from the full 24 h to the afternoon/evening.
    # The rule describes a "meal-anticipatory peak before habitual dinner
    # timing", which is a LOCAL maximum. Ghrelin's GLOBAL 24 h maximum is the
    # nocturnal peak around 01:00 (Cummings et al. 2001, Diabetes 50:1714), so a
    # 24 h argmax could never land in 18:00-22:00 no matter what the model did.
    # This was invisible while the soft-argmax was reporting the window centroid
    # (~12:00) for every marker; with the range-normalized sharpness the rule
    # would now report an 819-minute standing violation on a trajectory that has
    # a perfectly good pre-dinner peak. Window 14:00-23:00 makes the described
    # peak the window's own maximum; the band is relative to the window start.
    predicate=lambda traj, ctx: hinge_argmax_in_band(
        traj, ctx.col("ghrelin"),
        window=ctx.window(14 * 60.0, 23 * 60.0),
        target_min=4 * 60.0,   # 18:00
        target_max=8 * 60.0,   # 22:00
        step_min=ctx.step_min,
    ),
    scale=120.0,
)


LEPTIN_EVENING_PEAK = PhysiologyRule(
    name="leptin_evening_peak",
    source="Sinha et al. (1996); Saad et al. (1998)",
    description=(
        "Plasma leptin peaks at 22:00–02:00 local (the canonical "
        "night-time peak of the leptin diurnal rhythm). Currently "
        "leptin has no supervision rules; this is the load-bearing "
        "one for its phase."
    ),
    arms=(_CIRCADIAN_24H_WITH_MEALS_ARM, _SLEEP_WAKE_24H_ARM),
    # ITER 96 — the old band was `target_max = 26 h` on a window only 24 h long,
    # so a quarter of the stated band lay outside the trajectory and the rule
    # could only ever score the 22:00-24:00 half. The nocturnal leptin peak
    # straddles midnight, and these arms start at 00:00, so the peak sits on the
    # window BOUNDARY — the one place an argmax cannot be measured. Windowing to
    # 14:00-24:00 asks the reachable half of the claim ("leptin is still climbing
    # into the late evening") of a trajectory that can actually express it.
    predicate=lambda traj, ctx: hinge_argmax_in_band(
        traj, ctx.col("leptin"),
        window=ctx.window(14 * 60.0, 24 * 60.0),
        target_min=8 * 60.0,   # 22:00
        target_max=10 * 60.0,  # 24:00
        step_min=ctx.step_min,
    ),
    scale=120.0,
)


LEPTIN_STABLE_OVER_MEAL = PhysiologyRule(
    name="leptin_stable_over_meal",
    source="Considine et al. (1996); standard leptin kinetics",
    description=(
        "Plasma leptin drifts by < 3 ng/mL over a single 5-h "
        "postprandial window — leptin is a slow chronic-state "
        "marker, not a meal-response marker. Caps fast drift."
    ),
    arms=(_MIXED_MEAL_BREAKFAST_ARM, _MIXED_MEAL_DINNER_ARM),
    # Iter 97 (review 4.12): the dinner arm runs 19:00-24:00, INSIDE the
    # nocturnal leptin rise this file's own leptin_evening_peak demands. Sinha
    # et al. 1996 (J Clin Invest 97:1344): nocturnal amplitude ~30-50 % of the
    # daytime level, i.e. 3-5 ng/mL at a typical 10. A 3 ng/mL cap contradicted
    # the rhythm rule; 5 caps meal-driven fast drift without forbidding it.
    predicate=lambda traj, ctx: hinge_max_drift(
        traj, ctx.col("leptin"),
        window=ctx.window(0.0, float(ctx.arm.duration_min)),
        max_drift=5.0,
    ),
    scale=3.0,
)


GLP1_RISES_EARLY_POSTPRANDIAL = PhysiologyRule(
    name="glp1_rises_early_postprandial",
    source="Holst (2007); Vilsbøll et al. (2003)",
    description=(
        "GLP-1 rises ≥ 5 pmol/L within the first 30 min after a "
        "carb-containing meal — L-cell secretion kicks in early. "
        "Tighter than the existing peak-bounded rule (which caps "
        "the magnitude); this pins the timing."
    ),
    arms=(_MIXED_MEAL_BREAKFAST_ARM, _OGTT_75G_ARM),
    predicate=lambda traj, ctx: hinge_min_rise(
        traj, ctx.col("glp1"),
        pre=ctx.window(30.0, 60.0),
        post=ctx.window(75.0, 90.0),
        min_rise=5.0,
    ),
    scale=5.0,
)


GLP1_INSULIN_CORRELATION_POSTPRANDIAL = PhysiologyRule(
    name="glp1_insulin_correlation_postprandial",
    source="Drucker (2006) incretin review; Nauck et al. (1986)",
    description=(
        "Postprandial GLP-1 and insulin are positively correlated "
        "(incretin effect: ~50% of post-meal insulin is GLP-1 / "
        "GIP-driven). Corr ≥ +0.3 over 60–240 min post-meal."
    ),
    arms=(_MIXED_MEAL_BREAKFAST_ARM, _OGTT_75G_ARM),
    predicate=lambda traj, ctx: hinge_min_correlation(
        traj, ctx.col("glp1"), ctx.col("insulin"),
        window=ctx.window(60.0, 240.0),
        min_corr=0.3,
    ),
    scale=0.5,
)


# --- Stress / HPA axis (cortisol / acth) ---

CORTISOL_EVENING_TROUGH = PhysiologyRule(
    name="cortisol_evening_trough",
    source="Czeisler et al. (1999); standard cortisol diurnal data",
    description=(
        "Plasma cortisol soft-min over a 24h window lands at 22:00–"
        "02:00 (the canonical evening trough). Companion to "
        "cortisol_morning_peak — together they pin both ends of "
        "the diurnal swing."
    ),
    arms=(_CIRCADIAN_24H_ARM, _SLEEP_WAKE_24H_ARM),
    # Iter 97 (review 4.6): the band ran to 26 h on a 24 h window — half of it
    # was unreachable, and the trough straddles the arm boundary (these arms
    # start at 00:00). Same fix as leptin in iter 96: window the reachable half
    # of the claim, "cortisol is still falling into the late evening", so the
    # soft-argmin over 16:00-24:00 must land in 22:00-24:00.
    predicate=lambda traj, ctx: hinge_argmin_in_band(
        traj, ctx.col("cortisol"),
        window=ctx.window(16 * 60.0, 24 * 60.0),
        target_min=6 * 60.0,   # 22:00
        target_max=8 * 60.0,   # 24:00
        step_min=ctx.step_min,
    ),
    scale=120.0,
)


CORTISOL_CIRCADIAN_AMPLITUDE = PhysiologyRule(
    name="cortisol_circadian_amplitude",
    source="Czeisler et al. (1999); Weitzman et al. (1971)",
    description=(
        "Plasma cortisol shows a daily peak-to-trough amplitude of "
        "≥ 5 μg/dL (the diurnal swing is large — morning ~18 vs "
        "evening ~3 in healthy adults). Forces the model to "
        "actually express the circadian dynamics rather than "
        "flattening cortisol near typical."
    ),
    arms=(_CIRCADIAN_24H_ARM, _SLEEP_WAKE_24H_ARM),
    predicate=lambda traj, ctx: hinge_circadian_amplitude(
        traj, ctx.col("cortisol"),
        window=ctx.window(0.0, 24 * 60.0),
        min_amplitude=5.0,
    ),
    scale=5.0,
)


CORTISOL_RISES_WITH_EXERCISE = PhysiologyRule(
    name="cortisol_rises_with_exercise",
    source="Hill et al. (2008); Wahl et al. (2010)",
    description=(
        "Plasma cortisol rises ≥ 2 μg/dL during a 60-min moderate "
        "bout (acute HPA activation by physical stress)."
    ),
    arms=(_MODERATE_EXERCISE_ARM,),
    # Iter 97 (review 4.12): Hill et al. 2008 (J Endocrinol Invest 31:587):
    # cortisol rises at >= 60 % VO2max and does NOT rise at 40 %. The arm's
    # activity 0.7 is ~50-60 % — the threshold intensity — so a 2 ug/dL rise
    # (the 60 % mean) is not a lower bound there; 1 ug/dL is.
    predicate=lambda traj, ctx: hinge_min_rise(
        traj, ctx.col("cortisol"),
        pre=ctx.window(0.0, 30.0),
        post=ctx.window(75.0, 105.0),
        min_rise=1.0,
    ),
    scale=2.0,
)


ACTH_EVENING_TROUGH = PhysiologyRule(
    name="acth_evening_trough",
    source="Veldhuis et al. (1990); Weitzman et al. (1971)",
    description=(
        "Plasma ACTH soft-min in 22:00–02:00 (the diurnal nadir, "
        "anticipating cortisol's slightly later trough by the same "
        "cascade kinetics that put ACTH's morning peak ahead of "
        "cortisol's)."
    ),
    arms=(_CIRCADIAN_24H_ARM, _SLEEP_WAKE_24H_ARM),
    # Iter 97 (review 4.6): same 26 h-band fix as cortisol_evening_trough.
    predicate=lambda traj, ctx: hinge_argmin_in_band(
        traj, ctx.col("acth"),
        window=ctx.window(16 * 60.0, 24 * 60.0),
        target_min=6 * 60.0,   # 22:00
        target_max=8 * 60.0,   # 24:00
        step_min=ctx.step_min,
    ),
    scale=120.0,
)


ACTH_CIRCADIAN_AMPLITUDE = PhysiologyRule(
    name="acth_circadian_amplitude",
    source="Veldhuis et al. (1990)",
    description=(
        "Plasma ACTH peak-to-trough amplitude ≥ 15 pg/mL over 24h "
        "(morning ~50–60 vs evening ~10–15 in healthy adults)."
    ),
    arms=(_CIRCADIAN_24H_ARM, _SLEEP_WAKE_24H_ARM),
    predicate=lambda traj, ctx: hinge_circadian_amplitude(
        traj, ctx.col("acth"),
        window=ctx.window(0.0, 24 * 60.0),
        min_amplitude=15.0,
    ),
    scale=15.0,
)


# --- Cardiovascular axis (hr / hrv / sbp / dbp) ---

HR_RISES_WITH_EXERCISE = PhysiologyRule(
    name="hr_rises_with_exercise",
    source="Robergs & Landwehr (2002); ACSM exercise guidelines",
    description=(
        "Heart rate rises ≥ 30 bpm during a 60-min moderate bout "
        "(submaximal ~50–60% VO2max → HR ~110–130 from resting "
        "70). First HR rule in the codebase — the iter-60 0.13 "
        "MAPE is good but only because resting HR fits well; the "
        "exercise response was never under supervision."
    ),
    arms=(_MODERATE_EXERCISE_ARM,),
    predicate=lambda traj, ctx: hinge_min_rise(
        traj, ctx.col("hr"),
        pre=ctx.window(0.0, 30.0),
        post=ctx.window(60.0, 90.0),
        min_rise=30.0,
    ),
    scale=30.0,
)


HR_RECOVERY_POST_EXERCISE = PhysiologyRule(
    name="hr_recovery_post_exercise",
    source="Cole et al. (1999); Imai et al. (1994)",
    description=(
        "Heart rate drops ≥ 20 bpm within 30 min of stopping "
        "moderate exercise — the vagal-reactivation recovery "
        "curve, a clinical prognostic marker."
    ),
    arms=(_MODERATE_EXERCISE_ARM,),
    predicate=lambda traj, ctx: hinge_min_drop(
        traj, ctx.col("hr"),
        pre=ctx.window(75.0, 90.0),  # last 15 min of bout
        post=ctx.window(120.0, 150.0),  # 30–60 min into recovery
        min_drop=20.0,
    ),
    scale=20.0,
)


HR_FALLS_DURING_SLEEP = PhysiologyRule(
    name="hr_falls_during_sleep",
    source="Burgess et al. (1997); standard polysomnographic data",
    description=(
        "Heart rate is ≥ 10 bpm lower during sleep than during the "
        "daytime wake period — parasympathetic dominance during "
        "sleep. Tests the sleep_wake input is wired into HR "
        "dynamics."
    ),
    arms=(_SLEEP_WAKE_24H_ARM,),
    predicate=lambda traj, ctx: hinge_min_drop(
        traj, ctx.col("hr"),
        pre=ctx.window(15 * 60.0, 18 * 60.0),  # daytime 15:00–18:00
        post=ctx.window(2 * 60.0, 5 * 60.0),  # deep sleep 02:00–05:00
        min_drop=10.0,
    ),
    scale=10.0,
)


HRV_HIGHER_DURING_SLEEP = PhysiologyRule(
    name="hrv_higher_during_sleep",
    source="Trinder et al. (2001); standard HRV-sleep data",
    description=(
        "Heart rate variability (RMSSD) is ≥ 10 ms higher during "
        "sleep than daytime wake (vagal-mediated HRV rises with "
        "parasympathetic activity). First HRV rule."
    ),
    arms=(_SLEEP_WAKE_24H_ARM,),
    predicate=lambda traj, ctx: hinge_min_rise(
        traj, ctx.col("hrv"),
        pre=ctx.window(15 * 60.0, 18 * 60.0),
        post=ctx.window(2 * 60.0, 5 * 60.0),
        min_rise=10.0,
    ),
    scale=10.0,
)


HRV_INVERSE_TO_HR = PhysiologyRule(
    name="hrv_inverse_to_hr",
    source="Task Force ESC/NASPE (1996) standards",
    description=(
        "HRV (RMSSD) and HR are anti-correlated across diurnal "
        "fluctuation (parasympathetic-vagal autonomic balance: "
        "when HR is low, HRV is high). Corr ≤ -0.4 over 24h."
    ),
    arms=(_SLEEP_WAKE_24H_ARM, _CIRCADIAN_24H_WITH_MEALS_ARM),
    predicate=lambda traj, ctx: hinge_max_correlation(
        traj, ctx.col("hrv"), ctx.col("hr"),
        window=ctx.window(0.0, 24 * 60.0),
        max_corr=-0.4,
    ),
    scale=0.5,
)


SBP_RISES_WITH_EXERCISE = PhysiologyRule(
    name="sbp_rises_with_exercise",
    source="Lim et al. (1996); ACSM exercise testing guidelines",
    description=(
        "Systolic BP rises ≥ 15 mmHg during a 60-min moderate "
        "bout (cardiac output increase against modestly increased "
        "vascular resistance). Diastolic stays approximately flat."
    ),
    arms=(_MODERATE_EXERCISE_ARM,),
    predicate=lambda traj, ctx: hinge_min_rise(
        traj, ctx.col("sbp"),
        pre=ctx.window(0.0, 30.0),
        post=ctx.window(60.0, 90.0),
        min_rise=15.0,
    ),
    scale=15.0,
)


SBP_MORNING_SURGE = PhysiologyRule(
    name="sbp_morning_surge",
    source="Kario et al. (2003) Circulation 107:1401 — sleep-trough morning surge",
    description=(
        "Systolic BP shows a morning surge: the mean over the two hours after "
        "waking (06:30–09:30) exceeds the lowest night hours (02:00–05:00) by "
        "≥ 8 mmHg (sympathetic activation on waking)."
    ),
    arms=(_SLEEP_WAKE_24H_ARM,),
    # Iter 97 (review 4.12 / 1.2): the old predicate demanded that the 24 h
    # ARGMAX of SBP fall at 06:00-10:00. Kario's morning surge is a RISE relative
    # to the sleep trough (sleep-trough surge = post-wake 2 h mean minus lowest
    # night hour; non-surge group mean 17 mmHg, surge >= 35), not the daily
    # maximum, which ambulatory monitoring puts in the afternoon. Measured on the
    # corrected sleep arm the teacher's SBP argmax is ~12:00 (a 124-min
    # "violation") while its trough-to-morning rise is +15.6 mmHg — the old rule
    # penalised correct physiology. Floor 8 sits under the non-surge mean.
    predicate=lambda traj, ctx: hinge_min_rise(
        traj, ctx.col("sbp"),
        pre=ctx.window(2 * 60.0, 5 * 60.0),
        post=ctx.window(6.5 * 60.0, 9.5 * 60.0),
        min_rise=8.0,
    ),
    scale=10.0,
)


SBP_FALLS_DURING_SLEEP = PhysiologyRule(
    name="sbp_falls_during_sleep",
    source="O'Brien et al. (1988); standard ambulatory BP",
    description=(
        "Systolic BP drops ≥ 10 mmHg during sleep vs daytime wake "
        "(the 'nocturnal dip', ~10–20% drop in normotensives)."
    ),
    arms=(_SLEEP_WAKE_24H_ARM,),
    predicate=lambda traj, ctx: hinge_min_drop(
        traj, ctx.col("sbp"),
        pre=ctx.window(15 * 60.0, 18 * 60.0),
        post=ctx.window(2 * 60.0, 5 * 60.0),
        min_drop=10.0,
    ),
    scale=10.0,
)


# --- Thermoregulation axis (temp) ---

TEMP_CIRCADIAN_AMPLITUDE = PhysiologyRule(
    name="temp_circadian_amplitude",
    source="Czeisler et al. (1999); Refinetti & Menaker (1992)",
    description=(
        "Core body temperature swings 0.5–0.8 °C peak-to-trough over a day "
        "(the canonical 36.5 → 37.2 swing). Both bounds are supervised: the "
        "rhythm must exist AND must not be exaggerated."
    ),
    # Iter 94: this was `min_amplitude=0.5` alone — a one-sided floor that made
    # any overshoot free. The teacher had drifted to 1.04 °C, ~2x literature, and
    # the student inherited it to within 6 %; nothing in training ever objected.
    # The ceiling is the upper end of the same Czeisler/Refinetti range the floor
    # comes from, so this adds no specificity we do not already cite.
    arms=(_CIRCADIAN_24H_ARM, _SLEEP_WAKE_24H_ARM),
    predicate=lambda traj, ctx: hinge_circadian_amplitude(
        traj, ctx.col("temp"),
        window=ctx.window(0.0, 24 * 60.0),
        min_amplitude=0.5,
        max_amplitude=0.8,
    ),
    scale=0.5,
)


TEMP_LATE_AFTERNOON_PEAK = PhysiologyRule(
    name="temp_late_afternoon_peak",
    source="Refinetti & Menaker (1992); standard circadian temp data",
    description=(
        "Core body temperature peaks 16:00–19:00 (the late-"
        "afternoon temperature high before evening decline)."
    ),
    arms=(_CIRCADIAN_24H_ARM, _SLEEP_WAKE_24H_ARM),
    predicate=lambda traj, ctx: hinge_argmax_in_band(
        traj, ctx.col("temp"),
        window=ctx.window(0.0, 24 * 60.0),
        target_min=16 * 60.0,
        target_max=19 * 60.0,
        step_min=ctx.step_min,
    ),
    scale=120.0,
)


TEMP_PRE_DAWN_TROUGH = PhysiologyRule(
    name="temp_pre_dawn_trough",
    source="Refinetti & Menaker (1992)",
    description=(
        "Core body temperature reaches its diurnal trough at "
        "04:00–06:00 (the pre-dawn nadir, ~36.0–36.4 °C in "
        "healthy adults)."
    ),
    arms=(_CIRCADIAN_24H_ARM, _SLEEP_WAKE_24H_ARM),
    predicate=lambda traj, ctx: hinge_argmin_in_band(
        traj, ctx.col("temp"),
        window=ctx.window(0.0, 24 * 60.0),
        target_min=4 * 60.0,
        target_max=6 * 60.0,
        step_min=ctx.step_min,
    ),
    scale=120.0,
)


# --- Respiratory axis (rr / spo2) ---

RR_RISES_WITH_EXERCISE = PhysiologyRule(
    name="rr_rises_with_exercise",
    source="ACSM exercise guidelines; standard exercise pulmonary data",
    description=(
        "Respiratory rate rises ≥ 5 /min during a 60-min "
        "moderate-intensity bout (CO2 clearance + drive from "
        "central pattern generators in response to metabolic "
        "demand)."
    ),
    arms=(_MODERATE_EXERCISE_ARM,),
    predicate=lambda traj, ctx: hinge_min_rise(
        traj, ctx.col("rr"),
        pre=ctx.window(0.0, 30.0),
        post=ctx.window(60.0, 90.0),
        min_rise=5.0,
    ),
    scale=5.0,
)


SPO2_STABLE_AT_REST = PhysiologyRule(
    name="spo2_stable_at_rest",
    source="standard pulse oximetry data; ATS clinical practice guidelines",
    description=(
        "SpO2 drifts less than 3% across a resting / fasted "
        "protocol — pulse oximetry is a tight homeostatic signal "
        "in healthy adults absent respiratory pathology."
    ),
    arms=(_FAST_8H_TO_16H_ARM, _CIRCADIAN_24H_ARM),
    predicate=lambda traj, ctx: hinge_max_drift(
        traj, ctx.col("spo2"),
        window=ctx.window(0.0, float(ctx.arm.duration_min)),
        max_drift=3.0,
    ),
    scale=2.0,
)


# --- Prolonged fast continuation (24-48h, _FAST_24H_TO_48H_ARM) ---

BHB_CONTINUES_RISING_24_48H = PhysiologyRule(
    name="bhb_continues_rising_24_48h",
    source="Cahill (1970); Owen et al. (1969)",
    description=(
        "β-hydroxybutyrate continues to climb during a 24–48h "
        "prolonged fast (ketogenesis ramps as ketone-adaptation "
        "engages and glycogen is fully depleted). Rises ≥ 0.5 "
        "mmol/L across the 24h window."
    ),
    arms=(_FAST_24H_TO_48H_ARM,),
    predicate=lambda traj, ctx: hinge_min_rise(
        traj, ctx.col("bhb"),
        pre=ctx.window(0.0, 60.0),
        post=ctx.window(float(ctx.arm.duration_min) - 60.0, float(ctx.arm.duration_min)),
        min_rise=0.5,
    ),
    scale=0.5,
)


FFA_SUSTAINED_PROLONGED_FAST = PhysiologyRule(
    name="ffa_sustained_prolonged_fast",
    source="Cahill (1970); Coppack et al. (1992)",
    description=(
        "FFA stays ≥ 0.6 mmol/L throughout 24–48h of fasting "
        "(sustained lipolysis as the energy substrate). Floor on "
        "the prolonged-fast FFA trajectory."
    ),
    arms=(_FAST_24H_TO_48H_ARM,),
    predicate=lambda traj, ctx: hinge_min_value(
        traj, ctx.col("ffa"),
        window=ctx.window(60.0, float(ctx.arm.duration_min)),
        floor=0.6,
    ),
    scale=0.2,
)


LIVER_GLYCOGEN_DEPLETED_PROLONGED_FAST = PhysiologyRule(
    name="liver_glycogen_depleted_prolonged_fast",
    source="Rothman et al. (1991); Nilsson & Hultman (1973)",
    description=(
        "Hepatic glycogen is essentially exhausted (< 30g) by 36h "
        "into a fast. Caps the depletion floor — should NOT come "
        "back up between 36 and 48h of fasting."
    ),
    arms=(_FAST_24H_TO_48H_ARM,),
    predicate=lambda traj, ctx: hinge_max_value(
        traj, ctx.col("liver_glycogen"),
        window=ctx.window(12 * 60.0, float(ctx.arm.duration_min)),  # last 12h of the arm = 36–48h fasted
        ceiling=30.0,
    ),
    scale=20.0,
)


GLUCAGON_INSULIN_RATIO_HIGH_FAST = PhysiologyRule(
    name="glucagon_insulin_ratio_high_fast",
    source="Unger & Orci (1975); Cherrington (1999)",
    description=(
        "During prolonged fasting the glucagon/insulin ratio shifts "
        "strongly toward glucagon: glucagon ~60–100 pg/mL over insulin "
        "~3–6 μU/mL gives a ratio well above 10. Iter 74: implemented as "
        "a true ratio hinge (glucagon/insulin ≥ 8 over 24–48h fast) "
        "rather than the previous glucagon-only floor — the previous "
        "predicate hinged only glucagon ≥ 60 and never constrained the "
        "relationship the citation is about, so a model with both "
        "markers elevated satisfied it spuriously."
    ),
    arms=(_FAST_24H_TO_48H_ARM,),
    predicate=lambda traj, ctx: hinge_min_ratio(
        traj, ctx.col("glucagon"), ctx.col("insulin"),
        window=ctx.window(60.0, float(ctx.arm.duration_min)),
        floor_ratio=8.0,
    ),
    scale=8.0,
)


INSULIN_GLUCAGON_RATIO_HIGH_FED = PhysiologyRule(
    name="insulin_glucagon_ratio_high_fed",
    source="Unger & Orci (1981); Cherrington (1999)",
    description=(
        "Iter 74: the fed-state mirror of the fast ratio. Postprandially "
        "the balance inverts — insulin surges while glucagon is "
        "alpha-cell-suppressed, so insulin/glucagon rises far above its "
        "deep-fast value (~0.05). Pins insulin/glucagon ≥ 0.5 over the "
        "60–180 min post-meal window: a true ratio constraint on the "
        "fed-state coupling, complementing the anti-correlation rule "
        "(which only constrains shape, not the absolute balance)."
    ),
    arms=(_MIXED_MEAL_BREAKFAST_ARM, _OGTT_75G_ARM, _MIXED_MEAL_DINNER_ARM),
    predicate=lambda traj, ctx: hinge_min_ratio(
        traj, ctx.col("insulin"), ctx.col("glucagon"),
        window=ctx.window(60.0, 180.0),
        floor_ratio=0.5,
    ),
    scale=1.0,
)


# --- Postprandial replenishment + counter-regulation linkage ---

LIVER_GLYCOGEN_REPLENISHES_POSTPRANDIAL = PhysiologyRule(
    name="liver_glycogen_replenishes_postprandial",
    source="Petersen et al. (1999); Rothman et al. (1991)",
    description=(
        "Hepatic glycogen rises ≥ 10g over the 4-h postprandial "
        "window of a 75g carb-containing meal (gluconeogenesis "
        "switched off, glycogen synthesis on). Tests the "
        "GlycogenFluxHead's anabolic gate (iter-57) fires during "
        "nutrient appearance."
    ),
    arms=(_MIXED_MEAL_BREAKFAST_ARM,),
    predicate=lambda traj, ctx: hinge_min_rise(
        traj, ctx.col("liver_glycogen"),
        pre=ctx.window(0.0, 60.0),
        post=ctx.window(180.0, 300.0),  # 2–4 h post-meal
        min_rise=10.0,
    ),
    scale=10.0,
)


MUSCLE_GLYCOGEN_STABLE_AT_REST = PhysiologyRule(
    name="muscle_glycogen_stable_at_rest",
    source="Coyle (1991); Hultman & Bergström (1967)",
    description=(
        "Muscle glycogen drifts by < 20g over an 8h resting fast "
        "(no exercise, no breakdown gate — pool should be ~stable "
        "on this timescale)."
    ),
    arms=(_FAST_8H_TO_16H_ARM,),
    predicate=lambda traj, ctx: hinge_max_drift(
        traj, ctx.col("muscle_glycogen"),
        window=ctx.window(0.0, float(ctx.arm.duration_min)),
        max_drift=20.0,
    ),
    scale=20.0,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PHYSIOLOGY_RULES: list[PhysiologyRule] = [
    # iter 42–52: original five
    GLUCAGON_FALLS_POSTPRANDIAL,
    FFA_INVERSE_TO_INSULIN_POSTPRANDIAL,
    GHRELIN_FALLS_AFTER_MEAL,
    BHB_RISES_DURING_FAST,
    CORTISOL_MORNING_PEAK,
    # iter 64: HPA cascade companion rules (acth_precedes_cortisol_morning_peak
    # was replaced by acth_precedes_cortisol_relational in iter 65)
    ACTH_PRECEDES_CORTISOL_RELATIONAL,
    INSULIN_RISES_POSTPRANDIAL,
    # iter 65: Move C compact
    GLP1_RETURNS_TO_BASELINE_POSTPRANDIAL,
    GLP1_PEAK_BOUNDED,
    GHRELIN_RISES_PRE_MEAL,
    GLUCOSE_POSTPRANDIAL_BOUNDED,
    GLUCOSE_FASTING_FLOOR,
    FFA_RISES_DURING_FAST,
    BHB_PLATEAU_CAPPED,
    INSULIN_GLUCAGON_ANTAGONIST_POSTPRANDIAL,
    LIVER_GLYCOGEN_DEPLETES_IN_FAST,
    SBP_ABOVE_DBP_ALWAYS,
    # iter 66: Move C full — order-of-magnitude rule expansion
    # Metabolic
    GLUCOSE_RISES_WITH_MEAL,
    GLUCOSE_RETURNS_BELOW_140_POSTPRANDIAL,
    GLUCOSE_OVERNIGHT_STABILITY,
    INSULIN_FIRST_PHASE_RISE,
    INSULIN_RETURNS_TO_BASAL_POSTPRANDIAL,
    INSULIN_FALLS_DURING_FAST,
    GLUCAGON_RISES_DURING_FAST,
    FFA_FALLS_POSTPRANDIAL,
    BHB_RETURNS_TO_BASELINE_POSTPRANDIAL,
    HEPATIC_OUTPUT_INVERSE_TO_INSULIN,
    # Exercise / lactate / muscle
    LACTATE_RISES_WITH_EXERCISE,
    LACTATE_RETURNS_POST_EXERCISE,
    FFA_RISES_WITH_EXERCISE,
    MUSCLE_GLYCOGEN_FALLS_WITH_EXERCISE,
    # Appetite
    GHRELIN_EVENING_PEAK,
    LEPTIN_EVENING_PEAK,
    LEPTIN_STABLE_OVER_MEAL,
    GLP1_RISES_EARLY_POSTPRANDIAL,
    GLP1_INSULIN_CORRELATION_POSTPRANDIAL,
    # Stress / HPA
    CORTISOL_EVENING_TROUGH,
    CORTISOL_CIRCADIAN_AMPLITUDE,
    CORTISOL_RISES_WITH_EXERCISE,
    ACTH_EVENING_TROUGH,
    ACTH_CIRCADIAN_AMPLITUDE,
    # Cardiovascular
    HR_RISES_WITH_EXERCISE,
    HR_RECOVERY_POST_EXERCISE,
    HR_FALLS_DURING_SLEEP,
    HRV_HIGHER_DURING_SLEEP,
    HRV_INVERSE_TO_HR,
    SBP_RISES_WITH_EXERCISE,
    SBP_MORNING_SURGE,
    SBP_FALLS_DURING_SLEEP,
    # Thermoreg
    TEMP_CIRCADIAN_AMPLITUDE,
    TEMP_LATE_AFTERNOON_PEAK,
    TEMP_PRE_DAWN_TROUGH,
    # Respiratory
    RR_RISES_WITH_EXERCISE,
    SPO2_STABLE_AT_REST,
    # Prolonged fast (24–48h)
    BHB_CONTINUES_RISING_24_48H,
    FFA_SUSTAINED_PROLONGED_FAST,
    LIVER_GLYCOGEN_DEPLETED_PROLONGED_FAST,
    GLUCAGON_INSULIN_RATIO_HIGH_FAST,
    INSULIN_GLUCAGON_RATIO_HIGH_FED,
    # Postprandial replenishment + rest stability
    LIVER_GLYCOGEN_REPLENISHES_POSTPRANDIAL,
    MUSCLE_GLYCOGEN_STABLE_AT_REST,
]
