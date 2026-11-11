"""
Quantitative cohort statistic specs for population-level supervision.

A ``CohortStatisticSpec`` declares one or more virtual arms and a target
*scalar* statistic with a literature-derived value and standard error. The
loss is a Gaussian discrepancy ``((predicted - target) / sigma) ** 2``, so
weak literature (large sigma) automatically pulls less than strong
literature (small sigma). This is the PRD's cohort / summary-statistic
supervision in concrete form.

This replaces the older ordering-only ``CohortContrastSpec`` (which only
encouraged sign of an inter-arm difference) with quantitative effect-size
matching, so the model learns *how much*, not just *which way*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..types import SERIES_INDEX


class StatisticKind(str, Enum):
    # mean(arm[window]) per arm; per-arm target (single arm).
    MEAN_IN_WINDOW = "mean_in_window"
    # mean(arm[1].window) - mean(arm[0].window); requires exactly 2 arms.
    DELTA_MEANS = "delta_means"
    # max(arm[window]) per arm; per-arm target (single arm).
    PEAK_VALUE = "peak_value"
    # max(arm[1].window) - max(arm[0].window); requires exactly 2 arms.
    DELTA_PEAKS = "delta_peaks"
    # mean(arm[1].window) / mean(arm[0].window); requires exactly 2 arms. A
    # finding reported in a unit the model does not share (per kg of lean mass,
    # another assay's frame) is scale-free only as the ratio of its two arms.
    RATIO_MEANS = "ratio_means"
    # Area of (value − the arm's own baseline) on arm 1, over the same area on
    # arm 0. The baseline window is the run-in and is not part of the area.
    # This is a baseline-subtracted integral, which a ratio of window means is
    # not: the fasting level stays in a mean and cancels in an increment.
    RATIO_INCREMENTS = "ratio_increments"
    # Soft argmax over window (minutes from window start), single arm.
    TIME_TO_PEAK = "time_to_peak"
    # A whole published trajectory: the marker value at each ``CurvePoint`` of
    # ``curve``, single arm. The replacement for the teacher's dense tape — a
    # digitized mean +/- SEM at timepoints. A CURVE spec is never scored
    # directly; ``expand_curve_spec`` compiles it into one MEAN_IN_WINDOW spec
    # per point (see cohort_statistics.py), so the whole battle-tested batched
    # scorer handles it unchanged.
    CURVE = "curve"
    # Unbiased sample standard deviation, across people, of each person's mean
    # of the marker over the window. The published spread. Single arm.
    SPREAD = "spread"
    # Pearson correlation, across people, of the marker's window-mean with
    # ``partner`` (another marker's window-mean, or a named factor). Single arm.
    CORRELATION = "correlation"
    # mean(arm[window]) - mean(arm[baseline]) in one arm: the change of the
    # marker from the same person's own run-in. Single arm.
    CHANGE_FROM_BASELINE = "change_from_baseline"
    # Trapezoidal area of the marker over the paper's own ``sample_times``,
    # divided by the span of those times: the time-weighted mean, in the
    # marker's unit. Per person. Single arm.
    AREA_UNDER_CURVE = "area_under_curve"
    # Trapezoidal area, over ``sample_times``, of the marker minus that person's
    # baseline (the value at the first sample, or the mean over ``baseline``),
    # divided by the span. ``below_baseline`` says whether stretches under the
    # baseline count negative or not at all. Per person. Single arm.
    INCREMENTAL_AREA = "incremental_area"


class BelowBaseline(str, Enum):
    """What an incremental area does with the time the marker is under baseline.

    * ``subtracted`` — a net area: those stretches count negative.
    * ``ignored`` — only the area above baseline counts; a segment that crosses
      the baseline is cut where the straight line between its two samples
      crosses it (the usual "positive incremental area").
    * ``unstated`` — the paper does not say which of the two it printed, and
      most do not. The model answers for both: its number is the paper's value
      when that lies between the two readings of the model's own curves, and
      the nearer reading otherwise. Where a marker never goes under its
      baseline the readings are one and nothing is lost.
    """

    SUBTRACTED = "subtracted"
    IGNORED = "ignored"
    UNSTATED = "unstated"


# Named axes of the person code. A draw writes factor ``i`` into coordinate
# ``i`` of the 32-wide code; every other coordinate stays 0, which is the
# median person. Published cohorts restrict these axes (``Population``).
FACTORS: tuple[str, ...] = (
    "insulin_resistance",
    "beta_cell",
    "adiposity",
    "fitness",
    "chronotype",
    "hpa_reactivity",
    "gastric_emptying",
    "sex",
    "age",
)


def factor_index(name: str) -> int:
    """Coordinate of a named factor in the person code."""
    try:
        return FACTORS.index(name)
    except ValueError:
        raise ValueError(f"{name!r} is not a person factor") from None


# Kinds whose statistic is built from two arms, and kinds that read an arm as
# the mean over its window (the rest read the peak, or the soft argmax).
TWO_ARM_KINDS = frozenset({
    StatisticKind.DELTA_MEANS, StatisticKind.DELTA_PEAKS, StatisticKind.RATIO_MEANS,
    StatisticKind.RATIO_INCREMENTS,
})
_RATIO_KINDS = frozenset({StatisticKind.RATIO_MEANS, StatisticKind.RATIO_INCREMENTS})
# Kinds read at the paper's own sampling minutes (``sample_times``), not over a window.
SAMPLED_KINDS = frozenset({StatisticKind.AREA_UNDER_CURVE, StatisticKind.INCREMENTAL_AREA})
# Single-arm kinds that are a per-person quantity averaged over people, scored as a
# mean (sigma is the between-person SD, divided by sqrt(n_arm)); they have no entry in
# MEAN_KINDS because that reads a plain window mean.
PER_PERSON_KINDS = frozenset({StatisticKind.CHANGE_FROM_BASELINE}) | SAMPLED_KINDS
_SINGLE_ARM_ONLY = PER_PERSON_KINDS | {StatisticKind.SPREAD, StatisticKind.CORRELATION}
MEAN_KINDS = frozenset({
    StatisticKind.MEAN_IN_WINDOW, StatisticKind.DELTA_MEANS, StatisticKind.RATIO_MEANS,
})


def combine_arms(kind: StatisticKind, per_arm):
    """The statistic from its per-arm values (floats or tensors alike)."""
    if kind in _RATIO_KINDS:
        return per_arm[1] / per_arm[0]
    if kind in TWO_ARM_KINDS:
        return per_arm[1] - per_arm[0]
    return per_arm[0]


def statistic_cuts(spec: "CohortStatisticSpec", arm_idx: int = 0) -> tuple[int, ...]:
    """The minutes of an arm where a statistic starts or stops reading it.

    A meal that moves across one of them changes what the statistic measures. The
    window's two ends for every kind; for the kinds that read a baseline or the
    paper's sampling minutes, those too (a sample at ``t`` reads minute ``t`` only).
    """
    window = spec.per_arm_windows[arm_idx] if spec.per_arm_windows is not None else spec.window
    cuts = [window.start_min, window.end_min]
    if spec.kind in PER_PERSON_KINDS and spec.baseline is not None:
        cuts += [spec.baseline.start_min, spec.baseline.end_min]
    if spec.sample_times is not None:
        cuts += [x for t in spec.sample_times for x in (t, t + 1)]
    return tuple(cuts)


@dataclass(frozen=True)
class Population:
    """The people an arm was measured in.

    Unset fields draw from the reference prior: each continuous factor is a
    standard normal in its code slot, sex is a fair coin, age is centred at
    40 years and adiposity at BMI 25. ``sex`` is ``"female"`` or ``"male"``.
    ``age`` and ``bmi`` are inclusive ranges (years, kg/m²) drawn uniformly.
    ``healthy`` puts insulin resistance on the sensitive side of the median
    and beta-cell function on the competent side.
    """

    name: str = "reference"
    sex: str | None = None
    age: tuple[float, float] | None = None
    bmi: tuple[float, float] | None = None
    healthy: bool = False

    def __post_init__(self) -> None:
        if self.sex not in (None, "female", "male"):
            raise ValueError(f"{self.name}: sex must be 'female', 'male', or unset")
        for label, span in (("age", self.age), ("bmi", self.bmi)):
            if span is None:
                continue
            if len(span) != 2 or float(span[0]) > float(span[1]):
                raise ValueError(f"{self.name}: {label} range {span} is not lo <= hi")


@dataclass(frozen=True)
class CurvePoint:
    """One (time, mean, sigma) sample of a published trajectory.

    ``t_min`` is minutes from arm start. ``target`` / ``sigma`` are in the
    marker's native units; ``sigma`` is the paper's reported standard error at
    that timepoint (not yet corrected for the correlation between neighbouring
    points — ``expand_curve_spec`` applies that down-weight).
    """

    t_min: int
    target: float
    sigma: float


@dataclass(frozen=True)
class CohortArmSpec:
    """One virtual arm: a protocol the model rolls forward to produce a statistic.

    ``sleep_wake`` / ``activity`` are per-minute series (1 = awake, 0 = asleep;
    activity 0 = rest). When ``None`` the rollout uses the explicit frame
    constants in ``pulse.cohort_loss`` (awake, rest) — the same frame the
    teacher audits run in.

    The drug and IV inputs are per-minute series too, ``None`` for absent, and 0
    in a minute is absent in that minute, so a protocol can switch one on after a
    night or a meal (``held``). ``glucose_infusion`` is an IV glucose appearance
    rate in mg/dL/min of the patient's glucose space (the unit of the metabolic
    glucose rate). ``glucose_clamp`` is a closed-loop target in mg/dL: a
    proportional controller infuses ``relu(Kp · (target − G))`` every minute
    (``pulse.modules.metabolic.GLUCOSE_CLAMP_KP``) on top of any open-loop
    ``glucose_infusion``, and the realized infusion is the clamp's GIR — the
    ``glucose_infusion_rate`` observable. ``insulin_clamp`` is the plasma insulin
    plateau in µU/mL of the clamp's insulin arm, held by a constant infusion
    (``pulse.modules.metabolic.insulin_clamp_infusion_t``). ``sglt2`` is the
    empagliflozin exposure (``pulse.renal``: 1 is the steady 25 mg/day dose).
    ``bile_fistula`` drains bile before it meets the fat: 1 is an empty intestine.

    ``prefast_hours``: how long the subject has already been fasting when the
    arm starts. Cohort rollouts reach that state by spinning the student
    through a fed day and then this fast, not by copying a teacher row.

    ``population`` names who was measured. Unset is the reference prior.
    """

    label: str
    duration_min: int
    start_hour: float
    # Each meal is (minute, carbohydrate g, fat g, protein g): fat before protein.
    meals: tuple[tuple[float, float, float, float], ...]
    sleep_wake: tuple[float, ...] | None = None
    activity: tuple[float, ...] | None = None
    glucose_infusion: tuple[float, ...] | None = None  # mg/dL/min of the patient's glucose space
    glucose_clamp: tuple[float, ...] | None = None  # closed-loop glucose target, mg/dL
    insulin_clamp: tuple[float, ...] | None = None  # plasma insulin plateau, µU/mL
    sglt2: tuple[float, ...] | None = None  # empagliflozin exposure, 1 = 25 mg/day at steady state
    bile_fistula: tuple[float, ...] | None = None  # 1 drains the intestine of bile
    prefast_hours: float = 0.0
    population: Population | None = None


def held(level: float, duration_min: int, start_min: int = 0) -> tuple[float, ...]:
    """An input the protocol switches on at ``start_min`` and holds to the end of
    the arm: 0 before, ``level`` after."""
    if not 0 <= start_min <= duration_min:
        raise ValueError(f"start_min {start_min} outside [0, {duration_min}]")
    return (0.0,) * start_min + (float(level),) * (duration_min - start_min)


def is_on(series: tuple[float, ...] | None) -> bool:
    """Does this input act in any minute of the arm?"""
    return series is not None and any(x > 0.0 for x in series)


@dataclass(frozen=True)
class StatisticWindow:
    """Inclusive-exclusive minute range relative to arm start."""

    start_min: int
    end_min: int


class TargetShape(str, Enum):
    """How the batch-mean statistic is scored against ``target`` (iter 97, review 4.3).

    * ``point`` — Gaussian: ``((mean - target) / sem)^2``. For a literature
      mean with a standard error.
    * ``band`` — zero loss while ``|mean - target| <= band_halfwidth``, Gaussian
      on the excess outside. For "the literature puts it in a range".
    * ``at_most`` / ``at_least`` — one-sided: zero loss on the allowed side.
      For diagnostic criteria (a WHO 2-h OGTT glucose < 140 mg/dL is a ceiling,
      not a target of 120 +/- 15).
    """

    POINT = "point"
    BAND = "band"
    AT_MOST = "at_most"
    AT_LEAST = "at_least"


class InitMode(str, Enum):
    """How to seed the integration starting state for one spec.

    * ``cold`` — the student spins through a fed day and then the arm's
      pre-fast, and the protocol starts from that state. Typical OGTT /
      breakfast / sleep-restriction designs that overnight-fasted subjects.
    * ``norm_center`` — start from the marker typicals
      (``NORM_CENTER``). Use when the trial design is poorly captured
      by the cold model's fasting trajectory and anchoring there
      injects bias (e.g. ad-libitum free-living protocols, fed-state
      starts, and any spec where the cold model and the trial design
      disagree on the run-in state).
    """

    COLD = "cold"
    NORM_CENTER = "norm_center"


@dataclass(frozen=True)
class CohortStatisticSpec:
    """One quantitative literature finding as a differentiable target.

    ``target`` and ``sigma`` are in the marker's native units (e.g. mg/dL
    for glucose). ``marker_id`` is any series of the observed trajectory: a state
    marker, or one of ``types.OBSERVABLES``.
    """

    name: str
    source: str
    description: str
    arms: tuple[CohortArmSpec, ...]
    marker_id: str
    kind: StatisticKind
    window: StatisticWindow
    target: float
    sigma: float
    weight: float = 1.0
    # Soft argmax temperature; controls sharpness of TIME_TO_PEAK estimator.
    softargmax_beta: float = 0.05
    # Optional override: use a different window per arm (must align with arms).
    per_arm_windows: tuple[StatisticWindow, ...] | None = field(default=None)
    # Per-spec initial-state seeding strategy. Defaults to the cold-model
    # anchor used by the textbook benchmark; specs whose trial design
    # doesn't match the cold-model fasting assumption can opt into
    # NORM_CENTER to avoid biasing the starting state.
    init_mode: InitMode = InitMode.COLD
    # Iter 97 (review 4.3): scoring shape (see ``TargetShape``) and the band
    # half-width used by ``TargetShape.BAND`` (marker units).
    shape: TargetShape = TargetShape.POINT
    band_halfwidth: float = 0.0
    # Iter 97 (review 4.3): the loss scores the BATCH MEAN of the sampled
    # patients against the target with ``sem = sigma / sqrt(n)``. ``n`` is the
    # batch size by default (the arm the student's sample forms); a spec whose
    # ``sigma`` is already a standard error of the published mean can pin
    # ``n_arm=1`` so it is not tightened further.
    n_arm: int | None = None
    # A whole-trajectory finding (``kind=CURVE``): the marker's value at each
    # of these timepoints, in one arm. Compiled to per-point specs before
    # scoring. ``None`` for every scalar statistic.
    curve: tuple[CurvePoint, ...] | None = field(default=None)
    # CORRELATION only: the second series. A marker id correlates the two
    # window-means; a factor name correlates the marker's window-mean with
    # that factor's code.
    partner: str | None = None
    # RATIO_INCREMENTS: each arm's run-in. The area is the measured window minus
    # this window's mean, and the baseline ends where the window starts.
    # CHANGE_FROM_BASELINE: the window the change is measured from (anywhere in the
    # arm). INCREMENTAL_AREA: optional; the mean of this window is the baseline, in
    # place of the value at the first sample time.
    baseline: StatisticWindow | None = None
    # AREA_UNDER_CURVE and INCREMENTAL_AREA: the minutes (from arm start, strictly
    # increasing, inside the arm) at which the paper drew its samples. ``window`` is
    # ``[sample_times[0], sample_times[-1] + 1)``. ``target`` and ``sigma`` are in the
    # marker's unit: the area divided by ``sample_times[-1] - sample_times[0]``.
    sample_times: tuple[int, ...] | None = None
    # INCREMENTAL_AREA only (required there).
    below_baseline: BelowBaseline | None = None
    # The paper this finding comes from: a lowercase ASCII slug, ``firstauthor-year``,
    # read off ``source``. Sealing is decided per paper (``sealing.paper_is_sealed``),
    # so every item of a paper is held out or trained together. Required of every
    # registered item; the registry rejects an empty one.
    paper: str = ""
    # Held-out: a sealed item never enters a training registry; it is reachable
    # only through the scoring path. The registry sets it from ``paper`` and
    # ``sealing.HELD_OUT_ITEMS``; an author never does.
    sealed: bool = False

    def __post_init__(self) -> None:
        if not self.arms:
            raise ValueError(f"{self.name}: at least one arm required")
        if self.marker_id not in SERIES_INDEX:
            raise ValueError(f"{self.name}: {self.marker_id!r} is not an observed series")
        if self.kind is StatisticKind.CURVE:
            if not self.curve:
                raise ValueError(f"{self.name}: kind=curve needs a non-empty curve")
            if len(self.arms) != 1:
                raise ValueError(f"{self.name}: kind=curve is a single-arm trajectory")
            dur = self.arms[0].duration_min
            for cp in self.curve:
                if cp.t_min < 0 or cp.t_min >= dur:
                    raise ValueError(
                        f"{self.name}: curve point t={cp.t_min} outside [0, {dur})",
                    )
                if cp.sigma <= 0:
                    raise ValueError(f"{self.name}: curve point sigma must be positive")
        elif self.curve is not None:
            raise ValueError(f"{self.name}: curve is only for kind=curve")
        if self.sigma <= 0:
            raise ValueError(f"{self.name}: sigma must be positive (got {self.sigma})")
        if self.band_halfwidth < 0:
            raise ValueError(f"{self.name}: band_halfwidth must be >= 0")
        if self.shape is TargetShape.BAND and self.band_halfwidth <= 0:
            raise ValueError(f"{self.name}: shape=band needs band_halfwidth > 0")
        if self.n_arm is not None and self.n_arm < 1:
            raise ValueError(f"{self.name}: n_arm must be >= 1")
        if self.kind in TWO_ARM_KINDS and len(self.arms) != 2:
            raise ValueError(
                f"{self.name}: kind={self.kind.value} requires exactly 2 arms",
            )
        if self.kind is StatisticKind.RATIO_INCREMENTS:
            if self.baseline is None:
                raise ValueError(f"{self.name}: kind=ratio_increments needs a baseline window")
            if self.baseline.end_min <= self.baseline.start_min:
                raise ValueError(f"{self.name}: baseline window is empty")
            windows = self.per_arm_windows if self.per_arm_windows is not None else (self.window,) * len(self.arms)
            for arm, window in zip(self.arms, windows):
                if self.baseline.end_min > window.start_min:
                    raise ValueError(
                        f"{self.name}: the baseline has to end before the measured window",
                    )
                if window.end_min > arm.duration_min or self.baseline.end_min > arm.duration_min:
                    raise ValueError(f"{self.name}: a window runs past the arm")
        elif self.kind is StatisticKind.CHANGE_FROM_BASELINE:
            if self.baseline is None:
                raise ValueError(f"{self.name}: kind=change_from_baseline needs a baseline window")
            self._check_inside_arm("window", self.window)
            self._check_inside_arm("baseline window", self.baseline)
        elif self.kind is StatisticKind.INCREMENTAL_AREA and self.baseline is not None:
            self._check_inside_arm("baseline window", self.baseline)
        elif self.baseline is not None:
            raise ValueError(
                f"{self.name}: baseline is only for kinds ratio_increments, change_from_baseline "
                "and incremental_area",
            )
        if self.kind in SAMPLED_KINDS:
            times = self.sample_times
            if times is None or len(times) < 2:
                raise ValueError(f"{self.name}: kind={self.kind.value} needs at least two sample_times")
            if any(b <= a for a, b in zip(times, times[1:])):
                raise ValueError(f"{self.name}: sample_times must be strictly increasing")
            if times[0] < 0 or times[-1] >= self.arms[0].duration_min:
                raise ValueError(
                    f"{self.name}: sample_times {times[0]}..{times[-1]} lie outside the arm "
                    f"[0, {self.arms[0].duration_min})",
                )
            if (self.window.start_min, self.window.end_min) != (times[0], times[-1] + 1):
                raise ValueError(f"{self.name}: window must be [first sample time, last sample time + 1)")
        elif self.sample_times is not None:
            raise ValueError(f"{self.name}: sample_times is only for kinds area_under_curve and incremental_area")
        if self.kind is StatisticKind.INCREMENTAL_AREA:
            if self.below_baseline is None:
                raise ValueError(f"{self.name}: kind=incremental_area needs below_baseline")
        elif self.below_baseline is not None:
            raise ValueError(f"{self.name}: below_baseline is only for kind=incremental_area")
        if self.kind in _SINGLE_ARM_ONLY and len(self.arms) != 1:
            raise ValueError(f"{self.name}: kind={self.kind.value} is a single-arm statistic")
        if self.kind is StatisticKind.CORRELATION:
            if self.partner is None:
                raise ValueError(f"{self.name}: kind=correlation needs a partner")
            if self.partner not in SERIES_INDEX and self.partner not in FACTORS:
                raise ValueError(
                    f"{self.name}: partner {self.partner!r} is not a marker or a factor",
                )
        elif self.partner is not None:
            raise ValueError(f"{self.name}: partner is only for kind=correlation")
        if self.per_arm_windows is not None and len(self.per_arm_windows) != len(self.arms):
            raise ValueError(
                f"{self.name}: per_arm_windows length must equal arms length",
            )

    def _check_inside_arm(self, label: str, window: StatisticWindow) -> None:
        if not 0 <= window.start_min < window.end_min <= self.arms[0].duration_min:
            raise ValueError(
                f"{self.name}: {label} [{window.start_min}, {window.end_min}) is empty or runs "
                f"outside the arm [0, {self.arms[0].duration_min})",
            )
