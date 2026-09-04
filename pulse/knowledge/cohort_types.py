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


class StatisticKind(str, Enum):
    # mean(arm[window]) per arm; per-arm target (single arm).
    MEAN_IN_WINDOW = "mean_in_window"
    # mean(arm[1].window) - mean(arm[0].window); requires exactly 2 arms.
    DELTA_MEANS = "delta_means"
    # max(arm[window]) per arm; per-arm target (single arm).
    PEAK_VALUE = "peak_value"
    # max(arm[1].window) - max(arm[0].window); requires exactly 2 arms.
    DELTA_PEAKS = "delta_peaks"
    # Soft argmax over window (minutes from window start), single arm.
    TIME_TO_PEAK = "time_to_peak"


@dataclass(frozen=True)
class CohortArmSpec:
    """One virtual arm: a protocol the model rolls forward to produce a statistic.

    ``sleep_wake`` / ``activity`` are per-minute series (1 = awake, 0 = asleep;
    activity 0 = rest). When ``None`` the rollout uses the explicit frame
    constants in ``pulse.cohort_loss`` (awake, rest) — the same frame the
    teacher audits run in.

    ``prefast_hours`` (iter 97, review 4.5): how long the subject has been
    fasting when the arm STARTS. The cold initial state is then the teacher's
    row after that many hours of meal-free rest, not its fed row 0 — so an arm
    labelled "24 h -> 48 h fasted" really begins at 24 h fasted (liver glycogen
    ~55 g, not 100 g). 0 keeps the legacy row-0 start.
    """

    label: str
    duration_min: int
    start_hour: float
    meals: tuple[tuple[float, float, float, float], ...]
    sleep_wake: tuple[float, ...] | None = None
    activity: tuple[float, ...] | None = None
    prefast_hours: float = 0.0


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

    * ``cold`` — derive the starting state from
      ``cold_model_trajectory(PatientParams(), arm[0].meals, ...)``.
      Use when the trial protocol begins from the cold model's fasting
      assumption (typical OGTT / breakfast / sleep-restriction
      designs that overnight fasted subjects).
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
    for glucose).
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

    def __post_init__(self) -> None:
        if not self.arms:
            raise ValueError(f"{self.name}: at least one arm required")
        if self.sigma <= 0:
            raise ValueError(f"{self.name}: sigma must be positive (got {self.sigma})")
        if self.band_halfwidth < 0:
            raise ValueError(f"{self.name}: band_halfwidth must be >= 0")
        if self.shape is TargetShape.BAND and self.band_halfwidth <= 0:
            raise ValueError(f"{self.name}: shape=band needs band_halfwidth > 0")
        if self.n_arm is not None and self.n_arm < 1:
            raise ValueError(f"{self.name}: n_arm must be >= 1")
        per_arm = (
            self.kind in (StatisticKind.DELTA_MEANS, StatisticKind.DELTA_PEAKS)
        )
        if per_arm and len(self.arms) != 2:
            raise ValueError(
                f"{self.name}: kind={self.kind.value} requires exactly 2 arms",
            )
        if self.per_arm_windows is not None and len(self.per_arm_windows) != len(self.arms):
            raise ValueError(
                f"{self.name}: per_arm_windows length must equal arms length",
            )
