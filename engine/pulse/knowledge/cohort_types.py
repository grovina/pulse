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
    """One virtual arm: a protocol the model rolls forward to produce a statistic."""

    label: str
    duration_min: int
    start_hour: float
    meals: tuple[tuple[float, float, float, float], ...]
    sleep_wake: tuple[float, ...] | None = None
    activity: tuple[float, ...] | None = None


@dataclass(frozen=True)
class StatisticWindow:
    """Inclusive-exclusive minute range relative to arm start."""

    start_min: int
    end_min: int


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

    def __post_init__(self) -> None:
        if not self.arms:
            raise ValueError(f"{self.name}: at least one arm required")
        if self.sigma <= 0:
            raise ValueError(f"{self.name}: sigma must be positive (got {self.sigma})")
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
