"""Sparse evidence: the native training example.

A published statistic, a wearable check-in, a hinge rule, and a teacher
rate at a clamp state are the same kind of object: a protocol, an observation
operator, a reliability, and an authority.

The teacher remains a generator of protocols and of unobserved fill. It is
not a source of minute-by-minute hormone truth. Literature and real
measurements outrank it wherever they cover the same marker.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..types import MARKER_INDEX, STATE_DIM
from .cohort_statistics import ALL_COHORT_STATISTICS, SEALED_COHORT_STATISTICS
from .cohort_types import (
    CohortArmSpec,
    CohortStatisticSpec,
    InitMode,
    StatisticKind,
    StatisticWindow,
    TargetShape,
)
from .physiology_rules import (
    PHYSIOLOGY_RULES,
    SEALED_PHYSIOLOGY_RULES,
    PhysiologyRule,
)


class Authority(str, Enum):
    """Who is allowed to overrule whom.

    ``literature`` and ``real`` win over ``teacher`` on the same marker.
    The teacher tape is only the wearable-frequency vitals; internals that
    no paper reports as a time series may still be distilled from the
    generator (rate, or a long-window level for slow states).
    """

    LITERATURE = "literature"
    REAL = "real"
    TEACHER = "teacher"


class OperatorKind(str, Enum):
    WINDOW_MEAN = "window_mean"
    PEAK = "peak"
    TIME_TO_PEAK = "time_to_peak"
    DELTA_MEANS = "delta_means"
    DELTA_PEAKS = "delta_peaks"
    RATIO_MEANS = "ratio_means"
    HINGE = "hinge"
    TAPED_VITAL = "taped_vital"
    RATE = "rate"
    LONG_LEVEL = "long_level"
    DOSE_SLOPE = "dose_slope"
    DOSE_PEAK = "dose_peak"
    DOSE_RANK = "dose_rank"
    CALIBRATED_SERIES = "calibrated_series"
    SPREAD = "spread"
    CORRELATION = "correlation"


OPERATOR_OF_STATISTIC: dict[StatisticKind, OperatorKind] = {
    StatisticKind.MEAN_IN_WINDOW: OperatorKind.WINDOW_MEAN,
    StatisticKind.PEAK_VALUE: OperatorKind.PEAK,
    StatisticKind.TIME_TO_PEAK: OperatorKind.TIME_TO_PEAK,
    StatisticKind.DELTA_MEANS: OperatorKind.DELTA_MEANS,
    StatisticKind.DELTA_PEAKS: OperatorKind.DELTA_PEAKS,
    StatisticKind.RATIO_MEANS: OperatorKind.RATIO_MEANS,
    StatisticKind.SPREAD: OperatorKind.SPREAD,
    StatisticKind.CORRELATION: OperatorKind.CORRELATION,
}


# Wearable-frequency vitals the generator is allowed to put on the training
# tape. Everything else on a FullBody day is NaN: cohort stats, physiology
# rules, sweeps, and (for teacher-only internals) distillation own those.
TEACHER_TAPE_MARKERS: tuple[str, ...] = ("glucose", "hr", "sbp", "dbp", "temp")

# Generator-only internals: no literature time-series statistic owns them.
# Slow pair is scored as a day-scale level, not a 60-minute slope.
TEACHER_DISTILL_MARKERS: tuple[str, ...] = (
    "crh",
    "insulin_action",
    "insulin_slow",
    "mitochondrial_capacity",
    "fat_mass",
    "gallbladder_bile",
    "intestinal_bile",
    "sns_tone",
    "epinephrine",
)
TEACHER_DISTILL_LONG_ONLY: tuple[str, ...] = (
    "mitochondrial_capacity",
    "fat_mass",
)


def mask_trajectory(trajectory: np.ndarray, markers: tuple[str, ...]) -> np.ndarray:
    """Keep ``markers``; every other column is NaN (unsupervised on this tape)."""
    view = np.full((trajectory.shape[0], STATE_DIM), np.nan, dtype=np.float32)
    for m in markers:
        view[:, MARKER_INDEX[m]] = trajectory[:, MARKER_INDEX[m]]
    return view


def literature_markers() -> frozenset[str]:
    """Markers a cohort statistic already reports. Distill must not copy them."""
    return frozenset(spec.marker_id for spec in ALL_COHORT_STATISTICS)


def teacher_literature_collisions() -> tuple[str, ...]:
    return tuple(sorted(set(TEACHER_DISTILL_MARKERS) & literature_markers()))


@dataclass(frozen=True)
class Observation:
    """One operator on a simulated protocol.

    ``target`` / ``sigma`` are in the marker's native units for cohort-shaped
    items. Hinge rules carry their predicate on the parent ``EvidenceItem``.
    """

    marker_id: str
    kind: OperatorKind
    window: StatisticWindow | None = None
    target: float | None = None
    sigma: float | None = None
    shape: TargetShape = TargetShape.POINT
    band_halfwidth: float = 0.0


@dataclass(frozen=True)
class EvidenceItem:
    """One piece of evidence the student may be trained against."""

    name: str
    source: str
    description: str
    authority: Authority
    arms: tuple[CohortArmSpec, ...]
    observations: tuple[Observation, ...]
    init_mode: InitMode = InitMode.COLD
    weight: float = 1.0
    rule: PhysiologyRule | None = None


def from_cohort(spec: CohortStatisticSpec) -> EvidenceItem:
    return EvidenceItem(
        name=spec.name,
        source=spec.source,
        description=spec.description,
        authority=Authority.LITERATURE,
        arms=spec.arms,
        observations=(
            Observation(
                marker_id=spec.marker_id,
                kind=OPERATOR_OF_STATISTIC[spec.kind],
                window=spec.window,
                target=spec.target,
                sigma=spec.sigma,
                shape=spec.shape,
                band_halfwidth=spec.band_halfwidth,
            ),
        ),
        init_mode=spec.init_mode,
        weight=spec.weight,
    )


def from_rule(rule: PhysiologyRule) -> EvidenceItem:
    return EvidenceItem(
        name=rule.name,
        source=rule.source,
        description=rule.description,
        authority=Authority.LITERATURE,
        arms=rule.arms,
        observations=(Observation(marker_id="", kind=OperatorKind.HINGE),),
        init_mode=rule.init_mode,
        weight=rule.weight,
        rule=rule,
    )


def teacher_tape_item() -> EvidenceItem:
    """Documentary: the generator may tape these vitals, as a wide fence."""
    return EvidenceItem(
        name="teacher_wearable_tape",
        source="simulate_full_body (generator tape, not literature)",
        description="Banded pointwise supervision on wearable-frequency vitals only",
        authority=Authority.TEACHER,
        arms=(),
        observations=tuple(
            Observation(marker_id=m, kind=OperatorKind.TAPED_VITAL)
            for m in TEACHER_TAPE_MARKERS
        ),
    )


def teacher_distill_items() -> tuple[EvidenceItem, ...]:
    """Generator internals with no literature time series.

    Rate for the fast hidden states; day-scale level for the slow pair.
    """
    rate = tuple(m for m in TEACHER_DISTILL_MARKERS if m not in TEACHER_DISTILL_LONG_ONLY)
    return (
        EvidenceItem(
            name="teacher_internal_rate",
            source="simulate_full_body (generator internals, not literature)",
            description="Teacher-forced rate on unobserved internals no paper reports as a series",
            authority=Authority.TEACHER,
            arms=(),
            observations=tuple(
                Observation(marker_id=m, kind=OperatorKind.RATE) for m in rate
            ),
        ),
        EvidenceItem(
            name="teacher_internal_long_level",
            source="simulate_full_body (generator internals, not literature)",
            description="Day-scale level on slow states; not a 60-minute slope",
            authority=Authority.TEACHER,
            arms=(),
            observations=tuple(
                Observation(marker_id=m, kind=OperatorKind.LONG_LEVEL)
                for m in TEACHER_DISTILL_LONG_ONLY
            ),
        ),
    )


def teacher_meal_glucose_item() -> EvidenceItem:
    return EvidenceItem(
        name="teacher_meal_glucose",
        source="simulate_full_body per-patient 75 g meal (generator, wearable glucose)",
        description="Per-patient glucose peak rise, time-to-peak, and fasting level",
        authority=Authority.TEACHER,
        arms=(),
        observations=(
            Observation(marker_id="glucose", kind=OperatorKind.PEAK),
            Observation(marker_id="glucose", kind=OperatorKind.TIME_TO_PEAK),
            Observation(marker_id="glucose", kind=OperatorKind.WINDOW_MEAN),
        ),
    )


def dose_response_items() -> tuple[EvidenceItem, ...]:
    """Canonical literature dose-response (Wolever peak; GLP-1 rank)."""
    return (
        EvidenceItem(
            name="dose_glucose_peak",
            source="Wolever (1991, 1996)",
            description="Postprandial glucose Δpeak vs carb dose, 0.7 mg/dL/g",
            authority=Authority.LITERATURE,
            arms=(),
            observations=(
                Observation(
                    marker_id="glucose", kind=OperatorKind.DOSE_PEAK,
                    target=0.7, sigma=0.25,
                ),
            ),
        ),
        EvidenceItem(
            name="dose_insulin_peak",
            source="Wolever (1991, 1996); Brand-Miller (2003)",
            description="Postprandial insulin Δpeak vs carb dose",
            authority=Authority.LITERATURE,
            arms=(),
            observations=(
                Observation(
                    marker_id="insulin", kind=OperatorKind.DOSE_PEAK,
                    target=0.4, sigma=0.3,
                ),
            ),
        ),
        EvidenceItem(
            name="dose_glp1_rank",
            source="Brand-Miller (2003); incretin dose-ordering",
            description="GLP-1 peak ranks with carb dose",
            authority=Authority.LITERATURE,
            arms=(),
            observations=(
                Observation(marker_id="glp1", kind=OperatorKind.DOSE_RANK),
            ),
        ),
    )


def all_evidence() -> tuple[EvidenceItem, ...]:
    return (
        *(from_cohort(spec) for spec in ALL_COHORT_STATISTICS),
        *(from_rule(rule) for rule in PHYSIOLOGY_RULES),
        teacher_tape_item(),
        *teacher_distill_items(),
        teacher_meal_glucose_item(),
        *dose_response_items(),
    )


def sealed_evidence() -> tuple[EvidenceItem, ...]:
    """The held-out set (PLAN Phase 0): literature items absent from training.

    Only literature cohorts and rules are sealable — the teacher tape, the
    distill fill and the dose canon are not literature items and are never in
    the sealed set. The scoreboard scores these the way ``cgm_real`` is scored.
    """
    return (
        *(from_cohort(spec) for spec in SEALED_COHORT_STATISTICS),
        *(from_rule(rule) for rule in SEALED_PHYSIOLOGY_RULES),
    )


def real_evidence() -> tuple[EvidenceItem, ...]:
    """The individual-level rulers (PLAN Phase 0): measured people, each one
    calibrated on sparse early readings and predicted later under another
    protocol. The item name is the benchmark episode ``source``; the benchmark
    scores them against the population-median person and the person's own
    running mean. None is training evidence.
    """
    def ruler(name: str, source: str, description: str) -> EvidenceItem:
        return EvidenceItem(
            name=name, source=source, description=description,
            authority=Authority.REAL, arms=(),
            observations=tuple(
                Observation(marker_id=m, kind=OperatorKind.CALIBRATED_SERIES)
                for m in ("glucose", "hr")
            ),
        )

    return (
        ruler(
            "cgm_real",
            "One subject: FreeStyle Libre 3 CGM with Oura heart rate and sleep (2024-2025)",
            "Overnight windows: calibrate on the evening, predict the night",
        ),
        ruler(
            "cgmacros",
            "Gutierrez-Osuna, Kerr, Mortazavi, Das. CGMacros. PhysioNet 2025",
            "45 adults, healthy to type 2: calibrate across breakfast, predict the lunch response",
        ),
    )


ALL_EVIDENCE: tuple[EvidenceItem, ...] = all_evidence()
SEALED_EVIDENCE: tuple[EvidenceItem, ...] = sealed_evidence()
REAL_EVIDENCE: tuple[EvidenceItem, ...] = real_evidence()
