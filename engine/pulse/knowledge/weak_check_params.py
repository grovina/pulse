"""
Numeric targets for weak physiology checks (meal response, coupling, circadian, sleep, sanity).

Single source for ``verifier.evaluate_weak_checks`` and ``training_verifier_surrogate_loss``
so training-time gradients match post-hoc gate evaluation (PRD: priors as traceable signal).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MealWeakCheckParams:
    glucose_slope_per_g_carb: float = 0.045
    glucose_floor: float = 2.8
    insulin_slope_per_g_carb: float = 0.014
    insulin_floor: float = 1.1
    pre_window: int = 15
    post_start: int = 15
    post_end: int = 120
    glucose_soft_scale: float = 8.0
    insulin_soft_scale: float = 4.0
    glucose_weight: float = 1.3
    insulin_weight: float = 1.2

    def glucose_target(self, carbs: float) -> float:
        return max(self.glucose_floor, carbs * self.glucose_slope_per_g_carb)

    def insulin_target(self, carbs: float) -> float:
        return max(self.insulin_floor, carbs * self.insulin_slope_per_g_carb)


MEAL = MealWeakCheckParams()


@dataclass(frozen=True)
class CouplingWeakCheckParams:
    hr_sbp_corr_min: float = 0.10
    hr_sbp_soft_scale: float = 0.25
    hr_sbp_weight: float = 1.0
    glucose_glucagon_inverse_min: float = 0.05
    glucose_glucagon_soft_scale: float = 0.25
    glucose_glucagon_weight: float = 0.9
    sbp_dbp_margin_min: float = 5.0
    sbp_dbp_soft_scale: float = 8.0
    sbp_dbp_weight: float = 1.3


COUPLING = CouplingWeakCheckParams()


@dataclass(frozen=True)
class CircadianWeakCheckParams:
    cortisol_morning_evening_min: float = 1.0
    cortisol_soft_scale: float = 3.0
    cortisol_weight: float = 0.8
    morning_hour_lo: float = 6.0
    morning_hour_hi: float = 10.0
    evening_hour_lo: float = 18.0
    evening_hour_hi: float = 22.0
    temp_afternoon_early_min: float = 0.05
    temp_soft_scale: float = 0.1
    temp_weight: float = 0.7
    early_temp_hour_lo: float = 4.0
    early_temp_hour_hi: float = 8.0
    afternoon_temp_hour_lo: float = 14.0
    afternoon_temp_hour_hi: float = 18.0


CIRCADIAN = CircadianWeakCheckParams()


@dataclass(frozen=True)
class SleepWeakCheckParams:
    min_trajectory_len: int = 1440
    daytime_hour_lo: float = 10.0
    daytime_hour_hi: float = 16.0
    nighttime_hour_lo: float = 0.0
    nighttime_hour_hi: float = 5.0
    hr_dip_min: float = 3.0
    hr_dip_soft_scale: float = 4.0
    hr_dip_weight: float = 0.6
    sbp_dip_min: float = 2.0
    sbp_dip_soft_scale: float = 4.0
    sbp_dip_weight: float = 0.5


SLEEP = SleepWeakCheckParams()


@dataclass(frozen=True)
class SanityRange:
    lo: float
    hi: float
    weight: float


SANITY_RANGES: tuple[tuple[str, SanityRange], ...] = (
    ("glucose", SanityRange(50.0, 260.0, 1.1)),
    ("hepatic_output", SanityRange(0.0, 12.0, 0.75)),
    ("acth", SanityRange(5.0, 120.0, 0.85)),
    ("hr", SanityRange(40.0, 170.0, 1.0)),
    ("sbp", SanityRange(85.0, 190.0, 1.0)),
    ("temp", SanityRange(35.7, 38.8, 1.0)),
    ("spo2", SanityRange(90.0, 100.0, 0.9)),
)


def sanity_range_scale(lo: float, hi: float) -> float:
    """Scale for exp(-violation/scale) in numpy verifier and torch surrogate."""
    return max((hi - lo) * 0.1, 1e-6)
