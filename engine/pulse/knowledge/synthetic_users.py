"""
Synthetic user profiles for calibration-under-sparsity evaluation.

Each profile represents a distinct physiological archetype with specific
parameter overrides. Sparse observation schedules mimic what a real user
would actually log (a few glucose reads, occasional HR, meal logs, etc.)
over a 14-day period.
"""

from dataclasses import dataclass

import numpy as np

from ..types import STATE_DIM, MARKER_INDEX, MARKER_IDS
from ..benchmark import MeasurementPoint
from ..modules.gut import MealEvent
from .full_body import (
    PatientParams, randomize_params, generate_meal_plan,
    generate_sleep_wake, generate_activity, simulate_full_body,
)


@dataclass
class UserProfile:
    name: str
    description: str
    param_overrides: dict


PROFILES = [
    UserProfile(
        name="healthy_active",
        description="Fit adult with regular exercise, normal metabolic function",
        param_overrides={
            "Gb": 90.0, "Si": 0.0003, "HR0": 60.0, "HRV0": 55.0,
            "SBP0": 115.0, "DBP0": 72.0,
        },
    ),
    UserProfile(
        name="sedentary_office",
        description="Sedentary office worker with slightly elevated baseline glucose",
        param_overrides={
            "Gb": 102.0, "Si": 0.00015, "HR0": 78.0, "HRV0": 30.0,
            "SBP0": 128.0, "DBP0": 84.0, "Cort_b": 15.0,
        },
    ),
    UserProfile(
        name="insulin_resistant",
        description="Pre-diabetic with reduced insulin sensitivity",
        param_overrides={
            "Gb": 115.0, "Si": 0.00008, "Ib": 18.0, "gamma": 0.003,
            "HR0": 82.0, "HRV0": 25.0, "SBP0": 135.0, "DBP0": 88.0,
            "FFA_b": 0.7, "BHB_b": 0.15,
        },
    ),
    UserProfile(
        name="athletic_lean",
        description="Endurance athlete with high insulin sensitivity and low resting HR",
        param_overrides={
            "Gb": 85.0, "Si": 0.0004, "HR0": 52.0, "HRV0": 70.0,
            "SBP0": 108.0, "DBP0": 68.0, "RR0": 12.0,
            "Lac_b": 0.8,
        },
    ),
    UserProfile(
        name="shift_worker",
        description="Night shift worker with disrupted circadian rhythm",
        param_overrides={
            "Cort_b": 14.0, "cort_circ_amp": 3.0,
            "HR0": 75.0, "HRV0": 32.0, "T0": 36.8,
            "Lep_b": 13.0, "Ghr_b": 120.0,
        },
    ),
    UserProfile(
        name="anxious_stress",
        description="Chronically stressed with elevated cortisol baseline",
        param_overrides={
            "Cort_b": 18.0, "cort_circ_amp": 6.0,
            "ACTH_b": 42.0, "acth_circ_amp": 12.0,
            "HR0": 80.0, "HRV0": 28.0,
            "SBP0": 130.0, "DBP0": 85.0,
            "Gb": 100.0,
        },
    ),
    UserProfile(
        name="elderly_moderate",
        description="Older adult with reduced metabolic reserve",
        param_overrides={
            "Gb": 105.0, "Si": 0.00012, "Ib": 14.0,
            "HR0": 72.0, "HRV0": 20.0,
            "SBP0": 140.0, "DBP0": 82.0,
            "SpO2_0": 96.0, "RR0": 17.0,
        },
    ),
    UserProfile(
        name="young_healthy",
        description="Young healthy adult with robust physiological responses",
        param_overrides={
            "Gb": 88.0, "Si": 0.00025, "HR0": 68.0, "HRV0": 50.0,
            "SBP0": 112.0, "DBP0": 70.0, "T0": 37.1,
        },
    ),
]


@dataclass
class ObservationSchedule:
    """Defines how a synthetic user logs data."""
    glucose_per_day: float = 2.5
    hr_per_day: float = 1.0
    temp_per_day: float = 0.3
    bp_per_day: float = 0.2
    meal_log_probability: float = 0.7
    skip_day_probability: float = 0.15


DEFAULT_SCHEDULE = ObservationSchedule()

SCHEDULES_BY_PROFILE = {
    "healthy_active": ObservationSchedule(
        glucose_per_day=2.0, hr_per_day=3.0, meal_log_probability=0.8,
    ),
    "insulin_resistant": ObservationSchedule(
        glucose_per_day=4.0, hr_per_day=1.0, bp_per_day=0.5,
        meal_log_probability=0.6,
    ),
    "athletic_lean": ObservationSchedule(
        glucose_per_day=1.5, hr_per_day=4.0, meal_log_probability=0.9,
        temp_per_day=0.5,
    ),
    "shift_worker": ObservationSchedule(
        glucose_per_day=2.0, hr_per_day=1.0, meal_log_probability=0.5,
        skip_day_probability=0.3,
    ),
}


def generate_sparse_observations(
    ground_truth: np.ndarray,
    meals: list[tuple[float, float, float, float]],
    schedule: ObservationSchedule,
    n_days: int,
    rng: np.random.Generator,
    measurement_noise: float = 0.02,
) -> list[MeasurementPoint]:
    """Sample sparse observations from a dense trajectory."""
    duration_min = len(ground_truth)
    observations = []

    for day in range(n_days):
        if rng.random() < schedule.skip_day_probability:
            continue

        day_start = day * 1440

        # Glucose readings
        n_glucose = int(schedule.glucose_per_day) + (1 if rng.random() < (schedule.glucose_per_day % 1) else 0)
        for _ in range(n_glucose):
            t = day_start + int(rng.uniform(6 * 60, 22 * 60))
            if 0 <= t < duration_min:
                val = float(ground_truth[t, MARKER_INDEX["glucose"]])
                val *= (1 + rng.normal(0, measurement_noise))
                observations.append(MeasurementPoint(time=t, marker_id="glucose", value=val))

        # Heart rate
        n_hr = int(schedule.hr_per_day) + (1 if rng.random() < (schedule.hr_per_day % 1) else 0)
        for _ in range(n_hr):
            t = day_start + int(rng.uniform(7 * 60, 23 * 60))
            if 0 <= t < duration_min:
                val = float(ground_truth[t, MARKER_INDEX["hr"]])
                val *= (1 + rng.normal(0, measurement_noise))
                observations.append(MeasurementPoint(time=t, marker_id="hr", value=val))

        # Temperature
        if rng.random() < schedule.temp_per_day:
            t = day_start + int(rng.uniform(8 * 60, 20 * 60))
            if 0 <= t < duration_min:
                val = float(ground_truth[t, MARKER_INDEX["temp"]])
                val += rng.normal(0, 0.1)
                observations.append(MeasurementPoint(time=t, marker_id="temp", value=val))

        # Blood pressure
        if rng.random() < schedule.bp_per_day:
            t = day_start + int(rng.uniform(8 * 60, 20 * 60))
            if 0 <= t < duration_min:
                sbp = float(ground_truth[t, MARKER_INDEX["sbp"]])
                dbp = float(ground_truth[t, MARKER_INDEX["dbp"]])
                observations.append(MeasurementPoint(time=t, marker_id="sbp", value=sbp * (1 + rng.normal(0, measurement_noise))))
                observations.append(MeasurementPoint(time=t, marker_id="dbp", value=dbp * (1 + rng.normal(0, measurement_noise))))

    observations.sort(key=lambda o: o.time)
    return observations


def generate_synthetic_users(
    n_days: int = 14,
    seed: int = 123,
) -> list[dict]:
    """Generate all synthetic users with ground truth and sparse observations."""
    rng = np.random.default_rng(seed)
    users = []

    for profile in PROFILES:
        prng = np.random.default_rng(rng.integers(0, 2**32))
        params = randomize_params(prng)

        for attr, val in profile.param_overrides.items():
            setattr(params, attr, val)

        start_hour = 6.0
        duration_min = n_days * 1440
        meals = generate_meal_plan(n_days, prng, start_hour)
        sleep_wake = generate_sleep_wake(n_days, duration_min, start_hour, prng)
        activity = generate_activity(n_days, duration_min, start_hour, prng)

        trajectory, absorption_profile = simulate_full_body(
            params, meals, sleep_wake, activity,
            duration_min, start_hour, rng=prng,
        )

        schedule = SCHEDULES_BY_PROFILE.get(profile.name, DEFAULT_SCHEDULE)
        sparse_obs = generate_sparse_observations(
            trajectory, meals, schedule, n_days, prng,
        )

        # Logged meals (subset with noise)
        logged_meals = []
        for mt, mc, mf, mp in meals:
            if prng.random() < schedule.meal_log_probability:
                logged_meals.append(MealEvent(
                    time=mt + prng.normal(0, 5),
                    carbs=mc * prng.uniform(0.8, 1.2),
                    fats=mf * prng.uniform(0.7, 1.3),
                    proteins=mp * prng.uniform(0.7, 1.3),
                ))

        users.append({
            "profile_name": profile.name,
            "description": profile.description,
            "ground_truth": trajectory,
            "initial_state": trajectory[0].copy(),
            "meals": logged_meals,
            "duration_min": duration_min,
            "sparse_observations": sparse_obs,
            "sleep_wake": sleep_wake,
            "activity": activity,
        })

    return users
