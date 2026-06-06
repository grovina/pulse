"""
Cardiovascular dynamics with circadian variation, sleep modulation,
and autonomic coupling.

Sources:
  - Mancia (1993): "Ambulatory blood pressure monitoring: research and
    clinical applications"
  - Task Force of ESC/NASPE (1996): "Heart rate variability: standards
    of measurement, physiological interpretation and clinical use"
  - Somers et al. (1993): sleep-related cardiovascular changes

Generates HR, HRV, SBP, DBP, temperature, and respiratory trajectories
with circadian patterns, sleep/wake modulation, and cortisol coupling.
"""

import numpy as np

from ..types import STATE_DIM, MARKER_INDEX
from .base import Episode, KnowledgeContribution
from .full_body import generate_sleep_wake


def _circadian(t_abs_min: float, amplitude: float, peak_hour: float) -> float:
    hour = t_abs_min / 60.0
    return amplitude * np.cos(2 * np.pi * (hour - peak_hour) / 24.0)


def _simulate(rng: np.random.Generator, n_days: int, start_hour: float) -> tuple[np.ndarray, np.ndarray]:
    duration_min = n_days * 1440
    trajectory = np.full((duration_min, STATE_DIM), np.nan)

    HR0 = 70.0 * np.exp(rng.normal(0, 0.15))
    HRV0 = 40.0 * np.exp(rng.normal(0, 0.4))
    SBP0 = 120.0 * np.exp(rng.normal(0, 0.1))
    DBP0 = 80.0 * np.exp(rng.normal(0, 0.1))
    T0 = 37.0 + rng.normal(0, 0.2)
    RR0 = 15.0 * np.exp(rng.normal(0, 0.15))
    SpO2_0 = min(100, max(94, 98 + rng.normal(0, 1)))

    cort_amp = 5.0
    hr_circ_amp = 5.0
    temp_circ_amp = float(np.clip(0.45 * np.exp(rng.normal(0, 0.15)), 0.3, 0.55))

    sleep_hr_frac = float(np.clip(0.15 * np.exp(rng.normal(0, 0.2)), 0.08, 0.25))
    sleep_hrv_gain = float(np.clip(1.3 * np.exp(rng.normal(0, 0.15)), 1.1, 1.6))
    sleep_bp_frac = float(np.clip(0.12 * np.exp(rng.normal(0, 0.2)), 0.06, 0.20))
    sleep_temp_drop = float(np.clip(0.15 * np.exp(rng.normal(0, 0.2)), 0.05, 0.25))
    sleep_rr_drop = float(np.clip(3.0 * np.exp(rng.normal(0, 0.2)), 1.5, 5.0))

    sleep_wake = generate_sleep_wake(n_days, duration_min, start_hour, rng)

    HR, HRV, SBP, DBP = HR0, HRV0, SBP0, DBP0
    T, RR, SpO2 = T0, RR0, SpO2_0
    ns = 0.003

    for t in range(duration_min):
        t_abs = (start_hour * 60 + t) % 1440
        sw = float(sleep_wake[t])
        sleep_depth = 1.0 - sw

        circ_cort = _circadian(t_abs, cort_amp, peak_hour=8.0)
        cort_dev = circ_cort

        circ_hr = _circadian(t_abs, hr_circ_amp, peak_hour=14.0)
        sleep_hr_shift = -sleep_hr_frac * HR0 * sleep_depth
        dHR = -0.3 * (HR - HR0 - circ_hr - sleep_hr_shift) + 0.3 * cort_dev

        hrv_sleep_mult = 1.0 + (sleep_hrv_gain - 1.0) * sleep_depth
        dHRV = -0.1 * (HRV - HRV0 * HR0 / max(HR, 40) * hrv_sleep_mult) - 0.2 * max(cort_dev, 0)

        sleep_sbp_shift = -sleep_bp_frac * SBP0 * sleep_depth
        sleep_dbp_shift = -sleep_bp_frac * DBP0 * sleep_depth
        dSBP = -0.2 * (SBP - SBP0 - sleep_sbp_shift) + 0.15 * cort_dev
        dDBP = -0.2 * (DBP - DBP0 - sleep_dbp_shift) + 0.15 * cort_dev * 0.5

        circ_temp = _circadian(t_abs, temp_circ_amp, peak_hour=16.0)
        sleep_temp_shift = -sleep_temp_drop * sleep_depth
        dT = -0.025 * (T - T0 - circ_temp - sleep_temp_shift)

        sleep_rr_shift = -sleep_rr_drop * sleep_depth
        dRR = -0.1 * (RR - RR0 - sleep_rr_shift)
        dSpO2 = -0.5 * (SpO2 - SpO2_0)

        HR = max(HR + dHR + rng.normal(0, ns * 1), 30)
        HRV = max(HRV + dHRV + rng.normal(0, ns * 1), 1)
        SBP = max(SBP + dSBP + rng.normal(0, ns * 0.5), 60)
        DBP = max(DBP + dDBP + rng.normal(0, ns * 0.3), 30)
        T = T + dT + rng.normal(0, ns * 0.01)
        RR = max(RR + dRR + rng.normal(0, ns * 0.2), 4)
        SpO2 = min(100, max(SpO2 + dSpO2 + rng.normal(0, ns * 0.1), 70))

        trajectory[t, MARKER_INDEX["hr"]] = HR
        trajectory[t, MARKER_INDEX["hrv"]] = HRV
        trajectory[t, MARKER_INDEX["sbp"]] = SBP
        trajectory[t, MARKER_INDEX["dbp"]] = DBP
        trajectory[t, MARKER_INDEX["temp"]] = T
        trajectory[t, MARKER_INDEX["rr"]] = RR
        trajectory[t, MARKER_INDEX["spo2"]] = SpO2

    return trajectory, sleep_wake


class CardiovascularDynamics(KnowledgeContribution):
    def __init__(self):
        super().__init__(
            name="cardiovascular_dynamics",
            source="Mancia (1993); ESC/NASPE Task Force (1996); Somers et al. (1993)",
            description="Cardiovascular, temperature, and respiratory dynamics with sleep modulation",
        )

    def generate_episodes(self, n_episodes: int, rng: np.random.Generator) -> list[Episode]:
        episodes = []
        for _ in range(n_episodes):
            prng = np.random.default_rng(rng.integers(0, 2**32))
            n_days = 3
            start_hour = 6.0
            trajectory, sleep_wake = _simulate(prng, n_days, start_hour)

            episodes.append(Episode(
                trajectory=trajectory,
                duration_min=n_days * 1440,
                start_hour=start_hour,
                sleep_wake=sleep_wake,
                source=self.name,
            ))
        return episodes

    def trajectory_loss_mode(self) -> str:
        return "huber"
