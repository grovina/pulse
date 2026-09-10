"""
Cardiovascular dynamics with circadian variation, sleep modulation, and
autonomic coupling -- a VIEW of the full-body teacher (iter 97).

Sources:
  - Mancia (1993): "Ambulatory blood pressure monitoring: research and
    clinical applications"
  - Task Force of ESC/NASPE (1996): "Heart rate variability: standards
    of measurement, physiological interpretation and clinical use"
  - Somers et al. (1993): sleep-related cardiovascular changes

Through iter 96 this module carried its own cardiovascular ODE with the
constants full_body had long since re-measured: cortisol -> HR 1.0 bpm per
ug/dL (the 6.7x over-gain iter 96 cut to 0.15 with Adlan 2018), sleep_hr_frac
0.15 (0.09), sleep_hrv_gain 1.3 (1.08), temp_circ_amp 0.45 (0.25), and a
cortisol that was a bare cosine. It contributed 15% of the trajectory
distillation weight (item 3.12 of the 2026-09-04 review). It now generates its
episodes by running `simulate_full_body` and exposing only the cardiovascular,
thermal and respiratory markers. It is a diagnostic view, not a training
contribution.
"""

import numpy as np

from .base import Episode, KnowledgeContribution
from .evidence import mask_trajectory
from .full_body import (
    randomize_params, generate_meal_plan, generate_sleep_wake, generate_activity,
    simulate_full_body,
)

VIEW_MARKERS = ("hr", "hrv", "sbp", "dbp", "temp", "rr", "spo2")


class CardiovascularDynamics(KnowledgeContribution):
    def __init__(self, n_days: int = 3):
        super().__init__(
            name="cardiovascular_dynamics",
            source="Mancia (1993); ESC/NASPE Task Force (1996); Somers et al. (1993)",
            description="Cardiovascular, temperature, and respiratory dynamics with sleep modulation (full-body view)",
        )
        self.n_days = n_days

    def generate_episodes(self, n_episodes: int, rng: np.random.Generator) -> list[Episode]:
        episodes = []
        for _ in range(n_episodes):
            prng = np.random.default_rng(rng.integers(0, 2**32))
            params = randomize_params(prng)
            start_hour = 6.0
            duration_min = self.n_days * 1440
            meals = generate_meal_plan(self.n_days, prng, start_hour)
            sleep_wake = generate_sleep_wake(self.n_days, duration_min, start_hour, prng)
            activity = generate_activity(self.n_days, duration_min, start_hour, prng)
            trajectory, absorption_profile = simulate_full_body(
                params, meals, sleep_wake, activity, duration_min, start_hour, rng=prng,
            )
            episodes.append(Episode(
                trajectory=mask_trajectory(trajectory, VIEW_MARKERS),
                meals=meals,
                duration_min=duration_min,
                start_hour=start_hour,
                sleep_wake=sleep_wake,
                activity=activity,
                absorption_profile=absorption_profile,
                source=self.name,
            ))
        return episodes

    def trajectory_loss_mode(self) -> str:
        return "huber"
