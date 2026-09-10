"""
Cortisol circadian rhythm and HPA axis dynamics -- a VIEW of the full-body
teacher (iter 97).

Sources:
  - Weitzman et al. (1971): "Twenty-four hour pattern of the episodic
    secretion of cortisol in normal subjects"
  - Lightman & Conway-Campbell (2010): "The crucial role of pulsatile
    activity of the HPA axis for continuous dynamic equilibration"

Through iter 96 this module carried its own HPA/appetite ODE: cortisol double-
driven (its own circadian target plus an additive ACTH term; nadir 16.5 against
full_body's 4), ghrelin suppressed by ABSOLUTE insulin (night ghrelin 66 against
100), `k_lep = 0.001` (leptin phase-lagged 5 h). It contributed 18% of the
trajectory distillation weight and taught the student exactly what iters 91 and
96 had removed from the coupled teacher (item 3.12 of the 2026-09-04 review).
It now generates its episodes by running `simulate_full_body` and exposing only
the stress and appetite markers. It is a diagnostic view, not a training
contribution.
"""

import numpy as np

from .base import Episode, KnowledgeContribution
from .evidence import mask_trajectory
from .full_body import (
    randomize_params, generate_meal_plan, generate_sleep_wake, generate_activity,
    simulate_full_body,
)

VIEW_MARKERS = ("cortisol", "acth", "ghrelin", "leptin", "glp1")


class CortisolCircadian(KnowledgeContribution):
    def __init__(self, n_days: int = 3):
        super().__init__(
            name="cortisol_circadian",
            source="Weitzman et al. (1971); Lightman & Conway-Campbell (2010)",
            description="Cortisol circadian rhythm, HPA axis dynamics, and appetite hormone patterns (full-body view)",
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
