"""
Bergman minimal model for glucose-insulin dynamics -- a VIEW of the full-body
teacher (iter 97).

Source: Bergman, Ider, Bowden, Cobelli (1979).
        "Quantitative estimation of insulin sensitivity."

Through iter 96 this module carried its own copy of the glucose-insulin ODE, and
that copy taught what full_body had since corrected: `h` fixed at 95 while Gb was
sampled 63-138 (48% of patients had GSIR that never switched off at fasting), the
iter-92 kernel rate and gain, no glycogen pool, no incretin. It contributed 22%
of the trajectory distillation weight (item 3.12 of the 2026-09-04 review). The
PRD calls generators expendable: this one now generates its episodes by running
`simulate_full_body` and exposing only the metabolic markers, so the metabolic
view and the coupled teacher can never disagree again. Its contribution NAME,
loss mode and coupling priors are unchanged, so training specs keep working.
"""

import numpy as np

from ..types import STATE_DIM, MARKER_INDEX
from .base import Episode, KnowledgeContribution, CouplingPrior
from .full_body import (
    randomize_params, generate_meal_plan, generate_sleep_wake, generate_activity,
    simulate_full_body,
)

VIEW_MARKERS = ("glucose", "insulin", "glucagon", "ffa", "bhb", "lactate")


def _masked_view(trajectory: np.ndarray, markers: tuple[str, ...]) -> np.ndarray:
    """Keep only `markers`; every other column is NaN (unsupervised in this view)."""
    view = np.full_like(trajectory, np.nan)
    for m in markers:
        view[:, MARKER_INDEX[m]] = trajectory[:, MARKER_INDEX[m]]
    return view


class BergmanGlucoseInsulin(KnowledgeContribution):
    def __init__(self, n_days: int = 3):
        super().__init__(
            name="bergman_glucose_insulin",
            source="Bergman, Ider, Bowden, Cobelli (1979)",
            description="Minimal model glucose-insulin dynamics with meal responses (full-body view)",
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
                trajectory=_masked_view(trajectory, VIEW_MARKERS),
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

    def coupling_priors(self) -> list[CouplingPrior]:
        return [
            CouplingPrior("glucose", "insulin", sign=+1, magnitude_range=(0.001, 0.02)),
            CouplingPrior("insulin", "glucose", sign=-1, magnitude_range=(0.0001, 0.001)),
            CouplingPrior("glucose", "glucagon", sign=-1, magnitude_range=(0.001, 0.01)),
        ]
