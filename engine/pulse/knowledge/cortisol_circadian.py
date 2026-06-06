"""
Cortisol circadian rhythm and HPA axis dynamics.

Sources:
  - Weitzman et al. (1971): "Twenty-four hour pattern of the episodic
    secretion of cortisol in normal subjects"
  - Lightman & Conway-Campbell (2010): "The crucial role of pulsatile
    activity of the HPA axis for continuous dynamic equilibration"

Generates cortisol time series with morning peaks, evening troughs,
and stress-responsive dynamics. Also generates appetite hormone patterns
(ghrelin, leptin) that couple with cortisol and feeding state.
"""

import numpy as np

from ..types import STATE_DIM, MARKER_INDEX
from .base import Episode, KnowledgeContribution


def _circadian(t_abs_min: float, amplitude: float, peak_hour: float) -> float:
    hour = t_abs_min / 60.0
    return amplitude * np.cos(2 * np.pi * (hour - peak_hour) / 24.0)


def _simulate(rng: np.random.Generator, n_days: int, start_hour: float) -> tuple[np.ndarray, list]:
    duration_min = n_days * 1440
    trajectory = np.full((duration_min, STATE_DIM), np.nan)

    cort_b = 12.0 * np.exp(rng.normal(0, 0.3))
    cort_amp = 5.0 * np.exp(rng.normal(0, 0.3))
    k_cort = 0.02
    k_acth = 0.04
    acth_b = 30.0 * np.exp(rng.normal(0, 0.25))
    acth_amp = 8.0 * np.exp(rng.normal(0, 0.3))
    k_acth_to_cort = 0.006
    cort_feedback_acth = 0.025

    ghr_b = 100.0 * np.exp(rng.normal(0, 0.3))
    k_ghr = 0.02
    lep_b = 10.0 * np.exp(rng.normal(0, 0.5))
    k_lep = 0.001
    lep_circ_amp = 2.0
    glp1_b = 10.0 * np.exp(rng.normal(0, 0.3))
    k_glp1 = 0.2

    Cort = cort_b
    ACTH = acth_b
    Ghr, Lep, GLP1 = ghr_b, lep_b, glp1_b
    ns = 0.003

    meals = []
    for day in range(n_days):
        day_offset = day * 1440 - start_hour * 60
        meals.append((day_offset + rng.uniform(7 * 60, 9 * 60), rng.uniform(40, 80), 0.0, 0.0))
        meals.append((day_offset + rng.uniform(12 * 60, 14 * 60), rng.uniform(50, 100), 0.0, 0.0))
        meals.append((day_offset + rng.uniform(18 * 60, 20 * 60), rng.uniform(60, 110), 0.0, 0.0))
    meals = [(t, c, f, p) for t, c, f, p in meals if t >= 0]
    meals.sort()

    for t in range(duration_min):
        t_abs = (start_hour * 60 + t) % 1440

        circ_cort = _circadian(t_abs, cort_amp, peak_hour=8.0)
        circ_acth = _circadian(t_abs, acth_amp, peak_hour=7.5)
        cort_target = max(cort_b + circ_cort, 1.0)
        acth_target = max(acth_b + circ_acth, 5.0)
        dACTH = -k_acth * (ACTH - acth_target)
        dACTH -= cort_feedback_acth * max(Cort - cort_b, 0)
        dCort = -k_cort * (Cort - cort_target) + k_acth_to_cort * max(ACTH, 0)

        meal_active = any(0 < (t - mt) < 120 for mt, _, _, _ in meals)
        insulin_proxy = 15.0 if meal_active else 8.0
        ghr_supp = insulin_proxy / (insulin_proxy + 20.0)
        ghr_prod = ghr_b * k_ghr * (1 - ghr_supp)
        dGhr = ghr_prod - k_ghr * Ghr

        circ_lep = _circadian(t_abs, lep_circ_amp, peak_hour=2.0)
        dLep = -k_lep * (Lep - lep_b - circ_lep)

        glp1_meal_sig = 5.0 if meal_active and any(0 < (t - mt) < 60 for mt, _, _, _ in meals) else 0.0
        glp1_prod = glp1_b * k_glp1 + glp1_meal_sig
        dGLP1 = glp1_prod - k_glp1 * GLP1

        ACTH = max(ACTH + dACTH + rng.normal(0, ns * 0.8), 5.0)
        Cort = max(Cort + dCort + rng.normal(0, ns * 0.3), 0.5)
        Ghr = max(Ghr + dGhr + rng.normal(0, ns * 2), 5)
        Lep = max(Lep + dLep + rng.normal(0, ns * 0.1), 0.5)
        GLP1 = max(GLP1 + dGLP1 + rng.normal(0, ns * 0.5), 1)

        trajectory[t, MARKER_INDEX["cortisol"]] = Cort
        trajectory[t, MARKER_INDEX["acth"]] = ACTH
        trajectory[t, MARKER_INDEX["ghrelin"]] = Ghr
        trajectory[t, MARKER_INDEX["leptin"]] = Lep
        trajectory[t, MARKER_INDEX["glp1"]] = GLP1

    return trajectory, meals


class CortisolCircadian(KnowledgeContribution):
    def __init__(self):
        super().__init__(
            name="cortisol_circadian",
            source="Weitzman et al. (1971); Lightman & Conway-Campbell (2010)",
            description="Cortisol circadian rhythm, HPA axis dynamics, and appetite hormone patterns",
        )

    def generate_episodes(self, n_episodes: int, rng: np.random.Generator) -> list[Episode]:
        episodes = []
        for _ in range(n_episodes):
            prng = np.random.default_rng(rng.integers(0, 2**32))
            n_days = 3
            start_hour = 6.0
            trajectory, meals = _simulate(prng, n_days, start_hour)

            sleep_wake = np.ones(n_days * 1440, dtype=np.float32)
            for day in range(n_days):
                day_start = day * 1440
                sleep_start = int((23 - start_hour) * 60) + day_start
                sleep_end = int((31 - start_hour) * 60) + day_start
                sleep_wake[max(0, sleep_start):min(len(sleep_wake), sleep_end)] = 0.0

            episodes.append(Episode(
                trajectory=trajectory,
                meals=meals,
                duration_min=n_days * 1440,
                start_hour=start_hour,
                sleep_wake=sleep_wake,
                source=self.name,
            ))
        return episodes

    def trajectory_loss_mode(self) -> str:
        return "huber"
