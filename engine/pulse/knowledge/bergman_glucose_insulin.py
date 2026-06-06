"""
Bergman minimal model for glucose-insulin dynamics.

Source: Bergman, Ider, Bowden, Cobelli (1979).
        "Quantitative estimation of insulin sensitivity."

Generates glucose-insulin trajectories showing meal responses with
realistic inter-individual variation in insulin sensitivity, glucose
effectiveness, and beta-cell responsiveness.
"""

import numpy as np

from ..types import STATE_DIM, MARKER_INDEX
from .base import Episode, KnowledgeContribution, CouplingPrior


def _randomize_params(rng: np.random.Generator) -> dict:
    def vary(val, spread=0.3):
        return val * np.exp(rng.normal(0, spread))

    return {
        "Sg": vary(0.018),
        "Si": vary(0.0002, 0.5),
        "Gb": vary(95.0, 0.15),
        "Ib": vary(10.0, 0.4),
        "p2": 0.03,
        "n": vary(0.15),
        "gamma": vary(0.015),
        "h": 80.0,
        "Gnb": vary(70.0),
        "k_gn": 0.03,
        "alpha_gn": 1.5,
        "FFA_b": vary(0.5),
        "lip_max": 0.033,
        "IC50_lip": 15.0,
        "k_ffa": 0.04,
        "BHB_b": vary(0.1),
        "keto_max": 0.005,
        "IC50_keto": 10.0,
        "k_bhb": 0.03,
        "Lac_b": vary(1.0),
        "k_lac": 0.02,
    }


def _meal_absorption(t: float, meal_time: float, carbs: float, rate: float = 0.03) -> float:
    dt = t - meal_time
    if dt < 0 or dt > 300:
        return 0.0
    return carbs * rate * rate * dt * np.exp(-rate * dt) * 3.0


def _generate_meals(n_days: int, rng: np.random.Generator, start_hour: float) -> list[tuple[float, float, float, float]]:
    meals = []
    for day in range(n_days):
        day_offset = day * 1440 - start_hour * 60
        meals.append((day_offset + rng.uniform(7 * 60, 9 * 60), rng.uniform(30, 70), rng.uniform(5, 20), rng.uniform(10, 30)))
        meals.append((day_offset + rng.uniform(12 * 60, 14 * 60), rng.uniform(40, 90), rng.uniform(10, 35), rng.uniform(15, 40)))
        meals.append((day_offset + rng.uniform(18 * 60, 20 * 60), rng.uniform(50, 100), rng.uniform(15, 40), rng.uniform(20, 50)))
        if rng.random() > 0.5:
            meals.append((day_offset + rng.uniform(10 * 60, 16 * 60), rng.uniform(10, 30), rng.uniform(3, 15), rng.uniform(2, 15)))
    meals = [(t, c, f, p) for t, c, f, p in meals if t >= 0]
    meals.sort()
    return meals


def _simulate(params: dict, meals: list[tuple], duration_min: int, start_hour: float, rng: np.random.Generator) -> np.ndarray:
    trajectory = np.full((duration_min, STATE_DIM), np.nan)

    G, I, Gn = params["Gb"], params["Ib"], params["Gnb"]
    FFA, BHB, Lac = params["FFA_b"], params["BHB_b"], params["Lac_b"]
    X = 0.0
    ns = 0.003

    for t in range(duration_min):
        Ra = sum(_meal_absorption(t, mt, mc) for mt, mc, _, _ in meals)
        p3 = params["Si"] * params["p2"]

        dG = -(params["Sg"] + X) * (G - params["Gb"]) + Ra
        dX = -params["p2"] * X + p3 * max(I - params["Ib"], 0)
        glucose_ratio = min(G / max(params["Gb"], 1.0), 1.0)
        effective_Ib = params["Ib"] * glucose_ratio
        dI = -params["n"] * (I - effective_Ib) + params["gamma"] * max(G - params["h"], 0)

        glucagon_stim = params["alpha_gn"] * max(params["Gb"] - G, 0) / max(params["Gb"], 1)
        glucagon_supp = 0.5 * I / (params["Ib"] + 10.0)
        dGn = -params["k_gn"] * (Gn - params["Gnb"]) + glucagon_stim - glucagon_supp

        lipolysis = params["lip_max"] / (1 + I / params["IC50_lip"])
        dFFA = lipolysis - params["k_ffa"] * FFA

        ketogenesis = params["keto_max"] * FFA / (1 + I / params["IC50_keto"])
        dBHB = ketogenesis - params["k_bhb"] * BHB

        dLac = -params["k_lac"] * (Lac - params["Lac_b"])

        G = max(G + dG + rng.normal(0, ns * 2), 20)
        X = X + dX
        I = max(I + dI + rng.normal(0, ns * 0.5), 0.1)
        Gn = max(Gn + dGn + rng.normal(0, ns * 1), 1)
        FFA = max(FFA + dFFA + rng.normal(0, ns * 0.01), 0.01)
        BHB = max(BHB + dBHB + rng.normal(0, ns * 0.005), 0.001)
        Lac = max(Lac + dLac + rng.normal(0, ns * 0.02), 0.1)

        trajectory[t, MARKER_INDEX["glucose"]] = G
        trajectory[t, MARKER_INDEX["insulin"]] = I
        trajectory[t, MARKER_INDEX["glucagon"]] = Gn
        trajectory[t, MARKER_INDEX["ffa"]] = FFA
        trajectory[t, MARKER_INDEX["bhb"]] = BHB
        trajectory[t, MARKER_INDEX["lactate"]] = Lac

    return trajectory


class BergmanGlucoseInsulin(KnowledgeContribution):
    def __init__(self):
        super().__init__(
            name="bergman_glucose_insulin",
            source="Bergman, Ider, Bowden, Cobelli (1979)",
            description="Minimal model glucose-insulin dynamics with meal responses",
        )

    def generate_episodes(self, n_episodes: int, rng: np.random.Generator) -> list[Episode]:
        episodes = []
        for _ in range(n_episodes):
            prng = np.random.default_rng(rng.integers(0, 2**32))
            params = _randomize_params(prng)
            n_days = 3
            start_hour = 6.0
            duration_min = n_days * 1440
            meals = _generate_meals(n_days, prng, start_hour)
            trajectory = _simulate(params, meals, duration_min, start_hour, prng)
            episodes.append(Episode(
                trajectory=trajectory,
                meals=meals,
                duration_min=duration_min,
                start_hour=start_hour,
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
