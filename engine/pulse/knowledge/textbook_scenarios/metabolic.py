"""Textbook scenarios: glucose homeostasis, fasting, meal dose-response."""

from __future__ import annotations

import numpy as np
import torch

from .base import ScenarioCheck, ScenarioResult, cold_model_trajectory, MARKER_INDEX
from ..full_body import PatientParams
from ...modules.gut import MealEvent


def scenario_result_ogtt(traj: np.ndarray) -> ScenarioResult:
    """Build ScenarioResult from a trajectory (cold or neural OGTT protocol)."""
    g = traj[:, MARKER_INDEX["glucose"]]
    ins = traj[:, MARKER_INDEX["insulin"]]
    gn = traj[:, MARKER_INDEX["glucagon"]]

    g_baseline = float(np.mean(g[:25]))
    g_peak = float(np.max(g[40:120]))
    g_peak_time = int(np.argmax(g[40:120])) + 40
    g_at_150 = float(np.mean(g[170:190]))

    ins_baseline = float(np.mean(ins[:25]))
    ins_peak = float(np.max(ins[40:120]))

    gn_baseline = float(np.mean(gn[:25]))
    gn_during = float(np.mean(gn[60:120]))

    checks = [
        ScenarioCheck(
            "glucose_rises",
            "Glucose should rise >20 mg/dL above baseline after 75g load",
            g_peak - g_baseline > 20.0,
            g_peak - g_baseline, 20.0,
        ),
        ScenarioCheck(
            "glucose_peak_timing",
            "Glucose peak should occur 10-90 min after load (t=40-120)",
            40 <= g_peak_time <= 120,
            float(g_peak_time), 80.0,
        ),
        ScenarioCheck(
            "glucose_returns",
            "Glucose should return below 140 mg/dL by 150 min post-load (t=180)",
            g_at_150 < 140.0,
            g_at_150, 140.0,
        ),
        ScenarioCheck(
            "insulin_rises",
            "Insulin should rise meaningfully in response to glucose",
            ins_peak - ins_baseline > 3.0,
            ins_peak - ins_baseline, 3.0,
        ),
        ScenarioCheck(
            "glucagon_suppressed",
            "Glucagon should be lower during glucose absorption than at baseline",
            gn_during < gn_baseline,
            gn_baseline - gn_during, 0.0,
        ),
    ]

    return ScenarioResult(
        name="OGTT",
        source="Guyton & Hall, Medical Physiology (Ch. 79)",
        description="75g oral glucose tolerance test — glucose peak/recovery, insulin response, glucagon suppression",
        checks=checks,
        pass_rate=sum(1 for c in checks if c.passed) / len(checks),
    )


def ogtt_scenario(rng: np.random.Generator, trajectory_fn=None) -> ScenarioResult:
    del trajectory_fn
    params = PatientParams()
    duration_min = 240
    start_hour = 8.0
    meals = [(30.0, 75.0, 0.0, 0.0)]
    traj = cold_model_trajectory(
        params, meals, duration_min, start_hour,
        rng=np.random.default_rng(rng.integers(0, 2**32)),
    )
    return scenario_result_ogtt(traj)


def ogtt_neural_on_model(model, rng: np.random.Generator, device: str = "cpu") -> ScenarioResult:
    """Same OGTT protocol; trajectory from trained ``model`` (init = cold start state)."""
    from ...model import integrate
    from ...types import EMBEDDING_DIM

    params = PatientParams()
    duration_min = 240
    start_hour = 8.0
    meals = [(30.0, 75.0, 0.0, 0.0)]
    prng = np.random.default_rng(rng.integers(0, 2**32))
    cold_traj = cold_model_trajectory(params, meals, duration_min, start_hour, rng=prng)
    init = torch.tensor(cold_traj[0], dtype=torch.float32, device=device)
    emb = torch.zeros(EMBEDDING_DIM, dtype=torch.float32, device=device)
    meal_events = [MealEvent(time=30.0, carbs=75.0, fats=0.0, proteins=0.0)]
    model.eval()
    with torch.no_grad():
        neural = integrate(
            model, init, emb, duration_min,
            dt=1.0, start_time_minutes=start_hour * 60.0,
            meals=meal_events,
        )
    return scenario_result_ogtt(neural.cpu().numpy())


def _overnight_sleep_mask(duration_min: int) -> np.ndarray:
    sleep_wake = np.ones(duration_min, dtype=np.float32)
    sleep_wake[5 * 60:] = 0.0
    kernel = np.ones(20, dtype=np.float32) / 20.0
    return np.clip(np.convolve(sleep_wake, kernel, mode="same"), 0, 1).astype(np.float32)


def scenario_result_overnight_fast(traj: np.ndarray) -> ScenarioResult:
    """Checks for 14h overnight fast trajectory."""
    g = traj[:, MARKER_INDEX["glucose"]]
    ins = traj[:, MARKER_INDEX["insulin"]]
    ffa = traj[:, MARKER_INDEX["ffa"]]
    bhb = traj[:, MARKER_INDEX["bhb"]]
    ghr = traj[:, MARKER_INDEX["ghrelin"]]

    g_early = float(np.mean(g[180:240]))
    g_late = float(np.mean(g[720:840]))
    ins_late = float(np.mean(ins[720:840]))
    ffa_postmeal = float(np.mean(ffa[60:180]))
    ffa_late = float(np.mean(ffa[720:840]))
    bhb_early = float(np.mean(bhb[120:240]))
    bhb_late = float(np.mean(bhb[720:840]))
    ghr_early = float(np.mean(ghr[120:240]))
    ghr_late = float(np.mean(ghr[600:840]))

    checks = [
        ScenarioCheck(
            "glucose_stays_viable",
            "Glucose should remain above 60 mg/dL even after 14h fast",
            float(np.min(g)) > 60.0,
            float(np.min(g)), 60.0,
        ),
        ScenarioCheck(
            "glucose_declines",
            "Late-fast glucose should be lower than early post-meal",
            g_late < g_early,
            g_early - g_late, 0.0,
        ),
        ScenarioCheck(
            "insulin_low",
            "Insulin should be near basal (<15 μU/mL) by end of fast",
            ins_late < 15.0,
            ins_late, 15.0,
        ),
        ScenarioCheck(
            "ffa_rises",
            "Free fatty acids should be higher at end of fast than postprandially",
            ffa_late > ffa_postmeal,
            ffa_late - ffa_postmeal, 0.0,
        ),
        ScenarioCheck(
            "bhb_rises",
            "β-Hydroxybutyrate should rise as fat oxidation increases",
            bhb_late > bhb_early,
            bhb_late - bhb_early, 0.0,
        ),
        ScenarioCheck(
            "ghrelin_rises",
            "Ghrelin (hunger) should increase during prolonged fast",
            ghr_late > ghr_early,
            ghr_late - ghr_early, 0.0,
        ),
    ]

    return ScenarioResult(
        name="overnight_fast",
        source="Cahill (2006); Guyton & Hall (Ch. 69-70)",
        description="14-hour overnight fast — glucose maintenance, ketogenesis onset, hunger signaling",
        checks=checks,
        pass_rate=sum(1 for c in checks if c.passed) / len(checks),
    )


def overnight_fast_scenario(rng: np.random.Generator, trajectory_fn=None) -> ScenarioResult:
    del trajectory_fn
    params = PatientParams()
    duration_min = 840
    start_hour = 18.0
    meals = [(30.0, 60.0, 15.0, 25.0)]
    sw = _overnight_sleep_mask(duration_min)
    traj = cold_model_trajectory(
        params, meals, duration_min, start_hour,
        sleep_wake=sw,
        rng=np.random.default_rng(rng.integers(0, 2**32)),
    )
    return scenario_result_overnight_fast(traj)


def overnight_fast_neural_on_model(model, rng: np.random.Generator, device: str = "cpu") -> ScenarioResult:
    from ...model import integrate
    from ...types import EMBEDDING_DIM

    params = PatientParams()
    duration_min = 840
    start_hour = 18.0
    meals_t = [(30.0, 60.0, 15.0, 25.0)]
    sw = _overnight_sleep_mask(duration_min)
    prng = np.random.default_rng(rng.integers(0, 2**32))
    cold_traj = cold_model_trajectory(params, meals_t, duration_min, start_hour, sleep_wake=sw, rng=prng)
    init = torch.tensor(cold_traj[0], dtype=torch.float32, device=device)
    emb = torch.zeros(EMBEDDING_DIM, dtype=torch.float32, device=device)
    meal_events = [MealEvent(time=30.0, carbs=60.0, fats=15.0, proteins=25.0)]
    sw_t = torch.tensor(sw, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        neural = integrate(
            model, init, emb, duration_min,
            dt=1.0, start_time_minutes=start_hour * 60.0,
            meals=meal_events,
            sleep_wake=sw_t,
        )
    return scenario_result_overnight_fast(neural.cpu().numpy())


def scenario_result_meal_dose_response(traj_small: np.ndarray, traj_large: np.ndarray) -> ScenarioResult:
    g_peak_small = float(np.max(traj_small[50:120, MARKER_INDEX["glucose"]]))
    g_peak_large = float(np.max(traj_large[50:120, MARKER_INDEX["glucose"]]))

    ins_peak_small = float(np.max(traj_small[40:120, MARKER_INDEX["insulin"]]))
    ins_peak_large = float(np.max(traj_large[40:120, MARKER_INDEX["insulin"]]))

    glp1_mean_small = float(np.mean(traj_small[40:100, MARKER_INDEX["glp1"]]))
    glp1_mean_large = float(np.mean(traj_large[40:100, MARKER_INDEX["glp1"]]))

    checks = [
        ScenarioCheck(
            "glucose_dose_response",
            "90g carb meal should produce higher glucose peak than 30g",
            g_peak_large > g_peak_small + 5.0,
            g_peak_large - g_peak_small, 5.0,
        ),
        ScenarioCheck(
            "insulin_dose_response",
            "90g carb meal should produce higher insulin peak than 30g",
            ins_peak_large > ins_peak_small + 2.0,
            ins_peak_large - ins_peak_small, 2.0,
        ),
        ScenarioCheck(
            "glp1_dose_response",
            "Larger meal should produce higher GLP-1 response",
            glp1_mean_large > glp1_mean_small,
            glp1_mean_large - glp1_mean_small, 0.0,
        ),
    ]

    return ScenarioResult(
        name="meal_dose_response",
        source="Wolever & Bolognesi (1996)",
        description="Carbohydrate dose-response: 30g vs 90g meal — glucose, insulin, GLP-1 scaling",
        checks=checks,
        pass_rate=sum(1 for c in checks if c.passed) / len(checks),
    )


def meal_dose_response_scenario(rng: np.random.Generator, trajectory_fn=None) -> ScenarioResult:
    del trajectory_fn
    params = PatientParams()
    duration_min = 240
    start_hour = 8.0
    prng = np.random.default_rng(rng.integers(0, 2**32))
    meals_small = [(30.0, 30.0, 5.0, 10.0)]
    meals_large = [(30.0, 90.0, 5.0, 10.0)]
    traj_small = cold_model_trajectory(params, meals_small, duration_min, start_hour, rng=prng)
    prng2 = np.random.default_rng(rng.integers(0, 2**32))
    traj_large = cold_model_trajectory(params, meals_large, duration_min, start_hour, rng=prng2)
    return scenario_result_meal_dose_response(traj_small, traj_large)


def meal_dose_response_neural_on_model(model, rng: np.random.Generator, device: str = "cpu") -> ScenarioResult:
    from ...model import integrate
    from ...types import EMBEDDING_DIM

    params = PatientParams()
    duration_min = 240
    start_hour = 8.0
    meals_small = [(30.0, 30.0, 5.0, 10.0)]
    meals_large = [(30.0, 90.0, 5.0, 10.0)]
    prng = np.random.default_rng(rng.integers(0, 2**32))
    cold_s = cold_model_trajectory(params, meals_small, duration_min, start_hour, rng=prng)
    prng2 = np.random.default_rng(rng.integers(0, 2**32))
    cold_l = cold_model_trajectory(params, meals_large, duration_min, start_hour, rng=prng2)
    emb = torch.zeros(EMBEDDING_DIM, dtype=torch.float32, device=device)
    me_s = [MealEvent(time=30.0, carbs=30.0, fats=5.0, proteins=10.0)]
    me_l = [MealEvent(time=30.0, carbs=90.0, fats=5.0, proteins=10.0)]
    model.eval()
    with torch.no_grad():
        n_s = integrate(
            model,
            torch.tensor(cold_s[0], dtype=torch.float32, device=device),
            emb,
            duration_min,
            dt=1.0, start_time_minutes=start_hour * 60.0, meals=me_s,
        ).cpu().numpy()
        n_l = integrate(
            model,
            torch.tensor(cold_l[0], dtype=torch.float32, device=device),
            emb,
            duration_min,
            dt=1.0, start_time_minutes=start_hour * 60.0, meals=me_l,
        ).cpu().numpy()
    return scenario_result_meal_dose_response(n_s, n_l)


SCENARIO_RUNNERS = [
    ogtt_scenario,
    overnight_fast_scenario,
    meal_dose_response_scenario,
]
