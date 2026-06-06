"""Textbook scenarios: sleep–wake transitions and HPA / circadian patterns."""

from __future__ import annotations

import numpy as np
import torch

from .base import ScenarioCheck, ScenarioResult, cold_model_trajectory, MARKER_INDEX
from ..full_body import PatientParams


def _sleep_wake_transition_mask(duration_min: int) -> np.ndarray:
    sleep_wake = np.ones(duration_min, dtype=np.float32)
    sleep_wake[3 * 60:] = 0.0
    sleep_wake[10 * 60:] = 1.0
    kernel = np.ones(20, dtype=np.float32) / 20.0
    return np.clip(np.convolve(sleep_wake, kernel, mode="same"), 0, 1).astype(np.float32)


def _car_sleep_mask(duration_min: int) -> np.ndarray:
    sleep_wake = np.zeros(duration_min, dtype=np.float32)
    sleep_wake[2 * 60:] = 1.0
    kernel = np.ones(20, dtype=np.float32) / 20.0
    return np.clip(np.convolve(sleep_wake, kernel, mode="same"), 0, 1).astype(np.float32)


def scenario_result_sleep_wake_transition(traj: np.ndarray) -> ScenarioResult:
    hr = traj[:, MARKER_INDEX["hr"]]
    sbp = traj[:, MARKER_INDEX["sbp"]]
    temp = traj[:, MARKER_INDEX["temp"]]
    rr = traj[:, MARKER_INDEX["rr"]]

    hr_evening = float(np.mean(hr[:120]))
    sbp_evening = float(np.mean(sbp[:120]))
    rr_evening = float(np.mean(rr[:120]))

    hr_sleep = float(np.mean(hr[300:540]))
    sbp_sleep = float(np.mean(sbp[300:540]))
    rr_sleep = float(np.mean(rr[300:540]))

    temp_evening = float(np.mean(temp[:120]))
    temp_nadir = float(np.min(temp[300:600]))

    checks = [
        ScenarioCheck(
            "hr_dips_during_sleep",
            "HR should drop during sleep compared to evening",
            hr_evening - hr_sleep > 5.0,
            hr_evening - hr_sleep, 5.0,
        ),
        ScenarioCheck(
            "sbp_dips_during_sleep",
            "SBP should dip during sleep",
            sbp_evening - sbp_sleep > 3.0,
            sbp_evening - sbp_sleep, 3.0,
        ),
        ScenarioCheck(
            "temp_drops_at_night",
            "Core temperature should reach nadir during night",
            temp_evening - temp_nadir > 0.1,
            temp_evening - temp_nadir, 0.1,
        ),
        ScenarioCheck(
            "rr_drops_during_sleep",
            "Respiratory rate should decrease during sleep",
            rr_evening - rr_sleep > 1.0,
            rr_evening - rr_sleep, 1.0,
        ),
    ]

    return ScenarioResult(
        name="sleep_wake_transition",
        source="Somers et al. (1993); Borbely (1982)",
        description="Sleep-wake transition — HR/BP dipping, temperature nadir, RR reduction",
        checks=checks,
        pass_rate=sum(1 for c in checks if c.passed) / len(checks),
    )


def scenario_result_cortisol_awakening(traj: np.ndarray) -> ScenarioResult:
    cort = traj[:, MARKER_INDEX["cortisol"]]
    acth = traj[:, MARKER_INDEX["acth"]]

    cort_sleep = float(np.mean(cort[:100]))
    cort_post_wake = float(np.max(cort[120:180]))
    acth_peak_time = int(np.argmax(acth[110:175])) + 110
    cort_peak_time = int(np.argmax(cort[120:190])) + 120

    checks = [
        ScenarioCheck(
            "cortisol_rises_after_waking",
            "Cortisol should rise substantially after waking",
            cort_post_wake > cort_sleep * 1.3,
            cort_post_wake / max(cort_sleep, 0.1), 1.3,
        ),
        ScenarioCheck(
            "cortisol_morning_peak",
            "Cortisol should reach >15 μg/dL in the morning peak",
            cort_post_wake > 15.0,
            cort_post_wake, 15.0,
        ),
        ScenarioCheck(
            "acth_precedes_cortisol",
            "ACTH peak should precede or coincide with cortisol peak",
            acth_peak_time <= cort_peak_time + 10,
            float(cort_peak_time - acth_peak_time), 0.0,
        ),
    ]

    return ScenarioResult(
        name="cortisol_awakening_response",
        source="Pruessner et al. (1997); Fries et al. (2009)",
        description="Cortisol awakening response — post-wake cortisol surge, ACTH-cortisol timing",
        checks=checks,
        pass_rate=sum(1 for c in checks if c.passed) / len(checks),
    )


def sleep_wake_transition_scenario(rng: np.random.Generator, trajectory_fn=None) -> ScenarioResult:
    del trajectory_fn
    params = PatientParams()
    duration_min = 720
    start_hour = 20.0
    sw = _sleep_wake_transition_mask(duration_min)
    traj = cold_model_trajectory(
        params, [], duration_min, start_hour,
        sleep_wake=sw,
        rng=np.random.default_rng(rng.integers(0, 2**32)),
    )
    return scenario_result_sleep_wake_transition(traj)


def cortisol_awakening_response_scenario(rng: np.random.Generator, trajectory_fn=None) -> ScenarioResult:
    del trajectory_fn
    params = PatientParams()
    duration_min = 480
    start_hour = 4.0
    sw = _car_sleep_mask(duration_min)
    traj = cold_model_trajectory(
        params, [], duration_min, start_hour,
        sleep_wake=sw,
        rng=np.random.default_rng(rng.integers(0, 2**32)),
    )
    return scenario_result_cortisol_awakening(traj)


def sleep_wake_transition_neural_on_model(model, rng: np.random.Generator, device: str = "cpu") -> ScenarioResult:
    from ...model import integrate
    from ...types import EMBEDDING_DIM

    params = PatientParams()
    duration_min = 720
    start_hour = 20.0
    sw = _sleep_wake_transition_mask(duration_min)
    prng = np.random.default_rng(rng.integers(0, 2**32))
    cold_traj = cold_model_trajectory(params, [], duration_min, start_hour, sleep_wake=sw, rng=prng)
    init = torch.tensor(cold_traj[0], dtype=torch.float32, device=device)
    emb = torch.zeros(EMBEDDING_DIM, dtype=torch.float32, device=device)
    sw_t = torch.tensor(sw, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        neural = integrate(
            model, init, emb, duration_min,
            dt=1.0, start_time_minutes=start_hour * 60.0,
            meals=[],
            sleep_wake=sw_t,
        )
    return scenario_result_sleep_wake_transition(neural.cpu().numpy())


def cortisol_awakening_neural_on_model(model, rng: np.random.Generator, device: str = "cpu") -> ScenarioResult:
    from ...model import integrate
    from ...types import EMBEDDING_DIM

    params = PatientParams()
    duration_min = 480
    start_hour = 4.0
    sw = _car_sleep_mask(duration_min)
    prng = np.random.default_rng(rng.integers(0, 2**32))
    cold_traj = cold_model_trajectory(params, [], duration_min, start_hour, sleep_wake=sw, rng=prng)
    init = torch.tensor(cold_traj[0], dtype=torch.float32, device=device)
    emb = torch.zeros(EMBEDDING_DIM, dtype=torch.float32, device=device)
    sw_t = torch.tensor(sw, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        neural = integrate(
            model, init, emb, duration_min,
            dt=1.0, start_time_minutes=start_hour * 60.0,
            meals=[],
            sleep_wake=sw_t,
        )
    return scenario_result_cortisol_awakening(neural.cpu().numpy())


SCENARIO_RUNNERS = [
    sleep_wake_transition_scenario,
    cortisol_awakening_response_scenario,
]
