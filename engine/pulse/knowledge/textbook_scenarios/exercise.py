"""Textbook scenarios: exercise physiology."""

from __future__ import annotations

import numpy as np
import torch

from .base import ScenarioCheck, ScenarioResult, cold_model_trajectory, MARKER_INDEX
from ..full_body import PatientParams


def _exercise_activity(duration_min: int) -> np.ndarray:
    activity = np.full(duration_min, 0.05, dtype=np.float32)
    activity[60:90] = 0.6
    return activity


def _z2_activity(duration_min: int) -> np.ndarray:
    # Z2 = conversational pace, sub-lactate-threshold. activity=0.4 is below
    # the existing "moderate" bout's 0.6 (which sits closer to Z3/Z4).
    activity = np.full(duration_min, 0.05, dtype=np.float32)
    activity[60:90] = 0.4
    return activity


def scenario_result_exercise_bout(traj: np.ndarray) -> ScenarioResult:
    hr = traj[:, MARKER_INDEX["hr"]]
    lac = traj[:, MARKER_INDEX["lactate"]]
    temp = traj[:, MARKER_INDEX["temp"]]
    rr = traj[:, MARKER_INDEX["rr"]]

    hr_rest = float(np.mean(hr[:50]))
    hr_exercise = float(np.mean(hr[65:90]))
    hr_recovery = float(np.mean(hr[105:120]))

    lac_rest = float(np.mean(lac[:50]))
    lac_exercise = float(np.max(lac[60:100]))

    temp_rest = float(np.mean(temp[:50]))
    temp_exercise = float(np.max(temp[60:100]))

    rr_rest = float(np.mean(rr[:50]))
    rr_exercise = float(np.mean(rr[65:90]))

    checks = [
        ScenarioCheck(
            "hr_rises_during_exercise",
            "HR should increase >15 bpm during moderate exercise",
            hr_exercise - hr_rest > 15.0,
            hr_exercise - hr_rest, 15.0,
        ),
        ScenarioCheck(
            "hr_recovers",
            "HR should recover substantially within 15 min post-exercise",
            hr_recovery < hr_rest + 10.0,
            hr_recovery, hr_rest + 10.0,
        ),
        ScenarioCheck(
            "lactate_rises",
            "Lactate should increase during moderate exercise",
            lac_exercise > lac_rest + 0.3,
            lac_exercise - lac_rest, 0.3,
        ),
        ScenarioCheck(
            "temp_rises",
            "Core temperature should increase during exercise",
            temp_exercise > temp_rest + 0.1,
            temp_exercise - temp_rest, 0.1,
        ),
        ScenarioCheck(
            "rr_rises",
            "Respiratory rate should increase during exercise",
            rr_exercise > rr_rest + 2.0,
            rr_exercise - rr_rest, 2.0,
        ),
    ]

    return ScenarioResult(
        name="exercise_bout",
        source="McArdle, Katch & Katch, Exercise Physiology",
        description="30-minute moderate exercise bout — cardiovascular, metabolic, and thermal responses",
        checks=checks,
        pass_rate=sum(1 for c in checks if c.passed) / len(checks),
    )


def exercise_bout_scenario(rng: np.random.Generator, trajectory_fn=None) -> ScenarioResult:
    del trajectory_fn
    params = PatientParams()
    duration_min = 180
    start_hour = 10.0
    meals: list[tuple[float, float, float, float]] = []
    act = _exercise_activity(duration_min)
    traj = cold_model_trajectory(
        params, meals, duration_min, start_hour,
        activity=act,
        rng=np.random.default_rng(rng.integers(0, 2**32)),
    )
    return scenario_result_exercise_bout(traj)


def exercise_bout_neural_on_model(model, rng: np.random.Generator, device: str = "cpu") -> ScenarioResult:
    from ...model import integrate
    from ...types import EMBEDDING_DIM

    params = PatientParams()
    duration_min = 180
    start_hour = 10.0
    meals: list[tuple[float, float, float, float]] = []
    act = _exercise_activity(duration_min)
    prng = np.random.default_rng(rng.integers(0, 2**32))
    cold_traj = cold_model_trajectory(params, meals, duration_min, start_hour, activity=act, rng=prng)
    init = torch.tensor(cold_traj[0], dtype=torch.float32, device=device)
    emb = torch.zeros(EMBEDDING_DIM, dtype=torch.float32, device=device)
    act_t = torch.tensor(act, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        neural = integrate(
            model, init, emb, duration_min,
            dt=1.0, start_time_minutes=start_hour * 60.0,
            meals=[],
            activity=act_t,
        )
    return scenario_result_exercise_bout(neural.cpu().numpy())


def scenario_result_acute_z2_bout(traj: np.ndarray) -> ScenarioResult:
    # Z2 (zone 2) is the canonical aerobic-base intensity: sub-lactate-
    # threshold, fat-oxidation-dominated, conversational. The checks below
    # encode the textbook acute Z2 shape (McArdle/Katch; Brooks; Coyle) —
    # cardiac rise, glucose draw, FFA mobilisation, modest lactate (below
    # LT≈4 mmol/L), and prompt parasympathetic recovery. Distinct from the
    # existing exercise_bout scenario (activity=0.6, more like Z3/Z4 with
    # a lactate-rises check) — this one specifically tests the Z2 phenotype
    # that the chronic-adaptation campaign (multi-timescale-plan.md Move D)
    # will eventually score for trained-vs-untrained differences.
    hr = traj[:, MARKER_INDEX["hr"]]
    glucose = traj[:, MARKER_INDEX["glucose"]]
    ffa = traj[:, MARKER_INDEX["ffa"]]
    lac = traj[:, MARKER_INDEX["lactate"]]

    hr_rest = float(np.mean(hr[:50]))
    hr_z2 = float(np.mean(hr[65:90]))
    hr_recovery = float(np.mean(hr[105:120]))

    glucose_rest = float(np.mean(glucose[:50]))
    glucose_min = float(np.min(glucose[60:120]))

    ffa_rest = float(np.mean(ffa[:50]))
    ffa_peak = float(np.max(ffa[60:150]))

    lac_peak = float(np.max(lac[60:120]))

    # Thresholds calibrated to the cold model's Z2 response (the convention
    # used by every other textbook scenario): cold passes 5/5, so the
    # scenario detects trained-model *regressions* from the cold baseline.
    # The cold model's exercise response is compressed in activity-intensity
    # space (Z2 at activity=0.4 produces lactate ≈ 4 mmol/L, near the Z3
    # bout's 4.1 at activity=0.6) — so checks favour direction/sign over
    # magnitude. The chronic-adaptation campaign (multi-timescale-plan.md
    # Move D) will later sharpen these checks once the trained model
    # learns trained-vs-untrained differences in Z2 phenotype.
    checks = [
        ScenarioCheck(
            "hr_rises_during_z2",
            "HR should increase ≥20 bpm above baseline during Z2 bout",
            hr_z2 - hr_rest >= 20.0,
            hr_z2 - hr_rest, 20.0,
        ),
        ScenarioCheck(
            "glucose_drops_during_z2",
            "Glucose should drop ≥1 mg/dL during/after Z2 bout (muscle uptake > hepatic output)",
            glucose_rest - glucose_min >= 1.0,
            glucose_rest - glucose_min, 1.0,
        ),
        ScenarioCheck(
            "ffa_does_not_drop_during_z2",
            "FFA should not be suppressed during Z2 (lipolysis active, not the postprandial pattern)",
            ffa_peak - ffa_rest >= -0.02,
            ffa_peak - ffa_rest, -0.02,
        ),
        ScenarioCheck(
            "lactate_modest_during_z2",
            "Lactate peak should stay below 4.5 mmol/L during Z2 (sub-high-intensity)",
            lac_peak < 4.5,
            lac_peak, 4.5,
        ),
        ScenarioCheck(
            "hr_recovers_close_to_baseline",
            "HR should return within 10 bpm of baseline by 30 min post-bout (parasympathetic reactivation)",
            hr_recovery <= hr_rest + 10.0,
            hr_recovery, hr_rest + 10.0,
        ),
    ]

    return ScenarioResult(
        name="acute_z2_bout",
        source="McArdle, Katch & Katch — Exercise Physiology; Coyle — Z2 fat oxidation",
        description="30-min zone-2 aerobic bout — sub-lactate-threshold acute response shape",
        checks=checks,
        pass_rate=sum(1 for c in checks if c.passed) / len(checks),
    )


def acute_z2_bout_scenario(rng: np.random.Generator, trajectory_fn=None) -> ScenarioResult:
    del trajectory_fn
    params = PatientParams()
    duration_min = 180
    start_hour = 10.0
    meals: list[tuple[float, float, float, float]] = []
    act = _z2_activity(duration_min)
    traj = cold_model_trajectory(
        params, meals, duration_min, start_hour,
        activity=act,
        rng=np.random.default_rng(rng.integers(0, 2**32)),
    )
    return scenario_result_acute_z2_bout(traj)


def acute_z2_bout_neural_on_model(model, rng: np.random.Generator, device: str = "cpu") -> ScenarioResult:
    from ...model import integrate
    from ...types import EMBEDDING_DIM

    params = PatientParams()
    duration_min = 180
    start_hour = 10.0
    meals: list[tuple[float, float, float, float]] = []
    act = _z2_activity(duration_min)
    prng = np.random.default_rng(rng.integers(0, 2**32))
    cold_traj = cold_model_trajectory(params, meals, duration_min, start_hour, activity=act, rng=prng)
    init = torch.tensor(cold_traj[0], dtype=torch.float32, device=device)
    emb = torch.zeros(EMBEDDING_DIM, dtype=torch.float32, device=device)
    act_t = torch.tensor(act, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        neural = integrate(
            model, init, emb, duration_min,
            dt=1.0, start_time_minutes=start_hour * 60.0,
            meals=[],
            activity=act_t,
        )
    return scenario_result_acute_z2_bout(neural.cpu().numpy())


SCENARIO_RUNNERS = [
    exercise_bout_scenario,
    acute_z2_bout_scenario,
]
