"""End-to-end flow stories: nutrient and signal pathways across modules.

Each scenario encodes a narrative contract as explicit ScenarioCheck rows
against the cold full-body teacher unless trajectory_fn is supplied.
"""

from __future__ import annotations

import numpy as np

from .base import ScenarioCheck, ScenarioResult, TrajectoryProvider
from .flow_story_protocol import (
    DIETARY_CARB_FLOW_DURATION_MIN,
    DIETARY_CARB_FLOW_MEALS,
    DIETARY_CARB_FLOW_START_HOUR,
    DIETARY_CARB_FFA_POST_END,
    DIETARY_CARB_FFA_POST_START,
    DIETARY_CARB_GLUCAGON_POST_END,
    DIETARY_CARB_GLUCAGON_POST_START,
    DIETARY_CARB_GLUCOSE_EXCURSION_END,
    DIETARY_CARB_GLUCOSE_EXCURSION_START,
    DIETARY_CARB_GLUCOSE_RECOVERY_END,
    DIETARY_CARB_GLUCOSE_RECOVERY_START,
    DIETARY_CARB_GHRELIN_POST_END,
    DIETARY_CARB_GHRELIN_POST_START,
    DIETARY_CARB_GLP1_PULSE_END,
    DIETARY_CARB_GLP1_PULSE_START,
    DIETARY_CARB_INSULIN_PEAK_SCAN_END,
    DIETARY_CARB_PRE_MEAL_END,
    DIETARY_CARB_RA_WINDOW_END,
    DIETARY_CARB_RA_WINDOW_START,
    DIETARY_CARB_TEMP_POST_END,
    DIETARY_CARB_TEMP_POST_START,
)
from ..full_body import PatientParams, simulate_full_body
from ...types import GUT_OUTPUT_DIM, MARKER_INDEX, STATE_DIM


def dietary_carbohydrate_meal_flow_scenario(
    rng: np.random.Generator,
    trajectory_fn: TrajectoryProvider | None = None,
) -> ScenarioResult:
    """Life of dietary carbohydrate: logged mixed meal → appearance → glycemia → hormones → recovery.

    Uses the full-body cold model with an 8h horizon so glucose can return near
    baseline (the teacher's Bergman-style dynamics need longer than 4h for this meal).

    If ``trajectory_fn`` is set, it must be ``callable(rng) -> (trajectory, absorption)``
    with the same shapes as ``simulate_full_body`` for this protocol (minute grid,
    absorption[:,0] = glucose appearance). Typical use: neural model integrated with
    teacher gut outputs via ``make_dietary_carbohydrate_neural_trajectory_fn``.

    See: docs/flow-stories/dietary-carbohydrate.md
    """
    duration_min = DIETARY_CARB_FLOW_DURATION_MIN
    start_hour = DIETARY_CARB_FLOW_START_HOUR
    meals = DIETARY_CARB_FLOW_MEALS
    sleep_wake = np.ones(duration_min, dtype=np.float32)
    activity = np.full(duration_min, 0.05, dtype=np.float32)
    prng = np.random.default_rng(rng.integers(0, 2**32))
    if trajectory_fn is not None:
        traj, absorption = trajectory_fn(prng)
        if traj.shape != (duration_min, STATE_DIM):
            raise ValueError(
                f"trajectory_fn must return trajectory shape ({duration_min}, {STATE_DIM}), "
                f"got {traj.shape}",
            )
        if absorption.shape != (duration_min, GUT_OUTPUT_DIM):
            raise ValueError(
                f"trajectory_fn must return absorption shape ({duration_min}, {GUT_OUTPUT_DIM}), "
                f"got {absorption.shape}",
            )
    else:
        params = PatientParams()
        traj, absorption = simulate_full_body(
            params,
            meals,
            sleep_wake,
            activity,
            duration_min,
            start_hour,
            noise_scale=0.001,
            rng=prng,
        )

    ra = absorption[:, 0]
    g = traj[:, MARKER_INDEX["glucose"]]
    ins = traj[:, MARKER_INDEX["insulin"]]
    gn = traj[:, MARKER_INDEX["glucagon"]]
    ffa = traj[:, MARKER_INDEX["ffa"]]
    ghr = traj[:, MARKER_INDEX["ghrelin"]]
    glp1 = traj[:, MARKER_INDEX["glp1"]]
    temp = traj[:, MARKER_INDEX["temp"]]

    pre_end = DIETARY_CARB_PRE_MEAL_END
    g_pre = float(np.mean(g[:pre_end]))
    gn_pre = float(np.mean(gn[:pre_end]))
    ffa_pre = float(np.mean(ffa[:pre_end]))
    ghr_pre = float(np.mean(ghr[:pre_end]))
    glp1_pre = float(np.mean(glp1[:pre_end]))
    temp_pre = float(np.mean(temp[:pre_end]))

    win_start, win_end = DIETARY_CARB_RA_WINDOW_START, DIETARY_CARB_RA_WINDOW_END
    ra_peak_t = win_start + int(np.argmax(ra[win_start:win_end]))
    g_peak_t = win_start + int(np.argmax(g[win_start:win_end]))
    ins_scan_end = min(DIETARY_CARB_INSULIN_PEAK_SCAN_END, duration_min)
    ins_peak_t = win_start + int(np.argmax(ins[win_start:ins_scan_end]))

    g_excursion = float(
        np.max(g[DIETARY_CARB_GLUCOSE_EXCURSION_START:DIETARY_CARB_GLUCOSE_EXCURSION_END]) - g_pre
    )
    gn_post = float(np.mean(gn[DIETARY_CARB_GLUCAGON_POST_START:DIETARY_CARB_GLUCAGON_POST_END]))
    ffa_post = float(np.mean(ffa[DIETARY_CARB_FFA_POST_START:DIETARY_CARB_FFA_POST_END]))
    ghr_post = float(np.mean(ghr[DIETARY_CARB_GHRELIN_POST_START:DIETARY_CARB_GHRELIN_POST_END]))
    glp1_peak = float(np.max(glp1[DIETARY_CARB_GLP1_PULSE_START:DIETARY_CARB_GLP1_PULSE_END]))
    temp_post = float(np.mean(temp[DIETARY_CARB_TEMP_POST_START:DIETARY_CARB_TEMP_POST_END]))
    g_late = float(
        np.mean(g[DIETARY_CARB_GLUCOSE_RECOVERY_START:DIETARY_CARB_GLUCOSE_RECOVERY_END])
    )

    checks = [
        ScenarioCheck(
            "glucose_rises_after_meal",
            "Plasma glucose should rise meaningfully after the carb-containing meal",
            g_excursion > 12.0,
            g_excursion,
            12.0,
        ),
        ScenarioCheck(
            "appearance_leads_glucose_peak",
            "Peak glucose appearance (gut output) should not trail the glucose peak by much",
            ra_peak_t <= g_peak_t + 50,
            float(g_peak_t - ra_peak_t),
            -50.0,
        ),
        ScenarioCheck(
            "insulin_peak_follows_glucose",
            "Insulin peak should occur with or after the glycemic peak (within 30 min)",
            ins_peak_t + 30 >= g_peak_t,
            float(ins_peak_t - g_peak_t),
            -30.0,
        ),
        ScenarioCheck(
            "glucagon_suppressed_postprandial",
            "Mean glucagon after meal should stay below fasting mean",
            gn_post < gn_pre,
            gn_pre - gn_post,
            0.0,
        ),
        ScenarioCheck(
            "ffa_suppressed_antilipolysis",
            "Mean FFA mid-course should fall below pre-meal (insulin-mediated antilipolysis)",
            ffa_post < ffa_pre,
            ffa_pre - ffa_post,
            0.0,
        ),
        ScenarioCheck(
            "ghrelin_suppressed_after_feeding",
            "Ghrelin should be lower in the post-absorptive window than before the meal",
            ghr_post < ghr_pre,
            ghr_pre - ghr_post,
            0.0,
        ),
        ScenarioCheck(
            "glp1_incretin_pulse",
            "GLP-1 should show a clear post-meal pulse above fasting",
            glp1_peak > glp1_pre + 5.0,
            glp1_peak - glp1_pre,
            5.0,
        ),
        ScenarioCheck(
            "postprandial_warming",
            "Core temperature mean should rise slightly during the post-meal window",
            temp_post > temp_pre,
            temp_post - temp_pre,
            0.0,
        ),
        ScenarioCheck(
            "glucose_returns_near_baseline",
            "By late simulation, glucose mean should return near the pre-meal level",
            abs(g_late - g_pre) < 8.0,
            abs(g_late - g_pre),
            8.0,
        ),
    ]

    source_note = (
        "Pulse flow-story contract; cold model: pulse/knowledge/full_body.py"
        if trajectory_fn is None
        else "Pulse flow-story contract; neural trajectory with teacher gut appearance"
    )
    return ScenarioResult(
        name="dietary_carbohydrate_meal_flow",
        source=source_note,
        description=(
            "Dietary carbohydrate from a logged mixed meal: appearance, glycemic excursion, "
            "insulin/glucagon/FFA/ghrelin/GLP-1, DIT temperature, 8h glucose recovery"
        ),
        checks=checks,
        pass_rate=sum(1 for c in checks if c.passed) / len(checks),
    )


SCENARIO_RUNNERS = [
    dietary_carbohydrate_meal_flow_scenario,
]
