"""Protocol constants for flow stories (no torch — safe to import everywhere).

Also defines UI / textbook phase bands for the dietary-carbohydrate story so the
Pulse engine, scenario checks, and docs stay aligned on the same minute windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DIETARY_CARB_FLOW_DURATION_MIN = 480
DIETARY_CARB_FLOW_START_HOUR = 8.0
DIETARY_CARB_FLOW_MEALS: list[tuple[float, float, float, float]] = [
    (30.0, 50.0, 12.0, 18.0),
]

# --- Scenario slices (indices into the minute grid; same as flow_stories checks) ---
DIETARY_CARB_PRE_MEAL_END = 25
DIETARY_CARB_RA_WINDOW_START = 30
DIETARY_CARB_RA_WINDOW_END = 200
DIETARY_CARB_GLUCOSE_EXCURSION_START = 45
DIETARY_CARB_GLUCOSE_EXCURSION_END = 200
DIETARY_CARB_INSULIN_PEAK_SCAN_END = 220
DIETARY_CARB_GLUCAGON_POST_START = 60
DIETARY_CARB_GLUCAGON_POST_END = 200
DIETARY_CARB_FFA_POST_START = 80
DIETARY_CARB_FFA_POST_END = 240
DIETARY_CARB_GHRELIN_POST_START = 120
DIETARY_CARB_GHRELIN_POST_END = 280
DIETARY_CARB_GLP1_PULSE_START = 35
DIETARY_CARB_GLP1_PULSE_END = 200
DIETARY_CARB_TEMP_POST_START = 90
DIETARY_CARB_TEMP_POST_END = 240
DIETARY_CARB_GLUCOSE_RECOVERY_START = 420
DIETARY_CARB_GLUCOSE_RECOVERY_END = 475

# Reference meal minute in the canonical protocol (used to translate absolute → meal-relative).
DIETARY_CARB_PROTOCOL_MEAL_REF_MIN = 30.0


@dataclass(frozen=True)
class _PhaseDef:
    id: str
    label: str
    anchor: Literal["absolute", "meal_relative"]
    start: float
    end: float


# One row per narrative band in docs/flow-stories/dietary-carbohydrate.md
_DIETARY_CARB_PHASE_DEFS: tuple[_PhaseDef, ...] = (
    _PhaseDef("baseline", "Baseline", "absolute", 0.0, float(DIETARY_CARB_PRE_MEAL_END)),
    _PhaseDef(
        "appearance",
        "Glucose appearance (Ra window)",
        "meal_relative",
        float(DIETARY_CARB_RA_WINDOW_START - DIETARY_CARB_PROTOCOL_MEAL_REF_MIN),
        float(DIETARY_CARB_RA_WINDOW_END - DIETARY_CARB_PROTOCOL_MEAL_REF_MIN),
    ),
    _PhaseDef(
        "glycemic_excursion",
        "Glycemic excursion",
        "meal_relative",
        float(DIETARY_CARB_GLUCOSE_EXCURSION_START - DIETARY_CARB_PROTOCOL_MEAL_REF_MIN),
        float(DIETARY_CARB_GLUCOSE_EXCURSION_END - DIETARY_CARB_PROTOCOL_MEAL_REF_MIN),
    ),
    _PhaseDef(
        "insulin_response",
        "Insulin response",
        "meal_relative",
        float(DIETARY_CARB_RA_WINDOW_START - DIETARY_CARB_PROTOCOL_MEAL_REF_MIN),
        float(DIETARY_CARB_INSULIN_PEAK_SCAN_END - DIETARY_CARB_PROTOCOL_MEAL_REF_MIN),
    ),
    _PhaseDef(
        "glucagon_postprandial",
        "Glucagon (postprandial mean)",
        "meal_relative",
        float(DIETARY_CARB_GLUCAGON_POST_START - DIETARY_CARB_PROTOCOL_MEAL_REF_MIN),
        float(DIETARY_CARB_GLUCAGON_POST_END - DIETARY_CARB_PROTOCOL_MEAL_REF_MIN),
    ),
    _PhaseDef(
        "ffa_postprandial",
        "FFA (antilipolysis window)",
        "meal_relative",
        float(DIETARY_CARB_FFA_POST_START - DIETARY_CARB_PROTOCOL_MEAL_REF_MIN),
        float(DIETARY_CARB_FFA_POST_END - DIETARY_CARB_PROTOCOL_MEAL_REF_MIN),
    ),
    _PhaseDef(
        "ghrelin_postprandial",
        "Ghrelin (post-absorptive)",
        "meal_relative",
        float(DIETARY_CARB_GHRELIN_POST_START - DIETARY_CARB_PROTOCOL_MEAL_REF_MIN),
        float(DIETARY_CARB_GHRELIN_POST_END - DIETARY_CARB_PROTOCOL_MEAL_REF_MIN),
    ),
    _PhaseDef(
        "glp1_pulse",
        "GLP-1 (incretin pulse)",
        "meal_relative",
        float(DIETARY_CARB_GLP1_PULSE_START - DIETARY_CARB_PROTOCOL_MEAL_REF_MIN),
        float(DIETARY_CARB_GLP1_PULSE_END - DIETARY_CARB_PROTOCOL_MEAL_REF_MIN),
    ),
    _PhaseDef(
        "thermogenesis",
        "Core temperature (post-meal)",
        "meal_relative",
        float(DIETARY_CARB_TEMP_POST_START - DIETARY_CARB_PROTOCOL_MEAL_REF_MIN),
        float(DIETARY_CARB_TEMP_POST_END - DIETARY_CARB_PROTOCOL_MEAL_REF_MIN),
    ),
    _PhaseDef(
        "recovery",
        "Glucose recovery",
        "absolute",
        float(DIETARY_CARB_GLUCOSE_RECOVERY_START),
        float(DIETARY_CARB_GLUCOSE_RECOVERY_END),
    ),
)


def dietary_carb_flow_phases_for_ui(
    duration_min: int,
    meal_time_min: float | None,
) -> list[dict[str, str | float]]:
    """Phase bands aligned with flow_stories / dietary-carbohydrate.md windows.

    ``meal_time_min`` is the first carb meal in the *simulation* (minutes from t=0).
    Meal-relative bands shift with the meal; baseline and recovery stay absolute.
    If there is no carb meal, only absolute phases that fit ``duration_min`` are returned.
    """
    out: list[dict[str, str | float]] = []
    for p in _DIETARY_CARB_PHASE_DEFS:
        if p.anchor == "absolute":
            s = max(0.0, p.start)
            e = min(float(duration_min), p.end)
        else:
            if meal_time_min is None:
                continue
            s = max(0.0, float(meal_time_min) + p.start)
            e = min(float(duration_min), float(meal_time_min) + p.end)
        if s < e:
            out.append({"id": p.id, "label": p.label, "start_min": s, "end_min": e})
    return out
