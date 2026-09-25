"""Sealed fed-morning chain.

A 48 h awake rest started from the fed state, scored at the trained
prior-mean embedding. The links are literature bands on the student.
Nothing here is a loss, and nothing here is a gate: training on this
protocol would make the report a restatement of the loss.

The teacher trajectory is recorded beside the student so a miss can be
read against it. Pass or fail is the student's links only. The first
failed link, in causal order, is the result.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ..model import ModularPhysiologyNetwork, integrate
from ..types import MARKER_INDEX, NORM_CENTER
from .full_body import PatientParams, simulate_full_body

# Cahill 1970: ~2–3 mmol/L at 48 h. 6 mmol/L is a multi-day starvation
# value (the teacher's own guard against a 6.7 mmol/L 48 h fast).
_BHB_48H_LO = 2.0
_BHB_48H_HI = 6.0
# Rothman 1991 / the prolonged-fast rule's ceiling: hepatic glycogen
# essentially exhausted by the second morning.
_LIVER_48H_CEILING_G = 30.0
# A return to the setpoint is not a fast. Polonsky 1988: glucose falls on
# the order of 15% while insulin roughly halves. 5 mg/dL is the smallest
# drop that is still a drop; 0.7·Ib is the generous side of "halves".
# Marliss 1970 is about +0.5 pg/mL per fasted hour, so +10 pg/mL by 48 h
# is a low bar for "glucagon rose".
_GLUCOSE_DROP_MGDL = 5.0
_INSULIN_FALL_FRAC = 0.7
_GLUCAGON_RISE_PG = 10.0
# Two fasting glucoses the per-patient probe already uses (Gb 75 and Gb 120).
# The embedding is fit to an overnight of those check-ins; the 48 h fast is not
# in that fit. 60 mg/dL is the physiological floor that probe already requires.
# The two 48 h glucose/Gb ratios should agree: the floor is a fraction of the
# person's own basal, not a shared absolute glucose.
_FASTING_TARGETS_MGDL = (75.0, 120.0)
_GLUCOSE_FLOOR_MGDL = 60.0
_FRACTION_SPREAD = 0.10
_OVERNIGHT_MIN = 12 * 60

_HOURS = (8, 16, 24, 36, 48)
_MARKERS = (
    "glucose", "insulin", "glucagon", "ffa", "bhb", "liver_glycogen", "cortisol",
)


def judge_chain(
    course: dict[str, dict[str, float]],
    *,
    gb: float,
    ib: float,
    ffa_b: float,
    gn_b: float,
) -> dict[str, Any]:
    """Links in the order a fed-start fast has to travel.

    ``course`` is hour-label -> marker -> value, for 16 h, 24 h and 48 h
    at minimum. A link's ``passed`` is the literature or setpoint band.
    ``first_failure`` is the earliest link that missed.
    """
    def at(hour: int, marker: str) -> float:
        return float(course[f"{hour}h"][marker])

    links = [
        _link("glucose_down_16h", at(16, "glucose"), "<", gb,
              "glucose has left Gb by the first evening"),
        _link("insulin_down_16h", at(16, "insulin"), "<", ib,
              "insulin has followed glucose below Ib"),
        _link("glucagon_up_16h", at(16, "glucagon"), ">", gn_b,
              "glucagon is above Gnb while glucose and insulin are down"),
        _link("ffa_at_least_basal_16h", at(16, "ffa"), ">=", ffa_b,
              "FFA is at or above its fed basal"),
        _link("glucose_still_down_24h", at(24, "glucose"), "<", gb - _GLUCOSE_DROP_MGDL,
              "glucose is at least 5 mg/dL below Gb at the next 08:00"),
        _link("glucose_still_down_48h", at(48, "glucose"), "<", gb - _GLUCOSE_DROP_MGDL,
              "glucose is at least 5 mg/dL below Gb at 48 h"),
        _link("insulin_still_down_48h", at(48, "insulin"), "<", _INSULIN_FALL_FRAC * ib,
              "48 h insulin is below 0.7·Ib"),
        _link("glucagon_still_up_48h", at(48, "glucagon"), ">", gn_b + _GLUCAGON_RISE_PG,
              "48 h glucagon is at least 10 pg/mL above Gnb"),
        _link("liver_exhausted_48h", at(48, "liver_glycogen"), "<", _LIVER_48H_CEILING_G,
              "liver glycogen is under 30 g"),
        _link("bhb_cahill_48h", at(48, "bhb"), "in", (_BHB_48H_LO, _BHB_48H_HI),
              "BHB is in the 48 h Cahill band, 2–6 mmol/L"),
    ]
    first = next((lnk["name"] for lnk in links if not lnk["passed"]), None)
    return {
        "links": links,
        "first_failure": first,
        "passed": first is None,
    }


def fed_morning_chain(model: ModularPhysiologyNetwork) -> dict[str, Any]:
    """Roll the prior-mean person from 08:00 fed, awake, at rest, for 48 h."""
    model.eval()
    prior = getattr(model, "_embedding_prior_mean", None)
    if prior is None:
        embedding = torch.zeros(model.embedding_dim)
        embedding_source = "zero"
    else:
        embedding = prior.detach().float().cpu().view(-1)
        embedding_source = "embedding_prior_mean"
    return _chain_report(model, embedding, embedding_source)


def two_basal_fasts(model: ModularPhysiologyNetwork) -> dict[str, Any]:
    """Fit two embeddings to two fasting glucoses, then roll the sealed chain.

    Weights stay frozen. The fit is one asleep overnight of hourly glucose
    check-ins, the same ``calibrate_embedding`` the gate runs. The 48 h fast
    starts the next morning from the fed state and is not among those check-ins.
    """
    from ..calibration import CalibrationSettings, MeasurementPoint, calibrate_embedding

    model.eval()
    prior = getattr(model, "_embedding_prior_mean", None)
    prior_std = getattr(model, "_embedding_prior_std", None)
    people = []
    for target in _FASTING_TARGETS_MGDL:
        obs = [
            MeasurementPoint(time=t, marker_id="glucose", value=float(target))
            for t in range(60, _OVERNIGHT_MIN, 60)
        ]
        if prior is None or prior_std is None:
            people.append({
                "fasting_glucose_mgdl": float(target),
                "calibration": {"accepted": False, "reason": "no_prior"},
                "passed": False,
                "first_failure": "no_prior",
            })
            continue
        n_night = _OVERNIGHT_MIN
        cal = calibrate_embedding(
            model, obs, torch.tensor(NORM_CENTER, dtype=torch.float32), [], n_night,
            start_time_minutes=20.0 * 60.0,
            sleep_wake=torch.zeros(n_night), activity=torch.zeros(n_night),
            prior_mean=prior.detach().float().cpu().view(-1),
            prior_std=prior_std.detach().float().cpu().view(-1),
            settings=CalibrationSettings.from_env(),
        )
        report = _chain_report(model, cal.embedding, "calibrated_fasting_glucose")
        report["fasting_glucose_mgdl"] = float(target)
        report["calibration"] = cal.as_report()
        g48 = float(report["course"]["48h"]["glucose"])
        gb = float(report["setpoints"]["gb"])
        report["glucose_48h_over_gb"] = g48 / gb if gb else float("nan")
        if not cal.accepted:
            report["passed"] = False
            report["first_failure"] = "calibration_not_accepted"
        elif report["passed"] and g48 < _GLUCOSE_FLOOR_MGDL:
            report["passed"] = False
            report["first_failure"] = "glucose_floor_48h"
        people.append(report)

    ratios = [p.get("glucose_48h_over_gb") for p in people]
    fraction_ok = (
        len(ratios) == 2
        and all(r is not None and r == r for r in ratios)
        and abs(ratios[0] - ratios[1]) <= _FRACTION_SPREAD
        and all(p.get("calibration", {}).get("accepted") for p in people)
    )
    own = [p.get("first_failure") for p in people if not p.get("passed")]
    if own:
        first = own[0]
        passed = False
    elif not fraction_ok:
        first = "fractional_drop_disagrees"
        passed = False
    else:
        first = None
        passed = True
    return {
        "protocol": (
            "fit embedding to an asleep overnight of fasting glucose, "
            "then fed 08:00 awake rest, no meals, 48 h"
        ),
        "sealed": True,
        "gated": False,
        "targets_mgdl": list(_FASTING_TARGETS_MGDL),
        "fraction_spread": _FRACTION_SPREAD,
        "glucose_floor_mgdl": _GLUCOSE_FLOOR_MGDL,
        "people": people,
        "first_failure": first,
        "passed": passed,
    }


def _chain_report(
    model: ModularPhysiologyNetwork, embedding: torch.Tensor, embedding_source: str,
) -> dict[str, Any]:
    n = 48 * 60
    embedding = embedding.detach().float().cpu().view(-1)
    with torch.no_grad():
        traj = integrate(
            model, torch.tensor(NORM_CENTER, dtype=torch.float32), embedding, n,
            start_time_minutes=8.0 * 60.0, meals=[],
            sleep_wake=torch.ones(n), activity=torch.zeros(n),
        ).numpy()
        met_emb = model.embedding_projections["metabolic"](embedding.view(1, -1))
        gb = float(model.metabolic.glucose_setpoint_raw(met_emb))
        ib = float(model.metabolic.insulin_setpoint_raw(met_emb))
        ffa_b = float(model.metabolic.ffa_setpoint_raw(met_emb))
        gn_b = float(model.metabolic.gn_setpoint_raw(met_emb))
    course = {f"{h}h": _snap(traj, h * 60) for h in _HOURS}
    judged = judge_chain(course, gb=gb, ib=ib, ffa_b=ffa_b, gn_b=gn_b)
    return {
        "protocol": "fed 08:00, awake rest, no meals, 48 h",
        "sealed": True,
        "gated": False,
        "embedding_source": embedding_source,
        "setpoints": {"gb": gb, "ib": ib, "ffa_b": ffa_b, "gn_b": gn_b},
        "course": course,
        "teacher_course": _teacher_course(n),
        **judged,
    }


def _link(name: str, value: float, op: str, bound: float | tuple[float, float], detail: str) -> dict[str, Any]:
    if op == "<":
        passed = value < float(bound)
    elif op == ">":
        passed = value > float(bound)
    elif op == ">=":
        passed = value >= float(bound)
    elif op == "in":
        lo, hi = bound  # type: ignore[misc]
        passed = lo <= value <= hi
    else:
        raise ValueError(op)
    return {
        "name": name,
        "passed": bool(passed),
        "value": value,
        "op": op,
        "bound": list(bound) if isinstance(bound, tuple) else bound,
        "detail": detail,
    }


def _snap(traj: np.ndarray, minute: int) -> dict[str, float]:
    i = min(int(minute), len(traj) - 1)
    return {m: float(traj[i, MARKER_INDEX[m]]) for m in _MARKERS}


def _teacher_course(n: int) -> dict[str, dict[str, float]]:
    params = PatientParams()
    sw = np.ones(n, dtype=np.float32)
    act = np.zeros(n, dtype=np.float32)
    traj, _ = simulate_full_body(params, [], sw, act, n, start_hour=8.0, noise_scale=0.0)
    return {f"{h}h": _snap(traj, h * 60) for h in _HOURS}
