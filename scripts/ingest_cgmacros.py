"""Ingest CGMacros as calibrate-then-predict benchmark episodes (PLAN Phase 0).

CGMacros (Gutierrez-Osuna, Kerr, Mortazavi, Das; PhysioNet 2025, CC BY-NC-SA 4.0):
45 adults (healthy, pre-diabetic and type 2) over ten free-living days, with a
Libre Pro CGM, Fitbit heart rate and METs at one-minute rows, and logged meals
with their macronutrients. Breakfasts are protein shakes of varied composition,
lunches a restaurant meal at least three hours later.

One episode is one person's morning: sparse readings across breakfast calibrate
the person, and the prediction is the lunch response — a different meal, so a
different protocol, for the same body. Every meal row inside the window is part
of the protocol, snacks included.

Download (627 MB, mostly meal photos; only the CSVs are read):
    curl -L -o CGMacros.zip \\
        https://physionet-open.s3.amazonaws.com/cgmacros/1.0.0/CGMacros_dateshifted365.zip

Usage:
    uv run python scripts/ingest_cgmacros.py --zip CGMacros.zip --output /tmp/cgmacros.json
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pulse.benchmark import subject_of  # noqa: E402
from pulse.types import MARKER_INDEX, NORM_CENTER  # noqa: E402

SOURCE = "cgmacros"
CITATION = "Gutierrez-Osuna R, Kerr D, Mortazavi B, Das A. CGMacros. PhysioNet 2025 (v1.0.0)"

LEAD_IN_MIN = 30           # before breakfast: the person's own fasting morning
CALIBRATION_STRIDE_MIN = 30
EVAL_HORIZON_MIN = 180     # the lunch response
GLUCOSE_EVAL_STRIDE_MIN = 15   # the Libre Pro's native sampling period
HR_EVAL_STRIDE_MIN = 30
MEAL_GAP_MIN = (150, 360)  # breakfast to lunch; the protocol asked for three hours
MIN_HR_COVERAGE = 0.8

# A Fitbit MET of 1 is rest; the model's activity input is 0 at rest and 1 at a
# vigorous effort (about 9 METs, a run).
REST_MET = 1.0
VIGOROUS_MET = 9.0

_ROW_FILE = re.compile(r"CGMacros-(\d{3})/CGMacros-\d{3}\.csv$")


def _float(raw: str) -> float | None:
    raw = (raw or "").strip()
    try:
        return float(raw) if raw else None
    except ValueError:
        return None


def activity_from_mets(mets_x10: float | None) -> float:
    """Fitbit reports METs times ten."""
    if mets_x10 is None:
        return 0.0
    met = mets_x10 / 10.0
    return min(1.0, max(0.0, (met - REST_MET) / (VIGOROUS_MET - REST_MET)))


def glycaemic_group(a1c: float | None) -> str:
    """The study's own grouping by HbA1c (%)."""
    if a1c is None:
        return "unknown"
    if a1c < 5.7:
        return "healthy"
    return "prediabetes" if a1c <= 6.4 else "t2d"


def parse_bio(text: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for row in csv.reader(io.StringIO(text)):
        if not row or not row[0].strip().isdigit():
            continue
        a1c = _float(row[7])
        out[int(row[0])] = {"a1c": a1c, "group": glycaemic_group(a1c)}
    return out


def parse_rows(text: str) -> list[dict]:
    """One dict per minute: time, glucose, hr, activity, and the meal if one starts."""
    rows: list[dict] = []
    for r in csv.DictReader(io.StringIO(text)):
        try:
            ts = datetime.strptime(r["Timestamp"].strip(), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        except (KeyError, ValueError):
            continue
        meal = None
        kind = (r.get("Meal Type") or "").strip().lower()
        if kind:
            share = (_float(r.get("Amount Consumed ", r.get("Amount Consumed", ""))) or 100.0) / 100.0
            meal = {
                "kind": kind,
                "carbs": share * (_float(r.get("Carbs", "")) or 0.0),
                "proteins": share * (_float(r.get("Protein", "")) or 0.0),
                "fats": share * (_float(r.get("Fat", "")) or 0.0),
            }
        rows.append({
            "ts": ts,
            "glucose": _float(r.get("Libre GL", "")),
            "hr": _float(r.get("HR", "")),
            "activity": activity_from_mets(_float(r.get("METs", r.get("Mets", "")))),
            "meal": meal,
        })
    rows.sort(key=lambda x: x["ts"])
    return rows


def _initial_state(glucose: float, hr: float | None) -> list[float]:
    state = list(NORM_CENTER)
    state[MARKER_INDEX["glucose"]] = round(glucose, 1)
    if hr is not None:
        state[MARKER_INDEX["hr"]] = round(hr, 1)
    return state


def build_episode(user_id: str, rows: list[dict], breakfast_i: int, lunch_i: int) -> dict | None:
    """Rows are contiguous minutes; ``breakfast_i`` / ``lunch_i`` index the meal rows."""
    start_i = breakfast_i - LEAD_IN_MIN
    end_i = lunch_i + EVAL_HORIZON_MIN
    if start_i < 0 or end_i > len(rows):
        return None
    window = rows[start_i:end_i]
    t0 = window[0]["ts"]
    if any(int((w["ts"] - t0).total_seconds() // 60) != i for i, w in enumerate(window)):
        return None  # a gap in the minute rows
    if any(w["glucose"] is None for w in window):
        return None
    hr_share = sum(w["hr"] is not None for w in window) / len(window)
    use_hr = hr_share >= MIN_HR_COVERAGE

    lunch_t = lunch_i - start_i
    meals = [
        {"time": float(i), "carbs": round(w["meal"]["carbs"], 1),
         "fats": round(w["meal"]["fats"], 1), "proteins": round(w["meal"]["proteins"], 1)}
        for i, w in enumerate(window) if w["meal"] is not None
    ]

    calibration_check_ins = []
    for t in range(0, lunch_t, CALIBRATION_STRIDE_MIN):
        meas = {"glucose": round(window[t]["glucose"], 1)}
        if use_hr and window[t]["hr"] is not None:
            meas["hr"] = window[t]["hr"]
        calibration_check_ins.append({
            "time": t, "measurements": meas,
            "feelings": {}, "bodySignals": {}, "meal": {}, "waterIntakeMl": None,
        })

    eval_measurements = [
        {"time": t, "marker_id": "glucose", "value": round(window[t]["glucose"], 1)}
        for t in range(lunch_t + GLUCOSE_EVAL_STRIDE_MIN, len(window), GLUCOSE_EVAL_STRIDE_MIN)
    ]
    if use_hr:
        eval_measurements += [
            {"time": t, "marker_id": "hr", "value": window[t]["hr"]}
            for t in range(lunch_t + HR_EVAL_STRIDE_MIN, len(window), HR_EVAL_STRIDE_MIN)
            if window[t]["hr"] is not None
        ]

    return {
        "user_id": user_id,
        "source": SOURCE,
        "duration_min": len(window),
        "start_time_minutes": t0.hour * 60 + t0.minute,
        "initial_state": _initial_state(window[0]["glucose"], window[0]["hr"] if use_hr else None),
        "meals": meals,
        "calibration_check_ins": calibration_check_ins,
        "eval_measurements": eval_measurements,
        "sleep_wake": [1.0] * len(window),
        "activity": [round(w["activity"], 3) for w in window],
    }


def person_episodes(pid: int, rows: list[dict], per_person: int) -> list[dict]:
    """The person's first qualifying mornings: a breakfast, then lunch the same day."""
    episodes: list[dict] = []
    meal_rows = [(i, r) for i, r in enumerate(rows) if r["meal"] is not None]
    for k, (bi, b) in enumerate(meal_rows):
        if b["meal"]["kind"] != "breakfast":
            continue
        lunch = next(
            ((li, l) for li, l in meal_rows[k + 1:] if l["meal"]["kind"] == "lunch"
             and l["ts"].date() == b["ts"].date()),
            None,
        )
        if lunch is None:
            continue
        gap = int((lunch[1]["ts"] - b["ts"]).total_seconds() // 60)
        if not MEAL_GAP_MIN[0] <= gap <= MEAL_GAP_MIN[1]:
            continue
        ep = build_episode(f"{SOURCE}-{pid:03d}-day-{len(episodes) + 1:02d}", rows, bi, lunch[0])
        if ep is not None:
            episodes.append(ep)
        if len(episodes) >= per_person:
            break
    return episodes


def build_cgmacros_episodes(zip_path: Path, per_person: int) -> tuple[list[dict], dict]:
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        bio_name = next(n for n in names if n.endswith("bio.csv"))
        bio = parse_bio(zf.read(bio_name).decode("utf-8-sig"))
        episodes: list[dict] = []
        groups: dict[str, int] = {}
        for name in sorted(names):
            m = _ROW_FILE.search(name)
            if m is None:
                continue
            pid = int(m.group(1))
            rows = parse_rows(zf.read(name).decode("utf-8-sig"))
            eps = person_episodes(pid, rows, per_person)
            group = bio.get(pid, {}).get("group", "unknown")
            groups[group] = groups.get(group, 0) + len(eps)
            episodes += eps
    meta = {
        "source": CITATION,
        "license": "CC BY-NC-SA 4.0",
        "protocol": (
            "calibrate on readings every 30 min across breakfast; predict glucose every "
            "15 min and heart rate every 30 min over the 3 h after lunch"
        ),
        "episodes_by_group": groups,
        "activity_rest_level": 0.0,
    }
    return episodes, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, required=True, help="CGMacros_dateshifted365.zip")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--per-person", type=int, default=1, help="Mornings per participant")
    args = ap.parse_args()

    episodes, meta = build_cgmacros_episodes(args.zip, args.per_person)
    people = {subject_of(e["user_id"]) for e in episodes}
    print(f"CGMacros: {len(episodes)} episodes from {len(people)} people, by group {meta['episodes_by_group']}")
    payload = {
        "meta": {**meta, "generated_at": datetime.now(timezone.utc).isoformat(), "benchmark_source": SOURCE},
        "episodes": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {args.output} ({args.output.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
