"""
Ingest real CGM + Oura HR + sleep data into bench-format episodes.

Sources (all under the directory passed via ``--source-dir``):
- ``GabrielRovina_glucose_11-7-2025.csv`` — FreeStyle Libre 3 (5-min,
  Aug 2024 - Jul 2025).
- ``google_fit.zip``: per-marker JSONs from a Google Takeout export.
  We pull:
    * ``derived_com.google.heart_rate.bpm_com.ouraring.json``
      (continuous HR from Oura Ring, mostly during sleep)
    * ``derived_com.google.sleep.segment_com.google.an.json``
      (sleep stage segments — awake/light/deep/REM)
    * ``derived_com.google.activity.segment_com.google(15).json``
      (workout segments — walks/runs/biking/aerobics with start-end-type).
      Maps to per-minute ``activity[t]`` in [0, 1] matching the model's
      ``activity`` external input (0=rest, 1=vigorous). Resting minutes
      default to 0.1, matching the model's learned ``default_activity``.

Output: a JSON file in the same shape as
``benchmark.dataset.generated.json``, so the existing bench loader
and Bayesian-calibration tooling work unchanged. Each episode is a
12h window where CGM + Oura HR + sleep are all present.

Episodes have:
- ``calibration_check_ins``: every-60-min snapshots covering the first
  70% of the window (glucose + hr only — those are the markers we have).
- ``eval_measurements``: per-marker points at 30-min intervals across
  the last 30% of the window.
- ``sleep_wake``: per-minute mask derived from sleep stages.
- ``meals``: empty (we don't have meal labels in the source data).

The empty-meals choice means we're testing the model's *no-meal*
predictive accuracy on real data — which is a clean test of fasting-
period dynamics, baseline drift, sleep-mediated HR/glucose changes,
and the default-baseline regularizer. Meal-driven dynamics will need
either inferred meals (from CGM spikes) or a labelling pass — left
for a follow-up.
"""
from __future__ import annotations

import argparse
import csv
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

CGM_CSV = "GabrielRovina_glucose_11-7-2025.csv"
ZIP_NAME = "google_fit.zip"
HR_FILE = "Takeout/Fit/All data/derived_com.google.heart_rate.bpm_com.ouraring.json"
SLEEP_FILE = "Takeout/Fit/All data/derived_com.google.sleep.segment_com.google.an.json"
ACTIVITY_FILE = "Takeout/Fit/All data/derived_com.google.activity.segment_com.google(15).json"

# Google Fit activity codes → model ``activity[t]`` intensity (0=rest, 1=vigorous).
# Codes from com.google.android.gms.fitness.FitnessActivities. Anything not
# listed defaults to ``ACTIVITY_DEFAULT`` so unknown codes don't silently
# raise the baseline. Resting / sleep / unrecognized → matches model's
# learned ``default_activity = 0.1`` so activity-bearing episodes degrade
# gracefully when no segment covers a minute.
ACTIVITY_DEFAULT = 0.1
ACTIVITY_INTENSITY: dict[int, float] = {
    1: 0.7,    # biking
    7: 0.4,    # walking
    8: 0.95,   # running
    10: 0.6,   # elliptical
    24: 0.0,   # sleep (legacy)
    35: 0.6,   # strength training (legacy)
    45: 0.0,   # sleep
    80: 0.7,   # aerobics
    89: 0.7,   # treadmill
    97: 0.6,   # stair climbing
    100: 0.6,  # strength training
    108: 0.4,  # pilates
    114: 0.7,  # swimming
    122: 0.9,  # HIIT
}

EPISODE_DURATION_MIN = 12 * 60  # match existing bench shape
MIN_CGM_POINTS = 100  # ~8h at 5-min spacing
MIN_HR_POINTS = 50    # Oura is dense during sleep; this requires ~4h of sleep coverage

# Match types.MARKERS order so initial_state is shaped correctly.
MARKER_ORDER = [
    "glucose", "insulin", "glucagon", "ffa", "bhb", "lactate", "hepatic_output",
    "ghrelin", "leptin", "glp1", "cortisol", "acth",
    "hr", "hrv", "sbp", "dbp", "temp", "rr", "spo2",
]
BASELINE = {
    "glucose": 95, "insulin": 10, "glucagon": 70, "ffa": 0.5, "bhb": 0.1,
    "lactate": 1.0, "hepatic_output": 2.0, "ghrelin": 100, "leptin": 10,
    "glp1": 10, "cortisol": 12, "acth": 30, "hr": 70, "hrv": 40,
    "sbp": 120, "dbp": 80, "temp": 37, "rr": 15, "spo2": 98,
}


def parse_cgm(path: Path) -> list[tuple[datetime, float]]:
    """Return (utc_datetime, glucose_mg_dL) tuples sorted by time."""
    out: list[tuple[datetime, float]] = []
    with path.open() as f:
        next(f)  # metadata line
        reader = csv.DictReader(f)
        for row in reader:
            mmol = row.get("Historic Glucose mmol/L", "").strip()
            if not mmol:
                continue
            try:
                ts = datetime.strptime(row["Device Timestamp"], "%d-%m-%Y %H:%M")
            except ValueError:
                continue
            ts = ts.replace(tzinfo=timezone.utc)
            out.append((ts, float(mmol) * 18.0))  # mmol/L → mg/dL
    out.sort(key=lambda x: x[0])
    return out


def _ns_to_dt(ns: str | int) -> datetime:
    return datetime.fromtimestamp(int(ns) / 1e9, tz=timezone.utc)


def parse_oura_hr(zip_path: Path) -> list[tuple[datetime, float]]:
    out: list[tuple[datetime, float]] = []
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(HR_FILE) as f:
            data = json.load(f)
    for p in data.get("Data Points", []):
        try:
            v = float(p["fitValue"][0]["value"]["fpVal"])
        except (KeyError, IndexError, TypeError):
            continue
        if v <= 0:  # Oura emits 0 when off-wrist
            continue
        out.append((_ns_to_dt(p["startTimeNanos"]), v))
    out.sort(key=lambda x: x[0])
    return out


def parse_sleep_segments(zip_path: Path) -> list[tuple[datetime, datetime, int]]:
    """Return (start, end, stage) tuples. Google Fit codes: 1=awake, 4=light,
    5=deep, 6=REM (2/3 not seen in this export)."""
    out: list[tuple[datetime, datetime, int]] = []
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(SLEEP_FILE) as f:
            data = json.load(f)
    for p in data.get("Data Points", []):
        try:
            stage = int(p["fitValue"][0]["value"]["intVal"])
        except (KeyError, IndexError, TypeError):
            continue
        out.append((
            _ns_to_dt(p["startTimeNanos"]),
            _ns_to_dt(p["endTimeNanos"]),
            stage,
        ))
    out.sort(key=lambda x: x[0])
    return out


def parse_activity_segments(zip_path: Path) -> list[tuple[datetime, datetime, int]]:
    """Return (start, end, activity_code) tuples for workout segments."""
    out: list[tuple[datetime, datetime, int]] = []
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(ACTIVITY_FILE) as f:
            data = json.load(f)
    for p in data.get("Data Points", []):
        try:
            code = int(p["fitValue"][0]["value"]["intVal"])
        except (KeyError, IndexError, TypeError):
            continue
        out.append((
            _ns_to_dt(p["startTimeNanos"]),
            _ns_to_dt(p["endTimeNanos"]),
            code,
        ))
    out.sort(key=lambda x: x[0])
    return out


def build_activity_mask(
    window_start: datetime,
    duration_min: int,
    activity_segments: list[tuple[datetime, datetime, int]],
) -> list[float]:
    """Per-minute activity intensity in [0, 1]. Defaults to ACTIVITY_DEFAULT
    where no segment covers; activity segments override with ACTIVITY_INTENSITY
    lookup (unrecognized codes leave the default in place).
    """
    mask = [ACTIVITY_DEFAULT] * duration_min
    ws = window_start.timestamp()
    window_end = ws + duration_min * 60
    for seg_start, seg_end, code in activity_segments:
        ss = seg_start.timestamp()
        se = seg_end.timestamp()
        if se <= ws or ss >= window_end:
            continue
        intensity = ACTIVITY_INTENSITY.get(code)
        if intensity is None:
            continue
        lo = max(0, int((ss - ws) // 60))
        hi = min(duration_min, int((se - ws) // 60) + 1)
        for i in range(lo, hi):
            mask[i] = intensity
    return mask


def build_sleep_wake_mask(
    window_start: datetime,
    duration_min: int,
    sleep_segments: list[tuple[datetime, datetime, int]],
) -> list[float]:
    """Per-minute mask: 1.0 awake, 0.0 asleep. Stages 4/5/6 = asleep; 1 = awake."""
    mask = [1.0] * duration_min
    window_end = window_start.timestamp() + duration_min * 60
    ws = window_start.timestamp()
    for seg_start, seg_end, stage in sleep_segments:
        ss = seg_start.timestamp()
        se = seg_end.timestamp()
        if se <= ws or ss >= window_end:
            continue
        # Only stages 4/5/6 are actual sleep; stage 1 (awake) leaves mask=1.
        if stage not in (2, 4, 5, 6):
            continue
        lo = max(0, int((ss - ws) // 60))
        hi = min(duration_min, int((se - ws) // 60))
        for i in range(lo, hi):
            mask[i] = 0.0
    return mask


def find_overlapping_windows(
    cgm: list[tuple[datetime, float]],
    hr: list[tuple[datetime, float]],
    duration_min: int,
    stride_hours: int = 12,
) -> list[datetime]:
    """Find episode start times where CGM and HR both have dense coverage."""
    if not cgm or not hr:
        return []
    cgm_start = max(cgm[0][0], hr[0][0])
    cgm_end = min(cgm[-1][0], hr[-1][0])
    if cgm_end <= cgm_start:
        return []

    cgm_arr = np.array([(p[0].timestamp(), p[1]) for p in cgm])
    hr_arr = np.array([(p[0].timestamp(), p[1]) for p in hr])

    from datetime import timedelta
    starts: list[datetime] = []
    cur = cgm_start.replace(minute=0, second=0, microsecond=0)
    while cur.timestamp() + duration_min * 60 <= cgm_end.timestamp():
        ws = cur.timestamp()
        we = ws + duration_min * 60
        cgm_in = ((cgm_arr[:, 0] >= ws) & (cgm_arr[:, 0] < we)).sum()
        hr_in = ((hr_arr[:, 0] >= ws) & (hr_arr[:, 0] < we)).sum()
        if cgm_in >= MIN_CGM_POINTS and hr_in >= MIN_HR_POINTS:
            starts.append(cur)
        # Stride forward
        cur = datetime.fromtimestamp(ws + stride_hours * 3600, tz=timezone.utc)
    return starts


def build_episode(
    user_id: str,
    window_start: datetime,
    duration_min: int,
    cgm: list[tuple[datetime, float]],
    hr: list[tuple[datetime, float]],
    sleep_segments: list[tuple[datetime, datetime, int]],
    activity_segments: list[tuple[datetime, datetime, int]],
) -> dict | None:
    ws = window_start.timestamp()
    we = ws + duration_min * 60

    cgm_in = [(int((p[0].timestamp() - ws) // 60), p[1]) for p in cgm if ws <= p[0].timestamp() < we]
    hr_in = [(int((p[0].timestamp() - ws) // 60), p[1]) for p in hr if ws <= p[0].timestamp() < we]
    if len(cgm_in) < MIN_CGM_POINTS or len(hr_in) < MIN_HR_POINTS:
        return None

    sleep_wake = build_sleep_wake_mask(window_start, duration_min, sleep_segments)
    activity = build_activity_mask(window_start, duration_min, activity_segments)

    # Calibration: every-60-min snapshots over first 70%
    calib_end_min = int(duration_min * 0.7)
    calibration_check_ins: list[dict] = []
    for cm in range(0, calib_end_min, 60):
        cgm_near = [(t, v) for t, v in cgm_in if abs(t - cm) <= 5]
        hr_near = [(t, v) for t, v in hr_in if abs(t - cm) <= 30]
        if not cgm_near and not hr_near:
            continue
        meas: dict[str, float] = {}
        if cgm_near:
            meas["glucose"] = round(sum(v for _, v in cgm_near) / len(cgm_near), 1)
        if hr_near:
            meas["hr"] = round(sum(v for _, v in hr_near) / len(hr_near), 1)
        calibration_check_ins.append({
            "time": cm,
            "measurements": meas,
            "feelings": {}, "bodySignals": {}, "meal": {}, "waterIntakeMl": None,
        })

    if len(calibration_check_ins) < 2:
        return None

    # Eval: stride-6 CGM points (≈30-min spacing on a 5-min CGM trace) in
    # the last 30% of the window; HR sampled at most every 30 min over the
    # same range (Oura is dense — we don't need every point).
    eval_start_min = calib_end_min
    eval_measurements: list[dict] = []
    cgm_eval = [(t, v) for t, v in cgm_in if eval_start_min <= t < duration_min]
    for t, v in cgm_eval[::6]:
        eval_measurements.append({"time": t, "marker_id": "glucose", "value": round(v, 1)})
    seen_hr_buckets: set[int] = set()
    for t, v in hr_in:
        if eval_start_min <= t < duration_min and t // 30 not in seen_hr_buckets:
            eval_measurements.append({"time": t, "marker_id": "hr", "value": round(v, 1)})
            seen_hr_buckets.add(t // 30)

    if len(eval_measurements) < 5:
        return None

    # Initial state: mean of the earliest available CGM/HR readings
    # (within the first 60 min of *available data*, not the window — the
    # window may start before data does because Oura/CGM begin at slightly
    # different clock minutes).
    if cgm_in:
        first_cgm_t = cgm_in[0][0]
        early_cgm = [v for t, v in cgm_in if t < first_cgm_t + 60]
    else:
        early_cgm = []
    if hr_in:
        first_hr_t = hr_in[0][0]
        early_hr = [v for t, v in hr_in if t < first_hr_t + 60]
    else:
        early_hr = []
    init_glucose = round(sum(early_cgm) / len(early_cgm), 1) if early_cgm else BASELINE["glucose"]
    init_hr = round(sum(early_hr) / len(early_hr), 1) if early_hr else BASELINE["hr"]

    initial_state = []
    for marker in MARKER_ORDER:
        if marker == "glucose":
            initial_state.append(init_glucose)
        elif marker == "hr":
            initial_state.append(init_hr)
        else:
            initial_state.append(BASELINE[marker])

    return {
        "user_id": user_id,
        "duration_min": duration_min,
        "start_time_minutes": window_start.hour * 60 + window_start.minute,
        "initial_state": initial_state,
        "meals": [],
        "calibration_check_ins": calibration_check_ins,
        "eval_measurements": eval_measurements,
        "sleep_wake": sleep_wake,
        "activity": activity,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source-dir", type=Path, required=True,
        help="Local directory holding the CGM CSV + Google Takeout zip.",
    )
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--user-id", default="gabriel")
    ap.add_argument("--max-episodes", type=int, default=20)
    args = ap.parse_args()

    cgm = parse_cgm(args.source_dir / CGM_CSV)
    hr = parse_oura_hr(args.source_dir / ZIP_NAME)
    sleep = parse_sleep_segments(args.source_dir / ZIP_NAME)
    activity = parse_activity_segments(args.source_dir / ZIP_NAME)

    print(f"CGM: {len(cgm)} points, {cgm[0][0].date()} → {cgm[-1][0].date()}")
    print(f"Oura HR: {len(hr)} points, {hr[0][0].date()} → {hr[-1][0].date()}")
    print(f"Sleep: {len(sleep)} segments")
    print(f"Activity: {len(activity)} segments")

    starts = find_overlapping_windows(cgm, hr, EPISODE_DURATION_MIN)
    print(f"Candidate windows: {len(starts)}")

    episodes = []
    for s in starts:
        ep = build_episode(args.user_id, s, EPISODE_DURATION_MIN, cgm, hr, sleep, activity)
        if ep is None:
            continue
        episodes.append(ep)
        if len(episodes) >= args.max_episodes:
            break

    print(f"Built {len(episodes)} episodes")

    payload = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": "real CGM + Oura HR + sleep (gabriel)",
            "duration_min": EPISODE_DURATION_MIN,
            "episodes": len(episodes),
        },
        "episodes": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {args.output} ({args.output.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
