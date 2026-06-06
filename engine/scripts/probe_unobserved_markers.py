"""
Shadow probe — extract trained model's predictions for unobserved markers
(insulin, glucagon, ffa, bhb, lactate, hepatic_output, ghrelin, leptin,
glp1, cortisol, acth, hrv, rr, spo2) on the bench's 25 episodes and
report distributions / plausibility checks.

Motivated by the iter-36 calibration investigation: the bench eval
points only cover glucose/hr/sbp/dbp/temp because those are the markers
real users measure on their watches. The model produces ~14 additional
markers internally that have *never* been validated against held-out
data. The +15 bpm HR offset hidden across iters 27-35 was visible only
because HR happened to be one of the 5 measured markers — analogous
biases on insulin / glucagon / cortisol could be similarly large and
totally undetected.

This probe doesn't introduce a new training signal or eval gate. It
runs inference (with calibrated embedding) on each bench episode,
extracts trajectories for every marker, computes summary statistics
plus a small set of physiology-motivated plausibility checks, and
prints them. Lets a human (or a future automated pass) spot
implausible model behaviour the bench gate can't.

Usage:
    uv run python scripts/probe_unobserved_markers.py \\
        --checkpoint /tmp/iter38.pt \\
        --bench /tmp/bench.json \\
        [--calibrate-steps 32]   # default 32 to keep it fast on local CPU

Defaults to 32-step calibration so a full 25-episode run takes ~25 min
on M-series CPU. Use 512 to match production gate (much slower).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ENGINE_ROOT = Path(__file__).resolve().parents[1]
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from pulse.benchmark import (
    calibrate_embedding,
    deterministic_user_embedding,
    load_benchmark_dataset,
    measurement_points_from_check_ins,
)
from pulse.diagnostics.probe import load_model_from_checkpoint
from pulse.model import integrate
from pulse.types import MARKER_IDS, MARKER_INDEX, MARKERS, NORM_CENTER

# Markers we want to probe — everything the model produces *except* the 5
# directly measured ones the bench gate already tests.
OBSERVED = {"glucose", "hr", "sbp", "dbp", "temp"}
UNOBSERVED = [m for m in MARKER_IDS if m not in OBSERVED]


# Physiologically reasonable population ranges for the unobserved markers.
# These are loose envelopes meant to flag *gross* implausibility, not pin
# every distribution; the goal is to catch +15-bpm-HR-style biases, not
# to enforce strict literature compliance. Sourced from the same general
# physiology references the cohort specs use (Guyton & Hall, Polonsky,
# Pruessner, etc.) and from the marker `typical` values in types.py.
PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    "insulin": (3.0, 120.0),       # 3-120 µU/mL spans fasting + postprandial peak
    "glucagon": (40.0, 200.0),     # pg/mL, fasted ~70, suppressed/elevated headroom
    "ffa": (0.05, 1.5),            # mmol/L, fasted ~0.5, postprandially suppressed
    "bhb": (0.0, 3.0),             # mmol/L, normal <0.5, ketotic up to 3
    "lactate": (0.5, 8.0),         # mmol/L, rest ~1, exercise up to ~8
    "hepatic_output": (0.0, 4.0),  # mg/min, fasted ~2, suppressed postprandial
    "ghrelin": (40.0, 400.0),      # pg/mL, preprandial high, postprandial low
    "leptin": (1.0, 40.0),         # ng/mL, slow variation
    "glp1": (5.0, 80.0),           # pmol/L, postprandial peaks
    "cortisol": (3.0, 35.0),       # µg/dL, CAR peak ~20-25, nadir ~3-5
    "acth": (10.0, 100.0),         # pg/mL
    "hrv": (10.0, 150.0),          # ms RMSSD
    "rr": (8.0, 30.0),             # /min, rest ~12-16, exercise higher
    "spo2": (90.0, 100.0),         # %
}


def predict(model, ep, embedding):
    initial_state_t = torch.tensor(ep.initial_state, dtype=torch.float32)
    t0 = (
        float(ep.start_time_minutes) % 1440.0
        if ep.start_time_minutes is not None else 360.0
    )
    sw = (
        torch.tensor(ep.sleep_wake[: ep.duration_min], dtype=torch.float32)
        if ep.sleep_wake is not None else None
    )
    act = (
        torch.tensor(ep.activity[: ep.duration_min], dtype=torch.float32)
        if ep.activity is not None else None
    )
    with torch.no_grad():
        return integrate(
            model=model, initial_state=initial_state_t, embedding=embedding,
            n_steps=ep.duration_min, dt=1.0, start_time_minutes=t0,
            meals=ep.meals, sleep_wake=sw, activity=act,
        ).numpy()


def aggregate_marker(traj_per_episode: list[np.ndarray], marker: str) -> dict:
    """Across all episodes, return distribution stats for the marker."""
    idx = MARKER_INDEX[marker]
    all_vals = np.concatenate([t[:, idx] for t in traj_per_episode])
    per_ep_means = np.array([t[:, idx].mean() for t in traj_per_episode])
    per_ep_ranges = np.array([t[:, idx].max() - t[:, idx].min() for t in traj_per_episode])

    lo, hi = PLAUSIBLE_RANGES.get(marker, (-np.inf, np.inf))
    frac_in_range = float(((all_vals >= lo) & (all_vals <= hi)).mean())

    return {
        "global_mean": float(all_vals.mean()),
        "global_std": float(all_vals.std()),
        "global_p05": float(np.quantile(all_vals, 0.05)),
        "global_p50": float(np.quantile(all_vals, 0.50)),
        "global_p95": float(np.quantile(all_vals, 0.95)),
        "per_episode_mean_mean": float(per_ep_means.mean()),
        "per_episode_mean_std": float(per_ep_means.std()),
        "per_episode_range_mean": float(per_ep_ranges.mean()),
        "plausible_range": [lo, hi],
        "fraction_in_plausible_range": frac_in_range,
        "norm_center": float(NORM_CENTER[idx]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--bench", required=True, type=Path)
    ap.add_argument("--calibrate-steps", type=int, default=32)
    ap.add_argument("--calibrate-lr", type=float, default=0.05)
    ap.add_argument("--calibrate-l2", type=float, default=0.001)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    model, _ = load_model_from_checkpoint(args.checkpoint)
    model.eval()
    eps = load_benchmark_dataset(str(args.bench))
    print(f"[probe] {len(eps)} episodes, calibrate-steps={args.calibrate_steps}", flush=True)

    trajectories: list[np.ndarray] = []
    t_start = time.time()
    for i, ep in enumerate(eps, start=1):
        t_ep = time.time()
        cal_obs = measurement_points_from_check_ins(
            [c for c in ep.calibration_check_ins if isinstance(c, dict)],
            ep.duration_min,
        )
        if cal_obs and args.calibrate_steps > 0:
            t0 = (
                float(ep.start_time_minutes) % 1440.0
                if ep.start_time_minutes is not None else 360.0
            )
            sw = (
                torch.tensor(ep.sleep_wake[: ep.duration_min], dtype=torch.float32)
                if ep.sleep_wake is not None else None
            )
            act = (
                torch.tensor(ep.activity[: ep.duration_min], dtype=torch.float32)
                if ep.activity is not None else None
            )
            emb = calibrate_embedding(
                model=model, observations=cal_obs,
                initial_state=torch.tensor(ep.initial_state, dtype=torch.float32),
                meals=ep.meals, duration_min=ep.duration_min,
                start_time_minutes=t0,
                n_steps=args.calibrate_steps,
                lr=args.calibrate_lr, l2_weight=args.calibrate_l2,
                sleep_wake=sw, activity=act,
            ).embedding
        else:
            emb = deterministic_user_embedding(ep.user_id)

        trajectories.append(predict(model, ep, emb))
        print(
            f"[probe] {i}/{len(eps)} {ep.user_id}  "
            f"ep={time.time()-t_ep:.0f}s  total={time.time()-t_start:.0f}s",
            flush=True,
        )

    summary = {
        "checkpoint": str(args.checkpoint),
        "bench": str(args.bench),
        "calibrate_steps": args.calibrate_steps,
        "n_episodes": len(eps),
        "markers": {},
    }
    for marker in UNOBSERVED:
        summary["markers"][marker] = aggregate_marker(trajectories, marker)

    print()
    print(f"{'marker':<16} {'norm':>7} {'mean':>10} {'p05':>10} {'p50':>10} {'p95':>10} {'in-range':>9}  flag")
    for marker, stats in summary["markers"].items():
        plaus = stats["fraction_in_plausible_range"]
        flag = "" if plaus >= 0.95 else (" ⚠" if plaus >= 0.70 else " ✗")
        print(
            f"{marker:<16} "
            f"{stats['norm_center']:>7.2f} "
            f"{stats['global_mean']:>10.3f} "
            f"{stats['global_p05']:>10.3f} "
            f"{stats['global_p50']:>10.3f} "
            f"{stats['global_p95']:>10.3f} "
            f"{plaus:>8.1%}{flag}"
        )

    if args.output:
        args.output.write_text(json.dumps(summary, indent=2))
        print(f"\n[wrote] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
