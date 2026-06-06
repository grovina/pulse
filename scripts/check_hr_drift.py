"""
Real-data HR drift check — runs Bayesian calibration on every overnight
episode in a real-data corpus, fits an OLS slope through observed and
predicted HR points, prints a per-episode table and the cohort-mean
over-prediction (model_drift − observed_drift, bpm/hour).

Optionally diffs against a baseline checkpoint's saved table so iter-N
vs iter-(N-1) deltas land directly without re-running the baseline.

Usage:
    # Run + save iter-38 baseline
    uv run python scripts/check_hr_drift.py \\
        --checkpoint /tmp/pulse-diag/iter38-checkpoint.pt \\
        --episodes /tmp/pulse-diag/gabriel-real-episodes.json \\
        --output-json /tmp/pulse-diag/hr_drift_iter38.json

    # Run iter 40, diff against iter-38
    uv run python scripts/check_hr_drift.py \\
        --checkpoint /tmp/pulse-diag/iter40-checkpoint.pt \\
        --episodes /tmp/pulse-diag/gabriel-real-episodes.json \\
        --baseline-json /tmp/pulse-diag/hr_drift_iter38.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ENGINE_ROOT = Path(__file__).resolve().parents[1]
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from pulse.benchmark import (  # noqa: E402
    bayesian_calibrate,
    load_benchmark_dataset,
    measurement_points_from_check_ins,
    predictive_distribution,
)
from pulse.diagnostics.probe import load_model_from_checkpoint  # noqa: E402
from pulse.types import MARKER_INDEX  # noqa: E402


def run_episode(model, ep, *, n_steps: int, lr: float, l2_weight: float,
                n_samples: int, sigma_prior: float, seed: int) -> dict | None:
    cal_obs = measurement_points_from_check_ins(
        [c for c in ep.calibration_check_ins if isinstance(c, dict)],
        ep.duration_min,
    )
    sw = (
        torch.tensor(ep.sleep_wake[: ep.duration_min], dtype=torch.float32)
        if ep.sleep_wake is not None else None
    )
    act = (
        torch.tensor(ep.activity[: ep.duration_min], dtype=torch.float32)
        if ep.activity is not None else None
    )
    initial = torch.tensor(ep.initial_state, dtype=torch.float32)
    t0 = float(ep.start_time_minutes) % 1440.0 if ep.start_time_minutes is not None else 360.0

    bayes = bayesian_calibrate(
        model=model, observations=cal_obs, initial_state=initial,
        meals=ep.meals, duration_min=ep.duration_min,
        start_time_minutes=t0, n_steps=n_steps,
        lr=lr, l2_weight=l2_weight,
        sleep_wake=sw, activity=act,
        n_samples=n_samples, sigma_prior=sigma_prior, rng_seed=seed,
    )
    pred = predictive_distribution(
        model=model, samples=bayes.posterior_samples, initial_state=initial,
        meals=ep.meals, duration_min=ep.duration_min,
        start_time_minutes=t0, sleep_wake=sw, activity=act,
    )

    hr_idx = MARKER_INDEX["hr"]
    hr_obs = sorted(
        [(m.time, m.value) for m in ep.eval_measurements if m.marker_id == "hr"],
    )
    if len(hr_obs) < 2:
        return None
    ts = np.array([t for t, _ in hr_obs], dtype=float)
    vs_obs = np.array([v for _, v in hr_obs], dtype=float)
    vs_pred = np.array([float(pred["mean"][int(t), hr_idx]) for t in ts])
    drift_obs = float(np.polyfit(ts, vs_obs, 1)[0] * 60.0)
    drift_pred = float(np.polyfit(ts, vs_pred, 1)[0] * 60.0)
    sleep_frac = (
        sum(1 for x in ep.sleep_wake if x < 0.5) / len(ep.sleep_wake)
        if ep.sleep_wake is not None else float("nan")
    )
    return {
        "user_id": ep.user_id,
        "sleep_frac": sleep_frac,
        "hr_init_pred": float(pred["mean"][0, hr_idx]),
        "hr_init_obs": float(ep.initial_state[hr_idx]),
        "drift_obs_bpm_per_h": drift_obs,
        "drift_pred_bpm_per_h": drift_pred,
        "over_prediction_bpm_per_h": drift_pred - drift_obs,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--episodes", required=True, type=Path)
    ap.add_argument("--baseline-json", type=Path,
                    help="Path to a previous run's --output-json. Adds Δdrift_pred + Δover columns.")
    ap.add_argument("--output-json", type=Path,
                    help="Write the per-episode table to this path (consumed as --baseline-json next iter).")
    ap.add_argument("--calibrate-steps", type=int, default=128)
    ap.add_argument("--calibrate-lr", type=float, default=0.05)
    ap.add_argument("--calibrate-l2", type=float, default=0.001)
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--sigma-prior", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model, _ = load_model_from_checkpoint(args.checkpoint)
    model.eval()
    eps = load_benchmark_dataset(str(args.episodes))
    if not eps:
        print("no episodes loaded", file=sys.stderr)
        return 1

    baseline = None
    if args.baseline_json is not None:
        baseline = json.loads(args.baseline_json.read_text())["episodes"]

    rows: list[dict] = []
    for i, ep in enumerate(eps):
        row = run_episode(
            model, ep,
            n_steps=args.calibrate_steps, lr=args.calibrate_lr,
            l2_weight=args.calibrate_l2, n_samples=args.n_samples,
            sigma_prior=args.sigma_prior, seed=args.seed,
        )
        if row is None:
            print(f"ep {i}: skipped (fewer than 2 HR eval points)", file=sys.stderr)
            continue
        row["index"] = i
        rows.append(row)

    if baseline is not None:
        header = (
            f"{'ep':>3} {'sleep%':>7} {'init_p':>7} {'init_o':>7} "
            f"{'d_obs':>8} {'d_pred':>8} {'Δd_pred':>9} {'over':>8} {'Δover':>8}"
        )
    else:
        header = (
            f"{'ep':>3} {'sleep%':>7} {'init_p':>7} {'init_o':>7} "
            f"{'d_obs':>8} {'d_pred':>8} {'over':>8}"
        )
    print(header)
    base_by_idx = {b["index"]: b for b in baseline} if baseline else {}
    for r in rows:
        line = (
            f"{r['index']:>3} {r['sleep_frac']:>6.0%}  "
            f"{r['hr_init_pred']:>6.1f} {r['hr_init_obs']:>6.1f} "
            f"{r['drift_obs_bpm_per_h']:>+7.2f} {r['drift_pred_bpm_per_h']:>+7.2f}"
        )
        if baseline is not None:
            b = base_by_idx.get(r["index"])
            if b is not None:
                d_pred = r["drift_pred_bpm_per_h"] - b["drift_pred_bpm_per_h"]
                d_over = r["over_prediction_bpm_per_h"] - b["over_prediction_bpm_per_h"]
                line += f" {d_pred:>+8.2f} {r['over_prediction_bpm_per_h']:>+7.2f} {d_over:>+7.2f}"
            else:
                line += f" {'-':>9} {r['over_prediction_bpm_per_h']:>+7.2f} {'-':>8}"
        else:
            line += f" {r['over_prediction_bpm_per_h']:>+7.2f}"
        print(line)

    obs_mean = float(np.mean([r["drift_obs_bpm_per_h"] for r in rows]))
    pred_mean = float(np.mean([r["drift_pred_bpm_per_h"] for r in rows]))
    over_mean = float(np.mean([r["over_prediction_bpm_per_h"] for r in rows]))
    print()
    print(f"mean drift_obs        = {obs_mean:+.2f} bpm/h")
    print(f"mean drift_pred       = {pred_mean:+.2f} bpm/h")
    print(f"mean over-prediction  = {over_mean:+.2f} bpm/h  (target: 0.00)")
    if baseline is not None:
        base_over = float(np.mean([b["over_prediction_bpm_per_h"] for b in baseline]))
        print(f"baseline over-pred    = {base_over:+.2f} bpm/h  (Δ = {over_mean - base_over:+.2f})")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps({
            "checkpoint": str(args.checkpoint),
            "episodes_path": str(args.episodes),
            "n_samples": args.n_samples,
            "sigma_prior": args.sigma_prior,
            "summary": {
                "drift_obs_mean": obs_mean,
                "drift_pred_mean": pred_mean,
                "over_prediction_mean": over_mean,
            },
            "episodes": rows,
        }, indent=2))
        print(f"\nwrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
