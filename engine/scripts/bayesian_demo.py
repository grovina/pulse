"""
Bayesian-calibration demo — runs Laplace-approximation calibration on a
single bench-shaped episode, prints MAP estimate, posterior uncertainty,
predictive intervals at eval times, and 95%-PI coverage on held-out
eval points.

Designed to run on real-data episodes (e.g. produced by
``ingest_real_data.py``) with cheap calibration settings, so a single
episode finishes in a couple of minutes locally.

Usage:
    uv run python scripts/bayesian_demo.py \\
        --checkpoint /tmp/iter38.pt \\
        --episodes /tmp/gabriel-real-episodes.json \\
        --episode-index 0 \\
        --calibrate-steps 128 --n-samples 20
"""
from __future__ import annotations

import argparse
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--episodes", required=True, type=Path)
    ap.add_argument("--episode-index", type=int, default=0)
    ap.add_argument("--calibrate-steps", type=int, default=128)
    ap.add_argument("--calibrate-lr", type=float, default=0.05)
    ap.add_argument("--calibrate-l2", type=float, default=0.001)
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--sigma-prior", type=float, default=0.1,
                    help="Prior std on embedding dims (default 0.1, matches training-time init).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model, _ = load_model_from_checkpoint(args.checkpoint)
    model.eval()
    eps = load_benchmark_dataset(str(args.episodes))
    if not eps:
        print("no episodes loaded", file=sys.stderr)
        return 1
    if args.episode_index < 0 or args.episode_index >= len(eps):
        print(f"episode-index out of range (0..{len(eps)-1})", file=sys.stderr)
        return 1
    ep = eps[args.episode_index]

    print(f"=== {ep.user_id} ===")
    print(f"duration: {ep.duration_min} min  start_h: {(ep.start_time_minutes or 360)/60:.1f}")
    print(f"calibration check-ins: {len(ep.calibration_check_ins)}  eval points: {len(ep.eval_measurements)}")

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
    initial_state_t = torch.tensor(ep.initial_state, dtype=torch.float32)
    t0 = float(ep.start_time_minutes) % 1440.0 if ep.start_time_minutes is not None else 360.0

    bayes = bayesian_calibrate(
        model=model, observations=cal_obs,
        initial_state=initial_state_t,
        meals=ep.meals, duration_min=ep.duration_min,
        start_time_minutes=t0,
        n_steps=args.calibrate_steps,
        lr=args.calibrate_lr, l2_weight=args.calibrate_l2,
        sleep_wake=sw, activity=act,
        n_samples=args.n_samples, sigma_prior=args.sigma_prior,
        rng_seed=args.seed,
    )

    print()
    print(f"MAP loss          : {bayes.map_loss:.4f}")
    print(f"½·log|Σ| (entropy): {bayes.log_det_cov:.3f}  (lower = more constrained)")
    print(f"posterior std per dim: mean={float(torch.diagonal(bayes.posterior_cov).sqrt().mean()):.4f}  "
          f"max={float(torch.diagonal(bayes.posterior_cov).sqrt().max()):.4f}")

    pred = predictive_distribution(
        model=model, samples=bayes.posterior_samples,
        initial_state=initial_state_t, meals=ep.meals,
        duration_min=ep.duration_min, start_time_minutes=t0,
        sleep_wake=sw, activity=act,
    )

    # Per-marker predictive intervals at eval times + coverage.
    print()
    print(f"{'marker':<10} {'t':>5} {'actual':>8} {'mean':>8} {'std':>8} {'p05':>8} {'p95':>8}  in-95%")
    by_marker: dict[str, list[tuple[bool, float]]] = {}
    for m in ep.eval_measurements:
        idx = MARKER_INDEX.get(m.marker_id)
        if idx is None:
            continue
        t = int(m.time)
        if t < 0 or t >= pred["mean"].shape[0]:
            continue
        mn = float(pred["mean"][t, idx])
        sd = float(pred["std"][t, idx])
        p05 = float(pred["p05"][t, idx])
        p95 = float(pred["p95"][t, idx])
        inside = p05 <= m.value <= p95
        by_marker.setdefault(m.marker_id, []).append((inside, abs(mn - m.value)))
        flag = " ✓" if inside else " ✗"
        print(f"{m.marker_id:<10} {t:>5} {m.value:>8.1f} {mn:>8.1f} {sd:>8.2f} {p05:>8.1f} {p95:>8.1f}  {flag}")

    print()
    print("--- coverage summary ---")
    for marker, results in by_marker.items():
        coverage = sum(1 for inside, _ in results if inside) / len(results)
        mean_abs_err = sum(d for _, d in results) / len(results)
        print(f"{marker:<10}  n={len(results):<3}  in-95%-PI={coverage:.0%}  mean |error|={mean_abs_err:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
