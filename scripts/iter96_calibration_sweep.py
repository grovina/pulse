#!/usr/bin/env python3
"""Sweep the gate's embedding-calibration regularization on the real overnight episodes.

WHY THIS EXISTS. Measured on the iter-95 artifact over all 14 `cgm_real`
episodes, with the gate-exact path (512 steps, lr 0.05, l2 0.003,
prior_weight 0.0) and both `sleep_wake` and `activity` passed:

    pooled MAPE     calibrated (= the gate)     uncalibrated prior mean
      hr                 0.0806                        0.0759
      glucose            0.1057                        0.0717
    ||emb|| at the 3.0 clamp: 10 of 14 episodes

Calibration is WORSE than not calibrating, on both gate-blocking markers. The
mechanism is plain over-parameterization: 32 free embedding dimensions fitted to
8-9 noisy check-ins on 2 markers over an 8 h window, then scored on the 3 h that
follows. The clamp binding in 10 of 14 episodes is the tell -- a constraint that
active is not regularization, it is a wall the optimizer is pressed against.

WHY IT IS NOT SET IN THIS COMMIT. These are EVAL-TIME knobs, not trained ones,
so the honest place to set them is on the artifact that will actually be scored.
Tuning them against iter-95's weights and hoping they transfer through a
retrain that changes two module shapes is how a "free win" turns into a
regression. Run this after the iter-96 artifact lands and BEFORE the benchmark
job, then pass the winning values to that job as env vars.

Projected value, from the iter-95 measurement (an estimate, not a promise):
glucose skill -0.694 -> about -0.15, hr -1.165 -> about -1.04. Real and free,
but it clears NEITHER blocker on its own -- the rest is model quality.

Usage:
    python scripts/iter96_calibration_sweep.py MODEL.pt BENCHMARK.json [--workers N]

Run it on grovina-mini, not the laptop: 512-step calibration is ~30 min per
episode per setting single-threaded (see the pulse-grovina-mini-compute-box
note); 14 episodes x 6 settings is a fleet job, not a coffee break.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

from pulse import benchmark as bm
from pulse.diagnostics.probe import load_model_from_checkpoint
from pulse.model import integrate
from pulse.types import MARKER_INDEX

# (prior_weight, max_norm, l2_weight) -- the three ways to shrink the fit.
# The first row is the current gate default, i.e. the control.
SETTINGS: tuple[tuple[float, float, float], ...] = (
    (0.0,  3.0,  0.003),   # control = what the gate does today
    (0.0,  1.5,  0.003),   # tighter clamp only
    (0.0,  3.0,  0.030),   # 10x L2 only
    (0.25, 3.0,  0.003),   # light shrinkage toward the trained prior
    (0.50, 3.0,  0.003),
    (1.0,  3.0,  0.003),   # full prior -- iter-91 measured this as harmful on
                           # the OLD ruler; re-measured here on the new one
)


def _episode_scores(args):
    model_path, ep_index, ep, prior_weight, max_norm, l2 = args
    # Reconstruct with the checkpoint's TRAINING-TIME dims. Building
    # ModularPhysiologyNetwork() with library defaults does not load an iter-96
    # blob (hidden_dim=48 -> appetite/stress 24, thermoreg/respiratory 16), and
    # a default build fails on ~40 size mismatches.
    model, blob = load_model_from_checkpoint(model_path)
    # The checkpoint stores the prior as plain lists; calibrate_embedding calls
    # .detach() on it, so wrap exactly as train.py:1829 does.
    for attr in ("_embedding_prior_mean", "_embedding_prior_std"):
        raw = blob.get(attr.lstrip("_")) if isinstance(blob, dict) else None
        if raw is not None:
            setattr(model, attr, torch.as_tensor(raw, dtype=torch.float32))

    t0 = float(ep.start_time_minutes) % 1440.0 if ep.start_time_minutes is not None else 360.0
    sw = torch.tensor(ep.sleep_wake[:ep.duration_min], dtype=torch.float32) if ep.sleep_wake is not None else None
    act = torch.tensor(ep.activity[:ep.duration_min], dtype=torch.float32) if ep.activity is not None else None
    init = torch.tensor(ep.initial_state, dtype=torch.float32)
    cal_obs = bm.measurement_points_from_check_ins(
        [c for c in ep.calibration_check_ins if isinstance(c, dict)], ep.duration_min)

    prior_mean = getattr(model, "_embedding_prior_mean", None)
    prior_std = getattr(model, "_embedding_prior_std", None)

    old_norm = bm.BENCHMARK_GATE_CALIBRATE_MAX_NORM
    bm.BENCHMARK_GATE_CALIBRATE_MAX_NORM = max_norm
    try:
        emb = bm.calibrate_embedding(
            model=model, observations=cal_obs, initial_state=init, meals=ep.meals,
            duration_min=ep.duration_min, start_time_minutes=t0,
            n_steps=bm.BENCHMARK_GATE_CALIBRATE_STEPS, lr=bm.BENCHMARK_GATE_CALIBRATE_LR,
            l2_weight=l2, sleep_wake=sw, activity=act,
            prior_mean=prior_mean, prior_std=prior_std, prior_weight=prior_weight,
        ).embedding
    finally:
        bm.BENCHMARK_GATE_CALIBRATE_MAX_NORM = old_norm

    with torch.no_grad():
        pred = integrate(model=model, initial_state=init, embedding=emb,
                         n_steps=ep.duration_min, dt=1.0, start_time_minutes=t0,
                         meals=ep.meals, sleep_wake=sw, activity=act).numpy()

    last_cal: dict[str, float] = {}
    for obs in sorted(cal_obs, key=lambda o: o.time):
        last_cal[obs.marker_id] = obs.value

    out: dict[str, list[tuple[float, float]]] = {}
    for pt in ep.eval_measurements:
        idx = MARKER_INDEX.get(pt.marker_id)
        if idx is None:
            continue
        denom = max(abs(pt.value), 1e-6)
        model_err = abs(float(pred[pt.time, idx]) - pt.value) / denom
        base = last_cal.get(pt.marker_id, float(ep.initial_state[idx]))
        pers_err = abs(base - pt.value) / denom
        out.setdefault(pt.marker_id, []).append((model_err, pers_err))
    return ep_index, float(emb.norm()), out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("benchmark")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    eps = [e for e in bm.load_benchmark_dataset(args.benchmark) if e.source == "cgm_real"]
    print(f"{len(eps)} cgm_real episodes, {args.workers} workers\n")
    print(f"{'prior_w':>8}{'max_norm':>9}{'l2':>8} | "
          f"{'hr mape':>9}{'hr skill':>9}{'glu mape':>10}{'glu skill':>10}{'@clamp':>8}")
    print("-" * 74)

    for prior_weight, max_norm, l2 in SETTINGS:
        jobs = [(args.model, i, e, prior_weight, max_norm, l2) for i, e in enumerate(eps)]
        pooled: dict[str, list[tuple[float, float]]] = {}
        at_clamp = 0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for _i, norm, per_marker in pool.map(_episode_scores, jobs):
                if norm >= max_norm - 1e-3:
                    at_clamp += 1
                for mk, pairs in per_marker.items():
                    pooled.setdefault(mk, []).extend(pairs)
        cells = {}
        for mk in ("hr", "glucose"):
            pairs = pooled.get(mk, [])
            if not pairs:
                cells[mk] = (float("nan"), float("nan")); continue
            m = float(np.mean([a for a, _ in pairs]))
            p = float(np.mean([b for _, b in pairs]))
            cells[mk] = (m, 1.0 - m / p if p > 0 else float("nan"))
        tag = "   <- gate default" if (prior_weight, max_norm, l2) == SETTINGS[0] else ""
        print(f"{prior_weight:>8.2f}{max_norm:>9.1f}{l2:>8.3f} | "
              f"{cells['hr'][0]:>9.4f}{cells['hr'][1]:>9.3f}"
              f"{cells['glucose'][0]:>10.4f}{cells['glucose'][1]:>10.3f}"
              f"{at_clamp:>5}/{len(eps)}{tag}")

    print("\nPick the row that maximises BOTH skills, then pass it to the benchmark job:")
    print("  PULSE_BENCHMARK_PRIOR_WEIGHT / PULSE_BENCHMARK_CALIBRATE_MAX_NORM"
          " / PULSE_BENCHMARK_CALIBRATE_L2")
    print("A row that only helps one marker is not a win -- the gate needs both.")


if __name__ == "__main__":
    main()
