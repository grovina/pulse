#!/usr/bin/env python3
"""Sweep the gate's calibration knobs on the real overnight episodes.

WHY THIS EXISTS. Measured on the iter-95 artifact over all 14 `cgm_real`
episodes, with the gate-exact path of the time (512 steps, lr 0.05, l2 0.003,
prior_weight 0.0) and both `sleep_wake` and `activity` passed:

    pooled MAPE     calibrated (= the gate)     uncalibrated prior mean
      hr                 0.0806                        0.0759
      glucose            0.1057                        0.0717
    ||emb|| at the 3.0 clamp: 10 of 14 episodes

Calibration was WORSE than not calibrating, on both gate-blocking markers.

Iter 97 (review 1.3, 1.6, 5.3): three things about that measurement changed.
The calibration objective was integrating a different forward map from the
scorer (window > 0 re-integrated from t=0 with meals on the wrong clock) --
fixed, eval-side. The sweep's `max_norm=1.5` row was a no-op (the module
constant was bound at import) -- every knob is now passed explicitly through
`CalibrationSettings`. And the gate now runs the SHARED calibration
(pulse/calibration.py): chronological hold-out, early stop, acceptance only on
held-out improvement, a prior toward the trained mean, a soft norm penalty.
The knobs worth sweeping are therefore the prior weight and the soft-norm
weight (the old l2 / hard-clamp rows are gone: the clamp is a safety net now).

Skill is printed the way the gate scores it (review 5.1/5.7): episode-first,
in physical units, floored by the device noise:
    skill = 1 - MAE / max(persistence MAE, sigma_obs)

WHY IT IS NOT SET IN THIS COMMIT. These are EVAL-TIME knobs, not trained ones,
so the honest place to set them is on the artifact that will actually be
scored. Run this on the artifact, then pass the winning values to the benchmark
job as env vars.

Usage:
    python scripts/iter96_calibration_sweep.py MODEL.pt BENCHMARK.json [--workers N]

Run it on grovina-mini, not the laptop: a full-length calibration is minutes per
episode per setting single-threaded (early stopping makes it shorter than the
old 512 steps, but 14 episodes x 7 settings is still a fleet job).
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
from pulse.calibration import CalibrationSettings, calibrate_embedding
from pulse.diagnostics.probe import load_model_from_checkpoint
from pulse.model import integrate
from pulse.types import MARKER_INDEX

# (prior_weight, soft_norm_weight, max_norm). The first row is the gate default
# (CalibrationSettings defaults), i.e. the control. Every other knob (lr,
# hold-out fraction, patience, Huber delta) is the gate's, from the env.
SETTINGS: tuple[tuple[float, float, float], ...] = (
    (0.25, 0.10, 8.0),   # control = what the gate does today
    (0.00, 0.10, 8.0),   # no prior, soft norm only
    (0.00, 0.00, 3.0),   # closest to the pre-iter-97 gate (hard clamp does the work)
    (0.10, 0.10, 8.0),
    (0.50, 0.10, 8.0),
    (1.00, 0.10, 8.0),   # full prior -- iter-91 measured this harmful on the OLD ruler
    (0.25, 1.00, 8.0),   # stronger soft norm
)


def _episode_scores(args):
    model_path, ep_index, ep, prior_weight, soft_norm_weight, max_norm = args
    # Reconstruct with the checkpoint's TRAINING-TIME dims. Building
    # ModularPhysiologyNetwork() with library defaults does not load an iter-96
    # blob (hidden_dim=48 -> appetite/stress 24, thermoreg/respiratory 16), and
    # a default build fails on ~40 size mismatches.
    model, blob = load_model_from_checkpoint(model_path)
    for p in model.parameters():
        p.requires_grad_(False)
    # The checkpoint stores the prior as plain lists; wrap exactly as train.py does.
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

    # Iter 97 (review 1.6): EVERY swept knob is passed explicitly; nothing is
    # set by reassigning a module constant.
    settings = CalibrationSettings.from_env(
        prior_weight=prior_weight, soft_norm_weight=soft_norm_weight, max_norm=max_norm,
    )
    res = calibrate_embedding(
        model, cal_obs, init, ep.meals, ep.duration_min,
        start_time_minutes=t0, sleep_wake=sw, activity=act,
        prior_mean=prior_mean, prior_std=prior_std, settings=settings,
    )
    emb = res.embedding

    with torch.no_grad():
        pred = integrate(model=model, initial_state=init, embedding=emb,
                         n_steps=ep.duration_min, dt=1.0, start_time_minutes=t0,
                         meals=ep.meals, sleep_wake=sw, activity=act).numpy()

    last_cal: dict[str, float] = {}
    for obs in sorted(cal_obs, key=lambda o: o.time):
        last_cal[obs.marker_id] = obs.value

    # (model_abs_err, persistence_abs_err) per point; the episode-first,
    # noise-floored skill (review 5.1/5.7) is computed in main.
    out: dict[str, list[tuple[float, float]]] = {}
    for pt in ep.eval_measurements:
        idx = MARKER_INDEX.get(pt.marker_id)
        if idx is None:
            continue
        model_abs = abs(float(pred[pt.time, idx]) - pt.value)
        base = last_cal.get(pt.marker_id, float(ep.initial_state[idx]))
        out.setdefault(pt.marker_id, []).append((model_abs, abs(base - pt.value)))
    return ep_index, float(emb.norm()), bool(res.accepted), int(res.n_steps), out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("benchmark")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    eps = [e for e in bm.load_benchmark_dataset(args.benchmark) if e.source == "cgm_real"]
    print(f"{len(eps)} cgm_real episodes (ONE subject; read as n=1), {args.workers} workers")
    print(f"gate settings from env: {CalibrationSettings.from_env().as_dict()}\n")
    print(f"{'prior_w':>8}{'soft_w':>8}{'clamp':>7} | "
          f"{'hr mae':>8}{'hr skill':>9}{'glu mae':>9}{'glu skill':>10}"
          f"{'accepted':>10}{'steps':>7}{'||emb||':>9}   "
          f"(skill = 1 - MAE/max(persistence MAE, sigma_obs), episode-first)")
    print("-" * 110)

    for prior_weight, soft_norm_weight, max_norm in SETTINGS:
        jobs = [(args.model, i, e, prior_weight, soft_norm_weight, max_norm) for i, e in enumerate(eps)]
        per_episode: dict[str, list[tuple[float, float]]] = {}  # marker -> [(ep mae, ep pers mae)]
        accepted = 0
        steps: list[int] = []
        norms: list[float] = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for _i, norm, acc, n_steps, per_marker in pool.map(_episode_scores, jobs):
                accepted += int(acc)
                steps.append(n_steps)
                norms.append(norm)
                for mk, rows in per_marker.items():
                    per_episode.setdefault(mk, []).append((
                        float(np.mean([r[0] for r in rows])), float(np.mean([r[1] for r in rows]))))
        cells = {}
        for mk in ("hr", "glucose"):
            rows = per_episode.get(mk, [])
            if not rows:
                cells[mk] = (float("nan"), float("nan")); continue
            mae = float(np.mean([a for a, _ in rows]))
            pers = float(np.mean([b for _, b in rows]))
            cells[mk] = (mae, 1.0 - mae / max(pers, bm.sigma_obs_for(mk)))
        tag = "   <- gate default" if (prior_weight, soft_norm_weight, max_norm) == SETTINGS[0] else ""
        print(f"{prior_weight:>8.2f}{soft_norm_weight:>8.2f}{max_norm:>7.1f} | "
              f"{cells['hr'][0]:>8.3f}{cells['hr'][1]:>9.3f}"
              f"{cells['glucose'][0]:>9.3f}{cells['glucose'][1]:>10.3f}"
              f"{accepted:>6}/{len(eps):<3}{np.mean(steps):>7.0f}{np.mean(norms):>9.2f}{tag}")

    print("\nPick the row that maximises BOTH skills, then pass it to the benchmark job:")
    print("  PULSE_BENCHMARK_PRIOR_WEIGHT / PULSE_BENCHMARK_SOFT_NORM_WEIGHT"
          " / PULSE_BENCHMARK_CALIBRATE_MAX_NORM")
    print("A row that only helps one marker is not a win -- the gate needs both. A row where")
    print("nothing is accepted means the prior mean is the best available embedding on this")
    print("artifact: that is a model-quality finding, not a knob to hide.")


if __name__ == "__main__":
    main()
