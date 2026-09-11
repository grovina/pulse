#!/usr/bin/env python3
"""The 03:00-06:00 pre-dawn window on the 14 real overnight episodes.

Both remaining gate failures — skill[cgm_real].glucose and skill[cgm_real].hr —
live in this window, and iter-96's thesis is that the TEACHER is wrong there
first. This measures student, teacher and the real data on the same clock with
the same Oura masks, and decomposes the teacher's dHR into the four channels
that feed it, so a change can be attributed to a channel rather than guessed at.

The student runs at the PRIOR-MEAN (zero) embedding, not the calibrated one.
That is deliberate: calibration is a separate, measured problem (see
scripts/iter96_calibration_sweep.py) and 512-step fitting costs ~30 min per
episode, which would make this probe unusable as a routine check.

Usage:
    python scripts/iter96_dawn_probe.py BENCHMARK.json [MODEL.pt]

With no MODEL.pt it reports teacher-vs-real only, which needs no checkpoint and
is the form to use when auditing a teacher change.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

import pulse.knowledge.full_body as fb
from pulse.benchmark import load_benchmark_dataset
from pulse.model import ModularPhysiologyNetwork, integrate
from pulse.types import MARKER_INDEX

# Guard against the stale-shadowing failure mode (a scratchpad copy of `pulse`
# on sys.path[0] silently shadowing the repo, which once produced a chain of
# confident false physiology findings).
assert fb.__file__.startswith(str(_ROOT)), f"shadowed pulse: {fb.__file__}"

WINDOW = (180, 360)   # 03:00 -> 06:00, minutes of day
MARKERS = ("hr", "glucose")


def slope_per_hour(times_min, values) -> float:
    if len(times_min) < 2:
        return float("nan")
    t = np.asarray(times_min, float) / 60.0
    v = np.asarray(values, float)
    return float(np.linalg.lstsq(np.vstack([t, np.ones_like(t)]).T, v, rcond=None)[0][0])


def _load_model(path: str) -> ModularPhysiologyNetwork:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = ModularPhysiologyNetwork.from_checkpoint(blob)
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark")
    ap.add_argument("model", nargs="?", default=None)
    args = ap.parse_args()

    eps = [e for e in load_benchmark_dataset(args.benchmark) if e.source == "cgm_real"]
    if not eps:
        raise SystemExit("no cgm_real episodes in that dataset")
    model = _load_model(args.model) if args.model else None

    rows: list[dict] = []
    for e in eps:
        t0 = float(e.start_time_minutes)
        sw_np = np.asarray(e.sleep_wake, dtype=np.float32)
        act_np = np.asarray(e.activity, dtype=np.float32)

        # --- real data inside the window ---
        real: dict[str, list[tuple[float, float]]] = {}
        for pt in e.eval_measurements:
            if WINDOW[0] <= (t0 + pt.time) % 1440.0 <= WINDOW[1]:
                real.setdefault(pt.marker_id, []).append((pt.time, pt.value))
        for c in e.calibration_check_ins:
            if not isinstance(c, dict):
                continue
            if WINDOW[0] <= (t0 + c["time"]) % 1440.0 <= WINDOW[1]:
                for m, v in (c.get("measurements") or {}).items():
                    real.setdefault(m, []).append((c["time"], float(v)))

        # --- teacher: same masks, same clock, the episode's own start state ---
        p = fb.PatientParams()
        p.Gb = float(e.initial_state[MARKER_INDEX["glucose"]])
        p.HR0 = float(e.initial_state[MARKER_INDEX["hr"]])
        p.HRV0 = float(e.initial_state[MARKER_INDEX["hrv"]])
        p = fb.resolve_derived_params(p)
        traj, _ = fb.simulate_full_body(
            params=p, meals=[(m.time, m.carbs, m.fats, m.proteins) for m in e.meals],
            sleep_wake=sw_np, activity=act_np, duration_min=e.duration_min,
            start_hour=t0 / 60.0, noise_scale=0.0,
        )

        mins = np.arange(e.duration_min)
        mod = (t0 + mins) % 1440.0
        idx = np.where((mod >= WINDOW[0]) & (mod <= WINDOW[1]))[0]
        row = {"user": e.user_id}

        # --- student at the prior-mean embedding ---
        pred = None
        if model is not None:
            with torch.no_grad():
                pred = integrate(
                    model=model,
                    initial_state=torch.tensor(e.initial_state, dtype=torch.float32),
                    embedding=torch.zeros(model.embedding_dim),
                    n_steps=e.duration_min, dt=1.0, start_time_minutes=t0,
                    meals=e.meals,
                    sleep_wake=torch.tensor(sw_np), activity=torch.tensor(act_np),
                ).numpy()

        for mk in MARKERS:
            j = MARKER_INDEX[mk]
            row[f"teacher_{mk}"] = slope_per_hour(mins[idx], traj[idx, j])
            row[f"student_{mk}"] = (slope_per_hour(mins[idx], pred[idx, j])
                                    if pred is not None else float("nan"))
            pts = sorted(real.get(mk, []))
            row[f"real_{mk}"] = slope_per_hour(*zip(*pts)) if len(pts) >= 2 else float("nan")

        # cortisol nadir over the whole overnight episode (acceptance criterion 3)
        ci = MARKER_INDEX["cortisol"]
        row["teacher_cort_min"] = float(traj[:, ci].min())
        row["student_cort_min"] = float(pred[:, ci].min()) if pred is not None else float("nan")

        # --- teacher dHR channel decomposition over the window ---
        hrs = (t0 + mins) / 60.0
        circ = p.hr_circ_amp * np.cos(2 * np.pi * (hrs - 14.0) / 24.0)
        sleep_shift = -p.sleep_hr_frac * p.HR0 * (1.0 - sw_np)
        cort_ch = (p.cort_hr / p.k_hr) * (traj[:, ci] - p.Cort_b)
        act_ch = (p.act_hr_gain / p.k_hr) * act_np
        a, b = idx[0], idx[-1]
        row["d_circ"] = float(circ[b] - circ[a])
        row["d_sleep"] = float(sleep_shift[b] - sleep_shift[a])
        row["d_cort"] = float(cort_ch[b] - cort_ch[a])
        row["d_act"] = float(act_ch[b] - act_ch[a])
        row["d_hr_total"] = float(traj[b, MARKER_INDEX["hr"]] - traj[a, MARKER_INDEX["hr"]])
        rows.append(row)

    def mean(k: str) -> float:
        v = [r[k] for r in rows if not np.isnan(r.get(k, np.nan))]
        return float(np.mean(v)) if v else float("nan")

    print(f"\n{len(rows)} cgm_real episodes, window 03:00-06:00\n")
    print("SLOPE over the window (units/hour), mean across episodes")
    print(f"{'':<10}{'student':>10}{'teacher':>10}{'REAL':>10}")
    for mk in MARKERS:
        print(f"  {mk:<8}{mean(f'student_{mk}'):>10.2f}{mean(f'teacher_{mk}'):>10.2f}"
              f"{mean(f'real_{mk}'):>10.2f}")
    print(f"\ncortisol nadir (ug/dL): student {mean('student_cort_min'):.2f}   "
          f"teacher {mean('teacher_cort_min'):.2f}   physiological 3-5 (Weitzman 1971)")

    print("\nTEACHER dHR DECOMPOSITION -- change in the quasi-static HR target over the window")
    print(f"  circadian  (hr_circ_amp)          {mean('d_circ'):+7.2f} bpm")
    print(f"  sleep      (sleep_hr_frac)        {mean('d_sleep'):+7.2f} bpm")
    print(f"  cortisol   (cort_hr/k_hr)         {mean('d_cort'):+7.2f} bpm")
    print(f"  activity                          {mean('d_act'):+7.2f} bpm")
    print(f"  ---------------------------------------------")
    print(f"  realized teacher HR change        {mean('d_hr_total'):+7.2f} bpm")

    print("\nPER EPISODE (slopes /h)")
    hdr = f"{'user':<22}" + "".join(f"{w:>9}" for w in
                                    ("s_hr", "t_hr", "r_hr", "s_glu", "t_glu", "r_glu"))
    print(hdr)
    for r in rows:
        print(f"{r['user']:<22}"
              f"{r['student_hr']:>9.2f}{r['teacher_hr']:>9.2f}{r['real_hr']:>9.2f}"
              f"{r['student_glucose']:>9.2f}{r['teacher_glucose']:>9.2f}{r['real_glucose']:>9.2f}")


if __name__ == "__main__":
    main()
