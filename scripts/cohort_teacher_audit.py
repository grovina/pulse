#!/usr/bin/env python3
"""Audit the TEACHER against every cohort literature statistic.

Both the cold-model teacher and the cohort statistics supervise the same student, so
wherever they disagree the student is pulled two ways at once. This reports
z = (teacher_realized - target) / sigma for all of them, worst first.

Iter 93 was found this way: 8/31 contradicted at |z|>=2, and chasing the largest
exposed a teacher whose fasted state never engaged. Re-run it after ANY teacher
change -- a fix that quietly breaks three other literature anchors is not a fix.

Usage: python scripts/cohort_teacher_audit.py [N_PATIENTS]"""

from __future__ import annotations

import sys
from pathlib import Path

# Repo layout: scripts/ lives next to package pulse/
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

import pulse.knowledge.full_body as fb
from pulse.knowledge.cohort_statistics import ALL_COHORT_STATISTICS
from pulse.knowledge.cohort_types import StatisticKind
from pulse.types import MARKER_INDEX

N = int(sys.argv[1]) if len(sys.argv) > 1 else 12


def series(arm, params, marker, seed):
    n = arm.duration_min
    sw = np.array(arm.sleep_wake, dtype=float) if arm.sleep_wake is not None else np.ones(n)
    act = np.array(arm.activity, dtype=float) if arm.activity is not None else np.zeros(n)
    sw = np.resize(sw, n); act = np.resize(act, n)
    traj, _ = fb.simulate_full_body(params, list(arm.meals), sw, act,
                                    duration_min=n, start_hour=arm.start_hour,
                                    noise_scale=0.0, rng=np.random.default_rng(seed))
    return traj[:, MARKER_INDEX[marker]]


def arm_stat(spec, arm_idx, arm, params, seed):
    w = (spec.per_arm_windows[arm_idx] if spec.per_arm_windows else spec.window)
    s = series(arm, params, spec.marker_id, seed)
    seg = s[w.start_min:w.end_min]
    if spec.kind in (StatisticKind.MEAN_IN_WINDOW, StatisticKind.DELTA_MEANS):
        return float(seg.mean())
    if spec.kind in (StatisticKind.PEAK_VALUE, StatisticKind.DELTA_PEAKS):
        return float(seg.max())
    if spec.kind is StatisticKind.TIME_TO_PEAK:
        return float(seg.argmax())
    raise ValueError(spec.kind)


rows = []
for spec in ALL_COHORT_STATISTICS:
    vals = []
    rng = np.random.default_rng(17)
    for k in range(N):
        p = fb.randomize_params(rng)
        try:
            per_arm = [arm_stat(spec, i, a, p, 5000 + k) for i, a in enumerate(spec.arms)]
        except Exception as e:
            vals = []
            print(f"{spec.name}: ERROR {e}")
            break
        if spec.kind in (StatisticKind.DELTA_MEANS, StatisticKind.DELTA_PEAKS):
            vals.append(per_arm[1] - per_arm[0])
        else:
            vals.append(per_arm[0])
    if not vals:
        continue
    m, sd = float(np.mean(vals)), float(np.std(vals))
    z = (m - spec.target) / spec.sigma
    rows.append((abs(z), spec.name, spec.marker_id, m, sd, spec.target, spec.sigma, z))

rows.sort(reverse=True)
print(f"{'|z|':>6} {'statistic':44} {'marker':12} {'teacher':>10} {'sd':>7} "
      f"{'target':>9} {'sigma':>7}")
for az, name, marker, m, sd, tgt, sig, z in rows:
    flag = "  <<< CONTRADICTS" if az >= 2 else ("  <- watch" if az >= 1 else "")
    print(f"{z:+6.2f} {name:44} {marker:12} {m:10.2f} {sd:7.2f} {tgt:9.2f} {sig:7.2f}{flag}")
print(f"\n{sum(1 for r in rows if r[0] >= 2)}/{len(rows)} statistics contradict the teacher "
      f"at |z|>=2 (N={N} randomized patients)")
