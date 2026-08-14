#!/usr/bin/env python3
"""Does the STUDENT express the 24 h fasted cascade? (iter-93's primary criterion)

`scripts/iter93_teacher_validate.py` proves the TEACHER expresses it. This proves
whether the trained model inherited it — a separate question, and on the iter-93
artifact the answer was no (0/4 bands, liver glycogen frozen, insulin moving the
wrong way). See docs/iter94-proposal.md §0.1.

Targets are the same literature bands the teacher was validated against:
  glucose 70-85, insulin 2.5-6.5, FFA 0.8-1.4, BHB 0.8-2.2
The teacher lands 78.2 / 3.80 / 0.85 / 1.32 on the default healthy patient.

Protocol mirrors iter93_teacher_validate.py section A: no meals, no activity,
sleep_wake = 1 (awake) throughout, 48 h horizon, start_hour 6.0, read at t=1440
and t=2880.

Usage: uv run python scripts/iter94_student_fast_probe.py [CHECKPOINT.pt]
"""
from __future__ import annotations

import sys
from pathlib import Path

# Repo layout: scripts/ lives next to the pulse/ package. Insert the repo root
# explicitly — a script run from elsewhere puts ITS OWN dir on sys.path[0] and
# can silently import a stale shadowing copy of `pulse` (docs: this produced a
# chain of confident false physiology findings once already).
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

import pulse.knowledge.full_body as fb
from pulse.diagnostics.probe import load_model_from_checkpoint
from pulse.model import integrate
from pulse.types import MARKER_INDEX as MI, NORM_CENTER

import pulse.model as _pm

assert str(_ROOT) in fb.__file__, f"STALE ENGINE: {fb.__file__}"
assert str(_ROOT) in _pm.__file__, f"STALE ENGINE: {_pm.__file__}"
print(f"engine ok: {fb.__file__}")

if len(sys.argv) < 2:
    sys.exit("usage: iter94_student_fast_probe.py CHECKPOINT.pt\n"
             "  (fetch one with: gsutil cp "
             "gs://grovina-pulse-data/training/jobs/<iter>/model.pt /tmp/m.pt)")
CKPT = sys.argv[1]

model, ck = load_model_from_checkpoint(CKPT)
prior_mean = torch.tensor(ck["embedding_prior_mean"], dtype=torch.float32)
print(f"checkpoint: {CKPT}")
print(f"embedding_dim={ck.get('embedding_dim')} hidden={ck.get('hidden_dim')} "
      f"||prior_mean||={float(prior_mean.norm()):.3f}")

DUR = 2880
KEYS = ("glucose", "insulin", "ffa", "bhb", "liver_glycogen", "glucagon",
        "cortisol", "hr", "temp")
BANDS = {  # marker -> (lo, hi, teacher_value)
    "glucose": (70.0, 85.0, 78.2),
    "insulin": (2.5, 6.5, 3.80),
    "ffa": (0.8, 1.4, 0.85),
    "bhb": (0.8, 2.2, 1.32),
}

# --- STUDENT ---------------------------------------------------------------
init = torch.tensor(NORM_CENTER, dtype=torch.float32)
sw = torch.ones(DUR, dtype=torch.float32)      # awake, matches teacher sim()
act = torch.zeros(DUR, dtype=torch.float32)
with torch.no_grad():
    traj = integrate(model, init, prior_mean, n_steps=DUR, dt=1.0,
                     start_time_minutes=360.0, meals=[],
                     sleep_wake=sw, activity=act).numpy()

# --- TEACHER (same protocol, for a like-for-like reference) -----------------
tt, _ = fb.simulate_full_body(fb.PatientParams(), [], np.ones(DUR),
                              np.zeros(DUR), duration_min=DUR, start_hour=6.0,
                              noise_scale=0.0, rng=np.random.default_rng(0))

print("\n=== 24 h FAST CASCADE @ prior-mean embedding (iter-93 primary criterion) ===")
print(f"{'marker':<16}{'t=0':>9}{'@24h':>9}{'@48h':>9}   {'band':<14}"
      f"{'teacher@24h':>12}  verdict")
n_pass = n_banded = 0
for k in KEYS:
    i = MI[k]
    v0, v24, v48 = traj[0, i], traj[1440, i], traj[-1, i]
    t24 = tt[1440, i]
    if k in BANDS:
        lo, hi, _ref = BANDS[k]
        ok = lo <= v24 <= hi
        n_banded += 1
        n_pass += ok
        verdict = "PASS" if ok else "**FAIL**"
        bs = f"[{lo:g}, {hi:g}]"
    else:
        verdict = "(guard)"
        bs = "-"
    print(f"{k:<16}{v0:>9.2f}{v24:>9.2f}{v48:>9.2f}   {bs:<14}{t24:>12.2f}  {verdict}")

print(f"\nbanded criteria: {n_pass}/{n_banded} pass")

# Movement check: the iter-93 thesis is that these markers MOVE at all.
print("\n=== DID THE STUDENT MOVE AT ALL? (iter-93's actual thesis) ===")
for k in ("glucose", "insulin", "ffa", "bhb", "liver_glycogen"):
    i = MI[k]
    s_d = traj[1440, i] - traj[0, i]
    t_d = tt[1440, i] - tt[0, i]
    frac = (s_d / t_d) if abs(t_d) > 1e-6 else float("nan")
    print(f"  {k:<16} student delta_24h {s_d:>8.3f}   teacher delta_24h {t_d:>8.3f}"
          f"   ratio {frac:>6.2f}")

print("\niter-94 acceptance (docs/iter94-proposal.md): liver_glycogen ratio >= 0.5, "
      "bhb >= 0.5,\n  and insulin moving in the CORRECT direction. "
      "iter-93 shipped -0.00 / 0.02 / -0.17.")
