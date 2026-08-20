#!/usr/bin/env python3
"""Is each state's DYNAMICAL SHAPE the right shape for what it physically is?

iter 94 found that `typical` was an absorbing floor for the glycogen storage
pools — the mass-action assembly `prod·prod_scale − cons·cons_scale·norm_state`
has a vanishing breakdown term exactly at typical. That finding generalizes, and
this script is the instrument that shows how far.

It reports, per metabolic species: the head class, its emitted (prod, cons) on a
fasted rollout, the resulting EFFECTIVE RATE CONSTANT k = cons·cons_scale, and
the implied time constant τ = 1/k. A τ far longer than the marker's physiological
response time means the species cannot move on the timescale it is supposed to —
whatever the supervision does.

Two failure signatures it makes visible, both measured on the iter-94 artifact:

  RATE-CONSTANT COLLAPSE. `SetpointHead` emits k_factor = softplus(raw), and the
  rate is `cons_scale·k_factor·(target_z − norm_state)`. If k_factor → 0 the
  species freezes AND the gradient onto target_z vanishes with it — self-locking.
  Measured on bhb: k_factor = 1e-5, τ = 12 years, while target_z sat at +2.64
  (the head "wanted" 0.23 mmol/L and had no authority to get there). This is NOT
  a bug to constrain away: given a setpoint form for a species that is actually a
  driven flux product, zeroing the rate constant is the correct solution to the
  problem as posed. It is the shape reporting itself.

  CLAMP INSIDE THE PHYSIOLOGICAL RANGE. PHYSIOLOGICAL_MAX = center + 20·NORM_SCALE
  is meant to be a catastrophe bound, inactive in distribution. Where NORM_SCALE
  is mis-sized it is not: measured on bhb, NORM_SCALE = 0.05 puts the clamp at
  1.10 mmol/L, BELOW the teacher's own 1.315 at a 24 h fast (and 3.5 at 48 h).

Usage: uv run python scripts/iter95_head_shape_audit.py CHECKPOINT.pt
"""
from __future__ import annotations

import sys
from pathlib import Path

# Repo layout: scripts/ lives next to the pulse/ package. Insert the repo root
# explicitly — a script run from elsewhere puts ITS OWN dir on sys.path[0] and
# can silently import a stale shadowing copy of `pulse` (this produced a chain of
# confident false physiology findings once already).
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

import pulse.knowledge.full_body as fb
import pulse.model as _pm
from pulse.diagnostics.probe import load_model_from_checkpoint
from pulse.model import integrate
from pulse.types import MARKER_INDEX as MI, NORM_CENTER, NORM_SCALE, PHYSIOLOGICAL_MAX

assert str(_ROOT) in fb.__file__, f"STALE ENGINE: {fb.__file__}"
assert str(_ROOT) in _pm.__file__, f"STALE ENGINE: {_pm.__file__}"
print(f"engine ok: {_pm.__file__}")

if len(sys.argv) < 2:
    sys.exit("usage: iter95_head_shape_audit.py CHECKPOINT.pt\n"
             "  (fetch one with: gsutil cp "
             "gs://grovina-pulse-data/training/jobs/<iter>/model.pt /tmp/m.pt)")
CKPT = sys.argv[1]

model, ck = load_model_from_checkpoint(CKPT)
prior_mean = torch.tensor(ck["embedding_prior_mean"], dtype=torch.float32)
met = dict(model.named_modules())["metabolic"]

# Species order matches MODULE_MARKER_INDICES["metabolic"] (see modules/metabolic.py).
LOCAL = ["glucose", "insulin", "glucagon", "ffa", "bhb", "lactate", "hepatic_output",
         "liver_glycogen", "muscle_glycogen", "mitochondrial_capacity", "insulin_action"]
# Species whose rate is REPLACED by an explicit structural form in
# MetabolicModule.forward — their head's (prod, cons) is unused, so a collapsed
# rate constant there is expected and meaningless.
OVERRIDDEN = {"glucose", "insulin_action", "liver_glycogen", "muscle_glycogen"}

captured: dict[int, list[tuple[float, float]]] = {i: [] for i in range(len(met.heads))}


def _make_hook(i: int):
    def hook(_mod, _inp, out):
        prod, cons = out
        captured[i].append((float(prod.reshape(-1)[0]), float(cons.reshape(-1)[0])))
    return hook


for i, head in enumerate(met.heads):
    head.register_forward_hook(_make_hook(i))

# Protocol mirrors iter94_student_fast_probe.py: 48 h fast, awake, no meals.
DUR = 2880
READ_AT = 1440
with torch.no_grad():
    traj = integrate(model, torch.tensor(NORM_CENTER, dtype=torch.float32), prior_mean,
                     n_steps=DUR, dt=1.0, start_time_minutes=360.0, meals=[],
                     sleep_wake=torch.ones(DUR), activity=torch.zeros(DUR)).numpy()

print("\n=== EFFECTIVE RATE CONSTANTS @ 24 h fast ===")
print(f"{'species':<24}{'head':<26}{'prod':>10}{'cons':>10}{'tau (min)':>13}  note")
for i, name in enumerate(LOCAL):
    prod, cons = captured[i][READ_AT]
    k = cons * float(met.cons_scale[i])
    tau = (1.0 / k) if k > 1e-12 else float("inf")
    if name in OVERRIDDEN:
        note = "(head unused — structural rate)"
    elif cons < 1e-3:
        note = "<== RATE CONSTANT COLLAPSED"
    elif tau > 10_000:
        note = "<== inert on any protocol timescale"
    else:
        note = ""
    print(f"{name:<24}{met.heads[i].__class__.__name__:<26}"
          f"{prod:>10.5f}{cons:>10.5f}{tau:>13.1f}  {note}")

print("\n=== IS THE CATASTROPHE CLAMP INSIDE THE PHYSIOLOGICAL RANGE? ===")
teacher, _ = fb.simulate_full_body(fb.PatientParams(), [], np.ones(DUR), np.zeros(DUR),
                                   duration_min=DUR, start_hour=6.0, noise_scale=0.0,
                                   rng=np.random.default_rng(0))
print(f"{'marker':<22}{'center':>9}{'scale':>8}{'clamp_max':>11}"
      f"{'teach@24h':>11}{'teach@48h':>11}  verdict")
for name in ("glucose", "insulin", "glucagon", "ffa", "bhb", "lactate",
             "hepatic_output", "liver_glycogen", "cortisol"):
    i = MI[name]
    cmax = PHYSIOLOGICAL_MAX[i]
    t24, t48 = teacher[READ_AT, i], teacher[-1, i]
    bad = max(t24, t48) > cmax
    print(f"{name:<22}{NORM_CENTER[i]:>9.2f}{NORM_SCALE[i]:>8.2f}{cmax:>11.2f}"
          f"{t24:>11.3f}{t48:>11.3f}  {'**TEACHER EXCEEDS CLAMP**' if bad else 'ok'}")
