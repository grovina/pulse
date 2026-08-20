#!/usr/bin/env python3
"""Is each state's DYNAMICAL SHAPE the right shape for what it physically is?

iter 94 found that `typical` was an absorbing floor for the glycogen storage pools, and
read it as specific to `GlycogenFluxHead`. It is not. This script is the instrument that
shows how far it goes, and the answer measured on the iter-94 artifact is: **every
mass-action species in the metabolic module, including insulin.**

Three things it measures.

1. THE ABSORBING FLOOR AT `typical`. The assembly is

       rate = prod·prod_scale − cons·cons_scale·norm_state          prod, cons ≥ 0

   with `norm_state = (raw − typical)/NORM_SCALE`. At raw == typical the consumption
   term is exactly zero, so rate = prod ≥ 0. Below typical, `norm_state < 0` turns
   −cons·norm_state into a POSITIVE source. So no species whose head cannot emit signed
   production can ever fall below its typical value — measured, all six hit
   `min − typical = 0.0000` in every protocol (fast, fed, big meal, hard bout), while
   the teacher takes insulin to 2.81 and hepatic_output to 1.12.

   `SetpointHead` species escape this (their prod = k·target_z/typical is signed); that
   is what iter 51 was actually fixing, and only bhb and mitochondrial_capacity got it.

2. THE TIME CONSTANT, IN THE RIGHT FRAME. Modules receive the NORMALIZED state but their
   rates are applied to the RAW state (model.py: `norm_state = (state − norm_center)/
   norm_scale`, then `rates[:, idx] = met_rates[:, i]`). So

       d(raw)/dt = P − C·(raw − typical)/NORM_SCALE,        C = cons·cons_scale

   and the true relaxation time constant is **NORM_SCALE/C, not 1/C**. Getting this
   wrong misreports tau by a factor of NORM_SCALE per marker — the same frame-error class
   as the iter-90 Sg bug, so it is computed explicitly here rather than left implicit.

3. CLAMP INSIDE THE PHYSIOLOGICAL RANGE. `PHYSIOLOGICAL_MAX = center + 20·NORM_SCALE` is
   meant to be a catastrophe bound, inactive in distribution. Where NORM_SCALE is
   mis-sized it is not: for bhb, NORM_SCALE = 0.05 puts the clamp at 1.10 mmol/L, BELOW
   the teacher's own 1.315 at a 24 h fast (and 3.5 at 48 h).

Loads with strict=False so an OLDER artifact can be audited under NEWER code; any missing
parameter takes its fresh init and is reported, so the reader can see what is not the
artifact's own trained value.

Usage: uv run python scripts/iter95_head_shape_audit.py CHECKPOINT.pt
"""
from __future__ import annotations

import sys
from pathlib import Path

# Repo layout: scripts/ lives next to the pulse/ package. Insert the repo root explicitly
# — a script run from elsewhere puts ITS OWN dir on sys.path[0] and can silently import a
# stale shadowing copy of `pulse` (this produced a chain of false physiology findings once).
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

import pulse.knowledge.full_body as fb
import pulse.model as _pm
from pulse.model import ModularPhysiologyNetwork, integrate
from pulse.modules.gut import MealEvent
from pulse.types import (
    EMBEDDING_DIM, MARKER_INDEX as MI, NORM_CENTER, NORM_SCALE, PHYSIOLOGICAL_MAX,
)

assert str(_ROOT) in fb.__file__, f"STALE ENGINE: {fb.__file__}"
assert str(_ROOT) in _pm.__file__, f"STALE ENGINE: {_pm.__file__}"
print(f"engine ok: {_pm.__file__}")

if len(sys.argv) < 2:
    sys.exit("usage: iter95_head_shape_audit.py CHECKPOINT.pt\n"
             "  (fetch one with: gsutil cp "
             "gs://grovina-pulse-data/training/jobs/<iter>/model.pt /tmp/m.pt)")
CKPT = sys.argv[1]

ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
_h = int(ckpt.get("hidden_dim", 48))
model = ModularPhysiologyNetwork(
    embedding_dim=int(ckpt.get("embedding_dim", EMBEDDING_DIM)),
    metabolic_hidden=_h, appetite_hidden=max(24, _h // 2), stress_hidden=max(24, _h // 2),
    cardiovascular_hidden=_h, thermoreg_hidden=max(16, _h // 3),
    respiratory_hidden=max(16, _h // 3))
_missing, _unexpected = model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
model.eval()
if _missing or _unexpected:
    print(f"load (strict=False): missing={list(_missing)} unexpected={list(_unexpected)}")
    print("  ^ missing parameters hold their FRESH INIT, not this artifact's trained value")
prior_mean = torch.tensor(ckpt["embedding_prior_mean"], dtype=torch.float32)
met = dict(model.named_modules())["metabolic"]

# Species order matches MODULE_MARKER_INDICES["metabolic"] (see modules/metabolic.py).
LOCAL = ["glucose", "insulin", "glucagon", "ffa", "bhb", "lactate", "hepatic_output",
         "liver_glycogen", "muscle_glycogen", "mitochondrial_capacity", "insulin_action"]
# Species whose rate is REPLACED by an explicit structural form in MetabolicModule.forward
# — their head's (prod, cons) is unused, so a collapsed rate constant there is meaningless.
OVERRIDDEN = {"glucose", "insulin_action", "liver_glycogen", "muscle_glycogen"}

captured: dict[int, list[tuple[float, float]]] = {i: [] for i in range(len(met.heads))}


def _make_hook(i: int):
    def hook(_mod, _inp, out):
        prod, cons = out
        captured[i].append((float(prod.reshape(-1)[0]), float(cons.reshape(-1)[0])))
    return hook


for _i, _head in enumerate(met.heads):
    _head.register_forward_hook(_make_hook(_i))


def rollout(meals: list[MealEvent], dur: int, activity: float = 0.0) -> np.ndarray:
    for v in captured.values():
        v.clear()
    with torch.no_grad():
        return integrate(model, torch.tensor(NORM_CENTER, dtype=torch.float32), prior_mean,
                         n_steps=dur, dt=1.0, start_time_minutes=360.0, meals=meals,
                         sleep_wake=torch.ones(dur),
                         activity=torch.full((dur,), activity)).numpy()


# --- 1. the absorbing floor ------------------------------------------------------------
THREE_MEALS = [MealEvent(time=60.0, carbs=65.0, fats=12.0, proteins=25.0),
               MealEvent(time=360.0, carbs=80.0, fats=20.0, proteins=30.0),
               MealEvent(time=720.0, carbs=60.0, fats=15.0, proteins=30.0)]
PROTOCOLS = {
    "48h fast": ([], 2880, 0.0),
    "fed": (THREE_MEALS, 1080, 0.0),
    "big meal": ([MealEvent(time=60.0, carbs=120.0, fats=30.0, proteins=40.0)], 720, 0.0),
    "hard bout": ([], 720, 0.9),
}
FLOOR_MARKERS = ("insulin", "glucagon", "ffa", "bhb", "lactate", "hepatic_output")
mins: dict[str, tuple[float, str]] = {k: (float("inf"), "") for k in FLOOR_MARKERS}
fast_traj = None
for label, (meals, dur, act) in PROTOCOLS.items():
    tr = rollout(meals, dur, act)
    if label == "48h fast":
        fast_traj = tr
    for k in FLOOR_MARKERS:
        v = float(tr[:, MI[k]].min())
        if v < mins[k][0]:
            mins[k] = (v, label)

teacher, _ = fb.simulate_full_body(fb.PatientParams(), [], np.ones(2880), np.zeros(2880),
                                   duration_min=2880, start_hour=6.0, noise_scale=0.0,
                                   rng=np.random.default_rng(0))

print("\n=== 1. CAN A SPECIES GO BELOW ITS `typical` VALUE, IN ANY PROTOCOL? ===")
print(f"{'marker':<18}{'typical':>9}{'student min':>13}{'min - typ':>11}{'protocol':>11}"
      f"{'teacher min':>13}  verdict")
for k in FLOOR_MARKERS:
    typ = float(NORM_CENTER[MI[k]])
    v, lab = mins[k]
    floored = abs(v - typ) < 1e-3
    print(f"{k:<18}{typ:>9.2f}{v:>13.4f}{v - typ:>11.4f}{lab:>11}"
          f"{float(teacher[:, MI[k]].min()):>13.4f}"
          f"  {'**FLOORED AT typical**' if floored else 'ok'}")

# --- 2. time constants, in the raw frame -----------------------------------------------
READ_AT = 1440
rollout([], 2880)  # re-populate `captured` from the fast protocol
print("\n=== 2. EFFECTIVE TIME CONSTANTS @ 24 h fast (tau = NORM_SCALE/C) ===")
print(f"{'species':<24}{'head':<26}{'prod':>10}{'cons':>10}{'tau (min)':>13}  note")
for i, name in enumerate(LOCAL):
    prod, cons = captured[i][READ_AT]
    C = cons * float(met.cons_scale[i])
    tau = (float(NORM_SCALE[MI[name]]) / C) if C > 1e-12 else float("inf")
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

# --- 3. clamp vs physiological range ---------------------------------------------------
print("\n=== 3. IS THE CATASTROPHE CLAMP INSIDE THE PHYSIOLOGICAL RANGE? ===")
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
