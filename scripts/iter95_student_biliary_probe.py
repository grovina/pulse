#!/usr/bin/env python3
"""Did the STUDENT inherit the enterohepatic loop, or is the axis decorative?

`scripts/iter95_biliary_validate.py` proves the TEACHER expresses it. This proves
whether the trained model does — a separate question, and the one iter 93 got wrong
about the fasted cascade (the teacher was fixed; the student inherited nothing).

The failure mode this is built to catch is a marker that ships as a CONSTANT. Per the
iter-94 ruler finding, the gate cannot tell a constant from a simulator, so a new axis
that quietly flatlines would look like success. Hence the first check is simply: does
anything move at all?

Then the three quantitative behaviours, against docs/iter95-biliary-anchors.md:

  cck peak time and height        the FAST arm (~10 min, 6.5-7.1 pmol/L)
  bile-acid peak time             the SLOW arm (75-120 min)
  gallbladder ejection @60 min    >=35-38% (HIDA)

and the structural one no constant can fake: a second meal 90 min later must deliver a
far smaller ABSOLUTE amount of bile, because the reservoir has not refilled.

Usage: uv run python scripts/iter95_student_biliary_probe.py CHECKPOINT.pt
"""
from __future__ import annotations

import sys
from pathlib import Path

# Repo layout: scripts/ lives next to the pulse/ package. Insert the repo root
# explicitly — a script run from elsewhere puts ITS OWN dir on sys.path[0] and can
# silently import a stale shadowing copy of `pulse`.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

import pulse.knowledge.full_body as fb
import pulse.model as _pm
from pulse.model import ModularPhysiologyNetwork, integrate
from pulse.modules.gut import MealEvent
from pulse.types import EMBEDDING_DIM, MARKER_INDEX as MI, NORM_CENTER

assert str(_ROOT) in fb.__file__, f"STALE ENGINE: {fb.__file__}"
assert str(_ROOT) in _pm.__file__, f"STALE ENGINE: {_pm.__file__}"
print(f"engine ok: {_pm.__file__}")

if len(sys.argv) < 2:
    sys.exit("usage: iter95_student_biliary_probe.py CHECKPOINT.pt")
CKPT = sys.argv[1]

ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
_h = int(ckpt.get("hidden_dim", 48))
model = ModularPhysiologyNetwork(
    embedding_dim=int(ckpt.get("embedding_dim", EMBEDDING_DIM)),
    metabolic_hidden=_h, appetite_hidden=max(24, _h // 2), stress_hidden=max(24, _h // 2),
    cardiovascular_hidden=_h, thermoreg_hidden=max(16, _h // 3),
    respiratory_hidden=max(16, _h // 3))
missing, unexpected = model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
model.eval()
if missing or unexpected:
    print(f"load (strict=False): missing={list(missing)} unexpected={list(unexpected)}")
prior_mean = torch.tensor(ckpt["embedding_prior_mean"], dtype=torch.float32)

DUR, T0 = 720, 120
KEYS = ("cck", "gallbladder_bile", "intestinal_bile", "bile_acids")


def rollout(meals: list[MealEvent], dur: int = DUR) -> np.ndarray:
    with torch.no_grad():
        return integrate(model, torch.tensor(NORM_CENTER, dtype=torch.float32), prior_mean,
                         n_steps=dur, dt=1.0, start_time_minutes=360.0, meals=meals,
                         sleep_wake=torch.ones(dur), activity=torch.zeros(dur)).numpy()


tr = rollout([MealEvent(time=float(T0), carbs=70.0, fats=25.0, proteins=30.0)])
cck, gb, inte, ba = (tr[:, MI[k]] for k in KEYS)

print("\n=== 0. IS THE AXIS ALIVE AT ALL? (a constant would pass the gate) ===")
alive = True
for k in KEYS:
    v = tr[:, MI[k]]
    rng = float(v.max() - v.min())
    ok = rng > 1e-3
    alive &= ok
    print(f"  {k:<20} range {rng:>9.4f}   {'moves' if ok else '**INERT**'}")

print("\n=== 1. POSTPRANDIAL RESPONSE (meal at t=120) ===")
print(f"{'t-meal':>8}{'cck':>9}{'gallbladder':>13}{'intestinal':>12}{'serum BA':>10}")
for dt in (0, 10, 20, 30, 60, 90, 120, 180, 300):
    i = T0 + dt
    if i < DUR:
        print(f"{dt:>8}{cck[i]:>9.2f}{gb[i]:>13.3f}{inte[i]:>12.3f}{ba[i]:>10.2f}")

print("\n=== 2. vs LITERATURE (docs/iter95-biliary-anchors.md) ===")
n_pass = n_tot = 0


def chk(name: str, val: float, lo: float, hi: float) -> None:
    global n_pass, n_tot
    n_tot += 1
    ok = lo <= val <= hi
    n_pass += ok
    print(f"  {name:<40}{val:>8.1f}   target [{lo}, {hi}]  {'PASS' if ok else '**FAIL**'}")


gb0 = gb[T0]
chk("cck peak time (min after meal)", float(np.argmax(cck[T0:])), 5, 25)
chk("cck peak (pmol/L)", float(cck[T0:].max()), 5.0, 9.0)
chk("serum BA peak time (min after meal)", float(np.argmax(ba[T0:])), 60, 140)
chk("serum BA peak (umol/L)", float(ba[T0:].max()), 4.7, 20.2)
chk("gallbladder ejection @60min (%)", float(100.0 * (gb0 - gb[T0 + 60]) / gb0), 30, 75)

print("\n=== 3. SECOND MEAL 90 MIN LATER — the reservoir must be depleted ===")
tr2 = rollout([MealEvent(time=120.0, carbs=70.0, fats=25.0, proteins=30.0),
               MealEvent(time=210.0, carbs=70.0, fats=25.0, proteins=30.0)])
g2 = tr2[:, MI["gallbladder_bile"]]
d1, d2 = g2[120] - g2[180], g2[210] - g2[270]
print(f"  meal 1 delivered {d1:.3f} mmol   (gallbladder {g2[120]:.2f} -> {g2[180]:.2f})")
print(f"  meal 2 delivered {d2:.3f} mmol   (gallbladder {g2[210]:.2f} -> {g2[270]:.2f})")
depleted = d2 < 0.75 * d1
print(f"  second meal delivers less: {'PASS' if depleted else '**FAIL** — pool not depleting'}")

print(f"\nquantitative anchors {n_pass}/{n_tot};  axis alive: {alive};  "
      f"reservoir depletes: {depleted}")
