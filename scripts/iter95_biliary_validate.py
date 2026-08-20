#!/usr/bin/env python3
"""Does the TEACHER's enterohepatic loop match the literature it cites?

Checks every quantitative anchor in docs/iter95-biliary-anchors.md against the cold
model, plus two structural behaviours that no single constant can fake.

SECOND-MEAL DEPLETION. A gallbladder cannot be emptied twice. A meal 90 min after the
first must deliver a far smaller ABSOLUTE amount of bile, because the reservoir has
not refilled. This is the test that fails if the gallbladder is modelled as a
contraction fraction over a constant volume — see docs/iter95-proposal.md 3.2.1 for
why that parameterisation was rejected.

CHOLESTASIS RESPONSE. Lowering `k_canalicular` — the hepatocyte->bile export step —
must raise BOTH fasting and postprandial serum bile acids, because first-pass
extraction saturates and more of the portal return spills systemically. That is the
whole reason the export step is explicit rather than lumped into one clearance
constant: it is the hook ALP/GGT/ALT/bilirubin will hang off in a later iteration,
so the enzymes get a real driver instead of shipping inert.

On what is calibrated versus what is earned: the CCK peak (~10 min) and the serum
bile-acid peak (75-120 min) are separated by gallbladder emptying, intestinal
transit, ileal reabsorption and hepatic extraction acting in series. Only `k_ileal`
is calibrated against the serum peak time; everything upstream is pinned by the CCK
and ejection-fraction anchors, so the gap itself is a consequence, not a fit.

Usage: uv run python scripts/iter95_biliary_validate.py
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

import pulse.knowledge.full_body as fb
from pulse.types import MARKER_INDEX as MI

assert str(_ROOT) in fb.__file__, f"STALE ENGINE: {fb.__file__}"
print(f"engine ok: {fb.__file__}")

DUR = 720
# Mixed meal at t=120, after a fasting run-in so the gallbladder has filled.
MEALS = [(120.0, 70.0, 25.0, 30.0)]   # (time, carbs, fats, proteins)
tr, _ = fb.simulate_full_body(fb.PatientParams(), MEALS, np.ones(DUR), np.zeros(DUR),
                              duration_min=DUR, start_hour=6.0, noise_scale=0.0,
                              rng=np.random.default_rng(0))
t0 = 120
cck, gb, inte, ba = (tr[:, MI[k]] for k in
                     ("cck", "gallbladder_bile", "intestinal_bile", "bile_acids"))

print("=== POSTPRANDIAL BILIARY RESPONSE (meal at t=120) ===")
print(f"{'t-meal':>8}{'cck':>9}{'gallbladder':>13}{'intestinal':>12}{'serum BA':>10}")
for dt in (0, 10, 20, 30, 45, 60, 90, 120, 180, 300, 480):
    i = t0 + dt
    if i >= DUR: continue
    print(f"{dt:>8}{cck[i]:>9.2f}{gb[i]:>13.3f}{inte[i]:>12.3f}{ba[i]:>10.2f}")

post = slice(t0, DUR)
cck_peak_i = int(np.argmax(cck[post]))
ba_peak_i = int(np.argmax(ba[post]))
gb0 = gb[t0]
ef60 = 100.0 * (gb0 - gb[t0 + 60]) / gb0
print("\n=== vs LITERATURE (docs/iter95-biliary-anchors.md) ===")
def chk(name, val, lo, hi, unit=""):
    ok = lo <= val <= hi
    print(f"  {name:<38}{val:>8.1f}{unit}   target [{lo}, {hi}]  {'PASS' if ok else '**FAIL**'}")
chk("cck peak time (min after meal)", cck_peak_i, 5, 20)
chk("cck peak (pmol/L)", cck[post][cck_peak_i], 6.5, 7.1)
chk("cck at +30 min (pmol/L)", cck[t0 + 30], 2.5, 4.5)
chk("gallbladder ejection fraction @60min (%)", ef60, 35, 70)
chk("serum BA peak time (min after meal)", ba_peak_i, 75, 120)
chk("serum BA peak (umol/L)", ba[post][ba_peak_i], 4.7, 20.2)
chk("serum BA fasting (umol/L)", ba[t0], 4.4, 14.1)

print("\n=== SECOND MEAL 90 MIN LATER — the reservoir must be depleted ===")
tr2, _ = fb.simulate_full_body(fb.PatientParams(), [(120.0, 70.0, 25.0, 30.0),
                                                    (210.0, 70.0, 25.0, 30.0)],
                               np.ones(DUR), np.zeros(DUR), duration_min=DUR,
                               start_hour=6.0, noise_scale=0.0, rng=np.random.default_rng(0))
gb2 = tr2[:, MI["gallbladder_bile"]]
ef_1 = 100.0 * (gb2[120] - gb2[180]) / gb2[120]
ef_2 = 100.0 * (gb2[210] - gb2[270]) / gb2[210]
print(f"  meal 1 ejection fraction: {ef_1:.1f}%   (gallbladder {gb2[120]:.2f} -> {gb2[180]:.2f} mmol)")
print(f"  meal 2 ejection fraction: {ef_2:.1f}%   (gallbladder {gb2[210]:.2f} -> {gb2[270]:.2f} mmol)")
print(f"  absolute bile delivered:  meal 1 {gb2[120]-gb2[180]:.3f} mmol,"
      f" meal 2 {gb2[210]-gb2[270]:.3f} mmol")

print("\n=== CHOLESTASIS PROBE: drop canalicular export capacity ===")
print(f"{'k_canalicular':>15}{'fasting serum BA':>20}{'peak serum BA':>16}")
for k in (1.0, 0.5, 0.2, 0.05):
    pp = fb.PatientParams(); pp.k_canalicular = k
    trc, _ = fb.simulate_full_body(pp, MEALS, np.ones(DUR), np.zeros(DUR),
                                   duration_min=DUR, start_hour=6.0, noise_scale=0.0,
                                   rng=np.random.default_rng(0))
    b = trc[:, MI["bile_acids"]]
    print(f"{k:>15.2f}{b[t0]:>20.2f}{b[t0:].max():>16.2f}")
