#!/usr/bin/env python3
"""Iter-93 teacher validation against literature. Every number here is a spec claim.

Sections A/A2 are the iteration's thesis (the fasted state), B guards iter-92's meal
kinetics against regression, C/D are the meal->CV and exercise-lactate fixes, and E is
the guardrail set: things that must NOT have moved.

Literature bands apply to the DEFAULT (healthy) patient. The randomized population
deliberately includes impaired/diabetic patients, so its MEAN is not what Polonsky or
Cahill measured -- reporting a population mean against a healthy-subject band is how
you get a false failure. Population spread is printed unbanded, for stability only.

Usage: python scripts/iter93_teacher_validate.py [N_PATIENTS]"""

from __future__ import annotations

import sys
from pathlib import Path

# Repo layout: scripts/ lives next to package pulse/
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pulse.knowledge.full_body as fb
from pulse.types import MARKER_INDEX as MI

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
G, I, F, B, LG, HR, HRV, LAC = (MI["glucose"], MI["insulin"], MI["ffa"], MI["bhb"],
                                MI["liver_glycogen"], MI["hr"], MI["hrv"], MI["lactate"])


def sim(p, meals, dur, act=None, sw=None, start=6.0, seed=0):
    sw = np.ones(dur) if sw is None else sw
    act = np.zeros(dur) if act is None else act
    tr, _ = fb.simulate_full_body(p, meals, sw, act, duration_min=dur, start_hour=start,
                                  noise_scale=0.0, rng=np.random.default_rng(seed))
    return tr


def population(seed=42):
    rng = np.random.default_rng(seed)
    return [fb.randomize_params(rng) for _ in range(N)]


pop = population()


def band(vals, lo, hi, label, unit=""):
    v = np.array(vals, dtype=float)
    ok = lo <= v.mean() <= hi
    print(f"  {'PASS' if ok else 'FAIL'}  {label:46} {v.mean():8.2f}{unit} "
          f"[{v.min():.2f}, {v.max():.2f}]   target {lo}-{hi}")
    return ok


results = []
print("\n=== A. THE FASTED STATE (the iteration's thesis) ===")
print("   [literature bands apply to the DEFAULT healthy patient; the randomized")
print("    population deliberately includes impaired/diabetic patients]")
d = sim(fb.PatientParams(), [], 2880)
fast = [d]
results.append(band([t[1440, G] for t in fast], 70, 85, "glucose @ 24 h fast (Cahill 2006)", " mg/dL"))
results.append(band([t[1440, I] for t in fast], 2.5, 6.5, "insulin @ 24 h fast (Polonsky 1988)", " uU/mL"))
results.append(band([t[1440, F] for t in fast], 0.8, 1.4, "FFA @ 24 h fast (Cahill 2006)", " mmol/L"))
results.append(band([t[1440, B] for t in fast], 0.8, 2.2, "BHB @ 24 h fast (unchanged, guard)", " mmol/L"))
results.append(band([t[-1, G] for t in fast], 60, 75, "glucose @ 48 h fast (Cahill 2006)", " mg/dL"))
results.append(band([t[-1, F] for t in fast], 0.9, 1.6, "FFA @ 48 h fast", " mmol/L"))
popfast = [sim(p, [], 1440, seed=k) for k, p in enumerate(pop)]
print("   population spread @24h (no band): glucose %.1f [%.0f, %.0f]  insulin %.1f  FFA %.2f" % (
    np.mean([t[-1, G] for t in popfast]), min(t[-1, G] for t in popfast),
    max(t[-1, G] for t in popfast), np.mean([t[-1, I] for t in popfast]),
    np.mean([t[-1, F] for t in popfast])))

print("\n=== A2. OVERNIGHT (the real-data ruler's regime) ===")
# 12 h from 19:00, no meals — exactly the real episodes' shape
night = [sim(p, [(0.0, 70.0, 30.0, 25.0)], 720, start=19.0, seed=100 + k) for k, p in enumerate(pop)]
slopes = [np.polyfit(np.arange(240, 720), t[240:720, G], 1)[0] * 60 for t in night]
results.append(band(slopes, -3.5, -0.6, "deep-night glucose slope (user CGM: -2.37)", " mg/dL/h"))
results.append(band([t[120, G] - t[-1, G] for t in night], 8, 40, "post-dinner -> waking fall (user CGM: ~14)", " mg/dL"))

print("\n=== B. FED / POSTPRANDIAL MUST NOT MOVE (iter-92's win) ===")
meal = [sim(p, [(30.0, 65.0, 20.0, 25.0)], 300, seed=200 + k) for k, p in enumerate(pop)]
tpeaks = [int(t[30:, G].argmax()) for t in meal]
results.append(band(tpeaks, 45, 60, "glucose peak time (iter-92 held)", " min"))
results.append(band([t[30:, G].max() - t[30, G] for t in meal], 40, 55, "glucose rise", " mg/dL"))
results.append(band([t[:, I].max() for t in meal], 40, 60, "insulin peak", " uU/mL"))
ghr = MI["ghrelin"]
results.append(band([100 * (t[30:, ghr].min() - t[30, ghr]) / t[30, ghr] for t in meal],
                    -50, -30, "ghrelin nadir (Cummings 2001)", " %"))
results.append(band([100 * (t[30:, F].min() - t[30, F]) / t[30, F] for t in meal],
                    -80, -55, "postprandial FFA suppression (Frayn)", " %"))

print("\n=== C. MEAL -> CARDIOVASCULAR (was exactly zero) ===")
fasted_arm = [sim(p, [], 300, seed=300 + k) for k, p in enumerate(pop)]
hr_rise = [meal[k][:, HR].max() - fasted_arm[k][:, HR].max() for k in range(N)]
results.append(band(hr_rise, 4.0, 10.0, "postprandial HR rise (Brunzell 1971)", " bpm"))
hrv_drop = [meal[k][:, HRV].min() - fasted_arm[k][:, HRV].min() for k in range(N)]
print(f"   HRV follows inversely (no term of its own): {np.mean(hrv_drop):+.2f} ms")
bp_delta = max(abs(meal[k][:, MI['sbp']].max() - fasted_arm[k][:, MI['sbp']].max()) for k in range(N))
print(f"   SBP deliberately untouched: max |delta| {bp_delta:.3f} mmHg")

print("\n=== D. EXERCISE LACTATE (was 9.8 at moderate) ===")
ACT_MOD, ACT_MAX = 0.65, 1.0
for label, a, lo, hi in (("moderate bout (Brooks 1986)", ACT_MOD, 2.0, 4.0),
                         ("maximal effort", ACT_MAX, 8.0, 13.0)):
    act = np.zeros(360); act[60:180] = a
    lac = [sim(p, [], 360, act=act, seed=400 + k)[:, LAC].max() for k, p in enumerate(pop)]
    results.append(band(lac, lo, hi, f"peak lactate, {label}", " mmol/L"))

print("\n=== E. GUARDRAILS (things that must NOT have regressed) ===")
day = [sim(p, [(60., 60., 15., 20.), (420., 80., 25., 30.), (780., 70., 30., 25.)], 1080,
           seed=500 + k) for k, p in enumerate(pop)]
results.append(band([t[0, G] for t in day], 85, 115, "fasting glucose at t0 (population mean)", " mg/dL"))
results.append(band([t[60:, F].mean() for t in day], 0.25, 0.60, "mean daytime FFA (fed regime)", " mmol/L"))
results.append(band([t[-1, LG] / t[0, LG] for t in day], 0.55, 1.05, "liver glycogen retained over a fed day", ""))
unstable = sum(1 for t in day if not np.isfinite(t).all() or t[:, G].max() > 400 or t[:, G].min() < 30)
print(f"  {'PASS' if unstable == 0 else 'FAIL'}  unstable/divergent trajectories: {unstable}/{N}")
results.append(unstable == 0)

print(f"\n{sum(results)}/{len(results)} checks pass  (N={N} randomized patients)")
