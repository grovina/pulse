#!/usr/bin/env python3
"""Iter 97 teacher validation: the measurements behind the one-carbon-budget rewrite.

Prints, for the DEFAULT patient unless stated:
  1. resting table  -- fed night / pre-breakfast / 24 h / 36 h fast vs declared typicals
  2. carbon closure -- every glucose and glycogen flux on a eucaloric day, and the residual
  3. fast across Gb -- 12/24/36/48 h of fasting for Gb 70/95/130 (the floor must be absolute)
  4. standard meal  -- 75 g: glucose/insulin/ghrelin/GLP-1/HR peaks, incretin share, dose slope
  5. HPA timing     -- cortisol nadir/peak hours, cort/ACTH asleep vs awake, dawn HR and glucose

Run the anchor audit (scripts/cohort_teacher_audit.py) and the textbook scenarios
(scripts/run_textbook_scenarios.py) alongside. Usage: python scripts/iter97_teacher_validate.py
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

import pulse.knowledge.full_body as fb
from pulse.knowledge.full_body import (
    PatientParams, resolve_derived_params, simulate_full_body, glucose_fluxes,
    MG_DL_PER_G, BODY_MASS_KG,
)
from pulse.types import MARKER_INDEX as MI, MARKERS

assert Path(fb.__file__).resolve().is_relative_to(_ROOT), fb.__file__

START = 6.0
STD_DAY = [(8.0, 50, 12, 20), (13.0, 65, 22, 28), (19.0, 75, 28, 35)]
TYP = {m.id: m.typical for m in MARKERS}


def clock_meals(day_meals, n_days):
    return sorted((d * 1440 + (h - START) * 60.0, c, f, p)
                  for d in range(n_days) for h, c, f, p in day_meals if h >= START)


def sleep_wake(n_days, bed=23.0, wake=7.0):
    n = n_days * 1440
    sw = np.ones(n, dtype=np.float32)
    for d in range(n_days + 1):
        s = int((bed - START) * 60) + d * 1440
        e = int((wake + 24 - START) * 60) + d * 1440
        sw[max(0, s):min(n, e)] = 0.0
    k = np.ones(20, dtype=np.float32) / 20
    return np.clip(np.convolve(sw, k, mode="same"), 0, 1).astype(np.float32)


def run(p, meals, n_days, sw=None, act=None, start_hour=START):
    n = n_days * 1440
    sw = sleep_wake(n_days) if sw is None else sw
    act = np.zeros(n, dtype=np.float32) if act is None else act
    return simulate_full_body(p, meals, sw, act, n, start_hour=start_hour,
                              noise_scale=0.0, rng=np.random.default_rng(0))


def cs(day, h0, h1):
    return slice(int(day * 1440 + (h0 - START) * 60), int(day * 1440 + (h1 - START) * 60))


def section_resting(p):
    print("\n=== 1. RESTING TABLE (fed day 3 night 02-05 / pre-breakfast 06:30-07:30; fast 24 h / 36 h after dinner) ===")
    trajB, _ = run(p, clock_meals(STD_DAY, 3), 3)
    trajC, _ = run(p, clock_meals(STD_DAY, 1), 3)
    decl = dict(glucose=p.Gb, insulin=p.Ib, glucagon=p.Gnb, ffa=p.FFA_b, bhb=p.BHB_b, hepatic_output=p.Hep_b,
                ghrelin=p.Ghr_b, leptin=p.Lep_b, glp1=p.GLP1_b, cortisol=p.Cort_b, acth=p.ACTH_b, hr=p.HR0,
                hrv=p.HRV0, sbp=p.SBP0, dbp=p.DBP0, liver_glycogen=p.LGly_b, muscle_glycogen=p.MGly_b,
                cck=p.CCK_b, gallbladder_bile=p.GB_b, intestinal_bile=p.INT_b, bile_acids=p.BA_b)
    night, pre = cs(2, 26, 29), cs(1, 30.5, 31.5)
    f24, f36 = cs(1, 18.5, 19.5), cs(2, 6.5, 7.5)
    print(f"{'marker':18s}{'declared':>9s}{'typical':>9s}{'night':>9s}{'pre':>9s}{'fast24':>9s}{'fast36':>9s}")
    for m, d in decl.items():
        i = MI[m]
        print(f"{m:18s}{d:9.3g}{TYP[m]:9.3g}{trajB[night, i].mean():9.2f}{trajB[pre, i].mean():9.2f}"
              f"{trajC[f24, i].mean():9.2f}{trajC[f36, i].mean():9.2f}")


def section_carbon(p):
    print("\n=== 2. CARBON CLOSURE, eucaloric day 2 of 3 (190 g carbohydrate) ===")
    traj, absp = run(p, clock_meals(STD_DAY, 3), 3)
    day = cs(1, 6, 30)
    G, I, Gn = traj[day, MI["glucose"]], traj[day, MI["insulin"]], traj[day, MI["glucagon"]]
    Cort, FFA = traj[day, MI["cortisol"]], traj[day, MI["ffa"]]
    LG, MG, Hep = traj[day, MI["liver_glycogen"]], traj[day, MI["muscle_glycogen"]], traj[day, MI["hepatic_output"]]
    Ra = absp[day, 0]
    X = np.zeros_like(I); x = 0.0
    for k in range(len(I)):
        X[k] = x
        x += -p.p2 * x + p.Si * p.p2 * (I[k] - p.Ib)
    keys = ("ra", "syn_L", "syn_M", "egp", "uptake_ii", "uptake_id", "uptake_ex", "glyco_flux", "gng_rel_flux", "gng_divert", "brk_M_g")
    acc = {k: 0.0 for k in keys}
    for k in range(len(G)):
        fl = glucose_fluxes(p, G[k], I[k], X[k], Gn[k], Cort[k], FFA[k], LG[k], MG[k], Hep[k], Ra[k], 0.0)
        for kk in keys:
            acc[kk] += fl[kk]
    g = lambda v: v / MG_DL_PER_G
    kg = lambda v: v * BODY_MASS_KG / 1000.0
    print(f"  Ra integral {g(acc['ra']):.1f} g | liver synth direct {g(acc['syn_L']):.1f} g, indirect {kg(acc['gng_divert']):.1f} g "
          f"| glycogenolysis {kg(acc['glyco_flux']):.1f} g | GNG released {kg(acc['gng_rel_flux']):.1f} g")
    print(f"  uptake obligatory {g(acc['uptake_ii']):.1f} g, insulin-dependent {g(acc['uptake_id']):.1f} g | muscle synth {g(acc['syn_M']):.1f} g")
    pools = (G[-1] - G[0]) / MG_DL_PER_G + (LG[-1] - LG[0]) + (MG[-1] - MG[0])
    booked = (g(acc["ra"]) + kg(acc["gng_rel_flux"]) + kg(acc["gng_divert"])
              - g(acc["uptake_ii"]) - g(acc["uptake_id"]) - g(acc["uptake_ex"]) - acc["brk_M_g"])
    print(f"  d(pools) {pools:+.2f} g vs booked {booked:+.2f} g -> residual {pools - booked:+.3f} g/day")
    print(f"  LGly mean {LG.mean():.1f} (min {LG.min():.1f} max {LG.max():.1f}); hepatic_output mean {Hep.mean():.2f} mg/kg/min")


def section_fast(base):
    print("\n=== 3. FAST ACROSS Gb (hours after the day-0 dinner) ===")
    for gb in (70.0, 95.0, 130.0):
        p = copy.deepcopy(base); p.Gb = gb; p = resolve_derived_params(p)
        traj, _ = run(p, clock_meals(STD_DAY, 1), 3)
        t0 = int((19 - START) * 60)
        print(f"  Gb={gb:5.1f} Hep_b={p.Hep_b:.2f}  " + "  ".join(
            f"{h}h: G {traj[t0 + h*60 - 30:t0 + h*60 + 30, MI['glucose']].mean():5.1f} I {traj[t0 + h*60, MI['insulin']]:4.1f} "
            f"BHB {traj[t0 + h*60, MI['bhb']]:.2f} LGly {traj[t0 + h*60, MI['liver_glycogen']]:5.1f} ghr {traj[t0 + h*60, MI['ghrelin']]:5.1f}"
            for h in (12, 24, 48)))


def section_meal(p):
    print("\n=== 4. STANDARD 75 g MEAL at rest (08:00 start, meal at +120) ===")
    n, mt = 420, 120
    def probe(pp, carbs=75.0):
        return simulate_full_body(pp, [(float(mt), carbs, 5.0, 10.0)], np.ones(n), np.zeros(n, dtype=np.float32),
                                  n, start_hour=8.0, noise_scale=0.0, rng=np.random.default_rng(0))[0]
    traj = probe(p)
    for m, mode in (("glucose", "max"), ("insulin", "max"), ("ghrelin", "min"), ("glp1", "max"), ("hr", "max"),
                    ("glucagon", "min"), ("liver_glycogen", "max"), ("hepatic_output", "min"), ("bile_acids", "max")):
        x = traj[:, MI[m]]; pre = x[mt - 1]; post = x[mt:]
        k = int(np.argmax(post)) if mode == "max" else int(np.argmin(post))
        print(f"  {m:15s} pre {pre:7.2f} -> {post[k]:7.2f} ({post[k]-pre:+6.2f}) at +{k} min")
    g = traj[:, MI["glucose"]]; glp = traj[:, MI["glp1"]]
    gsir = p.gamma * np.maximum(g - p.h, 0); ex = np.maximum(glp - p.GLP1_b, 0)
    inc = 1 + p.incretin_gain * ex / (ex + p.K_incretin)
    print(f"  glucose at +120 {g[mt+120]:.1f}, +180 {g[mt+180]:.1f}; insulin mean +90..+240 {traj[mt+90:mt+240, MI['insulin']].mean():.1f}")
    print(f"  incretin share of glucose-stimulated secretion {(gsir*(inc-1)).sum()/(gsir*inc).sum():.2f} (Nauck 0.5-0.7)")
    print("  dose slope: " + ", ".join(f"{c:.0f} g +{probe(p, c)[mt:, MI['glucose']].max() - probe(p, c)[mt-1, MI['glucose']]:.1f}"
                                       for c in (25, 50, 75, 100, 150)))


def section_hpa(p):
    print("\n=== 5. HPA / DAWN (day 2, asleep 23-07) ===")
    traj, _ = run(p, clock_meals(STD_DAY, 3), 3)
    sw = sleep_wake(3)
    d = cs(1, 6, 30)
    C, A, H = traj[d, MI["cortisol"]], traj[d, MI["acth"]], traj[d, MI["hr"]]
    hrs = (START + np.arange(1440) / 60) % 24
    ratio = C / A
    print(f"  cortisol nadir {C.min():.2f} at {hrs[np.argmin(C)]:.1f} h, peak {C.max():.2f} at {hrs[np.argmax(C)]:.1f} h, "
          f"ratio {C.max()/C.min():.2f}, 24 h mean {C.mean():.2f}")
    print(f"  cort/ACTH asleep {ratio[sw[d] < 0.5].mean():.3f} vs awake {ratio[sw[d] > 0.5].mean():.3f}")
    i3, i6 = cs(1, 27, 27.05).start, cs(1, 30, 30.05).start - 1
    print(f"  dawn 03:00->06:00 asleep: HR {traj[i3, MI['hr']]:.2f} -> {traj[i6, MI['hr']]:.2f} ({traj[i6, MI['hr']]-traj[i3, MI['hr']]:+.2f}); "
          f"glucose {traj[i3, MI['glucose']]:.1f} -> {traj[i6, MI['glucose']]:.1f}; cortisol {traj[i3, MI['cortisol']]:.2f} -> {traj[i6, MI['cortisol']]:.2f}")
    print(f"  24 h mean HR {H.mean():.1f}; awake {H[sw[d] > 0.5].mean():.1f}; asleep {H[sw[d] < 0.5].mean():.1f}")


if __name__ == "__main__":
    p = resolve_derived_params(PatientParams())
    print(f"derived: Hep_b {p.Hep_b:.3f} mg/kg/min, Sg {p.Sg:.4f}, k_ba_synth {p.k_ba_synth:.5f}, "
          f"k_gb_basal {p.k_gb_basal:.5f}, ba_spill_gain {p.ba_spill_gain:.1f}")
    section_resting(p)
    section_carbon(p)
    section_fast(p)
    section_meal(p)
    section_hpa(p)
