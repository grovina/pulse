"""Iter-80 local pre-dispatch gate.

Cheap, local checks that the iter-80 structural mechanisms behave before a
~12 h / paid Cloud Run iter is spent on them. Two axes:

  1. POST-TRAIN (needs a checkpoint): the learned model's gut->glucose
     amplitude. Measures the carb->glucose-peak sensitivity (mg/dL per gram)
     by running meal episodes across doses at the population (zero) embedding.
     The iter-79 honest baseline realised ~0.16 mg/dL/g against a 0.7 target;
     a healthy model should land in the physiological band ~0.4-0.9.

  2. TEACHER (no checkpoint needed): the hepatic-output split + glycogen->
     ketosis coupling. Asserts (a) an acute OGTT is conservation-exact vs the
     pre-iter-80 teacher and (b) 24 h fasting BHB reaches the Cahill range.

Usage:
    python -m scripts.iter80_gate                 # teacher checks only
    python -m scripts.iter80_gate --checkpoint X  # + post-train amplitude

Exit code is non-zero if any enabled check fails, so this can gate a dispatch
script. Prints a one-line PASS/FAIL summary per check.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from pulse.model import integrate, precompute_gut_outputs
from pulse.modules.gut import MealEvent
from pulse.types import MARKER_INDEX, NORM_CENTER
from pulse.knowledge.full_body import PatientParams, simulate_full_body

# Physiological bands (the gate thresholds).
SENS_LO, SENS_HI = 0.40, 0.90        # mg/dL glucose peak per gram carb
ACUTE_TOL = 0.05                      # mg/dL — acute OGTT must match pre-iter-80 teacher
BHB_24H_LO, BHB_24H_HI = 0.8, 3.0     # mM — 24 h fasting ketones (Cahill 2006)

_GI = MARKER_INDEX["glucose"]
_II = MARKER_INDEX["insulin"]
_BI = MARKER_INDEX["bhb"]
_LGI = MARKER_INDEX["liver_glycogen"]


def _glucose_peak(model, carbs: float) -> float:
    emb = torch.zeros(model.embedding_dim)
    init = torch.tensor(NORM_CENTER, dtype=torch.float32)
    meals = [MealEvent(time=15.0, carbs=float(carbs), fats=20.0, proteins=25.0)] if carbs > 0 else []
    with torch.no_grad():
        gut = precompute_gut_outputs(model, emb, 180, dt=1.0, start_time_minutes=360.0, meals=meals)
        traj = integrate(model, init, emb, 180, dt=1.0, start_time_minutes=360.0, meals=meals, gut_outputs=gut)
    return float(traj[:, _GI].max())


def _load_tolerant(checkpoint: str):
    """Load a checkpoint into the current architecture, tolerating params the
    checkpoint predates (e.g. running this gate on a pre-iter-80 checkpoint to
    measure the baseline amplitude). New params keep their constructed init."""
    from pulse.model import ModularPhysiologyNetwork
    from pulse.types import EMBEDDING_DIM
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt)
    h = int(ckpt.get("hidden_dim", 48))
    model = ModularPhysiologyNetwork(
        embedding_dim=int(ckpt.get("embedding_dim", EMBEDDING_DIM)),
        metabolic_hidden=h, appetite_hidden=max(24, h // 2), stress_hidden=max(24, h // 2),
        cardiovascular_hidden=h, thermoreg_hidden=max(16, h // 3), respiratory_hidden=max(16, h // 3),
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        print(f"[warn] checkpoint has unexpected keys: {unexpected}")
    model.eval()
    return model


def check_amplitude(checkpoint: str) -> bool:
    model = _load_tolerant(checkpoint)
    p0, p120 = _glucose_peak(model, 0.0), _glucose_peak(model, 120.0)
    sens = (p120 - p0) / 120.0
    ok = SENS_LO <= sens <= SENS_HI
    print(f"[{'PASS' if ok else 'FAIL'}] gut->glucose sensitivity = {sens:.3f} mg/dL/g "
          f"(band {SENS_LO}-{SENS_HI}; 60 g rise = {_glucose_peak(model, 60.0) - p0:.1f} mg/dL)")
    return ok


def _run_teacher(duration_min, meals, params):
    sw = np.ones(duration_min)
    act = np.zeros(duration_min)
    traj, _ = simulate_full_body(params, meals, sw, act, duration_min, start_hour=6.0,
                                 noise_scale=0.0, rng=np.random.default_rng(0))
    return traj


def _pre_iter80() -> PatientParams:
    p = PatientParams()
    p.hep_glyco_frac = 0.0
    p.keto_glyc_gain = 0.0
    return p


def check_teacher() -> bool:
    meals = [(30.0, 75.0, 0.0, 0.0)]
    old = _run_teacher(180, meals, _pre_iter80())
    new = _run_teacher(180, meals, PatientParams())
    acute = max(float(np.abs(new[:, i] - old[:, i]).max()) for i in (_GI, _II, _BI))
    acute_ok = acute < ACUTE_TOL
    print(f"[{'PASS' if acute_ok else 'FAIL'}] acute OGTT conservation: max|Δ| = {acute:.4f} mg/dL "
          f"(< {ACUTE_TOL}; split is the identity acutely)")

    fast = _run_teacher(24 * 60, [], PatientParams())
    bhb = float(fast[-1, _BI])
    lgly_drop = float(fast[0, _LGI] - fast[-1, _LGI])
    bhb_ok = BHB_24H_LO <= bhb <= BHB_24H_HI
    print(f"[{'PASS' if bhb_ok else 'FAIL'}] 24 h fasting BHB = {bhb:.2f} mM "
          f"(band {BHB_24H_LO}-{BHB_24H_HI}; liver depleted {lgly_drop:.0f} g)")
    return acute_ok and bhb_ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", help="checkpoint to measure gut->glucose amplitude on")
    args = ap.parse_args()

    print("=== iter-80 local gate ===")
    ok = check_teacher()
    if args.checkpoint:
        ok = check_amplitude(args.checkpoint) and ok
    else:
        print("[skip] gut->glucose amplitude (no --checkpoint; teacher checks only)")
    print(f"=== {'ALL PASS' if ok else 'FAILURES PRESENT'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
