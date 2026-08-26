#!/usr/bin/env python3
"""Does the student's slow physiology hold still when nothing is asking it to move?

iter-95 measured this on its own artifact and found two states running away with
no supervision anywhere: over five identical eucaloric days liver_glycogen drained
100 -> 19 g while mitochondrial_capacity decayed 1.00 -> 0.46 (~14 %/day, toward a
mass-action equilibrium near 0.005). Neither is physiology. Both are the model
integrating an unopposed rate for longer than any training window ever looks.

iter-96 attacks them from three sides — the teacher's own fill taper (a eucaloric
day was glycogen-negative by 27 g), the student's leaking anabolic gate (63 % open
at zero appearance, so the pool REFILLED during a fast), and adding
mitochondrial_capacity to the distillation markers on the flat reference. This is
the probe that says whether it worked.

Also reports the structural check that should hold by construction, not by
training: the net liver-glycogen rate at the teacher's own 24 h-fast state must
be <= 0. It was +0.159 g/min on the iter-95 artifact.

Usage:
    python scripts/iter96_longclock_probe.py MODEL.pt [--days 5]
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
from pulse.model import ModularPhysiologyNetwork, integrate
from pulse.modules.gut import MealEvent
from pulse.types import MARKER_INDEX

assert fb.__file__.startswith(str(_ROOT)), f"shadowed pulse: {fb.__file__}"

# One eucaloric day, the same three meals the nutrition cohort arms use.
DAY_MEALS = ((120.0, 50.0, 8.0, 15.0), (360.0, 65.0, 12.0, 25.0), (720.0, 60.0, 15.0, 30.0))
WATCH = ("liver_glycogen", "mitochondrial_capacity", "glucose", "insulin", "cortisol", "hr")


def load_model(path: str) -> ModularPhysiologyNetwork:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    state = blob
    if isinstance(blob, dict):
        for key in ("model_state", "model_state_dict"):
            if key in blob:
                state = blob[key]
                break
    h = int(blob.get("hidden_dim", 48)) if isinstance(blob, dict) else 48
    model = ModularPhysiologyNetwork(
        metabolic_hidden=h, appetite_hidden=max(24, h // 2), stress_hidden=max(24, h // 2),
        cardiovascular_hidden=h, thermoreg_hidden=max(16, h // 3),
        respiratory_hidden=max(16, h // 3),
    )
    model.load_state_dict(state)
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--days", type=int, default=5)
    args = ap.parse_args()

    model = load_model(args.model)
    n = 1440 * args.days
    rng = np.random.default_rng(1)
    sw = fb.generate_sleep_wake(args.days, n, 6.0, rng)
    act = np.zeros(n, dtype=np.float32)
    meals = [MealEvent(time=m[0] + 1440 * d, carbs=m[1], fats=m[2], proteins=m[3])
             for d in range(args.days) for m in DAY_MEALS]

    params = fb.resolve_derived_params(fb.PatientParams())
    teacher, _ = fb.simulate_full_body(
        params, [(m.time, m.carbs, m.fats, m.proteins) for m in meals],
        sw, act, duration_min=n, start_hour=6.0, noise_scale=0.0)

    init = torch.tensor(teacher[0], dtype=torch.float32)
    with torch.no_grad():
        student = integrate(
            model=model, initial_state=init, embedding=torch.zeros(model.embedding_dim),
            n_steps=n, dt=1.0, start_time_minutes=360.0, meals=meals,
            sleep_wake=torch.tensor(sw), activity=torch.tensor(act),
        ).numpy()

    print(f"\n{args.days} EUCALORIC DAYS (3 meals/day, 175 g carbohydrate), zero embedding\n")
    for mk in WATCH:
        j = MARKER_INDEX[mk]
        s = "  ".join(f"{student[min(d * 1440, n - 1), j]:7.2f}" for d in range(args.days + 1))
        t = "  ".join(f"{teacher[min(d * 1440, n - 1), j]:7.2f}" for d in range(args.days + 1))
        print(f"  {mk:<24} student  {s}")
        print(f"  {'':<24} teacher  {t}")

    lg = MARKER_INDEX["liver_glycogen"]
    mc = MARKER_INDEX["mitochondrial_capacity"]
    lg_end = float(student[-1, lg])
    mc_end = float(student[-1, mc])
    print(f"\nACCEPTANCE 5 -- eucaloric stationarity")
    print(f"  liver_glycogen        day {args.days}: {lg_end:7.2f} g   "
          f"(iter-95 reached 19.0)   accept > 70 : {'PASS' if lg_end > 70 else 'FAIL'}")
    print(f"  mitochondrial_capacity day {args.days}: {mc_end:7.3f}     "
          f"(iter-95 reached 0.46)   accept > 0.9: {'PASS' if mc_end > 0.9 else 'FAIL'}")

    # --- ACCEPTANCE 4: the structural one ---
    # Put the student at the TEACHER's own 24 h-fast state and read its glycogen
    # rate. Nothing is being absorbed, so synthesis must be zero and the net rate
    # must be <= 0. On the iter-95 artifact this read +0.159 g/min.
    fast_traj, _ = fb.simulate_full_body(
        params, [], fb.generate_sleep_wake(1, 1440, 6.0, np.random.default_rng(2)),
        np.zeros(1440, dtype=np.float32), duration_min=1440, start_hour=6.0, noise_scale=0.0)
    fast_state = torch.tensor(fast_traj[-1], dtype=torch.float32)
    with torch.no_grad():
        two = integrate(
            model=model, initial_state=fast_state, embedding=torch.zeros(model.embedding_dim),
            n_steps=2, dt=1.0, start_time_minutes=360.0, meals=[],
        ).numpy()
    rate = float(two[1, lg] - two[0, lg])
    print(f"\nACCEPTANCE 4 -- the anabolic gate, at the teacher's 24 h-fast state")
    print(f"  teacher liver_glycogen there: {float(fast_traj[-1, lg]):.2f} g")
    print(f"  student net d(liver_glycogen)/dt: {rate:+.4f} g/min   "
          f"(iter-95: +0.1590)   accept <= 0: {'PASS' if rate <= 0 else 'FAIL'}")
    if rate > 0:
        print("  A positive rate here means the change did NOT take -- synthesis is")
        print("  supposed to be identically zero at zero appearance by construction,")
        print("  so this is not a 'training fell short' failure.")


if __name__ == "__main__":
    main()
