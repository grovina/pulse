#!/usr/bin/env python3
"""Audit the TEACHER against every physiology rule the student is trained on.

Iter 97 (review 4.5). The cold-model teacher and the physiology rules supervise
the same student, so a rule the teacher violates pulls the student two ways at
once — 22 of 61 rules did at HEAD 4c7a0a6, several of them entirely because of
the FRAME the rule was measured in (an inverted sleep mask, a "24 h fasted" arm
that started fed). This script evaluates every rule on the teacher in EXACTLY
the frame training uses (``pulse.training.arm_init``: declared pre-fast, arm
sleep series or awake, arm activity or rest), and:

* prints every (rule, arm) with the teacher's violation in native units and the
  loss ``(violation / scale)^2`` it would contribute;
* exits non-zero if a rule the teacher violates is not flagged
  ``teacher_correction=True`` in the registry. A deliberate correction (the rule
  encodes literature the teacher gets wrong) is allowed; an accidental
  contradiction is not, because it is either a wrong rule or a teacher bug and
  the registry must say which.

The teacher is being changed concurrently (review sections 2 and 3). RE-RUN
THIS after every teacher change; the flags below describe the teacher at the
commit that set them, and a fixed teacher should let a flag be removed.

Usage:
  uv run python scripts/rules_teacher_audit.py [--n-patients N] [--tolerance L]
                                               [--sleep-inverted] [--no-fail]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import pulse  # noqa: E402

assert pulse.__file__.startswith(str(_ROOT)), pulse.__file__

import pulse.knowledge.full_body as fb  # noqa: E402
from pulse.knowledge.physiology_rules import (  # noqa: E402
    _SLEEP_WAKE_24H_ARM,
    PHYSIOLOGY_RULES,
    PhysiologyRule,
)
from pulse.physiology_rules_loss import rule_context_for_arm  # noqa: E402
from pulse.training.arm_init import teacher_arm_trajectory  # noqa: E402

_HELPERS = [
    "hinge_min_drop", "hinge_min_rise", "hinge_max_correlation", "hinge_min_correlation",
    "hinge_argmax_in_band", "hinge_argmin_in_band", "hinge_a_precedes_b", "hinge_max_value",
    "hinge_min_value", "hinge_min_ratio", "hinge_monotone_decrease",
    "hinge_monotone_increase", "hinge_circadian_amplitude", "hinge_max_drift",
]


def _kind(rule: PhysiologyRule) -> str:
    names = getattr(rule.predicate, "__code__", None)
    if names is None:
        return "?"
    for h in _HELPERS:
        if h in names.co_names:
            return h
    return "custom"


def audit(
    rules: list[PhysiologyRule],
    *,
    n_patients: int = 0,
    tolerance: float = 0.01,
    sleep_inverted: bool = False,
    seed: int = 17,
) -> tuple[list[dict], list[str]]:
    """Evaluate every rule on the teacher. Returns (rows, unflagged_violations)."""
    arm_cache: dict[tuple, np.ndarray] = {}

    def arm_for(arm):
        if sleep_inverted and arm.label == _SLEEP_WAKE_24H_ARM.label and arm.sleep_wake is not None:
            return replace(arm, sleep_wake=tuple(1.0 - float(v) for v in arm.sleep_wake))
        return arm

    def teacher(arm, params, key):
        k = (arm.label, arm.prefast_hours, key)
        if k not in arm_cache:
            arm_cache[k] = teacher_arm_trajectory(arm_for(arm), params, noise_scale=0.0)
        return arm_cache[k]

    rng = np.random.default_rng(seed)
    patients = [("default", fb.PatientParams())]
    for k in range(n_patients):
        patients.append((f"p{k}", fb.randomize_params(rng)))

    rows: list[dict] = []
    unflagged: list[str] = []
    for rule in rules:
        per_arm: list[dict] = []
        for arm in rule.arms:
            ctx = rule_context_for_arm(arm_for(arm))
            vals = []
            for key, params in patients:
                traj = torch.tensor(teacher(arm, params, key), dtype=torch.float32)
                vals.append(float(rule.predicate(traj, ctx)))
            per_arm.append({
                "arm": arm.label, "violation_default": vals[0],
                "violation_mean": float(np.mean(vals)),
                "frac_violating": float(np.mean([v > 0 for v in vals])),
            })
        loss_default = float(np.mean([(a["violation_default"] / rule.scale) ** 2 for a in per_arm]))
        loss_mean = float(np.mean([(a["violation_mean"] / rule.scale) ** 2 for a in per_arm]))
        violated = loss_default > tolerance
        rows.append({
            "rule": rule.name, "kind": _kind(rule), "scale": rule.scale,
            "loss_default": loss_default, "loss_mean": loss_mean,
            "violated": violated, "flagged": rule.teacher_correction,
            "note": rule.teacher_correction_note, "arms": per_arm,
        })
        if violated and not rule.teacher_correction:
            unflagged.append(rule.name)
    return rows, unflagged


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-patients", type=int, default=0, help="randomized patients on top of the default one")
    ap.add_argument("--tolerance", type=float, default=0.01,
                    help="loss (violation/scale)^2 on the default patient above which a rule counts as violated")
    ap.add_argument("--sleep-inverted", action="store_true",
                    help="measure with the pre-iter-97 inverted sleep mask (review 1.2 before/after)")
    ap.add_argument("--no-fail", action="store_true", help="always exit 0 (report only)")
    ap.add_argument("--rules", type=str, default="", help="comma-separated subset of rule names")
    args = ap.parse_args()

    rules = list(PHYSIOLOGY_RULES)
    if args.rules:
        want = {r.strip() for r in args.rules.split(",") if r.strip()}
        rules = [r for r in rules if r.name in want]
    rows, unflagged = audit(
        rules, n_patients=args.n_patients, tolerance=args.tolerance, sleep_inverted=args.sleep_inverted,
    )
    frame = "INVERTED sleep mask (pre-iter-97)" if args.sleep_inverted else "training frame"
    print(f"teacher vs {len(rows)} rules, {frame}, default patient + {args.n_patients} randomized")
    print(f"{'rule':44} {'helper':24} {'loss':>8} {'loss_N':>8}  per-arm violation (default patient)")
    rows_sorted = sorted(rows, key=lambda r: -r["loss_default"])
    for r in rows_sorted:
        arms = " ".join(f"{a['arm']}={a['violation_default']:.3g}" for a in r["arms"])
        flag = ""
        if r["violated"]:
            flag = "  [teacher_correction]" if r["flagged"] else "  <<< TEACHER VIOLATES, UNFLAGGED"
        print(f"{r['rule']:44} {r['kind']:24} {r['loss_default']:8.3f} {r['loss_mean']:8.3f}  {arms}{flag}")
    n_viol = sum(1 for r in rows if r["violated"])
    print(f"\n{n_viol}/{len(rows)} rules violated by the teacher (loss > {args.tolerance}); "
          f"{sum(1 for r in rows if r['violated'] and r['flagged'])} flagged as deliberate corrections; "
          f"{len(unflagged)} unflagged")
    if unflagged:
        print("UNFLAGGED: " + ", ".join(unflagged))
    if unflagged and not args.no_fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
