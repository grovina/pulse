#!/usr/bin/env python3
"""Audit the benchmark ruler itself: can it tell a simulator from a constant?

Motivation (docs/iter94-proposal.md §0.6). Every iteration is steered by
`overall_weighted_mape` and the gate thresholds, but nothing had ever measured how
hard the ruler actually is. On the dataset iter-93 was graded against, carrying the
last calibration reading forward scores 0.0004-0.014 MAPE while the thresholds sit
at 0.02-0.20 — 11x to 50x looser. A constant passes the gate with an order of
magnitude to spare, so gate movements cannot be read as physiology.

This reports, per dataset: episode shape, whether any meal lands in the scored eval
window, the ground truth's own within-episode variability, and the persistence
baseline vs the shipped thresholds. Run it on any candidate ruler before adopting it.

Usage:
    uv run python scripts/iter94_ruler_audit.py DATASET.json [DATASET2.json ...]
    # defaults to the committed gate dataset when no argument is given
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# See the note in iter94_student_fast_probe.py — insert the repo root explicitly so a
# stale shadowing copy of `pulse` on sys.path[0] cannot be imported instead.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

GATE_MARKERS = ("glucose", "hr", "sbp", "dbp", "temp")
THRESHOLDS = json.loads((_ROOT / "pulse" / "benchmark.thresholds.json").read_text())
MAPE_MAX = THRESHOLDS["marker_mape_max"]


def audit(path: Path) -> None:
    payload = json.loads(path.read_text())
    eps = payload.get("episodes", [])
    print(f"\n{'=' * 78}\n{path.name}  —  {len(eps)} episodes")
    meta = payload.get("meta", {})
    print(f"  meta.source={meta.get('source')!r}  "
          f"benchmark_source={meta.get('benchmark_source')!r}  "
          f"generated={meta.get('generatedAt') or meta.get('generated_at')!r}")
    if not eps:
        print("  (no episodes)")
        return

    # --- shape -------------------------------------------------------------
    durs = sorted({e["duration_min"] for e in eps})
    starts = sorted({e.get("start_time_minutes") for e in eps})
    windows = sorted({(min(m["time"] for m in e["eval_measurements"]),
                       max(m["time"] for m in e["eval_measurements"])) for e in eps})
    starts_s = starts if len(starts) < 6 else f"{len(starts)} distinct"
    windows_s = windows if len(windows) < 4 else f"{len(windows)} distinct"
    print(f"  duration_min={durs}  start_time_minutes={starts_s}")
    print(f"  eval windows: {windows_s}")

    # A meal inside the scored window is what makes postprandial work measurable.
    in_win = 0
    for e in eps:
        lo = min(m["time"] for m in e["eval_measurements"])
        hi = max(m["time"] for m in e["eval_measurements"])
        in_win += sum(1 for m in e.get("meals", []) if lo <= m["time"] <= hi)
    n_meals = sum(len(e.get("meals", [])) for e in eps)
    print(f"  meals: {n_meals} total, {in_win} inside the scored eval window"
          f"{'   <-- postprandial dynamics are UNSCORED' if in_win == 0 and n_meals else ''}")

    # --- truth variability + persistence baseline --------------------------
    # Persistence = carry the last calibration observation forward, the same baseline
    # pulse/benchmark.py scores `skill_vs_persistence` against.
    within: dict[str, list[float]] = {}
    rng_: dict[str, list[float]] = {}
    uniq: dict[str, set[float]] = {}
    perr: dict[str, list[float]] = {}
    for e in eps:
        last: dict[str, float] = {}
        for c in sorted(e.get("calibration_check_ins", []), key=lambda x: x["time"]):
            for k, v in (c.get("measurements") or {}).items():
                if v is not None:
                    last[k] = float(v)
        for mk in GATE_MARKERS:
            vals = [m["value"] for m in e["eval_measurements"] if m["marker_id"] == mk]
            if not vals:
                continue
            within.setdefault(mk, []).append(float(np.std(vals)))
            rng_.setdefault(mk, []).append(float(np.ptp(vals)))
            uniq.setdefault(mk, set()).update(vals)
        for m in e["eval_measurements"]:
            base = last.get(m["marker_id"])
            if base is None or abs(m["value"]) < 1e-9:
                continue
            perr.setdefault(m["marker_id"], []).append(
                abs(base - m["value"]) / abs(m["value"]))

    print(f"\n  {'marker':<9}{'n':>5}{'uniq':>6}{'sd/ep':>9}{'range/ep':>10}"
          f"{'persist MAPE':>14}{'threshold':>11}{'thr/persist':>13}")
    for mk in GATE_MARKERS:
        if mk not in within:
            print(f"  {mk:<9}{'—':>5}  (absent from this ruler)")
            continue
        pe = perr.get(mk, [])
        p = float(np.mean(pe)) if pe else float("nan")
        thr = MAPE_MAX.get(mk, float("nan"))
        ratio = thr / p if p > 1e-12 else float("nan")
        flag = "  <-- a constant passes" if ratio > 5 else ""
        print(f"  {mk:<9}{len(pe):>5}{len(uniq[mk]):>6}{np.mean(within[mk]):>9.3f}"
              f"{np.mean(rng_[mk]):>10.3f}{p:>14.4f}{thr:>11.3f}{ratio:>12.1f}x{flag}")

    print("\n  skill_vs_persistence = 1 - mape/persist_mape, so a reported -7.2 means the"
          "\n  model is 8.2x WORSE than carrying the last reading forward.")


def main() -> int:
    paths = [Path(a) for a in sys.argv[1:]] or [
        _ROOT / "pulse" / "benchmark.dataset.generated.json"]
    for p in paths:
        if not p.exists():
            print(f"missing: {p}")
            continue
        audit(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
