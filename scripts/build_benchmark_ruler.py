#!/usr/bin/env python3
"""Assemble the benchmark ruler from its labelled sources (iter 94).

Why this exists: the ruler iter-93 was graded against was 24 episodes exported from
`pulse.check_ins` in April, ALL defaulting to `source="real"` because that is the
default in `pulse/benchmark.py`, not because anything asserted it. Their scored
window contains no meals and their ground truth barely moves, so a constant beats
every gate threshold by 11-50x (scripts/iter94_ruler_audit.py). Meanwhile the 14
CGM + Oura overnight episodes built during iter 92 were never added to the gate at
all.

This script makes the composition explicit and reproducible:

  legacy_static  the original 24 check-in episodes, tagged honestly and kept so the
                 headline number stays comparable across iterations
  cgm_real       real CGM + Oura overnight windows (glucose + hr only — those are
                 the markers we actually have)

The teacher-side episodes (`teacher`, `teacher_dynamic`) are generated in-process by
pulse/knowledge/benchmark_extras.py and merged at benchmark time, so they are NOT
written here.

Usage:
    uv run python scripts/build_benchmark_ruler.py \\
        --legacy pulse/benchmark.dataset.generated.json \\
        --cgm-source-dir ~/Documents/health \\
        --output /tmp/benchmark.dataset.iter94.json
    # then audit it, then upload:
    uv run python scripts/iter94_ruler_audit.py /tmp/benchmark.dataset.iter94.json
    gsutil cp /tmp/benchmark.dataset.iter94.json \\
        gs://grovina-pulse-data/benchmarks/benchmark.dataset.generated.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

LEGACY_SOURCE = "legacy_static"
CGM_SOURCE = "cgm_real"


def build_cgm_episodes(source_dir: Path, max_episodes: int) -> list[dict]:
    """Run the committed ingest and return its episodes, tagged `cgm_real`."""
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "cgm.json"
        cmd = [
            sys.executable, str(_ROOT / "scripts" / "ingest_real_data.py"),
            "--source-dir", str(source_dir), "--output", str(out),
            "--max-episodes", str(max_episodes),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_ROOT))
        if proc.returncode != 0:
            raise SystemExit(f"ingest_real_data.py failed:\n{proc.stdout}\n{proc.stderr}")
        print(proc.stdout.strip())
        eps = json.loads(out.read_text())["episodes"]
    for e in eps:
        e["source"] = CGM_SOURCE
    return eps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy", type=Path,
                    default=_ROOT / "pulse" / "benchmark.dataset.generated.json")
    ap.add_argument("--cgm-source-dir", type=Path, required=True,
                    help="Directory holding the CGM CSV + google_fit.zip.")
    ap.add_argument("--max-cgm-episodes", type=int, default=20)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    legacy_payload = json.loads(args.legacy.read_text())
    legacy = legacy_payload["episodes"]
    for e in legacy:
        # Per-episode source wins over the payload default in load_benchmark_dataset,
        # so tag explicitly rather than relying on meta.
        e["source"] = LEGACY_SOURCE
    print(f"legacy_static: {len(legacy)} episodes from {args.legacy}")

    cgm = build_cgm_episodes(args.cgm_source_dir.expanduser(), args.max_cgm_episodes)
    print(f"cgm_real: {len(cgm)} episodes")

    ids = [e["user_id"] for e in legacy + cgm]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate user_id across sources — episodes would collide")

    payload = {
        "meta": {
            "builtBy": "scripts/build_benchmark_ruler.py",
            "iter": 94,
            "benchmark_source": LEGACY_SOURCE,  # default for anything untagged
            "composition": {LEGACY_SOURCE: len(legacy), CGM_SOURCE: len(cgm)},
            "legacy_meta": legacy_payload.get("meta", {}),
            "note": (
                "teacher / teacher_dynamic episodes are generated in-process by "
                "pulse/knowledge/benchmark_extras.py and merged at benchmark time."
            ),
        },
        "episodes": legacy + cgm,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {args.output} — {len(payload['episodes'])} episodes "
          f"({args.output.stat().st_size:,} bytes)")
    print("Next: audit it with scripts/iter94_ruler_audit.py before uploading.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
