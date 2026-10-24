#!/usr/bin/env python3
"""Assemble the benchmark ruler from its labelled sources.

  legacy_static  the original 24 check-in episodes, tagged honestly and kept so the
                 headline number stays comparable across iterations
  cgm_real       one subject's CGM + Oura overnight windows (glucose + hr)
  cgmacros       CGMacros (PhysioNet 2025): 45 people, calibrate across breakfast,
                 predict the lunch response (scripts/ingest_cgmacros.py)

The measured sources are the individual-level rulers (``REAL_EVIDENCE``); each is
one real item among several. The teacher-side episodes (`teacher`,
`teacher_dynamic`) are generated in-process by pulse/knowledge/benchmark_extras.py
and merged at benchmark time, so they are NOT written here.

Usage:
    uv run python scripts/build_benchmark_ruler.py \\
        --legacy pulse/benchmark.dataset.generated.json \\
        --cgm-source-dir ~/Documents/health \\
        --cgmacros-zip CGMacros_dateshifted365.zip \\
        --output /tmp/benchmark.dataset.json
    # then audit it, then upload:
    uv run python scripts/iter94_ruler_audit.py /tmp/benchmark.dataset.json
    gsutil cp /tmp/benchmark.dataset.json \\
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
CGMACROS_SOURCE = "cgmacros"


def _run_ingest(script: str, args: list[str]) -> dict:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "episodes.json"
        cmd = [sys.executable, str(_ROOT / "scripts" / script), *args, "--output", str(out)]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_ROOT))
        if proc.returncode != 0:
            raise SystemExit(f"{script} failed:\n{proc.stdout}\n{proc.stderr}")
        print(proc.stdout.strip())
        return json.loads(out.read_text())


def build_source(source: str, script: str, args: list[str]) -> tuple[list[dict], dict]:
    """Run one committed ingest and return its episodes, tagged ``source``."""
    payload = _run_ingest(script, args)
    eps = payload["episodes"]
    for e in eps:
        e["source"] = source
    return eps, payload.get("meta", {})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy", type=Path,
                    default=_ROOT / "pulse" / "benchmark.dataset.generated.json")
    ap.add_argument("--cgm-source-dir", type=Path, default=None,
                    help="Directory holding the CGM CSV + google_fit.zip.")
    ap.add_argument("--max-cgm-episodes", type=int, default=20)
    ap.add_argument("--cgmacros-zip", type=Path, default=None,
                    help="CGMacros_dateshifted365.zip from PhysioNet.")
    ap.add_argument("--cgmacros-per-person", type=int, default=1)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.cgm_source_dir is None and args.cgmacros_zip is None:
        ap.error("at least one measured source (--cgm-source-dir or --cgmacros-zip) is required")

    legacy_payload = json.loads(args.legacy.read_text())
    legacy = legacy_payload["episodes"]
    for e in legacy:
        # Per-episode source wins over the payload default in load_benchmark_dataset,
        # so tag explicitly rather than relying on meta.
        e["source"] = LEGACY_SOURCE
    print(f"{LEGACY_SOURCE}: {len(legacy)} episodes from {args.legacy}")

    measured: dict[str, tuple[list[dict], dict]] = {}
    if args.cgm_source_dir is not None:
        measured[CGM_SOURCE] = build_source(CGM_SOURCE, "ingest_real_data.py", [
            "--source-dir", str(args.cgm_source_dir.expanduser()),
            "--max-episodes", str(args.max_cgm_episodes),
        ])
    if args.cgmacros_zip is not None:
        measured[CGMACROS_SOURCE] = build_source(CGMACROS_SOURCE, "ingest_cgmacros.py", [
            "--zip", str(args.cgmacros_zip.expanduser()),
            "--per-person", str(args.cgmacros_per_person),
        ])
    for source, (eps, meta) in measured.items():
        print(f"{source}: {len(eps)} episodes (activity_rest_level={meta.get('activity_rest_level')})")
        if meta.get("activity_rest_level") != 0.0:
            raise SystemExit(f"{source} does not declare a resting activity of 0")

    episodes = legacy + [e for eps, _ in measured.values() for e in eps]
    ids = [e["user_id"] for e in episodes]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate user_id across sources — episodes would collide")

    payload = {
        "meta": {
            "builtBy": "scripts/build_benchmark_ruler.py",
            "benchmark_source": LEGACY_SOURCE,  # default for anything untagged
            "composition": {
                LEGACY_SOURCE: len(legacy),
                **{source: len(eps) for source, (eps, _) in measured.items()},
            },
            "legacy_meta": legacy_payload.get("meta", {}),
            "source_meta": {source: meta for source, (_, meta) in measured.items()},
            # The measured ingests' resting activity, declared so the loader does not
            # apply the legacy 0.1 -> 0 remap to this file.
            "activity_rest_level": 0.0,
            "note": (
                "teacher / teacher_dynamic episodes are generated in-process by "
                "pulse/knowledge/benchmark_extras.py and merged at benchmark time."
            ),
        },
        "episodes": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {args.output} — {len(episodes)} episodes "
          f"({args.output.stat().st_size:,} bytes)")
    print("Next: audit it with scripts/iter94_ruler_audit.py before uploading.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
