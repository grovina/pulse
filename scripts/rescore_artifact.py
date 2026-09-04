#!/usr/bin/env python3
"""Re-score a trained artifact on the benchmark ruler, optionally FROZEN (iter 97).

Why this exists (review 2026-09-04, 5.2): a benchmark report never said what
ruler produced it -- no git SHA, no hash of the in-process teacher truth, no
dataset md5, no calibration settings -- and the `teacher*` persistence MAPE (a
property of the truth alone) changed between the iter-95 and iter-96 reports on
the same dataset file. The report now carries `ruler_fingerprint`, and the
in-process truth can be frozen to a file and reloaded so an old artifact is
re-scored on exactly the ruler a number was produced with.

This script is the local / non-Cloud-Run entry point around
`pulse.train._run_benchmark` (the same code path the trainer runs), for a
checkpoint on disk or in GCS. It touches none of the trainer's argparse.

Usage
-----
  # score an artifact on the LIVE ruler (current teacher), write the report
  uv run python scripts/rescore_artifact.py MODEL.pt --dataset DATASET.json \\
      --report report.json

  # freeze the current in-process truth to a file (and stop)
  uv run python scripts/rescore_artifact.py --freeze-ruler-to ruler.frozen.json

  # score on a frozen ruler
  uv run python scripts/rescore_artifact.py MODEL.pt --dataset DATASET.json \\
      --frozen-ruler ruler.frozen.json --report report.json

Calibration knobs go through the usual env vars (PULSE_BENCHMARK_PRIOR_WEIGHT,
PULSE_BENCHMARK_CALIBRATE_STEPS, ...); they are recorded in the fingerprint.
Runs ~17-30 min per episode single-threaded: use --parallel and a big box.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", help="checkpoint .pt (local path or gs://...)")
    ap.add_argument("--dataset", help="benchmark dataset JSON (local path or gs://...)")
    ap.add_argument("--thresholds", default=None,
                    help="thresholds JSON (gs://... only; local runs use pulse/benchmark.thresholds.json)")
    ap.add_argument("--report", default=None, help="where to write the report JSON (local or gs://)")
    ap.add_argument("--frozen-ruler", default=None,
                    help="score the in-process sources from this frozen file instead of the live teacher")
    ap.add_argument("--freeze-ruler-to", default=None,
                    help="write the CURRENT in-process truth to this file and exit (unless a model is given)")
    ap.add_argument("--parallel", type=int, default=None, help="benchmark worker processes")
    args = ap.parse_args()

    import pulse  # noqa: E402
    assert Path(pulse.__file__).resolve().is_relative_to(_ROOT), pulse.__file__

    if args.freeze_ruler_to:
        from pulse.benchmark import _git_sha
        from pulse.knowledge.benchmark_extras import export_frozen_ruler
        if os.environ.get("PULSE_BENCHMARK_FROZEN_RULER"):
            raise SystemExit("unset PULSE_BENCHMARK_FROZEN_RULER before freezing the LIVE ruler")
        meta = export_frozen_ruler(args.freeze_ruler_to, git_sha=_git_sha())
        print(f"frozen ruler written to {args.freeze_ruler_to}")
        print(json.dumps(meta, indent=2))
        if not args.model:
            return 0

    if not args.model or not args.dataset:
        ap.error("MODEL and --dataset are required to score")

    if args.frozen_ruler:
        os.environ["PULSE_BENCHMARK_FROZEN_RULER"] = str(Path(args.frozen_ruler).resolve())
    if args.parallel is not None:
        os.environ["PULSE_BENCHMARK_PARALLEL"] = str(args.parallel)
    os.environ["PULSE_BENCHMARK_DATASET_URI"] = args.dataset

    import torch
    from pulse.diagnostics.probe import load_model_from_checkpoint
    from pulse.train import _download_gs_uri_to_file, _run_benchmark

    model_path = args.model
    if model_path.startswith("gs://"):
        local = tempfile.mktemp(suffix=".pt")
        print(f"downloading {model_path} ...")
        _download_gs_uri_to_file(model_path, local)
        model_path = local

    model, blob = load_model_from_checkpoint(model_path)
    # Same rehydration the trainer's --benchmark-only path does: the diag-Gauss
    # prior needs the trained table's mean/std.
    pm, ps = blob.get("embedding_prior_mean"), blob.get("embedding_prior_std")
    if pm is not None and ps is not None:
        model._embedding_prior_mean = torch.tensor(pm, dtype=torch.float32)
        model._embedding_prior_std = torch.tensor(ps, dtype=torch.float32)
    else:
        print("checkpoint has no embedding prior: calibration falls back to iso-L2")
    for p in model.parameters():
        p.requires_grad_(False)

    return _run_benchmark(
        model, model_path, args.dataset, args.thresholds, args.report, gcs_bucket=None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
