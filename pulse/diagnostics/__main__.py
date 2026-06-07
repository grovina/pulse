"""CLI for pulse diagnostics.

Four subcommands:

- ``probe`` runs the physiological probe against a single checkpoint.
- ``compare`` diffs two runs (checkpoints + benchmark reports).
- ``signal-balance`` measures per-signal raw loss and per-module
  grad norms (gut, metabolic, all) on a checkpoint — the diagnostic
  that exposed the iter-13 gut-kernel collapse and informed the
  iter-14 weight retune, extended in iter 20 to the metabolic module.
- ``cohort-ablation`` un-averages the cohort statistic signal —
  per-spec z-residual + per-module grad norm. Catches dead literature
  specs and reveals which physiology pillar each spec actually pulls
  on.

All subcommands write structured JSON to ``--out`` if requested and
always print a human-readable rendering to stdout. To compare two runs,
download their artifacts from GCS (``gcloud storage cp``) and invoke
``compare`` against the resulting local files.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from .cohort_ablation import (
    cohort_ablation_for_checkpoint,
    render as render_cohort_ablation,
)
from .cold_calibration import (
    cold_calibration,
    render as render_cold_calibration,
)
from .compare import compare_reports, compare_runs, render
from .physiology_rules import (
    physiology_rules_for_checkpoint,
    render as render_physiology_rules,
)
from .probe import load_model_from_checkpoint, probe_checkpoint
from .signal_balance import (
    render as render_signal_balance,
    signal_balance_for_checkpoint,
)


def _localize(path: str | None, tmpdir: Path | None) -> str | None:
    """Return a local filesystem path. Downloads gs:// URIs lazily into
    ``tmpdir`` so the rest of the CLI can stay path-agnostic."""
    if path is None:
        return None
    if not path.startswith("gs://"):
        return path
    assert tmpdir is not None
    from google.cloud import storage as gcs

    rest = path[5:]
    bucket_name, _, blob_name = rest.partition("/")
    if not blob_name:
        raise ValueError(f"Invalid gs:// URI: {path}")
    local = tmpdir / blob_name.replace("/", "_")
    local.parent.mkdir(parents=True, exist_ok=True)
    gcs.Client().bucket(bucket_name).blob(blob_name).download_to_filename(str(local))
    return str(local)


def _upload_if_gs(local_path: str | None, dest: str | None) -> None:
    if not dest or not dest.startswith("gs://") or local_path is None:
        return
    from google.cloud import storage as gcs

    rest = dest[5:]
    bucket_name, _, blob_name = rest.partition("/")
    gcs.Client().bucket(bucket_name).blob(blob_name).upload_from_filename(local_path)


def _read_json(path: str | Path | None) -> dict | None:
    if path is None:
        return None
    return json.loads(Path(path).read_text())


def _cmd_probe(args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        ckpt = _localize(args.checkpoint, tmp)
        model, _ = load_model_from_checkpoint(ckpt)
        report = probe_checkpoint(model)
        payload = report.to_dict()
        if args.out:
            local_out = tmp / "probe.json" if args.out.startswith("gs://") else Path(args.out)
            local_out.write_text(json.dumps(payload, indent=2))
            _upload_if_gs(str(local_out), args.out)
        print(json.dumps(payload, indent=2))
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        new_ckpt = _localize(args.new_checkpoint, tmp)
        baseline_ckpt = _localize(args.baseline_checkpoint, tmp)
        new_rpt = _localize(args.new_report, tmp)
        baseline_rpt = _localize(args.baseline_report, tmp)

        cmp = compare_runs(
            new_checkpoint=new_ckpt,
            baseline_checkpoint=baseline_ckpt,
            new_report=_read_json(new_rpt),
            baseline_report=_read_json(baseline_rpt),
            new_label=args.new_label,
            baseline_label=args.baseline_label,
        )
        print(render(cmp))
        if args.out:
            local_out = tmp / "delta.json" if args.out.startswith("gs://") else Path(args.out)
            local_out.write_text(json.dumps(cmp.to_dict(), indent=2))
            _upload_if_gs(str(local_out), args.out)
    return 0


def _cmd_cold_calibration(args: argparse.Namespace) -> int:
    report = cold_calibration(seed=args.seed, misleading_z=args.misleading_z)
    print(render_cold_calibration(report))
    if args.out:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            local_out = (
                tmp / "cold-calibration.json"
                if args.out.startswith("gs://") else Path(args.out)
            )
            local_out.write_text(json.dumps(report.to_dict(), indent=2))
            _upload_if_gs(str(local_out), args.out)
    return 1 if any(r.misleading for r in report.rows) else 0


def _cmd_cohort_ablation(args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        ckpt = _localize(args.checkpoint, tmp)
        report = cohort_ablation_for_checkpoint(
            ckpt,
            n_patients=args.n_patients,
            sample_patients=args.sample_patients,
            seed=args.seed,
        )
        print(render_cohort_ablation(report))
        if args.out:
            local_out = (
                tmp / "cohort-ablation.json"
                if args.out.startswith("gs://") else Path(args.out)
            )
            local_out.write_text(json.dumps(report.to_dict(), indent=2))
            _upload_if_gs(str(local_out), args.out)
    return 0


def _cmd_physiology_rules(args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        ckpt = _localize(args.checkpoint, tmp)
        report = physiology_rules_for_checkpoint(
            ckpt,
            n_patients=args.n_patients,
            sample_patients=args.sample_patients,
            seed=args.seed,
        )
        print(render_physiology_rules(report))
        if args.out:
            local_out = (
                tmp / "physiology-rules.json"
                if args.out.startswith("gs://") else Path(args.out)
            )
            local_out.write_text(json.dumps(report.to_dict(), indent=2))
            _upload_if_gs(str(local_out), args.out)
    return 0


def _cmd_signal_balance(args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        ckpt = _localize(args.checkpoint, tmp)
        report = signal_balance_for_checkpoint(ckpt, n_patients=args.n_patients)
        payload = report.to_dict()
        print(render_signal_balance(report))
        if args.out:
            local_out = (
                tmp / "signal-balance.json"
                if args.out.startswith("gs://") else Path(args.out)
            )
            local_out.write_text(json.dumps(payload, indent=2))
            _upload_if_gs(str(local_out), args.out)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pulse.diagnostics")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe", help="Run physiological probe on one checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default=None, help="Optional path to write JSON")
    p.set_defaults(func=_cmd_probe)

    c = sub.add_parser("compare", help="Diff two runs (checkpoints + benchmark reports)")
    c.add_argument("--new-checkpoint", default=None)
    c.add_argument("--baseline-checkpoint", default=None)
    c.add_argument("--new-report", default=None)
    c.add_argument("--baseline-report", default=None)
    c.add_argument("--new-label", default="new")
    c.add_argument("--baseline-label", default="baseline")
    c.add_argument("--out", default=None, help="Optional path to write JSON delta")
    c.set_defaults(func=_cmd_compare)

    s = sub.add_parser(
        "signal-balance",
        help="Per-signal raw loss + grad norm at one checkpoint",
    )
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--n-patients", type=int, default=20)
    s.add_argument("--out", default=None, help="Optional path to write JSON")
    s.set_defaults(func=_cmd_signal_balance)

    cc = sub.add_parser(
        "cold-calibration",
        help="Verify each cohort spec is reachable from the cold model",
    )
    cc.add_argument("--seed", type=int, default=0)
    cc.add_argument("--misleading-z", type=float, default=2.0,
                    help="Flag a spec as misleading if |z| > this on cold model")
    cc.add_argument("--out", default=None, help="Optional path to write JSON")
    cc.set_defaults(func=_cmd_cold_calibration)

    a = sub.add_parser(
        "cohort-ablation",
        help="Per-spec z-residual + per-module grad norm at one checkpoint",
    )
    a.add_argument("--checkpoint", required=True)
    a.add_argument("--n-patients", type=int, default=20)
    a.add_argument("--sample-patients", type=int, default=4)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--out", default=None, help="Optional path to write JSON")
    a.set_defaults(func=_cmd_cohort_ablation)

    pr = sub.add_parser(
        "physiology-rules",
        help="Per-rule violation + per-module grad norm at one checkpoint",
    )
    pr.add_argument("--checkpoint", required=True)
    pr.add_argument("--n-patients", type=int, default=20)
    pr.add_argument("--sample-patients", type=int, default=4)
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--out", default=None, help="Optional path to write JSON")
    pr.set_defaults(func=_cmd_physiology_rules)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
