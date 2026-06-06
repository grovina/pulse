"""Structured deltas between two pulse training runs.

A "run" is a (checkpoint, benchmark report, optional meta) triple. The
comparison covers:

- the gate (pass/fail transition + symmetric diff of failures)
- top-line scalars: overall_weighted_mape, verifier.overall_score,
  textbook_mean_pass_rate, per-marker mean MAPE
- per-textbook-scenario pass rates and per-check transitions
- the three probes (gut sweep, fasting drift, counter-regulation) side by side

All printers go through ``compare_runs`` which returns a structured
``ComparisonReport`` and prints a human-readable rendering. The printers
read from the returned report so the on-disk JSON and the stdout view stay
in sync.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .probe import (
    DEFAULT_CARB_DOSES_G,
    ProbeReport,
    cold_gut_sweep_targets,
    load_model_from_checkpoint,
    probe_checkpoint,
)


@dataclass(frozen=True)
class CheckTransition:
    scenario: str
    name: str
    description: str
    baseline_passed: bool | None
    new_passed: bool | None
    baseline_value: float | None
    new_value: float | None
    threshold: float | None

    @property
    def kind(self) -> str:
        if self.baseline_passed is None and self.new_passed is not None:
            return "added"
        if self.new_passed is None and self.baseline_passed is not None:
            return "removed"
        if self.baseline_passed == self.new_passed:
            return "stable"
        return "regression" if self.baseline_passed else "improvement"


@dataclass
class ComparisonReport:
    new_label: str
    baseline_label: str
    new_has_report: bool = False
    baseline_has_report: bool = False
    gate_baseline_passed: bool | None = None
    gate_new_passed: bool | None = None
    gate_failures_added: list[str] = field(default_factory=list)
    gate_failures_cleared: list[str] = field(default_factory=list)
    gate_failures_persisting: list[str] = field(default_factory=list)
    overall_mape_delta: float | None = None
    verifier_overall_delta: float | None = None
    verifier_category_deltas: dict[str, float] = field(default_factory=dict)
    textbook_pass_rate_delta: float | None = None
    per_marker_mape_deltas: dict[str, float] = field(default_factory=dict)
    scenario_pass_rate_deltas: dict[str, float] = field(default_factory=dict)
    check_transitions: list[CheckTransition] = field(default_factory=list)
    baseline_probe: ProbeReport | None = None
    new_probe: ProbeReport | None = None

    @property
    def has_benchmark_diff(self) -> bool:
        return self.new_has_report and self.baseline_has_report

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "new": self.new_label,
            "baseline": self.baseline_label,
            "new_has_report": self.new_has_report,
            "baseline_has_report": self.baseline_has_report,
            "gate": {
                "baseline_passed": self.gate_baseline_passed,
                "new_passed": self.gate_new_passed,
                "failures_added": self.gate_failures_added,
                "failures_cleared": self.gate_failures_cleared,
                "failures_persisting": self.gate_failures_persisting,
            },
            "scalars": {
                "overall_weighted_mape_delta": self.overall_mape_delta,
                "verifier_overall_score_delta": self.verifier_overall_delta,
                "verifier_category_deltas": self.verifier_category_deltas,
                "textbook_pass_rate_delta": self.textbook_pass_rate_delta,
            },
            "per_marker_mape_deltas": self.per_marker_mape_deltas,
            "scenario_pass_rate_deltas": self.scenario_pass_rate_deltas,
            "check_transitions": [asdict(c) for c in self.check_transitions],
        }
        if self.baseline_probe is not None:
            d["baseline_probe"] = self.baseline_probe.to_dict()
        if self.new_probe is not None:
            d["new_probe"] = self.new_probe.to_dict()
        return d


def _delta(new: float | None, base: float | None) -> float | None:
    if new is None or base is None:
        return None
    return float(new) - float(base)


def _scenario_check_index(report: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for scenario in report.get("textbook_scenarios", []):
        sname = scenario.get("name", "")
        for check in scenario.get("checks", []):
            out[(sname, check.get("name", ""))] = check
    return out


def compare_reports(
    new_report: dict[str, Any],
    baseline_report: dict[str, Any],
    *,
    new_label: str,
    baseline_label: str,
    new_probe: ProbeReport | None = None,
    baseline_probe: ProbeReport | None = None,
) -> ComparisonReport:
    cmp = ComparisonReport(
        new_label=new_label,
        baseline_label=baseline_label,
        new_has_report=bool(new_report),
        baseline_has_report=bool(baseline_report),
        gate_baseline_passed=baseline_report.get("gate", {}).get("passed"),
        gate_new_passed=new_report.get("gate", {}).get("passed"),
        new_probe=new_probe,
        baseline_probe=baseline_probe,
    )

    if cmp.has_benchmark_diff:
        base_failures = set(baseline_report.get("gate", {}).get("failures", []) or [])
        new_failures = set(new_report.get("gate", {}).get("failures", []) or [])
        cmp.gate_failures_added = sorted(new_failures - base_failures)
        cmp.gate_failures_cleared = sorted(base_failures - new_failures)
        cmp.gate_failures_persisting = sorted(base_failures & new_failures)

    cmp.overall_mape_delta = _delta(
        new_report.get("overall_weighted_mape"),
        baseline_report.get("overall_weighted_mape"),
    )
    cmp.verifier_overall_delta = _delta(
        new_report.get("verifier", {}).get("overall_score"),
        baseline_report.get("verifier", {}).get("overall_score"),
    )
    base_cats = baseline_report.get("verifier", {}).get("category_mean", {}) or {}
    new_cats = new_report.get("verifier", {}).get("category_mean", {}) or {}
    for cat in sorted(set(base_cats) | set(new_cats)):
        d = _delta(new_cats.get(cat), base_cats.get(cat))
        if d is not None:
            cmp.verifier_category_deltas[cat] = d

    cmp.textbook_pass_rate_delta = _delta(
        new_report.get("textbook_mean_pass_rate"),
        baseline_report.get("textbook_mean_pass_rate"),
    )

    base_pm = baseline_report.get("per_marker", {}) or {}
    new_pm = new_report.get("per_marker", {}) or {}
    for marker in sorted(set(base_pm) | set(new_pm)):
        d = _delta(
            (new_pm.get(marker) or {}).get("mean_mape"),
            (base_pm.get(marker) or {}).get("mean_mape"),
        )
        if d is not None:
            cmp.per_marker_mape_deltas[marker] = d

    base_scen = {s.get("name", ""): s.get("pass_rate") for s in baseline_report.get("textbook_scenarios", [])}
    new_scen = {s.get("name", ""): s.get("pass_rate") for s in new_report.get("textbook_scenarios", [])}
    for name in sorted(set(base_scen) | set(new_scen)):
        d = _delta(new_scen.get(name), base_scen.get(name))
        if d is not None:
            cmp.scenario_pass_rate_deltas[name] = d

    base_checks = _scenario_check_index(baseline_report)
    new_checks = _scenario_check_index(new_report)
    for key in sorted(set(base_checks) | set(new_checks)):
        scenario, name = key
        b = base_checks.get(key)
        n = new_checks.get(key)
        cmp.check_transitions.append(CheckTransition(
            scenario=scenario,
            name=name,
            description=(n or b or {}).get("description", ""),
            baseline_passed=b.get("passed") if b else None,
            new_passed=n.get("passed") if n else None,
            baseline_value=b.get("value") if b else None,
            new_value=n.get("value") if n else None,
            threshold=(n or b or {}).get("threshold"),
        ))
    return cmp


def _try_probe(checkpoint: str | Path, side: str) -> dict[str, Any] | None:
    """Load a checkpoint and run the probe; gracefully degrade to ``None``
    if the checkpoint can't be loaded against the current model definition.

    Architecture changes between iterations (e.g. iter 16's gut-kernel
    factorization changed `gut.kernel.kernel.0.weight` shape from
    `[32, 12]` to `[32, 9]`) make `state_dict` loading raise
    ``RuntimeError``. We don't want that to wipe out the entire
    cross-run comparison — the scalar/gate sections from the JSON
    reports remain useful even when probes can't render.

    Other failure modes (missing file, corrupted checkpoint, etc.) are
    treated the same way: log to stderr, return ``None``, let the
    comparator render whatever sections it can.
    """
    import sys

    try:
        m, _ = load_model_from_checkpoint(checkpoint)
        return probe_checkpoint(m)
    except (RuntimeError, FileNotFoundError, KeyError) as exc:
        print(
            f"[compare] {side} probe skipped: cannot load {checkpoint}: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return None


def compare_runs(
    new_checkpoint: str | Path | None,
    baseline_checkpoint: str | Path | None,
    new_report: dict[str, Any] | None,
    baseline_report: dict[str, Any] | None,
    *,
    new_label: str,
    baseline_label: str,
) -> ComparisonReport:
    """Compare two runs end-to-end.

    Either side may pass ``None`` for the checkpoint (skip probes for that
    side) or for the report (skip benchmark scalars). At least one of each
    pair must be present for that section to render.

    If a checkpoint is provided but cannot be loaded against the current
    model definition (typical after an architecture change), that side's
    probe is skipped with a warning rather than aborting the whole
    comparison. The benchmark/gate scalars from the JSON reports still
    render — they don't depend on being able to instantiate the model.
    """
    new_probe = _try_probe(new_checkpoint, "new") if new_checkpoint else None
    baseline_probe = (
        _try_probe(baseline_checkpoint, "baseline")
        if baseline_checkpoint else None
    )

    return compare_reports(
        new_report=new_report or {},
        baseline_report=baseline_report or {},
        new_label=new_label,
        baseline_label=baseline_label,
        new_probe=new_probe,
        baseline_probe=baseline_probe,
    )


def _fmt_signed(x: float | None, digits: int = 4) -> str:
    if x is None:
        return "n/a"
    return f"{x:+.{digits}f}"


def _bool(v: bool | None) -> str:
    if v is None:
        return "n/a"
    return "PASS" if v else "FAIL"


def render(cmp: ComparisonReport) -> str:
    lines: list[str] = []
    lines.append(f"\n=== Comparison: {cmp.new_label} vs {cmp.baseline_label} ===\n")

    lines.append("--- Gate ---")
    if not cmp.has_benchmark_diff:
        missing = []
        if not cmp.new_has_report:
            missing.append(cmp.new_label)
        if not cmp.baseline_has_report:
            missing.append(cmp.baseline_label)
        lines.append(f"  no benchmark report for: {', '.join(missing)} (gate diff skipped)")
    else:
        lines.append(f"  baseline={_bool(cmp.gate_baseline_passed)}  new={_bool(cmp.gate_new_passed)}")
        if cmp.gate_failures_added:
            lines.append(f"  NEW failures ({len(cmp.gate_failures_added)}):")
            for f in cmp.gate_failures_added:
                lines.append(f"    + {f}")
        if cmp.gate_failures_cleared:
            lines.append(f"  CLEARED failures ({len(cmp.gate_failures_cleared)}):")
            for f in cmp.gate_failures_cleared:
                lines.append(f"    - {f}")
        if cmp.gate_failures_persisting:
            lines.append(f"  Persisting failures ({len(cmp.gate_failures_persisting)}):")
            for f in cmp.gate_failures_persisting:
                lines.append(f"    = {f}")

    if cmp.has_benchmark_diff:
        lines.append("\n--- Scalars (Δ = new − baseline; negative is better for MAPE) ---")
        lines.append(f"  overall_weighted_mape Δ : {_fmt_signed(cmp.overall_mape_delta)}")
        lines.append(f"  verifier.overall_score Δ: {_fmt_signed(cmp.verifier_overall_delta)}")
        lines.append(f"  textbook_pass_rate Δ    : {_fmt_signed(cmp.textbook_pass_rate_delta)}")
        if cmp.verifier_category_deltas:
            lines.append("  verifier categories:")
            for cat, d in sorted(cmp.verifier_category_deltas.items(), key=lambda kv: kv[0]):
                lines.append(f"    {cat:<14} {_fmt_signed(d)}")

    if cmp.per_marker_mape_deltas:
        lines.append("\n--- Per-marker mean MAPE Δ ---")
        for marker, d in sorted(cmp.per_marker_mape_deltas.items(), key=lambda kv: kv[1]):
            lines.append(f"  {marker:<10} {_fmt_signed(d)}")

    if cmp.scenario_pass_rate_deltas:
        lines.append("\n--- Textbook scenario pass-rate Δ ---")
        for name, d in sorted(cmp.scenario_pass_rate_deltas.items(), key=lambda kv: -abs(kv[1])):
            lines.append(f"  {name:<32} {_fmt_signed(d, 3)}")

    regressions = [t for t in cmp.check_transitions if t.kind == "regression"]
    improvements = [t for t in cmp.check_transitions if t.kind == "improvement"]
    if regressions or improvements:
        lines.append("\n--- Per-check transitions ---")
        for t in regressions:
            lines.append(
                f"  REGRESS  {t.scenario}.{t.name}  "
                f"value: {t.baseline_value!r} -> {t.new_value!r}  threshold={t.threshold!r}"
            )
        for t in improvements:
            lines.append(
                f"  IMPROVE  {t.scenario}.{t.name}  "
                f"value: {t.baseline_value!r} -> {t.new_value!r}  threshold={t.threshold!r}"
            )

    if cmp.baseline_probe and cmp.new_probe:
        lines.append("\n--- Gut sweep at zero embedding (carbs g -> AUC of glucose appearance) ---")
        cold = cold_gut_sweep_targets(cmp.new_probe.carb_doses_g)
        lines.append(
            f"  {'carbs':>7} {'tgt_AUC':>9} {'base_AUC':>9} "
            f"{'new_AUC':>9} {'Δ':>9} {'tgt_pk':>7} {'new_pk':>7}"
        )
        for c, b, n in zip(cold, cmp.baseline_probe.gut_sweep, cmp.new_probe.gut_sweep):
            lines.append(
                f"  {n.carbs_g:>7.1f} {c.auc:>9.2f} {b.auc:>9.2f} "
                f"{n.auc:>9.2f} {_fmt_signed(n.auc - b.auc, 2):>9} "
                f"{c.peak:>7.3f} {n.peak:>7.3f}"
            )
        lines.append(
            f"  monotone: baseline={cmp.baseline_probe.gut_sweep_monotone}  "
            f"new={cmp.new_probe.gut_sweep_monotone}"
        )

        lines.append("\n--- Fasting drift (zero embedding, no meals) ---")
        b = cmp.baseline_probe.fasting_drift
        n = cmp.new_probe.fasting_drift
        lines.append(
            f"  baseline: t0={b.glucose_t0:.1f} tend={b.glucose_tend:.1f} "
            f"drift={b.drift_mg_dl:+.1f} mg/dL ({b.hours:.1f} h)"
        )
        lines.append(
            f"  new     : t0={n.glucose_t0:.1f} tend={n.glucose_tend:.1f} "
            f"drift={n.drift_mg_dl:+.1f} mg/dL ({n.hours:.1f} h)"
        )
        lines.append(f"  drift Δ : {_fmt_signed(n.drift_mg_dl - b.drift_mg_dl, 1)}")

        lines.append("\n--- Counter-regulation (75 g mixed meal, zero embedding) ---")
        bdic = {d.marker: d for d in cmp.baseline_probe.counter_regulation.deltas}
        ndic = {d.marker: d for d in cmp.new_probe.counter_regulation.deltas}
        for marker in sorted(set(bdic) | set(ndic)):
            bd = bdic.get(marker)
            nd = ndic.get(marker)
            if bd is None or nd is None:
                continue
            lines.append(
                f"  {marker:<9} base Δ={bd.delta:+.3f}  new Δ={nd.delta:+.3f}  "
                f"ΔΔ={_fmt_signed(nd.delta - bd.delta, 3)}"
            )
    elif cmp.new_probe:
        lines.append("\n--- Probe (new only; baseline checkpoint not provided) ---")
        for r in cmp.new_probe.gut_sweep:
            lines.append(
                f"  carbs={r.carbs_g:>5.1f}  AUC={r.auc:>8.2f}  "
                f"peak={r.peak:.3f}  tpeak={r.tpeak_min}"
            )
        d = cmp.new_probe.fasting_drift
        lines.append(
            f"  fasting drift {d.hours:.1f}h: {d.drift_mg_dl:+.1f} mg/dL"
        )

    return "\n".join(lines) + "\n"
