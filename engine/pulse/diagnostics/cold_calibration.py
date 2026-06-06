"""
Cold-model calibration check: literature targets vs the hand-tuned cold model.

The PRD's literature-absorption strategy assumes the cold model
(``knowledge.full_body.simulate_full_body``) produces *plausible*
trajectories that the differentiable model can refine toward
literature targets. Iter 20's diagnostic uncovered the silent failure
mode: when the cold model's hand-tuned parameters disagree with the
cohort specs, the trajectory_rollout signal pulls the trained model
*away* from literature truth — and because trajectory pull is
typically 10-100× larger than cohort pull on any single module, the
literature signal loses every gradient-budget contest.

This diagnostic catches that class of miscalibration up-front:

* For every ``CohortStatisticSpec`` in the registry, run the cold
  model through the spec's protocol (one arm per spec; ``DELTA_*``
  kinds require both arms).
* Extract the same statistic the cohort signal extracts at training
  time.
* Report ``(predicted, target, z)`` plus a flag for ``|z| > 2`` —
  the threshold above which the cold model is *actively misleading*
  the differentiable model.

Use this:

* Before any training iteration that introduces new cohort specs,
  to confirm every new spec is at least *reachable* from the cold
  model state.
* After any change to ``PatientParams``, to verify the new params
  haven't silently broken specs that previously calibrated.
* When a cohort spec stays stuck despite weight bumps — if the cold
  model itself doesn't produce the target, no amount of training
  signal weight will fix it. The fix is in the cold model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ..knowledge import ALL_COHORT_STATISTICS
from ..knowledge.cohort_types import (
    CohortArmSpec,
    CohortStatisticSpec,
    StatisticKind,
    StatisticWindow,
)
from ..knowledge.full_body import PatientParams
from ..knowledge.textbook_scenarios.base import cold_model_trajectory
from ..types import MARKER_INDEX


@dataclass(frozen=True)
class ColdCalibrationRow:
    name: str
    source: str
    marker_id: str
    cold_predicted: float
    target: float
    sigma: float
    z_residual: float
    misleading: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ColdCalibrationReport:
    n_specs: int
    n_misleading: bool
    rows: tuple[ColdCalibrationRow, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_specs": self.n_specs,
            "n_misleading": int(sum(1 for r in self.rows if r.misleading)),
            "rows": [r.to_dict() for r in self.rows],
        }


def _arm_window(spec: CohortStatisticSpec, arm_idx: int) -> StatisticWindow:
    if spec.per_arm_windows is None:
        return spec.window
    return spec.per_arm_windows[arm_idx]


def _extract_statistic(
    traj: np.ndarray,
    marker_idx: int,
    window: StatisticWindow,
    kind: StatisticKind,
) -> float:
    series = traj[window.start_min:window.end_min, marker_idx]
    if kind in (StatisticKind.MEAN_IN_WINDOW, StatisticKind.DELTA_MEANS):
        return float(series.mean())
    if kind in (StatisticKind.PEAK_VALUE, StatisticKind.DELTA_PEAKS):
        return float(series.max())
    if kind == StatisticKind.TIME_TO_PEAK:
        return float(int(series.argmax()))
    raise ValueError(f"Unknown statistic kind: {kind}")


def _run_arm(
    arm: CohortArmSpec,
    params: PatientParams,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return cold_model_trajectory(
        params,
        list(arm.meals),
        arm.duration_min,
        arm.start_hour,
        rng=rng,
    )


def cold_calibration(
    *,
    params: PatientParams | None = None,
    specs: list[CohortStatisticSpec] | None = None,
    seed: int = 0,
    misleading_z: float = 2.0,
) -> ColdCalibrationReport:
    """Run every cohort spec against the cold model and report z-residuals."""
    spec_iter = list(specs) if specs is not None else list(ALL_COHORT_STATISTICS)
    p = params if params is not None else PatientParams()

    rows: list[ColdCalibrationRow] = []
    for i, spec in enumerate(spec_iter):
        marker_idx = MARKER_INDEX[spec.marker_id]
        per_arm: list[float] = []
        for arm_idx, arm in enumerate(spec.arms):
            traj = _run_arm(arm, p, seed=seed + i * 100 + arm_idx)
            win = _arm_window(spec, arm_idx)
            per_arm.append(_extract_statistic(traj, marker_idx, win, spec.kind))
        if spec.kind in (StatisticKind.DELTA_MEANS, StatisticKind.DELTA_PEAKS):
            pred = per_arm[1] - per_arm[0]
        else:
            pred = per_arm[0]
        z = (pred - float(spec.target)) / float(spec.sigma)
        rows.append(ColdCalibrationRow(
            name=spec.name,
            source=spec.source,
            marker_id=spec.marker_id,
            cold_predicted=float(pred),
            target=float(spec.target),
            sigma=float(spec.sigma),
            z_residual=float(z),
            misleading=bool(abs(z) > misleading_z),
        ))

    n_misleading = bool(any(r.misleading for r in rows))
    return ColdCalibrationReport(
        n_specs=len(rows),
        n_misleading=n_misleading,
        rows=tuple(rows),
    )


def render(report: ColdCalibrationReport, *, sort_by_z: bool = True) -> str:
    rows = sorted(report.rows, key=lambda r: -abs(r.z_residual)) if sort_by_z else list(report.rows)
    n_mis = sum(1 for r in rows if r.misleading)
    lines = [
        f"cold-calibration: {report.n_specs} specs, {n_mis} misleading (|z| > 2)",
        f"  {'spec':<40} {'marker':<14} {'cold_pred':>10} {'target':>10} {'sigma':>8} {'z':>7} mis",
        "  " + "-" * 96,
    ]
    for r in rows:
        flag = " *" if r.misleading else ""
        lines.append(
            f"  {r.name[:40]:<40} {r.marker_id:<14} "
            f"{r.cold_predicted:>10.3f} {r.target:>10.3f} {r.sigma:>8.3f} "
            f"{r.z_residual:>7.2f}{flag}"
        )
    return "\n".join(lines)
