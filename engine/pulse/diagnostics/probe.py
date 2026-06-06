"""Physiological probes for a trained pulse checkpoint at the zero embedding.

Three probes, each derived from a known-failure mode that integrated-state
benchmarks have a hard time exposing on their own:

- ``gut_sweep`` — directly query ``model.gut.forward_window`` with a carb dose
  grid and report AUC, peak, and time-to-peak of glucose appearance. Cold-model
  targets are provided by :func:`cold_gut_sweep_targets`. Monotonicity of AUC
  in carb dose is the first-order success criterion for any iteration that
  touches the gut kernel.
- ``fasting_drift`` — integrate from the cold initial state for N hours with
  no meals and report the glucose drift. Should be near zero at zero
  embedding; non-trivial drift indicates the model relies on ambient drift
  to "satisfy" supervision.
- ``counter_regulation`` — integrate a 75 g mixed meal from the cold initial
  state and report glucagon / FFA / ghrelin baseline → nadir deltas. The
  signs (all should suppress) and magnitudes are the canonical cold-model
  signature.

The same probes are used in tests and from the comparison CLI; treat this
module as the single source of truth.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..knowledge.full_body import PatientParams, compute_absorption_profile
from ..model import ModularPhysiologyNetwork, integrate
from ..modules.gut import MealEvent
from ..types import EMBEDDING_DIM, GUT_OUTPUT_DIM, MARKER_INDEX, NORM_CENTER


DEFAULT_CARB_DOSES_G: tuple[float, ...] = (0.0, 15.0, 30.0, 45.0, 60.0, 90.0, 120.0)
DEFAULT_FATS_G: float = 5.0
DEFAULT_PROTEINS_G: float = 10.0
DEFAULT_POST_WINDOW_MIN: int = 240
DEFAULT_FAST_HOURS: float = 4.0
DEFAULT_COUNTER_REG_MEAL = MealEvent(time=0.0, carbs=75.0, fats=20.0, proteins=25.0)
COUNTER_REG_MARKERS: tuple[str, ...] = ("glucagon", "ffa", "ghrelin")


@dataclass(frozen=True)
class GutSweepRow:
    carbs_g: float
    auc: float
    peak: float
    tpeak_min: int


@dataclass(frozen=True)
class DriftResult:
    glucose_t0: float
    glucose_tend: float
    drift_mg_dl: float
    hours: float


@dataclass(frozen=True)
class CounterRegulationDelta:
    marker: str
    baseline: float
    nadir: float
    delta: float


@dataclass(frozen=True)
class CounterRegulationResult:
    deltas: tuple[CounterRegulationDelta, ...]


@dataclass(frozen=True)
class ProbeReport:
    """Result of running all three probes on a single checkpoint.

    ``gut_sweep_monotone`` is true iff AUC is non-decreasing across the dose
    grid — the simplest invariant the gut kernel must satisfy at the zero
    embedding.
    """

    gut_sweep: tuple[GutSweepRow, ...]
    gut_sweep_monotone: bool
    fasting_drift: DriftResult
    counter_regulation: CounterRegulationResult
    carb_doses_g: tuple[float, ...] = field(default=DEFAULT_CARB_DOSES_G)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gut_sweep": [asdict(r) for r in self.gut_sweep],
            "gut_sweep_monotone": self.gut_sweep_monotone,
            "fasting_drift": asdict(self.fasting_drift),
            "counter_regulation": {
                "deltas": [asdict(d) for d in self.counter_regulation.deltas],
            },
            "carb_doses_g": list(self.carb_doses_g),
        }


def load_model_from_checkpoint(path: str | Path) -> tuple[ModularPhysiologyNetwork, dict[str, Any]]:
    """Load a checkpoint and reconstruct the model with its training-time dims."""
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt)
    h = int(ckpt.get("hidden_dim", 48))
    model = ModularPhysiologyNetwork(
        embedding_dim=int(ckpt.get("embedding_dim", EMBEDDING_DIM)),
        metabolic_hidden=h,
        appetite_hidden=max(24, h // 2),
        stress_hidden=max(24, h // 2),
        cardiovascular_hidden=h,
        thermoreg_hidden=max(16, h // 3),
        respiratory_hidden=max(16, h // 3),
    )
    model.load_state_dict(state)
    model.eval()
    return model, ckpt


def cold_gut_sweep_targets(
    carb_doses_g: tuple[float, ...] = DEFAULT_CARB_DOSES_G,
    fats_g: float = DEFAULT_FATS_G,
    proteins_g: float = DEFAULT_PROTEINS_G,
    post_window_min: int = DEFAULT_POST_WINDOW_MIN,
) -> tuple[GutSweepRow, ...]:
    """Cold-model AUC / peak / tpeak per dose, the target the model is distilled against."""
    params = PatientParams()
    rows: list[GutSweepRow] = []
    for d in carb_doses_g:
        meals = [(0.0, float(d), float(fats_g), float(proteins_g))]
        curve = np.zeros((post_window_min, GUT_OUTPUT_DIM), dtype=np.float32)
        for t in range(post_window_min):
            curve[t] = compute_absorption_profile(float(t), meals, params)
        glu = curve[:, 0]
        rows.append(GutSweepRow(
            carbs_g=float(d),
            auc=float(glu.sum()),
            peak=float(glu.max()),
            tpeak_min=int(glu.argmax()),
        ))
    return tuple(rows)


def _gut_sweep(
    model: ModularPhysiologyNetwork,
    embedding: torch.Tensor,
    carb_doses_g: tuple[float, ...],
    fats_g: float,
    proteins_g: float,
    post_window_min: int,
) -> tuple[GutSweepRow, ...]:
    emb_gut = model.embedding_projections["gut"](embedding)
    times = torch.arange(float(post_window_min))
    rows: list[GutSweepRow] = []
    with torch.no_grad():
        for d in carb_doses_g:
            meal = MealEvent(time=0.0, carbs=float(d), fats=fats_g, proteins=proteins_g)
            out = model.gut.forward_window(times, [meal], emb_gut)
            glu = out[:, 0]
            rows.append(GutSweepRow(
                carbs_g=float(d),
                auc=float(glu.sum()),
                peak=float(glu.max()),
                tpeak_min=int(glu.argmax()),
            ))
    return tuple(rows)


def _is_monotone(rows: tuple[GutSweepRow, ...], tol: float = 1e-6) -> bool:
    aucs = [r.auc for r in rows]
    return all(b >= a - tol for a, b in zip(aucs, aucs[1:]))


def _fasting_drift(
    model: ModularPhysiologyNetwork,
    embedding: torch.Tensor,
    hours: float,
    start_time_minutes: float,
) -> DriftResult:
    init = torch.tensor(NORM_CENTER, dtype=torch.float32)
    n = int(hours * 60.0)
    with torch.no_grad():
        traj = integrate(
            model, init, embedding,
            n_steps=n, dt=1.0,
            start_time_minutes=start_time_minutes,
            meals=[],
        )
    glu_idx = MARKER_INDEX["glucose"]
    return DriftResult(
        glucose_t0=float(traj[0, glu_idx]),
        glucose_tend=float(traj[-1, glu_idx]),
        drift_mg_dl=float(traj[-1, glu_idx] - traj[0, glu_idx]),
        hours=float(hours),
    )


def _counter_regulation(
    model: ModularPhysiologyNetwork,
    embedding: torch.Tensor,
    meal: MealEvent,
    start_time_minutes: float,
    n_steps: int,
    pre_window: int,
) -> CounterRegulationResult:
    init = torch.tensor(NORM_CENTER, dtype=torch.float32)
    with torch.no_grad():
        traj = integrate(
            model, init, embedding,
            n_steps=n_steps, dt=1.0,
            start_time_minutes=start_time_minutes,
            meals=[meal],
        )
    deltas: list[CounterRegulationDelta] = []
    for name in COUNTER_REG_MARKERS:
        i = MARKER_INDEX[name]
        baseline = float(traj[:pre_window, i].mean())
        nadir = float(traj[pre_window:, i].min())
        deltas.append(CounterRegulationDelta(
            marker=name, baseline=baseline, nadir=nadir, delta=nadir - baseline,
        ))
    return CounterRegulationResult(deltas=tuple(deltas))


def probe_checkpoint(
    model: ModularPhysiologyNetwork,
    embedding: torch.Tensor | None = None,
    *,
    carb_doses_g: tuple[float, ...] = DEFAULT_CARB_DOSES_G,
    fats_g: float = DEFAULT_FATS_G,
    proteins_g: float = DEFAULT_PROTEINS_G,
    post_window_min: int = DEFAULT_POST_WINDOW_MIN,
    fast_hours: float = DEFAULT_FAST_HOURS,
    counter_reg_meal: MealEvent = DEFAULT_COUNTER_REG_MEAL,
    start_time_minutes: float = 8.0 * 60.0,
    counter_reg_pre_window: int = 30,
) -> ProbeReport:
    """Run all three probes against a model.

    ``embedding`` defaults to the zero embedding (the manifold point the
    benchmark queries directly). Pass an arbitrary embedding to probe a
    specific patient.
    """
    if embedding is None:
        embedding = torch.zeros(model.embedding_dim, dtype=torch.float32)

    sweep = _gut_sweep(
        model, embedding, carb_doses_g, fats_g, proteins_g, post_window_min,
    )
    drift = _fasting_drift(model, embedding, fast_hours, start_time_minutes)
    cr = _counter_regulation(
        model, embedding, counter_reg_meal, start_time_minutes,
        n_steps=post_window_min, pre_window=counter_reg_pre_window,
    )
    return ProbeReport(
        gut_sweep=sweep,
        gut_sweep_monotone=_is_monotone(sweep),
        fasting_drift=drift,
        counter_regulation=cr,
        carb_doses_g=carb_doses_g,
    )
