"""Signal-balance probe: per-signal raw loss and ‖∇gut‖ at a checkpoint.

The physiological probes (``pulse.diagnostics.probe``) tell us how the
trained model behaves at inference. They don't tell us *why* training
ended where it did. When a signal weight escalation ships and a kernel
collapses (iter 13: gut output globally pulled toward a near-constant
~0.3 mg/min basin), the question we actually need to answer is:

  "At the converged checkpoint, how much does each signal want to move
  the gut kernel? And is that pull big enough to escape the local
  minimum, or has the kernel found a flat region where every signal's
  gradient on the gut params is small?"

This module measures exactly that. For each registered training signal
(plus the trajectory rollout signal's internal gut_loss term), it:

1. zero-grads the model;
2. calls ``signal.compute(model, embeddings, ctx)`` with a no-op
   optimizer that lets gradients survive ``compute()``;
3. records the raw scalar loss reported by the signal;
4. measures the L2 grad norm both for the gut-kernel parameter subset
   and for all model parameters.

Comparing the gut-grad-norm of two checkpoints (e.g. iter-N vs iter-N-1)
quantifies how much one iteration's hyperparameter change altered each
signal's leverage on the gut module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, cast

import numpy as np
import torch
import torch.nn as nn

from ..knowledge import ALL_COHORT_STATISTICS, ALL_CONTRIBUTIONS
from ..training import (
    CohortStatisticSignal,
    GutDoseSweepSignal,
    InsulinSweepSignal,
    SignalContext,
    TrainingSignal,
    TrajectoryRolloutSignal,
    WeightSchedule,
)
from ..types import EMBEDDING_DIM
from .probe import load_model_from_checkpoint


@dataclass(frozen=True)
class SignalBalance:
    """Per-signal training-time balance at one checkpoint."""

    name: str
    raw_loss: float                 # weight=1.0 magnitude of the loss
    grad_norm_gut: float            # L2 norm restricted to gut kernel + projection
    grad_norm_metabolic: float      # L2 norm restricted to metabolic + projection
    grad_norm_all: float            # L2 norm across all model params
    sub_metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SignalBalanceReport:
    """All-signal balance for one checkpoint, ready for serialization."""

    checkpoint: str
    n_gut_params: int
    n_metabolic_params: int
    n_all_params: int
    signals: tuple[SignalBalance, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "n_gut_params": self.n_gut_params,
            "n_metabolic_params": self.n_metabolic_params,
            "n_all_params": self.n_all_params,
            "signals": [asdict(s) for s in self.signals],
        }


class _NoopOptimizer:
    """Lets ``signal.compute`` call ``.step()`` / ``.zero_grad()`` without
    actually clearing grads, so we can read what each signal's
    ``backward()`` produced."""

    def step(self) -> None: pass

    def zero_grad(self, *_args: Any, **_kwargs: Any) -> None: pass


def _gut_kernel_params(model: nn.Module) -> list[nn.Parameter]:
    """Parameters whose gradient is the relevant lever for "the gut kernel" —
    the kernel itself plus the gut-projection of the patient embedding.
    """
    out: list[nn.Parameter] = []
    for name, p in model.named_parameters():
        if name.startswith("gut.") or "embedding_projections.gut" in name:
            out.append(p)
    return out


def _metabolic_params(model: nn.Module) -> list[nn.Parameter]:
    """The metabolic MLP and its embedding projection — the lever the
    insulin sweep signal lands on, and the lever downstream supervision
    signals (cohort statistics, dose-response) lean on for glucose /
    insulin / hepatic accuracy. Tracking ‖∇metabolic‖ alongside ‖∇gut‖
    is the iter 20 analog of the iter 19 gradient-budget analysis."""
    out: list[nn.Parameter] = []
    for name, p in model.named_parameters():
        if name.startswith("metabolic.") or "embedding_projections.metabolic" in name:
            out.append(p)
    return out


def _grad_norm(params: Iterable[nn.Parameter]) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum().item())
    return float(np.sqrt(total))


def _zero_grads(model: nn.Module) -> None:
    for p in model.parameters():
        p.grad = None


def _build_default_signals(n_patients: int) -> list[TrainingSignal]:
    """The signals that supervise the gut and downstream metabolic modules
    (directly or via coupling). Constructed at weight=1.0 so the report
    shows raw loss magnitudes; multiply by the spec.json weight to
    recover the in-spec contribution.
    """
    contrib_weights = {c.name: 1.0 for c in ALL_CONTRIBUTIONS}
    return [
        GutDoseSweepSignal(
            n_patients=n_patients,
            sample_patients=4,
            include_default_embedding=True,
            weight=WeightSchedule(1.0),
        ),
        InsulinSweepSignal(
            n_patients=n_patients,
            sample_patients=4,
            include_default_embedding=True,
            weight=WeightSchedule(1.0),
        ),
        CohortStatisticSignal(
            specs=list(ALL_COHORT_STATISTICS),
            n_patients=n_patients,
            sample_patients=4,
            include_default_embedding=True,
            weight=WeightSchedule(1.0),
        ),
        TrajectoryRolloutSignal(
            n_patients=n_patients,
            n_days=2,
            seed=0,
            contribution_weights=contrib_weights,
            windows_per_patient=1,
            meal_window_bias=0.55,
            input_dropout=0.0,
            huber_delta=1.0,
            gut_loss_weight=1.0,
            coupling_weight=WeightSchedule(0.0),
            verifier_weight=WeightSchedule(0.0),
            coupling_prior_samples=0,
            trajectory_band=0.2,
            trajectory_band_default=0.05,
            landmark_weight=WeightSchedule(0.0),
            n_default_patients=2,
        ),
    ]


def measure_signal_balance(
    model: nn.Module,
    *,
    n_patients: int = 20,
    seed: int = 0,
    epoch: int = 10,
    total_epochs: int = 80,
    signals: list[TrainingSignal] | None = None,
) -> tuple[SignalBalance, ...]:
    """Run each signal once and report (raw_loss, ‖∇gut‖, ‖∇all‖).

    ``model`` is mutated (training mode + grad accumulation) but its
    parameters are never updated — the optimizer is a no-op. Caller is
    responsible for passing a fresh / freshly-loaded model if they care.
    """
    model.train()
    embeddings = nn.Embedding(n_patients, EMBEDDING_DIM)
    nn.init.zeros_(embeddings.weight)
    rng = np.random.default_rng(seed)
    device = torch.device("cpu")
    params = list(model.parameters()) + list(embeddings.parameters())
    ctx = SignalContext(
        epoch=epoch,
        total_epochs=total_epochs,
        device=device,
        rng=rng,
        params=params,
        optimizer=cast(torch.optim.Optimizer, _NoopOptimizer()),
        grad_clip=1e9,
    )
    sigs = signals if signals is not None else _build_default_signals(n_patients)
    gut_params = _gut_kernel_params(model)
    metabolic_params = _metabolic_params(model)

    out: list[SignalBalance] = []
    for sig in sigs:
        _zero_grads(model)
        result = sig.compute(model, embeddings, ctx)
        out.append(SignalBalance(
            name=sig.name,
            raw_loss=float(result.loss_sum),
            grad_norm_gut=_grad_norm(gut_params),
            grad_norm_metabolic=_grad_norm(metabolic_params),
            grad_norm_all=_grad_norm(model.parameters()),
            sub_metrics=dict(result.sub_metrics) if result.sub_metrics else {},
        ))
    return tuple(out)


def signal_balance_for_checkpoint(
    checkpoint: str | Path,
    **kwargs: Any,
) -> SignalBalanceReport:
    """Convenience wrapper: load a checkpoint, measure, return a report."""
    model, _ = load_model_from_checkpoint(checkpoint)
    signals = measure_signal_balance(model, **kwargs)
    n_gut = sum(p.numel() for p in _gut_kernel_params(model))
    n_metabolic = sum(p.numel() for p in _metabolic_params(model))
    n_all = sum(p.numel() for p in model.parameters())
    return SignalBalanceReport(
        checkpoint=str(checkpoint),
        n_gut_params=n_gut,
        n_metabolic_params=n_metabolic,
        n_all_params=n_all,
        signals=signals,
    )


def render(report: SignalBalanceReport) -> str:
    """Human-readable rendering, suitable for pasting into a handoff doc."""
    lines: list[str] = []
    lines.append(f"signal-balance: {report.checkpoint}")
    gut_pct = report.n_gut_params / max(report.n_all_params, 1) * 100.0
    met_pct = report.n_metabolic_params / max(report.n_all_params, 1) * 100.0
    lines.append(
        f"  gut params: {report.n_gut_params} ({gut_pct:.1f}%); "
        f"metabolic params: {report.n_metabolic_params} ({met_pct:.1f}%); "
        f"total: {report.n_all_params}",
    )
    header = (
        f"  {'signal':<22} {'raw_loss':>12} "
        f"{'||g_gut||':>12} {'||g_met||':>12} {'||g_all||':>12}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for s in report.signals:
        lines.append(
            f"  {s.name:<22} {s.raw_loss:>12.4e} "
            f"{s.grad_norm_gut:>12.4e} {s.grad_norm_metabolic:>12.4e} "
            f"{s.grad_norm_all:>12.4e}",
        )
    return "\n".join(lines)
