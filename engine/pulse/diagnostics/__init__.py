"""Diagnostics utilities for pulse training runs.

Four reusable surfaces:

- ``probe`` runs a fixed set of physiological probes against a checkpoint at the
  zero embedding (gut absorption sweep, fasting glucose drift, counter-regulatory
  hormone response). It is the canonical, in-code answer to "how does this
  checkpoint behave on the queries that matter, separately from the benchmark".
- ``compare`` ingests two checkpoints + their benchmark reports and prints a
  structured delta covering both probes and benchmark scalars.
- ``signal_balance`` measures, per training signal, the raw loss magnitude and
  the L2 grad norm landed on the gut and metabolic module parameters at a
  converged checkpoint. Used to diagnose iterations where weight changes (or
  kernel-collapse pathologies) reshape the optimization landscape — see iter
  13/14, and the iter 19 → 20 metabolic-module extension.
- ``cohort_ablation`` un-averages the cohort statistic signal: per-spec
  z-residual + per-module grad-norm at one checkpoint. Catches dead
  literature contributions (zero gradient on every module) and reveals
  which physiology pillar each spec actually pulls on.

All four surfaces are exposed via ``python -m pulse.diagnostics``.
"""

from .cohort_ablation import (
    CohortAblationReport,
    CohortAblationRow,
    cohort_ablation,
    cohort_ablation_for_checkpoint,
)
from .cold_calibration import (
    ColdCalibrationReport,
    ColdCalibrationRow,
    cold_calibration,
)
from .probe import (
    CounterRegulationResult,
    DriftResult,
    GutSweepRow,
    ProbeReport,
    load_model_from_checkpoint,
    probe_checkpoint,
    cold_gut_sweep_targets,
)
from .signal_balance import (
    SignalBalance,
    SignalBalanceReport,
    measure_signal_balance,
    signal_balance_for_checkpoint,
)

__all__ = [
    "CohortAblationReport",
    "CohortAblationRow",
    "ColdCalibrationReport",
    "ColdCalibrationRow",
    "CounterRegulationResult",
    "DriftResult",
    "GutSweepRow",
    "ProbeReport",
    "SignalBalance",
    "SignalBalanceReport",
    "cohort_ablation",
    "cohort_ablation_for_checkpoint",
    "cold_calibration",
    "cold_gut_sweep_targets",
    "load_model_from_checkpoint",
    "measure_signal_balance",
    "probe_checkpoint",
    "signal_balance_for_checkpoint",
]
