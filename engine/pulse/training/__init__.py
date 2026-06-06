"""
Training-signal framework for Pulse.

Each gradient-producing source of medical knowledge (cold-model trajectory
distillation, cohort statistics, mechanism priors, scenario checks) is a
``TrainingSignal``. The trainer orchestrates them; signals own their data,
their inner loop, and their own backward+step calls.

This isolates the *what* (which knowledge source contributes) from the
*how* (rollout cadence, gradient handling) and lets us add new signals
without surgery on the main loop.
"""

from .signals import (
    SignalContext,
    SignalResult,
    TrainingSignal,
    WeightSchedule,
)
from .safe_step import NaNTrainingAbort, safe_step
from .trajectory_signal import TrajectoryRolloutSignal
from .cohort_signal import CohortStatisticSignal
from .dose_response_signal import DoseResponseSignal
from .gut_dose_sweep_signal import GutDoseSweepProtocol, GutDoseSweepSignal
from .insulin_sweep_signal import InsulinSweepProtocol, InsulinSweepSignal
from .fasting_stability_signal import FastingStabilitySignal
from .postprandial_recovery_signal import PostprandialRecoverySignal
from .default_baseline_signal import DefaultBaselineSignal
from .carb_mass_balance_signal import CarbMassBalanceSignal
from .cold_model_distillation_signal import ColdModelDistillationSignal
from .physiology_rules_signal import PhysiologyRulesSignal

__all__ = [
    "SignalContext",
    "SignalResult",
    "TrainingSignal",
    "WeightSchedule",
    "NaNTrainingAbort",
    "safe_step",
    "TrajectoryRolloutSignal",
    "CohortStatisticSignal",
    "DoseResponseSignal",
    "GutDoseSweepProtocol",
    "GutDoseSweepSignal",
    "InsulinSweepProtocol",
    "InsulinSweepSignal",
    "FastingStabilitySignal",
    "PostprandialRecoverySignal",
    "DefaultBaselineSignal",
    "CarbMassBalanceSignal",
    "ColdModelDistillationSignal",
    "PhysiologyRulesSignal",
]
