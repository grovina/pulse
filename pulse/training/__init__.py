"""
Training-signal framework for Pulse.

Each gradient-producing source of medical knowledge is a
``TrainingSignal``. The trainer orchestrates them; signals own their data,
their inner loop, and their own backward+step calls.

Rollout evidence (cohort statistics, physiology-rule hinges, dose-response,
per-patient meal glucose) shares one engine. Map, rate, and structure
signals stay their own engines.
"""

from .signals import (
    SignalContext,
    SignalResult,
    TrainingSignal,
    WeightSchedule,
)
from .safe_step import (
    NaNTrainingAbort,
    accumulate_grad,
    finalize_aux_accumulation,
    grad_snapshot,
    joint_aux_step,
    safe_step,
)
from .trajectory_signal import TrajectoryRolloutSignal
from .rollout_signal import (
    RolloutEvidenceSignal,
    perturb_dose_response_protocol,
    perturb_group_arms,
)
from .gut_dose_sweep_signal import GutDoseSweepProtocol, GutDoseSweepSignal
from .insulin_sweep_signal import InsulinSweepProtocol, InsulinSweepSignal
from .setpoint_supervision_signal import SetpointSupervisionSignal
from .embedding_prior_signal import EmbeddingPriorSignal
from .carb_mass_balance_signal import CarbMassBalanceSignal
from .cold_model_distillation_signal import ColdModelDistillationSignal

__all__ = [
    "SignalContext",
    "SignalResult",
    "TrainingSignal",
    "WeightSchedule",
    "NaNTrainingAbort",
    "safe_step",
    "accumulate_grad",
    "finalize_aux_accumulation",
    "grad_snapshot",
    "joint_aux_step",
    "TrajectoryRolloutSignal",
    "RolloutEvidenceSignal",
    "perturb_dose_response_protocol",
    "perturb_group_arms",
    "GutDoseSweepProtocol",
    "GutDoseSweepSignal",
    "InsulinSweepProtocol",
    "InsulinSweepSignal",
    "SetpointSupervisionSignal",
    "EmbeddingPriorSignal",
    "CarbMassBalanceSignal",
    "ColdModelDistillationSignal",
]
