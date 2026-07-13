"""
Thermoregulation module.

Core temperature. Receives metabolic rate proxy (glucose) and cortisol from other modules.

Iter 91 — PER-PATIENT SETPOINT. Through iter 90 this was a bare learned MLP with no restoring
structure at all: temperature had to hold its own resting level through 12h of integrated rate,
with nothing telling it what that level was. That is precisely the gap the cardiovascular module
had until iter 89, and it failed there in a measurable way (resting HR drifted to a constant
+15 bpm bias, and hr_mape sat at 0.19-0.21 for ~10 iterations). Giving CVS an explicit
per-patient setpoint fixed it in one iter (hr_mape 0.206 -> 0.158).

The same argument applies here, and the same evidence:
  * The TEACHER gives every patient a resting T0 (full_body.py PatientParams) and relaxes
    temperature toward `T0 + circadian + exercise + sleep_shift` with rate k_temp. So there IS
    a per-patient setpoint in the data; the student simply had no way to represent it.
  * temp is a GATE marker with a thin margin (mape 0.0182 against a 0.02 threshold on iter-89)
    and the physics review found it OVER-PREDICTS excursions -- the signature of a module with
    no restoring force, which drifts under its couplings instead of returning to a level.

So temperature now relaxes to a per-patient setpoint plus the learned drivers, exactly as
glucose does (-(Sg+X)(G - Gb_emb) + Ra) and as the four cardiovascular vitals do. This is the
ADDITIVE form, NOT the iter-36 input-centering that failed: the MLP still sees UNCENTERED
normalized state, so its coupling response is learned and evaluated in the same frame, and only
an explicit `-k*(state - setpoint)` force is added to the rate. The setpoint head is zero-init,
so at cold start the setpoint is exactly NORM_CENTER (37.0 C) and per-patient authority is
learned rather than assumed.

The teacher's T0 is now also supervised directly (SetpointSupervisionSignal), so this head has a
ground-truth target rather than having to discover the level from trajectories.
"""

import math

import torch
import torch.nn as nn

from .base import LearnedDynamicsModule
from ..types import MARKER_INDEX, NORM_SCALE

# Coupling inputs: glucose (metabolic rate proxy) (1) + cortisol (1) = 2
_N_COUPLING = 2

# External inputs: activity (1) + sleep_wake (1) = 2
_N_EXTERNAL = 2

_TEMP_NORM_SCALE = NORM_SCALE[MARKER_INDEX["temp"]]

# Per-patient resting-temperature offset, z-units (NORM_CENTER 37.0, NORM_SCALE 0.3).
# +-1.5 z => T0 in ~[36.55, 37.45] C, which covers the physiological resting spread (the teacher
# draws T0 = 37.0 + N(0, 0.2)) with margin, without letting calibration invent a fever.
_TEMP_SETPOINT_MAX_Z = 1.5

# Raw per-minute restoring rate (iter-90 frame convention: the normalized deviation is scaled by
# NORM_SCALE inside the rate, so k is a true 1/min constant). Teacher k_temp = 0.025 (tau 40 min
# -- core temperature is a slow, high-inertia variable). Band [0.005, 0.10] => tau 10-200 min.
_TEMP_K_MIN = 0.005
_TEMP_K_RANGE = 0.095
_TEMP_K_INIT = 0.025  # teacher full_body.py PatientParams.k_temp
_TEMP_LOG_K_INIT = math.log(
    ((_TEMP_K_INIT - _TEMP_K_MIN) / _TEMP_K_RANGE)
    / (1.0 - (_TEMP_K_INIT - _TEMP_K_MIN) / _TEMP_K_RANGE)
)


class ThermoregModule(LearnedDynamicsModule):
    def __init__(self, embedding_dim: int, hidden_dim: int = 24):
        super().__init__(
            n_state=1,
            n_coupling=_N_COUPLING,
            n_external=_N_EXTERNAL,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
        )
        # Per-patient resting-temperature setpoint. Zero-init final layer => offset is exactly 0
        # for every embedding at cold start, so the default patient rests at NORM_CENTER (37.0 C)
        # and per-patient authority grows during training. Mirrors metabolic.glucose_baseline_net
        # and cardiovascular.setpoint_net.
        _bh = max(8, hidden_dim // 4)
        self.setpoint_net = nn.Sequential(
            nn.Linear(embedding_dim, _bh), nn.Tanh(), nn.Linear(_bh, 1),
        )
        with torch.no_grad():
            self.setpoint_net[-1].weight.zero_()
            self.setpoint_net[-1].bias.zero_()
        self.log_k = nn.Parameter(torch.tensor(_TEMP_LOG_K_INIT))

    def forward(
        self,
        state: torch.Tensor,
        coupling: torch.Tensor,
        external: torch.Tensor,
        embedding: torch.Tensor,
        time_features: torch.Tensor,
    ) -> torch.Tensor:
        # Learned drivers (circadian via time_features, exercise via activity, sleep, and the
        # glucose/cortisol couplings). The MLP is unchanged and still sees UNCENTERED normalized
        # state — the iter-36 frame-consistency requirement.
        driver = super().forward(state, coupling, external, embedding, time_features)
        setpoint = _TEMP_SETPOINT_MAX_Z * torch.tanh(self.setpoint_net(embedding))  # [..., 1]
        k = _TEMP_K_MIN + _TEMP_K_RANGE * torch.sigmoid(self.log_k)
        # rate = driver - k*(state_raw - setpoint_raw), with k a true per-minute constant.
        return driver - k * (state - setpoint) * _TEMP_NORM_SCALE
