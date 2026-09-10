"""
Respiratory module.

Respiratory rate and SpO₂. Lactate and temperature are the coupling inputs.
RR relaxes to a per-patient setpoint; SpO₂ is stepped in logit coordinates of
(70, 100) by the integrator so saturation cannot leave that interval.
Above moderate effort, SpO₂ has the teacher's exercise dip
(``spo2_exercise_dip * relu(activity − 0.5)``) so the marker is not a
constant that a 95–100 band can pass.
"""

import math

import torch
import torch.nn as nn

from .base import LearnedDynamicsModule
from ..types import MARKER_INDEX, MODULE_COUPLING_CHANNELS, MODULE_MARKER_INDICES, NORM_CENTER, NORM_SCALE

_N_COUPLING = len(MODULE_COUPLING_CHANNELS["respiratory"])
_N_EXTERNAL = 2
_N_STATE = 2
_RR = 0
_SPO2 = 1

_RSP_CENTER = [NORM_CENTER[i] for i in MODULE_MARKER_INDICES["respiratory"]]
_RSP_SCALE = [NORM_SCALE[i] for i in MODULE_MARKER_INDICES["respiratory"]]

_RR_MAX_Z = 2.0
_SPO2_MAX_Z = 1.0
_K_MIN, _K_RANGE, _K_INIT = 0.02, 0.20, 0.08
_LOG_K_INIT = math.log(
    ((_K_INIT - _K_MIN) / _K_RANGE) / (1.0 - (_K_INIT - _K_MIN) / _K_RANGE)
)
_ACTIVITY_EXTERNAL_IDX = 0
# Teacher PatientParams.spo2_exercise_dip. Zero at rest and easy activity.
_SPO2_EXERCISE_DIP = 1.5
_SPO2_ACT_THRESH = 0.5


class RespiratoryModule(LearnedDynamicsModule):
    def __init__(self, embedding_dim: int, hidden_dim: int = 24):
        super().__init__(
            n_state=_N_STATE,
            n_coupling=_N_COUPLING,
            n_external=_N_EXTERNAL,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
        )
        _bh = max(8, hidden_dim // 4)
        self.setpoint_net = nn.Sequential(
            nn.Linear(embedding_dim, _bh), nn.Tanh(), nn.Linear(_bh, _N_STATE),
        )
        with torch.no_grad():
            self.setpoint_net[-1].weight.zero_()
            self.setpoint_net[-1].bias.zero_()
        self.log_k = nn.Parameter(torch.full((_N_STATE,), _LOG_K_INIT))
        self.register_buffer("rsp_center", torch.tensor(_RSP_CENTER, dtype=torch.float32))
        self.register_buffer("rsp_scale", torch.tensor(_RSP_SCALE, dtype=torch.float32))

    def setpoints_raw(self, embedding: torch.Tensor) -> torch.Tensor:
        o = self.setpoint_net(embedding)
        rr = self.rsp_center[_RR] + _RR_MAX_Z * self.rsp_scale[_RR] * torch.tanh(o[..., 0])
        spo2 = self.rsp_center[_SPO2] + _SPO2_MAX_Z * self.rsp_scale[_SPO2] * torch.tanh(o[..., 1])
        return torch.stack([rr, spo2], dim=-1)

    def forward(
        self,
        state: torch.Tensor,
        coupling: torch.Tensor,
        external: torch.Tensor,
        embedding: torch.Tensor,
        time_features: torch.Tensor,
    ) -> torch.Tensor:
        driver = super().forward(state, coupling, external, embedding, time_features)
        k = _K_MIN + _K_RANGE * torch.sigmoid(self.log_k)
        sp = self.setpoints_raw(embedding)
        raw = self.rsp_center + self.rsp_scale * state
        rate = driver - k * (raw - sp)
        act = external[..., _ACTIVITY_EXTERNAL_IDX]
        spo2_ex = _SPO2_EXERCISE_DIP * nn.functional.relu(act - _SPO2_ACT_THRESH)
        rate_spo2 = rate[..., _SPO2] - spo2_ex
        return torch.stack([rate[..., _RR], rate_spo2], dim=-1)
