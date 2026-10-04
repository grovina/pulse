"""
Respiratory module.

Respiratory rate and SpO₂. Lactate and temperature are the coupling inputs.
RR relaxes to a per-patient setpoint; SpO₂ is stepped in logit coordinates of
(70, 100) by the integrator so saturation cannot leave that interval.
Above moderate effort, SpO₂ has the teacher's exercise dip
(``spo2_exercise_dip * relu(activity − 0.5)``) so the marker is not a
constant that a 95–100 band can pass.

Setpoint decode (PLAN A1, 2026-10-04): the teacher draws RR0 lognormally and SpO2_0
additively, and the student's two decoders now match those families — RR relaxes to a
LOG-decoded setpoint (``15·exp(L·tanh(head))``, so the population mean sits above the
zero-embedding value by Jensen, as the teacher's does), SpO₂ to an ADDITIVE one. The bounds
below give the numbers, and why SpO₂ cannot be log-decoded.
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

# RR decodes in LOG space (A1): RR_sp = 15·exp(±0.52·tanh) → [8.9, 25.2] /min. It replaces the
# additive 15 ± 2·3 = 9-21 (L = ln(15/9) = 0.511 rounded up, so the 9 /min floor is kept; the
# ceiling rises because equal-ratio spans put it at center²/floor — the additive 21 had been out of
# reach for 1.2 % of the teacher's own patients). The teacher draws RR0 lognormally (σ 0.15), so
# its mean sits above its median (measured, 20 000 draws: 1.009); the log decode reproduces that by
# Jensen, where an additive one gives E[RR_sp] = center for any zero-mean embedding. A checkpoint
# through iter 109 loads unchanged in shape but its RR offset now acts 1.30× as strongly near a zero
# head (15·0.52 vs 2·3 /min per unit of head output), so it needs the retrain PLAN.md §4 schedules.
_RR_LOG_SP_MAX = 0.52
# SpO₂ stays ADDITIVE, ±1.5 % → [96.5, 99.5], and log space would be wrong for it. The teacher draws
# it additively and clipped (``min(100, max(94, 98 + N(0, 1)))``), so its population is symmetric
# about the median — measured over 20 000 draws: median 98.002, mean 97.992, the clip at 100
# pulling the mean DOWN by 0.01 — and an additive decode of a zero-mean embedding is already right
# at the median and the mean; there is no Jensen shift to reproduce. It is also a percentage 2
# points under its ceiling: any log span that reaches the teacher's 94 % floor (ln(98/94) = 0.042)
# also reaches 98·e^0.042 = 102.2 %, a resting saturation above 100 %. (The ±1.5 span itself is a
# separate question from the family: it misses 13.1 % of sampled patients, 6.4 % below 96.5 and
# 6.7 % above 99.5, against the teacher's σ = 1 draw.)
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
        # Setpoint head outputs: [log RR offset, SpO₂ offset (z)]. Final layer zero-init ⇒
        # tanh(0) = 0 ⇒ RR 15·exp(0) = 15 and SpO₂ 98 for the default patient.
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
        """Resting ``[RR, SpO₂]`` in raw units for ``embedding[..., E]``: RR is strictly
        positive for every embedding (log decode), SpO₂ is 98 ± 1.5 (additive decode)."""
        o = self.setpoint_net(embedding)
        rr = self.rsp_center[_RR] * torch.exp(_RR_LOG_SP_MAX * torch.tanh(o[..., 0]))
        spo2 = self.rsp_center[_SPO2] + _SPO2_MAX_Z * self.rsp_scale[_SPO2] * torch.tanh(o[..., 1])
        return torch.stack([rr, spo2], dim=-1)

    def setpoints_z(self, embedding: torch.Tensor) -> torch.Tensor:
        """Resting ``[RR, SpO₂]`` as z-scores ``(raw − NORM_CENTER)/NORM_SCALE`` — the frame a
        setpoint-supervision signal compares in. The module owns the decode (RR is no longer
        ``MAX_Z·tanh(head)``), the same hand-off ``CardiovascularModule.setpoints_z`` made in iter 97."""
        return (self.setpoints_raw(embedding) - self.rsp_center) / self.rsp_scale

    def constants(self, embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "k": _K_MIN + _K_RANGE * torch.sigmoid(self.log_k),
            "setpoint": self.setpoints_raw(embedding),
        }

    def drives(
        self,
        external: torch.Tensor,
        coupling: torch.Tensor,
        time_features: torch.Tensor,
        const: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        act = external[..., _ACTIVITY_EXTERNAL_IDX]
        return {"spo2_exercise": _SPO2_EXERCISE_DIP * nn.functional.relu(act - _SPO2_ACT_THRESH)}

    def step(
        self,
        state: torch.Tensor,
        coupling: torch.Tensor,
        const: dict[str, torch.Tensor],
        drv: dict[str, torch.Tensor],
        raw: dict[object, torch.Tensor],
    ) -> torch.Tensor:
        raw_s = self.rsp_center + self.rsp_scale * state
        rate = raw["driver"] - const["k"] * (raw_s - const["setpoint"])
        rate_spo2 = rate[..., _SPO2] - drv["spo2_exercise"]
        return torch.stack([rate[..., _RR], rate_spo2], dim=-1)
