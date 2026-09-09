"""
Stress / HPA Axis module.

The cascade is CRH → ACTH → cortisol. Circadian drive, sleep suppression,
hypoglycaemia and activity enter at CRH; ACTH tracks CRH; cortisol tracks ACTH.
Cortisol's negative feedback on CRH is one-sided saturating about the
patient's basal: high cortisol suppresses CRH; the nocturnal nadir is
sleep's. A rectifier at 12 µg/dL would leave the whole overnight range
inert.
"""

import math

import torch
import torch.nn as nn

from .base import ConstantFluxHead, MassActionModule
from ..types import MARKER_INDEX, MODULE_COUPLING_CHANNELS, MODULE_MARKER_INDICES, NORM_CENTER, NORM_SCALE

_N_COUPLING = len(MODULE_COUPLING_CHANNELS["stress"])
_N_EXTERNAL = 2

_CORTISOL_IDX = 0
_ACTH_IDX = 1
_CRH_IDX = 2

_TYPICALS = [NORM_CENTER[i] for i in MODULE_MARKER_INDICES["stress"]]
_NORM_SCALES = [NORM_SCALE[i] for i in MODULE_MARKER_INDICES["stress"]]
_CONS_SCALES = [0.02, 0.04, 0.08]

_CORT_B = float(_TYPICALS[_CORTISOL_IDX])
_ACTH_B = float(_TYPICALS[_ACTH_IDX])
_CRH_B = float(_TYPICALS[_CRH_IDX])

_GLUCOSE_CENTER = NORM_CENTER[MARKER_INDEX["glucose"]]
_GLUCOSE_SCALE = NORM_SCALE[MARKER_INDEX["glucose"]]

_K_CORT = 0.02
_K_ACTH = 0.04
_K_CRH = 0.08
_ACTH_PER_CRH = _ACTH_B / _CRH_B
_CORT_PER_ACTH = 0.45
_CRH_CIRC_AMP = 18.0 / _ACTH_PER_CRH   # same ACTH swing as the teacher
_HPA_SLEEP_SUPP = 0.35
_HPA_RISE_START_H = 2.0
_HPA_PEAK_H = 6.5
_HPA_FALL_TAU_H = 6.0
_FB_AMP = 0.35
_CORT_LOG_MAX = 0.5
_HYPO_GAIN = 0.025 * _K_CRH / (_K_ACTH * _ACTH_PER_CRH)
_ACT_GAIN = 0.35 * _K_CRH / (_K_ACTH * _ACTH_PER_CRH)


def _hpa_drive(hour: torch.Tensor) -> torch.Tensor:
    """Asymmetric 24 h HPA drive in [0, 1] (Weitzman 1971), matching the teacher."""
    rise = (_HPA_PEAK_H - _HPA_RISE_START_H) % 24.0
    fall = 24.0 - rise
    since_start = (hour - _HPA_RISE_START_H) % 24.0
    rising = since_start < rise
    rise_val = 0.5 * (1.0 - torch.cos(math.pi * since_start / rise))
    x = since_start - rise
    e_end = math.exp(-fall / _HPA_FALL_TAU_H)
    fall_val = (torch.exp(-x / _HPA_FALL_TAU_H) - e_end) / (1.0 - e_end)
    return torch.where(rising, rise_val, fall_val)


def _hour_from_time_features(time_features: torch.Tensor) -> torch.Tensor:
    sin_t = time_features[..., 0]
    cos_t = time_features[..., 1]
    theta = torch.atan2(sin_t, cos_t)
    return (theta / (2.0 * math.pi) * 24.0) % 24.0


class StressModule(MassActionModule):
    def __init__(self, embedding_dim: int, hidden_dim: int = 32):
        super().__init__(
            n_species=3,
            n_coupling=_N_COUPLING,
            n_external=_N_EXTERNAL,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            typicals=_TYPICALS,
            norm_scales=_NORM_SCALES,
            head_factories={
                _CORTISOL_IDX: lambda inp, hd: ConstantFluxHead(),
                _ACTH_IDX: lambda inp, hd: ConstantFluxHead(),
                _CRH_IDX: lambda inp, hd: ConstantFluxHead(),
            },
        )
        prod_scales = [c * t for c, t in zip(_CONS_SCALES, _TYPICALS)]
        self.prod_scale.copy_(torch.tensor(prod_scales, dtype=torch.float32))
        self.cons_scale.copy_(torch.tensor(_CONS_SCALES, dtype=torch.float32))

        self.phase_proj = nn.Linear(embedding_dim, 1)
        nn.init.normal_(self.phase_proj.weight, std=0.01)
        nn.init.zeros_(self.phase_proj.bias)
        self.log_k_crh = nn.Parameter(torch.tensor(math.log(_K_CRH)))
        self.log_k_acth = nn.Parameter(torch.tensor(math.log(_K_ACTH)))
        self.log_k_cort = nn.Parameter(torch.tensor(math.log(_K_CORT)))
        self._fb_raw = nn.Parameter(torch.tensor(math.log(math.expm1(_FB_AMP))))
        self._hypo_raw = nn.Parameter(torch.tensor(math.log(math.expm1(_HYPO_GAIN))))
        self._act_raw = nn.Parameter(torch.tensor(math.log(math.expm1(_ACT_GAIN))))
        _bh = max(8, hidden_dim // 4)
        self.cort_baseline_net = nn.Sequential(
            nn.Linear(embedding_dim, _bh), nn.Tanh(), nn.Linear(_bh, 1),
        )
        with torch.no_grad():
            self.cort_baseline_net[-1].weight.zero_()
            self.cort_baseline_net[-1].bias.zero_()

    def cort_setpoint_raw(self, embedding: torch.Tensor) -> torch.Tensor:
        return _CORT_B * torch.exp(
            _CORT_LOG_MAX * torch.tanh(self.cort_baseline_net(embedding).squeeze(-1)))

    def forward(
        self,
        state: torch.Tensor,
        coupling: torch.Tensor,
        external: torch.Tensor,
        embedding: torch.Tensor,
        time_features: torch.Tensor,
    ) -> torch.Tensor:
        raw = self.raw_state(state)
        cort = raw[..., _CORTISOL_IDX]
        acth = raw[..., _ACTH_IDX]
        crh = raw[..., _CRH_IDX]
        g = (_GLUCOSE_CENTER + _GLUCOSE_SCALE * coupling[..., 0]).clamp(min=1.0)
        sw = external[..., 0]
        act = external[..., 1]
        sleep_depth = 1.0 - sw

        hour = _hour_from_time_features(time_features)
        phase_h = 2.0 * torch.tanh(self.phase_proj(embedding).squeeze(-1))
        drive = _hpa_drive((hour - phase_h) % 24.0)
        sleep_suppression = 1.0 - _HPA_SLEEP_SUPP * sleep_depth
        cort_b = self.cort_setpoint_raw(embedding)
        fb = torch.tanh(torch.log((cort.clamp(min=0.05)) / cort_b))
        fb = torch.relu(fb)
        fb_amp = nn.functional.softplus(self._fb_raw)
        crh_target = (_CRH_B + _CRH_CIRC_AMP * (2.0 * drive - 1.0)).clamp(min=20.0)
        crh_target = crh_target * sleep_suppression * (1.0 - fb_amp * fb)

        k_crh = torch.exp(self.log_k_crh)
        k_acth = torch.exp(self.log_k_acth)
        k_cort = torch.exp(self.log_k_cort)
        hypo = nn.functional.softplus(self._hypo_raw) * torch.relu(70.0 - g)
        act_drive = nn.functional.softplus(self._act_raw) * act

        d_crh = -k_crh * (crh - crh_target) + hypo + act_drive
        d_acth = -k_acth * (acth - _ACTH_PER_CRH * crh)
        cort_target = (_CORT_PER_ACTH * acth.clamp(min=0.0)).clamp(min=0.5)
        d_cort = -k_cort * (cort - cort_target)

        rates = torch.zeros_like(state)
        rates[..., _CORTISOL_IDX] = d_cort
        rates[..., _ACTH_IDX] = d_acth
        rates[..., _CRH_IDX] = d_crh
        return rates
