"""
Appetite & Satiety module.

Ghrelin, leptin, GLP-1, and the 4 h insulin lag that leptin reads.
Receives insulin, duodenal delivery (fat/protein/carb), fat mass, and
systemic glucose appearance.

Ghrelin is suppressed by duodenal nutrient (Williams 2003) and by a signed
insulin term centred at basal. Leptin clears in ~30 min toward a target set
by fat mass, the nocturnal circadian, and lagged insulin (Saad 1998). GLP-1
peaks on dose-linear glucose appearance. insulin_slow is a first-order lag
of plasma insulin (τ = 4 h).
"""

import math

import torch
import torch.nn as nn

from .base import BasalPlusGatedPeakHead, ConstantFluxHead, MassActionModule
from ..types import (
    MARKER_INDEX, MODULE_COUPLING_CHANNELS, MODULE_MARKER_INDICES, NORM_CENTER, NORM_SCALE,
)

_N_COUPLING = len(MODULE_COUPLING_CHANNELS["appetite"])
_N_EXTERNAL = 1

_TYPICALS = [NORM_CENTER[i] for i in MODULE_MARKER_INDICES["appetite"]]
_NORM_SCALES = [NORM_SCALE[i] for i in MODULE_MARKER_INDICES["appetite"]]
# Ghrelin τ ~ 50 min; leptin plasma k = 0.025 (~30 min); GLP-1 fast; insulin_slow 4 h.
_CONS_SCALES = [0.02, 0.025, 0.2, 1.0 / 240.0]

_GHRELIN_IDX = 0
_LEPTIN_IDX = 1
_GLP1_IDX = 2
_INSULIN_SLOW_IDX = 3

# Coupling: insulin, duodenal.fat, duodenal.protein, duodenal.carb, fat_mass, gut.glucose
_INS_COUPLING = 0
_DUO_FAT = 1
_DUO_PROT = 2
_DUO_CARB = 3
_FAT_MASS_COUPLING = 4
_GUT_GLUCOSE = 5

_INSULIN_CENTER = NORM_CENTER[MARKER_INDEX["insulin"]]
_INSULIN_SCALE = NORM_SCALE[MARKER_INDEX["insulin"]]
_IB_LOG_MAX = 0.9
_FAT_CENTER = NORM_CENTER[MARKER_INDEX["fat_mass"]]
_FAT_SCALE = NORM_SCALE[MARKER_INDEX["fat_mass"]]
_GLP1_CENTER = NORM_CENTER[MARKER_INDEX["glp1"]]

_K_MEAL_GHR = 0.9
_GHR_INS_N = 1.0
_GHR_SUPP_MAX = 0.60
_GHR_ANTIC_AMP = 0.55
_LEP_CIRC_AMP = 2.0
_LEP_INS_GAIN = 2.5
_LEP_FAT_GAIN = 1.0  # target *= (Fat / Fat_b)^gain at gain=1 is linear in Fat/Fat_b
_K_INS_SLOW = 1.0 / 240.0
_K_LEP = 0.025
_K_GHR = 0.02
_GLP1_DUO_GAIN = 0.15  # small L-cell contribution from duodenal fat/protein


def _hour_from_time_features(time_features: torch.Tensor) -> torch.Tensor:
    """Hour of day in [0, 24) from ``[sin θ, cos θ, ...]``."""
    sin_t = time_features[..., 0]
    cos_t = time_features[..., 1]
    theta = torch.atan2(sin_t, cos_t)
    return (theta / (2.0 * math.pi) * 24.0) % 24.0


def _anticipation_drive(hour: torch.Tensor) -> torch.Tensor:
    """Entrained pre-meal drive in [0, 1] at the habitual 9 / 13 / 20 meal hours."""
    best = torch.zeros_like(hour)
    ramp_h, decay_h = 2.0, 1.0
    for hm in (9.0, 13.0, 20.0):
        dt = (hour - hm + 12.0) % 24.0 - 12.0
        ramp = 0.5 * (1.0 - torch.cos(math.pi * (dt + ramp_h) / ramp_h))
        decay = 0.5 * (1.0 + torch.cos(math.pi * dt / decay_h))
        v = torch.where(
            (dt >= -ramp_h) & (dt <= 0.0), ramp,
            torch.where((dt > 0.0) & (dt <= decay_h), decay, torch.zeros_like(dt)),
        )
        best = torch.maximum(best, v)
    return best


class AppetiteModule(MassActionModule):
    def __init__(self, embedding_dim: int, hidden_dim: int = 32):
        super().__init__(
            n_species=4,
            n_coupling=_N_COUPLING,
            n_external=_N_EXTERNAL,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            typicals=_TYPICALS,
            norm_scales=_NORM_SCALES,
            head_factories={
                _GHRELIN_IDX: lambda inp, hd: ConstantFluxHead(),
                _LEPTIN_IDX: lambda inp, hd: ConstantFluxHead(),
                _INSULIN_SLOW_IDX: lambda inp, hd: ConstantFluxHead(),
                _GLP1_IDX: lambda inp, hd: BasalPlusGatedPeakHead(
                    inp, hd, stimulus_idx=4 + _GUT_GLUCOSE, gate_dir=1,
                    init_thresh=0.1, init_log_temp=-1.6,
                ),
            },
        )
        prod_scales = [c * t for c, t in zip(_CONS_SCALES, _TYPICALS)]
        self.prod_scale.copy_(torch.tensor(prod_scales, dtype=torch.float32))
        self.cons_scale.copy_(torch.tensor(_CONS_SCALES, dtype=torch.float32))
        _bh = max(8, hidden_dim // 4)
        self.insulin_baseline_net = nn.Sequential(
            nn.Linear(embedding_dim, _bh), nn.Tanh(), nn.Linear(_bh, 1),
        )
        with torch.no_grad():
            self.insulin_baseline_net[-1].weight.zero_()
            self.insulin_baseline_net[-1].bias.zero_()

    def insulin_setpoint_raw(self, embedding: torch.Tensor) -> torch.Tensor:
        return _INSULIN_CENTER * torch.exp(
            _IB_LOG_MAX * torch.tanh(self.insulin_baseline_net(embedding).squeeze(-1)))

    def forward(
        self,
        state: torch.Tensor,
        coupling: torch.Tensor,
        external: torch.Tensor,
        embedding: torch.Tensor,
        time_features: torch.Tensor,
    ) -> torch.Tensor:
        raw = self.raw_state(state)
        ghr, lep, glp1, ins_slow = (
            raw[..., _GHRELIN_IDX], raw[..., _LEPTIN_IDX],
            raw[..., _GLP1_IDX], raw[..., _INSULIN_SLOW_IDX],
        )
        ins = (_INSULIN_CENTER + _INSULIN_SCALE * coupling[..., _INS_COUPLING]).clamp(min=1e-3)
        ib = self.insulin_setpoint_raw(embedding).clamp(min=1e-3)
        duo = (
            coupling[..., _DUO_FAT] + coupling[..., _DUO_PROT] + coupling[..., _DUO_CARB]
        ).clamp(min=0.0)
        fat = (_FAT_CENTER + _FAT_SCALE * coupling[..., _FAT_MASS_COUPLING]).clamp(min=1.0)
        ra = coupling[..., _GUT_GLUCOSE].clamp(min=0.0)

        # --- ghrelin: duodenal nutrient ∪ signed insulin, plus anticipatory rise ---
        ra_norm = duo / (duo + _K_MEAL_GHR)
        ins_ratio_n = (ins / ib) ** _GHR_INS_N
        insulin_supp = (ins_ratio_n - 1.0) / (ins_ratio_n + 1.0)
        meal_supp = 1.0 - (1.0 - insulin_supp) * (1.0 - ra_norm)
        hour = _hour_from_time_features(time_features)
        antic = _anticipation_drive(hour)
        ghr_prod = (
            _TYPICALS[_GHRELIN_IDX] * _K_GHR
            * (1.0 + _GHR_ANTIC_AMP * antic)
            * (1.0 - _GHR_SUPP_MAX * meal_supp)
        )
        d_ghr = ghr_prod - _K_GHR * ghr

        # --- insulin_slow: 4 h lag of plasma insulin ---
        d_slow = -_K_INS_SLOW * (ins_slow - ins)

        # --- leptin: fat mass + circadian + lagged insulin ---
        circ = _LEP_CIRC_AMP * torch.cos(2.0 * math.pi * (hour - 2.0) / 24.0)
        lep_ins = _LEP_INS_GAIN * (ins_slow / ib - 1.0)
        lep_target = (
            _TYPICALS[_LEPTIN_IDX] * (fat / _FAT_CENTER) ** _LEP_FAT_GAIN
            + circ + lep_ins
        )
        d_lep = -_K_LEP * (lep - lep_target)

        # --- GLP-1: learned meal-gated peak on appearance, plus a small duodenal term ---
        base_rate = super().forward(state, coupling, external, embedding, time_features)
        d_glp1 = (
            base_rate[..., _GLP1_IDX]
            + _GLP1_DUO_GAIN * (coupling[..., _DUO_FAT] + coupling[..., _DUO_PROT])
        )

        rates = torch.zeros_like(state)
        rates[..., _GHRELIN_IDX] = d_ghr
        rates[..., _LEPTIN_IDX] = d_lep
        rates[..., _GLP1_IDX] = d_glp1
        rates[..., _INSULIN_SLOW_IDX] = d_slow
        return rates
