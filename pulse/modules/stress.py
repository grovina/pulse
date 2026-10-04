"""
Stress / HPA Axis module.

The cascade is CRH → ACTH → cortisol. Circadian drive, sleep suppression,
hypoglycaemia and activity enter at CRH; ACTH tracks CRH; cortisol tracks ACTH.
Cortisol's negative feedback on CRH is one-sided saturating about the
patient's basal: high cortisol suppresses CRH; the nocturnal nadir is
sleep's. A rectifier at 12 µg/dL would leave the whole overnight range
inert.

The clock is not a person. The drive is one function of the hour for every
embedding. The teacher's ``randomize_params`` varies the circadian amplitude, the
basals and the gains, but never ``hpa_rise_start_h`` / ``hpa_peak_h`` /
``hpa_fall_tau_h``, so no patient has a phase for a head to recover. Through
iter 109 a ``phase_proj`` head shifted the drive by up to ±2 h per embedding
anyway: no ground truth, and a +1 h shift moves the five taped vitals by 0.07, so
the only dense loss barely sees it either. That is free authority, iter 109's
failure in another organ (the glucagon-basal head took the prior person from 70
to 98 pg/mL while liver, ketones and glucose were unchanged when it was zeroed,
because every downstream gate is normalized to the person's OWN level). A phase
that varies by person comes back as a state that person's sleep timing entrains,
which is evidence; it does not come back as an offset head.
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
    external_inputs = ("sleep_wake", "activity")

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

    def constants(self, embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "cort_b": self.cort_setpoint_raw(embedding),
            "fb_amp": nn.functional.softplus(self._fb_raw),
            "k_crh": torch.exp(self.log_k_crh),
            "k_acth": torch.exp(self.log_k_acth),
            "k_cort": torch.exp(self.log_k_cort),
            "hypo_gain": nn.functional.softplus(self._hypo_raw),
            "act_gain": nn.functional.softplus(self._act_raw),
        }

    def drives(
        self,
        external: torch.Tensor,
        coupling: torch.Tensor,
        time_features: torch.Tensor,
        const: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """The circadian CRH target under sleep suppression, and the activity drive."""
        sw = external[..., 0]
        act = external[..., 1]
        sleep_depth = 1.0 - sw
        hour = _hour_from_time_features(time_features)
        drive = _hpa_drive(hour)
        sleep_suppression = 1.0 - _HPA_SLEEP_SUPP * sleep_depth
        crh_target = (_CRH_B + _CRH_CIRC_AMP * (2.0 * drive - 1.0)).clamp(min=20.0)
        return {
            "crh_target_open_loop": crh_target * sleep_suppression,
            "act_drive": const["act_gain"] * act,
        }

    def step(
        self,
        state: torch.Tensor,
        coupling: torch.Tensor,
        const: dict[str, torch.Tensor],
        drv: dict[str, torch.Tensor],
        raw: dict[object, torch.Tensor],
    ) -> torch.Tensor:
        raw_s = self.raw_state(state)
        cort = raw_s[..., _CORTISOL_IDX]
        acth = raw_s[..., _ACTH_IDX]
        crh = raw_s[..., _CRH_IDX]
        g = (_GLUCOSE_CENTER + _GLUCOSE_SCALE * coupling[..., 0]).clamp(min=1.0)

        fb = torch.tanh(torch.log((cort.clamp(min=0.05)) / const["cort_b"]))
        fb = torch.relu(fb)
        crh_target = drv["crh_target_open_loop"] * (1.0 - const["fb_amp"] * fb)
        hypo = const["hypo_gain"] * torch.relu(70.0 - g)

        d_crh = -const["k_crh"] * (crh - crh_target) + hypo + drv["act_drive"]
        d_acth = -const["k_acth"] * (acth - _ACTH_PER_CRH * crh)
        cort_target = (_CORT_PER_ACTH * acth.clamp(min=0.0)).clamp(min=0.5)
        d_cort = -const["k_cort"] * (cort - cort_target)
        return torch.stack([d_cort, d_acth, d_crh], dim=-1)
