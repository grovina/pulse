"""
Cardiovascular module.

Heart rate, HRV, systolic BP, diastolic BP.
Receives cortisol from Stress, temperature from Thermoregulation, and
glucose + insulin from Metabolic so postprandial sympathetic / autonomic
effects can reach HR and BP.

Architectural constraints (iter 97 — now actually enforced, see below):
  SBP > DBP  (physical — systolic is during contraction, diastolic during
              relaxation) and HRV > 0 (an RMSSD is a root-mean-square).

Iter 32 note: glucose + insulin couplings added because hr_mape was
stuck at ~0.245 across iters 27-31. With cortisol+temperature only,
postprandial HR rise was structurally unreachable from the meal axis —
no amount of glucose-side weight tuning could move HR.

Iter 36 note (reverted): a per-user baseline head fed *centered* state
(``state_centered = state - tanh_bounded_offset(embedding)``) into the
learned MLP. Empirically (docs/iter36-calibration-investigation.md):
hr_mape 0.193 → 0.264, sbp_mape 0.062 → 0.173, verifier_coupling
0.876 → 0.541. The head competed with the dynamics pathway: the model
learned to respond to glucose/insulin couplings *in the centered frame*,
but at calibration time (zero embedding ⇒ offset=0) state arrived
uncentered and the coupling response misfired.

Iter 89 note: per-patient SETPOINT dynamics — the RIGHT way to house a
per-patient resting level, distinct from the iter-36 failure. Through
iter 88 this module was pure-learned rate (a bare MLP, no setpoint term),
so each patient's resting HR/BP had to be inferred implicitly through 12h
of integrated rate. The iter-36 investigation proved that fails: HR drifts
to a constant ~+15 bpm bias (hr_mape stuck ~0.19-0.21, the sole remaining
gate failure through iter 88). The TEACHER (full_body.py) models every vital
with an explicit per-patient setpoint — dHR = -k_hr·(HR − HR0 − circ − sleep)
+ drivers — and this module mirrors that: an ADDITIVE first-order restoring
term toward a per-patient setpoint, on top of the learned MLP which carries
the autonomic DRIVERS (cortisol / temperature / glucose / insulin / activity /
circadian). The MLP still sees UNCENTERED normalized state, so nothing shifts
between training and zero-embedding calibration; only an explicit
``-k·(state − setpoint)`` force is added, with a zero-init setpoint head so
the default patient rests at NORM_CENTER and per-patient authority grows from
zero. hr_mape 0.206 → 0.158 in one iter.

Iter 97 — SBP > DBP AND HRV > 0 ARE NOW TRUE BY CONSTRUCTION (review
2026-09-04, items 3.5 and 3.7). Through iter 96 the module docstring CLAIMED
"SBP > DBP" as an architectural constraint and nothing enforced it: the four
setpoints were independent tanh offsets, so an embedding at the calibration
leash gave resting SBP 96.1 / DBP 97.1 and a 12 h rest rollout ended SBP 96.2
/ DBP 113.1, inverted for 716 of 720 min. And 2 of 8 random embeddings at
||emb|| = 3 sat on the 0-ms HRV catastrophe clamp for 738 and 175 of 1440 min.

The iter-89 setpoint form is kept exactly; what changes is the COORDINATES
it acts in:

    hr   : rate = d_hr  − k_hr ·(HR − HR_sp)                       (unchanged)
    dbp  : rate = d_dbp − k_dbp·(DBP − DBP_sp)                     (unchanged)
    hrv  : d log HRV / dt = d_hrv / HRV_c − k_hrv·(log HRV − log HRV_sp)
    pp   : d log PP  / dt = d_pp  / PP_c  − k_pp ·(log PP  − log PP_sp),   PP = SBP − DBP
    sbp  : rate = rate_dbp + PP · d log PP / dt

with ``HRV_sp = 40·exp(±1.1·tanh)`` ∈ [13, 120] ms and ``PP_sp =
40·exp(±0.7·tanh)`` ∈ [20, 80] mmHg, so ``SBP_sp = DBP_sp + PP_sp > DBP_sp``
for EVERY embedding, and HRV / pulse pressure relax in log space, where zero
is unreachable. The MLP's hrv and pp outputs are divided by the marker's
center so they keep "raw units per minute at typical" scaling — near the
setpoint the log form reduces to the iter-89 additive one. ``forward`` still
returns raw ``d(state)/dt`` for all four markers (a rate-matching signal sees
what it always saw); ``model.integrate`` steps HRV and pulse pressure
multiplicatively (``x·exp(rate·dt/x)``), which is the same first-order update
and keeps both strictly positive along the whole trajectory, so the clamp
cannot bind on either. The setpoint head's third output is the log pulse-
pressure offset, not an SBP offset — decode setpoints through
``setpoints_raw`` / ``setpoints_z`` rather than reading the head directly.
"""

import math

import torch
import torch.nn as nn

from .base import LearnedDynamicsModule
from ..types import MARKER_INDEX, MODULE_COUPLING_CHANNELS, MODULE_MARKER_INDICES, NORM_CENTER, NORM_SCALE

# Per-vital NORM_CENTER / NORM_SCALE for [hr, hrv, sbp, dbp] (iter 90: k is a true
# per-minute constant, so the normalized deviation is converted back to raw units).
_CVS_NORM_CENTER = [NORM_CENTER[i] for i in MODULE_MARKER_INDICES["cardiovascular"]]
_CVS_NORM_SCALE = [NORM_SCALE[i] for i in MODULE_MARKER_INDICES["cardiovascular"]]

# Coupling inputs from MODULE_COUPLING_CHANNELS["cardiovascular"].
_N_COUPLING = len(MODULE_COUPLING_CHANNELS["cardiovascular"])

# External inputs: activity (1) + sleep_wake (1) = 2
_N_EXTERNAL = 2

# Module state order = MODULE_MARKER_INDICES["cardiovascular"] = [hr, hrv, sbp, dbp].
_N_STATE = 4
_HR = 0
_HRV = 1
_SBP = 2
_DBP = 3

# Per-patient setpoint offset bound for the ADDITIVE setpoints (hr, dbp), z-score
# units (per NORM_SCALE): hr 70±30 = 40-100 bpm (eval 49-77), dbp 80±24 = 56-104
# mmHg (eval 61-88). Still read by SetpointSupervisionSignal for the hr/dbp rows.
_CVS_BASELINE_MAX_Z = 3.0
# Log-space setpoint bounds for the POSITIVE quantities (iter 97).
#   HRV_sp = 40·exp(±1.1·tanh) → [13, 120] ms   (RMSSD spans ~15-100 in adults)
#   PP_sp  = 40·exp(±0.7·tanh) → [20, 80] mmHg  (physiological pulse pressure)
_HRV_LOG_SP_MAX = 1.1
_PP_LOG_SP_MAX = 0.7
_HRV_CENTER = float(NORM_CENTER[MARKER_INDEX["hrv"]])
_PP_CENTER = float(NORM_CENTER[MARKER_INDEX["sbp"]] - NORM_CENTER[MARKER_INDEX["dbp"]])
# Numerical epsilon inside log() only — HRV / PP are kept strictly positive by the
# integrator, so this never binds on a trajectory; it only guards an initial state
# handed in with a zero pulse pressure.
_LOG_EPS = 1e-6

# Iter 90 — RATE-CONSTANT FRAME FIX (k now means what it says).
# k is the RAW per-minute rate constant for every vital: `rate = driver - k·(state_raw -
# setpoint_raw)`. Band [0.02, 0.80]/min (τ 1.25-50 min), init 0.30 = the teacher's k_hr
# (full_body.py, τ ≈ 3.3 min). Euler-stable: k·dt ≤ 0.8 ≪ 2. For the log-space vitals
# k multiplies a log deviation, which near the setpoint is the same relative rate.
_CVS_K_MIN = 0.02
_CVS_K_RANGE = 0.78
_CVS_K_INIT = 0.30  # teacher full_body.py PatientParams.k_hr
_CVS_LOG_K_INIT = math.log(
    ((_CVS_K_INIT - _CVS_K_MIN) / _CVS_K_RANGE) / (1.0 - (_CVS_K_INIT - _CVS_K_MIN) / _CVS_K_RANGE)
)


class CardiovascularModule(LearnedDynamicsModule):
    def __init__(self, embedding_dim: int, hidden_dim: int = 48):
        super().__init__(
            n_state=_N_STATE,
            n_coupling=_N_COUPLING,
            n_external=_N_EXTERNAL,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
        )
        # Per-patient setpoint head: one shared hidden layer, four outputs read as
        #   [hr offset (z), log HRV offset, log pulse-pressure offset, dbp offset (z)].
        # Final layer zero-init ⇒ every offset is 0 at cold start ⇒ resting HR 70,
        # HRV 40, DBP 80, PP 40 (SBP 120) for the default patient.
        _bh = max(8, hidden_dim // 4)
        self.setpoint_net = nn.Sequential(
            nn.Linear(embedding_dim, _bh), nn.Tanh(), nn.Linear(_bh, _N_STATE),
        )
        with torch.no_grad():
            self.setpoint_net[-1].weight.zero_()
            self.setpoint_net[-1].bias.zero_()
        # Per-vital restoring rate for [hr, hrv, pp, dbp] (bounded, raw per-minute).
        self.log_k = nn.Parameter(torch.full((_N_STATE,), _CVS_LOG_K_INIT))
        self.register_buffer(
            "cvs_norm_center", torch.tensor(_CVS_NORM_CENTER, dtype=torch.float32)
        )
        self.register_buffer(
            "cvs_norm_scale", torch.tensor(_CVS_NORM_SCALE, dtype=torch.float32)
        )

    # ---- setpoints ------------------------------------------------------------

    def setpoints_raw(self, embedding: torch.Tensor) -> torch.Tensor:
        """Resting ``[HR, HRV, SBP, DBP]`` in raw units for ``embedding[..., E]``.

        ``SBP = DBP + PP`` with ``PP = 40·exp(±0.7·tanh) > 0``: SBP > DBP for every
        embedding, by construction.
        """
        o = self.setpoint_net(embedding)
        hr = self.cvs_norm_center[_HR] + _CVS_BASELINE_MAX_Z * self.cvs_norm_scale[_HR] * torch.tanh(o[..., 0])
        hrv = _HRV_CENTER * torch.exp(_HRV_LOG_SP_MAX * torch.tanh(o[..., 1]))
        pp = _PP_CENTER * torch.exp(_PP_LOG_SP_MAX * torch.tanh(o[..., 2]))
        dbp = self.cvs_norm_center[_DBP] + _CVS_BASELINE_MAX_Z * self.cvs_norm_scale[_DBP] * torch.tanh(o[..., 3])
        return torch.stack([hr, hrv, dbp + pp, dbp], dim=-1)

    def setpoints_z(self, embedding: torch.Tensor) -> torch.Tensor:
        """Resting ``[HR, HRV, SBP, DBP]`` as z-scores ``(raw − NORM_CENTER)/NORM_SCALE``
        — the frame SetpointSupervisionSignal compares against."""
        return (self.setpoints_raw(embedding) - self.cvs_norm_center) / self.cvs_norm_scale

    # ---- dynamics -------------------------------------------------------------

    def forward(
        self,
        state: torch.Tensor,
        coupling: torch.Tensor,
        external: torch.Tensor,
        embedding: torch.Tensor,
        time_features: torch.Tensor,
    ) -> torch.Tensor:
        # Learned autonomic drivers (cortisol/temp/glucose/insulin/activity/
        # circadian) — the MLP still sees UNCENTERED normalized state (the
        # iter-36 frame-consistency fix). Outputs are read as
        # [d_hr (bpm/min), d_loghrv·HRV_c, d_logpp·PP_c, d_dbp (mmHg/min)].
        driver = super().forward(state, coupling, external, embedding, time_features)
        k = _CVS_K_MIN + _CVS_K_RANGE * torch.sigmoid(self.log_k)
        sp = self.setpoints_raw(embedding)

        raw = self.cvs_norm_center + self.cvs_norm_scale * state
        hr, hrv, sbp, dbp = raw[..., _HR], raw[..., _HRV], raw[..., _SBP], raw[..., _DBP]
        pp = sbp - dbp
        hr_sp, hrv_sp, sbp_sp, dbp_sp = sp[..., 0], sp[..., 1], sp[..., 2], sp[..., 3]
        pp_sp = sbp_sp - dbp_sp

        rate_hr = driver[..., _HR] - k[_HR] * (hr - hr_sp)
        rate_dbp = driver[..., _DBP] - k[_DBP] * (dbp - dbp_sp)
        # Log-space relaxations, returned as RAW rates (x · d log x / dt).
        hrv_target = hrv_sp * hr_sp / hr.clamp(min=40.0)
        dlog_hrv = driver[..., _HRV] / _HRV_CENTER - k[_HRV] * (
            torch.log(hrv.clamp(min=_LOG_EPS)) - torch.log(hrv_target))
        dlog_pp = driver[..., _SBP] / _PP_CENTER - k[_SBP] * (
            torch.log(pp.clamp(min=_LOG_EPS)) - torch.log(pp_sp))
        rate_hrv = hrv * dlog_hrv
        rate_pp = pp * dlog_pp
        rate_sbp = rate_dbp + rate_pp
        return torch.stack([rate_hr, rate_hrv, rate_sbp, rate_dbp], dim=-1)
