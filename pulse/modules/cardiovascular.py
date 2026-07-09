"""
Cardiovascular module.

Heart rate, HRV, systolic BP, diastolic BP.
Receives cortisol from Stress, temperature from Thermoregulation, and
glucose + insulin from Metabolic so postprandial sympathetic / autonomic
effects can reach HR and BP.

Architectural constraint: SBP > DBP (physical — systolic is during
contraction, diastolic during relaxation).

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
gate failure through iter 88). This is exactly the gap the metabolic module
closed for glucose across iters 81-88: fully-learned dynamics cannot hold a
per-patient baseline, so the baseline must live *in the physics* as an
explicit setpoint. The TEACHER (full_body.py) already models every vital
this way — dHR = -k_hr·(HR − HR0 − circ − sleep) + cortisol·drive +
activity·gain, with per-patient HR0/HRV0/SBP0/DBP0. This module now mirrors
that structure: an ADDITIVE first-order restoring term toward a per-patient
setpoint, on top of the untouched learned MLP which carries the autonomic
DRIVERS (cortisol / temperature / glucose / insulin / activity / circadian).

Why this avoids the iter-36 failure: the MLP still sees UNCENTERED normalized
state, so the coupling response is learned and evaluated in the same frame —
nothing shifts between training and zero-embedding calibration. Only an
explicit ``-k·(state − setpoint)`` force is added to the rate (setpoint from
a zero-init head ⇒ 0 offset at cold start ⇒ resting level = NORM_CENTER for
the default patient). This is the additive-restoring form of glucose's
``-(Sg+X)·(G − Gb_emb) + Ra·appearance`` (metabolic.py), not the iter-36
input-centering. The equilibrium is the per-patient setpoint for ANY k>0,
true by construction; per-patient authority grows from zero during training.
"""

import math

import torch
import torch.nn as nn

from .base import LearnedDynamicsModule
from ..types import MARKER_INDEX, MODULE_MARKER_INDICES, NORM_SCALE

# Iter 90: per-vital NORM_SCALE for [hr, hrv, sbp, dbp], used to convert the module's
# normalized deviation back to raw units so k is a true per-minute rate constant.
_CVS_NORM_SCALE = [NORM_SCALE[i] for i in MODULE_MARKER_INDICES["cardiovascular"]]

# Coupling inputs: cortisol (1) + temperature (1) + glucose (1) + insulin (1) = 4
_N_COUPLING = 4

# External inputs: activity (1) + sleep_wake (1) = 2
_N_EXTERNAL = 2

# Module state order = MODULE_MARKER_INDICES["cardiovascular"] = [hr, hrv, sbp, dbp].
_N_STATE = 4

# Per-patient setpoint offset bound, z-score units (per NORM_SCALE). The learned
# tanh offset moves each vital's resting equilibrium by ±_CVS_BASELINE_MAX_Z·scale
# around NORM_CENTER. 3.0 covers the benchmark's per-patient spans with margin
# (NORM_CENTER ± 3·NORM_SCALE): hr 70±30 = 40-100 bpm (eval 49-77), hrv 40±45,
# sbp 120±30 = 90-150 mmHg (eval 94-133), dbp 80±24 = 56-104 mmHg (eval 61-88).
_CVS_BASELINE_MAX_Z = 3.0

# Iter 90 — RATE-CONSTANT FRAME FIX (k now means what it says).
#
# Through iter 89 k multiplied the NORMALIZED deviation while the rate was applied to RAW
# state, so the true per-minute constant was k / NORM_SCALE_i. That convention was applied
# knowingly here (τ = NORM_SCALE/k), but it made k's units differ per vital and hid the same
# 30× error that silently crippled glucose's Sg (see metabolic.py _SG_* block). k is now the
# RAW per-minute rate constant for every vital: `rate = driver - k·(state_raw - setpoint_raw)`.
#
# Band [0.02, 0.80]/min (τ 1.25-50 min), init 0.30 = the teacher's k_hr (full_body.py:563,
# τ ≈ 3.3 min). For reference the iter-89 model trained to a raw k of ~0.12-0.27 across the
# four vitals — already near the teacher — so this reframe is a small, safe correction that
# mainly makes the parameter honest. Euler-stable: k·dt ≤ 0.8 ≪ 2.
_CVS_K_MIN = 0.02
_CVS_K_RANGE = 0.78
_CVS_K_INIT = 0.30  # teacher full_body.py PatientParams.k_hr
# sigmoid(log_k) init so k == _CVS_K_INIT.
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
        # Per-patient resting-setpoint offsets for [hr, hrv, sbp, dbp], one head
        # emitting all four (shared hidden layer; the teacher varies HR0/HRV0/
        # SBP0/DBP0 independently, so the final layer separates them). Final
        # layer zero-init ⇒ setpoint_emb = 0 for every embedding at cold start ⇒
        # resting level = NORM_CENTER, and per-patient authority grows during
        # training (final-layer weight gets non-zero grad at step 1). Mirrors
        # metabolic.glucose_baseline_net / ra_baseline_net.
        _bh = max(8, hidden_dim // 4)
        self.setpoint_net = nn.Sequential(
            nn.Linear(embedding_dim, _bh), nn.Tanh(), nn.Linear(_bh, _N_STATE),
        )
        with torch.no_grad():
            self.setpoint_net[-1].weight.zero_()
            self.setpoint_net[-1].bias.zero_()
        # Per-vital restoring rate (bounded, raw per-minute; see _CVS_K_* above).
        self.log_k = nn.Parameter(torch.full((_N_STATE,), _CVS_LOG_K_INIT))
        # Converts the normalized deviation back to raw units (bpm / ms / mmHg).
        self.register_buffer(
            "cvs_norm_scale", torch.tensor(_CVS_NORM_SCALE, dtype=torch.float32)
        )

    def forward(
        self,
        state: torch.Tensor,
        coupling: torch.Tensor,
        external: torch.Tensor,
        embedding: torch.Tensor,
        time_features: torch.Tensor,
    ) -> torch.Tensor:
        # Learned autonomic drivers (cortisol/temp/glucose/insulin/activity/
        # circadian) — the MLP is unchanged and still sees UNCENTERED normalized
        # state (the iter-36 frame-consistency fix).
        driver = super().forward(state, coupling, external, embedding, time_features)
        # Per-patient setpoint offset (z-units), 0 at cold start.
        setpoint = _CVS_BASELINE_MAX_Z * torch.tanh(self.setpoint_net(embedding))
        k = _CVS_K_MIN + _CVS_K_RANGE * torch.sigmoid(self.log_k)
        # ADDITIVE first-order restoring toward the per-patient setpoint plus the
        # learned driver. Iter 90: scaling the normalized deviation by NORM_SCALE makes
        # this `rate = driver - k·(state_raw - setpoint_raw)` with k a true per-minute
        # constant. Equilibrium is `setpoint` once the driver averages to ~0 (which
        # DefaultBaselineSignal pins for the default patient). Mirrors glucose's
        # -(Sg + Si·Xa)·(G_raw - Gb_raw) + Ra·appearance.
        return driver - k * (state - setpoint) * self.cvs_norm_scale
