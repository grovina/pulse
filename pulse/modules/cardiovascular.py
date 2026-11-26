"""
Cardiovascular module.

Heart rate, HRV, systolic BP, diastolic BP.
Receives cortisol from Stress, temperature from Thermoregulation,
glucose + insulin from Metabolic, and gut glucose appearance. Meal HR is
driven by appearance (``meal_hr_gain * Ra / (Ra + K_ra)``), the same
stimulus the teacher uses — glucose and insulin lag absorption, so they
cannot time the +5–10 bpm rise.

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
the autonomic DRIVERS (cortisol / temperature / glucose / insulin / appearance /
activity / circadian). Meal HR is a structural term on appearance, matching the
teacher; the MLP still sees UNCENTERED normalized state, so nothing shifts
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

PLAN A1 (2026-10-04) — HR AND DBP SETPOINTS JOIN HRV AND PULSE PRESSURE IN LOG
SPACE. The teacher draws every resting vital lognormally (``vary``: val·exp(σ·z);
HR0 σ0.22, SBP0 0.14, DBP0 0.15, HRV0 0.4), so ``PatientParams()`` is the population
MEDIAN and the population mean sits above it: over 20 000 draws mean/median is
1.0232 for HR0, 1.0112 SBP0, 1.0105 DBP0, 1.0850 HRV0. Through iter 109 HR and DBP
were decoded ADDITIVELY (iter 89: ``center + 3z·tanh(head)``; iter 97 moved only HRV
and PP to log space), and for a zero-mean embedding population that gives
E[setpoint] = center exactly: population mean = median, which contradicts the
teacher and cannot be right at both points. Measured on a random zero-bias head (odd
in the embedding) over 20 000 antithetic N(0, I) codes
(tests/test_a1_log_setpoints.py), E[HR_sp] − HR_sp(0) was +0.000 bpm; it is now
+1.656 (the teacher's own: +1.757). HR and DBP are ``center·exp(L·tanh(head))`` like
HRV and PP, so E[sp] > sp(0) by Jensen with no loss term, and SBP = DBP + PP, now a
sum of two log-decoded terms, inherits the shift (E[SBP_sp] − SBP_sp(0): +1.66 →
+2.34 mmHg on that head). The spans keep the floors of the additive ones they
replace and reach further up, where the teacher's tail is: the additive 100 bpm /
104 mmHg ceilings were out of reach for 5.3 % / 4.2 % of sampled patients; the log
ceilings are out of reach for 0.5 % / 0.9 %.

Unchanged: the head layout, ``setpoints_raw``'s [HR, HRV, SBP, DBP] return, rest at
NORM_CENTER for a fresh model (tanh(0) = 0 ⇒ center·exp(0) = center, exact in
float32) and ``setpoints_z`` = (raw − center)/scale, which is why
SetpointSupervisionSignal needed no change. A checkpoint through iter 109 still
loads (same shapes) but was fit under the additive decode: near a zero head the HR /
DBP offsets now act 1.31× / 1.20× as strongly (70·0.56 vs 3·10 bpm, 80·0.36 vs 3·8
mmHg per unit of head output), so it needs the retrain PLAN.md §4 already schedules.
RR and SpO₂: see respiratory.py.
"""

import math

import torch
import torch.nn as nn

from .base import LearnedDynamicsModule
from ..types import MARKER_INDEX, MODULE_COUPLING_CHANNELS, MODULE_MARKER_INDICES, NORM_CENTER, NORM_SCALE

# Per-vital NORM_CENTER / NORM_SCALE for [hr, hrv, sbp, dbp] (iter 90: k is a true
# per-minute constant, so the normalized deviation is converted back to raw units).
_CVS_NORM_CENTER = [NORM_CENTER[i] for i in MODULE_MARKER_INDICES["cardiovascular"][:4]]
_CVS_NORM_SCALE = [NORM_SCALE[i] for i in MODULE_MARKER_INDICES["cardiovascular"][:4]]
_SLOW_CENTER = float(NORM_CENTER[MARKER_INDEX["hr_slow"]])
_SLOW_SCALE = float(NORM_SCALE[MARKER_INDEX["hr_slow"]])

# Coupling inputs from MODULE_COUPLING_CHANNELS["cardiovascular"].
_N_COUPLING = len(MODULE_COUPLING_CHANNELS["cardiovascular"])

# External inputs: activity (1) + sleep_wake (1) = 2
_N_EXTERNAL = 2

# Module state order = MODULE_MARKER_INDICES["cardiovascular"] = [hr, hrv, sbp, dbp, hr_slow].
# The four vitals are what the MLP reads and writes; ``hr_slow`` is a fifth state with its own
# law below, kept out of the MLP so a checkpoint without it loads every trained weight.
_N_STATE = 4
_N_OWN = 5
_HR = 0
_HRV = 1
_SBP = 2
_DBP = 3
_SLOW = 4

# Log-space setpoint bounds. All four head outputs decode as ``center·exp(L·tanh(head))``:
# strictly positive, a zero head is exactly the center, and a zero-mean embedding population
# has E[sp] > sp(0) by Jensen — the teacher's lognormal population, with no loss term.
#   HR_sp  = 70·exp(±0.56·tanh) → [40.0, 122.5] bpm   (A1; replaces 70±30 additive = 40-100,
#            eval 49-77; L = ln(70/40) = 0.5596 rounded up, so the 40 bpm floor is kept)
#   DBP_sp = 80·exp(±0.36·tanh) → [55.8, 114.7] mmHg  (A1; replaces 80±24 additive = 56-104,
#            eval 61-88; L = ln(80/56) = 0.3567 rounded up, so the 56 mmHg floor is kept)
#   HRV_sp = 40·exp(±1.1·tanh) → [13, 120] ms   (iter 97; RMSSD spans ~15-100 in adults)
#   PP_sp  = 40·exp(±0.7·tanh) → [20, 80] mmHg  (iter 97; physiological pulse pressure)
# Equal-ratio spans put the ceiling at center²/floor, which is why the HR / DBP ceilings rose
# (100 → 122.5, 104 → 114.7) while the floors held.
_HR_LOG_SP_MAX = 0.56
_DBP_LOG_SP_MAX = 0.36
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

# The slow post-exercise state ``hr_slow``: a leaky integral of the activity above rest,
#   d s / dt = (max(activity - _SLOW_REST, 0) - s) / tau,
# in the model's activity scale. Heart rate stays above its resting level for tens of minutes
# after a bout (sugimoto-2020, gahreman-2016) and the fast driver is back at rest within
# minutes, so this is the state the tail lives in. tau is a learned population number in
# [5, 240] min, initialised at 30. It reaches the heart rate through ``hr_slow_gain`` (bpm
# per unit of the state), which starts at ZERO: until a run trains it, heart rate is what it
# was. ``_SLOW_REST`` is the activity of ordinary waking (the default head's floor is 0.02).
_SLOW_REST = 0.1
_SLOW_TAU_MIN = 5.0
_SLOW_TAU_RANGE = 235.0
_SLOW_TAU_INIT = 30.0
_SLOW_LOG_TAU_INIT = math.log(
    ((_SLOW_TAU_INIT - _SLOW_TAU_MIN) / _SLOW_TAU_RANGE)
    / (1.0 - (_SLOW_TAU_INIT - _SLOW_TAU_MIN) / _SLOW_TAU_RANGE)
)

# Teacher PatientParams.meal_hr_gain / K_ra_norm. Appearance is mg/dL/min
# in the 70 kg reference space (same units as the gut glucose channel).
_MEAL_HR_GAIN = 3.0
_K_RA_NORM = 1.06
_APPEARANCE_COUPLING_IDX = MODULE_COUPLING_CHANNELS["cardiovascular"].index(
    "gut.glucose_appearance"
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
        # The MLP was built for the four vitals; the module integrates five states.
        self.n_state = _N_OWN
        # What fitness and the slow state add to heart rate, in bpm. Both start at zero, so
        # the module is exactly the one a checkpoint without them was trained as.
        #   hr_fit_gain  bpm per unit of activity per fitness SD BELOW the median (a lower
        #                VO2max needs a higher heart rate at the same absolute work);
        #   hr_slow_gain bpm per unit of the slow state.
        self.hr_fit_gain = nn.Parameter(torch.zeros(()))
        self.hr_slow_gain = nn.Parameter(torch.zeros(()))
        self.slow_tau_raw = nn.Parameter(torch.tensor(_SLOW_LOG_TAU_INIT))
        # Per-patient setpoint head: one shared hidden layer, four outputs read as
        #   [log HR offset, log HRV offset, log pulse-pressure offset, log DBP offset].
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

    def head_state_columns(self) -> list[int]:
        return list(range(_N_STATE))

    def bind_body(self, const, body):
        """The person's fitness code (a coordinate of the code, in SD), held once for the body."""
        if body is not None and "fitness_z" in body:
            return {**const, "fitness_z": body["fitness_z"]}
        return const

    # ---- setpoints ------------------------------------------------------------

    def setpoints_raw(self, embedding: torch.Tensor) -> torch.Tensor:
        """Resting ``[HR, HRV, SBP, DBP]`` in raw units for ``embedding[..., E]``.

        Every head output is ``center·exp(L·tanh(·))``, so HR, HRV, PP and DBP are
        strictly positive and ``SBP = DBP + PP`` with ``PP = 40·exp(±0.7·tanh) > 0``:
        SBP > DBP > 0 for every embedding, by construction.
        """
        o = self.setpoint_net(embedding)
        hr = self.cvs_norm_center[_HR] * torch.exp(_HR_LOG_SP_MAX * torch.tanh(o[..., 0]))
        hrv = _HRV_CENTER * torch.exp(_HRV_LOG_SP_MAX * torch.tanh(o[..., 1]))
        pp = _PP_CENTER * torch.exp(_PP_LOG_SP_MAX * torch.tanh(o[..., 2]))
        dbp = self.cvs_norm_center[_DBP] * torch.exp(_DBP_LOG_SP_MAX * torch.tanh(o[..., 3]))
        return torch.stack([hr, hrv, dbp + pp, dbp], dim=-1)

    def setpoints_z(self, embedding: torch.Tensor) -> torch.Tensor:
        """Resting ``[HR, HRV, SBP, DBP]`` as z-scores ``(raw − NORM_CENTER)/NORM_SCALE``
        — the frame SetpointSupervisionSignal compares against."""
        return (self.setpoints_raw(embedding) - self.cvs_norm_center) / self.cvs_norm_scale

    # ---- dynamics (see base.PhysiologyModule) ------------------------------------

    def constants(self, embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        k = _CVS_K_MIN + _CVS_K_RANGE * torch.sigmoid(self.log_k)
        sp = self.setpoints_raw(embedding)
        hr_sp, hrv_sp, sbp_sp, dbp_sp = sp[..., 0], sp[..., 1], sp[..., 2], sp[..., 3]
        pp_sp = sbp_sp - dbp_sp
        return {
            "k_hr": k[_HR], "k_hrv": k[_HRV], "k_pp": k[_SBP], "k_dbp": k[_DBP],
            "hr_sp": hr_sp, "hrv_sp": hrv_sp, "dbp_sp": dbp_sp,
            "log_pp_sp": torch.log(pp_sp),
            "slow_tau": _SLOW_TAU_MIN + _SLOW_TAU_RANGE * torch.sigmoid(self.slow_tau_raw),
            "hr_fit_gain": self.hr_fit_gain, "hr_slow_gain": self.hr_slow_gain,
        }

    def drives(
        self,
        external: torch.Tensor,
        coupling: torch.Tensor,
        time_features: torch.Tensor,
        const: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        ra = coupling[..., _APPEARANCE_COUPLING_IDX].clamp(min=0.0)
        return {"meal_hr": _MEAL_HR_GAIN * ra / (ra + _K_RA_NORM), "activity": external[..., 0]}

    def step(
        self,
        state: torch.Tensor,
        coupling: torch.Tensor,
        const: dict[str, torch.Tensor],
        drv: dict[str, torch.Tensor],
        raw: dict[object, torch.Tensor],
    ) -> torch.Tensor:
        # Learned autonomic drivers (cortisol/temp/glucose/insulin/activity/
        # circadian) — the MLP still sees UNCENTERED normalized state (the
        # iter-36 frame-consistency fix). Outputs are read as
        # [d_hr (bpm/min), d_loghrv·HRV_c, d_logpp·PP_c, d_dbp (mmHg/min)].
        driver = raw["driver"]
        raw_s = self.cvs_norm_center + self.cvs_norm_scale * state[..., :_N_STATE]
        hr, hrv, sbp, dbp = raw_s[..., _HR], raw_s[..., _HRV], raw_s[..., _SBP], raw_s[..., _DBP]
        pp = sbp - dbp
        hr_sp = const["hr_sp"]
        act = drv["activity"]
        slow = _SLOW_CENTER + _SLOW_SCALE * state[..., _SLOW]

        # Where the heart rate settles moves by what fitness and the slow state add (exactly
        # zero while both gains are zero). A lower fitness code needs more beats at the same
        # work; the slow state holds the tail after a bout.
        shift = const["hr_slow_gain"] * slow
        if "fitness_z" in const:
            shift = shift - const["hr_fit_gain"] * const["fitness_z"] * act
        rate_hr = driver[..., _HR] - const["k_hr"] * ((hr - hr_sp) - shift) + drv["meal_hr"]
        rate_slow = (torch.relu(act - _SLOW_REST) - slow) / const["slow_tau"]
        rate_dbp = driver[..., _DBP] - const["k_dbp"] * (dbp - const["dbp_sp"])
        # Log-space relaxations, returned as RAW rates (x · d log x / dt).
        hrv_target = const["hrv_sp"] * hr_sp / hr.clamp(min=40.0)
        dlog_hrv = driver[..., _HRV] / _HRV_CENTER - const["k_hrv"] * (
            torch.log(hrv.clamp(min=_LOG_EPS)) - torch.log(hrv_target))
        dlog_pp = driver[..., _SBP] / _PP_CENTER - const["k_pp"] * (
            torch.log(pp.clamp(min=_LOG_EPS)) - const["log_pp_sp"])
        rate_hrv = hrv * dlog_hrv
        rate_pp = pp * dlog_pp
        rate_sbp = rate_dbp + rate_pp
        return torch.stack([rate_hr, rate_hrv, rate_sbp, rate_dbp, rate_slow], dim=-1)
