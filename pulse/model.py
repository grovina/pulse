"""
Modular physiology network.

The body is modeled as a network of seven interconnected physiological
subsystems. Each module owns a slice of the state vector and computes
rates of change for its markers. Modules interact through explicit
coupling variables that reflect anatomical connectivity.

Architecture constraints:
  - Chemical species (Metabolic, Appetite, Stress) use mass-action kinetics
  - Vital signs (Cardiovascular, Thermoreg, Respiratory) have learned dynamics
  - The Gut module is a learned absorption kernel, not an ODE

External inputs (sleep/wake, activity) are optional. When missing, the
network substitutes a learned default conditioned on the person embedding and
the time of day (``default_external_inputs``); asleep implies rest (activity 0)
by construction, and 0 means rest everywhere.
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from .types import (
    STATE_DIM, EMBEDDING_DIM, GUT_OUTPUT_DIM, TIME_FEATURES_DIM, DUODENAL_DIM,
    MARKERS, MARKER_INDEX, MODULE_MARKER_INDICES, MODULE_COUPLING_CHANNELS,
    GUT_CHANNEL_INDEX, DUODENAL_CHANNEL_INDEX,
    NORM_CENTER, NORM_SCALE,
)
from .modules import (
    GutModule, MetabolicModule, AppetiteModule, StressModule,
    CardiovascularModule, ThermoregModule, RespiratoryModule,
    HepatobiliaryModule, DuodenalDeliveryKernel,
)
from .modules.base import PhysiologyModule, compute_time_features
from .modules.gut import MealEvent

# Iter 97: markers the integrator steps MULTIPLICATIVELY, `x·exp(rate·dt/x)`, so they
# stay strictly positive along every trajectory (review 2026-09-04, items 3.5/3.7). The
# cardiovascular module writes HRV's and pulse pressure's dynamics in log space and
# returns them as raw rates `x·dlog x/dt`; this step is the same first-order update as
# `x + rate·dt` and cannot cross zero. Pulse pressure is SBP − DBP, so SBP is rebuilt as
# DBP + PP after the step, which is what makes SBP > DBP hold by construction rather
# than by docstring.
_HRV_IDX = MARKER_INDEX["hrv"]
_SBP_IDX = MARKER_INDEX["sbp"]
_DBP_IDX = MARKER_INDEX["dbp"]
_SPO2_IDX = MARKER_INDEX["spo2"]
_SPO2_LO = 70.0
_SPO2_HI = 100.0
_SPO2_EPS = 1e-4

# Module evaluation order. Every marker belongs to exactly one module, so the
# modules' rates concatenated in this order are a permutation of the state.
_MODULE_ORDER: tuple[str, ...] = (
    "metabolic", "appetite", "stress", "cardiovascular", "thermoreg", "respiratory",
    "hepatobiliary",
)
_MODULE_ORDER_INDEX: tuple[int, ...] = tuple(
    i for name in _MODULE_ORDER for i in MODULE_MARKER_INDICES[name])
if sorted(_MODULE_ORDER_INDEX) != list(range(STATE_DIM)):
    raise RuntimeError("MODULE_MARKER_INDICES must partition the state vector")

# The PROTOCOL vector a rollout knows in advance, per member and minute (the head
# bank's exogenous first-layer input): [gut appearance (4) | duodenal delivery (3) |
# sleep_wake | activity | time features (4)].
_Z_GUT = 0
_Z_DUO = GUT_OUTPUT_DIM
_Z_EXTERNAL = {"sleep_wake": _Z_DUO + DUODENAL_DIM, "activity": _Z_DUO + DUODENAL_DIM + 1}
_Z_TIME = _Z_DUO + DUODENAL_DIM + 2
_Z_DIM = _Z_TIME + TIME_FEATURES_DIM


def _exp_step(x: torch.Tensor, rate: torch.Tensor, dt: float | torch.Tensor) -> torch.Tensor:
    """`x·exp(rate·dt/x)` for x > 0; falls back to `x + rate·dt` at x <= 0 (an initial
    state handed in at zero — never produced by the step itself). ``dt`` is the step
    length: a float, or the per-marker ETD1 step ``h`` (see ``_etd_step_size``)."""
    positive = x > 0
    denom = torch.where(positive, x, torch.ones_like(x))
    return torch.where(positive, x * torch.exp(rate * dt / denom), x + rate * dt)


def _logit_interval_step(
    x: torch.Tensor, rate: torch.Tensor, dt: float | torch.Tensor, lo: float, hi: float,
) -> torch.Tensor:
    """Step ``x`` in logit coordinates of ``(x − lo)/(hi − lo)`` so it stays in ``(lo, hi)``.
    ``dt`` is a float or the per-marker ETD1 step ``h``, as in ``_exp_step``."""
    width = hi - lo
    x_c = x.clamp(lo + _SPO2_EPS, hi - _SPO2_EPS)
    u = (x_c - lo) / width
    logit = torch.log(u / (1.0 - u))
    dsdlogit = (x_c - lo) * (hi - x_c) / width
    logit_new = logit + rate * dt / dsdlogit.clamp(min=_SPO2_EPS)
    return (lo + width * torch.sigmoid(logit_new)).clamp(lo + _SPO2_EPS, hi - _SPO2_EPS)


# PLAN B2 — EXPONENTIAL (ETD1) STEPPING OF THE LINEAR PART.
#
# Forward Euler advances a restoring term −k·(x − x*) by the factor (1 − k·dt) where the exact
# solution contracts by e^{−k·dt}, so the time constant a module declares is not the one the
# rollout integrates, and the trajectory depends on dt. On a pure relaxation at dt = 1 min:
#
#     k       Euler    exact    effective rate   error
#     0.15    0.850    0.861        0.163         +8 %
#     0.30    0.700    0.741        0.357        +19 %   cardiovascular default, the teacher's k_hr
#     0.80    0.200    0.449        1.609       +101 %   the cardiovascular band's ceiling
#
# and it is unstable once k·dt > 2, which closes a coarser step (PLAN D6) to any marker whose
# k·dt would pass that. A module may therefore report a per-marker decay rate k >= 0 (1/min)
# alongside its rate, and the step advances `x + rate·h` with
#
#     h(k, dt) = (1 − e^{−k·dt}) / k            → dt as k → 0,
#
# the exact update of the linear part with the rest of the rate held over the step: for
# rate = F − k·(x − x*) it lands on x* + F/k + (x − x* − F/k)·e^{−k·dt} for any dt, and it is
# stable for any k >= 0. `rates` stays the TOTAL rate, restoring term included (what `forward`,
# the rate-matching signals and the coupling probes always saw); the decay annotates it, so the
# k a module reports must be the coefficient of a `−k·x` already inside that rate — a k for a
# term that is not there shrinks a step that does not decay. The couplings between markers stay
# explicit: what is stable is each marker's own restoring term, not the system.
#
# THE COORDINATES. `h` replaces `dt` in whatever coordinate a marker is stepped in:
#   * ordinary markers: `x + rate·h`.
#   * HRV and pulse pressure are stepped in log space and their `rate` is `x·dlog x/dt`, so
#     their k is the rate on the LOG deviation (cardiovascular's `k_hrv`, `k_pp`), and
#     `x·exp(rate·h/x)` is exact for `dlog x/dt = −k·(log x − log x*)`, still strictly positive.
#     SBP has no update of its own (it is rebuilt as DBP + PP), so the `sbp` slot's decay IS the
#     pulse pressure's `k_pp`; a module cannot give SBP a separate one. SBP > DBP holds as before.
#   * SpO₂ is stepped in logit coordinates while its restoring term is linear in SpO₂ itself, so
#     no exponential is exact there: `h` scales the logit displacement, which is exact to first
#     order in the deviation and keeps the bound. The logit map's own curvature error, shared
#     with Euler, remains (x0 in [88, 99.5], x* in [96.5, 99.5], k in [0.02, 0.22], dt = 1: ETD
#     stays within 0.055 points of the SpO₂-space exponential on average, Euler 0.071; worst 0.51
#     against 0.74, from 99.5 down to 96.5). What ETD removes is the declared-vs-integrated rate
#     gap and the step-size dependence (k = 0.22, 94 -> 98 %, 60 min as 6 steps of 10: Euler ends
#     3.0 points from its own 1-minute trajectory, ETD 5e-5).
#
# NUMERICS. `h` is 0/0 at k = 0 (every marker that does not opt in) and its closed form loses
# digits to cancellation just above it (the float32 autograd derivative of −expm1(−z)/z is off
# by up to 300 % below z = 1e-6, 12 % below 1e-4 and 0.2 % below 1e-2, against float64), where a
# small learned k lands. Below |z| = 0.1 a five-term series is used instead (value within 1e-7
# and derivative within 4e-6 of float64 over |z| < 80). Both branches of the `where` are fed an
# argument that is safe for THEM: the series sees 0 outside its range (so it cannot overflow)
# and the closed form sees 1 inside it (so it never divides by zero), because `where`
# backpropagates through the branch it did not select too, and 0·inf is NaN. (A `clamp(min=0)`
# on k would fail the other way: torch's clamp has gradient 0 AT the bound, which silently
# deletes d(step)/dk exactly at k = 0.)
#
# ADOPTION (none of it wired in: a module that does not define `decay_rates` is plain Euler,
# bit for bit). `decay_rates(state, coupling, const, drv, raw)` takes `step`'s arguments and
# returns k for the module's own markers in its marker order ([..., n_state], zeros where there
# is no restoring term, or None for none at all). The explicit `−k·(x − x*)` terms that would
# report:
#   cardiovascular  [k_hr, k_hrv, k_pp, k_dbp] in marker order (hr, hrv, sbp→pp, dbp)
#   thermoreg       k (core temperature; already a true 1/min)
#   respiratory     k per marker (rr, spo2)
#   stress          k_cort, k_acth, k_crh
#   metabolic       k_ins (insulin), k_gn (glucagon), k_bhb, k_ffa, p2 (insulin_action),
#                   hep_cons·hep_cons_scale (hepatic_output), K_MITO, lac_cons·lac_cons_scale·mito
#   appetite        K_GHR, K_LEP, K_INS_SLOW and glp1's cons·cons_scale
#   hepatobiliary   cck's cons·cons_scale, k_ileal, k_ba
# and `MassActionModule.step`'s `cons·cons_scale·raw` consumption IS a decay rate, so a subclass
# that keeps that `step` can report `cons·cons_scale`. There is deliberately no such default on
# the base class: the four subclasses rebuild their species' rates, and a blanket one would give
# glucose (a ConstantFluxHead, cons = 1) a decay of 0.02 for a term its rate does not have.
#
# WHERE IT PAYS. ETD1 scales the whole rate by (1 − e^{−z})/z, z = k·dt, which is exact only for a
# forcing that is constant over the step; once z is small and the inflow is pulsatile that is
# below the explicit-forcing error, and it can cost. Measured on the iter-109 modules (fresh
# default model, every marker displaced 0.9 NORM_SCALE, 12 min, max error against a dt = 1/16
# rollout in NORM_SCALE units, one module adopted at a time, Euler -> ETD):
#   hr 0.054 -> 0.003, rr 0.013 -> 0.001, temp 0.0023 -> 0.0002, cck 0.035 -> 0.001,
#   glp1 0.022 -> 0.002, ffa 0.092 -> 0.033, insulin 0.136 -> 0.086;
#   but the bile loop (k 0.01-0.03: gallbladder 0.005 -> 0.013, intestinal 0.007 -> 0.017) and
#   lactate (0.002 -> 0.003) get slightly worse. Adopt the fast terms (z >~ 0.05) first, and
#   check any adoption this way: a k that is not the coefficient of a term in the rate makes its
#   marker worse.
# ONE CONDITION: the teacher (`full_body.simulate_full_body`) is explicit Euler at dt = 1, so a
# student k copied from a teacher constant realizes e^{−k} per minute where the teacher realizes
# 1 − k (k_hr: 0.741 vs 0.700, k_ffa: 0.819 vs 0.800). The first module to adopt should therefore
# land with the teacher on the same step, or refit those k as −ln(1 − k_teacher).
_ETD_SERIES_BELOW = 0.1


def _etd_step_size(k: torch.Tensor, dt: float) -> torch.Tensor:
    """ETD1's effective step ``(1 − e^{−k·dt})/k``, finite and differentiable at k = 0
    (value ``dt``, d/dk ``−dt²/2``). Exact for either sign of k (k < 0 would be growth), to
    float32 rounding; in float64 the series below |k·dt| = 0.1 truncates at ~1e-8."""
    z = k * dt
    small = z.abs() < _ETD_SERIES_BELOW
    zs = torch.where(small, z, 0.0)
    zc = torch.where(small, 1.0, z)
    series = 1.0 + zs * (-0.5 + zs * (1.0 / 6.0 + zs * (-1.0 / 24.0 + zs * (1.0 / 120.0))))
    closed = torch.expm1(-zc) / -zc
    return dt * torch.where(small, series, closed)


def _reports_decay(mod: nn.Module) -> bool:
    """Whether ``mod``'s class defines ``decay_rates`` (a base-class default means "none")."""
    fn = getattr(type(mod), "decay_rates", None)
    return fn is not None and fn is not getattr(PhysiologyModule, "decay_rates", None)


def euler_step(
    state: torch.Tensor, rates: torch.Tensor, dt: float, decay: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """One step of the raw state, with the positive-by-construction markers (HRV, pulse
    pressure) stepped multiplicatively and SpO₂ stepped in logit coordinates of (70, 100).
    Shared by ``integrate`` and anyone who steps the model by hand.

    ``decay`` (same shape as ``rates``, per-marker k >= 0 in 1/min) is the restoring rate
    already inside ``rates``; the step then uses ETD1's ``h(k, dt)`` in place of ``dt`` in
    every coordinate (see PLAN B2 above). ``None`` is all zeros: forward Euler, bit for bit.
    The ``sbp`` slot's decay is the pulse pressure's, since SBP is rebuilt as DBP + PP and has
    no update of its own."""
    if decay is None:
        step = h_hrv = h_pp = h_spo2 = dt
    else:
        step = _etd_step_size(decay, dt)
        h_hrv, h_pp, h_spo2 = step[..., _HRV_IDX], step[..., _SBP_IDX], step[..., _SPO2_IDX]
    new_state = state + rates * step
    hrv_new = _exp_step(state[..., _HRV_IDX], rates[..., _HRV_IDX], h_hrv)
    pp = state[..., _SBP_IDX] - state[..., _DBP_IDX]
    pp_new = _exp_step(pp, rates[..., _SBP_IDX] - rates[..., _DBP_IDX], h_pp)
    dbp_new = new_state[..., _DBP_IDX]
    spo2_new = _logit_interval_step(
        state[..., _SPO2_IDX], rates[..., _SPO2_IDX], h_spo2, _SPO2_LO, _SPO2_HI)
    new_state = new_state.clone()
    new_state[..., _HRV_IDX] = hrv_new
    new_state[..., _SBP_IDX] = dbp_new + pp_new
    new_state[..., _SPO2_IDX] = spo2_new
    return new_state


class ModularPhysiologyNetwork(nn.Module):
    norm_center: torch.Tensor
    norm_scale: torch.Tensor

    def __init__(
        self,
        embedding_dim: int = EMBEDDING_DIM,
        metabolic_hidden: int = 48,
        appetite_hidden: int = 32,
        stress_hidden: int = 32,
        cardiovascular_hidden: int = 48,
        thermoreg_hidden: int = 24,
        respiratory_hidden: int = 24,
        gut_hidden: int = 32,
        hepatobiliary_hidden: int = 32,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Per-module embedding projections. iter 60 widened these alongside
        # EMBEDDING_DIM 32->64 (the spec-R1 capacity pivot). appetite gets the
        # largest relative bump (6->16): it owns glp1+ghrelin, the only two
        # markers with mean_mape >1, which a 6-dim projection could not encode
        # as independent patient-specific spike manifolds.
        self._emb_dims = {
            "gut": 16,
            "metabolic": 20,
            "appetite": 16,
            "stress": 12,
            "cardiovascular": 16,
            "thermoreg": 8,
            "respiratory": 8,
            # Iter 95: the enterohepatic loop. 8 is deliberately modest — most of the
            # axis's behaviour is structural (see modules/hepatobiliary.py), so the
            # embedding only needs to carry per-patient gains, not the mechanism.
            "hepatobiliary": 8,
        }
        self.embedding_projections = nn.ModuleDict({
            name: nn.Linear(embedding_dim, dim)
            for name, dim in self._emb_dims.items()
        })

        # Modules
        self.gut = GutModule(self._emb_dims["gut"], gut_hidden)
        self.metabolic = MetabolicModule(self._emb_dims["metabolic"], metabolic_hidden)
        self.appetite = AppetiteModule(self._emb_dims["appetite"], appetite_hidden)
        self.stress = StressModule(self._emb_dims["stress"], stress_hidden)
        self.cardiovascular = CardiovascularModule(self._emb_dims["cardiovascular"], cardiovascular_hidden)
        self.thermoreg = ThermoregModule(self._emb_dims["thermoreg"], thermoreg_hidden)
        self.respiratory = RespiratoryModule(self._emb_dims["respiratory"], respiratory_hidden)
        self.hepatobiliary = HepatobiliaryModule(self._emb_dims["hepatobiliary"], hepatobiliary_hidden)
        # Duodenal delivery (gastric emptying) — NOT the gut module's systemic
        # appearance channels. Takes no embedding, so it needs no precompute path.
        self.duodenal = DuodenalDeliveryKernel()

        self._modules_by_name = {
            "metabolic": self.metabolic,
            "appetite": self.appetite,
            "stress": self.stress,
            "cardiovascular": self.cardiovascular,
            "thermoreg": self.thermoreg,
            "respiratory": self.respiratory,
            "hepatobiliary": self.hepatobiliary,
        }
        for name, mod in self._modules_by_name.items():
            n_expected = len(MODULE_COUPLING_CHANNELS[name])
            n_got = int(mod.n_coupling)
            if n_got != n_expected:
                raise ValueError(
                    f"{name} coupling width {n_got} != MODULE_COUPLING_CHANNELS "
                    f"({n_expected}: {MODULE_COUPLING_CHANNELS[name]})"
                )

        # Every constructor argument, so a checkpoint can rebuild the exact layout
        # without re-deriving module widths from `hidden_dim` (see `from_checkpoint`).
        self.constructor_kwargs = {
            "embedding_dim": int(embedding_dim),
            "metabolic_hidden": int(metabolic_hidden),
            "appetite_hidden": int(appetite_hidden),
            "stress_hidden": int(stress_hidden),
            "cardiovascular_hidden": int(cardiovascular_hidden),
            "thermoreg_hidden": int(thermoreg_hidden),
            "respiratory_hidden": int(respiratory_hidden),
            "gut_hidden": int(gut_hidden),
            "hepatobiliary_hidden": int(hepatobiliary_hidden),
        }

        # Learned defaults for missing external inputs (iter 97; review item 1.7 and
        # student-review item 13). Through iter 96 these were two GLOBAL scalars,
        # `default_sleep_wake = 0.5` and `default_activity = 0.1` — the PRD and the
        # docstring above promised a default "conditioned on embedding and time", and
        # 0.1 is not rest: at the teacher's meaning it is +8.3 bpm of activity drive on
        # every real-data night scored at that default. The default is now a small head
        # on (embedding, time features) emitting
        #     sleep_wake = σ(a),   activity = sleep_wake · σ(b)
        # so a default that says "asleep" implies rest, and rest = 0 is representable
        # (σ(b) → 0). Zero-init output weights; biases give sleep_wake 0.5 and an awake
        # NEAT floor of 0.02 at cold start (so activity 0.01 — not 0.1).
        self.default_inputs_net = nn.Sequential(
            nn.Linear(embedding_dim + TIME_FEATURES_DIM, 16), nn.Tanh(), nn.Linear(16, 2),
        )
        with torch.no_grad():
            self.default_inputs_net[-1].weight.zero_()
            self.default_inputs_net[-1].bias.copy_(
                torch.tensor([0.0, math.log(0.02 / 0.98)]))

        # State indices per module
        self._met_idx = MODULE_MARKER_INDICES["metabolic"]
        self._app_idx = MODULE_MARKER_INDICES["appetite"]
        self._str_idx = MODULE_MARKER_INDICES["stress"]
        self._cvs_idx = MODULE_MARKER_INDICES["cardiovascular"]
        self._thm_idx = MODULE_MARKER_INDICES["thermoreg"]
        self._rsp_idx = MODULE_MARKER_INDICES["respiratory"]
        self._hpb_idx = MODULE_MARKER_INDICES["hepatobiliary"]

        # Specific marker indices for coupling
        self._glucose_idx = MARKER_INDEX["glucose"]
        self._insulin_idx = MARKER_INDEX["insulin"]
        self._lactate_idx = MARKER_INDEX["lactate"]
        self._cortisol_idx = MARKER_INDEX["cortisol"]
        self._temp_idx = MARKER_INDEX["temp"]
        self._glp1_idx = MARKER_INDEX["glp1"]
        self._fat_mass_idx = MARKER_INDEX["fat_mass"]

        self.register_buffer("norm_center", torch.tensor(NORM_CENTER, dtype=torch.float32))
        self.register_buffer("norm_scale", torch.tensor(NORM_SCALE, dtype=torch.float32))
        # Rates are assembled by concatenating the modules' outputs in _MODULE_ORDER
        # and gathering them back into marker order (not persisted: derived from types).
        inverse = [0] * STATE_DIM
        for pos, marker in enumerate(_MODULE_ORDER_INDEX):
            inverse[marker] = pos
        self.register_buffer(
            "_marker_from_module_order", torch.tensor(inverse, dtype=torch.long), persistent=False)
        # Static index maps for planned rollouts, per (purpose, module, device).
        self._index_cache: dict[tuple[str, str, str], object] = {}

    @classmethod
    def from_checkpoint(cls, checkpoint: dict, strict: bool = True) -> "ModularPhysiologyNetwork":
        """Rebuild the model a checkpoint was trained with.

        Prefers ``checkpoint["model_config"]`` (the ``constructor_kwargs`` this class
        records; ``train.py`` should save it alongside ``model_state``). Falls back to
        the historical derivation of every module width from ``hidden_dim`` so older
        artifacts still load.

        If the checkpoint records ``coupling_channels``, they must match
        ``MODULE_COUPLING_CHANNELS`` in this tree — a Linear input-width mismatch
        otherwise surfaces as an opaque ``size mismatch`` on e.g.
        ``cardiovascular.network.0.weight``.
        """
        saved = checkpoint.get("coupling_channels")
        if saved is not None:
            current = {k: list(v) for k, v in MODULE_COUPLING_CHANNELS.items()}
            recorded = {k: list(v) for k, v in saved.items()}
            if recorded != current:
                raise ValueError(
                    "checkpoint coupling layout does not match this code. "
                    f"checkpoint={recorded} current={current}. "
                    "Load against the training-commit tree, or re-train."
                )
        cfg = checkpoint.get("model_config")
        if cfg is None:
            h = int(checkpoint.get("hidden_dim", 48))
            cfg = {
                "embedding_dim": int(checkpoint.get("embedding_dim", EMBEDDING_DIM)),
                "metabolic_hidden": h,
                "appetite_hidden": max(24, h // 2),
                "stress_hidden": max(24, h // 2),
                "cardiovascular_hidden": h,
                "thermoreg_hidden": max(16, h // 3),
                "respiratory_hidden": max(16, h // 3),
            }
        model = cls(**cfg)
        try:
            model.load_state_dict(checkpoint.get("model_state", checkpoint), strict=strict)
        except RuntimeError as exc:
            if "size mismatch" in str(exc).lower():
                raise RuntimeError(
                    f"{exc}\nCheckpoint was likely trained with a different "
                    "MODULE_COUPLING_CHANNELS (a module Linear changed input "
                    "width). Load it against the training-commit code."
                ) from exc
            raise
        pm = checkpoint.get("embedding_prior_mean")
        ps = checkpoint.get("embedding_prior_std")
        if pm is not None and ps is not None:
            model._embedding_prior_mean = torch.tensor(pm, dtype=torch.float32)
            model._embedding_prior_std = torch.tensor(ps, dtype=torch.float32)
        return model

    def default_external_inputs(
        self,
        embedding: torch.Tensor,
        time_feats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Learned ``(sleep_wake, activity)`` defaults for ``embedding[B, E]`` at
        ``time_feats[B, TIME_FEATURES_DIM]``. ``activity`` is gated by ``sleep_wake``
        (asleep ⇒ rest); both in [0, 1)."""
        out = self.default_inputs_net(torch.cat([embedding, time_feats], dim=-1))
        sw = torch.sigmoid(out[..., 0])
        act = sw * torch.sigmoid(out[..., 1])
        return sw, act

    def resolve_external_inputs(
        self,
        embedding: torch.Tensor,
        time_feats: torch.Tensor,
        sleep_wake: Optional[torch.Tensor],
        activity: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(sleep_wake, activity)`` over ``time_feats``' leading shape, missing entries
        filled with the learned default.

        A series that is ``None`` is missing everywhere; a ``NaN`` entry is missing at
        that member and minute (so one batch can mix logged and unlogged inputs). Where
        sleep is known but activity is not, the default activity is re-gated by the
        known sleep state (asleep ⇒ rest), exactly as in ``forward``.
        ``embedding`` is ``[B, E]`` against ``time_feats[B, T, 4]`` or ``[N, E]``
        against ``[N, 4]``.
        """
        shape = time_feats.shape[:-1]
        sw = None if sleep_wake is None else sleep_wake.expand(shape)
        act = None if activity is None else activity.expand(shape)
        sw_given = None if sw is None else ~torch.isnan(sw)
        act_given = None if act is None else ~torch.isnan(act)
        if sw_given is not None and act_given is not None and bool(sw_given.all()) and bool(act_given.all()):
            return sw, act
        emb = embedding
        if emb.dim() < time_feats.dim():
            emb = emb.unsqueeze(-2).expand(*shape, emb.shape[-1])
        sw_default, act_default = self.default_external_inputs(emb, time_feats)
        sw_res = sw_default if sw_given is None else torch.where(sw_given, sw, sw_default)
        if sw_given is None:
            fallback = act_default
        else:
            fallback = torch.where(
                sw_given, sw_res * act_default / torch.clamp(sw_default, min=1e-6), act_default)
        act_res = fallback if act_given is None else torch.where(act_given, act, fallback)
        return sw_res, act_res

    def coupling_for(
        self,
        module: str,
        norm_state: torch.Tensor,
        gut_outputs: torch.Tensor,
        duodenal: torch.Tensor,
    ) -> torch.Tensor:
        """Assemble ``module``'s coupling vector from ``MODULE_COUPLING_CHANNELS``.

        ``norm_state`` is the full normalized ODE state. Gut and duodenal
        channels are the absorption clocks (not ODE markers). Callers that
        invoke a module directly must use this rather than rebuilding the
        layout by hand.
        """
        pieces = []
        for ch in MODULE_COUPLING_CHANNELS[module]:
            if ch in GUT_CHANNEL_INDEX:
                i = GUT_CHANNEL_INDEX[ch]
                pieces.append(gut_outputs[..., i:i + 1])
            elif ch in DUODENAL_CHANNEL_INDEX:
                i = DUODENAL_CHANNEL_INDEX[ch]
                pieces.append(duodenal[..., i:i + 1])
            else:
                i = MARKER_INDEX[ch]
                pieces.append(norm_state[..., i:i + 1])
        return torch.cat(pieces, dim=-1)

    def _step_gather_indices(self, module: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Index maps for a planned step: ``module``'s markers out of the normalized
        state, and its coupling vector (``MODULE_COUPLING_CHANNELS`` order) out of
        ``cat([norm_state, gut, duodenal])``."""
        key = ("gather", module, str(device))
        cached = self._index_cache.get(key)
        if cached is None:
            coupling = []
            for ch in MODULE_COUPLING_CHANNELS[module]:
                if ch in GUT_CHANNEL_INDEX:
                    coupling.append(STATE_DIM + GUT_CHANNEL_INDEX[ch])
                elif ch in DUODENAL_CHANNEL_INDEX:
                    coupling.append(STATE_DIM + GUT_OUTPUT_DIM + DUODENAL_CHANNEL_INDEX[ch])
                else:
                    coupling.append(MARKER_INDEX[ch])
            cached = (
                torch.tensor(MODULE_MARKER_INDICES[module], dtype=torch.long, device=device),
                torch.tensor(coupling, dtype=torch.long, device=device),
            )
            self._index_cache[key] = cached
        return cached

    def _head_input_layout(self, module: str, device: torch.device) -> "_HeadInputLayout":
        key = ("heads", module, str(device))
        cached = self._index_cache.get(key)
        if cached is None:
            cached = _HeadInputLayout(module, self._modules_by_name[module], device)
            self._index_cache[key] = cached
        return cached

    def forward(
        self,
        state: torch.Tensor,
        embedding: torch.Tensor,
        t_minutes: torch.Tensor,
        meals: list[MealEvent],
        sleep_wake: Optional[torch.Tensor] = None,
        activity: Optional[torch.Tensor] = None,
        gut_override: Optional[torch.Tensor] = None,
        gut_clock_exempt: bool = False,
        duodenal_override: Optional[torch.Tensor] = None,
        return_decay: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute rates of change for all state variables.

        state: [batch, STATE_DIM] or [STATE_DIM]
        embedding: [batch, EMBEDDING_DIM] or [EMBEDDING_DIM]
        t_minutes: [batch] or scalar — absolute time of day (mod 1440)
        meals: list of MealEvent
        sleep_wake: [batch] or None — 0=asleep, 1=awake
        activity: [batch] or None — 0=rest, 1=vigorous
        duodenal_override: [batch, DUODENAL_DIM] duodenal (fat, protein, carb) delivery
            for this step, computed by ``integrate`` on the WINDOW-OFFSET clock. Same
            frame contract as ``gut_override``. When absent, delivery is zero rather
            than being computed on the absolute ``t_minutes`` clock.
        gut_clock_exempt: opt out of the meal frame-contract guard below. ONLY for
            callers whose result is provably independent of gut timing (the
            coupling-prior finite-difference probe, where the gut term is
            state-independent and cancels between the two evaluations).
        return_decay: also return the per-marker decay rate (PLAN B2, ``euler_step``)
            as ``(rates, decay)``, in the same marker order as ``rates``. ``decay`` is
            ``None`` when no module defines ``decay_rates``, and zeros at the markers of
            a module that does not. A module that reports one is evaluated a second
            time for it (its heads fire twice), which is why this is opt-in: the rate
            callers (coupling probes, distillation) never pay it.
        """
        is_unbatched = state.dim() == 1
        if is_unbatched:
            state = state.unsqueeze(0)
            embedding = embedding.unsqueeze(0)
            if t_minutes.dim() == 0:
                t_minutes = t_minutes.unsqueeze(0)

        batch = state.shape[0]

        # Compute time features
        time_feats = compute_time_features(t_minutes)
        if time_feats.dim() == 1:
            time_feats = time_feats.unsqueeze(0).expand(batch, -1)

        # Resolve external inputs (learned defaults, conditioned on embedding and time,
        # when missing). A provided sleep_wake still gates a defaulted activity, so a
        # sleeping patient with no activity log is at rest.
        if sleep_wake is None or activity is None:
            sw_default, act_default = self.default_external_inputs(embedding, time_feats)
        sw = sleep_wake if sleep_wake is not None else sw_default
        if sw.dim() == 0:
            sw = sw.expand(batch)
        if activity is not None:
            act = activity
        elif sleep_wake is None:
            act = act_default
        else:
            act = sw * act_default / torch.clamp(sw_default, min=1e-6)
        if act.dim() == 0:
            act = act.expand(batch)

        # Normalize state for module inputs
        norm_state = (state - self.norm_center) / self.norm_scale

        # Project embeddings per module
        emb = {name: proj(embedding) for name, proj in self.embedding_projections.items()}

        # --- Gut module (not an ODE — processes meals) ---
        # Per-batch gut output is required when batch > 1: gut depends on
        # (carbs, fats, proteins, t, embedding) and embeddings differ across
        # batch, so a single broadcast would silently use member 0 for
        # everyone (the iter 13 bug). Callers should precompute the whole
        # gut window via ``precompute_gut_outputs`` and pass the per-step
        # slice in ``gut_override``.
        if gut_override is not None:
            go = gut_override
            if go.dim() == 1:
                gut_out_batch = go.unsqueeze(0).expand(batch, -1)
            elif go.shape[0] == 1:
                gut_out_batch = go.expand(batch, -1)
            elif go.shape[0] == batch:
                gut_out_batch = go
            else:
                raise ValueError(
                    f"gut_override batch {tuple(go.shape)} incompatible with "
                    f"state batch {batch}",
                )
        elif batch == 1:
            # FRAME CONTRACT: this per-step fallback feeds ``t_minutes`` (an
            # ABSOLUTE minute-of-day, per this method's signature) straight into
            # the gut as the absorption clock. But ``meal.time`` is a WINDOW
            # OFFSET (0-based), so meal-accurate absorption requires the
            # window-offset clock — which this path CANNOT reconstruct from an
            # absolute ``t_minutes`` alone. Do NOT route trajectory integration
            # through here: ``integrate`` precomputes the whole gut window on the
            # correct offset clock (see precompute_gut_outputs, iter 87) and
            # passes it via ``gut_override``. This fallback is retained only for
            # pointwise sensitivity probes (coupling_prior_loss) where the gut
            # is state-independent and cancels in the finite difference, so its
            # timing frame does not affect the result.
            #
            # Iter 90: the contract above was enforced only by this comment. The
            # iter-87 meal-timing bug (absorption computed on the absolute clock,
            # silently shifting every meal out of the window) lived here for ~19
            # iters. Enforce it instead of narrating it: with meals present, the
            # caller must either supply gut_override from precompute_gut_outputs,
            # or explicitly assert (via gut_clock_exempt) that gut timing cannot
            # affect its result.
            if meals and not gut_clock_exempt:
                raise ValueError(
                    "model.forward with meals requires gut_override (the window-offset "
                    "absorption clock). Route trajectory integration through integrate(), "
                    "which precomputes it via precompute_gut_outputs. The batch==1 "
                    "per-step gut path uses an ABSOLUTE minute-of-day clock and would "
                    "silently mis-time meal absorption (the iter-87 frame bug). Pass "
                    "gut_clock_exempt=True only if the gut term provably cancels out.",
                )
            gut_out = self.gut(float(t_minutes[0].item()), meals, emb["gut"][0])
            gut_out_batch = gut_out.unsqueeze(0)
        else:
            raise ValueError(
                "model.forward at batch>1 requires gut_override; precompute "
                "via precompute_gut_outputs(model, embedding, n_steps, ...) "
                "and pass gut_outputs[:, step] each step.",
            )

        if duodenal_override is not None:
            duo = duodenal_override
            if duo.dim() == 1:
                duo = duo.unsqueeze(0)
            if duo.shape[0] != batch:
                duo = duo.expand(batch, -1)
        else:
            duo = torch.zeros(batch, DUODENAL_DIM, dtype=state.dtype, device=state.device)

        external = {"sleep_wake": sw, "activity": act}
        parts = []
        decays: list[Optional[torch.Tensor]] = []
        for name in _MODULE_ORDER:
            mod = self._modules_by_name[name]
            own = norm_state[:, MODULE_MARKER_INDICES[name]]
            coupling = self.coupling_for(name, norm_state, gut_out_batch, duo)
            ext = torch.stack([external[n] for n in mod.external_inputs], dim=-1)
            rate = mod(own, coupling, ext, emb[name], time_feats)
            parts.append(rate)
            if return_decay and _reports_decay(mod):
                # The same composition as ``PhysiologyModule.forward``, so the decay is a
                # function of exactly the constants, drives and head outputs the rate saw.
                const = mod.constants(emb[name])
                drv = mod.drives(ext, coupling, time_feats, const)
                raw = mod.head_outputs(own, coupling, ext, emb[name], time_feats)
                decays.append(_as_rate_shape(
                    mod.decay_rates(own, coupling, const, drv, raw), rate, name))
            else:
                decays.append(None)
        rates = torch.cat(parts, dim=-1).index_select(-1, self._marker_from_module_order)
        decay = _assemble_decay(parts, decays, self._marker_from_module_order)

        if is_unbatched:
            rates = rates.squeeze(0)
            decay = None if decay is None else decay.squeeze(0)

        return (rates, decay) if return_decay else rates


def _as_rate_shape(decay: Optional[torch.Tensor], rate: torch.Tensor, name: str) -> torch.Tensor:
    """A module's decay at its rate's shape: ``None`` is zeros, and a parameter-only decay
    (no batch dimension) is broadcast across the batch."""
    if decay is None:
        return torch.zeros_like(rate)
    if decay.shape[-1:] != rate.shape[-1:]:
        raise ValueError(
            f"{name}.decay_rates returned {tuple(decay.shape)}: one rate per marker of the "
            f"module ({rate.shape[-1]}, in its marker order) is expected")
    return torch.broadcast_to(decay, rate.shape)


def _assemble_decay(
    rates_by_module: list[torch.Tensor],
    decay_by_module: list[Optional[torch.Tensor]],
    marker_order: torch.Tensor,
) -> Optional[torch.Tensor]:
    """The modules' decays concatenated in ``_MODULE_ORDER`` and gathered into marker order,
    exactly as their rates are; zeros for a module that reports none, ``None`` if none does."""
    if all(d is None for d in decay_by_module):
        return None
    cat = torch.cat([
        torch.zeros_like(r) if d is None else d
        for r, d in zip(rates_by_module, decay_by_module)
    ], dim=-1)
    return cat.index_select(-1, marker_order)


class _HeadBank:
    """Every MLP head of every module, evaluated as ONE three-layer network per step.

    The model has ten small Linear-Tanh-Linear-Tanh-Linear heads; called one by one
    they cost ~70 matmuls and ~200 dispatches per simulated minute. Here their
    weights are stacked once per rollout (from the CURRENT parameters, so edits and
    optimizer steps between rollouts are always seen): the hidden layers become one
    block-diagonal matmul, and the first layer is split by where each input column
    comes from (``PhysiologyModule.head_input_sources``) —

    * ODE markers (a module's own state and its marker couplings): ``norm_state @ W1``
      per step;
    * the protocol (gut / duodenal channels, sleep, activity, time): one matmul over
      the whole ``[B, T]`` window up front;
    * the embedding projection and the bias: once per member.

    The blocks are built with differentiable slicing / scattering, so autograd carries
    every step's gradient back to the original per-head parameters. The arithmetic
    is the same as calling each head, summed in a different order (float rounding
    only).
    """

    def __init__(
        self,
        model: "ModularPhysiologyNetwork",
        emb: dict[str, torch.Tensor],
        z: torch.Tensor,
    ) -> None:
        w1_state: list[torch.Tensor] = []
        w1_exo: list[torch.Tensor] = []
        first: list[torch.Tensor] = []
        w2: list[torch.Tensor] = []
        b2: list[torch.Tensor] = []
        w3: list[torch.Tensor] = []
        b3: list[torch.Tensor] = []
        self.slots: dict[str, list[tuple[object, int]]] = {}
        splits: list[int] = []
        for name in _MODULE_ORDER:
            mod = model._modules_by_name[name]
            nets = mod.mlp_heads()
            if not nets:
                continue
            layout = model._head_input_layout(name, z.device)
            for key, net in nets.items():
                l1, l2, l3 = _three_layers(name, key, net)
                W1 = l1.weight
                H = W1.shape[0]
                w1_state.append(W1.new_zeros(STATE_DIM, H).index_add(
                    0, layout.state_rows, W1.index_select(1, layout.state_cols).t()))
                w1_exo.append(W1.new_zeros(_Z_DIM, H).index_add(
                    0, layout.exo_rows, W1.index_select(1, layout.exo_cols).t()))
                first.append(emb[name] @ W1.index_select(1, layout.emb_cols).t() + l1.bias)
                w2.append(l2.weight.t())
                b2.append(l2.bias)
                w3.append(l3.weight.t())
                b3.append(l3.bias)
                self.slots.setdefault(name, []).append((key, len(splits)))
                splits.append(l3.out_features)
        self.splits = splits
        self.w1_state = torch.cat(w1_state, dim=1)                     # [S, H]
        exo = torch.matmul(z, torch.cat(w1_exo, dim=1))               # [B, T, H]
        self.exo_steps = _per_step(exo + torch.cat(first, dim=-1).unsqueeze(1))
        self.w2 = torch.block_diag(*w2)
        self.b2 = torch.cat(b2)
        self.w3 = torch.block_diag(*w3)
        self.b3 = torch.cat(b3)

    def __call__(self, norm_state: torch.Tensor, exo_t: torch.Tensor) -> list[torch.Tensor]:
        h = torch.tanh(torch.addmm(exo_t, norm_state, self.w1_state))
        h = torch.tanh(torch.addmm(self.b2, h, self.w2))
        return list(torch.addmm(self.b3, h, self.w3).split(self.splits, dim=-1))


def _per_step(x: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """``[B, T, ...]`` → T contiguous ``[B, ...]`` slices, made once per rollout.

    Unbinding once matters for the backward: a per-step select of the window would
    materialise a full-size zero gradient on every step (O(T²) memory traffic). The
    time-major copy gives every slice the same canonical strides (``.contiguous()``
    would keep a batch-1 window's stride on its size-1 dim), which compiled steps
    guard on.
    """
    if _COMPILED_STEP is not None and torch.is_grad_enabled() and not x.requires_grad:
        # Compiled steps are specialised on which inputs carry gradient; making every
        # protocol input carry it keeps that from multiplying the variants (a logged
        # vs a defaulted sleep series, the teacher's absorption vs the student's).
        x = x.detach().requires_grad_(True)
    return x.transpose(0, 1).clone(memory_format=torch.contiguous_format).unbind(0)


def _three_layers(name: str, key: object, net: nn.Module) -> tuple[nn.Linear, nn.Linear, nn.Linear]:
    layers = list(net)
    shape_ok = (
        len(layers) == 5
        and all(isinstance(layers[i], nn.Linear) for i in (0, 2, 4))
        and all(isinstance(layers[i], nn.Tanh) for i in (1, 3))
    )
    if not shape_ok:
        raise TypeError(
            f"{name} head {key!r}: the fused integrator expects Linear-Tanh-Linear-Tanh-Linear, "
            f"got {net}",
        )
    return layers[0], layers[2], layers[4]


class _HeadInputLayout:
    """Where each column of one module's head input comes from (see ``_HeadBank``)."""

    def __init__(self, name: str, mod: nn.Module, device: torch.device) -> None:
        state_rows: list[int] = []
        state_cols: list[int] = []
        exo_rows: list[int] = []
        exo_cols: list[int] = []
        emb_cols: list[int] = []
        channels = MODULE_COUPLING_CHANNELS[name]
        for col, (kind, j) in enumerate(mod.head_input_sources()):
            if kind == "state":
                state_rows.append(MODULE_MARKER_INDICES[name][j])
                state_cols.append(col)
            elif kind == "coupling":
                ch = channels[j]
                if ch in GUT_CHANNEL_INDEX:
                    exo_rows.append(_Z_GUT + GUT_CHANNEL_INDEX[ch])
                    exo_cols.append(col)
                elif ch in DUODENAL_CHANNEL_INDEX:
                    exo_rows.append(_Z_DUO + DUODENAL_CHANNEL_INDEX[ch])
                    exo_cols.append(col)
                else:
                    state_rows.append(MARKER_INDEX[ch])
                    state_cols.append(col)
            elif kind == "external":
                exo_rows.append(_Z_EXTERNAL[mod.external_inputs[j]])
                exo_cols.append(col)
            elif kind == "embedding":
                emb_cols.append(col)
            elif kind == "time":
                exo_rows.append(_Z_TIME + j)
                exo_cols.append(col)
            else:
                raise ValueError(f"{name}: unknown head input source {kind!r}")

        def _t(xs: list[int]) -> torch.Tensor:
            return torch.tensor(xs, dtype=torch.long, device=device)

        self.state_rows, self.state_cols = _t(state_rows), _t(state_cols)
        self.exo_rows, self.exo_cols = _t(exo_rows), _t(exo_cols)
        self.emb_cols = _t(emb_cols)


class _ModulePlan:
    """One module's share of a rollout: its patient constants, its protocol drives
    unbound per minute, and the index maps from the step's state and coupling."""

    def __init__(
        self,
        name: str,
        mod: nn.Module,
        const: dict[str, torch.Tensor],
        drives: dict[str, torch.Tensor],
        B: int,
        T: int,
        state_idx: torch.Tensor,
        coupling_idx: torch.Tensor,
        head_slots: list[tuple[object, int]],
    ) -> None:
        self.name = name
        self.module = mod
        self.const = const
        self.state_idx = state_idx
        self.coupling_idx = coupling_idx
        self.head_slots = head_slots
        self.reports_decay = _reports_decay(mod)
        self.fixed: dict[str, torch.Tensor] = {}
        self.per_step: dict[str, tuple[torch.Tensor, ...]] = {}
        for k, v in drives.items():
            if v.dim() == 0:
                self.fixed[k] = v
            else:
                self.per_step[k] = _per_step(v.reshape(B, T, *v.shape[1:]))


class _RolloutPlan:
    """Everything a rollout knows before its first step, computed once.

    ``rates(state, inputs(t))`` then does only the work that depends on the ODE
    state: the head bank, each module's ``step`` and the coupling gathers. The pointwise
    ``ModularPhysiologyNetwork.forward`` and this plan call the same module code
    (``PhysiologyModule``), so they compute the same rates. ``rates_and_decay`` adds the
    modules' decay rates (PLAN B2) from the same constants, drives and head outputs; with no
    module reporting one the decay is ``None`` and a step is plain Euler.
    """

    def __init__(
        self,
        model: "ModularPhysiologyNetwork",
        embedding: torch.Tensor,
        n_steps: int,
        dt: float,
        start_time_minutes,
        gut: torch.Tensor,
        duo: torch.Tensor,
        sleep_wake: Optional[torch.Tensor],
        activity: Optional[torch.Tensor],
    ) -> None:
        B = int(embedding.shape[0])
        T = int(n_steps)
        device = embedding.device
        self.center = model.norm_center
        self.scale = model.norm_scale
        self.marker_order = model._marker_from_module_order

        # Absolute clock, per member: computed in float64 exactly like the historical
        # per-step ``(start + step·dt) % 1440`` Python float, then cast.
        start = torch.as_tensor(start_time_minutes, dtype=torch.float64, device=device).reshape(-1, 1)
        clock = torch.arange(T, dtype=torch.float64, device=device) * float(dt)
        t_abs = torch.remainder(start + clock, 1440.0).to(torch.float32).expand(B, T)
        time_feats = compute_time_features(t_abs)                       # [B, T, 4]
        sw, act = model.resolve_external_inputs(embedding, time_feats, sleep_wake, activity)
        emb = {name: proj(embedding) for name, proj in model.embedding_projections.items()}

        z = torch.cat([gut, duo, sw.unsqueeze(-1), act.unsqueeze(-1), time_feats], dim=-1)
        self.bank = _HeadBank(model, emb, z)
        self.gut_steps = _per_step(gut)
        self.duo_steps = _per_step(duo)

        # Drives see the protocol for the whole window at once, flattened to [B*T].
        # Marker coupling channels are NaN there: a drive must not read the state.
        nan_state = torch.full((B, T, STATE_DIM), float("nan"), dtype=gut.dtype, device=device)
        protocol_coupling = torch.cat([nan_state, gut, duo], dim=-1)
        external = {"sleep_wake": sw, "activity": act}
        tf_flat = time_feats.reshape(B * T, -1)
        self.modules: list[_ModulePlan] = []
        for name in _MODULE_ORDER:
            mod = model._modules_by_name[name]
            state_idx, coupling_idx = model._step_gather_indices(name, device)
            ext = torch.stack([external[n] for n in mod.external_inputs], dim=-1)
            const_window = mod.constants(emb[name].repeat_interleave(T, dim=0))
            drives = mod.drives(
                ext.reshape(B * T, -1),
                protocol_coupling.index_select(-1, coupling_idx).reshape(B * T, -1),
                tf_flat,
                const_window,
            )
            self.modules.append(_ModulePlan(
                name, mod, mod.constants(emb[name]), drives, B, T,
                state_idx, coupling_idx, self.bank.slots.get(name, []),
            ))

    def inputs(self, t: int) -> tuple[torch.Tensor, ...]:
        """Step ``t``'s protocol tensors, flat: head-bank first-layer offset, gut,
        duodenal, then every module's per-step drives in plan order."""
        return (
            self.bank.exo_steps[t], self.gut_steps[t], self.duo_steps[t],
            *(steps[t] for m in self.modules for steps in m.per_step.values()),
        )

    def rates(self, state: torch.Tensor, step_inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return self._evaluate(state, step_inputs, with_decay=False)[0]

    def rates_and_decay(
        self, state: torch.Tensor, step_inputs: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """``(rates, decay)`` for ``euler_step``; ``decay`` is ``None`` if no module reports one."""
        return self._evaluate(state, step_inputs, with_decay=True)

    def _evaluate(
        self, state: torch.Tensor, step_inputs: tuple[torch.Tensor, ...], with_decay: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        exo_t, gut_t, duo_t = step_inputs[:3]
        drives_t = iter(step_inputs[3:])
        norm = (state - self.center) / self.scale
        xin = torch.cat([norm, gut_t, duo_t], dim=-1)
        heads = self.bank(norm, exo_t)
        parts = []
        decays: list[Optional[torch.Tensor]] = []
        for m in self.modules:
            drv = dict(m.fixed)
            for k in m.per_step:
                drv[k] = next(drives_t)
            own = norm.index_select(-1, m.state_idx)
            coupling = xin.index_select(-1, m.coupling_idx)
            raw = {key: heads[i] for key, i in m.head_slots}
            rate = m.module.step(own, coupling, m.const, drv, raw)
            parts.append(rate)
            if with_decay and m.reports_decay:
                decays.append(_as_rate_shape(
                    m.module.decay_rates(own, coupling, m.const, drv, raw), rate, m.name))
            else:
                decays.append(None)
        rates = torch.cat(parts, dim=-1).index_select(-1, self.marker_order)
        return rates, _assemble_decay(parts, decays, self.marker_order)


def _planned_euler_step(
    plan: _RolloutPlan, state: torch.Tensor, step_inputs: tuple[torch.Tensor, ...], dt: float,
) -> torch.Tensor:
    """One planned minute: the rates (and decay rates) from the plan, then ``euler_step``.
    The unit ``enable_compiled_steps`` hands to ``torch.compile``."""
    rates, decay = plan.rates_and_decay(state, step_inputs)
    return euler_step(state, rates, dt, decay)


_COMPILED_STEP = None


def _step_kernel():
    return _guarded_compiled_step if _COMPILED_STEP is not None else _planned_euler_step


def _guarded_compiled_step(
    plan: "_RolloutPlan", state: torch.Tensor, step_inputs: tuple[torch.Tensor, ...], dt: float,
) -> torch.Tensor:
    """The compiled step, or — if compiling a new variant fails — the eager step for
    this and every later call (the step is a pure function, so recomputing is safe)."""
    global _COMPILED_STEP
    compiled = _COMPILED_STEP
    if compiled is not None:
        try:
            return compiled(plan, state, step_inputs, dt)
        except Exception as err:  # noqa: BLE001 — any backend failure means "go eager"
            _COMPILED_STEP = None
            print(f"[compile] compiled steps OFF mid-run: {type(err).__name__}: {err}", flush=True)
    return _planned_euler_step(plan, state, step_inputs, dt)


def enable_compiled_steps(model: Optional["ModularPhysiologyNetwork"] = None, *, enabled: bool = True) -> bool:
    """Run every planned minute through ``torch.compile`` (opt-in).

    The planned step is a fixed chain of ~500 small ops; compiled, its forward and
    backward each become a handful of fused C++ kernels — measured 4.9x on a batch-1
    240-min window (0.41 s vs 2.0 s forward + backward) on top of the planned
    integrator. The CPU backend needs a C++ compiler at runtime (the trainer image
    installs g++). Each new variant — batch 1 vs N, with or without grad — compiles
    once, lazily, in 30-60 s.

    With ``model``, the compiled step is warmed up here on short rollouts (batch 1
    and 2, with and without grad) and checked against the eager step, so a missing
    toolchain or a backend problem surfaces at startup, not mid-run; on any failure
    or mismatch compiled steps stay OFF and the reason is printed. Returns whether
    compiled steps are on.
    """
    global _COMPILED_STEP
    _COMPILED_STEP = None
    if not enabled:
        return False
    # Past the limit dynamo would silently run new variants eagerly.
    torch._dynamo.config.recompile_limit = max(int(torch._dynamo.config.recompile_limit), 64)
    # The per-step inputs are slices of one window tensor, so their storage offsets
    # vary; duck sizing would tie those to unrelated sizes and recompile at random.
    torch.fx.experimental._config.use_duck_shape = False
    compiled = torch.compile(_planned_euler_step, dynamic=True)
    if model is None:
        _COMPILED_STEP = compiled
        return True
    try:
        _COMPILED_STEP = compiled
        _check_compiled_steps(model)
    except Exception as err:  # noqa: BLE001 — any backend failure means "stay eager"
        _COMPILED_STEP = None
        print(f"[compile] compiled steps OFF: {type(err).__name__}: {err}", flush=True)
        return False
    print("[compile] compiled steps ON (warm-up matched the eager step)", flush=True)
    return True


def _check_compiled_steps(model: "ModularPhysiologyNetwork") -> None:
    global _COMPILED_STEP
    device = model.norm_center.device
    meals = [MealEvent(time=1.0, carbs=50.0, fats=10.0, proteins=10.0)]
    for batch in (1, 2):
        state = model.norm_center.unsqueeze(0).expand(batch, -1)
        emb = torch.zeros(batch, model.embedding_dim, device=device, requires_grad=True)
        results = []
        compiled = _COMPILED_STEP
        for kernel in (compiled, None):
            _COMPILED_STEP = kernel
            with torch.no_grad():
                frozen = integrate(model, state, emb, 3, meals=meals)
            traj = integrate(model, state, emb, 3, meals=meals)
            (grad,) = torch.autograd.grad(traj.pow(2).sum(), emb)
            results.append((frozen, traj.detach(), grad))
        _COMPILED_STEP = compiled
        for got, want in zip(results[0], results[1]):
            if not torch.allclose(got, want, rtol=1e-4, atol=1e-5):
                raise RuntimeError(f"compiled step disagrees with eager at batch {batch}")


def _per_member_meals(meals, batch: int) -> Optional[list[list[MealEvent]]]:
    """``meals`` as one list per batch member, or ``None`` when it is one shared list."""
    if not meals or isinstance(meals[0], MealEvent):
        return None
    if len(meals) != batch:
        raise ValueError(f"per-member meals: {len(meals)} lists for a batch of {batch}")
    return [list(m) for m in meals]


def integrate(
    model: ModularPhysiologyNetwork,
    initial_state: torch.Tensor,
    embedding: torch.Tensor,
    n_steps: int,
    dt: float = 1.0,
    start_time_minutes: float | torch.Tensor = 360.0,
    meals: Optional[list] = None,
    sleep_wake: Optional[torch.Tensor] = None,
    activity: Optional[torch.Tensor] = None,
    gut_outputs: Optional[torch.Tensor] = None,
    checkpoint_segments: int = 0,
    duodenal_outputs: Optional[torch.Tensor] = None,
    member_steps: Optional[torch.Tensor] = None,
    *,
    planned: bool = True,
) -> torch.Tensor:
    """Integrate the modular ODE forward in time.

    Shape-polymorphic on the leading dim:

    - Unbatched (``initial_state[STATE_DIM]``, ``embedding[EMB]``,
      ``gut_outputs[T, OUT]``) → ``[T, STATE_DIM]``
    - Batched (``initial_state[B, STATE_DIM]``, ``embedding[B, EMB]``,
      ``gut_outputs[B, T, OUT]``) → ``[B, T, STATE_DIM]``

    The protocol is shared across batch members or given per member:

    - ``meals``: one ``list[MealEvent]`` for everyone, or a list of B such lists.
    - ``start_time_minutes``: a float, or a ``[B]`` tensor.
    - ``sleep_wake`` / ``activity``: ``[T]`` (shared) or ``[B, T]``; ``None`` means
      missing everywhere and a ``NaN`` entry means missing at that member and minute
      — the model substitutes its learned default (``resolve_external_inputs``).
    - ``gut_outputs``: ``[T, GUT_OUTPUT_DIM]`` or ``[B, T, GUT_OUTPUT_DIM]``, or None to
      compute it from ``meals`` on the window-offset clock (``precompute_gut_outputs``).
    - ``duodenal_outputs``: ``[T, 3]`` or ``[B, T, 3]`` duodenal (fat, protein, carb)
      delivery on the WINDOW-OFFSET clock, or None to compute it from ``meals``
      (iter 97 — mirrors ``gut_outputs`` so a training signal that assembles its own
      window can hand the biliary axis its meal stimulus; review item 4.7).
    - ``member_steps``: ``[B]`` per-member horizons ≤ ``n_steps``, so protocols of
      different lengths can share one rollout. Member ``b``'s rows ``[0, member_steps[b])``
      are exactly its own rollout; after that its state is HELD (not integrated), so a
      padded minute can neither diverge nor feed a NaN into the shared backward.

    For a ``ModularPhysiologyNetwork`` the rollout is PLANNED (``_RolloutPlan``):
    everything that does not depend on the ODE state — per-patient constants, the
    protocol's drives, the embedding and protocol halves of every head's first layer
    — is computed once for the whole window, and each step runs only the state-
    dependent work through one fused head network. ``planned=False`` steps the same
    model through its pointwise ``forward`` one minute at a time instead: the
    reference the planned path is tested against, and the path to use when module or
    head forward hooks must fire (the planned step calls ``PhysiologyModule.step``
    directly). Any other rate model (test stubs) is always stepped pointwise.

    Each step is forward Euler, except at the markers of a module that defines
    ``decay_rates`` (PLAN B2): those advance by ETD1's ``(1 − e^{−k·dt})/k`` in place of
    ``dt``, which is the declared time constant and stops the trajectory depending on
    ``dt``. A model whose modules define none steps exactly as it always did.

    checkpoint_segments: if > 0 and grad is enabled, split the n_steps loop
        into roughly this many chunks and use torch.utils.checkpoint on each.
        Memory drops from O(n_steps) intermediate activations held in the
        autograd graph to O(n_steps / checkpoint_segments + checkpoint_segments)
        at the cost of one re-execution of each chunk during backward. The
        iter-67 saga found one ``loss.backward()`` through the unchunked 1440-
        step calibration rollout hanging > 1 h on the wider iter-66 model
        (EMBEDDING_DIM=64, n_species=4) — chunking is the structural fix.
        Default 0 = unchunked (back-compat with all other callers).
    """
    if meals is None:
        meals = []

    unbatched = initial_state.dim() == 1
    if unbatched:
        initial_state = initial_state.unsqueeze(0)
        embedding = embedding.unsqueeze(0)
        if gut_outputs is not None and gut_outputs.dim() == 2:
            gut_outputs = gut_outputs.unsqueeze(0)
    batch = int(initial_state.shape[0])
    if embedding.shape[0] != batch:
        embedding = embedding.expand(batch, -1)
    device = initial_state.device
    member_meals = _per_member_meals(meals, batch)

    if planned and isinstance(model, ModularPhysiologyNetwork):
        if gut_outputs is None:
            # The whole gut window up front, on the window-offset clock (see
            # precompute_gut_outputs) — never the per-step absolute-clock path.
            if member_meals is not None:
                gut_outputs = torch.stack([
                    precompute_gut_outputs(model, embedding[b], n_steps, dt=dt, meals=member_meals[b])
                    for b in range(batch)
                ])
            else:
                gut_outputs = precompute_gut_outputs(model, embedding, n_steps, dt=dt, meals=meals)
        gut = gut_outputs[..., :n_steps, :].expand(batch, n_steps, GUT_OUTPUT_DIM)
        duo = duodenal_outputs
        if duo is None:
            if member_meals is not None:
                duo = torch.stack([
                    precompute_duodenal_outputs(model, n_steps, dt=dt, meals=member_meals[b])
                    for b in range(batch)
                ])
            else:
                duo = precompute_duodenal_outputs(model, n_steps, dt=dt, meals=meals)
        duo = duo[..., :n_steps, :].to(gut.dtype).expand(batch, n_steps, DUODENAL_DIM)

        def _series(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return None if x is None else x[..., :n_steps].to(device=device, dtype=gut.dtype)

        plan = _RolloutPlan(
            model, embedding, n_steps, dt, start_time_minutes, gut, duo,
            _series(sleep_wake), _series(activity),
        )
        held: list[Optional[torch.Tensor]] = [None] * n_steps
        if member_steps is not None:
            horizon = torch.as_tensor(member_steps, device=device).reshape(batch, 1)
            # Row t+1 is written by step t; a member's last row is horizon − 1.
            hold = torch.arange(n_steps, device=device).unsqueeze(0) >= horizon - 1   # [B, T]
            held = [
                column if any_held else None
                for column, any_held in zip(hold.t().unsqueeze(-1).unbind(0), hold.any(dim=0).tolist())
            ]

        kernel = _step_kernel()

        def step_fn(state: torch.Tensor, step: int) -> torch.Tensor:
            # The compiled kernel (if on) is specialised on whether its inputs carry
            # grad; the initial state usually does not, so step 0 runs eagerly and
            # one compiled variant serves every later step.
            run = _planned_euler_step if step == 0 else kernel
            new_state = run(plan, state, plan.inputs(step), dt)
            if held[step] is not None:
                new_state = torch.where(held[step], state, new_state)
            return new_state
    else:
        if member_meals is not None or torch.is_tensor(start_time_minutes) or member_steps is not None:
            raise ValueError("per-member protocols need a ModularPhysiologyNetwork")
        # Precompute the whole gut window so meal absorption runs on the window-offset
        # clock even at batch 1: the per-step gut path in ``forward`` would compute
        # meal-dt from the absolute minute of day (the iter-87 frame bug). Skipped
        # without meals so gut-less stub models keep working.
        if gut_outputs is None and meals:
            gut_outputs = precompute_gut_outputs(
                model, embedding, n_steps, dt=dt,
                start_time_minutes=start_time_minutes, meals=meals,
            )
        elif batch > 1 and gut_outputs is None:
            raise ValueError(
                "integrate at batch>1 requires precomputed gut_outputs; call "
                "precompute_gut_outputs(model, embedding, n_steps, ...) first.",
            )
        # Duodenal delivery on the same window-offset clock, for the same reason.
        duo_outputs = duodenal_outputs
        if duo_outputs is None and meals and hasattr(model, "duodenal"):
            duo_outputs = precompute_duodenal_outputs(model, n_steps, dt=dt, meals=meals)

        # Other rate models (test stubs) return rates only and have no decay channel.
        decay_kw = {"return_decay": True} if isinstance(model, ModularPhysiologyNetwork) else {}

        def step_fn(state: torch.Tensor, step: int) -> torch.Tensor:
            t = torch.tensor(
                [(start_time_minutes + step * dt) % 1440.0],
                device=device,
            ).expand(batch)
            sw_step = sleep_wake[step].expand(batch) if sleep_wake is not None else None
            act_step = activity[step].expand(batch) if activity is not None else None
            gut_step = gut_outputs[:, step] if gut_outputs is not None else None
            duo_step = duo_outputs[step] if duo_outputs is not None else None
            out = model(
                state, embedding, t, meals,
                sleep_wake=sw_step, activity=act_step,
                gut_override=gut_step,
                duodenal_override=duo_step,
                **decay_kw,
            )
            rates, decay = out if decay_kw else (out, None)
            return euler_step(state, rates, dt, decay)

    use_checkpointing = (
        checkpoint_segments > 0
        and torch.is_grad_enabled()
        and (initial_state.requires_grad or embedding.requires_grad)
    )

    if not use_checkpointing:
        states: list[torch.Tensor] = []
        state = initial_state
        for step in range(n_steps):
            states.append(state)
            state = step_fn(state, step)
        out = torch.stack(states, dim=1)  # [B, T, STATE_DIM]
        return out.squeeze(0) if unbatched else out

    # Checkpointed path: split into K chunks, each chunk recomputes its
    # forward during backward so only the chunk-boundary states are saved
    # to the autograd graph. The per-step states (returned in `out`) DO go
    # in the graph because the caller may take the loss against any of them
    # — so within each chunk we materialize the chunk's step-state tensor
    # and return it; only the chunk's final exit state is the input to the
    # next chunk's checkpoint.
    chunk_size = max(1, (n_steps + checkpoint_segments - 1) // checkpoint_segments)

    def chunk_fn(state_in: torch.Tensor, start: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # `start` is a 0-d long tensor so checkpoint sees it as a graph input
        # (checkpoint requires the chunk to depend only on its tensor args).
        s0 = int(start.item())
        s1 = min(s0 + chunk_size, n_steps)
        states_local: list[torch.Tensor] = []
        s = state_in
        for step in range(s0, s1):
            states_local.append(s)
            s = step_fn(s, step)
        chunk_states = torch.stack(states_local, dim=1)  # [B, chunk_len, STATE_DIM]
        return s, chunk_states

    from torch.utils.checkpoint import checkpoint as _ckpt

    state = initial_state
    chunk_outputs: list[torch.Tensor] = []
    for chunk_idx in range((n_steps + chunk_size - 1) // chunk_size):
        start_t = torch.tensor(chunk_idx * chunk_size, device=device, dtype=torch.long)
        # use_reentrant=False is the modern checkpoint API; required when the
        # chunk's inputs include non-Tensor closures (model, meals) and we
        # want clean backward semantics without rng-warning noise.
        state, chunk_states = _ckpt(chunk_fn, state, start_t, use_reentrant=False)
        chunk_outputs.append(chunk_states)
    out = torch.cat(chunk_outputs, dim=1)  # [B, T, STATE_DIM]
    return out.squeeze(0) if unbatched else out


def precompute_duodenal_outputs(
    model: ModularPhysiologyNetwork,
    n_steps: int,
    dt: float = 1.0,
    meals: Optional[list[MealEvent]] = None,
) -> torch.Tensor:
    """Duodenal (fat, protein) delivery for a whole window → ``[T, 2]``.

    Same WINDOW-OFFSET frame contract as :func:`precompute_gut_outputs`: ``meal.time``
    is an offset from the window start, so the clock is ``arange(n_steps)·dt`` with no
    ``start_time_minutes`` and no daily wrap. Embedding-independent (the kernel takes
    none), so one call covers every batch member. Pass the result to
    ``integrate(..., duodenal_outputs=...)``.
    """
    if meals is None:
        meals = []
    device = model.duodenal.log_fast_rate.device
    times = torch.arange(n_steps, dtype=torch.float32, device=device) * dt
    return model.duodenal.forward_window(times, meals)


def precompute_gut_outputs(
    model: ModularPhysiologyNetwork,
    embedding: torch.Tensor,
    n_steps: int,
    dt: float = 1.0,
    start_time_minutes: float = 360.0,
    meals: Optional[list[MealEvent]] = None,
) -> torch.Tensor:
    """Vectorized precomputation of gut appearance for an entire window.

    Runs ``model.gut.forward_window`` once over all timesteps (and across
    every batch member if ``embedding`` is 2-D), returning a tensor that
    can be passed straight into ``integrate(..., gut_outputs=...)``.

    Shapes mirror :func:`integrate`:

    - ``embedding[EMB]``       → ``[T, GUT_OUTPUT_DIM]``
    - ``embedding[B, EMB]``    → ``[B, T, GUT_OUTPUT_DIM]``

    This single call replaces ``T`` per-step ``GutModule.forward`` calls
    (eliminating Python overhead) AND enables batched ``integrate`` over
    embeddings — the gut kernel is the only piece of the forward pass
    that depends on the embedding *outside* the ODE state, so once gut
    outputs are precomputed, the rest of ``model.forward`` parallelizes
    trivially across batch.
    """
    if meals is None:
        meals = []
    device = embedding.device
    # Gut absorption depends only on minutes-since-meal, and meal times are
    # window OFFSETS (0-based, matching the teacher's simulate_full_body frame,
    # the benchmark dataset, and the scenario/protocol generators). So the
    # absorption clock must be the window-offset frame, NOT absolute
    # minute-of-day. Adding start_time_minutes here (the behaviour through
    # iter 86) shifted every meal's absorption curve start_time_minutes earlier
    # than the window — e.g. an 08:00 start (480) pushed the whole absorption
    # peak out of the window and left only the tail leaking in — which crushed
    # the measured post-meal amplitude and was the real cause of the chronic
    # verifier_cat[meal] failure (the amplitude machinery itself is fine: with
    # meals on the correct clock the model already hits ~0.4-0.74 mg/dL/g). No
    # %1440 wrap either: a meal offset does not recur on a daily cycle.
    # ``start_time_minutes`` is retained for API compatibility (callers pass it
    # for the circadian clock used elsewhere in the forward pass) but is not
    # part of the absorption timebase.
    times = torch.arange(n_steps, dtype=torch.float32, device=device) * dt
    emb_gut = model.embedding_projections["gut"](embedding)
    return model.gut.forward_window(times, meals, emb_gut)
