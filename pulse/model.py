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
import os
import warnings
from contextlib import contextmanager
from typing import Iterator, Optional

import torch
import torch.nn as nn

from .types import (
    STATE_DIM, EMBEDDING_DIM, GUT_OUTPUT_DIM, TIME_FEATURES_DIM, DUODENAL_DIM,
    MARKER_INDEX, MODULE_MARKER_INDICES, MODULE_COUPLING_CHANNELS,
    GUT_CHANNEL_INDEX, DUODENAL_CHANNEL_INDEX,
    NORM_CENTER, NORM_SCALE,
    PHYSIOLOGICAL_MIN, PHYSIOLOGICAL_MAX,
)
from .modules import (
    GutModule, MetabolicModule, AppetiteModule, StressModule,
    CardiovascularModule, ThermoregModule, RespiratoryModule,
    HepatobiliaryModule, DuodenalDeliveryKernel,
)
from .modules.base import compute_time_features
from .modules.gut import MealEvent

# Hard physiological bounds applied to the integrated state each Euler step
# (iter 81). A no-op for any in-distribution trajectory (state stays well inside
# these extremes), this caps catastrophic off-manifold divergence — an
# embedding driven off the trained manifold by weakly-leashed calibration can
# explode the gut appearance kernel / a softplus production term and integrate a
# marker to nonphysical magnitudes (glucose ~18,000 mg/dL observed). See
# PHYSIOLOGICAL_MIN/MAX in types.py for the derivation and rationale.
_PHYS_MIN = torch.tensor(PHYSIOLOGICAL_MIN, dtype=torch.float32)
_PHYS_MAX = torch.tensor(PHYSIOLOGICAL_MAX, dtype=torch.float32)

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


def _exp_step(x: torch.Tensor, rate: torch.Tensor, dt: float) -> torch.Tensor:
    """`x·exp(rate·dt/x)` for x > 0; falls back to `x + rate·dt` at x <= 0 (an initial
    state handed in at zero — never produced by the step itself)."""
    positive = x > 0
    denom = torch.where(positive, x, torch.ones_like(x))
    return torch.where(positive, x * torch.exp(rate * dt / denom), x + rate * dt)


def _logit_interval_step(
    x: torch.Tensor, rate: torch.Tensor, dt: float, lo: float, hi: float,
) -> torch.Tensor:
    """Step ``x`` in logit coordinates of ``(x − lo)/(hi − lo)`` so it stays in ``(lo, hi)``."""
    width = hi - lo
    x_c = x.clamp(lo + _SPO2_EPS, hi - _SPO2_EPS)
    u = (x_c - lo) / width
    logit = torch.log(u / (1.0 - u))
    dsdlogit = (x_c - lo) * (hi - x_c) / width
    logit_new = logit + rate * dt / dsdlogit.clamp(min=_SPO2_EPS)
    return (lo + width * torch.sigmoid(logit_new)).clamp(lo + _SPO2_EPS, hi - _SPO2_EPS)


@contextmanager
def frozen_parameters(module: nn.Module) -> Iterator[nn.Module]:
    """Hold every parameter of ``module`` at ``requires_grad=False`` inside the block,
    restoring each flag on exit.

    For inner optimizations over an EMBEDDING (calibration): the model's weights are
    constants there. Without this, ``loss.backward()`` also accumulates the inner
    objective's gradient into every weight's ``.grad`` — the buffers the training
    step reads — and pays for computing it.
    """
    flags = [(p, p.requires_grad) for p in module.parameters()]
    for p, _ in flags:
        p.requires_grad_(False)
    try:
        yield module
    finally:
        for p, flag in flags:
            p.requires_grad_(flag)


# External-input layout per module: "as" = [activity, sleep_wake], "sa" = [sleep_wake,
# activity], "s" = [sleep_wake].
_MODULE_EXTERNAL = {
    "metabolic": "as",
    "appetite": "s",
    "stress": "sa",
    "cardiovascular": "as",
    "thermoreg": "as",
    "respiratory": "as",
    "hepatobiliary": "as",
}


def _module_input_columns(module: str) -> list[int]:
    """Columns of ``[normalized state ‖ gut ‖ duodenal]`` that form ``module``'s
    step input ``[own state ‖ coupling]`` (coupling per ``MODULE_COUPLING_CHANNELS``)."""
    cols = list(MODULE_MARKER_INDICES[module])
    for ch in MODULE_COUPLING_CHANNELS[module]:
        if ch in GUT_CHANNEL_INDEX:
            cols.append(STATE_DIM + GUT_CHANNEL_INDEX[ch])
        elif ch in DUODENAL_CHANNEL_INDEX:
            cols.append(STATE_DIM + GUT_OUTPUT_DIM + DUODENAL_CHANNEL_INDEX[ch])
        else:
            cols.append(MARKER_INDEX[ch])
    return cols


# Columns ``euler_step`` overwrites after the additive step, and where it reads them.
_STEP_SPECIAL_IDX = torch.tensor([_HRV_IDX, _SBP_IDX, _SPO2_IDX], dtype=torch.long)
_STEP_READ_IDX = torch.tensor([_HRV_IDX, _SBP_IDX, _DBP_IDX, _SPO2_IDX], dtype=torch.long)


def euler_step(state: torch.Tensor, rates: torch.Tensor, dt: float) -> torch.Tensor:
    """One forward-Euler step of the raw state, with the positive-by-construction
    markers (HRV, pulse pressure) stepped multiplicatively and SpO₂ stepped in
    logit coordinates of (70, 100). Shared by ``integrate`` and anyone who steps
    the model by hand."""
    new_state = state + rates * dt
    read = _STEP_READ_IDX.to(state.device)
    hrv, sbp, dbp, spo2 = state.index_select(-1, read).unbind(-1)
    r_hrv, r_sbp, r_dbp, r_spo2 = rates.index_select(-1, read).unbind(-1)
    # HRV and pulse pressure share the multiplicative step; one call for both.
    hrv_pp = _exp_step(
        torch.stack([hrv, sbp - dbp], dim=-1),
        torch.stack([r_hrv, r_sbp - r_dbp], dim=-1),
        dt,
    )
    dbp_new = new_state[..., _DBP_IDX]
    spo2_new = _logit_interval_step(spo2, r_spo2, dt, _SPO2_LO, _SPO2_HI)
    special = torch.stack([hrv_pp[..., 0], dbp_new + hrv_pp[..., 1], spo2_new], dim=-1)
    return new_state.index_copy(-1, _STEP_SPECIAL_IDX.to(state.device), special)


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

        # Step-time layout, resolved once. Each module reads ``x = [own state ‖
        # coupling]`` as ONE gather from ``[normalized state ‖ gut ‖ duodenal]``
        # (``_xidx_<module>``), and the modules' rate blocks, concatenated in
        # ``_modules_by_name`` order, are put back in MARKERS order by ONE gather
        # (``_rate_order``). Non-persistent: derived from ``types``, not trained.
        for name in self._modules_by_name:
            self.register_buffer(
                f"_xidx_{name}",
                torch.tensor(_module_input_columns(name), dtype=torch.long),
                persistent=False,
            )
        owned = [i for name in self._modules_by_name for i in MODULE_MARKER_INDICES[name]]
        if sorted(owned) != list(range(STATE_DIM)):
            raise ValueError("modules must own every marker exactly once")
        position = {marker: k for k, marker in enumerate(owned)}
        self.register_buffer(
            "_rate_order",
            torch.tensor([position[i] for i in range(STATE_DIM)], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer("_phys_min", _PHYS_MIN.clone(), persistent=False)
        self.register_buffer("_phys_max", _PHYS_MAX.clone(), persistent=False)

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
        sw, awake_act = self._default_input_parts(embedding, time_feats)
        return sw, sw * awake_act

    def _default_input_parts(
        self, embedding: torch.Tensor, time_feats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(σ(a), σ(b))``: the default sleep_wake and the default activity WHILE
        AWAKE. ``embedding`` broadcasts against ``time_feats``' leading dims."""
        lead = time_feats.shape[:-1]
        emb = embedding.expand(*lead, embedding.shape[-1])
        out = self.default_inputs_net(torch.cat([emb, time_feats], dim=-1))
        return torch.sigmoid(out[..., 0]), torch.sigmoid(out[..., 1])

    def resolve_external_inputs(
        self,
        embedding: torch.Tensor,
        time_feats: torch.Tensor,
        sleep_wake: Optional[torch.Tensor] = None,
        activity: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(sleep_wake, activity)`` with every missing value replaced by the learned
        default, shaped like ``time_feats.shape[:-1]``.

        Missing is ``None`` (the whole series) or ``NaN`` (an entry) — the second is
        what lets one batched rollout mix members with and without a sleep or
        activity log. A provided ``sleep_wake`` still gates a defaulted activity
        (``activity = sleep_wake · σ(b)``), so a sleeping patient with no activity
        log is at rest."""
        lead = time_feats.shape[:-1]

        def _as_series(v: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if v is None:
                return None
            return v.to(time_feats.dtype).expand(lead)

        sw_in, act_in = _as_series(sleep_wake), _as_series(activity)
        sw_missing = None if sw_in is None else torch.isnan(sw_in)
        act_missing = None if act_in is None else torch.isnan(act_in)
        need_default = (
            sw_in is None or act_in is None
            or bool(sw_missing.any()) or bool(act_missing.any())
        )
        if not need_default:
            return sw_in, act_in
        sw_def, awake_act = self._default_input_parts(embedding, time_feats)
        if sw_in is None:
            sw = sw_def
        else:
            sw = torch.where(sw_missing, sw_def, sw_in)
        act_def = sw * awake_act
        if act_in is None:
            act = act_def
        else:
            act = torch.where(act_missing, act_def, act_in)
        return sw, act

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
        full = torch.cat([norm_state, gut_outputs, duodenal], dim=-1)
        n_own = len(MODULE_MARKER_INDICES[module])
        return full.index_select(-1, getattr(self, f"_xidx_{module}")[n_own:])

    # ---- the two phases (see modules/base.py) -------------------------------------

    def prepare(
        self,
        embedding: torch.Tensor,
        time_feats: torch.Tensor,
        sleep_wake: Optional[torch.Tensor] = None,
        activity: Optional[torch.Tensor] = None,
    ) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, dict[str, torch.Tensor]]]:
        """Everything of the vector field that does not read the ODE state, per module.

        ``embedding`` is ``[B, E]``; ``time_feats`` is ``[B, 4]`` for one call or
        ``[T, B, 4]`` for a rollout, and ``sleep_wake`` / ``activity`` match its
        leading dims (``None`` / ``NaN`` = missing, see ``resolve_external_inputs``).
        Returns ``(const, seq)`` keyed by module; with a time axis every ``seq``
        tensor carries it in front.
        """
        sw, act = self.resolve_external_inputs(embedding, time_feats, sleep_wake, activity)
        externals = {
            "as": torch.stack([act, sw], dim=-1),
            "sa": torch.stack([sw, act], dim=-1),
            "s": sw.unsqueeze(-1),
        }
        const: dict[str, dict[str, torch.Tensor]] = {}
        seq: dict[str, dict[str, torch.Tensor]] = {}
        for name, mod in self._modules_by_name.items():
            emb = self.embedding_projections[name](embedding)
            const[name], seq[name] = mod.prepare(externals[_MODULE_EXTERNAL[name]], emb, time_feats)
        return const, seq

    def step(
        self,
        state: torch.Tensor,
        gut: torch.Tensor,
        duodenal: torch.Tensor,
        prepared: dict[str, dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Rates ``[B, STATE_DIM]`` for the raw ``state[B, STATE_DIM]``, this step's
        ``gut[B, 4]`` / ``duodenal[B, 3]``, and the merged per-step ``prepared``."""
        norm_state = (state - self.norm_center) / self.norm_scale
        full = torch.cat([norm_state, gut, duodenal], dim=-1)
        rates = [
            mod.step(full.index_select(-1, getattr(self, f"_xidx_{name}")), prepared[name])
            for name, mod in self._modules_by_name.items()
        ]
        return torch.cat(rates, dim=-1).index_select(-1, self._rate_order)

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
    ) -> torch.Tensor:
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

        if sleep_wake is not None and sleep_wake.dim() == 0:
            sleep_wake = sleep_wake.expand(batch)
        if activity is not None and activity.dim() == 0:
            activity = activity.expand(batch)

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
            emb_gut = self.embedding_projections["gut"](embedding.reshape(-1, embedding.shape[-1])[0])
            gut_out = self.gut(float(t_minutes[0].item()), meals, emb_gut)
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

        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        if embedding.shape[0] != batch:
            embedding = embedding.expand(batch, -1)
        const, seq = self.prepare(embedding, time_feats, sleep_wake, activity)
        prepared = {name: {**const[name], **seq[name]} for name in const}
        rates = self.step(state, gut_out_batch, duo, prepared)

        if is_unbatched:
            rates = rates.squeeze(0)

        return rates


def _advance(
    state: torch.Tensor,
    rates: torch.Tensor,
    dt: float,
    phys_min: torch.Tensor,
    phys_max: torch.Tensor,
    keep: Optional[torch.Tensor],
) -> torch.Tensor:
    """The Euler step of ``integrate``, clamped, with frozen rows held (``keep``)."""
    new_state = euler_step(state, rates, dt)
    # Physiological clamp (iter 81): a no-op in-distribution, it bounds
    # off-manifold divergence so a runaway term cannot integrate a marker to
    # nonphysical magnitudes. Straight-through: the FORWARD state is clamped
    # (so a blow-up never propagates and the benchmark/no-grad path is hard-
    # bounded), but the backward pass treats the clamp as the identity, so it
    # never zeros the gradient at the boundary — early training (when random
    # rollouts transiently hit the bounds) is not starved. In-distribution
    # the clamp never binds, so this is exactly `state + rates*dt`.
    clamped = torch.clamp(new_state, phys_min, phys_max)
    new_state = new_state + (clamped - new_state).detach()
    if keep is not None:
        new_state = torch.where(keep, new_state, state)
    return new_state


def _model_step(
    model: "ModularPhysiologyNetwork",
    state: torch.Tensor,
    gut: torch.Tensor,
    duodenal: torch.Tensor,
    prepared: dict[str, dict[str, torch.Tensor]],
    dt: float,
    phys_min: torch.Tensor,
    phys_max: torch.Tensor,
    keep: Optional[torch.Tensor],
) -> torch.Tensor:
    """One whole Euler step of a two-phase model: everything the loop repeats."""
    return _advance(state, model.step(state, gut, duodenal, prepared), dt, phys_min, phys_max, keep)


# Opt-in ``torch.compile`` of ``_model_step`` (``PULSE_COMPILE_STEP=1`` or
# ``set_step_compilation(True)``; ``python -m pulse.train --compile-step``). The step
# is ~600 small ops forward and as many autograd nodes; compiled, they fuse into a
# handful of kernels each way — ~4-5x per step on CPU (docs/training-efficiency.md).
# The price is a one-off compile (minutes) per distinct step signature (grad mode,
# which inputs require grad, frozen rows or not — the batch is dynamic, and a batch
# of 1 runs as 2, see ``integrate``), and a C++ compiler at runtime. Any failure to
# compile falls back to eager, once, loudly.
_STEP_COMPILE: dict[str, object] = {
    "enabled": os.environ.get("PULSE_COMPILE_STEP", "0") == "1",
    "fn": None,
    "failed": False,
}


def set_step_compilation(enabled: bool) -> None:
    """Turn compilation of the integrator's step on or off for this process."""
    _STEP_COMPILE["enabled"] = bool(enabled)


def _compiled_step():
    if not _STEP_COMPILE["enabled"] or _STEP_COMPILE["failed"]:
        return None
    if _STEP_COMPILE["fn"] is None:
        import torch._dynamo

        # One graph per step signature; a training run has a few dozen.
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
        _STEP_COMPILE["fn"] = torch.compile(_model_step, dynamic=True)
    return _STEP_COMPILE["fn"]


def _run_step(*args) -> torch.Tensor:
    fn = _compiled_step()
    if fn is None:
        return _model_step(*args)
    try:
        return fn(*args)
    except Exception as exc:  # noqa: BLE001 — any compile failure means "run eager"
        _STEP_COMPILE["failed"] = True
        warnings.warn(
            f"torch.compile of the integrator step failed ({type(exc).__name__}: {exc}); "
            "continuing in eager mode",
            RuntimeWarning,
        )
        return _model_step(*args)


def _steps_first(
    series: Optional[torch.Tensor], n_steps: int, batch: int, name: str,
) -> Optional[torch.Tensor]:
    """An input series as ``[T, B]``: ``[T]`` is shared by every member, ``[B, T]``
    is per member (the layout of ``gut_outputs``). ``None`` stays ``None``."""
    if series is None:
        return None
    if series.dim() == 1:
        return series[:n_steps].unsqueeze(1).expand(n_steps, batch)
    if series.dim() == 2 and series.shape[0] == batch:
        return series[:, :n_steps].transpose(0, 1)
    raise ValueError(f"{name}: expected [T] or [B={batch}, T], got {tuple(series.shape)}")


def integrate(
    model: ModularPhysiologyNetwork,
    initial_state: torch.Tensor,
    embedding: torch.Tensor,
    n_steps: int,
    dt: float = 1.0,
    start_time_minutes: float | torch.Tensor = 360.0,
    meals: Optional[list[MealEvent]] = None,
    sleep_wake: Optional[torch.Tensor] = None,
    activity: Optional[torch.Tensor] = None,
    gut_outputs: Optional[torch.Tensor] = None,
    checkpoint_segments: int = 0,
    duodenal_outputs: Optional[torch.Tensor] = None,
    active_steps: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Integrate the modular ODE forward in time.

    Shape-polymorphic on the leading dim:

    - Unbatched (``initial_state[STATE_DIM]``, ``embedding[EMB]``,
      ``gut_outputs[T, OUT]``) → ``[T, STATE_DIM]``
    - Batched (``initial_state[B, STATE_DIM]``, ``embedding[B, EMB]``,
      ``gut_outputs[B, T, OUT]``) → ``[B, T, STATE_DIM]``

    The batch may be heterogeneous: every per-step input can be shared or per
    member. ``meals`` is shared (it only feeds the gut / duodenal precompute; a
    batch of different meal plans passes its own ``gut_outputs`` /
    ``duodenal_outputs`` instead — see ``pulse.rollouts``).

    start_time_minutes: minute of day at step 0 — a float, or ``[B]`` per member.
    sleep_wake / activity: ``[T]`` shared, ``[B, T]`` per member, or None. Missing
        values (``None``, or ``NaN`` entries) take the model's learned default
        (``ModularPhysiologyNetwork.resolve_external_inputs``).
    gut_outputs: [T, GUT_OUTPUT_DIM] or [B, T, GUT_OUTPUT_DIM] or None (computed
        from ``meals`` — see :func:`precompute_gut_outputs`; zero without meals).
    duodenal_outputs: [T, 3] or [B, T, 3] duodenal (fat, protein, carb) delivery on the
        WINDOW-OFFSET clock, as returned by :func:`precompute_duodenal_outputs`, or None
        to compute it here from ``meals`` (iter 97 — mirrors ``gut_outputs`` so a
        training signal that assembles its own window, e.g. the distillation's level
        anchors, can hand the biliary axis its meal stimulus instead of zeros; review
        item 4.7).
    active_steps: ``[B]`` — row ``b`` evolves for its first ``active_steps[b]`` states
        and then holds its last one, so protocols of different lengths can share one
        batched rollout (inputs padded to ``n_steps``). A row's states up to its own
        length are exactly what a rollout of that length alone produces; the held
        tail is a constant, so nothing past a row's end reaches its gradient.

    Everything of the vector field that does not read the ODE state — setpoints,
    circadian drives, input projections, learned input defaults — is computed ONCE
    for the whole window by ``model.prepare`` and sliced per step; the Euler loop
    runs only ``model.step``. Any other rate model (``forward(state, embedding, t,
    meals, ...)``) is stepped through ``forward``.

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
    device = initial_state.device
    if embedding.dim() == 2 and embedding.shape[0] != batch:
        embedding = embedding.expand(batch, -1)
    # A compiled step never sees a batch of 1: a single rollout runs as two identical
    # rows and returns the first. Inductor's batch-1 specialization of the step
    # miscomputes some weight gradients (torch 2.12: the metabolic heads' gut-
    # appearance column; ``aot_eager`` and every batch >= 2 graph are exact). The twin
    # row carries no gradient (nothing reads it) and width is free; this also means
    # one compiled graph per step signature instead of two.
    single_twin = (
        batch == 1 and isinstance(model, ModularPhysiologyNetwork) and _compiled_step() is not None
    )
    if single_twin:
        batch = 2
        initial_state = initial_state.expand(2, -1)
        embedding = embedding.expand(2, -1)
        if sleep_wake is not None and sleep_wake.dim() == 2:
            sleep_wake = sleep_wake.expand(2, -1)
        if activity is not None and activity.dim() == 2:
            activity = activity.expand(2, -1)
        if active_steps is not None:
            active_steps = torch.as_tensor(active_steps, device=device).reshape(-1).expand(2)
    if gut_outputs is None and meals:
        # Precompute the whole gut window here so BOTH the batched and the
        # single-embedding paths take meal absorption on the window-offset
        # clock (see precompute_gut_outputs). Previously batch==1 fell through
        # to per-step ``model.forward`` gut calls that computed meal-dt from
        # absolute minute-of-day, re-introducing the start_time_minutes meal
        # shift on exactly the benchmark path (which calibrates then integrates
        # a single embedding). Precomputing centralizes the correct timebase.
        gut_outputs = precompute_gut_outputs(
            model, embedding, n_steps, dt=dt,
            start_time_minutes=0.0, meals=meals,
        )
        if gut_outputs.dim() == 2:
            gut_outputs = gut_outputs.unsqueeze(0)

    # Duodenal delivery (gastric emptying) for the whole window, on the WINDOW-OFFSET
    # clock — the same frame contract as the gut precompute above, and for the same
    # reason: meal.time is a window offset while the model's t_minutes is absolute time
    # of day, and mixing them is the iter-87 bug. Embedding-independent, so one call
    # covers every batch member.
    duo_outputs = duodenal_outputs
    if duo_outputs is None and meals and hasattr(model, "duodenal"):
        duo_outputs = precompute_duodenal_outputs(model, n_steps, dt=dt, meals=meals)

    # The clock: absolute minute of day per step (and per member when the start is).
    steps = torch.arange(n_steps, dtype=torch.float64, device=device) * dt
    start = torch.as_tensor(start_time_minutes, dtype=torch.float64, device=device).reshape(-1)
    t_abs = ((start.unsqueeze(0) + steps.unsqueeze(1)) % 1440.0).to(torch.float32)  # [T, 1|B]
    sw = _steps_first(sleep_wake, n_steps, batch, "sleep_wake")
    act = _steps_first(activity, n_steps, batch, "activity")

    def per_step(x: Optional[torch.Tensor], width: int, shared_2d: bool) -> list[torch.Tensor]:
        """[B|1, T, width] (or [T, width] when ``shared_2d``) → T tensors [B, width];
        zeros when absent. Laid out time-major first, so each step's slice is one
        contiguous block rather than a strided view across the whole window."""
        if x is None:
            zero = torch.zeros(batch, width, dtype=initial_state.dtype, device=device)
            return [zero] * n_steps
        if shared_2d and x.dim() == 2:
            x = x.unsqueeze(0)
        x = x[:, :n_steps].expand(batch, n_steps, width).transpose(0, 1).contiguous()
        return list(x.unbind(0))

    gut_steps = per_step(gut_outputs, GUT_OUTPUT_DIM, shared_2d=False)
    duo_steps = per_step(duo_outputs, DUODENAL_DIM, shared_2d=True)

    if isinstance(model, ModularPhysiologyNetwork):
        time_feats = compute_time_features(t_abs).expand(n_steps, batch, TIME_FEATURES_DIM)
        const, seq = model.prepare(embedding, time_feats, sw, act)
        seq_steps: dict[str, dict[str, tuple[torch.Tensor, ...]]] = {}
        for name, entries in seq.items():
            seq_steps[name] = {}
            for key, value in entries.items():
                if value.dim() < 2 or value.shape[0] != n_steps:
                    raise ValueError(
                        f"{name}.prepare: seq entry {key!r} has shape {tuple(value.shape)}; "
                        f"expected a leading time axis of {n_steps}",
                    )
                seq_steps[name][key] = value.unbind(0)
        phys_min, phys_max = model._phys_min, model._phys_max
    else:
        phys_min, phys_max = _PHYS_MIN.to(device), _PHYS_MAX.to(device)

    keep_steps: Optional[list[torch.Tensor]] = None
    if active_steps is not None:
        active_steps = torch.as_tensor(active_steps, device=device).reshape(1, batch, 1)
        steps_idx = torch.arange(n_steps, device=device).reshape(n_steps, 1, 1)
        keep_steps = list((steps_idx + 1 < active_steps).unbind(0))  # [B, 1] per step

    def step_fn(state: torch.Tensor, step: int) -> torch.Tensor:
        keep = keep_steps[step] if keep_steps is not None else None
        if isinstance(model, ModularPhysiologyNetwork):
            prepared = {
                name: {**const[name], **{k: v[step] for k, v in seq_steps[name].items()}}
                for name in const
            }
            return _run_step(
                model, state, gut_steps[step], duo_steps[step], prepared,
                dt, phys_min, phys_max, keep,
            )
        rates = model(
            state, embedding, t_abs[step].expand(batch), meals,
            sleep_wake=sw[step] if sw is not None else None,
            activity=act[step] if act is not None else None,
            gut_override=gut_steps[step],
            duodenal_override=duo_steps[step],
        )
        return _advance(state, rates, dt, phys_min, phys_max, keep)

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
        if single_twin:
            out = out[:1]
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

    from torch.utils.checkpoint import checkpoint as _ckpt, set_checkpoint_early_stop

    state = initial_state
    chunk_outputs: list[torch.Tensor] = []
    # A compiled step inside a non-reentrant checkpoint trips torch's recompute
    # early-stop (an internal assertion in its pack hook, torch 2.12). The backward
    # needs every recomputed tensor of the chunk anyway, so early stop saves nothing.
    with set_checkpoint_early_stop(_compiled_step() is None):
        for chunk_idx in range((n_steps + chunk_size - 1) // chunk_size):
            start_t = torch.tensor(chunk_idx * chunk_size, device=device, dtype=torch.long)
            # use_reentrant=False is the modern checkpoint API; required when the
            # chunk's inputs include non-Tensor closures (model, meals) and we
            # want clean backward semantics without rng-warning noise.
            state, chunk_states = _ckpt(chunk_fn, state, start_t, use_reentrant=False)
            chunk_outputs.append(chunk_states)
    out = torch.cat(chunk_outputs, dim=1)  # [B, T, STATE_DIM]
    if single_twin:
        out = out[:1]
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
