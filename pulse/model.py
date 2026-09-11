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


def euler_step(state: torch.Tensor, rates: torch.Tensor, dt: float) -> torch.Tensor:
    """One forward-Euler step of the raw state, with the positive-by-construction
    markers (HRV, pulse pressure) stepped multiplicatively and SpO₂ stepped in
    logit coordinates of (70, 100). Shared by ``integrate`` and anyone who steps
    the model by hand."""
    new_state = state + rates * dt
    hrv_new = _exp_step(state[..., _HRV_IDX], rates[..., _HRV_IDX], dt)
    pp = state[..., _SBP_IDX] - state[..., _DBP_IDX]
    pp_new = _exp_step(pp, rates[..., _SBP_IDX] - rates[..., _DBP_IDX], dt)
    dbp_new = new_state[..., _DBP_IDX]
    spo2_new = _logit_interval_step(
        state[..., _SPO2_IDX], rates[..., _SPO2_IDX], dt, _SPO2_LO, _SPO2_HI)
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

        rates = torch.zeros_like(state)
        ext_as = torch.stack([act, sw], dim=-1)   # [activity, sleep_wake]
        ext_sa = torch.stack([sw, act], dim=-1)   # [sleep_wake, activity]
        ext_s = sw.unsqueeze(-1)

        module_specs = (
            ("metabolic", self._met_idx, ext_as),
            ("appetite", self._app_idx, ext_s),
            ("stress", self._str_idx, ext_sa),
            ("cardiovascular", self._cvs_idx, ext_as),
            ("thermoreg", self._thm_idx, ext_as),
            ("respiratory", self._rsp_idx, ext_as),
            ("hepatobiliary", self._hpb_idx, ext_as),
        )
        for name, idx, external in module_specs:
            coupling = self.coupling_for(name, norm_state, gut_out_batch, duo)
            mod_rates = self._modules_by_name[name](
                norm_state[:, idx], coupling, external, emb[name], time_feats,
            )
            for i, state_idx in enumerate(idx):
                rates[:, state_idx] = mod_rates[:, i]

        if is_unbatched:
            rates = rates.squeeze(0)

        return rates


def integrate(
    model: ModularPhysiologyNetwork,
    initial_state: torch.Tensor,
    embedding: torch.Tensor,
    n_steps: int,
    dt: float = 1.0,
    start_time_minutes: float = 360.0,
    meals: Optional[list[MealEvent]] = None,
    sleep_wake: Optional[torch.Tensor] = None,
    activity: Optional[torch.Tensor] = None,
    gut_outputs: Optional[torch.Tensor] = None,
    checkpoint_segments: int = 0,
    duodenal_outputs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Integrate the modular ODE forward in time.

    Shape-polymorphic on the leading dim:

    - Unbatched (``initial_state[STATE_DIM]``, ``embedding[EMB]``,
      ``gut_outputs[T, OUT]``) → ``[T, STATE_DIM]``
    - Batched (``initial_state[B, STATE_DIM]``, ``embedding[B, EMB]``,
      ``gut_outputs[B, T, OUT]``) → ``[B, T, STATE_DIM]``

    ``meals``, ``sleep_wake``, and ``activity`` are shared across batch
    members (the typical training pattern is "same protocol, different
    embeddings"). ``gut_outputs`` should be precomputed by
    :func:`precompute_gut_outputs` whenever ``B > 1`` — gut output
    depends on the embedding, and per-step ``gut.forward`` cannot be
    invoked once for a heterogeneous batch.

    sleep_wake: [n_steps] tensor or None — per-step sleep/wake state
    activity: [n_steps] tensor or None — per-step activity level
    gut_outputs: [T, GUT_OUTPUT_DIM] or [B, T, GUT_OUTPUT_DIM] or None
    duodenal_outputs: [T, 3] duodenal (fat, protein, carb) delivery on the WINDOW-OFFSET clock,
        as returned by :func:`precompute_duodenal_outputs`, or None to compute it here
        from ``meals`` (iter 97 — mirrors ``gut_outputs`` so a training signal that
        assembles its own window, e.g. the distillation's level anchors, can hand the
        biliary axis its meal stimulus instead of zeros; review item 4.7).

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
    if gut_outputs is None and meals:
        # Precompute the whole gut window here so BOTH the batched and the
        # single-embedding paths take meal absorption on the window-offset
        # clock (see precompute_gut_outputs). Previously batch==1 fell through
        # to per-step ``model.forward`` gut calls that computed meal-dt from
        # absolute minute-of-day, re-introducing the start_time_minutes meal
        # shift on exactly the benchmark path (which calibrates then integrates
        # a single embedding). Precomputing centralizes the correct timebase.
        # Guarded on ``meals``: with no meals there is no absorption to compute,
        # and skipping keeps gut-less stub models (used in integrator tests)
        # and the trivial zero-gut path working unchanged.
        gut_outputs = precompute_gut_outputs(
            model, embedding, n_steps, dt=dt,
            start_time_minutes=start_time_minutes, meals=meals,
        )
        if gut_outputs.dim() == 2:
            gut_outputs = gut_outputs.unsqueeze(0)
    elif batch > 1 and gut_outputs is None:
        # Preserve the original contract: a heterogeneous batch cannot use the
        # per-step gut path (gut depends on the embedding), so callers must
        # precompute. Only reachable now for the no-meal batch>1 case.
        raise ValueError(
            "integrate at batch>1 requires precomputed gut_outputs; call "
            "precompute_gut_outputs(model, embedding, n_steps, ...) first.",
        )

    device = initial_state.device

    def step_fn(state: torch.Tensor, step: int) -> torch.Tensor:
        t = torch.tensor(
            [(start_time_minutes + step * dt) % 1440.0],
            device=device,
        ).expand(batch)
        sw_step = sleep_wake[step].expand(batch) if sleep_wake is not None else None
        act_step = activity[step].expand(batch) if activity is not None else None
        gut_step = gut_outputs[:, step] if gut_outputs is not None else None
        duo_step = duo_outputs[step] if duo_outputs is not None else None
        rates = model(
            state, embedding, t, meals,
            sleep_wake=sw_step, activity=act_step,
            gut_override=gut_step,
            duodenal_override=duo_step,
        )
        # Physiological clamp (iter 81): a no-op in-distribution, it bounds
        # off-manifold divergence so a runaway term cannot integrate a marker to
        # nonphysical magnitudes. Straight-through: the FORWARD state is clamped
        # (so a blow-up never propagates and the benchmark/no-grad path is hard-
        # bounded), but the backward pass treats the clamp as the identity, so it
        # never zeros the gradient at the boundary — early training (when random
        # rollouts transiently hit the bounds) is not starved. In-distribution
        # the clamp never binds, so this is exactly `state + rates*dt`.
        new_state = euler_step(state, rates, dt)
        clamped = torch.clamp(
            new_state,
            _PHYS_MIN.to(new_state.device),
            _PHYS_MAX.to(new_state.device),
        )
        return new_state + (clamped - new_state).detach()

    # Duodenal delivery (gastric emptying) for the whole window, on the WINDOW-OFFSET
    # clock — the same frame contract as the gut precompute above, and for the same
    # reason: meal.time is a window offset while the model's t_minutes is absolute time
    # of day, and mixing them is the iter-87 bug. Embedding-independent, so one call
    # covers every batch member.
    duo_outputs = duodenal_outputs
    if duo_outputs is None and meals and hasattr(model, "duodenal"):
        duo_outputs = precompute_duodenal_outputs(model, n_steps, dt=dt, meals=meals)

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
