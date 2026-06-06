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

External inputs (sleep/wake, activity) are optional. When missing, each
module uses a learned default conditioned on embedding and time.
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from .types import (
    STATE_DIM, EMBEDDING_DIM, GUT_OUTPUT_DIM,
    MARKERS, MARKER_INDEX, MODULE_MARKER_INDICES,
    NORM_CENTER, NORM_SCALE,
)
from .modules import (
    GutModule, MetabolicModule, AppetiteModule, StressModule,
    CardiovascularModule, ThermoregModule, RespiratoryModule,
)
from .modules.base import compute_time_features
from .modules.gut import MealEvent


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

        # Learned defaults for missing external inputs
        self.default_sleep_wake = nn.Parameter(torch.tensor(0.5))
        self.default_activity = nn.Parameter(torch.tensor(0.1))

        # State indices per module
        self._met_idx = MODULE_MARKER_INDICES["metabolic"]
        self._app_idx = MODULE_MARKER_INDICES["appetite"]
        self._str_idx = MODULE_MARKER_INDICES["stress"]
        self._cvs_idx = MODULE_MARKER_INDICES["cardiovascular"]
        self._thm_idx = MODULE_MARKER_INDICES["thermoreg"]
        self._rsp_idx = MODULE_MARKER_INDICES["respiratory"]

        # Specific marker indices for coupling
        self._glucose_idx = MARKER_INDEX["glucose"]
        self._insulin_idx = MARKER_INDEX["insulin"]
        self._lactate_idx = MARKER_INDEX["lactate"]
        self._cortisol_idx = MARKER_INDEX["cortisol"]
        self._temp_idx = MARKER_INDEX["temp"]

        self.register_buffer("norm_center", torch.tensor(NORM_CENTER, dtype=torch.float32))
        self.register_buffer("norm_scale", torch.tensor(NORM_SCALE, dtype=torch.float32))

    def forward(
        self,
        state: torch.Tensor,
        embedding: torch.Tensor,
        t_minutes: torch.Tensor,
        meals: list[MealEvent],
        sleep_wake: Optional[torch.Tensor] = None,
        activity: Optional[torch.Tensor] = None,
        gut_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute rates of change for all state variables.

        state: [batch, STATE_DIM] or [STATE_DIM]
        embedding: [batch, EMBEDDING_DIM] or [EMBEDDING_DIM]
        t_minutes: [batch] or scalar — absolute time of day (mod 1440)
        meals: list of MealEvent
        sleep_wake: [batch] or None — 0=asleep, 1=awake
        activity: [batch] or None — 0=rest, 1=vigorous
        """
        is_unbatched = state.dim() == 1
        if is_unbatched:
            state = state.unsqueeze(0)
            embedding = embedding.unsqueeze(0)
            if t_minutes.dim() == 0:
                t_minutes = t_minutes.unsqueeze(0)

        batch = state.shape[0]

        # Resolve external inputs (use learned defaults when missing)
        sw = sleep_wake if sleep_wake is not None else self.default_sleep_wake.expand(batch)
        act = activity if activity is not None else self.default_activity.expand(batch)
        if sw.dim() == 0:
            sw = sw.expand(batch)
        if act.dim() == 0:
            act = act.expand(batch)

        # Normalize state for module inputs
        norm_state = (state - self.norm_center) / self.norm_scale

        # Compute time features
        time_feats = compute_time_features(t_minutes)
        if time_feats.dim() == 1:
            time_feats = time_feats.unsqueeze(0).expand(batch, -1)

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
            gut_out = self.gut(float(t_minutes[0].item()), meals, emb["gut"][0])
            gut_out_batch = gut_out.unsqueeze(0)
        else:
            raise ValueError(
                "model.forward at batch>1 requires gut_override; precompute "
                "via precompute_gut_outputs(model, embedding, n_steps, ...) "
                "and pass gut_outputs[:, step] each step.",
            )

        # --- Extract coupling values from current state ---
        glucose = norm_state[:, self._glucose_idx: self._glucose_idx + 1]
        insulin = norm_state[:, self._insulin_idx: self._insulin_idx + 1]
        lactate = norm_state[:, self._lactate_idx: self._lactate_idx + 1]
        cortisol = norm_state[:, self._cortisol_idx: self._cortisol_idx + 1]
        temperature = norm_state[:, self._temp_idx: self._temp_idx + 1]
        nutrient_flag = gut_out_batch[:, 3:4]

        # --- Compute rates per module ---
        rates = torch.zeros_like(state)

        # Metabolic: coupling = [gut_outputs(4), cortisol(1)]
        met_coupling = torch.cat([gut_out_batch, cortisol], dim=-1)
        met_external = torch.stack([act, sw], dim=-1)
        met_rates = self.metabolic(
            norm_state[:, self._met_idx],
            met_coupling, met_external,
            emb["metabolic"], time_feats,
        )
        for i, idx in enumerate(self._met_idx):
            rates[:, idx] = met_rates[:, i]

        # Appetite: coupling = [insulin(1), nutrient_flag(1)]
        app_coupling = torch.cat([insulin, nutrient_flag], dim=-1)
        app_external = sw.unsqueeze(-1)
        app_rates = self.appetite(
            norm_state[:, self._app_idx],
            app_coupling, app_external,
            emb["appetite"], time_feats,
        )
        for i, idx in enumerate(self._app_idx):
            rates[:, idx] = app_rates[:, i]

        # Stress: coupling = [glucose, cortisol]; external = sleep_wake, activity
        str_coupling = torch.cat([glucose, cortisol], dim=-1)
        str_external = torch.stack([sw, act], dim=-1)
        str_rates = self.stress(
            norm_state[:, self._str_idx],
            str_coupling, str_external,
            emb["stress"], time_feats,
        )
        for i, idx in enumerate(self._str_idx):
            rates[:, idx] = str_rates[:, i]

        # Cardiovascular: coupling = [cortisol(1), temperature(1), glucose(1), insulin(1)]
        cvs_coupling = torch.cat([cortisol, temperature, glucose, insulin], dim=-1)
        cvs_external = torch.stack([act, sw], dim=-1)
        cvs_rates = self.cardiovascular(
            norm_state[:, self._cvs_idx],
            cvs_coupling, cvs_external,
            emb["cardiovascular"], time_feats,
        )
        for i, idx in enumerate(self._cvs_idx):
            rates[:, idx] = cvs_rates[:, i]

        # Thermoreg: coupling = [glucose(1), cortisol(1)]
        thm_coupling = torch.cat([glucose, cortisol], dim=-1)
        thm_external = torch.stack([act, sw], dim=-1)
        thm_rates = self.thermoreg(
            norm_state[:, self._thm_idx],
            thm_coupling, thm_external,
            emb["thermoreg"], time_feats,
        )
        for i, idx in enumerate(self._thm_idx):
            rates[:, idx] = thm_rates[:, i]

        # Respiratory: coupling = [lactate(1)]
        rsp_coupling = lactate
        rsp_external = torch.stack([act, sw], dim=-1)
        rsp_rates = self.respiratory(
            norm_state[:, self._rsp_idx],
            rsp_coupling, rsp_external,
            emb["respiratory"], time_feats,
        )
        for i, idx in enumerate(self._rsp_idx):
            rates[:, idx] = rsp_rates[:, i]

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
    if batch > 1 and gut_outputs is None:
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
        rates = model(
            state, embedding, t, meals,
            sleep_wake=sw_step, activity=act_step,
            gut_override=gut_step,
        )
        return state + rates * dt

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
    times = (
        torch.arange(n_steps, dtype=torch.float32, device=device) * dt
        + start_time_minutes
    ) % 1440.0
    emb_gut = model.embedding_projections["gut"](embedding)
    return model.gut.forward_window(times, meals, emb_gut)
