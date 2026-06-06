"""
Stress / HPA Axis module.

Iter 64 introduced "Move B compact" — mechanistic priors that closed the
EMBEDDING_DIM=64 shortcut directions on ACTH and cortisol. The plain
SetpointHead architecture lets the patient embedding bypass the canonical
HPA cascade — ACTH and cortisol fit as semi-independent free-floating
species via embedding shortcut directions, surfacing as the 0.65 / 0.49
mape regression that iter 61's eval-time diag-Gauss prior tried (and failed)
to fix at eval time.

Iter 64's mechanism (four sign-constrained terms on top of SetpointHead):

    cortisol_rate +=  α * relu((acth - typical_acth) / typical_acth) * prod_scale_cortisol
                  +   δ * (1 + diurnal_carrier(t, phase(emb)))      * prod_scale_cortisol
    acth_rate     +=  γ * (1 + diurnal_carrier(t, phase(emb)))      * prod_scale_acth
                  −   β * relu((cortisol - typical_cortisol) / typical_cortisol) * prod_scale_acth

with α, β, γ, δ ≥ 0 (softplus parameterised) and ``phase`` projected from
the embedding. Per-patient diurnal phase carries population-relative
OFFSET; the SetpointHead's time-feature input supplies the population-
level phase. The relu makes the prior surgical: baseline behaviour stays
with the SetpointHead, the architectural prior fires only when ACTH is
above typical (→ cortisol drive) or cortisol is above typical (→ ACTH
feedback). Iter 66 added the δ cosinor on cortisol — the adrenal cortex
has a direct ACTH-sensitivity rhythm on top of the cascade.

**Iter 73 reverts the CRH cascade for good** — back to iter-64 compact.
The CRH cascade (``diurnal → CRH → ACTH``) has now failed THREE distinct
ways: iter 69 (full replace: ACTH 0.091 → 0.761), iter 71 (indirect: ACTH
0.0755 → 0.8331), iter 72 (additive/dormant redundancy: ACTH 0.0755 →
0.3618 — drifted, not collapsed). The root cause is structural and is not
fixable by re-parameterising the cascade: **the cold knowledge model does
not simulate CRH** (see ``knowledge/full_body.py`` — CRH is padded with
the constant typical 100.0). CRH is therefore an unsupervised latent — the
cold-model-distillation signal supervises ACTH and cortisol directly (they
ARE in the cold ODE, with diurnal curves) but never CRH. Any ``λ·crh →
ACTH`` term lets ACTH's rate be explained by a free-floating variable with
no ground truth, so gradient drags it off the iter-64-proven direct
γ·(1+diurnal) drive. Three failures confirm: do not add an unsupervised
latent into a load-bearing rate. The iter-64 compact cascade mirrors the
cold model's own two-state ACTH/cortisol structure — which is exactly why
it lands ACTH ≈ 0.0755.

CRH is retained as a typed state at idx 22 (n_species stays at 3) — this
keeps the iter-69 benchmark dataset (23-D initial_state) working without
regeneration. CRH is mechanically inert: SetpointHead-only, no
participation in the ACTH/cortisol cascade. A CRH cascade could only be
re-introduced once the cold model is taught to simulate CRH (giving it
direct distillation supervision) — until then it stays inert.

Couplings (n_coupling=2): glucose, cortisol_feedback (legacy — duplicated
by our explicit β-feedback term; kept for parity with the rest of the
coupling graph and because the cohort generator still supplies it).
External inputs (n_external=2): sleep_wake, activity.
"""

import torch
import torch.nn as nn

from .base import MassActionModule, SetpointHead

# Coupling inputs: glucose (1) + cortisol feedback (1)
_N_COUPLING = 2

# External inputs: sleep_wake (1) + activity (1)
_N_EXTERNAL = 2

# Order matches MODULE_MARKER_INDICES["stress"] — defined by the order of
# System.STRESS entries in pulse.types.MARKERS. Currently: cortisol (idx
# 10), acth (idx 11), crh (idx 22 — internal tail). The module receives
# state[:, [10, 11, 22]] which becomes state[:, 0:3] locally with the
# below indices.
_CORTISOL_IDX = 0
_ACTH_IDX = 1
_CRH_IDX = 2

# Typicals (μg/dL for cortisol; pg/mL for ACTH and CRH).
_TYPICALS = [12.0, 30.0, 100.0]
# cons_scale per species (inverse time-constant proxy; SetpointHead uses
# this as base decay).
_CONS_SCALES = [0.02, 0.03, 0.04]

_MECHANISM_INIT_RAW = -3.0


class StressModule(MassActionModule):
    def __init__(self, embedding_dim: int, hidden_dim: int = 32):
        super().__init__(
            n_species=3,
            n_coupling=_N_COUPLING,
            n_external=_N_EXTERNAL,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            typicals=_TYPICALS,
            head_factories={
                _CORTISOL_IDX: lambda inp, hd: SetpointHead(inp, hd, typical=_TYPICALS[_CORTISOL_IDX]),
                _ACTH_IDX: lambda inp, hd: SetpointHead(inp, hd, typical=_TYPICALS[_ACTH_IDX]),
                _CRH_IDX: lambda inp, hd: SetpointHead(inp, hd, typical=_TYPICALS[_CRH_IDX]),
            },
        )
        prod_scales = [c * t for c, t in zip(_CONS_SCALES, _TYPICALS)]
        self.prod_scale.copy_(torch.tensor(prod_scales, dtype=torch.float32))
        self.cons_scale.copy_(torch.tensor(_CONS_SCALES, dtype=torch.float32))

        # Iter 64 compact mechanism — restored after iter 69's Move B FULL
        # regressed ACTH (0.091→0.761) by diluting the direct γ·diurnal drive.
        self._alpha_raw = nn.Parameter(torch.tensor(_MECHANISM_INIT_RAW))   # ACTH → cortisol drive
        self._delta_raw = nn.Parameter(torch.tensor(_MECHANISM_INIT_RAW))   # diurnal → cortisol drive (iter 66)
        self._gamma_raw = nn.Parameter(torch.tensor(_MECHANISM_INIT_RAW))   # diurnal → ACTH drive
        self._beta_raw = nn.Parameter(torch.tensor(_MECHANISM_INIT_RAW))    # cortisol → ACTH neg feedback

        # Per-patient diurnal phase (scalar) projected from the embedding.
        # Carries population-relative OFFSET; the SetpointHead's time-feature
        # input supplies the population-level phase.
        self.phase_proj = nn.Linear(embedding_dim, 1)
        nn.init.normal_(self.phase_proj.weight, std=0.01)
        nn.init.zeros_(self.phase_proj.bias)

    def forward(
        self,
        state: torch.Tensor,
        coupling: torch.Tensor,
        external: torch.Tensor,
        embedding: torch.Tensor,
        time_features: torch.Tensor,
    ) -> torch.Tensor:
        base_rate = super().forward(state, coupling, external, embedding, time_features)

        cortisol_raw = state[..., _CORTISOL_IDX]
        acth_raw = state[..., _ACTH_IDX]
        typical_cortisol = float(_TYPICALS[_CORTISOL_IDX])
        typical_acth = float(_TYPICALS[_ACTH_IDX])

        acth_excess = torch.relu((acth_raw - typical_acth) / typical_acth)
        cortisol_excess = torch.relu((cortisol_raw - typical_cortisol) / typical_cortisol)

        # Diurnal carrier with per-patient phase offset.
        # time_features[..., 1] = sin(2π·hour/24); [..., 2] = cos(2π·hour/24).
        phase = self.phase_proj(embedding).squeeze(-1)
        sin_t = time_features[..., 1]
        cos_t = time_features[..., 2]
        diurnal = sin_t * torch.cos(phase) - cos_t * torch.sin(phase)
        diurnal_carrier = 1.0 + diurnal  # range [0, 2]

        alpha = nn.functional.softplus(self._alpha_raw)
        delta = nn.functional.softplus(self._delta_raw)
        gamma = nn.functional.softplus(self._gamma_raw)
        beta = nn.functional.softplus(self._beta_raw)

        cortisol_drive = alpha * acth_excess * self.prod_scale[_CORTISOL_IDX]
        cortisol_diurnal = delta * diurnal_carrier * self.prod_scale[_CORTISOL_IDX]
        acth_drive = gamma * diurnal_carrier * self.prod_scale[_ACTH_IDX]
        acth_feedback = -beta * cortisol_excess * self.prod_scale[_ACTH_IDX]

        adjustments = torch.zeros_like(base_rate)
        adjustments[..., _CORTISOL_IDX] = cortisol_drive + cortisol_diurnal
        adjustments[..., _ACTH_IDX] = acth_drive + acth_feedback
        # _CRH_IDX: no mechanism adjustment — inert this iter.

        return base_rate + adjustments
