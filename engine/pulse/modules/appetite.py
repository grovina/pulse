"""
Appetite & Satiety module.

Hunger/fullness signaling: ghrelin, leptin, GLP-1.
Uses mass-action kinetics. Receives insulin from Metabolic
and nutrient sensing flag from Gut.
"""

import torch

from .base import BasalPlusGatedPeakHead, MassActionModule, SetpointHead

# Coupling inputs: insulin (1) + nutrient_flag (1) = 2
_N_COUPLING = 2

# External inputs: sleep_wake (1)
_N_EXTERNAL = 1

_TYPICALS = [100.0, 10.0, 10.0]
_CONS_SCALES = [0.02, 0.001, 0.2]

# Ghrelin (idx 0) is in the iter-51 dead trio — re-parameterised to
# SetpointHead. See modules/base.py:SetpointHead and docs/dead-pathways.md.
_GHRELIN_IDX = 0
# Iter 52: GLP-1 (idx 2) sits at MAPE 0.275 across iters — a stuck-near-typical
# regime with stimulus-driven postprandial peaks. Same softplus-saturation
# trap. See docs/architecture-roadmap.md "Move A". Leptin (idx 1) stays on
# SpeciesHead because its current MAPE (0.026) is already well within the
# gate; the small init-equivalence risk isn't worth the upside.
_GLP1_IDX = 2


class AppetiteModule(MassActionModule):
    def __init__(self, embedding_dim: int, hidden_dim: int = 32):
        super().__init__(
            n_species=3,
            n_coupling=_N_COUPLING,
            n_external=_N_EXTERNAL,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            typicals=_TYPICALS,
            # SetpointHead needs typical to map target_z (z-score units) ↔
            # prod (see modules/base.py). Factory closure captures it from
            # _TYPICALS so the head construction stays via head_factories.
            head_factories={
                _GHRELIN_IDX: lambda inp, hd: SetpointHead(inp, hd, typical=_TYPICALS[_GHRELIN_IDX]),
                # Iter 53: GLP-1 is meal-driven (sharp postprandial peaks) — iter 52's
                # SetpointHead regressed it (0.276→0.421) because target_z can drift
                # but can't fire a peak. Switch to BasalPlusGatedPeakHead gated by
                # nutrient_flag (the coupling[1] feature, head-input index =
                # n_species + 1 = 4): peak when nutrient_flag ≈ 1 (meal present).
                _GLP1_IDX: lambda inp, hd: BasalPlusGatedPeakHead(
                    inp, hd, stimulus_idx=3 + 1, gate_dir=1,
                    init_thresh=0.3, init_log_temp=-1.6,  # exp(-1.6) ≈ 0.2 sharp gate
                ),
            },
        )
        prod_scales = [c * t for c, t in zip(_CONS_SCALES, _TYPICALS)]
        self.prod_scale.copy_(torch.tensor(prod_scales, dtype=torch.float32))
        self.cons_scale.copy_(torch.tensor(_CONS_SCALES, dtype=torch.float32))
