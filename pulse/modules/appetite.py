"""
Appetite & Satiety module.

Hunger/fullness signaling: ghrelin, leptin, GLP-1.
Uses mass-action kinetics. Receives insulin from Metabolic, and from Gut both the
nutrient-sensing flag and the (dose-linear) glucose-appearance flux.

Iter 90 — DOSE BLINDNESS FIX. Through iter 89 this module's only meal input was
``nutrient_flag``, whose presence gate is ``1 - exp(-total_macro / 10 g)``. That
SATURATES: a 30 g-carb meal gives 0.9889 and a 90 g meal 0.99997 — a 1.1% difference.
The module was structurally unable to perceive meal SIZE. Its only other coupling,
insulin, does scale with dose, and appetite hormones are insulin-SUPPRESSED — so a
bigger meal produced LESS GLP-1. Measured on the iter-89 model:
``glp1_dose_response = -0.656`` (wrong-signed; the textbook meal_dose_response check
has been failing on it). The gut kernel already emits a dose-LINEAR glucose-appearance
flux by construction (macros @ K, see base.GutModuleBase), so the information existed
and simply was not routed here. It now is, and GLP-1's peak gate reads it rather than
the saturating flag — which is also the anatomically correct stimulus: intestinal
L-cells secrete GLP-1 in response to nutrient delivery rate, not to a binary fed flag.
"""

import torch

from .base import BasalPlusGatedPeakHead, MassActionModule, SpeciesHead
from ..types import MODULE_MARKER_INDICES, NORM_SCALE

# Coupling inputs: insulin (1) + nutrient_flag (1) + gut glucose_appearance (1) = 3
_N_COUPLING = 3

# External inputs: sleep_wake (1)
_N_EXTERNAL = 1

_TYPICALS = [100.0, 10.0, 10.0]
# Iter 95: NORM_SCALE per species, for the raw-concentration consumption term.
_NORM_SCALES = [NORM_SCALE[i] for i in MODULE_MARKER_INDICES["appetite"]]
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
            norm_scales=_NORM_SCALES,
            # SetpointHead needs typical to map target_z (z-score units) ↔
            # prod (see modules/base.py). Factory closure captures it from
            # _TYPICALS so the head construction stays via head_factories.
            head_factories={
                # Iter 95: was SetpointHead — unnecessary in the corrected concentration
                # frame (see modules/base.py:MassActionModule).
                _GHRELIN_IDX: SpeciesHead,
                # Iter 53: GLP-1 is meal-driven (sharp postprandial peaks) — iter 52's
                # SetpointHead regressed it (0.276→0.421) because target_z can drift
                # but can't fire a peak. BasalPlusGatedPeakHead fires a gated peak.
                #
                # Iter 90: the gate stimulus moves from nutrient_flag (coupling[1],
                # head-input index n_species+1 = 4) to the gut GLUCOSE-APPEARANCE flux
                # (coupling[2], index n_species+2 = 5). nutrient_flag saturates at ~1
                # for any real meal, so gating on it made both the gate AND the
                # MLP-derived peak amplitude dose-blind; appearance is dose-linear by
                # construction, so peak amplitude can now scale with meal size (and
                # the dose-response rank signal has a gradient that can move it).
                # Appearance is a ≥0 flux, ≈0 fasted and O(0.1-3) during absorption,
                # so the threshold sits just above zero with a sharp gate.
                _GLP1_IDX: lambda inp, hd: BasalPlusGatedPeakHead(
                    inp, hd, stimulus_idx=3 + 2, gate_dir=1,
                    init_thresh=0.1, init_log_temp=-1.6,  # exp(-1.6) ≈ 0.2 sharp gate
                ),
            },
        )
        prod_scales = [c * t for c, t in zip(_CONS_SCALES, _TYPICALS)]
        self.prod_scale.copy_(torch.tensor(prod_scales, dtype=torch.float32))
        self.cons_scale.copy_(torch.tensor(_CONS_SCALES, dtype=torch.float32))
