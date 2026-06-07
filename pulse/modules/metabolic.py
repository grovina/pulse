"""
Metabolic / Energy module.

Blood chemistry homeostasis: glucose, insulin, glucagon, FFA, BHB, lactate,
and hepatic glucose output (endogenous appearance flux). Uses mass-action
kinetics. Receives nutrient appearance from Gut and cortisol from Stress.
"""

import math

import torch
import torch.nn as nn

from .base import BasalPlusGatedPeakHead, MassActionModule, SetpointHead, SpeciesHead, gate_temp
from ..types import GUT_OUTPUT_DIM

# Coupling inputs: gut outputs (4) + cortisol (1) = 5
_N_COUPLING = GUT_OUTPUT_DIM + 1

# External inputs: activity (1) + sleep_wake (1) = 2
_N_EXTERNAL = 2

# Species order matches MODULE_MARKER_INDICES["metabolic"]:
# 0: glucose, 1: insulin, 2: glucagon, 3: ffa, 4: bhb, 5: lactate,
# 6: hepatic_output, 7: liver_glycogen (iter 56), 8: muscle_glycogen
# (iter 56), 9: mitochondrial_capacity (iter 55).
# Indices 7-9 are slow internal states (unobserved, marker_type="internal"
# in types.py) — physical pool size + protein turnover give them long τ
# naturally under the same mass-action primitive. See
# docs/multi-timescale-plan.md.
#
# Iter 56: the iter-55 lumped glycogen_pool (500 g, one τ ≈ 10 d) could
# not express a −60 g / 1-day fast delta. Split by tissue: liver pool
# (~100 g, τ ≈ 1 d — overnight-depleting) vs muscle pool (~400 g,
# τ ≈ 3 wk — rest-preserved, exercise-coupled). Now one cons_scale per
# tissue, each physically honest.
_TYPICALS = [95.0, 10.0, 70.0, 0.5, 0.1, 1.0, 2.0, 100.0, 400.0, 1.0]
# cons_scale = 1/τ in the rate equation.
#   liver_glycogen      τ ≈ 1 day  ≈ 1440 min  → cons_scale ≈ 7e-4
#   muscle_glycogen     τ ≈ 3 weeks ≈ 30240 min → cons_scale ≈ 3.3e-5
#   mitochondrial_cap.  τ ≈ 4 weeks ≈ 40320 min → cons_scale ≈ 2.5e-5
# These scales encode the slow timescale physically — the mass-action
# equation `rate = prod - cons·state` evolves them slowly without any
# special "slow integrator" mechanism. Liver's τ ≈ 1 d makes a −60 g
# delta reachable within the 1-day EXTENDED_FAST_GLYCOGEN protocol.
_CONS_SCALES = [0.02, 0.1, 0.03, 0.04, 0.03, 0.02, 0.04, 7e-4, 3.3e-5, 2.5e-5]
_INSULIN_IDX = 1
_GLUCOSE_IDX = 0
# Iter 51 dead-pathway species. The (prod, cons) parameterisation pins their
# state at `typical` with a flat gradient surface — see docs/dead-pathways.md
# and modules/base.py:SetpointHead.
_GLUCAGON_IDX = 2
_FFA_IDX = 3
# Iter 52 collateral-victim: bhb regressed from 0.214 → 0.346 when iter-51
# moved the embedding to fit ffa/glucagon. Same softplus-saturation trap; same
# fix. See docs/architecture-roadmap.md "Move A".
_BHB_IDX = 4
# Iter 55/56 slow internal states. SetpointHead for the saturation-free
# equilibrium-at-typical coordinate; the slow τ comes from cons_scale (small).
_LIVER_GLYCOGEN_IDX = 7
_MUSCLE_GLYCOGEN_IDX = 8
_MITO_IDX = 9

# GlycogenFluxHead input indices (iter 57). The head sees the module
# input x = cat([state(10), coupling(5), external(2), embedding, time]).
#   coupling[0] = gut glucose-appearance  → x index 10  (n_species + 0)
#   metabolic external = [activity, sleep_wake] (model.py met_external)
#     → activity = x index 15  (n_species + _N_COUPLING + 0)
# 10 == n_species for this module (see MetabolicModule below).
_N_SPECIES = 10
_GLUCOSE_APPEARANCE_X_IDX = _N_SPECIES + 0
_ACTIVITY_X_IDX = _N_SPECIES + _N_COUPLING + 0


class GlucoseGatedInsulinHead(nn.Module):
    """Insulin production = basal floor + glucose-gated peak amplitude.

    Iter-23 intervention C. The shared ``SpeciesHead`` MLP can in
    principle interpolate between low-basal (≈7 µU/mL at fasting
    glucose) and high-peak (≈60 µU/mL post-OGTT) by learning a
    glucose-shaped activation through its tanh layers, but iter-22
    showed the optimizer didn't find it: trained basal sat at 18 µU/mL
    while peak sat at 39 µU/mL — both wrong, in opposite directions.
    A single learned scale cannot satisfy basal-low *and* peak-high.

    This head structurally separates the two by emitting three logits:
    ``raw_basal``, ``raw_peak``, ``raw_cons``. Production is

        prod = softplus(raw_basal) + softplus(raw_peak) · σ((g - g_thresh) / g_temp)

    where ``g`` is the module's normalized glucose state, and
    ``g_thresh`` / ``g_temp`` are learnable scalars. At low glucose
    the gate is ≈0 and only the basal term survives; at high glucose
    the gate saturates and peak amplitude lands on top of basal.
    Consumption stays a single softplus output. Both basal and peak
    pass through the parent module's ``prod_scale`` exactly like the
    standard head, so the only behavioral change is the structural
    decoupling of basal and peak amplitudes.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3),
        )
        # Glucose enters the module as state[..., glucose_idx] already
        # normalized to (g - typical) / scale. With NORM_CENTER[glucose]=95
        # and NORM_SCALE[glucose]=30, fasting (~95 mg/dL) maps to 0,
        # OGTT peak (~150 mg/dL) maps to ~+1.8. Init g_thresh ≈ 0.5
        # (~110 mg/dL) so the gate already differentiates fasting from
        # postprandial at the start of training. Init temperature ≈ 0.5
        # (i.e. ~15 mg/dL transition width), which gives a smooth but
        # discriminative gate — narrow enough that fasting and OGTT-peak
        # land on opposite sides, wide enough to keep the gradient alive
        # across the full physiological range.
        self.g_thresh = nn.Parameter(torch.tensor(0.5))
        self.log_g_temp = nn.Parameter(torch.tensor(-0.7))  # exp(-0.7) ≈ 0.5

    def forward(self, x: torch.Tensor, state_self: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # state_self is normalized insulin; we need normalized glucose,
        # which lives at position _GLUCOSE_IDX of the module state slice
        # — i.e. the first n_species columns of x. (See MassActionModule.)
        glucose_norm = x[..., _GLUCOSE_IDX]
        gate = torch.sigmoid((glucose_norm - self.g_thresh) / gate_temp(self.log_g_temp))
        raw = self.network(x)
        basal = nn.functional.softplus(raw[..., 0])
        peak = nn.functional.softplus(raw[..., 1])
        cons = nn.functional.softplus(raw[..., 2])
        prod = basal + peak * gate
        return prod, cons


class GlycogenFluxHead(nn.Module):
    """Glycogen as a flux integrator, not a setpoint species.

    Iters 55-56 proved a SetpointHead glycogen state cannot be driven
    off ``typical`` by the indirect EXTENDED_FAST cohort delta. Root
    cause (iter-56 cohort-ablation, identical fast-vs-fed arms): the
    liver_glycogen spec lands a metabolic gradient ~6700x weaker than
    the `glucose` spec on the same arms — supervising a small-cons_scale
    setpoint reparam through a window-mean delta starves the gradient,
    and the equilibrium-at-typical prior actively cancels depletion.

    Glycogen is physically a flux integrator: dGly/dt = synthesis −
    breakdown. This head produces (prod, cons) so the parent's
    ``prod·prod_scale − cons·cons_scale·state`` realises exactly that,
    with NO setpoint attractor:

      synthesis  = softplus(net) · σ(glucose_appearance gate)
                   — glycogen fills only while gut nutrients are being
                     absorbed. Routes the −60 g fast-vs-fed gradient
                     through the SAME strong gut-coupling pathway the
                     `glucose` cohort spec uses (the fix for the
                     diagnosed gradient starvation).
      breakdown  = softplus(net)·basal + softplus(net)·σ(catabolic gate)
                   — tissue-specific catabolic drive:
                     liver:  gate on LOW insulin (systemic fast →
                             hepatic glycogenolysis; G6Pase releases
                             glucose to blood). catabolic_dir = −1.
                     muscle: gate on activity (muscle lacks G6Pase; its
                             glycogen is spent locally during exercise,
                             preserved in a resting fast — Coppack
                             1989). catabolic_dir = +1.

    The slow-τ separation is still carried by the parent's per-species
    ``cons_scale`` (liver 7e-4 ≈ 1 d, muscle 3.3e-5 ≈ 3 wk) — unchanged
    from iter 56; only the head's (prod, cons) computation changes.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        anabolic_idx: int,
        catabolic_idx: int,
        catabolic_dir: float,
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3),
        )
        self.anabolic_idx = int(anabolic_idx)
        self.catabolic_idx = int(catabolic_idx)
        self.catabolic_dir = float(catabolic_dir)
        # Anabolic gate on gut glucose-appearance: a >=0 flux, ~0 when
        # not absorbing and O(1)+ during a meal. Threshold just above 0
        # so the gate is OFF fasted and ON while nutrients arrive; sharp
        # temperature so fed/fasted land on opposite sides from the start
        # (mirrors GlucoseGatedInsulinHead's already-discriminative init).
        self.a_thresh = nn.Parameter(torch.tensor(0.1))
        self.log_a_temp = nn.Parameter(torch.tensor(-1.6))  # exp ≈ 0.2
        # Catabolic gate on a normalized (~0-centred) state/external.
        self.c_thresh = nn.Parameter(torch.tensor(0.0))
        self.log_c_temp = nn.Parameter(torch.tensor(-0.7))  # exp ≈ 0.5

    def forward(self, x: torch.Tensor, state_self: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.network(x)
        anab_stim = x[..., self.anabolic_idx]
        catab_stim = x[..., self.catabolic_idx]
        synth_gate = torch.sigmoid(
            (anab_stim - self.a_thresh) / gate_temp(self.log_a_temp)
        )
        catab_gate = torch.sigmoid(
            self.catabolic_dir * (catab_stim - self.c_thresh) / gate_temp(self.log_c_temp)
        )
        synth = nn.functional.softplus(raw[..., 0]) * synth_gate
        break_basal = nn.functional.softplus(raw[..., 1])
        break_active = nn.functional.softplus(raw[..., 2]) * catab_gate
        prod = synth
        cons = break_basal + break_active
        return prod, cons


class MetabolicModule(MassActionModule):
    def __init__(self, embedding_dim: int, hidden_dim: int = 48):
        super().__init__(
            n_species=10,
            n_coupling=_N_COUPLING,
            n_external=_N_EXTERNAL,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            typicals=_TYPICALS,
            head_factories={
                _INSULIN_IDX: GlucoseGatedInsulinHead,
                # SetpointHead needs typical to map target_z (z-score units) ↔
                # prod. The factory closure captures the per-species typical
                # from _TYPICALS.
                # Iter 72: ROLLBACK iter 71's SetpointHead revert →
                # BasalPlusGatedPeakHead (iter-70 architecture). Iter 71's
                # bench showed the SetpointHead revert was catastrophic on
                # the coupling graph: ACTH collapsed 0.0755 → 0.8331 (an
                # 11× regression on a marker the module code never touched),
                # glucose 0.224 → 0.404, bhb 0.195 → 0.355, coupling
                # category 0.789 → 0.625. The gated heads on glucagon/FFA
                # were structurally load-bearing for HPA coupling, not just
                # a metabolic refinement — they constrain the
                # cortisol↔metabolic coupling gradients so ACTH learns the
                # iter-64 cascade cleanly. Reverting them shifted the loss
                # landscape enough to break ACTH at training landscape
                # level (the stress module code is unchanged across
                # 70/71/72). Restoring iter-70's gates is the cheapest
                # path back to overall_weighted_mape ≈ 0.10.
                _GLUCAGON_IDX: lambda inp, hd: BasalPlusGatedPeakHead(
                    inp, hd, stimulus_idx=_GLUCOSE_IDX, gate_dir=-1,
                    init_thresh=-0.3, init_log_temp=-0.7,
                ),
                _FFA_IDX: lambda inp, hd: BasalPlusGatedPeakHead(
                    inp, hd, stimulus_idx=_INSULIN_IDX, gate_dir=-1,
                    init_thresh=-0.2, init_log_temp=-0.7,
                ),
                _BHB_IDX: lambda inp, hd: SetpointHead(inp, hd, typical=_TYPICALS[_BHB_IDX]),
                # Iter 57: glycogen is a flux integrator, not a setpoint
                # species (iters 55-56 proved a SetpointHead glycogen can't
                # be driven off typical by the indirect cohort delta —
                # gradient ~6700x weaker than the glucose spec on the same
                # arms). GlycogenFluxHead routes synthesis through the gut
                # glucose-appearance coupling (the strong gradient path)
                # and breakdown through a tissue-specific catabolic gate:
                # liver on LOW insulin (systemic fast), muscle on activity
                # (exercise; preserved in a resting fast — Coppack 1989).
                _LIVER_GLYCOGEN_IDX: lambda inp, hd: GlycogenFluxHead(
                    inp, hd,
                    anabolic_idx=_GLUCOSE_APPEARANCE_X_IDX,
                    catabolic_idx=_INSULIN_IDX,
                    catabolic_dir=-1.0,
                ),
                _MUSCLE_GLYCOGEN_IDX: lambda inp, hd: GlycogenFluxHead(
                    inp, hd,
                    anabolic_idx=_GLUCOSE_APPEARANCE_X_IDX,
                    catabolic_idx=_ACTIVITY_X_IDX,
                    catabolic_dir=1.0,
                ),
                # mito stays SetpointHead: it genuinely IS a slow
                # setpoint-like adaptation variable (not a flux pool), so
                # the setpoint primitive is correct for it.
                _MITO_IDX: lambda inp, hd: SetpointHead(inp, hd, typical=_TYPICALS[_MITO_IDX]),
            },
        )
        prod_scales = [c * t for c, t in zip(_CONS_SCALES, _TYPICALS)]
        self.prod_scale.copy_(torch.tensor(prod_scales, dtype=torch.float32))
        self.cons_scale.copy_(torch.tensor(_CONS_SCALES, dtype=torch.float32))
        # Structural glucose setpoint: learned clearance gain Sg.
        # Implements dG_extra = -Sg*(G-Gb), mirroring the knowledge model's
        # explicit setpoint term. Normalised state = 0 at Gb=95 mg/dL, so the
        # correction is zero at baseline and grows linearly with deviation.
        # Init: Sg ≈ 0.02 so that at G=Gb+30 mg/dL (norm=+1) this adds
        # ~0.02 clearance units/min — comparable to the base cons_scale.
        self.log_sg = nn.Parameter(torch.tensor(math.log(0.1)))

    def forward(
        self,
        state: torch.Tensor,
        coupling: torch.Tensor,
        external: torch.Tensor,
        embedding: torch.Tensor,
        time_features: torch.Tensor,
    ) -> torch.Tensor:
        rates = super().forward(state, coupling, external, embedding, time_features)
        sg = nn.functional.softplus(self.log_sg)
        # Subtract structural setpoint from glucose rate only.
        rates_out = rates.clone()
        rates_out[..., _GLUCOSE_IDX] = rates_out[..., _GLUCOSE_IDX] - sg * state[..., _GLUCOSE_IDX]
        return rates_out
