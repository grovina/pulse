"""
Metabolic / Energy module.

Blood chemistry homeostasis: glucose, insulin, glucagon, FFA, BHB, lactate,
hepatic glucose output, the two glycogen pools, mitochondrial capacity and the
remote-insulin latent. Mass-action kinetics where the state IS a species in
equilibrium; explicit structural forms where it is not (glucose, the pools,
hepatic output, insulin, insulin action — the PRD's "relaxation (c)"). Receives nutrient
appearance from Gut, cortisol from Stress and GLP-1 from Appetite.
Lipid appearance is an FFA source and amino appearance a glucagon source
(the teacher's ``0.01 * Ra_fat`` / ``0.02 * Ra_protein``), not just calories
into fat mass.

ITER 97 — Gb IS A DERIVED FIXED POINT, AND THE CARBON BUDGET CLOSES IN ONE UNIT
(review 2026-09-04, items 2.2, 2.6, 3.3, 3.4, 3.10, 3.11; mirrors the teacher's
iter-97 ``glucose_fluxes`` / ``resolve_derived_params``).

What was wrong, measured on the iter-96 artifact: glycogen was a shadow pool
(synthesis took ``relu(appearance)`` without debiting glucose; breakdown reached
blood only by moving the setpoint), muscle glycogen was spent at rest (400 ->
184 g in a resting fast), every glucose gate was a POPULATION threshold while
the setpoints were per patient (Gb = 120 fasted with insulin 23.75; Gb = 75 to
54 mg/dL), ``-(Sg + X)(G - Gb)`` made insulin action a glucose SOURCE below the
setpoint, ``mitochondrial_capacity`` fed all 11 heads, and BHB had no substrate
term. And once the units became mass-conserving, a restoring term
``-Sg(G - Gb_fasted)`` plus an absolute hepatic source could not have Gb as a
fixed point at all: the standing glycogenolysis credit had no counterpart at
rest and glucose sat at Gb + credit/Sg (Gb 75 -> 87.6).

The balance now, in ONE unit (mg/dL of the patient's glucose space per minute;
grams for the pools via ``mg = 1000/(mass·VG_DL_PER_KG)``). The gut kernel
emits appearance in the 70 kg reference space; grams are recovered with the
population ``MG_DL_PER_G``, then converted into this patient's mg/dL:

    dG    = ra·app·f_plasma                       meal appearance not stored
            + glyco + gng                          hepatic output (two fluxes, below)
            − k_ii·G − X·G − uptake_ex             obligatory, insulin-dependent, exercise
    dLGly = f_liver·app_g − glyco / mg
    dMGly = f_muscle·app_g + store·relu(X·G)/mg − brk_M
    dHep  = k·((glyco + gng)·VG_DL_PER_KG − Hep)   a lagged mg/kg/min readout

    EGP_b   = k_ii · Gb_emb                          per patient (basal EGP scales with Gb)
    glyco   = (1 − f_gng)·EGP_b · (LGly/LGly_b) · g_ins_glyco · g_gn · g_G
    gng     =      f_gng ·EGP_b · g_cort·√g_gn·g_ffa·g_ins_gng·g_G

``k_ii`` (obligatory uptake per mg/dL) and ``f_gng`` are learnable POPULATION
scalars. Every structural gate is normalized to exactly 1 at the patient's basal
state (``g_ins`` at I = Ib, ``g_gn`` at Gn = Gnb, ``g_cort`` at Cort_b, ``g_ffa``
at FFA_b, ``g_G`` at G ≤ Gb), so at the fasted reference ``dG = EGP_b − k_ii·Gb
= 0`` exactly: Gb is the fixed point, not an attractor. The liver head's
breakdown output and the hepatic head's production output do not multiply
these fluxes. A learned gain there cancelled the first-order pool on the
iter-104 weights: the same checkpoint with those gains at 1 holds a 48 h fast.
The fasting fall emerges from pool depletion — glycogenolysis is first order
in ``LGly``, so EGP falls toward GNG alone and glucose settles where obligatory
uptake balances it, an absolute floor ``gng/k_ii`` the same for every Gb
(item 3.3). ``Gb_fasted``, the drop and the floor are gone.

Carbon: ``d(G/mg) + dLGly + dMGly = app_g − brk_M − (k_ii·G + X·G +
uptake_ex − gng − store·relu(X·G))/mg`` identically. A heavier person
converts a gram of carbohydrate into fewer mg/dL. ``ra`` is only the
meal-appearance gain.

``Ib = 10·exp(±0.9·tanh(head))`` is a zero-init per-patient head like Gb.
Insulin is not mass-action. A learned basal × Hill could sit above Ib in a
fast (iter 100: 48 h ended at 12.2 µU/mL against the teacher's 3.8). The
rate is the teacher's restoring law:

    effective_Ib = Ib · max((min(G, Gb)/Gb)^5, 0.25)
    dI            = −k_ins · (I − effective_Ib) + γ · mod · relu(G − Gb) · incretin

GSIR is identically 0 at G ≤ Gb; fasting insulin is attracted to the
glucose-gated basal, not a learned floor. ``k_ins`` and ``γ`` are population
scalars (init 0.15 / 0.05, the teacher's ``n`` / ``gamma``); ``mod`` is a
learned state-dependent gain. The thresholds that leaked (glycogen catabolic
gates) are DELETED, not clamped: muscle breakdown is ``relu(act − 0.10)``,
liver breakdown is the insulin gate above. ``mitochondrial_capacity`` has
ONE role — a scale on the clearance of lactate — and no other
head sees it. ``insulin_action`` is lagged ``(I − Ib)/10``, signed; it is not
a concentration, so ``raw_state`` does not floor it at 0. The only bound on
how negative it can drive glucose is ``x_eff = max(Si·X, −0.05·k_ii)``.
BHB is the teacher's ketogenesis, not a mass-action head plus a rectifier
on FFA excess (iter 101: insulin tracked the 48 h teacher at 3.15 vs 3.83
and BHB still ended at 0.23 vs 3.14):

    ketogenesis = k_keto · FFA / (1 + I/IC50) · (1 + 13 · relu(1 − LGly/LGly_b))
    k_bhb       = (k_keto · FFA_b / (1 + Ib/IC50)) / BHB_b
    dBHB        = ketogenesis − k_bhb · BHB

``k_bhb`` is derived so BHB_b is the fed fixed point. Sub-basal insulin
raises production; an empty liver multiplies it by 14. ``k_keto`` and
``IC50`` are population scalars (init 0.005 / 15, the teacher's
``keto_max`` / ``IC50_keto``). The glycogen gain is a constant.

FFA is the teacher's lipolysis, not a gated peak on ``I − Ib`` (iter 102:
k_keto unmoved, 48 h BHB 1.24 vs 3.14, and FFA crashed to 0.14 at 16 h
while insulin was already 5.7 matching the teacher):

    lip_max   = FFA_b · k_ffa · (1 + Ib/IC50)
    lipolysis = lip_max / (1 + I/IC50)
    dFFA      = lipolysis − k_ffa · FFA + 0.01 · Ra_fat

``lip_max`` is derived so FFA_b is the fed fixed point, and ``k_ffa``
cancels at every equilibrium. It is the teacher's clearance, 0.20
(τ ≈ 5 min; Eaton 1969), not a parameter: a free copy walked from 0.20
to 0.038 and the sealed fast was unchanged when it was forced back.
``IC50`` stays a population scalar (init 5 µU/mL) — adipose
antilipolysis is the most insulin-sensitive action in the body.
Sub-basal insulin raises FFA; the gated peak could not be stopped from
crashing it.

Glucagon is the teacher's alpha-cell law, not a gated peak on
``(G − Gb)/30`` (iter 102 graft: at 16 h glucose was already 84 and
insulin 6.4, and glucagon sat at 54 against a basal of 65 — the peak
turns off whenever glucose returns to Gb, so insulin below basal cannot
keep the alpha cell on):

    stim = α · relu(Gb − G) / Gb
    supp = β · (I − Ib) / (Ib + 10)
    dGn  = −k_gn · (Gn − Gnb) + stim − supp + 0.02 · Ra_protein

At G = Gb and I = Ib both extras are zero, so Gnb is the fed fixed
point with nothing left to derive. ``k_gn``, ``α`` and ``β`` are
population scalars (init 0.03 / 3.5 / 0.4, the teacher's ``k_gn`` /
``alpha_gn`` / the 0.4 insulin coefficient). ``β`` stays positive, so
the sign lives in ``(I − Ib)``: insulin below basal disinhibits the
alpha cell even when glucose is back at Gb.
"""

import math

import torch
import torch.nn as nn

from .base import (
    ConstantFluxHead, MassActionModule, SpeciesHead,
)
from ..types import (
    BODY_MASS_KG, GUT_OUTPUT_DIM, MARKER_INDEX, MG_DL_PER_G, MODULE_COUPLING_CHANNELS,
    MODULE_MARKER_INDICES, NORM_CENTER, NORM_SCALE, VG_DL_PER_KG,
)

_GLUCOSE_NORM_SCALE = NORM_SCALE[MARKER_INDEX["glucose"]]
_GLUCOSE_CENTER = NORM_CENTER[MARKER_INDEX["glucose"]]
_INSULIN_NORM_SCALE = NORM_SCALE[MARKER_INDEX["insulin"]]
_INSULIN_CENTER = NORM_CENTER[MARKER_INDEX["insulin"]]
_CORT_NORM_SCALE = NORM_SCALE[MARKER_INDEX["cortisol"]]
_CORT_CENTER = NORM_CENTER[MARKER_INDEX["cortisol"]]
_GN_CENTER = NORM_CENTER[MARKER_INDEX["glucagon"]]
_GN_SCALE = NORM_SCALE[MARKER_INDEX["glucagon"]]
_FFA_CENTER = NORM_CENTER[MARKER_INDEX["ffa"]]
_FFA_SCALE = NORM_SCALE[MARKER_INDEX["ffa"]]
_GLP1_CENTER = NORM_CENTER[MARKER_INDEX["glp1"]]
_GLP1_SCALE = NORM_SCALE[MARKER_INDEX["glp1"]]

# Coupling: gut(4) + cortisol + glp1, from MODULE_COUPLING_CHANNELS.
_N_COUPLING = len(MODULE_COUPLING_CHANNELS["metabolic"])

# External inputs: activity (1) + sleep_wake (1) = 2
_N_EXTERNAL = 2

# Species order matches MODULE_MARKER_INDICES["metabolic"]:
# 0: glucose, 1: insulin, 2: glucagon, 3: ffa, 4: bhb, 5: lactate,
# 6: hepatic_output, 7: liver_glycogen, 8: muscle_glycogen,
# 9: mitochondrial_capacity, 10: insulin_action, 11: fat_mass.
_TYPICALS = [95.0, 10.0, 70.0, 0.5, 0.1, 1.0, 2.0, 100.0, 400.0, 1.0, 0.0, 18.0]
# NORM_SCALE per species, in the same order, so the module can rebuild the RAW
# concentration for the mass-action consumption term (iter 95).
_NORM_SCALES = [NORM_SCALE[i] for i in MODULE_MARKER_INDICES["metabolic"]]
# cons_scale = 1/τ in the mass-action rate equation (glycogen and glucose no longer use
# theirs — they have explicit flux forms — but the slots keep the protocol uniform).
_CONS_SCALES = [0.02, 0.1, 0.03, 0.04, 0.03, 0.02, 0.04, 7e-4, 3.3e-5, 2.5e-5, 0.02, 0.0]

_GLUCOSE_IDX = 0
_INSULIN_IDX = 1
_GLUCAGON_IDX = 2
_FFA_IDX = 3
_BHB_IDX = 4
_LACTATE_IDX = 5
_HEPATIC_IDX = 6
_LIVER_GLYCOGEN_IDX = 7
_MUSCLE_GLYCOGEN_IDX = 8
_MITO_IDX = 9
_INSULIN_ACTION_IDX = 10
_FAT_MASS_IDX = 11
_N_SPECIES = 12

# Index of the gut glucose-appearance channel within the `coupling` tensor.
_GUT_GLUCOSE_COUPLING_IDX = 0
# cortisol is metabolic coupling[GUT_OUTPUT_DIM]; activity is external[0].
_CORTISOL_COUPLING_IDX = GUT_OUTPUT_DIM
_GLP1_COUPLING_IDX = GUT_OUTPUT_DIM + 1
_LIPID_COUPLING_IDX = 1
_AMINO_COUPLING_IDX = 2
_ACTIVITY_EXTERNAL_IDX = 0
_SLEEP_EXTERNAL_IDX = 1

# Lag rate p2 = 1/τ for insulin action, bounded so τ ∈ [4, 100] min (teacher p2 = 0.02,
# τ = 50 min). Floored so the state can't freeze; capped so it can't collapse to the
# old instantaneous behaviour. Euler-stable (p2·dt ≤ 0.25 ≪ 2).
_P2_MIN = 0.01
_P2_RANGE = 0.24
_P2_INIT = 0.02  # teacher full_body.py PatientParams.p2
# Max per-patient fasting-glucose offset, z-score units around 95 mg/dL (iter 90: ±2.2
# gives Gb ∈ [29, 161], covering the teacher's lognormal spread and the benchmark's 60-120).
_GLUCOSE_BASELINE_MAX_Z = 2.2
# Iter 97: per-patient basal insulin in LOG space so it is positive by construction:
# Ib = 10·exp(±0.9·tanh) ∈ [4.1, 24.6] µU/mL — the teacher varies Ib with σ = 0.4
# lognormal loaded on insulin resistance, i.e. roughly this span at ±2σ.
_IB_LOG_MAX = 0.9
# Max per-patient meal-appearance (Ra) log-gain offset, pre-softplus units (iter 88).
_RA_BASELINE_MAX_Z = 1.0
# Iter 97: `ra` is the per-patient gain on the (now mass-conserving) gut appearance —
# 1.0 means the kernel's bioavailable mass appears in glucose space as-is. Init below 1
# so an UNTRAINED student's 75 g meal lands near the teacher's +56.6 mg/dL at 55 min
# (the teacher clears its load with a trained second-phase insulin response and
# first-pass hepatic uptake that the fresh student's heads cannot yet supply); the
# dose-response and mass-balance signals move it from here. Measured at init across
# 0.3/0.5/0.7/1.0: peak +19/+33/+47/+70 mg/dL; 0.8 lands +56 (the untrained peak
# sits at ~90 min, not 55 — the fresh insulin head has no second phase yet).
_RA_INIT = 0.8
# Iter 97: obligatory (insulin-independent) glucose uptake per mg/dL of glucose space —
# brain, blood cells, renal medulla. A POPULATION scalar, learnable within a band, init
# at the teacher's uptake_ii = 2.0 mg/kg/min / (1.85 dL/kg · 95 mg/dL) = 0.0114/min, i.e.
# the typical patient rests at the typical EGP of 2.0 mg/kg/min. Band [0.004, 0.03]/min.
_KII_MIN = 0.004
_KII_RANGE = 0.026
_KII_INIT = 2.0 / (1.85 * 95.0)
# Si: RAW per-minute per normalized-insulin-unit; X = Si·Xa with Xa the lagged
# signed (I − Ib)/10. Band brackets the Bergman literature range with margin.
_SI_MIN = 0.0005
_SI_RANGE = 0.0195
_SI_INIT = 0.004  # = 10 × teacher Si (insulin normalization)
_KACT_MIN, _KACT_RANGE, _KACT_INIT = 0.0, 0.06, 0.02          # teacher exercise uptake
# Fraction of basal EGP that is gluconeogenesis at the fasted reference (teacher
# Gng_b / Hep_b = 1.0 / 2.0; Landau 1996: 47 % at 14 h). Learnable population scalar.
_F_GNG_INIT = 0.5
# Structural gate constants (teacher iter 97). K's are learnable (softplus, init here);
# the Hill exponents are shapes and stay fixed.
_GLYC_INS_K_INIT, _GLYC_INS_N = 25.0, 2.0     # glycogenolysis insulin gate
_GNG_INS_K_INIT, _GNG_INS_N = 80.0, 1.0       # gluconeogenesis insulin gate
_HGO_GN_N = 2.0                               # glucagon Hill exponent on hepatic output
_GNG_CORT_AMP = 0.2                           # ±20 % GNG per e-fold of cortisol
_GNG_FFA_EXP = 0.2                            # (FFA/FFA_b)^0.2 substrate push on GNG
_HEP_AUTOREG_M = 0.6                          # (Gb/G)^m suppression above Gb, one-sided
# Withdrawable share of obligatory uptake at sub-basal insulin (teacher 0.05; Landau
# 1996). Floors signed X so remote insulin cannot reverse obligatory (brain) uptake.
_INS_DEP_BASAL_FRAC = 0.05
_INCRETIN_GAIN = 4.0
_K_INCRETIN = 10.0
_ID_STORE_FRAC = 0.25
_LAC_GLYCO_GAIN = 0.02
_KCAL_PER_KG_FAT = 7700.0
_BMR_KCAL_PER_MIN = 1.15
_ACT_KCAL_PER_MIN = 4.0
_LIPID_UNITS_PER_G = 3.0
# Teacher full_body.py: dFFA += 0.01 * Ra_fat, dGn += 0.02 * Ra_protein.
# Lipid/amino appearance already arrive on the coupling vector; without these
# terms they only entered the fat-mass calorie residual.
_FFA_FROM_LIPID = 0.01
_GN_FROM_AMINO = 0.02
_K_MITO = 1.0 / (14.0 * 1440.0)
_MITO_TRAIN_GAIN = 2.0e-5
_MITO_LOG_MAX = 0.4
_MASS_LOG_MAX = 0.25
_FFA_LOG_MAX = 0.5
_GN_LOG_MAX = 0.4
# Fasting insulin: the teacher's restoring law (Polonsky 1988; PatientParams.n / gamma /
# fast_ins_exp / fast_ins_floor). Not a learned Hill on a mass-action basal.
_FAST_INS_EXP = 5.0
_FAST_INS_FLOOR = 0.25
_K_INS_INIT = 0.15
_GAMMA_INIT = 0.05
# --- glycogen pools --------------------------------------------------------------------
_LIVER_GLY_CENTER = NORM_CENTER[MARKER_INDEX["liver_glycogen"]]      # 100 g
_MUSCLE_GLY_CENTER = NORM_CENTER[MARKER_INDEX["muscle_glycogen"]]    # 400 g
# Store capacity as a multiple of typical (supercompensation, Bergstrom & Hultman 1966 —
# a real ~20-40 % overshoot) and the WIDTH over which synthesis closes as the store nears
# capacity: `sigmoid((cap − pool)/width)` — 0.95 at typical, 0.5 at the cap.
_GLY_CAPACITY_FRAC = 1.3
_GLY_FILL_WIDTH_FRAC = 0.1
# Muscle breakdown availability: Michaelis in the pool, zero at zero by construction.
_MUSCLE_GLY_K = 150.0  # teacher glyc_K_M (g)
# Muscle breakdown flux scale (g/min per unit of the head's softplus drive, ≈0.7 at
# init): 3.0·0.7·0.73·relu(1 − 0.1) ≈ 1.4 g/min at a maximal bout, i.e. the −150 g / 2 h
# cohort target (teacher k = 3.5).
_MUSCLE_GLY_FLUX = 3.0
# Activity at or below this is rest for muscle glycogen: `relu(act − a_rest)` is zero by
# construction there (Coppack 1989; teacher act_rest_M). A CONSTANT, not a learned
# threshold — iter 96's S1 lesson is that a learnable threshold cannot be stopped from
# leaking (the previous gate learned its way to 40 % open at activity 0).
_MUSCLE_ACT_REST = 0.10
# Prior fractions of absorbed carbohydrate stored: liver 30 % (teacher glyc_syn_frac_L;
# Taylor 1996: 19 % of the meal by 5 h once the insulin drive is averaged in), muscle
# 15 %, the rest to plasma. Softmax logits relative to the plasma reference (logit 0).
_LIVER_STORE_FRAC_INIT = 0.30
_MUSCLE_STORE_FRAC_INIT = 0.15
_PLASMA_FRAC_INIT = 1.0 - _LIVER_STORE_FRAC_INIT - _MUSCLE_STORE_FRAC_INIT

# --- ketogenesis -----------------------------------------------------------------------
# Teacher full_body.ketogenesis (Cahill 2006; iter 80/93/97). Production on
# FFA as a concentration, insulin in the denominator as a concentration (not a
# rectifier on I−Ib), times a linear liver-emptying gain. Clearance is
# derived so BHB_b is the fed fixed point — the same construction as EGP_b.
_K_KETO_INIT = 0.005          # teacher keto_max
_KETO_INS_SUPP_INIT = 15.0    # teacher IC50_keto
_KETO_GLYC_GAIN = 13.0        # teacher keto_glyc_gain
_BHB_CENTER = NORM_CENTER[MARKER_INDEX["bhb"]]  # teacher BHB_b / typical

# --- lipolysis -----------------------------------------------------------------
# Teacher full_body.dFFA (Nurjhan 1986; Jensen 1989; Eaton 1969; iter 93).
# lip_max is derived so FFA_b is the fed fixed point. Clearance cancels
# there, so it is the teacher's time constant, not a learned level.
_K_FFA = 0.20                 # teacher k_ffa (τ ≈ 5 min)
_LIP_IC50_INIT = 5.0          # teacher IC50_lip

# --- glucagon ----------------------------------------------------------------------
# Teacher full_body.dGn (Unger & Orci 1981; Marliss 1970). Restoring to the
# patient's Gnb, glucose below Gb stimulates, insulin signed about Ib
# suppresses. At G = Gb and I = Ib the two extras are zero, so Gnb is the
# fed fixed point. The insulin offset in the denominator is the teacher's
# constant (Ib + 10), not a learned scale.
_K_GN_INIT = 0.03             # teacher k_gn (τ ≈ 33 min)
_ALPHA_GN_INIT = 3.5          # teacher alpha_gn
_GN_INS_INIT = 0.4            # teacher coefficient on (I − Ib) / (Ib + 10)
_GN_INS_OFFSET = 10.0


def _logit(p: float) -> float:
    """Inverse sigmoid — init a sigmoid-bounded parameter at a target value."""
    return math.log(p / (1.0 - p))


def _inverse_softplus(y: float) -> float:
    return math.log(math.expm1(y))


def _ins_gate(i: torch.Tensor, ib: torch.Tensor, k: torch.Tensor, n: float) -> torch.Tensor:
    """IC50 suppression normalized at basal insulin: 1 at I = Ib, → 0 at high insulin,
    → 1 + (Ib/K)^n as insulin falls to zero (signed, saturating). Teacher `_glyc_ins_gate`."""
    return (1.0 + (ib / k) ** n) / (1.0 + (i / k) ** n)


def _hill_centred(x: torch.Tensor, x_b: float, n: float) -> torch.Tensor:
    """2/(1 + (x_b/x)^n): 1 at basal, → 2 above, → 0 below. Teacher `_hill_centred`."""
    return 2.0 / (1.0 + (x_b / x.clamp(min=1e-6)) ** n)


class GlucoseStimulatedInsulinHead(nn.Module):
    """Learned modulation of glucose-stimulated insulin release.

    The rate is structural in ``MetabolicModule.fluxes``:

        dI = −k_ins · (I − effective_Ib) + γ · mod · relu(G − Gb) · incretin

    This head only emits ``mod`` (exp of an MLP, clamped). GSIR is identically 0
    at G ≤ Gb, so fasting insulin cannot sit above Ib because a leaky sigmoid
    or a learned basal said so. No gate temperature — the iter-76 collapse
    mode does not exist here.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, state_self: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        del state_self
        raw = self.network(x).squeeze(-1)
        mod = torch.exp(raw.clamp(-3.0, 4.0))
        return mod, torch.ones_like(raw)


class GlycogenFluxHead(nn.Module):
    """Glycogen as a flux integrator: the head emits the two learned GAINS of

        synthesis   = f_store · fill · appearance_g        (f_store from a softmax the
                                                            module takes over both pools
                                                            and plasma — see fluxes)
        breakdown   = structural                                 liver glycogenolysis
                      activity-gated · cons                      muscle only

    In the ``(prod, cons)`` protocol: ``prod`` is the STORE LOGIT (unbounded; the
    module softmaxes it against the other pool and a zero plasma reference so the
    fractions sum to one). ``cons`` scales muscle glycogenolysis. Liver
    glycogenolysis does not read it.

    Iter 97: the learned catabolic gate (``σ((s − c_thresh)/τ)``) is gone. It leaked
    the way every learned threshold here has leaked — 40 % open at activity 0 for
    muscle, 86 % ungated for liver — and the module now applies the gates
    structurally (``relu(act − a_rest)`` / the basal-normalized insulin gate).
    """

    def __init__(self, input_dim: int, hidden_dim: int, *, init_store_logit: float, emit_breakdown: bool = True):
        super().__init__()
        self.emit_breakdown = emit_breakdown
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2 if emit_breakdown else 1),
        )
        self.init_store_logit = float(init_store_logit)

    def forward(self, x: torch.Tensor, state_self: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.network(x)
        store_logit = raw[..., 0] + self.init_store_logit
        if self.emit_breakdown:
            break_mod = nn.functional.softplus(raw[..., 1])
        else:
            break_mod = torch.ones_like(store_logit)
        return store_logit, break_mod


def _without_mito(head_cls):
    """Factory adapter: build ``head_cls`` on the input WITHOUT the mito column."""
    return lambda inp, hd: head_cls(inp - 1, hd)


class MetabolicModule(MassActionModule):
    def __init__(self, embedding_dim: int, hidden_dim: int = 48):
        super().__init__(
            n_species=_N_SPECIES,
            n_coupling=_N_COUPLING,
            n_external=_N_EXTERNAL,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            typicals=_TYPICALS,
            norm_scales=_NORM_SCALES,
            # insulin_action is signed remote insulin, not a concentration. A 0-floor
            # here zeros the tracker whenever X < 0 and walks X to the catastrophe wall.
            raw_floors=[0.0 if i != _INSULIN_ACTION_IDX else float("-inf")
                        for i in range(_N_SPECIES)],
            # Iter 97: every head except mito's own reads the module input WITHOUT the
            # mitochondrial_capacity column (3.11); glucose, FFA, BHB and insulin_action
            # have fully structural rates and own no parameters.
            head_factories={
                _GLUCOSE_IDX: lambda inp, hd: ConstantFluxHead(),
                _INSULIN_IDX: _without_mito(GlucoseStimulatedInsulinHead),
                # Gated peak heads (iter 70/72: structurally load-bearing for the HPA
                # coupling — reverting them broke ACTH 11x). Stimuli are handed in by the
                # module as per-patient deviations; the idx here is the fallback.
                _GLUCAGON_IDX: lambda inp, hd: ConstantFluxHead(),
                _FFA_IDX: lambda inp, hd: ConstantFluxHead(),
                _BHB_IDX: lambda inp, hd: ConstantFluxHead(),
                _LACTATE_IDX: _without_mito(SpeciesHead),
                # hepatic_output: cons is the readout's relaxation rate.
                # There is no production output — it used to scale gluconeogenesis.
                _HEPATIC_IDX: lambda inp, hd: SpeciesHead(inp - 1, hd, emit_prod=False),
                _LIVER_GLYCOGEN_IDX: lambda inp, hd: GlycogenFluxHead(
                    inp - 1, hd,
                    init_store_logit=math.log(_LIVER_STORE_FRAC_INIT / _PLASMA_FRAC_INIT),
                    emit_breakdown=False),
                _MUSCLE_GLYCOGEN_IDX: lambda inp, hd: GlycogenFluxHead(
                    inp - 1, hd,
                    init_store_logit=math.log(_MUSCLE_STORE_FRAC_INIT / _PLASMA_FRAC_INIT)),
                _MITO_IDX: lambda inp, hd: ConstantFluxHead(),
                _INSULIN_ACTION_IDX: lambda inp, hd: ConstantFluxHead(),
                _FAT_MASS_IDX: lambda inp, hd: ConstantFluxHead(),
            },
        )
        prod_scales = [c * t for c, t in zip(_CONS_SCALES, _TYPICALS)]
        self.prod_scale.copy_(torch.tensor(prod_scales, dtype=torch.float32))
        self.cons_scale.copy_(torch.tensor(_CONS_SCALES, dtype=torch.float32))

        # Population scalars of the glucose balance (see the module docstring).
        self.log_k_ii = nn.Parameter(torch.tensor(_logit((_KII_INIT - _KII_MIN) / _KII_RANGE)))
        self.logit_f_gng = nn.Parameter(torch.tensor(_logit(_F_GNG_INIT)))
        self.log_glyc_ins_k = nn.Parameter(torch.tensor(_inverse_softplus(_GLYC_INS_K_INIT)))
        self.log_gng_ins_k = nn.Parameter(torch.tensor(_inverse_softplus(_GNG_INS_K_INIT)))
        self.log_k_ins = nn.Parameter(torch.tensor(_inverse_softplus(_K_INS_INIT)))
        self.log_gamma = nn.Parameter(torch.tensor(_inverse_softplus(_GAMMA_INIT)))
        # Structural rate-of-appearance gain Ra on the gut glucose-appearance flux
        # (iter 80): the meal-appearance gain, and nothing else (iter 97).
        self.log_ra = nn.Parameter(torch.tensor(_inverse_softplus(_RA_INIT)))
        self.log_si = nn.Parameter(torch.tensor(_logit((_SI_INIT - _SI_MIN) / _SI_RANGE)))
        self.log_p2 = nn.Parameter(torch.tensor(_logit((_P2_INIT - _P2_MIN) / _P2_RANGE)))
        self.log_k_act = nn.Parameter(torch.tensor(_logit((_KACT_INIT - _KACT_MIN) / _KACT_RANGE)))
        # Ketogenesis: IC50 on absolute insulin (µU/mL) and keto_max. The
        # glycogen-emptying gain is a constant; clearance is derived from these.
        self.log_keto_ins_supp = nn.Parameter(torch.tensor(_inverse_softplus(_KETO_INS_SUPP_INIT)))
        self.log_k_keto = nn.Parameter(torch.tensor(_inverse_softplus(_K_KETO_INIT)))
        # Lipolysis: IC50 on absolute insulin (µU/mL). Clearance is _K_FFA.
        # lip_max is derived so FFA_b is the fed fixed point.
        self.log_lip_ic50 = nn.Parameter(torch.tensor(_inverse_softplus(_LIP_IC50_INIT)))
        # Glucagon: speed, glucose-stimulus gain, signed-insulin coefficient.
        # Gnb is already the fed fixed point of the restoring term.
        self.log_k_gn = nn.Parameter(torch.tensor(_inverse_softplus(_K_GN_INIT)))
        self.log_alpha_gn = nn.Parameter(torch.tensor(_inverse_softplus(_ALPHA_GN_INIT)))
        self.log_gn_ins = nn.Parameter(torch.tensor(_inverse_softplus(_GN_INS_INIT)))

        # Per-patient setpoint heads. Final layers zero-init ⇒ Gb = 95, Ib = 10, Ra =
        # softplus(log_ra) for every embedding at cold start; authority grows in training.
        _bh = max(8, hidden_dim // 4)

        def _zero_head() -> nn.Sequential:
            net = nn.Sequential(nn.Linear(embedding_dim, _bh), nn.Tanh(), nn.Linear(_bh, 1))
            with torch.no_grad():
                net[-1].weight.zero_()
                net[-1].bias.zero_()
            return net

        self.glucose_baseline_net = _zero_head()   # Gb
        self.insulin_baseline_net = _zero_head()   # Ib
        self.ra_baseline_net = _zero_head()        # Ra
        self.body_mass_net = _zero_head()          # kg; 70 at zero embedding
        self.ffa_baseline_net = _zero_head()
        self.gn_baseline_net = _zero_head()
        self.mito_setpoint_net = _zero_head()

    # ---- per-patient setpoints -------------------------------------------------------

    def glucose_setpoint_raw(self, embedding: torch.Tensor) -> torch.Tensor:
        b_emb = _GLUCOSE_BASELINE_MAX_Z * torch.tanh(self.glucose_baseline_net(embedding).squeeze(-1))
        return _GLUCOSE_CENTER + _GLUCOSE_NORM_SCALE * b_emb

    def insulin_setpoint_raw(self, embedding: torch.Tensor) -> torch.Tensor:
        return _INSULIN_CENTER * torch.exp(
            _IB_LOG_MAX * torch.tanh(self.insulin_baseline_net(embedding).squeeze(-1)))

    def appearance_gain(self, embedding: torch.Tensor) -> torch.Tensor:
        ra_emb = _RA_BASELINE_MAX_Z * torch.tanh(self.ra_baseline_net(embedding).squeeze(-1))
        return nn.functional.softplus(self.log_ra + ra_emb)

    def body_mass_kg(self, embedding: torch.Tensor) -> torch.Tensor:
        return BODY_MASS_KG * torch.exp(
            _MASS_LOG_MAX * torch.tanh(self.body_mass_net(embedding).squeeze(-1)))

    def mg_dl_per_g(self, embedding: torch.Tensor) -> torch.Tensor:
        return 1000.0 / (self.body_mass_kg(embedding) * VG_DL_PER_KG)

    def ffa_setpoint_raw(self, embedding: torch.Tensor) -> torch.Tensor:
        return _FFA_CENTER * torch.exp(
            _FFA_LOG_MAX * torch.tanh(self.ffa_baseline_net(embedding).squeeze(-1)))

    def gn_setpoint_raw(self, embedding: torch.Tensor) -> torch.Tensor:
        return _GN_CENTER * torch.exp(
            _GN_LOG_MAX * torch.tanh(self.gn_baseline_net(embedding).squeeze(-1)))

    def mito_setpoint_raw(self, embedding: torch.Tensor) -> torch.Tensor:
        return torch.exp(
            _MITO_LOG_MAX * torch.tanh(self.mito_setpoint_net(embedding).squeeze(-1)))

    def k_ii(self) -> torch.Tensor:
        return _KII_MIN + _KII_RANGE * torch.sigmoid(self.log_k_ii)

    # ---- heads -----------------------------------------------------------------------

    @staticmethod
    def _head_input(state, coupling, external, embedding, time_features):
        x = torch.cat([state, coupling, external, embedding, time_features], dim=-1)
        x_no_mito = torch.cat([x[..., :_MITO_IDX], x[..., _MITO_IDX + 1:]], dim=-1)
        return x, x_no_mito

    def species_fluxes(
        self,
        state: torch.Tensor,
        coupling: torch.Tensor,
        external: torch.Tensor,
        embedding: torch.Tensor,
        time_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-species head outputs. Glucose, insulin, glucagon, FFA, BHB,
        insulin action, mito and fat mass have structural rates and do not
        read these outputs."""
        x, x_no_mito = self._head_input(state, coupling, external, embedding, time_features)
        prods: list[torch.Tensor] = []
        conss: list[torch.Tensor] = []
        for i, head in enumerate(self.heads):
            xi = x if i == _MITO_IDX else x_no_mito
            p, c = head(xi, state[..., i])
            prods.append(p)
            conss.append(c)
        return torch.stack(prods, dim=-1), torch.stack(conss, dim=-1)

    # ---- fluxes ----------------------------------------------------------------------

    def fluxes(
        self,
        state: torch.Tensor,
        coupling: torch.Tensor,
        external: torch.Tensor,
        embedding: torch.Tensor,
        time_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Every named term of the metabolic ODE, in raw units (mg/dL/min of glucose
        space; g/min for the pools). ``forward`` assembles the rates from these; the
        carbon-budget test and the probes read them directly."""
        relu = nn.functional.relu
        raw = self.raw_state(state)
        g = raw[..., _GLUCOSE_IDX]
        ins = raw[..., _INSULIN_IDX]
        gn = raw[..., _GLUCAGON_IDX]
        ffa = raw[..., _FFA_IDX]
        bhb = raw[..., _BHB_IDX]
        hep = raw[..., _HEPATIC_IDX]
        lgly = raw[..., _LIVER_GLYCOGEN_IDX]
        mgly = raw[..., _MUSCLE_GLYCOGEN_IDX]
        mito = raw[..., _MITO_IDX]
        xa = raw[..., _INSULIN_ACTION_IDX]
        cort = (_CORT_CENTER + _CORT_NORM_SCALE * coupling[..., _CORTISOL_COUPLING_IDX]).clamp(min=0.05)
        glp1 = (_GLP1_CENTER + _GLP1_SCALE * coupling[..., _GLP1_COUPLING_IDX]).clamp(min=0.1)
        act = external[..., _ACTIVITY_EXTERNAL_IDX]

        gb = self.glucose_setpoint_raw(embedding)
        ib = self.insulin_setpoint_raw(embedding)
        ra = self.appearance_gain(embedding)
        mg = self.mg_dl_per_g(embedding)
        mass_kg = self.body_mass_kg(embedding)
        ffa_b = self.ffa_setpoint_raw(embedding)
        gn_b = self.gn_setpoint_raw(embedding)
        mito_sp = self.mito_setpoint_raw(embedding)
        glucose_dev = (g - gb) / _GLUCOSE_NORM_SCALE
        insulin_dev = (ins - ib) / _INSULIN_NORM_SCALE

        prod_raw, cons_raw = self.species_fluxes(
            state, coupling, external, embedding, time_features)

        # Gut glucose appearance is in the 70 kg reference space. Grams are that
        # density / MG_DL_PER_G; this patient's mg/dL uses their own V_G.
        app_g = ra * coupling[..., _GUT_GLUCOSE_COUPLING_IDX] / MG_DL_PER_G
        app_eff = app_g * mg
        store_logits = torch.stack([
            prod_raw[..., _LIVER_GLYCOGEN_IDX],
            prod_raw[..., _MUSCLE_GLYCOGEN_IDX],
            torch.zeros_like(app_eff),
        ], dim=-1)
        frac = torch.softmax(store_logits, dim=-1)
        fill_l = torch.sigmoid(
            (_GLY_CAPACITY_FRAC * _LIVER_GLY_CENTER - lgly) / (_GLY_FILL_WIDTH_FRAC * _LIVER_GLY_CENTER))
        fill_m = torch.sigmoid(
            (_GLY_CAPACITY_FRAC * _MUSCLE_GLY_CENTER - mgly) / (_GLY_FILL_WIDTH_FRAC * _MUSCLE_GLY_CENTER))
        f_liver = frac[..., 0] * fill_l
        f_muscle = frac[..., 1] * fill_m
        f_plasma = 1.0 - f_liver - f_muscle
        syn_liver = f_liver * app_g
        syn_muscle_oral = f_muscle * app_g
        appearance_plasma = app_eff * f_plasma

        k_ii = self.k_ii()
        egp_b = k_ii * gb
        f_gng = torch.sigmoid(self.logit_f_gng)
        glyc_k = nn.functional.softplus(self.log_glyc_ins_k)
        gng_k = nn.functional.softplus(self.log_gng_ins_k)
        g_ins_glyco = _ins_gate(ins, ib, glyc_k, _GLYC_INS_N)
        g_ins_gng = _ins_gate(ins, ib, gng_k, _GNG_INS_N)
        g_gn = _hill_centred(gn, gn_b, _HGO_GN_N)
        g_cort = 1.0 + _GNG_CORT_AMP * torch.tanh(torch.log(cort / _CORT_CENTER))
        g_ffa = (ffa.clamp(min=1e-3) / ffa_b) ** _GNG_FFA_EXP
        g_g = (gb / torch.maximum(g, gb)) ** _HEP_AUTOREG_M
        glycogenolysis_plasma = ((1.0 - f_gng) * egp_b * (lgly / _LIVER_GLY_CENTER)
                                 * g_ins_glyco * g_gn * g_g)
        gng_plasma = (f_gng * egp_b * g_cort * torch.sqrt(g_gn) * g_ffa * g_ins_gng * g_g)
        brk_liver = glycogenolysis_plasma / mg

        avail_m = mgly / (mgly + _MUSCLE_GLY_K)
        brk_muscle = (_MUSCLE_GLY_FLUX * cons_raw[..., _MUSCLE_GLYCOGEN_IDX]
                      * relu(act - _MUSCLE_ACT_REST) * avail_m)

        si = _SI_MIN + _SI_RANGE * torch.sigmoid(self.log_si)
        uptake_ii = k_ii * g
        x_eff = torch.maximum(si * xa, -_INS_DEP_BASAL_FRAC * k_ii)
        uptake_id = x_eff * g
        k_act = _KACT_MIN + _KACT_RANGE * torch.sigmoid(self.log_k_act)
        exercise_uptake = k_act * act * relu(g - 0.8 * gb)
        syn_muscle_id = _ID_STORE_FRAC * relu(uptake_id) / mg * fill_m
        syn_muscle = syn_muscle_oral + syn_muscle_id

        glp1_excess = relu(glp1 - _GLP1_CENTER)
        incretin = 1.0 + _INCRETIN_GAIN * glp1_excess / (glp1_excess + _K_INCRETIN)
        k_ins = nn.functional.softplus(self.log_k_ins)
        gamma = nn.functional.softplus(self.log_gamma)
        glucose_ratio = torch.minimum(g / gb.clamp(min=1.0), torch.ones_like(g))
        effective_ib = ib * torch.clamp(glucose_ratio ** _FAST_INS_EXP, min=_FAST_INS_FLOOR)
        gsir_mod = prod_raw[..., _INSULIN_IDX]
        ins_gsir = gamma * gsir_mod * relu(g - gb) * incretin
        ins_restoring = -k_ins * (ins - effective_ib)
        ins_rate = ins_restoring + ins_gsir
        p2 = _P2_MIN + _P2_RANGE * torch.sigmoid(self.log_p2)
        xa_rate = p2 * (insulin_dev - xa)

        hep_target = (glycogenolysis_plasma + gng_plasma) * VG_DL_PER_KG
        hep_rate = cons_raw[..., _HEPATIC_IDX] * self.cons_scale[_HEPATIC_IDX] * (hep_target - hep)

        ic50_keto = nn.functional.softplus(self.log_keto_ins_supp)
        k_keto = nn.functional.softplus(self.log_k_keto)
        glyco_depletion = relu(1.0 - lgly / _LIVER_GLY_CENTER)
        ketogenesis = (k_keto * ffa / (1.0 + ins / ic50_keto)
                       * (1.0 + _KETO_GLYC_GAIN * glyco_depletion))
        k_bhb = (k_keto * ffa_b / (1.0 + ib / ic50_keto)) / _BHB_CENTER
        bhb_rate = ketogenesis - k_bhb * bhb

        lipid_app = coupling[..., _LIPID_COUPLING_IDX].clamp(min=0.0)
        amino_app = coupling[..., _AMINO_COUPLING_IDX].clamp(min=0.0)
        lipid_g = lipid_app / _LIPID_UNITS_PER_G
        amino_g = amino_app / _LIPID_UNITS_PER_G
        kcal_in = 4.0 * app_g + 9.0 * lipid_g + 4.0 * amino_g
        kcal_out = _BMR_KCAL_PER_MIN * (mass_kg / BODY_MASS_KG) + _ACT_KCAL_PER_MIN * act
        fat_rate = (kcal_in - kcal_out) / _KCAL_PER_KG_FAT
        ffa_from_lipid = _FFA_FROM_LIPID * lipid_app
        glucagon_from_amino = _GN_FROM_AMINO * amino_app
        k_gn = nn.functional.softplus(self.log_k_gn)
        alpha_gn = nn.functional.softplus(self.log_alpha_gn)
        gn_ins = nn.functional.softplus(self.log_gn_ins)
        glucagon_stim = alpha_gn * relu(gb - g) / gb.clamp(min=1.0)
        glucagon_supp = gn_ins * (ins - ib) / (ib + _GN_INS_OFFSET)
        glucagon_restoring = -k_gn * (gn - gn_b)
        gn_rate = glucagon_restoring + glucagon_stim - glucagon_supp + glucagon_from_amino
        k_ffa = ffa.new_tensor(_K_FFA)
        ic50_lip = nn.functional.softplus(self.log_lip_ic50)
        lip_max = ffa_b * k_ffa * (1.0 + ib / ic50_lip)
        lipolysis = lip_max / (1.0 + ins / ic50_lip)
        ffa_rate = lipolysis - k_ffa * ffa + ffa_from_lipid

        mito_rate = -_K_MITO * (mito - mito_sp) + _MITO_TRAIN_GAIN * relu(act - 0.2)
        lactate_from_glyco = _LAC_GLYCO_GAIN * brk_muscle

        return {
            "gb": gb, "ib": ib, "ra": ra, "egp_b": egp_b, "k_ii": k_ii, "f_gng": f_gng,
            "mg_dl_per_g": mg, "body_mass_kg": mass_kg, "ffa_b": ffa_b, "gn_b": gn_b,
            "glucose_dev": glucose_dev, "insulin_dev": insulin_dev,
            "effective_ib": effective_ib, "incretin": incretin, "k_ins": k_ins, "gamma": gamma,
            "prod_raw": prod_raw, "cons_raw": cons_raw, "mito": mito,
            "app_eff": app_eff, "app_g": app_g,
            "f_liver": f_liver, "f_muscle": f_muscle, "f_plasma": f_plasma,
            "syn_liver": syn_liver, "syn_muscle": syn_muscle, "syn_muscle_id": syn_muscle_id,
            "syn_muscle_oral": syn_muscle_oral,
            "brk_liver": brk_liver, "brk_muscle": brk_muscle,
            "appearance_plasma": appearance_plasma,
            "glycogenolysis_plasma": glycogenolysis_plasma, "gng_plasma": gng_plasma,
            "g_ins_glyco": g_ins_glyco, "g_ins_gng": g_ins_gng, "g_gn": g_gn,
            "g_cort": g_cort, "g_ffa": g_ffa, "g_g": g_g,
            "uptake_ii": uptake_ii, "uptake_id": uptake_id, "exercise_uptake": exercise_uptake,
            "ins_gsir": ins_gsir, "ins_restoring": ins_restoring, "ins_rate": ins_rate,
            "xa": xa, "xa_rate": xa_rate, "hep_target": hep_target, "hep_rate": hep_rate,
            "ketogenesis": ketogenesis, "glyco_depletion": glyco_depletion,
            "k_bhb": k_bhb, "k_keto": k_keto, "ic50_keto": ic50_keto,
            "bhb_rate": bhb_rate, "fat_rate": fat_rate, "mito_rate": mito_rate,
            "lactate_from_glyco": lactate_from_glyco, "mito_sp": mito_sp,
            "ffa_from_lipid": ffa_from_lipid, "glucagon_from_amino": glucagon_from_amino,
            "k_gn": k_gn, "alpha_gn": alpha_gn, "gn_ins": gn_ins,
            "glucagon_stim": glucagon_stim, "glucagon_supp": glucagon_supp,
            "glucagon_restoring": glucagon_restoring, "gn_rate": gn_rate,
            "lipolysis": lipolysis, "lip_max": lip_max, "k_ffa": k_ffa, "ic50_lip": ic50_lip,
            "ffa_rate": ffa_rate,
        }

    def forward(
        self,
        state: torch.Tensor,
        coupling: torch.Tensor,
        external: torch.Tensor,
        embedding: torch.Tensor,
        time_features: torch.Tensor,
    ) -> torch.Tensor:
        f = self.fluxes(state, coupling, external, embedding, time_features)
        prod_raw, cons_raw, mito = f["prod_raw"], f["cons_raw"], f["mito"]
        raw = self.raw_state(state)
        rates = prod_raw * self.prod_scale - cons_raw * self.cons_scale * raw
        out = rates.clone()
        out[..., _LACTATE_IDX] = (
            prod_raw[..., _LACTATE_IDX] * self.prod_scale[_LACTATE_IDX]
            - cons_raw[..., _LACTATE_IDX] * self.cons_scale[_LACTATE_IDX] * mito * raw[..., _LACTATE_IDX]
            + f["lactate_from_glyco"]
        )
        out[..., _FFA_IDX] = f["ffa_rate"]
        out[..., _GLUCAGON_IDX] = f["gn_rate"]
        out[..., _BHB_IDX] = f["bhb_rate"]
        out[..., _GLUCOSE_IDX] = (
            f["appearance_plasma"] + f["glycogenolysis_plasma"] + f["gng_plasma"]
            - f["uptake_ii"] - f["uptake_id"] - f["exercise_uptake"]
        )
        out[..., _INSULIN_IDX] = f["ins_rate"]
        out[..., _INSULIN_ACTION_IDX] = f["xa_rate"]
        out[..., _HEPATIC_IDX] = f["hep_rate"]
        out[..., _LIVER_GLYCOGEN_IDX] = f["syn_liver"] - f["brk_liver"]
        out[..., _MUSCLE_GLYCOGEN_IDX] = f["syn_muscle"] - f["brk_muscle"]
        out[..., _MITO_IDX] = f["mito_rate"]
        out[..., _FAT_MASS_IDX] = f["fat_rate"]
        return out
