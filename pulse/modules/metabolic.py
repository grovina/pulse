"""
Metabolic / Energy module.

Blood chemistry homeostasis: glucose, insulin, glucagon, FFA, BHB, lactate,
hepatic glucose output, the two glycogen pools, mitochondrial capacity and the
remote-insulin latent. Mass-action kinetics where the state IS a species in
equilibrium; explicit structural forms where it is not (glucose, the pools,
hepatic output, insulin action — the PRD's "relaxation (c)"). Receives nutrient
appearance from Gut, cortisol from Stress and GLP-1 from Appetite.

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

The balance now, in ONE unit (mg/dL of glucose space per minute; grams for the
pools via ``MG_DL_PER_G``):

    dG    = ra·app·f_plasma                       meal appearance not stored
            + glyco + gng                          hepatic output (two fluxes, below)
            − k_ii·G − X·G − uptake_ex             obligatory, insulin-dependent, exercise
    dLGly = f_liver·app_g − glyco / MG_DL_PER_G
    dMGly = f_muscle·app_g − brk_M
    dHep  = k·((glyco + gng)·VG_DL/BODY_MASS_KG − Hep)   a lagged mg/kg/min readout

    EGP_b   = k_ii · Gb_emb                          per patient (basal EGP scales with Gb)
    glyco   = (1 − f_gng)·EGP_b · (LGly/LGly_b) · g_ins_glyco · g_gn · g_G · mod_L/mod_L_ref
    gng     =      f_gng ·EGP_b · g_cort·√g_gn·g_ffa·g_ins_gng·g_G · mod_H/mod_H_ref

``k_ii`` (obligatory uptake per mg/dL) and ``f_gng`` are learnable POPULATION
scalars; every structural gate is normalized to exactly 1 at the patient's basal
state (``g_ins`` at I = Ib, ``g_gn`` at Gn = Gnb, ``g_cort`` at Cort_b, ``g_ffa``
at FFA_b, ``g_G`` at G ≤ Gb), and each learned modulation ``mod`` is the head's
output DIVIDED by the same head evaluated at the patient's fasted reference
state (typicals with Gb_emb / Ib_emb; not detached — the level is pinned to the
setpoint by construction and the head learns only shape). So at the fasted
reference ``dG = EGP_b − k_ii·Gb = 0`` exactly: Gb is the fixed point, not an
attractor. The fasting fall EMERGES from pool depletion — glycogenolysis is
first order in ``LGly`` so EGP falls toward GNG alone and glucose settles where
obligatory uptake balances it, an absolute floor ``gng/k_ii`` the same for
every Gb (item 3.3). ``Gb_fasted``, the drop and the floor are gone.

Carbon: ``d(G/MG_DL_PER_G) + dLGly + dMGly = app_g − brk_M − (k_ii·G + X·G +
uptake_ex − gng)/MG_DL_PER_G`` identically — one gram of glycogen is 7.72 mg/dL
for everyone; ``ra`` is only the meal-appearance gain it always was.

Gates: insulin and glucagon fire on ``(G − Gb)/30``, FFA on ``(I − Ib)/10``,
with ``Ib = 10·exp(±0.9·tanh(head))`` a zero-init per-patient head like Gb.
Basal insulin secretion falls with sub-Gb glucose through a Hill centred at
Gb with learnable steepness, applied to the BASAL term only (the ``(G/Gb)^5``
on total production cut insulin 41 % at 0.9·Gb — a x5 amplifier of any Gb_emb
error). The thresholds that leaked (glycogen catabolic gates) are DELETED, not
clamped: muscle breakdown is ``relu(act − 0.10)``, liver breakdown is the
insulin gate above. ``mitochondrial_capacity`` has ONE role — a scale on the
clearance of FFA, lactate and BHB — and no other head sees it.
"""

import math

import torch
import torch.nn as nn

from .base import (
    BasalPlusGatedPeakHead, ConstantFluxHead, MassActionModule, SpeciesHead, gate_temp,
)
from ..types import (
    BODY_MASS_KG, GUT_OUTPUT_DIM, MARKER_INDEX, MG_DL_PER_G, MODULE_MARKER_INDICES,
    NORM_CENTER, NORM_SCALE, VG_DL,
)

_GLUCOSE_NORM_SCALE = NORM_SCALE[MARKER_INDEX["glucose"]]
_GLUCOSE_CENTER = NORM_CENTER[MARKER_INDEX["glucose"]]
_INSULIN_NORM_SCALE = NORM_SCALE[MARKER_INDEX["insulin"]]
_INSULIN_CENTER = NORM_CENTER[MARKER_INDEX["insulin"]]
_CORT_NORM_SCALE = NORM_SCALE[MARKER_INDEX["cortisol"]]
_CORT_CENTER = NORM_CENTER[MARKER_INDEX["cortisol"]]
_GN_CENTER = NORM_CENTER[MARKER_INDEX["glucagon"]]
_FFA_CENTER = NORM_CENTER[MARKER_INDEX["ffa"]]

# Coupling inputs: gut outputs (4) + cortisol (1) + glp1 (1) = 6
#
# Iter 90 — INCRETIN PATH. The teacher potentiates glucose-stimulated insulin secretion by
# an incretin factor and the coupling prior registry has always DECLARED `glp1 -> insulin`,
# but glp1 was never fed into this module, so the declared prior had ZERO gradient. Placed
# LAST so the gut(0-3) and cortisol(4) coupling indices are unchanged.
_N_COUPLING = GUT_OUTPUT_DIM + 2

# External inputs: activity (1) + sleep_wake (1) = 2
_N_EXTERNAL = 2

# Species order matches MODULE_MARKER_INDICES["metabolic"]:
# 0: glucose, 1: insulin, 2: glucagon, 3: ffa, 4: bhb, 5: lactate,
# 6: hepatic_output, 7: liver_glycogen (iter 56), 8: muscle_glycogen
# (iter 56), 9: mitochondrial_capacity (iter 55), 10: insulin_action (iter 89).
_TYPICALS = [95.0, 10.0, 70.0, 0.5, 0.1, 1.0, 2.0, 100.0, 400.0, 1.0, 0.0]
# NORM_SCALE per species, in the same order, so the module can rebuild the RAW
# concentration for the mass-action consumption term (iter 95).
_NORM_SCALES = [NORM_SCALE[i] for i in MODULE_MARKER_INDICES["metabolic"]]
# cons_scale = 1/τ in the mass-action rate equation (glycogen and glucose no longer use
# theirs — they have explicit flux forms — but the slots keep the protocol uniform).
_CONS_SCALES = [0.02, 0.1, 0.03, 0.04, 0.03, 0.02, 0.04, 7e-4, 3.3e-5, 2.5e-5, 0.02]

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
_N_SPECIES = 11

# Index of the gut glucose-appearance channel within the `coupling` tensor
# (model.py metabolic_coupling = [gut_outputs(4), cortisol(1), glp1(1)], gut[0]=glucose).
_GUT_GLUCOSE_COUPLING_IDX = 0
# cortisol is metabolic coupling[GUT_OUTPUT_DIM]; activity is external[0].
_CORTISOL_COUPLING_IDX = GUT_OUTPUT_DIM
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
# relu((I − Ib)/10). Band brackets the Bergman literature range with margin.
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
# 1996). The student's insulin_action is rectified (its state floor is 0), so this is
# retained as documentation of the bound the signed form would use.
_INS_DEP_BASAL_FRAC = 0.05
# Basal insulin secretion vs sub-Gb glucose: 2/(1 + (Gb/G)^n) on the BASAL term — 1 at
# Gb, 0.79 at 0.9·Gb, 0.23 at 0.6·Gb (the teacher's floor). Learnable steepness, init 4
# (Polonsky 1988: basal insulin roughly halves while glucose falls ~15 %).
_INS_BASAL_HILL_N_INIT = 4.0
# mg/kg/min per mg/dL/min of glucose space (hepatic_output's unit is mg/kg/min).
_MG_KG_PER_MG_DL = VG_DL / BODY_MASS_KG

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
# Structural substrate term: production ∝ FFA above basal, suppressed by above-basal
# insulin with the teacher's IC50 shape. k_keto init: the teacher's 24 h-fast state (FFA
# ≈ 1.2, BHB 0.87-1.3, k_bhb 0.005) needs ≈ 0.006 mmol/L/min of production at 0.7 mmol/L
# of FFA excess → 0.01/min per mmol/L.
_K_KETO_INIT = 0.01
_KETO_INS_SUPP_INIT = 15.0  # teacher IC50_keto


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


class GlucoseGatedInsulinHead(nn.Module):
    """Insulin production = basal floor + glucose-gated peak amplitude.

    Iter-23 intervention C. A single learned scale cannot satisfy basal-low *and*
    peak-high, so the head structurally separates them:

        prod = basal · basal_gate + softplus(raw_peak) · σ((g − g_thresh) / g_temp)

    Iter 97: ``g`` is the PER-PATIENT deviation ``(G − Gb)/NORM_SCALE`` handed in by
    the module as ``stimulus`` (a population-frame read of ``x`` is the fallback), and
    ``basal_gate`` (handed in by the module) is the Hill-centred fall of basal secretion
    with sub-Gb glucose. Init ``g_thresh`` = 0.5 z (+15 mg/dL above the patient's own
    Gb) with a ~15 mg/dL transition width, so the gate already separates fasting from
    postprandial.
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
        self.g_thresh = nn.Parameter(torch.tensor(0.5))
        self.log_g_temp = nn.Parameter(torch.tensor(-0.7))  # exp(-0.7) ≈ 0.5

    def forward(
        self,
        x: torch.Tensor,
        state_self: torch.Tensor,
        stimulus: torch.Tensor | None = None,
        basal_gate: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if stimulus is None:
            stimulus = x[..., _GLUCOSE_IDX]
        gate = torch.sigmoid((stimulus - self.g_thresh) / gate_temp(self.log_g_temp))
        raw = self.network(x)
        basal = nn.functional.softplus(raw[..., 0])
        if basal_gate is not None:
            basal = basal * basal_gate
        peak = nn.functional.softplus(raw[..., 1])
        cons = nn.functional.softplus(raw[..., 2])
        prod = basal + peak * gate
        return prod, cons


class GlycogenFluxHead(nn.Module):
    """Glycogen as a flux integrator: the head emits the two learned GAINS of

        synthesis   = f_store · fill · appearance_g        (f_store from a softmax the
                                                            module takes over both pools
                                                            and plasma — see fluxes)
        breakdown   = structural flux · mod / mod_ref      (liver: the derived basal
                                                            glycogenolysis times gates;
                                                            muscle: activity-gated)

    In the ``(prod, cons)`` protocol: ``prod`` is the STORE LOGIT (unbounded; the
    module softmaxes it against the other pool and a zero plasma reference so the
    fractions sum to one) and ``cons`` is the non-negative breakdown modulation.

    Iter 97: the learned catabolic gate (``σ((s − c_thresh)/τ)``) is gone. It leaked
    the way every learned threshold here has leaked — 40 % open at activity 0 for
    muscle, 86 % ungated for liver — and the module now applies the gates
    structurally (``relu(act − a_rest)`` / the basal-normalized insulin gate). The
    liver's modulation is further divided by its own value at the patient's fasted
    reference state, so it can only reshape the flux, never move its basal level.
    """

    def __init__(self, input_dim: int, hidden_dim: int, *, init_store_logit: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )
        self.init_store_logit = float(init_store_logit)

    def forward(self, x: torch.Tensor, state_self: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.network(x)
        store_logit = raw[..., 0] + self.init_store_logit
        break_mod = nn.functional.softplus(raw[..., 1])
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
            # Iter 97: every head except mito's own reads the module input WITHOUT the
            # mitochondrial_capacity column (3.11); glucose and insulin_action have fully
            # structural rates and own no parameters (the 16.5 % dead-parameter finding).
            head_factories={
                _GLUCOSE_IDX: lambda inp, hd: ConstantFluxHead(),
                _INSULIN_IDX: _without_mito(GlucoseGatedInsulinHead),
                # Gated peak heads (iter 70/72: structurally load-bearing for the HPA
                # coupling — reverting them broke ACTH 11x). Stimuli are handed in by the
                # module as per-patient deviations; the idx here is the fallback.
                _GLUCAGON_IDX: lambda inp, hd: BasalPlusGatedPeakHead(
                    inp - 1, hd, stimulus_idx=_GLUCOSE_IDX, gate_dir=-1,
                    init_thresh=-0.3, init_log_temp=-0.7,
                ),
                _FFA_IDX: lambda inp, hd: BasalPlusGatedPeakHead(
                    inp - 1, hd, stimulus_idx=_INSULIN_IDX, gate_dir=-1,
                    init_thresh=-0.2, init_log_temp=-0.7,
                ),
                _BHB_IDX: _without_mito(SpeciesHead),
                _LACTATE_IDX: _without_mito(SpeciesHead),
                # hepatic_output: prod = GNG modulation (normalized at the reference),
                # cons = the readout's relaxation rate.
                _HEPATIC_IDX: _without_mito(SpeciesHead),
                _LIVER_GLYCOGEN_IDX: lambda inp, hd: GlycogenFluxHead(
                    inp - 1, hd,
                    init_store_logit=math.log(_LIVER_STORE_FRAC_INIT / _PLASMA_FRAC_INIT)),
                _MUSCLE_GLYCOGEN_IDX: lambda inp, hd: GlycogenFluxHead(
                    inp - 1, hd,
                    init_store_logit=math.log(_MUSCLE_STORE_FRAC_INIT / _PLASMA_FRAC_INIT)),
                _MITO_IDX: SpeciesHead,  # the one head that reads its own column
                _INSULIN_ACTION_IDX: lambda inp, hd: ConstantFluxHead(),
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
        self.log_ins_basal_n = nn.Parameter(torch.tensor(_inverse_softplus(_INS_BASAL_HILL_N_INIT)))
        # Structural rate-of-appearance gain Ra on the gut glucose-appearance flux
        # (iter 80): the meal-appearance gain, and nothing else (iter 97).
        self.log_ra = nn.Parameter(torch.tensor(_inverse_softplus(_RA_INIT)))
        self.log_si = nn.Parameter(torch.tensor(_logit((_SI_INIT - _SI_MIN) / _SI_RANGE)))
        self.log_p2 = nn.Parameter(torch.tensor(_logit((_P2_INIT - _P2_MIN) / _P2_RANGE)))
        self.log_k_act = nn.Parameter(torch.tensor(_logit((_KACT_INIT - _KACT_MIN) / _KACT_RANGE)))
        # Ketogenesis: half-suppression constant (µU/mL above Ib) and substrate gain.
        self.log_keto_ins_supp = nn.Parameter(torch.tensor(_inverse_softplus(_KETO_INS_SUPP_INIT)))
        self.log_k_keto = nn.Parameter(torch.tensor(_inverse_softplus(_K_KETO_INIT)))

        # Per-patient setpoint heads. Final layers zero-init ⇒ Gb = 95, Ib = 10, Ra =
        # softplus(log_ra) for every embedding at cold start; authority grows in training.
        _bh = max(8, hidden_dim // 4)

        def _zero_head() -> nn.Sequential:
            net = nn.Sequential(nn.Linear(embedding_dim, _bh), nn.Tanh(), nn.Linear(_bh, 1))
            with torch.no_grad():
                net[-1].weight.zero_()
                net[-1].bias.zero_()
            return net

        self.glucose_baseline_net = _zero_head()   # Gb (iter 81)
        self.insulin_baseline_net = _zero_head()   # Ib (iter 97)
        self.ra_baseline_net = _zero_head()        # Ra (iter 88)

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
        *,
        glucose_dev: torch.Tensor | None = None,
        insulin_dev: torch.Tensor | None = None,
        basal_gate: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-species head outputs. ``glucose_dev`` / ``insulin_dev`` are the
        per-patient deviations ``(G − Gb)/30`` and ``(I − Ib)/10`` that gate insulin,
        glucagon and FFA; ``basal_gate`` scales basal insulin secretion. Without them
        the gates fall back to the population frame."""
        x, x_no_mito = self._head_input(state, coupling, external, embedding, time_features)
        prods: list[torch.Tensor] = []
        conss: list[torch.Tensor] = []
        for i, head in enumerate(self.heads):
            xi = x if i == _MITO_IDX else x_no_mito
            if i == _INSULIN_IDX:
                p, c = head(xi, state[..., i], stimulus=glucose_dev, basal_gate=basal_gate)
            elif i == _GLUCAGON_IDX:
                p, c = head(xi, state[..., i], stimulus=glucose_dev)
            elif i == _FFA_IDX:
                p, c = head(xi, state[..., i], stimulus=insulin_dev)
            else:
                p, c = head(xi, state[..., i])
            prods.append(p)
            conss.append(c)
        return torch.stack(prods, dim=-1), torch.stack(conss, dim=-1)

    def reference_modulations(
        self,
        gb: torch.Tensor,
        ib: torch.Tensor,
        embedding: torch.Tensor,
        time_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The liver-breakdown and GNG head modulations at the patient's FASTED
        REFERENCE state: every species at typical except glucose at Gb_emb and insulin
        at Ib_emb, no appearance, cortisol and GLP-1 at basal, awake at rest, at the
        current time of day (so time-of-day cancels and cannot move the basal level).
        Not detached: the head learns shape only; the level is the setpoint's."""
        batch_shape = gb.shape
        state = torch.zeros(*batch_shape, _N_SPECIES, dtype=gb.dtype, device=gb.device)
        state[..., _GLUCOSE_IDX] = (gb - _GLUCOSE_CENTER) / _GLUCOSE_NORM_SCALE
        state[..., _INSULIN_IDX] = (ib - _INSULIN_CENTER) / _INSULIN_NORM_SCALE
        coupling = torch.zeros(*batch_shape, _N_COUPLING, dtype=gb.dtype, device=gb.device)
        external = torch.zeros(*batch_shape, _N_EXTERNAL, dtype=gb.dtype, device=gb.device)
        external[..., _SLEEP_EXTERNAL_IDX] = 1.0
        _, x_ref = self._head_input(state, coupling, external, embedding, time_features)
        _, mod_liver_ref = self.heads[_LIVER_GLYCOGEN_IDX](x_ref, state[..., _LIVER_GLYCOGEN_IDX])
        mod_gng_ref, _ = self.heads[_HEPATIC_IDX](x_ref, state[..., _HEPATIC_IDX])
        return mod_liver_ref, mod_gng_ref

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
        hep = raw[..., _HEPATIC_IDX]
        lgly = raw[..., _LIVER_GLYCOGEN_IDX]
        mgly = raw[..., _MUSCLE_GLYCOGEN_IDX]
        mito = raw[..., _MITO_IDX]
        xa = raw[..., _INSULIN_ACTION_IDX]
        cort = (_CORT_CENTER + _CORT_NORM_SCALE * coupling[..., _CORTISOL_COUPLING_IDX]).clamp(min=0.05)
        act = external[..., _ACTIVITY_EXTERNAL_IDX]

        gb = self.glucose_setpoint_raw(embedding)
        ib = self.insulin_setpoint_raw(embedding)
        ra = self.appearance_gain(embedding)
        glucose_dev = (g - gb) / _GLUCOSE_NORM_SCALE
        insulin_dev = (ins - ib) / _INSULIN_NORM_SCALE
        ins_excess = relu(ins - ib)
        ins_basal_n = nn.functional.softplus(self.log_ins_basal_n)
        # Basal secretion falls with sub-Gb glucose (1 at and above Gb, on the basal term).
        ins_basal_gate = _hill_centred(torch.minimum(g, gb), gb, ins_basal_n)

        prod_raw, cons_raw = self.species_fluxes(
            state, coupling, external, embedding, time_features,
            glucose_dev=glucose_dev, insulin_dev=insulin_dev, basal_gate=ins_basal_gate)
        mod_liver_ref, mod_gng_ref = self.reference_modulations(gb, ib, embedding, time_features)

        # --- appearance and its three destinations (mg/dL/min; grams via MG_DL_PER_G) ---
        app_eff = ra * coupling[..., _GUT_GLUCOSE_COUPLING_IDX]     # bioavailable appearance
        app_g = app_eff / MG_DL_PER_G                               # g/min
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
        f_plasma = 1.0 - f_liver - f_muscle                          # ≥ frac[..., 2] > 0
        syn_liver = f_liver * app_g                                  # g/min
        syn_muscle = f_muscle * app_g
        appearance_plasma = app_eff * f_plasma                       # mg/dL/min

        # --- hepatic glucose output: two fluxes from a derived basal EGP ----------------
        k_ii = self.k_ii()
        egp_b = k_ii * gb                                            # mg/dL/min, per patient
        f_gng = torch.sigmoid(self.logit_f_gng)
        glyc_k = nn.functional.softplus(self.log_glyc_ins_k)
        gng_k = nn.functional.softplus(self.log_gng_ins_k)
        g_ins_glyco = _ins_gate(ins, ib, glyc_k, _GLYC_INS_N)
        g_ins_gng = _ins_gate(ins, ib, gng_k, _GNG_INS_N)
        g_gn = _hill_centred(gn, _GN_CENTER, _HGO_GN_N)
        g_cort = 1.0 + _GNG_CORT_AMP * torch.tanh(torch.log(cort / _CORT_CENTER))
        g_ffa = (ffa.clamp(min=1e-3) / _FFA_CENTER) ** _GNG_FFA_EXP
        g_g = (gb / torch.maximum(g, gb)) ** _HEP_AUTOREG_M       # one-sided: 1 at G ≤ Gb
        mod_liver = cons_raw[..., _LIVER_GLYCOGEN_IDX] / mod_liver_ref
        mod_gng = prod_raw[..., _HEPATIC_IDX] / mod_gng_ref
        glycogenolysis_plasma = ((1.0 - f_gng) * egp_b * (lgly / _LIVER_GLY_CENTER)
                                 * g_ins_glyco * g_gn * g_g * mod_liver)
        gng_plasma = (f_gng * egp_b * g_cort * torch.sqrt(g_gn) * g_ffa * g_ins_gng * g_g
                      * mod_gng)
        brk_liver = glycogenolysis_plasma / MG_DL_PER_G             # g/min, the same flux

        # --- muscle glycogenolysis (activity-gated, oxidized in situ) ------------------
        avail_m = mgly / (mgly + _MUSCLE_GLY_K)
        brk_muscle = (_MUSCLE_GLY_FLUX * cons_raw[..., _MUSCLE_GLYCOGEN_IDX]
                      * relu(act - _MUSCLE_ACT_REST) * avail_m)

        # --- glucose uptake --------------------------------------------------------------
        si = _SI_MIN + _SI_RANGE * torch.sigmoid(self.log_si)
        uptake_ii = k_ii * g                                         # obligatory
        uptake_id = si * xa * g                                      # insulin-dependent, a sink
        k_act = _KACT_MIN + _KACT_RANGE * torch.sigmoid(self.log_k_act)
        exercise_uptake = k_act * act * relu(g - 0.8 * gb)

        # --- insulin, insulin action -----------------------------------------------------
        ins_production = prod_raw[..., _INSULIN_IDX] * self.prod_scale[_INSULIN_IDX]
        ins_clearance = cons_raw[..., _INSULIN_IDX] * self.cons_scale[_INSULIN_IDX] * ins
        p2 = _P2_MIN + _P2_RANGE * torch.sigmoid(self.log_p2)
        xa_rate = p2 * (relu(insulin_dev) - xa)

        # --- hepatic output readout (mg/kg/min), lagged ----------------------------------
        hep_target = (glycogenolysis_plasma + gng_plasma) * _MG_KG_PER_MG_DL
        hep_rate = cons_raw[..., _HEPATIC_IDX] * self.cons_scale[_HEPATIC_IDX] * (hep_target - hep)

        # --- ketogenesis -------------------------------------------------------------------
        keto_ins_supp = nn.functional.softplus(self.log_keto_ins_supp)
        ketogenesis = (nn.functional.softplus(self.log_k_keto) * relu(ffa - _FFA_CENTER)
                       / (1.0 + ins_excess / keto_ins_supp))

        return {
            "gb": gb, "ib": ib, "ra": ra, "egp_b": egp_b, "k_ii": k_ii, "f_gng": f_gng,
            "glucose_dev": glucose_dev, "insulin_dev": insulin_dev,
            "ins_basal_gate": ins_basal_gate,
            "prod_raw": prod_raw, "cons_raw": cons_raw, "mito": mito,
            "app_eff": app_eff, "app_g": app_g,
            "f_liver": f_liver, "f_muscle": f_muscle, "f_plasma": f_plasma,
            "syn_liver": syn_liver, "syn_muscle": syn_muscle,
            "brk_liver": brk_liver, "brk_muscle": brk_muscle,
            "appearance_plasma": appearance_plasma,
            "glycogenolysis_plasma": glycogenolysis_plasma, "gng_plasma": gng_plasma,
            "g_ins_glyco": g_ins_glyco, "g_ins_gng": g_ins_gng, "g_gn": g_gn,
            "g_cort": g_cort, "g_ffa": g_ffa, "g_g": g_g,
            "mod_liver": mod_liver, "mod_gng": mod_gng,
            "uptake_ii": uptake_ii, "uptake_id": uptake_id, "exercise_uptake": exercise_uptake,
            "ins_production": ins_production, "ins_clearance": ins_clearance,
            "xa_rate": xa_rate, "hep_target": hep_target, "hep_rate": hep_rate,
            "ketogenesis": ketogenesis,
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
        # Mass-action default (concentration frame, iter 95) for the species that are
        # genuinely in equilibrium: glucagon, mito. The oxidative species get mito's ONE
        # structural role — a scale on their clearance.
        rates = prod_raw * self.prod_scale - cons_raw * self.cons_scale * raw
        out = rates.clone()
        for idx in (_FFA_IDX, _BHB_IDX, _LACTATE_IDX):
            out[..., idx] = (prod_raw[..., idx] * self.prod_scale[idx]
                             - cons_raw[..., idx] * self.cons_scale[idx] * mito * raw[..., idx])
        out[..., _BHB_IDX] = out[..., _BHB_IDX] + f["ketogenesis"]
        out[..., _GLUCOSE_IDX] = (
            f["appearance_plasma"] + f["glycogenolysis_plasma"] + f["gng_plasma"]
            - f["uptake_ii"] - f["uptake_id"] - f["exercise_uptake"]
        )
        out[..., _INSULIN_IDX] = f["ins_production"] - f["ins_clearance"]
        out[..., _INSULIN_ACTION_IDX] = f["xa_rate"]
        out[..., _HEPATIC_IDX] = f["hep_rate"]
        out[..., _LIVER_GLYCOGEN_IDX] = f["syn_liver"] - f["brk_liver"]
        out[..., _MUSCLE_GLYCOGEN_IDX] = f["syn_muscle"] - f["brk_muscle"]
        return out
