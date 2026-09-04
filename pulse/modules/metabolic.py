"""
Metabolic / Energy module.

Blood chemistry homeostasis: glucose, insulin, glucagon, FFA, BHB, lactate,
hepatic glucose output, the two glycogen pools, mitochondrial capacity and the
remote-insulin latent. Mass-action kinetics where the state IS a species in
equilibrium; explicit structural forms where it is not (glucose, the pools,
hepatic output, insulin action — the PRD's "relaxation (c)"). Receives nutrient
appearance from Gut, cortisol from Stress and GLP-1 from Appetite.

ITER 97 — THE CARBON BUDGET CLOSES, AND EVERY GLUCOSE-SIDE GATE IS PER PATIENT
(review 2026-09-04, items 2.2, 2.6, 3.4, 3.10, 3.11). What was wrong, measured
on the iter-96 artifact:

  * Glycogen was a SHADOW POOL: synthesis consumed ``relu(appearance)`` without
    debiting glucose, and breakdown reached glucose only through the setpoint
    drop. On a eucaloric day the liver "synthesised" 77 g of a 210 g
    carbohydrate intake (37 %; Taylor 1996: 19 %) and none of it left plasma.
  * Muscle glycogen was spent AT REST (learned catabolic gate 0.40 at activity
    0): a 36 h resting fast took it 400 -> 184 g. The teacher is
    ``k·max(act − 0.10, 0)`` — zero at rest by construction. Liver breakdown
    was 86 % ungated.
  * Gates used POPULATION thresholds (insulin gate at 116.5 mg/dL, glucagon at
    83.4, FFA at insulin 7.8, ``insulin_action`` lagging ``relu(I − 10)``)
    while the setpoints were per patient: Gb = 120 fasted with insulin 23.75
    and a standing +34 % clearance; Gb = 75 fasted to 54 mg/dL.
  * ``−(Sg + X)(G − Gb)`` made insulin action a glucose SOURCE below the
    setpoint (+0.087 mg/dL/min at G = 85, Gb = 95, insulin 30). Bergman is
    ``−(Sg + X)·G + Sg·Gb``.
  * ``mitochondrial_capacity`` — an unsupervised weeks-τ latent — was an input
    to all 11 heads; 1.0 -> 0.7 moved the insulin rate by +28/h.
  * BHB had no substrate term: a 36 h fast reached 0.12 vs the teacher's 1.3.

The structural forms now (raw units throughout; ``fluxes`` returns every term):

    dG  = −Sg·(G − Gb_fasted) − X·G                       clearance (Bergman)
          + c·app_g·(1 − f_L·fill_L − f_M·fill_M)          appearance NOT stored
          + c·brk_L                                        hepatic glycogenolysis
          + EGP_extra(cortisol, glucagon) − exercise uptake
    dLGly = f_L·fill_L·app_g − brk_L      brk_L = F_L·softplus(net)·ins_gate·avail_L
    dMGly = f_M·fill_M·app_g − brk_M      brk_M = F_M·softplus(net)·relu(act − a_rest)·avail_M
    dHep  = k·(HEP_PER_G·brk_L + GNG(net) − Hep)          hepatic output READS breakdown
    dXa   = p2·(relu((I − Ib)/10) − Xa)                    per-patient Ib
    dBHB  = basal(net) + k_keto·relu(FFA − FFA_b)·keto_ins_gate − clearance·mito·BHB
    dFFA, dLac: production(net) − clearance(net)·mito·X    (mito's ONE role)

with ``(f_L, f_M, f_plasma) = softmax(store logits, 0)`` so the three
destinations of absorbed carbohydrate sum to exactly one, ``app_g`` the gut
appearance in grams/min, and ``c = Ra·U`` the per-patient conversion from grams
to mg/dL (``U`` = the gut's appearance-units-per-gram). So

    d(G/c) + dLGly + dMGly = app_g − (clearances + exercise − EGP_extra)/c − brk_M

holds identically — carbon that leaves plasma into a pool is debited, carbon
that leaves a pool into plasma is credited, and muscle glycogen is oxidised
locally (no G6Pase). ``tests/test_iter97_student_metabolic.py`` integrates a
eucaloric day and checks the ledger.

Gates: insulin and glucagon fire on ``(G − Gb)/30``, FFA on ``(I − Ib)/10``,
with ``Ib = 10·exp(±0.9·tanh(head))`` a zero-init per-patient head like Gb.
The fasting drop is ADDITIVE with an ABSOLUTE floor
(``Gb_fasted = max(Gb − drop·depleted, 60)``) so it no longer scales with Gb.
Thresholds that previously leaked (the glycogen catabolic gates) are DELETED,
not clamped: the liver gate is the teacher's saturating
``1/(1 + relu(I − Ib)/K)`` with K learned, the muscle gate is ``relu(act −
0.10)``.
"""

import math

import torch
import torch.nn as nn

from .base import (
    BasalPlusGatedPeakHead, ConstantFluxHead, MassActionModule, SpeciesHead, gate_temp,
)
from .gut import APPEARANCE_UNITS_PER_G
from ..types import GUT_OUTPUT_DIM, MARKER_INDEX, MODULE_MARKER_INDICES, NORM_CENTER, NORM_SCALE

_GLUCOSE_NORM_SCALE = NORM_SCALE[MARKER_INDEX["glucose"]]
_GLUCOSE_CENTER = NORM_CENTER[MARKER_INDEX["glucose"]]
_INSULIN_NORM_SCALE = NORM_SCALE[MARKER_INDEX["insulin"]]
_INSULIN_CENTER = NORM_CENTER[MARKER_INDEX["insulin"]]
_CORT_NORM_SCALE = NORM_SCALE[MARKER_INDEX["cortisol"]]
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

# Lag rate p2 = 1/τ for insulin action, bounded so τ ∈ [4, 100] min (teacher p2 ≈ 0.03,
# τ ≈ 33 min). Floored so the state can't freeze; capped so it can't collapse to the
# old instantaneous behaviour. Euler-stable (p2·dt ≤ 0.25 ≪ 2).
_P2_MIN = 0.01
_P2_RANGE = 0.24
# Max per-patient fasting-glucose offset, z-score units around 95 mg/dL (iter 90: ±2.2
# gives Gb ∈ [29, 161], covering the teacher's lognormal spread and the benchmark's 60-120).
_GLUCOSE_BASELINE_MAX_Z = 2.2
# Iter 97: per-patient basal insulin in LOG space so it is positive by construction:
# Ib = 10·exp(±0.9·tanh) ∈ [4.1, 24.6] µU/mL — the teacher varies Ib with σ = 0.4
# lognormal loaded on insulin resistance, i.e. roughly this span at ±2σ.
_IB_LOG_MAX = 0.9
# Max per-patient meal-appearance (Ra) log-gain offset, pre-softplus units (iter 88).
_RA_BASELINE_MAX_Z = 1.0
# Sg: RAW per-minute glucose effectiveness, literature band [0.005, 0.05]/min (τ 20-200
# min), init 0.018 (the teacher's value). See the iter-90 frame fix history in git.
_SG_MIN = 0.005
_SG_RANGE = 0.045
_SG_INIT = 0.018  # teacher full_body.py PatientParams.Sg
# Si: RAW per-minute per normalized-insulin-unit; X = Si·Xa with Xa the lagged
# relu((I − Ib)/10). Band brackets the Bergman literature range with margin.
_SI_MIN = 0.0005
_SI_RANGE = 0.0195
_SI_INIT = 0.004  # = 10 × teacher Si (insulin normalization)
# Counter-regulatory gains (iter 90), raw per-minute, init at the teacher's values.
_KCORT_MIN, _KCORT_RANGE, _KCORT_INIT = 0.0, 0.006, 0.0012    # teacher cort_gluco
_KGN_MIN, _KGN_RANGE, _KGN_INIT = 0.0, 0.10, 0.02             # teacher glucagon->glucose
_KACT_MIN, _KACT_RANGE, _KACT_INIT = 0.0, 0.06, 0.02          # teacher exercise uptake
# OFF-MANIFOLD SAFETY RAIL on the counter-regulatory extras (mg/dL/min) — NOT a
# physiological law; inactive in distribution (≤0.5 % deviation across the teacher's
# 0-1.8 range) and only bounds an untrained glucagon pinned at its state clamp, which at
# k_gn = 0.02 alone injected 8 mg/dL/min (the iter-83-85 runaway class).
_EGP_MAX = 8.0

# Iter 97 (was iter 95 A1): the defended level falls as the liver pool empties — Cahill
# 2006: as hepatic glycogen empties, gluconeogenesis cannot fully replace glycogenolysis.
#     Gb_fasted = max(Gb − drop·glyco_depleted, GB_FLOOR)      glyco_depleted = relu(1 − LGly/LGly_b)
# The drop is now ABSOLUTE (mg/dL, learned, init 35 = the teacher's 0.55·(1 − 0.62)·Gb at
# a typical Gb, where its own floor binds) and the floor is ABSOLUTE (60 mg/dL — the
# brain's floor is not a fraction of the setpoint; the teacher's own hypoglycemia response
# uses an absolute 70). Through iter 96 both scaled WITH Gb (`Gb·(1 − drop)`, floor
# `0.62·Gb`), so a Gb = 75 patient fasted to 54 mg/dL by construction. Identity at the fed
# state (LGly = LGly_b) either way, so the fed regime is untouched.
_GB_DROP_ABS_INIT = 35.0
_GB_FLOOR_ABS = 60.0
# Insulin's setpoint falls with the glucose ratio (iter 95 A5, Polonsky 1988):
# effective production ∝ min(G/Gb, 1)^5. Identity whenever G ≥ Gb. Kept at the
# teacher's calibrated shape; referenced to the FED gb, not to the falling one.
_FAST_INS_EXP = 5.0

# --- glycogen pools --------------------------------------------------------------------
_LIVER_GLY_CENTER = NORM_CENTER[MARKER_INDEX["liver_glycogen"]]      # 100 g
_MUSCLE_GLY_CENTER = NORM_CENTER[MARKER_INDEX["muscle_glycogen"]]    # 400 g
# Store capacity as a multiple of typical (supercompensation, Bergstrom & Hultman 1966 —
# a real ~20-40 % overshoot) and the WIDTH over which synthesis closes as the store nears
# capacity. Iter 97: the taper is `sigmoid((cap − pool)/width)` — 0.95 at typical, 0.5 at
# the cap — instead of `relu(1 − pool/cap)`, which was 0.23 AT THE FED LEVEL and gave the
# pool an implicit setpoint below typical (the shape error the teacher fixed in iter 96 and
# the review notes the student still carried).
_GLY_CAPACITY_FRAC = 1.3
_GLY_FILL_WIDTH_FRAC = 0.1
# Breakdown availability: Michaelis in the pool, `pool/(pool + K)`, zero at zero by
# construction (the pool cannot go negative) — the teacher's form and constants.
_LIVER_GLY_K = 35.0    # teacher glyc_K_L (g)
_MUSCLE_GLY_K = 150.0  # teacher glyc_K_M (g)
# Breakdown flux scales (g/min per unit of the head's softplus drive, ≈0.7 at init).
#   Liver: 0.08·0.7·0.74 ≈ 0.041 g/min at a full store = 59 g/day, the teacher's
#   post-absorptive glycogenolysis (0.062·0.74 = 0.046).
#   Muscle: 3.0·0.7·0.73·relu(1 − 0.1) ≈ 1.4 g/min at a maximal bout, i.e. the −150 g /
#   2 h cohort target (teacher k = 3.5).
_LIVER_GLY_FLUX = 0.08
_MUSCLE_GLY_FLUX = 3.0
# Activity at or below this is rest for muscle glycogen: `relu(act − a_rest)` is zero by
# construction there (Coppack 1989; teacher act_rest_M). A CONSTANT, not a learned
# threshold — iter 96's S1 lesson is that a learnable threshold cannot be stopped from
# leaking (the previous gate learned its way to 40 % open at activity 0).
_MUSCLE_ACT_REST = 0.10
# Above-basal insulin (µU/mL) that halves hepatic glycogenolysis; learned, init teacher.
_GLYC_INS_SUPP_INIT = 15.0  # teacher glyc_ins_supp
# Prior fractions of absorbed carbohydrate stored: liver ~20 % (Taylor 1996: 19 %), muscle
# ~15 %, the rest to plasma. Softmax logits relative to the plasma reference (logit 0).
_LIVER_STORE_FRAC_INIT = 0.20
_MUSCLE_STORE_FRAC_INIT = 0.15
_PLASMA_FRAC_INIT = 1.0 - _LIVER_STORE_FRAC_INIT - _MUSCLE_STORE_FRAC_INIT
# hepatic_output is reported in the teacher's unit (≈ mg/kg/min at 70 kg): 1 g/min of
# hepatic glucose release = 1000/70 of those units. Its GNG component is the head's learned
# production scaled by the marker's typical, so at init GNG ≈ 1.4 and glycogenolysis ≈ 0.6
# sum to the typical 2.0 (teacher hep_glyco_frac 0.6 early in a fast).
_HEP_UNITS_PER_G_MIN = 1000.0 / 70.0

# --- ketogenesis -----------------------------------------------------------------------
# Structural substrate term: production ∝ FFA above basal, suppressed by above-basal
# insulin with the teacher's IC50 shape. k_keto init: the teacher's 24 h-fast state (FFA
# ≈ 1.2, BHB 0.87-1.3, k_bhb 0.005) needs ≈ 0.006 mmol/L/min of production at 0.7 mmol/L
# of FFA excess → 0.01/min per mmol/L.
_K_KETO_INIT = 0.01
_KETO_INS_SUPP_INIT = 15.0  # teacher IC50_keto
_GLUCOSE_APPEARANCE_UNITS_PER_G = float(APPEARANCE_UNITS_PER_G[0])


def _logit(p: float) -> float:
    """Inverse sigmoid — init a sigmoid-bounded parameter at a target value."""
    return math.log(p / (1.0 - p))


def _inverse_softplus(y: float) -> float:
    return math.log(math.expm1(y))


class GlucoseGatedInsulinHead(nn.Module):
    """Insulin production = basal floor + glucose-gated peak amplitude.

    Iter-23 intervention C. A single learned scale cannot satisfy basal-low *and*
    peak-high, so the head structurally separates them:

        prod = softplus(raw_basal) + softplus(raw_peak) · σ((g − g_thresh) / g_temp)

    Iter 97: ``g`` is the PER-PATIENT deviation ``(G − Gb)/NORM_SCALE`` handed in by
    the module as ``stimulus`` (a population-frame read of ``x`` is the fallback).
    Init ``g_thresh`` = 0.5 z (+15 mg/dL above the patient's own Gb) with a ~15 mg/dL
    transition width, so the gate already separates fasting from postprandial.
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if stimulus is None:
            stimulus = x[..., _GLUCOSE_IDX]
        gate = torch.sigmoid((stimulus - self.g_thresh) / gate_temp(self.log_g_temp))
        raw = self.network(x)
        basal = nn.functional.softplus(raw[..., 0])
        peak = nn.functional.softplus(raw[..., 1])
        cons = nn.functional.softplus(raw[..., 2])
        prod = basal + peak * gate
        return prod, cons


class GlycogenFluxHead(nn.Module):
    """Glycogen as a flux integrator: the head emits the two learned GAINS of

        synthesis = f_store · fill · appearance_g        (f_store from a softmax the
                                                          module takes over both pools
                                                          and plasma — see forward)
        breakdown = flux_scale · softplus(net) · structural gate · availability

    In the ``(prod, cons)`` protocol: ``prod`` is the STORE LOGIT (unbounded; the
    module softmaxes it against the other pool and a zero plasma reference so the
    fractions sum to one) and ``cons`` is the non-negative breakdown drive.

    Iter 97: the learned catabolic gate (``σ((s − c_thresh)/τ)``) is gone. It leaked
    the way every learned threshold here has leaked — 40 % open at activity 0 for
    muscle, 86 % ungated for liver — and the module now applies the gate
    structurally (``relu(act − a_rest)`` / ``1/(1 + relu(I − Ib)/K)``). The head
    still sees insulin, activity, sleep and the embedding in ``x``, so the
    modulation the physiology genuinely has (insulin drive on synthesis, training
    status on breakdown capacity) remains learnable; only the zero-at-rest
    guarantee is taken out of the optimizer's hands.
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
        break_drive = nn.functional.softplus(raw[..., 1])
        return store_logit, break_drive


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

        self.log_sg = nn.Parameter(torch.tensor(_logit((_SG_INIT - _SG_MIN) / _SG_RANGE)))
        # Structural rate-of-appearance gain Ra on the gut glucose-appearance flux
        # (iter 80): Ra·appearance is the minimal-model Ra(t) source, and Ra·U is this
        # patient's grams → mg/dL conversion for the carbon budget.
        self.log_ra = nn.Parameter(torch.tensor(math.log(0.55)))
        self.log_si = nn.Parameter(torch.tensor(_logit((_SI_INIT - _SI_MIN) / _SI_RANGE)))
        _p2_p0 = (0.03 - _P2_MIN) / _P2_RANGE
        self.log_p2 = nn.Parameter(torch.tensor(math.log(_p2_p0 / (1.0 - _p2_p0))))
        self.log_k_cort = nn.Parameter(torch.tensor(_logit((_KCORT_INIT - _KCORT_MIN) / _KCORT_RANGE)))
        self.log_k_gn = nn.Parameter(torch.tensor(_logit((_KGN_INIT - _KGN_MIN) / _KGN_RANGE)))
        self.log_k_act = nn.Parameter(torch.tensor(_logit((_KACT_INIT - _KACT_MIN) / _KACT_RANGE)))
        # Iter 97: absolute fasting drop (mg/dL), softplus-positive.
        self.log_gb_drop_abs = nn.Parameter(torch.tensor(_inverse_softplus(_GB_DROP_ABS_INIT)))
        # Iter 97: half-suppression constants (µU/mL above Ib) for hepatic glycogenolysis
        # and ketogenesis, and the ketogenic substrate gain.
        self.log_glyc_ins_supp = nn.Parameter(torch.tensor(_inverse_softplus(_GLYC_INS_SUPP_INIT)))
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

    # ---- heads -----------------------------------------------------------------------

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-species head outputs. ``glucose_dev`` / ``insulin_dev`` are the
        per-patient deviations ``(G − Gb)/30`` and ``(I − Ib)/10`` that gate insulin,
        glucagon and FFA; without them the gates fall back to the population frame."""
        x = torch.cat([state, coupling, external, embedding, time_features], dim=-1)
        x_no_mito = torch.cat([x[..., :_MITO_IDX], x[..., _MITO_IDX + 1:]], dim=-1)
        prods: list[torch.Tensor] = []
        conss: list[torch.Tensor] = []
        for i, head in enumerate(self.heads):
            xi = x if i == _MITO_IDX else x_no_mito
            if i in (_INSULIN_IDX, _GLUCAGON_IDX):
                p, c = head(xi, state[..., i], stimulus=glucose_dev)
            elif i == _FFA_IDX:
                p, c = head(xi, state[..., i], stimulus=insulin_dev)
            else:
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
        """Every named term of the metabolic ODE, in raw units. ``forward`` assembles the
        rates from these; the carbon-budget test and the probes read them directly."""
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

        gb = self.glucose_setpoint_raw(embedding)
        ib = self.insulin_setpoint_raw(embedding)
        ra = self.appearance_gain(embedding)
        glucose_dev = (g - gb) / _GLUCOSE_NORM_SCALE
        insulin_dev = (ins - ib) / _INSULIN_NORM_SCALE
        ins_excess = relu(ins - ib)

        prod_raw, cons_raw = self.species_fluxes(
            state, coupling, external, embedding, time_features,
            glucose_dev=glucose_dev, insulin_dev=insulin_dev)

        # --- appearance and its three destinations -------------------------------------
        app = coupling[..., _GUT_GLUCOSE_COUPLING_IDX]           # teacher units (≥ 0)
        app_g = app / _GLUCOSE_APPEARANCE_UNITS_PER_G           # g/min
        c_plasma = ra * _GLUCOSE_APPEARANCE_UNITS_PER_G          # mg/dL per gram, per patient
        store_logits = torch.stack([
            prod_raw[..., _LIVER_GLYCOGEN_IDX],
            prod_raw[..., _MUSCLE_GLYCOGEN_IDX],
            torch.zeros_like(app),
        ], dim=-1)
        frac = torch.softmax(store_logits, dim=-1)
        fill_l = torch.sigmoid(
            (_GLY_CAPACITY_FRAC * _LIVER_GLY_CENTER - lgly) / (_GLY_FILL_WIDTH_FRAC * _LIVER_GLY_CENTER))
        fill_m = torch.sigmoid(
            (_GLY_CAPACITY_FRAC * _MUSCLE_GLY_CENTER - mgly) / (_GLY_FILL_WIDTH_FRAC * _MUSCLE_GLY_CENTER))
        f_liver = frac[..., 0] * fill_l
        f_muscle = frac[..., 1] * fill_m
        f_plasma = 1.0 - f_liver - f_muscle                      # ≥ frac[..., 2] > 0
        syn_liver = f_liver * app_g                              # g/min
        syn_muscle = f_muscle * app_g
        appearance_plasma = c_plasma * f_plasma * app_g          # mg/dL/min (= ra·app·f_plasma)

        # --- glycogen breakdown ----------------------------------------------------------
        glyc_ins_supp = nn.functional.softplus(self.log_glyc_ins_supp)
        liver_ins_gate = 1.0 / (1.0 + ins_excess / glyc_ins_supp)
        avail_l = lgly / (lgly + _LIVER_GLY_K)
        avail_m = mgly / (mgly + _MUSCLE_GLY_K)
        act = external[..., _ACTIVITY_EXTERNAL_IDX]
        brk_liver = _LIVER_GLY_FLUX * cons_raw[..., _LIVER_GLYCOGEN_IDX] * liver_ins_gate * avail_l
        brk_muscle = (_MUSCLE_GLY_FLUX * cons_raw[..., _MUSCLE_GLYCOGEN_IDX]
                      * relu(act - _MUSCLE_ACT_REST) * avail_m)
        glycogenolysis_plasma = c_plasma * brk_liver             # mg/dL/min

        # --- glucose clearance and counter-regulation ------------------------------------
        sg = _SG_MIN + _SG_RANGE * torch.sigmoid(self.log_sg)
        si = _SI_MIN + _SI_RANGE * torch.sigmoid(self.log_si)
        x_ins = si * xa
        glyco_depleted = relu(1.0 - lgly / _LIVER_GLY_CENTER)
        gb_drop = nn.functional.softplus(self.log_gb_drop_abs)
        gb_fasted = torch.maximum(gb - gb_drop * glyco_depleted, torch.full_like(gb, _GB_FLOOR_ABS))
        clearance_sg = sg * (g - gb_fasted)                      # signed: restoring toward Gb_fasted
        clearance_ins = x_ins * g                                # ≥ 0: insulin action is a SINK
        k_cort = _KCORT_MIN + _KCORT_RANGE * torch.sigmoid(self.log_k_cort)
        k_gn = _KGN_MIN + _KGN_RANGE * torch.sigmoid(self.log_k_gn)
        k_act = _KACT_MIN + _KACT_RANGE * torch.sigmoid(self.log_k_act)
        cort_excess = _CORT_NORM_SCALE * relu(coupling[..., _CORTISOL_COUPLING_IDX])
        gn_excess = relu(gn - _GN_CENTER)
        egp_extra = k_cort * cort_excess + k_gn * gn_excess
        egp_extra = _EGP_MAX * torch.tanh(egp_extra / _EGP_MAX)
        exercise_uptake = k_act * act * relu(g - 0.8 * gb)

        # --- insulin, insulin action -----------------------------------------------------
        fast_ins_supp = torch.clamp(g / gb, min=0.0, max=1.0) ** _FAST_INS_EXP
        ins_production = prod_raw[..., _INSULIN_IDX] * fast_ins_supp * self.prod_scale[_INSULIN_IDX]
        ins_clearance = cons_raw[..., _INSULIN_IDX] * self.cons_scale[_INSULIN_IDX] * ins
        p2 = _P2_MIN + _P2_RANGE * torch.sigmoid(self.log_p2)
        xa_rate = p2 * (relu(insulin_dev) - xa)

        # --- hepatic output: a lagged readout of glycogenolysis + learned GNG -----------
        hep_target = (_HEP_UNITS_PER_G_MIN * brk_liver
                      + _TYPICALS[_HEPATIC_IDX] * prod_raw[..., _HEPATIC_IDX])
        hep_rate = cons_raw[..., _HEPATIC_IDX] * self.cons_scale[_HEPATIC_IDX] * (hep_target - hep)

        # --- ketogenesis -------------------------------------------------------------------
        keto_ins_supp = nn.functional.softplus(self.log_keto_ins_supp)
        ketogenesis = (nn.functional.softplus(self.log_k_keto) * relu(ffa - _FFA_CENTER)
                       / (1.0 + ins_excess / keto_ins_supp))

        return {
            "gb": gb, "ib": ib, "ra": ra, "gb_fasted": gb_fasted,
            "glucose_dev": glucose_dev, "insulin_dev": insulin_dev,
            "prod_raw": prod_raw, "cons_raw": cons_raw, "mito": mito,
            "app_g": app_g, "c_plasma": c_plasma,
            "f_liver": f_liver, "f_muscle": f_muscle, "f_plasma": f_plasma,
            "syn_liver": syn_liver, "syn_muscle": syn_muscle,
            "brk_liver": brk_liver, "brk_muscle": brk_muscle,
            "appearance_plasma": appearance_plasma,
            "glycogenolysis_plasma": glycogenolysis_plasma,
            "clearance_sg": clearance_sg, "clearance_ins": clearance_ins,
            "egp_extra": egp_extra, "exercise_uptake": exercise_uptake,
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
            -f["clearance_sg"] - f["clearance_ins"]
            + f["appearance_plasma"] + f["glycogenolysis_plasma"]
            + f["egp_extra"] - f["exercise_uptake"]
        )
        out[..., _INSULIN_IDX] = f["ins_production"] - f["ins_clearance"]
        out[..., _INSULIN_ACTION_IDX] = f["xa_rate"]
        out[..., _HEPATIC_IDX] = f["hep_rate"]
        out[..., _LIVER_GLYCOGEN_IDX] = f["syn_liver"] - f["brk_liver"]
        out[..., _MUSCLE_GLYCOGEN_IDX] = f["syn_muscle"] - f["brk_muscle"]
        return out
