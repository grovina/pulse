"""
Metabolic / Energy module.

Blood chemistry homeostasis: glucose, insulin, glucagon, FFA, BHB, lactate,
and hepatic glucose output (endogenous appearance flux). Uses mass-action
kinetics. Receives nutrient appearance from Gut and cortisol from Stress.
"""

import math

import torch
import torch.nn as nn

from .base import BasalPlusGatedPeakHead, MassActionModule, SpeciesHead, gate_temp
from ..types import GUT_OUTPUT_DIM, MARKER_INDEX, MODULE_MARKER_INDICES, NORM_CENTER, NORM_SCALE

# Iter 90: converts the module's normalized glucose deviation back to raw mg/dL so the
# restoring term is `-(Sg + Si·Xa)·(G_raw - Gb_raw)` with Sg/Si as true per-minute rate
# constants (see the _SG_* block below).
_GLUCOSE_NORM_SCALE = NORM_SCALE[MARKER_INDEX["glucose"]]
_GLUCOSE_CENTER = NORM_CENTER[MARKER_INDEX["glucose"]]
# Iter 90 counter-regulation: raw-unit conversions for the states/couplings that feed
# glucose. cortisol and glucagon are centered at their basal values, so relu(normalized)
# is exactly their above-basal excess in z-units.
_HEP_CENTER = NORM_CENTER[MARKER_INDEX["hepatic_output"]]
_HEP_NORM_SCALE = NORM_SCALE[MARKER_INDEX["hepatic_output"]]
_CORT_NORM_SCALE = NORM_SCALE[MARKER_INDEX["cortisol"]]
_GN_NORM_SCALE = NORM_SCALE[MARKER_INDEX["glucagon"]]

# Coupling inputs: gut outputs (4) + cortisol (1) + glp1 (1) = 6
#
# Iter 90 — INCRETIN PATH. The teacher potentiates glucose-stimulated insulin secretion by
# an incretin factor `1 + GLP1/(GLP1 + K_incretin)` (full_body.py:446,452), and the coupling
# prior registry has always DECLARED `glp1 -> insulin` (+1, knowledge/coupling_priors/
# metabolism.py). But glp1 was never fed into this module, so ∂insulin_rate/∂glp1 ≡ 0: the
# declared prior had ZERO gradient and was silently inert, and the student had no incretin
# effect at all. Routing glp1 in makes the prior live and lets the insulin head learn the
# potentiation. Placed LAST so the gut(0-3) and cortisol(4) coupling indices are unchanged.
_N_COUPLING = GUT_OUTPUT_DIM + 2

# External inputs: activity (1) + sleep_wake (1) = 2
_N_EXTERNAL = 2

# Species order matches MODULE_MARKER_INDICES["metabolic"]:
# 0: glucose, 1: insulin, 2: glucagon, 3: ffa, 4: bhb, 5: lactate,
# 6: hepatic_output, 7: liver_glycogen (iter 56), 8: muscle_glycogen
# (iter 56), 9: mitochondrial_capacity (iter 55), 10: insulin_action (iter 89).
# Indices 7-9 are slow internal states (unobserved, marker_type="internal"
# in types.py) — physical pool size + protein turnover give them long τ
# naturally under the same mass-action primitive. See
# docs/multi-timescale-plan.md. Index 10 (insulin_action) is a fast latent
# (τ ≈ 33 min) whose rate is computed structurally in forward() (a low-pass of
# relu(insulin)), not by its mass-action head — like glucose (index 0), its
# SpeciesHead output is unused.
#
# Iter 56: the iter-55 lumped glycogen_pool (500 g, one τ ≈ 10 d) could
# not express a −60 g / 1-day fast delta. Split by tissue: liver pool
# (~100 g, τ ≈ 1 d — overnight-depleting) vs muscle pool (~400 g,
# τ ≈ 3 wk — rest-preserved, exercise-coupled). Now one cons_scale per
# tissue, each physically honest.
_TYPICALS = [95.0, 10.0, 70.0, 0.5, 0.1, 1.0, 2.0, 100.0, 400.0, 1.0, 0.0]
# Iter 95: NORM_SCALE per species, in the same order, so the module can rebuild the RAW
# concentration for the mass-action consumption term. Derived rather than hardcoded so it
# cannot drift from types.py.
_NORM_SCALES = [NORM_SCALE[i] for i in MODULE_MARKER_INDICES["metabolic"]]
# cons_scale = 1/τ in the rate equation.
#   liver_glycogen      τ ≈ 1 day  ≈ 1440 min  → cons_scale ≈ 7e-4
#   muscle_glycogen     τ ≈ 3 weeks ≈ 30240 min → cons_scale ≈ 3.3e-5
#   mitochondrial_cap.  τ ≈ 4 weeks ≈ 40320 min → cons_scale ≈ 2.5e-5
# These scales encode the slow timescale physically — the mass-action
# equation `rate = prod - cons·state` evolves them slowly without any
# special "slow integrator" mechanism. Liver's τ ≈ 1 d makes a −60 g
# delta reachable within the 1-day EXTENDED_FAST_GLYCOGEN protocol.
_CONS_SCALES = [0.02, 0.1, 0.03, 0.04, 0.03, 0.02, 0.04, 7e-4, 3.3e-5, 2.5e-5, 0.02]
_INSULIN_IDX = 1
_GLUCOSE_IDX = 0
# Iter 89: dynamic insulin action (remote insulin). Local species index 10.
# Its rate is computed in forward() as a first-order low-pass of
# relu(insulin_norm) (its mass-action SpeciesHead output is unused, like
# glucose's), giving the insulin→glucose clearance the teacher's ≈33-min lag.
_INSULIN_ACTION_IDX = 10
# Lag rate p2 = 1/τ, bounded so τ ∈ [4, 100] min (teacher p2 ≈ 0.03, τ ≈ 33 min).
# Floored so the state can't freeze (τ→∞); capped so it can't collapse to the
# old instantaneous behaviour (τ→0). Euler-stable (p2·dt ≤ 0.25 ≪ 2).
_P2_MIN = 0.01
_P2_RANGE = 0.24
# Iter 81: max per-patient fasting-glucose offset, z-score units, around the 95 mg/dL
# center (NORM_SCALE_glucose=30).
# Iter 90: widened 1.5 -> 2.2. The teacher's Gb spread (sigma=0.25 lognormal) spans
# ~58-157 mg/dL at ±2σ, i.e. z ∈ [-1.25, +2.05]. At the old ±1.5 bound (Gb ∈ [50, 140])
# the upper tail was UNREACHABLE, so SetpointSupervisionSignal would push the tanh into
# saturation for diabetic-range patients — correct direction, vanishing gradient. ±2.2
# gives Gb ∈ [29, 161], covering the teacher's range and the benchmark's 60-120 with
# margin. (The physiological state clamp still bounds anything pathological.)
_GLUCOSE_BASELINE_MAX_Z = 2.2
# Iter 88: max per-patient meal-appearance (Ra) log-gain offset, pre-softplus
# units. Ra = softplus(log_ra + ra_emb), ra_emb ∈ ±_RA_BASELINE_MAX_Z. With
# log_ra init = log(0.55), ±1.0 gives Ra ∈ ~[0.18, 0.91] around the ~0.44 init —
# a ~5× per-patient amplitude range, enough to cover the cohort's postprandial
# spread without letting a single patient's excursion run away.
_RA_BASELINE_MAX_Z = 1.0
# Iter 90 — RATE-CONSTANT FRAME FIX (Sg now means what it says).
#
# Glucose is the pure minimal-model setpoint (teacher full_body.py:448
# dG = -(Sg+X)·(G-Gb) + Ra). Through iter 89 the restoring term was computed on the
# NORMALIZED deviation (`-Sg·(G_norm - b_emb)`) while the resulting rate was applied to the
# RAW state, so the effective raw rate constant was Sg / NORM_SCALE_glucose = Sg/30 — thirty
# times smaller than the code claimed ("tc 1/Sg ≈ 3-33 min"). Measured on the iter-89
# checkpoint: trained Sg = 0.0583 gave k_eff = 0.00194/min, i.e. τ = 515 min for fasting
# glucose, versus the teacher's Sg = 0.018/min (τ = 56 min) and the literature
# glucose-effectiveness range 0.02-0.03/min (τ = 33-50 min). The old band [0.03, 0.30] spans
# k_eff [0.001, 0.010]/min, so matching the teacher would have needed Sg = 0.54 — ABOVE the
# old ceiling. The student was structurally incapable of physiological glucose effectiveness.
#
# The error was invisible to the gate because the scored eval window is fasting and flat: a
# too-weak restoring force still holds a flat line flat. Note also that iters 83-85 ran
# Sg ≈ 0.5-0.8, which in the CORRECT frame is k_eff 0.017-0.027/min — exactly the
# teacher/literature range. Those runs were labelled "brute-force clearance" and fenced off
# here, but iter-85's decisive test showed the aborts were the correlation sqrt(0) NaN (fixed
# in 9ffa316) plus the competing mass-action controller (removed in iter 86). This band was
# guarding against the physically correct value.
#
# Fix: Sg is now the RAW per-minute rate constant — the restoring term multiplies the
# normalized deviation by NORM_SCALE_glucose so `rate = -Sg·(G_raw - Gb_raw)` exactly.
# Band = literature glucose effectiveness with margin: [0.005, 0.05]/min (τ 20-200 min),
# init 0.018 (the teacher's value). Euler-stable by a wide margin: the raw coefficient
# (Sg + Si·Xa) peaks near 0.3/min, far below the dt=1 stability limit of 2.
_SG_MIN = 0.005
_SG_RANGE = 0.045
_SG_INIT = 0.018  # teacher full_body.py PatientParams.Sg

# Iter 90: Si is likewise a RAW rate constant. Insulin action enters glucose clearance as
# X = Si·Xa where Xa = relu(insulin_norm) = max(I - Ib, 0)/NORM_SCALE_insulin. The teacher
# uses X = Si_t·max(I-Ib,0) with Si_t = 0.0004 (full_body.py PatientParams.Si), so our Si
# absorbs the /10 insulin normalization: Si = 10·Si_t = 0.004 at the teacher's value.
# Band [0.0005, 0.02]/min per (normalized insulin unit) brackets the Bergman literature
# range (Si_t ≈ 0.0002-0.001) with margin. At a meal peak (Xa≈5) this contributes at most
# 0.1/min, so total clearance (Sg + Si·Xa) stays ≪ the dt=1 Euler limit of 2.
_SI_MIN = 0.0005
_SI_RANGE = 0.0195
_SI_INIT = 0.004  # = 10 × teacher Si (insulin normalization)

# Iter 90 — COUNTER-REGULATION RE-COUPLED. Through iter 89 the student's glucose rate was
# ONLY the minimal model: -(Sg + Si·Xa)(G - Gb) + Ra·appearance. The teacher's dG additionally
# carries four counter-regulatory terms (full_body.py:494-500):
#     + hep_to_glucose · Hep                      hepatic glucose output
#     + cort_gluco     · max(Cort - Cort_b, 0)    cortisol-driven gluconeogenesis
#     + 0.02           · max(Gn  - Gnb, 0)        glucagon-driven hepatic release
#     - act · 0.02     · max(G - 0.8·Gb, 0)       exercise glucose uptake
# The student folded all of it into the STATIC per-patient baseline Gb_emb. The consequence
# was that hepatic_output, glucagon and cortisol evolved as markers but had NO effect on
# glucose at all: fasting counter-regulation and exercise-induced glucose drop were not
# mechanistic, and those three markers had no gradient path from observed glucose (which also
# starved them as identifiability signals). This is the largest structural divergence from the
# teacher on the observed side (see the iter-90 physics review).
#
# Gains are RAW per-minute (matching the iter-90 frame convention), initialized at the
# teacher's own values and bounded to keep every term a bounded perturbation of the restoring
# force. Note the equilibrium now sits slightly ABOVE Gb (G* = Gb + hep_source/Sg, ≈ +4 mg/dL)
# — exactly as it does in the teacher, so Gb remains the SETPOINT parameter that
# SetpointSupervisionSignal supervises, not the fasting equilibrium.
_KHEP_MIN, _KHEP_RANGE, _KHEP_INIT = 0.005, 0.095, 0.038      # teacher hep_to_glucose
_KCORT_MIN, _KCORT_RANGE, _KCORT_INIT = 0.0, 0.006, 0.0012    # teacher cort_gluco
_KGN_MIN, _KGN_RANGE, _KGN_INIT = 0.0, 0.10, 0.02             # teacher glucagon->glucose
_KACT_MIN, _KACT_RANGE, _KACT_INIT = 0.0, 0.06, 0.02          # teacher exercise uptake

_HEPATIC_IDX = 6
# cortisol is metabolic coupling[GUT_OUTPUT_DIM]; activity is external[0].
_CORTISOL_COUPLING_IDX = GUT_OUTPUT_DIM
_ACTIVITY_EXTERNAL_IDX = 0

# Iter 95 (A1) — THE DEFENDED GLUCOSE LEVEL IS NOT A CONSTANT.
#
# Through iter 94 the student's fasting equilibrium was `b_emb` alone: a per-patient
# value read from the embedding and CONSTANT IN TIME. No fast of any length could lower
# it, so the student could not express the fasted state at all. Measured on the iter-94
# artifact over a 24 h fast (scripts/iter94_student_fast_probe.py): glucose ROSE
# 95.0 -> 100.2 while the teacher fell to 78.2, and with glucose held up the insulin gate
# stayed open (insulin rose 10 -> 12.5 against the teacher's fall to 3.8), which in turn
# suppressed lipolysis (FFA reached 0.29 of the teacher's delta). One missing term, three
# downstream failures.
#
# The teacher gained this in iter 93 (full_body.py:733-748, Cahill 2006): the minimal
# model is a 3-hour tool, and over that span Gb genuinely is the defended level, but over
# a fast it is not — as hepatic glycogen empties, gluconeogenesis cannot fully replace
# glycogenolysis and the defended level ITSELF falls.
#
#     Gb_fasted = max( Gb·(1 - fast_gb_drop·glyco_depleted),  Gb·fast_gb_floor_frac )
#     glyco_depleted = relu(1 - LGly/LGly_b)      LINEAR pool depletion
#
# Keyed to linear depletion, NOT to the Michaelis `glyco_avail` the hepatic-output split
# reads: that ratio describes how much glycogen the liver can still RELEASE and stays near
# 1 while the pool halves, whereas what the defended level tracks is how much of the pool
# is GONE. (The teacher's own comment makes this distinction; reusing glyco_avail here
# would be convenient and wrong.)
#
# Critically this is EXACTLY THE IDENTITY at the fed calibration state (LGly = LGly_b
# ⇒ glyco_depleted = 0 ⇒ Gb_fasted = Gb), so the entire fed/postprandial regime — the
# regime the gate, the dose-response signals and SetpointSupervisionSignal all score — is
# unchanged by construction, and `b_emb` remains the setpoint parameter that
# SetpointSupervisionSignal supervises. iter-95 A1 depends on iter-94's B4: only now that
# the glycogen pool can actually fall below `typical` does this term ever fire.
#
# The drop fraction is LEARNED (PRD: existence is architecture, strength is learned),
# bounded and initialized at the teacher's default exactly like every other rate constant
# in this file. The band reaches 0, so the model can switch the mechanism off if the data
# disagrees rather than being forced to use it.
#
# NOT per-patient. The teacher's population DOES vary fast_gb_drop over [0.25, 0.90]
# (full_body.py:494), so a per-patient head is the physiologically complete version and
# is deliberately deferred: it would add a third unsupervised embedding->physiology map,
# and iter-90's finding was that exactly those maps go wrong when nothing supervises them.
# A1's job is to make the mechanism EXIST; per-patient strength is a later question.
_GB_DROP_MIN, _GB_DROP_RANGE, _GB_DROP_INIT = 0.0, 0.95, 0.55  # teacher fast_gb_drop
# Floor on the defended level as a fraction of the fed setpoint — a healthy fast does not
# drive glucose arbitrarily low. Held at the teacher's value rather than learned: it binds
# only deep into a fast, so a learned version would carry almost no gradient signal.
_GB_FLOOR_FRAC = 0.62  # teacher fast_gb_floor_frac

# Iter 95 (A5) — FASTING HYPOINSULINEMIA. The second half of the iter-93 teacher fix, and
# it was never ported either.
#
# The teacher's iter-93 comment names the exact chain measured on the iter-94 student:
# "insulin never falls; and lipolysis is insulin-gated, so FFA never rises." iter 93 fixed
# it with TWO terms — `fast_gb_drop` (ported above as A1) and `fast_ins_exp` here. With A1
# alone, a 24 h fast now drops student glucose 95.0 -> 76.3 (teacher 78.2) but insulin
# still sits at 10.4 against the teacher's 3.8, because nothing lowers its setpoint.
#
#     effective_Ib = Ib · min(G/Gb, 1)^fast_ins_exp
#
# The exponent is not decoration: beta-cell secretion is sigmoid in glucose near threshold,
# so basal insulin roughly HALVES while glucose drops only ~15% (Polonsky 1988). A linear
# ratio cannot express that. Identity whenever G >= Gb, so the fed state is untouched, and
# the ratio is referenced to the FED setpoint `gb_raw` — NOT to A1's falling gb_fasted_raw,
# which would cancel exactly the signal this term exists to sense.
#
# APPLIED TO TOTAL INSULIN PRODUCTION, NOT ONLY TO A BASAL TERM — a deliberate divergence
# from the teacher's Ib-only application, on evidence. The teacher's secretion term
# `gamma·max(G-h,0)` with h=95 is exactly zero at fasting glucose, so suppressing Ib alone
# suppresses everything. `GlucoseGatedInsulinHead`'s equivalent is a SOFT sigmoid gate,
# and measured on the iter-94 artifact it leaks: peak·gate still supplies 18-47% of
# production across a 48 h fast (gate 0.008-0.063 at glucose 76-95). Suppressing only the
# basal channel would therefore leave up to half the fasting production untouched and the
# port would be incomplete. Scaling total production scales insulin's mass-action
# equilibrium (norm* = prod·typical/cons) by the same factor, which is exactly what
# `effective_Ib` means. At the meal peak the factor is clamped to 1, so meal kinetics —
# the iter-92 guard — are unaffected by construction.
#
# Exponent held at the teacher's calibrated value rather than learned: this is a SHAPE
# fitted to Polonsky, not a strength, and a learned exponent could drift to 0 and switch
# the mechanism off in exactly the way the bhb rate constant did.
_FAST_INS_EXP = 5.0  # teacher fast_ins_exp

# OFF-MANIFOLD SAFETY RAIL on total endogenous glucose production (mg/dL/min) — NOT a
# physiological law. Read it the way PHYSIOLOGICAL_LIMIT_K is read in types.py: a catastrophe
# bound that must be INACTIVE in distribution.
#
# The three source terms above are each linear in an unbounded state. That is safe for the
# teacher, whose glucagon stays near basal, but not for a student: on an untrained model
# glucagon reaches its state clamp at 470 pg/mL and at k_gn = 0.02/pg/mL that alone injects
# 8 mg/dL/min into glucose (observed: 24h-fast glucose ran to 187). That is the iter-83-85
# runaway class, so the sum gets a finite ceiling.
#
# Sizing it honestly. A first attempt used 3.0, reasoned from real basal EGP (~2 mg/kg/min ≈
# 1.25 mg/dL/min at a 70 kg distribution volume). That was a FRAME ERROR of the same family as
# the Sg bug: the teacher's `hep_to_glucose·Hep` term is not total EGP, it is a small
# correction on top of the balance already implied by Sg·(G−Gb) — its basal value is
# 0.046 mg/dL/min, ~27× smaller than physiological EGP. A 3.0 ceiling therefore bit hard inside
# the teacher's own operating range (its realistic combined source peaks near 1.8 mg/dL/min
# during a fasting glucagon rise, which 3.0 would shave to 1.56 — a 13% distortion of real
# physiology to fix an untrained transient).
#
# 8.0 keeps the rail inactive where the model actually lives (≤0.5% deviation across the
# teacher's 0–1.8 range) while still bounding the pathological limit. Applied as
# `EGP_MAX · tanh(src / EGP_MAX)`; the integrator's physiological state clamp remains the
# outer backstop, and trained glucagon is supervised by cold-distill so this should never bind.
_EGP_MAX = 8.0


def _logit(p: float) -> float:
    """Inverse sigmoid — init a sigmoid-bounded parameter at a target value."""
    return math.log(p / (1.0 - p))
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

# Iter 94: glycogen storage-pool flux constants (see the override at the end of
# MetabolicModule.forward for why the mass-action shape was replaced).
_LIVER_GLY_CENTER = NORM_CENTER[MARKER_INDEX["liver_glycogen"]]      # 100 g
_LIVER_GLY_NORM_SCALE = NORM_SCALE[MARKER_INDEX["liver_glycogen"]]   # 60 g
_MUSCLE_GLY_CENTER = NORM_CENTER[MARKER_INDEX["muscle_glycogen"]]    # 400 g
_MUSCLE_GLY_NORM_SCALE = NORM_SCALE[MARKER_INDEX["muscle_glycogen"]]  # 100 g
# Glycogen supercompensation after depletion+refeed is a real ~20-40 % overshoot
# (Bergstrom & Hultman 1966), so the store's ceiling sits above its typical value.
_GLY_CAPACITY_FRAC = 1.3
# Flux scale in RAW g/min. These are order-of-magnitude anchors: both bracket terms
# are O(1) at a normal store, and the head's learned softplus outputs (range ~0.1-3)
# modulate around them, so each scale needs to REACH its tissue's physiological rate,
# not to equal it.
#   Liver: ~100 g pool turning over across an overnight/24 h fast — the teacher falls
#   58 g/24 h = 0.04 g/min, and postprandial synthesis runs a few tenths of a g/min.
#   0.15 covers both directions with headroom.
_LIVER_GLY_FLUX = 0.15
#   Muscle: a hard 2 h bout costs 150-200 g (Bergstrom 1967; the cohort target is
#   -150 g), i.e. ~1.3-1.7 g/min sustained, while resting turnover is far slower and
#   the head's gate — not this constant — is what keeps rest quiet.
_MUSCLE_GLY_FLUX = 1.0

# GlycogenFluxHead input indices (iter 57). The head sees the module
# input x = cat([state(10), coupling(5), external(2), embedding, time]).
#   coupling[0] = gut glucose-appearance  → x index 10  (n_species + 0)
#   metabolic external = [activity, sleep_wake] (model.py met_external)
#     → activity = x index 15  (n_species + _N_COUPLING + 0)
# n_species for this module (see MetabolicModule below); iter 89: 10 → 11
# (insulin_action appended). The GlycogenFluxHead x-indices below are computed
# from _N_SPECIES so they track the coupling/external block automatically.
_N_SPECIES = 11
_GLUCOSE_APPEARANCE_X_IDX = _N_SPECIES + 0
_ACTIVITY_X_IDX = _N_SPECIES + _N_COUPLING + 0
# Index of the gut glucose-appearance channel within the `coupling` tensor
# alone (model.py met_coupling = [gut_outputs(4), cortisol(1)], gut[0]=glucose).
_GUT_GLUCOSE_COUPLING_IDX = 0


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

      synthesis  = softplus(net) · relu(glucose_appearance)
                   — glycogen fills only while gut nutrients are being
                     absorbed, and STRICTLY not otherwise (iter 96; the
                     sigmoid gate this replaced learned its way to 63 %
                     open at zero appearance). Routes the fast-vs-fed
                     gradient through the SAME strong gut-coupling
                     pathway the `glucose` cohort spec uses (the fix for
                     the diagnosed gradient starvation).
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
        # ITER 96 -- THE ANABOLIC GATE LEAKED, AND A LEARNABLE THRESHOLD CANNOT
        # BE STOPPED FROM LEAKING.
        #
        # Through iter 95 synthesis was `softplus(net) · σ((a − a_thresh)/temp)`
        # with `a_thresh` free (init +0.1). It learned a_thresh = −0.114, so at
        # ZERO gut glucose appearance the gate sat 63 % OPEN. Measured on the
        # iter-95 artifact at the teacher's own 24 h-fast state (liver_glycogen
        # 41.7 g), the student's net glycogen rate was **+0.159 g/min — it
        # REFILLED the liver during a fast**. Nothing downstream could then work:
        # `Gb_fasted` reads how much of the pool is GONE, so a pool that never
        # empties defends a glucose level that never falls. That is the student
        # half of the pre-dawn glucose blocker (student −0.14 mg/dL/h over
        # 03:00-06:00 against the teacher's −0.74 and the real −1.75).
        #
        # The physics is not a threshold. Glycogen synthase has no substrate to
        # act on when no glucose is arriving: the rate is PROPORTIONAL to the
        # appearance flux, exactly as the teacher writes it
        # (`syn_L = k · Ra_carb · ins_drive · fill`). So synthesis is now
        # `softplus(net) · relu(a)`, which is identically zero at zero
        # appearance — a structural guarantee, not a learned one — and linear in
        # the drive, which is a better-conditioned gradient than a saturating
        # sigmoid. The insulin dependence stays learnable: insulin is in `x`, so
        # `softplus(net)` can express `ins_drive` itself.
        #
        # `a_thresh` / `log_a_temp` are DELETED rather than clamped. A softplus
        # reparam would keep the gate closed at zero but leaves the same
        # saturating shape and one more parameter to mis-learn; the substrate
        # form removes the failure mode instead of bounding it.
        #
        # Catabolic gate on a normalized (~0-centred) state/external.
        self.c_thresh = nn.Parameter(torch.tensor(0.0))
        self.log_c_temp = nn.Parameter(torch.tensor(-0.7))  # exp ≈ 0.5

    def forward(self, x: torch.Tensor, state_self: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.network(x)
        anab_stim = x[..., self.anabolic_idx]
        catab_stim = x[..., self.catabolic_idx]
        catab_gate = torch.sigmoid(
            self.catabolic_dir * (catab_stim - self.c_thresh) / gate_temp(self.log_c_temp)
        )
        # Substrate-proportional, not gated: zero appearance -> zero synthesis.
        synth = nn.functional.softplus(raw[..., 0]) * torch.relu(anab_stim)
        break_basal = nn.functional.softplus(raw[..., 1])
        break_active = nn.functional.softplus(raw[..., 2]) * catab_gate
        prod = synth
        cons = break_basal + break_active
        return prod, cons


class MetabolicModule(MassActionModule):
    def __init__(self, embedding_dim: int, hidden_dim: int = 48):
        super().__init__(
            n_species=11,
            n_coupling=_N_COUPLING,
            n_external=_N_EXTERNAL,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            typicals=_TYPICALS,
            norm_scales=_NORM_SCALES,
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
                # Iter 95: was SetpointHead. In the corrected concentration frame a plain
                # SpeciesHead reaches any positive equilibrium with live gradients, so the
                # iter-51 workaround is unnecessary (see modules/base.py:MassActionModule).
                _BHB_IDX: SpeciesHead,
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
                _MITO_IDX: SpeciesHead,  # iter 95: was SetpointHead
            },
        )
        prod_scales = [c * t for c, t in zip(_CONS_SCALES, _TYPICALS)]
        self.prod_scale.copy_(torch.tensor(prod_scales, dtype=torch.float32))
        self.cons_scale.copy_(torch.tensor(_CONS_SCALES, dtype=torch.float32))
        # Structural glucose setpoint: clearance gain Sg pulling glucose toward
        # the patient baseline Gb (= 95 + NORM_SCALE*b_emb). Implements
        # dG_extra = -Sg*(G - Gb).
        #
        # Iter 83: floor Sg at _SG_MIN. The iter-81/82 Sg trained to ~0.11, far
        # too weak: with the glucose mass-action SpeciesHead anchoring ~95, the
        # setpoint could not pull glucose below ~85 mg/dL for ANY embedding
        # (verified by gradient-optimising the embedding to minimise fasting
        # glucose at every leash radius) — so the benchmark's low-baseline users
        # (60-84) were structurally unrepresentable and glucose_mape stuck at
        # 0.37 (the only remaining gate failure after iter-82 fixed HR). Training
        # will NOT raise Sg on its own (iter-82 kept 0.11 even with low-glucose
        # patients in the teacher). An sg-sweep showed the floor reachable scales
        # with Sg: 0.11->85, 0.5->65, 0.6->63, 0.8->~61, 1.0->59 mg/dL. _SG_MIN=0.5
        # lets the setpoint reach ~65 (covers the benchmark's low users at an
        # aggregate glucose_mape ~0.015). NOTE iter-83/84 aborted on a flat-window
        # NaN in the correlation physiology rules (not Sg stiffness), fixed in
        # iter 85 — see _SG_MIN comment up top. Physiologically honest: glucose is
        # tightly defended by counter-regulation, so a strong restoring force
        # toward the patient setpoint (and fast post-meal return to baseline) is
        # correct, not a benchmark hack. The meal excursion is carried by the Ra
        # appearance term (log_ra), which the dose-response signal raises to keep
        # postprandial amplitude despite the stronger clearance.
        # Iter 90: Sg = _SG_MIN + _SG_RANGE·sigmoid(log_sg) is now the RAW per-minute
        # glucose-effectiveness constant; init at the teacher's 0.018 (τ ≈ 56 min).
        self.log_sg = nn.Parameter(torch.tensor(_logit((_SG_INIT - _SG_MIN) / _SG_RANGE)))
        # Structural glucose rate-of-appearance: learned gain Ra on the gut
        # glucose-appearance flux. Implements dG_extra = +Ra·appearance,
        # mirroring the minimal-model Ra(t) source term (Dalla Man 2007) the
        # teacher carries explicitly. coupling[..., 0] is the gut
        # glucose-appearance channel (model.py met_coupling = [gut(4),
        # cortisol]); it is ≥0 and ≈0 between meals, so this term is silent
        # fasted (gate-safe) and fires only while nutrients are absorbed —
        # exactly the postprandial-amplitude axis dose-response supervises.
        #
        # Iter 80: glucose was the one metabolic species still on a plain
        # SpeciesHead, so the meal→glucose amplitude had to be discovered by
        # an MLP that simultaneously holds the fasting equilibrium — the same
        # softplus-saturation trap GlucoseGatedInsulinHead / BasalPlusGatedPeakHead
        # were built to escape. Despite peak-mode dose-response at weight 0.40
        # (target 0.7 mg/dL/g), the iter-79 model realised only ~0.16 mg/dL/g
        # (a 60 g meal raised glucose ~14 mg/dL, not ~35). The amplitude was a
        # parameter no strong gradient could reach; this term is the structural
        # fix — a single scalar the existing dose-response gradient moves
        # directly. Init Ra ≈ 0.55 (calibrated locally so 60 g lands ~0.5
        # mg/dL/g at the iter-79 operating point) so the cold-start run begins
        # near-physiological on amplitude and dose-response refines from there.
        self.log_ra = nn.Parameter(torch.tensor(math.log(0.55)))
        # Iter 81: patient-specific glucose baseline (fasting setpoint).
        # The clearance term above pulls glucose toward a setpoint that was
        # HARDCODED at Gb=95 mg/dL (normalised 0) — so the model could not
        # represent the fasting-glucose spread of real patients. Measured on the
        # iter-80 model: achievable fasting glucose over the calibration-reachable
        # embedding region floored at ~98 mg/dL (downward authority ≈ 0), while
        # the benchmark users span 60-120; ~2/3 of them were literally
        # unrepresentable, which dominated the glucose_mape gate failure (the
        # model also loses to persistence because its baseline is wrong, not just
        # its dynamics). This small head emits a per-patient baseline offset b_emb
        # (z-score units) from the embedding, so the clearance pulls toward
        # Gb = 95 + NORM_SCALE_glucose·b_emb — giving the embedding DIRECT linear
        # authority over the fasting setpoint (the SetpointHead philosophy applied
        # surgically to the existing structural term, with no head swap — the
        # iter-72 scar showed swapping metabolic heads wrecks the coupling graph).
        # tanh-bounded to ±GLUCOSE_BASELINE_MAX_Z so Gb stays physiological
        # (≈50-140 mg/dL); final layer zero-init so b_emb=0 at start (Gb=95,
        # byte-identical to the pre-iter-81 fasting equilibrium).
        _bh = max(8, hidden_dim // 4)
        self.glucose_baseline_net = nn.Sequential(
            nn.Linear(embedding_dim, _bh), nn.Tanh(), nn.Linear(_bh, 1),
        )
        with torch.no_grad():
            self.glucose_baseline_net[-1].weight.zero_()
            self.glucose_baseline_net[-1].bias.zero_()
        # Iter 88: per-patient meal-appearance gain Ra. Through iter 87 log_ra was
        # a single GLOBAL scalar while Gb_emb was the ONLY per-patient glucose
        # lever. When the iter-87 meal-timing fix put real excursions back into the
        # 240-min calibration window (which mixes meal peaks and fasting troughs
        # into one embedding fit), each patient's amplitude mismatch had nowhere to
        # go but Gb — biasing the fasting setpoint, so fasting glucose_mape
        # regressed 0.20->0.27 (and HR co-regressed via the cardiovascular
        # coupling). This head gives Ra its own per-patient offset, mirroring
        # glucose_baseline_net, so calibration fits the excursion amplitude
        # (ra_emb) and the baseline (Gb_emb) INDEPENDENTLY and Gb stops absorbing
        # amplitude error. Final layer zero-init ⇒ ra_emb=0 at start, so Ra is
        # byte-identical to the pre-iter-88 global softplus(log_ra) cold start; the
        # dose-response signal still supervises the amplitude axis and now moves it
        # per patient rather than globally.
        self.ra_baseline_net = nn.Sequential(
            nn.Linear(embedding_dim, _bh), nn.Tanh(), nn.Linear(_bh, 1),
        )
        with torch.no_grad():
            self.ra_baseline_net[-1].weight.zero_()
            self.ra_baseline_net[-1].bias.zero_()
        # Iter 86: insulin-action gain Si for the minimal-model clearance
        # (Sg + X)·(G - Gb), X = Si·max(insulin_above_baseline, 0). This is the
        # insulin->glucose suppression the dropped mass-action used to carry via
        # cons(insulin); now it lives in the physically-correct place (the
        # insulin-dependent glucose effectiveness of the Bergman minimal model).
        # Iter 90: bounded to the Bergman literature band and expressed as a RAW
        # per-minute constant (was an unbounded softplus in the normalized frame).
        self.log_si = nn.Parameter(torch.tensor(_logit((_SI_INIT - _SI_MIN) / _SI_RANGE)))
        # Iter 89: insulin-action lag rate p2 = _P2_MIN + _P2_RANGE·sigmoid(log_p2).
        # Init so p2 ≈ 0.03 (τ ≈ 33 min, teacher value): sigmoid(log_p2) = (0.03 -
        # _P2_MIN)/_P2_RANGE ≈ 0.0833. insulin_action low-passes relu(insulin_norm)
        # at this rate, and the glucose clearance reads the lagged state so the
        # insulin→glucose effect carries the teacher's delay instead of being
        # instantaneous. At the fasting equilibrium the lagged state → 0, so the
        # glucose dynamics are unchanged at rest (the fasting eval is protected);
        # the lag only reshapes the postprandial clearance.
        _p2_p0 = (0.03 - _P2_MIN) / _P2_RANGE
        self.log_p2 = nn.Parameter(torch.tensor(math.log(_p2_p0 / (1.0 - _p2_p0))))
        # Iter 90: counter-regulatory gains, raw per-minute, init at the teacher's values.
        self.log_k_hep = nn.Parameter(torch.tensor(_logit((_KHEP_INIT - _KHEP_MIN) / _KHEP_RANGE)))
        self.log_k_cort = nn.Parameter(torch.tensor(_logit((_KCORT_INIT - _KCORT_MIN) / _KCORT_RANGE)))
        self.log_k_gn = nn.Parameter(torch.tensor(_logit((_KGN_INIT - _KGN_MIN) / _KGN_RANGE)))
        self.log_k_act = nn.Parameter(torch.tensor(_logit((_KACT_INIT - _KACT_MIN) / _KACT_RANGE)))
        # Iter 95 (A1): fraction of the defended glucose level lost at full liver-glycogen
        # depletion. Init at the teacher's fast_gb_drop; zero at the fed state regardless.
        self.log_gb_drop = nn.Parameter(
            torch.tensor(_logit((_GB_DROP_INIT - _GB_DROP_MIN) / _GB_DROP_RANGE)))

    def forward(
        self,
        state: torch.Tensor,
        coupling: torch.Tensor,
        external: torch.Tensor,
        embedding: torch.Tensor,
        time_features: torch.Tensor,
    ) -> torch.Tensor:
        prod_raw, cons_raw = self.species_fluxes(
            state, coupling, external, embedding, time_features)
        # Iter 95: consumption is proportional to CONCENTRATION, not to the normalized
        # deviation. This module assembles its own rates instead of calling
        # super().forward() (it overrides several species below), so the frame fix has to
        # be applied here too — see modules/base.py:MassActionModule for why `typical` was
        # an absorbing floor for every species that reaches this line.
        rates = prod_raw * self.prod_scale - cons_raw * self.cons_scale * self.raw_state(state)
        # Sg: RAW per-minute glucose effectiveness, bounded to the literature band
        # [0.005, 0.05]/min (iter 90 frame fix). Equilibrium is Gb_emb for any Sg>0;
        # Sg sets the fasting return SPEED, which is now physiological (τ ≈ 56 min at init).
        sg = _SG_MIN + _SG_RANGE * torch.sigmoid(self.log_sg)
        # Iter 88: per-patient Ra = softplus(log_ra + ra_emb), ra_emb tanh-bounded
        # to ±_RA_BASELINE_MAX_Z. Decouples excursion amplitude from Gb_emb.
        ra_emb = _RA_BASELINE_MAX_Z * torch.tanh(
            self.ra_baseline_net(embedding).squeeze(-1)
        )
        ra = nn.functional.softplus(self.log_ra + ra_emb)
        # X: insulin action — extra glucose effectiveness when insulin is above
        # baseline (state is z-scored, so 0 = baseline). ≥0; ≈0 fasted.
        # Iter 89: read the LAGGED remote-insulin state (insulin_action, ≥0) rather
        # than instantaneous insulin, so the insulin→glucose clearance carries the
        # teacher's ≈33-min delay. insulin_action low-passes relu(insulin_norm)
        # (see its rate below), and its NORM_SCALE is 1.0 so state[..., idx] is the
        # raw non-negative lagged drive. relu() guards the transient in case the
        # straight-through clamp lets it dip fractionally below 0.
        # Iter 90: Si is a bounded RAW per-minute constant, so x_ins is /min like Sg.
        si = _SI_MIN + _SI_RANGE * torch.sigmoid(self.log_si)
        x_ins = si * nn.functional.relu(state[..., _INSULIN_ACTION_IDX])
        # Per-patient fasting setpoint Gb = 95 + NORM_SCALE·b_emb (z-score offset).
        b_emb = _GLUCOSE_BASELINE_MAX_Z * torch.tanh(
            self.glucose_baseline_net(embedding).squeeze(-1)
        )
        # gut glucose-appearance flux (≥0, ≈0 fasted).
        gut_glucose_appearance = coupling[..., _GUT_GLUCOSE_COUPLING_IDX]
        # Iter 86: glucose is the PURE minimal-model setpoint (teacher full_body
        # form), REPLACING the mass-action SpeciesHead rate for glucose — no
        # competing anchor toward ~95. dG = -(Sg + X)·(G - Gb_emb) + Ra·appearance.
        # The fasting equilibrium IS the per-patient Gb_emb (reachable 50-140 mg/dL
        # for ANY gentle Sg), true by construction; the meal excursion is Ra; the
        # insulin-mediated clearance is X. (The glucose SpeciesHead's prod/cons are
        # simply unused now — its dynamics were the wrong shape: a homeostatically
        # defended setpoint, not a chemical species seeking prod/cons balance.)
        rates_out = rates.clone()
        # Iter 90 frame fix: (state[glucose] - b_emb) is a NORMALIZED deviation, so scaling by
        # NORM_SCALE_glucose turns it into the raw (G - Gb) in mg/dL. The rate is applied to
        # raw state, so Sg and Si are now true per-minute rate constants and this line IS the
        # teacher's dG = -(Sg + X)·(G - Gb) + Ra·appearance in raw units.
        # Iter 90: counter-regulatory sources/sinks, in RAW mg/dL/min, mirroring the teacher.
        # Each reads a state or coupling the module already receives; converting normalized
        # deviations back to raw units keeps every gain a true per-minute rate constant.
        k_hep = _KHEP_MIN + _KHEP_RANGE * torch.sigmoid(self.log_k_hep)
        k_cort = _KCORT_MIN + _KCORT_RANGE * torch.sigmoid(self.log_k_cort)
        k_gn = _KGN_MIN + _KGN_RANGE * torch.sigmoid(self.log_k_gn)
        k_act = _KACT_MIN + _KACT_RANGE * torch.sigmoid(self.log_k_act)

        relu = nn.functional.relu
        # Hepatic glucose output (a >=0 flux state). raw = center + scale * normalized.
        hep_raw = relu(
            _HEP_CENTER + _HEP_NORM_SCALE * state[..., _HEPATIC_IDX]
        )
        # Above-basal cortisol and glucagon, back in raw units (their centers ARE their basals).
        cort_excess = _CORT_NORM_SCALE * relu(coupling[..., _CORTISOL_COUPLING_IDX])
        gn_excess = _GN_NORM_SCALE * relu(state[..., _GLUCAGON_IDX])
        # Exercise-driven uptake, gated on glucose above 80% of the patient's own setpoint.
        act = external[..., _ACTIVITY_EXTERNAL_IDX]
        g_raw = _GLUCOSE_CENTER + _GLUCOSE_NORM_SCALE * state[..., _GLUCOSE_IDX]
        gb_raw = _GLUCOSE_CENTER + _GLUCOSE_NORM_SCALE * b_emb
        # Iter 95 (A1): the defended level falls as the liver pool empties (see the
        # _GB_DROP_* block). Zero at the fed state, so this is the identity there.
        lgly_raw = _LIVER_GLY_CENTER + _LIVER_GLY_NORM_SCALE * state[..., _LIVER_GLYCOGEN_IDX]
        glyco_depleted = relu(1.0 - lgly_raw / _LIVER_GLY_CENTER)
        gb_drop = _GB_DROP_MIN + _GB_DROP_RANGE * torch.sigmoid(self.log_gb_drop)
        gb_fasted_raw = torch.maximum(
            gb_raw * (1.0 - gb_drop * glyco_depleted),
            gb_raw * _GB_FLOOR_FRAC,
        )
        # Iter 95 (A5): insulin's setpoint falls with the glucose ratio (see _FAST_INS_EXP).
        # Referenced to the FED gb_raw, not to gb_fasted_raw — referencing it to the falling
        # level would cancel exactly the signal it exists to sense (the teacher makes the
        # same note at full_body.py:754).
        fast_ins_supp = torch.clamp(g_raw / gb_raw, min=0.0, max=1.0) ** _FAST_INS_EXP
        # Consumption on the RAW concentration, as everywhere else in the corrected frame —
        # writing `state` here would silently restore the absorbing floor for insulin alone.
        rates_out[..., _INSULIN_IDX] = (
            prod_raw[..., _INSULIN_IDX] * fast_ins_supp * self.prod_scale[_INSULIN_IDX]
            - cons_raw[..., _INSULIN_IDX] * self.cons_scale[_INSULIN_IDX]
            * self.raw_state(state)[..., _INSULIN_IDX]
        )
        # NB: exercise uptake stays referenced to the FED setpoint, matching the teacher
        # (full_body.py `max(G - params.Gb * 0.8, 0)` uses Gb, not Gb_fasted) — it is a
        # threshold on absolute glucose availability, not on the defended level.
        exercise_uptake = k_act * act * relu(g_raw - 0.8 * gb_raw)

        # Total endogenous glucose production, soft-capped at a physiological Vmax (see
        # _EGP_MAX). In distribution this is the identity; it only bounds a runaway source
        # (e.g. an untrained glucagon pinned at its clamp).
        egp = k_hep * hep_raw + k_cort * cort_excess + k_gn * gn_excess
        egp = _EGP_MAX * torch.tanh(egp / _EGP_MAX)

        rates_out[..., _GLUCOSE_IDX] = (
            # Iter 95 (A1): written directly in raw mg/dL. Identical to the previous
            # `(state - b_emb)·NORM_SCALE` form when gb_fasted_raw == gb_raw (both equal
            # G_raw - Gb_raw by construction), so Sg and Si keep their iter-90 meaning as
            # true per-minute rate constants.
            -(sg + x_ins) * (g_raw - gb_fasted_raw)
            + ra * gut_glucose_appearance
            + egp
            - exercise_uptake
        )
        # Iter 89: insulin_action is a first-order low-pass of the insulin drive:
        # dXa/dt = p2·(relu(insulin_norm) − Xa). Its equilibrium is
        # relu(insulin_norm), so at rest Xa → 0 and x_ins above matches the old
        # instantaneous term; during a meal it lags with τ = 1/p2 ≈ 33 min. This
        # REPLACES the insulin_action SpeciesHead rate (its mass-action head output
        # is unused, exactly like glucose's). p2 bounded to a physiological band.
        p2 = _P2_MIN + _P2_RANGE * torch.sigmoid(self.log_p2)
        rates_out[..., _INSULIN_ACTION_IDX] = p2 * (
            nn.functional.relu(state[..., _INSULIN_IDX]) - state[..., _INSULIN_ACTION_IDX]
        )

        # Iter 94 — GLYCOGEN IS A STORAGE POOL, NOT A SPECIES IN EQUILIBRIUM.
        #
        # The mass-action assembly is `prod·prod_scale − cons·cons_scale·norm_state`,
        # and `norm_state` is ZERO at the pool's typical value. GlycogenFluxHead emits
        # prod = synthesis ≥ 0, so at typical the entire breakdown term vanishes and
        # the rate is ≥ 0. `typical` was therefore an ABSORBING FLOOR: measured on the
        # iter-93 artifact, both pools hit exactly `min - typical = 0.0000` in every
        # protocol — fasted, fed, and a 2 h hard bout — while the teacher falls 58 g
        # (liver) and 206 g (muscle) below start on the same input. No amount of
        # gradient can fix that; iters 55-57 read it as a supervision/timescale
        # problem and it was never either. (Doc: docs/iter94-spec.md.)
        #
        # Physically, glycogenolysis flux is set by DEMAND — fasting for the liver,
        # contraction for muscle — not by how far the pool sits from a reference
        # level, and it stops only as the pool empties. So the two pools get an
        # explicit flux balance in RAW g/min, exactly as glucose gets the explicit
        # minimal-model form above and for the same reason: the mass-action shape is
        # the wrong shape here, and the PRD's rule is to change the structure rather
        # than fit against it. The head's own (synthesis, breakdown) outputs and its
        # learned gut/insulin/activity gates are reused UNCHANGED; only how they are
        # turned into a rate changes.
        #
        #   d(pool)/dt = flux_scale · ( synth·headroom − breakdown·fullness )
        #     headroom = relu(1 − pool/capacity)  — synthesis stops at a full store,
        #                                            capacity = 1.3·typical (glycogen
        #                                            supercompensation is real, ~20-40%)
        #     fullness = clamp(pool/typical, 0, 1) — breakdown has FULL authority at a
        #                                            normal store and fades to zero as
        #                                            the pool empties, so the pool
        #                                            cannot go negative by construction
        # Both bracket terms are O(1) at a normal store, so `flux_scale` alone sets the
        # physiological magnitude; it is NOT 1/tau any more (the old cons_scale was,
        # which is why 3.3e-5 could never express an hours-scale bout).
        for idx, center, nscale, flux_scale in (
            (_LIVER_GLYCOGEN_IDX, _LIVER_GLY_CENTER, _LIVER_GLY_NORM_SCALE,
             _LIVER_GLY_FLUX),
            (_MUSCLE_GLYCOGEN_IDX, _MUSCLE_GLY_CENTER, _MUSCLE_GLY_NORM_SCALE,
             _MUSCLE_GLY_FLUX),
        ):
            pool = center + nscale * state[..., idx]
            fullness = torch.clamp(pool / center, min=0.0, max=1.0)
            headroom = relu(1.0 - pool / (_GLY_CAPACITY_FRAC * center))
            rates_out[..., idx] = flux_scale * (
                prod_raw[..., idx] * headroom - cons_raw[..., idx] * fullness
            )
        return rates_out
