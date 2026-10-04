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

    dG    = app·f_plasma                          meal appearance not stored
            + glyco + gng_released                 hepatic release (two fluxes, below)
            − k_ii·G − X·G − uptake_ex             obligatory, insulin-dependent, exercise
    dLGly = f_liver·app_g + gng_divert/mg − glyco/mg
    f_liver = 0.30·(0.5 + 0.5·ins_drive)·fill      direct pathway, not a head
    gng_divert = gng · ins_drive^0.25               indirect pathway, insulin above basal
    gng_released = gng − gng_divert                 what actually reaches plasma
    ins_drive = relu(I − Ib) / (relu(I − Ib) + Ib)
    dMGly = f_muscle·app_g + store·relu(X·G)/mg − brk_M
    dHep  = k·((glyco + gng_released)·VG_DL_PER_KG − Hep)  a lagged mg/kg/min readout

    EGP_b   = k_ii · Gb_emb                          per patient (basal EGP scales with Gb)
    glyco   = (1 − f_gng)·EGP_b · (LGly/LGly_b) · g_ins_glyco · g_gn · g_G
    gng     =      f_gng ·EGP_b · g_cort·√g_gn·g_ffa·g_ins_gng·g_G

``k_ii`` (obligatory uptake per mg/dL) and ``f_gng`` are the teacher's
population constants. ``k_ii = 2 / (VG_DL_PER_KG · 95)`` is the uptake that
puts a typical person at 2 mg/kg/min. ``f_gng`` is one half, Landau's
post-absorptive share (the teacher's ``Gng_b / Hep_b``). On the iter-107
weights both had walked, uptake from 0.0114 to 0.0083 per minute and the
share from 0.50 to 0.37. Uptake cancels at the glucose fixed point, so the
loss does not defend it, and the smaller turnover is fewer grams of
glycogenolysis per hour. The share then fell to keep a fed night spending
liver, which drops a long fast through 60 mg/dL. Restoring both on those
frozen weights returns the fed night to the teacher and the 48 h fast to
69.9 mg/dL, with both basals above 60.

Every structural gate is normalized to exactly 1 at the patient's basal
state (``g_ins`` at I = Ib, ``g_gn`` at Gn = Gnb, ``g_cort`` at Cort_b, ``g_ffa``
at FFA_b, ``g_G`` at G ≤ Gb), so at the fasted reference ``dG = EGP_b − k_ii·Gb
= 0`` exactly: Gb is the fixed point, not an attractor. The liver head's
breakdown output and the hepatic head's production output do not multiply
these fluxes. A learned gain there cancelled the first-order pool on the
iter-104 weights: the same checkpoint with those gains at 1 holds a 48 h fast.
Liver storage is the same kind of constant. The direct pathway takes 30 % of
appearance, half of it before insulin rises (the portal signal; Taylor 1996
lands near 19 % once insulin is averaged in). While insulin is above basal the
indirect pathway books gluconeogenic carbon into glycogen instead of blood
(Katz & McGarry 1984; the teacher's ``ins_drive^0.25``). At I ≤ Ib that share
is zero, so the fasted fixed point is unchanged. On the iter-106 weights the
store logit had gone to zero: a day of meals synthesized 0 g and glycogenolysis
removed 61 g, and the second morning was a fast.
The fasting fall emerges from pool depletion — glycogenolysis is first order
in ``LGly``, so EGP falls toward GNG alone and glucose settles where obligatory
uptake balances it, an absolute floor ``gng/k_ii`` the same for every Gb
(item 3.3). ``Gb_fasted``, the drop and the floor are gone.

Carbon: ``d(G/mg) + dLGly + dMGly = app_g − brk_M − (k_ii·G + X·G +
uptake_ex − gng − store·relu(X·G))/mg`` identically. A heavier person
converts a gram of carbohydrate into fewer mg/dL. There is no ``ra`` factor:
A10 deleted it, so ``app_g`` is the gut kernel's bioavailable mass and nothing
rescales it on the way in (the A10 note among the constants below says why).

A11: ``hepatic_output`` is the two fluxes that REACH PLASMA — glycogenolysis
plus ``gng_released`` — not ``glyco + gng``. The diverted gluconeogenic carbon
is on the liver-glycogen ledger, and the HGO cohort's Rizza-1981 / Basu-2000
tracer EGP does not count it. At the fasted reference ``ins_drive = 0``, so the
basal readout is unchanged.

``Ib = 10·exp(±0.9·tanh(head))`` is a zero-init per-patient head, and since A1
so is ``Gb = 95·exp(±0.45·tanh(head))`` — both decoded in the lognormal family
the teacher draws them from, so zero is the median person and the population
mean sits above it (PLAN §1).

B1 (PLAN §3) — THE INSULIN-SENSITIVITY FAMILY IS PER PERSON, AND MUSCLE IS NOT
LIVER. Five quantities now decode from the embedding as
``population_scalar · exp(L·tanh(head))`` with a zero-init head, the same
lognormal family as Gb and Ib and for the same reason — the teacher DRAWS all
five lognormally, so a zero-mean code must decode to the MEDIAN and the
population mean must sit above it, which Jensen gives for free and an additive
decode cannot give at all:

    si            peripheral (muscle) Si   teacher Si               σ0.5   L 1.25
    hep_ins_k     hepatic gate IC50        teacher glyc_ins_K       σ0.25  L 0.75
    gamma         β-cell gain              teacher gamma            σ0.3   L 0.75
    k_ins         insulin clearance        teacher n                σ0.3   L 0.75
    act_ins_sens  exercise sensitisation   teacher act_insulin_sens σ0.3   L 0.75

L is 2.5σ of the teacher's own draw, except the hepatic gate's 3σ, which covers
its clip ([12, 50] around 25) end to end. Measured over 4,000
``randomize_params`` draws the teacher's ±2σ log-ratios are Si [−1.00, +0.98],
glyc_ins_K [−0.49, +0.50], gamma [−0.61, +0.60], n [−0.60, +0.60] and
act_insulin_sens [−0.59, +0.60] against the reachable ±1.25 / ±0.75, so no
sampled patient sits on a bound. A zero embedding decodes to the population
scalar exactly, so the cold start is bit-identical to iter 109.

This is the item the plan is about. The PRD's own example of individual
variation is "exact insulin sensitivity", and through iter 109 all four of the
student's copies were ONE population scalar. Measured over 150 sampled teacher
patients after a 75 g mixed meal, incremental AUC over the 240 min after the
meal: corr(log Si, glucose iAUC) = −0.48, i.e. Si alone explains 22.5 % of
between-person glucose iAUC and 20.7 % of insulin iAUC, and the bottom-vs-top Si
decile is 6114 vs 3215 = 1.90× on glucose and 6491 vs 2990 = 2.17× on insulin.
(PLAN's headline 34 % / 31 % is the same quantity measured on TOTAL post-meal
AUC — corr −0.58, R² 33.8 % — which also carries the fasting level, and Gb loads
+0.50 on the same insulin-resistance latent that Si loads −0.70 on, so part of
that share is Gb. The direction, the decile ratio and the conclusion are
unchanged: it is the largest single axis by a wide margin.) The student could
imitate that only through meal amplitude, so glucose and insulin both moved for
the wrong reason, and no insulin-sensitising intervention (training, metformin,
weight loss, sleep) had a lever to act on. It now realizes 1.77× between those
same two Si deciles, against 1.81× on the teacher's own pure-Si sweep.

Muscle and liver are SEPARATE quantities, not one "insulin sensitivity": ``si``
multiplies ``xa`` into ``uptake_id`` (peripheral disposal) while the hepatic
factor moves the IC50 of both hepatic gates (``g_ins_glyco``, ``g_ins_gng``).
Donga 2010's clamp after sleep restriction is peripheral −29 % with hepatic
essentially unchanged, which is not expressible at all if the two are one
number — PLAN §3 records that as the reason the carve is exactly here, and C3's
sleep debt is the consumer. ONE factor moves BOTH hepatic gates: the teacher
varies only ``glyc_ins_K`` (``gng_ins_K`` is 80 for every patient), so a second
per-person decode on the GNG gate would be a head no ground truth can reach —
the iter-109 glucagon-basal failure mode — whereas one liver with one insulin
sensitivity costs one degree of freedom that ``glyc_ins_k`` supervises.

``act_ins_sens`` is not a re-parameterization: the teacher's ``si_effective =
Si·(1 + act_insulin_sens·act)`` had NO counterpart here, so acute exercise
reached glucose only through the insulin-INDEPENDENT ``exercise_uptake`` and did
not sensitise insulin action at all. It enters where the teacher puts it, INSIDE
the remote-insulin lag (``xa_rate``), not at the uptake site: the teacher's tape
writes ``insulin_action`` as ``X/(Si·10)``, which is exactly this product with
Si divided out, and ``cold_model_distillation_signal`` scores the student's
column against it — a gain applied after the lag would leave the student short
by ``(1 + act_ins_sens·act)`` for every minute of every bout, and would
sensitise instantly where the teacher ramps with τ = 1/p2 = 50 min. At the
median 0.3 a moderate bout (act 0.5) raises insulin-dependent uptake 15 % and a
maximal one 30 %.

All five are supervised against the teacher's own draw by
``SetpointSupervisionSignal`` (PLAN A4), in log-ratio-to-default units; without
that they would be five free per-person heads, which is the disease PLAN is
about.
Insulin is not mass-action. A learned basal × Hill could sit above Ib in a
fast (iter 100: 48 h ended at 12.2 µU/mL against the teacher's 3.8). The
rate is the teacher's restoring law:

    effective_Ib = Ib · max((min(G, Gb)/Gb)^5, 0.25)
    dI            = −k_ins · (I − effective_Ib) + γ · mod · relu(G − Gb) · incretin

GSIR is identically 0 at G ≤ Gb; fasting insulin is attracted to the
glucose-gated basal, not a learned floor. ``k_ins`` and ``γ`` are per-person
about a population scalar (centre 0.15 / 0.05, the teacher's ``n`` / ``gamma``;
B1 above); ``mod`` is a learned state-dependent gain. The thresholds that leaked (glycogen catabolic
gates) are DELETED, not clamped: muscle breakdown is ``relu(act − 0.10)``,
liver breakdown is the insulin gate above. ``mitochondrial_capacity`` has
ONE role — a scale on the clearance of lactate — and no other
head sees it. ``insulin_action`` is lagged ``(1 + act_ins_sens·act)·(I − Ib)/10``,
signed; it is not a concentration, so ``raw_state`` does not floor it at 0. The
only bound on how negative it can drive glucose is
``x_eff = max(Si·X, −0.05·k_ii)``.
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
point with nothing left to derive. Gnb is the marker's typical, 70 pg/mL,
not a head. The hepatic gates are centered on whatever basal the patient
declares, so a free head moves only the glucagon level. On the iter-108
weights that head took the prior person from 70 to 98, and the fed-day
trace sat near 101 against the teacher's 73, while liver, ketones and
glucose were unchanged when the head was put back at zero. ``α`` is
the teacher's 3.5, not a parameter. A free copy walked 3.5 → 0.31, and
on the iter-109 weights the 48 h rise was 8 pg/mL against a bar of 10.
Forcing 3.5 leaves a fed day where it is, because the stimulus is zero
while glucose sits on its basal, and lifts the fast to glucagon 99 and
glucose 71. ``k_gn`` and ``β`` stay population scalars (init 0.03 /
0.4). ``β`` stays positive, so the sign lives in ``(I − Ib)``: insulin
below basal disinhibits the alpha cell even when glucose is back at Gb.
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
# A1 (PLAN §1/§3) — Gb IS DECODED IN LOG SPACE, like Ib above and for the same
# reason: the teacher DRAWS it lognormally (`full_body.py`
# `clip(vary(p.Gb, 0.25, ir=0.50), 70, 130)`), and "the decoder family must
# match the generative family". The additive decode it replaces
# (`95 + 30·2.2·tanh`) has E[Gb] = 95 for a zero-mean code, i.e. the mean of a
# right-skewed population equal to its median — it cannot be right at both, and
# zero is defined to be the MEDIAN person (PLAN §2). `95·exp(L·tanh)` makes
# both true by Jensen with no loss term. L = 0.45 spans Gb ∈ [60.6, 148.9]
# mg/dL (z ∈ [−1.15, +1.80]), which covers the teacher's clipped [70, 130] with
# 9.4 mg/dL of margin below and 19.0 above. The old ±2.2 z reached [29, 161]:
# half of that span was glucose no living patient defends, and iter 96 had
# already had to clip the TEACHER at 70 for exactly that reason. The floor is
# 0.6 mg/dL above the 60 the benchmark's observed fasting glucose reaches, which
# is not a gap — Gb is the setpoint and a drawn-down liver settles BELOW it, at
# the absolute `gng/k_ii` floor that is the same for every Gb.
_GB_LOG_MAX = 0.45
# Iter 97: per-patient basal insulin in LOG space so it is positive by construction:
# Ib = 10·exp(±0.9·tanh) ∈ [4.1, 24.6] µU/mL — the teacher varies Ib with σ = 0.4
# lognormal loaded on insulin resistance, i.e. roughly this span at ±2σ.
_IB_LOG_MAX = 0.9
# A10 (PLAN §3): there is no `Ra` here any more. Through iter 109 meal amplitude
# carried THREE multiplying per-person gains — the gut kernel's `f_bio`, this
# module's `ra` (`log_ra` + `ra_baseline_net`, iter 80/88) and `1/V_G` via body
# mass — and glucose data identifies only their product (measured: scaling Ra and
# body mass together by 1.2 moves glucose 0.18 mg/dL, i.e. the pair is flat).
# The gain now lives once, as the gut kernel's bioavailable FRACTION, which is
# also the only one of the three that is bounded by a conservation law; V_G stays
# a known scale. `Ra_init = 0.8` moved into `GutModuleBase.F_BIO_INIT_FRACTION`,
# so the cold start is unchanged.

# Obligatory (insulin-independent) glucose uptake per mg/dL of glucose space —
# brain, blood cells, renal medulla. The teacher's uptake_ii: 2.0 mg/kg/min
# at Gb 95, so a typical person rests at the typical EGP of 2.0 mg/kg/min.
# A constant. A free copy walked 0.0114 → 0.0083 and the loss never saw it,
# because it cancels at the glucose fixed point.
_K_II = 2.0 / (VG_DL_PER_KG * 95.0)
# Si: RAW per-minute per normalized-insulin-unit; X = Si·Xa with Xa the lagged
# signed (I − Ib)/10. Band brackets the Bergman literature range with margin —
# the band bounds the POPULATION centre; B1's per-person factor multiplies it.
_SI_MIN = 0.0005
_SI_RANGE = 0.0195
_SI_INIT = 0.004  # = 10 × teacher Si (insulin normalization)
_KACT_MIN, _KACT_RANGE, _KACT_INIT = 0.0, 0.06, 0.02          # teacher exercise uptake
# Fraction of basal EGP that is gluconeogenesis at the fasted reference (teacher
# Gng_b / Hep_b = 1.0 / 2.0; Landau 1996: 47 % at 14 h). A constant. A free
# copy walked 0.50 → 0.37 to buy back the grams of glycogenolysis that the
# shrunken uptake had lost, and that walk is what drops a long fast through 60.
_F_GNG = 0.5
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
# Body mass: 70·exp(±0.45·tanh) ∈ [44.6, 109.8] kg. A4/B1 — 0.25 reached only
# [54.5, 89.9], and the teacher clips mass at [52, 110] with 1.7 % of 4,000 draws
# ABOVE 89.9 (max 110.0): those patients sat on a saturated tanh, where the head has
# no gradient and the decoded mass is simply wrong. 0.45 leaves 0.025 % outside.
# Widening it is only safe now: A10 deleted the `Ra` gain mass was confounded with
# (×1.2 on both moved glucose 0.18 mg/dL, i.e. the pair was flat), and this iteration
# supervises mass against the teacher's own kg, so the span is identified rather than
# free. Zero still decodes to 70 kg, so no default person moves.
_MASS_LOG_MAX = 0.45
_FFA_LOG_MAX = 0.5
# Fasting insulin: the teacher's restoring law (Polonsky 1988; PatientParams.n / gamma /
# fast_ins_exp / fast_ins_floor). Not a learned Hill on a mass-action basal.
_FAST_INS_EXP = 5.0
_FAST_INS_FLOOR = 0.25
_K_INS_INIT = 0.15
_GAMMA_INIT = 0.05
# B1 (PLAN §3) — the per-person log HALF-WIDTHS of the insulin-sensitivity family.
# Each decode is `population_scalar · exp(L·tanh(head))`, so the reachable factor is
# [e^−L, e^+L] about whatever the scalar holds and the zero code decodes to the
# scalar exactly. L = 2.5σ of the teacher's own draw; realized σ(log) over 4,000
# `randomize_params` draws is 0.499 / 0.247 / 0.301 / 0.303 / 0.297 against the
# declared 0.5 / 0.25 / 0.3 / 0.3 / 0.3, and the teacher's ±2σ log-ratios measured
# on the same draws are:
#   si            L 1.25 → ×[0.287, 3.490]   [−1.00, +0.98]
#   hep_ins_k     L 0.75 → ×[0.472, 2.117]   [−0.49, +0.50] — 3σ, chosen so the span
#                   also covers the teacher's CLIP [12, 50] = [−0.73, +0.69] end to
#                   end: 0 of 4,000 draws is unreachable
#   gamma         L 0.75 → ×[0.472, 2.117]   [−0.61, +0.60]
#   k_ins         L 0.75 → ×[0.472, 2.117]   [−0.60, +0.60]
#   act_ins_sens  L 0.75 → ×[0.472, 2.117]   [−0.59, +0.60] — here the clip's FLOOR
#                   (0.1, i.e. −1.10 = 3.7σ) is below the span's 0.142, which bounds
#                   the 0.5 % of draws the clip's own lower tail produces
# so every span contains ±2σ with margin — the tightest is gamma's lower edge at 0.46σ
# (0.137 in log units), the loosest the hepatic gate's 1.0σ — and no sampled patient
# saturates a tanh. Wider is not free: the span IS the per-person authority, and PLAN
# §3's rule is that authority needs the evidence that identifies it
# (`SetpointSupervisionSignal`).
_SI_LOG_MAX = 1.25
_HEP_INS_K_LOG_MAX = 0.75
_GAMMA_LOG_MAX = 0.75
_K_INS_LOG_MAX = 0.75
_ACT_INS_SENS_LOG_MAX = 0.75
# Teacher `si_effective = Si·(1 + act_insulin_sens·act)` (full_body.py). The student
# had NO activity term on insulin action at all, so this population scalar is new
# rather than moved; 0.3 is the teacher's default (clip [0.1, 0.6], loading fit +0.35,
# i.e. exercise sensitisation is itself a trainable adaptation).
_ACT_INS_SENS_INIT = 0.3
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
# Direct pathway into liver glycogen (teacher glyc_syn_frac_L). Half of the
# fraction is taken as absorption starts, before insulin has risen; the rest
# scales with ins_drive. Taylor 1996: 19 % of the meal by 5 h once that drive
# is averaged in. Muscle cold-start logit is its old 15 % share of the remainder.
_LIVER_DIRECT_FRAC = 0.30
_GNG_DIVERT_EXP = 0.25            # teacher gng_divert_exp
# x^0.25 is the teacher's shape and its derivative blows up at x = 0, which
# is exactly where a fast sits. x / (x + eps)^(1 − exp) matches it for any
# drive that is not tiny and is finite at zero.
_GNG_DIVERT_EPS = 1e-3
_MUSCLE_STORE_FRAC_INIT = 0.15
_PLASMA_FRAC_INIT = 1.0 - _LIVER_DIRECT_FRAC - _MUSCLE_STORE_FRAC_INIT

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
_ALPHA_GN = 3.5               # teacher alpha_gn. A free copy walked 3.5 → 0.31.
_GN_INS_INIT = 0.4            # teacher coefficient on (I − Ib) / (Ib + 10)
_GN_INS_OFFSET = 10.0


# B1/A4 — the student's MEDIAN PERSON for every quantity `person_params` decodes, i.e.
# the value a zero embedding gets. `SetpointSupervisionSignal` compares the decode
# against the teacher's draw as a log-ratio to each side's own default, and this is the
# student's half of that frame: it is what makes `si` (per normalized insulin unit) and
# the teacher's `Si` (per µU/mL) the same quantity despite the 10× normalization.
PERSON_PARAM_CENTERS: dict[str, float] = {
    "body_mass_kg": BODY_MASS_KG,
    "si": _SI_INIT,
    "glyc_ins_k": _GLYC_INS_K_INIT,
    "gamma": _GAMMA_INIT,
    "k_ins": _K_INS_INIT,
    "act_insulin_sens": _ACT_INS_SENS_INIT,
}
# The log half-width of each decode above, for whoever needs to know what is reachable
# (the span tests, and a calibration that wants to know when a target is out of range).
PERSON_PARAM_LOG_MAX: dict[str, float] = {
    "body_mass_kg": _MASS_LOG_MAX,
    "si": _SI_LOG_MAX,
    "glyc_ins_k": _HEP_INS_K_LOG_MAX,
    "gamma": _GAMMA_LOG_MAX,
    "k_ins": _K_INS_LOG_MAX,
    "act_insulin_sens": _ACT_INS_SENS_LOG_MAX,
}


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
        return self.post(self.network(x))

    def post(self, raw: torch.Tensor, stimulus=None) -> tuple[torch.Tensor, torch.Tensor]:
        raw = raw.squeeze(-1)
        mod = torch.exp(raw.clamp(-3.0, 4.0))
        return mod, torch.ones_like(raw)


class GlycogenFluxHead(nn.Module):
    """Glycogen as a flux integrator: the head emits the two learned GAINS of

        synthesis   = f_store · fill · remainder           muscle only; liver storage is the
                                                            direct-pathway fraction in fluxes
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
        return self.post(self.network(x))

    def post(self, raw: torch.Tensor, stimulus=None) -> tuple[torch.Tensor, torch.Tensor]:
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
                # Liver storage is the direct-pathway fraction, not a logit.
                _LIVER_GLYCOGEN_IDX: lambda inp, hd: ConstantFluxHead(),
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
        # Uptake and the gluconeogenic share are constants, not parameters.
        self.log_glyc_ins_k = nn.Parameter(torch.tensor(_inverse_softplus(_GLYC_INS_K_INIT)))
        self.log_gng_ins_k = nn.Parameter(torch.tensor(_inverse_softplus(_GNG_INS_K_INIT)))
        self.log_k_ins = nn.Parameter(torch.tensor(_inverse_softplus(_K_INS_INIT)))
        self.log_gamma = nn.Parameter(torch.tensor(_inverse_softplus(_GAMMA_INIT)))
        self.log_si = nn.Parameter(torch.tensor(_logit((_SI_INIT - _SI_MIN) / _SI_RANGE)))
        # B1: the teacher's `act_insulin_sens`, which the student had no counterpart for.
        self.log_act_ins_sens = nn.Parameter(torch.tensor(_inverse_softplus(_ACT_INS_SENS_INIT)))
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
        self.log_gn_ins = nn.Parameter(torch.tensor(_inverse_softplus(_GN_INS_INIT)))

        # Per-patient setpoint heads. Final layers zero-init ⇒ Gb = 95 and Ib = 10 for
        # every embedding at cold start; authority grows in training. A10 deleted the
        # Ra head: meal amplitude is the gut kernel's bioavailable fraction times V_G.
        _bh = max(8, hidden_dim // 4)

        def _zero_head() -> nn.Sequential:
            net = nn.Sequential(nn.Linear(embedding_dim, _bh), nn.Tanh(), nn.Linear(_bh, 1))
            with torch.no_grad():
                net[-1].weight.zero_()
                net[-1].bias.zero_()
            return net

        self.glucose_baseline_net = _zero_head()   # Gb
        self.insulin_baseline_net = _zero_head()   # Ib
        self.body_mass_net = _zero_head()          # kg; 70 at zero embedding
        self.ffa_baseline_net = _zero_head()
        self.mito_setpoint_net = _zero_head()
        # B1: the insulin-sensitivity family, each a log FACTOR on its population
        # scalar (so a zero code is bit-identical to iter 109). `hepatic_ins_k_net`
        # moves both hepatic gate IC50s together — one liver, one insulin
        # sensitivity — and is separate from `insulin_sens_net`, which is muscle.
        self.insulin_sens_net = _zero_head()         # si, peripheral
        self.hepatic_ins_k_net = _zero_head()        # glyc_ins_K and gng_ins_K
        self.beta_cell_gain_net = _zero_head()       # γ
        self.insulin_clearance_net = _zero_head()    # k_ins
        self.act_insulin_sens_net = _zero_head()     # act_insulin_sens

    # ---- per-patient setpoints -------------------------------------------------------

    def glucose_setpoint_raw(self, embedding: torch.Tensor) -> torch.Tensor:
        """Gb in mg/dL, decoded in LOG space: ``95·exp(0.45·tanh(head))`` ∈ [60.6, 148.9].

        A1: the teacher draws Gb lognormally, so this decoder's family is the
        generative family (PLAN §1) and E[Gb] > Gb(0) by Jensen — the zero code is
        the MEDIAN person and the population mean sits above it, both without a
        loss term. The additive form it replaces forced E[Gb] = median = 95.
        """
        return _GLUCOSE_CENTER * torch.exp(
            _GB_LOG_MAX * torch.tanh(self.glucose_baseline_net(embedding).squeeze(-1)))

    def glucose_setpoint_z(self, embedding: torch.Tensor) -> torch.Tensor:
        """Gb as a glucose z-score, for whoever needs to compare against a target.

        The decode's SHAPE is this module's business: `SetpointSupervisionSignal`
        used to rebuild `_GLUCOSE_BASELINE_MAX_Z · tanh(head)` itself, which was
        the same z only while the decode stayed additive. Going through here, a
        later change of family moves one line instead of every consumer.
        """
        return (self.glucose_setpoint_raw(embedding) - _GLUCOSE_CENTER) / _GLUCOSE_NORM_SCALE

    def insulin_setpoint_raw(self, embedding: torch.Tensor) -> torch.Tensor:
        return _INSULIN_CENTER * torch.exp(
            _IB_LOG_MAX * torch.tanh(self.insulin_baseline_net(embedding).squeeze(-1)))

    def body_mass_kg(self, embedding: torch.Tensor) -> torch.Tensor:
        return BODY_MASS_KG * torch.exp(
            _MASS_LOG_MAX * torch.tanh(self.body_mass_net(embedding).squeeze(-1)))

    def mg_dl_per_g(self, embedding: torch.Tensor) -> torch.Tensor:
        return 1000.0 / (self.body_mass_kg(embedding) * VG_DL_PER_KG)

    def ffa_setpoint_raw(self, embedding: torch.Tensor) -> torch.Tensor:
        return _FFA_CENTER * torch.exp(
            _FFA_LOG_MAX * torch.tanh(self.ffa_baseline_net(embedding).squeeze(-1)))

    def gn_setpoint_raw(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.log_k_gn.new_full(embedding.shape[:-1], _GN_CENTER)

    def mito_setpoint_raw(self, embedding: torch.Tensor) -> torch.Tensor:
        return torch.exp(
            _MITO_LOG_MAX * torch.tanh(self.mito_setpoint_net(embedding).squeeze(-1)))

    @staticmethod
    def _person_factor(net: nn.Module, log_max: float, embedding: torch.Tensor) -> torch.Tensor:
        """``exp(L·tanh(head(e)))`` — the per-person log factor every B1 decode shares.

        Positive for every embedding including the ‖e‖ = 8 calibration clamp, bounded
        by ``[e^−L, e^+L]`` because tanh saturates, and exactly 1 at a zero-init head.
        """
        return torch.exp(log_max * torch.tanh(net(embedding).squeeze(-1)))

    def person_params(self, embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        """The per-person quantities under the teacher's OWN keys (B1 + A4).

        These are the keys of ``Episode.patient_params``, so
        ``SetpointSupervisionSignal`` can compare the decode against the teacher's
        draw without a translation table on the model side. ``si`` is in the
        student's frame (per NORMALIZED insulin unit, hence 10× the teacher's ``Si``);
        the supervision compares log-ratios to each side's own median person, which
        is the only frame in which the two are the same quantity.

        ``hep_ins_k_factor`` is dimensionless and multiplies BOTH hepatic gate IC50s
        (see the module docstring on why one factor, not two heads).
        """
        hep = self._person_factor(self.hepatic_ins_k_net, _HEP_INS_K_LOG_MAX, embedding)
        glyc_k = nn.functional.softplus(self.log_glyc_ins_k) * hep
        return {
            "body_mass_kg": self.body_mass_kg(embedding),
            "si": ((_SI_MIN + _SI_RANGE * torch.sigmoid(self.log_si))
                   * self._person_factor(self.insulin_sens_net, _SI_LOG_MAX, embedding)),
            "glyc_ins_k": glyc_k,
            "gng_ins_k": nn.functional.softplus(self.log_gng_ins_k) * hep,
            "hep_ins_k_factor": hep,
            "gamma": (nn.functional.softplus(self.log_gamma)
                      * self._person_factor(self.beta_cell_gain_net, _GAMMA_LOG_MAX, embedding)),
            "k_ins": (nn.functional.softplus(self.log_k_ins)
                      * self._person_factor(self.insulin_clearance_net, _K_INS_LOG_MAX, embedding)),
            "act_insulin_sens": (
                nn.functional.softplus(self.log_act_ins_sens)
                * self._person_factor(self.act_insulin_sens_net, _ACT_INS_SENS_LOG_MAX, embedding)),
        }

    def k_ii(self) -> torch.Tensor:
        return self.log_si.new_tensor(_K_II)

    # ---- heads -----------------------------------------------------------------------

    def head_state_columns(self) -> list[int]:
        """Iter 97 (3.11): no head reads ``mitochondrial_capacity`` — it has one role,
        the lactate clearance scale — so every head's input drops that column."""
        return [i for i in range(_N_SPECIES) if i != _MITO_IDX]

    # ---- the rate, split by what it depends on (see base.PhysiologyModule) ------------

    def constants(self, embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        """Per-patient setpoints and the population scalars of the balance."""
        gb = self.glucose_setpoint_raw(embedding)
        ib = self.insulin_setpoint_raw(embedding)
        pp = self.person_params(embedding)
        mass_kg = pp["body_mass_kg"]
        mg = 1000.0 / (mass_kg * VG_DL_PER_KG)
        ffa_b = self.ffa_setpoint_raw(embedding)
        gn_b = self.gn_setpoint_raw(embedding)
        mito_sp = self.mito_setpoint_raw(embedding)
        k_ii = self.k_ii()
        ic50_keto = nn.functional.softplus(self.log_keto_ins_supp)
        k_keto = nn.functional.softplus(self.log_k_keto)
        k_ffa = ffa_b.new_tensor(_K_FFA)
        ic50_lip = nn.functional.softplus(self.log_lip_ic50)
        return {
            "gb": gb, "ib": ib, "mg_dl_per_g": mg, "body_mass_kg": mass_kg,
            "ffa_b": ffa_b, "gn_b": gn_b, "mito_sp": mito_sp,
            "k_ii": k_ii, "egp_b": k_ii * gb, "f_gng": k_ii.new_tensor(_F_GNG),
            # B1: four of these are per-person factors on their population scalar,
            # decoded once in `person_params` so nothing recomputes a head.
            "glyc_k": pp["glyc_ins_k"],
            "gng_k": pp["gng_ins_k"],
            "si": pp["si"],
            "k_act": _KACT_MIN + _KACT_RANGE * torch.sigmoid(self.log_k_act),
            "k_ins": pp["k_ins"],
            "gamma": pp["gamma"],
            "act_ins_sens": pp["act_insulin_sens"],
            "p2": _P2_MIN + _P2_RANGE * torch.sigmoid(self.log_p2),
            "ic50_keto": ic50_keto, "k_keto": k_keto,
            "k_bhb": (k_keto * ffa_b / (1.0 + ib / ic50_keto)) / _BHB_CENTER,
            "k_gn": nn.functional.softplus(self.log_k_gn),
            "alpha_gn": self.log_k_gn.new_tensor(_ALPHA_GN),
            "gn_ins": nn.functional.softplus(self.log_gn_ins),
            "k_ffa": k_ffa, "ic50_lip": ic50_lip,
            "lip_max": ffa_b * k_ffa * (1.0 + ib / ic50_lip),
            "lac_prod_scale": self.prod_scale[_LACTATE_IDX],
            "lac_cons_scale": self.cons_scale[_LACTATE_IDX],
            "hep_cons_scale": self.cons_scale[_HEPATIC_IDX],
        }

    def drives(
        self,
        external: torch.Tensor,
        coupling: torch.Tensor,
        time_features: torch.Tensor,
        const: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Meal appearance, the energy ledger and the activity terms — the protocol."""
        relu = nn.functional.relu
        act = external[..., _ACTIVITY_EXTERNAL_IDX]
        # Gut glucose appearance is in the 70 kg reference space, and it is ALREADY
        # the bioavailable mass (the kernel's f_bio ≤ MG_DL_PER_G per gram). Grams
        # are that density / MG_DL_PER_G; this patient's mg/dL uses their own V_G.
        # A10: no `ra` factor here — a second gain on the same mass is what made
        # absorbed-vs-ingested unidentifiable, and it is the gut's number to own.
        app_g = coupling[..., _GUT_GLUCOSE_COUPLING_IDX] / MG_DL_PER_G
        lipid_app = coupling[..., _LIPID_COUPLING_IDX].clamp(min=0.0)
        amino_app = coupling[..., _AMINO_COUPLING_IDX].clamp(min=0.0)
        lipid_g = lipid_app / _LIPID_UNITS_PER_G
        amino_g = amino_app / _LIPID_UNITS_PER_G
        kcal_in = 4.0 * app_g + 9.0 * lipid_g + 4.0 * amino_g
        kcal_out = _BMR_KCAL_PER_MIN * (const["body_mass_kg"] / BODY_MASS_KG) + _ACT_KCAL_PER_MIN * act
        return {
            "app_g": app_g,
            "app_eff": app_g * const["mg_dl_per_g"],
            "act_above_rest": relu(act - _MUSCLE_ACT_REST),
            "exercise_gain": const["k_act"] * act,
            # B1: the teacher's `si_effective = Si·(1 + act_insulin_sens·act)`. `act`
            # is an external input and the sensitivity is a per-person constant, so
            # the product is a drive and `state_fluxes` reads it rather than reaching
            # for `external` again. `relu` on `act`, not on the product: a malformed
            # negative activity would otherwise be able to INVERT insulin action,
            # which is the catastrophe wall `x_eff`'s floor exists to keep away from.
            "ins_sens_act_gain": 1.0 + const["act_ins_sens"] * relu(act),
            "fat_rate": (kcal_in - kcal_out) / _KCAL_PER_KG_FAT,
            "ffa_from_lipid": _FFA_FROM_LIPID * lipid_app,
            "glucagon_from_amino": _GN_FROM_AMINO * amino_app,
            "mito_training": _MITO_TRAIN_GAIN * relu(act - 0.2),
        }

    def state_fluxes(
        self,
        state: torch.Tensor,
        coupling: torch.Tensor,
        c: dict[str, torch.Tensor],
        d: dict[str, torch.Tensor],
        raw_heads: dict[object, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Every state-dependent term of the metabolic ODE, in raw units (mg/dL/min of
        glucose space; g/min for the pools)."""
        relu = torch.relu
        (g, ins, gn, ffa, bhb, lac, hep, lgly, mgly, mito, xa, _fat) = self.raw_state(state).unbind(-1)
        cort = (_CORT_CENTER + _CORT_NORM_SCALE * coupling[..., _CORTISOL_COUPLING_IDX]).clamp(min=0.05)
        glp1 = (_GLP1_CENTER + _GLP1_SCALE * coupling[..., _GLP1_COUPLING_IDX]).clamp(min=0.1)
        gb, ib, mg = c["gb"], c["ib"], c["mg_dl_per_g"]
        k_ii, egp_b, f_gng = c["k_ii"], c["egp_b"], c["f_gng"]
        app_g = d["app_g"]

        gsir_mod, _ = self.heads[_INSULIN_IDX].post(raw_heads[_INSULIN_IDX])
        lac_prod, lac_cons = self.heads[_LACTATE_IDX].post(raw_heads[_LACTATE_IDX])
        _, hep_cons = self.heads[_HEPATIC_IDX].post(raw_heads[_HEPATIC_IDX])
        store_logit, muscle_break = self.heads[_MUSCLE_GLYCOGEN_IDX].post(
            raw_heads[_MUSCLE_GLYCOGEN_IDX])

        glucose_dev = (g - gb) / _GLUCOSE_NORM_SCALE
        insulin_dev = (ins - ib) / _INSULIN_NORM_SCALE

        fill_l = torch.sigmoid(
            (_GLY_CAPACITY_FRAC * _LIVER_GLY_CENTER - lgly) / (_GLY_FILL_WIDTH_FRAC * _LIVER_GLY_CENTER))
        fill_m = torch.sigmoid(
            (_GLY_CAPACITY_FRAC * _MUSCLE_GLY_CENTER - mgly) / (_GLY_FILL_WIDTH_FRAC * _MUSCLE_GLY_CENTER))
        ins_excess = relu(ins - ib)
        ins_drive = ins_excess / (ins_excess + ib.clamp(min=1e-3))
        f_liver = _LIVER_DIRECT_FRAC * (0.5 + 0.5 * ins_drive) * fill_l
        muscle_share = torch.sigmoid(store_logit) * fill_m
        f_muscle = muscle_share * (1.0 - f_liver)
        f_plasma = 1.0 - f_liver - f_muscle
        syn_liver = f_liver * app_g
        syn_muscle_oral = f_muscle * app_g
        appearance_plasma = d["app_eff"] * f_plasma

        g_ins_glyco = _ins_gate(ins, ib, c["glyc_k"], _GLYC_INS_N)
        g_ins_gng = _ins_gate(ins, ib, c["gng_k"], _GNG_INS_N)
        g_gn = _hill_centred(gn, c["gn_b"], _HGO_GN_N)
        g_cort = 1.0 + _GNG_CORT_AMP * torch.tanh(torch.log(cort / _CORT_CENTER))
        g_ffa = (ffa.clamp(min=1e-3) / c["ffa_b"]) ** _GNG_FFA_EXP
        g_g = (gb / torch.maximum(g, gb)) ** _HEP_AUTOREG_M
        glycogenolysis_plasma = ((1.0 - f_gng) * egp_b * (lgly / _LIVER_GLY_CENTER)
                                 * g_ins_glyco * g_gn * g_g)
        gng_plasma = (f_gng * egp_b * g_cort * torch.sqrt(g_gn) * g_ffa * g_ins_gng * g_g)
        gng_divert = gng_plasma * ins_drive / (
            ins_drive + _GNG_DIVERT_EPS).pow(1.0 - _GNG_DIVERT_EXP)
        gng_released = gng_plasma - gng_divert
        brk_liver = glycogenolysis_plasma / mg

        avail_m = mgly / (mgly + _MUSCLE_GLY_K)
        brk_muscle = _MUSCLE_GLY_FLUX * muscle_break * d["act_above_rest"] * avail_m

        uptake_ii = k_ii * g
        x_eff = torch.maximum(c["si"] * xa, -_INS_DEP_BASAL_FRAC * k_ii)
        uptake_id = x_eff * g
        exercise_uptake = d["exercise_gain"] * relu(g - 0.8 * gb)
        syn_muscle_id = _ID_STORE_FRAC * relu(uptake_id) / mg * fill_m
        syn_muscle = syn_muscle_oral + syn_muscle_id

        glp1_excess = relu(glp1 - _GLP1_CENTER)
        incretin = 1.0 + _INCRETIN_GAIN * glp1_excess / (glp1_excess + _K_INCRETIN)
        glucose_ratio = torch.minimum(g / gb.clamp(min=1.0), torch.ones_like(g))
        effective_ib = ib * torch.clamp(glucose_ratio ** _FAST_INS_EXP, min=_FAST_INS_FLOOR)
        ins_gsir = c["gamma"] * gsir_mod * relu(g - gb) * incretin
        ins_restoring = -c["k_ins"] * (ins - effective_ib)
        ins_rate = ins_restoring + ins_gsir
        # B1: remote insulin relaxes toward the ACTIVITY-SENSITISED insulin deviation,
        # which is the teacher's `dX = −p2·X + p3·(I − Ib)` with `p3 = Si_eff·p2`
        # exactly (divide both by Si: the student's `xa` is Si-free by construction).
        # Putting the gain here and not on `uptake_id` is what makes the student's
        # `insulin_action` column the same quantity the teacher writes on its tape
        # (`X/(Si·10)`), and gives the sensitisation the teacher's τ = 1/p2 ramp
        # instead of switching it on the minute a bout starts.
        xa_rate = c["p2"] * (insulin_dev * d["ins_sens_act_gain"] - xa)

        # A11: hepatic_output is RELEASE INTO PLASMA, so the gluconeogenic carbon
        # routed into glycogen (`gng_divert`) is not part of it. The marker's own
        # cohort spec (`knowledge/cohorts/glucose_handling.meal_hgo_suppression`)
        # cites Rizza 1981 / Basu 2000, which measure tracer EGP — appearance in
        # plasma — so a target that included the diverted share was scoring the
        # student against a quantity the literature does not report. The teacher
        # already does it this way (`full_body.glucose_fluxes`: `hep_target =
        # glyco_t + gng_rel_t`); this closes the gap. At the fasted fixed point
        # ins_drive = 0, so gng_divert = 0 and the basal readout is unchanged at
        # the textbook 2.0 mg/kg/min. The postprandial one falls further, which is
        # the direction the cohort's −1.0 ± 0.5 mg/kg/min wants: measured on a
        # fresh model (75 g meal, 150-240 min window) the suppression goes from
        # −0.65 to −1.22 mg/kg/min, i.e. from 0.7σ short of the target to 0.4σ
        # past it, with the fasted arm at 1.95 in both.
        hep_target = (glycogenolysis_plasma + gng_released) * VG_DL_PER_KG
        hep_rate = hep_cons * c["hep_cons_scale"] * (hep_target - hep)

        glyco_depletion = relu(1.0 - lgly / _LIVER_GLY_CENTER)
        ketogenesis = (c["k_keto"] * ffa / (1.0 + ins / c["ic50_keto"])
                       * (1.0 + _KETO_GLYC_GAIN * glyco_depletion))
        bhb_rate = ketogenesis - c["k_bhb"] * bhb

        glucagon_stim = c["alpha_gn"] * relu(gb - g) / gb.clamp(min=1.0)
        glucagon_supp = c["gn_ins"] * (ins - ib) / (ib + _GN_INS_OFFSET)
        glucagon_restoring = -c["k_gn"] * (gn - c["gn_b"])
        gn_rate = glucagon_restoring + glucagon_stim - glucagon_supp + d["glucagon_from_amino"]
        lipolysis = c["lip_max"] / (1.0 + ins / c["ic50_lip"])
        ffa_rate = lipolysis - c["k_ffa"] * ffa + d["ffa_from_lipid"]

        mito_rate = -_K_MITO * (mito - c["mito_sp"]) + d["mito_training"]
        lactate_from_glyco = _LAC_GLYCO_GAIN * brk_muscle
        lac_rate = (lac_prod * c["lac_prod_scale"]
                    - lac_cons * c["lac_cons_scale"] * mito * lac
                    + lactate_from_glyco)
        glucose_rate = (appearance_plasma + glycogenolysis_plasma + gng_released
                        - uptake_ii - uptake_id - exercise_uptake)
        liver_rate = syn_liver + gng_divert / mg - brk_liver
        muscle_rate = syn_muscle - brk_muscle

        return {
            "glucose_dev": glucose_dev, "insulin_dev": insulin_dev,
            "effective_ib": effective_ib, "incretin": incretin,
            "mito": mito, "gsir_mod": gsir_mod, "muscle_store_logit": store_logit,
            "f_liver": f_liver, "f_muscle": f_muscle, "f_plasma": f_plasma,
            "ins_drive": ins_drive,
            "syn_liver": syn_liver, "syn_muscle": syn_muscle, "syn_muscle_id": syn_muscle_id,
            "syn_muscle_oral": syn_muscle_oral,
            "brk_liver": brk_liver, "brk_muscle": brk_muscle,
            "appearance_plasma": appearance_plasma,
            "glycogenolysis_plasma": glycogenolysis_plasma, "gng_plasma": gng_plasma,
            "gng_divert": gng_divert, "gng_released": gng_released,
            "g_ins_glyco": g_ins_glyco, "g_ins_gng": g_ins_gng, "g_gn": g_gn,
            "g_cort": g_cort, "g_ffa": g_ffa, "g_g": g_g,
            "uptake_ii": uptake_ii, "uptake_id": uptake_id, "exercise_uptake": exercise_uptake,
            "ins_gsir": ins_gsir, "ins_restoring": ins_restoring, "ins_rate": ins_rate,
            "xa": xa, "xa_rate": xa_rate, "xa_drive": insulin_dev * d["ins_sens_act_gain"],
            "hep_target": hep_target, "hep_rate": hep_rate,
            "ketogenesis": ketogenesis, "glyco_depletion": glyco_depletion,
            "bhb_rate": bhb_rate, "mito_rate": mito_rate,
            "lactate_from_glyco": lactate_from_glyco, "lactate_rate": lac_rate,
            "glucagon_stim": glucagon_stim, "glucagon_supp": glucagon_supp,
            "glucagon_restoring": glucagon_restoring, "gn_rate": gn_rate,
            "lipolysis": lipolysis, "ffa_rate": ffa_rate,
            "glucose_rate": glucose_rate, "liver_glycogen_rate": liver_rate,
            "muscle_glycogen_rate": muscle_rate,
        }

    def step(
        self,
        state: torch.Tensor,
        coupling: torch.Tensor,
        const: dict[str, torch.Tensor],
        drv: dict[str, torch.Tensor],
        raw: dict[object, torch.Tensor],
    ) -> torch.Tensor:
        f = self.state_fluxes(state, coupling, const, drv, raw)
        return torch.stack([
            f["glucose_rate"], f["ins_rate"], f["gn_rate"], f["ffa_rate"], f["bhb_rate"],
            f["lactate_rate"], f["hep_rate"], f["liver_glycogen_rate"],
            f["muscle_glycogen_rate"], f["mito_rate"], f["xa_rate"], drv["fat_rate"],
        ], dim=-1)

    def fluxes(
        self,
        state: torch.Tensor,
        coupling: torch.Tensor,
        external: torch.Tensor,
        embedding: torch.Tensor,
        time_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Every named term of the metabolic ODE at one time point — the patient's
        constants, the protocol's drives and the state's fluxes in one dict. ``step``
        assembles the rates from the same terms; the carbon-budget test and the
        probes read them directly."""
        c = self.constants(embedding)
        d = self.drives(external, coupling, time_features, c)
        raw = self.head_outputs(state, coupling, external, embedding, time_features)
        return {**c, **d, **self.state_fluxes(state, coupling, c, d, raw)}
