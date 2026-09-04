"""
Coherent full-body physiology simulator.

Generates multi-day trajectories where all state markers evolve jointly
in a single coupled ODE. Cross-system couplings (cortisol → glucose,
insulin → ghrelin, glucose → cortisol, etc.) are active during data
generation so the model trains on coherent joint dynamics.

This is training data generation — it can use any model or hand-tuning
that produces plausible trajectories. The dynamics here come from the
individual knowledge contributions (Bergman, circadian, autonomic CV)
wired together into one system.

There is deliberately NO arterial baroreflex term in dHR; see the
CARDIOVASCULAR block below for the measurements behind that decision.

Sources:
  - Bergman et al. (1979): glucose-insulin minimal model
  - Weitzman et al. (1971): cortisol circadian rhythm
  - Mancia (1993): ambulatory blood pressure dynamics
  - Task Force ESC/NASPE (1996): heart rate variability
"""

from dataclasses import dataclass

import numpy as np

from ..types import STATE_DIM, MARKER_INDEX
from .base import Episode, KnowledgeContribution, CouplingPrior


def _circadian(t_abs_min: float, amplitude: float, peak_hour: float) -> float:
    hour = t_abs_min / 60.0
    return amplitude * np.cos(2 * np.pi * (hour - peak_hour) / 24.0)


def _anticipation_drive(t_abs_min: float, meal_hours, ramp_h: float, decay_h: float) -> float:
    """Entrained pre-meal drive in [0, 1]: half-cosine ramp over `ramp_h` before
    each habitual meal hour, half-cosine decay over `decay_h` after it."""
    hour = (t_abs_min / 60.0) % 24.0
    best = 0.0
    for hm in meal_hours:
        dt = (hour - hm + 12.0) % 24.0 - 12.0        # signed hours from the habitual hour
        if -ramp_h <= dt <= 0.0:
            v = 0.5 * (1.0 - np.cos(np.pi * (dt + ramp_h) / ramp_h))
        elif 0.0 < dt <= decay_h:
            v = 0.5 * (1.0 + np.cos(np.pi * dt / decay_h))
        else:
            v = 0.0
        best = max(best, v)
    return best


def _hpa_drive(t_abs_min: float, rise_start_h: float, peak_h: float,
               fall_tau_h: float = 6.0) -> float:
    """Asymmetric 24 h HPA drive in [0, 1]: a half-cosine rise from `rise_start_h`
    to `peak_h`, then an exponential decline (time constant `fall_tau_h`) that is
    normalized to reach exactly 0 at the start of the next rise. Weitzman 1971:
    cortisol is ~60% of its peak by noon, ~25% by 20:00, quiescent 20:00-02:00."""
    hour = (t_abs_min / 60.0) % 24.0
    rise = (peak_h - rise_start_h) % 24.0
    fall = 24.0 - rise
    since_start = (hour - rise_start_h) % 24.0
    if since_start < rise:
        return 0.5 * (1.0 - np.cos(np.pi * since_start / rise))
    x = since_start - rise
    e_end = np.exp(-fall / fall_tau_h)
    return (np.exp(-x / fall_tau_h) - e_end) / (1.0 - e_end)


# --- Iter 97: THE GLUCOSE LEDGER IS IN MASS UNITS ------------------------------
# Glucose space is a distribution volume: Bergman's V_G ~ 1.85 dL/kg (the same
# figure Cobelli/Dalla Man use for the oral minimal model). Every flux below is
# expressed in mg/dL/min OF THAT SPACE, and every conversion goes through these
# three constants, so a gram of carbohydrate eaten, a gram of glycogen stored and
# a milligram of glucose in blood are the SAME CARBON on both sides of every
# equation. Body mass is fixed at 70 kg for the teacher population (the literature
# anchors are per-kg or 70-kg-normalized; a per-patient mass adds nothing that the
# per-patient Gb/Si/Hep do not already carry).
# The four constants live in ``pulse.types`` so the student's gut kernel and
# metabolic ledger share them without importing teacher code.
from ..types import BODY_MASS_KG, VG_DL_PER_KG, VG_DL, MG_DL_PER_G  # noqa: E402

# Iter 97: the carbohydrate appearance kernel is MASS-CONSERVING. The integral of
# `_meal_absorption` over all time is carbs (g) * MG_DL_PER_G, i.e. the whole
# ingested carbohydrate appears in glucose space and nowhere else. Through iter 96
# this gain was 2.55 -- a phenomenological scale that put 32% of the meal into
# glucose space and left the other 68% unaccounted for -- because the kernel was
# ~3x too peaky for a real absorption curve and the gain was the knob that held the
# glucose excursion at the literature +40-60 mg/dL. Mass now comes from the kernel;
# amplitude comes from where physiology puts it: hepatic first-pass uptake into
# glycogen (`glyc_syn_frac_L`, debited from glucose -- see the metabolic block) and
# insulin-stimulated disposal (`Si`), which is why both were re-sized in this
# iteration (measured, see PatientParams).
CARB_APPEARANCE_GAIN = MG_DL_PER_G

# Iter 97: kernels integrate to >= 99.7% of their mass. A gamma-2 kernel
# rate^2 t e^{-rate t} has retained 1 - (1 + x) e^{-x} of its mass at x = rate * t;
# x = 8 gives 0.997. The fixed 300-min cutoff destroyed 12.6% of the slow fraction
# at the default slow rate and 66% at the floor of the sampled range -- 3.8% of all
# ingested carbohydrate on average across the population, 14.4% in the worst
# patient, silently. The truncation is now a property of the kernel, not a number.
_KERNEL_CUTOFF_X = 8.0


def _kernel_cutoff_min(rate: float) -> float:
    return _KERNEL_CUTOFF_X / max(rate, 1e-6)


def _meal_absorption(t: float, meal_time: float, carbs: float,
                     rate: float = 0.03) -> float:
    dt = t - meal_time
    if dt < 0 or dt > _kernel_cutoff_min(rate):
        return 0.0
    return carbs * rate * rate * dt * np.exp(-rate * dt) * CARB_APPEARANCE_GAIN


def _fat_absorption(t: float, meal_time: float, fats: float,
                    rate: float = 0.015) -> float:
    dt = t - meal_time
    if dt < 0 or dt > _kernel_cutoff_min(rate):
        return 0.0
    return fats * rate * rate * dt * np.exp(-rate * dt) * 3.0


def _duodenal_delivery(t: float, meal_time: float, grams: float,
                       fast_rate: float = 0.10, slow_rate: float = 0.005,
                       slow_frac: float = 0.35) -> float:
    """Nutrient delivery INTO THE DUODENUM (g/min) — gastric emptying.

    Iter 95. Distinct from `_fat_absorption` / `_protein_absorption`, which are
    SYSTEMIC appearance: fat reaches the blood via chylomicrons and peaks around
    60 min, whereas duodenal I-cells see nutrient arriving within minutes. Driving
    CCK from systemic appearance put the modelled peak at +68 min against a
    literature +10 min — the wrong signal, not the wrong gain.

    TWO COMPONENTS, following the same fast/slow split this module already uses for
    carbohydrate absorption, because one component cannot fit the CCK data:

      fast  gamma-shaped, peaking at 1/fast_rate = 10 min — the rapid liquid-phase
            emptying (plus cephalic phase) that produces the sharp early CCK peak
            and its fall to roughly half by 30 min.
      slow  exponential, tau = 1/slow_rate = 200 min — continued emptying of the
            solid phase, which is what keeps CCK elevated for the observed 3-5 h.

    A single fast kernel reproduces the peak but loses the plateau; a single slow one
    reproduces neither. Measured: with one exponential, peak and the +30 min value
    moved together and no (rate, gain) pair satisfied both anchors at once.
    """
    dt = t - meal_time
    if dt < 0 or dt > 480:
        return 0.0
    fast = fast_rate * fast_rate * dt * np.exp(-fast_rate * dt)
    slow = slow_rate * np.exp(-slow_rate * dt)
    return grams * ((1.0 - slow_frac) * fast + slow_frac * slow)


def _protein_absorption(t: float, meal_time: float, proteins: float,
                        rate: float = 0.02) -> float:
    dt = t - meal_time
    if dt < 0 or dt > _kernel_cutoff_min(rate):
        return 0.0
    return proteins * rate * rate * dt * np.exp(-rate * dt) * 3.0


@dataclass
class PatientParams:
    # Glucose-Insulin (Bergman minimal model). Iter 21 recalibration:
    # gamma 0.015 -> 0.07 and h 80 -> 95 align cold-model OGTT insulin
    # peak with DeFronzo (~60 uU/mL) and zero out GSIR at fasting glucose
    # so insulin can drop below Ib during a fast. Si 0.0002 -> 0.0004
    # tightens late-glucose clearance toward the OGTT 120-min target.
    # --- Iter 97: Gb IS A FIXED POINT, NOT AN ATTRACTOR -------------------------
    # Through iter 96 the glucose ODE was `-(Sg + X)(G - Gb) + Ra + small terms`:
    # the whole basal turnover (Sg*Gb ~ 222 mg/min) was implicit in one restoring
    # term, hepatic output was booked a second time as a 5.8 mg/min `Hep` state and
    # a third time as liver-glycogen breakdown that never reached blood, and the
    # fasting fall was produced by MOVING the attractor (`Gb_fasted`), not by a flux
    # deficit. Below the setpoint, insulin action was a glucose SOURCE (item 3.10).
    #
    # Now: dG = Ra - synthesis + EGP - uptake, with every term a mass flux (see the
    # module constants), and Gb is the level at which EGP_b = k_ii * Gb -- the
    # obligatory (insulin-independent) uptake balances basal hepatic output. `Sg`
    # is DERIVED: the linearized restoring rate is the uptake constant plus the
    # liver's own glucose suppression, Sg = uptake_ii * (1 + hep_autoreg_m).
    # Default 0.01138 * 1.6 = 0.0182 -- the value iters 21-96 carried, now with
    # its two halves named.
    Sg: float = 0.0182
    # Iter 97: UNCHANGED at 0.0004, and that is a finding. When the kernel first
    # became mass-conserving the same +58.7 mg/dL excursion seemed to need Si 2-4x
    # higher; it did not -- it needed the absorption curve to have its real shape
    # (a plateau, not a spike), hepatic first-pass uptake to respond to the portal
    # load from the first minute, and the second phase of secretion. With those
    # three structures in place the swept optimum for Si is 3.5-5 x 10^-4 /min per
    # uU/mL: Bergman's IVGTT range for healthy adults (Bergman 1979; 4-8 x 10^-4).
    Si: float = 0.0004
    Gb: float = 95.0
    Ib: float = 10.0
    # Iter 97: 0.03 -> 0.02 (Bergman's p2 is 0.02-0.03; the slower remote-insulin
    # lag holds the postprandial tail, measured across the sweep in the iter-97
    # report).
    p2: float = 0.02
    n: float = 0.15
    # Iter 97: 0.07 -> 0.05, re-sized with the incretin effect (see K_incretin):
    # glucose-stimulated insulin release is now amplified up to 3.5x by ABOVE-BASAL
    # GLP-1 rather than 1.4x by basal GLP-1, so the same ~55 uU/mL peak needs less
    # gamma.
    gamma: float = 0.05
    # --- Iter 97: SECOND-PHASE INSULIN SECRETION ------------------------------
    # Bergman's insulin equation is dI = -n(I - Ib) + gamma (G - h)+ * t: secretion
    # grows with time above threshold (the second phase). The teacher had dropped
    # the t, so insulin could only track instantaneous glucose and fell to ~17
    # uU/mL by 90-240 min of an OGTT against 30-45 in the literature (the
    # `ogtt_75g_insulin_mean_3h` anchor). Bergman's literal `t` integrates without
    # bound (measured here: insulin 100+, glucose 60-70 at 180 min -- reactive
    # hypoglycaemia by construction), which is why Toffolo & Cobelli (1980; 2001)
    # replaced it with a DELAYED PROPORTIONAL static component:
    #     dPot = -k_pot * (Pot - (G - h)+),
    #     secretion = gamma * ((1 - w) * (G - h)+ + w * Pot) * incretin.
    # Steady-state secretion is gamma * (G - h) either way; the response is spread
    # over tau = 1/k_pot = 40 min, so insulin persists while glucose is already
    # falling. w = 0.3: measured, it lifts 120-min OGTT insulin from 27 to 33 uU/mL
    # with a 3-4 h dip to ~93 mg/dL (the mild late undershoot real OGTTs show);
    # 0.5 deepens the dip to ~90 for another +3 uU/mL.
    ins_phase2_frac: float = 0.3
    k_pot: float = 0.025
    # Iter 97: obligatory glucose uptake per mg/dL of glucose space (brain, blood
    # cells, renal medulla -- the insulin-independent disposal). A POPULATION
    # constant: it is what makes the prolonged-fast floor absolute (item 3.3), since
    # at the floor EGP is gluconeogenesis alone and G_floor = GNG / uptake_ii
    # regardless of the patient's fed setpoint. The value makes the typical patient
    # (Gb 95) rest at the typical EGP of 2.0 mg/kg/min: 2.0 / (1.85 * 95).
    uptake_ii: float = 2.0 / (VG_DL_PER_KG * 95.0)
    # Fraction of basal disposal that sub-basal insulin can withdraw. Bergman's
    # insulin action X is signed (it was rectified here in iter 80); brain uptake
    # is not insulin-dependent, so the withdrawable part is small. 0.05: tracer
    # studies find disposal per mg/dL at 22 h of fasting within a few percent of the
    # post-absorptive value (Rd 1.8 at 82 mg/dL vs 2.0 at 92; Landau 1996), so the
    # withdrawable share is near zero; measured here, 0.15 held the 24 h fasting
    # glucose at 86 against Cahill's 70-80 by withdrawing 14% of uptake.
    ins_dep_basal_frac: float = 0.05
    # Hepatic autoregulation exponent: hyperglycaemia suppresses hepatic output as
    # (Gb/G)^m ABOVE the setpoint (glucokinase sensing; ~50% suppression at +100
    # mg/dL with basal insulin). Below the setpoint counter-regulation is hormonal
    # (glucagon, cortisol -- both explicit gates), not autoregulatory, so the term
    # is one-sided by physiology rather than by rectifier.
    hep_autoreg_m: float = 0.6
    # Iter 96: `h` -- the glucose threshold above which glucose-stimulated insulin
    # release engages -- is now DERIVED per patient in resolve_derived_params as
    # Gb * h_frac, not held at the population 95. Iter 21 set 80 -> 95 precisely to
    # "zero out GSIR at fasting glucose"; that intent is correct and was expressed
    # against ONE patient's fasting glucose while `Gb` is sampled over 54-126.
    # MEASURED on 12 randomized patients: 5 of 12 have Gb > 95, so their GSIR never
    # switches off and fasted insulin runs 27-53 uU/mL against the 7 +/- 4 anchor
    # (`extended_fast_insulin_basal`); corr(Gb, fasted insulin) = +0.835 and the
    # standing GSIR contribution averages 7.0 uU/mL -- the whole anchor's budget.
    # Same family as the iter-95 frame error: a rate law referenced to a POPULATION
    # constant where the physiology references the INDIVIDUAL's set point. In the
    # Bergman minimal model h is a per-subject fitted parameter, not a constant.
    h: float = 95.0
    h_frac: float = 1.0

    # --- Iter 93 -> 97: THE FASTED STATE ---------------------------------------
    # Iter 93 found the fasted state never engaged (48 h fast: glucose 97 -> 100,
    # insulin 16 at 24 h) because Gb was a hard attractor, and fixed it by making
    # the attractor itself fall with liver glycogen (`fast_gb_drop`, `fast_gb_floor
    # _frac`). That reproduced the numbers but not the mechanism: the fall was
    # PROPORTIONAL to Gb (a Gb-130 patient fasted to 97, a Gb-70 patient to 56 --
    # item 3.3) and the glycogen pool still never touched blood glucose.
    #
    # Iter 97 removes both parameters. The fasting fall is now a flux deficit:
    # glycogenolysis is first-order in the liver pool, so as the pool empties EGP
    # falls toward gluconeogenesis alone and glucose settles where obligatory
    # uptake balances it -- an ABSOLUTE floor of ~65-70 mg/dL (Cahill 2006) for
    # every patient, because `uptake_ii` and `Gng_b` are population-level while
    # only the glycogenolytic share of EGP scales with Gb.
    #
    # Insulin's fall in fasting is far steeper than the glucose fall that
    # drives it (Polonsky 1988: basal insulin roughly halves while glucose
    # drops ~15%) because beta-cell secretion is sigmoid in glucose near
    # threshold. A linear ratio cannot express that; the exponent can.
    # Identity whenever G >= Gb, so the fed state is untouched.
    fast_ins_exp: float = 5.0
    fast_ins_floor: float = 0.25   # basal secretion never falls below this fraction of Ib

    # Glucagon
    Gnb: float = 70.0
    k_gn: float = 0.03
    # Iter 97 follow-up: 1.5 -> 3.5. The alpha cell's response to glucose falling
    # below the setpoint is what carries the EARLY fasting rise (8-16 h, before
    # insulin has fallen much): Marliss 1970 108 -> 158 pg/mL over 3 days (+46%),
    # Aguilar-Parada 1969 +50% by 48-72 h. Measured after: +0.5 pg/mL/h over
    # 8-16 h, +19% at 24 h, +43% at 48 h. Zero above Gb, so postprandial
    # suppression (the OGTT anchors) is untouched.
    alpha_gn: float = 3.5

    # Free fatty acids. Iter 93: FFA was very nearly INERT — measured across a
    # 48 h fast it moved 0.50 -> 0.44 mmol/L (literature: a 2-3x RISE, Cahill
    # 2006), and postprandially it fell only -41% against Frayn's -60 to -80%.
    # Both ends are the same defect: IC50_lip = 15 uU/mL made lipolysis far too
    # insulin-INSENSITIVE across the physiological insulin range. Adipose
    # antilipolysis is the most insulin-sensitive action in the body — half-
    # maximal near 5-10 uU/mL, well below the ~50 uU/mL half-max for glucose
    # disposal (Nurjhan 1986; Jensen 1989).
    #
    # IC50_lip 15 -> 5 with lip_max 0.033 -> 0.06 holds the ANCHORED observable
    # exactly: basal FFA at I = Ib is (lip_max/(1+Ib/IC50))/k_ffa = 0.5 mmol/L
    # both before and after. Only the RESPONSE changes — same discipline as
    # iter 92's CARB_APPEARANCE_GAIN, which held the glucose excursion while
    # the kernel sharpened.
    # k_ffa 0.04 -> 0.20 is a SPEED change, not a level change: FFA relaxes to
    # lip_max/(1+I/IC50)/k_ffa, and lip_max is solved from FFA_b *including*
    # k_ffa (see randomize_params), so every equilibrium is identical and only
    # the time constant moves — 25 min -> 5 min. The old tau was the reason
    # postprandial suppression measured only -41%: FFA never reached the
    # equilibrium its insulin level implied before insulin came back down.
    # Plasma FFA turnover is genuinely fast, t1/2 ~2-4 min (Eaton 1969), so
    # 25 min was never defensible; 5 min is still conservative.
    FFA_b: float = 0.5
    lip_max: float = 0.30
    IC50_lip: float = 5.0
    k_ffa: float = 0.20

    # Beta-hydroxybutyrate. Iter 21 recalibration: k_bhb 0.03 -> 0.005
    # (much slower BHB clearance) + IC50_keto 10 -> 15 (less insulin
    # suppression of ketogenesis) so overnight fasting BHB rises to
    # ~0.25 mmol/L (matches the cohort target of +0.2 over baseline).
    BHB_b: float = 0.1
    keto_max: float = 0.005
    IC50_keto: float = 15.0
    # Iter 97 follow-up: DERIVED in resolve_derived_params so that BHB_b is the
    # fixed point of the fed equations: k_bhb = keto_max * FFA_b / (1 + Ib/IC50)
    # / BHB_b. At the old 0.005 the fed state produced +0.001 mmol/L/min and the
    # "basal" BHB actually rested at 0.3 (typical 0.1). The derived value, 0.015
    # /min, is also the physiological one: ketone-body MCR is ~20 mL/kg/min at
    # low concentration (Balasse & Fery 1989), i.e. tau ~1 h, not 200 min.
    k_bhb: float = 0.015           # default = derived value
    # Iter 80: hepatic ketogenesis ramps as the liver glycogen pool empties —
    # the prolonged-fast fuel switch (Cahill 2006: BHB ~1-2 mM by 24 h fast,
    # the teacher previously plateaued ~0.25). Gated on liver-glycogen
    # depletion (1 − glyco_avail), so it is exactly 0 at the fed calibration
    # state (conservation-exact) and rises only as the liver empties. This is
    # the marker that *actually moves* in fasting, so it gives the slow
    # glycogen pool a strong, observable gradient — what the glucose-side
    # coupling alone could not (glucose is homeostatically defended).
    # Iter 93: 22.0 -> 5.0. This gain was set in iter 80 to force the fasting
    # ketosis ramp at a time when FFA — ketogenesis' actual SUBSTRATE — was
    # frozen (it moved 0.50 -> 0.44 across a 48 h fast, so the FFA factor in
    # `ketogenesis` contributed nothing and the glycogen gain had to carry the
    # whole fuel switch by itself). That was a compensation for a defect, and
    # with FFA now mobilizing properly it double-counts: BHB at 24 h overshot
    # to 3.66 mmol/L against Cahill's 0.8-2.2. Retuned so BHB lands 0.55 / 1.32
    # / 3.51 mmol/L at 12 / 24 / 48 h, and it now rises for the RIGHT reason —
    # substrate supply — rather than being driven open-loop by the glycogen pool.
    # Iter 97 follow-up: 5 -> 13, and the gate now reads LINEAR pool depletion
    # (1 - LGly/LGly_b)+ rather than the saturating `glyco_avail` ratio, which the
    # iter-93 comment already noted "stays near 1 while the pool halves". With
    # BHB_b a true fixed point (clearance derived, 3x faster than before) the
    # fed-state BHB is 0.1 rather than 0.3, and the fuel switch has to come from
    # the pool running down: measured ~0.75 at 24 h and ~2.2 at 48 h after a
    # dinner, ~2.6 at 24 h when the fast starts from a full liver at dawn
    # (Cahill 0.8-2.2 at 24 h, 2-3 at 48 h).
    keto_glyc_gain: float = 13.0

    # --- Hepatobiliary / enterohepatic circulation (iter 95) ---------------------
    # Sourced anchors and the design rationale: docs/iter95-biliary-anchors.md.
    # Entered at the MEDIATOR (bile acids) rather than the damage readouts
    # (ALP/GGT/ALT/bilirubin), because those have no driver in this state vector and
    # would ship inert — and the iter-94 ruler finding is that the gate cannot tell a
    # constant from a simulator.
    #
    # CCK. Duodenal I-cells respond to intraluminal FAT and PROTEIN (not carbohydrate),
    # so the drive reads Ra_fat and Ra_protein. Literature: fasting 0.8-1.2 pmol/L,
    # peak 6.5-7.1 within ~10 min of a mixed meal, ~3.5 at 30 min, elevated 3-5 h.
    # k_cck = 1/tau; tau ~ 8 min gives the fast rise and the fall to ~3.5 by 30 min.
    CCK_b: float = 1.0            # pmol/L, fasting
    # CCK is cleared fast — plasma half-life ~1-3 min — so it tracks duodenal delivery
    # closely rather than smoothing it. tau = 1/k_cck = 2.9 min.
    k_cck: float = 0.35           # /min
    # Gains are per (g/min) of DUODENAL delivery, not of systemic appearance.
    # Calibrated (not guessed) against the three QUANTITATIVE CCK anchors at once:
    # peak 6.5-7.1 pmol/L, peak time 5-20 min, and ~2.5-4.5 pmol/L at +30 min.
    # The "elevated for 3-5 h" statement in the same sources is qualitative, so it is
    # NOT fitted to: the model realizes +17% over basal at +3 h, which is in the right
    # direction but is not evidence of anything and must not be quoted as a match.
    cck_fat_gain: float = 2.25    # fat is the strongest I-cell stimulus
    cck_prot_gain: float = 0.79   # protein is weaker (0.35x)
    # Gastric emptying, two-component (see _duodenal_delivery).
    duo_fast_rate: float = 0.10          # /min, gamma peaking at 10 min (liquid phase)
    duo_slow_rate: float = 0.005         # /min, tau 200 min tail (solid phase)
    duo_slow_frac: float = 0.35          # share of the meal on the slow route
    # Gallbladder. A POOL in mmol of bile acids (content, not volume — the gallbladder
    # concentrates bile ~10x, so content is what conserves round the loop). Emptying is
    # CCK-gated and PROPORTIONAL TO CONTENT, which makes it exponential: that is what
    # produces the observed biphasic "early rapid, late slow" shape without a second
    # mechanism, and it is why a second meal 90 min later empties far less.
    GB_b: float = 4.0             # mmol, fasting (fills between meals)
    GB_max: float = 6.0           # mmol, capacity
    k_gb_eject: float = 0.030     # /min at saturating CCK -> ~35-40% ejected by 60 min
    K_cck_gb: float = 2.5         # pmol/L above basal for half-maximal ejection
    # --- Iter 97: THE ENTEROHEPATIC LOOP IS CLOSED ------------------------------
    # Through iter 96 the gallbladder refilled from an infinite source
    # (`k_gb_fill * (GB_max - GB)`) and the 90% of portal return the liver
    # extracted vanished; the pools therefore rested at the CAP (gallbladder 5.97 =
    # GB_max against a declared 4.0) and the intestine at 0.08 against 1.0. Now
    # hepatic bile secretion = extracted portal return + cleared serum bile acids +
    # de-novo synthesis, and it is DIVIDED between the gallbladder (a fraction
    # `gb_divert_frac`, tapering to zero as the gallbladder fills -- bile bypasses a
    # full gallbladder to the duodenum) and the duodenum directly. Interdigestive
    # emptying `k_gb_basal` (the MMC-related partial emptying, 10-30% per ~100 min
    # cycle) is DERIVED so that GB_b is the fasting fixed point; synthesis is
    # derived to replace exactly the faecal loss at INT_b; the serum spill gain is
    # derived so BA_b is the fixed point of a pure mass-action serum pool.
    # Interdigestively ~70-80% of hepatic bile is diverted into the gallbladder
    # (the sphincter of Oddi is closed between MMC phases); the remainder flows to
    # the duodenum. 0.8 refills a 40%-emptied gallbladder in ~2-3 h, as measured by
    # ultrasound (Howard 1991), now that MMC emptying is suppressed while fed.
    gb_divert_frac: float = 0.8   # share of hepatic bile diverted to the gallbladder interdigestively
    # Fed-state suppression of interdigestive (MMC) emptying: duodenal nutrient
    # delivery (g/min) at which the MMC is half-suppressed. The MMC does not occur
    # in the fed state (Vantrappen 1977); a 30 g course over an hour (~0.5 g/min)
    # suppresses it ~90%.
    K_mmc_fed: float = 0.05
    gb_fill_width: float = 2.0    # mmol below GB_max over which diversion tapers to zero
    k_gb_basal: float = 0.0026    # /min, DERIVED (fasting fixed point at GB_b); default = derived value
    # Intestine. Carries the transit delay AND the 95%/5% split, so the CCK-peak-at-10min
    # to serum-peak-at-75-120min gap is a CONSEQUENCE of transport in series rather than
    # a fitted lag. k_ileal = 1/tau of transit-to-ileal-uptake.
    INT_b: float = 1.0            # mmol resident in the gut lumen at rest
    # Calibrated so serum bile acids peak MID-band (75-120 min), not on its edge:
    # tau 50 min put the peak at 73 min, two minutes outside. tau ~77 min lands it at
    # 83. This one constant sets the peak time, so it is the honest place to calibrate
    # — everything upstream is already pinned by the CCK and ejection-fraction anchors.
    k_ileal: float = 0.013        # /min (tau ~77 min) transit to ileal uptake
    f_ileal: float = 0.95         # fraction reabsorbed; 5% lost to faeces
    # Hepatic first-pass. THE CHOLESTASIS SITE. `k_canalicular` is the export capacity of
    # the hepatocyte->bile step; extraction saturates against it, so when it falls, more
    # of the portal return spills into serum and serum bile acids rise. That is the
    # clinical picture falling out of the mass balance, and it is the whole reason the
    # step is explicit rather than lumped. Healthy value = generous (extraction ~90%).
    k_canalicular: float = 1.0    # relative export capacity; 1.0 = healthy
    hep_extraction: float = 0.90  # fraction of portal return cleared on first pass
    # Serum. Fasting reference 4.4-14.1 umol/L; postprandial 4.7-20.2, peak 75-120 min.
    BA_b: float = 6.0             # umol/L
    k_ba: float = 0.030           # /min systemic clearance (tau ~ 33 min)
    # umol/L per mmol/min of unextracted portal return. Iter 97: DERIVED as
    # k_ba * BA_b / spillover_b, i.e. the serum pool is mass action with BA_b as its
    # fixed point (it used to relax to BA_b with a non-negative source, the
    # absorbing-floor pattern). Its reciprocal is an effective volume: 1000 /
    # gain ~ 7 L, plasma plus the albumin-bound interstitial share.
    ba_spill_gain: float = 145.7   # default = derived value
    # Hepatic de-novo synthesis, mmol/min. Iter 97: DERIVED as the faecal loss at
    # the fixed point, (1 - f_ileal) * k_ileal * INT_b = 0.00065 -> 0.94 mmol/day
    # ~ 0.4-0.5 g/day (literature 0.2-0.6 g/day).
    k_ba_synth: float = 0.00065   # default = derived value

    # Lactate. Iter 93: the drive was LINEAR in activity (`act * 0.3`), which
    # put a *moderate* 0.65 bout at 9.8 mmol/L — near-maximal, anaerobic
    # territory — against Brooks (1986) / Wasserman's 2-4 mmol/L for moderate
    # steady state. Blood lactate is not linear in intensity: it stays near
    # rest until the lactate threshold (~55-65% VO2max), then climbs steeply.
    # A thresholded quadratic reproduces both anchors — 2.4 mmol/L at act=0.65
    # and 11.6 at act=1.0 (maximal effort, literature 10-12) — where no linear
    # gain could satisfy both at once.
    Lac_b: float = 1.0
    k_lac: float = 0.02
    lac_thresh: float = 0.45     # activity fraction at the lactate threshold
    lac_act_gain: float = 0.7    # supra-threshold production, quadratic in excess

    # --- Iter 97: HEPATIC GLUCOSE OUTPUT, ONE FLUX ON TWO LEDGERS --------------
    # `hepatic_output` is endogenous glucose production in mg/kg/min (its declared
    # unit and typical, 2.0, are exactly the textbook basal EGP -- DeFronzo; Rothman
    # 1991). It is the SUM of glycogenolysis and released gluconeogenesis, and the
    # glycogenolytic part is the same number that leaves the liver-glycogen pool.
    #
    # Hep_b is DERIVED in resolve_derived_params as uptake_ii * Gb * VG_DL_PER_KG:
    # the EGP that holds this patient's declared fasting glucose. That makes fasting
    # hyperglycaemia an EGP excess (Gb 130 -> 2.7 mg/kg/min), which is what it is
    # clinically (DeFronzo 1989: the fasting glucose of type-2 diabetes correlates
    # with EGP, not with disposal).
    Hep_b: float = 2.0
    # Basal gluconeogenesis, mg/kg/min. Landau 1996 (JCI): 47% of EGP at 14 h of
    # fasting, 67% at 22 h, 93% at 42 h; Rothman 1991 (Science): 64% at 22 h. 1.2
    # is 60% of the typical EGP -- the post-absorptive share -- and it is
    # POPULATION-level (varied only mildly), because it is what sets the absolute
    # prolonged-fast glucose floor. Glycogenolysis at basal = Hep_b - Gng_b.
    # 1.0 = 50% of the typical EGP (Landau's 47% at 14 h). MEASURED: at 1.2 the
    # 48 h floor landed at 84 mg/dL because the glucagon/FFA/insulin gates lift GNG
    # ~25% in the fast; at 1.0 it lands at ~67 (Cahill 65-70).
    Gng_b: float = 1.0
    # Hepatic response lag to its hormonal drive (tau 25 min: hepatic insulin
    # action on glycogenolysis, Cherrington 1999). `hepatic_output` relaxes to the
    # instantaneous target; BOTH ledgers use the lagged state, split in the same
    # proportion, so carbon is closed at every step and not just on average.
    k_hep: float = 0.04
    # Glycogenolysis insulin gate: the IC50 form NORMALIZED AT BASAL,
    #     g = (1 + (Ib/K)^n) / (1 + (I/K)^n),
    # equal to 1 at I = Ib, falling to ~0.1 at 5x basal insulin and RISING to
    # 1 + (Ib/K)^n (1.44) as insulin falls toward zero. Sub-basal insulin therefore
    # accelerates glycogenolysis instead of being invisible to it (item 3.2): the
    # overnight fall in insulin is precisely the liver's cue to keep glucose up.
    # K 25 uU/mL, n 2: glycogenolysis is half-suppressed at ~+15 uU/mL above basal
    # (Rizza 1981 dose-response) and ~85% at +40; measured, the standard meal
    # takes EGP to a nadir of ~0.3 mg/kg/min (85% suppression; literature 70-90%).
    glyc_ins_K: float = 25.0
    glyc_ins_n: float = 2.0
    # Gluconeogenesis is far less insulin-sensitive than glycogenolysis (its
    # suppression runs through substrate supply and glucagon; Gastaldelli 2001):
    # first order, half-effect at +70 above basal; the same form's sub-basal boost
    # saturates at 1 + Ib/K = 1.125, so the insulin of a deep fast cannot lift GNG
    # by more than that (measured: at K 40 it added 22% and the floor sat at 76).
    gng_ins_K: float = 80.0
    gng_ins_n: float = 1.0
    # Share of the gluconeogenic flux diverted into glycogen (the indirect pathway)
    # as a function of insulin drive: ins_drive^0.25, so the modest insulin of the
    # absorptive tail (+2..+10 uU/mL) already diverts 60-80% -- the fed liver is a
    # net glucose consumer and routes its gluconeogenic carbon to glycogen (Katz &
    # McGarry 1984). Taylor 1996: the indirect pathway supplies roughly as much
    # postprandial hepatic glycogen as the direct one.
    gng_divert_exp: float = 0.25
    # Glucagon on hepatic output: a Hill centred at the patient's basal glucagon,
    #     g = 2 / (1 + (Gnb/Gn)^n)  -- 1 at basal, -> 2 with rising glucagon,
    # -> 0 as it falls. Glucagon acts fully on glycogenolysis and as sqrt on
    # gluconeogenesis (Cherrington: the acute glucagon response is glycogenolytic;
    # its gluconeogenic effect is slower and weaker).
    hgo_gn_n: float = 2.0
    # Cortisol on gluconeogenesis, SIGNED and saturating:
    #     g = 1 + a * tanh(ln(Cort / Cort_b)),
    # so the 14 h/day cortisol spends below its reference lower GNG (item 3.2),
    # and the morning peak raises it. a = 0.2: the nadir (4 ug/dL) takes GNG to
    # 0.84x and the peak (19) to 1.08x -- cortisol's whole diurnal swing moves EGP
    # by ~+/-8%. Dinneen 1993: cortisol held at ~35 ug/dL for 5 h raised EGP 14%;
    # Bolli 1984: the healthy dawn rise in glucose is +2-5 mg/dL.
    gng_cort_amp: float = 0.2
    # Gluconeogenic substrate: glycerol from lipolysis tracks FFA. (FFA/FFA_b)^0.2
    # takes the 48-h doubling of FFA to +15% GNG; with glucagon's +7% and the
    # weak insulin gate's +10% that is Landau's +25-30% rise in absolute
    # gluconeogenesis over a 42 h fast.
    gng_ffa_exp: float = 0.2
    # LEGACY FIELDS -- no longer read by simulate_full_body. Retained because
    # `pulse.training.insulin_sweep_signal._cold_metabolic_rates` still carries a
    # copy of the iter-80 hepatic equations and reads them by name; that copy is
    # outside the teacher's file ownership and is flagged for the student layer.
    cort_hep: float = 0.06
    ins_hep: float = 0.08
    gn_hep: float = 0.018
    hep_to_glucose: float = 0.038

    # Ghrelin
    Ghr_b: float = 100.0
    # k_ghr 0.02 => half-life ln2/k = 35 min, matching ghrelin's measured plasma
    # half-life (~30 min). Iter 92 checked whether raising it would pull the
    # postprandial nadir (112 min) into the literature's 60-90 min window -- it does
    # (k=0.035 gives 89 min), but only by making the half-life 20 min, i.e. by
    # contradicting a directly measured constant to fix a downstream timing symptom.
    # NOT changed. The residual nadir lag reflects how long nutrient appearance stays
    # elevated (the absorption kernel, already advanced this iteration), not ghrelin
    # kinetics.
    k_ghr: float = 0.02
    # Iter 97: insulin's action on ghrelin is a SIGNED saturating function centred
    # at basal insulin (see the appetite block); the old `(I - Ib)+ / (.. + IC50)`
    # rectifier made 24/36/48 h of fasting produce ghrelin = 100.000 exactly. The
    # exponent sets the curvature: n = 1 reproduces the postprandial suppression at
    # the old IC50 (half-suppression at 3x basal insulin) and gives +20% at the
    # insulin of a 24 h fast (Espelund 2005: +15-30%).
    ghr_ins_n: float = 1.0
    # Half-saturation of the nutrient suppressor, now g/min of DUODENAL delivery
    # (see the appetite block). A 75/5/10 standard meal delivers ~2.2 g/min at its
    # 10-min peak, so 0.9 gives ~70% suppression drive at the peak.
    K_meal_ghr: float = 0.9
    # --- Iter 97 follow-up: THE ANTICIPATORY PRE-MEAL RISE --------------------
    # Ghrelin rises before HABITUAL meal times independently of the previous
    # meal's suppression wearing off: Natalucci 2005 (Eur J Endocrinol 152:845)
    # saw the meal-locked pattern persist through a 33 h fast; Cummings 2001
    # (Diabetes 50:1714) measured +78% from the post-meal nadir over the 1-2 h
    # before a habitual meal; Frecka & Mattes 2008 showed it entrains to feeding
    # schedule. Basal production is multiplied by 1 + a * ramp(t), a half-cosine
    # ramp over the `ghr_antic_ramp_h` hours before each habitual meal hour that
    # decays over the hour after it (the meal itself then suppresses). a = 0.55
    # gives +21 pg/mL over the pre-meal hour and ~+30% at the habitual hour when
    # no meal comes -- Natalucci's fasting-day amplitude.
    ghr_antic_amp: float = 0.55
    ghr_antic_ramp_h: float = 2.0
    ghr_antic_decay_h: float = 1.0
    habitual_meal_hours: tuple[float, ...] = (9.0, 13.0, 20.0)
    # Iter 92: cap on the fraction of basal ghrelin production a meal can suppress.
    # Cummings (2001) puts the postprandial nadir 30-50% below fasting; without a cap
    # the suppressors drove production to ~6% of basal (nadir -78%). See the appetite
    # block for why the two suppressors are unioned rather than multiplied.
    # 0.60 measured (N=30): population nadir mean -41%, centring Cummings' -30..-50% band.
    ghr_supp_max: float = 0.60

    # Leptin
    Lep_b: float = 10.0
    # Iter 96: 0.001 -> 0.025. Leptin is driven by a 24 h circadian target peaking
    # at 02:00, and a first-order filter with k = 0.001 (tau = 1000 min) applied to
    # a 24 h sinusoid has a phase lag of atan(omega/k)/omega = 5.14 h and retains
    # only 1/sqrt(1+(omega/k)^2) = 23 % of the amplitude. MEASURED, the teacher's
    # leptin peaked at 07:00 against a 02:00 target with a realized range of 0.94
    # against a +/-2.0 target -- which is the whole of why leptin has read as
    # "inert" for many iterations, and why `leptin_nocturnal_peak` reported a
    # standing violation. Plasma leptin's half-life is ~25-30 min (Klein et al.
    # 1996, J Clin Invest 97:2152), i.e. k = ln2/27 = 0.026/min; 0.025 gives a
    # 0.6 h lag and 98 % amplitude retention. NOT fixed here: leptin still has no
    # meal coupling at all, so `leptin_fed_vs_fasted` is structurally 0.00 against
    # its +2 target -- that needs an insulin->leptin term (Saad et al. 1998), which
    # is a new mechanism rather than a rate constant. Left as an open item.
    k_lep: float = 0.025
    lep_circ_amp: float = 2.0
    # --- Iter 97: THE INSULIN -> LEPTIN COUPLING (the iter-96 open item) --------
    # Leptin tracks insulin over HOURS, not minutes: Saad 1998 (JCEM 83:453) -- a
    # day's leptin follows the day's insulin AUC with a ~4 h lag; Boden 1996 (JCEM
    # 81:3419) / Kolaczynski 1996 -- a 16-24 h fast lowers leptin 30-50% while the
    # circadian rhythm persists. The teacher had no meal coupling at all, so the
    # `leptin_fed_vs_fasted` anchor was structurally 0.00 against +2. Now a slow
    # insulin state (tau 4 h) moves the leptin target, SIGNED about basal insulin:
    #     target = Lep_b + circ + lep_ins_gain * (Ins_slow / Ib - 1).
    # A fed day (Ins_slow ~ 1.4 Ib) sits +1, a 16 h fast (~0.6 Ib) -1: the +2
    # difference Boden/Kolaczynski report, and a 48 h fast (~0.35 Ib) gives -16%,
    # the low end of the cited fall.
    lep_ins_gain: float = 2.5
    k_ins_slow: float = 1.0 / 240.0

    # GLP-1. Iter 21 recalibration: glp1_meal_gain 5.0 -> 1.5 to match
    # the large-meal GLP-1 peak of ~22 uU/mL (was overshooting to ~64
    # at the previous gain — z=4.76 vs the literature target).
    GLP1_b: float = 10.0
    k_glp1: float = 0.2
    # Iter 97: 1.5 -> 0.7. Per unit of carbohydrate appearance, which tripled when
    # the kernel became mass-conserving and then broadened (a lower, longer Ra);
    # the realized GLP-1 peak for 75 g stays in the cited 20-25 pmol/L.
    glp1_meal_gain: float = 0.7
    # --- Iter 97: THE INCRETIN EFFECT WAS 7% ------------------------------------
    # The incretin factor was `1 + GLP1 / (GLP1 + 15)` on ABSOLUTE GLP-1: basal
    # GLP-1 (10) already gave 1.40, the meal peak (26.5) 1.64, so removing the meal
    # rise changed the insulin AUC by 7%. Nauck 1986: the oral/IV isoglycaemic
    # insulin ratio is 2-3, i.e. 50-70% of the oral insulin response is incretin.
    # Same absolute-vs-above-basal shape iter 91 fixed for glucagon and ghrelin.
    # Now `1 + incretin_gain * g / (g + K_incretin)` with g = GLP-1 above basal:
    # 1.0 at basal, ~3.5 at the meal peak (GLP-1 +16 pmol/L). gamma was re-sized
    # alongside. MEASURED as the share of glucose-stimulated secretion carried by
    # the incretin factor along the standard-meal trajectory: 0.62 (Nauck 0.5-0.7);
    # at 2.5 it was 0.50.
    incretin_gain: float = 4.0
    K_incretin: float = 10.0

    # HPA: ACTH drives cortisol; cortisol feeds back on ACTH
    Cort_b: float = 12.0
    k_cort: float = 0.02
    cort_circ_amp: float = 5.0   # legacy (iter<=90 cortisol's own circadian; unused since iter 91)
    cort_gluco: float = 0.0012
    ACTH_b: float = 30.0
    k_acth: float = 0.04
    # Iter 91: 8.0 -> 18.0. ACTH now carries the ENTIRE HPA circadian (cortisol's separate
    # circadian target was removed -- it was double-driving the rhythm), so ACTH's own swing
    # must be deep enough to move cortisol across its real 4-5x range. Measured: ACTH 12-46
    # pg/mL (physiological 10-60), giving cortisol nadir 4.2 / peak 18.2 / ratio 4.35.
    acth_circ_amp: float = 18.0
    # --- Iter 97: THE HPA RHYTHM IS ASYMMETRIC AND SUPPRESSED ONCE --------------
    # Through iter 96 the drive was a symmetric cosine peaking at 07:30 (trough
    # 19:30) and the sleep suppression was applied to the ACTH target AND to the
    # cortisol target, so cort/ACTH was 0.26 asleep and 0.47 awake and the correct
    # nadir was two errors cancelling (item 3.9). Weitzman 1971 / Van Cauter 1996:
    # ACTH and cortisol are QUIESCENT from ~20:00 to ~02:00, rise steeply from
    # 02:00-03:00 to a peak at 07:00-08:00 that coincides with (and is amplified
    # by) awakening, then decline through the day -- ~60% of the peak by noon, ~25%
    # by 20:00. That is a half-cosine rise over ~5 h and an exponential fall with a
    # ~6 h time constant (a 19 h half-cosine, tried first, left cortisol cresting at
    # 09:40 because the drive was still at 97% two hours after its peak). ACTH_b
    # and acth_circ_amp now mean the MID-RANGE and half-range (peak ACTH_b + amp,
    # trough ACTH_b - amp); the 24 h MEAN ACTH is ~0.35 of the way up the range and
    # the 24 h mean cortisol ~9 ug/dL, which is the literature's mean (Weitzman:
    # nadir 2-4, peak 15-20, mean 7-10). Cort_b (12) stays the reference level
    # for the downstream couplings, as before.
    hpa_rise_start_h: float = 2.0
    hpa_peak_h: float = 6.5      # ACTH crests ~07:15 with the 25-min lag; cortisol ~08:00
    hpa_fall_tau_h: float = 6.0
    # Sleep (NREM) suppression of ACTH release, applied ONCE, to ACTH; cortisol
    # follows its secretagogue with a fixed ratio asleep and awake. 0.35 keeps the
    # nadir at 3-5 ug/dL and leaves a +6 awakening response when it is lifted.
    hpa_sleep_supp: float = 0.35
    k_acth_to_cort: float = 0.006   # legacy (iter<=90 additive ACTH->cortisol term; unused since iter 91)
    # Iter 91: cortisol's relaxation target is cort_per_acth * ACTH (see the HPA block). Set so
    # the ACTH rhythm carries cortisol across its physiological range; tuned by measurement below.
    # Iter 97: 0.42 -> 0.45 with the asymmetric drive (the mid-range ACTH is no
    # longer the mean): peak 17-18 ug/dL, nadir 3.7, 24 h mean ~9.5.
    cort_per_acth: float = 0.45
    cort_feedback_acth: float = 0.025
    hypo_acth: float = 0.025
    # Sympathetic / exercise-associated HPA drive on ACTH (activity in [0, 1])
    cort_activity: float = 0.35

    # Cardiovascular
    HR0: float = 70.0
    k_hr: float = 0.3
    # Iter 94 considered lowering this 5.0 -> 3.5 and REVERTED it. Recorded because the
    # reasoning is the trap, not the number: no cohort statistic constrains this term
    # (it cancels in `sleep_hr_dip`, whose arms share a clock window), so the only
    # justification available was a whole-day sleep-vs-wake contrast — which is not an
    # encoded, cited quantity at all. Measured properly, the change also does not do
    # what it was invoked for: whole-day contrast is 26.7 % at 5.0 and 24.6 % at 3.5,
    # both ABOVE the 10-20 % ambulatory band, so 3.5 bought nothing. (And the protocol
    # matters in the opposite direction to the intuition: adding realistic daytime
    # activity WIDENS the contrast, 24.1 % -> 26.7 %, because it lifts the daytime mean.)
    # The whole-day HR contrast running ~25 % is a real open discrepancy — see
    # docs/iter94-spec.md; it needs a cited contribution, not a tuned constant.
    # Iter 97: 5.0 -> 2.5. The cited contribution exists: under constant-routine
    # conditions (posture, activity, meals and sleep all held constant) the
    # ENDOGENOUS circadian amplitude of heart rate is 2-4 bpm (Krauchi & Wirz-
    # Justice 1994; Hu et al. 2004, trough ~05:00). The rest of the ambulatory
    # day/night swing is sleep, posture and activity, which this model carries
    # separately (`sleep_hr_frac`, `act_hr_gain`). At 5.0 the cosine alone made 10
    # bpm peak-to-trough and contributed +2.33 of the teacher's +2.64 bpm asleep
    # 03:00-06:00 rise against a real +1.6 -- the current gate blocker window.
    hr_circ_amp: float = 2.5
    HRV0: float = 40.0
    k_hrv: float = 0.1
    SBP0: float = 120.0
    DBP0: float = 80.0
    k_bp: float = 0.2
    # --- Iter 96: THE CORTISOL -> CARDIOVASCULAR GAINS WERE 7-13x TOO STRONG ---
    #
    # These three coefficients enter dHR/dHRV/dSBP as rate terms, so the level a
    # standing cortisol deviation buys is coefficient/k. The teacher realized
    #     hr 1.000 bpm, sbp 0.750 mmHg, hrv 2.000 ms   per ug/dL of cortisol
    # i.e. cortisol's own 4-18 ug/dL diurnal swing moved HR by 14 bpm on its own.
    #
    # The measurement (Adlan et al. 2018, J Physiol 596(20):4847-4861): 200 mg IV
    # hydrocortisone vs placebo, n=10 healthy males, 3 h post-dose ->
    #     HR +7 +/- 4 bpm, SBP +5 +/- 5 mmHg, rMSSD 84 +/- 38 -> 59 +/- 29 ms.
    # Serum cortisol: placebo 93.7 +/- 37.0 nmol/L; on hydrocortisone it exceeded
    # the assay ceiling in 7 of 10 (censored to 1400; the 3 measurable read
    # 2637 +/- 42). Using the CENSORED floor gives delta-cortisol 47.3 ug/dL --
    # deliberately the conservative choice, because a smaller denominator
    # OVERSTATES the gain. Even so:
    #     hr  0.148 bpm, sbp 0.106 mmHg, hrv -0.528 ms   per ug/dL
    # and against the actually-measured concentration those halve again. The
    # teacher was 6.8x the conservative gain and 13.2x the measured one.
    #
    # WHAT THIS FIXES. Measured on the 14 real overnight episodes with their own
    # Oura masks, the teacher's 03:00-06:00 HR rise decomposes as circadian
    # +2.33, sleep-shift +3.14, CORTISOL +6.84 bpm -- so cortisol alone carried
    # 56% of a +12.2 bpm dawn rise against a real +3.2. Re-gaining it is the
    # single largest correction available in that window, and it also relieves
    # `sleep_hr_dip`, which sat 1 sd too DEEP (-11.9 vs -8 +/- 4) precisely
    # because sleep's cortisol suppression was being amplified 7x on the way to HR.
    #
    # HONEST LIMIT: a 200 mg bolus is supraphysiological and a receptor-mediated
    # effect may well be steeper (not flatter) inside the 4-18 ug/dL range, so
    # this could under-gain. It is still the only direct human dose-response
    # measurement available, and 7x is far outside any plausible curvature.
    cort_hr: float = 0.045
    cort_hrv: float = 0.053
    cort_bp: float = 0.021
    # Iter 93: there was NO meal -> cardiovascular coupling at all. Measured,
    # a 75 g mixed meal and a fasted arm produced BIT-IDENTICAL HR, HRV, SBP
    # and DBP trajectories (max delta 0.0000 over 5 h) against a literature
    # postprandial HR rise of +5 to +10 bpm peaking at 30-60 min (Brunzell
    # 1971; Kearney 1995; Marfella 2000 — the same sources the cohort
    # statistic `postprandial_hr_rise` already cites, and which the teacher
    # was failing at z = -2.33). The mechanism is meal-induced sympathetic
    # activation plus splanchnic vasodilation, so it is driven by NUTRIENT
    # APPEARANCE (ra_norm), which times it to absorption rather than to the
    # meal clock. HRV needs no term of its own: it already tracks HR
    # inversely through `HRV0 * HR0 / HR`, so postprandial HRV suppression
    # falls out of the coupling that is already there (see the registered
    # hr -> hrv inverse prior, d8735a6).
    #
    # BP deliberately gets NO term. Postprandial BP in healthy adults is
    # roughly flat — splanchnic pooling offsets the cardiac-output rise
    # (postprandial HYPOtension is an autonomic-failure/elderly phenomenon,
    # not the healthy default), so adding one would assert an effect the
    # literature does not support at this population's age.
    # Iter 97: 4.0 -> 3.0. Measured on the 75 g standard meal the rise was +12.1
    # bpm against the cited +5-10 (Kearney 1995) and `postprandial_hr_rise` sat at
    # z = +1.21; 3.0 lands the same probe at ~+9.
    meal_hr_gain: float = 3.0
    # Half-saturation of nutrient appearance for the HR meal drive, mg/dL/min of
    # glucose space. Iter 97: 0.35 -> 1.06 with the mass-conserving kernel (x3.03),
    # shape unchanged.
    K_ra_norm: float = 1.06

    # Thermal
    T0: float = 37.0
    k_temp: float = 0.025
    # Iter 94: 0.45 -> 0.25. `circ_temp` is amp*cos, so peak-to-trough is 2*amp and
    # the sleep drop lands on top of it; 0.45 + 0.15 gave a 1.04 °C daily swing vs
    # the 0.5-0.8 °C of Czeisler (1999) / Refinetti & Menaker (1992), and
    # temp_circadian_nadir realized -1.02 against its -0.5 +/- 0.3 target.
    # Measured after the change: nadir -0.60 (z=-0.34), daily swing 0.61 °C.
    temp_circ_amp: float = 0.25
    temp_exercise_gain: float = 0.8
    # Iter 97: diet-induced thermogenesis is driven by the ENERGY absorbed, not by
    # a sum of three kernels in three different units (the carbohydrate kernel's
    # amplitude tripled this iteration and would have tripled DIT with it). Thermic
    # effects: protein 25%, carbohydrate 8%, fat 3% of ingested energy (Westerterp
    # 2004); gain in degC per (kcal/min of heat). Re-sized to hold the standard-meal
    # rise at ~+0.2 degC (literature +0.1-0.3).
    temp_dit_gain: float = 0.017
    sleep_temp_drop: float = 0.12

    # Sleep modulation
    # Iter 94: sleep_hr_frac 0.15 -> 0.06 and sleep_hrv_gain 1.3 -> 1.08. Measured on
    # the cohort arms these two are scored against (same clock window, awake vs
    # asleep, so the circadian term cancels), the teacher gave sleep_hr_dip -20.3
    # against -8 +/- 4 and hrv_sleep_rise +42.8 against +12 +/- 7.5. They interact:
    # HRV relaxes to HRV0*HR0/HR*gain, so shrinking the HR dip shrinks the HRV rise
    # through the inverse-coupling channel before `gain` is touched at all — which is
    # why `gain` moves much further than a naive ratio would suggest. Landing values
    # (N=12 randomized): sleep_hr_dip -11.9 (z=-0.99), hrv_sleep_rise +19.4 (z=+0.99).
    #
    # WHAT THIS NUMBER IS NOT. Setting sleep_hr_frac=0 still leaves a -6.9 bpm dip, so
    # only ~38 % of the realized sleep bradycardia is produced by this term; the rest
    # comes through sleep's suppression of cortisol and thence `cort_hr`. Both channels
    # are real physiology and NOTHING WE HAVE ENCODED DETERMINES THE SPLIT — lowering
    # `cort_hr` instead would satisfy the same target equally well. So 0.06 fixes the
    # TOTAL, which is the only thing the literature here constrains, and should not be
    # read as "the sleep bradycardia is 6 % of HR0". Separating the channels needs a
    # contribution that isolates them (a beta-blockade or cortisol-suppression arm),
    # not a further tuning pass. Recorded as open in docs/iter94-spec.md.
    #
    # sleep_bp_frac deliberately NOT changed — see sbp_sleep_dip in cohorts/breadth_floor.py.
    # Iter 96: 0.06 -> 0.09. The iter-94 note below asked for "a contribution that
    # isolates them" before this could be set honestly -- Adlan 2018 (see cort_hr)
    # IS that contribution. With cortisol's chronotropic gain fixed at its measured
    # value, the explicit sleep term is no longer double-counted, and `sleep_hr_dip`
    # falls to -5.2 bpm at 0.06. 0.09 restores it to -7.3 against the -8 +/- 4
    # target (z -0.99 -> +0.17, i.e. BETTER anchored than before), and leaves the
    # whole-day contrast at 23.7% -- see DAY_NIGHT_HR_CONTRAST in
    # cohorts/cardiovascular.py, which is the cited anchor iter-94 said was missing.
    sleep_hr_frac: float = 0.09
    sleep_hrv_gain: float = 1.08
    sleep_bp_frac: float = 0.12
    sleep_rr_drop: float = 3.0

    # Activity coupling
    act_hr_gain: float = 25.0
    act_sbp_gain: float = 8.0
    act_dbp_gain: float = -2.0
    act_insulin_sens: float = 0.3

    # Respiratory
    RR0: float = 15.0
    k_rr: float = 0.1
    SpO2_0: float = 98.0
    k_spo2: float = 0.5
    rr_lactate_gain: float = 0.4
    spo2_exercise_dip: float = 1.5

    # Meal absorption
    # Iter 92: fast-carb rate 0.03 -> 0.040. The gamma-2 kernel peaks at 1/rate, so
    # 0.03 put carb appearance at 33 min and the resulting glucose peak at 68 min --
    # late against the literature's 45-60 min for a mixed meal, and it dragged the
    # whole postprandial cascade with it (insulin peak 74 min vs 30-60, ghrelin nadir
    # 126 min vs 60-90). 0.040 measured (N=30): glucose peak 57 min, insulin 62 min.
    # Total carb appearance is unchanged -- the kernel's integral is rate-independent
    # (see CARB_APPEARANCE_GAIN, which holds the excursion amplitude fixed).
    # Iter 97: the kernel is now MASS-CONSERVING, so its shape has to be the shape
    # of a real absorption curve rather than a peak scaled to fit. Tracer studies of
    # a 75 g oral load (Dalla Man 2004/2005; Ferrannini 1985): Ra rises to 5-6
    # mg/kg/min by 30-45 min, PLATEAUS there through ~90 min, is still ~3 at 120
    # and ~1.5 at 180. A single gamma-2 cannot plateau. Two components: a fast
    # gamma-2 at 0.030 (peak 33 min) carrying 30% and a slow one at 0.012 (peak 83
    # min) carrying 70% reproduce that shape with the whole load.
    meal_absorption_fast_rate: float = 0.030
    meal_absorption_slow_rate: float = 0.012
    meal_absorption_slow_fraction: float = 0.70

    # Glycogen pools (iter 76 — Move D first concrete step). Liver and muscle
    # glycogen as flux integrators (dGly/dt = synthesis − breakdown), no longer
    # padded constants. Anatomically separate tissues with separate turnover
    # (the iter-56 split): the liver pool depletes overnight to defend blood
    # glucose (Cahill 2006 — ~60 g of the ~100 g pool gone by a ~24 h fast);
    # the muscle pool is rest-preserved (no hepatic G6Pase — muscle glycogen is
    # spent locally during activity, not released to blood: Coppack 1989) and
    # is the slow exercise-coupled reservoir that the chronic-exercise north
    # star (mitochondrial_capacity, deferred) will eventually ride on.
    LGly_b: float = 100.0       # liver glycogen typical / fed level (g)
    MGly_b: float = 400.0       # muscle glycogen typical / fed level (g)
    # Iter 96 CONSIDERED 110 -> 125 and REVERTED it; recorded because the reasoning
    # is the trap. With the width taper (glyc_fill_width_L) the eucaloric pool
    # equilibrates near LGly_max - 25 g, so 125 lands it at LGly_b = 100 g, which
    # looks like the right answer for "a eucaloric day is glycogen-neutral". But
    # the cap is not only a cap: raising it also lets each meal deposit more, and
    # MEASURED, the 10-16 h fasted arm went from 82.6 g to 97 g -- the pool stopped
    # emptying overnight, and four fasted-state anchors degraded together
    # (insulin_basal z +2.07 -> +3.09, bhb -0.93 -> -1.59, ffa -1.71 -> -1.91).
    # The per-meal deposition is pinned by Taylor 1996 (19% of meal carbohydrate)
    # and the 24 h fast level by Cahill/Rothman (~40 g); both are already satisfied
    # at 110. That a 175 g-carbohydrate day is then mildly glycogen-negative is not
    # obviously an error -- it is a low-carbohydrate day. What WAS an error was the
    # 9%-of-capacity throttle at the fed level, and the width taper fixes that
    # without touching the cap.
    # Iter 97: 110 -> 150. LGly_b is the TYPICAL level, not the maximum: the liver
    # holds up to ~120-150 g after large carbohydrate meals (Nilsson & Hultman
    # 1973). With the cap 10 g above typical and a 30 g taper, synthesis was
    # throttled to ~50% at the pool's own normal level -- the iter-96 defect
    # again, one notch higher -- and the eucaloric pool drained to 51 g once the
    # ledger was honest. The taper now closes over 120-150 g.
    LGly_max: float = 150.0     # liver storage cap (synthesis tapers as it fills)
    # --- Iter 97: GLYCOGEN SYNTHESIS IS DEBITED FROM GLUCOSE --------------------
    # Liver: the direct pathway takes a FRACTION of portal carbohydrate appearance
    # into glycogen under insulin drive: glyc_syn_frac_L * Ra_carb * ins_drive *
    # fill. Taylor 1996 (JCI, 13C-NMR): 19% of meal carbohydrate is in liver
    # glycogen by ~5 h. Net hepatic glucose uptake is driven by the PORTAL SIGNAL
    # (the portal-arterial glucose gradient) as much as by insulin -- Cherrington
    # 1999: the portal signal roughly doubles NHGU at a given insulin -- so the
    # drive is (0.5 + 0.5 * ins_drive): half of the fraction is taken from the
    # first minute of absorption, before insulin has risen. That front-loading is
    # what lets `Si` sit in its measured range while the peak holds. ins_drive
    # averages ~0.45 over the absorptive tail, so 0.30 * 0.72 lands ~21%. This flux
    # is subtracted from the glucose ledger: it IS hepatic first-pass uptake, which
    # is the physiological reason a 75 g load raises glucose by 60 and not by 180
    # mg/dL. The indirect pathway -- gluconeogenic carbon diverted into glycogen
    # while insulin is high (the "glucose paradox", Katz & McGarry 1984) -- is a
    # share of the GNG flux booked into glycogen instead of blood (gng_divert_exp).
    glyc_syn_frac_L: float = 0.30
    # Muscle: glycogen synthesis is PART of insulin-stimulated disposal, booked
    # here to a pool with its own ledger and debited from glucose so the two books
    # agree. g/min per (mg/dL/min of appearance) at full insulin drive and full
    # deficit. 0.08 gives ~0.25-0.4 g/min at peak absorption -- resynthesis of a
    # 150 g exercise deficit over ~10-20 h on a carbohydrate diet (Ivy 1988).
    k_glyc_syn_M: float = 0.08
    # Iter 97: `k_glyc_brk_L` is gone -- basal glycogenolysis is DERIVED as
    # Hep_b - Gng_b (mg/kg/min) and is first order in the pool (LGly / LGly_b), so
    # that as the pool empties EGP falls toward gluconeogenesis alone. Landau
    # 1996's fractions (glycogenolysis 1.06 -> 0.55 -> 0.1 mg/kg/min at 14 / 22 /
    # 42 h while the pool goes ~75 -> 30 -> 10 g) are close to proportional.
    k_glyc_brk_M: float = 3.5   # muscle glycogenolysis gain (activity-driven)
    # Liver saturation constant for the KETOGENESIS gate only (`glyco_avail`, the
    # fraction of the pool the liver can still mobilize); the glycogenolysis flux
    # itself no longer reads it.
    glyc_K_L: float = 35.0
    glyc_K_M: float = 150.0     # muscle depletion-saturation constant (g)
    # Iter 96: width (g) over which synthesis tapers off as the pool approaches
    # its cap. See the dLGly block -- `1 - LGly/LGly_max` throttled refill to 9%
    # of capacity exactly at the fed level, which made a eucaloric day
    # glycogen-NEGATIVE and gave the pool an implicit setpoint near 51 g.
    glyc_fill_width_L: float = 30.0
    # Iter 97: the MUSCLE taper closes at MGly_b, not at a cap 50 g above it.
    # Muscle has no glycogenolysis at rest (by construction, below), so any resting
    # synthesis accumulates until SOMETHING stops it -- and what stopped it was the
    # cap: the teacher's muscle glycogen rested at 449.9 g against a declared
    # typical of 400 (item 2.5). Glycogen synthase is allosterically inhibited by
    # glycogen content, so the replete level is a genuine fixed point: synthesis
    # runs only into a deficit below MGly_b. Supercompensation is not modelled.
    glyc_fill_width_M: float = 60.0
    act_rest_M: float = 0.10    # activity below this is rest — no muscle glycogenolysis (Coppack 1989)


def randomize_params(rng: np.random.Generator) -> PatientParams:
    """Sample a virtual patient.

    Iter 90 — CORRELATED POPULATION. Every parameter used to be drawn INDEPENDENTLY
    (`val * exp(N(0, sigma))`), so the "population" was a product of marginals: the teacher
    could emit a lean-athlete resting HR alongside a diabetic fasting glucose and a
    hypertensive blood pressure in the same "patient". Real physiology is strongly
    correlated, and the consequence was concrete: with only ~20 such patients defining a
    64-dim embedding space, the model baked the sample's *spurious* correlations into the
    embedding geometry. Measured on iter-89: resting HR correlated -0.55 with SBP and +0.41
    with HRV across prior-sampled patients — both WRONG-SIGNED (higher sympathetic tone
    raises SBP and lowers HRV). Calibrating one marker then dragged the others along a fake
    axis, and per-person recovery was worse than simply predicting the population mean.

    Parameters now load on two latent physiological axes:

      z_ir   insulin resistance / metabolic syndrome
             -> lower Si and Sg, higher Gb, Ib, FFA, leptin, SBP/DBP, resting HR;
                lower HRV and ghrelin.
      z_fit  cardiorespiratory fitness
             -> lower resting HR and RR, higher HRV, higher Si and muscle glycogen,
                greater activity-induced insulin sensitivity, modestly lower BP.

    Each parameter's log-deviation is `sigma * (a_ir*z_ir + a_fit*z_fit + sqrt(1 - a_ir^2 -
    a_fit^2) * eps)`. Because the loadings and the idiosyncratic term have unit total
    variance, the MARGINAL spread of every parameter is EXACTLY what it was before (the
    sigmas were tuned across iters 82-88 to cover the benchmark ranges, so they are
    preserved); only the joint structure changes. Loadings are signed from physiology, with
    magnitudes deliberately moderate so no parameter becomes a deterministic function of a
    latent.
    """
    p = PatientParams()

    # Latent physiological axes (standard normal, independent).
    z_ir = float(rng.normal())
    z_fit = float(rng.normal())

    def vary(val, spread=0.3, ir: float = 0.0, fit: float = 0.0):
        shared = ir * z_ir + fit * z_fit
        resid_var = 1.0 - ir * ir - fit * fit
        if resid_var < 0.0:
            raise ValueError(f"loadings exceed unit variance: ir={ir}, fit={fit}")
        z = shared + np.sqrt(resid_var) * rng.normal()
        return val * np.exp(spread * z)

    # Iter 97: Sg is derived (uptake_ii * (1 + hep_autoreg_m)); the insulin-resistance
    # loading it used to carry (Bergman: Sg and Si co-degrade) moves to the hepatic
    # autoregulation exponent, which is the part of glucose effectiveness that lives
    # in the liver. NB the ORDER and COUNT of `vary` calls below is preserved from
    # iter 96 (one draw here where Sg was; five in the hepatic block where
    # fast_gb_drop/Hep_b/k_hep/cort_hep/hep_to_glucose were) so that a seed still
    # samples the same patient: the anchor audit is compared across iterations at a
    # fixed seed, and a shifted stream would show up as a phantom HR change.
    p.hep_autoreg_m = float(np.clip(vary(p.hep_autoreg_m, 0.3, ir=-0.30), 0.2, 1.5))
    # Insulin sensitivity: the defining axis of z_ir; training raises it.
    p.Si = vary(p.Si, 0.5, ir=-0.70, fit=0.40)
    # Iter 82: widen fasting-glucose baseline diversity 0.15 -> 0.25. The
    # iter-81 population (sigma 0.15 ~ 82-110 mg/dL) was clinically too narrow:
    # the per-patient baseline (b_emb) only learned authority over the trained
    # range, so the model's achievable fasting glucose floored at ~88 and could
    # not represent low-baseline patients (benchmark spans 60-118; gate FAILED
    # glucose_mape 0.49). 0.25 spans ~59-171 mg/dL at +/-2 sigma (mild hypo to
    # diabetic-range fasting) -- realistic clinical diversity; teacher verified
    # sane across the range (0/40 unstable trajectories).
    # Iter 96: CLIPPED. The unclipped 0.25 spread reached Gb = 54 mg/dL, and that
    # patient's simulated fasted glucose settled at 42 mg/dL -- neuroglycopenic, not
    # a phenotype, and it was being distilled as one. Clinical fasting glucose spans
    # ~70 (low-normal) to ~130 (diabetic range); clip there and keep the spread.
    p.Gb = float(np.clip(vary(p.Gb, 0.25, ir=0.50), 70.0, 130.0))
    p.Ib = vary(p.Ib, 0.4, ir=0.60)           # compensatory hyperinsulinemia
    p.n = vary(p.n)
    p.gamma = vary(p.gamma)
    p.Gnb = vary(p.Gnb)
    p.FFA_b = vary(p.FFA_b, ir=0.35)          # impaired lipolysis suppression
    # Iter 93: make FFA_b the level the patient actually DEFENDS. FFA relaxes to
    # lipolysis/k_ffa, so with a global lip_max every patient converged to the
    # same ~0.5 mmol/L and FFA_b only set a transient initial condition — the
    # per-patient spread was decorative. Solving lip_max for FFA_b at basal
    # insulin makes fasting FFA equal FFA_b exactly, per patient, and restores
    # the physiological IR direction (FFA_b loads +0.35 on z_ir; without this,
    # sharpening IC50_lip would have made insulin-resistant patients converge to
    # LOWER FFA, the wrong sign — Boden 1997).
    p.lip_max = p.FFA_b * p.k_ffa * (1.0 + p.Ib / p.IC50_lip)
    p.BHB_b = vary(p.BHB_b)
    p.Lac_b = vary(p.Lac_b)
    # Fitter patients cross the lactate threshold later and clear lactate faster.
    p.lac_thresh = float(np.clip(vary(p.lac_thresh, 0.12, fit=0.45), 0.30, 0.65))
    p.lac_act_gain = vary(p.lac_act_gain, 0.2, fit=-0.30)
    # Iter 97: Hep_b is derived from Gb (resolve_derived_params); the fasted fall
    # is a flux deficit, so `fast_gb_drop` no longer exists. Five draws, as before.
    # Obligatory uptake per mg/dL varies little between people.
    p.uptake_ii = vary(p.uptake_ii, 0.06)
    # Basal gluconeogenesis: mildly higher with insulin resistance (Magnusson 1992:
    # the EGP excess of type-2 diabetes is largely gluconeogenic). Kept narrow because
    # it sets the ABSOLUTE prolonged-fast glucose floor.
    p.Gng_b = vary(p.Gng_b, 0.10, ir=0.40)
    p.k_hep = float(np.clip(vary(p.k_hep, 0.35), 0.02, 0.09))
    p.glyc_ins_K = float(np.clip(vary(p.glyc_ins_K, 0.25, ir=0.40), 12.0, 50.0))  # hepatic insulin resistance
    p.gng_cort_amp = float(np.clip(vary(p.gng_cort_amp, 0.3), 0.10, 0.45))
    p.Ghr_b = vary(p.Ghr_b, 0.3, ir=-0.25)    # ghrelin lower in obesity/IR
    p.Lep_b = vary(p.Lep_b, 0.5, ir=0.45)     # leptin tracks adiposity, co-travels with IR
    p.GLP1_b = vary(p.GLP1_b, 0.3)
    p.Cort_b = vary(p.Cort_b, 0.3)
    p.cort_circ_amp = vary(p.cort_circ_amp, 0.3)
    p.ACTH_b = vary(p.ACTH_b, 0.25)
    p.k_acth = float(np.clip(vary(p.k_acth, 0.3), 0.02, 0.08))
    p.acth_circ_amp = vary(p.acth_circ_amp, 0.35)
    p.k_acth_to_cort = float(np.clip(vary(p.k_acth_to_cort, 0.35), 0.002, 0.012))
    p.cort_feedback_acth = float(np.clip(vary(p.cort_feedback_acth, 0.35), 0.012, 0.045))
    p.hypo_acth = float(np.clip(vary(p.hypo_acth, 0.35), 0.012, 0.045))
    p.cort_activity = float(np.clip(vary(p.cort_activity, 0.35), 0.12, 0.8))
    # Iter 82: widen resting-vital baseline diversity to cover the benchmark /
    # clinical range (same flat-baseline gap as glucose — the model's achievable
    # resting HR floored at ~68 but the benchmark spans 49-77 bpm; hr_mape gate
    # FAILED at 0.18). HR is learned-dynamics (no hardcoded anchor), so a wider
    # training range alone should unlock the low end. HR0 0.15->0.22 (~46-111 bpm
    # at +/-2 sigma: athlete to tachycardic); SBP0/DBP0 0.10->0.14/0.15 to reach
    # the benchmark lows (sbp 94, dbp 61). Teacher verified sane across the range.
    p.HR0 = vary(p.HR0, 0.22, ir=0.20, fit=-0.60)   # trained athletes rest low
    p.HRV0 = vary(p.HRV0, 0.4, ir=-0.30, fit=0.55)  # vagal tone up with fitness, down with IR
    p.SBP0 = vary(p.SBP0, 0.14, ir=0.40, fit=-0.30)
    p.DBP0 = vary(p.DBP0, 0.15, ir=0.40, fit=-0.30)
    p.T0 = p.T0 + rng.normal(0, 0.2)
    # Iter 94: these four clip ranges moved WITH their defaults. Three of them
    # (temp_circ_amp, sleep_hr_frac, sleep_hrv_gain) would otherwise have clipped
    # every patient back UP to the old range and no patient would have received the
    # recalibrated value — the exact self-inflicted bug iter 93 caught. Ranges are
    # ~+/-2.5 sigma of the lognormal `vary` around each new default.
    p.temp_circ_amp = float(np.clip(vary(p.temp_circ_amp, 0.15), 0.16, 0.38))
    p.temp_exercise_gain = float(np.clip(vary(p.temp_exercise_gain, 0.2), 0.4, 1.1))
    p.sleep_temp_drop = float(np.clip(vary(p.sleep_temp_drop, 0.2), 0.05, 0.21))
    p.sleep_hr_frac = float(np.clip(vary(p.sleep_hr_frac, 0.2), 0.03, 0.11))
    # sleep_hrv_gain is a MULTIPLIER on the HRV setpoint, so the physiological
    # quantity is the excess over 1.0, not the gain itself. With the default now at
    # 1.08, varying the gain multiplicatively (as every other param is varied) would
    # push most patients below 1.0 — i.e. HRV *falling* during sleep, inverting an
    # effect the literature is unambiguous about. Vary the excess instead: the sign
    # is then correct by construction rather than by clipping.
    p.sleep_hrv_gain = 1.0 + float(np.clip(
        (p.sleep_hrv_gain - 1.0) * np.exp(0.30 * rng.normal()), 0.02, 0.20))
    p.sleep_bp_frac = float(np.clip(vary(p.sleep_bp_frac, 0.2), 0.06, 0.20))
    p.sleep_rr_drop = float(np.clip(vary(p.sleep_rr_drop, 0.2), 1.5, 5.0))
    p.act_hr_gain = vary(p.act_hr_gain, 0.15)
    p.meal_hr_gain = vary(p.meal_hr_gain, 0.25)
    p.act_insulin_sens = float(np.clip(vary(p.act_insulin_sens, 0.3, fit=0.35), 0.1, 0.6))
    p.rr_lactate_gain = vary(p.rr_lactate_gain, 0.2)
    p.RR0 = vary(p.RR0, 0.15, fit=-0.20)
    p.SpO2_0 = min(100, max(94, p.SpO2_0 + rng.normal(0, 1)))
    p.meal_absorption_fast_rate = float(np.clip(vary(p.meal_absorption_fast_rate, 0.2), 0.01, 0.08))
    p.meal_absorption_slow_rate = float(np.clip(vary(p.meal_absorption_slow_rate, 0.25), 0.004, 0.04))
    p.meal_absorption_slow_fraction = float(np.clip(vary(p.meal_absorption_slow_fraction, 0.2), 0.35, 0.8))
    # Glycogen pool sizes vary across patients (training/diet history); the
    # flux gains stay fixed so the dynamics shape is consistent. Caps track
    # the baselines so a larger pool can still fill. Cold-distill references
    # use PatientParams() defaults, so this only diversifies full_body episodes.
    p.LGly_b = float(np.clip(vary(p.LGly_b, 0.15), 70.0, 130.0))
    p.MGly_b = float(np.clip(vary(p.MGly_b, 0.15, fit=0.50), 300.0, 520.0))
    p.LGly_max = 1.5 * p.LGly_b
    return resolve_derived_params(p)


def resolve_derived_params(params: PatientParams) -> PatientParams:
    """Recompute parameters that are DERIVED from other parameters.

    Iter 93: ``lip_max`` is not free — it is solved so that FFA's equilibrium at
    basal insulin lands exactly on ``FFA_b`` (see randomize_params). Anything
    that mutates ``FFA_b``, ``Ib``, ``IC50_lip`` or ``k_ffa`` after the fact
    must call this, or the patient silently defends an FFA level nobody asked
    for. ``synthetic_users`` does exactly that — its ``insulin_resistant``
    profile overrides FFA_b 0.7 AND Ib 18 post-randomization, which without
    this would have left it defending a much LOWER FFA than the profile
    declares, i.e. the opposite of the phenotype it exists to represent.

    Iter 97 extends the same discipline to every pool: the glucose, liver-
    glycogen, gallbladder, intestinal-bile and serum-bile-acid fixed points are
    all SOLVED from the declared ``_b`` levels here, so a declared typical is the
    level the ODE actually rests at, by construction.

    Idempotent, so it is safe to call more than once.
    """
    params.lip_max = params.FFA_b * params.k_ffa * (1.0 + params.Ib / params.IC50_lip)
    # Iter 96: GSIR threshold tracks the patient's own defended fasting glucose.
    params.h = params.Gb * params.h_frac

    # --- Ketone fixed point (iter 97 follow-up): clearance from the declared basal ---
    params.k_bhb = (params.keto_max * params.FFA_b / (1.0 + params.Ib / params.IC50_keto)) / params.BHB_b

    # --- Glucose fixed point (iter 97) ---
    # Basal EGP is what holds the declared fasting glucose against obligatory
    # uptake; its glycogenolytic share is what is left after gluconeogenesis.
    params.Hep_b = params.uptake_ii * params.Gb * VG_DL_PER_KG
    params.Gng_b = float(min(params.Gng_b, 0.85 * params.Hep_b))   # glycogenolysis >= 15% of EGP
    # Glucose effectiveness = obligatory uptake + hepatic autoregulation, linearized.
    params.Sg = params.uptake_ii * (1.0 + params.hep_autoreg_m)

    # --- Enterohepatic fixed points (iter 97) ---
    # Ileal uptake at the fixed point is the whole recirculating flux.
    ileal_b = params.k_ileal * params.INT_b                                  # mmol/min
    params.k_ba_synth = (1.0 - params.f_ileal) * ileal_b                      # replaces faecal loss
    extraction_b = min(max(params.hep_extraction, 0.0), 0.995)
    spillover_b = (1.0 - extraction_b) * params.f_ileal * ileal_b             # mmol/min to serum
    params.ba_spill_gain = params.k_ba * params.BA_b / max(spillover_b, 1e-9)  # BA_b is the serum fixed point
    # Hepatic secretion at rest = extracted portal return + cleared serum + synthesis
    # = ileal_b exactly (conservation), split between gallbladder and duodenum.
    divert_b = params.gb_divert_frac * min(1.0, max(0.0, (params.GB_max - params.GB_b) / params.gb_fill_width))
    params.k_gb_basal = ileal_b * divert_b / params.GB_b                      # GB_b is the fasting fixed point
    return params


def _glyc_ins_gate(I: float, Ib: float, K: float, n: float) -> float:
    """IC50 suppression normalized at basal insulin: 1 at I = Ib, -> 0 at high
    insulin, -> 1 + (Ib/K)^n as insulin falls to zero (signed, saturating)."""
    return (1.0 + (Ib / K) ** n) / (1.0 + (I / K) ** n)


def _hill_centred(x: float, x_b: float, n: float) -> float:
    """2 / (1 + (x_b/x)^n): 1 at basal, -> 2 above, -> 0 below. Signed, saturating."""
    x = max(x, 1e-6)
    return 2.0 / (1.0 + (x_b / x) ** n)


def glucose_fluxes(
    params: PatientParams,
    G: float, I: float, X: float, Gn: float, Cort: float, FFA: float,
    LGly: float, MGly: float, Hep: float, Ra_carb: float, act: float,
) -> dict[str, float]:
    """Every glucose-carbon flux at one instant, in ONE set of units.

    Returns mg/dL/min of glucose space for the glucose ledger and g/min for the
    glycogen ledgers, plus the hepatic target in mg/kg/min. This is the single
    place the balance is written; ``simulate_full_body`` integrates it and the
    iter-97 tests re-evaluate it along a trajectory to check that carbon closes.

    Glucose ledger:   dG  = ra - syn_L - syn_M + egp - uptake_ii - uptake_id - uptake_ex
    Liver ledger:     dLGly = syn_L + gng_divert - glycogenolysis
    Muscle ledger:    dMGly = syn_M - brk_M   (brk_M is oxidized in situ, no G6Pase)
    Hepatic output:   Hep -> glycogenolysis_target + gng_released_target (lagged);
                      the realized glycogenolysis and released GNG are the lagged
                      state split in the target's proportion, so the flux that
                      leaves the pool is the flux that reaches blood.
    """
    Ib, Gb = params.Ib, params.Gb
    ins_excess = max(I - Ib, 0.0)
    ins_drive = ins_excess / (ins_excess + Ib)                      # 0 at basal -> 1

    # --- uptake (mg/dL/min) ---
    x_eff = max(X, -params.ins_dep_basal_frac * params.uptake_ii)  # signed insulin action, floored
    uptake_ii = params.uptake_ii * G
    uptake_id = x_eff * G
    uptake_ex = act * 0.02 * max(G - Gb * 0.8, 0.0)

    # --- glycogen synthesis, debited from glucose ---
    fill_L = min(1.0, max(0.0, (params.LGly_max - LGly) / params.glyc_fill_width_L))
    fill_M = min(1.0, max(0.0, (params.MGly_b - MGly) / params.glyc_fill_width_M))
    syn_L = params.glyc_syn_frac_L * Ra_carb * (0.5 + 0.5 * ins_drive) * fill_L  # mg/dL/min
    syn_M_g = params.k_glyc_syn_M * Ra_carb * ins_drive * fill_M                # g/min
    syn_M = syn_M_g * MG_DL_PER_G                                               # mg/dL/min

    # --- hepatic output target (mg/kg/min) ---
    g_ins_glyco = _glyc_ins_gate(I, Ib, params.glyc_ins_K, params.glyc_ins_n)
    g_ins_gng = _glyc_ins_gate(I, Ib, params.gng_ins_K, params.gng_ins_n)
    g_gn = _hill_centred(Gn, params.Gnb, params.hgo_gn_n)
    g_cort = 1.0 + params.gng_cort_amp * np.tanh(np.log(max(Cort, 0.05) / params.Cort_b))
    g_ffa = (max(FFA, 1e-3) / params.FFA_b) ** params.gng_ffa_exp
    g_G = (Gb / max(G, 1.0)) ** params.hep_autoreg_m if G > Gb else 1.0    # one-sided by physiology
    glyco_b = params.Hep_b - params.Gng_b
    glyco_t = glyco_b * (LGly / params.LGly_b) * g_ins_glyco * g_gn * g_G
    gng_total = params.Gng_b * g_cort * np.sqrt(g_gn) * g_ffa * g_ins_gng * g_G
    gng_divert = gng_total * ins_drive ** params.gng_divert_exp                 # indirect pathway -> glycogen
    gng_rel_t = gng_total - gng_divert
    hep_target = glyco_t + gng_rel_t

    # --- realized (lagged) hepatic output, split in the target's proportion ---
    f_glyco = glyco_t / hep_target if hep_target > 1e-9 else 0.0
    glyco_flux = Hep * f_glyco                                                  # mg/kg/min
    gng_rel_flux = Hep - glyco_flux
    egp = Hep / VG_DL_PER_KG                                                    # mg/dL/min

    # --- muscle glycogenolysis (activity-gated, oxidized locally) ---
    act_ex = max(act - params.act_rest_M, 0.0)
    brk_M_g = params.k_glyc_brk_M * act_ex * (MGly / (MGly + params.glyc_K_M))  # g/min

    dG = Ra_carb - syn_L - syn_M + egp - uptake_ii - uptake_id - uptake_ex
    dLGly = (syn_L / MG_DL_PER_G
             + gng_divert * BODY_MASS_KG / 1000.0
             - glyco_flux * BODY_MASS_KG / 1000.0)                              # g/min
    dMGly = syn_M_g - brk_M_g
    return {
        "dG": dG, "dLGly": dLGly, "dMGly": dMGly, "hep_target": hep_target,
        "ra": Ra_carb, "syn_L": syn_L, "syn_M": syn_M, "egp": egp,
        "uptake_ii": uptake_ii, "uptake_id": uptake_id, "uptake_ex": uptake_ex,
        "glyco_flux": glyco_flux, "gng_rel_flux": gng_rel_flux, "gng_divert": gng_divert,
        "brk_M_g": brk_M_g, "ins_drive": ins_drive, "x_eff": x_eff,
    }


def generate_meal_plan(
    n_days: int,
    rng: np.random.Generator,
    start_hour: float = 6.0,
) -> list[tuple[float, float, float, float]]:
    meals = []
    for day in range(n_days):
        day_offset = day * 1440 - start_hour * 60
        meals.append((day_offset + rng.uniform(7 * 60, 9 * 60),
                       rng.uniform(30, 70), rng.uniform(5, 20), rng.uniform(10, 30)))
        meals.append((day_offset + rng.uniform(12 * 60, 14 * 60),
                       rng.uniform(40, 90), rng.uniform(10, 35), rng.uniform(15, 40)))
        meals.append((day_offset + rng.uniform(18 * 60, 20 * 60),
                       rng.uniform(50, 100), rng.uniform(15, 40), rng.uniform(20, 50)))
        if rng.random() > 0.5:
            meals.append((day_offset + rng.uniform(10 * 60, 16 * 60),
                           rng.uniform(10, 30), rng.uniform(3, 15), rng.uniform(2, 15)))
    meals = [m for m in meals if m[0] >= 0]
    meals.sort()
    return meals


def generate_sleep_wake(
    n_days: int,
    duration_min: int,
    start_hour: float,
    rng: np.random.Generator,
) -> np.ndarray:
    sleep_wake = np.ones(duration_min, dtype=np.float32)
    for day in range(n_days + 1):
        day_start = day * 1440
        bedtime_hour = rng.uniform(22.0, 24.0)
        wake_hour = rng.uniform(6.0, 8.0) + 24
        sleep_start = int((bedtime_hour - start_hour) * 60) + day_start
        sleep_end = int((wake_hour - start_hour) * 60) + day_start
        sleep_wake[max(0, sleep_start):min(duration_min, sleep_end)] = 0.0
    kernel = np.ones(20, dtype=np.float32) / 20.0
    sleep_wake = np.convolve(sleep_wake, kernel, mode='same').astype(np.float32)
    return np.clip(sleep_wake, 0.0, 1.0)


def generate_activity(
    n_days: int,
    duration_min: int,
    start_hour: float,
    rng: np.random.Generator,
) -> np.ndarray:
    # Iter 97: rest is 0. The 0.05 floor (applied even asleep) was worth +4.2 bpm
    # HR, +2 mmHg SBP and +2.5 br/min of fictitious activity drive in every teacher
    # episode, and `base.py` defines 0 = rest for the student's input. No awake
    # NEAT floor either: a floor gated on sleep_wake would only re-encode
    # sleep_wake in a second input.
    activity = np.zeros(duration_min, dtype=np.float32)
    for day in range(n_days):
        day_offset = day * 1440
        if rng.random() > 0.4:
            ex_start = int(day_offset + (rng.uniform(7, 19) - start_hour) * 60)
            ex_duration = int(rng.uniform(20, 60))
            ex_intensity = rng.uniform(0.3, 0.9)
            s = max(0, ex_start)
            e = min(duration_min, ex_start + ex_duration)
            activity[s:e] = ex_intensity
    return activity


def compute_absorption_profile(
    t: float,
    meals: list[tuple[float, float, float, float]],
    params: PatientParams,
) -> tuple[float, float, float, float]:
    """Compute nutrient appearance rates at time t.

    Returns (glucose_appearance, lipid_appearance, amino_appearance, nutrient_flag).
    """
    fast_rate = params.meal_absorption_fast_rate
    slow_rate = params.meal_absorption_slow_rate
    slow_frac = params.meal_absorption_slow_fraction
    fast_frac = 1.0 - slow_frac

    carb_total = 0.0
    fat_total = 0.0
    protein_total = 0.0
    for mt, mc, mf, mp in meals:
        carb_total += fast_frac * _meal_absorption(t, mt, mc, fast_rate)
        carb_total += slow_frac * _meal_absorption(t, mt, mc, slow_rate)
        fat_total += _fat_absorption(t, mt, mf)
        protein_total += _protein_absorption(t, mt, mp)

    nutrient_flag = 1.0 if (carb_total + fat_total + protein_total) > 0.01 else 0.0
    return carb_total, fat_total, protein_total, nutrient_flag


def simulate_full_body(
    params: PatientParams,
    meals: list[tuple[float, float, float, float]],
    sleep_wake: np.ndarray,
    activity: np.ndarray,
    duration_min: int,
    start_hour: float = 6.0,
    noise_scale: float = 0.003,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate full marker state jointly with cross-system coupling.

    Returns (trajectory, absorption_profile).
    trajectory: (duration_min, STATE_DIM)
    absorption_profile: (duration_min, 4) — glucose/lipid/amino appearance + nutrient flag
    """
    if rng is None:
        rng = np.random.default_rng(42)
    # Iter 97 follow-up: a PatientParams is only a patient once its derived fields
    # are resolved (Hep_b, Sg, k_bhb, the enterohepatic constants). Callers that
    # build a raw ``PatientParams()`` or override a field after ``randomize_params``
    # (tests, textbook scenarios, synthetic profiles) used to run a subtly different
    # patient -- the iter-80 ketosis test ran with a 3x slower ketone clearance
    # than the population. Resolving here is idempotent and makes the invariant
    # "declared setpoints are the fixed points" hold by construction for every
    # caller; the dataclass defaults are also set to their derived values so a raw
    # default and a resolved default are the same patient.
    params = resolve_derived_params(params)

    trajectory = np.zeros((duration_min, STATE_DIM))
    absorption_profile = np.zeros((duration_min, 4))

    G, I, Gn = params.Gb, params.Ib, params.Gnb
    FFA, BHB, Lac = params.FFA_b, params.BHB_b, params.Lac_b
    Hep = params.Hep_b
    Ghr, Lep, GLP1 = params.Ghr_b, params.Lep_b, params.GLP1_b
    Cort, ACTH = params.Cort_b, params.ACTH_b
    HR, HRV = params.HR0, params.HRV0
    SBP, DBP = params.SBP0, params.DBP0
    T, RR, SpO2 = params.T0, params.RR0, params.SpO2_0
    LGly, MGly = params.LGly_b, params.MGly_b
    CCK, GB, INT, BA = params.CCK_b, params.GB_b, params.INT_b, params.BA_b
    X = 0.0
    Pot = 0.0   # delayed glucose signal for second-phase secretion (internal, see ins_phase2_frac)
    Ins_slow = params.Ib   # 4 h low-pass of insulin (internal, see lep_ins_gain)
    ns = noise_scale

    # Iter 91: the `(1 + Ib/IC50_ghr)` factor was a COMPENSATION for the standing suppression
    # that basal insulin used to apply (insulin_supp_ghr used ABSOLUTE insulin, so it was ~0.33
    # even at rest). With suppression now correctly keyed to insulin ABOVE basal, there is
    # nothing to compensate for -- keeping the boost overshot ghrelin to 138.8 against a basal
    # of 100. Basal production is simply Ghr_b * k_ghr, whose equilibrium is exactly Ghr_b.
    ghr_base_prod = params.Ghr_b * params.k_ghr
    glp1_base_prod = params.GLP1_b * params.k_glp1

    for t in range(duration_min):
        t_abs = (start_hour * 60 + t) % 1440
        sw = float(sleep_wake[t])
        act = float(activity[t])
        sleep_depth = 1.0 - sw

        # --- Gut absorption ---
        Ra_carb, Ra_fat, Ra_protein, nutrient_flag = compute_absorption_profile(t, meals, params)
        absorption_profile[t] = [Ra_carb, Ra_fat, Ra_protein, nutrient_flag]
        Ra = Ra_carb

        # --- Metabolic: one carbon budget (iter 97; see glucose_fluxes) ---
        si_effective = params.Si * (1.0 + params.act_insulin_sens * act)
        p3 = si_effective * params.p2
        # Incretin effect on GLP-1 ABOVE basal (item 3.8).
        glp1_excess = max(GLP1 - params.GLP1_b, 0.0)
        incretin_factor = 1.0 + params.incretin_gain * glp1_excess / (glp1_excess + params.K_incretin)

        # Ketogenesis gate: how much of the liver pool can still be mobilized.
        phi_L = LGly / (LGly + params.glyc_K_L)
        phi_L0 = params.LGly_b / (params.LGly_b + params.glyc_K_L)
        glyco_avail = min(phi_L / phi_L0, 1.0)

        fl = glucose_fluxes(params, G, I, X, Gn, Cort, FFA, LGly, MGly, Hep, Ra, act)
        dG = fl["dG"]
        dLGly = fl["dLGly"]
        dMGly = fl["dMGly"]
        dHep = -params.k_hep * (Hep - fl["hep_target"])
        # Bergman's remote insulin action is SIGNED (item 3.2/3.10): sub-basal insulin
        # withdraws insulin-dependent disposal; the floor lives in glucose_fluxes.
        dX = -params.p2 * X + p3 * (I - params.Ib)
        glucose_ratio = min(G / max(params.Gb, 1.0), 1.0)
        # Iter 97: floored at fast_ins_floor * Ib. With an absolute fasting floor a
        # Gb-130 patient fasts to G/Gb = 0.6, and 0.6^5 = 0.08 took its insulin to
        # 0.8 uU/mL and BHB to 6.7 mM at 48 h (a 5-7 day starvation value). Basal
        # secretion never switches off entirely; fasted insulin in the insulin-
        # resistant stays 3-5 (Polonsky 1988).
        effective_Ib = params.Ib * max(glucose_ratio ** params.fast_ins_exp, params.fast_ins_floor)
        g_above = max(G - params.h, 0.0)
        dPot = -params.k_pot * (Pot - g_above)
        w2 = params.ins_phase2_frac
        dI = (-params.n * (I - effective_Ib)
              + params.gamma * ((1.0 - w2) * g_above + w2 * Pot) * incretin_factor)

        glucagon_stim = params.alpha_gn * max(params.Gb - G, 0) / max(params.Gb, 1)
        # Iter 91: suppression responds to insulin ABOVE BASAL, not absolute insulin. With
        # `0.5 * I / (Ib + 10)` the basal insulin level already suppressed glucagon (0.25 at
        # I=Ib), so the teacher's glucagon rested at 60.3 against its own declared basal of 70
        # -- and the student faithfully distilled that 10-unit deficit. Physiologically,
        # alpha-cells are suppressed by a RISE in insulin (Unger & Orci); at basal insulin the
        # hormone sits AT its basal.
        # Iter 97: 0.5 -> 0.4. Measured with the 75 g standard meal the nadir was -56%
        # against the cited -20..-30% (and the OGTT anchor's -25 +/- 12).
        # Iter 97 follow-up: SIGNED about basal. Intra-islet insulin tonically
        # restrains the alpha cell, so insulin falling BELOW basal disinhibits it
        # (Unger & Orci 1981; the "switch-off" signal). That disinhibition is what
        # carries the fasting glucagon rise -- Marliss 1970: 108 -> 158 pg/mL over
        # 3 days, ~0.5-0.7 pg/mL per fasted hour -- which a rectified term could not
        # produce (the teacher rose 0.25/h, from the glucose signal alone).
        glucagon_supp = 0.4 * (I - params.Ib) / (params.Ib + 10.0)
        dGn = -params.k_gn * (Gn - params.Gnb) + glucagon_stim - glucagon_supp + 0.02 * Ra_protein
        # Glucagon reaches glucose through hepatic output (g_gn in glucose_fluxes),
        # not through a separate additive term.
        ra_norm = Ra / (Ra + params.K_ra_norm)

        lipolysis = params.lip_max / (1 + I / params.IC50_lip)
        dFFA = lipolysis - params.k_ffa * FFA + 0.01 * Ra_fat

        # glyco_depletion ∈ [0,1): 0 when the liver is full (fed calibration —
        # ketosis unchanged), rising as it empties to drive the fuel switch.
        glyco_depletion = max(0.0, 1.0 - LGly / params.LGly_b)   # linear (see keto_glyc_gain)
        ketogenesis = (
            params.keto_max * FFA / (1 + I / params.IC50_keto)
            * (1.0 + params.keto_glyc_gain * glyco_depletion)
        )
        dBHB = ketogenesis - params.k_bhb * BHB

        # Iter 93: thresholded, not linear — see lac_thresh / lac_act_gain.
        lac_supra = max(act - params.lac_thresh, 0.0)
        dLac = -params.k_lac * (Lac - params.Lac_b) + params.lac_act_gain * lac_supra ** 2

        # --- Hepatobiliary: the enterohepatic circulation (iter 95) ---
        # Four states in series. The delay structure is the point: CCK peaks ~10 min
        # after a meal but serum bile acids peak at 75-120 min, and that gap is
        # gallbladder emptying -> intestinal transit -> ileal reabsorption -> hepatic
        # first-pass extraction happening in sequence. Nothing here fits the gap
        # directly; it falls out. Anchors: docs/iter95-biliary-anchors.md.
        #
        # CCK: duodenal I-cells read FAT and PROTEIN, not carbohydrate.
        # Duodenal delivery, NOT systemic appearance — see _duodenal_delivery.
        fat_duo = sum(_duodenal_delivery(t, mt, mf, params.duo_fast_rate,
                                         params.duo_slow_rate, params.duo_slow_frac)
                      for mt, _mc, mf, _mp in meals)
        prot_duo = sum(_duodenal_delivery(t, mt, mp, params.duo_fast_rate,
                                          params.duo_slow_rate, params.duo_slow_frac)
                       for mt, _mc, _mf, mp in meals)
        carb_duo = sum(_duodenal_delivery(t, mt, mc, params.duo_fast_rate,
                                          params.duo_slow_rate, params.duo_slow_frac)
                       for mt, mc, _mf, _mp in meals)
        duo_total = fat_duo + prot_duo + carb_duo                    # g/min into the duodenum
        fed_gate = duo_total / (duo_total + params.K_mmc_fed)        # 0 fasted -> 1 fed
        cck_drive = (params.cck_fat_gain * fat_duo
                     + params.cck_prot_gain * prot_duo)
        dCCK = -params.k_cck * (CCK - params.CCK_b) + cck_drive
        # Gallbladder: CCK-gated emptying, PROPORTIONAL TO CONTENT. Proportionality is
        # what makes emptying exponential (the observed early-rapid / late-slow shape)
        # and what makes a second meal empty far less than the first — a gallbladder
        # cannot be emptied twice. Contraction is a GATE computed here, not a state;
        # see docs/iter95-proposal.md 3.2.1.
        cck_excess = max(CCK - params.CCK_b, 0.0)
        contraction = cck_excess / (cck_excess + params.K_cck_gb)   # in [0, 1)
        # CCK-driven ejection plus the interdigestive (MMC) partial emptying that keeps
        # the loop turning between meals; both proportional to content. The MMC is a
        # FASTED-state motor pattern -- suppressed while nutrient is in the duodenum --
        # which is what lets the gallbladder refill after a meal instead of leaking.
        gb_empty = (params.k_gb_eject * contraction
                    + params.k_gb_basal * (1.0 - fed_gate)) * GB          # mmol/min
        # Intestine: what the gallbladder delivers transits and is reabsorbed in the
        # ileum at ~95%; the remaining ~5% is the faecal loss that hepatic synthesis
        # replaces. This state is where the transit delay lives.
        ileal_uptake = params.k_ileal * INT                          # mmol/min
        portal_return = params.f_ileal * ileal_uptake                # mmol/min
        # Hepatic first pass. Extraction saturates against canalicular export capacity:
        # when k_canalicular falls (cholestasis), the liver cannot clear the portal
        # load into bile, extraction drops, and the unextracted remainder spills into
        # the systemic circulation. Serum bile acids then rise as a CONSEQUENCE of the
        # mass balance rather than by assertion — which is the whole reason this step
        # is explicit rather than lumped into a single clearance constant.
        extraction = params.hep_extraction * (
            params.k_canalicular / (params.k_canalicular + 0.15)
        ) / (1.0 / (1.0 + 0.15))     # == hep_extraction at k_canalicular = 1
        extraction = min(max(extraction, 0.0), 0.995)
        spillover = (1.0 - extraction) * portal_return               # mmol/min
        # Serum: a mass-action pool. What the liver clears from serum returns to bile.
        serum_return = params.k_ba * BA / params.ba_spill_gain       # mmol/min
        dBA = params.ba_spill_gain * spillover - params.k_ba * BA
        # De-novo synthesis under the FXR/FGF19 loop: ileal enterocytes secrete FGF19
        # in proportion to the bile acids they absorb, and FGF19 represses hepatic
        # CYP7A1. Synthesis therefore scales INVERSELY with the returning flux
        # (Inagaki 2005) -- equal to the derived basal at the fixed point, falling
        # while a meal's bolus is being reabsorbed, rising as the pool runs low. It is
        # what holds the pool: faecal loss is 5% of whatever flux passes the ileum, so
        # a fed day (three boluses) loses more than a constant synthesis replaces.
        # Squared: CYP7A1 repression is steep (synthesis rises 2-3x with modest pool
        # depletion, bile acid sequestrant studies); first order left the pool
        # recovering a fed day's loss over ~5 days.
        ileal_b = params.k_ileal * params.INT_b
        ba_synth = params.k_ba_synth * (ileal_b / max(ileal_uptake, 0.25 * ileal_b)) ** 2
        # Hepatic bile secretion = everything the liver takes up + de-novo synthesis
        # (iter 97: the loop is closed; nothing enters from or leaves to nowhere).
        # Canalicular export capacity gates how much of it reaches bile.
        hep_secretion = (extraction * portal_return + serum_return + ba_synth)
        hep_secretion *= min(params.k_canalicular, 1.0)
        # Split between the gallbladder (interdigestive diversion, tapering to zero as it
        # fills) and direct duodenal flow.
        divert = params.gb_divert_frac * min(1.0, max(0.0, (params.GB_max - GB) / params.gb_fill_width))
        gb_fill = hep_secretion * divert
        dGB = gb_fill - gb_empty
        dINT = gb_empty + hep_secretion * (1.0 - divert) - ileal_uptake

        # --- Glycogen pools (iter 76) ---
        # Flux integrators, not setpoints. Synthesis is gated on gut carb
        # appearance (Ra_carb) and insulin drive, tapering to zero as the pool
        # fills — this is the SAME gut-coupling pathway the GlycogenFluxHead's
        # anabolic gate reads, so the cold-distill target is expressible by the
        # learned head. Breakdown is tissue-specific:
        #   liver: glycogenolysis runs in the post-absorptive / fasted state
        #          (suppressed by above-basal insulin) and saturates as the
        #          pool empties — so it cannot release glucose it no longer
        #          has. This is what depletes the liver pool overnight.
        #   muscle: drains ONLY during activity (no hepatic G6Pase — muscle
        #          glycogen is consumed locally, preserved in a resting fast:
        #          Coppack 1989), matching the head's activity-gated catabolism.
        # COUPLING (iter 80): liver glycogen now feeds back into glucose via the
        # hepatic-output split above (glyco_avail reads LGly). This closes the
        # glycogen→glucose loop the iter-76 comment flagged as "the next iter":
        # liver depletion lowers the glycogenolytic share of hepatic output, so
        # the *strong* glucose gradient (real data + gate + dose-response) now
        # reaches LGly through a conservation-exact edge instead of relying on
        # the weak open-loop cold-distill trajectory alone (iters 55-57 never
        # got the SetpointHead pools off `typical` — the teacher had no signal
        # to give; now it does). The split is the IDENTITY at the fed
        # calibration state, so the acute/fed observed-marker ODE is unchanged.
        # ITER 96 -- THE FILL TAPER GAVE THE POOL AN IMPLICIT SETPOINT (the taper
        # `1 - LGly/LGly_max` throttled refill to 9% at the fed level; it is a WIDTH
        # now). ITER 97 -- the pools are ledgers of the SAME fluxes the glucose
        # balance books: dLGly, dMGly come out of glucose_fluxes above, together
        # with dG. Synthesis is debited from glucose, glycogenolysis credited to it.

        # --- Appetite (ghrelin, leptin, GLP-1) ---
        # Iter 97: ghrelin's nutrient suppressor is DUODENAL delivery (g/min), the
        # same signal CCK reads, not systemic appearance. Ghrelin is suppressed by
        # nutrient sensing in the small intestine (Williams 2003: gastric distension
        # alone does nothing; intestinal infusion does), so with the absorption curve
        # now at its real breadth a systemic drive put the nadir at 130 min against
        # Cummings' 60-90. Same half-saturation constant, now in g/min.
        Ra_norm = duo_total / (duo_total + params.K_meal_ghr)
        # Iter 91: same fix as glucagon -- above-basal insulin, not absolute. At I=Ib the old
        # form gave 10/30 = 0.33 of standing suppression, so ghrelin rested at 94.9 against a
        # basal of 100. Ghrelin falls POSTPRANDIALLY (Cummings 2001), i.e. in response to the
        # insulin RISE, not to the existence of basal insulin.
        # Iter 97: and it RISES in fasting (item 3.2). The rectifier `(I - Ib)+` made
        # ghrelin exactly 100.000 at 12/24/36/48 h of fasting. Signed saturating form
        # centred at basal:  u = ((I/Ib)^n - 1) / ((I/Ib)^n + 1)  in (-1, 1),
        # 0 at basal, half-suppression at 3x basal (the old IC50), -0.35 at the insulin
        # of a 24 h fast -> ghrelin +21% (Espelund 2005 / Natalucci 2005: +15-30%).
        ins_ratio_n = (max(I, 1e-3) / params.Ib) ** params.ghr_ins_n
        insulin_supp_ghr = (ins_ratio_n - 1.0) / (ins_ratio_n + 1.0)
        # Iter 92 -- GHRELIN WAS OVER-SUPPRESSED (measured nadir -78% vs Cummings 2001's
        # -30 to -50%). Two independent causes, both fixed here:
        #
        # 1. DOUBLE-COUNTING. The old form multiplied the two suppressors:
        #        ghr_prod = base * (1 - insulin_supp) * (1 - Ra_norm)
        #    But insulin-above-basal and nutrient appearance are two readings of the SAME
        #    postprandial event, not independent inhibitors. Multiplying them compounds:
        #    measured at the trough, (1-0.646)*(1-0.834) = 0.059, i.e. production fell to
        #    ~6% of basal. They are now combined as a saturating union (1 - prod of the
        #    complements), so either signal alone can suppress but the two cannot stack
        #    past saturation.
        # 2. NO CEILING. Even ONE suppressor at full strength was too strong on its own
        #    (Ra_norm alone implies ~-85% at equilibrium, insulin alone ~-60%), so the
        #    union still needs a cap. ghr_supp_max bounds the suppressible fraction of
        #    basal production, which is what sets the achievable nadir.
        #
        # Insulin is deliberately RETAINED as a contributor (rather than dropping to the
        # nutrient signal alone) so the registered insulin->ghrelin -1 coupling prior
        # continues to be supported by the trajectory signal.
        meal_supp_ghr = 1.0 - (1.0 - insulin_supp_ghr) * (1.0 - Ra_norm)
        antic = _anticipation_drive(t_abs, params.habitual_meal_hours,
                                    params.ghr_antic_ramp_h, params.ghr_antic_decay_h)
        ghr_prod = (ghr_base_prod * (1.0 + params.ghr_antic_amp * antic)
                    * (1.0 - params.ghr_supp_max * meal_supp_ghr))
        dGhr = ghr_prod - params.k_ghr * Ghr

        circ_lep = _circadian(t_abs, params.lep_circ_amp, peak_hour=2.0)
        dIns_slow = -params.k_ins_slow * (Ins_slow - I)
        lep_ins = params.lep_ins_gain * (Ins_slow / params.Ib - 1.0)
        dLep = -params.k_lep * (Lep - params.Lep_b - circ_lep - lep_ins)

        glp1_prod = glp1_base_prod + params.glp1_meal_gain * Ra
        dGLP1 = glp1_prod - params.k_glp1 * GLP1

        # --- Stress / HPA (ACTH → cortisol + feedback) ---
        #
        # Iter 91 — CORTISOL WAS DOUBLE-DRIVEN. Through iter 90 cortisol relaxed toward a
        # circadian target (Cort_b + circ_cort) AND received an additive k_acth_to_cort·ACTH
        # term. But ACTH is itself circadian, so the same rhythm was injected twice and the
        # ACTH term added a standing offset on top. Measured equilibrium:
        #     Cort* ≈ (Cort_b 12 ± circ 5) + k_acth_to_cort·ACTH/k_cort (≈ +6.9) ≈ 18.9 ± 5
        # so the simulated teacher produced nadir 11.75 / peak 25.5 µg/dL with a peak:nadir
        # ratio of 2.17. Physiology: nadir 3-5, peak 15-20, ratio ~4-5x (Weitzman 1971;
        # Pruessner 1997). The teacher's cortisol was ~3x too high overnight and its rhythm
        # was half as deep as it should be -- and cortisol drives glucose (cort_gluco, added to
        # the student in iter-90 C3), heart rate, BP and thermoregulation, so the error
        # propagated into every one of those.
        #
        # The fix is the cascade this model already claims to implement: ACTH DRIVES CORTISOL.
        # The circadian belongs in ACTH (where it already is, and where the SCN→PVN→pituitary
        # pathway actually puts it); cortisol simply follows its own secretagogue. So cortisol's
        # relaxation target is now proportional to ACTH, with no second circadian of its own.
        # ACTH's own rhythm is deepened (acth_circ_amp) because the previous amplitude was far
        # too compressed to carry cortisol's real 4-5x swing.
        #
        # Cort_b is RETAINED as the per-patient reference level -- it is the threshold for
        # cortisol's downstream effects (cort_feedback_acth, cort_gluco, cort_hr) and for the
        # student's normalization -- but it is no longer cortisol's relaxation target.
        # Iter 97 (item 3.9): asymmetric drive, quiescent evening, steep pre-dawn rise
        # peaking at awakening; sleep suppression applied ONCE, to ACTH.
        drive = _hpa_drive(t_abs, params.hpa_rise_start_h, params.hpa_peak_h, params.hpa_fall_tau_h)
        acth_target = max(params.ACTH_b + params.acth_circ_amp * (2.0 * drive - 1.0), 5.0)
        sleep_suppression = 1.0 - params.hpa_sleep_supp * sleep_depth
        dACTH = -params.k_acth * (ACTH - acth_target * sleep_suppression)
        dACTH += params.hypo_acth * max(70.0 - G, 0)
        dACTH += params.cort_activity * act
        dACTH -= params.cort_feedback_acth * max(Cort - params.Cort_b, 0)

        # Cortisol tracks its secretagogue: target = cort_per_acth · ACTH, the same
        # ratio asleep and awake (the suppression already lives in ACTH).
        cort_target = max(params.cort_per_acth * max(ACTH, 0.0), 0.5)
        dCort = -params.k_cort * (Cort - cort_target)

        cort_dev = Cort - params.Cort_b

        # --- Cardiovascular ---
        #
        # NO ARTERIAL BAROREFLEX TERM (evaluated and rejected 2026-07-20; measured, not
        # assumed). The obvious addition here is dHR += -k_baro*(SBP - SBP0) with k_baro
        # ~0.5-1.5 bpm/mmHg from BRS ~15 ms/mmHg (Guyton & Hall 14th ed. ch. 18). It is
        # wrong at this model's resolution, for four independent reasons:
        #
        # 1. TIMESCALE. The baroreflex is a beat-to-beat mechanism (~1-5 s). This is a
        #    MINUTE-mean Euler sim, so the reflex fully equilibrates inside a single step.
        #    It is not a dynamic state here -- it is already implicitly folded into the
        #    effective act_hr_gain / k_hr gains.
        # 2. NOTHING TO REFLEX AGAINST. The baroreflex exists to buffer pressure
        #    perturbations (orthostasis, Valsalva, hemorrhage, vasoactive drugs). This sim
        #    has none of them, and BP process noise is ns*0.5 <= 0.0005 mmHg/min. Measured
        #    over 40 patients x 24 h: SBP sd is 9.97 mmHg but the ACTIVITY-INDEPENDENT part
        #    -- the only part a correctly-gated reflex could act on -- has sd 0.40 mmHg in
        #    steady periods. That is a ~1 bpm HR effect: below the noise floor.
        # 3. THE RESIDUAL IS A LAG ARTIFACT, AND ACTING ON IT WOULD DO HARM. What
        #    activity-independent SBP deviation does exist is concentrated entirely at
        #    exercise transitions, where first-order SBP lags its new equilibrium: peak
        #    residual -17.7 mmHg at bout onset, +17.7 mmHg at offset. Feeding that to
        #    -k_baro*resid would ADD ~+18 bpm/min to the exercise HR rise and SUBTRACT
        #    ~18 bpm/min from recovery -- corrupting dynamics that are currently CORRECT
        #    (measured HRR1 at maximal intensity = 22.0 bpm, vs Cole 1999 healthy 20-30).
        # 4. SIGN. Central command RESETS the baroreflex operating point upward during
        #    exercise, so HR and SBP rise together; realized corr(HR, SBP) in the teacher
        #    is +0.964. A naive reflex would fight that co-rise rather than model it.
        #
        # Consequence for the student: the sbp->hr NEGATIVE coupling prior is likewise NOT
        # registered (see coupling_priors/cardiovascular.py) -- it would contradict a
        # trajectory signal whose realized HR-SBP correlation is +0.96.
        circ_hr = _circadian(t_abs, params.hr_circ_amp, peak_hour=14.0)
        sleep_hr_shift = -params.sleep_hr_frac * params.HR0 * sleep_depth
        dHR = (-params.k_hr * (HR - params.HR0 - circ_hr - sleep_hr_shift)
                + params.cort_hr * cort_dev
                + params.act_hr_gain * act
                + params.meal_hr_gain * ra_norm)
        hrv_sleep_mult = 1.0 + (params.sleep_hrv_gain - 1.0) * sleep_depth
        # Iter 97: cortisol's vagal effect is two-sided, as it already is for HR and BP
        # (item 3.2): low overnight cortisol RAISES HRV instead of being invisible.
        dHRV = (-params.k_hrv * (HRV - params.HRV0 * params.HR0 / max(HR, 40) * hrv_sleep_mult)
                 - params.cort_hrv * cort_dev)
        sleep_sbp_shift = -params.sleep_bp_frac * params.SBP0 * sleep_depth
        sleep_dbp_shift = -params.sleep_bp_frac * params.DBP0 * sleep_depth
        dSBP = (-params.k_bp * (SBP - params.SBP0 - sleep_sbp_shift)
                + params.cort_bp * cort_dev + params.act_sbp_gain * act)
        dDBP = (-params.k_bp * (DBP - params.DBP0 - sleep_dbp_shift)
                + params.cort_bp * cort_dev * 0.5 + params.act_dbp_gain * act)

        # --- Thermoregulation ---
        circ_temp = _circadian(t_abs, params.temp_circ_amp, peak_hour=16.0)
        # Diet-induced thermogenesis from absorbed ENERGY (iter 97): kernels -> g/min
        # (carb kernel integrates to grams x MG_DL_PER_G; fat/protein to 3 x grams),
        # x kcal/g x thermic fraction (protein 25%, carbohydrate 8%, fat 3%).
        dit_kcal = (0.08 * 4.0 * Ra_carb / MG_DL_PER_G
                    + 0.03 * 9.0 * Ra_fat / 3.0
                    + 0.25 * 4.0 * Ra_protein / 3.0)
        dit = params.temp_dit_gain * dit_kcal
        sleep_temp_shift = -params.sleep_temp_drop * sleep_depth
        exercise_temp_target = params.temp_exercise_gain * act
        dT = (-params.k_temp * (T - params.T0 - circ_temp - sleep_temp_shift - exercise_temp_target)
              + dit)

        # --- Respiratory ---
        sleep_rr_shift = -params.sleep_rr_drop * sleep_depth
        lac_excess = max(Lac - params.Lac_b, 0)
        lactate_drive = params.rr_lactate_gain * lac_excess / (lac_excess + 2.0)
        dRR = -params.k_rr * (RR - params.RR0 - sleep_rr_shift) + act * 5.0 + lactate_drive
        spo2_exercise_effect = params.spo2_exercise_dip * max(act - 0.5, 0)
        dSpO2 = -params.k_spo2 * (SpO2 - params.SpO2_0) - spo2_exercise_effect

        # Euler integration with process noise
        G = max(G + dG + rng.normal(0, ns * 2), 20)
        X = X + dX
        Pot = max(Pot + dPot, 0.0)
        Ins_slow = max(Ins_slow + dIns_slow, 0.1)
        I = max(I + dI + rng.normal(0, ns * 0.5), 0.1)
        Gn = max(Gn + dGn + rng.normal(0, ns * 1), 1)
        FFA = max(FFA + dFFA + rng.normal(0, ns * 0.01), 0.01)
        BHB = max(BHB + dBHB + rng.normal(0, ns * 0.005), 0.001)
        Lac = max(Lac + dLac + rng.normal(0, ns * 0.02), 0.1)
        Hep = max(Hep + dHep + rng.normal(0, ns * 0.05), 0.05)
        Ghr = max(Ghr + dGhr + rng.normal(0, ns * 2), 5)
        Lep = max(Lep + dLep + rng.normal(0, ns * 0.1), 0.5)
        GLP1 = max(GLP1 + dGLP1 + rng.normal(0, ns * 0.5), 1)
        ACTH = max(ACTH + dACTH + rng.normal(0, ns * 0.8), 5.0)
        Cort = max(Cort + dCort + rng.normal(0, ns * 0.3), 0.5)
        HR = max(HR + dHR + rng.normal(0, ns * 1), 30)
        HRV = max(HRV + dHRV + rng.normal(0, ns * 1), 1)
        SBP = max(SBP + dSBP + rng.normal(0, ns * 0.5), 60)
        DBP = max(DBP + dDBP + rng.normal(0, ns * 0.3), 30)
        T = T + dT + rng.normal(0, ns * 0.01)
        RR = max(RR + dRR + rng.normal(0, ns * 0.2), 4)
        SpO2 = min(100, max(SpO2 + dSpO2 + rng.normal(0, ns * 0.1), 70))
        LGly = max(LGly + dLGly + rng.normal(0, ns * 0.5), 1.0)
        MGly = max(MGly + dMGly + rng.normal(0, ns * 0.5), 1.0)
        CCK = max(CCK + dCCK + rng.normal(0, ns * 0.05), 0.05)
        # The diversion taper makes GB_max unreachable; the clip is a safety only.
        GB = min(max(GB + dGB, 0.0), params.GB_max)
        INT = max(INT + dINT, 0.0)
        BA = max(BA + dBA + rng.normal(0, ns * 0.2), 0.1)

        trajectory[t] = [
            G, I, Gn, FFA, BHB, Lac, Hep,
            Ghr, Lep, GLP1, Cort, ACTH,
            HR, HRV, SBP, DBP, T, RR, SpO2,
            # Iter 76 (Move D): liver_glycogen + muscle_glycogen are now
            # SIMULATED flux integrators (see the glycogen block above), giving
            # cold-model distillation a real trajectory target for the slow
            # pools — the measurement that iters 55-57 lacked.
            LGly,  # liver_glycogen (g) — overnight-depleting fast pool
            MGly,  # muscle_glycogen (g) — rest-preserved, exercise-coupled
            # mitochondrial_capacity stays padded: its τ ≈ weeks means a ≤1-day
            # distillation protocol can't exercise it — it needs the chronic-
            # block protocols (the next Move D iter), so simulating it here
            # would only emit a flat reference. crh likewise stays padded (the
            # cold ODE does not simulate it; see the HPA note below).
            1.0,    # mitochondrial_capacity (× population mean)
            # Iter 69 Move B FULL — CRH as latent first stage of the
            # HPA cascade. The cold model does not simulate CRH, so it
            # is padded with the population-typical value; the learned
            # model discovers CRH dynamics by satisfying the cascade-
            # derived ACTH and cortisol trajectories (which the cold
            # model does simulate).
            100.0,  # crh (pg/mL) — typical resting level
            # Iter 89 — insulin_action (remote insulin) latent. The cold ODE
            # DOES track a remote-insulin X internally (dX above), but the
            # student's insulin_action is a differently-scaled low-pass of
            # relu(insulin_above_baseline) and is unsupervised (not a
            # cold-distill marker), so this is padded at the student's fasting
            # equilibrium (0) — the value only seeds a rollout start-state, and
            # 0 = relu(insulin-baseline) at rest, matching the benchmark loader's
            # typical-padding of the same index.
            0.0,    # insulin_action (a.u.) — fasting equilibrium
            # Iter 95 — hepatobiliary. SIMULATED (not padded): the whole point of the
            # axis is that it is dynamic and meal-locked, so the distillation gets a
            # real trajectory target from day one rather than a flat reference.
            CCK,    # cck (pmol/L)
            GB,     # gallbladder_bile (mmol) — the pool
            INT,    # intestinal_bile (mmol) — transit delay + the 95%/5% split
            BA,     # bile_acids (µmol/L) — the observable
        ]

    return trajectory, absorption_profile


# Iter 91 — standard meal used to characterise a patient's postprandial glucose response.
# 75 g carbohydrate is the OGTT dose (Guyton & Hall Ch. 79), so the resulting peak-rise is a
# clinically meaningful per-patient quantity rather than an arbitrary probe.
STANDARD_MEAL_CARBS_G = 75.0
STANDARD_MEAL_FATS_G = 5.0
STANDARD_MEAL_PROTEINS_G = 10.0
# Iter 91: the meal sits at t=120 (not t=30) so glucose has ~2.2 time-constants (tau ~56 min)
# to SETTLE at the patient's own fasting equilibrium first. That settled level is itself a
# supervision target: it is the OBSERVABLE fasting glucose, which is NOT params.Gb -- the
# standing hepatic source puts the equilibrium at Gb + egp/Sg, measured at +0.85..+5.09 mg/dL
# and VARYING per patient. Calibration only ever sees the observable, so supervising the latent
# Gb alone injects a per-patient bias.
_MEAL_RESPONSE_DURATION_MIN = 360
_MEAL_RESPONSE_MEAL_TIME_MIN = 120


def _standard_meal_response(params: PatientParams) -> dict[str, float]:
    """This patient's glucose response to a standard 75 g meal, from the teacher itself.

    Returns the peak RISE above the patient's own fasting equilibrium (mg/dL) and the time to
    that peak (minutes after the meal). Both are per-patient consequences of the sampled
    PatientParams -- Si, Sg, the absorption rates and the insulin response all feed in -- so
    they are exactly the amplitude information the student's Ra head needs and has never had.

    Deterministic (noise_scale=0): this is a reference quantity, not a training trajectory.
    Cost is one 240-minute simulation per patient at dataset-generation time, i.e. negligible
    beside the 14-day episode already being simulated.
    """
    meals = [(
        float(_MEAL_RESPONSE_MEAL_TIME_MIN),
        STANDARD_MEAL_CARBS_G, STANDARD_MEAL_FATS_G, STANDARD_MEAL_PROTEINS_G,
    )]
    n = _MEAL_RESPONSE_DURATION_MIN
    sleep_wake = np.ones(n)          # awake
    activity = np.zeros(n)           # at rest
    traj, _ = simulate_full_body(
        params, meals, sleep_wake, activity, n,
        start_hour=8.0, noise_scale=0.0, rng=np.random.default_rng(0),
    )
    g = traj[:, MARKER_INDEX["glucose"]]
    # Fasting reference = the level just before the meal (the patient's own equilibrium,
    # which sits slightly above params.Gb because of the standing hepatic source -- so a RISE
    # is the honest amplitude measure, independent of that offset).
    pre = float(g[_MEAL_RESPONSE_MEAL_TIME_MIN - 1])
    post = g[_MEAL_RESPONSE_MEAL_TIME_MIN:]
    peak_idx = int(np.argmax(post))
    return {
        # The OBSERVABLE fasting glucose (settled), not params.Gb. See the note above.
        "glucose_fasting": pre,
        "glucose_peak_rise": float(post[peak_idx] - pre),
        "glucose_time_to_peak_min": float(peak_idx),
    }


class FullBody(KnowledgeContribution):
    def __init__(self, n_days: int = 14):
        super().__init__(
            name="full_body",
            source="Bergman (1979); Weitzman (1971); Mancia (1993); ESC/NASPE (1996)",
            description="Coherent full-body simulation with cross-system coupling (incl. HPA)",
        )
        self.n_days = n_days

    def generate_episodes(self, n_episodes: int, rng: np.random.Generator) -> list[Episode]:
        episodes = []
        for _ in range(n_episodes):
            prng = np.random.default_rng(rng.integers(0, 2**32))
            params = randomize_params(prng)
            start_hour = 6.0
            duration_min = self.n_days * 1440
            meals = generate_meal_plan(self.n_days, prng, start_hour)
            sleep_wake = generate_sleep_wake(self.n_days, duration_min, start_hour, prng)
            activity = generate_activity(self.n_days, duration_min, start_hour, prng)

            trajectory, absorption_profile = simulate_full_body(
                params, meals, sleep_wake, activity,
                duration_min, start_hour, rng=prng,
            )

            episodes.append(Episode(
                trajectory=trajectory,
                meals=meals,
                duration_min=duration_min,
                start_hour=start_hour,
                sleep_wake=sleep_wake,
                activity=activity,
                absorption_profile=absorption_profile,
                source=self.name,
                # Iter 90: the ground-truth per-patient resting setpoints this episode was
                # simulated from. These are exactly the quantities the model's per-patient
                # heads decode from the embedding (metabolic.glucose_baseline_net ->
                # Gb; cardiovascular.setpoint_net -> HR0/HRV0/SBP0/DBP0), so they give the
                # embedding->physiology map direct supervision instead of leaving it to be
                # discovered through 12h of integrated rate. Only markers with a dedicated
                # per-patient head are listed.
                setpoints={
                    "glucose": float(params.Gb),
                    "hr": float(params.HR0),
                    "hrv": float(params.HRV0),
                    "sbp": float(params.SBP0),
                    "dbp": float(params.DBP0),
                    # Iter 91: thermoreg gained a per-patient setpoint head (modules/thermoreg.py),
                    # so its resting level now has a ground-truth target instead of having to be
                    # discovered from trajectories. temp is a thin-margin gate marker.
                    "temp": float(params.T0),
                },
                # Iter 91: this patient's TRUE postprandial glucose peak-rise for a standard
                # meal. Measured on iter-90: the student's per-patient meal gain (Ra) is FROZEN
                # (trained std 0.01), because nothing ever supervised per-patient meal
                # amplitude -- SetpointSupervisionSignal (iter 90) covered Gb/HR0/HRV0/SBP0/DBP0
                # but not Ra, and dose-response only supplies a POPULATION target (Wolever's
                # 0.7 mg/dL/g, identical for every patient). The consequence is severe: with a
                # meal in the calibration window the optimizer cannot fit a person's meal
                # amplitude, so it compensates with the only lever it has and INFLATES their Gb
                # (measured: pushes Gb to ~110 whether the truth is 100 or 85). That single bias
                # explains BOTH failures of iter 90 -- calibration recovers Gb to 0.96 mg/dL with
                # no meal in the window but is off by 17.84 mg/dL with one, and the same inflated
                # Gb drives the fasting eval prediction upward (glucose_mape 0.193 -> 0.210).
                #
                # The teacher knows each patient's true meal response exactly, and we were
                # throwing it away -- the same oversight as the setpoints above. Recording it
                # gives ra_baseline_net a real per-patient gradient.
                meal_response=_standard_meal_response(params),
            ))
        return episodes

    def trajectory_loss_mode(self) -> str:
        return "mse"

    def coupling_priors(self) -> list[CouplingPrior]:
        return [
            CouplingPrior("glucose", "insulin", sign=+1, magnitude_range=(0.001, 0.02)),
            CouplingPrior("insulin", "glucose", sign=-1, magnitude_range=(0.0001, 0.001)),
            CouplingPrior("glucose", "glucagon", sign=-1, magnitude_range=(0.001, 0.01)),
            CouplingPrior("cortisol", "glucose", sign=+1, magnitude_range=(0.001, 0.01)),
            CouplingPrior("cortisol", "hepatic_output", sign=+1, magnitude_range=(0.02, 0.15)),
            CouplingPrior("insulin", "hepatic_output", sign=-1, magnitude_range=(0.02, 0.12)),
            CouplingPrior("insulin", "ghrelin", sign=-1, magnitude_range=(0.01, 0.1)),
            CouplingPrior("cortisol", "hr", sign=+1, magnitude_range=(0.1, 1.0)),
        ]
