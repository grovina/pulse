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


# Iter 92: carb appearance gain, split out from the shared `* 3.0` the three
# absorption kernels used to share. Sharpening the carb kernel (see
# meal_absorption_fast_rate) advances the glucose peak but also raises its
# amplitude, because a gamma-2 kernel's peak FLUX scales with rate while its
# integral (3*carbs) does not. The excursion amplitude was already correct
# (+45 mg/dL vs literature +40-50), so this gain absorbs the amplitude side
# effect and leaves timing as the only thing the rate change moves.
# 2.55 measured (N=30): rise +45.5 mg/dL, i.e. baseline amplitude preserved.
# Fat and protein keep the original 3.0 -- their kernels are unchanged.
#
# TRADEOFF, stated honestly: holding the PEAK fixed costs ~10% of the 3 h
# incremental AUC (5189 -> 4675 mg/dL*min). The alternative -- sharpen the kernel
# and keep gain at 3.0 -- holds AUC (5343) but pushes the peak to +54 mg/dL, out of
# the literature's +40-50 band. Peak excursion is the well-anchored observable and
# AUC is not, so the peak wins. The absolute carb->appearance conversion implied by
# this gain is still NOT independently validated against literature (it is a
# phenomenological scale, not a mass-conserving one); that remains open.
CARB_APPEARANCE_GAIN = 2.55


def _meal_absorption(t: float, meal_time: float, carbs: float,
                     rate: float = 0.03) -> float:
    dt = t - meal_time
    if dt < 0 or dt > 300:
        return 0.0
    return carbs * rate * rate * dt * np.exp(-rate * dt) * CARB_APPEARANCE_GAIN


def _fat_absorption(t: float, meal_time: float, fats: float,
                    rate: float = 0.015) -> float:
    dt = t - meal_time
    if dt < 0 or dt > 420:
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
    if dt < 0 or dt > 360:
        return 0.0
    return proteins * rate * rate * dt * np.exp(-rate * dt) * 3.0


@dataclass
class PatientParams:
    # Glucose-Insulin (Bergman minimal model). Iter 21 recalibration:
    # gamma 0.015 -> 0.07 and h 80 -> 95 align cold-model OGTT insulin
    # peak with DeFronzo (~60 uU/mL) and zero out GSIR at fasting glucose
    # so insulin can drop below Ib during a fast. Si 0.0002 -> 0.0004
    # tightens late-glucose clearance toward the OGTT 120-min target.
    Sg: float = 0.018
    Si: float = 0.0004
    Gb: float = 95.0
    Ib: float = 10.0
    p2: float = 0.03
    n: float = 0.15
    gamma: float = 0.07
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

    # --- Iter 93: THE FASTED STATE NEVER ENGAGED ---------------------------
    # Measured over 12 randomized patients, a 48 h fast produced: glucose
    # 97.4 -> 99.9 mg/dL (literature: 70-80 by 24 h), insulin 16.0 uU/mL at
    # 24 h (literature 3-5), FFA 0.50 -> 0.44 (literature 2-3x rise). One root
    # cause, three symptoms: in `dG = -(Sg+X)(G-Gb) + Ra`, Gb is a HARD
    # ATTRACTOR, so with no meal G cannot fall below it. Insulin's fasted
    # setpoint keys off `min(G/Gb, 1)`, which therefore never leaves 1.0, so
    # insulin never falls; and lipolysis is insulin-gated, so FFA never rises.
    # The iter-21 comment above already INTENDED "insulin can drop below Ib
    # during a fast" — the trigger simply could not fire.
    #
    # The minimal model is a 3-hour tool: over that span Gb genuinely is the
    # defended level. Over a fast it is not — as hepatic glycogen empties,
    # gluconeogenesis cannot fully replace glycogenolysis and the defended
    # level itself falls (Cahill 2006). So Gb becomes glycogen-dependent,
    # gated on the SAME `glyco_avail` the hepatic-output split and ketogenesis
    # already read — which makes it exactly the IDENTITY at the fed
    # calibration state (glyco_avail = 1), so no fed/postprandial behaviour
    # moves. Confirmed against the user's own CGM: 14/14 overnight episodes
    # fall, -2.37 mg/dL/h through 23:00-07:00.
    # Calibrated on the DEFAULT (healthy) patient against five independent
    # anchors at once — see the iter-93 spec. Population means are NOT used for
    # this calibration: the population deliberately includes impaired/diabetic
    # patients, so its mean fasting insulin is not what Polonsky measured.
    fast_gb_drop: float = 0.55
    # Prolonged starvation does not extrapolate to zero — glucose plateaus at
    # ~60-70 mg/dL and holds there for weeks on gluconeogenesis alone (Cahill
    # 2006). The floor makes that asymptote explicit rather than trusting the
    # liver pool never to empty. It is slack in every protocol shorter than
    # ~3 days; it exists so the linear law cannot run somewhere absurd.
    fast_gb_floor_frac: float = 0.62
    # Insulin's fall in fasting is far steeper than the glucose fall that
    # drives it (Polonsky 1988: basal insulin roughly halves while glucose
    # drops ~15%) because beta-cell secretion is sigmoid in glucose near
    # threshold. A linear ratio cannot express that; the exponent can.
    # Identity whenever G >= Gb, so again the fed state is untouched.
    fast_ins_exp: float = 5.0

    # Glucagon
    Gnb: float = 70.0
    k_gn: float = 0.03
    alpha_gn: float = 1.5

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
    k_bhb: float = 0.005
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
    keto_glyc_gain: float = 5.0

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
    k_gb_fill: float = 0.004      # /min, interdigestive refill toward GB_max
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
    ba_spill_gain: float = 45.0   # umol/L per mmol/min of unextracted portal return
    # Hepatic synthesis replaces faecal loss (~5% of the pool per cycle), holding the
    # ~3 g total pool steady over a day.
    k_ba_synth: float = 0.0010    # mmol/min

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

    # Hepatic endogenous glucose output (slow flux into glucose mass balance)
    Hep_b: float = 1.2
    k_hep: float = 0.04
    cort_hep: float = 0.06
    ins_hep: float = 0.08
    gn_hep: float = 0.018
    hep_to_glucose: float = 0.038
    meal_suppress_hep: float = 0.38
    # Hepatic-output split (iter 80). The lumped `Hep` is partitioned into a
    # glycogenolytic share (scaled by liver-glycogen availability) and a
    # gluconeogenic share. CONSERVATION-EXACT at the fed calibration state
    # (LGly = LGly_b): the two shares sum to the old `Hep`, so fed/acute
    # trajectories are byte-identical to iter 79. They diverge only as the
    # liver pool depletes (~12 h+ fast): the glycogenolytic component falls
    # with availability, gluconeogenesis partially compensates, and net
    # hepatic glucose output declines — the Cahill-2006 prolonged-fast
    # picture, and the edge that finally makes glucose *downstream of* the
    # slow glycogen pool (so glucose's strong gradient reaches LGly).
    hep_glyco_frac: float = 0.6   # glycogenolytic share of basal hepatic output (early fast)
    hep_gng_comp: float = 0.5     # gluconeogenic compensation as the liver empties (0..1)

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
    IC50_ghr: float = 20.0
    K_meal_ghr: float = 0.3
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

    # GLP-1. Iter 21 recalibration: glp1_meal_gain 5.0 -> 1.5 to match
    # the large-meal GLP-1 peak of ~22 uU/mL (was overshooting to ~64
    # at the previous gain — z=4.76 vs the literature target).
    GLP1_b: float = 10.0
    k_glp1: float = 0.2
    glp1_meal_gain: float = 1.5
    K_incretin: float = 15.0

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
    k_acth_to_cort: float = 0.006   # legacy (iter<=90 additive ACTH->cortisol term; unused since iter 91)
    # Iter 91: cortisol's relaxation target is cort_per_acth * ACTH (see the HPA block). Set so
    # the ACTH rhythm carries cortisol across its physiological range; tuned by measurement below.
    cort_per_acth: float = 0.42
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
    hr_circ_amp: float = 5.0
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
    meal_hr_gain: float = 4.0

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
    temp_dit_gain: float = 0.0008
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
    meal_absorption_fast_rate: float = 0.040
    meal_absorption_slow_rate: float = 0.012
    meal_absorption_slow_fraction: float = 0.25

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
    LGly_max: float = 110.0     # liver storage cap (synthesis tapers as it fills)
    MGly_max: float = 450.0     # muscle storage cap
    k_glyc_syn_L: float = 0.6   # liver synthesis gain (per unit carb-appearance·insulin-drive)
    k_glyc_syn_M: float = 0.45  # muscle synthesis gain
    k_glyc_brk_L: float = 0.062  # liver glycogenolysis gain (post-absorptive)
    k_glyc_brk_M: float = 3.5   # muscle glycogenolysis gain (activity-driven)
    glyc_ins_supp: float = 15.0  # above-basal insulin (µU/mL) that halves liver glycogenolysis
    glyc_K_L: float = 35.0      # liver depletion-saturation constant (g)
    glyc_K_M: float = 150.0     # muscle depletion-saturation constant (g)
    # Iter 96: width (g) over which synthesis tapers off as the pool approaches
    # its cap. See the dLGly block -- `1 - LGly/LGly_max` throttled refill to 9%
    # of capacity exactly at the fed level, which made a eucaloric day
    # glycogen-NEGATIVE and gave the pool an implicit setpoint near 51 g.
    glyc_fill_width_L: float = 30.0
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

    # Glucose effectiveness falls with insulin resistance (Bergman: Sg and Si co-degrade).
    p.Sg = vary(p.Sg, ir=-0.30)
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
    # How far the defended glucose level falls once the liver empties.
    # Clip brackets the calibrated 0.55 default (see PatientParams); an
    # insulin-resistant liver defends its glucose harder, hence ir=-0.30.
    p.fast_gb_drop = float(np.clip(vary(p.fast_gb_drop, 0.25, ir=-0.30), 0.25, 0.90))
    p.Hep_b = float(np.clip(vary(p.Hep_b, 0.3), 0.4, 3.5))
    p.k_hep = float(np.clip(vary(p.k_hep, 0.35), 0.02, 0.09))
    p.cort_hep = float(np.clip(vary(p.cort_hep, 0.35), 0.02, 0.12))
    p.hep_to_glucose = float(np.clip(vary(p.hep_to_glucose, 0.3), 0.018, 0.055))
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
    p.meal_absorption_slow_fraction = float(np.clip(vary(p.meal_absorption_slow_fraction, 0.3), 0.05, 0.6))
    # Glycogen pool sizes vary across patients (training/diet history); the
    # flux gains stay fixed so the dynamics shape is consistent. Caps track
    # the baselines so a larger pool can still fill. Cold-distill references
    # use PatientParams() defaults, so this only diversifies full_body episodes.
    p.LGly_b = float(np.clip(vary(p.LGly_b, 0.15), 70.0, 130.0))
    p.MGly_b = float(np.clip(vary(p.MGly_b, 0.15, fit=0.50), 300.0, 520.0))
    p.LGly_max = p.LGly_b + 10.0
    p.MGly_max = p.MGly_b + 50.0
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

    Idempotent, so it is safe to call more than once.
    """
    params.lip_max = params.FFA_b * params.k_ffa * (1.0 + params.Ib / params.IC50_lip)
    # Iter 96: GSIR threshold tracks the patient's own defended fasting glucose.
    params.h = params.Gb * params.h_frac
    return params


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
    activity = np.full(duration_min, 0.05, dtype=np.float32)
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

        # --- Metabolic (Bergman + glucagon + FFA + BHB + lactate) ---
        si_effective = params.Si * (1.0 + params.act_insulin_sens * act)
        p3 = si_effective * params.p2
        incretin_factor = 1 + GLP1 / (GLP1 + params.K_incretin)

        # Iter 93: hoisted above the glucose ODE (it used to be computed with the
        # hepatic split, further down) so the DEFENDED GLUCOSE LEVEL can read it.
        # Depends only on LGly, which is a state carried in from the previous
        # step, so hoisting changes no value — only availability.
        phi_L = LGly / (LGly + params.glyc_K_L)
        phi_L0 = params.LGly_b / (params.LGly_b + params.glyc_K_L)
        glyco_avail = min(phi_L / phi_L0, 1.0)

        # Iter 93: the defended level falls as hepatic glycogen empties (see
        # fast_gb_drop). Zero at the fed calibration state (LGly = LGly_b), so
        # this is the IDENTITY there and the entire fed/postprandial regime —
        # the regime the gate and the dose-response signals score — is
        # unchanged.
        #
        # Keyed to LINEAR pool depletion, NOT to `glyco_avail`. glyco_avail is
        # a saturating Michaelis ratio built to describe how much glycogen the
        # liver can still RELEASE, and it stays near 1 while the pool halves
        # (measured: LGly 100 -> 43.5 over 24 h moves glyco_avail only
        # 1.00 -> 0.75). Reusing it here would have been convenient and wrong:
        # what the defended level tracks is how much of the pool is GONE.
        glyco_depleted = max(0.0, 1.0 - LGly / max(params.LGly_b, 1e-6))
        Gb_fasted = max(
            params.Gb * (1.0 - params.fast_gb_drop * glyco_depleted),
            params.Gb * params.fast_gb_floor_frac,
        )

        dG = -(params.Sg + X) * (G - Gb_fasted) + Ra
        dX = -params.p2 * X + p3 * max(I - params.Ib, 0)
        # NB: the ratio stays referenced to the FED Gb, not Gb_fasted — it is
        # what senses the fall. Referencing it to the falling level would
        # cancel exactly the signal it exists to carry.
        glucose_ratio = min(G / max(params.Gb, 1.0), 1.0)
        effective_Ib = params.Ib * glucose_ratio ** params.fast_ins_exp
        dI = -params.n * (I - effective_Ib) + params.gamma * max(G - params.h, 0) * incretin_factor
        dG += params.hep_to_glucose * Hep + params.cort_gluco * max(Cort - params.Cort_b, 0)
        dG -= act * 0.02 * max(G - params.Gb * 0.8, 0)

        glucagon_stim = params.alpha_gn * max(params.Gb - G, 0) / max(params.Gb, 1)
        # Iter 91: suppression responds to insulin ABOVE BASAL, not absolute insulin. With
        # `0.5 * I / (Ib + 10)` the basal insulin level already suppressed glucagon (0.25 at
        # I=Ib), so the teacher's glucagon rested at 60.3 against its own declared basal of 70
        # -- and the student faithfully distilled that 10-unit deficit. Physiologically,
        # alpha-cells are suppressed by a RISE in insulin (Unger & Orci); at basal insulin the
        # hormone sits AT its basal.
        glucagon_supp = 0.5 * max(I - params.Ib, 0.0) / (params.Ib + 10.0)
        dGn = -params.k_gn * (Gn - params.Gnb) + glucagon_stim - glucagon_supp + 0.02 * Ra_protein
        dG += 0.02 * max(Gn - params.Gnb, 0)

        hep_target = (
            params.Hep_b
            + params.cort_hep * max(Cort - params.Cort_b, 0)
            + params.gn_hep * max(Gn - params.Gnb, 0)
            - params.ins_hep * max(I - params.Ib, 0) / (params.Ib + 5.0)
        )
        ra_norm = Ra / (Ra + 0.35) if (Ra + 0.35) > 1e-9 else 0.0
        hep_target *= max(0.2, 1.0 - params.meal_suppress_hep * min(ra_norm, 1.0))
        hep_target = max(hep_target, 0.12)
        # Iter-80 hepatic-output split. glyco_avail = 1 at the fed calibration
        # state (LGly = LGly_b) and falls toward 0 as the liver depletes, so
        # this partition is the IDENTITY until the pool empties (conservation-
        # exact) and only then bends hepatic output downward.
        hep_glyco = params.hep_glyco_frac * hep_target * glyco_avail
        hep_gng = (
            (1.0 - params.hep_glyco_frac) * hep_target
            + params.hep_gng_comp * params.hep_glyco_frac * hep_target * (1.0 - glyco_avail)
        )
        hep_target = hep_glyco + hep_gng
        dHep = -params.k_hep * (Hep - hep_target)

        lipolysis = params.lip_max / (1 + I / params.IC50_lip)
        dFFA = lipolysis - params.k_ffa * FFA + 0.01 * Ra_fat

        # glyco_depletion ∈ [0,1): 0 when the liver is full (fed calibration —
        # ketosis unchanged), rising as it empties to drive the fuel switch.
        glyco_depletion = max(0.0, 1.0 - glyco_avail)
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
        gb_empty = params.k_gb_eject * contraction * GB             # mmol/min
        # Interdigestive refill from hepatic secretion, tapering as the store fills.
        # Gated on canalicular export capacity — the cholestasis site.
        gb_fill = (params.k_gb_fill * params.k_canalicular
                   * max(params.GB_max - GB, 0.0))
        dGB = gb_fill - gb_empty
        # Intestine: what the gallbladder delivers transits and is reabsorbed in the
        # ileum at ~95%; the remaining ~5% is the faecal loss that hepatic synthesis
        # replaces. This state is where the transit delay lives.
        ileal_uptake = params.k_ileal * INT                          # mmol/min
        dINT = gb_empty + params.k_ba_synth - ileal_uptake
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
        dBA = params.ba_spill_gain * spillover - params.k_ba * (BA - params.BA_b)

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
        ins_drive = max(I - params.Ib, 0.0) / (max(I - params.Ib, 0.0) + params.Ib)
        # ITER 96 -- THE FILL TAPER GAVE THE POOL AN IMPLICIT SETPOINT.
        # The taper used to be `1 - LGly/LGly_max`, which at the fed level
        # LGly=100 with LGly_max=110 is 0.091: synthesis was throttled to 9% of
        # capacity exactly where the pool is supposed to be refilling, while
        # breakdown ran at LGly/(LGly+K) = 0.74 of its own. Measured on the
        # eucaloric cohort day (175 g carbohydrate, 3 meals): synthesis totalled
        # 21 g against 48 g of breakdown, i.e. **a normal eating day was
        # glycogen-NEGATIVE by 27 g**, and the pool ran down to an implicit
        # equilibrium of 51 g over five such days. A storage pool whose fed state
        # cannot hold its own level has stopped being a storage pool: it no longer
        # discriminates fed from fasted in EITHER direction, which is exactly what
        # `extended_fast_liver_glycogen_overnight` had been reporting (teacher
        # delta -7.8 g) and what the student inherited (its own pool drains to
        # 19 g over five eucaloric days -- docs/iter95 notes).
        #
        # The taper is now a WIDTH: full synthesis through the working range,
        # closing over the last `glyc_fill_width` grams before the cap. Same
        # physical statement (you cannot overfill a liver), without the throttle
        # that was ALSO applied through the whole normal operating range.
        # Measured after the fix: the pool is stationary -- day-1 and day-5
        # eucaloric levels agree to 0.1 g at every setting swept.
        #
        # This is the same shape error as the student's GlycogenFluxHead anabolic
        # gate (`headroom` grows as the pool empties while `fullness` shrinks its
        # breakdown authority) -- see the iter-96 proposal. Both are fixed here.
        fill_L = min(1.0, max(0.0, (params.LGly_max - LGly) / params.glyc_fill_width_L))
        fill_M = min(1.0, max(0.0, (params.MGly_max - MGly) / params.glyc_fill_width_M))
        syn_L = params.k_glyc_syn_L * Ra_carb * ins_drive * fill_L
        syn_M = params.k_glyc_syn_M * Ra_carb * ins_drive * fill_M
        fast_gate_L = 1.0 / (1.0 + max(I - params.Ib, 0.0) / params.glyc_ins_supp)
        brk_L = params.k_glyc_brk_L * fast_gate_L * (LGly / (LGly + params.glyc_K_L))
        act_ex = max(act - params.act_rest_M, 0.0)  # only supra-rest activity spends muscle glycogen
        brk_M = params.k_glyc_brk_M * act_ex * (MGly / (MGly + params.glyc_K_M))
        dLGly = syn_L - brk_L
        dMGly = syn_M - brk_M

        # --- Appetite (ghrelin, leptin, GLP-1) ---
        Ra_norm = Ra / (Ra + params.K_meal_ghr) if (Ra + params.K_meal_ghr) > 1e-9 else 0.0
        # Iter 91: same fix as glucagon -- above-basal insulin, not absolute. At I=Ib the old
        # form gave 10/30 = 0.33 of standing suppression, so ghrelin rested at 94.9 against a
        # basal of 100. Ghrelin falls POSTPRANDIALLY (Cummings 2001), i.e. in response to the
        # insulin RISE, not to the existence of basal insulin.
        insulin_supp_ghr = max(I - params.Ib, 0.0) / (max(I - params.Ib, 0.0) + params.IC50_ghr)
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
        ghr_prod = ghr_base_prod * (1.0 - params.ghr_supp_max * meal_supp_ghr)
        dGhr = ghr_prod - params.k_ghr * Ghr

        circ_lep = _circadian(t_abs, params.lep_circ_amp, peak_hour=2.0)
        dLep = -params.k_lep * (Lep - params.Lep_b - circ_lep)

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
        circ_acth = _circadian(t_abs, params.acth_circ_amp, peak_hour=7.5)
        acth_target = max(params.ACTH_b + circ_acth, 5.0)
        sleep_suppression = 1.0 - 0.3 * sleep_depth
        dACTH = -params.k_acth * (ACTH - acth_target * sleep_suppression)
        dACTH += params.hypo_acth * max(70.0 - G, 0)
        dACTH += params.cort_activity * act
        dACTH -= params.cort_feedback_acth * max(Cort - params.Cort_b, 0)

        # Cortisol tracks its secretagogue: target = cort_per_acth · ACTH.
        cort_target = max(params.cort_per_acth * max(ACTH, 0.0), 0.5)
        dCort = -params.k_cort * (Cort - cort_target * sleep_suppression)

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
        dHRV = (-params.k_hrv * (HRV - params.HRV0 * params.HR0 / max(HR, 40) * hrv_sleep_mult)
                 - params.cort_hrv * max(cort_dev, 0))
        sleep_sbp_shift = -params.sleep_bp_frac * params.SBP0 * sleep_depth
        sleep_dbp_shift = -params.sleep_bp_frac * params.DBP0 * sleep_depth
        dSBP = (-params.k_bp * (SBP - params.SBP0 - sleep_sbp_shift)
                + params.cort_bp * cort_dev + params.act_sbp_gain * act)
        dDBP = (-params.k_bp * (DBP - params.DBP0 - sleep_dbp_shift)
                + params.cort_bp * cort_dev * 0.5 + params.act_dbp_gain * act)

        # --- Thermoregulation ---
        circ_temp = _circadian(t_abs, params.temp_circ_amp, peak_hour=16.0)
        Ra_total = Ra_carb + Ra_fat + Ra_protein
        dit = params.temp_dit_gain * Ra_total
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
