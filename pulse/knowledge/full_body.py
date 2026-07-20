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


def _meal_absorption(t: float, meal_time: float, carbs: float,
                     rate: float = 0.03) -> float:
    dt = t - meal_time
    if dt < 0 or dt > 300:
        return 0.0
    return carbs * rate * rate * dt * np.exp(-rate * dt) * 3.0


def _fat_absorption(t: float, meal_time: float, fats: float,
                    rate: float = 0.015) -> float:
    dt = t - meal_time
    if dt < 0 or dt > 420:
        return 0.0
    return fats * rate * rate * dt * np.exp(-rate * dt) * 3.0


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
    h: float = 95.0

    # Glucagon
    Gnb: float = 70.0
    k_gn: float = 0.03
    alpha_gn: float = 1.5

    # Free fatty acids
    FFA_b: float = 0.5
    lip_max: float = 0.033
    IC50_lip: float = 15.0
    k_ffa: float = 0.04

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
    keto_glyc_gain: float = 22.0

    # Lactate
    Lac_b: float = 1.0
    k_lac: float = 0.02

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
    k_ghr: float = 0.02
    IC50_ghr: float = 20.0
    K_meal_ghr: float = 0.3

    # Leptin
    Lep_b: float = 10.0
    k_lep: float = 0.001
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
    hr_circ_amp: float = 5.0
    HRV0: float = 40.0
    k_hrv: float = 0.1
    SBP0: float = 120.0
    DBP0: float = 80.0
    k_bp: float = 0.2
    cort_hr: float = 0.3
    cort_hrv: float = 0.2
    cort_bp: float = 0.15

    # Thermal
    T0: float = 37.0
    k_temp: float = 0.025
    temp_circ_amp: float = 0.45
    temp_exercise_gain: float = 0.8
    temp_dit_gain: float = 0.0008
    sleep_temp_drop: float = 0.15

    # Sleep modulation
    sleep_hr_frac: float = 0.15
    sleep_hrv_gain: float = 1.3
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
    meal_absorption_fast_rate: float = 0.03
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
    LGly_max: float = 110.0     # liver storage cap (synthesis tapers as it fills)
    MGly_max: float = 450.0     # muscle storage cap
    k_glyc_syn_L: float = 0.6   # liver synthesis gain (per unit carb-appearance·insulin-drive)
    k_glyc_syn_M: float = 0.45  # muscle synthesis gain
    k_glyc_brk_L: float = 0.062  # liver glycogenolysis gain (post-absorptive)
    k_glyc_brk_M: float = 3.5   # muscle glycogenolysis gain (activity-driven)
    glyc_ins_supp: float = 15.0  # above-basal insulin (µU/mL) that halves liver glycogenolysis
    glyc_K_L: float = 35.0      # liver depletion-saturation constant (g)
    glyc_K_M: float = 150.0     # muscle depletion-saturation constant (g)
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
    p.Gb = vary(p.Gb, 0.25, ir=0.50)          # fasting hyperglycemia tracks IR
    p.Ib = vary(p.Ib, 0.4, ir=0.60)           # compensatory hyperinsulinemia
    p.n = vary(p.n)
    p.gamma = vary(p.gamma)
    p.Gnb = vary(p.Gnb)
    p.FFA_b = vary(p.FFA_b, ir=0.35)          # impaired lipolysis suppression
    p.BHB_b = vary(p.BHB_b)
    p.Lac_b = vary(p.Lac_b)
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
    p.temp_circ_amp = float(np.clip(vary(p.temp_circ_amp, 0.15), 0.3, 0.55))
    p.temp_exercise_gain = float(np.clip(vary(p.temp_exercise_gain, 0.2), 0.4, 1.1))
    p.sleep_temp_drop = float(np.clip(vary(p.sleep_temp_drop, 0.2), 0.05, 0.25))
    p.sleep_hr_frac = float(np.clip(vary(p.sleep_hr_frac, 0.2), 0.08, 0.25))
    p.sleep_hrv_gain = float(np.clip(vary(p.sleep_hrv_gain, 0.15), 1.1, 1.6))
    p.sleep_bp_frac = float(np.clip(vary(p.sleep_bp_frac, 0.2), 0.06, 0.20))
    p.sleep_rr_drop = float(np.clip(vary(p.sleep_rr_drop, 0.2), 1.5, 5.0))
    p.act_hr_gain = vary(p.act_hr_gain, 0.15)
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
    return p


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

        dG = -(params.Sg + X) * (G - params.Gb) + Ra
        dX = -params.p2 * X + p3 * max(I - params.Ib, 0)
        glucose_ratio = min(G / max(params.Gb, 1.0), 1.0)
        effective_Ib = params.Ib * glucose_ratio
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
        phi_L = LGly / (LGly + params.glyc_K_L)
        phi_L0 = params.LGly_b / (params.LGly_b + params.glyc_K_L)
        glyco_avail = min(phi_L / phi_L0, 1.0)
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

        dLac = -params.k_lac * (Lac - params.Lac_b) + act * 0.3

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
        syn_L = (params.k_glyc_syn_L * Ra_carb * ins_drive
                 * max(0.0, 1.0 - LGly / params.LGly_max))
        syn_M = (params.k_glyc_syn_M * Ra_carb * ins_drive
                 * max(0.0, 1.0 - MGly / params.MGly_max))
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
        ghr_prod = ghr_base_prod * (1 - insulin_supp_ghr) * (1 - Ra_norm)
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
                + params.act_hr_gain * act)
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
