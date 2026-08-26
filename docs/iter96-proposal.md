# iter-96 proposal — the pre-dawn window, and two more wrong rulers

Predecessor: `docs/iter95-proposal.md`. iter-95's training and benchmark both landed
(artifact `gs://grovina-pulse-data/training/jobs/iter95/model.pt`, report
`.../iter95/benchmark-report.json`). Every measurement below was taken on **this tree**,
either against that artifact or against the teacher directly; each is re-runnable from
the script named beside it.

---

## 0. Where iter-95 left the gate

iter-95 was a clear net win — overall weighted MAPE 0.0825 → 0.0723, textbook pass rate
0.95 (best ever), `cgm_real` glucose skill −1.698 → −0.694 (best ever). **Two gate
failures remain, and they are the same two as last iteration:**

```
skill[cgm_real].glucose  -0.694   (iter-94: -1.698)
skill[cgm_real].hr       -1.165   (iter-94: -0.817)   <- REGRESSED, now the worse blocker
```

Post-benchmark diagnosis (2026-08-24/25) established three things, one of which corrected
an earlier claim:

1. **Both blockers live in the 03:00–06:00 window of the 14 real overnight episodes.**
2. **Embedding calibration is genuinely worse than not calibrating** (pooled MAPE hr
   0.0806 calibrated vs 0.0759 at the prior mean; glucose 0.1057 vs 0.0717; ‖emb‖ at the
   3.0 clamp in 10 of 14 episodes). But — the correction — killing that overfit clears
   **neither** blocker: projected glucose −0.694 → −0.15, hr −1.165 → −1.04. The first
   run that said otherwise was blind to `sleep_wake`.
3. **The teacher is wrong in that window first.** Its own dawn HR rise was +12.3 bpm
   against a real +1.6. No student-side work can beat a distillation target that is
   itself wrong.

So iter-96 is primarily a **teacher** iteration, with the two student-side amplifiers
that the same window's attribution named.

---

## 1. The finding: the teacher's dawn response, decomposed

`scratchpad/dawn.py` — the 14 `cgm_real` episodes, each with its own Oura `sleep_wake`
and `activity` masks, the episode's own initial state, the episode's own clock.

**Slopes over 03:00–06:00, mean across episodes (units/hour), all three on the same
ruler** (student at the prior-mean embedding, so calibration is not a confound):

```
                student   teacher    real
hr               +5.46     +4.08    +1.06     teacher 3.9x the real rise; student 5.2x
glucose          -0.15     -0.74    -1.75     teacher 42% of the real fall; student 9%
cortisol nadir    9.51      4.20    3-5 (Weitzman 1971)
```

Read the glucose row carefully: **the student loses 80% of even the teacher's own
decline.** So glucose has a student problem on top of a teacher problem, while HR is
mostly the teacher's error passed through and amplified.

HR is a rate equation, `dHR = −k_hr·(HR − HR0 − circ_hr − sleep_hr_shift) + cort_hr·cort_dev
+ …`, with `k_hr = 0.3` (τ = 3.3 min), so HR tracks its quasi-static target closely and the
target decomposes cleanly. Change in each channel's contribution, 03:00 → 06:00:

```
  circadian   (hr_circ_amp = 5, peak 14 h)     +2.33 bpm    19%
  sleep shift (sleep_hr_frac = 0.06)           +3.14 bpm    26%
  CORTISOL    (cort_hr/k_hr = 1.0 per µg/dL)   +6.84 bpm    56%   <---
  activity                                     +0.00 bpm
  ------------------------------------------------------
  realized teacher HR change                  +12.17 bpm    against a real +3.2
```

**Cortisol carries the majority of a dawn rise that is nearly four times too large.**

### 1.1 The gain is 7–13x the only human measurement of it

Adlan et al. (2018), *J Physiol* 596(20):4847-4861. 200 mg IV hydrocortisone vs placebo,
n = 10 healthy males, crossover, measured 3 h post-dose:

```
HR      +7 ± 4 bpm      (placebo 50.9 ± 9.7 -> 57.8 ± 9.0)
SBP     +5 ± 5 mmHg     (113.6 ± 7.9 -> 118.8 ± 6.8)
rMSSD   84 ± 38 ms -> 59 ± 29 ms
serum cortisol: placebo 93.7 ± 37.0 nmol/L; on hydrocortisone it exceeded the assay
  ceiling in 7 of 10 (censored to 1400; the three measurable read 2637 ± 42)
```

Using the **censored floor** — deliberately the conservative choice, because a smaller
denominator *overstates* the gain — Δcortisol = 47.3 µg/dL:

```
                   literature gain      teacher's realized gain (coefficient / k)
hr      per µg/dL       0.148 bpm                1.000 bpm       6.8x
sbp     per µg/dL       0.106 mmHg               0.750 mmHg      7.1x
hrv     per µg/dL      -0.528 ms                -2.000 ms        3.8x
```

Against the actually-measured concentration those literature gains halve again (hr 13.2x).
**The entire cortisol → cardiovascular block is over-gained by the same factor**, which is
what a single mis-scaled coupling looks like when it was fitted by eye rather than measured.

Honest limit, stated because it cuts against the fix: a 200 mg bolus is supraphysiological,
and a receptor-mediated effect could be *steeper*, not flatter, inside the physiological
4–18 µg/dL range. It is still the only direct human dose-response measurement available,
and a factor of 7 is far outside any plausible curvature.

### 1.2 The corroboration is that four other anchors moved for free

`cort_hr` 0.3 → 0.045 (and `cort_bp` 0.15 → 0.021, `cort_hrv` 0.2 → 0.053) alone drops
`sleep_hr_dip` from −11.9 bpm to −5.2, because sleep's cortisol suppression was being
amplified 7x on its way to HR. Restoring the explicit sleep term to
`sleep_hr_frac = 0.09` puts the dip back at −7.3 against its −8 ± 4 literature target.

That single re-tune was the **only** thing fitted. Three further anchors then landed on
their own (`scripts/cohort_teacher_audit.py`, N=12):

```
                     iter-95 z      iter-96 z
sleep_hr_dip           -0.99          -0.10
hrv_sleep_rise         +0.99          +0.09
sbp_sleep_dip          -0.87          +0.01
dbp_sleep_dip          -0.41          +0.14
```

Four independent literature anchors, each 0.4–1.0 sd off, all inside 0.15 sd after one
measured coefficient change. That is the signature of having fixed the right thing.

The `hr_circ_amp` comment in `full_body.py` recorded iter-94's reason for not touching any
of this: *"no cohort statistic constrains this term … the only justification available was
a whole-day sleep-vs-wake contrast — which is not an encoded, cited quantity at all."*
It is now: `DAY_NIGHT_HR_CONTRAST` and `NIGHT_HR_LEVEL` (24 h Holter reference cohort,
n = 134, PMID 2424396: day 82 ± 10, night 64 ± 8 bpm). With Adlan pinning `cort_hr`,
`sleep_hr_dip` pinning the sleep term, and these pinning the total, **the three HR channels
are separately identified for the first time.** Realized: 87.9 (z +0.59) and 64.4 (z +0.05).

**Result: the teacher's dawn HR rise falls +12.17 → +8.11 bpm** (slope +4.08 → +2.89 /h).
The residual is now dominated by the sleep→wake transition, which is real physiology —
Oura says these people are 86% awake by 06:00.

---

## 2. The second finding: a normal eating day was glycogen-negative

`scratchpad/glyco.py`. The teacher's liver glycogen synthesis was

```
syn_L = k_glyc_syn_L · Ra_carb · ins_drive · (1 − LGly/LGly_max)
```

with `LGly_b = 100` and `LGly_max = 110`. **At the fed level that taper is 0.091** —
synthesis throttled to 9% of capacity exactly where the pool is supposed to be refilling —
while breakdown ran at `LGly/(LGly+35) = 0.74` of its own. On the eucaloric cohort day
(175 g carbohydrate, three meals):

```
  total synthesis   21.00 g/day
  total breakdown   48.14 g/day
  net              -27.14 g/day
```

Five identical eucaloric days: 100 → 72.6 → 61.0 → 55.4 → 52.7 → 51.4 g. **The pool has an
implicit setpoint near 51 g and no longer discriminates fed from fasted in either
direction.** This is the same structure iter-95 recorded for the *student* ("liver glycogen
drains to 19 g over 5 eucaloric days") — the student inherited it.

**Fix:** the taper becomes a *width*, `clip((LGly_max − LGly)/30, 0, 1)` — full synthesis
through the working range, closing only over the last 30 g before the cap. Same physical
statement (you cannot overfill a liver), without the throttle that was also applied through
the entire normal operating range. **Measured after the fix, the pool is stationary:** day-1
and day-5 eucaloric levels agree to 0.1 g at every setting in a 4×3×3 sweep.

### 2.1 CONSIDERED AND REVERTED: raising the cap

With the width taper the eucaloric pool equilibrates near `LGly_max − 25`, so
`LGly_max = 125` lands it at exactly `LGly_b = 100` — which looks like the right answer for
"a eucaloric day is glycogen-neutral". **Measured, it was not.** The cap is not only a cap;
raising it also lets each meal deposit more, and the 10–16 h fasted arm went from 82.6 g to
97 g — the pool stopped emptying overnight, and four fasted-state anchors degraded together
(`extended_fast_insulin_basal` z +2.07 → +3.09, `bhb` −0.93 → −1.59, `ffa` −1.71 → −1.91).
Per-meal deposition is pinned by Taylor 1996 and the 24 h fast level by Cahill/Rothman;
both are already satisfied at 110. That a 175 g-carbohydrate day is then mildly
glycogen-negative is not obviously an error — it is a low-carbohydrate day. Recorded here
because the reasoning is the trap, not the number.

---

## 3. The third finding: the −60 g anchor could never have been satisfied

`extended_fast_liver_glycogen_overnight` has been contradicted at |z| ≈ 2 for roughly forty
iterations. It is not the teacher that is wrong.

Cahill's −60 g is an **absolute** depletion: fed pool ~100 g → ~40 g after a 16–24 h fast.
It was encoded as `DELTA_MEANS` between two arms that differ by **one 60 g-carbohydrate
dinner** — i.e. asking that dinner to deposit 60 g of glycogen, 100% of its carbohydrate,
into one organ.

**Measured: a 5×4×2 sweep over (`k_glyc_syn_L`, `k_glyc_brk_L`, fill width) moved this delta
only between −14.6 and −32.6**, and every setting that reached −32 had already driven the
24 h fast to 4.8 g against Cahill's ~40. The two are inconsistent, and the 24 h level is
the one that is actually the cited quantity.

What a meal really deposits — Taylor et al. (1996), *J Clin Invest* 97:126-132, ¹³C NMR
with an acetaminophen tracer: liver glycogen 207 ± 22 → 316 ± 19 mmol/l, peaking at
318 ± 31 min, **net 28.3 ± 3.7 g, ≈ 19% of meal carbohydrate.** For a 60 g dinner that is
11.4 g.

**Two changes:**
- `extended_fast_liver_glycogen_overnight`: target −60 ± 25 → **−13 ± 6** (Taylor). Teacher
  realizes −14.6, z −0.26.
- new `extended_fast_liver_glycogen_level`: `MEAN_IN_WINDOW` on the fasted arm alone,
  **60 ± 20 g** (Cahill; Rothman et al. 1991 *Science* 254:573, whose 64 ± 5% gluconeogenic
  fraction over the first 22 h implies ~3 g/h of net glycogenolysis). This is the absolute
  depletion the −60 was reaching for, now expressed as the statistic it actually is.
  Teacher realizes 83.7 g, z +1.18 — a live, reachable gradient pushing the right way.

---

## 4. The fourth finding: `h` was a population constant where the physiology is per-patient

`scratchpad/hframe.py`. In the Bergman insulin equation, `h` is the glucose threshold above
which glucose-stimulated insulin release engages. iter-21 moved it 80 → 95 precisely to
"zero out GSIR at fasting glucose" — correct intent, expressed against **one** patient's
fasting glucose, while `Gb` is sampled across 54–126 mg/dL.

Twelve randomized patients, extended-fast arm:

```
 pt     Gb     h    Gb-h |  fasted glucose  fasted insulin   standing GSIR
  4  122.9  95.0    27.9 |          122.0           31.31          11.59
  5  108.5  95.0    13.5 |          109.5           27.19          14.27
  6  126.4  95.0    31.4 |          125.7           33.55           9.55
  9  120.6  95.0    25.6 |          121.4           45.41          19.90
 11  121.7  95.0    26.7 |          122.2           53.46          28.70
 ...
mean fasted insulin 19.76 against the 7 ± 4 anchor;  corr(Gb, fasted insulin) = +0.835
patients with Gb > h: 5 of 12
```

**For 5 of 12 patients GSIR never switches off**, and fasted insulin runs 27–53 µU/mL.
The standing GSIR contribution averages 7.0 µU/mL — the entire budget of the anchor it
violates. Same family as the iter-95 frame error: a rate law referenced to a **population**
constant where the physiology references the **individual's** set point.

Fix: `h` is derived per patient in `resolve_derived_params` as `Gb · h_frac` (default 1.0).
`extended_fast_insulin_basal` z **+2.62 → +0.86**; `bhb` −1.17 → −0.71; `ffa` −1.73 → −1.51.

Also fixed alongside: `Gb` sampling is now clipped to **[70, 130] mg/dL**. Unclipped it
reached Gb = 54, and that patient's simulated fasted glucose settled at **42 mg/dL** —
neuroglycopenic, not a phenotype, and it was being distilled as one.

---

## 5. The fifth finding: six timing rules were measuring the middle of the window

iter-95 left this open, flagged but unmeasured. `scratchpad/beta.py` measures it.

Every soft-argmax in `knowledge/physiology_rules.py` used an absolute `beta = 0.05`, i.e.
`softmax(0.05 · value)`. Sharpness is only meaningful relative to how far the values
spread; the useful quantity is `beta · range`, and it must be ≳ 20 before the weights
concentrate on the peak at all. On a teacher 24 h trajectory:

```
marker      range   true argmax   reported @ beta=0.05   beta*range
cortisol    14.13     508 min          654 min             0.71
sbp         15.88     513              687                 0.79
acth        31.98     459              570                 1.60
temp         0.51     920              722                 0.03
leptin       0.94     418              714                 0.05
ghrelin     49.48       0              455                 2.47
```

Every one collapses toward **719.5 — the centroid of the 1440-minute window.** These rules
were not measuring peaks. Consequences on a teacher that satisfies them:

```
cortisol_morning_peak   true peak 508 IS INSIDE band [360,540]  -> reported 114 min violation (loss 0.95)
sbp_morning_surge       true peak 513 IS INSIDE band [360,600]  -> reported  87 min violation (loss 0.73)
temp_afternoon_peak     true error 40 min                       -> reported 238 min violation (loss 1.98)
```

**Two rules were penalising the teacher for peaks it puts in exactly the right place**, with
a gradient that pushes the whole trajectory toward midday rather than moving the peak.

Fix: sharpness is normalized by the marker's own (detached) range, so `beta_scale = 20` is
dimensionless and self-calibrating — a marker added later cannot inherit a silently broken
sharpness the way `bile_acids` did in iter-95. After: cortisol violation 114 → **0**,
sbp 87 → **0**, temp 238 → **0**, acth exact.

### 5.1 What the correct measurement then exposed

With a sharp argmax, `ghrelin` and `leptin` reported *larger* violations (819 and 903 min),
because both rules were mis-windowed and the blunt measurement had been hiding it:

- **ghrelin** — the rule describes "a meal-anticipatory peak before habitual dinner", which
  is a *local* maximum; ghrelin's global 24 h maximum is the nocturnal peak around 01:00
  (Cummings et al. 2001, *Diabetes* 50:1714). A 24 h argmax could never land in 18:00–22:00.
  Re-windowed to 14:00–23:00 → violation **0** (the teacher's pre-dinner peak is at 19:00,
  exactly where the rule wanted it).
- **leptin** — the band was `target_max = 26 h` on a 24 h window, so a quarter of it lay
  outside the trajectory entirely; and with arms starting at 00:00 the nocturnal peak sits
  on the window *boundary*, the one place an argmax cannot be measured. Re-windowed to
  14:00–24:00, which asks the reachable half of the claim.

### 5.2 …and that exposed why leptin has read as inert

Re-windowed, leptin still violated by 446 min. `k_lep = 0.001` (τ = 1000 min) applied to a
24 h circadian target gives a first-order phase lag of `atan(ω/k)/ω = 5.14 h` and retains
`1/√(1+(ω/k)²) = 23%` of the amplitude. **Measured: the teacher's leptin peaked at 07:00
against a 02:00 target, with a realized range of 0.94 against a ±2.0 target.** Plasma
leptin's half-life is ~25–30 min (Klein et al. 1996, *J Clin Invest* 97:2152), i.e.
k ≈ 0.026/min. At `k_lep = 0.025` the lag is 0.6 h and 98% of the amplitude survives —
violation **0**.

**Not fixed:** leptin still has no meal coupling at all, so `leptin_fed_vs_fasted` is
structurally 0.00 against its +2 target. That needs an insulin → leptin term (Saad et al.
1998), which is a new mechanism rather than a rate constant. Left as an open item; it is
one of the two remaining teacher contradictions.

---

## 6. Student side — the two amplifiers the attribution named

One-at-a-time attribution at 04:00 (substitute one teacher component into the student's
state, re-read the rate) said **hr is driven by cortisol, glucose by liver glycogen**.
Section 1 fixes the cortisol channel in the teacher; these two fix the student's ability
to follow it.

### S1 — the anabolic gate leaked, and a learnable threshold cannot be stopped from leaking

`GlycogenFluxHead` synthesis was `softplus(net) · σ((a − a_thresh)/temp)` with `a_thresh`
free (init +0.1). It learned **a_thresh = −0.114**, so at zero gut glucose appearance the
gate sat **63% open**. At the teacher's own 24 h-fast state (liver_glycogen 41.7 g) the
student's net glycogen rate was **+0.159 g/min — it refilled the liver during a fast.**
Nothing downstream could then work: `Gb_fasted` reads how much of the pool is *gone*, so a
pool that never empties defends a glucose level that never falls. Student pre-dawn glucose
slope −0.15 mg/dL/h, against the teacher's −0.74 and the real −1.75.

The physics is not a threshold — glycogen synthase has no substrate to act on when no
glucose is arriving. Synthesis becomes `softplus(net) · relu(a)`: identically zero at zero
appearance as a **structural** guarantee rather than a learned one, linear in the drive
(better-conditioned than a saturating sigmoid), and the same form the teacher already uses.
The insulin dependence stays learnable — insulin is in `x`. `a_thresh`/`log_a_temp` are
deleted rather than clamped: a softplus reparam would keep the gate shut at zero but leaves
the same saturating shape and one more parameter to mis-learn.

This is the same shape error as §2, in the student.

### S2 — cortisol was double-driven in the student; iter-91's teacher fix was never ported

`StressModule` had **both** `cortisol_drive = α·relu(ACTH−typical)` and
`cortisol_diurnal = δ·(1+diurnal)·prod_scale`. The teacher stopped doing exactly this in
iter 91 — ACTH is itself circadian, so a second circadian on cortisol injects the same
rhythm twice, and since the carrier is in [0,2] and δ = softplus(·) ≥ 0 it adds a **strictly
non-negative standing offset** on top. A rectified source term is a production *floor* —
the same shape of error iter-95 found in the mass-action frame. Measured on the iter-95
artifact across the 14 real episodes, the student's cortisol nadir sat at
**9.51 µg/dL** against the teacher's 4.20 and a physiological 3–5 (Weitzman 1971).

Two changes, both mirroring the teacher's iter-91 cascade:
- `cortisol_diurnal` **removed**. The circadian lives in ACTH alone, where the
  SCN→PVN→pituitary pathway puts it. ACTH keeps the [0,2] carrier, which reaches 0 at the
  trough, so ACTH retains a real nadir of its own.
- the ACTH→cortisol drive becomes **proportional** to the secretagogue (`ACTH/typical`)
  rather than rectified at typical. Rectifying made the map *flat* below typical ACTH —
  cortisol could not tell an ACTH of 5 from an ACTH of 12, which is precisely the overnight
  range the nadir lives in.

Two new regression tests assert both properties (`tests/test_stress_module.py`).

### S3 — `mitochondrial_capacity` is now distilled, on the flat reference

The distillation comment treated a flat teacher reference as a reason to leave the marker
out. That has a hole, and the iter-95 artifact walked straight through it: **with no
supervision the student does not hold flat either** — over five eucaloric days it decays
1.00 → 0.85 → 0.73 → 0.63 → 0.54 → 0.46, ~14%/day, toward a mass-action equilibrium near
0.005. "This does not move measurably in a day" is a true statement about a state whose τ
is weeks, and it is exactly the statement the student is violating. The flat reference is
the correct supervision at this protocol length — not a placeholder for the chronic-block
work, which is still owed.

Alongside: `--cold-distill-anchor-long-window` 480 → **1440** (one full-day free-running
arc, `--anchor-long-samples=1`), which is the "supervise the long clock" item iter-95 left
open, and `train.py`'s `cold_distill_markers` default is brought in line with
`_DEFAULT_DISTILL_MARKERS` — it had silently lagged the dispatch recipe since iter 76.

### S4 — calibration shrinkage, scripted but deliberately NOT set here

`scripts/iter96_calibration_sweep.py` sweeps `prior_weight` × `max_norm` × `l2` over the 14
real episodes. These are **eval-time** knobs, so the honest place to set them is on the
artifact that will actually be scored; tuning them against iter-95's weights and hoping
they survive a retrain that changes two module shapes is how a free win becomes a
regression. **Run it after the iter-96 artifact lands and before the benchmark job.**
Projected value from the iter-95 measurement: glucose skill −0.694 → about −0.15, hr
−1.165 → about −1.04. Real and free — and it clears neither blocker on its own.

---

## 7. Where the teacher stands after all of it

`scripts/cohort_teacher_audit.py`, N=12: **contradictions at |z| ≥ 2 fall from 4/37 to
2/39**, and the two survivors are both pre-existing and off the pre-dawn path:

```
-3.04  fasting_breakfast_glucose_morning   glucose  -30.22  vs  -12.00 +/- 6.00
-2.00  leptin_fed_vs_fasted                leptin     0.00  vs   +2.00 +/- 1.00
```

`leptin_fed_vs_fasted` is §5.2's open item (leptin has no meal coupling, so the statistic
is structurally 0). `fasting_breakfast_glucose_morning` — the teacher over-drops morning
glucose when a meal is skipped — was already at −2.85 before any iter-96 change and is
**not** touched here; it points the opposite way to the pre-dawn blocker and deserves its
own diagnosis rather than a coefficient nudged mid-iteration.

---

## 8. What this iteration is expected to do, and what would falsify it

**Primary criteria — measured on the artifact, not the gate:**

1. **Pre-dawn HR.** Student slope over 03:00–06:00 on the 14 real episodes, at the
   prior-mean embedding. Teacher is now +2.89 /h (was +4.08); real is +1.06. Student was
   +5.41 /h. **Accept if the student lands at or below the teacher's +2.89.**
2. **Pre-dawn glucose.** Student slope, same protocol. Teacher −0.74 /h, real −1.75,
   student was −0.15. **Accept if the student reaches at least −0.5 /h** — i.e. it follows
   the teacher it is distilled from instead of losing 80% of it.
3. **Cortisol nadir.** Student minimum over the overnight episodes. Was **9.51 µg/dL**
   against the teacher's 4.20 and a physiological 3–5. **Accept below 7.0.**
4. **Glycogen gate.** Net liver-glycogen rate at the teacher's 24 h-fast state must be
   **≤ 0** (it was +0.159 g/min). This is a structural guarantee now, so a failure here
   means the change did not take, not that training fell short.
5. **Eucaloric stationarity.** Five simulated eucaloric days: liver glycogen must not fall
   below 70 g (student reached 19 g) and `mitochondrial_capacity` must stay above 0.9
   (reached 0.46).

**Guards — a fix that breaks three other things is not a fix:**

- iter-92 meal kinetics: glucose peak 45–60 min, ghrelin nadir −30…−50% at 60–90 min.
- iter-95's wins: biliary axis alive and 5/5 anchors; fasted cascade bhb/insulin ratios.
- `legacy_static` absolute MAPEs must not move materially.
- `cgm_real` glucose skill must not regress below iter-95's −0.694.

**What would falsify the thesis.** If the student's pre-dawn HR slope does *not* fall after
the teacher's dawn rise was cut by a third, then HR in that window is not being driven by
the channel the attribution named, and the remaining error is the sleep→wake transition —
which would make `sleep_hr_frac` and the wake-transition shape the iter-97 target rather
than anything hormonal.

**Known risk, stated in advance.** §1 and §5 both change signals the student has been
trained against for many iterations. The cortisol re-gain is a 7x reduction in one coupling;
if cortisol was silently doing load-bearing work for some *other* marker, that marker will
regress and the cohort audit will show it. The audit is the instrument: 4/37 → 2/39 with
four CV anchors converging is the evidence that it is not happening, but it is measured on
the teacher, and the student is a separate question that only the run answers.

---

## 9. Run plan

Training and benchmarking are **separate jobs** (iter-93 spent 12.2 h of 20 h in the
benchmark; the ruler costs ~21 h on its own).

1. `deploy/deploy.sh` with `TASK_TIMEOUT=158400` (44 h, as iter-95).
2. Training: `--spec=train/spec.json`, artifact
   `gs://grovina-pulse-data/training/jobs/iter96/model.pt`. **Verify from the logs that the
   SPEC recipe is in force, not the argparse defaults** — the log line must read
   "Training: 55 epochs" and "Phase 1 = 30 epochs, Phase 2 = 25", 40 patients.
   **Check completion via the ARTIFACT, never `succeededCount`.**
3. `scripts/iter96_calibration_sweep.py` on the artifact (on grovina-mini, not the laptop).
4. Benchmark: `--benchmark-only` against
   `gs://grovina-pulse-data/benchmarks/benchmark.dataset.iter94.json`, report to
   `training/jobs/iter96/benchmark-report.json`, with the env vars step 3 selected.
   Compare against `training/jobs/iter95/benchmark-report.json`.

Probes, all re-runnable: `scripts/iter94_student_fast_probe.py`,
`scripts/iter95_student_biliary_probe.py`, `scripts/iter95_head_shape_audit.py`,
`scripts/iter96_dawn_probe.py`, `scripts/cohort_teacher_audit.py`.
