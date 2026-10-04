# Plan — the person model, the frozen constants, and the missing hubs

*Written 2026-10-04, against HEAD `ea16d0d` (iter 109). Companion to
`docs/review-2026-09-04.md` (the iter-96 review, whose A–E list iter 97 executed)
and `docs/modeling-state.md` (the completeness map). This document is the
successor to both: the review's three diseases are fixed, and what the
2026-10-03 analysis found is a **fourth** that iters 97–109 made worse while
fixing the third.*

## 0. The picture in one paragraph

Layer 1 of the PRD hierarchy is now real: carbon closes, the gut kernel is a
normalized density, SBP > DBP and HRV > 0 hold in the coordinates rather than in
a docstring, Gb is a fixed point, and the ruler is honest. What iters 98–109
did to get there was **replace learned physiology with the teacher's own laws
and constants** — 68,293 parameters to 37,343, with ghrelin, leptin, the whole
HPA cascade, ketogenesis, lipolysis, glucagon and the liver's carbon budget now
structural. Each of those was individually the right call against a flat
gradient direction. Collectively they moved the model *down* the PRD hierarchy:
quantities that the PRD assigns to layer 3 (medical knowledge — "priors, not
constraints") and layer 4 (individual variation — "learned from this person")
are now layer 1 (imposed architecture). The model is a differentiable copy of
its teacher, and a copy cannot overturn the thing it copies.

The fourth disease is therefore: **the model has lost the ability to represent a
person.** Its most important individual axis — insulin sensitivity, the PRD's own
example — does not exist. Nine per-patient heads have no supervision path at
all. The embedding is 40 memorized codes in 32 dimensions, three different
"default persons" are in use at once, and two signals actively pull every
patient toward the median one.

## 1. Principles this plan is held to

Beyond the PRD's four layers, three rules decide every choice below.

**Right by construction beats fitted.** Where an invariant is physics
(conservation, positivity, ordering) it belongs in the coordinates or the
functional form, not in a loss. This is the iter-97 lesson and it holds.

**Flat is not the same as fixed.** When a parameter's gradient cancels at the
fixed point, the remedy is a *prior plus the evidence that identifies it*, not a
constant. A δ-prior is infinitely confident; the PRD forbids exactly that
("strong claims become tight constraints, weak claims loose ones"). A frozen
constant also deletes real between-person variation: the teacher varies Gnb,
k_ffa, uptake_ii and the gluconeogenic share per patient, and iters 106–109
replaced all four with one number.

**The decoder family must match the generative family.** The teacher draws
almost every parameter lognormally, so the median person is `PatientParams()`
and the population *mean* is above it (Si mean/median 1.14, Ib 1.08, HRV₀ 1.09).
A student that decodes a setpoint as `center + scale·tanh(head)` has
E[setpoint] = center for a zero-mean code — it cannot be simultaneously right at
the median and in the mean. Decoding positive quantities in log space makes both
true by Jensen, with no loss term. This single observation resolves the "is zero
the median or the mean" tension that otherwise forces a choice.

## 2. What zero means, decided

**Zero is the median person, and the embedding table's mean is pinned to zero.**

Both halves are needed, and together they are consistent *only* with log-space
decoders (§1). The consequences, each of which is a work item:

- `PatientParams()` is the median patient (measured: it sits at the 44th–57th
  percentile of every daily-mean marker), so supervising zero against it is
  correct, and the textbook scenarios and the lab keep their meaning.
- The calibration prior becomes `N(0, diag σ)` by construction instead of a
  post-hoc fit to whatever cloud the reconstruction loss left behind. The soft
  norm penalty and the hard clamp, which are already centred on zero, stop
  disagreeing with the prior mean, which is currently not.
- Every consumer uses the same person. The server's user-id-seeded random vector
  goes; a new user gets the median person and a band.
- Population statistics stop being estimated by "mean of 2 random rows + zero".

## 3. Work items

Grouped by wave. Within a wave, items are independent and can land in parallel.
Each carries its **invariant** (what becomes true by construction) or its
**evidence** (what identifies it), because an item with neither is how we got
here.

### Wave A — one person, consistently (no new states; no `STATE_DIM` change)

| # | Item | Invariant / evidence |
|---|---|---|
| A1 | Decode every positive per-person setpoint in log space: HR₀, SBP₀, DBP₀, RR₀, Gb (T0 and SpO₂ stay additive — the teacher draws those additively) | E[setpoint] > setpoint(0) by Jensen, matching the teacher's lognormal draws; median and mean both right with no loss term |
| A2 | Pin the table mean: `embedding_prior` penalises `‖mean(E)‖²` at a weight that makes it a constraint, not a nudge; keep the existing per-row norm term weak | prior mean = 0 by construction; `_embedding_prior_mean` stops disagreeing with the clamp and the soft norm |
| A3 | Person-level constants: one `model.person_constants(embedding)` owning basal insulin Ib and body mass, consumed by metabolic AND appetite | one person has one basal insulin (today: two untied heads); removes the appetite copy |
| A4 | Supervise every per-person head the teacher has ground truth for: Ib, body mass, FFA_b, mito setpoint, cort_b, RR₀, SpO₂₀ — extend `SetpointSupervisionSignal` | each head gets the evidence that identifies it; kills the Ra↔mass confound (measured: ×1.2 on both changes glucose 0.18 mg/dL) |
| A5 | Delete the heads the teacher does not vary and nothing supervises: HPA phase, CCK basal | no unsupervised per-person authority (the Gnb-drift failure mode). **Not finished:** the `cck` and `gallbladder_bile` species heads still read the embedding while `randomize_params` draws *no* hepatobiliary parameter at all, so a person's resting CCK is still free against a now-pinned gate reference. Those are species heads rather than baseline nets, so they need the C6 couplings (something must read a bile state) before deleting or supervising them is the right move |
| A6 | Insulin sweep and gut sweep supervise the zero embedding only (they score against `PatientParams()` rates and kernel) | removes a 0.30-weight and a 0.10-weight pull of sampled patients toward the median; per-row targets return in B4 |
| A7 | Cohort population batches: drop zero, debias the batch-mean loss by its own sampling variance (**landed, floored — see the note below**) | the loss stops rewarding between-person shrinkage (the term is ≈(2/9)σ²_between/SEM², 1.4–14 per spec). Rule hinges are NOT batch-mean statistics — each member's own violation is scored — so the debias does not apply and zero stays in the rule batch, because a rule holds for the median person too |
| A8 | Server default → median person; `last_good.pt` carries the prior stats; `--init-from` loads `embeddings_state` | one default person everywhere; a rolling checkpoint re-scores on its own prior; warm start keeps patient identity |
| A9 | Bioavailability bounded: `f_bio = MG_DL_PER_G · σ(·)` ≤ 1 per channel | absorbed ≤ ingested by construction (today softplus: >1 creates carbon, <1 deletes it unbooked) |
| A10 | Delete the `Ra` gain and `ra_baseline_net` | meal amplitude has one per-person gain (f_bio) and one known scale (V_G), not three multiplying ones |
| A11 | `hepatic_output` is release into plasma (exclude `gng_divert`) | the marker means what the tracer literature the HGO cohort cites measures |

**A7's floor is a deliberate, reversible choice, and the alternative is written and
tested.** An unbiased estimate of a square must sometimes be negative, and a loss
may not be, so `relu` clips it — which reintroduces part of the shrinkage force the
debias removes. Measured on Gaussian members as the surviving share of the
un-debiased force at B = 2: **0.64 at zero bias (exactly 2/π), 0.56 / 0.22 / 0.03 at
one / two / three between-person sds.** Dropping the zero member even makes the
zero-bias case *worse* for the seven `n_arm = 1` specs (force 0.44 → 0.64 sd/SEM²),
because zero had been diluting the variance term; in a toy SGD against a true sd of
3, the spread settles at 2.08 under the old batch, 1.83 floored, and 3.00 with an
unbiased gradient.

The straight-through form — `est + (relu(est) − est).detach()`, value floored for the
adaptive-weight EMA and the logs, gradient unbiased — is a one-line change and a
ready patch. **It is not adopted, for a reason that is about estimator quality, not
caution:** the term being subtracted is `s²/B`, and for exactly the specs where the
bias was largest (`n_arm = 1`, whose σ is a published standard error while
individuals spread 2.5–8× wider, so `v/SEM²` is 6–64) it is estimated from **1 degree
of freedom at B = 2**. Passing that straight into the gradient injects noise of the
same order as the signal it corrects. The floor clips the noise, and clips the
correct negative gradient with it. So the honest reading is that **at B = 2–4 the
debias cannot be made both unbiased and quiet, and the real fix is more members** —
D2's deterministic quadrature over `N(0, I)`, where the "sampling variance" is zero
by construction and the question disappears. Until then: keep the floor, raise
`--cohort-sample-patients` above 2, and flip to straight-through in one line if a
retrain shows the residual shrinkage dominating (watch `setpoint_supervision`'s
per-marker MAE and the spread of the decoded setpoints).

### Wave B — right by construction (architecture; needs a retrain to mean anything)

| # | Item | Invariant / evidence |
|---|---|---|
| B1 | **Per-person insulin sensitivity.** Log-space heads for muscle Si, hepatic insulin sensitivity (the glycogenolysis/GNG gate K), β-cell γ, insulin clearance k_ins | The teacher varies all four per patient (Si σ0.5 loading ir −0.70; glyc_ins_K σ0.25 ir +0.40; γ σ0.3; n σ0.3), so A4's mechanism supervises them directly. **Landed, with this row's own number corrected.** Si alone is **22.5 %** of between-person glucose *incremental* AUC and 20.7 % of insulin iAUC, decile ratio 1.90× / 2.17× (N=150). The "34 % / 31 %" this row first claimed was measured on TOTAL post-meal AUC, which also carries the fasting level — and Gb loads +0.50 on the same insulin-resistance latent that Si loads −0.70 on, so part of that share was Gb, not Si. The conclusion is unchanged (still the largest single axis by a wide margin) but the headline was confounded through the shared latent. The student now realizes **1.77×** between those deciles against the teacher's own pure-Si sweep at 1.81× |
| B2 | **Exponential (ETD1) stepping of the linear part.** A module may report a decay rate alongside its rate; the integrator steps `x + rate·(1−e^{−k·dt})/k`, which is Euler as k→0 and exact for a pure relaxation | literature time constants are what gets simulated, and the trajectory stops depending on `dt` (today: effective rate +19 % at the CVS default k=0.3, +101 % at its bound 0.8) |
| B3 | **Second-phase insulin.** The teacher's delayed-proportional potentiator state (Toffolo & Cobelli) | OGTT 120-min insulin; the student currently has no way to hold insulin up while glucose falls |
| B4 | Per-row sweep targets: carry each patient's `PatientParams` into the dataset so the insulin/gut sweeps can score a row against its own teacher | restores per-patient absorption and rate supervision that A6 removes |
| B5 | Priors replace the iter-106/108/109 freezes: k_ffa, uptake_ii, the gluconeogenic share, Gnb become learnable with log-normal/Beta priors carrying the cited uncertainty, plus the tracer evidence that identifies them | a flat direction with a prior does not drift; a frozen one cannot learn. Restores the teacher's own per-patient spread in all four |
| B6 | Identifiability audit as a gate artifact: eigen-spectrum of the Fisher information of the total loss w.r.t. the population scalars and the per-person heads | every flat direction is covered by evidence or a prior, and we find out *before* the next freeze |

### Wave C — the missing hubs (new states; `STATE_DIM` grows; teacher must change in step)

Each hub lands as: teacher mechanism → student mechanism → supervision
(cohort/rule/anchor) → benchmark dataset migration. A hub without supervision is
the inert-glycogen failure repeated, so none of these are "add the state and
move on".

| # | Hub | New state(s) | What it connects | Why it is the next one |
|---|---|---|---|---|
| C1 | **Sympathoadrenal arm** (lives in the Stress module: the HPA axis and the sympathoadrenal system are the two arms of one central stress response, so this is not a new organ) | `sns_tone` (neural, τ≈1–2 min), `epinephrine` (adrenal medullary, pg/mL, τ≈2 min) | in: activity, hypoglycaemia, the stress input (C8), sleep; out: HR, HRV (the vagal readout), BP, HGO, lipolysis, glucagon, α2-inhibition of insulin secretion, thermogenesis | Wearable HR/HRV is the densest signal a user has, and this is what turns it into metabolic information. Exercise and hypoglycaemia are both broken without it: during a bout HGO and FFA do not rise, and muscle uptake is instead switched off below 0.8·Gb; counter-regulation is glucagon plus slow CRH→cortisol with no fast arm. **No baroreflex term** — see the note below |
| C2 | **Energy expenditure** | none — a flux node (VO₂, VCO₂, RQ) exported as coupling | BMR + activity + DIT + thermogenesis → heat, CO₂, O₂ demand, substrate oxidation | PRD says RR tracks CO₂ production; today RR reads lactate and temperature. Closes the energy ledger against the carbon ledger (fat mass currently counts carbon that went to glycogen) |
| C3 | **Circadian phase + sleep homeostat** | `circ_phase_offset` (hours of internal-vs-external phase, bounded, so there is no angle to wrap), `sleep_debt` (hours, ≥ 0) | sleep midpoint → phase offset; debt → Si, cortisol, ghrelin/leptin, HR/BP | Replaces three imposed waveforms (HPA drive, ghrelin's 09/13/20 anticipation, leptin's 02:00 cosine) with one entrained clock every module reads as `hour − offset` — the PRD's "learned consequence of SCN input, not an imposed waveform". It is also where A5's deleted per-person HPA phase head belongs: phase becomes a state driven by the person's own sleep behaviour, not free per-person authority. The sleep cohort's +6 mg/dL has nowhere to live today (measured: 5 nights of 4 h vs 8 h changes next-day glucose by <0.05 %) |
| C4 | **Gastric emptying as the shared upstream** | `gastric_fat`, `gastric_protein`, `gastric_carb` (grams in the stomach) | emptying → duodenal delivery → absorption; slowed by GLP-1, CCK, hyperglycaemia, meal fat and total load | One mechanism replaces two independent kernels in both models. It also **upgrades** the conservation argument rather than losing it: iter 97 made `∫K = f_bio·mass` true by parameterizing a fixed normalized density, which conserves mass only because the shape is frozen; an explicit stomach compartment conserves it by accounting, and can therefore be *modulated* — which a fixed kernel cannot be. Carb absorption currently ignores the meal's fat and size (a pizza absorbs like juice), and no GLP-1-agonist counterfactual is expressible without it |
| C5 | **Incretin and islet** | `gip` | GIP (the larger incretin), amino acids → insulin, GLP-1 → glucagon suppression | Protein raises insulin only indirectly today, because secretion is gated by `relu(G − Gb)`; glucagon has no GLP-1 term |
| C6 | **Hepatobiliary links out** | none | intestinal bile → fat absorption; bile acids → GLP-1 (TGR5) | Four states nothing reads: the module contributes zero to coupling amplification |
| C7 | **Renal glucose + Cori cycle** | none — flux terms | glycosuria above ~180 mg/dL; lactate → gluconeogenic substrate | carbon closure at high glucose, and the SGLT2 counterfactual |
| C8 | **Inputs the PRD already promises** | none — external inputs | user-reported stress; a pharmacology channel (GLP-1 RA, metformin, SGLT2i) | PRD lists stress as a Stress-module input and it was never implemented; today stress and drugs can only be explained by changing *who the person is* |
| C9 | **Slow states for adaptation** | `fitness`, `liver_fat` | fitness → Si, HR₀, HRV₀, lactate threshold; liver fat → hepatic Si, FFA_b | The north star. Measured today: 8 weeks of training moves mito capacity +0.8 % (plan expects ~30 %) and resting HR not at all |

**The supervision for C1 and C3 already exists in the literature, and it is
quantitative.** Two anchors, both of which also say something about the
architecture:

- **The counter-regulatory hierarchy** (Cryer et al. 1987, *J Clin Invest*
  [112884](https://www.jci.org/articles/view/112884)): glycaemic thresholds for
  activation are epinephrine **69 ± 2** mg/dL, glucagon **68 ± 2**, growth
  hormone **66 ± 2**, cortisol **58 ± 3**, symptoms **53 ± 2**. This is a
  ranked constraint, which is the strongest kind of weak evidence — an ordering
  is cheap to encode as hinges and hard to satisfy by accident. It also convicts
  the current model: both teacher and student fire the hypoglycaemia term into
  **CRH** at an absolute 70 mg/dL (`stress.py`, `full_body.py` `hypo_acth`), so
  cortisol is the *first* responder at 70 when the literature puts it *last* at
  58, and the fast arm that should lead at 69 does not exist. C1 fixes the
  ordering and the mechanism together, and `_hypo_raw` — the one parameter group
  no signal currently reaches, because it only acts below 70 mg/dL — gets
  evidence for the first time.
- **Sleep restriction is a sensitivity effect, split by tissue** (Donga et al.
  2010, hyperinsulinaemic euglycaemic clamp, n=9: insulin sensitivity
  **−19 to −25 %** after a single 4 h night; a companion 4 h-vs-8 h clamp study
  reports whole-body **−25 %**, peripheral **−29 %**, and **hepatic
  essentially unchanged** with the gluconeogenic percentage rising). Two things
  follow. First, C3's sleep debt should act on insulin sensitivity, not on
  glucose directly — which is why the existing `sleep_restriction_next_day_glucose`
  cohort (+6 mg/dL) has no mechanism to work through today. Second, the effect is
  *peripheral and not hepatic*, so it can only be expressed if muscle and hepatic
  insulin sensitivity are separate quantities — exactly the split B1 introduces.
  B1 and C3 are therefore the same change seen from two directions, and B1 should
  land first.

**C1 carries no baroreflex, deliberately.** `full_body.py:1737-1765` records an
evaluated-and-rejected baroreflex term with four measured reasons: at minute
resolution the reflex equilibrates inside one step (so it is already folded into
the effective `act_hr_gain`/`k_hr`); there is nothing to reflex against, because
the activity-independent part of SBP has sd 0.40 mmHg in steady periods (a ~1 bpm
effect, below the noise floor); the residual that does exist is a first-order lag
artifact concentrated at exercise transitions (peak ±17.7 mmHg), so feeding it to
`−k_baro·resid` would add ~18 bpm/min to the exercise rise and subtract the same
from recovery, corrupting an HRR1 that currently matches Cole 1999; and the sign
is wrong anyway, since central command resets the operating point upward so HR and
SBP co-rise (realized corr +0.964). That analysis stands, and the sbp→hr prior is
correctly absent. A baroreflex becomes meaningful only once the model has
something to buffer — orthostasis, a vasoactive drug (C8), or blood-volume change
— and it should be added with those, not before. The PRD lists the baroreflex
under medical knowledge, which is an argument for a prior once the mechanism is
observable, not for a term that fires on a lag artifact.

Eight coupling priors are already registered for edges the model cannot express
(cortisol→FFA, glucagon→FFA, mito→FFA, cortisol→ACTH, GLP-1→ghrelin,
leptin→ghrelin, glucose→cortisol, glucose→ACTH). Their loss is a gradient-free
constant today. C1/C3/C5/C6 make most of them real; the three HPA ones should be
rerouted to **CRH**, which is where both models actually act (cortisol feeds back
on `crh_target`; the hypoglycaemia term enters `dCRH`), and the FFA ones become
real when C1 gives lipolysis its epinephrine and glucagon drives.

*A hypothesis worth recording as refuted.* One-sided couplings — glucose→glucagon
is `α·relu(Gb − G)/Gb`, exactly zero above the patient's own setpoint — have a
true sensitivity of zero wherever they are inactive, and `coupling_band_hinge`
declares a band with `lo > 0`, so it should in principle penalise correct
physiology at every fed sample. Measured on a perturbed model: at G = 140 the
normalized sensitivity is exactly 0.00000 and the hinge is **0.0014**, because
`lo` (0.001) is tiny against the band width (0.019). So the effect is real in
form and negligible in size, and is not worth a change on its own; the same probe
showed the *active* regime at 2.8× above the declared band, which is just an
untrained model. Setting `lo = 0` for gated edges is still the more honest
declaration and can ride along with the reroute.

### Wave D — the person as an inference problem

| # | Item | Invariant / evidence |
|---|---|---|
| D1 | Low-dimensional named latent (≈6–10 factors: insulin resistance, β-cell, fitness, adiposity, chronotype, HPA reactivity, gastric emptying) with a fresh teacher population every epoch and an amortized encoder `q(z | observations)` | no table, so no warm-start identity problem; the calibration prior is exact; the encoder *is* the PRD's "future: amortized inference" |
| D2 | Population statistics by deterministic quadrature over `N(0, I)` (sigma points), not by a 3-member random batch | the cohort loss stops carrying sampling variance that only shrinkage reduces |
| D3 | Calibration estimates `(z, x₀, disturbances)`, with x₀ spun up from the person's own quasi-steady state | a short night or an unlogged meal stops being absorbed into *who the person is* |
| D4 | Measurement models: CGM interstitial lag + noise; learned subjective-report likelihoods including negative reports | PRD "epistemological humility" with the sensor physics in the right place |
| D5 | An amplification metric: calibrate on marker set A, score held-out set B | the PRD success criterion "coupling amplifies information" currently has no number |
| D6 | Multi-rate integration (vitals 1 min, metabolic 1–5 min, slow pools hourly) | the only affordable route to weeks–months; a 16-week protocol is 160k single-minute steps per patient |

### Wave E — grounding

Teacher-side defects the analysis measured, plus the real-data gap.

- **High-Gb ketosis.** `Hep_b = uptake·Gb·V_G` with `Gng_b` fixed near 1.0 books
  a person's whole elevated fasting output as glycogenolysis. On 190 g carb/day a
  Gb-130 patient ends day 4 at 19 g liver glycogen and 1.6 mM BHB; ~20 % of
  sampled patients exceed 1 mM. This contradicts the teacher's own Magnusson-1992
  citation (the T2D excess is gluconeogenic). The student's fixed 50/50 share has
  the same shape, milder.
  **This document's first prescription — "scale `Gng_b` with Gb" — was wrong, and
  was measured to be wrong.** Proportional scaling across the whole range
  re-creates the iter-93 / review-3.3 failure that iter 97 fixed: the Gb-130
  patient then fasts to 97 mg/dL at 48 h (iter 93's exact number) and the Gb-70
  patient to 60.7, breaking `glucose_fasting_floor` at 63.7 in the 24 h arm. The
  excess must be split **one-sidedly about the population-typical EGP**:
  `Gng_b = f_gng·min(Hep_b, 2.0) + relu(Hep_b − 2.0)`, so glycogenolysis absorbs
  the deficit (iter 97's Cahill argument for an absolute prolonged-fast floor) and
  gluconeogenesis absorbs the excess (Magnusson measured GNG higher *and* net
  hepatic glycogenolysis lower in T2D). The sampled quantity becomes the *fraction*
  `f_gng`, which is also what B5 needs a Beta prior over. Measured over 80
  patients: day-4 BHB above 1 mM **26.3 % → 2.5 %**, corr(Gb, BHB) **+0.75 →
  +0.12**, the Gb-130 patient's `extended_fast_bhb_overnight` **+6.39σ → +0.36σ**.
  **The student still carries the defect**: `_F_GNG = 0.5` multiplies the patient's
  own `egp_b = k_ii·gb`, so its fed high-Gb patient still spends its pool. Mirroring
  the teacher's split there is parameter-free and differentiable.
- **A separate defect this uncovered, which this plan mis-attributed.** Wave E
  expected the fix to stop the default patient's liver glycogen drifting from 100 g.
  It structurally cannot: the new split is an *identity* at the typical EGP, so the
  default patient is bit-identical. The drift (−6 g/day, decaying to an ~82 g
  asymptote, bounded not runaway) is a pre-existing imbalance in
  `glyc_syn_frac_L` / `LGly_b` / the basal glycogenolytic flux, and
  `extended_fast_liver_glycogen_level` is **+0.99σ for the default patient both
  before and after**. The old Gb-coupled depletion had been *cancelling* that error
  in the population mean, which is why no anchor caught it. Its own item, since
  fixing it moves the 24 h-fast reference numbers.
- **A third, in the residual.** The worst remaining fed-ketotic patient has Gb 96 —
  a normal fasting glucose. `LGly_b` is sampled over 70–130 g against a fixed fed
  carbohydrate load, so a large-pool patient permanently sits at ~45 % of its *own*
  declared pool and `keto_glyc_gain`'s gate `1 + 13·(1 − LGly/LGly_b)` reads that as
  a fast. The gate should read an absolute pool level, or `LGly_b` should co-vary
  with intake.
- **`extended_fast_liver_glycogen_level` may be the wrong anchor for this
  population.** It is a healthy-cohort mean (60 ± 20 g at 16–22 h) applied to a
  population spanning Gb 70–130. After the fix the population mean is 88 g (+1.40σ)
  *because* high-Gb patients correctly hold their glycogen — which is what Magnusson
  measured in T2D. Either the anchor needs a cohort restriction or the pool needs
  re-sizing; the two errors were cancelling.
- Respiratory rate equilibrated at RR₀ + 50·activity (63 /min at 0.9); **fixed**,
  by moving both drives inside the relaxation as equilibrium offsets (the ×10 hidden
  in 1/k_rr is how "5 /min" became 50) and saturating the activity term (Hill n=2),
  since ventilation below the ventilatory threshold is carried mostly by tidal
  volume. Equilibrium at act 0.9: **63.1 → 35.0** /min; minutes above the marker's
  declared max of 40 across six 14-day episodes: **926 → 0**; resting value exact.
- The habitual meal hours (9/13/20) sit *after* the generated meals (7–9, 12–14,
  18–20), so most "anticipatory" ghrelin arrives post-meal. C3 subsumes this.
- `Cort_b` (12) is 3 µg/dL above the default patient's actual 24 h mean (8.95),
  so every cortisol gate runs at a standing offset.
- Three sampled HPA parameters are never read; `Sg` is computed and unused.
- Schedule diversity: every teacher person eats and sleeps on one template, and
  activity is zero except one bout on 60 % of days. No chronotype variation, and
  none of the low-level daily movement a wearable actually reports.
- Real data is one subject's 14 nights.
  [CGMacros](https://physionet.org/content/cgmacros/) (45 subjects across
  healthy/prediabetes/T2D, two CGMs, Fitbit, meal macros) and the
  [BIG IDEAs Lab set](https://physionet.org/content/big-ideas-glycemic-wearable/)
  (Dexcom G6 + Empatica E4, standardized breakfasts) are the obvious next rulers.

## 4. How each wave is judged

**Wave A and B1–B3 are structural claims, provable without a trained
checkpoint**: an invariant either holds for every embedding and every protocol or
it does not. Those are tests, and they are the acceptance criteria.

**Everything that changes what training can learn needs a retrain to mean
anything.** No trained artifact is reachable from this session (GCS returns 401),
so nothing here can be scored against iter 109's gate from here. The honest
sequencing is:

1. Land A + B1–B3 behind tests, with the invariant probes as the gate.
2. Retrain once on Cloud Run; compare against iter 109 on a frozen ruler with a
   matching `ruler_fingerprint`.
3. Only then decide whether B5's priors move the frozen constants anywhere, and
   whether Wave C's hubs are reachable by the supervision we have.

**Every pre-change artifact is now strictly unloadable, and that is correct.**
A5 removed `stress.phase_proj.*` and `hepatobiliary.cck_baseline_net.*`, A10
removed `metabolic.log_ra` / `ra_baseline_net`, so `from_checkpoint(strict=True)`
raises "Unexpected key(s)" on anything through iter 109. That breaks
`--resume-from`, the server's `MODEL_URI`, `diagnostics/probe.py`'s loader (and
therefore every diagnostics CLI and most `scripts/`), and several scripts' raw
strict loads. **No key-dropping shim should be added**: A1 re-decodes Gb, HR, DBP
and RR in log space, so an old artifact's per-person heads were fitted in a
different frame, and silently dropping the stale keys would produce a
plausible-looking model that is wrong rather than a loud failure. `--init-from`
(strict=False) still works and prints the dropped keys, which is the right route
for seeding a retrain. One case was fixed rather than left: `lab_sim.py` caught
every load failure and rendered UNTRAINED cold weights with `trained: False`
buried in its metadata — the fallback is fine, the silence was not.

**Wave C grows `STATE_DIM`**, which invalidates every checkpoint and every
benchmark `initial_state` array (`scripts/migrate_benchmark_initial_state.py` is
the migration path). Each hub therefore lands with its teacher mechanism, its
supervision, and its dataset migration in one piece — never the state alone.

**The falsifier for this plan as a whole.** If, after A + B1–B3 and a retrain,
per-patient recovery of Gb and Si is still no better than predicting the
population mean, then the person model is not the fourth disease and the next
iteration is about the *evidence* a calibration window carries, not about how the
person is parameterized.
