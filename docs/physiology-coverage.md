# Physiology coverage matrix

## Status as of iter 69 (2026-05-26)

The breadth-first floor goal — every marker gets ≥1 uncontroversial
anchor — has been met since iter 66's order-of-magnitude rule
expansion (60 rules across 10 cohort arms, vs 5 / 7 at the time this
doc was first written). Every observable marker plus the internal-
state markers has ≥1 rule; lactate, hepatic_output, HR, HRV, RR,
SpO2, temp, and leptin moved from zero to 2–5 each. The matrix below
remains the conceptual scorecard; **the live rule registry in
`pulse/knowledge/physiology_rules.py` is the source
of truth** for current coverage.

Structural coverage updates since this doc was written:
- **iter 64** closed the largest coupling gap: ACTH→cortisol is now
  encoded as an architectural prior in the StressModule (Move B
  compact from `architecture-roadmap.md`), not just as a coupling-
  graph sign. Cortisol gained its own diurnal cosinor term in iter 66.
- **iter 69** finally landed **Move B FULL** — `crh` added as a new
  internal marker (idx 22, STATE_DIM 22→23), StressModule cascade
  restructured to `diurnal → CRH → ACTH → cortisol` with cortisol
  negative feedback rerouted from ACTH to CRH (anatomically correct).
  But iter 69 regressed across most markers (overall MAPE 0.0976 →
  0.146; acth 0.091 → 0.761 — the new mechanism params likely sat
  near softplus(−3) init and the iter-64-proven γ·diurnal ACTH drive
  was diluted). See `architecture-roadmap.md` chronicle.
- **iter 69 also added a multi-marker dose-response axis**
  (`DoseResponseSignal` extended with `MarkerDoseTarget` for slope
  and ranking-monotonicity modes) — the supervision surface that
  attacks amplitude scaling, which `meal_dose_response 0/3` at iter
  68 r8 had cleanly diagnosed as missing.

The remaining named gaps (cold-model coverage, coupling-prior depth,
latent reservoirs for gut hormones / adipose FFA pool, and the
multi-timescale Move D north star — see `multi-timescale-plan.md`)
are still open.

## Why this doc exists

Iters 55-57 chased one mechanism (slow-state glycogen) hard. That
discipline works for *making a mechanism correct*, but it leaves a
strategic question unanswered: **is the chunk of physiology we already
model actually complete enough to be a trustworthy reference for
everything built on top of it?**

The honest answer today is no — and not for aesthetic reasons. 12 of 22
state variables have zero population-level supervision. The embedding
bottleneck (see `dead-pathways.md`) freely reshuffles representational
capacity between unanchored markers: ghrelin swings +0.71 MAPE while
glp1 recovers −0.46 *in the same iter*, uncorrelated with the change
under test. Every ~12-15 h training iteration reads its signal through
that noise. **An unanchored marker is not just unmodelled — it is an
active source of attribution noise that makes the whole model an
unreliable reference.**

This doc is the scorecard that makes "comprehensive" measurable. It
inventories what physiological knowledge is *actually encoded* in each
of the four supervision surfaces, marks the gaps, and prescribes a
breadth-first floor (every marker gets ≥1 uncontroversial anchor)
before any depth campaign.

Method: for each marker, four columns —
- **cohort** — `CohortStatisticSpec` population-mean/delta (the
  `differentiability` surface: literature RCT → loss).
- **rule** — `PhysiologyRule` per-trajectory hinge (qualitative shape).
- **coupling** — `CouplingPrior` signed graph edge (sign imposed,
  strength learned).
- **cold** — does `simulate_full_body` model it (cold-distill target).

A cell is either a cited real symbol (covered) or `GAP` + the
uncontroversial textbook fact that should fill it.

## The matrix

Inventory as of commit `feb1b786` (20 cohort specs, 5 physiology
rules, 8 coupling priors).

### Metabolic module

| marker | cohort | rule | coupling | cold |
|---|---|---|---|---|
| glucose | ✓ ×6 (`fasting_breakfast_glucose_morning`, `extended_fast_glucose_morning`, `sleep_restriction_next_day_glucose`, `ogtt_75g_glucose_peak`, `ogtt_75g_glucose_120min`, `small_carb_glucose_peak`) | GAP — no shape rule (e.g. *glucose returns to baseline by 120 min post-OGTT*) | ✓ in/out (`glucose↔insulin`, `glucose→glucagon`) | ✓ Bergman |
| insulin | ✓ ×3 (`extended_fast_insulin_basal`, `ogtt_75g_insulin_peak`, `ogtt_75g_insulin_mean_3h`) | GAP — *insulin peak follows glucose peak within 30 min* | ✓ (`glucose→insulin`, `insulin→glucose/hepatic_output/ghrelin`) | ✓ Bergman |
| glucagon | ✓ ×2 (`ogtt_glucagon_suppression`, `mixed_meal_glucagon_suppression`) | ✓ `glucagon_falls_postprandial` | ✓ (`glucose→glucagon`) | ✓ |
| ffa | ✓ ×2 (`extended_fast_ffa_overnight`, `meal_ffa_suppression`) | ✓ `ffa_inverse_to_insulin_postprandial` | GAP — `insulin→ffa` sign −1 (antilipolysis) is textbook, unencoded | ✓ |
| bhb | ✓ ×1 (`extended_fast_bhb_overnight`) | ✓ `bhb_rises_during_fast` | GAP — `ffa→bhb` +1 (hepatic ketogenesis from FFA), `glucagon→bhb` +1 | ✓ |
| lactate | **GAP** — no spec | **GAP** — *lactate rises during exertion, clears in recovery* | **GAP** — `activity→lactate` +1 | ✓ (term exists) |
| hepatic_output | ✓ ×1 (`meal_hgo_suppression`) | GAP — *HGO suppressed by insulin, raised by glucagon/cortisol* | ✓ (`insulin/cortisol→hepatic_output`) | ✓ |
| liver_glycogen | ✓ ×1 (`extended_fast_liver_glycogen_overnight`) | GAP — *glycogen depletion precedes/【correlates with】bhb rise in fast* | GAP — `liver_glycogen→hepatic_output` +1 (substrate for gluconeogenesis) | ✗ padded (by design — flux-head learned) |
| muscle_glycogen | **GAP** | **GAP** — *muscle glycogen spent in exercise, spared in resting fast* | **GAP** — `activity→muscle_glycogen` −1 | ✗ padded |
| mitochondrial_capacity | **GAP** | **GAP** — *rises ≥30 % over 6-8 wk aerobic training* (Holloszy 1967) | **GAP** — `mito→ffa/lactate` (fat-ox capacity, lactate clearance) | ✗ padded |

### Appetite module

| marker | cohort | rule | coupling | cold |
|---|---|---|---|---|
| ghrelin | ✓ ×1 (`meal_ghrelin_suppression`) | ✓ `ghrelin_falls_after_meal` | ✓ (`insulin→ghrelin` −1) | ✓ |
| leptin | **GAP** — *fasted < fed; tracks adiposity/energy balance; circadian nadir ~noon, peak ~midnight* (Sinha 1996) | **GAP** | **GAP** — `insulin→leptin` +1 (postprandial), `leptin→ghrelin` −1 | ✓ |
| glp1 | ✓ ×1 (`large_meal_glp1_peak`) | GAP — *GLP-1 rises within 15 min of nutrient entry* | **GAP** — `glp1→insulin` +1 (incretin effect — a load-bearing missing edge) | ✓ |

### Stress / HPA module

| marker | cohort | rule | coupling | cold |
|---|---|---|---|---|
| cortisol | **GAP** — *CAR: +50-75 % within 30-45 min of waking* (Pruessner 1997); *circadian amplitude ~5× trough→peak* | ✓ `cortisol_morning_peak` | ✓ out (`cortisol→glucose/hepatic_output/hr`) | ✓ ACTH→cortisol |
| acth | **GAP** — *ACTH precedes cortisol by ~15 min; pulsatile, circadian* | **GAP** — *ACTH peak precedes cortisol peak* | **GAP** — `acth→cortisol` +1 (the core HPA edge — unencoded as a prior!) | ✓ |

### Cardiovascular module

| marker | cohort | rule | coupling | cold |
|---|---|---|---|---|
| hr | ✓ ×2 (`postprandial_hr_rise`, `sleep_hr_dip`) | GAP — *HR rises ≥15 bpm in moderate exercise, recovers <15 min post* | ✓ in (`cortisol→hr`) | ✓ |
| hrv | **GAP** — *HRV drops with sympathetic activation (exercise/stress), rises in sleep* (Task Force 1996) | **GAP** — *HRV inversely tracks HR within-subject* | **GAP** — `activity→hrv` −1, `hr→hrv` −1 | ✓ |
| sbp | **GAP** — *nocturnal dip 10-20 % vs daytime* (dipper pattern); *rises with exercise/stress* | **GAP** — *SBP dips during sleep; rises during exertion* | **GAP** — `activity→sbp` +1, `cortisol→sbp` +1 | ✓ |
| dbp | **GAP** — *nocturnal dip ~10-15 %; smaller exercise rise than SBP* | **GAP** | **GAP** — `activity→dbp` +1 | ✓ |

### Thermoregulation module

| marker | cohort | rule | coupling | cold |
|---|---|---|---|---|
| temp | **GAP** — *circadian amplitude ~0.5 °C, nadir ~04-05 h, peak ~late afternoon* (core-temp rhythm); *postprandial thermogenesis +0.1-0.3 °C; rises in exercise* | **GAP** — *core temp nadir at night; rises during exercise* | **GAP** — `activity→temp` +1 | ✓ |

### Respiratory module

| marker | cohort | rule | coupling | cold |
|---|---|---|---|---|
| rr | **GAP** — *RR rises with exercise/metabolic demand; falls in sleep* | **GAP** — *RR drops in sleep, rises in exertion* | **GAP** — `activity→rr` +1 | ✓ |
| spo2 | **GAP** — *normal 95-100 %; mild nocturnal dip; transient exertional desaturation* | **GAP** | **GAP** — weak `activity→spo2` −1 | ✓ |

## What the matrix shows

**Coverage is collapsed onto the metabolic/nutrition axis.** Every one
of the 20 cohort specs is a meal, fast, OGTT, or sleep-restriction
glucose protocol. The four non-metabolic modules
(stress/cardio/thermo/respiratory) have **one** cohort spec between
them that targets their own markers (`postprandial_hr_rise`,
`sleep_hr_dip` — both HR) and **one** rule (`cortisol_morning_peak`).

**Zero-supervision markers (12):** lactate, muscle_glycogen,
mitochondrial_capacity, leptin, acth, hrv, sbp, dbp, temp, rr, spo2 —
plus cortisol has only a rule, no population anchor. These float on
cold-distill + embedding alone.

**Missing mechanism *families* (cross-cutting, not just per-marker):**
1. **Exercise** — acute *or* chronic. No `activity→{hr,lactate,ffa,
   temp,rr,hrv,sbp}` couplings, no exercise cohort/rule. This is the
   single largest hole; it also blocks the multi-timescale north star.
2. **Circadian** — only cortisol AM peak. temp/HR/BP/cortisol/leptin
   all have textbook circadian rhythms; none are constrained as such.
3. **HPA dynamics** — `acth→cortisol`, the *defining* edge of the
   module, is not even a coupling prior. No ACTH supervision at all.
4. **Autonomic / baroreflex** — HR/HRV/BP co-regulation absent.
5. **Incretin** — `glp1→insulin` (+1) absent; GLP-1 currently a
   dead-end output that influences nothing.
6. **Energy-balance loop** — leptin↔ghrelin, adiposity signalling
   absent; leptin is wholly unconstrained.

## Prescription: breadth floor first (iter 58 candidate)

**Principle.** Every zero-supervision marker gets ≥1 *gentle*,
uncontroversial, literature-cited anchor — chosen so it cannot be
"wrong" physiologically and is wide enough not to fight existing fits.
Gentle = a sign-only `CouplingPrior` or a wide-σ `CohortStatisticSpec`
mean, *not* a tight hinge. Probe per-module so attribution stays clean
within the breadth pass.

Per-marker minimum anchor (one each; surface in brackets):

| marker | anchor (gentle, uncontroversial) | citation | surface |
|---|---|---|---|
| lactate | rises ≥1 mmol/L during moderate exercise vs rest | Brooks 1986 | cohort DELTA_MEANS, wide σ |
| leptin | fed-state mean > 16 h-fasted mean (Δ ≥ +2 ng/mL) | Boden 1996 | cohort DELTA_MEANS |
| acth | `acth→cortisol` sign +1 | textbook HPA | coupling prior |
| cortisol | `cortisol_awakening`: +50 % 0-45 min post-wake | Pruessner 1997 | cohort DELTA_MEANS |
| hrv | sleep-window mean > evening-wake mean | Task Force 1996 | cohort DELTA_MEANS |
| sbp | sleep dip: night mean ≤ day mean − 8 mmHg | dipper pattern | cohort DELTA_MEANS |
| dbp | sleep dip: night mean ≤ day mean − 5 mmHg | dipper pattern | cohort DELTA_MEANS |
| temp | circadian: 04-06 h mean ≤ 16-20 h mean − 0.3 °C | core-temp rhythm | cohort DELTA_MEANS |
| rr | sleep mean < awake mean (Δ ≤ −2 /min) | textbook | cohort DELTA_MEANS |
| spo2 | mean within 95-100 % across all windows | textbook | cohort MEAN_IN_WINDOW |
| muscle_glycogen | `activity→muscle_glycogen` sign −1 | Bergström 1967 | coupling prior |
| mito | `mitochondrial_capacity→ffa` sign +1 (fat-ox capacity) | Holloszy 1967 | coupling prior |

All twelve are facts no physiologist would dispute and that the
current model has *no reason to already satisfy*. Encoded gently they
anchor each marker's level/phase without dictating fine dynamics —
exactly the "many weak constraints" the PRD calls for, applied where
there are currently *zero*.

This also lets several missing-family edges land for free as sign-only
coupling priors (no cohort cost): `insulin→ffa` −1, `acth→cortisol`
+1, `glp1→insulin` +1, `ffa→bhb` +1. These are pure
`encode-the-sign-learn-the-strength` additions.

## Campaign sequence

- **iter 58 — breadth floor.** The 12 anchors above + the 4 free
  sign-only edges. One iter, but probe each module's anchored markers
  separately (per-module attribution). Success = every marker now
  responds to its anchor in a forward-rollout probe AND overall_mape
  does not regress past iter-57's baseline. This kills the
  attribution-noise floor.
- **iter 59+ — depth campaign, audit-prioritised.** With the floor in
  place and measurement reliable, deepen one mechanism family per iter
  to literature fidelity, ordered by leverage: (1) exercise
  acute+chronic [also unblocks the multi-timescale north star], (2)
  circadian (temp/HR/BP/cortisol/leptin phase), (3) HPA dynamics
  (ACTH-cortisol pulsatility, CAR), (4) autonomic/baroreflex
  (HR-HRV-BP), (5) incretin + energy-balance loop.

## iter 58 — RESULT (2026-05-18): thesis half-validated; glp1 is the residual sink

Run `train-20260517T195127Z` (exec `pulse-trainer-vr4q9`, commit
`34b96b79`; the 1st dispatch `x4msq`/`958a2d79` OOM-died at Phase 2
start — see multi-timescale-plan.md and the breadth_floor.py MEMORY
BUDGET note; arms were minimized + cohort-sample-patients 16→12 to fit
32 Gi). Gate **FAIL** (baseline iter-57 also FAIL).

`overall_weighted_mape` 0.1189 → **0.1191 (Δ +0.0002, flat)** — no
recovery toward iter-56's 0.0857. Gate fails: glucose_mape=0.248
(improved from 0.283), **hr_mape=0.162 (NEW failure, was passing)**,
verifier_cat[meal]=0.586 (worse, was 0.630).

But the per-marker MAPE Δ vs iter-57 is the real signal:

- **Improved:** glucagon **−0.33**, ghrelin −0.15, insulin −0.14,
  ffa −0.13, cortisol −0.05, glucose −0.035, leptin −0.032, bhb
  −0.028, temp ≈0.
- **Regressed:** glp1 **+0.477**, acth +0.03, hr +0.02, sbp +0.011,
  dbp +0.006.

**The floor worked on what it anchored.** Every one of the 12 floor
markers held or improved, with large fuel-metabolism gains and glucose
down. The embedding-bottleneck reshuffle did not vanish — it
**concentrated into glp1**, the one major high-variance marker the
floor did *not* strengthen (glp1 had only the single, default-weight
`large_meal_glp1_peak` cohort spec — effectively under-anchored, the
same gradient-starvation leptin had before its iter-58 weight bump).
glp1's +0.477 alone cancels the broad improvement, so overall is flat.
This is the floor thesis confirming itself by counter-example: pin
everything and the noise migrates to whatever remains least anchored.
leptin specifically improved −0.032 and held, vindicating the
weight=12 bump.

**iter 59 = complete the floor: properly anchor glp1.** It is now the
designated noise sink; its existing peak spec is gradient-starved.
Apply the established remedy (gentle anchor + weight bump, leptin
precedent) — likely a GLP-1 fed/level or incretin-kinetics cohort plus
a weight increase on the GLP-1 supervision. Thesis prediction: with
glp1 pinned too, the broad per-marker improvement finally surfaces in
overall_mape. Secondary: investigate the new `hr_mape` 0.162 failure
(hr +0.02; the new hrv/sbp/dbp sleep-dip anchors may perturb the HR
fit — check before the depth campaign). The depth campaign (exercise →
circadian → …) waits until the floor is genuinely complete (glp1
closed, hr understood).

## iter 59 RESULT — the floor thesis is falsified; capacity is the ceiling

iter 59 (train-20260518T145947Z) bumped `large_meal_glp1_peak` 12x and
added a `MEAL_GLP1_RISE` level anchor. glp1 did **not** recover:
2.0824 → 2.1177 (+0.0353, slightly *worse*). The rest of the pack was
noise-floor static (insulin +0.0001, ffa +0.0002, leptin +0.0007,
glucagon +0.0013, bhb −0.0009). The only real move, glucose
−0.0151, was the orthogonal gut-amplitude work, not the glp1 lever.
overall 0.1191 → 0.1170; textbook byte-identical 0.8127;
verifier_cat[meal] 0.586 → 0.577 (worse).

**A 12x supervision bump that does not move its target is the
capacity-not-weight signature.** Three consistent diagnostics pin the
mechanism:

1. **Shape, not mass.** iter-59 gut-sweep at zero embedding: glucose
   appearance AUC near-perfect at every dose (120 g: tgt 338.61 vs
   336, 0.06%) but the PEAK compressed to ~45% (120 g: tgt_pk 3.274
   vs new_pk 1.457). The model conserves total glucose and smears it
   flat.
2. **The two catastrophes share the thinnest module.** glp1 (2.12)
   and ghrelin (1.08) are the only markers with mean_mape >1. Both
   are high-variance sharp meal-locked hormones owned by the
   AppetiteModule, whose per-module embedding projection was the
   smallest in the model (`_emb_dims`: appetite 6 vs metabolic 10,
   gut 8). leptin (also appetite, 0.029) is fine because it is a
   slow low-variance setpoint. 6 dims cannot carry two independent
   patient-specific spike manifolds → both collapse to the
   population mean → MAPE >1.
3. **The roadmap already named it.** architecture-roadmap.md #8
   ("per-module shared embedding_projection is a coupling channel …
   mitigatable by widening the projection") and the spec's
   pre-registered R1 ("the bottleneck capacity itself is the limit")
   both nominate exactly this pivot.

## iter 60 = widen the embedding bottleneck (the R1 pivot)

One structural lever, attribution-clean (zero cohort/head/loss/trainArg
changes — byte-identical trainArgs to iter 59). `EMBEDDING_DIM`
32 → 64 (types.py constant; the train table, benchmark MAP-fit, and
server all auto-scale off it) and every per-module projection widened
proportionally, appetite getting the largest relative bump because it
provably owns the two catastrophic markers: `_emb_dims` appetite
6→16, metabolic 10→20, gut 8→16, stress 6→12, cardiovascular 8→16,
thermoreg 4→8, respiratory 4→8. SUCCESS = overall below 0.1170 toward
iter-56's 0.0857, driven by glp1/ghrelin falling from >1 toward the
pack, pack holds, glucose peak-compression eases. The experiment
cannot fail to discriminate: if glp1/ghrelin recover → capacity was
the limit (continue capacity/attention); if they stay flat → the
embedding-FITTING procedure or the linear projection is the limit
(iter 61 goes architectural on the projection, roadmap #8's attention
alternative); if they recover but textbook drops → eval-time MAP-fit
overfit (iter 61 strengthens the embedding prior).

## Generator capability check (resolved 2026-05-17)

`CohortArmSpec` already exposes per-step `sleep_wake` *and* `activity`
series (both optional tuples), and `cohort_loss._rollout_arm_batched`
already plumbs both into the rollout. Consequences:

- **Sleep-dip anchors are a solved pattern.** `cohorts/sleep.py` and
  `cohorts/cardiovascular.py` already build day/night arms with
  `sleep_wake=_adequate_sleep_24h()` etc. The sbp/dbp/temp/hrv/rr
  nocturnal-dip anchors need no generator work — copy that pattern with
  a day-window vs night-window `StatisticWindow`.
- **Activity arms are expressible but unused.** No existing spec sets
  `activity=`, but the field is plumbed end-to-end. The lactate and
  muscle_glycogen exercise anchors are new authoring, not new
  machinery.

**Therefore the breadth floor is 12/12, not 10/12 — no generator
extension required.** This is the iter-58 scope.

## Remaining open question

- Gentle-σ calibration: how wide is "won't fight existing fits"? Start
  each new cohort σ at ~½ the marker's NORM_SCALE and tighten only in
  the depth campaign. Validate per-module with the cohort-ablation
  diagnostic before dispatch (same pre-flight that caught the iter-56
  gradient starvation and validated iter-57's 112× fix) — every new
  anchor should land non-trivial gradient on its own module and not
  swamp existing specs.
