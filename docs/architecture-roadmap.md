# Pulse architecture roadmap

Written immediately after iter 51 (2026-05-14). The post-iter-51 picture is
crisp enough to step back from per-iter tuning and ask: *what would actually
make the model work, not 5-10% better per iter, but qualitatively better?*

## What iter 51 just showed

The SetpointHead re-parameterisation (move from (prod, cons) softplus-output
to (target_z, k_factor) for ffa/glucagon/ghrelin) worked exactly as designed:

| marker   | iter 50 | iter 51 | Δ                                    |
|----------|--------:|--------:|--------------------------------------|
| ghrelin  |  1.928  |  0.812  | −1.12 (mechanism confirmed)           |
| insulin  |  1.080  |  0.389  | −0.69 (collateral *win*, see below)   |
| ffa      |  0.432  |  0.240  | −0.19                                 |
| glucagon |  0.444  |  0.272  | −0.17                                 |
| glucose  |  0.209  |  0.204  | −0.01 (still over 0.20 gate)          |
| **acth** |  0.488  |  0.772  | **+0.28**  collateral regression      |
| **cortisol** | 0.477 | 0.609 | **+0.13**  collateral regression       |
| **bhb**  |  0.214  |  0.346  | **+0.13**  collateral regression      |
| **hr**   |  0.138  |  0.153  | **+0.01**  newly over 0.15 gate       |

Gate still fails (glucose 0.204, hr 0.153, verifier_meal 0.636). textbook +0.005.

Two deep facts to extract:

1. **The softplus-saturation trap is generic, not specific to the dead trio.**
   Every chemical-species head whose data wants it pinned near `typical` most
   of the time and excursing only on specific stimuli is at risk. Iter 51
   moved three of them and *exposed three more* (acth, cortisol, bhb) that
   were ALSO stuck — they only looked “OK” at iter 50 because the
   patient-embedding fit happened to settle in a place that gave them a
   half-decent error. As soon as the embedding moved to accommodate the now-
   movable ffa/glucagon/ghrelin, the stuck heads couldn't track and their
   error jumped.

2. **Insulin going from 1.08 → 0.39 *without* touching its head** is the
   cleanest evidence yet that the patient embedding was being held hostage
   by the dead trio. Once the dead heads could move on their own coordinates,
   the embedding stopped having to compensate for three unsolvable species,
   and the insulin head's existing glucose-gated structure could converge.
   *Unblocking dead species frees the patient embedding to do its real job.*

This reframes the problem. The dead-pathways thread is not “fix 3 markers”;
it is **“the head parameterisation choice for chemical species was wrong,
codebase-wide”** — wrong in a way that makes the embedding carry an
impossible load. Iter 51 fixed it for three; the right next step is to fix
it for the rest of the relevant heads. But that alone is not enough — there
are structural gaps below that we should plan around at the same time.

## What's structurally missing

Inventory of the current model's load-bearing assumptions and the gaps under
each:

### 1. Head parameterisation (partial fix in flight)
   - `SpeciesHead` (default) emits `(softplus(raw_prod), softplus(raw_cons))`.
     At equilibrium-at-typical, the head must push `raw_prod` very negative,
     where softplus' derivative is exponential-small. Saturation trap.
   - `SetpointHead` (iter 51, dead trio) emits `(target_z, softplus(k))`,
     decoupling the equilibrium location from the rate constant. No
     saturation.
   - `GlucoseGatedInsulinHead` (iter 23) emits `(raw_basal, raw_peak, raw_cons)`
     with a glucose-gated mixture. **This is structurally the best of the
     three** because it separates baseline behaviour from stimulus-driven
     peaks — a thing the data demands for every hormone with a meal/stress
     pulse (insulin, glucagon, FFA, GLP-1, cortisol, ACTH, ghrelin pre-meal
     rise).
   - **Gap**: only insulin gets the gated-peak treatment. Every other
     stimulus-driven hormone is asked to fit basal + pulse from a single
     scalar emission.

### 2. Stress / HPA axis has no structural prior
   - `StressModule`: two species (cortisol, ACTH), both plain `SpeciesHead`,
     `_N_COUPLING=2` (glucose + cortisol feedback), `_N_EXTERNAL=2`
     (sleep_wake + activity). No diurnal carrier wired into the head, no
     ACTH→cortisol explicit cascade structure, no negative-feedback term.
   - Yet the textbook dynamics are very specific: ACTH pulsatile with
     6-12 pulses/day, cortisol response amplifies and lags ACTH by 15-30 min,
     cortisol negative-feedback on hypothalamic CRH (proxied through ACTH
     here). Morning peak (CAR) at 30-45 min post-wake, evening trough.
   - **Gap**: the model has to discover all of this from a sin/cos
     time feature + a single physiology rule (`cortisol_morning_peak`).
     This is unrealistic supervision pressure for a feedback loop.

### 3. No latent reservoir states (every state is an observed marker)
   - Glycogen pool (liver ~80g + muscle ~400g, depleted over 12-24h fasting,
     resynthesised over hours after a meal) drives gluconeogenesis kinetics
     and the timing of the fast→ketosis transition. Currently has to emerge
     from the BHB rate function operating purely on observables.
   - Adipose FFA pool (multi-day storage capacity, modulated by leptin)
     bounds FFA mobilisation rate ceiling. Not represented.
   - Gut hormone reservoirs (K-cell GIP, L-cell GLP-1, parietal-cell ghrelin)
     have characteristic refill timescales after meal-induced depletion.
     Not represented.
   - CRH / pre-cortisol pool would give the HPA axis its integrator.
   - Sleep debt, insulin sensitivity (slow drift over days), hydration —
     all are physiology variables that drive observables on timescales the
     current architecture cannot represent without observed-state proxies.
   - **Gap**: dynamics with characteristic time > step horizon cannot be
     cleanly represented. Loss must compensate by overfitting fast variables.

### 4. Coupling graph is thin
   - Metabolic ← gut(4) + cortisol(1).
   - Appetite ← insulin(1) + nutrient_flag(1).
   - Stress ← glucose(1) + cortisol_feedback(1).
   - Cardio ← (TBD — likely sympathetic / external).
   - **Gap inventory** (textbook-canonical but absent in the wiring):
     - leptin → POMC / hypothalamus → ACTH and ghrelin suppression
     - FFA → metabolic.bhb (ketogenesis is FFA-driven)
     - insulin → FFA (antilipolysis tight loop)
     - glucagon → hepatic_output (gluconeogenesis driver)
     - cortisol → glucagon (permissive on counter-regulation)
     - GLP-1 → insulin (incretin effect, ~50% of post-meal insulin)
     - ghrelin → meal_anticipation rise
     - catecholamines (missing module) → HR, BP, FFA, hepatic glucose
   - Some of these *might* be internal to a module via shared coupling vector;
     others (catecholamines) just aren't represented.

### 5. Physiology rules: 5 rules, should be ~50
   - Current: `glucagon_falls_postprandial`, `ffa_inverse_to_insulin_postprandial`,
     `ghrelin_falls_after_meal`, `bhb_rises_during_fast`, `cortisol_morning_peak`.
   - Standard endocrinology / metabolism textbooks (Guyton, Boron & Boulpaep,
     Berne & Levy, Williams Textbook of Endocrinology) contain hundreds of
     testable inequalities — monotonicities, antagonist pairs, asymptotes,
     extrema, phase relationships, dose-response shapes, mass-conservation
     identities. We use five.
   - Each rule weight is 0.05/N — at N=5, ~0.01 per rule. At N=50 the
     individual gradient would be smaller but the *cumulative* shaping power
     would be far more discriminating. Combined with rule-rebalancing (give
     more weight to the rules a candidate model violates most), this is a
     high-leverage axis.

### 6. Cold-distill anchor is single-point
   - `--cold-distill-pool=bench_cohorts` distills *only* at the bench cohort
     protocols, *at* the bench-cohort calibrated embedding.
   - Generalisation to other patient embeddings is implicit — but a) the
     training distribution of embeddings is bench-cohort dominated during
     phase 2, b) iter 51's collateral regression on cortisol/acth/bhb is
     consistent with “fit gets better at the anchor, gets worse off-anchor”.
   - **Gap**: no multi-anchor distillation. Should distill at bench-cohort
     embedding + zero embedding + sampled embedding-table rows + interpolated
     anchors. Cost is linear in number of anchors; current is 2 protocols/epoch.

### 7. Cohort coverage
   - Bench scenarios are dominated by meal-postprandial dynamics. Coverage:
     fasting (yes), meal (yes), sleep (yes), exercise (limited), stress
     (limited), IVGTT/OGTT clinical-trial-like protocols (no), prolonged
     fast 24/48/72h (no), cold exposure (no), recovery-from-illness (no).
   - **Gap**: the model is never asked to extrapolate to >24h dynamics or
     to clinical-trial perturbations. Long-horizon ground truth from the
     cold ODE model is free.

### 8. Per-module shared embedding_projection is a coupling channel
   - Each module has its own `embedding_projection` that maps the global
     patient embedding into a module-local representation. Every head in
     that module reads from the same projection.
   - When a head pulls hard on the projection (as the SetpointHead heads did
     in iter 51 to fit cold-rate curves), every other head in the module
     sees the new projection too — and if it was sitting in a flat region,
     it cannot follow.
   - **Gap**: no per-head adapter / LoRA / decoupling. Possibly mitigatable
     by widening the projection or by replacing it with an attention
     mechanism that lets each head pull its own slice.
   - **iter 60 acts on the "widen" arm.** EMBEDDING_DIM 32→64 and every
     per-module projection widened proportionally (appetite 6→16, it owns
     glp1+ghrelin — the only two markers with mean_mape >1 — see
     physiology-coverage.md "iter 59 RESULT"). If width alone does not
     resolve the reshuffle, the attention-projection arm is iter 61.

## Three candidate "big moves"

### Move A — Generalise the head parameterisation
*Cost: ~1 day. Risk: low (already validated for 3 heads). Upside: large.*

- Convert every chemical-species head that data wants near typical to
  `SetpointHead` (cortisol, acth, bhb, glp1, leptin (already ≈ OK numerically
  but stuck), lactate, hepatic_output).
- Generalise `GlucoseGatedInsulinHead`'s structural decoupling into a
  reusable `BasalPlusGatedPeakHead(stimulus_source, gate_threshold_init)` —
  parameterise the dimension of stimulus + the gate location. Apply to:
  - Insulin (already) → glucose-gated
  - Glucagon → glucose-anti-gated (peak when glucose is LOW)
  - FFA → insulin-anti-gated (peak when insulin is LOW)
  - GLP-1 → nutrient_flag-gated (peak when meal present)
  - Cortisol → diurnal-gated (peak in morning hours)
  - Ghrelin → meal-anti-gated (peak pre-meal, suppress post-meal)
- This is *the same conceptual fix that worked for insulin in iter 23, plus
  the saturation fix that worked for ffa/glucagon/ghrelin in iter 51,
  applied uniformly*.

### Move B — HPA axis structural prior
*Cost: ~2-3 days. Risk: medium (changes a 2-species module structurally).
Upside: directly addresses iter-51 regression, unlocks the meal-verifier
circadian-meal interaction.*

- Build a `StressModule` subclass that hard-wires:
  - A CRH pool latent state (integrator), driven by stress inputs + diurnal
    cosinor.
  - ACTH rate is a function of CRH pool with first-order kinetics + pulsatile
    modulator.
  - Cortisol rate is a function of ACTH with 15-30 min delay (Gamma kernel
    or learnable two-pool relay) + cortisol-on-CRH negative feedback.
  - Diurnal drive: explicit cosinor (24h period, learnable phase + amplitude
    per patient via embedding-projected scalar) as an *input to CRH rate*,
    not just a time feature the head must learn to extract.
- Adds 1 hidden state (CRH pool), keeps 2 observed states, gains a delay
  primitive that the codebase doesn't have.

### Move C — 10× the physiology rules
*Cost: ~1-2 days. Risk: low (additive constraints). Upside: high-leverage
shaping pressure with broad coverage.*

- Author 45+ additional `PhysiologyRule` instances from textbook canon:
  - Antagonist pairs: insulin⊥glucagon, insulin⊥FFA, leptin⊥ghrelin,
    cortisol↑↑→glucagon↑, parasympathetic↔sympathetic.
  - Asymptotes: ketosis BHB plateau under prolonged fast; glucose floor
    under fast (~65 mg/dL); GLP-1 returns to baseline ~3h post-meal.
  - Monotonicities: liver glycogen monotonically depletes during fast;
    GLP-1 monotonically increases with carb meal size up to a saturation;
    HR/SBP increase monotonically with exercise intensity.
  - Mass-conservation: gut macro intake → glucose+lipid+amino appearance
    matches stoichiometric upper bound.
  - Phase relationships: ACTH leads cortisol by 15-30 min; ghrelin rises
    before meals (anticipatory); insulin pulse leads glucose recovery.
  - Magnitude order: post-meal glucose excursion < 60 mg/dL above baseline
    in non-diabetic; cortisol AUC over 24h is roughly constant.
- Rebalance rule weighting: instead of uniform 0.05/N, weight each rule by
  its *current model violation magnitude* (auto-tuned to the worst
  rule-violators).
- Add 1-2 "long-horizon" rules requiring 24h+ protocols, which forces the
  cohort generator to add a prolonged-fast protocol.

## What to do next (concrete iter-52 proposal)

**The pragmatic combination**: Move A + a small slice of Move B + cohort
breadth from Move C.

Specifically iter 52:
- Convert cortisol, acth, bhb, leptin, glp1, lactate, hepatic_output to
  `SetpointHead`. (Move A, broadly.)
- Hold the gated-peak generalisation for iter 53 (clean separation: prove the
  saturation fix scales first, *then* layer in gated peaks).
- Keep training params identical to iter 51 except the new head wiring, so
  the delta is attributable.

Expected effect:
- cortisol, acth, bhb regressions reverse (these were the iter-51 collateral
  victims — they fall back into the same trap iter 51 broke for the dead
  trio).
- glp1, lactate, hepatic_output get gradient access; current iter 51
  numbers stay or improve.
- Overall_weighted_mape: target 0.075-0.085 (from iter-51's 0.1005, iter-50's
  0.098 reference). Glucose gate (0.204) probably doesn't move on its own;
  hr gate (0.153) doesn't move; verifier_meal climbs further but unlikely
  to clear 0.65.
- Gate: still fails on glucose+hr+verifier_meal, but ALL non-vital chemical
  markers should be in their respective gates. That's the first time that's
  ever been true.

After iter 52, two iters in sequence:
- iter 53: gated-peak generalisation (Move A, completion). This is the
  precise mechanism for fixing the stimulus-driven peaks (glucagon at
  hypoglycemia, FFA at low-insulin, GLP-1 at meal, cortisol at morning,
  ghrelin pre-meal). Probably the single biggest verifier-meal mover.
- iter 54: HPA axis structural prior (Move B). CRH latent + ACTH→cortisol
  cascade. Probably the iter that finally clears verifier_meal AND fixes
  circadian.

iter 55+ is open — pick from the remaining gaps (latent reservoirs, coupling
graph fill-in, multi-anchor distillation, cohort breadth, embedding
projection per-head adapter, physiology rules 5→50).

**Status update — iter 60 through 66 (2026-05-22):**

- **iter 60** widened EMBEDDING_DIM 32→64 (the gap #8 "widen" arm).
  Overall MAPE **0.117→0.0875** (tied with iter 56 all-time best);
  glp1 catastrophe collapsed 2.12→0.43; cortisol/sbp/dbp big wins.
  R1 fired: acth 0.156→0.65, insulin 0.29→0.45, ffa 0.21→0.31 — the
  wider per-episode MAP-fit opened embedding shortcut directions for
  these markers.
- **iter 61** tried to close the shortcuts via trained-table diagonal
  Gaussian prior at eval time. acth 0.65→0.13 ✓ but glucose/hr/glucagon
  regressed (overall 0.0875→0.146).
- **iter 62 / 63** ablated `PULSE_BENCHMARK_PRIOR_WEIGHT` ∈ {0.5, 0}
  on iter 61's checkpoint (no retrain). w=0 reproduced iter 60's
  numbers exactly — the prior plumbing IS the regression, not the
  checkpoint. The diag-Gauss eval-prior was retired in iter 64 as
  default (`BENCHMARK_GATE_PRIOR_WEIGHT` default 1.0 → 0.0).
- **iter 64** = **Move B compact**: HPA cascade structural prior in
  StressModule. α·relu(acth excess) drives cortisol; γ·diurnal_carrier
  drives ACTH; β·relu(cortisol excess) negative-feeds ACTH. All
  sign-constrained (softplus), per-patient diurnal phase projected
  from embedding. acth 0.65→**0.20**, cortisol 0.18, insulin 0.37 —
  the structural priors closed the iter-60 shortcuts at training
  time. But glp1 0.42→1.74 (catastrophe), ghrelin 1.30, glucose 0.29:
  the embedding's freed capacity reallocated badly.
- **iter 65** = **Move C compact**: 11 new physiology rules + new
  relational predicate helper `hinge_a_precedes_b` replacing iter 64's
  argmax-band pair. Total rules 7→17. Designed to close iter 64's
  collateral damage. (Result-in-flight at time of this update.)
- **iter 66** = **Move C FULL**: order-of-magnitude rule expansion —
  60 total rules (was 7 at iter 64, ~8.5x). 3 new cohort arms
  (prolonged fast 24-48h, moderate exercise bout, sleep-wake 24h).
  4 new helpers (`hinge_min_correlation`, `hinge_circadian_amplitude`,
  `hinge_argmin_in_band`, `hinge_max_drift`). Every marker now has
  ≥1 rule (lactate, hepatic_output, HR, HRV, RR, SpO2, temp, leptin
  moved from zero to 2-5 each). Plus a small "easy from B" pickup:
  explicit cosinor δ on cortisol_rate mirroring iter 64's γ on ACTH.

**Open after iter 66:**
- Move A part 2 (BasalPlusGatedPeakHead generalisation) is still open
  but conditional on cohort arms covering the gate stimuli (e.g.
  glucagon@low-glucose needs a hypoglycemia challenge arm — not yet
  added). iter 53 tried this naively and regressed; the prerequisite
  is arm breadth.
- Move B FULL (CRH latent state + Gamma-kernel ACTH→cortisol delay)
  is the natural iter 67 follow-up if iter 66's circadian amplitude
  + relational ACTH-precedes rules don't fully resolve the diurnal
  shape (would require cold-ODE updates for the new latent state).
- Move D (multi-timescale latent reservoirs) — still the north star
  per `docs/multi-timescale-plan.md`.

- **iter 65/66 produced no bench** (iter 65 NaN-crashed phase 2 ep 58
  via cold-distill embedding calibration without finiteness guards;
  iter 66 was never dispatched — content rolled into iter 67).
  Baseline reverts to iter 64 (overall 0.133).

- **iter 67** = **ALL MOVES bundle** (commit `8632c371`). Bundled three
  changes onto iter 66's tree:
  - (1) **Cold-distill numerics fix** — clamp emb L∞≤5 in Adam inner
    loop, abort steps on non-finite loss/grad, warm-start fallback,
    loud RuntimeError on non-finite rates.
  - (2) **Move B FULL** — two new internal markers (`crh`, `acth_pool`,
    idx 22/23), StressModule n_species 2→4, two-stage cascade with
    explicit relay rate for ACTH→cortisol delay (τ softplus + 5 min
    floor, init ≈ 15 min per Veldhuis pulsatility). Cortisol drive
    rerouted from `α·acth_excess` (instant) to `α·acth_pool_excess`
    (delayed). Also fixed a silent iter-64 units bug:
    `relu((acth_norm − typical)/typical)` mixed normalized state with
    raw typicals → α/β feedback terms have been dead since iter 64
    (iter 64's gains came from γ·(1+diurnal) alone).
  - (3) **Adaptive rule weighting** (`--physiology-rules-adaptive`) —
    per-rule EMA of violation_mean (α=0.1), reweight so per-rule
    coefficient ∝ EMA-violation, preserving Σ. 2% floor.
  - **Verdict: OOM'd 8 dispatch rounds in phase-2 ep 50.**
    Slow-growth pattern (per-step memory accumulation). Cuts attempted
    (cold-distill 2→1, cohort 12→8, windows 3→2, rules-sample 4→1,
    verifier off) didn't unblock. Cloud Run ceiling 32Gi @ 8 CPU is a
    hard cap. Diagnosed: n_species 2→4 doubled autograd-graph branches
    in StressModule across long phase-2 rollouts.

- **iter 67'** (commit `86cd54be`, dispatched 2026-05-23 07:51Z, exec
  `mwkph`) reverted Move B FULL to ship a bench. Kept: numerics fix,
  adaptive weighting, faulthandler watchdog. **Did not produce a bench
  report** — outcome unclear from job artifacts (no
  `benchmark-report.json` at `train-20260523T074502Z/`); status
  considered failed.

- **iter 68 = Move B FULL retry + memory-profile fixes.** Rounds 1-7
  all OOM'd in phase 2. The eater was found via per-epoch RSS +
  autograd-tensor instrumentation: gradient checkpointing on
  cold-distill calibration (r3, `43b3e1ce`), per-spec gradient
  accumulation in cohort_statistic (r5, `9209b3a0`), per-rule backward
  in physiology_rules (r7, `4c35e9f9`), watchdog 1h→4h + phase-2 cut
  30→10 (r8, `fd8578a4`).

- **iter 68 r8 LANDED** (job `train-20260525T011534Z`, 16h16m, 2026-05-25
  17:32Z). **Overall MAPE 0.0976** — best yet (vs iter 64 baseline
  0.133, −27%). Gate failed by narrow margins: `glucose=0.224 > 0.20`,
  `hr=0.156 > 0.15`. acth 0.091 ✓ (HPA prior working), cortisol 0.347,
  ghrelin 0.615 (halved from iter-64's 1.30), glp1 0.401 (quartered
  from 1.74). Textbook 74.1%. **`meal_dose_response: 0/3`** — glucose
  Δpeak 0.015 mg/dL vs 5 threshold, insulin 1.64<2, glp1 wrong sign.
  Diagnosis: model nails shape, fails amplitude.
  `--dose-response-weight=0.0` has been off since iter 32 — the only
  signal that supervises amplitude scaling with stimulus.

- **iter 69** (commit `c9c4cf4d`, exec `mmz62`, started 2026-05-25
  18:51Z) = two structural moves together:
  - (1) **Multi-marker dose-response axis** — `DoseResponseSignal`
    extended to `MarkerDoseTarget` dataclass (per-marker mode `slope`
    or `rank`). Spec: `glucose:slope:0.7:0.25; insulin:rank:0.5;
    glp1:rank:0.3`, weight 0.0→0.30. Existing single-glucose-slope
    path preserved for back-compat.
  - (2) **Move B FULL, smaller footprint** — single new `crh` marker
    (idx 22, STATE_DIM 22→23), StressModule n_species 2→**3** (not 4
    as in iter 67), no separate `acth_pool` relay. Cortisol negative
    feedback rerouted from ACTH to CRH (anatomically correct: PVN/CRH
    is the dominant glucocorticoid feedback target). Cascade:
    `diurnal → CRH → ACTH → cortisol`. New mechanism params (λ, ε,
    decay) softplus sign-constrained, all init at raw −3 (≈0.05).

- **iter 69 LANDED** (job `train-20260525T185159Z`, 2026-05-26 16:07Z).
  **REGRESSION on most markers**:

  | marker | iter 64 | iter 68 r8 | **iter 69** | δ vs r8 |
  |---|---:|---:|---:|---:|
  | overall_weighted_mape | 0.133 | **0.0976** | 0.146 | +0.049 |
  | glucose | 0.29 | 0.224 | 0.296 | +0.07 |
  | hr | — | 0.156 | 0.288 | +0.13 |
  | **acth** | 0.20 | **0.091** | 0.761 | **+0.67 (catastrophe)** |
  | cortisol | 0.18 | 0.347 | **0.129** | −0.22 |
  | insulin | 0.37 | 0.599 | 0.641 | +0.04 |
  | ghrelin | 1.30 | 0.615 | 0.471 | −0.14 |
  | glp1 | 1.74 | 0.401 | 0.332 | −0.07 |
  | ffa | — | 0.260 | 0.229 | −0.03 |
  | bhb | — | 0.200 | 0.209 | +0.01 |

  Gate failed: glucose 0.296 > 0.20, hr 0.288 > 0.15, verifier_cat[meal]
  0.644 < 0.65. Textbook pass-rate 78.9%. **meal_dose_response 1/3** —
  glp1 now passes (the rank hinge worked); glucose Δpeak still 0.23
  mg/dL (rank-monotonic but flat), insulin Δpeak still slightly
  negative. cortisol_awakening_response 2/3 (cortisol_rises 1.232,
  just below 1.30 threshold).

  **Dominant signal: ACTH collapsed from 0.091 to 0.761.** Move B
  FULL's CRH cascade and cortisol-feedback re-route broke what the
  iter-64 compact cascade had cleanly fixed. Hypothesis: the new
  mechanism params (λ, ε, decay) sat near softplus(−3)≈0.05, leaving
  ACTH driven mostly through emergent CRH dynamics without the
  iter-64-proven γ·diurnal direct ACTH drive. Cortisol *did* improve
  (0.347→0.129) — the morning peak still found a path via CRH, but
  ACTH lost its.

  **Confound: two moves landed in one iter**, so attribution between
  the dose-response axis and the CRH cascade is harder than a single-
  move iter. Subsequent analysis needs to isolate which move drove
  which delta.

- **iters 70-72 = the CRH-cascade fight (all three failed).** A
  four-iter detour trying to land the CRH cascade, none beating iter
  68 r8 (0.0976):
  - **iter 70** (revert Move B FULL, retry gated peaks): overall
    **0.1094**, **acth 0.0755** ✓, cortisol 0.2235. The compact iter-64
    two-state cascade restored — best HPA numbers of the arc.
  - **iter 71** (revert gated peaks too): overall **0.1679**, **acth
    0.0755 → 0.8331** 💥. Reverting the *metabolic* gated heads broke
    HPA coupling via cross-module gradient flow — gated heads were
    load-bearing for the HPA cascade's learned positions.
  - **iter 72** (rollback to iter-70 metabolic + CRH cascade ADDITIVE
    on iter-64 paths, softplus-dormant init): overall **0.1181**,
    **acth 0.0755 → 0.3618**. The dormant-init redundancy contained the
    damage (no 0.8 collapse) but ACTH still **drifted** — the iter-72
    R1 risk materialised. insulin 1.09 (worst marker), cortisol 0.364,
    glucose 0.292. Gate failed: glucose 0.292, hr 0.182, verifier_meal
    0.603.

  **Structural verdict — the CRH cascade is unsupervisable, do not
  retry it as-is.** Three failure modes (69 full-replace, 71 indirect,
  72 additive) share one root cause: **the cold knowledge model does
  not simulate CRH** (`knowledge/full_body.py` pads `crh` with the
  constant typical 100.0). Cold-model distillation supervises ACTH and
  cortisol *directly* (both are in the cold ODE with diurnal curves)
  but never CRH. Any `λ·crh → ACTH` term therefore lets ACTH's rate be
  explained by a free-floating latent with zero ground truth — gradient
  drags it off the iter-64-proven direct γ·(1+diurnal) drive every time.
  The iter-64 compact two-state cascade works *because* it mirrors the
  cold model's own ACTH/cortisol structure. A CRH cascade is only
  viable after the cold model is taught to simulate CRH (giving it
  direct distillation supervision). Until then, CRH stays an inert
  latent.

- **iter 73 = revert CRH for good + absolute meal-amplitude
  supervision** (commit TBD). Two attribution-clean moves on disjoint
  clusters. (A) Restore the exact iter-70 `stress.py` (iter-64 compact
  cascade; CRH inert) — recovers ACTH 0.0755 for free; metabolic stays
  iter-70. (B) New dose-response **`peak` mode**: Gaussian z² on
  *absolute* Δpeak vs a literature line through the origin
  (`target_per_g · dose`, dose-proportional σ), pinning amplitude not
  just slope. glucose slope→peak (0.7/g), insulin rank→peak (0.4/g),
  glp1 stays rank; dose-response weight 0.30→0.40. Rationale: the
  standing gate (glucose, hr, verifier_meal) is the **meal-amplitude**
  cluster — the model learns shape but systematically-too-low magnitude
  because `slope` is offset-invariant, `rank` is ordering-only, and the
  verifier targets are loose floors the model still undershoots. The
  deeper thesis: across 30+ iters the bottleneck has been supervision
  *strength*, not architecture (dead markers, inert glycogen, CRH
  drift, the amplitude gap — all the same under-supervision pattern).
  This also reframes **Move D**: minimal-Move-D-for-HPA would add
  *another* unsupervised latent (the plan's `crh_pool` /
  `cortisol_baseline_drift` are undetailed and would face the same
  gradient-starvation that left iters 55-57's slow states inert) — so
  Move D is deferred until supervision strength, not capacity, is shown
  to be the binding constraint.

---

**Update (2026-05-15): the multi-timescale work — Move D, "latent reservoirs"
above — has been promoted from a single-line gesture to a full plan in
`docs/multi-timescale-plan.md`. That plan is now the north star:
a new Adaptation module (`LearnedDynamicsModule`, same pattern as
cardiovascular) carrying latent slow-timescale states (Si_chronic,
RHR_chronic, glycogen_pool, autonomic_tone, cortisol_curve_health), all
PRD-compliant (initial values projected from the embedding, no new
per-patient params; supervision via cohort specs + physiology rules from
literature deltas; coupling sign priors honoured). When `chronic_exercise_block_8w`
passes as a bench gate, the platform is a multi-timescale digital
physiology twin rather than a 19-marker autoregressive imputer. Iter 55
is the first step — instrumentation only (time-since-last-X covariates,
chronic-block synthetic protocol, acute-z2-bout bench scenario, no
architecture change).**

## North star

The benchmark gate's standing failures as of iter 69 (regressed from iter 68
r8's near-miss): glucose 0.296, hr 0.288, verifier_cat[meal] 0.644.
At iter 68 r8 — the best-yet checkpoint before iter 69's regression — those
were 0.224, 0.156, and (passing). Glucose and hr were within a few percent of
clearing; verifier_meal is the load-bearing semantic gate (meal-time
counter-regulation textbook actually firing).

The real ambition is not "pass the gate". It is: *the model reproduces the
canonical physiology trajectories well enough that textbook scenarios pass at
>= 0.90, the patient embedding actually represents endocrinotype variation,
and the model becomes useful for counterfactual reasoning* (what does
patient X look like under sustained sleep restriction? semaglutide dose?
shift work?). That's an iter-60+ horizon, and it's where the architecture
choices we make now (latent reservoirs, gated peaks, HPA cascade, multi-
anchor distillation) compound the most.
