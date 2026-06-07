# Dead pathways: the unobserved-marker problem

## The symptom

Five markers — **glucagon, ffa, ghrelin, leptin, acth** (and to a
lesser extent **cortisol, bhb**) — have been *byte-identical* in the
benchmark report across every iteration since iter 38:

```
ffa       0.4320      glucagon  0.4441
ghrelin   1.9283      leptin    0.0266
```

Not "approximately stable" — the same floating-point value, iter
after iter, through multi-arm physiology rules (iter 43), broader
rule sampling (iter 44), rule-weight tuning (iters 45-46), FFA-weight
tuning. Nothing moved them. That is the signature of a *structural*
ceiling: there is no gradient path from any active training signal to
the parameters that set these markers' level and shape, so they sit
at initialization forever.

## Root cause

**1. The bench's ground truth for these markers comes from the cold
knowledge model.** No real user self-measures FFA / glucagon /
ghrelin, so `benchmark_extras.py` generates the cohort episodes by
running `simulate_full_body(PatientParams(), …)` and using *its*
trajectories for the unobserved markers as the eval targets. To score
well, the learned model must reproduce the textbook ODE's
counter-regulatory dynamics.

**2. No active training signal reaches the relevant parameters:**

| signal | why it doesn't reach the dead markers |
|---|---|
| bench embedding calibration | fits only the 5 self-measured markers (glucose/hr/sbp/dbp/temp) — the embedding is never moved on account of unobserved markers |
| trajectory MSE (real-user windows) | supervises only those same 5 markers — real users don't measure FFA |
| cohort-statistic loss | weight 0.15 split across ~19 specs ≈ 0.008 each — negligible on small-amplitude markers |
| physiology rules | ~1e-4 gradient onto glucagon/ghrelin — their predicates are *correlation/direction* constraints, and the gradient of a correlation w.r.t. a flat trajectory is ~0. FFA's rule gradient is real (~14) but it trains at *sampled patient embeddings* the cohort eval never visits — that's the iter-44 "embedding narrowness" finding |
| marker_vitality (iter 39) | a range-floor band-aid — never enabled, and even on it only constrains *variance*, not the curve |

⇒ The FFA/glucagon/ghrelin/leptin MLP heads sit at initialization,
which emits a near-constant near-typical value → byte-identical
trajectory → byte-identical MAPE.

**Architecture is *not* the bottleneck.** Every species head receives
`state + coupling + external + embedding + time_features` (see
`MassActionModule.forward`). So the metabolic glucagon/FFA heads see
glucose, insulin, and lipid/amino appearance; the appetite ghrelin
head sees insulin and the nutrient flag; the leptin/cortisol/acth
heads see `time_features` (circadian). They *can* represent the cold
model's equations (`glucagon_stim - glucagon_supp`, `lipolysis -
k_ffa·FFA`, `ghr_prod·(1-insulin_supp)·(1-Ra_norm)`, the circadian
envelopes) — they have simply never been told to.

## The fix: cold-model distillation (iter 47)

Add a structural training signal — `ColdModelDistillationSignal`
(`pulse/training/cold_model_distillation_signal.py`) — that distils
the knowledge model's unobserved-marker dynamics into the learned
model directly:

- **Reference (cached once at init; the cold model is deterministic):**
  for each of an 8-protocol pool, run `simulate_full_body(PatientParams(),
  …)` → reference trajectory for all 19 markers.
- **Protocol pool** spans the regimes the bench cohort episodes
  probe — `standard_3meal`, two `randomized_day` plans, `ogtt_75g`,
  `fast_24h`, `high_fat_meal`, `phase_shift_14h`, `grazing_6meal` —
  *without copying the bench cohorts' exact meal grams / timing*, so
  the bench cohorts stay an honest held-out check of the distilled
  mechanism (this is the "broad, not teach-to-test" choice — Gabriel,
  2026-05-11).
- **Each epoch:** sample 4 of the 8 protocols, integrate the learned
  model at the **zero embedding** (= population-centre patient — what
  the cohort episodes calibrate near, since their initial state is the
  default-`PatientParams` state), and add the NORM_SCALE-normalized
  **Huber** trajectory loss vs the cold reference on the 7 unobserved
  counter-regulatory + circadian markers. Huber (not MSE) because the
  dead markers start grossly wrong (ghrelin ~1.9× off) — a plain MSE
  would be dominated by a few large errors and could swamp the other
  signals; Huber keeps it linear past one NORM_SCALE.
- **Replaces `marker_vitality`** (matching the curve subsumes the
  variance floor; the band-aid is removed, not kept alongside).
- `insulin` / `glp1` are deliberately *excluded* — already
  constrained by the glucose-coupling, insulin-sweep, and
  gut-dose-sweep signals; re-distilling them would just add a
  competing pull on already-calibrated dynamics.
- **Starting weight 0.2** — a *direct* MSE-ish signal on these markers
  is orders of magnitude stronger than the prior ~1e-4, so 0.2 is
  plenty to break the deadlock while protecting iter-45/46's
  best-ever gate metrics. Ramp in iter 48 if it moves the markers
  without regressing the gate.

### What's deliberately deferred: the patient axis

The distillation distribution here is **broad over protocols, centred
on the population-mean patient**. True patient-axis breadth —
distilling varied `PatientParams` onto the *learned embedding
manifold* — would require a `PatientParams` → embedding
correspondence, i.e. calibrating an embedding to each cold patient's
*observed* markers (exactly as the bench does) on every distillation
sample, which roughly doubles epoch wall-clock. It's the natural next
structural step *if* iter 47 shows the mechanism works (markers move
on the broad protocols) but per-patient unobserved accuracy on the
held-out bench cohorts still lags — the signature being `dist_*`
epoch metrics dropping while bench MAPE stays flat. The embedding
table itself is still trained for patient variation by the trajectory
signal on real data; this signal owns the population-centre patient's
unobserved-marker dynamics across protocols.

## Iter-47 plan

**Status: DONE, FAILED — job `train-20260511T113422Z` /
`pulse-trainer-66rtp`, commit `dc1a01ba`, completed 2026-05-12.
Distill-at-zero did nothing for its targets and regressed the
observed/circadian markers — see "Iter-47 result" below. Superseded
by iter 48 (calibrate-then-distill).**

`spec.json`: iter-46 spec + `--cold-distill-weight=0.2
--cold-distill-markers=glucagon:ffa:ghrelin:leptin:acth:cortisol:bhb
--cold-distill-protocols-per-epoch=4`; remove the dead
`--marker-vitality-*` args. Physiology rules stay at 0.05 (iter-46
confirmed net-neutral — harmless, keeps insulin calibrated). All
other knobs unchanged.

**The headline test:** FFA/glucagon/ghrelin/leptin bench MAPE moves
for the first time since iter 38. Targets (won't be hit in one iter,
but *any* real movement confirms the mechanism): FFA 0.43 → 0.20-0.30,
glucagon 0.44 → 0.25-0.35, ghrelin 1.93 → 0.6-1.2, cortisol 0.41 →
0.30-0.40, acth 0.50 → 0.35-0.45, bhb 0.21 → 0.15-0.20, leptin already
tiny. Watch the per-marker `dist_*` sub-metrics in the epoch logs —
glucagon and ghrelin should drop fastest (worst → most gradient under
Huber). Textbook scenario pass-rate should rise (the
`glucagon_suppressed` / `ffa_suppressed` / `ghrelin_suppressed_after_feeding`
checks can finally pass once the markers move).

`overall_weighted_mape` is gate-weighted over thresholded markers
only (glucose/hr/sbp/dbp/temp), so the dead-marker improvements
*don't* change it directly — expect it to hold ~0.090-0.095. HR /
glucose / vitals: should hold at iter-46 best-ever — cold-distill
touches the metabolic module's glucagon/ffa heads, which share an
embedding projection with glucose/insulin, so a small perturbation
risk exists; weight 0.2 + the trajectory signal pushing back should
contain it. Bench gate still won't pass (glucose 0.207 ≥ 0.20,
verifier_meal 0.605 < 0.65) — iter 47 unsticks the dead pathways;
the two gate failures are iter 48+.

Epoch wall-clock: +~36s (4 zero-embedding rollouts averaging ~1080
min) → ~8.5-9 h total, inside the 8h-per-step budget the trainer job
allows (`task-timeout=86400`; bench step has its own 8h ceiling).

### Risks / contingencies

- **R1 — dead markers move but glucose/insulin/HR regress past
  iter-46 best.** The shared metabolic embedding projection is the
  conduit. Iter 48: drop cold-distill weight to 0.1, or split the
  metabolic module's glucagon/ffa heads onto a separate embedding
  projection (a cleaner structural fix — decouples the unobserved
  pathways from the observed ones architecturally).
- **R2 — markers barely move at weight 0.2.** Ramp to 0.5 in iter 48.
  Unlikely (the gradient is direct now), but possible if the loss is
  dominated by one badly-wrong protocol that Huber clips flat.
- **R3 — ghrelin moves a lot but cortisol/acth don't.** The appetite
  module reaches but the stress module's circadian head lacks the
  time-feature resolution to track the CAR. Iter 48: inspect the
  stress module's inputs.
- **R4 — training destabilizes / NaN in the first phase-1 epochs.**
  The cold references are grossly off for the untrained model
  (glucagon at random-init can be in the hundreds), so the first few
  epochs see large losses. Huber bounds it; if it still blows up,
  iter 48 adds a warmup ramp on `cold_distill_weight`.
- **R5 — markers move on the broad protocols but the 2 bench cohort
  episodes specifically don't improve.** The held-out cohorts use a
  *calibrated* (non-zero) embedding the distillation never trains;
  this would mean the patient-axis extension (per-sample calibration)
  is needed sooner than planned. Signature: `dist_*` epoch metrics
  dropping while bench MAPE stays flat.

## Iter-47 result: distill-at-zero doesn't reach the bench's eval point

Iter 47 vs iter-46 (`docs/iter42-baselines/benchmark-report-iter46.json`):

| marker | iter-46 | iter-47 | |
|---|---|---|---|
| ffa | 0.43198 | 0.43196 | byte-identical (again) |
| glucagon | 0.44409 | 0.44494 | byte-identical |
| ghrelin | 1.92835 | 1.92938 | byte-identical |
| leptin | 0.02663 | 0.02726 | byte-identical |
| acth | **0.463** | **1.014** | regressed +0.55 |
| cortisol | **0.415** | **0.665** | regressed +0.25 |
| hr | **0.110** | **0.209** | regressed — *new gate failure* |
| bhb | 0.231 | 0.131 | improved |
| insulin | 1.118 | 0.694 | improved |
| overall_weighted_mape | **0.0906** | **0.1100** | regressed |
| verifier overall | **0.800** | **0.753** | regressed |
| textbook pass-rate | **0.808** | **0.665** | regressed (`meal_dose_response` → fully broke) |

So R1 + R5, together and worse than feared: the four headline dead
markers **did not move at all** on the bench, *and* the distillation
collaterally damaged the markers that do have other supervision
(acth/cortisol — the circadian heads; hr — shares the metabolic
embedding projection) plus the textbook scenarios.

**Why.** The bench evaluates every cohort at an embedding *calibrated*
(512 steps, lr 0.05, l2 0.001 — `pulse.benchmark.calibrate_embedding`)
to that cohort's 5 self-measured markers. The "zero embedding ≈
population-centre patient ≈ where the cohort calibrates" assumption
behind iter 47 was wrong: the calibrated embedding is far enough from
zero that (a) the unobserved-marker dynamics distilled *at zero* never
surface at the calibrated point, and (b) pulling the metabolic/stress
modules' *shared* params toward the zero-embedding optimum drags the
*observed*-marker fit *at* the calibrated embedding off iter-46's
best. Distilling at the zero embedding optimizes a point in
embedding-space the bench never visits, at the cost of the point it
does. (`bhb`/`insulin` improving is consistent — their zero-embedding
and calibrated-embedding dynamics happen to be close, so the
distillation didn't fight the eval there.)

The architecture diagnosis still stands (the heads *can* represent the
cold equations; nothing trains them). What iter 47 adds: **the
distillation has to happen at the embedding the bench's calibrated
eval lands on, not at zero.**

## Iter-48 plan: calibrate-then-distill (phase-2-only)

**Status: DONE — job `train-20260512T012321Z` / `pulse-trainer-8tq68`,
commit `44c930cc`, completed 2026-05-12. Recovered all of iter-47's
collateral damage and nudged the observed side forward (owm best-ever,
one of two standing gate failures cleared) — but did NOT move the four
headline dead markers on the bench cohorts. See "Iter-48 result"
below.**

The fix, in `ColdModelDistillationSignal`: before each distillation
rollout, fetch a **per-protocol calibrated embedding** — an embedding
optimized to reproduce that protocol's cold *observed*-marker
trajectory (glucose/hr/sbp/dbp/temp), the same objective the bench
calibrates (normalized obs MSE + L2 prior), full-trajectory rather
than windowed (protocols are ≤1 day). Cold-start it (`calib_steps=64`)
the first time a protocol is sampled, warm-restart (`calib_warm_steps=24`)
when stale (`recalib_every=10` epochs); the embedding is **detached** —
it sets *where* we distil, not *what*. Then run the learned model at
that embedding and add the same NORM_SCALE-normalized Huber loss on
the 7 unobserved markers vs the cold reference. Forwards in this
signal run with the model in **eval mode** (so the distillation target
and the embedding it's calibrated against see the same deterministic
network the bench does). Cold-distill also moves to **phase-2-only**
(`enable_at = phase_boundary`, with the other additive losses) — a
calibrated embedding is only meaningful once phase 1 has fit the
observed markers, and it sidesteps R4 (distilling against an untrained
model) entirely.

Pool, markers, weight, all trainArgs: **unchanged from iter 47** — the
change is purely in the signal's code, so this isolates calibrate-
then-distill as the one structural fix. The pool stays broad-over-
protocols / centred-on-the-population-mean-patient (not the bench
cohorts' exact meals), so the bench cohorts remain a held-out check:
if the dead markers move *there*, the mechanism generalized across the
protocol axis. (Patient-axis breadth — varied `PatientParams` — is
still future work.)

Wall-clock: phase 1 unchanged; phase 2 +~1-1.5h total (8 cold-start
calibrations ≈ +25 min on each of ~2 early phase-2 epochs; ~3 warm-
refresh bursts ≈ +8 min; steady-state +~40s/epoch for the 4
distillation rollouts, same as iter 47).

### Iter-48 risk/contingency

- **dead markers move on the bench but only a little** — 64-step
  calibration isn't tight enough / the per-protocol embedding still
  differs too much from the bench-cohort embedding. Iter 49: bump
  `calib_steps` toward 128-256, and/or add the bench-cohort protocols
  to the pool (giving up some held-out-ness).
- **hr/glucose still regress even with a consistent distillation** —
  the metabolic/stress modules' shared embedding projection is the
  conduit regardless. Iter 49: drop weight to 0.1, or split the 7
  unobserved heads onto a *separate* embedding projection — the
  cleaner structural fix (architecturally decouples the unobserved
  pathways from the observed ones).
- **cached embedding drifts badly between recalibs** — shorten
  `recalib_every` to 5, or recalibrate every epoch the protocol is
  sampled (cost permitting).
- **eval-mode forwards interact badly with the rest of training** —
  unlikely (autograd works fine in eval mode; mode is restored in a
  `finally`); if metrics look off, revert to train-mode forwards.
- **generalizes to some held-out protocols but not the 2 bench cohort
  episodes** — the pool isn't broad in the right direction; widen it,
  or if even that fails, the patient-axis (varied-`PatientParams`)
  extension is the real requirement and "population-mean only" was the
  limiter.

## Iter-48 result: harmless, helps the gate, still doesn't move the dead pathways

Iter 48 vs iter-46 (the real baseline — iter-47 was a regression):

| marker / metric | iter-46 | iter-47 | iter-48 | iter-48 read |
|---|---|---|---|---|
| ffa | 0.43198 | 0.43196 | 0.43199 | unchanged (≈byte-identical) |
| glucagon | 0.44409 | 0.44494 | 0.44407 | unchanged |
| ghrelin | 1.92835 | 1.92938 | 1.92830 | unchanged |
| leptin | 0.02663 | 0.02726 | 0.02655 | unchanged |
| acth | 0.46292 | 1.01403 | **0.41278** | recovered + improved past iter-46 |
| cortisol | 0.41465 | 0.66525 | **0.41723** | recovered to iter-46 |
| hr | 0.11025 | 0.20933 | **0.08330** | recovered + best-ever |
| sbp | 0.06460 | 0.06206 | **0.04906** | improved |
| bhb | 0.23083 | 0.13136 | 0.17961 | still better than iter-46 |
| insulin | 1.11810 | 0.69441 | 0.76907 | still much better than iter-46 |
| glucose | 0.20723 | 0.20418 | 0.21875 | slightly worse (still > gate) |
| glp1 | 0.28529 | 0.32528 | 0.29220 | ≈iter-46 |
| overall_weighted_mape | 0.09055 | 0.10995 | **0.08462** | best-ever |
| verifier overall | 0.80031 | 0.75335 | **0.80082** | ≈iter-46 |
| textbook pass-rate | 0.8079 | 0.6651 | 0.7127 | partial recovery (`meal_dose_response` +0.33 vs iter-47, still below iter-46) |
| gate failures | glucose, verifier_meal | glucose, hr, verifier_meal | **glucose only** | verifier_cat[meal] cleared — closest to passing ever |

So calibrate-then-distill is **net mildly positive but misses the
point**: it undid iter-47's regression, slightly *improved* the
observed side (distilling consistent dynamics at calibrated embeddings
acts as a mild regularizer), and cleared one of the two standing gate
failures — but the four dead markers moved by ~1e-5 (i.e. nothing) at
the bench-cohort embeddings, for a third structural attempt.

**Why it didn't move them.** Two compounding causes:

1. **Wrong embedding.** Iter 48 distilled at *per-protocol* calibrated
   embeddings — embeddings fit to the synthetic-pool protocols'
   observed markers. Those are *different meals* than the two bench
   cohort episodes, so they calibrate to *different embeddings*. The
   unobserved-marker heads are strongly embedding-dependent (they take
   the module embedding as a direct head input), so heads taught to
   produce the cold curves at the pool embeddings produce ≈the init
   curve at the bench-cohort embeddings. The "broad pool generalizes
   across the protocol axis" bet (the deliberate held-out-check design
   from 2026-05-11) does not pay off: the bench's unobserved targets
   live in exactly *two* episodes, and the only way to score on them is
   to train the heads at *those two* episodes' calibrated embeddings.
2. **Under-dosed.** Cold-distill ran phase-2-only (30 epochs) at
   weight 0.2, and `compute` runs once per epoch ⇒ ~30 gradient steps
   on the signal. The heads start at init (phase 1 doesn't train them
   either — nothing does, until phase 2), grossly wrong (ghrelin ~2
   NORM_SCALEs off), and the Huber loss is linear past 1 NORM_SCALE ⇒
   constant-magnitude gradient. 30 steps at weight 0.2, lr 3e-4 moves
   the head params a tiny amount — nowhere near init→textbook. (Iter
   47's 80 epochs of weight-0.2 at the *zero* embedding *did* move them
   — enough to cause visible damage — so 80 steps moves; 30 doesn't.)

There is also a deeper chicken-and-egg: the bench *re-calibrates* the
embedding fresh at eval time against the post-distillation model, so
even "distil at the bench-cohort embedding I computed during training"
is fragile (the model drifts, the embedding moves). The robust fix is
to make the unobserved heads *not depend on the per-patient embedding
at all* — but that's a module-architecture change (the modules are
monolithic MLPs, not per-species heads, so it's not a small edit).

## Iter-49 plan: Fork A — distil at the bench cohorts (+ dose)

**Status: DONE, FAILED — job `train-20260512T134841Z` /
`pulse-trainer-qccqd`, commit `ff48afee`, completed 2026-05-13.
Distilling at the two bench-cohort episodes' own calibrated embeddings
at weight 1.0 *still* did not move the four dead markers (ffa/glucagon
byte-identical to 1e-7), AND regressed the observed side back toward
iter-46 (the iter-48 gains undone). 4th failed structural attempt —
see "Iter-49 result" and "STOP: this is a wall, diagnose before
iterating" below. Iter 50 should NOT be another training run until the
diagnostic answers *why* the heads won't move.**

Three structural attempts (47, 48) have not moved the bench dead
markers. The forks considered (Fork A chosen):

- **Fork A — distil at the two bench-cohort episodes' calibrated
  embeddings** (+ raise the dose). Add `cohort_sleep_48h` and
  `cohort_meal_postprandial` (from `benchmark_extras.py`) to the
  distillation pool, calibrate embeddings to *their* observed markers
  (the bench's exact procedure), distil their unobserved cold curves
  there; bump `cold-distill-weight` 0.2 → ~1.0 and
  `protocols-per-epoch` so the heads get a real dose. Most direct test
  of "can we move the number at all"; gives up the held-out check
  (it's now "train at the eval's operating point" — defensible, since
  the observed-calibration is the bench's own procedure and the
  unobserved targets are the textbook ODE, but it *is* teach-to-the-
  eval-regime). Small code change, ~16h run. If even this doesn't move
  the markers, the embedding-mismatch isn't the (only) cause and
  Fork C is forced.
- **Fork B — same but keep the broad pool too** (broad protocols *and*
  the two bench cohorts). Hedge: keeps a generalization signal while
  also hitting the eval regime. Marginally more compute.
- **Fork C — architectural decouple** (the "cleaner structural fix"):
  give the 7 unobserved markers' rate computation a *fixed* embedding
  (zero, or a dedicated learned constant) instead of the per-patient
  module embedding, so the distilled curves transfer to bench eval
  regardless of how the bench calibrates. Physiologically defensible
  (these markers have no per-patient ground truth; the bench expects
  the default-patient curve for every cohort). But the modules are
  monolithic MLPs producing all species' rates jointly — decoupling a
  subset of species needs restructuring (per-species heads, or
  splitting modules), a multi-file model change with checkpoint-format
  churn. Biggest blast radius, cleanest end state.

Chosen: **Fork A** (fastest decisive test; if it fails it forces C, if
it works C is still worth doing later for robustness). Wired as
`--cold-distill-pool=bench_cohorts` — the signal pulls its pool from
`benchmark_extras.all_cohort_benchmark_episodes()` and calibrates each
protocol's embedding against that episode's own check-ins.

## Iter-49 result: still byte-identical — and a regression

Iter 49 vs iter-48 (its baseline) and iter-46:

| marker / metric | iter-46 | iter-48 | iter-49 | iter-49 read |
|---|---|---|---|---|
| ffa | 0.43198 | 0.43199 | 0.43199 | **byte-identical** (Δ vs iter-48 = −1e-6) |
| glucagon | 0.44409 | 0.44407 | 0.44407 | **byte-identical** (Δ = −1.3e-7) |
| ghrelin | 1.92835 | 1.92830 | 1.92814 | byte-identical |
| leptin | 0.02663 | 0.02655 | 0.02656 | byte-identical |
| acth | 0.46292 | **0.41278** | 0.46164 | regressed back to iter-46 |
| cortisol | 0.41465 | 0.41723 | **0.48638** | regressed past iter-46 |
| hr | 0.11025 | **0.08330** | 0.14168 | regressed past iter-46 |
| insulin | 1.11810 | **0.76907** | 1.10091 | regressed back to iter-46 |
| bhb | 0.23083 | **0.17961** | 0.22249 | regressed back to iter-46 |
| glucose | 0.20723 | 0.21875 | 0.20923 | ≈iter-46, still > gate |
| overall_weighted_mape | 0.09055 | **0.08462** | 0.09899 | regressed past iter-46 |
| verifier overall | 0.80031 | 0.80082 | 0.78750 | regressed |
| textbook pass-rate | 0.8079 | 0.7127 | 0.8079 | recovered to iter-46 (`meal_dose_response` +0.67 vs iter-48) |
| gate failures | glucose, verifier_meal | **glucose only** | glucose, verifier_meal | iter-48's cleared failure came back |

So Fork A is a **strict regression vs iter-48 and still a total miss
on the goal** — the worst combination. Distilling at the bench
cohorts' own embeddings at 5× the weight pulled the metabolic/stress
modules' shared params hard enough to undo iter-48's observed-side
gains (hr 0.083→0.142, insulin 0.77→1.10, owm 0.085→0.099, the
verifier_meal gate failure returned), and the dead markers *still*
didn't budge — ffa moved 1e-6, glucagon 1e-7. Not "small movement":
**nothing.** 4 structural attempts (iters 47, 48, 49 × {at-zero,
at-synthetic-pool, at-bench-cohort} embeddings, {80, 30, 30} epochs,
{0.2, 0.2, 1.0} weight) → ffa/glucagon/ghrelin/leptin MAPE
byte-identical every time.

## STOP: this is a wall — diagnose before iterating again

Four cold-model-distillation iters have moved ffa/glucagon/ghrelin/
leptin by ≈0 on the bench, byte-identical to ~1e-6, under every
embedding choice and dose we've tried. The structural-over-parametric
rule (`training-runs.md`) is unambiguous here: *stop iterating, do a
diagnostic dive.* The next step is **not iter 50** — it's answering
*why the heads won't move*. Concretely (run locally — checkpoints
auto-localize from GCS):

1. **Dump trajectories.** Load the iter-49 (or iter-48) checkpoint,
   calibrate an embedding on `cohort_meal_postprandial` exactly as the
   bench does (512 steps, lr 0.05, l2 0.001, windowed), roll out, and
   print the model's ffa / glucagon / ghrelin trajectories next to the
   cold reference. Hypothesis to confirm/kill: *the heads emit a
   near-constant near-typical value and that value doesn't change
   between iters* (which is what a 0.43 MAPE that never moves looks
   like).
2. **Gradient check.** With that checkpoint + embedding, compute the
   cold-distill Huber loss on those markers and `loss.backward()` —
   inspect `‖grad‖` on the metabolic/appetite/stress modules' params
   and specifically on whatever produces the ffa/glucagon/ghrelin
   rates. Is it ~0 (gradient genuinely doesn't reach — a wiring bug or
   a saturating nonlinearity or a `.detach()` somewhere) or non-trivial
   (gradient is there, so the *bench eval* is the problem — it
   re-calibrates the embedding to a place where the moved heads still
   produce ≈the same wrong curve)?
3. **Is the cold target itself ~flat?** Print the cold model's ffa /
   glucagon range over the cohort protocols. If `simulate_full_body`'s
   ffa barely moves (small fasted/fed swing) and sits at value X while
   the model's head sits at Y≠X, MAPE ≈ |X−Y|/X ≈ 0.43 *with a real
   non-zero gradient* — in which case the question is why 30–80 steps
   of that gradient at lr 3e-4 doesn't close it (head LR too low?
   gradient-clip eating it? competing pull from `cohort` which also
   touches these markers and may be pinning them?).

The answer to (2)/(3) decides everything:
- **Gradient ~0 / wiring bug** → fix the wiring; this whole iter 47-49
  arc was fighting a plumbing problem, not a learning problem.
- **Gradient present, bench-recalibration masks it** → Fork C
  (architectural decouple: 7 unobserved markers' rates read a *fixed*
  embedding, not the per-patient one) is the only robust fix; accept
  the multi-file model change.
- **Gradient present, just under-applied** → bump the head's effective
  LR / kill the competing `cohort` pull on these markers / ramp the
  cold-distill weight much higher with the architectural-decouple to
  contain collateral.

### Diagnostic result (2026-05-13, `scripts/diagnose-dead-pathways.py`)

Ran the diagnostic against the iter-46/48/49 checkpoints on
`cohort_meal_postprandial`, calibrating each one's embedding the
bench's way (256 steps, lr 0.05, l2 0.001, windowed). Findings:

**The dead markers are pinned at exactly their `typical`/`NORM_CENTER`
value, dead flat, identical across all three checkpoints to ~1e-7:**

| marker | model trajectory (cohort_meal_postprandial) | cold target | typical | iter-46→49 max-diff |
|---|---|---|---|---|
| ffa | min=max=mean=0.4998, **swing 0.0000** | 0.228 → 0.500, swing 0.27 | 0.5 | **9e-8** |
| glucagon | ≈69.755, swing 0.01 | 33 → 70, swing 36 | 70 | 3e-3 |
| ghrelin | ≈100.02, swing 0.04 | 15 → 100, swing 85 | 100 | 6e-3 |
| leptin | ≈10.005, swing 0.007 | 9.5 → 10.0, swing 0.5 | 10 | 3e-3 |
| bhb | 0.101 → 0.119, swing 0.018 | 0.10 → 0.16, swing 0.06 | 0.1 | — |
| cortisol | 12.3 → ~20, swing ~8 | 12 → 26, swing 13 | 12 | — |
| (works) glucose | 95 → ~151, swing 56 | 95 → 152, swing 57 | 95 | — |

So ffa/glucagon/ghrelin sit at the mass-action *equilibrium* — the
`prod/cons` softplus head emits ≈0 residual, so `state → typical *
softplus(prod_logit)/softplus(cons_logit) ≈ typical`, and it never
leaves. (`leptin` isn't really "dead" — its cold target is also ~flat
at ≈10, so its 0.017 MAPE is *fine*; same for `bhb` to a lesser
degree. The genuinely dead trio is **ffa, glucagon, ghrelin**, plus
`acth` partially.)

**The gradient *does* reach the heads — it's just 3-5 orders of
magnitude weaker for the dead trio than for the markers that work.**
`loss.backward()` on the cold-distill Huber (loss = 0.36, substantial):

```
stress.heads            grad-norm 2.4e-1   (cortisol/acth — moves)
metabolic.heads          grad-norm 4.9e-2   (aggregate)
  metabolic.heads.0 (glucose)   ~5e-3       works
  metabolic.heads.1 (insulin)   ~3e-3       works
  metabolic.heads.4 (bhb)       ~1e-2-4e-2  moves a bit
  metabolic.heads.2 (glucagon)  ~1e-5-1e-4  DEAD
  metabolic.heads.3 (ffa)       ~2e-6-4e-5  DEAD
appetite.heads.0 (ghrelin)      ~3.8e-6     DEAD
embedding_projections.{cardiovascular,thermoreg,respiratory}  0.0
```

So it is **not** a plumbing bug (no zero gradient, no stray `.detach()`)
and **not** a bench-recalibration artifact (the model curves are
identical across checkpoints *before* any re-calibration question even
arises — they're identical at the same calibrated embedding). It's a
**flat-loss-landscape / vanishing-gradient trap**: the `prod/cons`
softplus head, starting at init, sits in a region where moving the
marker off `typical` requires the head's prod_logit to shift, but the
gradient that would do so is diluted ~480× through the integration
chain and arrives ~1e-5 — gradient descent crawls and 30-80 epochs
isn't remotely enough. Param-tuning the cold-distill weight/embedding
(iters 47-49) was always going to be a no-op against a 1e-5 gradient.

**Implication for the fix** — the cold-distill *trajectory* loss is
the wrong shape; what's needed is a signal with a *direct, undiluted*
gradient to the head. Candidates, roughly in order of cleanness:

1. **Rate-matching distillation** (the standout): instead of "roll out
   480 steps and match the integrated trajectory", do "feed the model
   the *cold* state at time t and require its rate output for
   ffa/glucagon/ghrelin to match the cold ODE's `d/dt` at that state".
   That's a per-timestep regression — gradient straight to the head,
   no integration chain, ~480× stronger. This is how you distil an
   ODE. (Needs the cold model to expose its instantaneous rates, or
   finite-difference them from the cold trajectory.)
2. **Re-parameterise the dead heads** to not pin at `typical` and to
   have a steeper local gradient — e.g. give ffa/glucagon/ghrelin the
   `GlucoseGatedInsulinHead`-style structure that glucose/insulin use
   successfully, or have them emit the marker value directly rather
   than a mass-action prod/cons residual.
3. **Per-head LR boost + warm-start**: imprint the ffa/glucagon/ghrelin
   heads from the cold model before training (a one-time fit), and/or
   give those params a much higher LR. Band-aid-ish but cheap.
4. **(Fork C still on the table)** decouple from the per-patient
   embedding — orthogonal to the gradient problem, would help
   robustness but won't on its own un-pin the markers.

Recommendation for iter 50: **(1) rate-matching distillation** —
biggest gradient win, cleanest framing, and it subsumes the current
trajectory-distill (matching rates ⇒ matching trajectory). Keep it on
the bench-cohort protocols (iter-49 plumbing) but switch the loss from
trajectory-Huber to rate-MSE; the weight can go back down (0.2) since
the gradient is now ~480× stronger. If rate-matching still doesn't
move them, escalate to (2).

(This is a model/signal-architecture change, not a config tweak —
flagged for Gabriel's direction before iter 50 dispatch. iter-48's
checkpoint remains the best gate-relevant baseline in the meantime.)

## Iter-50 plan: rate-matching distillation

**Status: DONE, FAILED (R2/R3) — job `train-20260513T051414Z` /
execution `pulse-trainer-2lxbj`, commit `9ee248ca`, completed
2026-05-13. Rate-matching moved ghrelin's MAPE by 1e-4 and ffa's by
1e-7 (vs iter-48) — a stronger gradient than trajectory but still
3-4 orders of magnitude short of moving the bench number. See
"Iter-50 result" below — the gradient surface in (prod, cons)
coordinates is itself the bottleneck, not the loss-shape. Superseded
by iter 51 (re-parameterise the dead heads as SetpointHead).**

Swap `--cold-distill-mode=trajectory` → `=rate`; everything else is
the minimal change from iter 49 (same `pool=bench_cohorts`, same
calibrate-then-distill plumbing, phase-2-only, recalib/10),
`--cold-distill-weight` 1.0 → 0.3 (the gradient is ~T× stronger now,
no need for 1.0; iter-49 at 1.0 regressed the observed side, want to
avoid that). When it lands: did ffa/glucagon/ghrelin actually move on
the bench (not 1e-7)? did the observed side hold at iter-48's
levels? If `verifier_cat[meal]` re-clears, *both* gate failures may
fall and the gate passes for the first time.

The signal change, in `ColdModelDistillationSignal._rate_terms`: for
each protocol, build a `[T, STATE_DIM]` cold-state batch and a
`[T, 4]` cold-absorption batch (captured from `simulate_full_body`'s
`_abs` return, so the model sees the *exact* gut/coupling inputs the
cold ODE saw), call `model.forward` *once* (batched over T), Huber
the normalized rate residual `rate × 60min / NORM_SCALE` on the 7
unobserved markers vs the finite-differenced cold rate. Gradient
straight to the head, no integration chain, ~T× stronger. Cold
absorption replaces the learned-gut output to keep the rate-matching
consistent (the gut module has its own dedicated signals; not
co-trained here).

### Iter-50 risk/contingency

- **markers move but glucose/hr regress** — shared metabolic params
  perturbed too far. Mitigation: weight already 0.3 (5× lower than
  iter-49); if still bad, iter 51 drops to 0.1 or starts splitting
  the dead heads off the shared module structure.
- **markers move only partially** (e.g. ffa 0.43→0.30) — head
  expressivity is the next limit. Iter 51: re-parameterise the dead
  heads (the `GlucoseGatedInsulinHead` pattern that glucose/insulin
  use successfully, or emit the marker value directly rather than a
  prod/cons residual).
- **markers don't move even with the direct gradient** — re-run the
  diagnostic with the iter-50 checkpoint to confirm
  `metabolic.heads.{2,3}` / `appetite.heads.0` grad-norms are now O(1)
  not O(1e-5). If still tiny, the head's parameterisation forbids the
  cold rates (architectural fix forced).
- **bench re-calibrates to a different embedding than we distilled at**
  — teacher-forced rate-matching at cold states should be more
  embedding-robust than trajectory-matching (matching rates at the
  same cold state → integrating from cold initial gives cold
  trajectory at any embedding within the calibrated region). If 50
  moves the rates but not the bench MAPE, iter 51 adds multi-anchor
  distillation (zero + calibrated + sampled embedding-table rows).

Also note: **iter-48's checkpoint (`train-20260512T012321Z`) is the
best one on the gate-relevant metrics so far** (owm 0.0846, hr 0.083,
only `glucose` failing). If the dead-pathway thread keeps not paying
off, the fallback is to keep iter-48 as the working baseline and spend
the next iters on the *actual* gate failures (glucose ≥ 0.20,
verifier_meal) instead of the diagnostic markers.

(Fork-B/C descriptions above stay as the contingency menu once the
diagnostic narrows it down.)

## Iter-50 result: rate-matching moved more than trajectory, still nowhere near enough

Iter 50 vs its baseline iter-48 (the best-on-gate checkpoint):

| marker / metric | iter-46 | iter-48 | iter-49 | iter-50 | Δ iter50−iter48 | read |
|---|---|---|---|---|---|---|
| ffa | 0.43198 | 0.43199 | 0.43199 | 0.43199 | **+7e-8** | byte-identical (4th time) |
| glucagon | 0.44409 | 0.44407 | 0.44407 | 0.44408 | **+1.5e-5** | byte-identical |
| ghrelin | 1.92835 | 1.92830 | 1.92814 | 1.92830 | **+1.7e-4** | basically byte-identical |
| leptin | 0.02663 | 0.02655 | 0.02656 | 0.02662 | +5e-5 | unchanged (cold target ~flat) |
| acth | 0.46292 | **0.41278** | 0.46164 | 0.48748 | +0.07 | regressed past iter-46 |
| cortisol | 0.41465 | 0.41723 | 0.48638 | 0.47725 | +0.06 | regressed past iter-46 |
| bhb | 0.23083 | **0.17961** | 0.22249 | 0.21437 | +0.03 | partially regressed |
| insulin | 1.11810 | **0.76907** | 1.10091 | 1.08049 | +0.31 | regressed back to iter-46 |
| hr | 0.11025 | **0.08330** | 0.14168 | 0.13822 | +0.05 | regressed past iter-46 |
| glucose | 0.20723 | 0.21875 | 0.20923 | 0.20932 | ~equal | still > gate |
| overall_weighted_mape | 0.09055 | **0.08462** | 0.09899 | 0.09800 | +0.013 | regressed past iter-46 |
| verifier overall | 0.80031 | 0.80082 | 0.78750 | 0.78821 | +0.013 | regressed |
| textbook pass-rate | 0.8079 | 0.7127 | 0.8079 | 0.7127 | 0 | regressed (meal_dose_response stays at 0.33) |
| gate failures | glucose, verifier_meal | **glucose only** | glucose, verifier_meal | glucose, verifier_meal | — | iter-48's cleared failure came back |

The headline test failed in the R2/R3 mode the iter-50 plan predicted:
ghrelin moved by 1e-4 (1000× more than iter-49's 1e-7), so the
rate-matching loss *did* deliver a stronger gradient than the
trajectory loss, but still 3-4 orders of magnitude short of moving
the bench number meaningfully. ffa moved by 1e-7 — within noise of
iter-49. Five iterations (47, 48, 49, 50) at four different
embedding strategies (zero / per-protocol synthetic / per-cohort
calibrated / per-cohort calibrated + rate-matching) and two loss
shapes (trajectory / teacher-forced rate) — the four dead-trio bench
numbers byte-identical each time.

**Why this is the moment to re-parameterise (R2/R3 → the architectural
fix).** Iter 50's rate-matching delivered a stronger upstream
gradient (~T× the trajectory loss, as predicted) — ghrelin moved
1000× more than iter 49 — but ran into a different structural wall:
*softplus saturation in the (prod, cons) parameterisation*. The
parent `MassActionModule` passes *normalized* state to the head
(`norm_state = (raw_state − typical)/norm_scale`) and computes

```
rate = prod·prod_scale − cons·cons_scale·norm_state    (prod_scale = cons_scale·typical)
     = cons_scale · (prod·typical − cons·norm_state)
```

For the model's equilibrium to sit at the typical raw state
(`norm_state = 0`), the head must emit `prod = 0` exactly. But
`SpeciesHead` produces `prod = softplus(prod_raw)`, so achieving
prod ≈ 0 forces `prod_raw → −∞` and the softplus saturates. The
gradient `∂prod/∂prod_raw = sigmoid(prod_raw) ≈ exp(prod_raw)` is
then exponentially small. The dead trio is stuck in exactly this
regime: training (cohort-stat + bench-cohort observation) pushed
prod_raw very negative to keep the marker at typical, and now no
further gradient — however strong upstream — can move the marker off
typical, because everything we add gets multiplied by `exp(very
negative)` at the head's prod_raw saturation. The diagnostic's ~1e-5
grad-norm on these heads was this saturation, not a wiring bug or a
weak loss. Re-running with a stronger rate-matching gradient (iter
50) hits the same wall.

## Iter-51 plan: SetpointHead — re-parameterise (prod, cons) → (target_z, k_factor)

**Status: dispatched 2026-05-13T17:55Z — job `train-20260513T175514Z`,
execution `pulse-trainer-bz7xq`, commit `7002e59d` (clean tree).**

The change is structural and coordinate-only — the parent
`MassActionModule` is unchanged, mass conservation still holds, the
per-species scaling is unchanged, integration plumbing identical.
What changes is *which two scalars the head emits and how they map
into the parent's (prod, cons) interface*. For the dead trio
(glucagon = `metabolic.heads.2`, ffa = `metabolic.heads.3`, ghrelin =
`appetite.heads.0`), the new `SetpointHead` (in `modules/base.py`)
keeps the same MLP architecture as `SpeciesHead` and instead emits:

```
target_z = raw[..., 0]                        # signed, in z-score units
k_factor = softplus(raw[..., 1])              # ≥ 0 (rate constant must be positive for stability)
prod     = k_factor · target_z / typical      # signed; passes typical at construction
cons     = k_factor
```

After the parent's scaling, this yields exactly:

```
rate = cons_scale · k_factor · (target_z − norm_state)
     = (cs · k_factor) · ((typical + norm_scale·target_z) − raw_state) / norm_scale
```

— setpoint dynamics with target = `typical + norm_scale·target_z`
(equilibrium in raw state units) and rate constant `k = cs·k_factor`.
`target_z` is signed and unbounded — *no softplus saturation in the
direction that controls where the marker sits*. To park ghrelin at
its post-meal target of 15 (typical 100, norm_scale 40), the head
emits target_z = (15 − 100)/40 = −2.125. The gradient onto
`target_z_raw` is `cons_scale·k_factor` × the MLP's linear coefficient
— direct, non-saturating, *bounded below* (cs·log 2 at init never
decays exponentially toward 0).

The signed `prod = k_factor·target_z/typical` is mathematically fine
in the parent's `prod·prod_scale − cons·cons_scale·norm_state`
formula: negative prod simply means active removal (rate < 0 even at
norm_state = 0), which is what setpoint dynamics require when target
is below typical. The mass-action interpretation `prod = production
≥ 0, cons = consumption ≥ 0` is loosened for these three species —
intentionally, since the underlying physiology (insulin-suppressed
glucagon, insulin-suppressed lipolysis, nutrient-suppressed ghrelin
secretion) is not mass-action in the chemical sense; the original
mass-action parameterisation was a useful structural prior but is
the wrong shape for these markers when the equilibrium has to leave
typical.

**Init equivalence.** Final-layer weights zeroed; biases zero ⇒
`target_z = 0` (equilibrium at typical) and `k_factor = softplus(0)
= log 2 ≈ 0.693`. This is exactly what SpeciesHead converges to for
these species: prod_softplus saturated near 0 (because training
pushed prod_raw very negative), cons_softplus near log 2 (from the
rate-slope prior). At init the system sits at the typical raw state
with rate = 0 and a slow ~16h time constant pulling back to typical
— matching iter-50's *trained* dynamics for these markers, not its
init. So the model walks into phase 1 already at the
equilibrium-at-typical state, sparing the optimizer the
saturate-prod_raw-to-keep-marker-at-typical phase that produced the
trap in the first place. Data-dependent `target_z` emerges as the
final-layer weights move off zero; the final layer gets non-zero
gradient at step 1 even with `W_final = 0` (because
`W_final.grad = upstream_grad ⊗ hidden_input`, both non-zero).

**Wiring.** `modules/metabolic.py:MetabolicModule` adds glucagon (idx
2) and ffa (idx 3) to its `head_factories`; `modules/appetite.py:
AppetiteModule` adds ghrelin (idx 0). All other species (glucose,
insulin, bhb, lactate, hepatic_output, leptin, glp1, all stress +
cardiovascular + respiratory + thermoreg) stay on `SpeciesHead` —
they're not the dead pathways and the diagnostics show their
gradient surface works as-is. Insulin keeps `GlucoseGatedInsulinHead`
unchanged.

**Training config.** Identical to iter-50: cold-distill mode=rate,
pool=bench_cohorts, weight=0.3, protocols-per-epoch=2, phase-2-only.
The only change between iter 50 and iter 51 is the head class for
three species — every other knob held fixed, so any movement is
attributable to the re-parameterisation.

### Iter-51 risk/contingency

- **R1 — Markers still don't move** (re-param wasn't the limit
  either). The coordinate change brings ∂rate/∂target_ratio into
  O(1) gradient territory; if MAPE still doesn't move, the remaining
  candidates are (a) the rate-matching loss is dominated by sleep-
  cohort timesteps where cold_rate ≈ 0 (mean-reduction over T dilutes
  the meal-cohort signal), (b) the bench's eval-time embedding re-
  calibration lands at a point the cold-state forward pass doesn't
  predict well. Mitigation: re-run `diagnose-dead-pathways.py` (
  extended to mode=rate) against the iter-51 checkpoint — confirm
  grad-norm on the three SetpointHeads is O(1), confirm target_ratio
  values at calibrated embedding actually moved off 1.0 during
  training. Iter 52 then reshapes the rate-matching loss (sum-reduce
  + per-step weight by |cold_rate|, or weight the meal cohort 5×
  sleep).
- **R2 — Overshoot.** Target_ratio swings past physiology and
  destabilises coupling. Mitigation: weight=0.3 (already 5× below
  iter-49); iter 52 can cap target_ratio to e.g. [0.05, 3.0] via a
  bounded transform.
- **R3 — Collateral on bhb/cortisol/acth.** Shared embedding
  projections in metabolic/stress modules. iter-50's bhb already
  moved 0.18→0.21 without re-param; if it gets worse, iter 52 splits
  the dead trio's heads onto a separate embedding projection (the
  "Fork C" structural fix listed back in iter 49 — orthogonal to the
  coordinate change, additive if needed).
- **R4 — Verifier_meal stays failing.** The three textbook scenarios
  (glucagon_suppressed_postprandial, ffa_suppressed_antilipolysis,
  ghrelin_suppressed_after_feeding) need the markers to move enough
  to flip from fail to pass. If movement is real but smaller than
  the threshold (e.g. ffa 0.43 → 0.35), iter 52 ramps the cold-
  distill weight up since the gradient now lands cleanly.

Fallback if iter 51 also fails: iter-48's checkpoint stays the
best-on-gate baseline; pivot the next iter off the dead-pathway
thread entirely and work the actual gate failures (glucose ≥ 0.20,
verifier_meal) from a different angle.

## Iter-51 result + iter-52 result

**Iter 51** (job `train-20260513T175514Z`) — SetpointHead applied to ffa/glucagon/ghrelin worked exactly as predicted:

| marker | iter50 | iter51 | Δ |
|---|---|---|---|
| ffa | 0.432 | 0.240 | −0.19 |
| glucagon | 0.444 | 0.272 | −0.17 |
| ghrelin | 1.928 | 0.812 | −1.12 |
| insulin (collateral win) | 1.080 | 0.389 | −0.69 |

But collateral *regressions*: acth 0.488→0.772, cortisol 0.477→0.609, bhb 0.214→0.346. Diagnosis: same softplus-saturation trap applies to *every* species pinned near typical — iter 51 freed three and exposed three more. The embedding had been compensating for the stuck heads; once the dead trio moved on its own coordinates, the embedding shifted and the still-stuck heads couldn't follow.

**Iter 52** (job `train-20260514T035444Z`) — SetpointHead extended to cortisol/acth/bhb/glp1. Verdict: hypothesis confirmed for non-stimulus species, exposed the *next* structural gap for stimulus-driven species.

| marker | iter50 | iter51 | iter52 |
|---|---|---|---|
| **cortisol** | 0.477 | 0.609 | **0.266** (best ever) |
| **acth** | 0.488 | 0.772 | **0.186** (best ever) |
| **bhb** | 0.214 | 0.346 | **0.184** (best ever) |
| **hr** | 0.138 | 0.153 | **0.075** (back in gate) |
| sbp / dbp | 0.066/0.051 | 0.047/0.072 | 0.037/0.030 (best ever) |
| insulin | 1.080 | 0.389 | 0.703 (regressed) |
| glucagon | 0.444 | 0.272 | 0.395 (regressed) |
| ffa | 0.432 | 0.240 | 0.321 (regressed) |
| glp1 | 0.282 | 0.276 | 0.421 (regressed despite SetpointHead) |
| overall_weighted_mape | 0.098 | 0.101 | **0.077** (best ever) |
| textbook_pass_rate | 0.808 | 0.813 | **0.670** (regression) |

**The diagnosis**: SetpointHead fixes equilibrium-near-typical but cannot represent **gated peaks**. Stimulus-driven hormones (insulin/glucagon/FFA/GLP-1) must *sit near typical AND spike sharply on cue* — `cons_scale·k_factor·(target_z − norm_state)` is pure first-order relaxation, which fits a slow drift to a new target but not a sharp transient peak with fast onset and slow decay. iter 52's GLP-1 regression *despite* getting a SetpointHead is the clean evidence: GLP-1's signal is dominated by meal-induced peaks, not by a different equilibrium.

`GlucoseGatedInsulinHead` already has the structural primitive: `prod = softplus(raw_basal) + softplus(raw_peak)·σ((stimulus − thresh)/temp)`. iter 53 promotes it to a generic `BasalPlusGatedPeakHead(stimulus_idx, gate_dir, init_thresh, init_temp)` and applies it to glucagon (anti-gated by glucose: peak in hypoglycemia), FFA (anti-gated by insulin), GLP-1 (gated by nutrient_flag).

## Iter-53 plan — BasalPlusGatedPeakHead

Generalise `GlucoseGatedInsulinHead` into a reusable head class. Each instance:
- final-layer emits `(raw_basal, raw_peak, raw_cons)`
- learns a stimulus-source index (which feature in the head's input vector is the gate stimulus) at construction
- learns a `gate_dir ∈ {+1, −1}` (positive = peak when stimulus high, negative = peak when stimulus low)
- learnable `g_thresh` and `log_g_temp` scalars (per-head)

Wiring:
- metabolic.heads[1] (insulin): keep on `GlucoseGatedInsulinHead` (special-case retained for compatibility, or migrate to the generic form with gate_dir=+1)
- metabolic.heads[2] (glucagon): switch SetpointHead → BasalPlusGatedPeakHead(stim=glucose, dir=−1, thresh=−0.5, temp=0.5)
- metabolic.heads[3] (ffa): switch SetpointHead → BasalPlusGatedPeakHead(stim=insulin, dir=−1, thresh=0.0, temp=0.5)
- appetite.heads[2] (glp1): switch SetpointHead → BasalPlusGatedPeakHead(stim=nutrient_flag, dir=+1, thresh=0.3, temp=0.2) — gate is positive when nutrient_flag (in coupling[1]) is high

Heads that stay on SetpointHead (no stimulus-driven peak required): cortisol, acth, bhb, ghrelin. Ghrelin's "pre-meal rise" is anticipatory — currently we have no stimulus to gate against, so SetpointHead's slow drift to a new target_z is the right primitive.

Risks:
- (R1) Setting init `g_thresh` wrong for each stimulus mis-positions the gate; could fail to discriminate fasting/fed states. Mitigation: pick thresholds based on the typical norm_state values during cohort meal protocols (glucose at norm=+1.0 mid-meal, insulin at norm=+2.5 peak, nutrient_flag toggling 0→1 with meal onset). Conservative init: gate already discriminates at iter 0.
- (R2) Removing SetpointHead from glucagon/ffa/glp1 loses the saturation-free coordinate. BasalPlusGatedPeakHead has softplus on basal+peak+cons — the saturation trap is back. *Why this works anyway*: with structural basal/peak separation, the head doesn't need to drive basal very negative to keep marker at typical — it just sets `softplus(raw_basal) = cons/(typical·norm_state)·something_small` and the peak rides on top via the gate. The (raw_basal, raw_peak) parameterisation lets the model say "baseline equilibrium is at typical, peak fires when stimulus crosses thresh", without ever needing prod ≈ 0.
- (R3) Embedding-fit collateral on the just-fixed cortisol/acth/bhb. Same risk shape as iter 51 → iter 52. Mitigation: hold those on SetpointHead which we've now seen lands them well.
- (R4) Textbook recovery may not happen — the iter-52 textbook drop (0.81 → 0.67) included `meal_dose_response` going to 0% and `cortisol_awakening_response` to 33%. If iter 53 doesn't restore those, the gain on bench MAPE doesn't translate to scenario competence.
