# Pulse — Bayesian calibration + real CGM data

Two related investments after iter 38's gate clearance, queued behind
iter 39 (marker vitality):

1. **Laplace-approximation Bayesian calibration**: replace point-estimate
   embedding `ε̂` with a posterior `p(ε | obs)`. Production users get
   uncertainty bands "for free"; we get a built-in calibration-quality
   metric ("how well do we know this user?").

2. **Real CGM data ingestion** (`apps/caravel/tester/data/gabriel/GabrielRovina_glucose_11-7-2025.csv`):
   8018 readings, 28 days, FreeStyle Libre 3, glucose only. No labelled
   meals/insulin. Becomes a *validation* corpus for the Bayesian
   posteriors before we ever consider using it for training.

Both can be built and shipped in parallel with the iter-39 train loop.

## Bayesian via Laplace approximation — design

Currently `calibrate_embedding` runs Adam for 512 steps from `ε=0`,
returns a point. The Bayesian extension keeps Adam (gives the MAP) and
adds two steps:

1. **Hessian at MAP**. Evaluate `H = ∂²L/∂ε∂ε^T` at `ε̂` where
   `L(ε) = − log p(ε|obs) = MSE_loss(ε) + ‖ε‖²/(2 σ_prior²)`. PyTorch's
   `torch.autograd.functional.hessian` works directly because the loss
   is already differentiable and the embedding is 32-dim — Hessian is
   32×32, trivially small.

2. **Posterior covariance**. Σ = (H + λI)⁻¹ where λ is a small numerical
   regularizer to handle near-singular H (happens when some embedding
   dimensions are unconstrained by the observations — exactly the
   "what we don't know" we want to expose). Cholesky factor `L` so we
   can sample efficiently.

Predictive distribution at any time `t`:

```
ε_k ~ N(ε̂, Σ),  k = 1..K
y_k(t) = integrate(model, ε_k)
predictive ≈ {mean: y_k.mean(0), std: y_k.std(0),
              p05/p50/p95: quantiles}
```

K = 20 samples gives reasonable predictive mean / 95% PI. K = 100 if
we want tail percentiles. The cost per sample is one full integration
— same as running bench once. So Bayesian calibration costs roughly
`(MAP cost) + (K × bench inference cost)`, which is bounded.

### Calibration-quality metrics

- **Posterior entropy** `½ log det(Σ) + const` — single scalar, "how
  much do we know about this user?" Higher = less constrained.
- **Per-marker predictive std** at the calibration midpoint — pinpoints
  *which* markers we're uncertain about, not just average uncertainty.
- **Effective dimensionality** `tr(I − ε̂_var × prior_var⁻¹)` — how
  many degrees of the embedding the observations actually constrain.

### Held-out coverage as a calibration check

The killer feature with real CGM data: take a 12h slice, calibrate on
the first 8h, predict the last 4h, compute the fraction of held-out
CGM readings inside the 95% predictive interval. If that fraction is
~95%, our posterior is well-calibrated. If it's 70% we're
overconfident; if it's 99% we're underconfident.

This is a much sharper signal than MAPE: MAPE says "we're off by 12
bpm on average"; coverage says "we *know* we're within ±X bpm 95% of
the time." The latter is what production users need to act on.

## Real CGM data — what it lets us do

8018 5-min readings across 28 effective days, glucose only. With no
labelled meals/insulin, we can't directly run the bench pipeline.
What we *can* do:

1. **Fasting-period validation**. Find continuous 4–8h segments where
   glucose stays in a narrow range (no meal-shaped spikes). The model's
   `fasting_stability` and `default_baseline` (HR) signals predict
   fasting dynamics directly; this tests them on real data.

2. **Inferred-meal validation**. Detect spikes in the CGM trace,
   infer approximate meal events (timing + crude carb load from peak
   amplitude), feed those as model inputs, predict the rest of the
   trajectory. Lossy but honest: tells us whether the model's
   meal-driven dynamics match real meal-driven dynamics on this user.

3. **Posterior calibration check** (per above). The Bayesian posterior
   gives 95% PIs; real CGM is the gold-standard hold-out.

We deliberately don't put this into training yet. The model is trained
against synthetic + literature; mixing real CGM data into training
introduces user-specific overfitting risk that we'd then have to
disentangle from any signal we measure. Validation first.

## Plan

**Workstream A — Laplace** (~150 LOC, no retrain):
1. `bayesian_calibrate(model, obs, ...)` in `benchmark.py` —
   returns `(eps_map, posterior_chol, posterior_samples)`.
2. `predictive_distribution(model, samples, ...)` — runs K
   integrations, returns mean/std/quantiles per marker per time.
3. `apps/pulse/engine/scripts/bayesian_demo.py` — runs Laplace on
   iter-38 checkpoint over a bench episode, prints predictive
   intervals around eval points and the in-95%-PI coverage rate.

**Workstream B — CGM ingestion** (~200 LOC, no retrain):
1. `apps/pulse/engine/scripts/ingest_cgm.py` — reads the CSV,
   normalizes timestamps + units (mmol/L → mg/dL), splits into
   continuous segments, writes a `cgm-validation-episodes.json` in
   the same shape as `benchmark.dataset.generated.json` (so all
   downstream tooling works unchanged).
2. `apps/pulse/engine/scripts/cgm_validation.py` — runs Bayesian
   calibration on each CGM episode, computes per-episode coverage
   metrics.

A and B compose: A gives us the tool, B gives us the data to point
the tool at. Together they tell us — for the first time — whether
the model trained on synthetic + literature *actually* captures real
glucose dynamics on a real user.

## Follow-ups (queued behind Bayesian demo working)

- **Exercise / activity ingestion.** `google_fit.zip` also contains
  `activity.segment_*`, `step_count.delta_*`, `distance.delta_*`,
  `calories.expended_*`, `heart_minutes_*` — collectively the workout
  database. The model has an ``activity`` external input alongside
  ``sleep_wake`` (currently ``None`` in our episodes). Wiring it up
  would let us:
  1. Populate per-minute activity intensity in real-data episodes.
  2. Build dedicated workout episodes (very different HR regime than
     resting/overnight — much bigger amplitude, different time
     constants).
  3. Replace the synthetic textbook `exercise_bout` check with a real
     `hr_rises_during_workout` validation against Gabriel's logged
     exercise sessions.

  Worth doing once the basic resting/sleep-period validation works.

## HR drift across all 14 episodes (iter 38, real data)

`/tmp/pulse-diag/check_hr_drift.py` (ephemeral) ran Bayesian
calibration + 8 posterior samples on every overnight episode and
fit a slope through observed vs predicted HR points:

```
ep  sleep%  hr_init_pred  hr_init_obs  drift_obs(bpm/h)  drift_pred(bpm/h)
 0    66%        58.8         58.8           +0.57            +3.46
 1    66%        69.6         69.6           +3.75            +5.68
 2    71%        78.9         78.9           +1.03            +2.39
 3    48%        69.8         69.8           +4.06            +7.27
 4    52%        61.4         61.4           +1.81            +4.15
 5    52%        63.3         63.3           -1.27            +2.35
 6    50%        70.6         70.6           -3.25            -0.18
 7    58%        76.8         76.8           +0.33            +3.73
 8    51%        57.5         57.5           -1.37            +0.69
 9    68%        63.9         63.9           +6.76            +2.74
10    64%        59.8         59.8           -0.05            +2.37
11    67%        67.2         67.2           +1.79            +2.69
12    70%        70.1         70.1           +4.11            +3.03
13    62%        64.0         64.0           +4.27            +2.90

mean(drift_obs)  = +1.55 bpm/h
mean(drift_pred) = +2.99 bpm/h
mean(over-prediction) = +1.44 bpm/h
```

In 9 of 14 episodes the model over-predicts drift by 0.9–3.6 bpm/h.
Real HR is more variable than the model's monotonic creep upward
(observed range -3 to +7 bpm/h vs predicted -0.2 to +7.3).

This motivated iter 40's SLEEP_HR_DIP cohort weight bump 1.0 → 3.0.

## Real-data validation — first results on iter 38

Ran `bayesian_demo.py` on episode 0 of Gabriel's real CGM + Oura HR
overnight episodes (12h, 19:00 → 07:00, 66% asleep). σ_prior = 0.1
(matches training-time embedding init), aleatoric σ_obs from real
device specs (CGM = 8 mg/dL, Oura HR = 3 bpm).

```
glucose   100% in 95% PI  mean |error| = 5.79 mg/dL  ✓
hr         43% in 95% PI  mean |error| = 7.71 bpm    ✗
```

**Glucose**: well-calibrated. Mean prediction ~100 mg/dL matches actual
95-106. PIs capture all 8 held-out points. Mean error ≈ CGM measurement
noise. Iter 38's overnight glucose dynamics are right.

**HR**: predictions drift upward over the 12h. At minute 504 the model
predicts 67 bpm (actual 64 — close). By minute 669 it predicts 77 bpm
while actual stays at 65. **Drift rate ≈ +0.05 bpm/min ≈ +3 bpm/hour
during overnight sleep.** PIs are the right *width* (~4 bpm) — the
*mean* is moving away from truth.

This is a real flaw in iter 38 visible only on real overnight data.
The bench gate cleared HR at 0.135 because most bench eval points are
scattered through the day at a single time per episode, masking
trajectory-shape errors. Real continuous data exposes the drift.

The default-baseline signal pulls the [60-240] window mean toward
NORM_CENTER=70 but doesn't constrain drift in either direction; the
dynamics creep upward over hours, especially during sleep.

## Implications for iter 40+

Two distinct findings to act on:

1. **Iter 39's MarkerVitalitySignal at weight 0.10 broke HR**
   (hr_mape 0.135 → 0.218). Roll back to iter 38 spec. The
   counter-regulatory revival problem (glucagon/FFA/ghrelin still
   flat after iter 39) needs a different signal shape — not a range
   floor but a per-spec cohort weight bump on the existing
   suppression specs.

2. **Iter 38's HR has a sleep-time drift problem.** The `default_baseline`
   regularizer needs companions:
   - A drift penalty (penalize d(HR)/dt during sleep windows).
   - Or an endpoint constraint (HR at t=duration should still be near
     NORM_CENTER, not just the [60-240] window mean).
   - Or a sleep-mask-aware target (real HR *decreases* during sleep —
     the model should too, not stay flat or drift up).

   Option (c) is the most physiologically informed: pull HR DOWN during
   sleep (e.g., target 5 bpm below daytime mean for 22:00-07:00
   windows). Aligns with the existing `sleep_hr_dip` cohort spec
   (Somers 1993) which already exists but at insufficient weight.

## Iter 40 result (job `train-20260508T092719Z`)

Hypothesis: SLEEP_HR_DIP cohort weight 1.0→3.0 should reduce the
+1.48 bpm/h overnight HR over-prediction by tightening the sleep-arm
contrast. Outcome:

```
                          iter 38   iter 40    Δ
overall_weighted_mape     0.095     0.101     +0.006   (regression)
hr_mape (bench)           0.135     0.158     +0.023   (gate fail, > 0.15)
glucose_mape (bench)      0.203     0.213     +0.010   (gate fail, > 0.20)
verifier[meal]            0.581     0.581      0.00    (unchanged)
real-data over-pred       +1.48     +1.53     +0.05    (no movement)
```

Both axes failed. The cohort spec is **contrast-shaped**, not
drift-shaped: SLEEP_HR_DIP penalizes (sleep-arm mean − awake-arm
mean), so the model can satisfy the contrast by lowering the
sleep-window mean while still drifting upward inside the window.
The drift is a *trajectory* phenomenon; population summary
statistics never see it.

Per-episode delta is tiny (Δ over-pred between ±0.27 bpm/h, mean
+0.05, smaller than inter-episode noise) — confirming the cohort
gradient never reached the drift dynamics in any meaningful way.

## Iter 41: A + β bundled

A re-read of the post-iter-38 brainstorm surfaced that **item A** ("expand
the bench to cover unobserved markers, ~1 day, highest value-per-effort")
was never actually attacked. Iter 39 attempted **item B** (response-amplitude
signal — `MarkerVitalitySignal`) and was reverted as a regression on HR;
iter 40 tried a cohort-weight bump on SLEEP_HR_DIP and failed both axes.
Both verdicts were *partially blind* because the bench gate only scores
5 measured markers (glucose, hr, sbp, dbp, temp), so we couldn't measure
whether iter 39's vitality signal moved glucagon/FFA/ghrelin — the actual
target. Item A removes that blind spot; β provides a properly-shaped
revival mechanism.

### A — bench expansion for unobserved markers

`apps/pulse/engine/pulse/knowledge/benchmark_extras.py`:

- Existing `cohort_sleep_48h_benchmark_episodes` adds 99 eval samples
  across 9 dead-pathway markers (insulin, glucagon, ffa, bhb, ghrelin,
  leptin, glp1, cortisol, acth) at 11 timestamps spanning the protocol.
- New `cohort_meal_postprandial_benchmark_episodes` (8h OGTT-style)
  adds 100 more samples — 10 timestamps × 10 markers. Postprandial
  dynamics force insulin / glucagon / glp1 into observable regimes.
- `all_cohort_benchmark_episodes` aggregator wired into `train.py`.

`apps/pulse/engine/pulse/benchmark.py`:

- `evaluate_model_against_benchmark` now restricts `overall_weighted_mape`
  to gate-thresholded markers (glucose, hr, sbp, dbp, temp). The new
  markers appear in `per_marker[]` with `in_gate=False` and don't inflate
  the gate's overall metric — including them would change the gate's
  meaning since the 0.16 threshold was sized for the 5 measured markers.

Diagnostic output only — no per-marker thresholds added. The first
expanded-bench numbers from iter 38 (and iter 41) become the baseline
that iter 42+ thresholds can be sized against.

### β — re-add counter-regulatory NADIR landmarks

`apps/pulse/engine/pulse/landmarks.py`:

- `DEFAULT_LANDMARK_SPECS` now `(HR PEAK, glucagon NADIR, ffa NADIR,
  ghrelin NADIR)` — re-instates the 3 specs that iter 34 narrowed
  away. Glucose PEAK and insulin PEAK stay excluded (those were the
  iter-33 regressors; trajectory MSE + insulin-sweep already cover
  them).

`apps/pulse/train/spec.json`:

- `--landmark-weight=0.20 → 0.40` so per-spec average lands at 0.10
  across the 4 specs. HR's effective weight halves (0.20 → 0.10),
  mitigated by DefaultBaselineSignal at 0.30 being the dominant HR
  anchor since iter 38.

### Iter 38 baseline on the expanded bench (Cloud Build `f1b9a780`)

Re-ran iter 38 against the expanded bench (cohort eval points across 9
dead-pathway markers). Per-marker baseline numbers iter 41 has to beat:

| marker     | mean MAPE | median | n  | notes                                     |
|------------|-----------|--------|----|-------------------------------------------|
| insulin    | **1.08**  | 0.50   | 21 | predictions ~2× actual                    |
| ghrelin    | **1.93**  | 1.73   | 21 | DEAD pathway (iter-39 probe confirmed)    |
| glucagon   | 0.44      | 0.44   | 21 | β target — NADIR landmark                 |
| ffa        | 0.43      | 0.41   | 21 | β target — NADIR landmark                 |
| cortisol   | 0.42      | 0.34   | 21 | not in β spec set — iter 42+              |
| acth       | 0.41      | 0.36   | 21 | not in β spec set — iter 42+              |
| glp1       | 0.29      | 0.29   | 21 | suboptimal but moves                      |
| bhb        | 0.19      | 0.21   | 21 | OK-ish                                    |
| leptin     | 0.03      | 0.03   | 21 | already excellent (slow dynamics)         |
| glucose ⓖ  | 0.20      | 0.13   | 181| unchanged from prior bench                |
| hr ⓖ       | 0.13      | 0.15   | 180| unchanged                                 |
| sbp ⓖ      | 0.07      | 0.07   | 169| unchanged                                 |
| dbp ⓖ      | 0.05      | 0.04   | 168| unchanged                                 |
| temp ⓖ     | 0.02      | 0.02   | 169| unchanged                                 |

ⓖ = in gate (counted toward overall_weighted_mape = 0.095).

The 4 gate markers' MAPEs match the original iter 38 in-job bench
(0.203 / 0.135 / 0.067 / 0.051 / 0.015) within run-to-run noise,
confirming the bench expansion is a pure-add: no regression on
existing measured markers.

### What β tests that iter 39 couldn't

Landmarks penalize `(predicted_residual_at_nadir / scale)²` directly —
trajectory-shape supervision, not a range floor. The model can't satisfy
it by keeping the marker flat (a flat trajectory has nadir = baseline,
residual = full target Δ). The iter-39 vitality range floor failed
because `relu(target_range − actual_range)²` is satisfied by *any* small
oscillation, including representation-collapse-shaped noise that doesn't
respect physiology timing.

Iter 41's success criterion: the per-marker MAPE on glucagon, ffa,
ghrelin at the new bench eval points drops dramatically vs iter 38's
expanded-bench baseline. Anything in 0.10–0.30 means the markers
actually move with meals; staying near 1.0 means revival failed
(consistent with iter 39's MarkerVitalitySignal).

### Deferred for iter 42+

- **Sleep-window drift penalty** for the +1.5 bpm/h overnight HR
  over-prediction on real data. The β trade-off (HR landmark weight
  halving) might already worsen this; we'd want to land β first and
  measure. The SleepWindowDriftSignal idea (~30 LOC, supervises
  trajectory shape over a zero-embedding sleep mask) is the natural
  next step.
- **SLEEP_HR_DIP cohort weight** stays at 3.0 from iter 40 since
  reverting it is independent of β. If iter 41 shows it's still
  inert, drop to 1.0 in iter 42.
- **Promoting unobserved markers into the gate** — once iter 41
  produces real per-marker MAPE numbers, set conservative thresholds
  in `benchmark.thresholds.json` so gate failures actually fire on
  dead markers.

## State at end of session (2026-05-08)

- **Iter 38** is bench-gate champion: overall 0.095, hr 0.135 ✓.
- **Iter 39** (MarkerVitalitySignal at 0.10) was a regression on
  bench, reverted in iter 40.
- **Iter 40** (commit `0958859a`) failed: bench `hr_mape 0.135→0.158`
  (gate fail), real-data drift `+1.48→+1.53` (no movement). See
  the iter-41 plan above.
- **Iter 41**: A (bench expansion) + β (counter-regulatory NADIR
  landmarks). Item A's benchmark-rerun on iter 38 dispatched as
  Cloud Build to surface dead-pathway baseline numbers. Iter 41
  training run dispatched against `origin/main` (this commit).
- **Bayesian validation pipeline** shipped (commits `36044462`,
  `8ad68ed6`, `fd6e7c5a`). Reusable: `bayesian_calibrate` +
  `predictive_distribution` in `benchmark.py`, demo in
  `apps/pulse/engine/scripts/bayesian_demo.py`, real-data
  ingestion in `apps/pulse/engine/scripts/ingest_real_data.py`.
- **HR-drift validation script** at `apps/pulse/engine/scripts/check_hr_drift.py`
  — runs Bayesian calibration + drift slopes across 14 overnight
  episodes, compares against a saved baseline JSON. Iter-38 baseline
  at `/tmp/pulse-diag/hr_drift_iter38.json`, iter-40 at
  `/tmp/pulse-diag/hr_drift_iter40.json`.
- **Real-data corpus** at `/tmp/pulse-diag/gabriel-real-episodes.json`
  (ephemeral) — 14 overnight 12h episodes from CGM + Oura HR + sleep.
  Re-generate with `uv run python scripts/ingest_real_data.py
  --output <path>`.

To resume a fresh context: read this doc + iter36-calibration-investigation.md
+ iter39-unobserved-markers-investigation.md, then act on the iter-41
plan above.

## Open questions

- **Prior σ²**: currently calibration uses `l2_weight=0.001` which is
  effectively a wide prior. For Bayesian posterior to be meaningful
  we want a *real* prior. Two options: (a) keep wide (`σ_prior ≈ 1`)
  → the data dominates and posterior is mostly likelihood-shape,
  (b) learn `σ_prior` from training-time embedding distribution
  (sample mean & std of the 20 trained embeddings) → tighter prior,
  more meaningful when observations are sparse. Start with (a) and
  switch to (b) if the wide-prior posteriors are too diffuse.

- **Hessian conditioning**: with sparse observations and a 32-dim
  embedding, H may be near-rank-deficient (some embedding dims
  unconstrained). Symptom: huge posterior uncertainty in those dims,
  predictive intervals blow up. Two fixes: (a) numerical regularizer
  `H + λI`, (b) project posterior to the constrained subspace via
  pseudoinverse. Try (a) first; revisit if predictive intervals are
  unreasonable.

- **Production deployment**: each Bayesian calibration is K extra
  integrations (~K × 720 = K × 720 forward steps for a 12h episode).
  For K=20 that's an extra ~14 seconds per user on CPU. Acceptable
  for the bench pipeline; for a per-user-API call we'd cache the
  posterior and re-sample lazily.
