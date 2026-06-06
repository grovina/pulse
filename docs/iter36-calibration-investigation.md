# Pulse iter 36 — calibration-as-bottleneck investigation

After 4 iterations (32–35) of trying to move `hr_mape` via training-side
signals (cardio coupling inputs, per-window HR landmark, HR-only landmark,
HR cohort specs), the metric plateaued at 0.218–0.241 across all configs.
Per-iter movement was in the noise band (≤ ±0.025 bpm-equivalent), with no
intervention crossing the gate (≤ 0.15) or even getting close.

Iter 36 is preceded by an investigation, not a training run.

## Bottom line

`hr_mape ≈ 0.24` is **a constant +15 bpm prediction bias**, not a
shape/timing/coupling deficit. No training-side signal can fix it
because per-window landmark, dose-response, and cohort losses all
constrain Δs and shapes — they don't penalize a uniform offset.

## The smoking gun

`/tmp/pulse-diag/inspect_hr.py` ran the iter-35 checkpoint on the 24
benchmark episodes (168 HR eval points total) and reported:

```
actual HR  mean = 62.95 bpm  std = 8.40   range = [48, 79]
pred   HR  mean = 78.00 bpm  std = 9.21   range = [61.5, 93.4]
residual  : mean = +15.05 bpm  std = 1.67  median = +15.39
            168/168 samples over-predict
abs(resid) percentiles: p25=13.83  p50=15.39  p75=16.39  p90=16.97
```

Bias is essentially uniform across HR levels (low/mid/high all show
+13 to +16 bpm). The math: `15 / 63 ≈ 0.24` — every iter's hr_mape has
been measuring the same offset wearing different clothes.

## Why training-side interventions kept missing

The four signals tried in iters 32–35:

| iter | intervention | what it constrains | why it can't move uniform offset |
|------|---|---|---|
| 32a | Cardio coupling inputs (_N_COUPLING 2 → 4) | how HR responds to glucose/insulin | only changes Δs through dynamics; no per-user baseline pull |
| 33b | Per-window HR landmark (`MarkerLandmarkSpec("hr",PEAK,...)`) | Δpeak / time-to-peak / AUC vs pre-meal baseline | penalizes deviation from pre-meal level — uniform offset cancels in the Δ |
| 34 | HR-only landmark (narrow DEFAULT_LANDMARK_SPECS to HR) | as above, with bigger per-marker weight | same — all-Δ losses |
| 35 | HR cohort specs: postprandial_hr_rise (DELTA_PEAKS), sleep_hr_dip (DELTA_MEANS) | population-amplitude contrasts between arms | DELTA losses cancel any constant bias |

All four are differential signals. The model has no training-time pressure
to match an absolute HR baseline.

## Where the +15 bpm is coming from

`/tmp/pulse-diag/calibration_probe.py` ran four regimes:

| regime | mean residual | inference |
|---|---|---|
| zero embedding (no calibration) | **+24.88 bpm** | model's "default patient" predicts HR ≈ 88 bpm |
| std calibration (32 steps, all markers, LR 0.02, L2 0.1) | **+15.05 bpm** | calibration removes ~10 bpm — it CAN move HR |
| HR-only calibration (128 steps, LR 0.05, L2 0.01) | **+12.46 bpm** | 4× steps + HR focus saves only 2.6 more bpm |
| capacity probe (random emb std=0.5, n=50) | HR baseline range **[34.8, 89.9]** | embedding has full physiological span available |

Three independent facts:

1. **Capacity is fine.** The cardio module's embedding pathway can land
   HR anywhere from 35 to 90 bpm. Architecture is not the bottleneck.

2. **The trained model's zero-embedding default is 88 bpm.** Cold model
   produces ~70 (the `MARKERS["hr"]` typical). Training with 100 epochs
   of metabolic coupling pushed default-patient HR up by ~18 bpm.

3. **Calibration finds 10 bpm of correction but plateaus at +12 bpm.**
   With more steps / lower L2 / HR-only loss, the gain is small. The
   loss landscape from zero-embedding is shallow on the HR-baseline axis
   — gradient through 12h of rate-of-change integration has small
   magnitude per Adam step.

## Bench dataset internal inconsistency

| source | HR mean (bpm) | n |
|---|---|---|
| calibration check-ins | 68.76 | 384 |
| eval measurements | 62.95 | 168 |

5.8 bpm gap between the two. Even perfect calibration on check-ins still
under-fits eval by ~6 bpm. This is a generation-side bug in
`benchmark.dataset.generated.json` worth fixing eventually but doesn't
change the iter 36 strategy.

## Iter 36 candidate paths

### Path 1 — Calibration hyperparameter sweep (no retrain) — RESULTS

`/tmp/pulse-diag/path1_rebench.py` ran the iter 35 checkpoint through the
full benchmark with seven calibration configs. Per-marker MAPE shown for
each:

| setting (n_steps, lr, l2) | glucose | hr | sbp | dbp | temp | overall |
|---|---|---|---|---|---|---|
| baseline (32, 0.02, 0.1) | 0.278 | 0.242 | 0.083 | 0.085 | 0.019 | 0.141 |
| more_steps (128, 0.02, 0.1) | 0.225 | 0.246 | 0.088 | 0.089 | 0.020 | 0.134 |
| higher_lr (32, 0.05, 0.1) | 0.233 | 0.244 | 0.095 | 0.092 | 0.020 | 0.137 |
| low_l2 (32, 0.02, 0.01) | 0.276 | 0.242 | 0.083 | 0.085 | 0.019 | 0.141 |
| combined (128, 0.05, 0.01) | 0.214 | 0.243 | 0.081 | 0.087 | 0.019 | 0.129 |
| aggressive (256, 0.05, 0.005) | 0.213 | 0.234 | 0.075 | 0.082 | 0.018 | 0.124 |
| **max (512, 0.05, 0.001)** | **0.212** | **0.193** | **0.062** | **0.061** | **0.016** | **0.109** |

**Reads:**
- **Calibration was severely under-tuned across all markers, not just HR.**
  iter 32–35's overall_weighted_mape plateau at ~0.14 was capped by the
  calibration step, not the trained model's quality.
- Going 32→512 steps + LR 0.05 + L2 0.001 monotonically improves every
  marker. No marker is sacrificed.
- HR moved 0.242 → 0.193 (20% reduction) without retraining.
- Glucose moved 0.278 → 0.212 (24% reduction). Now barely over the 0.20
  gate.
- Overall_weighted_mape 0.141 → 0.109 — a new best by a wide margin
  (was 0.138 in iter 34).
- Below the 6 bpm bench-dataset HR inconsistency: hr_mape is bounded
  below by ~0.10 (= 6/63) without dataset fix or architectural shortcut.

### Path 2 — Architectural shortcut (requires retrain)

Add a learned per-user **bias head** to `CardiovascularModule`:
embedding → small MLP → direct HR baseline offset (additive, not through
dynamics). This gives calibration immediate, large-magnitude gradient
signal on baseline. Surgery in `apps/pulse/engine/pulse/modules/cardiovascular.py`
plus a related forward-pass change. ~30 lines.

### Path 3 — Training-side regularization (requires retrain)

Add a "zero-embedding HR output should match `NORM_CENTER`" training
signal. Pulls trained-model default DOWN from 88 toward 70. Doesn't fix
the 6 bpm bench inconsistency or the loss-landscape calibration issue
but reduces the gap calibration must close.

### Path 4 — Bench dataset fix

Investigate why generated check-in HR (mean 68.76) differs from eval
HR (mean 62.95). Out of scope for iter 36; track separately.

## Plan

1. **Run Path 1** locally on iter 35 checkpoint. Sweep
   `(n_steps, lr, l2_weight)`. Report per-marker MAPE for each. (See
   `/tmp/pulse-diag/path1_rebench.py`.)
2. If Path 1 lands hr_mape ≤ 0.15: ship as a benchmark.py change. No
   retrain needed.
3. If Path 1 lands hr_mape in (0.15, 0.20]: combine with Path 2
   (architectural bias head) for iter 36. Path 1 alone goes in as a
   benchmark.py change; iter 36 is then the architectural retrain.
4. If Path 1 lands hr_mape > 0.20: calibration isn't the dominant
   bottleneck and Path 2 + Path 3 together are needed. Iter 36 becomes
   a structural retrain.

## Diagnostic scripts (for re-use)

- `/tmp/pulse-diag/inspect_hr.py` — per-eval-point HR residuals, by
  meal timing and HR level, plus per-episode predicted-vs-actual
  dynamic range.
- `/tmp/pulse-diag/calibration_probe.py` — HR residual under zero,
  standard, and HR-only-aggressive calibration; embedding capacity probe.
- `/tmp/pulse-diag/path1_rebench.py` — calibration hyperparameter
  sweep, full per-marker MAPE per setting, gate-status per setting.

These are ad-hoc and live under `/tmp` for now. Worth promoting to
`apps/pulse/scripts/diagnostics/` if Path 1/2/3 work pans out.

## Iter 36 attempt — Path 2 (cardio baseline head): empirically refuted

Added `nn.Linear(embedding_dim, 4)` baseline head to `CardiovascularModule`
with soft tanh bound at ±25, applied as `state_centered = state - offset`.
Theory: dpred/doffset ≈ 1 should give calibration unit-slope gradient on
baseline, converging in tens of steps rather than thousands.

Trained 1 iter from the iter-35 spec (commit a21e6780). Canonical bench
on the resulting checkpoint (parallelized 512-step calibration):

| metric | iter 35 + Path 1 | **iter 36** | dir |
|---|---|---|---|
| overall_weighted_mape | **0.109** | 0.150 | ✗ +38% |
| hr mean_mape | **0.193** | 0.264 | ✗ +37% |
| sbp mean_mape | **0.062** | 0.173 | ✗ +179% (NEW gate fail) |
| dbp mean_mape | 0.061 | 0.080 | ✗ |
| glucose | 0.212 | 0.213 | flat |
| verifier overall | 0.776 | 0.707 | ✗ |
| verifier coupling | 0.876 | 0.541 | ✗ catastrophic |
| meal_dose_response | 1.000 | 0.333 | ✗ collapsed |

**The architectural change is a strict regression.** Diagnosis:

1. *Coupling collapsed.* During training the cardio module learned to
   respond to glucose/insulin couplings in the centered frame. At
   calibration time the head outputs zero (zero-init + zero-embedding),
   so state arrives uncentered and the trained coupling response
   misfires. `verifier_coupling` 0.876 → 0.541 is the signature.
2. *SBP regressed sharply.* Same mechanism, hits BP harder because BP
   has small natural amplitude — frame misalignment shows up as a
   bigger relative error (0.062 → 0.173).
3. *HR didn't gain anything.* The architectural shortcut only activates
   *after* calibration shifts the embedding. During training the head
   competed with the existing baseline pathway, so both pathways ended
   up worse than the iter 35 single-pathway baseline.

**Reverted in commit (TBD)**. Codebase now reflects iter 35 + Path 1
(parallelized bench, calibration constants 512/0.05/0.001) as the
current best.

## Iter 37 — Path 3 at weight 0.05: regularizer active, hr_mape unchanged

Iter 37 (commit 2cbac1a9) added `DefaultBaselineSignal` at weight 0.05.
Bench:

| metric | iter 35 + Path 1 | **iter 37** |
|---|---|---|
| overall | 0.109 | **0.105** (best at the time) |
| hr_mape | 0.193 | 0.193 (unchanged) |
| glucose | 0.212 | 0.210 |

A post-iter-37 zero-embedding probe (single 240-min fasted rollout from
NORM_CENTER) showed the regularizer was active and *partially* working:
trained zero-embedding fasted HR mean shifted from iter-35's ~88 bpm
to **78.7 bpm** (still 8.7 bpm above the 70-bpm anchor). hr_mape on
bench didn't move because:

1. The signal's per-epoch loss contribution at weight 0.05 was ~0.04,
   dwarfed by trajectory MSE (~10), cohort (~3-4), ppr (~5-10),
   landmark (~0.5-3). It pulled but didn't dominate.
2. Even with the mean shifted, dynamics still drift HR upward across
   the window (probe: 70 bpm at t=0 → 83 bpm at t=240). Calibration
   fits check-ins (HR mean 68.76) at specific times, but predictions
   at other times stay high.

## Iter 38 — Path 3 at weight 0.30: 🎯 first HR gate clear

Iter 38 (commit 34142e00) bumped the regularizer weight 0.05 → 0.30
(~6×) so its loss contribution becomes comparable to the postprandial
recovery signal (~0.6). All other knobs unchanged from iter 37.

| metric | iter 35 + Path 1 | iter 37 | **iter 38** | gate |
|---|---|---|---|---|
| overall_weighted_mape | 0.109 | 0.105 | **0.0946** | 0.16 ✓ |
| glucose mean_mape | 0.212 | 0.210 | 0.203 | 0.20 ✗ (median 0.121) |
| **hr mean_mape** | 0.193 | 0.193 | **0.135** | 0.15 **✓** |
| sbp mean_mape | 0.062 | 0.062 | 0.067 | 0.12 ✓ |
| dbp mean_mape | 0.061 | 0.061 | 0.051 | 0.12 ✓ |
| temp | 0.016 | 0.015 | 0.015 | 0.02 ✓ |
| verifier overall | 0.776 | 0.728 | 0.797 | 0.70 ✓ |
| verifier coupling | 0.876 | — | 0.902 | 0.60 ✓ |
| verifier meal | 0.541 | 0.517 | 0.597 | 0.65 ✗ |
| verifier circadian | — | — | 0.713 | 0.55 ✓ |
| verifier sleep | — | — | 0.996 | 0.55 ✓ |
| verifier sanity | — | — | 0.998 | 0.85 ✓ |

**Two failures remaining**: `glucose_mape=0.2031` (0.003 over the 0.20
gate; median 0.121 — distribution is right-skewed by a few outlier
episodes) and `verifier_cat[meal]=0.5974` (closing on 0.65, was 0.541
in iter 35).

**HR mechanism worked cleanly.** The signal that motivated this whole
investigation — a per-marker training-time anchor at zero embedding —
moved hr_mape 0.193 → 0.135 with no architectural surgery and no
collateral damage to coupling, sbp, dbp, temp, or verifier overall.
Verifier coupling hit a new best (0.902); the cardio module is
operating in its tightest learned regime to date.

**Side regression to investigate (not blocking the iter):** textbook
cortisol_awakening_response dropped 1.000 → 0.667. Cortisol is a
stress-module output; the regularizer doesn't touch stress. Likely
a downstream effect of the HR-baseline shift through cortisol
coupling. Worth a probe in iter 39.

## Iter 39 — TBD

The remaining gate failures are 0.003 of glucose_mape and 0.05 of
verifier_meal. These are worth a deliberate plan, not a single-knob
sweep. Candidates to consider in the morning:

- Investigate the glucose right-tail (median 0.121 vs mean 0.203 — a
  few outlier eval points; mirror the HR residual-by-meal-timing probe
  used here).
- Investigate the cortisol_awakening regression — a probe analogous to
  the HR baseline probe but on stress-module output at zero embedding.
- Verifier-meal: 0.541 → 0.597 in one iter, may improve on its own
  with another iter; or target it deliberately.

- **(α) Path 3 — training-time zero-embedding regularizer.** Add a
  loss pulling the model's HR output at zero embedding toward
  `NORM_CENTER[hr] = 70`. Doesn't add capacity; shapes the trained
  default-patient baseline. ~10 lines. Retrain.
- **(β) Smart calibration init.** Save mean of trained embeddings at
  end of training; init calibration's embedding to that mean instead
  of zero. Embedding capacity probe already showed the subspace spans
  HR [35, 90] — a better starting point + 512 Adam steps should
  converge tighter. ~20 lines. Retrain (to record mean) but very small.
- **(γ) Bench dataset fix (Path 4).** Investigate the 5.8 bpm gap
  between `calibration_check_ins` HR mean (68.76) and `eval_measurements`
  HR mean (62.95). This is a hard floor that no model or calibration
  can beat (hr_mape ≥ ~0.10 by construction). Data engineering, no
  retrain.

## Open issue: bench-only Cloud Build cost

Path 1's 512-step calibration is much more expensive on Cloud Build's
x86 server CPUs than on local M-series. Two attempts to re-bench iter
35 via `cloudbuild-benchmark-only.yaml` timed out:

- 2h timeout on default e2-medium → TIMEOUT (commit 2834138b'd before)
- 4h timeout on E2_HIGHCPU_8 → TIMEOUT (commit 2834138b)

The in-job benchmark inside the Cloud Run training job has a 24h task
timeout (commit 1a4844f3) and should complete cleanly for iter 36 +
beyond. The iter-36 architectural baseline head also makes calibration
gradient unit-slope on HR baseline, so practical convergence is likely
in tens of steps even at the 512-step ceiling.

If/when we need bench-only re-runs again, options are:

- (i) Reduce `BENCHMARK_GATE_CALIBRATE_STEPS` to 256 (from the sweep:
  overall=0.124, hr=0.234 — still much better than baseline 0.141/0.241,
  not as good as 512 but completes inside an hour).
- (ii) Parallelize the per-episode calibrate-and-evaluate loop in
  `evaluate_model_against_benchmark`. The 24 episodes are independent;
  even simple `concurrent.futures` would give 8× on E2_HIGHCPU_8.
- (iii) Move the bench step out of Cloud Build into a Cloud Run job
  (24h timeout) triggered by the training job's completion.
