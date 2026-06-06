# Pulse Iter 17 — Resume Handoff

Self-contained resume point. Read only this file to pick up the work.

## Where we are

**Iter 17 is staged.** Code changes are committed; the Cloud Build
submission is the next action.

- Spec: `apps/pulse/train/spec.json`.
- Cloud Build config: `apps/pulse/train/cloudbuild.yaml`.
- Submission: `bash apps/pulse/scripts/train-submit.sh --gcp` from repo root.
- Job ID assigned at submit time. Artifact root:
  `gs://grovina-pulse/training/jobs/<JOB_ID>/`.

## What iter 16 did and what it revealed

Iter 16 factorized the gut kernel —
`appearance[..., j] = Σᵢ macros[..., i] · K_{i,j}(t, embedding)` — to make
zero-at-zero, dose-linearity, and dose-monotonicity *analytic* rather
than learned. All three structural properties landed perfectly:

| property              | iter 15        | **iter 16**          |
| --------------------- | -------------- | -------------------- |
| `monotone`            | False (1 wobble) | **True** (first time ever) |
| `AUC(0g)`             | 40.5           | **2.4** (≈0; small ODE-integration drift) |
| per-gram slope        | varied 4-6 with saturation | **constant 4.24 across full 0-120g** |
| `tpeak` consistency   | broken         | uniform across all doses |

But the kernel converged to **uniform 1.5× over-amplitude** vs the cold
target it was supposed to track:

| dose | cold target | iter 16 | ratio |
| ---- | ----------- | ------- | ----- |
| 15   | 42.3        | 65.7    | 1.55× |
| 30   | 84.7        | 128.9   | 1.52× |
| 45   | 127.0       | 192.2   | 1.51× |
| 60   | 169.3       | 255.4   | 1.51× |
| 90   | 254.0       | 381.9   | 1.50× |
| 120  | 338.6       | 508.4   | 1.50× |

Per-gram slope: cold = 2.82 mg-min/g, kernel = **4.24 mg-min/g**.

This caused the metrics that mattered to regress:

| metric            | iter 15 | **iter 16** | direction |
| ----------------- | ------- | ----------- | --------- |
| glucose_mape      | 0.5220  | **0.5653**  | ✗ +0.04   |
| fasting_drift     | +53     | **+76**     | ✗ +23     |
| overall_mape      | 0.1952  | 0.1956      | flat      |
| verifier_overall  | 0.6742  | **0.7897**  | ✓ +0.12   |
| verifier[coupling]| —       | **+0.103 vs iter15** | ✓ |
| meal_flow textbook | —      | **+0.111 vs iter15** | ✓ |

## Diagnosis: compensatory equilibrium

`pulse.diagnostics signal-balance` on the iter 16 checkpoint:

| signal              | iter 15 ‖∇gut‖ | **iter 16 ‖∇gut‖** | direction |
| ------------------- | -------------- | -------------------- | --------- |
| gut_dose_sweep      | 0.72           | **0.38** (raw_loss=0.07) | half |
| cohort_statistic    | 3.24           | **0.34**             | one-tenth |
| trajectory_rollout  | 4.14           | **0.92**             | one-quarter |

Two things became clear:

1. The factorized form satisfies `cohort_statistic` and
   `trajectory_rollout` so efficiently that they **stopped pulling the
   gut around at all** (gradients dropped 4-10×).
2. `gut_dose_sweep` MSE was the only remaining direct pull on
   amplitude, but at weight 0.10 it was too soft to drag the kernel
   from 1.5× cold-target down to 1.0×. The kernel found a stable
   force-balance point at 1.5× via shared parameters with the rest of
   the model.

**Hypothesis:** The 1.5× over-amplitude is compensating for
under-tuned downstream modules (insulin response / glucose clearance
/ hepatic glucose output). With the iter-15-and-earlier saturation
removed, the kernel now has the *capacity* to over-deliver, and
trajectory MSE is happy because integrated state matches the cold
trajectory in the over-strong-gut + under-tuned-rest equilibrium.
Counter-regulation deltas growing for the first time
(glucagon Δ = -3.34 vs iter 15's +0.009) supports this — other modules
are starting to engage but at sub-physiological magnitudes.

## What iter 17 changes

**One spec change:** `--gut-dose-sweep-weight=0.10 → 0.50` (5×).

Rationale: with the factorized kernel, MSE pulls smoothly with no
architectural barriers. We just need more weight to make `gut_dose_sweep`
the dominant signal on gut amplitude. A 5× bump should:

1. Force the kernel to track cold-target amplitude (slope ~2.82 vs 4.24).
2. Trigger `trajectory_rollout` MSE to rise, because the under-tuned
   downstream modules can no longer rely on over-strong gut input.
3. Force SGD to start fixing those downstream modules instead of
   leaning on gut compensation.

All other hyperparams unchanged from iter 16. `gut-loss-weight` stays
at 0.10 (the trajectory-window gut MSE; secondary signal). Ranking
term stays in place at `ranking_weight=1.0` as defense-in-depth (the
factorized form should keep its loss component near zero anyway).

**One CI change:** `pulse.diagnostics.compare_runs` now gracefully
degrades when checkpoint loading fails. Iter 16's `CompareToPrevious`
crashed because the iter 15 baseline checkpoint had the old gut-shape
parameters (`gut.kernel.kernel.0.weight: [32, 12]`) and the iter 16
image's model expected the new shape (`[32, 9]`). Architecture changes
between iters now log a warning and skip that side's probe, rather
than aborting the whole comparison. The benchmark/gate sections from
the JSON reports still render — they don't depend on being able to
instantiate the model.

## Code changes (commit-level)

1. **`apps/pulse/train/spec.json`** — `gut-dose-sweep-weight` 0.10 →
   0.50; hypothesis & expectedEffect rewritten.
2. **`apps/pulse/engine/pulse/diagnostics/compare.py`** — new
   `_try_probe()` helper wraps `load_model_from_checkpoint` +
   `probe_checkpoint` in try/except; `compare_runs` uses it for both
   sides. RuntimeError / FileNotFoundError / KeyError are caught and
   the side's probe is skipped with a stderr warning.
3. **`apps/pulse/docs/iter17-handoff.md`** — this file.

## Local sanity (CI fix verified)

Cross-architecture compare command that crashed in CI for iter 16:

```bash
.venv/bin/python -m pulse.diagnostics compare \
  --new-label iter16  --new-checkpoint /tmp/iter16/checkpoint.pt \
  --new-report /tmp/iter16/benchmark-report.json \
  --baseline-label iter15  --baseline-checkpoint /tmp/iter15.pt \
  --baseline-report /tmp/iter15-benchmark.json
```

Now exits cleanly with:
```
[compare] baseline probe skipped: cannot load /tmp/iter15.pt:
  RuntimeError: Error(s) in loading state_dict for ModularPhysiologyNetwork:
        size mismatch for gut.kernel.kernel.0.weight: copying [32, 12] vs [32, 9]
        ...
=== Comparison: iter16 vs iter15 ===
--- Gate ---
  baseline=FAIL  new=FAIL
  ...
--- Scalars (Δ = new − baseline; negative is better for MAPE) ---
  overall_weighted_mape Δ : +0.0003
  verifier.overall_score Δ: +0.0200
  ...
```

The "Probes" section is now silently skipped (both probes need
loadable checkpoints to render Δ); everything else renders normally.

## Success criteria for iter 17

Hard gates:

1. **Per-gram slope ≤ 3.5 mg-min/g** (iter 16: 4.24; cold target: 2.82).
   Probe via `(AUC(120g) − AUC(0g)) / 120`.
2. **AUC(120g) ≤ 400** (iter 16: 508).
3. **glucose_mape ≤ 0.45** (iter 16: 0.57; this is a recovery
   milestone — not the eventual ≤0.30 target, just a clear improvement).
4. **CompareToPrevious produces `delta-vs-baseline.json`** (iter 16
   crashed; iter 17 should succeed because both checkpoints have the
   factorized architecture).
5. **Structural guarantees hold:** monotone=True, AUC(0g)<5.

Soft gates:

- `fasting_drift` ≤ +50 mg/dL (iter 16: +76).
- `verifier_overall` ≥ 0.75 (iter 16: 0.79; should hold).
- `gut_dose_sweep` becomes the dominant ‖∇gut‖ in signal-balance.
- `trajectory_rollout` raw_loss visibly higher than iter 16 (this is
  the *expected* side effect of stripping gut compensation).

## What to look at if iter 17 fails

| symptom | likely cause | next move |
| ------- | ------------ | --------- |
| Slope drops but glucose_mape doesn't recover | downstream modules genuinely under-capacity | iter 18: direct insulin/clearance probe-style supervision |
| Trajectory loss skyrockets and never recovers | weight bump too aggressive; equilibrium broken faster than SGD can heal | iter 18: ramp weight 0.10→0.50 across phase 2 instead of jumping |
| Slope unchanged at 4.24 | gut_dose_sweep MSE is being countered by something else | check signal-balance ‖∇gut‖ from each signal; identify the counter-pull |
| New AUC(0g) regression | structural property somehow broke during training | inspect `pulse/modules/base.py` for accidental changes; add stricter test |

## Diagnostic commands

```bash
# Pull iter 17 artifacts
mkdir -p /tmp/iter17 && cd /tmp/iter17 && \
  gsutil cp gs://grovina-pulse/training/jobs/<JOB_ID>/{benchmark-report.json,delta-vs-baseline.json,checkpoint.pt} .

# Probe the trained kernel (slope, AUC, monotonicity, drift, counter-reg)
cd apps/pulse/engine && .venv/bin/python -m pulse.diagnostics probe \
  --checkpoint /tmp/iter17/checkpoint.pt

# Signal-balance (verify gut_dose_sweep dominates ‖∇gut‖)
.venv/bin/python -m pulse.diagnostics signal-balance \
  --checkpoint /tmp/iter17/checkpoint.pt

# Cross-iter compare (both have factorized arch; should produce probe Δ)
.venv/bin/python -m pulse.diagnostics compare \
  --new-label iter17  --new-checkpoint /tmp/iter17/checkpoint.pt \
  --new-report /tmp/iter17/benchmark-report.json \
  --baseline-label iter16  --baseline-checkpoint /tmp/iter16/checkpoint.pt \
  --baseline-report /tmp/iter16/benchmark-report.json

# Cloud Build status / logs
gcloud builds list --project=grovina --limit=5
gcloud builds log <BUILD_ID> --project=grovina
```
