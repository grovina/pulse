# Pulse Iter 27 — Resume Handoff

Self-contained resume point. Iter 27 is a **training signal addition**, not
an architecture change. Same iter-23/24/25/26 architecture, same loss weights
except one new signal: `FastingStabilitySignal` at weight 0.20.

## Why iter 26 failed the benchmark gate

Build `f361b1a1-9bcb-4831-aa0b-c5ed75b106e4` (`train-20260426T105353Z`).
Training **completed without NaN abort** — the iter-26 `_safe_corr_torch`
fix was correct and the stability gate passed. But:

| metric | value | threshold |
|--------|-------|-----------|
| glucose_mape | 0.6078 | 0.2 |
| hr_mape | 0.2409 | 0.15 |
| overall_weighted_mape | 0.2090 | 0.16 |

Diagnostic probe (`new_probe.fasting_drift`):
- `glucose_t0`: 95.0 mg/dL
- `glucose_tend`: 152.3 mg/dL
- `drift_mg_dl`: **+57.3 mg/dL over 4h, no meals, zero embedding**

This is a resting state that is not a fixed point. The neural metabolic
module has no explicit setpoint mechanism — unlike the knowledge model's
`dG = -(Sg+X)*(G-Gb) + Ra` which pulls glucose back to `Gb=95`, the
learned production/consumption heads settled at resting production >
clearance. The cascade:

1. Glucose drifts up at rest → MAPE is high against observed ~95 mg/dL
2. Glucose doesn't respond proportionally to meals → dose-response collapse
   (90g vs 30g carbs: 0.2 mg/dL difference, threshold 5 mg/dL)
3. Insulin doesn't rise appropriately → glucagon/FFA/ghrelin don't suppress
   (all show ≈0 delta when they should suppress postprandially)

## What iter 27 changes

### `FastingStabilitySignal`

New file: `apps/pulse/engine/pulse/training/fasting_stability_signal.py`

One forward pass per epoch: integrate 120 min from `NORM_CENTER` with no
meals at the zero embedding. Loss = mean-squared glucose drift from initial:

```python
drift = ((pred[:, glc_idx] - initial[glc_idx]) / glc_scale).pow(2).mean()
safe_step(w * drift, ctx, signal="fasting_stability", ...)
```

- Weight: `0.20` (spec.json `--fasting-stability-weight=0.20`)
- Enabled from epoch 0 (not phased — glucose homeostasis is a prerequisite)
- Zero embedding only — the benchmark queries this embedding for all textbook
  scenarios; patient embeddings are supervised by trajectory distillation
- Logged as `fst=X.XXXX` in the 5-epoch print headline

### Cloud Build fix

`apps/pulse/train/cloudbuild.yaml` step 7 (`EnforceBenchmarkGate`) used
`python` which is absent from `cloud-sdk:slim`. Fixed to `python3`.

### What iter 27 does NOT change

- No architecture change (same per-species heads + GlucoseGatedInsulinHead)
- No LR change (phase1=0.003, phase2=0.0003)
- No change to insulin/met_coupling (Fix B from iter-26 analysis — deferred
  to preserve the clean A/B signal)
- No trajectory_band change
- All other signal weights unchanged

## Success criteria

**Primary gate:** `new_probe.fasting_drift.drift_mg_dl` < 15 mg/dL after
120-min fasting rollout at zero embedding. (Currently: +57.3 mg/dL)

**Benchmark gate:** `glucose_mape` ≤ 0.2, `hr_mape` ≤ 0.15,
`overall_weighted_mape` ≤ 0.16 → `gate.passed=true`.

**Cascade checks** (expected to follow from fasting drift fix):
- `meal_dose_response`: 90g vs 30g glucose delta ≥ 5 mg/dL
- `glucagon_suppressed_postprandial`: mean glucagon post-meal < fasting mean
- `glucose_returns_near_baseline` in dietary_carbohydrate_meal_flow

**Stability:** No NaN abort (confirmed by iter 26 `_safe_corr_torch` fix).

## Risk paths

| symptom | likely cause | iter 28 move |
|---------|--------------|--------------|
| `fst=` in headline decreases to < 0.02 by epoch 50 AND glucose_mape drops below 0.2 | Fix A worked cleanly | declare glucose fixed; target remaining failures (hr_mape, counter-regulation) |
| `fst=` drops but glucose_mape stays high (> 0.3) | Fasting drift fixed but meal-response has additional gap | Add explicit dose-response supervision or increase landmark weight |
| `fst=` stays high (> 0.5) all 100 epochs | Weight 0.20 insufficient vs production/clearance imbalance | Bump `--fasting-stability-weight=0.50` in iter 28 |
| `fst=` drops but trajectory loss degrades (> 2x iter 26's average) | Stability signal fights trajectory signal on glucose rises during meals | Reduce weight to 0.10 OR add a gate on meal windows (apply fasting signal only when gut_output ≈ 0) |
| Counter-regulation (glucagon/FFA/ghrelin) remains dead even after glucose improves | Confirms coupling gap: insulin not routed to metabolic module | Add insulin to `met_coupling` (model.py:190), bump metabolic coupling_dim by 1 |
| NaN abort fires | New autograd singularity (iter-26 _safe_corr_torch fix eliminated that site) | Follow iter-25 decision tree against new abort dump |

## Reading `fst=` in the training headline

The `fst=` value is the **normalized** glucose drift loss
(units: (mg/dL / 30)² averaged over 120 steps). Rough calibration:

- `fst= > 1.0`: drift ≥ 30 mg/dL — severe, signal active
- `fst= 0.1–0.5`: drift 10–20 mg/dL — improving
- `fst= < 0.04`: drift < 6 mg/dL — near-zero resting deviation, target range

In iter 26 the drift was 57 mg/dL → `fst ≈ (57/30)² × 0.5 ≈ 1.8`
(averaged over a ramping 120-step window, ~1.0 expected loss).
Watch for `fst=` to decrease from epoch 0 onward.

## How to resume after reboot

**State as of 2026-04-27:** Training is complete. `checkpoint.pt` is in GCS.
The benchmark-rerun Cloud Build is in progress (or has completed — check GCS).

```bash
cd /Users/grovina/Projects/grovina/platform/.claude/worktrees/pulse

# 1. Verify worktree state.
git status   # should be clean on worktree-pulse / main

# 2. Check if benchmark-report.json has landed yet:
gsutil ls gs://grovina-pulse/training/jobs/train-20260427T042129Z/

# If benchmark-report.json is missing, rerun:
bash apps/pulse/scripts/benchmark-rerun.sh train-20260427T042129Z

# 3. Once benchmark-report.json exists, fetch and read results:
mkdir -p /tmp/iter27
python3 -c "
from google.cloud import storage
from google.oauth2 import service_account
import os, json
creds = service_account.Credentials.from_service_account_file(os.environ['GOOGLE_APPLICATION_CREDENTIALS'])
client = storage.Client(credentials=creds, project='grovina')
bucket = client.bucket('grovina-pulse')
for fname in ['benchmark-report.json', 'delta-vs-baseline.json']:
    blob = bucket.blob(f'training/jobs/train-20260427T042129Z/{fname}')
    open(f'/tmp/iter27/{fname}', 'wb').write(blob.download_as_bytes())

r = json.load(open('/tmp/iter27/benchmark-report.json'))
print(f'gate.passed={r[\"gate\"][\"passed\"]}')
print(f'overall={r[\"overall_weighted_mape\"]:.4f}')
print(f'glucose_mape={r[\"per_marker\"][\"glucose\"][\"mean_mape\"]:.4f}')
d = json.load(open('/tmp/iter27/delta-vs-baseline.json'))
print(f'fasting_drift_new={d[\"new_probe\"][\"fasting_drift\"][\"drift_mg_dl\"]:.1f} mg/dL')
"
```

## Cloud Build / job IDs

| iter | job ID | build status | outcome |
| ---- | ------ | ------------ | ------- |
| 21 | train-20260423T175447Z | FAILURE (gate) | overall=0.190, glu_mean=0.543 |
| 22 | train-20260424T071900Z | FAILURE (gate) | overall=0.193, glu_mean=0.566 |
| 23 | train-20260424T163240Z | FAILURE (NaN + timeout) | no checkpoint produced |
| 24 | train-20260425T153910Z | FAILURE (NaN-corrupt checkpoint) | textbook=0.108, all metrics NaN |
| 25 | train-20260426T062232Z | FAILURE (strict abort at Phase 2 epoch 61) | abort dump → _safe_corr_torch |
| 26 | train-20260426T105353Z | FAILURE (gate; CB step 7 python→python3 bug) | stability PASSED, glu_mape=0.608, drift=+57 mg/dL |
| 27 | train-20260427T042129Z | FAILURE (Cloud Run 12h timeout during benchmark) | checkpoint.pt saved; benchmark-rerun pending |

## Benchmark rerun (Cloud Run timeout recovery)

Training completed (checkpoint saved) but the Cloud Run task hit its 43200s
limit before the benchmark ran. Rerun just the benchmark:

```bash
bash apps/pulse/scripts/benchmark-rerun.sh train-20260427T042129Z
```

This builds a fresh image (with `--benchmark-only` support) from current
source and runs the benchmark in-process in a Cloud Build step — no Cloud
Run needed, inference needs far less RAM than training.

## Pointer references

- `apps/pulse/engine/pulse/training/fasting_stability_signal.py` — new signal
- `apps/pulse/train/cloudbuild.yaml` — python3 fix in EnforceBenchmarkGate
- `apps/pulse/train/cloudbuild-benchmark-only.yaml` — benchmark rerun config
- `apps/pulse/scripts/benchmark-rerun.sh` — submit a benchmark-only build
- `apps/pulse/docs/iter26-handoff.md` — diagnostic dump + fasting drift context
- `apps/pulse/docs/iter25-handoff.md` — strict-abort framework, abort-dump decision tree
