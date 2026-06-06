# Pulse Iter 18 — Resume Handoff

Self-contained resume point. Read only this file to pick up the work.

## Where we are

**Iter 18 is staged.** Code changes are committed; the Cloud Build
submission is the next action.

- Spec: `apps/pulse/train/spec.json`.
- Cloud Build config: `apps/pulse/train/cloudbuild.yaml` (unchanged from iter 17).
- Submission: `bash apps/pulse/scripts/train-submit.sh --gcp` from repo root.
- Job ID assigned at submit time. Artifact root:
  `gs://grovina-pulse/training/jobs/<JOB_ID>/`.

## What iter 17 did and what it revealed

Iter 17 cranked `gut-dose-sweep-weight` 0.10 → 0.50 (5×) on the iter 16
factorized kernel, hypothesizing that more weight on the existing MSE
term would drag the kernel from its 1.5× over-amplitude down to cold
target. **It didn't.** The probe at zero embedding shows uniform
1.58× over-amplitude across every active dose:

| dose | cold target | iter 16 | iter 17 | trend     |
| ---- | ----------- | ------- | ------- | --------- |
| 15   | 42.3        | 65.7    | 67.9    | 1.55→1.60 |
| 30   | 84.7        | 128.9   | 134.4   | 1.52→1.59 |
| 45   | 127.0       | 192.2   | 201.0   | 1.51→1.58 |
| 60   | 169.3       | 255.4   | 267.6   | 1.51→1.58 |
| 90   | 254.0       | 381.9   | 400.7   | 1.50→1.58 |
| 120  | 338.6       | 508.4   | 533.8   | 1.50→1.58 |

Per-gram slope: cold = 2.82 mg-min/g, iter 16 = 4.24, **iter 17 = 4.45**.

Headline metrics regressed (but not dramatically):

| metric                | iter 16 | **iter 17** | direction |
| --------------------- | ------- | ----------- | --------- |
| glucose mean_mape     | 0.5653  | **0.5793**  | ✗ +0.014  |
| glucose median_mape   | 0.4506  | **0.4622**  | ✗ +0.012  |
| overall_weighted_mape | 0.1956  | **0.1983**  | ✗ +0.003  |
| verifier_overall      | 0.7897  | **0.7929**  | ✓ +0.003  |
| verifier[meal]        | 0.6138  | **0.6288**  | ✓ +0.015  |
| verifier[coupling]    | (high)  | -0.017      | ✗ slight  |

## Diagnosis: per-element MSE is structurally diluted

Computed locally on `iter17/checkpoint.pt` against the unmodified
gut_dose_sweep targets:

```
mse  = 0.05454      (per-element, mean over [B, T=240, C=4])
auc  = 0.46719      (per-(dose, batch, channel) integral; see code)
rank = 0.00000      (no inversions)
```

The MSE term is the loss the iter 17 weight-bump multiplied through.
At raw value 0.054, weighting it by 0.50 gave an effective contribution
of 0.027 — and that's *with* a uniform 60% over-amplitude on every
active region. Why so small?

The cold-target window is [240 min × 4 channels] but only the active
nutrient channels carry meaningful magnitude, and even on those the
post-meal absorption rate is concentrated in the first ~120 min. The
per-element mean averages the squared error across mostly-zero target
slots; multiplicative over-amp on the active region gets divided by
~B·T·C inactive entries. **Increasing weight just multiplies through
that dilution.**

The same dilution affects `trajectory_signal.gut_loss` (the line at
`trajectory_signal.py:434` that takes `.mean()` over the same window).

## What iter 18 changes

**One signal change (no architecture change):** add an AUC-matching
term to `GutDoseSweepSignal._losses`.

```python
auc_targets = self._auc_targets.to(device).unsqueeze(1)   # [D, 1, C]
auc_scale   = self._auc_scale.to(device)                  # [C]
pred_aucs   = pred_stack.sum(dim=2)                       # [D, B, C]
auc_loss    = ((pred_aucs - auc_targets) / auc_scale).pow(2).mean()
```

The integral `pred_aucs.sum(dim=2)` collapses the time axis *before*
squaring, so a uniform multiplicative over-amp lands a per-dose loss
proportional to the cold-target AUC squared — undiluted. Then the mean
over [D, B, C] = ~112 already-collapsed scalars (vs the MSE's
~B·D·T·C ≈ 27000 mostly-zero per-time-step errors) keeps the term
numerically meaningful.

`auc_scale[c] = abs_scale[c] · post_window_min / 4` — the typical AUC
under a kernel-shaped profile of width `post_window_min` and peak
`abs_scale`. This normalizes channels so the binary nutrient_flag
(small AUC) and the glucose absorption rate (large AUC) contribute on
comparable footing.

`auc_weight = 1.0` by default; gut-dose-sweep-weight reverts 0.50 →
0.10 (iter 16's value). The existing per-element MSE and ranking term
stay in place — they pin shape and ordering, the new term pins
amplitude.

**Spec change:** `--gut-dose-sweep-weight=0.50 → 0.10` (revert to
iter 16). The new AUC term provides the amplitude pull; the iter 17
weight bump is no longer needed and is now actively harmful (it would
amplify the diluted MSE's noisy gradient on shape).

All other hyperparams unchanged from iter 17.

## Code changes (commit-level)

1. **`apps/pulse/engine/pulse/training/gut_dose_sweep_signal.py`** —
   - Added `auc_weight: float = 1.0` field to `GutDoseSweepSignal`.
   - Cached `_auc_targets` (cold target integrals, [D, C]) and
     `_auc_scale` (per-channel normalization, [C]) in `__post_init__`.
   - `_losses()` now returns 4-tuple `(mse, rank, auc, n_inv_zero)`;
     `compute()` adds `self.auc_weight * auc_loss` to the total loss
     and reports `auc` as a sub-metric.
2. **`apps/pulse/engine/tests/test_gut_dose_sweep_signal.py`** —
   - Updated existing `_losses` consumers for the new 4-tuple.
   - New `TestAucMatchingTerm` class with four tests:
     - AUC = 0 when pred matches cold target.
     - AUC scales as `(α − 1)²` under multiplicative over-amp.
     - `auc_weight=0` ⇒ total loss equals `mse + ranking_weight·rank`.
     - End-to-end: AUC supervision with `mse + rank` zeroed pulls a
       fresh kernel's AUC at 120 g toward the cold target.
3. **`apps/pulse/train/spec.json`** — `gut-dose-sweep-weight`
   0.50 → 0.10; hypothesis/expectedEffect rewritten.
4. **`apps/pulse/docs/iter18-handoff.md`** — this file.

## Local sanity (the iter 17 ckpt, predicting iter 18's pull)

```
iter17 @ zero-emb against cold targets:
  mse  = 0.05454    (existing per-element MSE)
  rank = 0.00000    (no inversions; structural)
  auc  = 0.46719    <-- new term
  n_inversions = 0
```

The new AUC term is **8.6× larger** than the existing MSE term at the
exact failure mode (uniform 1.58× over-amp). Quadratic scaling means
it shrinks toward 0 as the kernel approaches 1.0× ratio — verified
analytically: `auc_loss(α=1.25) / auc_loss(α=1.5) = 4.0` (unit test
`test_auc_grows_quadratically_with_overamp`).

All 114 engine tests pass with the new code.

## Success criteria for iter 18

Hard gates:

1. **Per-dose pred/cold ratio ≤ 1.25× on every dose ≥ 30g** (iter 17:
   uniform 1.58×; iter 16: uniform 1.50×). Probe via `out[:, 0].sum()`
   at zero embedding.
2. **AUC(120g) ≤ 420** (iter 17: 534, iter 16: 508, cold: 339).
3. **Per-gram slope ≤ 3.5 mg-min/g** (iter 17: 4.45, iter 16: 4.24,
   cold: 2.82). Probe via `(AUC(120g) − AUC(0g)) / 120`.
4. **Structural guarantees hold:** monotone=True, AUC(0g) < 5,
   `(AUC(2d) − AUC(d)) / AUC(d) < 0.05` for any dose pair (linearity).

Soft gates:

5. **glucose mean_mape ≤ 0.45** (iter 17: 0.5793, iter 16: 0.5653;
   recovery milestone, not the eventual ≤ 0.30 target).
6. **overall_weighted_mape ≤ 0.18** (iter 17: 0.1983, gate 0.16).
7. **verifier_overall ≥ 0.75** (iter 17: 0.79; should hold).
8. **signal-balance:** `gut_dose_sweep` `auc` sub-metric raw value
   < 0.10 (down from 0.467); `gut_dose_sweep` becomes the dominant
   ‖∇gut‖ contributor; `trajectory_rollout` raw_loss visibly higher
   than iter 17 (the *expected* side effect of stripping over-amp).

## What to look at if iter 18 fails

| symptom | likely cause | next move |
| ------- | ------------ | --------- |
| AUC drops cleanly but per-element MSE rises (kernel hits the right integral with the wrong shape — tall narrow spike) | AUC alone underspecifies the profile | Iter 19: per-time-step inverse-target weighting on MSE so the active region carries proportionally more loss |
| Amplitude drops on target but glucose_mape doesn't recover | downstream modules (insulin, clearance, hepatic output) genuinely under-tuned and were leaning on gut over-delivery | Iter 19: probe-style direct supervision for those modules (analogous to gut_dose_sweep but on insulin/clearance/HGO) |
| Counter-regulation deltas don't grow | same as above | same as above |
| Amplitude unchanged at 1.5×+ | another signal is countering the AUC pull | Run signal-balance; if `cohort_statistic` or `trajectory_rollout` has a positive-amp gradient on gut, isolate which one and either down-weight it or add a structural fix |
| Trajectory loss skyrockets and never recovers | downstream modules can't co-adapt fast enough | Iter 19: ramp `auc_weight` 0.0 → 1.0 across phase 2 instead of jumping; or warm-start downstream module supervision early |

## Diagnostic commands

```bash
# Pull iter 18 artifacts
mkdir -p /tmp/iter18 && cd /tmp/iter18 && \
  gsutil cp gs://grovina-pulse/training/jobs/<JOB_ID>/{benchmark-report.json,delta-vs-baseline.json,checkpoint.pt} .

# Probe the trained kernel (slope, AUC, monotonicity, drift, counter-reg)
cd apps/pulse/engine && .venv/bin/python -m pulse.diagnostics probe \
  --checkpoint /tmp/iter18/checkpoint.pt

# Signal-balance (verify AUC term shrank from 0.467, not just MSE pulled)
.venv/bin/python -m pulse.diagnostics signal-balance \
  --checkpoint /tmp/iter18/checkpoint.pt

# Cross-iter compare (both have factorized arch; should produce probe Δ)
.venv/bin/python -m pulse.diagnostics compare \
  --new-label iter18  --new-checkpoint /tmp/iter18/checkpoint.pt \
  --new-report /tmp/iter18/benchmark-report.json \
  --baseline-label iter17  --baseline-checkpoint /tmp/iter17/checkpoint.pt \
  --baseline-report /tmp/iter17/benchmark-report.json

# Direct AUC measurement at zero-emb (the headline iter 18 number)
.venv/bin/python -c "
import torch
from pulse.training.gut_dose_sweep_signal import GutDoseSweepSignal, GutDoseSweepProtocol
from pulse.training.signals import WeightSchedule
from pulse.diagnostics.probe import load_model_from_checkpoint
from pulse.modules.gut import MealEvent

model, _ = load_model_from_checkpoint('/tmp/iter18/checkpoint.pt')
model.eval()
proto = GutDoseSweepProtocol()
sig = GutDoseSweepSignal(n_patients=0, sample_patients=0,
    include_default_embedding=True, weight=WeightSchedule(1.0), protocol=proto)
T = proto.post_window_min
times = torch.arange(T, dtype=torch.float32)
EMBED = next(model.embedding_projections['gut'].parameters()).shape[1]
emb_gut = model.embedding_projections['gut'](torch.zeros(EMBED))
preds = []
with torch.no_grad():
  for d in proto.carb_doses_g:
    out = model.gut.forward_window(times, [MealEvent(0.0, float(d), 5.0, 10.0)], emb_gut)
    preds.append(out.unsqueeze(0))
mse, rank, auc, n_inv = sig._losses(preds, sig._targets, sig._abs_scale, T)
print(f'mse={float(mse):.4f}  rank={float(rank):.4f}  auc={float(auc):.4f}')
for d, p in zip(proto.carb_doses_g, torch.stack(preds, 0).sum(2)[:,0,0].tolist()):
  c = float(sig._auc_targets[proto.carb_doses_g.index(d), 0])
  print(f'  {d:6.0f}g: pred={p:7.1f} cold={c:7.1f} ratio={p/c if c>0.1 else 0:.3f}')
"

# Cloud Build status / logs
gcloud builds list --project=grovina --limit=5
gcloud builds log <BUILD_ID> --project=grovina
```
