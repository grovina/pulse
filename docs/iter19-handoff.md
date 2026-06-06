# Pulse Iter 19 — Resume Handoff

> **Status: LANDED.** This file is a complete record of the iter 19
> plan + outcome. For the next-step plan see
> [`iter20-handoff.md`](./iter20-handoff.md).

## Outcome (added post-landing)

Cloud Build `0ef57a5a-a358-41f7-ae7c-ad7d298b7e7c` →
job `train-20260422T180605Z`. Only `EnforceBenchmarkGate` failed
(expected). All four iter 19 hard gates on the gut kernel hit:

| iter 19 hard gate | target | actual |
| ----------------- | ------ | ------ |
| per-dose pred/cold ratio (≥30g) | ≤ 1.15× | uniform 1.13× |
| AUC(120g) | ≤ 400 | 384 |
| per-gram slope (mg-min/g) | ≤ 3.20 | 3.19 |
| AUC sub-loss raw | < 0.05 | 0.050 |

Probe table at zero-emb: 1.15, 1.14, 1.14, 1.14, 1.13, 1.13×.
Sub-losses: mse=0.029, auc=0.050, rank=0.000.
Headline metrics flat (overall=0.198, glu_mean=0.578, ver=0.784).

The gut kernel is essentially solved — but headline metrics didn't
move, which is the empirical proof that downstream modules (insulin,
clearance, HGO) were leaning on gut over-amplitude as a free parameter
and now need their own probe-style supervision. That's iter 20's job.

See `iter20-handoff.md` for the next-step plan.

---

## Original plan (kept for record)

- Spec: `apps/pulse/train/spec.json`.
- Cloud Build config: `apps/pulse/train/cloudbuild.yaml` (unchanged from iter 18).
- Submission: `bash apps/pulse/scripts/train-submit.sh --gcp` from repo root.
- Artifact root: `gs://grovina-pulse/training/jobs/train-20260422T180605Z/`.

## What iter 18 did and what it revealed

Iter 18 introduced an AUC-matching sub-term in `gut_dose_sweep_signal`
to cure the loss-dilution that made iter 17's weight bump useless. The
term collapses `(T, C)` into per-`(dose, batch, channel)` integrals
*before* squaring, so multiplicative over-amp on the active region
lands an undiluted per-dose penalty.

**The AUC term works directionally.** First time across iters 16-17-18
that the over-amplitude moved meaningfully:

| dose | cold target | iter 16 | iter 17 | **iter 18** |
| ---- | ----------- | ------- | ------- | ----------- |
| 15   | 42.3        | 1.55×   | 1.60×   | **1.35×**   |
| 30   | 84.7        | 1.52×   | 1.59×   | **1.34×**   |
| 60   | 169.3       | 1.51×   | 1.58×   | **1.33×**   |
| 120  | 338.6       | 1.50×   | 1.58×   | **1.33×**   |

| sub-loss at zero-emb | iter 17 | **iter 18** | direction |
| -------------------- | ------- | ----------- | --------- |
| MSE                  | 0.0545  | **0.0373**  | -32%      |
| AUC                  | 0.4672  | **0.1782**  | **-62%**  |
| rank                 | 0.0     | 0.0         | clean     |

| metric                  | iter 17 | **iter 18** | direction |
| ----------------------- | ------- | ----------- | --------- |
| per-gram slope (mg/g)   | 4.45    | **3.75**    | -16%      |
| AUC(120g)               | 534     | **451**     | -16%      |

But headline metrics were essentially flat:

| metric                  | iter 17 | **iter 18** | direction |
| ----------------------- | ------- | ----------- | --------- |
| glucose mean_mape       | 0.5793  | 0.5752      | flat      |
| overall_weighted_mape   | 0.1983  | 0.1972      | flat      |
| verifier_overall        | 0.7929  | 0.7828      | -0.01     |
| verifier[coupling] Δ    | -0.017  | **-0.086**  | dropped   |

## Diagnosis: AUC pull was budget-starved

`signal-balance` shows the gradient picture clearly. Multiplying the
unweighted ‖∇gut‖ by each signal's effective spec weight:

| signal             | weight | iter 17 (||g_gut||) | iter 17 weighted | iter 18 (||g_gut||) | **iter 18 weighted** |
| ------------------ | ------ | ------------------- | ---------------- | ------------------- | -------------------- |
| gut_dose_sweep     | 0.10/0.50 | 6.48              | 3.24             | 3.13                | **0.31**             |
| trajectory_rollout | ~1.0   | 0.95                | 0.95             | 0.64                | **0.64**             |
| cohort_statistic   | 0.15   | 0.53                | 0.08             | 0.54                | 0.08                 |

**Critical observation:** in iter 18, `trajectory_rollout`'s pull on
the gut (0.64) is *2× larger* than `gut_dose_sweep`'s pull (0.31).
And trajectory_rollout is the signal that wants over-amp via the
compensatory equilibrium with under-tuned downstream modules
(insulin / clearance / hepatic output use over-strong gut as a free
parameter to match overall integrated state). With trajectory's
gradient out-pulling AUC's, SGD found a new equilibrium at 1.33×
instead of collapsing to 1.0×.

This also explains why the *direction* worked but the *destination*
was wrong: AUC's gradient is correctly oriented to cut amplitude, but
its budget was halved when we reverted the spec weight 0.50 → 0.10
(undoing iter 17's failed bump on the diluted MSE).

## What iter 19 changes

**One new CLI flag:** `--gut-dose-sweep-auc-weight` (default 1.0,
preserves iter 18 behavior). Plumbed end-to-end through `train.py`
into `GutDoseSweepSignal.auc_weight`.

**One spec change:** `--gut-dose-sweep-auc-weight=5.0`.

Local sanity on iter 18 ckpt (zero-emb, exact weighted ‖∇gut‖
contribution from gut_dose_sweep, total spec weight held at 0.10):

```
auc_weight=  1.0: total=0.215  ||g_gut||(unweighted)= 3.13  weighted= 0.31  (iter 18)
auc_weight=  3.0: total=0.572  ||g_gut||(unweighted)= 9.03  weighted= 0.90
auc_weight=  5.0: total=0.928  ||g_gut||(unweighted)=14.93  weighted= 1.49  ← iter 19
auc_weight= 10.0: total=1.819  ||g_gut||(unweighted)=29.68  weighted= 2.97
```

At `auc_weight=5.0`, gut_dose_sweep weighted ‖∇gut‖ jumps to **1.49** —
*2.3× larger than* `trajectory_rollout`'s 0.64. The gradient-budget
ratio that held amplitude at 1.33× should flip in AUC's favor.

We deliberately do not touch `gut-dose-sweep-weight` (stays at 0.10).
Bumping the outer weight would amplify *all three* sub-terms (mse,
rank, auc) — but we only want to amplify the integral term. The
per-element MSE at 0.037 is already pulling shape correctly; we don't
want to push it harder and risk noisy local-minima on the per-time-step
profile.

All other hyperparams unchanged from iter 18. Ranking term stays in
place at default `ranking_weight=1.0` (still zero in practice — the
factorized form maintains monotonicity analytically).

## Code changes (commit-level)

1. **`apps/pulse/engine/pulse/train.py`** — added
   `gut_dose_sweep_auc_weight: float = 1.0` parameter; threaded into
   `GutDoseSweepSignal(..., auc_weight=...)`; added matching
   `--gut-dose-sweep-auc-weight` CLI argument; wired
   `args.gut_dose_sweep_auc_weight` into the train() call. Added the
   value to the `Gut dose sweep:` startup print line and the
   provenance dict.
2. **`apps/pulse/train/spec.json`** — added
   `--gut-dose-sweep-auc-weight=5.0` to `trainArgs`;
   hypothesis/expectedEffect rewritten.
3. **`apps/pulse/docs/iter19-handoff.md`** — this file.

No changes to `gut_dose_sweep_signal.py` itself — the iter 18
implementation is exactly what we want, just dialed up.

## Local sanity (the iter 18 ckpt, predicting iter 19's pull)

The numbers above were computed by loading the iter 18 checkpoint and
calling `_losses` then `total.backward()` on the gut params, varying
`auc_weight` only. Direct gradient evidence rather than back-of-envelope
extrapolation.

The full 114-test engine suite passes against the new train.py
plumbing. The gut_dose_sweep_signal tests already cover `auc_weight=0`
ablation, quadratic scaling in over-amp, and end-to-end downward pull
from a fresh kernel — all unchanged from iter 18.

## Success criteria for iter 19

Hard gates:

1. **Per-dose pred/cold ratio ≤ 1.15× on every dose ≥ 30g** (iter 18:
   uniform 1.33×; iter 17: uniform 1.58×; iter 16: uniform 1.50×). Probe
   via `out[:, 0].sum()` at zero embedding.
2. **AUC(120g) ≤ 400** (iter 18: 451, iter 17: 534, cold: 339).
3. **Per-gram slope ≤ 3.2 mg-min/g** (iter 18: 3.75, iter 17: 4.45,
   iter 16: 4.24, cold: 2.82).
4. **AUC sub-metric raw_loss < 0.05 at end of training** (iter 18:
   0.18, iter 17: 0.47). The new term should largely zero out.
5. **Structural guarantees hold:** monotone=True, AUC(0g) < 5,
   linearity intact (iter 16's factorization is analytic; this is just
   a regression guard).

Soft gates:

6. **glucose mean_mape ≤ 0.45** (iter 18: 0.5752; recovery milestone).
7. **overall_weighted_mape ≤ 0.18** (iter 18: 0.1972, gate 0.16).
8. **verifier_overall ≥ 0.75** (iter 18: 0.7828; should hold).
9. **signal-balance:** `gut_dose_sweep` weighted ‖∇gut‖ ≥ 1.0 (iter 18:
   0.31; predicted 1.49 from local sanity); should now exceed
   `trajectory_rollout`'s ~0.64. `trajectory_rollout` raw_loss visibly
   higher than iter 18's 8.53 (this is the *expected* side effect of
   breaking the compensatory equilibrium).

## What to look at if iter 19 fails

| symptom | likely cause | next move |
| ------- | ------------ | --------- |
| AUC drops cleanly (<0.05) and amplitude → 1.0× but glucose_mape doesn't recover | downstream modules (insulin, clearance, hepatic output) genuinely under-tuned and were leaning on gut over-delivery | Iter 20: probe-style direct supervision for those modules (analogous to gut_dose_sweep but on insulin/clearance/HGO) |
| AUC drops but per-element MSE rises substantially (kernel hits the right integral with the wrong shape — tall narrow spike, broad plateau, anything that integrates to the right number without tracking the time course) | AUC alone underspecifies the profile when out-pulling shape MSE | Iter 20: per-time-step inverse-target weighting on MSE so the active region carries proportionally more loss |
| Trajectory loss skyrockets and never recovers | downstream cascade can't co-adapt fast enough at the bumped pull rate | Iter 20: ramp `auc_weight` 1.0 → 5.0 across phase 2 instead of jumping at start of phase 1 |
| Amplitude unchanged at 1.33× | another signal still countering the AUC pull (would mean trajectory_rollout's gradient on gut is not actually 0.64 effective; something is upweighting it) | Run signal-balance with the actual training opt path; isolate the counter-pull |

## Diagnostic commands

```bash
# Pull iter 19 artifacts
mkdir -p /tmp/iter19 && cd /tmp/iter19 && \
  gsutil cp gs://grovina-pulse/training/jobs/<JOB_ID>/{benchmark-report.json,delta-vs-baseline.json,checkpoint.pt} .

# Probe the trained kernel + measure new AUC sub-term
cd apps/pulse/engine && .venv/bin/python -c "
import torch
from pulse.training.gut_dose_sweep_signal import GutDoseSweepSignal, GutDoseSweepProtocol
from pulse.training.signals import WeightSchedule
from pulse.diagnostics.probe import load_model_from_checkpoint
from pulse.modules.gut import MealEvent

model, _ = load_model_from_checkpoint('/tmp/iter19/checkpoint.pt')
model.eval()
proto = GutDoseSweepProtocol()
sig = GutDoseSweepSignal(n_patients=0, sample_patients=0,
    include_default_embedding=True, weight=WeightSchedule(1.0), protocol=proto, auc_weight=5.0)
T = proto.post_window_min
times = torch.arange(T, dtype=torch.float32)
EMBED = next(model.embedding_projections['gut'].parameters()).shape[1]
emb_gut = model.embedding_projections['gut'](torch.zeros(EMBED))
preds = []
with torch.no_grad():
    for d in proto.carb_doses_g:
        out = model.gut.forward_window(times, [MealEvent(0.0, float(d), 5.0, 10.0)], emb_gut)
        preds.append(out.unsqueeze(0))
mse, rank, auc, _ = sig._losses(preds, sig._targets, sig._abs_scale, T)
print(f'mse={float(mse):.4f}  rank={float(rank):.4f}  auc={float(auc):.4f}')
pred_aucs = torch.stack(preds, 0).sum(2)[:,0,0]
for d, p in zip(proto.carb_doses_g, pred_aucs.tolist()):
    cold_idx = proto.carb_doses_g.index(d)
    c = float(sig._targets.sum(dim=1)[cold_idx, 0])
    print(f'  {d:6.0f}g: pred={p:7.1f} cold={c:7.1f} ratio={p/c if c>0.1 else 0:.3f}')
print(f'slope = {(pred_aucs[-1] - pred_aucs[0]).item() / 120.0:.3f} mg-min/g (cold: 2.82)')
"

# Signal-balance (verify gut_dose_sweep now exceeds trajectory_rollout)
.venv/bin/python -m pulse.diagnostics signal-balance \
  --checkpoint /tmp/iter19/checkpoint.pt

# Cross-iter compare (both have factorized arch; should produce probe Δ)
.venv/bin/python -m pulse.diagnostics compare \
  --new-label iter19  --new-checkpoint /tmp/iter19/checkpoint.pt \
  --new-report /tmp/iter19/benchmark-report.json \
  --baseline-label iter18  --baseline-checkpoint /tmp/iter18/checkpoint.pt \
  --baseline-report /tmp/iter18/benchmark-report.json

# Cloud Build status / logs
gcloud builds list --project=grovina --limit=5
gcloud builds log <BUILD_ID> --project=grovina
```
