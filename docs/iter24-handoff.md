# Pulse Iter 24 — Resume Handoff

Self-contained resume point. Iter 24 is a **stability re-run of iter 23**:
the architecture (per-species `MassActionModule` heads + `GlucoseGatedInsulinHead`,
intervention A + C from the iter-23 plan) is unchanged. Only training-loop
stability and Cloud Run timing are touched.

## Why iter 23 produced no usable checkpoint

Cloud Build `ab7bd13d-af70-45c4-8be3-7d19aadeadb6` (`train-20260424T163240Z`):

* Phase 1 (50 epochs, ~185s/epoch, ~2.6h) ran cleanly. Phase 2 epoch 50
  printed `loss=0.826576` (valid).
* Phase 2 epoch 55: `loss=nan gut=nan v_sur=nan lm=nan cohort=nan dose=nan
  gut_sweep=nan ins_sweep=nan`. NaN persisted through epoch 95.
* Cloud Run task hit the 36000s task timeout at 02:41 UTC the next day,
  while still running NaN forward passes. No checkpoint, no benchmark
  report, no delta-vs-baseline.

Two distinct failures:

1. **NaN explosion at the Phase 1 → Phase 2 boundary**. Phase 2 restarts
   Adam at `--phase2-lr=0.001` after Phase 1's cosine decayed to ~0.00015
   — a **6.7× LR jump**. Phase 2 also turns on the cohort, verifier
   surrogate, landmark, and dose-response signals which the per-species
   heads have *zero* gradient history with (Phase 1 only ran gut +
   gut_sweep + insulin_sweep). Fresh signals × fresh heads × LR shock
   drove one or more heads into a runaway regime; the resulting Inf
   z² loss produced NaN gradients; `clip_grad_norm_` does not filter
   non-finite gradients, so the optimizer step poisoned every parameter
   and every subsequent epoch was NaN.

2. **Wall-clock exceeded the 10h task timeout**. With per-species heads
   the model is 39,524 params (3× iter 22's ~14k). Phase 2 ran ~580s/epoch
   (3× Phase 1's ~185s — Phase 2 adds cohort, verifier, landmark,
   dose-response on top of Phase 1's signals). Total expected: 50×185 +
   50×580 = ~10.7h vs the 36000s (10h) `--task-timeout`.

## What iter 24 changes (stability fixes only)

### Fix 1 — drop `--phase2-lr` 0.001 → 0.0003

In `apps/pulse/train/spec.json`. Still 2× the Phase 1 cosine end-LR but
no longer a shock restart. The iter-21 / iter-22 model absorbed a 6.7×
restart because (a) it was 3× smaller, (b) the cohort signal had been
fitting through a shared trunk that already had stable curvature. The
iter-23 architecture has neither property.

### Fix 2 — bump task / build timeouts

In `apps/pulse/train/cloudbuild.yaml`:

* `--task-timeout`: `36000` → `43200` (10h → 12h Cloud Run task limit)
* top-level `timeout`: `39600s` → `46800s` (Cloud Build timeout, must
  exceed task-timeout + step overhead)

12h gives ~1.3h headroom over the 10.7h expected wall-clock.

### Fix 3 — defensive NaN guards in all 5 backward sites

In `apps/pulse/engine/pulse/training/`:

* `cohort_signal.py`
* `dose_response_signal.py`
* `gut_dose_sweep_signal.py`
* `insulin_sweep_signal.py`
* `trajectory_signal.py`

Pattern at every backward site:

```python
if torch.isfinite(loss):
    (w * loss).backward()
    nn.utils.clip_grad_norm_(ctx.params, max_norm=ctx.grad_clip)
    ctx.optimizer.step()
ctx.optimizer.zero_grad()
```

A non-finite loss costs one signal-step, not the whole run. The cohort
signal additionally reports a `nan_skip` sub-metric so we can tell
post-hoc whether the guard ever fired and on which epochs.

### What iter 24 does NOT change

Everything else from iter 23. Same `SpeciesHead` / `GlucoseGatedInsulinHead`
implementations (`apps/pulse/engine/pulse/modules/base.py`,
`metabolic.py`), same cohort spec list and weights (iter-21 baseline,
no per-spec weight bumps), same trajectory pipeline, same PatientParams.
**Cloud Run remains `--cpu=8 --memory=32Gi`** — speeding up the run with
more CPUs / GPU is a separate decision, deferred until iter 24 lands.

## Success criteria

Architectural gates (carry over from iter 23 — make-or-break test for
the per-species head hypothesis):

1. **`large_meal_glp1_peak`** predicted Δ ≥ 6 pmol/L (iter 22: 0.14).
2. **`ogtt_glucagon_suppression`** predicted Δ ≤ -10 pg/mL (iter 22:
   -0.016).
3. **`extended_fast_bhb_overnight`** predicted ≥ 0.10 mmol/L (iter 22:
   -0.003).
4. **`meal_hgo_suppression`** predicted ≤ -0.5 mg/min (iter 22: 0.007).
5. **`meal_ffa_suppression`** predicted ≤ -0.10 mmol/L (iter 22: -0.000).
6. **`meal_ghrelin_suppression`** predicted ≤ -10 pg/mL (iter 22: -0.094).
7. **`extended_fast_insulin_basal`** predicted ≤ 12 µU/mL (iter 22: 18.1).
8. **`ogtt_75g_insulin_peak`** predicted ≥ 45 µU/mL (iter 22: 39.1).

Both 7 *and* 8 must hold simultaneously — the falsifiable test for
intervention C.

Sanity gates:

9. No regression in gut probe metrics: ratio ≤ 1.20× on every dose ≥
   30g; per-gram slope ≤ 3.5; AUC sub-loss < 0.10.
10. Cold-calibration misleading-spec count ≤ 2.

Soft gates:

11. glucose `mean_mape` ≤ 0.50.
12. `overall_weighted_mape` ≤ 0.18.
13. verifier `overall_score` within -0.02 of iter 22's 0.862.
14. `trajectory_rollout` raw_loss ≤ 10.5.

Stability gates (new for iter 24):

* **S1**: Phase 2 reaches epoch 99/99 with non-NaN loss at every
  5-epoch print.
* **S2**: total wall-clock < 12h.
* **S3**: `cohort.nan_skip` total across all epochs is 0, or < 5
  transient skips not clustered at the phase boundary.

If S1–S3 hold but the architectural gates don't move, iter 23's
per-species head hypothesis is **falsified with stable training** and
iter 25 needs a different intervention (long-fast windows / long-fast
cohort signal). If S1–S3 fail, lower phase2-lr further (iter 25:
0.0003 → 0.0001) or add per-spec output clamping in the heads.

## Risks (carry over from iter 23, plus iter-24-specific)

| symptom | likely cause | next move |
| ------- | ------------ | --------- |
| Some collapsed species recover (GLP-1, glucagon, HGO, ghrelin) but BHB and FFA stay near zero | recovered ones live in time scales the 240-min windows can resolve; BHB and FFA need 16-22h fasted dynamics windows never visit | **Iter 25**: extend `TRAIN_WINDOW` for a fraction of windows OR add a long-fast knowledge contribution |
| Insulin basal drops below 5 µU/mL | glucose-gate `g_thresh` initialized too high or `g_temp` too narrow → at fasting the gate ≈ 0 and `raw_basal` absorbed all amplitude → optimizer overshoots when amplitude lifts | **Iter 25**: retune `g_thresh` init 0.5 → 0.3 (~104 mg/dL midpoint) and add a soft regularizer on `raw_basal` |
| `ogtt_75g_insulin_peak` recovers (≥45) but `extended_fast_insulin_basal` regresses upward | gate isn't decoupling — `raw_peak` and `raw_basal` share an MLP trunk before the 3-output linear | **Iter 25**: split `raw_basal` and `raw_peak` into two sub-MLPs inside the head |
| Verifier meal coherence regresses below iter 22 (+0.167) | per-species heads broke joint dynamics that iter 21/22 had via the shared trunk | **Iter 25**: re-tune `coupling_priors/metabolism.py` magnitude bands |
| Glucose `mean_mape` regresses while collapsed species recover | per-species heads gave each species too much capacity | **Iter 25**: tighten `--trajectory-band-default` 0.05 → 0.03, OR reduce per-species hidden_dim 48 → 24 |
| `trajectory_rollout` raw_loss climbs above iter 22's 10.80 | heads have more freedom to over-fit cohort against trajectory | **Iter 25**: tighten trajectory_band, OR add L2 penalty on per-species head weights |
| Nothing moves: collapsed specs at 0 AND glucose still bad | failure isn't head topology — upstream coupling carries no signal OR supervision data lacks information | **Iter 25**: instrument signal-balance per-species (not just per-signal); if `‖g_met_per_species‖` is healthy but predictions don't move, it's a data visibility problem |
| **NEW (iter 24)**: `cohort.nan_skip > 0` repeatedly | phase2-lr=0.0003 still too high OR a specific spec produces extreme outputs at session start | **Iter 25**: drop phase2-lr 0.0003 → 0.0001 OR add per-spec output clamping in `MassActionModule.forward` |
| **NEW (iter 24)**: wall-clock still > 11.5h | Phase 2 per-epoch time grew (more rollouts? gradient explosions slowing autograd?) | Bump `--cpu` 8 → 16 + set `OMP_NUM_THREADS=8` in cloudbuild.yaml — separate change after iter 24 lands |

## Code state (iter 24 changes ready to commit)

* `apps/pulse/train/spec.json` — `--phase2-lr=0.001` → `0.0003`,
  hypothesis + expectedEffect rewritten as iter-24 stability re-run.
* `apps/pulse/train/cloudbuild.yaml` — `--task-timeout=36000` →
  `43200`, top-level `timeout: 39600s` → `46800s`.
* `apps/pulse/engine/pulse/training/cohort_signal.py` — NaN guard
  around backward + step; reports `nan_skip` sub-metric.
* `apps/pulse/engine/pulse/training/dose_response_signal.py` — NaN
  guard; added `import torch` (only `from torch import nn` was present).
* `apps/pulse/engine/pulse/training/gut_dose_sweep_signal.py` — NaN
  guard.
* `apps/pulse/engine/pulse/training/insulin_sweep_signal.py` — NaN
  guard.
* `apps/pulse/engine/pulse/training/trajectory_signal.py` — NaN guard;
  also moves `loss_sum / gut_sum / n_windows` accumulation inside the
  isfinite branch so NaN windows don't poison the average.

## How to resume after reboot

```bash
cd /Users/grovina/Projects/grovina/platform/.claude/worktrees/pulse

# gcloud is already wired to pulse-iteration-runner SA via .claude/settings.local.json
# (env: GOOGLE_APPLICATION_CREDENTIALS + CLOUDSDK_ACTIVE_CONFIG_NAME=grovina-pulse).

# Verify iter 24 status
gsutil ls gs://grovina-pulse/training/jobs/ | tail -3

# When checkpoint.pt + benchmark-report.json + delta-vs-baseline.json all present,
# pull them. parallel_process_count=1 because gsutil multiprocessing is broken on macOS.
JOB_ID=train-<paste-here>
mkdir -p /tmp/iter24 && \
  gsutil -o "GSUtil:parallel_process_count=1" cp \
    gs://grovina-pulse/training/jobs/${JOB_ID}/{benchmark-report.json,delta-vs-baseline.json,checkpoint.pt,spec.json} \
    /tmp/iter24/

# First check: did the NaN guard fire?
python3 -c "
import json
r = json.load(open('/tmp/iter24/benchmark-report.json'))
# nan_skip lives in cohort training metrics, not in the benchmark report —
# read it from Cloud Logging via build ID instead, or parse the train logs
# during the run.
print(f'gate.passed={r[\"gate\"][\"passed\"]} overall={r[\"overall_weighted_mape\"]:.4f} verifier={r[\"verifier\"][\"overall_score\"]:.4f}')
"

# Iter 22 baseline for comparison (delta-vs-baseline.json computes this in-build)
mkdir -p /tmp/iter22 && \
  gsutil -o "GSUtil:parallel_process_count=1" cp \
    gs://grovina-pulse/training/jobs/train-20260424T071900Z/{benchmark-report.json,checkpoint.pt} \
    /tmp/iter22/

# Diagnostics on iter 24 checkpoint
cd apps/pulse/engine && .venv/bin/python -m pytest tests/ -x -q
.venv/bin/python -m pulse.diagnostics cold-calibration
.venv/bin/python -m pulse.diagnostics signal-balance --checkpoint /tmp/iter24/checkpoint.pt --n-patients 20
.venv/bin/python -m pulse.diagnostics cohort-ablation --checkpoint /tmp/iter24/checkpoint.pt --sample-patients 4
```

Note: iter-22 (and earlier) checkpoints remain **not loadable** by the
iter-24 model — same per-species `nn.ModuleList` head topology as iter
23, parameter shapes and names changed. Compare via the saved
benchmark / delta-vs-baseline JSON only.

## Cloud Build / job IDs

| iter | job ID | build status | benchmark-report key metrics |
| ---- | ------ | ------------ | ---------------------------- |
| 21 | train-20260423T175447Z | FAILURE (gate) | overall=0.190, glu_mean=0.543 |
| 22 | train-20260424T071900Z | FAILURE (gate) | overall=0.193, glu_mean=0.566 |
| 23 | train-20260424T163240Z | FAILURE (NaN + timeout) | no checkpoint produced |
| 24 | train-20260425T153910Z | WORKING at handoff time (build `4392b9bb-6b1b-43d5-bd32-79969537fe1c`) | (pending) |

## Pointer references (only read if needed)

* `apps/pulse/docs/iter23-handoff.md` — full detail on the per-species
  heads + glucose-gated insulin head architecture and why iter 22's
  cohort-amplification approach was falsified.
* `apps/pulse/docs/iter22-handoff.md` — gradient-budget vs
  representation-collapse hypothesis test, the per-spec ‖g_met‖ table
  iter-23's architecture was designed around.
* `apps/pulse/docs/iter21-handoff.md` — five-PatientParams cold-model
  recalibration; the cold-calibration diagnostic spec.
* `apps/pulse/docs/iter16-handoff.md` — factorized kernel structural
  guarantees on the gut module. Iter-23/24's `SpeciesHead` /
  `GlucoseGatedInsulinHead` are the metabolic-axis analog.
* `apps/pulse/docs/training-runs.md` — **commit before submit** workflow.
* `apps/pulse/engine/pulse/knowledge/AUTHORING.md` — guide for encoding
  new medical knowledge.
* `apps/pulse/docs/prd.md` — top-level vision.
