# Pulse iter 79 — rollout-dedup refactor (measurement-first)

**Status at handoff: code complete + unit-validated, NOT yet committed or
dispatched.** Working-tree changes are on disk (survive a context clear); a
local smoke was healthy and mid-phase-2 at handoff time.

## Why this iter (the decision)

iter 78's joint-aux-step fix was validated in phase 1 but its **honest benchmark
was deferred**: a full run never completed because one phase-2 epoch is ~1.6 h
(the three heavy signals are sequential Python loops — `cohort_statistic`
~1754 s over 31 specs, `physiology_rules` ~2100 s over 61 rules,
`cold_model_distillation` ~1339 s), so ~10 phase-2 epochs ≈ 16 h and the
dispatched runs timed out before writing a report.

Direction chosen with the user: **measurement-first**. Make the heavy signals
structurally cheaper *without changing what they compute*, so the run completes
and emits the **true iter-78 numbers** (config held byte-identical). This is the
clean prerequisite for any later modeling move (the teacher mass-balance work,
multi-timescale, etc.) being attributable. The user's broader steer: don't be
constrained by the repo's existing plan — aim for the best final outcome; the
teacher-completeness / multi-timescale frontier is the north star *after* we
have an honest baseline.

## What changed (two gradient-identical refactors)

Both are pure rollout-dedup — same losses, same gradients, fewer ODE rollouts.

### 1. `physiology_rules` → arm-major (`pulse/training/physiology_rules_signal.py`)
The 61 rules reference only **10 distinct arm protocols**, but the iter-68
rule-major loop re-rolled each arm once *per rule* (115 rollouts = 77,880
integration steps/epoch). Inverted to **arm-major**: group rules by
`(arm.label, init_mode)`, roll each distinct protocol ONCE, score every rule
that uses it against the shared trajectory (10 rollouts = **8,040 steps/epoch,
9.7× less**). One backward per arm-group; per-rule loss/diagnostics
reconstructed from detached accumulators. `physiology_rules_loss.py` unchanged
(its `physiology_rule_loss_one_rule` / `physiology_rules_epoch_loss` kept for
diagnostics + the equivalence test).

### 2. `cohort_statistic` → protocol batching (`pulse/training/cohort_signal.py` + `pulse/cohort_loss.py`)
31 specs collapse to **18 distinct arm protocols** (four extended-fast 1440-step
specs, four OGTT, four sleep-dip specs each share a protocol, differing only in
marker/window/target). Group specs by their frozen `.arms` tuple and roll each
protocol ONCE over the group's **stacked per-spec cold-init states** (cold init
is seeded per spec, so states differ even within a protocol — the batched
rollout stacks `[S×B]` rows: spec s's init repeated across the B embeddings).
45,240 → **28,500 steps/epoch, 1.59×**. New `cohort_statistic_loss_group()` in
`cohort_loss.py`; `_rollout_arm_batched` refactored into a thin wrapper over the
new `_rollout_arm_states` (per-row initial states). One backward per group.

**Why gradient-identical:** both losses are means over (units × patients) of
per-rollout terms; mean/sum are linear, so each rollout's contribution is a
separable partial loss and one backward per group accumulates into `.grad`
exactly as one backward per rule/spec did (PyTorch additive-grad — the same
linearity the iter-68 per-rule/per-spec memory fix relies on). Peak memory is
≤ before (one group's graph live at a time; largest cohort group = 4 specs × 7
embeddings = 28-wide, far below the all-31 graph that OOM'd in iter 67).

## Validation done

- **189/189 tests pass** (`uv run python -m pytest -q`, ~20 min).
- **Two new per-parameter gradient-equivalence tests** assert the refactored
  path reproduces the old path's reported loss AND every model+embedding
  gradient to float-32 tolerance (`atol=1e-6, rtol=1e-4`):
  - `tests/test_physiology_rules_signal.py::TestPhysiologyRulesArmMajorEquivalence`
    (covers shared-arm rules + a multi-arm rule + cold init).
  - `tests/test_cohort_signal.py::TestCohortStatisticProtocolBatchingEquivalence`
    (covers same-protocol specs with distinct cold inits + singleton + 2-arm delta).
- Real-set dedup factors confirmed by counting (physiology 9.7×, cohort 1.59×).
- Local smoke (`uv run python -m pulse.train --spec train/spec.json
  --n-patients=1 --phase1-epochs=1 --phase2-epochs=1`) was healthy: phase 1
  done (heavy signals correctly gated 0s), phase 2 trajectory ran, refactored
  cohort/physiology executing. **Re-run/verify this smoke first thing on resume**
  — confirm phase-2 cohort_statistic + physiology_rules complete and a
  benchmark prints, then proceed.

## Files changed (uncommitted)

- `pulse/training/physiology_rules_signal.py` — arm-major compute()
- `pulse/training/cohort_signal.py` — protocol-grouped compute()
- `pulse/cohort_loss.py` — `_rollout_arm_states`, `cohort_statistic_loss_group`
- `tests/test_physiology_rules_signal.py` — equivalence test + import
- `tests/test_cohort_signal.py` — equivalence test + import
- `train/spec.json` — iter 78→79, hypothesis/expectedEffect; **trainArgs
  byte-identical** (verified: `git diff train/spec.json` = 3 lines)

## Remaining steps (resume here)

1. **Verify the smoke** completed phase 2 cleanly (above). Fix anything it surfaces.
2. **Commit to main** (the repo's iter workflow; ties the trainer image to a
   clean SHA per the repro doctrine). Suggested message: `iter 79: rollout-dedup
   (physiology arm-major 9.7x, cohort protocol-batching 1.59x) — gradient-
   identical, unblocks deferred iter-78 honest baseline`.
3. **Build the trainer image** at that SHA (deploy.sh dies on the engine step —
   build directly):
   ```bash
   TAG=$(git rev-parse --short HEAD)
   gcloud builds submit . --project grovina-pulse \
     --config deploy/cloudbuild.yaml \
     --service-account projects/grovina-pulse/serviceAccounts/pulse-dev@grovina-pulse.iam.gserviceaccount.com \
     --substitutions _REGION=europe-west1,_REPO=pulse,_TAG=$TAG
   ```
4. **Pin the job** to the immutable SHA + 24 h timeout:
   ```bash
   gcloud run jobs update trainer --project grovina-pulse --region europe-west1 \
     --image europe-west1-docker.pkg.dev/grovina-pulse/pulse/trainer:$TAG \
     --cpu 4 --memory 16Gi --max-retries 0 --task-timeout 86400s
   ```
5. **Dispatch** the honest baseline run:
   ```bash
   gcloud run jobs execute trainer --project grovina-pulse --region europe-west1 \
     --args=--spec=train/spec.json,--gcs-bucket=grovina-pulse-data,\
   --gcs-object=training/jobs/iter79/model.pt,\
   --benchmark-dataset-uri=pulse/benchmark.dataset.generated.json
   ```
6. **Poll for completion** (`succeededCount` LIES — a timeout-kill exits 0; see
   [[pulse-cloud-run-training-ops]]). Wait for the artifact:
   ```bash
   gsutil ls gs://grovina-pulse-data/training/jobs/iter79/   # benchmark-report.json = real success
   ```
   Expected ~12 h wall-clock (phase-2 epoch ~0.9 h × 10 + ~2.7 h phase-1).

## What to read off the run (= the deferred iter-78 baseline)

Same checklist as iter 78 (numbers reflect the iter-78 model exactly — refactor
is gradient-identical): `per_marker_by_source['real']` skill_vs_persistence for
glucose/hr/sbp/dbp/temp; `overall_weighted_mape_by_source` (real vs teacher);
`textbook_mean_pass_rate` (gate 0.45); `dist_insulin/lactate/hepatic_output` in
cold-distill telemetry; aux losses tracking weights coherently.

## Risks / fallbacks

- **(R1)** If it still times out, the heavy remainder is
  `cold_model_distillation` (~1339 s; its inner Adam anchor-calibration is
  sequential, deliberately NOT batched — would need an `integrate()` API change
  to accept per-batch sleep/activity/start-time). Next levers: batch its 8
  level-anchor windows, or tune recalib cadence (`--cold-distill-*`), or the
  documented lighter-eval flags (`--cohort-sample-patients=1
  --physiology-rules-sample-patients=1 --cold-distill-protocols-per-epoch=1
  --phase2-epochs=6`).
- **(R2)** Cohort grouping peak memory: largest group 4 specs × 7 = 28-wide ×
  1440 steps checkpointed — ~2-3 GB, well within 16 Gi. If OOM, lower
  `--cohort-sample-patients` or cap group size.
- **(R3)** Grouping key is `spec.arms` value-equality — near-identical-but-
  unequal protocols just stay separate (correctness preserved, less speedup).

## Next iter (iter 80, after the baseline lands)

The first fundamentally-correct modeling move, measured against this baseline:
**close the glycogen↔glucose mass-balance loop in the teacher** (`knowledge/
full_body.py`) — split `Hep` into glycogenolytic + gluconeogenic so liver-
glycogen depletion drives fasting hepatic output + ketosis (true-by-
construction mass conservation; makes the iter-76 slow pools load-bearing;
advances the multi-timescale north star). See `docs/modeling-state.md` and the
`full_body.py` glycogen-block comment that names this as the next step.
