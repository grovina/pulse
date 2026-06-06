# Pulse Iter 14 — Resume Handoff

Self-contained resume point. Read only this file to pick up the work.

## Where we are

**Iter 14 is staged.** Code changes are committed; the Cloud Build
submission is the next action.

- Spec: `apps/pulse/train/spec.json`.
- Cloud Build config: `apps/pulse/train/cloudbuild.yaml`.
- Submission: `bash apps/pulse/scripts/train-submit.sh --gcp` from repo root.
- Job ID assigned at submit time. Artifact root:
  `gs://grovina-pulse/training/jobs/<JOB_ID>/`.

When the run finishes, five artifacts land under that root:

- `checkpoint.pt`, `benchmark-report.json`, `spec.json`, `meta.json`,
  `delta-vs-baseline.json` (output of `CompareToPrevious`).

## What iter 13 actually did, and why it regressed

Iter 13 fixed the `GUT_OUTPUT_SCALE` to per-channel cold-model
magnitudes `(2.0, 0.3, 0.5, 1.0)` mg/min so that gut MSE wasn't silently
flattened ~225–1100× per channel. The intent was correct. The realized
outcome was a hard regression on every benchmark axis:

| metric                 | iter 12 | iter 13 |
| ---------------------- | ------- | ------- |
| overall_mape           | ~0.18   | 0.2077  |
| **glucose_mape**       | ~0.30   | **0.5792** |
| hr_mape                | ~0.21   | 0.2516  |
| verifier_cat[meal]     | ~0.66   | 0.6238  |
| benchmark gate         | failed  | failed  |

Diagnostic on the iter-13 checkpoint (`pulse.diagnostics.signal_balance`
on `/tmp/iter13.pt`, n=20 patients, zero embeddings):

```
signal             raw_loss   ‖∇gut‖     ‖∇all‖
gut_dose_sweep     0.0976     0.4274     0.4274
cohort_statistic   3.18       2.70       4.47
trajectory_rollout 9.11       0.318      20.61
```

Reference (iter-11 / iter-12 checkpoint, same probe): gut_dose_sweep raw
loss ≈ 1.5, ‖∇gut‖ for cohort ≈ 25–80×. **The kernel collapsed into a
flat region of the gut loss landscape** — gut output is near-constant,
dose-insensitive, and no signal can move the kernel out because gradient
norms are small everywhere.

Mechanism: the SI-correct `GUT_OUTPUT_SCALE` made
`((pred − tgt) / abs_scale)²` admit a degenerate global minimum at the
dose-averaged target shape (~0.3 mg/min everywhere). With
`gut-dose-sweep-weight=0.10` and `gut-loss-weight=0.10`, the gut signal
dominated optimization for the first few epochs, slammed the kernel into
that basin, and overshot. From there the integrated-state signals can't
recover dose-sensitivity because they barely see gut.

## What iter 14 changes

### 1. Gut weights drop 5× (fix the over-regularization)

```
--gut-dose-sweep-weight   0.10 → 0.02
--gut-loss-weight         0.10 → 0.02
```

Keeps the SI-correct scale (no scale fudging — that fix was right) but
makes the gut signals nudge instead of dominate. Trajectory + cohort
signals carry dose-response learning indirectly through integrated state,
which is how iter 11 (last good gut behavior) was actually trained.

### 2. Compute refactor: batched embeddings + gut-window precompute

Iter-13 profiling showed `cohort_statistic` was 91% of training
wall-clock and ran ~110 sequential `integrate` calls per epoch. The hot
path is now batched at the embedding dimension across the four signals
that were previously a per-embedding Python loop:

- `pulse.modules.gut.GutModule.forward_window` — accepts batched
  embeddings `[B, EMB]`, returns `[B, T, GUT_OUTPUT_DIM]`.
- `pulse.model.ModularPhysiologyNetwork.forward` — accepts
  `gut_override` of shape `[B, ...]`; explicitly requires
  `gut_override` when `batch > 1` (the previous code silently used
  `embedding[0]` for gut output, which was a latent bug).
- `pulse.model.integrate` — accepts batched `initial_state` /
  `embedding`; requires precomputed `gut_outputs` when `batch > 1`.
- New `pulse.model.precompute_gut_outputs(model, embeddings, t_eval)`
  — single-call gut window for the whole batch, designed to be passed
  to `integrate`.
- `cohort_loss._rollout_arm_batched` / `_arm_statistic_batched` —
  stack all K supervised embeddings, one `integrate` per arm.
- `dose_response.predicted_slopes_batched` — one `integrate` per dose
  for the whole patient batch.
- `gut_dose_sweep_signal.compute` — one `forward_window` per dose for
  the whole patient batch.
- `trajectory_signal.compute` — precomputes `gut_window` once per
  integration window and reuses it for both `integrate` and the
  internal `gut_loss` term.

Math is preserved (same per-embedding outputs, just batched). Expected
speedup: ~3–5× on cohort, 3–5× on dose_response, ~7× on gut_dose_sweep,
~1.3× on trajectory. Net wall-clock should drop substantially from
iter 13's ~9 h.

`torch.compile` was tested as a third tier and reverted: micro-bench
showed 2.9× speedup but full-loop hit recompile thrashing on dynamic
shapes / `requires_grad` flips and ended up slower (3:24 vs 2:52). See
the comment block in `pulse/train.py`.

### 3. CI fixes

Two latent CI flaws surfaced during iter 13 triage:

- **`--fail-on-benchmark` short-circuited comparison.** When the trainer
  exited non-zero on gate failure, `ResolveBaseline` and
  `CompareToPrevious` were skipped — exactly the steps you most need on
  a regression. Removed from `train.py` and `train-submit.sh`. Gate
  enforcement moved to a dedicated `EnforceBenchmarkGate` Cloud Build
  step that runs *after* `CompareToPrevious` and exits 1 if
  `gate.passed != True`. The trainer now always exits 0 on physical
  completion; CI orchestration owns "is this run shippable".
- **`ResolveBaseline` picked baselines without benchmark reports.**
  Now requires both `checkpoint.pt` *and* `benchmark-report.json` to
  exist in the candidate prior run. Prevents wasted compare passes.

### 4. Diagnostic-as-code: signal balance probe

Promoted the iter-13 triage script (formerly `/tmp/diag_iter13.py`)
into `pulse.diagnostics.signal_balance`, exposed as
`python -m pulse.diagnostics signal-balance --checkpoint <path>`.

Same pattern as `probe` and `compare`: importable module, JSON-
serializable report, CLI that accepts local paths or `gs://` URIs.
Use it whenever a weight change ships and the post-run benchmark looks
unexpected — it tells you, per signal, how much that signal *wants* to
move the gut kernel at the converged checkpoint.

### 5. Trainer prints runtime config

`_set_runtime` now prints CPU/threading config at startup:
`torch_threads`, `interop`, `os_cpus`, `sched_affinity`, `deterministic`.
Cheap, makes Cloud Run pinning behavior visible without re-instrumenting.

## Submission

```bash
bash apps/pulse/scripts/train-submit.sh --gcp
```

This commits nothing on its own — the working tree is the source. Verify
`git status` is clean (or carries only the spec.json edits you intend)
before submitting; the meta records `gitSha` + `gitClean` and the
`pulse.train` entrypoint refuses to run on a dirty tree.

Cloud Build streams logs in real-time now (`PYTHONUNBUFFERED=1` in the
engine Dockerfile, async `gcloud run jobs execute` + `gcloud beta logging
tail` in cloudbuild.yaml). You can follow along in the Cloud Build UI
without waiting for the job to finish.

## How to check the run

```bash
./apps/pulse/train/get-results.sh grovina-pulse <JOB_ID>

gsutil cat gs://grovina-pulse/training/jobs/<JOB_ID>/delta-vs-baseline.json | jq '.'

./apps/pulse/train/compare-runs.sh <JOB_ID>

cd apps/pulse/engine && .venv/bin/python -m pulse.diagnostics probe \
  --checkpoint gs://grovina-pulse/training/jobs/<JOB_ID>/checkpoint.pt

cd apps/pulse/engine && .venv/bin/python -m pulse.diagnostics signal-balance \
  --checkpoint gs://grovina-pulse/training/jobs/<JOB_ID>/checkpoint.pt
```

## Success criteria

- **Gate failures recede.** `glucose_mape` returns to ≤ 0.30 (iter 13:
  0.58); `overall_mape` ≤ 0.20.
- **Probe at zero embedding:** gut sweep AUC monotone in carb dose;
  glucose-appearance peak time ~30–50 min for nonzero doses.
- **Counter-regulation deltas** (glucagon, FFA, ghrelin) negative under
  a 75 g mixed meal.
- **Signal balance:** `gut_dose_sweep` raw loss > 0.5 (vs iter 13: 0.10
  — confirms the kernel is no longer collapsed).
- **Wall-clock:** noticeably under iter 13's ~9 h. If it's not, the
  cohort batching didn't kick in — check the runtime print at startup
  and inspect `_arm_statistic_batched` is being called (not the legacy
  per-embedding path; that path is gone, but worth verifying the rest of
  the dataset uses K > 1).

## Decision tree for iter 15

1. **Gate passes, probe monotone, signal balance healthy.** Move to the
   pending Phase 3 cohort specs (counter-regulatory: see Pending work).
2. **Probe monotone but gate still failing on glucose_mape.** Trajectory
   weight is over-tight. Try `--trajectory-band-default 0.05 → 0.08` or
   bump `--cohort-statistic-weight` 0.15 → 0.20.
3. **Probe still flat / inverted, signal balance shows ‖∇gut‖ small for
   gut_dose_sweep.** The 5× weight cut was too aggressive. Try
   `--gut-dose-sweep-weight 0.02 → 0.04`. Don't go above 0.05 without a
   second look — that's where iter 13 got into trouble.
4. **Probe inverted but ‖∇gut‖ large.** Different problem: the kernel
   is responsive but pointing the wrong way. Inspect
   `GutDoseSweepProtocol` targets — they may need a longer 0 g window
   or different quantization.
5. **Wall-clock unchanged from iter 13.** Compute refactor didn't take
   effect. Bisect: run `python -m pulse.train` locally with `--n-epochs 2
   --n-patients 4` and verify the cohort signal logs the batched path.

## Pending work from the broader plan

These remain queued behind getting a stable, gate-passing baseline.

### Phase 3 — counter-regulatory cohort specs

In `apps/pulse/engine/pulse/knowledge/cohort_statistics/`, modeled on the
existing 11 specs:

- `OGTT_GLUCAGON_SUPPRESSION` (Müller 1970, Unger 1971)
- `MIXED_MEAL_GLUCAGON_SUPPRESSION`
- `MEAL_FFA_SUPPRESSION` (Frayn 2002)
- `EXTENDED_FAST_FFA_RISE`
- `EXTENDED_FAST_INSULIN` (basal floor)

Each is a `CohortStatisticSpec` with literature effect size + SE.
Citations in the spec docstring. Bump `cohort_statistic_weight` in
`spec.json` to absorb the spec-count growth.

### Future perf work

- **Cloud Run → Vertex AI on `c3-highcpu-22` with multi-thread.**
  Cloud Run pins to ~6 effective cores; Vertex on c3-highcpu-22 unlocks
  22. Combined with the existing opt-in `--deterministic` flag, an
  expected 2-3× wall-clock improvement on top of iter 14's batching.
- **Re-evaluate `torch.compile` once shapes stabilize.** The current
  blocker is dynamic batch sizes / `requires_grad` flips. If we land
  fixed-K cohort batching and freeze `requires_grad` during certain
  phases, the recompile thrashing should disappear.

## Key files (iter 14 changes)

- `apps/pulse/engine/pulse/modules/gut.py` — `forward_window` batched.
- `apps/pulse/engine/pulse/model.py` — `forward` accepts batched
  `gut_override`; `integrate` accepts batched state/embedding;
  new `precompute_gut_outputs` helper.
- `apps/pulse/engine/pulse/cohort_loss.py` — `_rollout_arm_batched`,
  `_arm_statistic_batched`, batched `cohort_statistic_loss_one_spec`.
- `apps/pulse/engine/pulse/dose_response.py` —
  `predicted_slopes_batched`, batched `dose_response_epoch_loss`.
- `apps/pulse/engine/pulse/training/gut_dose_sweep_signal.py` —
  per-dose batched `forward_window`.
- `apps/pulse/engine/pulse/training/trajectory_signal.py` —
  precomputes `gut_window` once per integration window.
- `apps/pulse/engine/pulse/diagnostics/signal_balance.py` (new) —
  per-signal raw loss + ‖∇gut‖ probe.
- `apps/pulse/engine/pulse/diagnostics/{__init__,__main__}.py` —
  exports + CLI subcommand.
- `apps/pulse/engine/pulse/train.py` — removed
  `--fail-on-benchmark`; runtime config print; `torch.compile` left
  out (see comment).
- `apps/pulse/scripts/train-submit.sh` — removed `fail_args`.
- `apps/pulse/train/cloudbuild.yaml` — removed `--fail-on-benchmark`;
  `ResolveBaseline` requires `checkpoint.pt` + `benchmark-report.json`;
  new `EnforceBenchmarkGate` step after `CompareToPrevious`.
- `apps/pulse/engine/Dockerfile` — `ENV PYTHONUNBUFFERED=1`.
- `apps/pulse/engine/tests/test_cohort_statistic_loss.py`,
  `test_dose_response.py` — updated for batched API.
- `apps/pulse/docs/prd.md` — gate enforcement mechanism note.
- `apps/pulse/train/spec.json` — iter 14 hyperparameters.

## Architectural / philosophical context

Three lessons compound into iter 14:

**Iter 12 → 13:** when introducing a new loss term, the per-element
scale matters as much as the weight. A signal whose residual is
normalized by 100× the typical excursion contributes the same gradient
as a signal at weight/100 — silently. Iter 13 put the scale in one
place (`GUT_OUTPUT_SCALE`).

**Iter 13 → 14:** when a per-element scale lets the loss admit a flat
trivial minimum (like the dose-averaged target shape), even a
moderate-weight signal can pull the optimizer into that minimum during
early training and overshoot. The fix is not to revert the scale (it's
physically correct) but to weaken the signal so it nudges instead of
dominates, and let indirect signals do the dose-response learning.
This is also why `signal_balance` is now a first-class diagnostic:
the post-mortem on iter 13 needed this exact view, and it'll be needed
again the next time a weight change ships.

**CI lesson:** the trainer should report; orchestration should decide.
Coupling "did this run pass the gate" to the trainer's exit code meant
that exactly the runs we most want to compare against the prior baseline
were the ones that skipped the comparison. Now `RunTraining` always
completes (assuming physical success), `CompareToPrevious` always runs,
and a separate `EnforceBenchmarkGate` step decides whether the build is
green.

## Resume protocol for the next agent

1. Read this file.
2. Confirm the run was submitted (commit log or `gcloud builds list`).
3. When it finishes, compare against the baseline:
   `./apps/pulse/train/compare-runs.sh <JOB_ID>`
   — or `gsutil cat gs://.../delta-vs-baseline.json | jq` for the in-flight
   delta uploaded by the cloudbuild step.
4. If the gate fails or probe is suspicious, run
   `python -m pulse.diagnostics signal-balance --checkpoint <ckpt>`
   to see whether any signal collapsed or escalated.
5. Apply the decision tree above to scope iter 15.
