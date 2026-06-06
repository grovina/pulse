# Pulse Iter 13 — Resume Handoff

Self-contained resume point. Read only this file to pick up the work.

## Where we are

**Iter 13 is staged.** Code is in working tree; the Cloud Build submission is
the next action.

- Spec: `apps/pulse/train/spec.json` (committed at HEAD once submitted).
- Cloud Build config: `apps/pulse/train/cloudbuild.yaml`.
- Submission: `bash apps/pulse/scripts/train-submit.sh --gcp` from repo root.
- Job ID will be assigned at submit time. Artifact root:
  `gs://grovina-pulse/training/jobs/<JOB_ID>/`.

When the run finishes, four artifacts land under that root:

- `checkpoint.pt`, `benchmark-report.json`, `spec.json`, `meta.json`
- **new in iter 13:** `delta-vs-baseline.json` (output of the
  `CompareToPrevious` cloudbuild step — see below).

## What iter 12 missed and iter 13 fixes

Iter 12 introduced the right two architectural changes (`GutDoseSweepSignal`
at the zero embedding + `trajectory_band_default=0.05`), but the gut-related
loss landed in the wrong magnitude regime: both signals divided gut-output
residuals by `abs_scale=[30,10,15,1]` — a scale inherited from blood-state
mg/dL — when actual cold-model gut appearance peaks are about
`[2.0, 0.3, 0.5, 1.0]` mg/min (per channel: glucose, lipid, amino,
nutrient_flag). Per-channel gut MSE was flattened ~225×, ~1100×, ~900× — so
even with `weight=0.30` the gut signal contributed negligible gradient and
the kernel barely moved.

Iter 12 vs iter 11 probe deltas (zero embedding, run on local checkpoints):

| metric                              | iter 11 | iter 12 | Δ      |
| ----------------------------------- | ------- | ------- | ------ |
| AUC at 0 g carbs                    | 122.07  | 134.99  | +12.92 |
| sweep monotone in dose              | False   | False   | —      |
| fasting-drift over 4 h              | +68.7   | +55.6   | -13.1  |
| counter-reg glucagon Δ              | -0.022  | +0.004  | +0.026 |

Iter 13 fixes the scale at the source:

- `pulse.modules.gut.GUT_OUTPUT_SCALE = (2.0, 0.3, 0.5, 1.0)` — single
  source of truth.
- `TrajectoryRolloutSignal._abs_scale` and `GutDoseSweepSignal._abs_scale`
  both import it.
- Weights drop in proportion: `--gut-dose-sweep-weight 0.30 → 0.10`,
  `--gut-loss-weight 0.5 → 0.1`. Net effect: ~25× stronger gut gradient
  in aggregate than iter 12, comfortably bounded.

## Compute changes (free win)

A 4-epoch × 8-patient profile of iter-12-spec training showed the time
breakdown is overwhelmingly dominated by `cohort_statistic`:

| signal              | mean ms / epoch | % of wall  |
| ------------------- | --------------: | ---------: |
| trajectory_rollout  |          12 654 |       6.4% |
| **cohort_statistic**|     **179 491** |  **91.3%** |
| dose_response       |           4 473 |       2.3% |
| gut_dose_sweep      |              14 |     0.007% |

Iter 13 does the one cheap change `cohort_signal` allows: cold-model
initial states are now precomputed once at signal construction (in
`__post_init__`) instead of recomputed every epoch. The closure that used
to wrap an epoch-local cache now reads from the precomputed dict.
Expected ≤5% wall-clock win — small relative to the architecture-level
opportunity.

The bigger compute opportunity (vectorize `integrate` across embeddings
inside cohort signal — currently ~110 sequential `integrate` calls per
epoch) is a separate iteration: it changes the optimization regime
(per-epoch loss landscape changes, LR re-tuning required) and shouldn't
be coupled with the abs_scale fix.

Cloud Build / Cloud Run timeouts bumped to absorb the iter 12 8-hour
overrun: `--task-timeout 28800 → 36000` (10 h), top-level `timeout
32400 → 39600` (11 h).

## Diagnostics-as-code (new in iter 13)

The probe and the run-vs-run comparator are now first-class modules,
not throwaway scripts.

- `apps/pulse/engine/pulse/diagnostics/probe.py` — gut sweep, fasting
  drift, counter-regulation; same checks the iter 11 → iter 12 →
  iter 13 chain has been using by hand. Importable, JSON-serializable,
  used both by the CLI and by `apps/pulse/engine/tests/test_diagnostics_probe.py`.
- `apps/pulse/engine/pulse/diagnostics/compare.py` — diff two runs
  (gate, scalars, per-marker MAPE, scenario pass-rates, per-check
  transitions, side-by-side probe).
- `apps/pulse/engine/pulse/diagnostics/__main__.py` — CLI. Accepts both
  local paths and `gs://` URIs (downloads via google-cloud-storage).
- `apps/pulse/train/compare-runs.sh` — convenience wrapper that resolves
  the most recent prior baseline job from GCS and invokes the CLI.
- `apps/pulse/train/cloudbuild.yaml` — added `ResolveBaseline` (cloud-sdk)
  and `CompareToPrevious` (engine image) steps after `RunTraining`.
  Every run now uploads `delta-vs-baseline.json` next to its checkpoint.

## How to check the run

```bash
./apps/pulse/train/get-results.sh grovina-pulse <JOB_ID>

gsutil cat gs://grovina-pulse/training/jobs/<JOB_ID>/delta-vs-baseline.json | jq '.'

./apps/pulse/train/compare-runs.sh <JOB_ID>

cd apps/pulse/engine && .venv/bin/python -m pulse.diagnostics probe \
  --checkpoint gs://grovina-pulse/training/jobs/<JOB_ID>/checkpoint.pt
```

`compare-runs.sh` auto-resolves the previous job; pass an explicit
baseline job ID as the second arg to override.

## Success criteria

- **Gut sweep at zero embedding becomes monotone in carb dose.** AUC(120 g)
  > AUC(60 g) > AUC(0 g) ≈ 0.
- **Glucose-appearance peak time** moves from end-of-window (~239 min in
  iter 12) toward the cold-model peak (~30-50 min) for nonzero doses.
- **Fasting drift over 4 h** continues to fall (iter 11: +69, iter 12: +56).
- **Counter-regulation deltas** (glucagon, FFA, ghrelin) become clearly
  negative at the zero embedding under a 75 g mixed meal.
- **Hard benchmark `meal_dose_response.glucose_dose_response`** slope flips
  positive (iter 11/12: -29). `dietary_carbohydrate_meal_flow.{glucagon,
  ffa, ghrelin}_suppressed_*` move from near-zero to negative.
- **Stable or better:** overall weighted MAPE, glucose MAPE, verifier
  overall and meal category.

## Decision tree for iter 14

1. **Gut monotone + dose response positive + drift small.** Move on to the
   pending Phase 3 cohort specs (counter-regulatory: see Pending work).
2. **Gut monotone but dose response flat.** Drift is dominating the
   integrated curve. Tighten `--trajectory-band-default` further (try
   0.0) or extend `GutDoseSweepProtocol` to longer 0 g windows.
3. **Gut still partially inverted.** The new abs_scale wasn't aggressive
   enough relative to other signals. Crank `--gut-dose-sweep-weight`
   (0.10 → 0.25) before touching the kernel design.
4. **Gut monotone but training too slow / timing out.** Ship the
   patient-vectorized `integrate` rewrite (its own iter; see "Future
   perf work").

## Pending work from the broader plan

These were queued behind the gut/drift fix because that fix is upstream
of all of them.

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

### Future perf work (separate iterations)

- **Vectorize `integrate` across the embedding/patient dim inside
  `cohort_signal`.** Switches the cohort regime from sequential to
  batched; needs LR re-tuning. Largest single throughput lever (cohort
  is 91% of training wall-clock).
- **Cloud Run → Vertex AI on `c3-highcpu-22` with multi-thread.**
  Cloud Run pins to ~6 effective cores; Vertex on c3-highcpu-22 unlocks
  22. Combined with the existing opt-in `--deterministic` flag, an
  expected 2-3× wall-clock improvement.

## Key files (iter 13 changes)

- `apps/pulse/engine/pulse/modules/gut.py` — `GUT_OUTPUT_SCALE` constant.
- `apps/pulse/engine/pulse/training/trajectory_signal.py` — imports
  `GUT_OUTPUT_SCALE`.
- `apps/pulse/engine/pulse/training/gut_dose_sweep_signal.py` — imports
  `GUT_OUTPUT_SCALE`.
- `apps/pulse/engine/pulse/training/cohort_signal.py` — `__post_init__`
  precomputes cold-model initial states once.
- `apps/pulse/engine/pulse/diagnostics/{__init__,probe,compare,__main__}.py`
  — new diagnostics module.
- `apps/pulse/engine/tests/test_diagnostics_probe.py` — new tests.
- `apps/pulse/engine/tests/test_gut_dose_sweep_signal.py` — `TestAbsScale`
  pinning the per-channel scale invariant.
- `apps/pulse/train/compare-runs.sh` — new local wrapper.
- `apps/pulse/train/cloudbuild.yaml` — `ResolveBaseline` +
  `CompareToPrevious` steps; bumped timeouts.
- `apps/pulse/train/spec.json` — iter 13 hyperparameters.

## Architectural / philosophical context

The supervision pattern matters: the gut module is a *stateless absorption
kernel*. Its correctness is independent of any downstream ODE, so it is
supervised independently — that is what `GutDoseSweepSignal` does, and
why it succeeds where the integrated-state signals could not. When adding
new training signals, look for the same separability: if a module's
correctness is well-defined without integration, supervise it without
integration.

The asymmetric-band insight (`trajectory_band` vs `trajectory_band_default`)
is a more general one: sampled-patient embeddings benefit from slack
(per-patient variation should not be penalized as deviation from the cold
model), but the zero embedding has no patient identity and the benchmark
queries it directly — it should be held to a tighter standard.

The iter 12 → iter 13 lesson is a third one: when introducing a new
loss term, the per-element scale matters as much as the weight.
A signal whose residual is normalized by 100× the typical excursion
contributes the same gradient as a signal with weight/100 — silently.
The iter 13 fix puts the scale in one place (`GUT_OUTPUT_SCALE`) so
this class of mistake can't drift between signals again.

## Resume protocol for the next agent

1. Read this file.
2. Confirm the run was submitted (commit log or `gcloud builds list`).
3. When it finishes, compare against the baseline:
   `./apps/pulse/train/compare-runs.sh <JOB_ID>`
   — or `gsutil cat gs://.../delta-vs-baseline.json | jq` for the in-flight
   delta uploaded by the cloudbuild step.
4. Apply the decision tree above to scope iter 14.
