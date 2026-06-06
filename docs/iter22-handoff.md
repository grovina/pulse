# Pulse Iter 22 — Resume Handoff

Self-contained resume point. **Read only this file to pick up the work
after a reboot.** Iters 12-21 are documented in their own
`iter<N>-handoff.md` files but you do not need to read them to proceed.

> **Resume action:** iter 22 was submitted at the time this handoff was
> written. Job ID and Build ID are in the *In-flight* section at the
> bottom. Skip straight there to check whether the build finished
> before doing anything else — do **not** re-submit before verifying.

## Headline status

**Iter 21 produced a partial structural success.** The cold-model
recalibration broke the metabolic insulin head out of its 15-25 μU/mL
saturated band, the make-or-break gate from iter 20:

| metric                              | iter 19 | iter 20 | **iter 21** | direction |
| ----------------------------------- | ------- | ------- | ----------- | --------- |
| `ogtt_75g_insulin_peak` z-residual  | -1.59   | -1.59   | **-0.87**   | gate #1 essentially met |
| `ogtt_75g_insulin_mean_3h` z-residual | -1.02 | -1.02   | **-0.21**   | improved  |
| verifier overall_score Δ            | —       | +0.110  | **+0.086**  | sustained |
| verifier meal Δ                     | —       | —       | **+0.163**  | big gain  |

**But scalar MAPE regressed and a new failure mode appeared:**

| metric                  | iter 20 | **iter 21** | gate | result |
| ----------------------- | ------- | ----------- | ---- | ------ |
| glucose mean_mape       | 0.5141  | **0.5432**  | ≤ 0.42 | ✗ |
| overall_weighted_mape   | 0.1859  | **0.1896**  | ≤ 0.17 | ✗ |
| trajectory_rollout raw_loss | 9.61 | **10.87** | drop | ✗ (rose 13%) |

The new failure mode: **small-magnitude metabolic-axis species
collapsed toward zero predictions** despite the cold model now producing
target-near values for all of them. Cohort-ablation on iter 21:

| spec | iter 20 trained | **iter 21 trained** | iter 21 cold | iter 21 z |
| ---- | --------------- | ------------------- | ------------ | --------- |
| `large_meal_glp1_peak`           | 1.75    | **0.030** | 12.16  | -2.00 |
| `ogtt_glucagon_suppression`      | -4.00   | **-0.012** | -23.23 | +2.08 |
| `mixed_meal_glucagon_suppression`| -3.69   | **-0.011** | -18.95 | +1.25 |
| `meal_hgo_suppression`           | -0.095  | **0.008** | -0.61  | +2.02 |
| `extended_fast_bhb_overnight`    | ~0      | **-0.003** | 0.06  | -2.03 |
| `meal_ffa_suppression`           | 0.000   | **-0.000** | -0.165| +2.00 |
| `extended_fast_ffa_overnight`    | 0.000   | **0.000** | 0.019 | -2.00 |
| `meal_ghrelin_suppression`       | -0.159  | **-0.042** | -67.65 | +2.00 |
| `extended_fast_insulin_basal`    | 15.40   | **20.28** | 11.36 | +3.32 |

(`extended_fast_insulin_basal` over-shot — opposite problem, same
gradient-budget root cause.)

## What's actually wrong

**The mechanism is gradient-budget starvation on the cohort signal,
not loss saturation.** The cohort loss is `z.pow(2).mean()` —
fully unbounded z². But for the collapsed species, ∂pred/∂params ≈ 0
across a wide manifold of internal states, so even with z² growing
quadratically the gradient on metabolic params is dead (||g_met|| ~
10⁻³–10⁻¹ on collapsed specs). Two compounding effects:

1. **Metabolic head shared across 7 species.** When insulin's target
   dynamics shifted dramatically (cold pred 24→54 between iters 20
   and 21), the shared MLP representation reorganized around it and
   other species' fits got perturbed. Same architectural issue iter
   16 fixed in the gut module.
2. **trajectory_rollout dominates the gradient budget.** Signal-balance
   on iter 21: cohort_statistic ||g_all||=5.9 vs trajectory_rollout
   ||g_all||=26.0 (~4.4× weaker). With harder new cold dynamics,
   trajectory's pull intensified rather than relaxed. Most of the
   metabolic head's capacity went to fitting insulin amplitude
   (||g_met|| 18-78 on insulin specs), starving the small-magnitude
   species.

The cohort signal can't fight back at its current relative strength.

## What iter 22 changes

### One conceptual intervention: amplify the cohort signal, more for collapsed species

This is mathematically two coupled knob moves because `cohort_loss`
takes a weighted average across specs (`total / weight_sum`), so a
per-spec bump alone would dilute the working specs:

**(a) Per-spec `weight=1.0 → 3.0` on 9 stuck/collapsed specs:**

`apps/pulse/engine/pulse/knowledge/cohorts/nutrition.py`:

* `EXTENDED_FAST_BHB`
* `EXTENDED_FAST_FFA`
* `EXTENDED_FAST_INSULIN` (over-shoot, basal too high)
* `OGTT_GLUCAGON_SUPPRESSION`
* `MIXED_MEAL_GLUCAGON_SUPPRESSION`
* `MEAL_FFA_SUPPRESSION`
* `LARGE_MEAL_GLP1`
* `MEAL_GHRELIN_SUPPRESSION`

`apps/pulse/engine/pulse/knowledge/cohorts/glucose_handling.py`:

* `MEAL_HGO_SUPPRESSION`

The other 8 specs stay at the default `weight=1.0` — these are the
glucose / insulin specs that already work (peak, 120-min, mean_3h,
small_carb, fasting_breakfast) plus `extended_fast_glucose_morning`
and `sleep_restriction_next_day_glucose` (both glucose, partially
collapsed but glucose has plenty of trajectory data; let trajectory
keep doing its job there).

**(b) Global `--cohort-statistic-weight 0.15 → 0.40`** in
`apps/pulse/train/spec.json` to compensate for the weighted-average
dilution that (a) would otherwise inflict on working specs.

### Net effect on per-spec gradient pressure

| group | iter 21 cohort weight per spec | iter 22 cohort weight per spec | ratio |
| ----- | ------------------------------ | ------------------------------ | ----- |
| collapsed (×9) | 0.15 × 1/17 = 0.0088   | 0.40 × 3/35 = 0.0343           | **3.89×** |
| working (×8)   | 0.15 × 1/17 = 0.0088   | 0.40 × 1/35 = 0.0114           | **1.30×** |

So collapsed specs get ~4× more cohort pull, working specs get a
mild boost (so insulin_peak which is at z=-0.87 nudges further
toward zero), and no spec loses weight.

### Training pipeline: nothing else changes

Strictly two knobs, but conceptually one intervention — the math
*requires* both to move together. Every other parameter (trajectory
weights, sample sizes, LRs, epochs, all module architectures, all
coupling priors, all PatientParams) stays identical to iter 21. So
any movement is attributable to "amplified cohort pull on collapsed
species".

## What iter 22 does NOT change

* `PatientParams` — the iter 21 cold-model recalibration is locked
  in. Cold-calibration diagnostic still shows misleading-spec
  count = 2 (ghrelin, fasting_breakfast_glucose).
* Trajectory pipeline (windows, bands, contributions, default
  patients, landmark, gut sweep, insulin sweep) — all identical.
* Module architectures — metabolic head still the 48-dim shared MLP
  emitting 7 rates. Iter 23 may need to factorize this; iter 22 first
  tests whether the gradient-budget contest alone is the issue.
* Training-pipeline knobs other than `--cohort-statistic-weight`.

If iter 22 recovers the collapsed species without regressing the
gut/glucose specs that work, the iter 23 plan changes from "factorize
the metabolic head" to "tackle the remaining ghrelin + fasting-glucose
cold-model miscalibrations + dial back if any over-pull was observed".

## Code state (iter 22 changes ready to commit)

Modified:

* `apps/pulse/engine/pulse/knowledge/cohorts/nutrition.py` — added
  `weight=3.0` (with iter 22 inline rationale) to 8 specs:
  `EXTENDED_FAST_BHB`, `EXTENDED_FAST_FFA`, `EXTENDED_FAST_INSULIN`,
  `OGTT_GLUCAGON_SUPPRESSION`, `MIXED_MEAL_GLUCAGON_SUPPRESSION`,
  `MEAL_FFA_SUPPRESSION`, `LARGE_MEAL_GLP1`, `MEAL_GHRELIN_SUPPRESSION`.
* `apps/pulse/engine/pulse/knowledge/cohorts/glucose_handling.py`
  — added `weight=3.0` to `MEAL_HGO_SUPPRESSION`.
* `apps/pulse/train/spec.json` — `--cohort-statistic-weight=0.15` →
  `0.40`; iter 22 hypothesis + expectedEffect rewritten.

No new files. No new diagnostics. No new tests required (the per-spec
`weight` field already exists in `CohortStatisticSpec`; no schema
change). All 130 engine tests pass.

## In-flight: iter 22 was submitted

* Job ID:   `train-20260424T071900Z`
* Build ID: `525c31cd-41f2-4380-839c-ff92448faebf`
* Submitted: 2026-04-24T07:19:00Z
* Expected finish: ~5-6h after submit (matches iter 19-21 wall clock);
  benchmark report and delta-vs-baseline land last.

The Cloud Build *will* fail at the final `EnforceBenchmarkGate` step
because we haven't cleared `overall_mape ≤ 0.16` yet — same expected
failure pattern as iters 17-21. Training, benchmark, comparison, and
artifact upload all complete normally; the gate step is a separate
post-training check. Treat the build's FAILURE status as *expected*
and look at the artifacts directly.

## How to resume after reboot

```bash
cd /Users/grovina/Projects/grovina/platform
gcloud auth list
ls apps/pulse/engine/.venv/bin/python

# Check whether iter 22 has finished (artifacts land last)
gsutil ls -l gs://grovina-pulse/training/jobs/train-20260424T071900Z/

# When checkpoint.pt + benchmark-report.json + delta-vs-baseline.json
# are all present, pull them. Use parallel_process_count=1 because
# gsutil multiprocessing on macOS is broken (see iter 21 handoff).
mkdir -p /tmp/iter22 && \
  gsutil -o "GSUtil:parallel_process_count=1" cp \
    gs://grovina-pulse/training/jobs/train-20260424T071900Z/{benchmark-report.json,delta-vs-baseline.json,checkpoint.pt,spec.json} \
    /tmp/iter22/

# And iter 21 for compare baselines
mkdir -p /tmp/iter21 && \
  gsutil -o "GSUtil:parallel_process_count=1" cp \
    gs://grovina-pulse/training/jobs/train-20260423T175447Z/{benchmark-report.json,delta-vs-baseline.json,checkpoint.pt} \
    /tmp/iter21/

# Tests should still pass identically
cd apps/pulse/engine && .venv/bin/python -m pytest tests/ -x -q

# Diagnostic suite on iter 22 checkpoint
cd apps/pulse/engine && .venv/bin/python -m pulse.diagnostics cold-calibration
cd apps/pulse/engine && .venv/bin/python -m pulse.diagnostics signal-balance \
  --checkpoint /tmp/iter22/checkpoint.pt --n-patients 20
cd apps/pulse/engine && .venv/bin/python -m pulse.diagnostics cohort-ablation \
  --checkpoint /tmp/iter22/checkpoint.pt --sample-patients 4

# Headline check: did the gates move?
.venv/bin/python -c "
import json
r21 = json.load(open('/tmp/iter21/benchmark-report.json'))
r22 = json.load(open('/tmp/iter22/benchmark-report.json'))
print(f'overall:  iter21={r21[\"overall_weighted_mape\"]:.4f}  iter22={r22[\"overall_weighted_mape\"]:.4f}')
print(f'glucose:  iter21={r21[\"per_marker\"][\"glucose\"][\"mean_mape\"]:.4f}  iter22={r22[\"per_marker\"][\"glucose\"][\"mean_mape\"]:.4f}')
"
```

## Cloud Build / job IDs

| iter | job ID | build status when noted | benchmark-report key metrics |
| ---- | ------ | ----------------------- | ---------------------------- |
| 17   | train-20260422T065354Z | FAILURE (gate) | overall=0.198, glu_mean=0.579 |
| 18   | train-20260422T140536Z | FAILURE (gate) | overall=0.197, glu_mean=0.575 |
| 19   | train-20260422T180605Z | FAILURE (gate) | overall=0.198, glu_mean=0.578 |
| 20   | train-20260423T122419Z | FAILURE (gate) | overall=0.186, glu_mean=0.514 |
| 21   | train-20260423T175447Z | FAILURE (gate) | **overall=0.190, glu_mean=0.543** (insulin_peak +18 μU/mL structural win) |
| 22   | train-20260424T071900Z | WORKING at handoff time | (pending) |

## Success criteria for iter 22

Hard gates:

1. **`large_meal_glp1_peak` predicted Δ ≥ 5 pmol/L** (iter 21: 0.03;
   cold model produces 12.16 — model just needs to track cold).
2. **`ogtt_glucagon_suppression` predicted Δ ≤ -10 pg/mL** (iter 21:
   -0.012; cold produces -23.23).
3. **`extended_fast_bhb_overnight` predicted ≥ 0.10 mmol/L** (iter
   21: -0.003; cold produces 0.06 so this also tests whether the
   training data's BHB visibility is sufficient).
4. **`extended_fast_insulin_basal` predicted ≤ 14 μU/mL** (iter 21:
   20; target 7).
5. **`ogtt_75g_insulin_peak` z-residual stays ≤ |1.0|** (iter 21:
   -0.87 — must hold or improve; the iter-21 structural win must
   not regress).
6. **No regression in gut probe metrics:** ratio stays ≤ 1.20× on
   every dose ≥ 30g; per-gram slope ≤ 3.5; AUC sub-loss < 0.10.
7. **Cold-calibration diagnostic on iter 22 PatientParams shows
   misleading-spec count ≤ 2.** (PatientParams unchanged from iter
   21, so this should automatically hold; assert as a guard.)

Soft gates:

8. **glucose mean_mape ≤ 0.50** (iter 21: 0.543, iter 20: 0.514).
   Recovery from the iter 21 regression.
9. **overall_weighted_mape ≤ 0.18** (iter 21: 0.190).
10. **verifier overall_score holds** (iter 21: +0.086; iter 22 must
    stay within -0.02 of that).
11. **trajectory_rollout raw_loss visibly drops** (iter 21: 10.87;
    rising trajectory loss meant the model wasn't tracking the new
    cold dynamics — expect partial drop as the cohort signal anchors
    species amplitudes that trajectory was missing).

## Risks / what to look for if iter 22 doesn't recover the collapsed species

| symptom | likely cause | next move |
| ------- | ------------ | --------- |
| Some collapsed specs recover (e.g. GLP-1, glucagon) but others stay at 0 (e.g. BHB, FFA) | the recovered ones are visible in 240-min training windows; the others (BHB esp.) need long-fast contribution data | **Iter 23**: extend `TRAIN_WINDOW` for a fraction of windows, OR add a long-fast knowledge contribution that visits 16-24h fasted states. |
| All collapsed specs recover but glucose mean_mape regresses further | cohort signal over-pulled past trajectory's glucose anchor | **Iter 23**: dial global cohort weight back to 0.30 (intermediate between 0.15 and 0.40); per-spec weights stay. |
| `extended_fast_insulin_basal` drops (good) but `ogtt_75g_insulin_peak` also drops back below iter 21's 38 | shared metabolic head can't simultaneously satisfy basal-low and peak-high | **Iter 23**: factorize the metabolic insulin head — sigmoid-on-glucose × scale separates basal from peak gain. The iter 16 architectural surgery applied to metabolic. |
| Verifier meal coherence regresses below iter 21 (+0.16) | amplified cohort pull on small species broke joint dynamics that gave iter 21 its coherence win | **Iter 23**: re-tune `coupling_priors/metabolism.py` magnitude bands on the species we just amplified (looser bands so the amplified cohort signal doesn't conflict with coupling constraints). |
| Nothing moves: collapsed specs stay at 0 AND glucose stays bad | gradient-budget contest is not the actual blocker → representation collapse is structural | **Iter 23**: factorize the metabolic head per-species (the iter 16 gut-fix analog). This is the architectural surgery the iter-21 risks table flagged. |
| trajectory_rollout raw_loss rises further | amplified cohort signal disrupted trajectory tracking | **Iter 23**: anneal `--cohort-statistic-weight` (start at 0.15 in phase 1, ramp to 0.40 in phase 2) so the model establishes trajectory anchoring before the cohort-pull intensifies. |

## Pointer references (only read if needed)

* `apps/pulse/docs/iter21-handoff.md` — five-PatientParams cold-model
  recalibration details, the cold-calibration diagnostic spec, and
  the iter 21 in-flight artifact paths. Read for the iter 21
  baseline state and the per-spec cohort-ablation table.
* `apps/pulse/docs/iter20-handoff.md` — five-pronged metabolic-axis
  intervention. Read for the InsulinSweepSignal / cohort-ablation
  / coupling-priors-registry / InitMode wiring.
* `apps/pulse/docs/iter16-handoff.md` — factorized kernel structural
  guarantees. **Read this if iter 22 fails the small-species recovery
  gates** — the iter 23 fix will mirror this iter's architectural
  surgery applied to the metabolic head.
* `apps/pulse/docs/iter18-handoff.md` — full justification of the AUC
  term + dilution diagnosis. Read only if iter 22 reveals a new
  dilution problem on some signal.
* `apps/pulse/engine/pulse/knowledge/AUTHORING.md` — guide for
  encoding new medical knowledge (cohort specs, coupling priors,
  scenarios, sweep signals).
* `apps/pulse/docs/prd.md` — top-level vision. Read for context
  but not required for iter 22 mechanics.
