# Pulse Iter 21 — Resume Handoff

Self-contained resume point. **Read only this file to pick up the work
after a reboot.** Iters 12-20 are documented in their own
`iter<N>-handoff.md` files but you do not need to read them to proceed.

> **Resume action:** iter 21 was already submitted before this handoff
> was written. Build ID `7713d58c-17f1-420a-b8a7-baada3ebefb5`,
> job `train-20260423T175447Z`, started 2026-04-23T17:56:05Z. Skip
> straight to *In-flight* and *How to resume after reboot* below to
> pick up where we left off — do **not** re-submit before verifying
> whether the build finished.

## Headline status

**Iter 20 produced the first real glucose movement since iter 12** —
the five-pronged metabolic-axis intervention (`InsulinSweepSignal` +
6 new glucose-handling cohort specs + `cohort_sample_patients` 4→16
+ `InitMode` + coupling-priors registry + cohort-ablation diagnostic)
moved the needle:

| metric                  | iter 18 | iter 19 | **iter 20** | Δ vs iter 19 |
| ----------------------- | ------- | ------- | ----------- | ------------ |
| glucose mean_mape       | 0.5752  | 0.5778  | **0.5141**  | **−0.064 (−11%)** |
| overall_weighted_mape   | 0.1972  | 0.1980  | **0.1859**  | −0.012      |
| verifier coupling Δ     | -0.086  | +0.025  | **+0.110**  | +0.085      |

Cohort-ablation on iter 20's checkpoint confirmed the new literature
contributions were doing real work:

| spec | iter 19 z | **iter 20 z** | direction |
| ---- | --------- | ------------- | --------- |
| `small_carb_glucose_peak` | 5.03 | **3.15** | -1.88, recovering |
| `ogtt_75g_glucose_120min` | 2.79 | **1.87** | -0.92, recovering |
| `ogtt_75g_glucose_peak`   | 0.48 | **-0.07** | satisfied |

But **two specs stayed structurally stuck**:

| spec | predicted | target | z | symptom |
| ---- | --------- | ------ | -- | ------- |
| `ogtt_75g_insulin_peak` | 20 μU/mL | 60 | **-1.59** | model can't escape ~15-25 μU/mL band |
| `extended_fast_bhb_overnight` | ~0 | 0.2 | -2.0 | ketogenesis pathway flat |
| `extended_fast_insulin_basal` | 15 | 7 | **+2.10** | fasting insulin too high |

Initial diagnosis was "insulin head saturated, bump
`InsulinSweepSignal` weight". The deeper investigation revealed the
*actual* root cause: **the cold model itself is miscalibrated against
the literature targets**. trajectory_rollout (built from cold-model
trajectories) was actively pulling the trained model *away* from
literature truth, and trajectory pull is 10-100× larger than cohort
pull on any single module — so the literature signal lost every
gradient-budget contest no matter how heavy we made the per-spec
weight.

**Iter 21's job: recalibrate the cold model against the literature
targets. One-knob iteration — every training-pipeline parameter stays
identical so attribution is clean.** This is also a permanent fix for
a class of silent failure: a new `cold-calibration` diagnostic now
runs every cohort spec against the cold model and flags any spec the
cold model can't reach, so this miscalibration can never silently land
again.

## The cold-calibration finding

`pulse.diagnostics cold-calibration` (new in iter 21) runs every
cohort spec through the hand-tuned `cold_model_trajectory` (i.e.
`simulate_full_body` with default `PatientParams`) and reports the
z-residual the cold model alone produces against each literature
target. On iter 20's `PatientParams`:

```
cold-calibration: 17 specs, 4 misleading (|z| > 2)
  large_meal_glp1_peak                glp1     cold=40.5  target=12   sigma=6   z=+4.76 *
  fasting_breakfast_glucose_morning   glucose  cold=-32.8 target=-12  sigma=6   z=-3.46 *
  meal_ghrelin_suppression            ghrelin  cold=-62.1 target=-30  sigma=15  z=-2.14 *
  ogtt_75g_glucose_120min             glucose  cold=150.6 target=120  sigma=15  z=+2.04 *
  extended_fast_bhb_overnight         bhb      cold=0.001 target=0.2  sigma=0.1 z=-1.99
  extended_fast_ffa_overnight         ffa      cold=0.004 target=0.4  sigma=0.2 z=-1.98
  meal_ffa_suppression                ffa      cold=-0.05 target=-0.3 sigma=0.15 z=+1.65
  ogtt_75g_insulin_peak               insulin  cold=24.6  target=60   sigma=25  z=-1.42
  extended_fast_insulin_basal         insulin  cold=12.4  target=7    sigma=4   z=+1.35
  ...
```

Four specs (`*`) the cold model is *actively misleading* on. Six more
borderline. The trained model's iter-20 stuck-points line up exactly
with the cold-model miscalibration:

* `ogtt_75g_insulin_peak` trained=20, cold=24, target=60 → trained
  model converged to the cold target (which is 2.5× under the
  literature target).
* `extended_fast_bhb_overnight` trained=0, cold=0.001, target=0.2 →
  same story; trajectory data has BHB ≈ 0 throughout, so the model
  learned to keep it there.
* `extended_fast_insulin_basal` trained=15, cold=12, target=7 → cold
  model has insulin *rising* during fasting (because GSIR fires at
  G=95 > h=80), pulling the trained model into a similar pattern.

This is the silent-miscalibration failure mode the diagnostic now
catches.

## What iter 21 changes

### Cold-model recalibration (5 PatientParams adjustments)

`apps/pulse/engine/pulse/knowledge/full_body.py`:

| param | iter 20 | **iter 21** | rationale |
| ----- | ------- | ----------- | --------- |
| `gamma` | 0.015 | **0.07** | OGTT insulin peak with this γ matches literature target ~60 μU/mL (was 24) |
| `h` | 80.0 | **95.0** | GSIR threshold = basal glucose, so GSIR is zero during fasting → insulin can drop below Ib (was pinned at 12.4) |
| `k_bhb` | 0.03 | **0.005** | BHB clearance slow enough for ketogenesis to accumulate to ~0.25 mmol/L overnight (was 0.001) |
| `IC50_keto` | 10.0 | **15.0** | less insulin suppression of ketogenesis at fasting insulin levels |
| `glp1_meal_gain` | 5.0 | **1.5** | large-meal GLP-1 peak drops from 64 to 22 (target 22) — fixes the worst miscalibration (z=+4.76) |
| `Si` | 0.0002 | **0.0004** | tighter late-glucose clearance — OGTT 120-min drops from 150 to 132 (target 120) |

After recalibration, the cold-calibration diagnostic shows
**misleading specs collapse 4 → 2** (remaining: ghrelin and fasting
glucose drop, both now |z| in 2.3-2.5 range — borderline, candidates
for iter 22).

Major wins:

| spec | iter 20 cold z | **iter 21 cold z** |
| ---- | -------------- | ------------------ |
| `large_meal_glp1_peak` | +4.76 | **+0.03** |
| `ogtt_75g_glucose_120min` | +2.04 | **+0.82** |
| `ogtt_glucagon_suppression` | +1.46 | **+0.15** |
| `ogtt_75g_insulin_peak` | -1.42 | **-0.26** |
| `ogtt_75g_insulin_mean_3h` | -0.89 | **+0.32** |
| `extended_fast_bhb_overnight` | -1.99 | **-1.41** |

### New diagnostic — `cold-calibration`

`pulse.diagnostics cold-calibration` (with optional
`--misleading-z=<threshold>` and `--out=<gs://...>`). Permanently in
the toolchain. Runs every cohort spec through the cold model and
reports z-residual + a misleading flag.

Use this:

* Before any iteration that introduces new cohort specs (verify each
  is reachable from the cold model).
* After any change to `PatientParams` (verify nothing silently
  regressed).
* When a cohort spec stays stuck despite weight bumps (if the cold
  model itself doesn't produce the target, no amount of training
  signal weight will fix it — fix the cold model).

`apps/pulse/engine/pulse/diagnostics/cold_calibration.py` plus tests
in `tests/test_cold_calibration.py` (5 cases).

### Training pipeline: zero changes

This is intentional. iter 21 is a **one-knob iteration** —
recalibrating the cold model is the only variable. Every other
parameter (signal weights, sample sizes, learning rates, epochs)
stays identical to iter 20 so we can attribute any movement (or
non-movement) directly to the cold-model recalibration.

The recalibration cascades naturally through:

* `trajectory_rollout` — built from `simulate_full_body(...)`, now
  produces correct insulin amplitude, fasting BHB rise, GLP-1 peak.
* `InsulinSweepSignal` — `_cold_metabolic_rates(glucose, insulin,
  params)` uses `params.gamma`, `params.h`, etc.; cold targets now
  match the literature.
* `cohort_statistic_signal` — for `InitMode.COLD` specs, the initial
  state is now correct.

So all three signals on the metabolic axis pull the same direction.
The compensatory equilibrium that pinned insulin at 20 μU/mL and BHB
at 0 should dissolve because there's no longer an opposing pull from
miscalibrated trajectory data.

## What iter 21 does NOT change

* Training-pipeline parameters (signal weights, sample sizes,
  learning rates, epochs) — all identical to iter 20.
* Module architectures (`pulse/modules/{base,metabolic,gut,...}.py`).
* Cohort spec definitions (`pulse/knowledge/cohorts/*.py`).
* Coupling priors registry — still the iter 20 12+8 standard edges.
* Diagnostic CLI subcommands except the new `cold-calibration`.

If iter 21's recalibration alone moves insulin and BHB to within
spec, the iter 22 plan changes from "factorize the metabolic insulin
head" to "tackle the remaining ghrelin + fasting-glucose
miscalibrations + extend cold-calibration to vital-sign pillars".

## Code state (iter 21 changes ready to commit)

New files:

* `apps/pulse/engine/pulse/diagnostics/cold_calibration.py`
* `apps/pulse/engine/tests/test_cold_calibration.py`

Modified:

* `apps/pulse/engine/pulse/knowledge/full_body.py` — six
  `PatientParams` defaults updated: `gamma 0.015→0.07`,
  `h 80.0→95.0`, `k_bhb 0.03→0.005`, `IC50_keto 10.0→15.0`,
  `glp1_meal_gain 5.0→1.5`, `Si 0.0002→0.0004`. Inline comments
  reference iter 21.
* `apps/pulse/engine/pulse/diagnostics/__init__.py` — exports
  `cold_calibration`, `ColdCalibrationReport`, `ColdCalibrationRow`.
* `apps/pulse/engine/pulse/diagnostics/__main__.py` — adds
  `cold-calibration` subcommand.
* `apps/pulse/train/spec.json` — iter 21 hypothesis + expectedEffect
  describe the one-knob recalibration; trainArgs unchanged from iter
  20.

Unchanged from iter 20 and depended on:

* `apps/pulse/engine/pulse/training/insulin_sweep_signal.py` — the
  cold-target function reads from `params: PatientParams`, so it
  auto-picks up the new calibration.
* `apps/pulse/engine/pulse/training/cohort_signal.py` —
  `InitMode.COLD` initial states auto-pick up the new calibration.
* `apps/pulse/engine/pulse/training/trajectory_signal.py` — uses
  `cold_model_trajectory`, auto-updated.
* `apps/pulse/engine/pulse/diagnostics/cohort_ablation.py` — works
  unchanged.

All 130 engine tests pass (125 + 5 new for cold-calibration).

## In-flight: iter 21 was already submitted

**Iter 21 is RUNNING in Cloud Build at the moment this handoff was
written.** Do not re-submit until you confirm whether it has
completed. Status is in the table at the bottom of this section.

* Job ID:   `train-20260423T175447Z`
* Build ID: `7713d58c-17f1-420a-b8a7-baada3ebefb5`
* Submitted: 2026-04-23T17:56:05Z
* Expected finish: ~5h after submit (matches iter 19 / iter 20 wall
  clock); benchmark report and delta-vs-baseline land last.

The Cloud Build *will* fail at the final `EnforceBenchmarkGate` step
because we haven't cleared the `overall_mape ≤ 0.16` gate yet — that
is the same expected failure pattern we saw in iters 17–20. Training,
benchmark, comparison, and artifact upload all complete normally; the
gate step is a separate post-training check. So treat the build's
FAILURE status as *expected* and look at the artifacts directly.

## How to resume after reboot

```bash
cd /Users/grovina/Projects/grovina/platform
gcloud auth list
ls apps/pulse/engine/.venv/bin/python

# Check whether iter 21 has finished (artifacts land last)
gsutil ls -l gs://grovina-pulse/training/jobs/train-20260423T175447Z/

# When checkpoint.pt + benchmark-report.json + delta-vs-baseline.json
# are all present, pull them
mkdir -p /tmp/iter21 && \
  gsutil cp gs://grovina-pulse/training/jobs/train-20260423T175447Z/{benchmark-report.json,delta-vs-baseline.json,checkpoint.pt} /tmp/iter21/

# And iter 20 for compare baselines
mkdir -p /tmp/iter20 && \
  gsutil cp gs://grovina-pulse/training/jobs/train-20260423T122419Z/{benchmark-report.json,delta-vs-baseline.json,checkpoint.pt} /tmp/iter20/

# Tests should still pass identically
cd apps/pulse/engine && .venv/bin/python -m pytest tests/ -x -q

# Diagnostic runs against the iter 21 checkpoint
cd apps/pulse/engine && .venv/bin/python -m pulse.diagnostics cold-calibration
cd apps/pulse/engine && .venv/bin/python -m pulse.diagnostics signal-balance \
  --checkpoint /tmp/iter21/checkpoint.pt --n-patients 20
cd apps/pulse/engine && .venv/bin/python -m pulse.diagnostics cohort-ablation \
  --checkpoint /tmp/iter21/checkpoint.pt --sample-patients 4

# Headline check: did we hit the make-or-break gates?
.venv/bin/python -c "
import json
r20 = json.load(open('/tmp/iter20/benchmark-report.json'))
r21 = json.load(open('/tmp/iter21/benchmark-report.json'))
print(f'overall:  iter20={r20[\"overall_weighted_mape\"]:.4f}  iter21={r21[\"overall_weighted_mape\"]:.4f}')
print(f'glucose:  iter20={r20[\"per_marker\"][\"glucose\"][\"mean_mape\"]:.4f}  iter21={r21[\"per_marker\"][\"glucose\"][\"mean_mape\"]:.4f}')
"
```

## Useful trivia learned this iter

**Training is bit-deterministic across re-runs of the same spec.**
The first iter-21 submit (`train-20260423T122419Z`, dated
12:24Z) was actually a re-submit of the iter 20 spec (the iter 21
edits hadn't landed on disk yet at submit time). Its metrics were
*byte-identical* to the original iter 20 run we'd analyzed earlier
in the day. So don't trust an unfamiliar job ID's metrics until you
verify its `spec.json`. The actual iter 21 run is
`train-20260423T175447Z` (above).

## Cloud Build / job IDs

| iter | job ID | build status when noted | benchmark-report key metrics |
| ---- | ------ | ----------------------- | ---------------------------- |
| 17   | train-20260422T065354Z | FAILURE (gate) | overall=0.198, glu_mean=0.579 |
| 18   | train-20260422T140536Z | FAILURE (gate) | overall=0.197, glu_mean=0.575 |
| 19   | train-20260422T180605Z | FAILURE (gate) | overall=0.198, glu_mean=0.578 |
| 20   | train-20260423T122419Z | FAILURE (gate) | **overall=0.186, glu_mean=0.514** |
| 21   | train-20260423T175447Z | **WORKING** at handoff time | (pending) |

(For iter 20 there is also an earlier job whose ID we no longer have;
its metrics matched 122419Z exactly thanks to the determinism noted
above, so the table entry above is canonical.)

## Success criteria for iter 21

Hard gates:

1. **`ogtt_75g_insulin_peak` z-residual ≤ |1.0|** (iter 20: -1.59 —
   model under-secretes insulin by 3×). This is the make-or-break
   gate: if the recalibration alone fixes insulin amplitude, the
   structural-correctness-of-targets lesson generalizes from gut
   (iter 16) to the metabolic module without an architectural change.
2. **`extended_fast_bhb_overnight` predicted ≥ 0.15 mmol/L** (iter
   20: ~0). Demonstrates the ketogenesis pathway is no longer flat.
3. **glucose mean_mape ≤ 0.42** (iter 20: 0.514). Sustained movement,
   not a one-off.
4. **No regression in gut probe metrics:** ratio stays ≤ 1.20× on
   every dose ≥ 30g; slope stays ≤ 3.5; AUC sub-loss stays < 0.10.
5. **Cold-calibration diagnostic on iter 21 checkpoint shows
   misleading-spec count ≤ 2.** (iter 22 will attack the remaining
   two.)

Soft gates:

6. **overall_weighted_mape ≤ 0.17** (iter 20: 0.186, gate 0.16).
7. **trajectory_rollout raw_loss visibly drops** — *significantly*
   this iter, not flat like iter 19. The expected signature that the
   targets it pulls toward are no longer wrong.
8. **`extended_fast_insulin_basal` z ≤ |1.5|** (iter 20: +2.10 —
   fasting insulin too high; cold-model recalibration drops it from
   12.4 to 11.4).
9. **`small_carb_glucose_peak` z continues toward zero** (iter 20:
   3.15, iter 19: 5.03 — should keep moving).

## Risks / what to look for if iter 21 doesn't move insulin_peak

| symptom | likely cause | next move |
| ------- | ------------ | --------- |
| Cold-model insulin peak hits 60 in `cold-calibration` but trained model still stuck at 20 | metabolic module is architecturally insulin-saturated despite correct cold targets | **Iter 22**: factorize the metabolic insulin head analogous to iter 16's gut fix — sigmoid-on-glucose × scale instead of arbitrary MLP. The hidden-dim 48 MLP may have hit a representational ceiling. |
| Insulin peak moves to ~40 but stalls there | shared embedding projection or coupling priors pull insulin back into compensation when amplitude rises | Tighten the relevant `coupling_priors/metabolism.py` magnitude_range bands; consider a regularizer on the metabolic embedding projection. |
| BHB rises but FFA stays at 0 | ketogenesis pathway works but lipolysis pathway is structurally muted | Raise `insulin_sweep_species_weights[3]` (FFA) from 0.3 → 1.0; probe whether cold `lip_max` needs further tuning (cold-calibration on FFA is borderline at -1.90). |
| Glucose moves but HR/BP regress | cold-model recalibration changed coherent joint dynamics in ways autonomic modules can't track | Iter 22 expands `cold-calibration` to vital-sign pillars (cardiovascular, thermoreg) so cascades are caught early; re-tune autonomic coupling priors. |
| Cold-calibration diagnostic shows new misleading specs (>2) after training | trained model drifted PatientParams-equivalent dynamics in unexpected ways | Inspect whether trajectory_rollout was satisfying its loss by *over-correcting* past the literature target on some specs — may need tighter sigma on those specs to flatten the loss landscape. |
| trajectory_rollout raw_loss skyrockets | the new cold-model dynamics are too far from what the iter 20 model learned; phase 1 can't co-adapt fast enough | Iter 22 ramps the recalibration-derived signals (sweep + cohort) more gradually; or warm-starts iter 21 from cold instead of from iter 20 ckpt. |

## Pointer references (only read if needed)

* `apps/pulse/docs/iter20-handoff.md` — five-pronged metabolic-axis
  intervention details. Read for the iter 20 baseline state and the
  per-spec cohort-ablation table.
* `apps/pulse/docs/iter18-handoff.md` — full justification of the AUC
  term + dilution diagnosis. Read only if iter 21's recalibration
  still leaves a dilution problem on some signal.
* `apps/pulse/docs/iter19-handoff.md` — gradient-budget analysis,
  exact per-loss ‖∇gut‖ measurements. Read only if you need to
  re-derive the gradient-budget reasoning for a new module.
* `apps/pulse/docs/iter16-handoff.md` — factorized kernel structural
  guarantees. **Read this if iter 21 fails the insulin-peak gate** —
  the iter 22 fix will mirror this iter's architectural surgery.
* `apps/pulse/engine/pulse/knowledge/AUTHORING.md` — guide for
  encoding new medical knowledge (cohort specs, coupling priors,
  scenarios, sweep signals).
* `apps/pulse/docs/prd.md` — top-level vision. Read for context
  but not required for iter 21 mechanics.
