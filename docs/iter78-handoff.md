# Pulse iter 78 — joint auxiliary-signal optimization

This iter ships the one structural change iter 77 deferred: the optimizer.
It is **not** a parameter sweep — the training config is held byte-identical to
iter 77 on purpose (see "Why config is frozen").

## What iter 77 actually did (read this first)

iter 77 **never tested its own hypothesis.** It NaN-aborted in **phase-1 epoch
39/50** — before phase 2 (where its level anchors engage) and before any
`benchmark-report.json` was written. The only artifacts in
`gs://grovina-pulse-data/training/jobs/iter77/` are `abort_diagnostics.json` and
`last_good_pre_nan.pt` (epoch 38). So iter 77's two hypotheses (honest benchmark
views + insulin/lactate/hepatic level anchors) are **untested** — the code is
committed and correct, it just never ran to completion.

**Root cause** (from `abort_diagnostics.json`): the `postprandial_recovery` aux
signal runs a single 480-step rollout and penalizes residual glucose 7h
post-meal. Over epochs 35→38 its loss went **0.009 → 0.70 → 0.63 → 116.9**, the
recovery-window glucose ran away to **303 mg/dL**, and backprop through that
unstable long rollout produced **NaN grads in 113/159 params**. The strict
`safe_step` (first non-finite grad halts) killed the run. The main `trajectory`
signal was healthy the whole time (~0.19). The damage was entirely in the
auxiliary steps.

## Why this is structural, not bad luck

Every one of the ~10 auxiliary signals did its **own** `backward +
clip_grad_norm_(10) + step + zero_grad`, in sequence. So each epoch the model
took ~10 independent clipped steps in fighting directions:

1. **Weights were erased.** Whenever a signal's grad-norm exceeded the clip (10),
   its step was scaled to norm 10 *regardless of its configured weight* — which
   is why 30+ iters of weight sweeps produced byte-identical benchmarks (the
   canonical dead-pathway symptom; see `dead-pathways.md` and `training-runs.md`).
2. **One signal could kill the run.** `postprandial_recovery` walked the model
   into an unstable regime over a few solo steps, then its own gradient NaN'd and
   aborted everything — before the rest of training could counterbalance it.

## What this iter changes

**Auxiliary signals now ACCUMULATE into one joint step** (`pulse/training/safe_step.py`):

- `accumulate_grad(loss, ctx, ...)` — single-tensor aux signals (`dose_response`,
  `gut_dose_sweep`, `insulin_sweep`, `fasting_stability`, `postprandial_recovery`,
  `default_baseline`, `carb_mass_balance`, `cold_model_distillation`): `backward`
  only, no solo clip/step/zero.
- `finalize_aux_accumulation(ctx, snapshot, ...)` — the two signals that already
  accumulate internally per-spec/rule for memory (`cohort`, `physiology_rules`):
  they keep their per-term `backward()` + graph-drop loop but **defer** the
  clip+step to the joint one.
- `joint_aux_step(ctx)` — the trainer (`pulse/train.py`, after the signals loop)
  applies **one** `clip_grad_norm_ + step + zero_grad` over the combined aux
  gradient per epoch, iff any aux signal accumulated (`ctx.aux_accumulated`).
- The **`trajectory` signal is untouched** — it keeps its per-window SGD (~84
  steps/epoch, the main data fit). Folding it into one joint step/epoch would cut
  its steps ~90× and be untrainable. This is *auxiliary*-signal accumulation, by
  design.

Effect: aux weights **compose** (the joint clip scales the whole combined
gradient, preserving relative magnitudes), signals stop fighting sequentially,
and per-epoch aux drift drops from ~10 clipped steps to one — the drift that
drove iter 77's runaway.

**Isolation policy (robustness):** a single aux signal whose loss or gradient is
non-finite has its contribution **rolled back to a pre-signal snapshot and
dropped** (logged `[SKIP-NONFINITE] signal=… cause=…`), not aborted. So no single
unstable aux rollout can kill a run before it produces numbers. Strict abort is
preserved for the `trajectory` signal and — defense-in-depth — for the combined
gradient in `joint_aux_step` (it would surface as `signal=joint_aux`).

## Why config is frozen

iter 78 changes **only the optimizer structure**; the entire `trainArgs` recipe
is identical to iter 77. Two reasons: (1) it makes the optimizer change
*attributable* — no confounded knobs; (2) it finally lets iter 77's honest
benchmark + level anchors reach phase 2 and emit the baseline that was the whole
point. Constraint-strength retuning (the joint clip now gives *all* aux signals
combined the budget *one* had alone — a net reduction in aux influence) is a
deliberate follow-up against that now-real baseline, not bundled here.

## What to read off the run (the real iter-78 baseline)

1. **It completes both phases and writes `benchmark-report.json`** (the bar iter
   77 never cleared). If it still aborts, the abort `signal` will be `joint_aux`
   or `trajectory_rollout/window` — never an aux signal (those self-isolate now) —
   which localizes the next fix.
2. `per_marker_by_source['real']` `skill_vs_persistence` for
   glucose/hr/sbp/dbp/temp — the first honest baseline. Glucose skill is a hard
   bar (persistence ~0.014); near-zero/negative is honest, not a regression.
3. `overall_weighted_mape_by_source` (real vs teacher) and
   `textbook_mean_pass_rate` (gate live at 0.45).
4. `dist_insulin` / `dist_lactate` / `dist_hepatic_output` in the
   cold-distill telemetry — finite and falling over phase 2 (iter 77's anchors,
   finally exercised).
5. **Aux losses should now track their weights** across epochs (gut_sweep,
   ins_sweep, fasting, postprandial, cohort, physiology) instead of moving
   byte-identically — the weight-erasure ceiling lifting.
6. `[SKIP-NONFINITE]` log lines: a few isolated skips = the safety net working;
   *persistent* skips of one signal = that rollout still destabilizes and wants a
   targeted fix (e.g. a saturating postprandial residual loss) in iter 79.

## How to run

```bash
# local smoke (proves the joint step wires up)
uv run python -m pulse.train --spec train/spec.json --n-patients=1 \
  --phase1-epochs=1 --phase2-epochs=1

# cloud (full iter)
bash deploy/deploy.sh
gcloud run jobs execute trainer --region europe-west1 \
  --args=--spec=train/spec.json,--gcs-bucket=grovina-pulse-data,\
--gcs-object=training/jobs/iter78/model.pt,\
--benchmark-dataset-uri=pulse/benchmark.dataset.generated.json
```

## Risks

- **(R1) Constraints under-trained.** Joint clip (norm 10 over all aux combined,
  vs 10 each) reduces total aux influence per epoch. If cold-distill/anchor or
  sweep constraints look weak in the report, raise `grad_clip` or split a
  trajectory-vs-aux clip budget in iter 79 — not here (attribution).
- **(R2) Cross-signal grad pollution.** Accumulation never zeroes between aux
  signals; cold-distill's internal embedding-calibration `backward` rides into the
  joint step exactly as it rode into cold-distill's solo step before
  (behavior-preserving, but watched).
- **(R3) Fewer aux steps.** ~1 joint aux step/epoch vs ~10 before. Over 60 epochs
  that is enough to enforce regularizer-style constraints *and* they now compose;
  but if a constraint needs more cadence, that is an iter-79 structural call.
