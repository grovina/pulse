# Pulse iter 77 — honest benchmark + level anchors

This iter came out of a full-stack modeling review (not a parameter sweep). The
review found that the 30+ iterations of "fighting parameters" were the symptom
of three compounding structural problems, two of which this iter fixes and one
of which it deliberately defers (with a design).

## What the review found

1. **The benchmark could not see dynamics.** Over the real-user scored window
   (t=510–690) the markers barely move — glucose ~2.9 mg/dL, hr ~3 bpm, temp
   ~0.1 °C. Pointwise MAPE there rewards "land a calibrated constant," not
   "predict a trajectory." A persistence baseline (carry the last calibration
   reading forward) scores ~0.014 glucose MAPE — so the bar a passing model
   cleared was mostly *baseline landing*. Worse, the hormone numbers come from
   cohort episodes whose ground truth **is** the cold model the network is
   distilled from (circular: distillation fidelity, not physiology), and the one
   fit-free non-circular check (`textbook_mean_pass_rate`) was silently disabled
   locally because `default_thresholds()` had drifted from
   `benchmark.thresholds.json` (0.0 vs 0.45).

2. **Absolute levels were under-supervised for the worst markers.** Insulin
   (worst marker, ~0.90) had every *strong* signal level-blind by construction:
   `insulin_sweep` (w=0.30) matches dI/dt teacher-forced (never integrated →
   offset-invariant); `dose_response` (w=0.40) matches Δpeak above the model's
   own free baseline; and insulin was **excluded** from cold-distill. Its only
   level pin was ~3 cohort specs at effective weight ~2.6e-3. Same disease for
   `lactate` (insulin_sweep species-weight 0.0; only a delta cohort + a
   zero-embedding default-baseline) and `hepatic_output` (sweep rate + one
   delta).

3. **The optimizer mixes signals by sequential per-signal SGD, not a joint
   loss.** Each signal does its own `backward + clip_grad_norm_(10) + step`, so
   a signal's "weight" is erased whenever its grad-norm exceeds the clip (this is
   why weight sweeps went byte-identical for 30 iters), and signals fight in
   sequence. **Caveat found during scoping:** the *main* data-fit signal
   (trajectory) already does proper per-window SGD (~84 steps/epoch); only the
   ~10 auxiliary/constraint signals suffer the clip-erasure + fighting. So the
   fix is *auxiliary-signal gradient accumulation*, not a global "one joint step
   per epoch" (which would cut steps ~90× and be untrainable). That makes it a
   training-loop change needing retuning — **deferred to iter 78**, to be
   designed against this iter's honest baseline rather than launched blind.

## What this iter ships

**Honest benchmark (measurement only — cannot move training):**
- `skill_vs_persistence` per marker (1 = perfect, 0 = no better than persistence,
  <0 = worse than carrying the last reading forward).
- Real-vs-teacher source split: every `BenchmarkEpisode` carries a `source`
  (`real` measured data vs `teacher` cold-model cohort); the report exposes
  `overall_weighted_mape_by_source` and `per_marker_by_source` so circular
  numbers never enter the measured-data headline.
- `default_thresholds()` now **loads** `benchmark.thresholds.json` (single source
  of truth) instead of a drifted copy — re-enabling the textbook gate (0.45).
- The gate's pass/fail field is unchanged (still mixed `overall_weighted_mape`);
  only honest *views* were added, so this cannot regress the gate. Switching the
  gate to real-only is a deliberate follow-up.

**Level anchors (the mechanism under test):**
- `insulin`, `lactate`, `hepatic_output` added to the cold-distill **anchored**
  marker set, pinning each marker's absolute *level* via the short free-rollout
  depth anchor (mode was already `anchored` from iter 76). All three are
  simulated in the cold ODE (indices 1/5/6), so the reference exists.

**Reproducibility (fixes a review finding of its own):**
- The full recipe was being lost (it lived in an external file dropped from the
  repo). `train/spec.json` is restored and `python -m pulse.train --spec
  train/spec.json` injects it (explicit CLI flags override). The trainer image
  now ships `train/` (Dockerfile) and a `.dockerignore` keeps the build context
  small.

## How to run

```bash
# local (tiny smoke)
uv run python -m pulse.train --spec train/spec.json --n-patients=1 \
  --phase1-epochs=1 --phase2-epochs=1

# cloud (full iter)
bash deploy/deploy.sh
gcloud run jobs execute trainer --region europe-west1 \
  --args=--spec=train/spec.json,\
--gcs-bucket=grovina-pulse-data,--gcs-object=training/jobs/iter77/model.pt,\
--benchmark-dataset-uri=pulse/benchmark.dataset.generated.json
```

## What to read off the run (the new iter-78 baseline)

- `per_marker['insulin'].mean_mape` — should drop materially from ~0.90 as the
  level anchor engages; `dist_insulin` finite + falling in phase 2.
- `dist_lactate` / `dist_hepatic_output` finite + falling (levels stop floating).
- `per_marker_by_source['real']` skill_vs_persistence per marker — record as the
  honest baseline (glucose skill is a hard bar; persistence ~0.014).
- `overall_weighted_mape_by_source` (real vs teacher) and
  `textbook_mean_pass_rate` now that its gate is live.

## Risks

- (R1) anchoring insulin could fight `insulin_sweep`'s rate term — complementary
  (level vs rate), but watch for insulin oscillation; if it regresses, drop
  insulin from the anchor set first.
- (R2) three extra anchored markers add modest phase-2 cost (same per-protocol
  rollouts, just more scored markers); `anchor_skips`/OOM unlikely but watched.
- (R3) the textbook gate is now live at 0.45 — if the rate sits below, the gate
  fails. That is the check doing its job (it was disabled before), not a bug.
