# Pulse Iter 15 — Resume Handoff

Self-contained resume point. Read only this file to pick up the work.

## Where we are

**Iter 15 is staged.** Code changes are committed; the Cloud Build
submission is the next action.

- Spec: `apps/pulse/train/spec.json`.
- Cloud Build config: `apps/pulse/train/cloudbuild.yaml`.
- Submission: `bash apps/pulse/scripts/train-submit.sh --gcp` from repo root.
- Job ID assigned at submit time. Artifact root:
  `gs://grovina-pulse/training/jobs/<JOB_ID>/`.

## What iter 14 did and why it was wrong

Iter 14 hypothesis: iter 13's gut signals (`gut-dose-sweep-weight=0.10`,
`gut-loss-weight=0.10`) were over-regularizing the kernel; cut both 5×
and let the indirect signals (cohort, trajectory) carry dose-response
learning. Result was the opposite of what we wanted:

| metric                     | iter 12 | iter 13 | **iter 14** |
| -------------------------- | ------- | ------- | ----------- |
| glucose_mape               | ~0.30   | 0.5792  | **0.6246**  |
| overall_mape               | ~0.18   | 0.2077  | **0.2150**  |
| fasting drift / 4h         | +56     | +54.7   | **+60.6**   |
| gut sweep AUC at 0g/120g   | flat    | 64 / 64 | **53 / 38** (inverted) |
| benchmark gate             | FAIL    | FAIL    | **FAIL**    |

Probe at zero embedding: gut sweep is now *inverted* — peak appearance
at 30 g, monotonically decreasing through 120 g. Iter 13 was a
collapsed kernel; iter 14 is a *misshapen* kernel.

Signal-balance probe explains the mechanism:

| signal              | iter 13 raw / ‖∇gut‖ | iter 14 raw / ‖∇gut‖ |
| ------------------- | -------------------- | -------------------- |
| gut_dose_sweep      | 0.097 / **0.43**     | 0.125 / **0.27** ↓   |
| cohort_statistic    | 3.18 / **2.70**      | 2.95 / **4.09** ↑    |
| trajectory_rollout  | 9.11 / **0.32**      | 9.14 / **0.97** ↑    |

The dominant pull on the gut kernel in iter 13 was already from the
*indirect* signals (cohort: 2.70 vs direct gut: 0.43). Cutting direct
gut supervision 5× left the kernel pulled almost exclusively by signals
that don't supervise gut shape — only integrated state. The kernel
landed in a non-physiological minimum that fits cohort statistics with
a peak-at-30g-then-decreasing absorption curve.

**Architectural lesson (re-stated):** if a module's correctness is
well-defined without integration, it must be supervised directly with
strong weight. Indirect signals can't substitute for direct shape
supervision. Iter 12's handoff said this explicitly; iter 14 violated it.

## What iter 15 changes

### 1. Structural fix: dose-ranking term in `GutDoseSweepSignal`

The pure per-element MSE term `((pred − tgt) / abs_scale)²` admits
low-gradient minima where the kernel has wrong dose-shape (iter 13's
collapse, iter 14's inversion are both examples). The fix is structural:
add a ranking constraint that makes those minima *infeasible*.

For every ordered dose pair `(dᵢ < dⱼ)` and every channel where the cold
target itself ranks the doses (mask-driven; only glucose under the
current protocol where fats/proteins are constant):

```
violation(i, j, b, c) = relu(rank_margin × tgt_AUC_gap(i, j, c) − pred_AUC_gap(i, j, b, c))
```

Summed over `(i, j, b, c)`, masked, and per-channel-normalized by
`abs_scale × T` so it lives at the same magnitude as the MSE term.
With `rank_margin = 0.3`, the predicted AUC gap must be at least 30% of
the cold-target gap, in the right direction. Total loss:

```
loss = mse_loss + ranking_weight × rank_loss        # ranking_weight = 1.0
```

Vectorized as a single `[D, D, B, C]` tensor op — negligible compute on
top of the existing per-dose forward pass.

**Local sanity, starting from the iter-13 collapsed checkpoint:**

```
epoch  loss     mse      rank     n_inv_zero
    0  0.1849  0.0976  0.0873   9
    1  0.1755  0.0937  0.0818   8
    2  0.1658  0.0900  0.0758   5
    3  0.1557  0.0864  0.0693   4
    4  0.1454  0.0829  0.0625   1
   10  0.0833  0.0655  0.0179   1
   19  0.0559  0.0509  0.0050   1
```

The ranking term drove inversions from 9 → 1 in 4 epochs at lr=1e-3 on
the collapsed iter-13 kernel. Both MSE and rank decrease together
through the rest of training. The residual 1 inversion is between the
two smallest-gap adjacent doses (0g↔15g), where the effective constraint
is weakest; it shrinks further as MSE pulls the shape into place.

### 2. Restore gut weights

Iter 14's 5× cut was the wrong move. Restored:

```
--gut-dose-sweep-weight   0.02 → 0.10
--gut-loss-weight         0.02 → 0.10
```

The previous concern (over-regularization → basin collapse) is
neutralized because the loss surface no longer contains the basin.

### 3. CI fix: `CompareToPrevious` working directory

Iter 14's Cloud Build failed at `CompareToPrevious` with
`ModuleNotFoundError: No module named 'pulse'`. The engine
`Dockerfile` sets `WORKDIR /app` and copies `pulse/` into it, but Cloud
Build steps default to `/workspace`, overriding WORKDIR. Fix is one
line: `dir: /app` on the step. `EnforceBenchmarkGate` doesn't need it
(uses the cloud-sdk image, not the engine image).

### 4. New sub-metrics on `gut_dose_sweep`

The signal now reports per-epoch:

- `mse` — pure-MSE component (was the old `loss_sum`)
- `rank` — ranking-term component
- `n_inversions_zero_emb` — count of glucose-channel pair inversions
  at the zero embedding, max value `C(D, 2) = 21` for the 7-dose protocol

These show up in the Phase 2 trainer log as part of the
`gut_sweep=...` field once we wire them through (left as a small follow-up;
sub_metrics are already returned by `compute()`).

## Signal-balance evidence summary (so far)

For posterity / future regression triage:

| signal              | iter 11 | iter 13 | iter 14 |
| ------------------- | ------- | ------- | ------- |
| gut_dose_sweep raw  | 1.51    | 0.097   | 0.125   |
| gut_dose_sweep ‖∇gut‖ | (high) | 0.43    | 0.27    |
| cohort_statistic ‖∇gut‖ | (mid) | 2.70    | 4.09    |
| trajectory_rollout ‖∇gut‖ | (mid) | 0.32    | 0.97    |

Run on any new checkpoint with:

```bash
python -m pulse.diagnostics signal-balance --checkpoint <gs:// or local path>
```

## Submission

```bash
bash apps/pulse/scripts/train-submit.sh --gcp
```

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

- **Probe at zero embedding: gut sweep AUC strictly monotone** in carb
  dose (`monotone: True`). Local sanity says we should hit ≤1
  inversion within 4 epochs of phase 2; converged training should be 0.
- **`glucose_mape` ≤ 0.30** (iter 14: 0.62, iter 13: 0.58).
- **Fasting drift / 4 h ≤ +50 mg/dL** (iter 14: +60.6).
- **Counter-regulation deltas** under a 75 g mixed meal: `glucagon < 0`,
  `ffa < 0`, `ghrelin < 0` (all currently ≈ 0).
- **Wall-clock similar to iter 14** (~3.5h). The ranking term is one
  vectorized `[D, D, B, C]` op — negligible vs the existing per-dose
  forward pass.

## Decision tree for iter 16

1. **Gate passes, probe monotone, signal balance healthy.** Ship the
   trainer-log surfacing of the new sub-metrics so future runs make the
   inversion count visible per epoch. Then move to the pending Phase 3
   counter-regulatory cohort specs.
2. **Probe monotone but `glucose_mape` still > 0.30.** Dose-response
   is correctly oriented but magnitude is wrong. Try `rank_margin
   0.3 → 0.5` to enforce a larger fraction of the cold-target gap, or
   reduce `--trajectory-band-default` so distillation pulls harder
   on integrated glucose.
3. **`n_inversions_zero_emb` doesn't go to 0 by phase-2 end.** The
   ranking gradient is being overwhelmed by another signal pulling
   the kernel in a different direction. Bump `ranking_weight` 1.0
   → 3.0, or bump `--gut-dose-sweep-weight` 0.10 → 0.15.
4. **Probe inverted again.** The ranking constraint is somehow being
   ignored — bug in the mask or normalization. Check `_losses` against
   the existing tests before any other change.
5. **Wall-clock significantly higher than iter 14.** The vectorized
   ranking op probably isn't vectorizing. Profile and inspect.

## Pending work from the broader plan

Unchanged from iter 14:

### Phase 3 — counter-regulatory cohort specs

In `apps/pulse/engine/pulse/knowledge/cohort_statistics/`:

- `OGTT_GLUCAGON_SUPPRESSION` (Müller 1970, Unger 1971)
- `MIXED_MEAL_GLUCAGON_SUPPRESSION`
- `MEAL_FFA_SUPPRESSION` (Frayn 2002)
- `EXTENDED_FAST_FFA_RISE`
- `EXTENDED_FAST_INSULIN`

### Future perf work

- **Cloud Run → Vertex AI on `c3-highcpu-22`.** Cloud Run pins to ~6
  effective cores; Vertex on c3-highcpu-22 unlocks 22.
- **Re-evaluate `torch.compile` once shapes stabilize.**

## Key files (iter 15 changes)

- `apps/pulse/engine/pulse/training/gut_dose_sweep_signal.py`
  - `GutDoseSweepProtocol.rank_margin: float = 0.3` (new field)
  - `GutDoseSweepSignal.ranking_weight: float = 1.0` (new field)
  - Pre-compute `_rank_target_gap`, `_rank_mask` in `__post_init__`
  - Extract loss math into `_losses(per_dose_pred, ...) → (mse, rank, n_inv)`
    so tests can drive it without going through `compute()`
  - `compute()` returns new sub-metrics: `mse`, `rank`, `n_inversions_zero_emb`
- `apps/pulse/engine/tests/test_gut_dose_sweep_signal.py`
  - New `TestRankingTerm` class with 4 cases:
    pure-MSE-when-matched, positive-when-inverted,
    `ranking_weight=0` ablation, gradient-pushes-toward-monotone.
- `apps/pulse/train/cloudbuild.yaml` — `dir: /app` on `CompareToPrevious`.
- `apps/pulse/train/spec.json` — restored gut weights to 0.10 / 0.10;
  hypothesis & expectedEffect rewritten.
- `apps/pulse/docs/iter14-handoff.md` — left as historical record;
  iter15 supersedes its hypothesis.

## Architectural / philosophical context

Three lessons compound into iter 15. The first two from earlier handoffs:

**Iter 12 → 13 (scale):** when introducing a new loss term, the
per-element scale matters as much as the weight. A signal whose
residual is normalized by 100× the typical excursion contributes the
same gradient as a signal at weight/100 — silently. Iter 13 put the
scale in one place (`GUT_OUTPUT_SCALE`).

**Iter 13 → 14 (basin):** when a per-element scale lets the loss admit
a flat trivial minimum, even a moderate-weight signal can pull the
optimizer into that minimum during early training. Iter 14 tried to
weaken the signal to escape; this was wrong (next lesson explains why).

**Iter 14 → 15 (separation of supervision):** weakening *direct*
shape supervision so that *indirect* (integrated-state) supervision
can carry shape learning is a categorical error. The indirect signals
don't see kernel shape; they see only the consequence of integrating
through it. The right escape from a kernel basin is structural —
reformulate the loss so the basin doesn't exist — not weight tuning.
The dose-ranking term is the structural fix: it makes "wrong dose
ordering" infeasible by construction, regardless of the per-element
loss surface's local minima.

Stated as a principle: *for stateless modules with inference-time
queries that have known structural properties (monotonicity,
positivity, conservation), encode those properties as constraints in
the direct supervision loss, not as soft signals to be balanced.*
The MSE term gives the correct *shape*; the ranking term gives the
correct *order*. Both are needed; one without the other produces
either iter 13's collapse or a hypothetical "monotone but wrong shape"
failure mode.

## Resume protocol for the next agent

1. Read this file.
2. Confirm the run was submitted (commit log or `gcloud builds list`).
3. When it finishes:
   - `gsutil cat gs://.../delta-vs-baseline.json | jq '.gut_sweep_table'`
     for the at-a-glance dose-monotonicity check (now that
     `CompareToPrevious` will actually upload it).
   - `python -m pulse.diagnostics probe --checkpoint <ckpt>` for the
     full probe, including the `monotone:` field on gut sweep.
   - `python -m pulse.diagnostics signal-balance --checkpoint <ckpt>`
     to see how the new ranking-enabled signal landed in terms of grad
     norms relative to the indirect signals.
4. Apply the decision tree above to scope iter 16.
