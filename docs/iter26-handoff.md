# Pulse Iter 26 — Resume Handoff

Self-contained resume point. Iter 26 is a **bug fix**, not a
hyperparameter exploration. Same iter-23/24/25 architecture, same iter-25
loss weights. The only behavioral changes are (1) one corrected divisor
in `_safe_corr_torch` and (2) a new `cpl` field in the 5-epoch print
headline.

## Why iter 25 aborted

Build `2ffc4b7f-6762-458b-be2d-3201e45bdbd1` (`train-20260426T062232Z`)
fired the iter-25 strict abort at Phase 2 epoch 61. From the abort
diagnostic JSON:

```
abort.signal       trajectory_rollout/window
abort.cause        grad
abort.epoch        61
abort.phase        2
abort.lr           0.000267
abort.loss_value   0.0578  (finite — forward was clean)

extra.patient_id   13
extra.win_start    3569 min  (~day 3, mid-day)
extra.n_meals_in_window 1
extra.coupling_loss     0.561
extra.verifier_loss     0.113
extra.gut_loss          0.000
extra.landmark_loss     0.000

extra.n_params_with_grad   115
extra.n_finite_params      44
extra.n_nan_params         71
extra.n_inf_params         0     ← not Adam denormal
extra.max_abs_grad_finite  0.015 ← surviving grads tiny
```

Pattern: 71/115 params NaN, 0 Inf, surviving params have small grads.
That's NaN propagation, not Inf overflow — and the NaN reaches every
parameter the loss flows through, which means the trap is in a path
broad enough to cover most heads.

Code audit found exactly one match: `_safe_corr_torch` in
`apps/pulse/engine/pulse/training_verifier_loss.py:25`. Pre-fix:

```python
denom = sa * sb
return torch.where(
    denom < 1e-8,
    a.new_tensor(0.0),
    (am * bm).mean() / denom,
)
```

This is the canonical PyTorch `torch.where` NaN-grad trap. Forward
selects 0 when denom is tiny, **but autograd evaluates BOTH branches in
backward**. The unselected `(am*bm).mean() / denom` produces a
non-finite gradient at `denom≈0` (chain rule yields `1/denom²`), and
`torch.where` then propagates NaN through the unselected branch into
every parameter that fed `a` or `b`.

The verifier surrogate calls `_safe_corr_torch` for `hr↔sbp`,
`glucose↔glucagon`, etc. correlations on each window's `pred_traj`. For
windows where any of those marker series happens to be near-constant
(zero variance) the divisor approaches zero, the trap fires, and 1+
backward minutes of model param flow through the verifier loss
produces NaN gradients across the metabolic / cardiovascular / stress
heads.

## What iter 26 changes

### Surgical fix — clamp before divide

`apps/pulse/engine/pulse/training_verifier_loss.py`:

```python
denom_safe = denom.clamp(min=1e-8)
corr = (am * bm).mean() / denom_safe
return torch.where(denom < 1e-8, a.new_tensor(0.0), corr)
```

After clamping, both branches of the `where` are finite during
backward; the unselected branch's gradient still flows but multiplied
by 0 (`0 * finite = 0`), which is correct. The where condition still
uses the unclamped `denom` so forward behavior is bit-for-bit
unchanged for normal inputs.

### Regression test

`apps/pulse/engine/tests/test_safe_corr_grad.py` — 3 cases:

1. Zero-variance `a` flowing into a leaf parameter via `b` →
   `corr.backward()` must produce a finite grad on the parameter. (The
   exact failing pattern from iter 25; pre-fix this would NaN.)
2. Both sides zero-variance — degenerate but still must not NaN.
3. Normal correlated input — the clamp must NOT bias the gradient when
   denom is well above the floor.

### `cpl` in 5-epoch print

`trajectory_signal` now exposes `coupling` as a sub-metric (averaged
over windows) when coupling priors are active. `train.py` prints it as
`cpl=X.XXXXXX` in the 5-epoch headline alongside `gut`, `v_sur`, `lm`.
The iter-25 epoch-58 trajectory spike (0.63 → 1.08, recovered to 0.59
at 59) was a leading indicator we couldn't decompose mid-run; with
`cpl` visible we'll see whether spikes drive through coupling, the
trajectory base, or something else.

### Audit completeness

Full grep for `torch.where` in `apps/pulse/engine/pulse/`:

| site | safe? | reason |
| --- | --- | --- |
| `training_verifier_loss.py:35` | **fixed in iter 26** | clamp-before-divide |
| `training/trajectory_signal.py:391` | safe | both branches (`diff`, `zeros_like`) finite |
| `training/trajectory_signal.py:401` | safe | both branches (`quad`, `lin`) finite for any finite `excess` |

No other `torch.where` calls in the engine. No `torch.sqrt(0)` /
`torch.log(0)` / `1/x` patterns elsewhere (only one division-by-tensor
site: `total_macro / FLAG_GATE_SCALE_G` in `modules/base.py:266` — the
divisor is a class constant, safe).

## What iter 26 does NOT change

* **No LR change.** `--phase2-lr=0.0003` (same as iter 24/25).
* **No architecture change.** Same per-species heads +
  `GlucoseGatedInsulinHead`.
* **No `coupling_prior_weight` / `verifier_loss_weight` change.** The
  bug was a numerical-stability defect, not a weight imbalance.
* **No `trajectory_band` change.** Same 0.2/0.05.
* **No Adam-epsilon change.** Default 1e-8.

The bug-fix-only scope makes iter 26 a clean A/B against iter 25 modulo
this single change. If iter 26 reaches Phase 2 epoch 99 without
aborting, the iter-25 NaN was caused by exactly this bug.

## Success criteria

**Stability gate (primary):** Phase 2 reaches epoch 99/99 with non-NaN
loss at every 5-epoch print. Total wall-clock < 12h.

**Architectural gates** (carried from iter 23/24, finally testable
under stable training):

1. `large_meal_glp1_peak` predicted Δ ≥ 6 pmol/L (iter 22: 0.14)
2. `ogtt_glucagon_suppression` predicted Δ ≤ -10 pg/mL (iter 22: -0.016)
3. `extended_fast_bhb_overnight` predicted ≥ 0.10 mmol/L (iter 22: -0.003)
4. `extended_fast_insulin_basal` predicted ≤ 12 µU/mL (iter 22: 18.1)
5. `ogtt_75g_insulin_peak` predicted ≥ 45 µU/mL (iter 22: 39.1)
6. (5) AND (4) simultaneously — falsifiable test for intervention C.
7. `large_meal_glp1_peak` and `ogtt_glucagon_suppression` simultaneously
   moving — falsifiable test for intervention A.

**Soft gates:**

* glucose `mean_mape` ≤ 0.50 (iter 22: 0.566)
* `overall_weighted_mape` ≤ 0.18 (iter 22: 0.193)
* verifier `overall_score` within -0.02 of iter 22's 0.862

**Sanity:**

* Cold-calibration misleading-spec count ≤ 2
* Gut probe ratio ≤ 1.20× on every dose ≥ 30g

## Risk paths

| symptom | likely cause | iter 27 move |
| --- | --- | --- |
| Reaches epoch 99 cleanly + architectural gates move | iter-23 architecture vindicated; first stable run since iter 22 with these heads | continue with current architecture, attack next-priority gap (e.g. BHB long-fast visibility per iter-23/24 risk paths) |
| Reaches epoch 99 cleanly but architectural gates DON'T move | iter-23 per-species head hypothesis falsified under stable training (the test we couldn't run before) | revert to iter-22 shared-trunk architecture, OR try long-fast windows / long-fast cohort signal (iter 23/24 R2) |
| Aborts at a NEW signal/site | new bug surfaced by the stricter abort path now that the verifier site is clean | read iter-26 abort dump, follow iter-25 decision tree against the new fingerprint |
| Aborts at the SAME signal (`trajectory_rollout/window` cause=grad) | the verifier wasn't the only NaN trap on that path — coupling_prior_loss or trajectory base has its own | inspect the new abort dump's `extra` for which sub-component dominates; clamp/audit that path |
| Coupling component (`cpl=`) spikes in headline before any abort | a specific coupling prior is amplifying through training | reduce `coupling_prior_weight` 0.03 → 0.01 OR remove the offending prior |

## How to read the next abort dump (if it fires)

Same shape as iter 25's. The iter-25 handoff
(`apps/pulse/docs/iter25-handoff.md`) has the full decision tree. The
two highest-signal fields:

* `abort.cause`: `loss` means a forward path went non-finite (different
  failure mode from iter 25); `grad` means another autograd singularity
  (need to audit which op).
* `abort.extra`: signal-specific. For `trajectory_rollout/window` it
  includes per-window patient_id / win_start / per-component losses, so
  you can re-run that window in isolation against the last-good
  checkpoint.

`last_good_pre_nan.pt` is uploaded next to `abort_diagnostics.json` —
that's a frozen end-of-last-good-epoch state ready for diagnostics:

```bash
.venv/bin/python -m pulse.diagnostics signal-balance \
    --checkpoint /tmp/iter26/last_good_pre_nan.pt --n-patients 20
.venv/bin/python -m pulse.diagnostics cohort-ablation \
    --checkpoint /tmp/iter26/last_good_pre_nan.pt --sample-patients 4
```

## How to resume after reboot

```bash
cd /Users/grovina/Projects/grovina/platform/.claude/worktrees/pulse

# gcloud is wired to pulse-iteration-runner SA via .claude/settings.local.json
# (env: GOOGLE_APPLICATION_CREDENTIALS + CLOUDSDK_ACTIVE_CONFIG_NAME=grovina-pulse).

# 1. Verify worktree state.
git status
.venv/bin/python -m pytest apps/pulse/engine/tests/ -x -q   # 137 pass

# 2. Submit iter 26.
bash apps/pulse/scripts/train-submit.sh

# 3. While running, watch build URL for the 5-epoch print pattern:
#    Phase X Epoch NN/100  loss=...  gut=...  cpl=...  v_sur=...  lm=...(N)  cohort=...  ...
#    The new `cpl=` field will surface coupling-component spikes.

# 4. After completion:
LATEST=$(gsutil ls gs://grovina-pulse/training/jobs/ | tail -1 | awk -F/ '{print $(NF-1)}')
if gsutil -q stat "gs://grovina-pulse/training/jobs/$LATEST/abort_diagnostics.json"; then
  echo "ABORTED — fetching diagnostics + last-good"
  mkdir -p /tmp/iter26 && \
    gsutil -o "GSUtil:parallel_process_count=1" cp \
      gs://grovina-pulse/training/jobs/$LATEST/{abort_diagnostics.json,last_good_pre_nan.pt,spec.json,meta.json} \
      /tmp/iter26/
  jq . /tmp/iter26/abort_diagnostics.json
else
  echo "Completed — fetching benchmark + delta + checkpoint"
  mkdir -p /tmp/iter26 && \
    gsutil -o "GSUtil:parallel_process_count=1" cp \
      gs://grovina-pulse/training/jobs/$LATEST/{benchmark-report.json,delta-vs-baseline.json,checkpoint.pt,spec.json} \
      /tmp/iter26/
  python3 -c "import json; r = json.load(open('/tmp/iter26/benchmark-report.json')); \
    print(f'gate.passed={r[\"gate\"][\"passed\"]} overall={r[\"overall_weighted_mape\"]:.4f} verifier={r[\"verifier\"][\"overall_score\"]:.4f}')"
fi
```

## Code state (iter 26 changes ready to commit)

* `apps/pulse/engine/pulse/training_verifier_loss.py` — `_safe_corr_torch`
  clamp-before-divide fix.
* `apps/pulse/engine/tests/test_safe_corr_grad.py` — new, 3 regression
  tests for the where-NaN-trap pattern.
* `apps/pulse/engine/pulse/training/trajectory_signal.py` — accumulate
  `coupling_sum` per window, surface `coupling` sub-metric.
* `apps/pulse/engine/pulse/train.py` — print `cpl=X.XXXXXX` in 5-epoch
  headline.
* `apps/pulse/train/spec.json` — iter-26 hypothesis + expectedEffect.

## Cloud Build / job IDs

| iter | job ID | build status | outcome |
| ---- | ------ | ------------ | ------- |
| 21 | train-20260423T175447Z | FAILURE (gate) | overall=0.190, glu_mean=0.543 |
| 22 | train-20260424T071900Z | FAILURE (gate) | overall=0.193, glu_mean=0.566 |
| 23 | train-20260424T163240Z | FAILURE (NaN + timeout) | no checkpoint produced |
| 24 | train-20260425T153910Z | FAILURE (NaN-corrupt checkpoint) | textbook=0.108, all metrics NaN |
| 25 | train-20260426T062232Z | FAILURE (strict abort fired correctly at Phase 2 epoch 61) | abort dump pinpointed `_safe_corr_torch` |
| 26 | train-20260426T105353Z | WORKING at handoff time (build `f361b1a1-9bcb-4831-aa0b-c5ed75b106e4`) | iter-26 = surgical verifier fix + cpl logging |

## Pointer references

* `apps/pulse/docs/iter25-handoff.md` — strict-abort + diagnostic-dump
  framework; decision tree for reading abort dumps.
* `apps/pulse/docs/iter24-handoff.md` — iter-24 silent-NaN failure mode
  and why skip-and-continue guards were insufficient.
* `apps/pulse/docs/iter23-handoff.md` — per-species heads +
  `GlucoseGatedInsulinHead` architecture.
* `apps/pulse/docs/training-runs.md` — commit-before-submit workflow,
  GCP wiring, train-submit.sh details.
