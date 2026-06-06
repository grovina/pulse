# Pulse Iter 25 — Resume Handoff

Self-contained resume point. Iter 25 changes the **NaN-handling strategy
only** — it does not change the model architecture, the LR schedule, the
loss weights, or any cohort/spec definitions. Same iter-23/24 architecture
(per-species `MassActionModule` heads + `GlucoseGatedInsulinHead`).

## Why iter 24 failed silently

Cloud Build `4392b9bb-6b1b-43d5-bd32-79969537fe1c` (`train-20260425T153910Z`):

* Phase 1 (epochs 0–49) ran cleanly. Loss settled in 0.20–0.40 range.
* Phase 2 epochs 50–60 ran cleanly. Loss 0.5–0.9, all aux metrics
  finite, ~510s/epoch.
* **Phase 2 epoch 65: every aux loss flipped to NaN simultaneously**
  (`v_sur=nan, lm=nan, dose=nan, gut_sweep=nan, ins_sweep=nan`). Step
  time collapsed 510s → 205s.
* Epochs 65–99: `loss=0.000000` printed every 5 epochs while the model
  produced NaN inference. Final checkpoint had every parameter NaN.
* Benchmark gate failed on `textbook_mean_pass_rate=0.108 < 0.45` —
  every per-marker MAPE / verifier score / scenario `new_value` came
  back NaN; the four "cleared" gates (glucose/hr/overall/temp MAPE)
  flipped only because they were now NaN.

The iter-24 inline NaN guards in all 5 backward sites (`cohort_signal`,
`dose_response_signal`, `gut_dose_sweep_signal`, `insulin_sweep_signal`,
`trajectory_signal`) skip backward+step on a non-finite **loss** but do
NOT revert model parameters. Once parameters turned NaN at epoch 65 the
guard zeroed the loss every step and there was no recovery path. They
also did not check **gradients** for finiteness — `clip_grad_norm_`
specifically does not filter non-finite grads, so a single bad
backward (loss finite but grad NaN/Inf) corrupts every parameter on the
next `optimizer.step`.

## What iter 25 changes

### Strict abort path — `safe_step` helper

`apps/pulse/engine/pulse/training/safe_step.py` (new). Centralizes the
backward + clip + step + zero_grad sequence with two abort conditions:

```python
def safe_step(loss: torch.Tensor, ctx: SignalContext, *, signal: str,
              extra: dict[str, Any] | None = None) -> float:
    if not torch.isfinite(loss):
        raise NaNTrainingAbort(cause="loss", ...)
    loss.backward()
    if any non-finite gradient:
        ctx.optimizer.zero_grad()  # leave optimizer state clean
        raise NaNTrainingAbort(cause="grad", ...)  # extras include grad stats
    nn.utils.clip_grad_norm_(...)
    ctx.optimizer.step()
    ctx.optimizer.zero_grad()
    return float(loss.detach().item())
```

`NaNTrainingAbort` carries `signal`, `epoch`, `cause` ("loss" or "grad"),
`loss_value`, and a free-form `extra` dict for signal-specific
breakdown. Caller passes the **already-weighted** loss so the gradient
magnitude matches what the optimizer would have actually seen.

### All 5 signals refactored to use `safe_step`

* `cohort_signal.py` — extras: per-spec z² (`z_<spec_name>`), n_emb,
  n_specs, raw_loss, weight.
* `dose_response_signal.py` — extras: predicted_slope, target_slope,
  n_emb, raw_loss, weight.
* `gut_dose_sweep_signal.py` — extras: mse / rank / auc components,
  n_dose_emb_pairs, n_inversions_zero_emb, raw_loss, weight.
* `insulin_sweep_signal.py` — extras: mse / rank / auc components, plus
  per-axis g_mse / g_rank / g_auc / i_mse / i_rank / i_auc, raw_loss,
  weight.
* `trajectory_signal.py` — extras (per-window!): patient_idx,
  patient_id (or -1 for default-embedding), is_default, window_idx,
  win_start, win_end, win_steps, n_meals_in_window, traj_mode (huber=1
  vs mse=0), band, gut_loss, coupling_loss, verifier_loss,
  landmark_loss, landmark_meals_window. The trajectory signal does many
  per-window backwards per epoch; the abort dump pinpoints exactly
  which window.

The iter-24 `nan_skip` sub-metric is removed — under strict abort it's
either always 0 or the run aborts.

### `train.py` — abort handler around the epoch loop

```python
try:
    for epoch in range(total_epochs):
        # ... per-signal compute, scheduler.step(), epoch-end save ...
        torch.save(..., last_good_path)  # rolling, overwritten each epoch
except NaNTrainingAbort as abort:
    diagnostic = {
        "outcome": "aborted_nan",
        "abort": {signal, epoch, cause, loss_value, phase, lr, extra},
        "recent_metrics": [last 5 epoch summaries],
        "config": {hyperparameters},
        "aborted_at": ISO timestamp,
    }
    write to /tmp/<output>.abort.json
    upload abort_diagnostics.json + last_good_pre_nan.pt to GCS
    return 42  # CLI raises SystemExit(42)
```

### Per-signal wall-time in 5-epoch print

Every 5-epoch headline now ends with `[traj=Xs coh=Ys dose=Zs gsw=Ws
isw=Vs]`. The iter-24 step-time collapse from 510s → 205s would have
been an immediate leading indicator if it had been logged. Per-signal
times also flow into the abort diagnostic dump.

### `cloudbuild.yaml` — surface abort in build log

The `RunTraining` step now checks for `abort_diagnostics.json` after
the Cloud Run execution completes; if present, `gsutil cat`s it into
the build log inside a clear `TRAINING_ABORTED_NAN` banner, then
`exit 1`s. Downstream `ResolveBaseline` / `CompareToPrevious` /
`EnforceBenchmarkGate` skip automatically (waitFor chain).

### Tests

`tests/test_safe_step.py` — 4 unit tests:

1. Finite loss + finite grad → step taken, value returned, grads cleared.
2. NaN loss → abort (`cause="loss"`), parameters untouched, no grad accumulated.
3. Inf loss → abort (`cause="loss"`), parameters untouched.
4. Finite loss + NaN grad (forced via custom `_NaNGrad` autograd Function) →
   abort (`cause="grad"`), grad-stats merged into `extra`, optimizer cleared.

All 134 engine tests pass (130 pre-existing + 4 new).

## What iter 25 does NOT change

* **No LR change.** `--phase2-lr=0.0003` (same as iter 24).
* **No architecture change.** Same per-species heads + glucose-gated
  insulin head.
* **No per-spec output clamps.** Tempting, but a guess until we know
  which spec / which head produces the offending output.
* **No Adam-epsilon bump.** Tempting, but a guess.
* **No trajectory-band tightening.** Tempting, but a guess.

All of these are candidate fixes for iter 26. Choosing one before
reading the abort dump is exactly the iter-24 mistake.

## How to read `abort_diagnostics.json`

```bash
gsutil cat gs://grovina-pulse/training/jobs/<job-id>/abort_diagnostics.json | jq .
```

Top-level keys:

| key | what it tells you |
| --- | ------ |
| `abort.signal` | which signal hit NaN first (`cohort_statistic`, `dose_response`, `gut_dose_sweep`, `insulin_sweep`, `trajectory_rollout/window`) |
| `abort.cause` | `loss` (loss aggregate non-finite) vs `grad` (loss finite, ≥1 param grad non-finite) |
| `abort.epoch` + `abort.phase` | when in the schedule. `phase=1` Phase-1 distillation; `phase=2` full-signal phase. |
| `abort.lr` | LR at time of abort. If `phase=2`, this is on the cosine decay from `phase2-lr`. |
| `abort.extra` | signal-specific breakdown — see decision tree below. |
| `recent_metrics` | last 5 epochs of summary (loss, per-signal time, n_units). Look for a signal whose loss was already climbing or whose time was already changing. |
| `config` | full hyperparameter snapshot for reproducibility. |

For `cause="grad"`, `extra` also includes `n_params_with_grad`,
`n_finite_params`, `n_nan_params`, `n_inf_params`, `max_abs_grad_finite`,
`grad_l2_finite_only` — distinguishes "isolated parameter exploded"
(`max_abs_grad_finite` huge, `n_finite_params` ≈ all) from "broad
denormal blowup" (`n_inf_params` >> `n_nan_params`, indicating Adam
overflow or division-by-near-zero).

## Decision tree for iter 26

| abort.signal | abort.cause | extra signal | likely cause | iter 26 move |
| --- | --- | --- | --- | --- |
| `cohort_statistic` | `loss` | one `z_<spec>` >> 1, others normal | spec's literature target too tight OR arms drive model into unsupervised region | soft-clip that spec's z² (`min(z, 5)`) OR remove spec pending re-tuning |
| `cohort_statistic` | `loss` | many `z_*` huge | broad miscalibration, likely architectural | tighten band / reduce capacity / different from cohort |
| `trajectory_rollout/window` | `loss` | `coupling_loss` huge | coupling prior overshooting | reduce `coupling_prior_weight` OR remove specific prior |
| `trajectory_rollout/window` | `loss` | `verifier_loss` huge, others normal | verifier surrogate pathological on a specific window | inspect that patient/win_start, lower `verifier_loss_weight` |
| `trajectory_rollout/window` | `loss` | all components reasonable, raw `loss` is the killer | ODE integrator hitting numerical edge under fresh embedding | check `pred_traj` clipping; consider `RK45` → `Heun` step downgrade for that band |
| `gut_dose_sweep` / `insulin_sweep` | `loss` | `rank` huge, `mse` normal | hinge overflow when one species' rate becomes extreme | clamp pre-hinge differences |
| any | `grad` | `max_abs_grad_finite` huge, `n_inf_params=0` | isolated parameter exploding | per-spec output clamping in `MassActionModule.forward` |
| any | `grad` | `n_inf_params >> n_nan_params` | Adam denormal / division blowup | Adam `eps` 1e-8 → 1e-6, OR drop phase2-lr 0.0003 → 0.0001 |
| any | abort at `epoch < 50` | (Phase 1, no cohort/verifier/landmark/dose signals active yet) | Phase 1 destabilizes alone — would falsify iter-23 architecture independent of additive signals | reduce `--lr 0.003 → 0.001`, OR fall back to iter-22 architecture |

## Cross-iteration analysis tips

* The `recent_metrics` buffer captures per-signal time across the last 5
  epochs. If `time_*_s` for any signal halves between epochs E-1 and E,
  that signal is silently doing less work even with strict abort active —
  weight schedule mis-fire, empty embedding selection, etc.
* `last_good_pre_nan.pt` has the model state from the END of the last
  successful epoch. Load it with the iter-25 model topology and run:
  ```bash
  cd apps/pulse/engine
  .venv/bin/python -m pulse.diagnostics signal-balance \
      --checkpoint /tmp/iter25/last_good_pre_nan.pt --n-patients 20
  .venv/bin/python -m pulse.diagnostics cohort-ablation \
      --checkpoint /tmp/iter25/last_good_pre_nan.pt --sample-patients 4
  ```
  If a specific spec / head was already producing extreme outputs in the
  last-good state, the abort came after but the cause started earlier.

## How to resume after reboot

```bash
cd /Users/grovina/Projects/grovina/platform/.claude/worktrees/pulse

# gcloud is wired to pulse-iteration-runner SA via .claude/settings.local.json
# (env: GOOGLE_APPLICATION_CREDENTIALS + CLOUDSDK_ACTIVE_CONFIG_NAME=grovina-pulse).

# 1. Submit (commit-before-submit is enforced by the script).
bash apps/pulse/scripts/train-submit.sh --gcp

# 2. Tail build / job logs while it runs (copy build ID from script output).

# 3. After completion, check outcome.
LATEST=$(gsutil ls gs://grovina-pulse/training/jobs/ | tail -1 | awk -F/ '{print $(NF-1)}')
echo "Latest job: $LATEST"

# Did it abort?
if gsutil -q stat "gs://grovina-pulse/training/jobs/$LATEST/abort_diagnostics.json"; then
  echo "ABORTED — fetching diagnostics + last-good"
  mkdir -p /tmp/iter25 && \
    gsutil -o "GSUtil:parallel_process_count=1" cp \
      gs://grovina-pulse/training/jobs/$LATEST/{abort_diagnostics.json,last_good_pre_nan.pt,spec.json,meta.json} \
      /tmp/iter25/
  jq . /tmp/iter25/abort_diagnostics.json
  # Then follow the decision tree above.
else
  echo "Completed — fetching benchmark + delta + checkpoint"
  mkdir -p /tmp/iter25 && \
    gsutil -o "GSUtil:parallel_process_count=1" cp \
      gs://grovina-pulse/training/jobs/$LATEST/{benchmark-report.json,delta-vs-baseline.json,checkpoint.pt,spec.json} \
      /tmp/iter25/
  python3 -c "import json; r = json.load(open('/tmp/iter25/benchmark-report.json')); \
    print(f'gate.passed={r[\"gate\"][\"passed\"]} overall={r[\"overall_weighted_mape\"]:.4f} verifier={r[\"verifier\"][\"overall_score\"]:.4f}')"
  cd apps/pulse/engine && .venv/bin/python -m pytest tests/ -x -q
fi
```

Note: iter-23 / iter-24 checkpoints remain **not loadable** by iter-25's
model — same `nn.ModuleList` per-species head topology as iter 23/24
but if you've changed `hidden_dim` or anything in `modules/metabolic.py`
since iter 23 the state_dict keys/shapes will mismatch. Compare via
`benchmark-report.json` and `delta-vs-baseline.json`, not via direct
checkpoint loading.

## Code state (iter 25 changes ready to commit)

* `apps/pulse/engine/pulse/training/safe_step.py` — new. `safe_step`
  helper, `NaNTrainingAbort` exception, `_grad_stats` finiteness check.
* `apps/pulse/engine/pulse/training/__init__.py` — re-export
  `safe_step` and `NaNTrainingAbort`.
* `apps/pulse/engine/pulse/training/cohort_signal.py` — replaced
  inline `nan_skip` guard with `safe_step`; per-spec z² in extras.
* `apps/pulse/engine/pulse/training/dose_response_signal.py` — same
  refactor; predicted/target slope in extras.
* `apps/pulse/engine/pulse/training/gut_dose_sweep_signal.py` — same;
  mse/rank/auc components in extras.
* `apps/pulse/engine/pulse/training/insulin_sweep_signal.py` — same;
  full per-axis component breakdown in extras.
* `apps/pulse/engine/pulse/training/trajectory_signal.py` — same per-
  window; full per-window context in extras (patient/window/sub-loss).
* `apps/pulse/engine/pulse/train.py` — try/except NaNTrainingAbort
  around the epoch loop, rolling last-good checkpoint, recent-metrics
  buffer, abort artifact upload, exit code 42.
* `apps/pulse/train/cloudbuild.yaml` — RunTraining step surfaces the
  abort artifact in the build log before failing.
* `apps/pulse/train/spec.json` — iter-25 hypothesis + expectedEffect.
* `apps/pulse/engine/tests/test_safe_step.py` — 4 unit tests.

## Cloud Build / job IDs

| iter | job ID | build status | notes |
| ---- | ------ | ------------ | ---- |
| 21 | train-20260423T175447Z | FAILURE (gate) | overall=0.190, glu_mean=0.543 |
| 22 | train-20260424T071900Z | FAILURE (gate) | overall=0.193, glu_mean=0.566 |
| 23 | train-20260424T163240Z | FAILURE (NaN + timeout) | no checkpoint produced |
| 24 | train-20260425T153910Z | FAILURE (NaN-corrupt checkpoint) | textbook=0.108, all metrics NaN; build `4392b9bb-6b1b-43d5-bd32-79969537fe1c` |
| 25 | (pending) | (pending) | iter-25 = strict NaN abort + diagnostic dump |

## Pointer references (only read if needed)

* `apps/pulse/docs/iter24-handoff.md` — full detail on the iter-24
  stability re-run plan and why its skip-and-continue NaN guards were
  insufficient.
* `apps/pulse/docs/iter23-handoff.md` — per-species heads +
  glucose-gated insulin head architecture and the cohort-amplification
  hypothesis falsification.
* `apps/pulse/docs/training-runs.md` — **commit before submit**
  workflow, GCP wiring, train-submit.sh details.
* `apps/pulse/docs/prd.md` — top-level vision.
