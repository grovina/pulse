# Pulse Iter 30 — Resume Handoff

Self-contained resume point. Iter 30 applies a structural glucose setpoint
to MetabolicModule, reverts insulin from met_coupling, and drops the fasting
stability signal weight to 0.10. Signal-based drift correction is exhausted;
this is the architectural fix.

## Why iters 27/28/29 all failed

Three iterations of signal-weight tuning hit the same wall:

| iter | fst_weight | insulin_coupling | glucose_mape | fasting_drift |
|------|-----------|-----------------|-------------|--------------|
| 27 | 0.20 | no | 0.5710 | 52.2 mg/dL |
| 28 | 0.50 | yes | 0.7252 | 46.0 mg/dL |
| 29 | 0.30 | yes | 0.7126 | 48.0 mg/dL |

**Signal ceiling:** Weights 0.20, 0.30, and 0.50 all produce 46–52 mg/dL
fasting drift. The 120-min training rollout can't anchor the 240-min probe.

**Insulin double-counting:** Adding insulin to met_coupling (iters 28/29)
consistently made glucose_mape worse (0.57 → 0.71–0.73). Root cause:
insulin appears twice in each species head's input — once in
`norm_state[:, met_idx]` (position 1) and once in `coupling`. The resulting
gradient instability shows up as insulin dose-response flipping between
+0.343 (iter 28) and −0.070 (iter 29) with no architecture change.

**Dose-response plateau:** `dose=` loss stuck at ~8.0 every Phase-2 epoch
across both iters 28 and 29 — neither weight change moved it.

## What iter 30 changes

### 1. Structural glucose setpoint in MetabolicModule

`apps/pulse/engine/pulse/modules/metabolic.py` — new learnable parameter
`log_sg` and `forward()` override:

```python
self.log_sg = nn.Parameter(torch.tensor(math.log(0.02)))

def forward(self, state, coupling, external, embedding, time_features):
    rates = super().forward(state, coupling, external, embedding, time_features)
    sg = nn.functional.softplus(self.log_sg)
    rates_out = rates.clone()
    rates_out[..., _GLUCOSE_IDX] = rates_out[..., _GLUCOSE_IDX] - sg * state[..., _GLUCOSE_IDX]
    return rates_out
```

This mirrors the knowledge model's `Sg*(G-Gb)` clearance term. The
correction is zero at NORM_CENTER (normalized glucose = 0 at Gb=95 mg/dL)
and grows linearly with deviation — guaranteed restoring force regardless of
what the learned production/consumption heads settle at. `sg` is learnable
so the model can tune the strength during training.

Init: `sg = softplus(log(0.02)) ≈ 0.02`. At G=Gb+30 mg/dL (norm=+1), this
provides ~0.02 clearance units/min, comparable to the base `cons_scale[0]`.

### 2. Insulin removed from met_coupling

`metabolic.py`: `_N_COUPLING = GUT_OUTPUT_DIM + 1` (back to 5).
`model.py`: `met_coupling = torch.cat([gut_out_batch, cortisol], dim=-1)`.
`insulin_sweep_signal.py`: coupling reverted to `[zero_gut, cortisol]`.

### 3. Fasting stability weight 0.30 → 0.10

`spec.json`: `--fasting-stability-weight=0.10`. Structural term handles the
bulk; signal becomes a minor regularizer for the zero-embedding specifically.

## What iter 30 does NOT change

- Per-species heads + GlucoseGatedInsulinHead — unchanged
- LR schedule (phase1=0.003, phase2=0.0003) — unchanged
- All other signal weights — unchanged
- n_epochs=80, trajectory_band, Adam epsilon — unchanged

Architecture change (new `log_sg` parameter + forward override): fresh
training run. Cannot warm-start from iter 27/28/29 checkpoints.

## Success criteria

**Fasting drift < 15 mg/dL** — structural setpoint guarantees restoring
force; the steady-state drift will be where `prod ≈ sg*(G-Gb)`.

**glucose_mape < 0.40** — provisional milestone. Once resting glucose is
anchored, trajectory errors should drop substantially. Target remains 0.20.

**dose-response recovering** — `dose=` loss should start decreasing in
Phase 2 (was flatlined at ~8.0 in iters 28/29).

**gate.passed=true** — glucose_mape ≤ 0.20, hr_mape ≤ 0.15,
overall_mape ≤ 0.16, temp_mape ≤ 0.02.

## Risk paths

| symptom | likely cause | iter 31 move |
|---------|--------------|--------------|
| fasting_drift still > 30 mg/dL | log_sg gradient not flowing / sg too small | Check that log_sg appears in optimizer param groups; init at log(0.1) |
| glucose_mape < 0.5 but dose-response still fails | Setpoint anchors rest; meal response has independent gap | Increase dose-response weight from 0.20 to 0.40 |
| Setpoint fights postprandial rise (mape worse during meals) | sg too high, suppresses meals | Reduce init to log(0.005); or clamp sg < 0.05 |
| NaN abort | log_sg / safe_step interaction | Follow iter-25 abort dump decision tree |
| hr_mape persists after glucose fixed | Independent failure | Add glucose to cardiovascular coupling |
| temp_mape persists > 0.02 | thermoreg coupling insufficient | Investigate thermoreg module |

## Cloud Build / job IDs

| iter | job ID | outcome |
|------|--------|---------|
| 27 | train-20260427T042129Z | glu_mape=0.571, drift=52 |
| 28 | train-20260428T063053Z | glu_mape=0.725, drift=46; insulin DR fixed |
| 29 | train-20260428T153634Z | glu_mape=0.713, drift=48; DR regressed |
| 30 | train-20260429T085342Z | glu_mape=0.587, drift=55; sg frozen (0.023) — window mismatch |

## How to resume after reboot

```bash
cd /Users/grovina/Projects/grovina/platform/.claude/worktrees/pulse

# Check GCS
gsutil ls gs://grovina-pulse/training/jobs/<JOB_ID>/

# Read results
gsutil cat gs://grovina-pulse/training/jobs/<JOB_ID>/benchmark-report.json \
  | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(f'gate.passed={d[\"gate\"][\"passed\"]}')
print(f'glucose_mape={d[\"per_marker\"][\"glucose\"][\"mean_mape\"]:.4f}')
print(f'hr_mape={d[\"per_marker\"][\"hr\"][\"mean_mape\"]:.4f}')
[print(' -', f) for f in d['gate']['failures']]
"

gsutil cat gs://grovina-pulse/training/jobs/<JOB_ID>/delta-vs-baseline.json \
  | python3 -c "
import json, sys
d = json.load(sys.stdin)
fd = d['new_probe']['fasting_drift']
print(f'fasting_drift={fd[\"drift_mg_dl\"]:.1f} mg/dL')
"

# Cloud Run timeout recovery:
bash apps/pulse/scripts/benchmark-rerun.sh <JOB_ID>
```

## Pointer references

- `apps/pulse/engine/pulse/modules/metabolic.py` — log_sg + forward override
- `apps/pulse/engine/pulse/model.py` — met_coupling reverted to [gut, cortisol]
- `apps/pulse/engine/pulse/training/insulin_sweep_signal.py` — coupling reverted
- `apps/pulse/train/spec.json` — fasting-stability-weight=0.10
- `apps/pulse/docs/iter29-handoff.md` — iter 29 diagnostic
