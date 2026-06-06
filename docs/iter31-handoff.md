# Pulse Iter 31 — Resume Handoff

Self-contained resume point. Iter 31 fixes the 120-min/240-min training/probe
window mismatch that caused iter 30's structural setpoint to produce no
meaningful gradient on log_sg.

## Why iter 30 failed

| metric | iter 27 | iter 28 | iter 29 | iter 30 | threshold |
|--------|---------|---------|---------|---------|-----------|
| glucose_mape | 0.5710 | 0.7252 | 0.7126 | 0.5873 | 0.20 |
| hr_mape | — | — | 0.2077 | 0.2425 | 0.15 |
| temp_mape | — | — | — | 0.0199 ✓ | 0.02 |
| fasting_drift | 52.2 | 46.0 | 48.0 | 55.0 mg/dL | — |

**log_sg didn't learn**: `metabolic.log_sg` went from -3.912 to -3.754 over 80
epochs (sg: 0.0198 → 0.0232). Gradient was flowing but nearly zero. Root cause:
the fasting rollout trains on 120 min while the probe measures 240 min. At 120
min with sg≈0.02, glucose hasn't drifted far enough to create a strong gradient
— the optimizer sees no incentive to grow sg.

**temp_mape finally cleared**: 0.0199 < 0.02 threshold. One metric down.

## What iter 31 changes

### 1. Fasting rollout window 120 → 240 min

`train.py`: new `--fasting-stability-window` arg wired to
`FastingStabilitySignal(window_min=...)`.

`spec.json`: `--fasting-stability-window=240`.

At 240 min the full +55 mg/dL drift is visible during training. The gradient
through `log_sg` is proportional to glucose deviation at each step — with 120
min of additional drift this becomes the strong signal that iter 30 lacked.

### 2. log_sg init log(0.02) → log(0.1)

`metabolic.py`: `self.log_sg = nn.Parameter(torch.tensor(math.log(0.1)))`.

sg ≈ 0.10 at epoch 0 instead of 0.02. At +30 mg/dL (norm=+1) the correction
is 0.10 clearance units/min vs 0.023 in iter 30 — 4× stronger restoring force
from the very first epoch.

## What iter 31 does NOT change

- `fasting-stability-weight=0.10` — unchanged
- LR schedule (phase1=0.003, phase2=0.0003) — unchanged
- n_epochs=80, all other signal weights — unchanged
- Met coupling, GlucoseGatedInsulinHead — unchanged

## Success criteria

**fasting_drift < 15 mg/dL** — 240-min rollout directly trains the setpoint
against the observed drift; sg≈0.10 provides meaningful restoring force.

**glucose_mape < 0.40** — provisional milestone once resting is anchored.

**gate.passed=true** — glucose_mape ≤ 0.20, hr_mape ≤ 0.15,
overall_mape ≤ 0.16, temp_mape ≤ 0.02.

## Risk paths

| symptom | likely cause | iter 32 move |
|---------|--------------|--------------|
| fasting_drift still > 30 mg/dL | gradient still too weak at sg=0.10 | Make sg non-learnable (fixed 0.10) or raise fasting weight to 0.20 |
| Setpoint fights postprandial rise (mape worse) | sg=0.10 suppresses meal response | Reduce init to log(0.05); or clamp sg < 0.10 |
| NaN abort | longer 240-min rollout gradient | Follow iter-25 abort dump decision tree |
| hr_mape persists after glucose fixed | Independent failure | Add glucose to cardiovascular coupling |

## Cloud Build / job IDs

| iter | job ID | outcome |
|------|--------|---------|
| 27 | train-20260427T042129Z | glu_mape=0.571, drift=52 |
| 28 | train-20260428T063053Z | glu_mape=0.725, drift=46 |
| 29 | train-20260428T153634Z | glu_mape=0.713, drift=48 |
| 30 | train-20260429T085342Z | glu_mape=0.587, drift=55; sg frozen at 0.023 |
| 31 | train-20260430T062140Z | glu_mape=0.570 (−1.8%), hr=0.245, overall=0.199; gate still failing 3-of-3. Fasting at 14h essentially unchanged (95.0→95.0 mg/dL). OGTT cleared, but `insulin_dose_response` regressed to −0.95 (wrong sign), `meal_dose_response` pass-rate stuck at 0.33 (5/5 iters). |

## Iter 32 — review, not a tune

`apps/pulse/docs/iter32-review.md` — 10,000ft review of iters 27–31. Two
structural problems identified (HR has no metabolic coupling;
fasting-stability trains a regime disjoint from the failing post-meal
recovery check). Recommended next moves are structural (cardiovascular
wiring, post-meal-recovery signal) plus a knob-rationalisation ablation,
**not** another setpoint tune.

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

- `apps/pulse/engine/pulse/modules/metabolic.py` — log_sg init bumped to log(0.1)
- `apps/pulse/engine/pulse/train.py` — --fasting-stability-window arg added
- `apps/pulse/train/spec.json` — --fasting-stability-window=240
- `apps/pulse/docs/iter30-handoff.md` — iter 30 diagnostic
