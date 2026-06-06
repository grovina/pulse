# Pulse Iter 28 — Resume Handoff

Self-contained resume point. Iter 28 applies two changes on top of the
iter-27 architecture: (A) fasting stability weight 0.20 → 0.50 and (B)
insulin added to MetabolicModule coupling inputs.

## Why iter 27 failed the benchmark gate

Build `904bdbd6-1495-4d9f-90be-5c2761053448` (`train-20260427T042129Z`).
Training completed; benchmark ran via benchmark-rerun.sh.

| metric | value | threshold |
|--------|-------|-----------|
| glucose_mape | 0.5710 | 0.20 |
| hr_mape | 0.2444 | 0.15 |
| overall_weighted_mape | 0.2017 | 0.16 |

Delta vs iter 26 (`train-20260426T105353Z`):

| metric | iter 26 | iter 27 | Δ |
|--------|---------|---------|---|
| overall_mape | 0.2090 | 0.2017 | −0.007 |
| glucose_mape | 0.6078 | 0.5710 | −0.037 |
| hr_mape | 0.2409 | 0.2444 | +0.004 |
| verifier_overall | 0.828 | 0.817 | −0.011 |
| verifier_coupling | ~0.832 | 0.745 | **−0.087** |
| textbook_pass_rate | 0.760 | 0.821 | +0.060 |

`new_probe.fasting_drift.drift_mg_dl`: **52.2 mg/dL** (was 57.3 in iter 26).
FastingStabilitySignal at weight=0.20 is having effect but is clearly
insufficient — drift needs to reach < 15 mg/dL.

### Root cause analysis

Two iter-27 risk paths triggered:

**R2 (weight insufficient):** `fst=` decreased during training (the signal
is working) but drift only moved from 57→52 mg/dL. The weight=0.20 cannot
overcome the production-vs-clearance imbalance in the metabolic module.
Fix: bump to 0.50.

**R5 (coupling gap):** Coupling verifier category dropped −0.087. The meal
dose-response (90g vs 30g carbs: 0.41 mg/dL difference vs threshold 5.0
mg/dL) and insulin dose-response (negative delta vs threshold +2.0) both
still failing. The metabolic module only sees gut outputs and cortisol —
it cannot directly observe insulin and thus cannot use insulin as a clearance
signal. With the fasting stability signal now providing some glucose anchoring,
the coupling gap (insulin not in met_coupling) became the visible bottleneck.
Fix: add insulin to met_coupling (Fix B from iter-26 analysis).

## What iter 28 changes

### 1. Fasting stability weight 0.20 → 0.50

`apps/pulse/train/spec.json`: `--fasting-stability-weight=0.50`

No change to the signal itself — same 120-min fasting rollout from
NORM_CENTER at zero embedding. 2.5× gradient pressure on the glucose
setpoint. Expected to bring drift from ~52 toward <15 mg/dL.

### 2. Insulin added to MetabolicModule coupling (Fix B)

`apps/pulse/engine/pulse/modules/metabolic.py`:
```python
# Before: gut outputs (4) + cortisol (1) = 5
_N_COUPLING = GUT_OUTPUT_DIM + 1

# After: gut outputs (4) + cortisol (1) + insulin (1) = 6
_N_COUPLING = GUT_OUTPUT_DIM + 2
```

`apps/pulse/engine/pulse/model.py`:
```python
# Before:
met_coupling = torch.cat([gut_out_batch, cortisol], dim=-1)

# After:
met_coupling = torch.cat([gut_out_batch, cortisol, insulin], dim=-1)
```

The metabolic module can now observe insulin at each step of the ODE
integration. This is the physiologically correct mechanism: insulin
drives glucose uptake (muscles, adipose), suppresses hepatic glucose
output, and mediates antilipolysis (FFA↓) and glucagon suppression.

With insulin visible, the per-species heads for glucose and its coupled
markers (FFA, glucagon, ghrelin) can learn insulin-mediated dynamics
directly rather than inferring them through gut outputs alone.

**Architecture note:** This is a warm-start-incompatible change — the
MetabolicModule weight matrices are resized. Fresh training run only;
do NOT attempt to load the iter-27 checkpoint.

## What iter 28 does NOT change

- No architecture change beyond met_coupling dimension
- Same per-species heads + GlucoseGatedInsulinHead
- LR unchanged: phase1=0.003, phase2=0.0003
- All other signal weights unchanged
- trajectory_band, Adam epsilon, n_epochs unchanged

## Success criteria

**Primary:** `new_probe.fasting_drift.drift_mg_dl` < 15 mg/dL (was 52.2 in iter 27)

**Benchmark gate:** `glucose_mape` ≤ 0.20, `hr_mape` ≤ 0.15,
`overall_weighted_mape` ≤ 0.16 → `gate.passed=true`

**Coupling recovery:**
- `verifier_coupling` ≥ 0.832 (iter 26 level, before the regression)
- `meal_dose_response`: 90g vs 30g glucose delta ≥ 5 mg/dL
- `insulin_dose_response` positive (> 2 µU/mL)
- `glucagon_suppressed_postprandial` passing
- `ghrelin_suppressed_after_feeding` passing

## Risk paths

| symptom | likely cause | iter 29 move |
|---------|--------------|--------------|
| fst= → < 0.04 but glucose_mape stays > 0.3 | Drift fixed; meal-response has additional gap | Add explicit dose-response supervision on carb scaling |
| fst= still > 0.5 after epoch 50 | 0.50 still insufficient vs production imbalance | Structural setpoint: explicit glucose homeostasis layer in MetabolicModule |
| Coupling verifier still regressing | Insulin in met_coupling insufficient; glucose not clearing via insulin | Audit cardiovascular coupling — glucose may need to enter cvs_coupling |
| Stability signal vs trajectory tug-of-war (trajectory loss > 2× iter 27) | 0.50 weight too aggressive, fights meal glucose rise | Reduce to 0.30 OR add gut_output gate (only penalize steps where gut_output[:carbs] < 0.01) |
| hr_mape still > 0.15 after glucose fixes | HR failure is independent of glucose | Next target: add cortisol or activity coupling to cardiovascular module |
| NaN abort | New autograd singularity at insulin coupling site | Follow iter-25 abort dump decision tree |

## Cloud Build / job IDs

| iter | job ID | build status | outcome |
|------|--------|--------------|---------|
| 21 | train-20260423T175447Z | FAILURE (gate) | overall=0.190, glu_mean=0.543 |
| 22 | train-20260424T071900Z | FAILURE (gate) | overall=0.193, glu_mean=0.566 |
| 23 | train-20260424T163240Z | FAILURE (NaN + timeout) | no checkpoint produced |
| 24 | train-20260425T153910Z | FAILURE (NaN-corrupt checkpoint) | textbook=0.108, all metrics NaN |
| 25 | train-20260426T062232Z | FAILURE (strict abort at Phase 2 epoch 61) | abort dump → _safe_corr_torch |
| 26 | train-20260426T105353Z | FAILURE (gate) | stability PASSED, glu_mape=0.608, drift=+57 mg/dL |
| 27 | train-20260427T042129Z | FAILURE (gate) | glu_mape=0.571, drift=+52 mg/dL, coupling −0.087 |
| 28 | train-20260427T204704Z | in progress | — |

## How to resume after reboot

```bash
cd /Users/grovina/Projects/grovina/platform/.claude/worktrees/pulse

# 1. Check build status
gsutil ls gs://grovina-pulse/training/jobs/<JOB_ID>/

# 2. Read results once benchmark-report.json exists
gsutil cat gs://grovina-pulse/training/jobs/<JOB_ID>/benchmark-report.json \
  | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(f'gate.passed={d[\"gate\"][\"passed\"]}')
print(f'overall={d[\"overall_weighted_mape\"]:.4f}')
print(f'glucose_mape={d[\"per_marker\"][\"glucose\"][\"mean_mape\"]:.4f}')
print(f'hr_mape={d[\"per_marker\"][\"hr\"][\"mean_mape\"]:.4f}')
[print(' -', f) for f in d['gate']['failures']]
"

# 3. If benchmark didn't run (Cloud Run timeout), rerun it:
bash apps/pulse/scripts/benchmark-rerun.sh <JOB_ID>
```

## Pointer references

- `apps/pulse/engine/pulse/modules/metabolic.py` — _N_COUPLING changed to GUT_OUTPUT_DIM+2
- `apps/pulse/engine/pulse/model.py` — met_coupling now includes insulin
- `apps/pulse/engine/pulse/training/fasting_stability_signal.py` — unchanged
- `apps/pulse/train/spec.json` — --fasting-stability-weight=0.50
- `apps/pulse/docs/iter27-handoff.md` — iter-27 diagnostic + risk table
