# Pulse Iter 29 — Resume Handoff

Self-contained resume point. Iter 29 is a single-knob change: drop
`--fasting-stability-weight` from 0.50 (iter 28) to **0.30**. All other
code is identical to iter 28 (insulin in met_coupling, per-species heads,
GlucoseGatedInsulinHead).

## Why iter 28 failed the benchmark gate

Build `ea3154d7-2d70-40d5-8b8c-89cea74ccbb4` (`train-20260428T063053Z`).
Training completed; benchmark ran inline.

| metric | iter 27 | iter 28 | Δ | threshold |
|--------|---------|---------|---|-----------|
| glucose_mape | 0.5710 | **0.7252** | +0.154 | 0.20 |
| hr_mape | 0.2444 | **0.2204** | −0.024 | 0.15 |
| temp_mape | 0.0198 | **0.0216** | +0.002 | 0.02 |
| overall_mape | 0.2017 | **0.2238** | +0.022 | 0.16 |
| verifier_overall | 0.817 | **0.844** | +0.027 | — |
| verifier_coupling | 0.745 | **0.720** | −0.025 | — |
| verifier_meal | — | **0.791** | +0.076 | — |
| fasting_drift | 52.2 mg/dL | **46.0 mg/dL** | −6.2 | — |

### Root cause: weight=0.50 too aggressive

The risk table entry that triggered: *"fst= drops but trajectory loss
degrades → stability signal fights trajectory signal on glucose rises
during meals → Reduce weight to 0.10 OR add a gate on meal windows."*

Weight=0.50 gave the fasting stability signal more gradient pressure than
any other signal. The model learned to globally suppress glucose dynamics
to minimize the MSE drift penalty. Evidence:

- `meal_dose_response glucose_dose_response`: 90g vs 30g carb meal
  produced only 0.15 mg/dL difference (was 0.41 in iter 27; threshold 5.0).
  The model barely lets glucose rise for any meal size.
- `dietary_carbohydrate_meal_flow` lost two checks that iter 27 had fixed:
  `ffa_suppressed_antilipolysis` and `glucagon_suppressed_postprandial`
  — postprandial dynamics suppressed globally.
- `overnight_fast insulin_low`: 19.0 → 22.2 µU/mL (worse — insulin stays
  elevated because the model suppresses glucose clearance everywhere).

### Fix B (insulin in met_coupling) confirmed working

Despite the glucose regression, Fix B produced a clean result:
- `insulin_dose_response`: −0.068 (iter 27) → +0.343 (iter 28). Insulin
  now correctly scales with carb dose. This is the direct effect of the
  metabolic module having insulin as a coupling input.
- Verifier_meal improved +0.076 — the model's meal physiology improved
  overall, obscured by the glucose suppression.

Fix B is kept in iter 29. The regression is from the weight, not the Fix.

## What iter 29 changes

**Only `--fasting-stability-weight`: 0.50 → 0.30.**

No code changes — identical architecture to iter 28:
- Insulin in met_coupling (`_N_COUPLING = GUT_OUTPUT_DIM + 2`)
- Per-species heads + GlucoseGatedInsulinHead
- `insulin_sweep_signal._sweep_rates` coupling fixed

## Success criteria

**Fasting drift < 40 mg/dL** — weight=0.30 should land between iter 27
(52 mg/dL at weight=0.20) and iter 28 (46 mg/dL at weight=0.50). If the
relationship isn't monotone, that reveals the signal is ineffective at any
fixed weight and the architecture fix (R1) is needed.

**glucose_mape < 0.571** — recover to at least iter-27 level and continue
toward the 0.20 threshold.

**meal_dose_response glucose_dose_response > 5 mg/dL** — 90g vs 30g
carb meal should produce distinguishable glucose peaks.

**Benchmark gate pass**: glucose_mape ≤ 0.20, hr_mape ≤ 0.15,
overall_mape ≤ 0.16, temp_mape ≤ 0.02.

## Risk paths

| symptom | likely cause | iter 30 move |
|---------|--------------|--------------|
| glucose_mape still > 0.5 at weight=0.30 | Fasting signal cannot isolate fasting vs meal dynamics at any fixed weight | Architectural setpoint: add explicit Gb restoring force (structural term) to MetabolicModule |
| Fasting drift > 46 mg/dL (no improvement from iter 28) | 0.30 too weak, <0.50 is in the suppression zone | Try weight=0.40 + gut_output gate in signal (penalize only when gut_output_carbs < 0.01) |
| glucose_mape < 0.571 but > 0.3 | Partial recovery — weight 0.30 helps but meal-response has additional gap | Add explicit dose-response supervision per-carb-dose on the trajectory signal |
| temp_mape stays > 0.02 | New independent failure; thermoreg coupling insufficient | Investigate thermoreg module — possibly add glucose or insulin to thermoreg coupling |
| hr_mape stays > 0.20 | HR improvement in iter 28 (+0.024) is encouraging but threshold is 0.15 | After glucose fixed: add glucose or activity to cardiovascular coupling |
| NaN abort | New autograd singularity | Follow iter-25 abort dump decision tree |

## Cloud Build / job IDs

| iter | job ID | build status | outcome |
|------|--------|--------------|---------|
| 25 | train-20260426T062232Z | FAILURE (NaN abort) | → _safe_corr_torch fix |
| 26 | train-20260426T105353Z | FAILURE (gate) | glu_mape=0.608, drift=+57 |
| 27 | train-20260427T042129Z | FAILURE (gate) | glu_mape=0.571, drift=+52 |
| 28 | train-20260428T063053Z | FAILURE (gate) | glu_mape=0.725, drift=+46; insulin DR fixed |
| 29 | TBD | in progress | — |

## How to resume after reboot

```bash
cd /Users/grovina/Projects/grovina/platform/.claude/worktrees/pulse

# Check status
gsutil ls gs://grovina-pulse/training/jobs/<JOB_ID>/

# Read results
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

# Get fasting drift from delta report
gsutil cat gs://grovina-pulse/training/jobs/<JOB_ID>/delta-vs-baseline.json \
  | python3 -c "
import json, sys
d = json.load(sys.stdin)
fd = d['new_probe']['fasting_drift']
print(f'fasting_drift={fd[\"drift_mg_dl\"]:.1f} mg/dL')
print(f'insulin_dr={[c[\"new_value\"] for c in d[\"check_transitions\"] if c[\"name\"]==\"insulin_dose_response\"]}')
"

# If benchmark didn't run (Cloud Run timeout):
bash apps/pulse/scripts/benchmark-rerun.sh <JOB_ID>
```

## Pointer references

- `apps/pulse/engine/pulse/modules/metabolic.py` — _N_COUPLING=GUT_OUTPUT_DIM+2
- `apps/pulse/engine/pulse/model.py` — met_coupling includes insulin
- `apps/pulse/engine/pulse/training/insulin_sweep_signal.py` — coupling fixed
- `apps/pulse/train/spec.json` — --fasting-stability-weight=0.30
- `apps/pulse/docs/iter28-handoff.md` — iter-28 diagnostic
