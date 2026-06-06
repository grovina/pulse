# Pulse Iter 32 — Resume Handoff

Self-contained resume point. Iter 32 is a **structural pivot, not a setpoint
tune.** It bundles three changes (32a/32b/32c) in one training run because
they touch orthogonal axes; the per-bench-check breakdown will give us
attribution without paying for three separate Cloud Run jobs.

For the diagnostic that motivated this iter, see `iter32-review.md`.

## What this iter changes

### 32a — Cardiovascular module gets metabolic couplings

`pulse/modules/cardiovascular.py`: `_N_COUPLING = 2 → 4`. Coupling tuple is
now `[cortisol, temperature, glucose, insulin]`.

`pulse/model.py:222-223`: routes `glucose` and `insulin` (norm_state slices)
into the cardio coupling tensor.

**Why:** `hr_mape` was stuck at ~0.245 across iters 27-31. With cortisol +
temperature only, postprandial HR rise was structurally unreachable from
the meal axis — no glucose-side weight tune could move HR by construction.

### 32b — Replace fasting-stability with postprandial-recovery signal

`pulse/training/postprandial_recovery_signal.py`: new `PostprandialRecoverySignal`
mirrors the dietary_carbohydrate_meal_flow bench protocol exactly:
- start at 08:00 (start_time_minutes=480) from NORM_CENTER, zero embedding
- single mixed meal at minute 30: 50g carbs / 12g fats / 18g proteins
- duration 480 min (= bench)
- baseline = mean glucose over minutes [0, 25)
- recovery = mean glucose over minutes [420, 475)
- loss = ((recovery − baseline) / glc_scale) ** 2

`spec.json`: `--fasting-stability-weight=0.10 → 0.0` (off),
`--postprandial-recovery-weight=0.10` (on).

**Why:** the failing bench check `glucose_returns_near_baseline` measures
*post-meal* recovery; the fasting-stability signal trains a *no-meal*
regime. The model has never received gradient on long-horizon post-meal
recovery — exactly the failure mode the bench probes. Iter 31's
fst term grew during training but the bench metric didn't move,
confirming the regime was disjoint.

### 32c — Knob-rationalisation ablation (zero out three weights)

`spec.json`:
- `--gut-loss-weight=0.10 → 0.0` (duplicative with `gut-dose-sweep-auc-weight=5.0` since iter 18)
- `--dose-response-weight=0.20 → 0.0` (loss stuck at ~8.0 across iters 28-30, no movement)
- `--landmark-weight=0.15 → 0.0` (set pre-iter12, never named as a lever or failure since)

**Why:** these are flags the iter32-review knob-rationalisation pass
flagged as "unclear" — old, untouched, with no defensible attribution
today. If the bench doesn't materially regress on the checks they
nominally support, they were vestigial and we drop them permanently.
If something specific regresses, that flag was load-bearing after all
and we restore just that one — and now we have a defensible
attribution.

## What iter 32 does NOT change

- `log_sg` init stays at log(0.10) (iter 31 setting). The setpoint is
  not the lever this iter.
- `--cohort-statistic-weight=0.15`, `--insulin-sweep-weight=0.30`,
  `--gut-dose-sweep-weight=0.10` (with `auc-weight=5.0`) — all
  load-bearing per the review.
- `--n-default-patients=8`, `--trajectory-band-default=0.05` — both
  load-bearing zero-embedding fixes; touching either reopens iter9-era
  pathology.
- LR schedule, epoch counts, hidden dim, n_patients — unchanged.

## Success criteria — per sub-change

Bundled run, but each sub-change has a separable bench signal so we can
attribute outcomes:

**32a (cardio coupling):**
- `hr_mape` 0.245 → ~0.18 (target: 0.15 gate). The only path to HR
  improvement; can only be 32a's doing.
- `sbp_mape` / `dbp_mape` should not regress materially (currently
  0.082 / 0.075). If they get worse, the coupling broke BP-side
  dynamics; revert coupling to glucose-only.

**32b (postprandial recovery):**
- `dietary_carbohydrate_meal_flow.glucose_returns_near_baseline` 60 →
  <20 mg/dL. Was 60-70 mg/dL across all 5 iters in 27-31.
- `glucose_mape` should improve via better postprandial trajectories.
  Iter 31 was 0.570; if 32b lands cleanly we'd hope for 0.45-0.50.
- `insulin_dose_response` may shift: was −0.95 in iter 31. With the
  setpoint no longer doing the postprandial clearing, the optimizer
  has more pressure to make insulin scale with meals.

**32c (vestigial-knob ablation):**
- `meal_dose_response.glucose_dose_response` (was 0.19): if this drops
  toward 0, `dose-response-weight` was load-bearing and we restore it.
- `glucose_rises_after_meal` / landmark-driven checks: if they regress,
  `landmark-weight` was load-bearing — restore to 0.15.
- Trajectory MSE on gut: if it regresses, `gut-loss-weight` was still
  contributing — restore to 0.10.

**Aggregate gate:** glucose_mape ≤ 0.20, hr_mape ≤ 0.15,
overall_mape ≤ 0.16. Iter 31 was 0.570 / 0.245 / 0.199 — clearing
hr alone would put overall close to 0.16.

## Risk paths

| symptom | likely cause | iter 33 move |
|---------|--------------|--------------|
| hr_mape unchanged | cardio coupling not enough; need HR-specific cohort spec or landmark term | Add HR landmark / postprandial HR cohort spec |
| ppr loss explodes on phase-1 random model | safe_step catches it, but phase 1 may slow | Lower ppr weight to 0.05; warmup window |
| sbp/dbp regress | metabolic coupling broke BP-side dynamics | Revert cardio coupling to cortisol+temp+glucose only (drop insulin) |
| All 3 of 32c regress | one or more flags was load-bearing | Bench breakdown identifies which; restore that one |
| glucose_mape still ~0.57 | structure problem deeper than these three changes | Run Bergman minimal-model parity check before any further tuning |
| NaN abort | longer 480-min meal-bearing rollout in ppr signal | Follow iter-25 abort dump tree; consider half-window=240 |

## Cloud Build / job IDs

| iter | job ID | outcome |
|------|--------|---------|
| 27 | train-20260427T042129Z | glu_mape=0.571, hr=0.244, drift=52 |
| 28 | train-20260428T063053Z | glu_mape=0.725, hr=0.220, drift=46 |
| 29 | train-20260428T153634Z | glu_mape=0.713, hr=0.208, drift=48 |
| 30 | train-20260429T085342Z | glu_mape=0.587, hr=0.243, drift=55; sg frozen |
| 31 | train-20260430T062140Z | glu_mape=0.570, hr=0.245, overall=0.199; meal_dose_response 0.33 (5/5 iters) |
| 32 | TBD | bundled 32a/b/c |

## How to resume after reboot

```bash
cd /Users/grovina/Projects/grovina/platform

# Check GCS
gcloud storage ls gs://grovina-pulse/training/jobs/<JOB_ID>/

# Read results
gcloud storage cat gs://grovina-pulse/training/jobs/<JOB_ID>/benchmark-report.json \
  | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(f'glucose_mape={d[\"per_marker\"][\"glucose\"][\"mean_mape\"]:.4f}')
print(f'hr_mape={d[\"per_marker\"][\"hr\"][\"mean_mape\"]:.4f}')
print(f'overall={d[\"overall_weighted_mape\"]:.4f}')
for s in d['textbook_scenarios']:
    if s['name'] in ('meal_dose_response','dietary_carbohydrate_meal_flow','overnight_fast'):
        print(f'  {s[\"name\"]}: pass_rate={s[\"pass_rate\"]:.2f}')
        for c in s['checks']:
            if not c['passed']:
                print(f'    [FAIL] {c[\"name\"]}: v={c[\"value\"]:.4f} thr={c[\"threshold\"]}')
"

# Cloud Run timeout recovery (if benchmark phase gets cut):
bash apps/pulse/scripts/benchmark-rerun.sh <JOB_ID>
```

## Pointer references

- `apps/pulse/docs/iter32-review.md` — the diagnostic motivating this iter
- `apps/pulse/engine/pulse/modules/cardiovascular.py` — `_N_COUPLING=4`
- `apps/pulse/engine/pulse/model.py:222-223` — cardio coupling routing
- `apps/pulse/engine/pulse/training/postprandial_recovery_signal.py` — new signal
- `apps/pulse/engine/pulse/knowledge/textbook_scenarios/flow_story_protocol.py:35-36`
  — bench recovery window (training signal mirrors these constants)
- `apps/pulse/docs/iter31-handoff.md` — prior iteration
