# Pulse Iter 32 — 10,000ft review

Stepping back from the per-iter weight-tuning loop. After 5 iterations (27–31)
of glucose_mape bouncing in [0.57, 0.73] with no monotonic improvement, the
question is no longer "which weight do we tune next" but "are we tuning the
right thing at all."

This document is the diagnostic that should precede any further training. It
is **not** an iter32 spec. The next training run should be a deliberate
response to the conclusions here.

## Bottom line

Two structural problems are masquerading as tuning problems, and a third
structural problem (knob accumulation) is making them harder to see:

1. **`hr_mape` is unreachable from anything we've been tuning.**
   `CardiovascularModule._N_COUPLING = 2`, accepting only cortisol +
   temperature. No glucose, insulin, gut, or lactate inputs. There is also
   no HR-specific cohort spec or landmark term. The *only* gradient path to
   HR is per-step trajectory MSE on the cold model. Five iterations of
   glucose-side tuning cannot move HR by construction. Confirmed in
   `cardiovascular.py:14-18` and zero hits across `knowledge/cohorts/` for
   HR-targeted terms.

2. **The fasting-stability signal trains a regime disjoint from the failing
   bench check.** Iter 31's headline change (window 120→240 min) was meant
   to anchor fasting glucose. But the no-meal 240-min training rollout and
   the *post-meal 420–475-min recovery* bench check (`knowledge/textbook_scenarios/flow_story_protocol.py:35-36`)
   are different dynamical regimes. The trained model never sees the long
   post-meal tail. This explains why iter 31's in-training `fst` term grew
   0.77→1.22 while the bench fasting metric (95.0→95.0 mg/dL at 14h) and
   `glucose_returns_near_baseline` (60 mg/dL above baseline at end of flow,
   threshold 8) barely moved.

3. **The flag set has accumulated cruft that's hiding what's actually
   load-bearing.** Of the ~30 flags in `trainExtraArgs`, ~10 equal their
   `train.py` default and are carried for documentation only; another
   handful (`landmark-weight`, `dose-response-weight`,
   `meal-window-bias`) were set pre-iter12 and have not been the subject
   of any iteration since. We can't argue cleanly about iter32 until we
   know which knobs are doing real work.

The recommended next move is **not** another weight tune. It's two scoped
structural changes (HR coupling + post-meal recovery training signal) and
a knob-rationalisation ablation, before any further setpoint tuning.

## Bench across iters 27–31

| metric          | 27     | 28     | 29     | 30     | 31     | gate |
|-----------------|--------|--------|--------|--------|--------|------|
| glucose_mape    | 0.5710 | 0.7252 | 0.7126 | 0.5873 | 0.5697 | 0.20 |
| hr_mape         | 0.2444 | 0.2204 | 0.2077 | 0.2425 | 0.2445 | 0.15 |
| temp_mape       | 0.0198 | 0.0216 | 0.0211 | 0.0199 | 0.0199 | 0.02 |
| overall_mape    | 0.2017 | 0.2238 | 0.2200 | 0.2049 | 0.1992 | 0.16 |
| verifier        | 0.818  | 0.844  | 0.810  | 0.815  | 0.826  | —    |
| textbook        | 0.821  | 0.789  | 0.789  | 0.776  | 0.805  | —    |

| scenario pass-rate              | 27   | 28   | 29   | 30   | 31   |
|---------------------------------|------|------|------|------|------|
| OGTT                            | 0.80 | 0.80 | 0.80 | 0.60 | 0.80 |
| overnight_fast                  | 0.83 | 0.83 | 0.83 | 0.83 | 0.83 |
| **meal_dose_response**          | 0.33 | 0.33 | 0.33 | 0.33 | 0.33 |
| dietary_carbohydrate_meal_flow  | 0.78 | 0.56 | 0.56 | 0.67 | 0.67 |
| exercise_bout                   | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| sleep_wake_transition           | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| cortisol_awakening_response     | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

**meal_dose_response has been at 0.33 for every iteration in this window.**
Three checks underneath it — and they're the most damning evidence that we're
not actually learning what we think we're learning.

## Persistent failing checks (failed in ≥3/5 iters)

```
  [meal_dose_response] glucose_dose_response       threshold=5.0
      iter27 ✗ v=0.4111   iter28 ✗ v=0.1519   iter29 ✗ v=-0.0485
      iter30 ✗ v=0.2111   iter31 ✗ v=0.1885

  [meal_dose_response] insulin_dose_response       threshold=2.0
      iter27 ✗ v=-0.0676  iter28 ✗ v=0.3428   iter29 ✗ v=-0.0701
      iter30 ✗ v=0.0078   iter31 ✗ v=-0.9548

  [dietary_carb] glucose_returns_near_baseline     threshold=8.0
      iter27 ✗ v=69.19    iter28 ✗ v=68.63    iter29 ✗ v=65.73
      iter30 ✗ v=69.82    iter31 ✗ v=60.07

  [overnight_fast] insulin_low                     threshold=15.0
      iter27 ✗ v=19.00    iter28 ✗ v=22.22    iter29 ✗ v=21.22
      iter30 ✗ v=18.86    iter31 ✗ v=17.47

  [OGTT] glucagon_suppressed                       threshold=0.0
      iter27 ✗ v=-0.081   iter28 ✗ v=-0.003   iter29 ✗ v=-0.003
      iter30 ✗ v=-0.057   iter31 ✗ v=-0.006

  [dietary_carb] ghrelin_suppressed_after_feeding  threshold=0.0
      iter27 ✗ v=-0.0067  iter28 ✗ v=-0.0066  iter29 ✗ v=-0.0064
      iter30 ✗ v=-0.0067  iter31 ✗ v=-0.0064
```

What this tells us:

- **`glucose_dose_response` is moving randomly, not converging** (0.41 →
  0.15 → −0.05 → 0.21 → 0.19). That's the signature of an undertrained signal,
  not an over-trained one.
- **`insulin_dose_response` is wrong-signed in 4 of 5 iters** and gets worse
  the more we tune. Iter 31 hit −0.95 — the model is now *strongly*
  inversely-dosing insulin. This is the smoking gun for "the setpoint is
  doing the clearing work."
- **`glucose_returns_near_baseline` improved 13% across 5 iters** while we've
  changed almost everything around it. That's noise, not learning.
- **`insulin_low` after fasting trends down monotonically** (19→17.5) but
  isn't going to clear 15 at this rate.
- **The two `glucagon_suppressed` checks** are tiny-magnitude failures
  (values within ±0.08 of threshold 0). Likely not the priority.

## Architectural constraints (the load-bearing findings)

### A1. HR cannot couple to metabolism

`pulse/modules/cardiovascular.py:14-18`:
```
_N_COUPLING = 2
COUPLING_NAMES = ('cortisol', 'temperature')
```

`pulse/model.py:222-231` only routes cortisol + temperature to the
cardiovascular module. There is no path for glucose, insulin, gut output,
or lactate (lactate goes only to respiratory at `model.py:245`).

There is also **no HR-specific cohort spec, no HR landmark term, no HR
sweep**. The only gradient path to HR weights is the per-step trajectory
MSE on the cold-model rollout — exactly the term we cannot bias toward
HR specifically.

**Implication:** continuing to tune `landmark-weight`, `dose-response-weight`,
or any glucose-axis knob is *guaranteed* not to move `hr_mape`. To move HR,
either:

- **Wire metabolic couplings into `CardiovascularModule`** (insulin, glucose,
  or sympathetic proxy), or
- **Add an HR cohort spec / HR landmark / HR sweep** so HR receives gradient
  independent of the cold-model trajectory.

### A2. Fasting stability trains the wrong dynamical regime

The bench failure is `glucose_returns_near_baseline = 60 mg/dL` at
**420–475 min after a meal** (`knowledge/textbook_scenarios/flow_story_protocol.py:35-36`).

The training signal `--fasting-stability-window=240` runs:
- Zero-embedding rollout
- **No meal**
- 240 min duration
- Targets glucose drift from `NORM_CENTER`

These are disjoint dynamical regimes. The model is being trained to
hold glucose at baseline *with no perturbation*; the bench is asking for
glucose to *return* to baseline *after a meal-driven excursion*. The
mechanisms involved (insulin clearance, hepatic glucose output suppression,
gut absorption tail) are completely different from the resting-equilibrium
mechanisms.

This is also why iter 31's `fst` term grew during training (the no-meal
fast is improving) while the post-meal recovery on the bench did not.

**Implication:** if `glucose_returns_near_baseline` is something we want to
fix, we need a training signal that runs a meal and measures recovery at
~7h, not just a longer no-meal fast.

### A3. `insulin_dose_response` wrong sign is structurally permitted but not forced

`GlucoseGatedInsulinHead.forward` (`metabolic.py:79-90`):
```
prod = softplus(basal) + softplus(peak) · sigmoid((G_norm − g_thresh)/g_temp)
```

`basal` and `peak` are MLP outputs that read both glucose state *and*
gut couplings. Both are wrapped in softplus (≥0), so the gate guarantees
monotonic-in-glucose insulin **only if the MLP doesn't itself reduce
basal/peak as gut input rises**. The wrong sign on the bench therefore
comes from the MLP learning that bigger meals → smaller (basal+peak), which
nothing in the architecture prevents — but nothing in the architecture
forces, either. Capacity exists; we have not exerted enough training
pressure on the right axis to force the right shape.

The `--insulin-sweep-weight=0.30` term *should* push monotonicity, but its
sweep is on `(G, I)` axis, not on `(carb dose, I)` axis. The dose-response
loss exists (`--dose-response-weight=0.20`) but per the knob audit it's been
"stuck at ~8.0" since iter 28 with no movement, suggesting it isn't producing
useful gradient.

## Train → bench signal mapping

| Training term                       | What it supervises                                        | Bench check it should improve              |
|-------------------------------------|-----------------------------------------------------------|--------------------------------------------|
| `gut-loss-weight`                   | Per-step MSE on `gut.forward_window` vs cold              | indirect (`appearance_leads_glucose_peak`) |
| `gut-dose-sweep-weight` (+ AUC)     | Gut kernel at 0/15/30/45/60/90/120 g vs cold              | `appearance_leads_glucose_peak`            |
| `landmark-weight`                   | Δpeak / time-to-peak / AUC around carb meals              | `glucose_dose_response`, `insulin_dose_response`, `glucose_rises_after_meal` |
| `dose-response-weight`              | OLS slope of glucose Δpeak vs carb dose at 30/60/90 g     | `glucose_dose_response` (1:1 protocol)     |
| `insulin-sweep-weight` (+AUC, rank) | dI/dt and dHep/dt at (G,I) sweep points                   | `insulin_dose_response`, `glucose_returns_near_baseline` (via dHep) |
| `fasting-stability-weight`          | Drift from NORM_CENTER over window_min (no meal, zero-emb) | nominally "fasting" — but **disjoint** from `glucose_returns_near_baseline` (post-meal) |
| `cohort-statistic-weight`           | Population-level z² on OGTT/FFA/BHB/GLP-1 etc.            | overall_weighted_mape, marker MAPEs        |
| `verifier-loss-weight`              | Surrogate of verifier directional rules                   | `verifier.overall_score`                   |
| trajectory MSE (always on)          | Per-step state MSE vs cold                                | overall_weighted_mape, all marker MAPEs    |

**Bench failures with NO direct training signal:**

- `hr_mape` — only generic trajectory MSE; no HR-specific term anywhere.
- `glucose_returns_near_baseline` (post-meal, 420 min) — closest is
  fasting-stability, which runs no-meal 240 min. **Disjoint.**
- `insulin_peak_follows_glucose` (timing) — only landmark, only when
  a meal lands inside a window with `landmark-weight > 0`.
- `glucagon_suppressed` (small magnitude failures) — only cohort-statistic.

## Knob rationalisation summary

Of the 30+ flags in `trainExtraArgs`:

**Likely vestigial** (added pre-iter12, never tuned in iters 13–31, no
current failure cites them, value often equals `train.py` default):
`--landmark-pre-window=15`, `--landmark-post-window=120`,
`--landmark-min-carbs=5.0`, `--landmark-weight=0.15`,
`--dose-response-sample-patients=4`, `--gut-dose-sweep-sample-patients=4`,
`--insulin-sweep-sample-patients=4`, `--insulin-sweep-ranking-weight=1.0`,
`--meal-window-bias=0.55`, `--verifier-loss-weight=0.03`.

`--dose-response-weight=0.20` is **borderline-stale**: iter 30 noted the
loss has been stuck at ~8.0 across iters 28–30 with no movement, and no
recent iter touches the weight.

**Load-bearing** (recent or large-vs-default, addressing live failure):
`--fasting-stability-window=240` (iter 31, hypothesis-critical),
`--fasting-stability-weight=0.10` (iter 27–30 churn),
`--gut-dose-sweep-auc-weight=5.0` and `--insulin-sweep-auc-weight=5.0`
(5× default, named as the dilution-defeating mechanism in iters 18–20),
`--insulin-sweep-weight=0.30` (iter 20's headline intervention),
`--cohort-statistic-weight=0.15` (3× default),
`--n-default-patients=8` (zero-embedding fix; removing reopens iter 9-era
pathology), `--trajectory-band-default=0.05` (asymmetric tightening on
zero-embedding).

**Suspicious interactions:**

1. **`gut-loss-weight=0.10` vs `gut-dose-sweep-weight=0.10` are duplicative.**
   Both train the gut kernel. Iter 18 explicitly moved amplitude-pull into
   the AUC term inside `gut-dose-sweep`; `gut-loss-weight` is now arguably
   redundant gradient on the same parameters. No handoff since iter 15
   defends keeping both.

2. **`fasting-stability-weight` vs `landmark-weight`/`dose-response-weight`
   collide during meal windows.** Iter 29 already root-caused a version of
   this ("fasting weight=0.50 fights trajectory signal on glucose rises
   during meals"). Iter 31 dropping the fasting weight to 0.10 plus
   doubling the window to 240 min means the collision is *more*, not less,
   likely during meal-anchored windows (recall `meal-window-bias=0.55`).

3. **`cohort-statistic-weight=0.15`** lives on without a defended
   attribution. Iter 22 falsified the "more cohort weight" hypothesis;
   iter 20's win was attributed to the *sample-patients* bump, not the
   weight.

## Recommended path forward

The right next iteration is **not** a weight tune. In rough priority
order:

### Iter 32a (highest leverage, lowest cost): cardiovascular wiring

Add metabolic couplings to `CardiovascularModule._N_COUPLING`. At minimum,
`insulin` (postprandial sympathetic withdrawal) and `glucose` (acute
hyperglycemia HR effect) are physiologically motivated. Optionally
add an HR cohort spec for postprandial HR rise to give a non-trajectory
gradient path.

This is the only way `hr_mape` moves. Expected effect: `hr_mape` drops
from 0.245 toward 0.15 over training. No effect on glucose (orthogonal
change), so we can validate cleanly.

### Iter 32b (test the "setpoint vs postprandial" hypothesis cleanly): replace fasting-stability with post-meal-recovery

Add a `--postprandial-recovery-weight` signal that runs a meal-bearing
rollout and penalizes residual elevation at +6h. Replace (do not stack)
`--fasting-stability-weight` for one iter to isolate the effect.

If `glucose_returns_near_baseline` clears 8 mg/dL while the no-meal fast
holds, confirms the structural-setpoint regime is the wrong axis. If
fast regresses badly, we know the no-meal signal was load-bearing for
something we hadn't named.

### Iter 32c (knob-rationalisation ablation): zero out the borderline-vestigial set

Single ablation run with these flags removed/zeroed:
- `--gut-loss-weight=0` (replaced by gut-dose-sweep-auc)
- `--dose-response-weight=0` (loss has not moved in 3 iters)
- `--landmark-weight=0` (untouched 19+ iters, never named as lever)

Compare bench to iter 31. If overall MAPE doesn't regress materially,
they were vestigial and we drop them permanently. If something specific
regresses, we keep just that one — and now we have a defensible
attribution.

### What to NOT do for at least one iter

- **Do not tune `log_sg`, `fasting-stability-weight`, or any setpoint knob.**
  We've spent 5 iters on this axis and persistent failures
  (meal_dose_response stuck at 0.33 for *every* iter, glucose_returns at
  ~65 mg/dL) point at the wrong axis.
- **Do not add new loss terms** until 32c rationalisation is done.
  Adding more weights to a system we can't yet ablate cleanly compounds
  the problem.

### Stronger-baseline check (deferred but flagged)

A Bergman minimal model fit per patient is worth running in parallel to
benchmark our complex multi-signal trainer against. If a 3-parameter
minimal model gets glucose_mape competitive with our 30-flag setup, we
need to re-justify the architecture entirely. This isn't urgent — but if
iters 32a/b/c don't clear at least one of the persistent failures, this
becomes the next mandatory step.

## Pointer references

- `apps/pulse/engine/pulse/modules/cardiovascular.py:14-18` — HR coupling
  set (cortisol, temperature only)
- `apps/pulse/engine/pulse/modules/metabolic.py:79-90` —
  `GlucoseGatedInsulinHead.forward`
- `apps/pulse/engine/pulse/modules/metabolic.py:124-127` — `log_sg`
  restoring rate
- `apps/pulse/engine/pulse/training/fasting_stability_signal.py:5-8` —
  documents the "production > clearance at rest" failure mode
- `apps/pulse/engine/pulse/knowledge/textbook_scenarios/flow_story_protocol.py:35-36`
  — post-meal recovery evaluation window (`DIETARY_CARB_GLUCOSE_RECOVERY_START/END = 420/475`)
- `apps/pulse/docs/iter27-handoff.md` through `iter31-handoff.md` — per-iter
  motivations
