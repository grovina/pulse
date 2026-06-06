# Pulse iter 39 — unobserved-marker shadow probe (the dead-pathway problem)

After iter 38 cleared the HR gate, a fundamental review surfaced a
much bigger blind spot than the +15 bpm HR offset that cost us 9
iterations to find.

## The probe

`apps/pulse/engine/scripts/probe_unobserved_markers.py` runs the
trained model on the bench's 25 episodes (using deterministic
embeddings — no calibration, pure inference) and reports per-marker
distribution statistics for the 14 markers the bench gate **doesn't**
test. The bench eval points cover only 5 markers (glucose, hr, sbp,
dbp, temp) because those are the markers real users measure on their
watches. The remaining 14 (insulin, glucagon, FFA, BHB, lactate,
hepatic_output, ghrelin, leptin, GLP-1, cortisol, ACTH, HRV, RR,
SpO₂) are produced internally by the model and have **never** been
held-out-validated in any iter.

## Result on iter 38 checkpoint

```
marker            norm        mean        p05        p50        p95
insulin          10.00       34.49      21.09      36.54      41.78
glucagon         70.00       70.00      70.00      70.00      70.01    ← essentially constant
ffa               0.50        0.50       0.50       0.50       0.50    ← essentially constant
bhb               0.10        0.15       0.11       0.15       0.18
lactate           1.00        1.84       1.11       1.95       2.29
hepatic_output    2.00        2.15       2.05       2.06       2.71
ghrelin         100.00      100.03     100.00     100.03     100.06    ← essentially constant
leptin           10.00       10.01      10.01      10.01      10.02    ← essentially constant
glp1             10.00       18.51      13.12      18.69      23.06
cortisol         12.00       18.86      12.88      19.83      22.04
acth             30.00       30.90      30.19      30.97      31.41
hrv              40.00       19.51       5.27      16.27      41.32    ← below physiological floor
rr               15.00       18.20      15.83      18.31      20.46
spo2             98.00       98.00      97.53      97.93      98.60
```

**Five markers are essentially flat lines** across the entire 25-episode
12h bench: glucagon, ffa, ghrelin, leptin, acth. Their across-episode
mean spans are <0.1% of the variable's range. The dynamics module
exists; the gradient signal during training apparently doesn't reach
it; the model has learned to leave these markers at their initial
state.

**Two more markers are tightly suppressed**: bhb (range too narrow for
fasting cycles), cortisol (no afternoon nadir, daily range compressed
to ~20% of expected).

## This is the root cause of 9+ iterations of `dietary_carb` failures

The textbook bench checks have been failing across every iter since 27:

| check | iter 38 v | threshold | problem |
|---|---|---|---|
| `glucagon_suppressed_postprandial` | -0.06 | < 0 (post < pre) | glucagon ≈ 70 always, no signal |
| `ffa_suppressed_antilipolysis` | -0.00 | < 0 | ffa ≈ 0.5 always, no signal |
| `ghrelin_suppressed_after_feeding` | -0.02 | < 0 | ghrelin ≈ 100 always, no signal |
| `OGTT glucagon_suppressed` | -0.01 | < 0 | same |
| `dietary_carb glucose_returns_near_baseline` | 28 mg/dL | ≤ 8 | model can't recover postprandially — counter-regulatory hormones aren't kicking in because they never move |

Every per-iter weight tuning for the postprandial regime hit a wall
because **the counter-regulatory pathway is dead at the marker level**.

## Why training failed to drive these markers

`MassActionModule` (apps/pulse/engine/pulse/modules/base.py) already
fixed an iter-23 "representation collapse" problem where small-magnitude
species were left on a flat manifold by gradient asymmetry. Each species
got its own production/consumption head. But the structural fix only
helps if there's *gradient pressure* on those species. Sources of
gradient that *should* drive glucagon / FFA / ghrelin:

- **Cohort statistics** (`meal_ghrelin_suppression`, `mixed_meal_glucagon_suppression`,
  `meal_ffa_suppression`, etc. in `nutrition.py` / `glucose_handling.py`).
  Weight 0.15 across 17–19 specs sharing budget; glucagon/FFA/ghrelin
  contributions are individually tiny.
- **Trajectory MSE on default-patient distillation**. Cold-model HR /
  glucose / insulin trajectories have large amplitude → large MSE
  gradients. Glucagon / FFA / ghrelin trajectories from cold model
  are themselves narrow → small MSE → dominated by other markers'
  gradients.
- **Landmark loss** — was on these markers in iter-33's full default
  spec set, but iter 34 narrowed `DEFAULT_LANDMARK_SPECS` to HR-only
  to fix the gradient-budget problem on glucose. We never put them
  back; nothing has filled that gradient gap since.

## Iter 39 plan: revive the dead pathways

**(α) New training signal: `MarkerVitalitySignal`.** For each named
marker in a configurable list, run a 12h zero-embedding rollout with
a standard meal pattern (matches the bench scenario shape), measure
the absolute peak-to-trough variation, and apply a *floor* loss — penalise
when range falls below a target. This directly attacks "marker is
flat" rather than fitting a specific shape. Per-marker target ranges
sourced from the same physiology references that anchor the cohort
specs.

```
loss_marker = relu(target_range - actual_range) ** 2 / scale ** 2
```

Disabled by default (weight=0.0); enabled for `glucagon, ffa, ghrelin,
leptin, acth, bhb, cortisol` at meaningful weight. Won't conflict with
trajectory MSE or cohort statistics — those constrain *level/shape*,
this constrains *that the marker moves at all*.

**(β) Bump landmark-weight subset on counter-regulatory markers.**
Re-introduce `glucagon` (NADIR), `ffa` (NADIR), `ghrelin` (NADIR) into
`DEFAULT_LANDMARK_SPECS` alongside HR. The iter 33 → 34 narrowing was
right at the time (HR was the broken one), but with HR now constrained
by the iter-37/38 default-baseline regularizer, the landmark capacity
can re-target the dead markers.

**(γ) Audit cohort-spec weights**. The total cohort_statistic_weight
(0.15) is split across 19 specs. The glucagon / FFA / ghrelin specs
get diluted to ~0.008 each. Either bump the global weight or add a
per-spec weight (already supported in `CohortStatisticSpec.weight`,
defaulting to 1.0 — currently unused) to triple the budget on the
suppression specs.

α is the most direct: "marker must move." β is opportunistic re-use
of existing infra. γ is a budget rebalance.

**Iter 39 = α only**, plus the probe automated as part of the bench
pipeline so we surface this kind of regression in real time. β and
γ deferred until α's effect is measured.

## On the bench dataset internal HR inconsistency

Side note from the iter-36 investigation: bench `calibration_check_ins`
HR mean is 68.76 vs `eval_measurements` HR mean 62.95 — a 5.8 bpm
floor on hr_mape that no model fix can beat. The bench is generated
from real user check-ins; the gap is plausibly a sampling artifact
(eval points happen to fall at lower-HR moments — late-night, post-
meal slow decline, etc.). With iter 38 at hr_mape 0.135 (well under
0.15), this is no longer the binding constraint and can stay deferred.

## Productionising the probe

The probe currently runs against a downloaded checkpoint with no
calibration, ~10s for 25 episodes. Worth wiring into
`cloudbuild-benchmark-only.yaml` as a step that runs after the bench
gate evaluation — it adds tens of seconds, not minutes, and gives
every future iter a per-marker plausibility report alongside the gate
metrics. Tracked as a follow-up; not blocking iter 39.
