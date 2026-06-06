# Pulse iter 42 — physiology-rule supervision

After iters 39 (range-floor vitality), 40 (cohort weight bump), and
41 (NADIR landmarks) all failed to revive the dead-pathway markers
(glucagon / FFA / ghrelin / cortisol / ACTH still flat against
iter-38 baseline), a fundamental review surfaced the principle now
written into the PRD as *Calibrated supervision*:

> Match supervision to the evidence we have — no more, no less.

The failed iters were all over-enforcing in some shape: range floor
satisfied by any oscillation, contrast-shape satisfied by means,
NADIR landmark on flat data has tiny residuals. None of them
encoded the *qualitative* physiology facts the literature gives us —
direction-of-change after meals, anti-correlation between FFA and
insulin, timing windows for circadian peaks.

## Surface: physiology rules

A new knowledge surface alongside `CohortStatisticSpec` and
`CouplingPrior`. Source code:

- Schema: `apps/pulse/engine/pulse/knowledge/physiology_rules.py`
- Loss: `apps/pulse/engine/pulse/physiology_rules_loss.py`
- Training signal: `apps/pulse/engine/pulse/training/physiology_rules_signal.py`
- Diagnostic: `apps/pulse/engine/pulse/diagnostics/physiology_rules.py`
  (`uv run python -m pulse.diagnostics physiology-rules --checkpoint <ckpt>`)
- CLI: `--physiology-rules-weight` (off by default), `--physiology-rules-sample-patients`

Each `PhysiologyRule` has a literature-cited predicate that returns a
non-negative *violation amount* in marker-native units. Loss is
squared-hinge: `(violation / scale) ** 2`. The hinge form is
load-bearing — once the rule says "glucagon decreased after the
meal," gradient stops, never enforcing more than physiology says.

Per-trajectory evaluation (vs cohort statistics' per-population mean)
removes the iter-40 escape hatch where a contrast was satisfied by
inter-patient averaging without trajectory-shape movement.

## Starter rules (5 across the dead-pathway markers)

| rule | source | shape |
|---|---|---|
| glucagon_falls_postprandial | Unger & Orci 1981 | direction-after-event |
| ffa_inverse_to_insulin_postprandial | Frayn 2003; Boden 1996 | anti-correlation |
| ghrelin_falls_after_meal | Cummings 2001 | direction-after-event |
| bhb_rises_during_fast | Cahill 2006; Owen 1969 | direction-after-event |
| cortisol_morning_peak | Czeisler 1999; Weitzman 1971 | timing band |

Predicate helpers (`hinge_min_drop`, `hinge_min_rise`,
`hinge_max_correlation`, `hinge_argmax_in_band`) cover the common
shapes. Novel rule shapes write their own torch ops.

## iter-38 baseline diagnostic

Saved at `apps/pulse/docs/iter42-baselines/physiology-rules-iter38.json`.

```
rule                                violation  sat%  ||g_gut||  ||g_met||  ||g_oth||
glucagon_falls_postprandial             5.00    0%   2.8e-04  4.5e-03  2.2e-04
ffa_inverse_to_insulin_postprandial     0.06    0%   2.0e+00  3.6e+00  1.1e+00
ghrelin_falls_after_meal               30.00    0%   2.2e-05  8.6e-05  1.8e-03
bhb_rises_during_fast                   0.00  100%   0        0        0
cortisol_morning_peak                 183.36    0%   0        1.1e-01  4.0e+00
```

Key reads:

- **Four of five rules violate** — the dead-pathway story made
  measurable per-rule, not per-aggregate-MAPE.
- **bhb is alive** (100% satisfied, no gradient needed) — matches
  iter-38 bench MAPE 0.19, the lowest of the dead-pathway markers.
  The bhb rule will exert no pressure once training starts; we keep
  it in the registry for regression detection.
- **ffa relational rule has 3.6 metabolic-grad** — much stronger
  pull than iter-41's NADIR landmark on the same marker, suggesting
  the relational form (anti-correlation with insulin) reaches
  parameters the extremum-residual form didn't.
- **ghrelin gradient is 1–2 orders of magnitude weaker** than
  glucagon (8.6e-5 vs 4.5e-3 onto metabolic). Predicts ghrelin will
  be the hardest revival case even with this surface — same finding
  iter-41 surfaced indirectly (ghrelin MAPE stayed 1.93 → 1.93).
- **cortisol gradient lands on Stress module** (4.0 in the "other"
  bucket) — the right module owns cortisol; gradient reaches.

## Iter-42 plan

**A — measurement-only iter** (recommended first step):
Submit a training run with `--physiology-rules-weight=0.10` (looser
than cohort-statistic-weight=0.15 to start), iter-38 spec otherwise.
Phase-2 enabled. Compare bench results against iter-38 baseline:

- *Primary*: per-marker MAPE on glucagon / ffa / ghrelin /
  cortisol (the four violated rules) on the expanded bench.
  Target: drop from baseline (0.44 / 0.43 / 1.93 / 0.42) toward
  0.10–0.30 range on at least 2 of 4.
- *Per-rule satisfaction*: re-run the diagnostic on the iter-42
  checkpoint, compare violation amounts against iter-38. Target
  the four currently-violated rules satisfied at sat% ≥ 50% on
  the supervised embeddings.
- *Gate*: hr_mape ≤ 0.16, glucose_mape ≤ 0.20, overall ≤ 0.12.
  Any regression here means the rule weight is too high or one
  rule conflicts with the existing signal mix.

**Risk paths**:

- (R1) ghrelin grad is so weak that even with the rule, ghrelin
  doesn't move. If iter-42 shows ghrelin MAPE flat *despite* the
  rule weight, the issue is upstream — Appetite module
  parameterization or coupling-graph reach. Move to (B) below.
- (R2) FFA anti-correlation grad is *too* strong (3.6 vs glucagon's
  4.5e-3) — could over-pull and break glucose dynamics. Watch
  glucose_mape closely. If regressed, drop FFA rule weight or move
  to per-rule weights.
- (R3) Cortisol timing-band rule integrates a 24h trajectory with
  no meals. Long rollout, expensive. May want to drop sample
  patients to 2 for this rule specifically if epoch time blows up.

**B — if (R1) hits**: investigate whether the appetite module's
ghrelin head can express the postprandial drop at all from the
zero-embedding starting state. If ghrelin parameter gradients are
zero across multiple supervision shapes, the issue is structural
(coupling graph or factorization), not loss design.

**C — registry expansion** (deferred): once the starter 5 prove
out, expand to ~20–30 rules covering the rest of the markers and
the OGTT / fasting / circadian regimes. The cost is per-rule
authoring (~30 min if predicate helpers cover the shape).

## Why this is different from iter-39 vitality

Iter-39's `MarkerVitalitySignal` was structurally similar — a
hinge floor on marker behaviour. It failed because the predicate
("range > target") was satisfied by any oscillation, including
representation-collapse-shaped noise that doesn't respect physiology
timing. Physiology rules use *strict directional* predicates anchored
to specific events ("post-meal mean - pre-meal mean > 5 pg/mL"),
which a flat trajectory cannot satisfy by adding noise. The
diagnostic's per-rule grad-norm view also makes dead rules visible
*before* training starts — preventing the iter-39 failure mode of
launching a full run on a signal that turns out to be inert.

## Open questions

- **Per-rule weights**: currently rules share a single global
  `--physiology-rules-weight`. If FFA over-pulls (R2), we'd want a
  `weight` field on the rule itself. Already present in the schema
  (`PhysiologyRule.weight`); not yet exposed at the CLI level.
- **Cold init for circadian rule**: the cortisol rule uses a 24h
  arm with no meals. Cold-model init from `PatientParams()` may
  not be the right anchor; NORM_CENTER might be cleaner. Need to
  check whether the rollout produces sensible cortisol dynamics
  from each.
- **Promotion to bench gate**: once rules prove out, the rule
  satisfaction fractions are themselves a candidate gate metric.
  Worth wiring into `benchmark.thresholds.json` as
  `physiology_rules.min_satisfied_fraction` after iter-42 numbers
  are in.

## Iter-42 result (job `train-20260509T093245Z`, commit `b36964ae`)

Mixed: rule mechanism worked (FFA satisfied) but bench unchanged on
dead-pathway markers. Per-rule diagnostic on iter-42 checkpoint vs
iter-38 baseline:

| rule | viol 38 | viol 42 | sat% 42 | g_met 38 | g_met 42 |
|---|---|---|---|---|---|
| glucagon_falls_postprandial | 5.00 | 5.00 | 0% | 4.5e-3 | 1.3e-2 |
| ffa_inverse_to_insulin | 0.06 | **0.00** | **100%** | 3.6 | 0 |
| ghrelin_falls_after_meal | 30.00 | 30.01 | 0% | 8.6e-5 | 1.3e-4 |
| bhb_rises_during_fast | 0.00 | 0.00 | 100% | 0 | 0 |
| cortisol_morning_peak | 183.4 | 180.3 | 0% | 0.11 | 0.10 |

Bench (relative to iter-38):

```
overall_weighted_mape   0.095 → 0.107  (+0.012)
hr_mape (gate)          0.135 → 0.162  (+0.027, gate fail)
glucose_mape (gate)     0.203 → 0.215  (+0.012, gate fail)
verifier meal           0.581 → 0.558  (-0.023, gate fail)
glucagon                0.44  → 0.44   (0)
ffa                     0.43  → 0.43   (0)
ghrelin                 1.93  → 1.93   (0)
cortisol                0.42  → 0.42   (0)
insulin                 1.08  → 0.94   (-0.14, incidental coupling)
```

Gate failed on hr / glucose / verifier-meal. Per-marker MAPE on the
four targeted dead-pathway markers is **identical to iter-38** even
though FFA is fully satisfied at the supervised embeddings.

Diagnostics persisted at `apps/pulse/docs/iter42-baselines/`:
`physiology-rules-iter42.json`, `benchmark-report-iter42.json`.

### What we learned

**The rule mechanism works.** FFA satisfied 0% → 100% under its own
arm protocol — the per-trajectory hinge supervision did move
parameters as designed. Glucagon partially moved (gradient nearly
tripled but didn't reach satisfaction at this weight). Ghrelin
remained stuck — confirms the structural-reach hypothesis (the
Appetite module's ghrelin head doesn't receive meaningful gradient
from any direction).

**Training-vs-eval distribution mismatch is the bench-blocking
issue.** The FFA rule supervises one specific arm
(`mixed_meal_postprandial`, 75/25/20g, start_hour=8.0, 300min).
The bench evaluates *different* arms (cohort sleep 48h, cohort
meal postprandial 8h OGTT, real-user 12h overnights). The model
satisfied FFA↔insulin under the rule's protocol without
generalizing to bench protocols. This is the iter-39 vitality
failure mode in a different shape — predicate too narrow.

**HR regressed despite reverting landmark-weight to 0.20.**
Iter-38 baseline hr_mape was 0.135; iter-42 returned 0.162. With
landmark configuration matching iter-38, the most likely cause is
parameter drift from the new physiology-rules rollouts interfering
through the model's shared hidden state. Worth a `signal-balance`
diff between iter-38 and iter-42 to localize.

### Iter-43 candidate moves

1. **Multi-arm rules** — each rule evaluated on 2–4 arm protocols
   that span the bench's regimes. FFA rule on
   {mixed_meal_postprandial, OGTT, fasted-postprandial}. Forces
   the relationship to hold robustly, not just on one breakfast.
2. **Ghrelin coupling-graph investigation** — separate from rule
   design; the persistent 1.3e-4 metabolic gradient across iters
   38, 41, 42 says "no signal can reach the Appetite module's
   ghrelin head". Worth diffing the Appetite module's coupling
   inputs vs the canonical references.
3. **HR regression localization** — `pulse.diagnostics signal-
   balance` on iter-42 vs iter-38 to find which signal pushed HR
   off baseline.

(1) is the direct fix for the bench-translation problem. (2) is
load-bearing for ghrelin specifically. (3) is a regression hunt
that should land before the next training run.

## Iter-43 result (job `train-20260509T180927Z`, commit `aacedba5`)

**Multi-arm hypothesis confirmed.** FFA satisfied 33% → 67% — exactly
the protocol-coverage improvement we predicted. HR regression fully
recovered. But the bench-MAPE-on-dead-pathways problem persists at
a different layer (embedding distribution narrowness instead of
arm-protocol narrowness).

Per-rule diagnostic (multi-arm, iter-38 → iter-42 → iter-43):

| rule | sat% 38 | sat% 42 | sat% 43 | viol 43 | g_met 43 |
|---|---|---|---|---|---|
| glucagon (3 arms) | 0% | 0% | 0% | 5.00 | 4.1e-3 |
| **ffa (3 arms)** | 0% | 33% | **67%** | 0.23 | **19.8** |
| ghrelin (3 arms) | 0% | 0% | 0% | 30.0 | 9.1e-5 |
| bhb (2 arms) | 50% | 50% | 50% | 0.006 | 1.15 |
| cortisol (2 arms) | 0% | 0% | 0% | 180 | 0.085 |

Bench (iter-38 → iter-42 → iter-43):

```
overall_weighted_mape   0.095 → 0.107 → 0.092   (best ever)
verifier overall        0.788 → 0.768 → 0.798   (best ever)
hr_mape (gate)          0.135 → 0.162 → 0.114   (best ever)
glucose_mape (gate)     0.203 → 0.215 → 0.207   (gate just over)
verifier meal           0.581 → 0.558 → 0.597   (gate just under)
verifier coupling       0.872 → 0.870 → 0.904   (best ever)
sbp_mape                0.073 → 0.074 → 0.065   (best ever)
dbp_mape                0.051 → 0.056 → 0.048   (best ever)
**insulin**             1.080 → 0.938 → 1.232   (regressed past iter-38)
glucagon                0.44  → 0.44  → 0.44    (flat)
ffa                     0.43  → 0.43  → 0.43    (flat)
ghrelin                 1.93  → 1.93  → 1.93    (flat)
cortisol                0.42  → 0.42  → 0.42    (flat)
```

### What we learned

**Multi-arm worked.** FFA jumped 33% → 67% under multi-arm
supervision — the model now satisfies 2 of 3 arms. Hypothesis
confirmed: single-arm rules let the model overfit to one protocol.

**HR regression was a transient.** With multi-arm rules and the
FFA pull more diffused, HR landed at 0.114 — better than iter-38's
0.135. Probably the iter-42 HR regression came from one specific
rule-arm interaction that multi-arm averaged out.

**Dead-pathway bench MAPE didn't move despite rule progress.**
Even with FFA at 67% satisfied, the bench FFA MAPE stayed at
0.43. Same story for glucagon/ghrelin/cortisol. The rule mechanism
fixes specific (embedding, arm) regimes but doesn't generalize to
the bench's calibrated embeddings + different arms. **This is the
next layer of overfitting:** we fixed protocol-narrowness with
multi-arm; we still have embedding-narrowness with sample_patients=4.

**Insulin regressed.** FFA rule has `||g_met||=19.8` — by far the
largest gradient in the registry, larger than the sum of all
others. Pulled metabolic params away from iter-38's well-tuned
insulin dynamics. Per-rule weighting is the lever to dial this back.

### Iter-44 plan

Two changes, both small:

1. **`FFA_INVERSE_TO_INSULIN.weight = 0.5`** (down from default 1.0).
   The rule still pulls but at half strength, leaving room for
   cohort/insulin-sweep to keep insulin calibrated. iter-43 made
   FFA significantly harder than iter-42 (gradient is real and
   working), so half-pull should still drive the 67% → 80%+
   trajectory while letting insulin stabilize.
2. **`--physiology-rules-sample-patients 4 → 10`**. Each rule now
   evaluates on 10 patient embeddings + zero default per epoch
   (vs 4+0). Forces the model to satisfy the rule across a broader
   embedding distribution — closer to what calibrate_embedding
   produces at bench time. Per-epoch wall-clock goes ~480s → 540s
   (+10%); still inside the 8h budget.

Other knobs unchanged. The hope: FFA bench MAPE finally moves
because the model can no longer satisfy the rule by overfitting
to 4 specific embeddings under 3 arms — has to make the
relationship hold for 10 embeddings × 3 arms = 30 (emb, arm)
combinations, much closer to the bench's embedding distribution.

## Iter-44 result (job `train-20260510T022858Z`, commit `b032f614`)

**Insulin recovered (1.23 → 0.99) — FFA half-pull worked.** But broader
embedding sampling did NOT move dead-pathway bench MAPE and corrupted
HR/vitals/verifier-meal. The two iter-44 changes had decoupled effects;
isolating them is iter-45's job.

Per-rule diagnostic (multi-arm, iter-43 → iter-44 on iter-44 ckpt):

| rule | sat% 43 (4-pt) | sat% 44 (4-pt) | sat% 44 (10-pt) | viol 44 | g_met 44 |
|---|---|---|---|---|---|
| glucagon (3 arms) | 0% | 0% | 0% | 5.00 | 8.5e-3 |
| **ffa (3 arms)** | **67%** | **0%** | **0%** | **0.70** | **25.85** |
| ghrelin (3 arms) | 0% | 0% | 0% | 30.0 | 1.2e-4 |
| **bhb (2 arms)** | **50%** | **100%** | **100%** | 0.00 | 0.00 |
| cortisol (2 arms) | 0% | 0% | 0% | 182 | 0.10 |

Bench (iter-43 → iter-44):

```
overall_weighted_mape   0.092 → 0.115   (regressed past iter-42)
verifier overall        0.798 → 0.772   (regressed)
hr_mape (gate)          0.114 → 0.194   (catastrophic, +0.079)
glucose_mape (gate)     0.207 → 0.214   (just over gate)
verifier meal           0.597 → 0.557   (further from gate)
verifier coupling       0.904 → 0.874   (regressed)
sbp_mape                0.065 → 0.083   (regressed +0.018)
dbp_mape                0.048 → 0.057   (regressed +0.008)
**insulin**             1.232 → 0.990   (recovered, FFA half-pull worked)
glucagon                0.44  → 0.44    (flat, dead pathway)
ffa                     0.43  → 0.43    (flat, dead pathway)
ghrelin                 1.93  → 1.93    (flat, dead pathway)
cortisol                0.42  → 0.44    (slight regression)
textbook pass           0.808 → 0.760   (regressed)
```

Textbook regression is single-check: `meal_dose_response.insulin_dose_response`
went from 2.83 → 1.95 (threshold 2.0) — not a model failure, just the
insulin-sweep AUC ratio crossing back under the textbook threshold as
insulin recovered. Expected side-effect of the half-pull.

### What we learned

**FFA half-pull (1.0 → 0.5) decoupled the rule from insulin.** Insulin
recovered cleanly toward iter-38's 1.08. ✓ predicted. But: ||g_met||
on FFA grew 19.8 → 25.85 even at half weight (the gradient is over a
larger violation now), and FFA rule satisfaction collapsed 67% → 0%
even at the iter-43-equivalent 4-patient diagnostic. The half-pull
removed enough force that the model walked away from FFA satisfaction
under the iter-44 broader-sampling regime. Bench FFA MAPE flat at
0.43 — confirming the rule's local progress doesn't translate to
bench-calibrated embeddings (now firmly established across iter-42,
43, 44). bhb is the only rule that improved: 50% → 100% sat,
||g_met||=0 — meaning broader sampling DID help bhb generalize.

**Broader sampling (4 → 10) corrupted the bench.** HR collapsed
+0.079, sbp +0.018, dbp +0.008, verifier meal -0.040, verifier
coupling -0.030, sanity -0.014, glucose +0.007 (just over gate).
Hypothesis: with 5 rules × {2,3} arms × 10 patients evaluated each
epoch, the rule signal dominates Phase-2 gradient budget at
weight=0.10, washing out default-baseline (HR), trajectory rollout,
and the cohort signal that was carrying verifier-meal. The
sample_patients=10 setting raised effective rule weight past what
0.10 was designed to buffer.

**Dead-pathway bench MAPE is structurally bottlenecked, not
sampling-bottlenecked.** FFA, glucagon, ghrelin all stayed at iter-38
levels through every rule-mechanism experiment iter-42 → iter-43 →
iter-44. Per-rule local satisfaction has hit its ceiling for moving
bench. Per spec R3: next pivot is coupling-prior strengthening
(appetite←insulin / appetite←nutrient) or a separate trajectory-
supervised signal on the gut→ghrelin pathway. Iter 46 territory.

### Iter-45 plan

Pure revert: keep FFA weight at 0.5 (insulin recovery is real),
revert `--physiology-rules-sample-patients` 10 → 4. Single variable
swap. Cleanest possible isolation of which iter-44 change caused
the HR/vitals collapse.

Expected: HR returns to ~0.114, sbp/dbp recover, verifier-meal
returns toward 0.597, insulin stays in the 0.99–1.10 band, dead-
pathway bench MAPE flat (still no structural mechanism for them).
If HR fully recovers, we have iter-43-best-ever + insulin fix.
Still misses the gate (glucose 0.207 ≥ 0.20, verifier_meal 0.597
< 0.65) — those are iter-46+ territory.

Risks: (R1) HR doesn't fully recover → FFA half-pull also
contributes to vitals destabilization → iter-46 reverts FFA weight
to 0.75 or 1.0 with a balancing knob (cohort_statistic_weight up).
(R2) FFA rule satisfaction stays at 0% under 4-patient training
with weight=0.5 → confirms half-pull was too aggressive → iter-46
tries weight=0.75. (R3) Insulin regresses back toward 1.23 →
contradicts the iter-44 reading; rules out FFA half-pull as the
cause and means broader sampling somehow helped insulin via a
coupling we don't understand.

## Iter-45 result (job `train-20260510T104956Z`, commit `1aa796bc`)

**Best run yet on every metric except insulin.** sample_patients=10
was 100% the cause of iter-44's HR/vitals/verifier-meal collapse;
iter-45 reverted to 4 and recovered everything. FFA half-pull
(weight=0.5) held — insulin landed in the 1.14 band, between
iter-43 (1.23) and iter-44 (0.99). All three earlier risks (R1, R2,
R3) cleared.

Per-rule diagnostic (multi-arm, iter-43 → iter-44 → iter-45 at 4 sampled):

| rule | sat% 43 | sat% 44 | sat% 45 | viol 45 | g_met 45 |
|---|---|---|---|---|---|
| glucagon (3 arms) | 0% | 0% | 0% | 5.00 | 4.5e-3 |
| **ffa (3 arms)** | **67%** | 0% | **67%** | **0.28** | **15.89** |
| ghrelin (3 arms) | 0% | 0% | 0% | 30.0 | 9.3e-5 |
| bhb (2 arms) | 50% | 100% | 50% | 0.006 | 1.19 |
| cortisol (2 arms) | 0% | 0% | 0% | 180 | 0.086 |

Bench (iter-43 → iter-44 → iter-45):

```
overall_weighted_mape   0.092 → 0.115 → 0.091   (NEW BEST)
verifier overall        0.798 → 0.772 → 0.799   (NEW BEST)
hr_mape (gate)          0.114 → 0.194 → 0.110   (NEW BEST)
glucose_mape (gate)     0.207 → 0.214 → 0.207   (gate by 0.007)
verifier meal           0.597 → 0.557 → 0.605   (gate short by 0.045)
verifier coupling       0.904 → 0.874 → 0.898
verifier sanity         0.999 → 0.985 → 0.998
sbp_mape                0.065 → 0.083 → 0.065   (NEW BEST)
dbp_mape                0.048 → 0.057 → 0.046   (NEW BEST)
temp_mape               0.015 → 0.016 → 0.015   (NEW BEST)
insulin                 1.232 → 0.990 → 1.138   (improved vs iter-43)
glucagon                0.44  → 0.44  → 0.44    (flat, structural)
ffa                     0.43  → 0.43  → 0.43    (flat, structural)
ghrelin                 1.93  → 1.93  → 1.93    (flat, structural)
cortisol                0.42  → 0.44  → 0.41    (best vs iter-43)
textbook pass           0.808 → 0.760 → 0.808
```

### What we learned

**Single-variable isolation worked.** iter-45 confirmed sample_patients=10
was the sole cause of the HR/vitals/verifier-meal regression; FFA
half-pull was a clean win and the two iter-44 changes had decoupled
effects.

**FFA rule is at its local ceiling.** sat% sits at 67% (2 of 3 arms)
across iter-43 → 45 regardless of weight (1.0 → 0.5 → 0.5) or sampling
(4 → 10 → 4). ||g_met|| stays in the 15-25 range. The rule pulls
hard but the model has found a local solution where 1 of the 3 arms
remains violated. That last arm is probably structural.

**Dead pathways are structurally unmovable.** FFA, glucagon, ghrelin,
leptin bench MAPE are literally byte-identical to iter-38: 0.4320,
0.4441, 1.9285, 0.0266. Across three iters of rule-mechanism
experiments (multi-arm, weight tuning, broader sampling), nothing
moved. The bench cohort embeddings live in a region of latent space
the rule signal cannot reach via the current coupling graph. Per
spec R3: iter 46+ must pivot to coupling-prior strengthening on
appetite←insulin / appetite←nutrient or a separate trajectory-
supervised signal on the gut→ghrelin pathway.

**bhb sampling sensitivity is the one signal of rule generalization.**
bhb went 50% → 100% under sample_patients=10, then back to 50% at 4.
The rule actually does generalize across embeddings for bhb (g_met
non-zero). Could be a hint for designing the structural mechanism —
bhb's coupling is reachable, FFA/glucagon/ghrelin aren't.

### Iter-46 plan

The cleanest next experiment, and the one that informs the
structural pivot, is to **halve `--physiology-rules-weight` 0.10 →
0.05.** Hypothesis: at 0.10 the rule signal eats ~70s/epoch of
gradient budget (10% of phase-2 epoch wall-clock) and isn't moving
dead-pathway bench MAPE; it might be net-neutral or net-negative on
overall accuracy. Halving makes the rule's gradient contribution
smaller while keeping the mechanism in place to preserve insulin
calibration via the FFA half-pull.

Outcomes:
- **Improves overall_weighted_mape further** → rules at 0.10 were
  costing bench accuracy; iter-47 zeros them out and the registry
  becomes a diagnostic tool only, not a training surface.
- **Holds at iter-45 levels** → rules contribute approximately zero
  net; iter-47 zeros them with confidence and we move to structural
  work.
- **Regresses (especially insulin)** → rules are meaningfully holding
  insulin/FFA calibration even though they don't move dead pathways;
  the right next move is structural (coupling-prior or trajectory-
  supervised signal) rather than dropping the rule signal further.

Single-variable change. All other args from iter-45 unchanged.
Risks: (R1) insulin regresses back toward 1.23 — partial-but-not-
complete recovery of iter-44's insulin gain confirms rules-at-0.10
were carrying insulin; iter-47 explores 0.075. (R2) verifier-meal
drops below 0.60 — the rule was buffering meal-response accuracy
via FFA-postprandial; iter-47 reverts and pivots to structural.
(R3) FFA rule satisfaction collapses 67% → 0% — confirms 0.05 is
below the activation threshold for FFA; useful boundary information
even if the bench regresses. (R4) Glucose moves under the gate
(0.20) — a happy surprise; iter-47 then attacks verifier-meal
directly.

## Iter-46 result (job `train-20260510T233609Z`, commit `8c3ccb60`)

**Net-neutral — outcome (b).** Halving `--physiology-rules-weight`
0.10 → 0.05 changed nothing material: overall_weighted_mape 0.0907 →
0.0906, verifier_overall 0.799 → 0.800 (microscopic new best), HR
0.110 → 0.110, glucose/sbp identical, insulin 1.14 → 1.12 (slight
improvement — even at 0.05 the FFA rule still caps insulin). Dead
pathways byte-identical *again* (FFA 0.4320, glucagon 0.4441, ghrelin
1.9283, leptin 0.0266). Per-rule diagnostic at 4 sampled: FFA 67%,
bhb 50%, glucagon/ghrelin/cortisol 0% — unchanged from iter-45.

The rule mechanism is confirmed net-neutral at 0.05-0.10: it
preserves insulin calibration (the FFA-postprandial cap) but
contributes nothing to bench accuracy and **cannot move the dead
pathways at any weight**. The dead-pathway numbers have now been
identical across *four* rule-mechanism iters (42→43→44→45→46) and
across three different levers (multi-arm, broader sampling, weight).
This closes the rule-mechanism line of attack on the dead pathways.

→ **The structural diagnosis and the pivot — cold-model distillation
— are written up in `dead-pathways.md`.** Iter 47 onward lives
there. The physiology rules stay at 0.05 (harmless, keeps insulin
calibrated); this doc is closed at iter 46.
