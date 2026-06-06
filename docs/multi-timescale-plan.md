# Multi-timescale physiology — plan

The architecture-roadmap doc listed "latent reservoirs" as a Move D
— an abstract gap labelled "the model needs hidden integrators."
A user question on 2026-05-15 made it concrete: *can the model
handle Z2 exercise's acute-vs-chronic split?* The honest answer
surfaced the largest single architectural lift remaining in Pulse —
bigger than head re-parameterisation, bigger than HPA structure,
bigger than the physiology rule library.

This doc scopes that lift, and frames it so it lands as a natural
extension of the PRD's existing primitives rather than a new layer of
modeling machinery on top.

## The gap, restated

The model represents *instantaneous* physiology: a 19-marker state
vector + per-timestep ODE forward pass. Every dynamics input is
either acute (the meal you just ate, the current activity level, the
time of day) or static (your patient embedding — your endocrinotype).

There is no representation of *adaptation*: the slow accumulator
that turns yesterday's training session into a slightly lower
resting HR this week, or last week's overeating into slightly higher
fasting insulin. No state variable in any module has τ on the order
of weeks.

This matters because most things humans actually want to model
operate at multiple timescales:

| stimulus            | acute (mins-hours)                       | chronic (weeks-months)                              |
|---------------------|------------------------------------------|-----------------------------------------------------|
| Z2 exercise         | HR↑ glucose↓ FFA↑ cortisol↑ lactate↑ temp↑ | Si↑ RHR↓ HRV↑ fasting glucose↓ cortisol diurnal sharper |
| chronic overfeeding | HR↑ insulin↑↑ FFA↓ leptin↑               | Si↓ fasting glucose↑ leptin resistance, ghrelin set-point↑ |
| sleep restriction   | cortisol↑ ghrelin↑ leptin↓ HR↑           | Si↓ inflammation up, ACTH dysregulation, BP↑ baseline   |
| caloric restriction | glucose↓ insulin↓ FFA↑ ghrelin↑↑         | RMR↓ leptin↓ T3↓ cortisol baseline drift                |
| chronic stress      | cortisol↑ glucagon↑ glucose↑             | HPA dysregulation, cortisol curve flattens, Si↓         |

The acute column is what the current architecture can in principle
represent. The chronic column is what *no* current state variable,
signal, or cohort protocol exercises. Worse: cohort-statistic loss
treats any acute deviation from typical as a penalty, which rewards
"predict that exercise is bad" — the acute deviation is large and
the chronic adaptation is invisible.

## North star

A model that, given a patient and a multi-week stimulus protocol,
produces trajectories whose acute responses AND chronic adaptations
both match physiology-textbook expectations within tolerance.
Concretely: a new bench scenario `chronic_exercise_block_8w` that
scores both the acute bout shape (HR peak, glucose drop, recovery
time) and the chronic delta (ΔRHR by week 8, Δfasting glucose, ΔSi
from OGTT slope). When that scenario gates, the platform can do
counterfactual reasoning across weeks-to-months — the primary use
case it exists to enable.

## Mapping the work onto PRD primitives

This plan deliberately avoids introducing new architectural concepts.
Every piece slots into a surface the codebase already has, used in
the way the PRD intends.

A first pass of this doc proposed a new "Adaptation" module — a
seventh subsystem grouping all the slow variables together. That
framing was a band-aid. Anatomically there is no "adaptation organ":
adaptation happens *in the tissue that adapts*. Muscle GLUT4 density
rises in the muscle. Vagal tone shifts in the heart. β-cell
secretory capacity changes in the pancreas. Grouping these into a
separate module is anatomically incorrect and violates the PRD's
"anatomy as structure" principle.

### What we impose: slow state variables in the existing modules
The naturalistic version: **add slow state variables to the modules
that own them**. The dynamics primitive is unchanged — mass-action
for chemical species, learned for vital signs. The slowness isn't a
special "slow integrator" mechanism; it emerges from the physical
reality that large pool sizes and slow protein turnover *are* long
timescales under the standard rate equations.

Per-module additions:

| Module          | New state variable             | Why it's slow                            | Approx τ      |
|-----------------|--------------------------------|------------------------------------------|---------------|
| Metabolic       | `glycogen_pool`                | ~500 g pool, ~50 g/day flux              | ~10 days      |
| Metabolic       | `mitochondrial_capacity`       | Mito protein turnover + PGC-1α biogenesis response (Holloszy 1967 onward) | ~4-6 weeks    |
| Metabolic       | `muscle_glut4_density`         | Protein half-life ~2-3 weeks (well-characterised) | ~14 days      |
| Metabolic       | `beta_cell_capacity`           | Months-years remodelling                 | ~months       |
| Cardiovascular  | `vagal_tone_setpoint`          | Cardiac autonomic remodelling, weeks     | ~weeks        |
| Cardiovascular  | `stroke_volume_scaling`        | Cardiac remodelling                      | ~weeks        |
| Stress          | `crh_pool`                     | HPA neuropeptide reservoir + slow turnover | hours-days   |
| Stress          | `cortisol_baseline_drift`      | HPA flexibility, weeks                   | ~weeks        |
| Appetite        | `leptin_receptor_sensitivity`  | Leptin signalling drift                  | ~weeks        |

`mitochondrial_capacity` is the most mechanistically load-bearing of
these for the chronic-exercise story — it is the structural
explanation for why every other exercise adaptation happens.
Mitochondrial biogenesis (PGC-1α pathway) is the central
exercise-response programme. Within the metabolic module's existing
coupling graph, it drives: FFA utilisation rate (fat oxidation
capacity), lactate clearance (lactate threshold shift — the
canonical Z2 adaptation), BHB production ceiling (ketogenesis
requires hepatic mitochondria), and glucose-vs-fat substrate
selection at submaximal intensity. Supervision: 40% increase in
muscle mitochondrial volume over 8 weeks of aerobic training
(Holloszy 1967); lactate threshold shifts upward with training
(Coyle 1991); fat oxidation rate at submaximal effort scales with
mitochondrial density (Coyle 1995). All cleanly expressible as
cohort-spec mean deltas + physiology rules.

All of these are STATE variables — same status as glucose / insulin /
HR — owned by the module corresponding to their tissue. They obey
the module's existing dynamics primitive. They are unobserved (no
direct measurement) but supervised via cohort statistics + physiology
rules (PRD: "many weak constraints", "calibrated supervision"). Lack
of observation isn't novel — `hepatic_output` is already unobserved.

The cross-module coupling story is handled by the existing coupling
graph: `vagal_tone_setpoint` (cardio) couples into metabolic via a
named edge, just like `glucose` (metabolic) currently couples into
appetite. No new architectural primitive.

### What we learn from each person: still the embedding (and only initial conditions)
The PRD is explicit: "Calibration adjusts only the embedding, not
the model weights." We honour this fully. Per patient, the
embedding now sets the **initial conditions** of the slow state
variables — not the patient's full endocrinotype baked into a 32-d
vector by feel, but a concrete starting state ("this person's
current glycogen pool, muscle GLUT4 density, vagal tone setpoint").
The dynamics then evolve all of it.

This is a second-order win on top of the timescale fix: the
embedding shrinks to its proper job (initial conditions) and
adaptation history becomes part of the state. A patient who started
training 6 months ago is represented as someone with elevated
muscle GLUT4 density and lowered vagal tone setpoint *as initial
conditions on simulation start*. Their identical twin who didn't
train is the same patient embedding mapped to different initial
slow-state values via the trained projection. From there, the
dynamics simulate forward — and they diverge naturally if their
lifestyles diverge.

A useful consequence: counterfactual reasoning becomes mechanistic.
"What if this patient trained Z2 4×/week for 3 months?" → extend
the simulation forward with that activity schedule, and the slow
state variables evolve under the model's learned dynamics. No
special "future-prediction" mode; just integration.

### Time-since-last-X covariates
The `external_inputs` primitive already accepts arbitrary scalar
features per timestep (currently `[sleep_wake, activity]`). Extend
to:
- `time_since_last_meal_min`
- `time_since_last_exercise_min`
- `exercise_dose_trailing_7d` (rolling integral)
- `sleep_debt_trailing_3d` (rolling sleep deficit)

These let the head input directly read "this person hasn't eaten in
14h" without the embedding having to encode it via a sin/cos
proxy. No architecture change — just configuration. The PRD's
"missing data is normal" stance is preserved: each new input has a
sentinel value for "unknown" and the dropout-during-training pattern
already in place handles it.

### Acute-shape supervision: more physiology rules
The `PhysiologyRule` primitive in `knowledge/physiology_rules.py`
was built exactly for per-trajectory hinge predicates. Today it has
5 rules; the literature supports many dozens. For acute exercise
specifically:
- `hr_peaks_during_z2` — HR rises ≥30 bpm above baseline during a
  Z2 bout
- `glucose_drops_during_exercise` — glucose drops ≥5 mg/dL within
  20 min of activity onset
- `ffa_rises_during_exercise` — FFA rises ≥0.1 mmol/L within
  30 min of activity onset
- `lactate_peaks_z2` — lactate peaks at 2-4 mmol/L during Z2
- `recovery_to_baseline` — HR returns within 5 bpm of baseline by
  30 min post-bout

All hinge-shaped, sized to literature evidence (PRD's "calibrated
supervision"), additive to the existing rule library.

### Chronic-delta supervision: more cohort specs
The `CohortStatisticSpec` primitive already accepts arbitrary
population-level mean comparisons between arms. Add multi-week
training-block cohort specs:
- 8-week Z2 vs sedentary control → ΔRHR ≥ 4 bpm
  (Helgerud 2007)
- 12-week aerobic vs control → Δfasting glucose ≤ −3 mg/dL
  (DPP study group 2002)
- 16-week Mediterranean intervention → as cited in PRD's
  differentiability example

The cohort protocol generator needs to know how to simulate weeks of
days with periodic exercise/meal/sleep events. That's a generator
extension, not an architecture change.

### Trajectory coherence: rules that read latent state
A `PhysiologyRule` predicate is `(trajectory, ctx) -> tensor`.
Today it reads observed marker columns; the same predicate API
trivially reads adaptation-state columns from the trajectory if the
trajectory exposes them. Add rules like:
- `si_rises_under_chronic_exercise` — Si_chronic at week 8 ≥
  Si_chronic at week 1 + Δ
- `rhr_drops_under_chronic_exercise` — RHR_chronic by week 8 is
  below week-1 baseline by ≥ 3 bpm
- `si_falls_under_chronic_overfeeding` — Si_chronic drops over a
  caloric-excess multi-week arm

These constrain *how* the Adaptation module's state evolves, not
just whether the right cohort statistics are reproduced. They're
the analogue of the existing `glucagon_falls_postprandial` rule —
qualitative claim, hinge-shaped, literature-cited.

## Why this is natural (PRD audit)

Mapping each PRD principle to this plan:

| Principle                                  | How the plan honours it                                                                                                   |
|--------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| Impose physics / chemistry only            | No new physics. LearnedDynamicsModule for Adaptation (the PRD's pattern for "no closed-form equation").                  |
| Anatomy as structure                       | New module is the *anatomical fact* of slow adaptation — fitness, glycogen, autonomic tone, HPA flexibility.              |
| Medical knowledge as prior, not constraint | Adaptation dynamics learned, not hand-tuned. Literature deltas enter as cohort specs and physiology rules.                |
| Individual variation via embedding only    | Adaptation state initial values projected from embedding. NO new per-patient parameters.                                  |
| Differentiability                          | Every literature finding (training-block RCTs, longitudinal studies) becomes a cohort-spec mean-delta loss.               |
| Many weak constraints                      | Each new rule and each new cohort spec contributes a small gradient on the relevant adaptation axes.                      |
| Calibrated supervision                     | Hinge-shaped, literature-sized. Strong claims tight; "Si goes up" loose.                                                  |
| Epistemological humility                   | New external inputs (time-since-X) follow dropout-during-training; missing data still handled.                            |
| Modularity                                 | Slow variables live in the modules whose tissue they belong to. No separate "adaptation organ" because there isn't one anatomically. |
| Coupling propagation                       | An observed glucose reading constrains acute glucose → constrains glycogen_pool dynamics (intra-module) → constrains hepatic_output / bhb. |

The plan doesn't extend the PRD; it *uses* the PRD as written.

## Iter sequence

### iter 55 — instrumentation + first two slow state variables (combined)
Originally scoped as two iters (55 = instrumentation only, 56 =
slow state variables in metabolic). Merging into one because the
parts are complementary, not orthogonal: instrumentation has
nothing to evaluate without slow states, and the slow states have
nothing to supervise without the chronic-block protocol.

Concrete scope:
1. **External inputs**: extend from `[sleep_wake, activity]` →
   `[sleep_wake, activity, time_since_last_meal_min,
   time_since_last_exercise_min, exercise_dose_trailing_7d]`.
   Sentinel values for unknowns; dropout-during-training pattern
   already in place.
2. **State extension**: metabolic module 7 → 9 species. Add
   `glycogen_pool` (typical=500 g, τ ≈ 10d) and
   `mitochondrial_capacity` (typical=1.0 unitless multiplier of
   population mean, τ ≈ 4-6 weeks). Both `SetpointHead`. Both
   unobserved.
3. **Intra-metabolic coupling**: hepatic_output / bhb / ffa /
   lactate heads can read the two new slow states. New coupling-
   input dimension; weights zero-init so iter 55 starts
   behaviourally identical to iter 54.
4. **Initial-condition projection**: shared embedding → 2-dim
   initial-values vector. Zero-init weights so all patients start
   at population mean; literature deltas pull weights up only if
   supervision rewards it.
5. **Synthetic protocol**: `chronic_exercise_block_4w` (M/W/F
   30-min Z2 sessions × 4 weeks). Added to the cold-distill pool.
6. **Bench scenario**: `acute_z2_bout` (30-min Z2 bout, scored
   against textbook response shape — HR peak in zone, glucose
   drop, FFA rise, recovery time).
7. **Supervision**:
   - Cohort spec: glycogen_pool drops 40-80 g over
     `_FAST_8H_TO_16H_ARM` (Cahill 2006).
   - Cohort spec: mitochondrial_capacity rises ≥ 30% over
     `chronic_exercise_block_4w` (Holloszy 1967).
   - Physiology rule: glycogen depletion correlates with bhb rise
     during prolonged fast.
   - Physiology rule: lactate clearance scales with
     mitochondrial_capacity during Z2 bout.

The "bolder" call: this is a larger iter than usual, but the
zero-init story keeps it a strict superset of iter 54's
behaviour at init. Any divergence is downstream of supervision
pulling on the new degrees of freedom — which is exactly what we
want.

(Scope-trimmed at execution to: state extension + the single
`EXTENDED_FAST_GLYCOGEN` cohort spec only. external_inputs,
chronic_exercise_block, acute_z2_bout, mito supervision all
deferred — attribution-clean test of whether the slow-state
mechanism works at all before piling on.)

### iter 55 — RESULT (2026-05-16): mechanism inert (R2 confirmed)

Training completed (commit `0b9ea0f`); benchmark blocked first by a
stale 19-D dataset (state grew 19→21; fixed in `d6c964c7` —
benchmark-dataset migration + re-benchmark of the saved checkpoint).

Observed markers: overall_weighted_mape 0.114 → 0.124 (+0.010,
mild regression), textbook 0.694 → 0.718 (+0.024), verifier
0.784 → 0.816 (+0.032), `verifier_meal` gate **cleared**. The
glp1↔ghrelin swing (glp1 −0.455 recovered, ghrelin +0.714 broke)
is the known embedding-bottleneck capacity reshuffle (see
dead-pathways.md), not attributable to the glycogen architecture.

Core hypothesis: **FAILED**. A pure forward rollout of the saved
checkpoint on the `EXTENDED_FAST_GLYCOGEN` arms (default embedding,
no calibration) gives glycogen_pool window-mean 499.99 g
(extended_fast) vs 500.00 g (normal_eating) — **Δ = −0.01 g vs the
−60 g target**. The state never moved off `typical`.
mitochondrial_capacity drifted 1.000 → 1.007 (≈silent, as expected
for an unsupervised surface).

Root cause = pre-registered risk **R2**, sharpened: a *single
lumped 500 g pool* with `cons_scale = 7e-5` (τ ≈ 10 days) **cannot
express a −60 g delta over a 1-day (1440-min) protocol**. The
SetpointHead is structurally able to read fasting signals, but the
rate is scaled by `cons_scale`; with τ ≈ 10 d the maximum
1-day excursion is a fraction of a gram. The memory's mitigation
("raise cons_scale 10×") is rejected: it would make the *entire*
500 g pool (incl. muscle) deplete in a day, which is
physiologically wrong and destroys the slow-state purpose. The
lumped pool was a first-draft simplification; the timescale
contradiction is structural, not a tuning miss.

### iter 56 — split glycogen by tissue (the structural fix)

Anatomy as structure (the plan's own core principle): there is no
single "glycogen pool" organ. Liver glycogen and muscle glycogen
are different tissues with different turnover, different drivers,
and different timescales. Lumping them forced one `cons_scale` to
serve two incompatible roles. Split:

- **`liver_glycogen`**: typical ≈ 100 g, τ ≈ 1 day
  (`cons_scale ≈ 7e-4`). The overnight-depleting pool. Largely
  exhausted by a ≈16 h fast (Cahill 2006) — so the −60 g
  fast-vs-fed delta is now *physically reachable* within the
  1-day protocol while keeping τ honest.
- **`muscle_glycogen`**: typical ≈ 400 g, τ ≈ 3 weeks
  (`cons_scale ≈ 3.3e-5`). The genuinely-slow, exercise-coupled
  state — this is the chronic-exercise north-star reservoir
  (Move D). Preserved at rest during a 1-day fast
  (Coppack 1989), so its fast-arm delta ≈ 0.

Supervision (attribution-clean — *exactly one* mechanism, nothing
else this iter):
- `EXTENDED_FAST_GLYCOGEN` retargeted to `marker_id="liver_glycogen"`,
  target −60 g (σ 25) over the existing fast-vs-normal arms. This is
  the only new/changed supervision.
- `muscle_glycogen` left unsupervised (like mitochondrial_capacity).
  It is a *separate* state variable with its own SetpointHead, so the
  liver spec's gradient does not flow into it — no "muscle preserved"
  companion spec is needed. (A Δ≈0 target on a zero-init SetpointHead
  carries no gradient anyway: pred==target==0 at init — it would only
  be a degenerate no-op that also trips the cohort-ablation
  zero-gradient guard.) Muscle preservation under a resting fast
  falls out of its slow τ + absence of an exercise drive; its
  exercise supervision lands in iter 57.
- mitochondrial_capacity unchanged (architecture only, still
  unsupervised — its supervision waits until the slow-state
  *mechanism* is proven by liver_glycogen actually moving).

Verification when it lands: re-run the same forward-rollout probe;
success = liver_glycogen window-mean drops ≥ 35 g under
extended_fast vs normal_eating, muscle_glycogen stays within ±10 g,
observed-marker overall_mape does not regress beyond iter-55's
0.124.

### iter 56 — RESULT (2026-05-17): timescale fixed, gradient starved (R5)

Training completed (commit `f78f548b`, exec `pulse-trainer-8mrfl`);
in-job benchmark loaded 26 episodes cleanly (22-D migration works).
Forward-rollout probe of the saved checkpoint:

- liver_glycogen: normal_eating window-mean 98.81 g vs extended_fast
  98.66 g → **Δ = −0.15 g** vs the −60 g target (z = 2.39). Liver
  drifted only ~1.4 g off 100 g typical, *near-identically in both
  arms* — no fast-vs-fed differentiation.
- muscle_glycogen: frozen at 400.00 g (unsupervised, as expected).
- mitochondrial_capacity: drifted 1.00 → ~0.92 (unsupervised noise).

Observed markers (in-job benchmark, train-20260516T104208Z):
overall_weighted_mape **0.0857** (iter 55 was 0.124, iter 54
0.114 — best in the arc), textbook 0.741, verifier 0.778,
glucose 0.206 (iter 55: 0.291), sbp/dbp/temp ≈ 0.02. So the
tissue split, though it *failed its primary glycogen hypothesis*,
**substantially improved observed markers** — two well-separated
SetpointHead pools relieved embedding-bottleneck pressure vs the
one over-constrained lumped pool. ghrelin 1.63 / glp1 0.95 are
the usual dead-pathway reshuffle, not glycogen-attributable.
**0.0857 is therefore the real iter-57 observed-marker baseline.**

**FAILED again** [primary glycogen hypothesis] — but the timescale
fix worked (liver *can* now move within a 1-day window; it just
didn't). This is the pre-registered **R5**: the binding constraint
is the cohort-statistic gradient path, not τ. Cohort-ablation on a fresh
model, identical fast-vs-fed arms:

| spec (same arms)   | metabolic grad norm |
|--------------------|---------------------|
| `glucose`          | **739.9**           |
| `ffa`              | 0.27                |
| `liver_glycogen`   | **0.11**  (~6700× weaker than glucose) |

Supervising a small-`cons_scale` SetpointHead via an indirect
window-mean cohort delta produces a gradient 3-4 orders of
magnitude weaker than the observed-marker specs sharing the
*same arms*. In the summed multi-spec cohort loss, Adam treats it
as noise — `target_z` never moves off typical. Compounding it: the
SetpointHead imposes an equilibrium-at-typical attractor that
*actively pulls glycogen back up*, cancelling depletion. The head
*can* see the distinguishing signal (gut glucose-appearance is in
its coupling inputs) — it never learns to use it because the
gradient is starved. Cohort loss did fall during training
(3.28→2.59) but by satisfying the 11 loud specs; liver's z stayed
~2.4 throughout.

**Structural conclusion:** glycogen is not a homeostatically
regulated *setpoint* species — it is a **flux integrator**:
dGly/dt = synthesis(glucose/insulin available) −
breakdown(fasting/glucagon). SetpointHead is the wrong primitive
for it (it was designed for dead-pathway markers that genuinely
*do* relax to a setpoint). Two iters proved a setpoint-relaxation
state cannot be driven off `typical` by an indirect cohort delta.

### iter 57 — GlycogenFluxHead (mechanistic, signed coupling)

Replace the glycogen SetpointHead with a purpose-built
`GlycogenFluxHead` (precedent: `GlucoseGatedInsulinHead` is already
a custom mechanistic head in the same module — this is an
established pattern, not a new architectural concept):

- **synthesis flux** gated by glucose-appearance / insulin from the
  module's gut-coupling inputs. Sign +, magnitude learned. Glycogen
  fills when nutrients are being absorbed.
- **breakdown flux** gated by the fasting signal (low insulin and/or
  absent gut absorption; glucagon). Sign +, magnitude learned.
  Glycogen drains when fasting.
- rate = synthesis − breakdown. No equilibrium-at-typical attractor
  — the pool level is whatever net flux integrates to, which is what
  a glycogen pool physically is.

Why this fixes the gradient starvation: the −60 g delta now flows
through the *same strong gut-coupling pathway* that gives the
`glucose` cohort spec its 739 gradient, not through a
cons_scale-shrunk setpoint reparam. Gradient magnitude becomes
comparable to the observed-marker specs, so the optimizer can't
ignore it. This is the PRD "impose chemistry + encode the sign of
effects, learn the strength" principle applied correctly.

Both `liver_glycogen` and `muscle_glycogen` move to the new head
(muscle's synthesis/breakdown additionally gated by exercise —
which sets up the iter-58 chronic-exercise story cleanly). τ
separation is preserved via the per-flux scale, not a lumped
`cons_scale`. Supervision unchanged from iter 56 (liver −60 g spec
only; attribution-clean). mito stays SetpointHead+unsupervised
(it genuinely *is* a slow setpoint-like adaptation variable, not a
flux pool — the setpoint primitive is correct *for it*).

Verification: same forward-rollout probe. Success = liver drops
≥ 35 g under fast vs fed, muscle within ±10 g, observed
overall_mape not worse than iter-56's. If liver STILL doesn't move
with a direct mechanistic flux path + comparable gradient, the
problem is upstream of the head entirely (cohort-loss formulation
or the rollout-through-integrator gradient) — that would redirect
to a direct per-step glycogen supervision signal rather than the
population-delta cohort spec.

If iter 57 works, *then* resume the deferred scope (external_inputs,
chronic_exercise_block_4w, acute_z2_bout, muscle + mito exercise
supervision).

### iter 57 — RESULT (2026-05-17): rejected on observed-marker regression; core hypothesis unmeasured

Run `train-20260517T021059Z` (exec `pulse-trainer-b46zk`). Gate
**FAILED** (baseline iter-56 `train-20260516T104208Z` also FAIL):

- `overall_weighted_mape` **regressed 0.0857 → 0.1189** (+0.033).
- `glucose_mape` worse: 0.206 → 0.283 (gate: `>0.2`).
- New failure `verifier_cat[meal]=0.630 < 0.65`.
- Verifier overall +0.018 (0.778 → 0.796), textbook pass +0.071 —
  small structural wins, swamped by the glucose/MAPE regression.

Per-marker MAPE Δ (− = better): ghrelin **−0.40**, acth −0.19,
insulin −0.059 improved; glp1 **+0.66**, glucagon +0.27, ffa +0.10,
glucose +0.077 regressed. This is exactly the **R5 embedding-
bottleneck reshuffle** the spec pre-registered as accepted noise —
and exactly the attribution-noise floor `physiology-coverage.md`
exists to kill. The head change helped the markers it touched and
the bottleneck paid for it elsewhere, uncorrelated with the
mechanism under test.

**Core hypothesis (does liver_glycogen now move ≥35 g fast-vs-fed)
is UNMEASURED from available artifacts.** The compare step's
baseline probe was skipped (state_dict mismatch — iter-56 ckpt
can't load the new head shape; expected and unavoidable on a head
swap), and the only probe captured was the gut-dose glucose probe,
not the EXTENDED_FAST_GLYCOGEN rollout. `benchmark-report.json`
carries no glycogen probe. So R6 ("liver still doesn't move")
cannot be adjudicated here; the reject is purely on the observed-
marker regression, which is decisive on its own.

**Decision: reject iter 57; proceed to the committed breadth-floor
iter 58** (`physiology-coverage.md`, decided direction). GlycogenFlux
Head stays in the codebase unchanged — the breadth floor is
orthogonal (it supervises the ~9 unanchored *observed* markers that
generate the R5 reshuffle noise) and attribution-clean. Re-measuring
glycogen depth is deferred to the post-floor depth campaign, when
measurement is no longer read through that noise.

### (superseded) iter 56 — generalise the pattern across modules
Add `glycogen_pool` and `mitochondrial_capacity` to the metabolic
module's state simultaneously. Both belong here (both are
muscle-dominant tissue properties driving metabolic dynamics), both
are first-order for the chronic-exercise story, and they share the
same wiring template — efficient to add together.

**`glycogen_pool`** (liver + muscle):
- Well-quantified physically (~80 g liver + ~400 g muscle = ~500 g
  pool; flux 30-50 g/h post-meal absorption; 12-24h fast depletes
  liver pool). Mass-conservation maths gives τ ≈ 10 days for free.
- Fast enough that a single fasting protocol moves it observably.
- Coupling within metabolic: drives `hepatic_output` (gluconeogenesis
  substrate when low), gates `bhb` (ketosis fires when exhausted),
  modulates `ffa` (lipolysis ramps as glycogen depletes).
- Supervision: cohort-mean glycogen drops 40-80 g over 8h fast
  (Cahill 2006); physiology rule: glycogen depletion correlates with
  bhb rise during prolonged fast.

**`mitochondrial_capacity`** (muscle-dominant, whole-body abstraction):
- The mechanistic anchor for the chronic-exercise story. Drives
  fat oxidation rate, lactate clearance (the lactate-threshold
  shift), BHB production ceiling, glucose-vs-fat substrate selection.
- τ ≈ 4-6 weeks (mito protein turnover + PGC-1α biogenesis response).
- Coupling within metabolic: FFA's SetpointHead reads it (higher mito
  → FFA decay constant grows), lactate's head reads it (clearance
  rate scales), bhb's SetpointHead reads it (ceiling).
- Cross-module coupling: out to cardiovascular as a VO2max proxy
  (limits HR/SBP response curves at high intensity).
- Supervision: cohort-mean Δ mitochondrial_capacity ≥ +30% over 8
  weeks of aerobic training (Holloszy 1967); physiology rule:
  lactate-threshold shift correlates with mitochondrial_capacity
  delta (Coyle 1991); fat-oxidation rate at submaximal intensity
  scales with mitochondrial_capacity (Coyle 1995).

Wiring (same template for both):
- Extend metabolic state from 7 to 9 species.
- Initial value per patient: embedding-projected scalar (same pattern
  as every other state's initial condition).
- SetpointHead with appropriate `typical` (500 g for glycogen; 1.0
  for mitochondrial_capacity as a unitless multiplier of population
  mean).
- Mass-action dynamics: `prod` and `cons` reflect anabolic and
  catabolic fluxes appropriate to each variable.
- Coupling: glycogen_pool and mitochondrial_capacity both readable
  by other metabolic heads via the existing intra-module input
  channel.

Risk: both are unobserved state variables whose initial values are
embedding-projected. Bench calibration adjusts the embedding to
match observed markers, but the slow-state initial values are
invisible to it — they land wherever the projection puts them.
Mitigation: initial-condition projection weights start near zero so
all patients start at population mean; training pulls weights off
zero only if literature deltas reward it.

### iter 56 — generalise across the other modules
(Was iter 57 before the iter-55/56 merge.) Add the rest of the slow
state variables, each into the module that owns the tissue:

- Metabolic: `muscle_glut4_density` (insulin-sensitivity proxy),
  `beta_cell_capacity`
- Cardiovascular: `vagal_tone_setpoint`, `stroke_volume_scaling`
- Stress: `crh_pool`, `cortisol_baseline_drift`
- Appetite: `leptin_receptor_sensitivity`

If iter 55 went smoothly, this can be one large iter; if iter 55
exposed integration issues at multi-scale, stagger into iter-56a
(rest of metabolic), iter-56b (cardio), iter-56c (stress +
appetite).

Each follows the same iter-56 template: state variable in its
module, embedding-projected initial value, dynamics through the
module's existing head type, supervision via cohort specs +
physiology rules. Cross-module coupling for variables that affect
markers outside their own module (e.g. vagal_tone_setpoint in cardio
couples into metabolic via the existing coupling graph).

If iter 56 went smoothly, this can be one large iter; if iter 56
exposed integration issues at multi-scale (it shouldn't, but in
case), this can stagger into iter-57a (cardio), iter-57b (stress),
iter-57c (appetite).

### iter 57 — multi-week protocols in production
Extend cold-model distillation to multi-week protocol arms. This
requires either (a) extending `simulate_full_body` to model chronic
adaptation (real effort) or (b) generating multi-week protocol
arms whose targets come from literature deltas rather than ODE
output. (b) is faster and PRD-aligned ("knowledge as data" plus
"differentiability").

### iter 58+ — hormesis-aware loss; bench scenarios
First-class hormesis scoring: a model that says "exercise is bad"
should LOSE relative to one that says "spike then recover." This
likely requires a new acute-shape scoring signal that rewards
spike-then-recovery patterns. The `physiology_rules` surface
already supports the predicate shape; we just need rules in that
form.

`chronic_exercise_block_8w` bench scenario as the gate. When it
passes (acute shape within 20%, ΔRHR ≥ 3 bpm, Δfasting glucose ≤
−3 mg/dL), the platform has counterfactual-over-time competence.

## Open design questions

1. **Integration timestep for slow states.** Main ODE runs at 1-5
   min; updating Si_chronic at every step is wasteful (τ = weeks).
   Options: (a) subsample slow-state updates (every 60 min); (b)
   parameterise the slow integrator as a closed-form decay-toward-
   target with state update at simulation step that integrates
   the target-and-time-elapsed. Probably (b) — it's a one-line
   update and avoids subsampling complexity.
2. **Adaptation state at calibration time.** The bench calibrates
   the embedding from 256-step observed windows. The embedding
   projection gives initial Si_chronic; the simulation then
   evolves it. So calibration is unchanged — only the embedding
   moves. PRD-compliant.
3. **Knowledge-model chronic coverage.** The cold ODE
   (`simulate_full_body`) doesn't currently encode chronic
   adaptation. Two paths: extend it (large effort, but gives
   ground-truth trajectories for cold-distill multi-week protocols);
   or skip cold-distill for multi-week arms and rely purely on
   cohort-statistic + physiology-rule supervision (PRD's "many
   weak constraints" framing). Probably the second to start.
4. **Coupling sign priors for Adaptation outputs.** The PRD's
   coupling-graph principle says we encode the SIGN of effects and
   learn the strength. Si_chronic → insulin peak amplitude is
   suppressive (higher Si = lower insulin needed for same glucose
   response). RHR_chronic → cardiovascular HR setpoint is positive
   (resting HR baseline is the resting HR baseline). These signs
   go into the coupling priors.

## Success criteria

The plan works when `chronic_exercise_block_8w` passes as a bench
gate. The signature would be:
- Acute bout shape within 20% of textbook ranges
- ΔRHR week-8-vs-week-1 ≥ 3 bpm
- Δfasting glucose ≤ −3 mg/dL
- All markers stay within physiological bounds throughout
- The model produces qualitatively correct trajectories on
  counterfactual queries ("what if this patient does Z2 4×/week for
  3 months?") that an external clinician would recognise as
  physiologically grounded.

That's the platform's primary use case. When it lands, Pulse stops
being a 19-marker autoregressive imputer and starts being a
multi-timescale digital physiology twin.

## What this doc replaces

The architecture-roadmap.md "Move D" line (latent reservoirs).
That was a single-line gesture toward this work; this doc
supersedes it as the concrete plan.
