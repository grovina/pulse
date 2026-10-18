# Plan — one physiology, found by the literature

*Written 2026-10-17. Supersedes the 2026-10-09 plan ("a body a gradient
can move"). That plan's hard-won measurements still bind; they are
restated in §10. The completeness map remains `docs/modeling-state.md`.
The authoring surfaces remain `pulse/knowledge/AUTHORING.md`.*

*This plan changes direction. The 2026-10-09 plan replaced stalled
learned laws with the teacher's constants (iters 98–109: 68,293 → 37,343
parameters) and then landed eight physiology hubs the same way — a
mechanism copied from the teacher, confirmed by a noise-free test, with
no degree of freedom a later paper could move. That is a copy, and a
copy cannot be overturned by evidence. This plan unfreezes those
numbers, turns each citation into evidence on the protocol it was
measured in, and sets a course where the teacher is removed.*

## 1. Destination

One simulation of human physiology, found by the scientific record. Every
quantity a physiologist measures in a body is learned. The only things
imposed are the ones that are not about physiology at all: the
conservation laws of physics, the stoichiometry of chemistry, and the
anatomical fact of which organ connects to which.

Each published finding is a soft constraint on a reproducible protocol —
a slice with a hole in it. One slice barely narrows the model. Enough
independent slices, each cut from a different study, intersect on the one
model that passes through all of them. That intersection is the
physiology we are hunting. Personalization is part of the same object: a
population law plus a distribution over people, so the slices that report
spread and correlation narrow the person axis the same way a mean narrows
the law.

The simulation is never "done." It is more or less constrained. A later
paper can always move it, including a law that today sits at a value
nothing has yet measured against.

## 2. What is imposed, and what is learned

The hierarchy is the PRD's. This plan sharpens the boundary and holds to
it without exception.

| Layer | Nature | In the model |
|---|---|---|
| Physics / chemistry | Universal laws | Enforced in the architecture: conservation of mass, the carbon ledger, positivity, 4/4/9 kcal/g, the Weir equation, unit conversions |
| Anatomy | How the body is built | Module boundaries and the coupling graph: which marker reaches which module |
| Physiology | What the body does | **Learned.** Every rate, threshold, volume, gain, time constant, fraction and setpoint |
| Individual variation | This person | Learned from their own data, as a draw from the population |

What is imposed is the **space**, never a **point** in it. "Glucose above
some threshold leaves in the urine" is anatomy: the pathway exists. The
threshold is 190 mg/dL *today*, initialised there, with nothing pulling
it back. A tracer study moves it; a better study moves it again.

Three consequences, which are the rules of this plan:

- **No forced value.** A number a physiologist could measure is a
  learnable population parameter, initialised at the current best value,
  with no restoring pull toward it. `2.13 kcal/min`, `190 mg/dL`, the
  potentiator's `0.025`, glucagon's `3.5`, the incretin split, the bile
  floor, the sleep-debt gains — all of them. The value today is a
  starting point, not a target.

- **Functional forms are defaults, not constraints.** Mass action, a Hill
  curve, the one-compartment stomach — each carries a learned correction
  term, zero-initialised, so the truth is reachable if the shape is
  wrong. Where the evidence cannot decide between models the tie-break is
  the simplest one (corrections near zero, by a shrinkage term on the
  correction only). That is a preference among models the evidence
  allows, not a value imposed on physiology.

- **A parameter that drifts into nonsense means a slice is missing.** In
  iter 107 obligatory uptake walked 0.0114 → 0.0083 because every slice
  then written permitted it. The fix is never to pin the value. It is to
  write the finding that forbids the drift (Landau's tracer shares, the
  prolonged-fast floor) and let it hold the parameter by evidence. The
  identifiability report (§6, Phase 0) is the standing list of which
  parameters no slice yet constrains, each paired with the study that
  would constrain it. That list is the reading list.

## 3. What a finding is

The native training example is an `EvidenceItem`
(`pulse/knowledge/evidence.py`): a protocol, an observation operator, a
reliability, and an authority. **Knowledge acts on protocols and
observables, never on the model's internal parameters.** A prior on a
parameter is meaningful only inside one way of writing the model; an
observation on a protocol is meaningful to every model, which is what
lets slices accumulate without being rewritten when the structure
changes.

A finding enters at the smallest surface that captures it.

| The finding | Where it goes |
|---|---|
| X raises or lowers Y, with a magnitude band | `CouplingPrior` in `pulse/knowledge/coupling_priors/` |
| In protocol P, marker M had statistic S ± σ over window W | `CohortStatisticSpec` in `pulse/knowledge/cohorts/` |
| In protocol P, marker M followed this curve (mean ± SEM at timepoints) | `CohortStatisticSpec` with `StatisticKind.CURVE` — the replacement for the teacher's dense tape |
| Across people, marker/parameter M had spread S or correlation R | `StatisticKind.SPREAD` / `StatisticKind.CORRELATION` — the replacement for teacher-drawn setpoint supervision |
| Under an IV glucose, insulin, or hormone infusion (incl. a clamp) | an infusion arm (`CohortArmSpec` infusion/clamp series), then any operator above |
| A direction, floor, ceiling, or timing on one trajectory | `PhysiologyRule` in `pulse/knowledge/physiology_rules.py` |
| A dose scales a peak, rank, or slope | `OperatorKind.DOSE_*`, scored by `RolloutEvidenceSignal` |
| For one subject, these sparse readings under this protocol | an individual-level item (`authority=real`), scored like the real-data ruler |
| Conservation, positivity, ordering, a fixed point | the coordinates or functional form, in `pulse/modules/` |

Every item carries a sigma, a source string, and an authority. Literature
and real measurements outrank the teacher on any marker they both cover.
A finding with no sigma is a hinge or a prior, not a cohort target with an
invented one.

**The slice is soft.** A sigma tighter than the study supports can wrongly
exclude the truth, so two items on the same protocol and marker whose
targets disagree beyond their sigmas are recorded as a conflict (§6, M3.2),
not averaged into a number neither paper reported. A conflict that is
real — the model cannot satisfy both — is the signal to change structure,
not to widen a sigma. This is the PRD's "small persistent discrepancies
are structural signals."

## 4. The scoreboard, not a gate

There is no binary gate. Every checkpoint is scored on a continuous
board, and a run that regresses on one line tells us something; it is not
rejected. Three lines:

- **Fit to held-out literature.** The median miss on the sealed evidence
  set, in units of each item's own sigma, and the share of sealed items
  within 1σ and 2σ, by protocol family.

- **Individual prediction skill.** On held-out people from public
  individual-level datasets: calibrate on sparse early readings, predict
  later readings under a different protocol. Skill is reported against two
  baselines — the population-median person, and the person's own running
  mean.

- **How open the model still is.** The count of parameter directions the
  current evidence does not constrain (the Fisher spectrum of §6, Phase
  0), each flat direction named with the study that would close it.

Teacher-tape agreement is logged. It is not on the board.

## 5. The teacher is scaffolding

The teacher (`pulse/knowledge/full_body.py`) is a lossy copy of the
literature with no sigmas, no citations, and invented minute-by-minute
detail. It is a warm start in a reasonable direction and a generator of
protocols. It is on a path to removal. It does five jobs today; each
needs a literature replacement before it can go, and each is removed by
ablation — set its signal to zero weight, and keep the removal if the
scoreboard holds.

| Teacher role today | Literature replacement | Removed in |
|---|---|---|
| Dense tape of glucose/HR/BP/temp | `CURVE` items: digitized 24 h profiles, mean ± SEM at timepoints | Phase 6 |
| Distilling internals (CRH, remote insulin, bile pools, fat mass) | latent states with priors on time constants; report as unidentified if nothing pins them | Phase 6 |
| Generating the population (setpoint + sweep supervision) | `SPREAD` / `CORRELATION` items from published distributions; fresh draws from a named latent | Phase 3 → 6 |
| Ground truth for identifiability and calibration-recovery tests | synthetic people sampled from the trained student itself | Phase 3 |
| Stable cold start | the literature's own constants as the init | kept until last |

The ablation is the test for when each role can go: train with that
teacher signal at weight zero, and keep the deletion if the held-out
literature fit and the individual-prediction skill do not fall.

## 6. The phases

The phases below are ordered by dependency, not calendar. Phase 0 and the
Phase 1 run start together. Phases 2–4 can proceed in parallel and must
all land before Phase 5.

### Phase 0 — the scoreboard, before anything is tuned

The board of §4 exists before any run is judged by it.

- **A sealed evidence set.** Evidence items carry a `sealed` flag. Sealed
  items never enter any training registry (`ALL_COHORT_STATISTICS`,
  `PHYSIOLOGY_RULES`, `all_evidence()`); they are reachable only through a
  scoring path. New items are sealed at random (about one in five) within
  each protocol family at filing time, so no one seals only the easy
  ones. Existing items stay in training — the warm start has already seen
  them. A test asserts no sealed item appears in any training registry.

- **The identifiability report.** A checkpoint artifact: the eigen-
  spectrum of the Fisher information of the total training loss with
  respect to the population parameters and the per-person heads. Every
  flat direction is either covered by a slice (named) or listed as open
  (named, with the study that would close it). This is the "map of
  holes." It is a run artifact, never a reason to delay a run, and it is
  the thing that tells us a parameter may be unfrozen without drifting.

- **An individual-level ruler.** One public dataset ingested as sparse
  calibrate-then-predict episodes (CGMacros first; the BIG IDEAs Lab set
  and OhioT1DM next). The single-subject CGM ruler stops being special:
  it is one `real` item among several.

### Phase 1 — one baseline run

Rewrite the iter 112 spec for the current 39-state tree (it was written
for 31 states), warm-start from iter 111, and score it on the Phase 0
board. This is both the number to beat and the warm start for everything
after. The teacher mix is not retuned after this run.

### Phase 2 — every measured number becomes a learnable parameter

Unfreeze every forced physiological constant (the inventory is the
explore pass of 2026-10-17; it is ~150 numbers, ~40 already learnable).
Each becomes an `nn.Parameter` on its owning module, in log or logit
space for positivity or bounds, initialised at today's value, **with no
prior pull**. A module that is fully forced today (`CircadianModule`) and
the hub files (`incretin.py`, `renal.py`, `bile.py`, `expenditure.py`,
`sympathoadrenal.py`) are where this changes the most.

Unfreezing alone is net-negative: a free parameter the loss does not
identify drifts (iter 107). So unfreezing is coupled to writing the slice
that identifies it. Each landed hub already cites the paper it copied;
that paper becomes an evidence item on its protocol, using the operators
of Phase 4:

- renal threshold / clearance / empagliflozin → a `CURVE` or cohort on
  the Diabetes 2017;66:1999 urinary-glucose regression.
- the incretin split, GIP ratio, GLP-1-on-glucagon → cohorts on Vilsbøll
  2003, Vollmer 2008, Salehi 2008.
- the bile floor → a cohort on Borgström 1985.
- gas exchange → a `CURVE` on Wahren 1971 Table I.
- sleep-debt gains → the Donga 2010 clamp ratios, via an infusion arm.

`PopulationPriorSignal` (the four former pins under a log-normal / Beta
prior) is removed only when the identifying slices exist — Landau's
tracer curve for `f_gng` and `k_ii`, Eaton's clearance for `k_ffa`. Until
then it stays, because removing a pin with no slice behind it restores
the iter-107 drift. This is the one place a prior survives, and it
survives only as a bridge.

### Phase 3 — the population becomes a distribution

Replace the 40-row memorised embedding table with a prior over a small
set of named factors, sampled fresh each step: insulin resistance,
β-cell function, adiposity, fitness, chronotype, HPA reactivity, gastric
emptying, plus sex and age. Named, because published cohorts are
restricted populations: each arm declares the population it was measured
in ("healthy, 20–35 y, BMI 20–25").

- The `SPREAD` and `CORRELATION` operators carry published distributions,
  replacing teacher-drawn setpoint and sweep supervision.
- Arm initial states come from the student's own spin-up (a standard fed
  day, then the protocol's pre-fast), not a teacher row — so even cold
  cohort arms stop depending on the teacher.
- The calibration-window question (can a glucose window separate Gb from
  Si?) becomes a recovery test on people sampled from the student.

Done when setpoint supervision and the sweeps can be switched off without
the between-person spread collapsing.

### Phase 4 — the operators the corpus needs, each added once

Each operator lands once, then every later paper of that shape uses it.

- **`CURVE`** — mean ± SEM at published timepoints, with a down-weight for
  the correlated error between adjacent points. Replaces the dense tape.
- **Infusion / clamp arms** — IV glucose, insulin, or hormone as source
  terms threaded through `integrate` the way the SGLT2 input already is; a
  clamp is an infusion under a differentiable proportional controller
  that holds a target, with the infusion rate as the observed statistic.
  Unblocks Donga's clamp ratios, Cryer's thresholds, Salehi's exendin
  arm, and the whole clamp / IVGTT literature — the gold standard for
  insulin sensitivity.
- **Individual-level item** — sparse per-subject observations from public
  datasets and supplementary tables.
- **Per-item residuals and conflict records** — written next to every
  checkpoint (M3.2), so a break (a residual moving from inside σ to well
  outside it) and a conflict (two items that cannot both be met) are
  visible.

### Phase 5 — invert the training loop

Today the loop walks teacher windows and runs a literature step every k
windows (iter 97 measured literature taking 25 of a day's optimizer
steps). Flip it: each step samples a batch of evidence items, simulates
their arms on population draws, and sums their Gaussian log-likelihoods.
**Sum, not average** — adding a paper then adds gradient instead of
diluting the others (the iter-74 dilution disease). The teacher terms
become one extra term whose weight decays to zero across the run.

### Phase 6 — remove the teacher, one role at a time

By ablation, in this order: the dense tape (→ curves), the teacher
meal-response item, the sweeps, setpoint supervision (→ spread /
correlation), distillation (internals → latent states with priors). Each
removal is kept if the Phase 0 board holds. Then delete `full_body.py`,
the distillation and sweep signals, and the teacher-side tests.
Ground-truth checks sample from the student instead.

### Phase 7 — read at scale

Writing evidence items becomes the main activity and the bottleneck. An
agent reads a paper, writes the item (protocol, operator, population,
sigma, citation), runs the gradient check (`cohort-ablation`) and the
conflict check, and the item is sealed at random or released before
review. The reading order: first the metabolic canon where the states
already exist (OGTT, IVGTT, clamp, mixed meal, fasting), then exercise,
sleep, and cardiovascular; within a hub, its §-equivalent evidence list.
Textbooks first (a sign or time constant, one line each), then the papers
they cite for a number, then primary papers that contradict an item
already in the corpus — a contradiction is a conflict record, and it is
how a copied teacher law gets overturned.

## 7. Risks

- **Compute.** Making evidence the main step means many protocol rollouts
  per step. Short protocols dominate; long ones (48 h fasts, training
  blocks) run on a slower cadence until multi-rate stepping exists (M2.9
  in the old plan).
- **Identifiability.** Many parameters will sit at their init. That is
  honest; the report names each and the study that would move it. The
  risk is a parameter the loss *does* reach for the wrong reason (k_ii
  cancelling at the fixed point) — caught by the report, fixed by the
  slice, never by a pin.
- **Population mismatch.** Averaging over the wrong people biases every
  cohort item. Arm populations (Phase 3) must land before the loop
  inversion (Phase 5).
- **The model space may not contain the truth.** Then two well-measured
  findings cannot both be met and the holes stop overlapping. That
  conflict is the signal to change a functional form, which is why the
  correction terms of §2 exist and why conflict records (Phase 4) are
  first-class.

## 8. How a step is judged

A structural claim (a fixed point, a conservation bound, a decode family,
an operator's gradient) is a test. It passes for every embedding and
every protocol, or it does not. It needs no checkpoint.

Anything that changes what training can learn needs a run, scored on the
Phase 0 board against the previous accepted checkpoint on a frozen ruler
with a matching `ruler_fingerprint`. Accepted means: held-out literature
fit does not fall, individual-prediction skill does not fall, the
open-direction count is reported, and no slice already in the corpus
breaks. Teacher agreement is logged, not judged.

## 9. The queue

`pulse/knowledge/queue.py` is the list of findings not yet in the
registries. The "blocked on a hub" status is now mostly stale: the hubs
landed, and the real blocker is a missing operator (an infusion/clamp
arm, a pmol/L table, a curve). Phase 4 removes those blocks. An item is
**ready** when its marker exists, a gradient can reach the pathway, and
the arm we can run is the paper's protocol; an agent lands it, seals it at
random or releases it, and sets its `spec_name`. The ready list is
allowed to be empty. When an agent finishes an item it reads another paper
and files it.

## 10. Measurements that still bind

These were measured while writing earlier plans. A later item that
contradicts one of them has to remeasure it. They are facts about the
model and the teacher, and they survive the change of direction.

**Log-space decodes.** The teacher draws almost every positive parameter
lognormally, so the median person is `PatientParams()` and the population
mean lies above it. A setpoint decoded as `center + scale·tanh` has
expectation equal to the center for a zero-mean code, and cannot be right
at the median and in the mean. Log-space decoding makes both true. Zero is
the median person; the embedding table's mean is pinned to zero.

**Cohort debias floor.** An unbiased estimate of a square can be negative
and a loss cannot, so the debias is clipped. At batch size 2 the clip
keeps about 0.64 of the shrinkage force at zero bias. The fix is more
members, then the quadrature of Phase 3. `--cohort-sample-patients` is 4
until then.

**Insulin sensitivity's share.** Si is 22.5% of between-person glucose
incremental AUC and 20.7% of insulin iAUC (N = 150). Gb loads +0.50 on the
same insulin-resistance latent that Si loads −0.70 on. It is the largest
single person-axis by a wide margin.

**High-Gb ketosis.** Proportional scaling of the gluconeogenic share with
Gb recreates the iter-93 failure (Gb 130 fasts to 97 mg/dL at 48 h). The
one-sided split is the form that held: the split is an identity at the
typical EGP, so the median patient does not move. The median patient's
liver-glycogen drift (−6 g/day) is a different defect, in
`glyc_syn_frac_L` / `LGly_b` / basal glycogenolytic flux.

**Blood pressure.** The teacher draws diastolic and pulse pressure, with
pulse pressure clipped to the span the student can represent (20–80 mmHg).
Independent lognormals for SBP and DBP made 12.6% of patients unfittable.
An invariant installed on one side of a distillation pair is installed on
both.

**Respiratory rate.** Both drives sit inside the relaxation as equilibrium
offsets; the activity term is a Hill of n = 2. The old form equilibrated
at RR₀ + 50·activity.

**Checkpoint compatibility.** A strict load of an artifact from a
different state vector or decode family raises. That is correct;
silently dropping keys serves a model whose setpoints were fitted in
another frame. `--init-from` (strict=False) seeds a retrain.

**Cortisol basal.** `Cort_b` is 12 µg/dL and the default patient's 24 h
mean is 8.95. Every cortisol gate runs at a standing offset until the
basal and the mean name the same number.

**Exponential stepping vs the teacher's Euler.** The teacher integrates
explicit Euler at dt = 1. A student that copies a teacher `k` and steps
ETD1 realizes `e^{−k}` where the teacher realizes `1 − k` (k_hr 0.741 vs
0.700). A module that adopts ETD1 lands with the teacher on the same step
or refits `k = −ln(1 − k_teacher)`. The integrator is in `pulse/model.py`;
nothing defines `decay_rates` yet.
