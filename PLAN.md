# Plan — one physiology, found by the literature

*Written 2026-11-11. Supersedes the 2026-10-17 plan. That plan's doctrine
stands and is restated in §1–§5. Most of its phases landed (§6 lists
them); what is true at this commit is `STATE.md`. The completeness map
remains `docs/modeling-state.md`. The authoring surfaces remain
`pulse/knowledge/AUTHORING.md`.*

*This plan changes the order of work, not the destination. Three weeks of
runs under the last plan showed where the limit now is. A run no longer
finishes: an evidence epoch takes three hours and iters 114 and 115 both
reached the 72 h timeout. The held-out set is nine points from two
curves, so it cannot tell a better checkpoint from a worse one. And a
calibrated person predicts glucose no better than the median person. The
physiology is not what is holding the model back. The loop, the board and
the size of the corpus are, so they come first.*

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

What it is for, in two sentences. It predicts the result of a physiology
study it has never read, with an interval. It predicts a person's next
response from a few days of their own data. The board of §4 has one line
for each, and the stages of §6 are ordered by what moves those two lines.

## 2. What is imposed, and what is learned

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
  with no restoring pull toward it. The value today is a starting point,
  not a target.

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
  write the finding that forbids the drift and let it hold the parameter
  by evidence. The identifiability report is the standing list of which
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
| X raises or lowers Y, with a magnitude band | `CouplingPrior` |
| In protocol P, marker M had statistic S ± σ over window W | `CohortStatisticSpec` |
| In protocol P, marker M followed this curve (mean ± SEM at timepoints) | `StatisticKind.CURVE` |
| Across people, marker or parameter M had spread S or correlation R | `StatisticKind.SPREAD` / `StatisticKind.CORRELATION` |
| Under an IV glucose, insulin, or hormone infusion (incl. a clamp) | an infusion arm, then any operator above |
| A direction, floor, ceiling, or timing on one trajectory | `PhysiologyRule` |
| A dose scales a peak, rank, or slope | `OperatorKind.DOSE_*` |
| For one subject, these sparse readings under this protocol | an individual-level item (`authority=real`) |
| Conservation, positivity, ordering, a fixed point | the coordinates or functional form, in `pulse/modules/` |

Every item names its paper, where in the paper the number is (table or
figure, and the reported text), what kind of uncertainty it carries (a
between-person SD with its n, a standard error, or a range), and the
population it was measured in. A number with no place in a paper is a
rule or a prior, not a cohort target with an invented sigma. Literature
and real measurements outrank the teacher on any marker they both cover.

**The slice is soft.** A sigma tighter than the study supports can wrongly
exclude the truth, so two items on the same protocol and marker whose
targets disagree beyond their sigmas are recorded as a conflict, not
averaged into a number neither paper reported. A conflict that is real —
the model cannot satisfy both — is the signal to change structure, not to
widen a sigma.

## 4. The scoreboard, not a gate

There is no binary gate. Every checkpoint is scored on a continuous
board, and a run that regresses on one line tells us something; it is not
rejected. Three lines:

- **Fit to held-out literature.** Sealing is by paper: one paper in five,
  drawn by a hash of the paper's name when it is filed, never chosen. The
  line is the median miss on the items of sealed papers, in units of each
  item's own sigma, with an interval from resampling papers, and beside
  it the same number for the untrained model at its published constants.
  Items held out before sealing was by paper are reported on their own
  line, as held-out items whose papers also train.

- **Individual prediction skill.** On people from public individual-level
  datasets who were never used to develop the calibration: calibrate on
  sparse early readings, predict later readings under a different
  protocol. Skill is reported against the population-median person and
  the person's own running mean.

- **How open the model still is.** The count of parameter directions the
  current evidence does not constrain, each flat direction named with the
  study that would close it.

Teacher-tape agreement is logged. It is not on the board.

## 5. The teacher is scaffolding

The teacher (`pulse/knowledge/full_body.py`) is a lossy copy of the
literature with no sigmas, no citations, and invented minute-by-minute
detail. It is a warm start in a reasonable direction and a generator of
protocols. It is on a path to removal. Each role it still plays needs a
literature replacement before it can go, and each is removed by ablation:
train with that signal at weight zero, and keep the removal if the board
holds.

| Teacher role | Literature replacement |
|---|---|
| Dense tape of glucose, HR, BP, temperature | `CURVE` items: digitized 24 h profiles, mean ± SEM at timepoints |
| Distilling internals (CRH, remote insulin, bile pools, fat mass) | latent states with priors on time constants; reported as unidentified if nothing pins them |
| Generating the population | `SPREAD` / `CORRELATION` items from published distributions |
| Ground truth for identifiability and recovery tests | synthetic people sampled from the trained student itself |
| Stable cold start | the literature's own constants as the init; kept until last |

## 6. The work, in order

Stages 1 and 2 start together and everything after leans on them. Stages
3 and 4 run side by side once they land. Each stage ends on something
measured.

### Next steps, as of 2026-11-11

Written while iter 116 runs. Measurements cited here are in `STATE.md`.
Rewrite this section when iter 116's board is read.

**1. Read iter 116** (`trainer-jt2n9`, finished 2026-11-12)

Read on 2026-11-12 from the run's own boards (`STATE.md`, The board): sealed
papers 1.21 against the untrained student's 1.19, so branch 2b applies. The
file papers it trained on fell to 0.62 from 0.85. Still to do from this
step: the `--baseline` scoring of the checkpoint itself, the open-direction
pass and the ruler.


- Score `training/jobs/iter116/model.pt` beside the untrained student
  (`python -m pulse.diagnostics scoreboard --no-identifiability --baseline`)
  and write the board and the run's row into `STATE.md`.
- Read the three boards in order (phase 1, phase 2, final). They separate
  what the teacher tape does to the sealed papers from what the literature
  does.
- The numbers to beat, median |z| on the same people. Sealed papers (117
  points, 8 papers): untrained 1.19, iter 113 (the warm start) 1.18, iter
  114 1.32. File papers that train (202 points, 18 papers): untrained 0.85,
  iter 113 1.42, iter 114 1.16.
- Run the two passes left out of the job on its checkpoint: the
  open-direction pass and the CGMacros ruler (`--benchmark-only`). For iter
  114 they took 3 h 11 min; at 4 vCPU that is about $1.20. Their memory at
  16 GiB is unmeasured, so watch the first run.

**Done on 2026-11-13: the two split runs of 2b.** Dropping the teacher
gave 1.09 on the sealed papers; a cold start gave 1.20. Neither moved the
insulin bias, which the literature itself causes (`STATE.md`, The board).
Next: no teacher with `--evidence-paper-weighting sqrt`, to test whether
the three papers that hold 42 of 46 trained insulin points set it.

**2a. If the sealed papers come in under the untrained student**

Training on papers read as files generalises, and the corpus is the lever
(stage 3).

- Widen the sealed set first. Eight papers give an interval of 1.15 to
  1.68; a difference of 0.1 cannot be judged on it. Read 20 to 30 more
  papers, each by one reader and checked by a second.
- Close the format gaps that keep printed numbers out, in the order
  `python -m pulse.admit --missing` ranks them: C-peptide as a marker (10
  papers), indices derived from a curve (13), a median with its
  interquartile range (5), insulin clearance (4), noradrenaline (4), values
  per lean mass or body surface (5), the condition of the day before (4),
  an insulin infusion rate as a finding (3).
- After each gap closes, re-read the papers that named it, as was done for
  the three single-arm statistics.

**2b. If they do not**

Fitting papers is moving the model away from unseen ones, and more papers
will not fix that.

- Start where iter 114 lost most: glucose (1.17 against 0.69 untrained on
  110 points), fatty acids (1.39 against 0.43), lactate (2.61 against
  0.67), and fasting GLP-1 (52 pmol/L against a measured 4).
- Split the question with runs of about $9 each: a cold start against the
  warm start from iter 113; the literature without the teacher tape; the
  registries alone against the registries plus files.
- Look for findings that pull against each other on shared parameters (the
  board's conflicts line, with the open-direction pass on), and for one
  paper's many correlated points outweighing another's one.

**3. The loop, whichever way step 2 goes** (stage 1 is not done)

The recipe takes 23 h and stage 1 asks for three. The evidence step is 54 s
of a 68 s step, and 24 s of that is the spin-up.

- Try `--compile-steps` with the batched step: first an equivalence test
  against the eager step, then a timed execution of 10 steps (about $0.30).
- Use the job's four cores: one worker per rollout group, as
  `docs/training-runs.md` sketches under "Across processes".
- Cut the spin-up: people of the same population, start hour and pre-fast
  already share one rollout within a step; measure whether a shorter fed
  day reaches the same start state.
- Raise `--evidence-batch-size`. Rows are nearly free up to about 75, and a
  step of 4 items sees 1.5% of the corpus.

**4. Defects that bias what a run learns** (`STATE.md`, Known wrong)

- The sigma of the 35 registry points with no `n_arm` depends on a training
  flag. State each point's n, or drop the point.
- 693 gut-kernel parameters receive no gradient, and the stomach's split
  correction never reaches absorption.
- The cortisol gain on glucose production is centred on the population's
  cortisol.
- `peak_value` reads a dense maximum where a paper's Cmax is the largest of
  its samples; give it `times`.
- The open-direction report still starts everyone from the median person's
  spin-up.
- Three tests fail.

**5. People** (stage 4)

- A calibrated person predicts CGMacros glucose no better than the median
  person (28.17 against 28.06 mg/dL). Rescore on iter 116 before changing
  anything.
- Calibration fits interstitial glucose while evaluation reads plasma.

**6. Keeping it honest**

- The test suite has no home: it does not fit the lab box (16 GiB on four
  workers) and there is no CI. Run it as a Cloud Build step or a small Cloud
  Run job per push; a GitHub workflow needs a credential with the
  `workflow` scope.
- The papers' full texts are not kept. Store the PubMed Central XML in the
  project's bucket, outside the repository, so `admit --paper` can be rerun.
- The `engine` service does not start.

**7. With soma** (low priority; the lab repository's `NOTES.md`)

- Soma has no board beside an untrained student on unseen papers.
- Soma's files lack `assumed`, `left_out` and the quote check.
- The first finding both can read: a human-islet incretin dose-response in
  picomolar.

### Where the 2026-10-17 phases stand

Code and tests that cite "PLAN Phase N" mean these.

| Phase | What it was | Now |
|---|---|---|
| 0 | Sealed set, identifiability report, an individual ruler | Landed; the sealed set is too small to judge by (stage 2) |
| 1 | One baseline run | Landed (iter 113) |
| 2 | Every measured number a learnable parameter | Half done: 95 scalar laws learn, about 100 numbers are still forced (stage 6) |
| 3 | The population is a distribution | The factor prior landed; one spread item, no correlation item (stage 4) |
| 4 | Curve, infusion and clamp operators, residuals, conflicts | Landed |
| 5 | The loop samples evidence and sums likelihoods | Landed (iter 114) |
| 6 | Remove the teacher | Stage 5 |
| 7 | Read at scale | Stage 3 |

### Stage 1 — a loop that finishes

A step rolls all of its evidence together. The people of a step are spun
up once, the arms that share a horizon share a rollout, and the
likelihoods are summed as before. A rollout costs about the same for one
person as for a hundred, so the step can see far more of the corpus for
less than four items cost before. The minute step itself sheds the work
nothing reads. The cores of the training machine are used. A run writes a
board at every phase boundary and stops itself before its deadline, so no
run ends without one.

Done when a full run of the current recipe takes under three hours on the
training job, and a step that scores every item in the corpus takes under
a minute.

### Stage 2 — a board that can judge

Sealing by paper, decided by the registry (§4). An interval on every
line. The untrained model beside every line. `STATE.md`, rewritten in the
commit that changes what it describes and checked against the code by a
test, takes over from the iteration handoffs as the record of what is
true.

Done when the sealed line holds at least 60 items from at least 20
sealed papers. It starts near empty, because every paper already in
training is exempt. Stage 3 is what fills it.

### Stage 3 — the corpus, at the scale of the claim

A paper is a file in `corpus/findings/`, the way soma keeps its corpus.
`python -m pulse.admit` is the gate. It checks that the protocol runs,
that a gradient reaches a parameter, that the number has a place in the
paper, that its uncertainty is of a stated kind, and that it does not
silently contradict an item already admitted. A paper the model cannot
express becomes a file in `corpus/refusals/` that names the missing
piece: a marker, an input, a population, a timescale. That pile is the
specification for the next change to the model.

Reading is done by agents, one paper each: read, write the finding or the
refusal, run the gate. A second agent checks every number against the
paper before the file is admitted. The existing registries move into
files the same way, and the targets that are textbook ranges rather than
reported numbers (about 45% of the cohort items today) are re-sourced or
become rules.

The order: the metabolic canon where the states already exist (OGTT,
IVGTT, clamp, mixed meal, fasting), then exercise, sleep and circulation;
after that, whatever the open-direction list and the refusal pile ask
for. Open-access papers first.

Done when the corpus holds 1,000 items traceable to a table or figure,
from at least 300 papers. Then 10,000.

### Stage 4 — real people

The population prior is fitted to published distributions: the joint
spread of fasting glucose, insulin, blood pressure, heart rate and
lipids by age, sex and BMI (NHANES), as spread and correlation items.
Prediabetes and type 2 diabetes become regions of the person space with
their own findings; today every population in the corpus is healthy,
while two thirds of the CGMacros people are not.

The individual ruler grows from one morning per person to every day of
CGMacros and at least two more public datasets, with the people split
once, by a hash of their id, into those used to develop calibration and
those used only to report it. What a record already says about a person
(age, sex, BMI) sets those coordinates directly. Calibration then becomes
a posterior over the person, not a point.

Done when a calibrated person beats the median person by 20% on next-meal
glucose on three datasets, on the people held back for reporting.

### Stage 5 — no teacher

The ablations of §5, in this order: the dense tape, the teacher
meal-response item, the sweeps, setpoint supervision, distillation. Each
is one run, which stage 1 makes cheap. Then delete `full_body.py`, the
distillation and sweep signals, and the teacher-side tests.

Done when the board holds without it.

### Stage 6 — wider, and longer

Longer first, because the product question is about weeks. Slow states
step at their own rate, so a twelve-week rollout costs about what a day
does now. Then the findings that need it: a training block, a
sleep-restriction week, weight loss.

Wider, one hub at a time, each landing with the papers that constrain it
and each prompted by the refusal pile: drugs as inputs with their own
kinetics (GLP-1 receptor agonists, metformin, insulin, β-blockers);
pressure and volume (cardiac output, the baroreflex, the kidney's handling
of salt and water, renin–angiotensin–aldosterone); lipids and
lipoproteins; body composition that follows the energy balance; thyroid;
sex hormones and the menstrual cycle.

The remaining forced numbers (§6 table, phase 2) become parameters as the
slices that identify them arrive, never before.

Done when a twelve-week training block and a GLP-1 agonist trial arm are
both scored on the board.

### Stage 7 — the cell inside the body

Soma is the same thesis at the grain of one cell, and its first cell is
the pancreatic β-cell: the cell behind pulse's insulin release, which
here is one learned gain and a few constants. The two models should agree
where they overlap, and a module at this grain can be a model at that
one.

First a probe: drive soma's cell with the plasma pulse predicts and see
where the two disagree. Then one finding format both corpora read, so a
perifusion study and an oral glucose test constrain the same cell. Then
soma's cell as the reference pulse's secretion law is fitted to, and, if
the cost allows, as the secretion module itself.

Done when one finding constrains both models.

### Stage 8 — in the open

The corpus published as a dataset others can cite and correct. The board
for every accepted checkpoint, public. The same sealed papers scored
against an established hand-built simulator. A preprint. The inference
service running again on a current checkpoint, with an explorer where
anyone can run a protocol.

## 7. Risks

- **Reading at scale invents numbers.** An agent that cannot open a paper
  will still produce a plausible target. The gate requires the reported
  text and its place in the paper, a second agent checks it, and a
  finding that cannot be checked is refused.
- **The sealed line stays thin.** One paper in five is slow to fill it.
  If stage 3 stalls, the board judges by the weaker held-out-items line,
  and says so.
- **Batching changes the optimisation.** A step that sees forty items
  takes a different path from one that sees four. The mechanism lands
  first, proven equivalent on the same items; the recipe that uses it is
  its own run with its own hypothesis.
- **Identifiability.** Many parameters will sit at their init. That is
  honest; the report names each and the study that would move it. The
  risk is a parameter the loss reaches for the wrong reason, caught by
  the report and fixed by a slice, never by a pin.
- **Population mismatch.** Averaging over the wrong people biases every
  cohort item. Each arm declares its population, and the admit gate
  refuses an arm that does not.
- **The model space may not contain the truth.** Then two well-measured
  findings cannot both be met. That conflict is the signal to change a
  functional form, which is why the correction terms of §2 exist and why
  conflicts and refusals are first-class.

## 8. How a step is judged

A structural claim (a fixed point, a conservation bound, a decode family,
an operator's gradient, an equivalence between two ways of computing the
same thing) is a test. It passes for every embedding and every protocol,
or it does not. It needs no checkpoint.

Anything that changes what training can learn needs a run, scored on the
board against the previous accepted checkpoint, with matching evidence
fingerprints. Accepted means: held-out literature fit does not fall
outside its interval, individual skill does not fall, the open-direction
count is reported, and no item already inside one sigma moves beyond two.
Teacher agreement is logged, not judged.

A run costs money. Its expected cost is stated before it is launched, and
it is launched from a commit.

## 9. Measurements that still bind

These were measured while writing this plan and its predecessors. A later
item that contradicts one of them has to remeasure it.

**Where the time went.** Iter 115: an evidence epoch of 40 steps took
3–3.7 h on one busy core of eight, 57 s per evidence item, and 63% of all
simulated minutes were the fed-day spin-up repeated for every arm of every
item. A 240-minute rollout with gradient costs about 1.2× as much at
batch 128 as at batch 1.

**The sealed nine.** The iter 114 board's held-out fit was a median miss
of 1.21σ on nine points: seven timepoints of one hand-chosen OGTT curve
and two of one VO₂ curve, against 0.09σ on 131 training items.

**Calibration buys nothing on glucose yet.** CGMacros, iter 114: 28.2
mg/dL calibrated against 28.1 for the median person over 44 people; of
the 26 whose prediction changed, 11 improved.

**Log-space decodes.** The teacher draws almost every positive parameter
lognormally, so the median person is `PatientParams()` and the population
mean lies above it. A setpoint decoded as `center + scale·tanh` has
expectation equal to the center for a zero-mean code, and cannot be right
at the median and in the mean. Log-space decoding makes both true. Zero is
the median person.

**Cohort debias floor.** An unbiased estimate of a square can be negative
and a loss cannot, so the debias is clipped. At batch size 2 the clip
keeps about 0.64 of the shrinkage force at zero bias. The fix is more
members. `--cohort-sample-patients` is 4.

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
