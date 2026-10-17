# Plan — a simulation of human physiology, trained by the literature

*Written 2026-10-09. Supersedes the 2026-10-04 plan ("the person model, the
frozen constants, and the missing hubs"). That plan's measurements still
bind; they are restated under the item that would otherwise repeat the
mistake. The completeness map remains `docs/modeling-state.md`. The
authoring surfaces remain `pulse/knowledge/AUTHORING.md`.*

## 1. Destination

A simulation of human physiology. What the scientific record has
established about the body enters as gradients, at the granularity the
evidence supports, through one schema. A later paper can move the model,
including a law that was copied from the teacher.

The simulation is done enough to extend when three things are true at
once:

- A published finding has a state and a pathway to act on.
- That finding is an evidence item — protocol, observation, sigma,
  citation, authority — and the training step gives it a gradient.
- Held-out findings and real measurements move the right way. Agreement
  with the teacher tape is not the score.

Personalization is a consequence. A body that cannot represent
between-person variation cannot absorb the literature that is about
that variation. It is not a separate product goal.

## 2. What a finding is

The native training example is an `EvidenceItem`
(`pulse/knowledge/evidence.py`): a protocol, an observation operator, a
reliability, and an authority. Literature and real measurements outrank
the teacher on any marker they both cover.

A finding enters at the smallest surface that captures it.

| The finding | Where it goes |
|---|---|
| X raises or lowers Y, with a magnitude band | `CouplingPrior` in `pulse/knowledge/coupling_priors/` |
| In protocol P, marker M had statistic S ± σ | `CohortStatisticSpec` in `pulse/knowledge/cohorts/`, registered in `cohort_statistics.py` |
| A direction, floor, ceiling, or timing on one trajectory | `PhysiologyRule` in `pulse/knowledge/physiology_rules.py` |
| A dose scales a peak, a rank, or a slope | `OperatorKind.DOSE_*`, scored by `RolloutEvidenceSignal` |
| A local input→output map the trajectory loss cannot reach | A sweep signal, after `signal-balance` shows the trajectory term dominating that module |
| Conservation, positivity, ordering, a fixed point | The coordinates or the functional form, in `pulse/modules/` |

The teacher tape (`TEACHER_TAPE_MARKERS`: glucose, hr, sbp, dbp, temp)
is wearable-frequency fill. Distillation
(`TEACHER_DISTILL_MARKERS`: crh, insulin_action, insulin_slow,
mitochondrial_capacity, fat_mass, gallbladder_bile, intestinal_bile)
covers internals no paper reports as a time series. A marker that gains
a cohort statistic leaves the distill list. `teacher_literature_collisions()`
stays empty.

Today the registries hold 40 cohort statistics, 61 physiology rules, and
37 coupling priors. That is the schema, exercised on one region of
physiology. It is not the literature.

## 3. How work is ordered

Three rules decide what is next.

**A finding needs a degree of freedom.** Encoding a result whose pathway
does not exist produces a zero gradient or a parameter that gets frozen.
Iters 98–109 did the second: learned laws were replaced by the teacher's
constants, and the parameter count fell from 68,293 to 37,343. Each
replacement was locally right. Together they made a copy, and a copy
cannot be overturned by the next paper.

**A new state lands with the evidence that identifies it.** Teacher
mechanism, student mechanism, and the cohort or rule that supervises it
arrive together, with the benchmark `initial_state` migration
(`scripts/migrate_benchmark_initial_state.py`). A state alone repeats
the inert-glycogen failure.

**The schema grows when a finding has no home, and it grows once.** A
paper does not get its own loss class. If the decision table above
cannot express the finding, the table gains one operator, and later
papers of that shape use it.

## 4. Where this stands (2026-10-09)

The simulator is ten modules and 39 states: metabolic, appetite,
stress, cardiovascular, thermoregulation, respiratory, gut,
hepatobiliary, gastric, circadian. Carbon closes. Glucose's basal is a fixed point.
A meal leaves the stomach at one calorie rate, and appearance is that
outflow times the bioavailable fraction. GIP and GLP-1 split the
incretin effect, and protein raises insulin while glucose stays put.
Intestinal bile sets the share of leaving fat that is absorbed, and the
hepatobiliary heads do not read the person. Glucose above 190 mg/dL
leaves in the urine. Empagliflozin 25 mg/day, when a protocol turns it
on, moves that threshold to 34 mg/dL and the clearance to 126 mL/min.
A fed day of this model then loses about 104 g, against the free-living
collection of 42 ± 5 g. The line is the clamp regression and is not
retuned to the collection.
The embedding is
32-dimensional; zero is the median person; the table mean is pinned
to zero. Twelve per-person setpoints are supervised. Muscle insulin
sensitivity is a different quantity from hepatic insulin sensitivity,
and on a pure sensitivity sweep the student realizes 1.77× between
deciles against the teacher's 1.81×.

Iter 111 is the cold run of that person model. The next hypothesis
records its gate as real-CGM glucose MAE 7.23 mg/dL, skill +0.10,
glucagon MAE 1.08, cortisol MAE 0.31. The insulin potentiator is in
the student, the same delayed law as the teacher (`k_pot` 0.025,
`w` 0.3), and `ogtt_75g_insulin_120min` scores the tail.

Iter 112 trained that tree. `sns_tone` and `epinephrine` are the
sympathoadrenal states (M2.1).

The four former pins are population parameters under a prior, with no
per-person head: obligatory uptake `k_ii`, the gluconeogenic fraction
`f_gng` (mode one half), FFA clearance `k_ffa` (mode 0.20), and glucagon
basal (mode 70 pg/mL). The teacher varies uptake, the fraction and
glucagon basal per patient; clearance is one value for every patient.
Second-phase insulin is the teacher's delayed potentiator in both models
(`ins_phase2_frac` 0.3, `k_pot` 0.025). Appetite reads the metabolic
module's basal insulin. The exponential
integrator exists and no module reports a decay.

## 5. Movement 1 — a body a gradient can move

This movement ends when a published number can change a person-axis or a
population law, the literature items already in the registry receive
real optimizer steps, and a flat direction is reported before anyone
freezes it.

M1.4 added `insulin_potentiator`. Nothing else in this movement adds a state.

### M1.1 Calibration window — in progress

**Question.** Can a glucose window tell fasting glucose (Gb) apart from
insulin sensitivity (Si)? Both move post-meal glucose. Gb also sets the
fasting level. Calibration freezes the weights and fits the embedding.

**What is already written.** `scripts/si_gb_calibration_window.py` opens
the two heads onto orthogonal embedding axes and compares a 180-minute
fast with a 75 g meal. `pulse/measurement.py` lags plasma glucose by
8 minutes (Rebrin & Steil; the usual CGM delay is 5–10). The calibration
changes keep the window's initial state on that window and spin it up
from the person's quasi-steady state.

**Acceptance.** A meal gradient carries an Si component a fast does not.
A glucose observation is interstitial, not plasma. A shifted start or an
unlogged meal moves the window state and does not become part of the
embedding that is kept.

**Branch.** If, on teacher patients, recovering Gb and Si from a window
is no better than predicting the population mean, stop adding person
heads. The next work stays inside this item: the observation model
(lag, noise, the disturbance of an unlogged meal). Further per-person axes wait until a window can see
the ones that exist.

### M1.2 Per-person sweep targets — iter 112, spec staged

**Change.** `GutDoseSweepSignal` and `InsulinSweepSignal` score each
sampled row against that row's teacher `PatientParams`. The zero
embedding stays the median person and is scored against
`PatientParams()`. A sampled row with no `patient_params` aborts.
`--gut-dose-sweep-sample-patients` and
`--insulin-sweep-sample-patients` go from 0 back to 4. Warm start is
`gs://grovina-pulse-data/training/jobs/iter111/model.pt`.

**Why it was removed.** Iter 111 set both flags to 0 so the two sweeps
(weights 0.10 and 0.30) would stop pulling sampled patients toward the
median. Restoring them without per-row targets would restore that pull.

**Acceptance.** The zero row's target is unchanged. A non-zero row's
target is its own teacher. On the gate, real-CGM glucose skill stays
positive against iter 111's +0.10, and the teacher-side gains hold
(glucagon MAE 1.08, cortisol MAE 0.31). Setpoint-supervision MAE and
the spread of the decoded setpoints are the shrinkage watch: the
centring term must leave per-patient structure in place.

### M1.3 One basal insulin, one body mass — landed

The appetite decode is gone. `person_quantities` reads basal insulin and
body mass from the metabolic projection of the person code, and appetite
binds that insulin before it steps. A test sets the metabolic head to a
teacher patient's Ib of 18 µU/mL; both the metabolic fixed point and the
value appetite reads are that patient's. Body mass had only the one
decode already. The sweeps still supervise it against the teacher's
`body_mass_kg`; they do not decode a second one.

### M1.4 Second-phase insulin — landed in the equations

The teacher's secretion was already
`gamma * ((1 − w) * (G − h)+ + w * Pot) * incretin`, with
`dPot = −k_pot * (Pot − (G − h)+)`, `w = 0.3`, `k_pot = 0.025`.
The student now has the same state (`insulin_potentiator`) and the
same constants. At a constant glucose the stimulus collapses to
`relu(G − Gb)`, so steady secretion and the fasting fixed point are
unchanged. While glucose falls, secretion is still the previous half
hour. `ogtt_75g_insulin_120min` is the tail: 34.6 ± 1.44 µU/mL
(PMID 3525269, the SEM, `n_arm` 1) over minutes 110–130 after the load.
Whether a trained checkpoint sits in that band is the next run's question.

### M1.5 Priors replace the four frozen constants — population parameters landed

**Constants.** `k_ii` (uptake that puts a typical person at
2 mg/kg/min), `f_gng` (one half; Landau's post-absorptive share),
`k_ffa` (0.20, Eaton's τ ≈ 5 min), glucagon basal (70 pg/mL). On the
iter-107 weights the first two had walked — uptake 0.0114 → 0.0083 per
minute, share 0.50 → 0.37 — and the walk is why they were pinned. A pin
deletes the teacher's per-patient spread and cannot be moved by a later
paper.

**Change.** Each becomes learnable. The prior is the distribution the
citation already implies: log-normal for the rate constants, Beta for
the fraction. The evidence that identifies them is the tracer literature
those citations are (Landau for the gluconeogenic share and the
prolonged-fast floor, Eaton for the FFA time constant, the glucagon
basal the marker's typical already states). `f_gng` is the fraction in
the one-sided split, not a rescaling of all EGP with Gb. That
rescaling was measured to be wrong: a Gb-130 patient then fasts to
97 mg/dL at 48 h, and a Gb-70 patient to 60.7, breaking
`glucose_fasting_floor`.

**The split, unchanged at the median.**
`Gng_b = f_gng · min(Hep_b, 2.0) + relu(Hep_b − 2.0)`, with
glycogenolysis absorbing the deficit and gluconeogenesis the excess,
and glycogenolysis kept at least 15% of EGP. At the typical EGP the
split is an identity, so the default patient does not move. There is no
per-person `f_gng` head until the Beta prior and the tracer evidence
exist; a head without them is the free-parameter failure this movement
exists to stop.

**Landed.** Each is one population parameter, initialised at the old pin,
with `PopulationPriorSignal` (default weight 1). `k_ii` and `k_ffa` and
glucagon basal are log-normal; `f_gng` is Beta, concentration matched to
the teacher's `vary(f_gng, 0.10)`. Eaton's clearance prior is
`k_ffa` in the queue. Landau's 14 h / 22 h / 42 h shares stay blocked:
hepatic output is the sum, and the prior only holds the post-absorptive
mode. There is still no per-person head. The spread acceptance below is
open until a head arrives with a target.

**Acceptance.** At the median person, basal glucose, the 48 h fast, and
the fed night match the pinned law within the prior width. Across
teacher patients the student's spread in the four quantities is the
teacher's spread, not a point. A direction the loss does not identify
stays at the prior. It does not walk, and it is not pinned. The mode
and the restoring gradient are tested. The spread is not.

### M1.6 Exponential stepping, one module at a time

The integrator is in `pulse/model.py`. A module that does not define
`decay_rates` is Euler, bit for bit. Nothing defines it yet.

**The trap.** The teacher integrates explicit Euler at dt = 1. A student
that copies a teacher `k` and then steps ETD1 realizes `e^{−k}` where
the teacher realizes `1 − k`. At `k_hr = 0.3` that is 0.741 against
0.700, a 6% per-step divergence introduced by the fix. The first module
to adopt either lands together with the teacher on the same step, or
uses `k = −ln(1 − k_teacher)`.

**Where it helps.** Measured on the teacher constants: hr 0.054 → 0.003,
cck 0.035 → 0.001, glp1 0.022 → 0.002, ffa 0.092 → 0.033. The slow
pulsatile states get worse (gallbladder, intestinal bile, lactate,
`insulin_slow`), because ETD1 is exact only for forcing that is constant
across the step. There is no `MassActionModule` default: handing
metabolic glucose a spurious decay would be a bug, not a cleanup.

**Acceptance, per module.** A `dt = 1/16` rollout of that module agrees
with the ETD1 step more closely than Euler does, and the marker's
teacher residual falls. A module whose residual rises does not adopt.
Adoption order follows that measurement, starting with hr, cck, glp1,
and ffa.

### M1.7 Identifiability, before the corpus grows

**Change.** A gate artifact: the eigen-spectrum of the Fisher
information of the total loss with respect to the population scalars and
the per-person heads.

**Acceptance.** Every flat direction is either covered by a prior plus
the evidence that identifies it, or listed. The list is what the next
iteration is allowed to freeze, and only after the prior exists. M1.5
is the first consumer of this artifact.

### M1.8 The literature items already written must receive steps — the log lands

Iter 97 measured the failure mode this item exists to close: in a
21-hour run the literature stack took 25 optimizer steps, while
pointwise teacher imitation took the rest. The current spec puts cohort
statistics at weight 0.15, physiology rules at 0.05, dose response at
0.40, cold distill at 0.30, with an auxiliary step every 12 trajectory
windows and cold distill on a slower cadence than the cohorts.

**Change.** The training log prints, per epoch, the optimizer-step count
and the gradient-norm share of three families: literature (cohorts,
rules, dose-response), teacher tape, and distillation. The cadence is
then set so the literature family is a real fraction of the step, on
the order of the other two, not a few dozen steps in a day.

**Landed.** Each epoch prints `[LITSHARE]`: contributions and the
gradient-norm share of literature (cohorts, rules, dose-response),
teacher tape, distillation, and everything else. The cadence is not
retuned here. That waits for a run, and a run whose literature count
is the iter-97 pattern is not a completed movement.

**Acceptance.** A run records those shares. A movement-1 retrain whose
literature step count is the iter-97 pattern is not a completed
movement, whatever the gate says.

### Movement 1 retrain

M1.1's branch is read before more person heads are added. M1.2 trains
on the iter 111 checkpoint as iter 112. M1.3, M1.4 and the population
half of M1.5 are in the tree. A warm start drops appetite's old basal
head and starts the four priors at their modes. The per-person spread
of those four is still open. They change what a warm start means, so
they ride the next run with the potentiator.

M1.6 lands inside whichever module it touches and does not force that run. M1.7 is an
artifact of the run, not a reason to delay it. M1.8 is a property of
the run.

The run is scored against iter 111 on a frozen ruler with a matching
`ruler_fingerprint`. The score is real-CGM skill, the cohort z-residuals,
and setpoint spread. Teacher-tape MAE is reported and is not the
acceptance criterion.

## 6. Movement 2 — physiology, in the order the literature unlocks

Each hub below is one landing: teacher mechanism in
`pulse/knowledge/full_body.py`, student mechanism in the module named,
the cohort statistics and rules that identify it, and the benchmark
initial-state migration. `STATE_DIM` grows, so every checkpoint from
before that landing is unloadable. That is the same rule as the person
model: a strict load fails loudly. `--init-from` (strict=False) is how
a retrain is seeded. No key-dropping shim. An old head fitted in a
different state vector is a wrong model that looks plausible.

A hub is accepted when its new evidence items move and the items already
in the registry still hold, on the frozen ruler.

### M2.1 Sympathoadrenal arm — states landed, Cryer still blocked

Lives in the stress module. The HPA axis and the sympathoadrenal system
are the two arms of one stress response.

**States.** `sns_tone` (neural, τ = 1.5 min, rest 0.2). `epinephrine`
(pg/mL, τ = 2 min, rest 30). Appended at the tail of `MARKERS`.

**Connections.** In: activity, glucose below 69 mg/dL, sleep. The
reported-stress input stays with M2.8. Out, in the teacher: heart rate,
blood pressure and the exercise temperature follow settled tone, with
the same gain they had on activity. Out, in both models: epinephrine
raises glycogenolysis and lipolysis, lifts glucagon's target, and
inhibits glucose-stimulated insulin. Exercise blood-glucose uptake is
contraction-mediated, proportional to arterial glucose and to work, and
zero at rest. The student's cardiovascular module still reads activity;
the tape carries the vital. Cortisol's hypoglycaemic drive starts at
58 mg/dL.

The hepatic gain, the lipolysis gain and the uptake coefficient are
student parameters, initialised at the teacher's 5, 0.5 and 0.015.
Resting epinephrine is an excess of zero, and resting uptake is zero,
so fed fixed points do not move.

**Evidence.** `wahren_800_splanchnic_glucose` is the 800 kg-m/min
splanchnic rise, 3.63 ± 1.29 mg/kg/min, on activity 0.7.
`wahren_800_arterial_glucose` is the same bout's arterial rise,
8.28 ± 6.32 mg/dL. The teacher sits on both: splanchnic output about
3.5, glucose about 9. `ffa_rises_with_exercise` moves the lipolysis gain.

Cryer et al. 1987, *J Clin Invest* 112884, is not yet a training item.
The thresholds are epinephrine 69 ± 2, glucagon 68 ± 2, growth hormone
66 ± 2, cortisol 58 ± 3, symptoms 53 ± 2. Growth hormone and symptoms
have no state. Glucagon's response below a person's own basal is a
separate law. The order cannot be scored: an arm has no insulin clamp,
and a day without food stays near 83 mg/dL.

**Baroreflex stays out.** It was measured and rejected
(`full_body.py` around the activity–HR block): at minute resolution the
reflex equilibrates inside one step; steady-period SBP has sd 0.40 mmHg;
the residual that does exist is a first-order lag at exercise
transitions (peak ±17.7 mmHg), and feeding it back would add ~18 bpm/min
to the rise and subtract it from recovery, corrupting an HRR1 that
matches Cole 1999; HR and SBP co-rise under central command
(correlation +0.964). A baroreflex returns with orthostasis, a
vasoactive drug, or a change in blood volume — M2.10 — not with this hub.

### M2.2 The stomach

**States.** `gastric_carb`, `gastric_protein`, `gastric_fat`, in grams.
Landed. The teacher, the student's appearance, duodenal delivery and the
ODE pool are the same update.

**Mechanism.** One compartment. Brener, Hendrix and McHugh,
Gastroenterology 1983: glucose solutions of 0.05, 0.125 and 0.25 g/ml
all left at 2.13 kcal/min once the loop was closed. Carbohydrate,
protein and fat share that rate in proportion to the calories still in
the stomach, so fat lowers the carbohydrate gram-rate and lengthens the
meal. The integral is the grams swallowed. The student fraction stays
0.8 at init; the teacher's integral is the whole gram. The rate is a
parameter on the student (`log_empty_kcal`, init 2.13) and the constant
on the teacher.

**Measured, noise-free, meal at minute 60.** A 75 g OGTT peaks at 168
mg/dL in the 60–180 window (target 150 ± 25) and the 90–150 min-post
mean is 116 (ceiling 140). Insulin peaks at 79 (target 60 ± 25). Insulin
at 120 min is 40 against 34.6 ± 1.44. Fifty grams of carbohydrate alone
appears at 4.11 mg/dL/min and peaks glucose at 168; the same
carbohydrate with 30 g of fat appears at 1.75 and peaks at 127.

**Not this increment.** GLP-1 and hyperglycaemia cannot slow a rate that
is computed from the meal before the step. The CCK cohort (peak 6.8 ±
0.4 within 10 min of a liquid mixed meal) is the open-loop spike this
cap does not make: on the mixed-meal arm CCK rises to about 1.8 and
holds for the emptying. Cunningham and Read 1989 is the glucose curve
for fat slowing emptying; the table is not in hand, so that cohort stays
queued. Do not put the two kernels back to recover the CCK peak.

### M2.3 Clock and sleep debt

Depends on M1's split between muscle and hepatic insulin sensitivity,
which has landed.

**States.** `circ_phase_offset`, in hours, bounded to ±6, so there is
no angle to wrap. `sleep_debt`, in hours. During the biological night
the debt state counts hours slept (a negative number). At 09:00 that
count becomes the shortfall against an 8 h night, and it holds until
the next night opens at 21:00. A morning with no bedtime behind it is
not charged.

**Mechanism.** The modules that had an imposed clock — the HPA drive, ghrelin's
anticipation, leptin's cosine, and the teacher's heart-rate and
temperature cosines — read `hour − offset`. The student's heart and
heat modules still see the wall clock, because their daily wave is in
the learned heads. The offset is pulled
by sleep toward a night centered at 03:00. One night centered 4 h late
moves it by 0.71 h; the Donga nights, which share that center, move it
by less than a tenth of an hour. Debt scales three insulin actions,
each of them 1 at no debt and at basal insulin: peripheral disposal,
the hepatic insulin gates, and the insulin suppression of lipolysis.
Cortisol, glucagon, ghrelin, leptin, heart rate and blood pressure have
no debt gain. The table measured cortisol and glucagon and they did
not change.

**Evidence.** Donga et al., J Clin Endocrinol Metab 2010;95:2963-2968,
Table 1, n = 9. Sleep allowed 23:00–07:30 or 01:00–05:00, both centered
at about 03:00. Basal glucose, insulin, NEFA, endogenous production,
glucagon and cortisol the next morning were unchanged. At the clamp
(insulin 22 mU/L against a basal of 3.1): disposal 32.5 ± 3.6 versus
40.7 ± 5.1 µmol/kg LBM/min, endogenous production 4.4 ± 0.3 versus
3.6 ± 0.2, NEFA 68 ± 5 versus 57 ± 4 µmol/L, glucose infusion
27.8 ± 3.7 versus 36.9 ± 5.1. The rollout has no clamp, so the cohort
is the basal glucose row (`donga_2010_basal_glucose`, target 0, sigma
the unpaired SE of the two ±0.18 mmol/L arms). The three clamp ratios
are the debt factors, checked at 22/3.1 times basal insulin.

**Measured, noise-free, the table's protocol.** The 4 h night commits
4.0 h of debt in the 11:00–12:00 window and the 8.5 h night commits 0.
Basal glucose is 84.6 versus 84.8 mg/dL. Basal hepatic output is 1.666
versus 1.668 mg/kg/min. Phase differs by 0.08 h.

**Still open.** The anticipation hours are still 09:00, 13:00 and 20:00
on the biological clock. They sit after the generated meals, so on the
reference schedule most of that ghrelin still arrives after the meal.
A shifted sleeper shifts the three hours together; the schedule itself
is not derived from the person's meals.

### M2.4 Energy expenditure

No new stored species. VO₂, VCO₂ and the respiratory quotient are a flux,
computed from the same function in the teacher and the student.

**The table.** Wahren et al., J Clin Invest 1971;50:2715, Table I.
Pulmonary oxygen uptake 256 ± 19 ml/min at rest and 1811 ± 69 ml/min
after 40 min at 800 kg-m/min. Respiratory quotient 0.77 ± 0.01 and
0.82 ± 0.01. Activity 0.7 is that load, the map the splanchnic cohort
already uses. Weir's protein-free equation turns the pair into kcal/min.
At the reference mass the resting flux is 1.227 kcal/min and the load is
8.780 kcal/min, VO₂ 1811 ml/min, RQ 0.820.

**What the flux drives.** Diet-induced thermogenesis is the existing
thermic fractions (protein 25%, carbohydrate 8%, fat 3%) added to that
expenditure, and it is the heat term temperature already had. Fat mass
is the energy that remains after that expenditure and after the glycogen
step: a gram that entered liver or muscle glycogen is not also adipose,
and glycogen breakdown covers part of the expenditure. Respiratory
frequency keeps the saturated Hill. Its input is CO₂ production, scaled
so the 800 kg-m/min point sits where activity 0.7 sat. Lactate and
temperature do not move it. At rest, with no meal, the offset is zero
and the rate equilibrates at RR₀.

**Measured, noise-free, the default patient, eight hours.** RR at
activity 0 / 0.3 / 0.5 / 0.7 / 0.9 / 1.0 is 15.00 / 20.03 / 24.83 /
28.24 / 30.41 / 31.16 per minute. The 0.7 point is RR₀ plus the old
Hill at 0.7. The earlier 35 /min at activity 0.9 included the lactate
term this hub removed.

**Still open.** The student's heart and heat modules do not read the
flux. Temperature's exercise term is still the sympathetic drive, not
the kcal. A heavier student's breathing uses the reference mass; the
teacher uses the person's mass, and with no meal the two agree because
the ratio cancels.

### M2.5 Incretin and the islet

**State.** `gip`, appended. Thirty-nine states.

**The split.** Vilsbøll, Krarup, Madsbad and Holst, Diabetologia
2003;46:183: at ordinary glucose the two incretins contribute about
equally, and GLP-1, not GIP, suppresses glucagon. The factor that used
to sit entirely on GLP-1 is now half on each. When the two excesses are
equal it is the old factor, so the incretin effect is not copied. On the
standard meal the share of glucose-stimulated secretion carried by the
factor is 0.62, still inside Nauck's 0.5–0.7. Glucose rise 60 mg/dL,
insulin peak 55 µU/mL.

**GIP.** Vollmer et al., Diabetes 2008;57:678. Integrated GIP over 240 min
was 186 ± 17% higher after an 820 kcal breakfast (360 kcal carbohydrate,
353 fat, 107 protein) than after 75 g of glucose. One compartment is
still emptying the breakfast at 240 min, so the fat weight is solved on
what has left by then. The simulated ratio is 2.86. While that breakfast
is emptying, GIP's excess equals GLP-1's (both 6.2 pmol/L above 10). K
cells read duodenal carbohydrate and fat. The paper's concentrations are
in the figures, so there is no pmol/L cohort; the ratio is the test.

**Amino acids.** Westphal, Gannon and Nuttall, Am J Clin Nutr
1990;52:267, n = 7. Fifty grams of protein alone did not change serum
glucose. Fasting glucose was 4.8 ± 0.1 mmol/L, and the difference was
not published, so the cohort sigma is that SEM as an unpaired SE
(2.55 mg/dL). Secretion from protein is a term of its own, zero when no
protein is leaving, so it fires while glucose is at basal. Glucagon
from the same grams is what keeps the glucose from falling, and it is
what puts the mixed meal's glucagon suppression back on its published
−10 pg/mL once GLP-1 is holding glucagon down. Measured, noise-free:
glucose delta −1.9 mg/dL over the 4 h, insulin +1.3 µU/mL while the
protein is emptying, mixed-meal glucagon delta −9.3.

**GLP-1 on glucagon.** Salehi, Vahl and D'Alessio, J Clin Endocrinol
Metab 2008;93:4909, Table. Glucose clamped near 8.9 mmol/L, 75 g oral
glucose, n = 10. Incremental glucagon area 95–270 min: saline 221 ± 353,
exendin-(9-39) 1657 ± 699 pg/mL·min. The difference over those 175 min
is 8.2 pg/mL. The term is zero at basal GLP-1, and at the excess a 75 g
load produces while it is emptying the glucagon equilibrium is 8.2 pg/mL
lower. There is no clamp in a rollout. The free OGTT's glucagon delta
is −31 pg/mL, inside −25 ± 12.

**Still open.** A GIP peak in pmol/L, once a table is in hand. The
exendin arm itself. GLP-1 still does not brake the stomach inside the
minute.

### M2.6 Hepatobiliary links out

**No new state.** Borgström, Barrowman and Lindström, New Comprehensive
Biochemistry 1985, chapter 14: in the absence of bile, 60–70% of a
normal fat load is still absorbed. The floor is the midpoint, 0.65. The
resting intestinal content (1 mmol) is bile present, and at that content
and above the fat that leaves the stomach is absorbed in full. Below it
the fraction falls in a straight line to the floor. Carbohydrate is not
touched. The same fraction multiplies the teacher's fat appearance, the
student's free-fatty-acid source and calorie ledger, the heat from the
meal, and the CO₂ that sets respiratory rate.

On a 40 g fat meal the intestine stays at or above its resting content,
so the absorbed fat is the swallowed fat. The floor is the empty
intestine, which a rollout does not reach: canalicular secretion is not
an arm input.

**The open A5 item closes here.** `randomize_params` draws no
hepatobiliary parameter. The cck and gallbladder heads were the free
person, and they are gone. Resting CCK is the typical for every person.
Gallbladder ejection is the structural 0.030 per minute, the teacher's
rate. Synthesis is the population constant.

**Still open.** Wu et al., JCEM 2013;98:E718: jejunal taurocholate does
not move fasting GLP-1, and with glucose it does. The abstract gives no
pmol/L, so the term is not sized. Serum bile acids still have no reader
outside the loop that fills the intestine.

### M2.7 Renal glucose and the Cori cycle

**No new state.** Diabetes 2017;66:1999–2006, people without diabetes,
n = 15. Glucose appears in the urine at 190 ± 2 mg/dL. Creatinine
clearance is 139 ± 7 mL/min, and above the threshold the excretion is
that clearance times the concentration above 190. At 200 mg/dL that is
13.9 mg/min. Below the threshold the term is zero, so fasting glucose
and every meal this model actually reaches are unchanged. The lost
grams leave the carbon ledger.

**Still open.** Lactate returning as glucose. Basal gluconeogenesis
already contains the resting Cori flux, and the extra share from a
raised lactate has no published fraction in hand.

### M2.8 Inputs the schema already promises

**No new state.** Empagliflozin 25 mg/day is an input on the metabolic
module. Zero is the untreated kidney. One is the day-13 kidney in
Diabetes 2017;66:1999–2006, people without diabetes, n = 15: the
regression of urinary glucose on plasma glucose crosses zero at 34
mg/dL, and creatinine clearance is 126 mL/min. The same line as M2.7,
with those two numbers. A value between them mixes the two measured
kidneys. It is not a dose titration, and the paper has no intermediate
dose. A missing log is no drug.

**The collection.** That paper's day-13 urine is 42 ± 5 g/day, from
people eating their own food. A fed day here (50, 65 and 75 g of
carbohydrate, asleep from 23:00) loses about 104 g. Glucose stays near
90 mg/dL, and the clamp line is steep there. The threshold and the
clearance stay at the measured numbers.

**Still open.** A reported stress, a GLP-1 receptor agonist, and
metformin. Each one waits for a protocol with a sigma. A drug that
accumulates is a state, and that state is M2.10.

### M2.9 Slow adaptation

**States.** `fitness`. `liver_fat`.

**Mechanism.** Fitness raises insulin sensitivity, lowers HR₀, raises
HRV₀, and moves the lactate threshold. Liver fat lowers hepatic insulin
sensitivity and raises FFA basal.

**Evidence.** A training-block cohort: weeks of exercise, with the
published change in insulin sensitivity and resting heart rate. A
hepatic-fat cohort: liver fat against hepatic insulin sensitivity.
Measured on the current teacher, 8 weeks of training moves mitochondrial
capacity +0.8% and resting heart rate not at all. Those are the
residuals this hub has to beat, not the targets.

**Depends on.** M1.8 and M2.4. A 16-week protocol at one-minute steps is
about 160,000 steps per patient. This hub is the one that forces
multi-rate integration: vitals at 1 minute, metabolic state at 1–5
minutes, slow pools hourly. The multi-rate step lands with this hub,
not as a separate refactor. Shorter hubs stay on the one-minute step.

### M2.10 Systems the nine hubs still do not cover

Same landing contract. Each waits until the findings that need it are
the ones being encoded, and until the hub it depends on exists. They
are named so the destination stays a body, not a better glucose model.

| System | What it has to represent | Waits on |
|---|---|---|
| Blood volume and vascular tone | Orthostasis, haemorrhage, a vasoactive drug. Pressure as a consequence of volume and tone, not four independent vitals | M2.1. The baroreflex returns here |
| Kidney beyond glucose | Water, sodium, potassium, acid–base. Thirst and urine as observations | M2.7 is the glucose term only |
| Inflammation | A slow cytokine state, so illness and recovery are a protocol | M2.9's slow timescale |
| Sex steroids | A cycle, because a large part of the endocrine literature is stratified by it and the model is one body | The HPA and gonadotropin axes sharing CRH's pattern: a driver, a pituitary state, a gland |
| Drug kinetics | A compartment with a half-life. M2.8 is an input; a drug that accumulates is a state | The target pathway for that drug |

## 7. Movement 3 — the intake scales

Movement 2 adds places a finding can act. This movement makes adding the
finding the same operation every time, and makes the corpus the thing
the optimizer integrates.

### M3.1 One door

A published finding enters through the table in §2. `RolloutEvidenceSignal`
is already the scorer for cohorts, rules, and dose operators. A new
paper appends a spec to a registry. It does not add a module under
`pulse/training/`.

A sweep signal remains the exception it already is: a local map the
trajectory loss cannot reach, justified by `signal-balance`, removed
when an evidence item covers that marker. The gut and insulin sweeps
are teacher-rate supervision of maps the cohorts do not pin pointwise.
They stay until M1.2's per-person targets and the cohort residuals say
the maps are held by evidence. They are not a pattern to copy per organ.

### M3.2 Residuals and conflicts are visible

Every evidence item already has a name, a source string, and a sigma.
The training artifact gains a per-item z-residual, written next to the
checkpoint the way the benchmark report is.

Two items on the same protocol and marker whose targets disagree beyond
their sigmas are recorded as a conflict. They are not averaged into a
target neither paper reported. Resolving a conflict means narrowing the
protocol until the papers describe different conditions, or widening a
sigma that was tighter than its source.

### M3.3 A held-out set

A sealed list of evidence items, absent from the training registries,
scored at the gate the way `cgm_real` is scored today. The first sealed
set is small and quantitative: OGTT and clamp results that are not
already the anchors in `cohorts/glucose_handling.py`, a sleep-restriction
clamp that is not Donga once Donga is in the training set, and an
exercise-plus-meal study. The sealed list lives with the benchmark
inputs and is cited. Adding an item to training means it leaves the
sealed list.

### M3.4 What gets read, and in what order

The corpus fills by reading, into the existing schema, in an order that
follows Movement 2. A hub's evidence list in §6 is the reading list for
that landing. Beyond those lists, the order is:

1. The textbooks the project already treats as the source of a sign or a
   time constant (Guyton, Boron & Boulpaep, Williams), encoded as
   coupling priors and hinges. One line each. These are the cheap
   breadth.
2. The quantitative papers those textbooks cite for a number, encoded as
   cohort statistics with the paper's own sigma. These are the amplitude.
3. Primary papers that contradict an item already in the corpus.
   Contradiction is a conflict record (M3.2), and it is how a copied
   teacher law gets overturned.

A finding with no sigma is a hinge or a prior, not a cohort target with
an invented one. The authoring note stands: sigma is the
literature-reported SD when the paper gives it; an estimate is marked as
an estimate in the spec, and `cohort-ablation` has to show the spec a
non-zero gradient before it is trusted.

Real datasets are rulers, not a substitute for this reading.
`cgm_real` is one subject's fourteen nights. CGMacros (45 subjects,
healthy / prediabetes / T2D, two CGMs, Fitbit, meal macros) and the
BIG IDEAs Lab set (Dexcom G6, Empatica E4, standardized breakfasts) are
the next rulers. They score a checkpoint. They do not replace a cohort
statistic that has a citation.

### M3.5 The person as an inference problem

This waits until M1.1 says a window can identify the axes that exist,
and until M1.7 says which of those axes carry information.

The embedding table is 40 memorized codes. The teacher draws each person
from latent axes plus per-parameter noise. The replacement is a named
latent of a handful of factors — insulin resistance, β-cell function,
fitness, adiposity, chronotype, HPA reactivity, gastric emptying — with
a fresh teacher population every epoch and an encoder from observations
to that latent. Population statistics are then a deterministic
quadrature over that latent, which removes the sampling variance that
M1's cohort debias can only floor at a batch of four. Calibration
estimates the latent, the initial state, and a disturbance, and keeps
only the latent.

Until M1.1's branch is resolved, building the encoder would amortize a
recovery that may not exist.

## 8. How a step is judged

A structural claim (a fixed point, a conservation bound, a decode family)
is a test. It passes for every embedding and every protocol, or it does
not. It does not need a checkpoint.

Anything that changes what training can learn needs a retrain. The
comparison is against the previous accepted checkpoint, on a frozen ruler
with a matching `ruler_fingerprint`.

Accepted means:

- The new evidence items' z-residuals fall.
- The evidence items already in the registry do not break. A break is a
  residual that moves from inside its sigma to well outside it.
- Real-CGM glucose skill stays at or above the last accepted run. Iter
  111's +0.10 is the current floor.
- The literature-family step share (M1.8) is reported.
- Flat directions (M1.7) are listed, not silently pinned.

Teacher-tape agreement is logged. It is not acceptance.

## 9. Measurements that still bind

These were measured while writing the 2026-10-04 plan or the September
review. A later item that contradicts one of them has to remeasure it.

**Log-space decodes.** The teacher draws almost every positive parameter
lognormally, so the median person is `PatientParams()` and the population
mean lies above it. A setpoint decoded as `center + scale·tanh` has
expectation equal to the center for a zero-mean code, and cannot be right
at the median and in the mean. Log-space decoding makes both true. Zero
is the median person. The embedding table's mean is pinned to zero. The
centring weight in the spec is 10; weight 1 does not pin at the spec's
cadence, because the penalty lands only on auxiliary steps and Adam
scales the step by the RMS of every gradient that coordinate sees.

**Cohort debias floor.** An unbiased estimate of a square can be
negative, and a loss cannot, so the debias is clipped. At batch size 2
the clip keeps about 0.64 of the shrinkage force at zero bias. The
straight-through gradient is written and tested and not adopted: for the
specs where the bias is largest, the variance term has one degree of
freedom. The fix is more members, and then the quadrature in M3.5.
`--cohort-sample-patients` is 4 until then.

**Insulin sensitivity's share.** Si is 22.5% of between-person glucose
incremental AUC and 20.7% of insulin iAUC (N = 150). The earlier 34% /
31% was total post-meal AUC, which includes the fasting level. Gb loads
+0.50 on the same insulin-resistance latent that Si loads −0.70 on.

**High-Gb ketosis.** Proportional scaling of the gluconeogenic share
with Gb recreates the iter-93 failure (Gb 130 fasts to 97 mg/dL at 48 h).
The one-sided split in M1.5 is the form that held: day-4 BHB above 1 mM
fell from 26.3% to 2.5% of 80 patients, and corr(Gb, BHB) from +0.75 to
+0.12. The median patient's liver-glycogen drift (−6 g/day toward about
82 g) is a different defect, in `glyc_syn_frac_L` / `LGly_b` / basal
glycogenolytic flux. The split cannot fix it, because the split is an
identity at the typical EGP. `extended_fast_liver_glycogen_level` is a
healthy-cohort anchor (60 ± 20 g) applied to a population whose
high-Gb patients correctly hold glycogen. Fixing the drift moves the
24 h-fast reference numbers; the anchor needs a cohort restriction or
the pool needs re-sizing, and the two errors have been cancelling.

**Blood pressure.** The teacher now draws diastolic pressure and pulse
pressure, with pulse pressure clipped to the span the student can
represent (20–80 mmHg). Independent lognormals for SBP and DBP made
12.6% of patients unfittable. An invariant installed on one side of a
distillation pair is installed on both.

**Respiratory rate.** Both drives sit inside the relaxation as
equilibrium offsets, and the activity term is a Hill function of n = 2.
The old form equilibrated at RR₀ + 50·activity.

**Checkpoint compatibility.** The person-model landing removed heads and
changed decode families. A strict load of an earlier artifact raises.
That is correct. Silently dropping keys would serve a model whose
setpoints were fitted in another frame.

**Coupling-prior hinge on a gated edge.** Above the patient's glucose
setpoint the glucagon response to glucose is exactly zero, and a hinge
with a positive lower band penalizes that. Measured, the penalty is
0.0014 against a band width of 0.019. Real in form, negligible in size.
Setting the lower band to zero for gated edges can ride along with the
M2.1 reroute. It is not its own item.

**Cortisol basal.** `Cort_b` is 12 µg/dL, and the default patient's 24 h
mean is 8.95. Every cortisol gate runs at a standing offset until the
basal and the mean name the same number. M2.3 is the natural place,
because the waveform is what produces the mean.

## 10. The queue

The operational form of Movement 3 is `pulse/knowledge/queue.py`. It is
the list of findings not already in the cohort, rule, and prior
registries.

An item is **ready** when its marker exists, a gradient can reach the
pathway, and the arm we can run is the paper's protocol. An agent lands
a ready item on the surface the item names, then sets its status to
integrated and its `spec_name` to the registry entry.

An item is **blocked** on a hub from this plan (`m1.4`, `m1.5`,
`m2.1`–`m2.3`, `m2.5`–`m2.9`). An item is **backlog** when the number is
real but the protocol or the population is not the one we simulate.
Cole 1999 is the example: median heart-rate recovery of 17 bpm
(IQR 12–23) in referred adults during a cool-down, which is not the
moderate-bout arm and not the median person.

When an agent finishes an item, it reads another paper and files it.
A blocked paper stays blocked. The ready list is allowed to be empty.
As of the first filing it is empty: the one ready finding, the diastolic
ceiling during aerobic exercise, is landed as
`dbp_does_not_rise_with_exercise`. The two-hour OGTT insulin from
PMID 3525269 is `ogtt_75g_insulin_120min`, scored against the potentiator.
