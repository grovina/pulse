# State

Snapshot of **this commit**. Rewrite it in the same commit as the change it
describes; `git show <rev>:STATE.md` is pulse at that point. The count table
is checked against the code by `tests/test_state.py`, so it cannot go stale
quietly. Everything below the table is a measurement with its date, and has
to be re-read by hand. Where pulse is going is [`PLAN.md`](PLAN.md).

## Counts

| Metric | Now |
|---|---|
| Modules | 9 (metabolic 13 states, appetite 5, stress 5, cardiovascular 4, hepatobiliary 4, gastric 3, respiratory 2, circadian 2, thermoreg 1) |
| States | 39 |
| Observables | 10 |
| Parameters | 37,290 at the default widths |
| Published numbers | 46 in 10 tables, each learned from its published value |
| Corrections | 3 |
| Cohort specs | 1141 authored, 140 of them curves; 1215 training points and 332 held-out points |
| Rules | 62 training, 0 held out |
| Coupling priors | 32 |
| Papers | 192, of which 31 are sealed |
| Held-out items | 5 |
| Corpus files | 139 papers as findings, 57 as refusals, 0 queued to be read |
| Queue | 23 (15 integrated, 7 blocked, 1 backlog, 0 ready) in the older Python queue |

## The model

One network of nine modules over 39 states, stepped by explicit Euler at one
minute for every state. A person is a 32-number embedding; the median person
is the zero embedding. The checkpoints of iters 113–115 have 35,551
parameters (`--hidden-dim=48`).

`--meal-details` (off in the recipe; `pulse/modules/food.py`) adds a meal
reader of 262 parameters. It reads a meal's fibre, sugars, fat by type,
protein source and form (25 features: shares of the macronutrient, one-hot
source and form, and a stated-or-not mask for each field) and gives the meal
two modifiers per macronutrient. Speed, between 1/3 and 3, scales the meal's
share of the stomach's 2.13 kcal/min, so it changes when the meal reaches
the duodenum and the blood. Share is a shift of the logit of the person's
absorbed fraction, at most ln 3 either way, so it changes how much reaches
the blood. Its last layer starts at zero. Without the flag there is no
reader and the model is the one before it, bit for bit, and so is a meal
that states none of these details. `fat_source` and `food` are free text
and are not read, so `wong-2025`'s butter, coconut-oil and vegetable-oil
days, which name only the fat, are one meal to the reader. No run has
trained with it.
An evidence step of items with meal details costs the same with it as
without (24.5 s against 26.5 s batched, mean of three steps on the lab box,
2026-11-15, `python -m pulse.diagnostics step-time --only-meal-details`).

Insulin secretion in the islet (`pulse/incretin.py`, `metabolic.state_fluxes`)
is the glucose-driven term (`gamma`, on glucose above basal and the
potentiator, times the incretin factor) plus, since 2026-11-19, a
nutrient-driven one: `ins_nut_gain` x incretin factor x (`ins_nut_w_p` x
protein + `ins_nut_w_f` x fat) leaving the stomach, starting at 2, 2 and 0.5
(µU/mL per minute per gram). It replaces the protein-only `amino_ins_gain`
(1 per gram, dropped; the teacher keeps its own). Glucagon gains
`gn_nut_gain` x protein, starting at 3 x 2 x `protein_gn_gain` = 21 pg/mL per
minute per gram. The basal insulin setpoint is 0.8 of what it was
(`INSULIN_SETPOINT_SCALE`, applied in `insulin_setpoint_raw`; the
normative centre is unchanged). Carbohydrate is not in the nutrient drive.
Not yet trained: the four numbers start at these values and no run has moved
them. Iter 118's checkpoint loads (`load_model_state` starts the new keys at their
defaults and ignores `log_amino_ins_gain`). Free tests of this form on it
(2026-11-19, variant `n2_m3`) are the case for it; the checkpoint has not
been rescored on the full board.

## Training

The recipe is `train/spec.json` (iter 116, the flags of iter 115: 58 epochs,
30 on trajectories alone, then 28 with the literature). The job is 4 vCPU, 16 GiB, 72 h, no
retries (8 vCPU and 32 GiB until 2026-11-11).

A step now spins up all of its people in one rollout and rolls its items in
groups of similar horizon (`pulse/training/evidence_step.py`). A run writes a
board at the end of each phase, and `--deadline-hours` stops it cleanly
before the job's timeout. Iter 116 is the first run on this loop; it
finished in 23 h (Runs, below). It was launched without the open-direction pass and
without the CGMacros ruler, whose memory on the smaller job has not been
measured; both are owed on its checkpoint.

A step draws its items uniformly, so a paper weighs as many items as it
printed: three papers are 36% of the draws, and `atkinson-2020` alone 13%.
`--evidence-paper-weighting sqrt` weighs a paper by the square root of its
item count (the three: 17%) and `paper` weighs every paper alike. The
recipe keeps `item`; `trainer-lxsdb` is the first run with `sqrt`.

An arm's pre-fast now begins with a meal and its nights are slept
(`pulse/training/arm_init.py`); before, the last meal was the fed day's 19:00
dinner whatever the pre-fast, and the pre-fast was spent awake. Iter 116
is the first run to train from these start states. Measured on iter 114's median person:
at 08:00 after a 10-hour fast (food 10 h ago, not 13) insulin starts at 5.7
µU/mL where it was 6.3 and β-hydroxybutyrate 13% higher; at 12:00 after a
4-hour fast (food 4 h ago, not 17) glucose starts at 101 mg/dL where it was
83 and insulin at 19 µU/mL where it was 8.6. An arm with no pre-fast is
unchanged. Every file in `corpus/findings/` still passes the gate from the
new start states.

Measured on iter 115's log (2026-10-31 to 2026-11-03):

- An epoch with the literature takes 3 to 3.7 h for 40 steps. The 160
  evidence items of an epoch take 9,167 s (57 s each), the rules 2,418 s, the
  dose response 679 s, the carbohydrate balance 537 s, the teacher tape 356 s.
- One core of eight is busy (`--torch-threads` 1, `--compile-steps` off). Peak
  memory is 10.7 GB.
- 63% of the simulated minutes are the 1,440-minute fed day that starts every
  arm, redone for each arm of each item.

Measured on the training job on 2026-11-11 (`trainer-spvj8`: iter 116's
recipe cut to one epoch per phase of 10 steps, warm start from iter 113):

- A step with the literature takes 62 to 73 s, where iter 115's took 270 to
  330 s. The evidence step (4 items and 1 rule) is 54 s of it, 24 s of that
  the spin-up; the dose response is 4 s and the carbohydrate balance 4 s.
- A step on trajectories alone takes 4 s, where iter 115's took 7 s.
- A board without the open-direction pass takes 9 minutes. Peak memory is
  5.3 GB.
- An epoch on trajectories takes 162 s at 8 vCPU, 159 s at 4 and 269 s at 2
  (`trainer-qvcvj`, `trainer-ppqff`, `trainer-rkxxs`). The run is not held to
  one core, and 4 vCPU is the smallest size at full speed, at half the price.
- At those rates the 58 epochs of the recipe take about 23 h (1.4 h on
  trajectories, then 28 epochs of about 45 minutes), not the three hours
  PLAN's stage 1 asks for. The evidence step is four fifths of the time.

Measured on the lab box on 2026-11-11 (`python -m pulse.diagnostics
step-time`, untrained model, 4 items and 1 rule, CPU seconds):

- Arms as the papers ran them, mean of three steps, with the box at 8 CPUs
  and 16 GiB: 46.5 s item by item, 22.0 s batched (spin-up 10.6 s, rollout
  and scoring 3.9 s, backward 7.4 s), peak memory 3.4 GB.

With arms capped at 300 minutes, on the same box at 2 CPUs:


- Item by item, as iter 115 ran: 47.5 s. Batched, on this commit: 9.9 s
  (spin-up 3.4 s, rollout and scoring 2.3 s, backward 4.2 s).
- A simulated minute costs 979 tensor operations, down from 1,167. With a
  different meal list for every member of a batch it cost 8,009.

## Runs

| Execution | Iter | Started | Ended |
|---|---|---|---|
| `trainer-42rqr` | 118, no teacher, gut reads meal details | 2026-11-16 | stopped by its 24 h deadline after epoch 26 of 28 (23 h 30 min), from the corpus of 2026-11-16 with the fasting and clamp topics: `--meal-details`, warm start; output `training/jobs/iter118-meals/` |
| `trainer-ss2p8` | 118, no teacher | 2026-11-16 | finished in 18 h 23 min, the same run without `--meal-details`; output `training/jobs/iter118-base/` |
| `trainer-c7m8g` | 119, no teacher, insulin follows absorbed nutrients | 2026-11-19 | stopped by its 24 h deadline (2026-11-20 19:09 UTC) on image `7c781a5`, warm start from `iter118-base` (the four new islet gains at their defaults), the corpus of 2026-11-19 with every topic read; output `training/jobs/iter119-insulin/` |
| `trainer-gxk99` | 117, no teacher, gut reads meal details | 2026-11-15 | stopped by its 24 h deadline after epoch 25 of 28, from `87cd5b3`: `--meal-details`, warm start from iter 113; output `training/jobs/iter117-meals/` |
| `trainer-kn8wz` | 117, no teacher | 2026-11-15 | finished in 19 h 30 min from `87cd5b3`, the same run without `--meal-details`; output `training/jobs/iter117-base/` |
| `trainer-lxsdb` | 116, no teacher, papers by square root | 2026-11-13 | stopped by its 24 h deadline after epoch 21, the end of phase 2, from `758cc2f`: `--teacher-tape-weight=0 --phase1-epochs=0 --evidence-paper-weighting=sqrt`; output `training/jobs/iter116-sqrt/` |
| `trainer-gqtrs` | 116, no teacher | 2026-11-12 | finished in 19 h 40 min from `b30be0a`: `--teacher-tape-weight=0 --phase1-epochs=0`, warm start from iter 113; output `training/jobs/iter116-noteacher/` |
| `trainer-rh7j2` | 116, cold start | 2026-11-12 | stopped by its 26 h deadline after epoch 51, the end of phase 2 (no phase 3), from `b30be0a`: `--init-from=`; output `training/jobs/iter116-cold/` |
| `trainer-jt2n9` | 116 | 2026-11-11 | finished in 23 h from commit `b30be0a` at 4 vCPU / 16 GiB; boards at each phase, no open-direction pass or ruler |
| `trainer-ppqff` | 116, cut down | 2026-11-11 | cancelled after 10 min: two epochs at 4 vCPU, to time them |
| `trainer-rkxxs` | 116 | 2026-11-11 | cancelled in epoch 5: at 2 vCPU / 8 GiB an epoch took 1.7 times as long |
| `trainer-qvcvj` | 116 | 2026-11-11 | cancelled in epoch 27 of 58: 8 vCPU / 32 GiB, twice the price for the same speed |
| `trainer-spvj8` | 116, cut down | 2026-11-11 | finished in 54 min: one epoch of 10 steps per phase, to time a step; its output is `training/jobs/iter116-smoke/` |
| `trainer-rgg6f` | 115 | 2026-10-31 | timed out at 72 h in epoch 49 of 58; left `last_good.pt` and no board |
| `trainer-bgdhx` | 114 | 2026-10-28 | finished in 3 h 11 min; the last checkpoint with a board |
| `trainer-kd75r` | 114 | 2026-10-28 | out of memory after 10 h |
| `trainer-sfb64` | 114 | 2026-10-24 | timed out at 72 h |
| `trainer-krqfh` | 114 | 2026-10-24 | out of memory after 4.6 h |
| `trainer-2j88z` | 113 | 2026-10-19 | finished in 1 h 38 min |

## The board

The board reports sealed papers and held-out items as separate lines, each
with its paper count and a 90% interval from resampling papers, and can score
the untrained student beside them (`pulse/scoreboard.py`). It scores an item
as a training step does: on people drawn from each arm's own population, each
started from their own fed day and pre-fast.

Iter 116 is the last checkpoint. Its own boards, written by the run
(`training/jobs/iter116/`), median |z| at the end of each phase:

| Points | n | Phase 1 | Phase 2 | Final | Untrained |
|---|---|---|---|---|---|
| Sealed papers | 117 | 1.36 | 1.19 | 1.21 (90%: 0.81 to 1.40) | 1.19 |
| File papers that train | 202 | 1.21 | 0.55 | 0.62 | 0.85 |
| Registry points that train | 73 | 0.68 | 0.34 | 0.33 | 0.65 |
| Held-out items of registry papers | 12 | 0.49 | 0.76 | 0.84 | 0.48 |

- On the sealed papers iter 116 is level with the untrained student, where
  iter 114 was further (1.32). It fits what it trained on, including the 18
  papers read into files.
- Phase 1 (the teacher tape alone) moves the sealed papers away (1.36); the
  literature of phase 2 brings them back to where the untrained student
  stands, not past it.
- The untrained column is from the scoring of 2026-11-11 below, on the same
  items and people. Iter 116 has not been scored with `--baseline` itself.
- Insulin is what holds the sealed line back. Iter 116 is under the measured
  value on 84% of the 31 sealed insulin points (median z -1.53, against
  +0.06 on the 60 it trained on). Fasting insulin sits near 4 µU/mL whatever
  the cohort (`battezzati-2023` 4.1 against 6.7 and 7.3, `vislocky-2008` 3.3
  and 3.5 against 9.2 and 6.4), and after a mixed meal it predicts 32% of
  the measured insulin in `battezzati-2023` and 6% of the incremental area
  in `lindgren-2020`. The insulin it trained on is mostly glucose drinks,
  and 42 of its 46 trained insulin points come from three papers
  (`atkinson-2020` 18, `meyergerspach-2016` 14, `horie-2018` 10). Glucose
  moves the other way: nearer than the untrained student on 23 of 35 sealed
  points.
- Fasting GLP-1 in `zilstorff-2024` is 33 pmol/L against a measured 4 (z
  19); the warm start from iter 113 had 54.

Two runs split iter 116, each changing one thing (their own final boards;
the untrained student as above):

| Run | Sealed papers | File papers that train | Sealed insulin under target | Fasting GLP-1, `zilstorff-2024` (measured 4) |
|---|---|---|---|---|
| Iter 116 | 1.21 (0.81 to 1.40) | 0.62 | 26 of 31 | 33 |
| No teacher (`trainer-gqtrs`) | 1.09 (0.82 to 1.44) | 0.56 | 25 of 31 | 29 |
| Cold start (`trainer-rh7j2`) | 1.20 (0.74 to 1.47) | 0.53 | 24 of 31 | 23 |
| No teacher, papers by square root (`trainer-lxsdb`) | 1.29 (0.79 to 1.56) | 0.58 | 26 of 31 | 24 |

- Without the teacher the sealed papers come in under the untrained
  student (1.09 against 1.19), the first run that does; the interval is too
  wide to call it.
- Neither the teacher nor iter 113's laws make insulin low. After the
  teacher alone, fasting insulin on the sealed arms is about 7.3 µU/mL
  (iter 116) or 8.8 (cold start); the literature takes it to 5.2 to 6.0 in
  all three runs.
- Weighing papers by the square root of their items changes nothing that
  matters: at the end of phase 2 it is 1.29 against the no-teacher run's
  1.22 at the same point, and the sealed arms' fasting insulin is 79% of the
  measured value (85% without it), mixed-meal insulin 32% (33%). The three
  papers that hold most of the trained insulin points are not the cause.
  Its epochs took about 63 minutes, against 49 drawing items uniformly.
- Training barely teaches insulin after a mixed meal. Of the 60 trained
  insulin points, 19 follow a mixed meal and 18 of those are one meal of one
  paper (`atkinson-2020`); 22 of the 31 sealed insulin points follow a mixed
  meal. The islet's amino-acid gain (`log_amino_ins_gain`, since replaced, below)
  starts at 0 and ends between -0.04 and +0.10 in all four runs, and its
  incretin gain (`log_k_incretin`) moves by at most 0.09 from 2.30.
- The cold start stopped at the end of phase 2: its epochs with the
  literature took about 59 minutes, against 49 for the warm start.

Iter 118, the same two runs on the corpus with the fasting and clamp topics
(2026-11-17; their own boards, 164 sealed points in 12 papers, against 128
in 10 for iter 117):

| Run | Sealed papers, end of phase 2 | Sealed papers, final | Sealed insulin under target | Sealed glucose, median \|z\|, final |
|---|---|---|---|---|
| No teacher (`trainer-ss2p8`) | 1.08 (0.86 to 1.27) | 1.01 (0.90 to 1.16) | 25 of 38 | 0.75 |
| No teacher, gut reads meal details (`trainer-42rqr`) | 1.03 (0.80 to 1.21) | 1.01 (0.76 to 1.17) | 25 of 38 | 0.64 |

- Level with iter 117 (1.02, 1.01) on a larger sealed set. On the 140
  sealed items both iters score, the median |z| moves from 1.02 to 0.94
  (no meal details) and from 0.96 to 0.95 (with), inside the intervals.
- Insulin is still under target on two thirds of the sealed points. On the
  common items the mean z rose from -0.86 to -0.43 without meal details and
  from -0.87 to -0.60 with them: less short, not right.
- Free tests on the no-meal-details checkpoint (2026-11-19, no training)
  give the miss one shape: after mixed meals insulin is about 0.6 of target
  at 10-90 min and 0.34 at 120-180 min, while glucose is back at baseline;
  yet fasting and exercise insulin sit 1.4-1.7 times target (sealed mean z
  +1.5 and +2.6). The glucose drink's insulin is right at 2 h. Raising
  `log_amino_ins_gain` 3, 5 or 10 times, or slowing the potentiator
  (`_K_POT` 0.025 to 0.01, `_PHASE2_FRAC` 0.5), barely moves the meal miss
  and worsens the drink: secretion runs on glucose above basal, which a
  mixed meal barely raises. The candidate is secretion driven by nutrient
  absorption plus a lower basal rate; it needs code and a paid run.
- The meal reader again gives the better sealed glucose (0.64 against 0.75)
  and is better on the 36 sealed items new to this corpus (median |z| 1.17
  against 1.35); otherwise the two runs are level.
- Spend: about $6.80 and $9 at Cloud Run list rates (4 vCPU, 16 GiB).

Iter 119, insulin driven by absorbed protein and fat and a basal 0.8 times
lower (2026-11-19; its own board against iter 118's, 176 sealed and 821
trained items in common):

- Sealed papers 0.99 (0.88 to 1.14) on 312 points in 30 papers, 0.92 at
  the end of phase 2; on the common sealed items 0.99 to 0.96, and the
  common trained items 0.59 to 0.59. Level, slightly better.
- Insulin is no longer mostly short: 28 of 65 sealed points under target
  and 37 over (iter 118: 25 under, 13 over of 38). After mixed meals the
  median ratio rose from 0.78 to 0.88 (mean z -0.46 to -0.16), but from
  120 min on only from 0.40 to 0.57. Fasting insulin fell from 1.59 to
  1.12 times target; exercise insulin from 2.57 to 2.14 (two points).
- Mixed-meal glucose moved lower (mean z -0.35 to -0.65); the glucose
  drink did not move (insulin +1.21 to +1.14, glucose +0.12 to +0.06).
  Sealed glucose median |z| 0.84 on its larger set (0.75 on iter 118's).
- Against the bar set before the run: met for the sealed and trained
  miss, the drink and fasting insulin; not met for late mixed-meal
  insulin (0.57 against 0.8) or exercise insulin. Spend about $9.
- Free frozen tests on iter 119 (2026-11-22): at the late minutes the
  model's mixed-meal glucose sits 10-15 mg/dL below baseline (battezzati
  75-78 against 80-83), so any extra secretion pushes it lower; tripling the
  nutrient gain gives late insulin 1.06 but glucose mean z -1.36. Slower
  gastric emptying makes late insulin worse (0.43 at x0.7) and moves the
  drink. Exercise insulin is two thirds basal level: the restoring term has
  no epinephrine gate; gating it moves exercise insulin 1.77 to 1.24 at no
  cost to meals. Stronger glucagon from protein and an extra hepatic output
  below baseline are nearly inert; only damping insulin-dependent uptake
  below baseline (`min(g/gb,1)^n`) holds glucose up. No frozen variant meets
  the bar; the best (glucagon x10, uptake gate ^16) has glucose z -0.84 and
  overall mean |z| 1.04 against 0.90.
- Where the dip comes from (same day): fasting balance is right (EGP 1.80,
  uptake 1.81 mg/kg/min) and meal EGP suppression (about -0.8) is inside its
  cohort target. The small flux is meal glucose reaching plasma: battezzati
  absorbs 46 of 136 g carbohydrate by 150 min and about half of it goes to
  liver and muscle glycogen (`f_liver`, `f_muscle`), so plasma sees about
  18% of the printed grams by then. Halving the muscle share lifts mixed-meal
  glucose (z -0.67 to -0.54) but raises the drink (+0.32 to +0.58), so a flat
  change is not the fix. No corpus paper gives a mixed-meal plasma Ra to set
  this against; a tracer topic was read for it (below).
- The tracer papers settle the direction, not yet a target (their oral
  appearance prints no spread, so it is in `left_out` as
  `oral_glucose_appearance`): on polidori-2013's meal the model absorbs 79
  g from the gut by 6 h, as the paper does, but only 45 g reaches plasma
  against the paper's about 79 g (whelehan-2024: 38 g against 53 g). At 1 h
  and 2 h the model has 46% and 41% of the paper's oral appearance. The
  loss is mostly the first-pass split (54-57% of what is absorbed reaches
  plasma), with gut absorption too slow in the first 2 h. A flat cut to the
  muscle share moves the drink, which fits today.

Iter 117, the first runs on the mixed-meal insulin papers and the meal
details (2026-11-16; their own boards, the untrained student as above):

| Run | Sealed papers, end of phase 2 | Sealed papers, final | Sealed insulin under target | Sealed glucose, median \|z\|, phase 2 |
|---|---|---|---|---|
| No teacher (`trainer-kn8wz`) | 1.08 (0.77 to 1.19) | 1.02 (0.80 to 1.16) | 25 of 32 | 0.65 |
| No teacher, gut reads meal details (`trainer-gxk99`) | 0.97 (0.77 to 1.16) | 1.01 (0.66 to 1.31) | 24 to 25 of 32 | 0.50 |

- Both runs are under the untrained student (1.19) on the sealed papers;
  the no-teacher run on the older corpus was 1.09. Insulin is still under
  target on most sealed points.
- The meal reader improves sealed glucose and otherwise moves the board
  within its interval. It does not separate the equal-gram pairs it trains
  on: `almoraie-2026`'s pomegranate-seed-oil and control meals get the same
  predictions to three figures. On the held-out items of training papers it
  is worse (0.91 against 0.51 at the end of phase 2, 12 points).
- The meal-details run's epochs took longer, and its deadline stopped it two
  epochs into phase 3, so the end-of-phase-2 boards are the even comparison.

Iter 114 was scored on commit `0f4da88` on 2026-11-11,
with the untrained student on the same items and people (`python -m
pulse.diagnostics scoreboard --no-identifiability --baseline`, 4 people per
population, 8 minutes on 2 CPUs). |z| is the miss of the predicted mean in
between-person standard deviations. Iter 114 trained on the Python registries
only, so every paper read from a file is new to it, sealed or not:

| Points | n | Papers | Iter 114, median \|z\| | Untrained |
|---|---|---|---|---|
| Registry points it trained on | 73 | 38 | 0.21 | 0.65 |
| Sealed papers | 117 | 8 | 1.32 (90%: 1.15 to 1.68) | 1.19 (1.10 to 1.35) |
| File papers that will train | 202 | 18 | 1.16 | 0.85 |
| Held-out items of registry papers | 12 | 4 | 1.09 | 0.48 |

- Iter 114 fits what it trained on and is no nearer than the untrained
  student to anything else. On the 319 points from the 26 papers read into
  files it is further than the untrained student on 186 and nearer on 131.
- By marker, over every cohort point: it is further on glucose (1.17 against
  0.69 on 110 points), fatty acids (1.39 against 0.43) and lactate (2.61
  against 0.67), and nearer on heart rate (0.74 against 2.00) and cortisol
  (0.31 against 0.55).
- The corrected pre-fast is not the cause. From the old start states, on the
  same 319 points, iter 114 scores 1.23 and the untrained student 1.07 (0.98
  from the new ones).
- Fasting GLP-1 is 52 pmol/L in iter 114 against a measured 4
  (`zilstorff-2024`), the largest single miss.
- The four rows are the board's items split by whether the paper is a file in
  `corpus/findings/`; the board does not know what a checkpoint trained on.

Not rescored, from the board of 2026-10-29:

- People (CGMacros, 44 episodes): glucose error 28.17 mg/dL calibrated against
  28.06 for the median person. Of 36 accepted calibrations 18 left the median
  person unchanged, and 11 of the 26 that changed improved. Heart-rate skill
  is +0.14.
- 46 of 101 scalar directions are open.

## The corpus

- A new paper is a file: `corpus/findings/<paper>.json` when the model can
  express it, `corpus/refusals/<paper>.json` naming the missing piece when it
  cannot, `corpus/queue/<paper>.json` while it waits to be read
  ([`corpus/README.md`](corpus/README.md)). `python -m pulse.admit <file>` is
  the gate; it takes 6 CPU-seconds on a ten-finding file. The file states
  what the paper printed, in the paper's unit, with the kind of its
  uncertainty and its n; the loader derives the sigma.
- Papers are found by a standing search list, `corpus/searches.json`: one
  PubMed Central open-access query per topic, worked in batches until its
  hits are all screened. All ten topics are worked (insulin after a meal;
  meals equal in grams, different in kind; the oral glucose tolerance test;
  fasting; intravenous glucose and clamps; a bout of exercise; sleep loss
  and timing; heart and circulation; stress and cortisol; glucose fluxes
  traced with isotopes), and all 4,088 hits found so far are screened. Screening runs on Haiku.
- A paper is read by one agent and checked against the paper, number by
  number, by a second that sees only the file and the paper. Only then is it
  committed. First readers run on Haiku since 2026-11-16: re-reading three
  checked papers, it matched all 54 of their numbers; the second reader
  stays on the larger model, since its corrections are judgement calls
  (what belongs, which group, which unit).
- A paper whose people are a sealed paper's people is refused
  (`population`/`shares_people_with_sealed_paper`): garces-2020 prints the
  same obese men as sealed sandovalalzate-2016. The gate checks this
  (`pulse.admit`, "shares no people with a sealed paper"): two or more
  distinctive `value ± spread` pairs, or a trial registration, shared between a
  new paper and a paper on the other side of the seal fail it; one shared pair
  is noted. The same evidence between two papers on the same side of the
  seal is a warning: the reader enters each of those people's measures in one
  paper only (pilis-2025 and stec-2023, one programme's men). Run over the 68
  findings files and the 68 Python-registry papers on 2026-11-16 it found no
  match across the seal.
  The table counts the files. Given the paper's full text
  (`admit --paper`), the gate checks that each quote is in it; the full texts
  are not in the repository. Run over the 35 files on 2026-11-11: 267 quotes,
  none missing.
- The 68 papers filed before that are still Python registries. Each has a
  training item, so none of them is sealed (`pulse/knowledge/sealing.py`). A
  paper read from a file is held out when the draw on its slug says so, a
  fifth of them; Mari 2013 was the first.
- The five held-out items predate sealing by paper. Four share a paper or an
  arm with training and two have targets equal to a starting constant; the
  board prints what each shares.
- About 45% of the cohort targets are textbook ranges, not numbers reported
  by a study (review of 2026-11-11). Their papers are the slugs `textbook-*`,
  `ada-2024`, `who-1999`, `acsm-guidelines`, `icp-reference-range-2022`,
  `digestion-1978`, `diabetes-2017-66-1999`, `pmid-2424396`, `pmid-3525269`
  and `task-force-1996`.
- The format takes three single-arm statistics, `change_from_baseline`,
  `area_under_curve` and `incremental_area`; an area is carried as its
  time-weighted mean over the paper's own sample minutes, and an incremental
  area whose paper does not say what it does under baseline is entered as
  `unstated` and held to the nearer of the two readings. Six papers use them:
  8 changes, 4 areas and 38 incremental areas, 26 of them `unstated`
  (`atkinson-2020`, which trains, and `lindgren-2020`, which is sealed). No
  `left_out` entry names one of the three as missing any more: an area that is
  still out lacks its sampling minutes, its time unit or a writable arm in the
  paper itself. An area the paper prints already divided by its span is
  entered as printed with `per_span: true` in place of a time unit
  (`oberoi-2022`'s net areas over 0-180 min).
- Insulin converts from U/L (x1000) and, like GIP and GLP-1, from a mass
  concentration (pg/mL, ng/L, ng/mL, µg/L) through the hormone's molar mass
  (`pulse/knowledge/corpus_units.py`). The four files that had left values
  out for want of these were read again: `mok-2016` gained its insulin
  iAUCs, `wong-2025` its GIP and GLP-1, `oberoi-2022` its net areas;
  `deru-2023`'s insulin stays out, now as `assay_not_comparable` (a
  multiplex panel's levels, high and flat through a 36 h fast).
- A meal in a file may state what the paper prints beyond its grams:
  `fibre_g`, `sugars_g`, `fat_g_by_type`, `fat_source`, `protein_source`,
  `form`, `food` (`corpus/README.md`, "What a meal says"); `carbs_g` is
  available carbohydrate. They are carried beside the meal tuple
  (`CohortArmSpec.meal_details`), and on each `MealEvent` of the arm's
  rollout. 54 meal arms in 16 papers carry them; 53 state something the
  meal reader reads (`suzuki-2012`'s glucose load names only its food), and
  300 of the 634 training points are on such arms. The recipe's model does
  not read them; a model trained with `--meal-details` does (The model,
  above). The gate's
  check 5 treats equal grams with a detail both arms state differently as two
  protocols. The gate still probes the model without the reader, so a
  finding that differs only through a detail stays out
  (`meal_details_not_read`), and `corpus/README.md` still says the model does
  not read them.
- One item is a spread across people and none is a correlation. Every
  population is healthy adults, except the groups three papers read from
  files enter as `unselected` (`meyergerspach-2016`'s obese group,
  `skurvydas-2025`, `welch-2023`). No patient group is in.
- About 50 items cite several papers; the first is taken as the paper.

## Known wrong

- A cohort item with no `n_arm` has its sigma divided by the square root of
  the number of people drawn (`pulse/cohort_loss.py`,
  `pulse/training/evidence_step.py`), so how tightly it binds depends on a
  training flag. After the 2026-11-19 sweep no scored point is in that case:
  the 34 that were had their n looked up in the repository. Two state it
  (`day_night_hr_contrast`, `night_hr_level`: n = 134, in their `source`)
  and carry `n_arm=134`; the other 32 were Python-registry points that cite a
  review or a textbook with no n anywhere in the repository, and were removed
  from the registries (15 papers left training with them). The code path stays
  for `n_arm=None` because the unit tests exercise it and a spread has no n
  to divide by (`cryer_1987_fasting_glucose_spread`, the one remaining
  `None`). Training iters up to 119 trained on the 32 dropped points (119 was launched on `7c781a5`, before the drop).
- `pytest` from the repository root fails while collecting
  `scripts/iter89_recovery_test.py`. Run `pytest tests/`.
- There is no CI. The suite is run by hand:
  `uv run --group dev pytest tests/ -n auto`.
- No test fails (run file by file on 2026-11-19, on an 8-CPU, 16 GiB box, each
  file capped at 5 minutes). Three files passed 5 minutes and were not run:
  `test_iter81_stability.py`, `test_iter97_ruler_calibration.py` and
  `test_training_signals.py`. The other 111 files pass, `test_cohort_ablation.py`
  in 299 s. Ten tests in 8 files had failed; all were expectations made stale
  by deliberate changes, and none was a code bug:
  - Since 7c781a5 (insulin follows absorbed nutrients; basal insulin 0.8 of
    the centre): parameter counts (+3: four published gains added, one retired;
    `test_b1`, `test_circadian_learnable`, `test_planned_integrator`), the
    planned integrator's recorded rollout, a no-meal drift of 1.8 mg/dL where
    the test wanted under 1 (`test_a9`), a fed-day peak of 104.75 under an
    absolute 105 (`test_iter97_student_model`), and the fasting
    gluconeogenic fraction of 0.27, which read the state at the old insulin
    centre of 10 (`test_population_prior`).
  - `test_dg_is_zero_at_the_decoded_setpoint` perturbed every parameter by
    N(0, 1), including the renal threshold (glucosuria below Gb, landed
    after the test) and the epinephrine threshold (a non-zero drive at the
    population's epinephrine adds hepatic output the basal split does not
    carry). Both are now left alone. A learned epinephrine threshold below the
    population's epinephrine does move the fasted set point off `k_ii * Gb`;
    left as it is.
  - The gut tests took `argmax` of a flat plateau, which float noise (1e-7)
    puts at minute 1; they now ask that minute 0 be on the plateau.
  A fresh model starts at the old normative insulin (10, `NORM_CENTER`), above
  its new set point of 8, so its first hours drift; not changed, since it would
  move trained behaviour.
- 693 parameters of the gut kernel (the mixture logits of its last layer,
  rows 0-20 of `gut.kernel.kernel[-1]`) receive no gradient from any loss.
  They are dead rows, not a closed gate: `GutModule.appearance` discards
  the mixture weights since appearance comes from the stomach simulation,
  and only the bioavailability rows 21-23 are read (they do get gradient).
  Harmless (about 2% of parameters, no output); meal-kind differences go
  through the meal reader, which is live. Checked on iter 118 (2026-11-19).
- The stomach's split correction acts on the three gastric pools and never
  reaches absorption: no module reads a gastric state, so the pools are
  decorative and the correction is a dead branch (zero gradient, exactly
  zero in iter 114). Appearance comes from the by-meal stomach
  (`gastric.simulate_by_meal`), which the meal reader's speeds do drive
  (live gradient on a reader grafted onto iter 118, 2026-11-19); fat slowing
  carbohydrate is in its shared-calorie law. Harmless; retire it when the
  module is next touched.
- The cortisol gain on glucose production is centred on the population's
  12 µg/dL, not the person's own basal cortisol, which the stress module
  learns (7.3 to 19.6 across 512 draws). Basal production is off by up to 9%
  for a person away from the median.

- The activity input ran only to 1.0 (VO2 35.4 mL/kg/min), so hard bouts were
  left out as `exercise_above_activity_scale`. It now runs to 2.0 (VO2 67;
  `pulse/knowledge/corpus.py`, `tests/test_activity_above_one.py`), but above
  1.0 the model extrapolates: RQ is clamped at 1.0, sympathetic tone at 1, and
  the heart-rate driver has never been trained above 1. A cold test of iter
  117 (no training, 2026-11-16) on the 21 findings this unlocked (li-2024,
  crabtree-2025, garnachocastano-2018, roberts-2023) gave a median |z| of
  about 0.7, against 1.05 on the sealed exercise items below 1.0, so no fix or
  retrain was made for it. Heart rate runs low at every Bruce stage of
  crabtree-2025 alike (-1.9 at level 0.45, -2.0 at 1.07): a bias of its own,
  not of the extrapolation. All-out sprints stay out.

Reported by the review of 2026-11-11 and not yet confirmed:

- calibration fits interstitial glucose while evaluation reads plasma.

## Live

- The `engine` service is not ready: its container fails the health check
  (2026-11-11). `models/prod.pt` dates from 2026-06-07 and has an earlier
  state layout.
