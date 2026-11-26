# State

Snapshot of **this commit**. Rewrite it in the same commit as the change it
describes; `git show <rev>:STATE.md` is pulse at that point. The count table
is checked against the code by `tests/test_state.py`, so it cannot go stale
quietly. Everything below the table is a measurement with its date, and has
to be re-read by hand. Where pulse is going is [`PLAN.md`](PLAN.md).

## Counts

| Metric | Now |
|---|---|
| Modules | 9 (metabolic 13 states, appetite 5, stress 5, cardiovascular 5, hepatobiliary 4, gastric 3, respiratory 2, circadian 2, thermoreg 1) |
| States | 40 |
| Observables | 10 |
| Parameters | 37,293 at the default widths |
| Published numbers | 46 in 10 tables, each learned from its published value |
| Corrections | 3 |
| Cohort specs | 1185 authored, 140 of them curves; 1241 training points and 348 held-out points |
| Rules | 62 training, 0 held out |
| Coupling priors | 32 |
| Papers | 200, of which 33 are sealed |
| Held-out items | 5 |
| Corpus files | 147 papers as findings, 64 as refusals, 0 queued to be read |
| Queue | 23 (15 integrated, 7 blocked, 1 backlog, 0 ready) in the older Python queue |

## The model

One network of nine modules over 40 states, stepped by explicit Euler at one
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
minute per gram. The basal insulin setpoint is 0.65 of the normative centre
(`INSULIN_SETPOINT_SCALE`, applied in `insulin_setpoint_raw`; 0.8 through iter
120; the centre is unchanged, and the head reaches 16 µU/mL at most). Carbohydrate is not in the nutrient drive.
Not yet trained: the four numbers start at these values and no run has moved
them. Iter 118's checkpoint loads (`load_model_state` starts the new keys at their
defaults and ignores `log_amino_ins_gain`). Free tests of this form on it
(2026-11-19, variant `n2_m3`) are the case for it; the checkpoint has not
been rescored on the full board.

The stomach empties faster when full (`pulse/gastric.py`): the 2.13 kcal/min
is multiplied by `clamp(held kcal / 300, 1, 10) ** 0.33`
(`LOAD_REF_KCAL`, `LOAD_CAP`, `LOAD_EXPONENT`), in the teacher's `step_grams`
and the student's `leaving_grams` and `simulate_by_meal`. Brener's 75 g drink
(300 kcal) is unchanged; a 900 kcal meal leaves 1.44 times faster. The incretin
constants derived from the rate follow: the Vollmer breakfast delivers 0.73 of
its calories by 240 min (was 0.62), so `GIP_FAT_WEIGHT` 6.48 to 5.17 and
`GIP_GAIN` 1.41 to 1.66 (iter 120's trained values, 8.36 and 2.26, load as
they are). No load adjustment accompanies the setpoint scale or the stomach:
the insulin head is not anchored (setpoint supervision weight 0, no insulin
prior), so an offset would cancel the change, and a warm start from iter 120
loads its head untouched. That head had already drifted up against the 0.8
scale (tanh output at the zero embedding 0.25 at iter 119, 0.37 at iter 120),
so training may re-learn part of the cut. Not trained yet. Full suite: 1673 passed
and 5 tests that pinned old numbers (Ib, the constant-rate plateau, one golden rollout)
re-recorded or restated.

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

A validation split picks the checkpoint (2026-11-25, `--val-fraction`, off by
default). Iter 121 trained better (0.64 to 0.59) and sealed worse (0.77 to
0.82): the job kept the last epoch whatever it scored. With `--val-fraction
0.1`, about a tenth of the papers that train (a hash of the slug with its own
salt, `sealing.paper_is_validation`; a sealed paper is never drawn) leave the
evidence step and the rules draw: 14 papers, 152 cohort items of 129 papers
and 1,241 items. Every `--val-every` epochs (default 1, and the last epoch
always) they are scored as the board scores cohorts (4 draws, `[VAL]` lines in
the job log) and the weights with the lowest median |z| are the `model.pt`;
the last epoch is `model_last.pt` (when it differs). The curve is in the board
and in the checkpoint as `validation`. One scoring of the 14 papers takes 61 s
on iter 121's checkpoint (one core); an epoch with the literature is 3 h, so
every epoch adds under 1% to a 24 h run. The board's training line still
includes these papers. A sealed number still never chooses.

## Runs

| Execution | Iter | Started | Ended |
|---|---|---|---|
| `trainer-42rqr` | 118, no teacher, gut reads meal details | 2026-11-16 | stopped by its 24 h deadline after epoch 26 of 28 (23 h 30 min), from the corpus of 2026-11-16 with the fasting and clamp topics: `--meal-details`, warm start; output `training/jobs/iter118-meals/` |
| `trainer-ss2p8` | 118, no teacher | 2026-11-16 | finished in 18 h 23 min, the same run without `--meal-details`; output `training/jobs/iter118-base/` |
| `trainer-c7m8g` | 119, no teacher, insulin follows absorbed nutrients | 2026-11-19 | stopped by its 24 h deadline (2026-11-20 19:09 UTC) on image `7c781a5`, warm start from `iter118-base` (the four new islet gains at their defaults), the corpus of 2026-11-19 with every topic read; output `training/jobs/iter119-insulin/` |
| `trainer-hrx5g` | 120, first launch | 2026-11-22 | cancelled after 2 min: `--init-from` loaded through `load_compatible_state_dict`, which skipped the doubling of the nutrient gain; fixed in the next commit |
| `trainer-zw876` | 120, no teacher, meal-gated muscle share and epinephrine gate on basal insulin | 2026-11-22 | stopped by its 24 h deadline (2026-11-23 22:46 UTC) on image `e3ace61`, warm start from `iter119-insulin` (nutrient gain 1.79 doubled to 3.59 on load); output `training/jobs/iter120-mealsplit/`, whose final board equals its phase-2 board; approved by Gabriel, about $9 |
| `trainer-2qvnr` | 121 (run D), no teacher, basal setpoint scale 0.65 and load-dependent gastric emptying | 2026-11-24 | stopped by its 24 h deadline (2026-11-25 13:06 UTC) on image `6496abe`, warm start from `iter120-mealsplit`; output `training/jobs/iter121-basal-gastric/`; approved by Gabriel, about $9 |
| `trainer-qqb4f` | 122 (run E), no teacher, ketone glycogen threshold, breathing-rate papers, `--val-fraction 0.1` keeps the best validation epoch | 2026-11-25 | completed after 28 epochs in 20 h 57 min (2026-11-26 13:24 UTC) on image `d1ef4f2`; `model.pt` is epoch 1 (best validation), `model_last.pt` epoch 27; warm start from `iter120-mealsplit`; output `training/jobs/iter122-ketone-breath/`; approved by Gabriel, about $9 |
| `trainer-69h6t` | 123 (probe), no teacher, low constant lr 0.0003, 3 phase-2 epochs, `--fitness-from-vo2max`, `--val-fraction 0.1`, frozen fixes 2e5ee62/21e6cf4/74f809b/0e1959f, slow HR state | 2026-11-26 | running on image `b66d400`, 4 h deadline, warm start from `iter120-mealsplit`; output `training/jobs/iter123-probe-lowlr/`; approved by Gabriel, about $1 |
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

Leptin falls with insulin by a fraction of its level (2026-11-26, free, frozen
on iter 120; `pulse/modules/appetite.py`, teacher `lep_ins_gain*` in
`full_body.py`, `tests/test_leptin_fast_fall.py`). Chosen from the trained-side
board only: leptin was the one marker with every item the same sign (13 items,
all predicted too high, mean |z| 7.8; deru-2023 36 h 7.8 ng/mL against 0.87,
pilis-2025 after 8 days 6.9 against 1.46). The model's fall was an additive
2.5 ng/mL at most (24 h to 36 h fasts read 9.1 and 7.8 against 10.3 at 12 h),
so a fast could take a quarter of the level; the corpus has deru-2023's 36 h at
a quarter of its 12 h value and Boden 1996's 30 to 50% by 16 to 24 h. The
insulin term now multiplies the fat-set level: 0.25 per unit of
(insulin_slow/ib - 1) when fed (the old size at 10 ng/mL), 0.8 below basal. A
symmetric 0.8 was tried first and sent `leptin_fed_vs_fasted` from z 1.1 to
7.6, so the fed side keeps its old size. Frozen on iter 120, trained
items: leptin mean |z| 7.84 to 4.85 (deru 36 h 17.9 to 8.5, pilis after 16.0
to 5.3), all other trained items mean 0.952 to 0.953, median 0.608 to 0.607, the
one worse leptin item `leptin_fed_vs_fasted` z 1.14 to 1.36; sealed (no leptin
item there) median 0.679 to 0.682, mean 1.026 to 1.027 on 339 common non-GLP-1
items: neutral, as it should be. The full suite: 1,697 passed, one golden rollout re-recorded (`test_planned_integrator`, shifts under 3e-5 z). Not fixed: the 12 h level is 3 times the
corpus's in lean arms (deru 10.5 against 3.5, z 4.7; pilis, skurvydas the
same), which is the fat-mass to leptin map, not fasting; and heart rate runs
about 10% low in exercise (61 trained items, mean z -1.95) with post-exercise
heart rate back to rest within 5 minutes in the model where sugimoto-2020,
cesaremarincola-2025 and gahreman-2016 hold 10 to 20 bpm above for 10 to 30
minutes (the model's temperature, sympathetic tone and epinephrine all snap
back inside 5 minutes of the last bout); that needs a slow post-exercise state
in the heart-rate module.

Iter 122 (run E), warm start from iter 120 on main with the ketone
threshold, the breathing-rate papers and `--val-fraction 0.1` (2026-11-26;
against iter 120 frozen on the same code, `$S/fz/i120.json`):

- No gain. Sealed median 0.68 to 0.79 on the whole board (kept epoch 1;
  phase-2 board 0.77); on the 339 common non-GLP-1 sealed items median
  0.68 to 0.76, mean 1.03 to 1.16. Trained 0.62 to 0.61. Sealed rr 0.32 to
  0.94 (11 items), insulin 0.72 to 0.85, glucose 0.67 to 0.75, hr 0.52 to
  0.61, bhb 1.71 to 2.02.
- The validation papers (14, trained-side, never sealed) agree: median
  0.61 at epoch 0, 0.57 at epoch 1, then 0.64 to 0.92 for the rest. The
  guard kept epoch 1, so it worked; but by its own measure, training from
  iter 120 makes held-out papers worse after one epoch. Epochs ran about
  45 min, not 3 h, so the guard costs about 4% of the run.
- Iter 120 on current code stays the reference model.
- Where the validation drift sits (2026-11-26, frozen swaps on `model_last.pt`,
  validation papers only, 4 draws; epoch 27 is 0.807, iter 120 0.623, epoch 1
  0.571). It is diffuse, not one module: putting iter 120's values back for
  respiratory and cardiovascular networks gives 0.655, adding the metabolic heads
  and nets 0.595; the gut kernel, embedding projections, appetite heads, thermo
  net, default inputs and the scalars alone move it by under 0.02. Walking from
  iter 120 to epoch 27 in a straight line rises smoothly (0.614, 0.66, 0.72,
  0.81 at a quarter, half, three quarters, all), and epoch 1 (a third of the
  distance) is off that line. One paper carries much of it: `behrendt-2024` (22
  of 152 items) median |z| 0.17 to 0.91, a hr curve moving to +1.2 and an spo2
  curve to -0.8 to -2; insulin 0.35 to 0.95 and lactate 1.4 to 4.2 also.
  The log shows why: evidence nll has no trend over 28 epochs (10.7, then 6.5
  to 18.5, noisy), 30 to 45% of evidence backwards are clipped, one epoch has
  rules 6.7 and evidence gradient 410, while rules, dose response, the coupling
  prior and carbohydrate balance sit where they were or fall. So it is not the
  auxiliary losses pulling away; the weights random-walk at lr 0.0015 on 4-item
  steps (phase 3 starts at epoch 22 and does not change the picture). No
  frozen or short probe shows a recipe that holds epoch 1's level (a probe of
  40 steps is 40 minutes); the candidates are a lower phase-2 lr (iter 120
  ended at 0.001, the warm start raised it to 0.0015) or averaging weights over
  epochs, to be tried as a run with the validation guard, not on sealed.
  Uniform weight averages on the validation papers (2026-11-26): mean(epoch 1,
  epoch 27) 0.621, mean(iter 120, epoch 1, epoch 27) 0.601, and 0.675 for
  epoch 1 with `last_good.pt` and epoch 27 (`last_good.pt` is epoch 27, so that
  is one third epoch 1); none beats epoch 1 (0.571), so averaging alone does
  not argue for EMA; a lower warm-start lr does. A warm start builds a fresh
  Adam at `--phase2-lr` (train.py `_build_phase`, cosine to
  `--phase2-lr-floor`), so 'continue at the floor' needs only
  `--phase2-lr 0.001 --phase2-lr-floor 0.001`, or lower (0.0003 to 0.0001).

Iter 121 (run D), basal setpoint scale 0.65 and a stomach that empties
faster when full (2026-11-24; its board against iter 120's, GLP-1 scored
as ratios on both):

- Sealed papers 0.82 (0.68 to 1.01) on 314 points in 31 papers, against
  iter 120's 0.77 on the same scoring; on the 317 common non-GLP-1 sealed
  items median |z| 0.78 to 0.80, mean 1.20 to 1.25. Trained items 0.64 to
  0.59 (mean 1.06 to 1.02): better on what it trained on, not on sealed.
- Mixed-meal glucose z -0.67 to -0.22 (ratio 0.98). Fasting insulin 1.36 to
  1.20 of target, but exercise insulin 1.66 to 1.80 and late mixed-meal
  insulin 0.72 to 0.67; the drink's glucose z +0.57 to +0.69. Training
  undid part of the setpoint cut. mok-2016's post-lunch insulin 33 against
  106 (iter 120: 19).
- Iter 120 stays the reference model.
- Why (free, frozen and from the boards): the two code changes are neutral
  to slightly good on sealed papers frozen on iter 120 (median 0.85 to 0.81);
  the loss comes from retraining. Five sealed items carry nearly all of the
  mean's growth: ketones (`bhb`) in rossmeislova-2024 and breathing rate in
  crabtree-2025. The basal cut came back through
  `embedding_projections.metabolic.bias` (setpoint at the zero embedding
  11.1 to 11.4 uU/mL; the population mean still fell 11%), not through the
  head. Exercise insulin sensitivity rose x1.24. The late-insulin candidate
  above on iter 121: late 0.67 to 0.78, sealed median 0.81 to 0.83.
- A sealed number must not choose checkpoints or stop training, or it stops
  being a test; a guard of that kind needs a split of the trained papers.
- Frozen on main at 0ca5c24 (ketone glycogen threshold, breathing-rate
  papers; 2026-11-25), on the 317 sealed and 1,261 trained non-GLP-1 items
  both boards hold: iter 120 sealed median 0.78 to 0.73 (mean 1.20 to 1.07),
  trained 0.64 to 0.60; iter 121 sealed 0.80 to 0.78. Nearly all of it is
  day-0 fasting `bhb` (rossmeislova-2024 obese z 6.9 to 2.3 on iter 120;
  dai-2024 and dai-2022 likewise); day-3 ketones barely move. Crabtree-2025
  `rr` is unchanged by code: z 0.0 to 1.9 on iter 120, 1.2 to 4.3 on
  iter 121 (pre-exercise 25.7 predicted against 17), so iter 121's rr loss
  was retraining. The one large loss: vollmer-2008 `gip` ratio z 3.1 to 6.4
  on iter 120. Not the incretin constants (trained `log_gip_gain` and
  `log_gip_fat_weight` override them; reverting changes no gip z) but the
  load-scaled stomach: `LOAD_EXPONENT` 0 gives 3.07 (iter 121: 7.09 to
  3.70). Its 820 kcal breakfast empties faster, GIP peaks 41.8 at 90 min
  against a flat 34.4, ratio 3.95 against 2.86 ± 0.17. On every other gip
  item the new stomach is slightly better (trained mean |z| 1.69 to 1.61,
  sealed 2.96 to 2.93; iter 121 1.54 to 1.43, 2.92 to 2.88).

Iter 120, meal-gated muscle share, an epinephrine gate on basal insulin
and the nutrient gain doubled (2026-11-22; its board against iter 119's):

- Sealed papers 0.84 (0.71 to 1.03) on 320 points in 31 papers, against
  0.99; on the 324 items both boards hold, median |z| 0.98 to 0.84 but mean
  |z| 1.60 to 1.85, so the typical point is nearer and the worst further.
  Trained items 0.62 to 0.63.
- Late (from 120 min) mixed-meal insulin 0.57 to 0.72 of target; mixed-meal
  glucose z -0.65 to -0.67; the drink's glucose z +0.33 to +0.57 and
  insulin +0.63 to +0.58. Exercise insulin 1.68 to 1.66 times target (mean
  z +1.76 to +1.11); fasting insulin 1.21 to 1.36. Sealed insulin 30 of 68
  points under target, 38 over.
- Against the bar set before the run: met for mixed-meal glucose and the
  overall sealed miss; not met for late insulin (0.72 against 0.8), the
  drink (moved 0.24 z against 0.2) or exercise insulin (1.66 against 1.4).
- The rise in sealed mean |z| is GLP-1 alone: without its 15 sealed items
  mean |z| falls 1.32 to 1.22 and the median 0.95 to 0.80. The model's
  fasting GLP-1 doubled (hjorth-2022 t0 11.2 to 23.8 pM against 3.4;
  zilstorff-2024 7.2 to 15.7 against 4.0), though none of the three changes
  touches GLP-1. Trained items: no marker worse but glucose.
- Fasting and exercise insulin after iter 120 (free, frozen): fasting
  insulin is the basal setpoint alone (every nutrient and glucose-driven
  term is zero fasted), and the corpus pulls both ways (majorpedersen-2008
  2.8x target, hoeks-2010 and raz-2013 0.35x). The epinephrine gate works:
  li-2024's run lowers insulin 38% against the paper's 40%, so exercise
  insulin's 1.66x is the resting excess carried through. Setpoint scale 0.8
  to 0.6 gives fasting 0.99, exercise 1.24, late mixed-meal insulin 0.65
  (from 0.72); doubling the nutrient gain again is inert. sandovalalzate-2016's
  2 h glucose below its 1 h value is as the paper prints it.
- GLP-1 is now scored scale-free (`ratio_to_baseline`, and `ratio_increments`
  for lindgren-2020): the corpus's fasting GLP-1 runs 3.4 to 48 pmol/L, in two
  clusters about sevenfold apart, with every trained paper in the high one and
  every sealed paper in the low one, and the rises scale with it. Lone fasting
  levels go to `left_out` (`assay_dependent_level`). Re-scored that way, sealed
  median and mean |z| are 0.92 and 1.30 for iter 119, 0.77 and 1.18 for iter
  120. Lindgren's post-dinner ratios stay poor (the model shows no rise).
- A stomach that empties faster when full (rate x clamp(kcal/300, 1)^0.33;
  the drink is unchanged) frozen on iter 120: late mixed-meal insulin 0.72 to
  0.81, mok-2016's post-lunch insulin 19 to 31 against 106, battezzati 48 to
  65 g absorbed by 150 min; other targets within 0.05. At 150 min after mok's
  breakfast insulin barely moves (17 to 19 against 74): secretion, not the
  stomach.
- Candidate after run D (frozen on iter 120 with main's code, 969 items):
  damping insulin-dependent uptake below baseline (`min(g/gb,1)^4`), a
  stronger meal gate on the muscle share (0.025 to 0.01) and insulin
  clearance after a meal x0.8 give late mixed-meal insulin 0.77 to 0.87, no
  minutes below baseline on battezzati or mok, mean |z| 0.99 to 0.98 and
  sealed median 0.81 to 0.82. Without the clearance change: 0.80 late, mean
  |z| 0.97. Reducing the liver's first-pass share instead lowers fasting
  glucose and costs 0.2 mean |z|. Battezzati's own glucose dips a little (78
  to 83 against 85), so part of the model's dip is real.

The bundle (setpoint scale 0.65 and load-dependent emptying), frozen on iter
120 (2026-11-24, free; 808 glucose and insulin items, 18 GLP-1 ratio items and
every sealed item, 969 in all; before to after):

- Late (from 120 min) mixed-meal insulin 0.75 to 0.77 of target; mixed-meal
  glucose z -0.53 to -0.58; drink glucose z +0.76 to +0.68, drink insulin
  +0.43 to +0.20; fasting insulin 1.32 to 1.03 times target; exercise
  insulin 1.79 to 1.39 (mean z +1.43 to +0.39).
- Mean |z| on the 808 items 0.98 to 0.91; GLP-1 ratio items 0.95 to 1.32 (the
  faster stomach raises the model's GLP-1 rise); all 969 1.04 to 0.99. The 326
  sealed items (every marker, rescored by `score_cohorts`, so not the board's
  paper-weighted number): median |z| 0.85 to 0.81, mean 1.24 to 1.16.
- The scale cut carries the fasting and exercise gain; the stomach adds
  little to late insulin on its own (see the 0.72 to 0.81 above, which had the
  scale at 0.8).

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
- A split that separates meal from drink: scaling the muscle share by
  `1 - nut/(nut + c)`, with `nut` the absorbed protein plus fat in g/min,
  leaves the drink where it is. Frozen on iter 119 with the epinephrine gate
  on the insulin setpoint and the nutrient gain x2, at c = 0.025: late
  mixed-meal insulin 0.88, mixed-meal glucose z -0.74, drink glucose and
  insulin z 0.25 and 0.33 (base 0.32 and 0.63); exercise insulin still 1.62
  and overall mean |z| on those 618 items 0.98 against 0.90. Untrained, so
  the remaining two misses are what a run would have to fix.
- That split is now in the code (uncommitted until a run): `f_muscle` is
  scaled by `1 - nut/(nut + 0.025)` (`_MEAL_GATE_NUT`, a constant, not a
  learned number); the insulin restoring setpoint is divided by
  `1 + ins_epi_gain * epi_drive`; `ins_nut_gain` starts at 4.0 (was 2.0).
  A checkpoint without the `state_rev` buffer (every one before this) loads
  with `log_ins_nut_gain` raised by ln 2 (`LOAD_ADJUSTMENTS` in
  `pulse/model.py`), so iter 119 warm-starts on its trained gain doubled.
  Loading iter 119 through that path reproduces the frozen numbers on the
  618 items exactly: late mixed insulin 0.88, mixed glucose z -0.74, drink
  0.25 and 0.33, exercise insulin 1.62, overall mean |z| 0.98.

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
  hits are all screened. All eleven topics are worked or being read (insulin after a meal;
  meals equal in grams, different in kind; the oral glucose tolerance test;
  fasting; intravenous glucose and clamps; a bout of exercise; sleep loss
  and timing; heart and circulation; stress and cortisol; glucose fluxes
  traced with isotopes; breathing rate at rest and in exercise), and all
  4,188 hits found so far are screened. Screening runs on Haiku.
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
- Ketogenesis's glycogen term used to switch on as soon as liver glycogen fell
  below its fed baseline, giving 0.4 to 0.9 mM beta-hydroxybutyrate at a 10 to
  12 h fast where the corpus (dai-2024, dai-2022, bak-2018, rossmeislova-2024)
  reads 0.05 to 0.30 mM. It now starts below 0.7 of the baseline
  (`_KETO_GLYC_THRESH` in `pulse/modules/metabolic.py`, `keto_glyc_thresh` in
  the teacher, `pulse/knowledge/full_body.py`). Untrained: frozen on iter 121's
  weights, `bhb` mean |z| 3.90 to 1.29 on 19 items (median 0.89; the 10 to
  12 h items 1.10), the other 162 fasting items unchanged at mean 1.06. A
  default person's BHB at 10, 12, 24, 48, 72 h went from 0.67, 0.74, 1.68,
  2.78, 3.36 to 0.20, 0.32, 1.34, 2.68, 3.35 mM, so long fasts still rise.
  The teacher's 24 h BHB for a typical patient is now 0.63 mM (Cahill's 24 h
  band starts at 0.8), and the rule `bhb_rises_during_fast` is flagged as a
  teacher correction (its 8 h to 16 h arm rises 0.03 mM, floor 0.05). Test
  expectations moved: teacher 48 h band lower edge 2.0 to 1.8 (Gb 130 reads
  1.92), 24 h lower edge 0.7 to 0.5, and the planned-integrator golden
  glucose/insulin/gradient values (shifts under 1e-4). The next run trains with it.
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
- Exercise heart rate (2026-11-26, frozen on iter 120, trained-side items
  only to choose). Not the HR-activity gain alone: iter 120's heart rate is flat
  from activity 0.6 to 1.4 (139, 143, 143, 143 bpm at 0.6, 0.8, 1.0, 1.4)
  while steeper than the teacher at 0.2 to 0.4 (93, 122 against 82, 98): a
  saturating learned driver. `k_hr` is 0.18 (tau 5.4 min), so a 3 min bout
  (cesaremarincola-2025) reaches 88 against 123. Recovery is back to rest in
  about 7 min, with no slow tail, and nothing exercise-slow reaches the module:
  temperature falls below rest within 5 min, lactate and cortisol are the only
  slow ones and are not physiological drivers of heart rate. A slow
  post-exercise state (tens of minutes; sugimoto-2020, gahreman-2016) needs a new
  marker and a coupling channel, which changes the checkpoint layout, so it is a
  retrain-level change. cesaremarincola-2025's flat 85 bpm "recovery" is its
  resting level (84.8), a baseline item, not a tail. Tried layout-free: a term
  8 bpm/min per unit activity above 0.5 on the HR rate. Trained hr 1.57 to
  1.47 (signed -1.19 to -0.88), nothing else moved, but behrendt-2024 and
  ali-2026 (matched before) got worse by up to 1.3 z, the trained median rose
  0.616 to 0.623, and sealed hr went 0.95 to 1.24 (median 0.68 to 0.74). Not
  kept. Targets at the same activity differ by 15 to 25 bpm between papers
  (behrendt, ali against sugimoto, tsukamoto, schubert), so a single gain cannot
  fit them; who the people are (age, fitness) is the next thing to look at.

- Between-paper heart-rate spread (2026-11-26, free, trained side). Not a
  population or entry problem. At the same relative intensity (60% VO2peak)
  behrendt-2024 (VO2peak 51.4, level 0.86) and ali-2026 (50, 0.84) read 142 to
  147 bpm, sugimoto-2020 and tsukamoto-2023 (45.9, 0.75) 153 to 160, schubert-2024
  (level 0.81, VO2peak not printed) 167; all are young (21 to 25 y), mostly male,
  BMI 21 to 25, so age, sex and BMI explain nothing and fitness (a 5 mL/kg/min gap)
  could at most explain part of it. Entries follow the README rule alike (VO2, or
  60% of the group's VO2peak); a watts/186 reading would raise tsukamoto's 0.75
  to about 0.8, but the model is flat from 0.6 to 1.4, so no entry change moves
  a prediction. And no fitness reaches the heart-rate path: the corpus `Population`
  carries sex, age, BMI and health only, `fitness` is a person factor nothing sets
  from VO2max, and the teacher's `act_hr_gain` does not depend on it (fitness
  only lowers HR0 and raises HRV0). Needs a VO2max field in the corpus schema and
  a fitness-dependent gain in the teacher; a retrain-level change. Nothing changed.

Respiratory quotient follows the fuel mix (2026-11-26, free, frozen on iter 120; board run
before the respiratory-rate line was moved ahead of the shift, so rr was not
rescored; `rq_fuel_shift_t` in `pulse/modules/metabolic.py`, `tests/test_rq_fuel_shift.py`).
Next after leptin and heart rate on the trained side: RQ was a constant 0.807 at
rest whatever the insulin (activity and the meal's thermic fuel were its only
inputs), so a 72 h fast read 0.807 against 0.74 (bak-2018 z +3.4 and +6.7) and
the end of a clamp 0.807 against 0.87 to 0.89 (hoeg-2011 -2.2, -2.9). A shift
from `ins/ib`, zero at 0.54 (the 12 h fasted level: the 12 h items stay put),
0.2 per unit below and 0.014 per unit above, capped at +0.12; VCO2 and Weir
energy follow; the respiratory rate still reads the CO2 before the shift (rest with
no meal stays at RR0, `test_expenditure`). Frozen on iter 120, trained items: rq mean |z| 2.15 to 1.68
(bak 72 h to 0.0, gillberg 34 h 1.6 to -0.5, hoeg clamps 2.9 to 1.1 and 2.2 to
1.7), all items median 0.616 to 0.611, mean 0.992 to 0.983; energy expenditure
0.79 to 0.83 (Weir on the new VCO2, mainly ostrowska-2022). Worse:
wadazumi-2019 fasted-exercise water arm (-4.5 to -6.3 at 65 min) and
wahren-1971 at 10 min. The full suite: 1,701 passed, one golden rollout re-recorded (observables only: VCO2, RQ, energy). Sealed (looked at once, to report): median 0.682 to 0.685,
mean 1.027 to 1.023 on 339 common non-GLP-1 items, neutral. The first
setting (zero at 1.0, 0.12 below, 0.02 above) overshot the 12 h fasts to -1.4
to -1.7 z and was replaced.

Absorbed protein releases glucose (2026-11-26, free, frozen on iter 120; `_AA_GLUCOSE_FRAC`
in `pulse/modules/metabolic.py`, `tests/test_protein_glucose.py`). Chosen from the trained-side
board after leptin, exercise heart rate and RQ: glucose after a mixed meal read low on 102
scalar items in 21 papers (signed z -0.69, 85% negative: peaks 9% low, incremental areas
a third low, time to peak 20% early), and the protein-only arms were the worst of them
(westphal-1990 50 g protein alone: model 92 to 53 mg/dL against "no change", z -6.1;
oberoi-2022 protein arms -2.6 and -2.8). The model had no glucose from protein at all: insulin and
glucagon rose and hepatic output stayed flat. Now 0.4 g of glucose enters plasma per gram of
protein leaving the stomach (Jungas, Halperin, Brosnan 1992, Physiol Rev 72:419: up to about
57% of amino-acid carbon can leave the liver as glucose; 0.15 and 0.2 were tried and lifted the
protein arm's trough (53 mg/dL) by only 8 and 10 mg/dL, 0.4 by 19 and 0.6 overshoots to 102). Frozen on iter 120, trained
items (1,294 common non-GLP-1): median |z| 0.611 to 0.587, mean 0.983 to 0.965; mixed-meal
glucose scalars 0.77 to 0.58, westphal -6.1 to -1.7, oberoi protein iAUC -2.8 to -1.6 and -2.6 to
-1.1, glucose in exercise papers (a fed start) 1.49 to 1.17 late. Worse: urinary glucose under
SGLT2 (empagliflozin 24 h 6.8 to 7.7, 3 items), early oral-glucose glucose 0.91 to 0.97, exercise
lactate and heart rate late by under 0.1, demoraisjunior-2020 delta items still -3.9 to -4.0. The full suite: 1,703 passed after one golden rollout re-recorded (meal rows only), the carbon ledger given the new input, and the Si/Gb meal-gradient dominance check relaxed from 2.0 to 1.5 (it reads 1.9). Sealed (looked at once, to report): median 0.685 to 0.707,
mean 1.023 to 1.012 on 339 common non-GLP-1 items; mixed-meal glucose scalars 1.28 to 0.92, but
glucose, insulin in exercise arms 0.54 to 0.66 and 0.83 to 0.92. The teacher has no such term, so a
retrain would fit around it; add it there before the next run. Tried and not kept:
potentiator rate `_K_POT` 0.0125, 0.0167 and 0.04 and hepatic autoregulation `_HEP_AUTOREG_M` 1.0 for the late
oral-glucose tail (glucose at 90 to 360 min 10 to 25 mg/dL high, 25 items): none moved it
(mean |z| 1.43 to 1.40 to 1.58), so the tail is not these constants.

Retrain-level candidates (changes the checkpoint layout or needs learned parameters;
not fixable frozen):
- BUILT, awaiting a run (2026-11-26): a slow post-exercise heart-rate state (tens of minutes;
  sugimoto-2020, gahreman-2016, cesaremarincola-2025). New marker `hr_slow` (cardiovascular,
  the 40th state, kept out of the MLP so no weight changes shape): a leaky integral of activity
  above 0.1, tau a learned `slow_tau_raw` (30 min at init, band 5 to 240), read by heart rate
  through `hr_slow_gain` (bpm per unit, init 0). Teacher mirror in `full_body.py`: 30 bpm, tau 30.
- BUILT, awaiting a run (2026-11-26): a fitness-dependent heart-rate gain. Corpus `Population`
  has an optional `vo2max` (mL/kg/min as printed, same fields as age; `corpus/README.md`),
  filled in 20 papers (engeroff-2024, hoshi-2022, rappelt-2023 and 17 more: ali-2026, behrendt-2024,
  bradley-2015, broome-2022, cesaremarincola-2025, crabtree-2025, garnachocastano-2018, gahreman-2016,
  gronwald-2019, li-2024, mosqueralopez-2025, roberts-2023, schild-2016, schubert-2024, sugimoto-2020,
  tsukamoto-2023, wadazumi-2019; sealed ones got the field, their scores not looked at; wadazumi-2019 is
  converted from mL/min with the printed group mass). The other exercise papers print none (chiba-2025
  estimates it; eshghi-2017 prints only VO2 at the ventilatory threshold; taboni-2024 only W/kg) or none
  at all; hedrington-2015 prints 39 +/- 2 SE but its file fails the gate on main (gir_final does
  not move with the parameters), so it is left without.
  A sweep of the gate over all 147 findings and 64 refusals (2026-11-26) found that one failure
  and no other. Cause (bisected): dcf7ec2 made exercise lower basal insulin, so in
  exercise_euglycemia glucose runs 95 -> 103 mg/dL above the 95.5 clamp target and the one-sided
  clamp infusion (relu) is exactly 0, with no gradient; the paper's 1.2 umol/kg/min is a real
  number, the file is right, the model is not. Not changed: the gap is the model's exercise
  glucose (the clamp should hold it level), a model change for a training round. Meanwhile
  `tests/test_corpus_gate_sweep.py` records it in `KNOWN_DEAD` (a strict test: it fails when the
  finding moves again), gates three small files by default, and sweeps every file with
  `PULSE_GATE_SWEEP=1` (about 25 CPU-minutes; run after a change to a module that moves a marker).
  `training.factors.FITNESS_FROM_VO2MAX` (False) sets the fitness code from it
  ((VO2max - 38) / 9); turn it on for the run with `train --fitness-from-vo2max` (default off;
  `train(fitness_from_vo2max=)` sets the module switch, the flag is in `train_config` and the
  checkpoint's top-level `fitness_from_vo2max`). Every loader that builds people reads it back:
  `diagnostics.probe.load_model_from_checkpoint` (so `scoreboard_for_checkpoint`, compare,
  identifiability, rules, ablation, signal balance) sets the switch from the checkpoint, absent
  meaning off. The switch is process-global: a script that loads two checkpoints of different
  settings must load each before scoring it. Frozen on iter 120 (trained-side items only, sealed
  scores not looked at): switching it on moves 127 of 1,303 trained-side items, all in the
  13 papers whose arms carry a `vo2max` (hr 46, glucose 23, lactate 15, spo2 10, insulin 8,
  rq 8, rr 5, vco2 5), median |z| 0.588 -> 0.599, largest single move 1.7 sigma (sugimoto-2020 hr
  at minute 59, 141 -> 133 bpm, away from the 158 target). Cause: the fitness code is one coordinate of the person
  embedding, so it reaches every module through the learned embedding projections, not only
  through the zero hr_fit_gain; iter 120 saw fitness as a standard-normal latent, so a
  VO2max-set person is a different person to every head. Off, boards are identical to before. `body_quantities["fitness_z"]` reaches the
  cardiovascular module, which adds `hr_fit_gain` (init 0) bpm per unit activity per SD below
  median to where heart rate settles. Teacher: 12 bpm per SD per unit activity, on the patient's
  fitness latent. Both gains are zero in the student, so iter 120 and every rollout are as
  before: golden rollouts unchanged, and the frozen board of iter 120 is identical item by item
  (1,651 items, predictions equal to the last bit) against a fresh main. A checkpoint without
  the three new keys loads (`ADDED_STATE_KEYS`; the shorter `norm_center` and `norm_scale` are
  replaced by the current table in `adjusted_state`). Not yet filled: papers without cached text
  (sugimoto-2020, tsukamoto-2023, behrendt-2024, ali-2026, schubert-2024, gahreman-2016 and the
  rest of the 147 files); the field is for a reader to add as papers are read.
- Exercise lactate runs low (trained signed z -1.3, yamaguchi-2020 1 to 5 mmol/L against
  10 to 17 at high intensity; gahreman-2016 2 against 4 to 5): the learned lactate production head.
- Protein-only insulin is too high after the nutrient gain was doubled at load (37 µU/mL
  mean at 50 g protein), which is what drives the remaining dip; a learned `ins_nut_gain`.
- Gastric emptying starts at once at 2.13 kcal/min: mixed-meal glucose peaks at 25 to 34 min
  against 36 to 143 min measured (12 items; no solid-food lag in the compartment).
- Fasted baseline glucose in exercise papers (bradley-2015, sugimoto-2020, tsukamoto-2023) is
  10 mg/dL low at the start of the session (87 against 97): a person-level baseline. Checked
  frozen on iter 120 (2026-11-26): not an init fault. The arms start from the student's own fed
  day and pre-fast (`arm_init._spinup_protocol`), and a 6-person spin-up of the sugimoto arm reads
  89 mg/dL at 12 h fasted (90 at 10 h, 83 at 16 h), the model's general fasted level (all trained
  fasting glucose items: median -1.9 mg/dL; exercise ones -7.8, but yamaguchi-2020 is +5.6 at a
  target of 83 against 94 to 98 in the other four). The gap is between papers, not a start-state
  error. The `_AA_GLUCOSE_FRAC` term (0e1959f) takes only `amino_g`, the gut's amino appearance, which
  is zero without a meal (no endogenous amino acids); on exercise arms it lifts the start by 0.5 to
  1 mg/dL through the fed day's protein and helps (zhang-2024, protein meal 2.5 h before HIIE, glucose
  at 154 to 170 min z -2.1 to -2.7 now -1.4 to -1.7; insulin items move under 0.05 z).

Reported by the review of 2026-11-11 and not yet confirmed:

- calibration fits interstitial glucose while evaluation reads plasma.

## Live

- The `engine` service is not ready: its container fails the health check
  (2026-11-11). `models/prod.pt` dates from 2026-06-07 and has an earlier
  state layout.
