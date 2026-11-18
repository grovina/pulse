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
| Parameters | 37,287 at the default widths |
| Published numbers | 43 in 10 tables, each learned from its published value |
| Corrections | 3 |
| Cohort specs | 1062 authored, 137 of them curves; 1135 training points and 316 held-out points |
| Rules | 62 training, 0 held out |
| Coupling priors | 32 |
| Papers | 189, of which 29 are sealed |
| Held-out items | 5 |
| Corpus files | 121 papers as findings, 46 as refusals, 8 queued to be read |
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
  meal. The islet's amino-acid gain (`metabolic.islet.log_amino_ins_gain`)
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
- The meal reader again gives the better sealed glucose (0.64 against 0.75)
  and is better on the 36 sealed items new to this corpus (median |z| 1.17
  against 1.35); otherwise the two runs are level.
- Spend: about $6.80 and $9 at Cloud Run list rates (4 vCPU, 16 GiB).

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
  hits are all screened. Seven topics are worked (insulin after a meal;
  meals equal in grams, different in kind; the oral glucose tolerance test;
  fasting; intravenous glucose and clamps; a bout of exercise; sleep loss
  and timing), and 3,739 of 4,056 hits found so far are screened. Screening runs on Haiku.
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
  training flag. That is 35 of the 275 training points, all from the Python
  registries; a file always states its n.
- `pytest` from the repository root fails while collecting
  `scripts/iter89_recovery_test.py`. Run `pytest tests/`.
- There is no CI. The suite is run by hand:
  `uv run --group dev pytest tests/ -n auto`.
- Three tests fail (run file by file on 2026-11-11, on a 2-CPU, 4 GB box):
  `test_dg_is_zero_at_the_decoded_setpoint` in
  `test_a1_log_glucose_setpoint.py` (glucose moves 0.28 mg/dL a minute at a
  set point of 60.6), and `test_kernel_decays_monotonically_after_its_peak`
  and `test_appearance_is_at_the_cap_from_the_first_minute` in
  `test_iter97_student_gut.py`. Four files could not be run there:
  `test_cohort_ablation.py` and `test_identifiability.py` ran out of memory,
  `test_iter81_stability.py` and `test_training_signals.py` passed 30 minutes.
  The other 89 files pass, in 49 minutes one after another.

- 693 parameters of the gut kernel (the mixture logits of its last layer)
  receive no gradient from any loss.
- The stomach's split correction acts on the three gastric pools and never
  reaches absorption: no module reads a gastric state. It is exactly zero in
  iter 114. The meal reader's speeds do not reach those pools either.
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
