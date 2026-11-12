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
| Cohort specs | 304 authored, 33 of them curves; 275 training points and 129 held-out points |
| Rules | 62 training, 0 held out |
| Coupling priors | 32 |
| Papers | 94, of which 8 are sealed |
| Held-out items | 5 |
| Corpus files | 26 papers as findings, 9 as refusals, 0 queued to be read |
| Queue | 23 (15 integrated, 7 blocked, 1 backlog, 0 ready) in the older Python queue |

## The model

One network of nine modules over 39 states, stepped by explicit Euler at one
minute for every state. A person is a 32-number embedding; the median person
is the zero embedding. The checkpoints of iters 113–115 have 35,551
parameters (`--hidden-dim=48`).

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
recipe keeps `item`; no run has used the others.

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
- A paper is read by one agent and checked against the paper, number by
  number, by a second that sees only the file and the paper. Only then is it
  committed. The table counts the files. Given the paper's full text
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
  paper itself.
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
  iter 114.
- The cortisol gain on glucose production is centred on the population's
  12 µg/dL, not the person's own basal cortisol, which the stress module
  learns (7.3 to 19.6 across 512 draws). Basal production is off by up to 9%
  for a person away from the median.

Reported by the review of 2026-11-11 and not yet confirmed:

- calibration fits interstitial glucose while evaluation reads plasma.

## Live

- The `engine` service is not ready: its container fails the health check
  (2026-11-11). `models/prod.pt` dates from 2026-06-07 and has an earlier
  state layout.
