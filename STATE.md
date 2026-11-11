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
30 on trajectories alone, then 28 with the literature). The job is 8 vCPU, 32 GiB, 72 h, no
retries.

A step now spins up all of its people in one rollout and rolls its items in
groups of similar horizon (`pulse/training/evidence_step.py`). A run writes a
board at the end of each phase, and `--deadline-hours` stops it cleanly
before the job's timeout. No run has used any of this yet: the recipe is
unchanged, and the numbers below are from before.

An arm's pre-fast now begins with a meal and its nights are slept
(`pulse/training/arm_init.py`); before, the last meal was the fed day's 19:00
dinner whatever the pre-fast, and the pre-fast was spent awake. No checkpoint
has trained from these start states. Measured on iter 114's median person:
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

Measured on a 2-CPU box on 2026-11-11 (`python -m pulse.diagnostics
step-time`, untrained model, 4 items and 1 rule, arms capped at 300 minutes,
CPU seconds):

- Item by item, as iter 115 ran: 47.5 s. Batched, on this commit: 9.9 s
  (spin-up 3.4 s, rollout and scoring 2.3 s, backward 4.2 s).
- A simulated minute costs 979 tensor operations, down from 1,167. With a
  different meal list for every member of a batch it cost 8,009.
- An uncapped step and a whole epoch have not been timed since the change;
  this box cannot hold a day-long rollout with its gradient.

## Runs

| Execution | Iter | Started | Ended |
|---|---|---|---|
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

Iter 114 is the last checkpoint. It was scored on this commit on 2026-11-11,
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
