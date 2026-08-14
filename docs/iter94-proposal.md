# iter-94 proposal — fix the instrument, then make slow physiology learnable

Written 2026-08-14, after landing iter-93. Every number below was measured against the
committed tree at `d9c64f7` and the shipped artifact
`gs://grovina-pulse-data/training/jobs/iter93/model.pt`. Reproduction scripts are named
per section.

## Summary

iter-93 fixed the teacher's fasted state and the gate improved to a new best
(`overall_weighted_mape` 0.0609). Both halves of that sentence are true and neither means
what it looks like:

1. **The student did not inherit the fix.** iter-93's own primary acceptance criterion —
   the 24 h fast cascade at the prior-mean embedding — fails **0 of 4** bands. Liver
   glycogen is frozen; BHB moves 2 % of the teacher's amount; insulin moves the *wrong
   way*.
2. **The gate could not have told us.** On the gate's 24 "real" episodes, carrying the
   last calibration reading forward scores `glucose 0.0098, hr 0.0141, sbp 0.0054,
   dbp 0.0058, temp 0.0004`. The thresholds are `0.20 / 0.15 / 0.12 / 0.12 / 0.02` —
   **11× to 50× looser than a constant already achieves.** The gate cannot distinguish a
   physiology simulator from a flat line.

So iter-94 has to do two things in one pass: make the ruler capable of reading the
answer, and fix the mechanism that stopped iter-93's physics from reaching the student.

---

## Part 0 — the evidence

### 0.1 The student fails iter-93's primary criterion (`iter93_student_fast.py`)

24 h fast, no meals, prior-mean embedding, same protocol as
`scripts/iter93_teacher_validate.py` section A.

| marker | band | teacher @24 h | **student @24 h** | verdict |
|---|---|---|---|---|
| glucose | 70–85 mg/dL | 78.23 | **90.32** | FAIL |
| insulin | 2.5–6.5 µU/mL | 3.80 | **11.02** | FAIL |
| FFA | 0.8–1.4 mmol/L | 0.85 | **0.56** | FAIL |
| BHB | 0.8–2.2 mmol/L | 1.32 | **0.13** | FAIL |

Fraction of the teacher's 24 h movement the student reproduces:

| marker | student Δ | teacher Δ | ratio |
|---|---|---|---|
| glucose | −4.69 | −16.82 | 0.28 |
| insulin | **+1.02** | −6.20 | **−0.17 (wrong sign)** |
| FFA | +0.06 | +0.35 | 0.18 |
| BHB | +0.03 | +1.21 | 0.02 |
| liver_glycogen | +0.13 | −58.23 | **−0.00** |

Liver glycogen is the state that *carries* iter-93's mechanism (fix A1 keyed the defended
glucose fall to linear liver-glycogen depletion). It is frozen, so the cascade has nothing
to stand on. The 28 % of the glucose decline the student does show is coming from
somewhere else.

### 0.2 Why it did not transfer — slow drift is ~3 orders of magnitude under-weighted (`window_drift.py`)

`ColdModelDistillationSignal` runs in `mode="anchored"`: the protocol is tiled into
60-minute windows, the model is **reset to the teacher's truth at each window start**, and
a Huber (δ=1) is applied to the `NORM_SCALE`-normalized *level* inside the window.

The loss a marker can generate is therefore bounded by how far it moves **within one
60-minute window**. Worst-case normalized within-window excursion, and the Huber a
perfectly flat predictor would pay:

| marker | fast_24h | standard_3meal | flat-predictor Huber, fast | …meal | ratio |
|---|---|---|---|---|---|
| glucose | 0.033 | 2.005 | 0.00055 | 1.505 | **1 : 2700** |
| insulin | 0.051 | 4.417 | 0.00129 | 3.917 | **1 : 3000** |
| ffa | 0.098 | 1.776 | 0.00476 | 1.276 | 1 : 270 |
| liver_glycogen | 0.044 | 0.081 | 0.00098 | 0.00328 | 1 : 3 |
| glucagon | 0.026 | 0.989 | 0.00034 | 0.489 | 1 : 1500 |

**A model that ignores fasting physiology entirely pays roughly a thousandth of what a
model that is 2 % wrong about postprandial insulin pays.** The residual is normalized by
each marker's *global* `NORM_SCALE` — a scale sized to its full physiological range, which
is dominated by meal and exercise excursions. Slow states move a tiny fraction of that per
window, so their gradient is attenuated by the ratio above for the same *relative* error.

Note the contrast that makes this diagnosis load-bearing rather than a story: the same
table gives `hr 1.20` and `temp 0.70` normalized per window — large. And those are exactly
the markers where the student **does** faithfully reproduce the teacher (§0.4). The
markers with visible within-window signal transferred; the ones without did not.

### 0.3 …and the knowledge signals only run for 10 of 60 epochs

From `pulse/train.py:361`, `enable_at = phase_boundary if phased else 0`. With
`phase_schedule = {'phase1_epochs': 50, 'phase2_epochs': 10}`, **every** knowledge signal —
coupling priors, verifier, landmark, cohort statistics, dose-response, meal-response,
carb mass balance, cold distillation, physiology rules — is off for epochs 0–49 and runs
only in epochs 50–59. Confirmed in the iter-93 logs (`sig=cold_model_distillation
dt=0.0s` for every phase-1 epoch), and phase-2 LR decays across those 10 epochs from
`3e-4` to `1e-5`. The literature-grounded half of the training doctrine gets ten epochs at
a collapsing learning rate.

### 0.4 The one place transfer *did* work — and what it says about the circadian family (`iter94_diag.py`)

| quantity | literature | teacher | student | teacher/lit | student/teacher |
|---|---|---|---|---|---|
| core temp peak-to-trough | 0.5 °C | 1.04 | 1.10 | 2.07 | 1.06 |
| sleep HR dip | −8 bpm | −22.21 | −23.63 | 2.78 | 1.06 |
| sleep HRV rise | +12 ms | +34.12 | +36.22 | 2.84 | 1.06 |
| sleep SBP dip | −10 mmHg | −18.37 | −18.76 | 1.84 | 1.02 |

The student tracks the teacher to within 6 % on all four. The circadian overshoot is a
**pure teacher defect that the student faithfully inherits** — so a teacher constant change
will transfer, unlike iter-93's fasted work.

The mechanism that let it drift is in the repo: `TEMP_CIRCADIAN_AMPLITUDE`
(`pulse/knowledge/physiology_rules.py:1555`) uses
`hinge_circadian_amplitude(..., min_amplitude=0.5)`, which returns
`relu(0.5 − amplitude)` — a **one-sided floor**. Any amplitude above 0.5 °C is free. Same
shape in the verifier: `sbp_dip_min = 2.0`. This is the PRD's under-enforcement failure
mode with a ceiling missing, not a tuning miss.

### 0.5 Muscle glycogen is *structurally* frozen — a different failure from the rest

Student range ÷ teacher range over a full day:

| protocol | liver_glycogen | muscle_glycogen |
|---|---|---|
| fed day | 0.783 | **0.001** |
| fast day | 0.312 | **0.004** |
| exercise day | 0.782 | **0.000** |

Liver glycogen's anabolic side works (0.78 on a fed day) and only its catabolic side is
starved — a supervision problem, fixable by §2. Muscle is different. From
`pulse/modules/metabolic.py:441`, `prod_scale = cons_scale × typical`, so for muscle
`3.3e-5 × 400 = 0.0132 /min` and `rate = 0.0132 · (prod − cons)`. The teacher depletes
68 units in a 45-minute bout; that needs `prod − cons ≈ 114` out of a softplus head. It is
not reachable, so no amount of gradient will fix it. The 3-week τ is the right constant for
**repletion at rest** and the wrong constant for **depletion during exercise**; one scale is
being asked to carry both.

### 0.6 The gate cannot read any of this

The benchmark dataset used by iter-93 (`gs://grovina-pulse-data/benchmarks/
benchmark.dataset.generated.json`, generated 2026-04-13, still the file in GCS today):

- 24 episodes, **all** `duration_min = 720`, **all** `start_time_minutes = 214` — one phase.
- Calibration 0–510 min (16 check-ins); evaluation 510–690 min, 7 timestamps × 5 markers.
- **Zero meals in the eval window.** All 72 meals across all episodes fall at t = 90–330,
  entirely inside the calibration window. The scored window is always quiescent.
- Mean within-episode standard deviation of the ground truth:
  `glucose 1.03 mg/dL, hr 1.09 bpm, sbp 0.80 mmHg, dbp 0.56 mmHg, temp 0.035 °C`.
- **`temp` takes exactly two distinct values — 36.6 and 36.7 — across all 168 points.**

Hence the persistence baseline vs the thresholds:

| marker | persistence MAPE | gate threshold | threshold ÷ persistence |
|---|---|---|---|
| glucose | 0.0098 | 0.20 | **20×** |
| hr | 0.0141 | 0.15 | **11×** |
| sbp | 0.0054 | 0.12 | **22×** |
| dbp | 0.0058 | 0.12 | **21×** |
| temp | 0.0004 | 0.02 | **50×** |

`skill_vs_persistence = 1 − mape/persistence_mape`, so iter-93's reported row
(`glucose −7.18, temp −50.05`) states that the model is **8.2× worse than persistence on
glucose and 51× worse on temp** — while passing every threshold with an order of magnitude
to spare. This also resolves a standing contradiction in the notes: the iter-77/87 reading
("`skill_vs_persistence` is a structural artifact, do not chase it") is correct, and
iter-93's plan to justify a circadian iteration by `temp = −50.05` was reading an artifact
of a ruler whose temperature truth has two distinct values.

Two further consequences worth stating plainly:

- The `source="real"` label is a default (`pulse/benchmark.py:160,199`), not an assertion.
  These 24 episodes come from `pulse.check_ins`; the **14 CGM + Oura overnight episodes
  built during iter-92 were never added to the gate** — GCS still holds the April file.
  iter-93's "confirmed on real data" conclusion rests on an offline CGM analysis, which is
  legitimate evidence about physiology, but the reported `(real)` numbers do not come from
  it.
- Only 2 of the 26 scored episodes (`benchmark-cohort-meal-postprandial`,
  `benchmark-cohort-sleep-48h-adequate`, both `source="teacher"`) put a meal response
  inside the eval window. Every postprandial iteration since the ruler was built has been
  graded almost entirely on quiescent windows.

---

## Part A — make the ruler honest (no training cost)

**A1. Stop mislabelling.** Tag the legacy 24 episodes `"benchmark_source": "legacy_static"`
in the dataset meta. Their number stays comparable across iterations; it stops being
called "real".

**A2. Add the real CGM + Oura episodes as `source="cgm_real"`.** Rebuilt today from
`/Users/grovina/Documents/health` with the committed
`scripts/ingest_real_data.py` — 14 episodes, and materially more dynamic than the legacy
set:

| | legacy_static | cgm_real |
|---|---|---|
| mean within-episode glucose range | 2.90 mg/dL | **13.24 mg/dL** |
| mean within-episode hr range | 3.00 bpm | 6.21 bpm |
| persistence MAPE, glucose | 0.0098 | **0.0624** |
| persistence MAPE, hr | 0.0141 | 0.0372 |

These carry glucose and hr only (no temp/sbp/dbp), which is honest — those are the markers
we actually have. Per-source reporting already exists, so nothing breaks.

**A3. Add teacher episodes whose eval window contains a meal.** Today exactly two do. The
generator already exists (`pulse/knowledge/benchmark_extras.py`); widen it to a handful of
protocols spanning meal size and timing. Without this, postprandial work is unscoreable.

**A4. Gate on skill, not only on absolute MAPE.** Add a per-source
`skill_vs_persistence_min` threshold on the dynamic sources and promote it from report
line to gate metric. An absolute MAPE threshold on a quiescent window is not a test. Keep
the existing absolute thresholds for continuity.

**A5. Establish the honest baseline before training anything.** Run `--benchmark-only`
with the existing iter-93 checkpoint against the new ruler. One cheap cloud job, no
training. This is what iter-94's model work gets compared against.

> **Expect the honest gate to fail, and do not treat that as a regression.** A ruler that a
> constant passes with 20× margin was never reporting the model's quality. iter-94's
> acceptance criterion is *not* `gate.passed`.

## Part B — make slow physiology learnable (one training run)

**B1. Normalize the distillation residual by the teacher's own within-window dynamic
range, floored** — not by the global `NORM_SCALE`. This is the direct fix for §0.2 and
costs no extra compute: it changes what the existing rollout is compared against, so a
relative error of 50 % on liver glycogen finally costs about what a relative error of 50 %
on insulin costs. Floor the divisor so genuinely-flat regions do not amplify noise.

**B2. Add long-horizon anchors for the slow protocols.** Keep the 60-minute windows for
meal kinetics; add a small number of `W = 360–720` windows on `fast_24h`,
`high_fat_meal` and the circadian protocols so the *integrated* arc is scored and not only
the local slope. Budget: iter-93's distillation already cost up to 1713 s of a 3366 s
epoch, so add samples sparingly and measure.

**B3. Give the knowledge signals more than ten epochs.** Move `enable_at` earlier (target
~25 of 60) or lengthen phase 2. This is the cheapest structural change available and it
conditions everything in Part B and C. Watch for the phase-1 instability the phased
schedule was introduced to avoid — ramp rather than step if needed.

**B4. Unfreeze muscle glycogen structurally.** Separate the exercise-depletion flux scale
from the resting-repletion τ, so an activity-gated breakdown term can move the pool on an
hours timescale while rest-preservation still holds. Do not re-falsify the −150 target.
Blast radius is small — check whether muscle glycogen feeds glucose/lactate before
committing.

## Part C — the circadian family (fold in; re-scoped, not dropped)

**C1. Make the temp amplitude constraint two-sided.** `relu(0.5 − amplitude)` becomes a
band. Same for `sbp_dip_min`. This is the actual bug: nothing in training has ever
penalized an overshoot.

**C2. Bring the teacher's amplitudes to literature** — temp peak-to-trough 1.04 → ~0.6 °C,
sleep HR dip −22 → −10 bpm, sleep HRV rise +34 → +15 ms. §0.4 shows the student inherits
these at ~1.06×, so the change will transfer.

**C3. Leave SBP alone.** −18.4 mmHg on a 120 mmHg baseline is a 15 % nocturnal dip, inside
the normal dipper range (O'Brien 1988; Staessen 1997 — 10–20 %). The repo's own
`sbp_sleep_dip` target of −10 ± 5 is at the shallow end of that range; the teacher is not
contradicting the literature here, and "fixing" it would be fitting the encoded target
rather than the evidence.

**C4. Justify this on the literature contradiction alone.** Do *not* cite
`temp skill = −50.05`. That said, C2 is expected to help `temp_mape` on its own merits: at
0.019 the model is ~0.70 °C off against a truth that moves 0.10 °C in-window, and its own
1.10 °C spurious swing is most of that error. `temp` is the single marker whose threshold
(0.02) sits near the model's actual ability, so this is the one place a circadian fix is
also gate-relevant.

## What not to do

- Do not chase `skill_vs_persistence` on `legacy_static`. It is a ratio against a ~0.001
  floor and no physiology work will move it.
- Do not re-open ghrelin suppression or the carb kernel — closed in iter-92.
- Do not re-propose the arterial baroreflex — rejected with measurements in `d13b385`.
- Do not treat `extended_fast_liver_glycogen_overnight` as a student failure. The
  **teacher** realizes −11.97 g against its own −60 ± 25 target on those arms (the student
  tracks it at −9.75). Either the target or the arms are wrong; settle that against the
  literature before pointing the loss at it.

## Acceptance criteria

Ordered by what actually decides whether iter-94 worked.

1. **Primary.** The 24 h fast cascade at the prior-mean embedding moves toward the
   teacher: `liver_glycogen` ratio ≥ 0.5 (from −0.00), `bhb` ≥ 0.5 (from 0.02), insulin
   moving in the **correct direction** (from −0.17). Re-run `iter93_student_fast.py`.
2. **Secondary.** On `cgm_real`, `skill_vs_persistence` for glucose improves against the
   Part-A baseline from A5. This is the first time that number will mean anything.
3. **Circadian.** Student temp peak-to-trough ≤ 0.7 °C, sleep HR dip within
   −8 ± 4 bpm, HRV rise within 12 ± 7.5 ms.
4. **Guard.** iter-92's meal kinetics hold: glucose peak 45–60 min, ghrelin nadir
   −30…−50 % at 60–90 min.
5. **Guard.** `legacy_static` absolute MAPEs do not regress materially — it is a weak
   instrument, but a large move there still means something broke.

## Sequencing

Part A first and entirely, including the A5 baseline benchmark, **before** dispatching any
training. Part A costs one cheap benchmark-only job; Part B+C is one ~20 h run, the same
price as any iteration — but the first one in a while whose result can be read.
