# iter-94 spec — as built

Proposal and evidence: `docs/iter94-proposal.md`. This file records what was actually
implemented, what the measurements said, and — most importantly — what is still open.
Every number below was measured on this tree; reproduction commands are given.

## What shipped

### Part A — the ruler

| change | where |
|---|---|
| legacy 24 episodes tagged `source="legacy_static"` (was defaulting to `"real"`) | `scripts/build_benchmark_ruler.py` |
| 14 real CGM + Oura overnight episodes added as `source="cgm_real"` | same; built from `~/Documents/health` via the committed `scripts/ingest_real_data.py` |
| 8 teacher episodes whose **scored window contains a meal**, `source="teacher_dynamic"` | `pulse/knowledge/benchmark_extras.py` |
| gate on `skill_vs_persistence` per source | `pulse/train.py`, `pulse/benchmark.thresholds.json` |
| per-episode ids for the CGM nights (were all `gabriel`) | `scripts/ingest_real_data.py` |
| skill printout covers every source, not just `real` | `pulse/train.py` |

Ruler composition is now 24 `legacy_static` + 14 `cgm_real` (on disk) + 2 `teacher` +
8 `teacher_dynamic` (generated in-process) = **48 episodes**.

Eight dynamic arms rather than four because per-marker statistics are **per-episode**
MAPEs (one per episode, not one per eval point), so the arm count *is* the sample size
behind the new skill threshold.

Noted while wiring the gate: **`min_samples_per_marker` in the thresholds file is dead**
— nothing in the codebase reads it. Left in place (removing it is unrelated churn) but
it should not be mistaken for a guard.

Dynamic range of the scored window, ground truth (mean within-episode):

| source | glucose sd | glucose range | meals in window |
|---|---|---|---|
| legacy_static | 1.03 mg/dL | 2.9 mg/dL | **0 of 72** |
| cgm_real | 4.32 mg/dL | 13.2 mg/dL | 0 (overnight by design) |
| teacher_dynamic | 9.0–23.8 mg/dL | 25.6–74.0 mg/dL | 1 each |

**`teacher_dynamic` episodes are excluded from the distillation pool**
(`benchmark_extras.distillation_pool_episodes`). The cohort episodes double as
training protocols via `--cold-distill-pool=both`, so putting the new gate episodes in
the pool would have meant training on the test — the 2 pre-existing `teacher` episodes
already carry that circularity knowingly.

### Part B — making slow physiology learnable

**B1 — local-scale normalization** (`cold_model_distillation_signal._level_terms`,
`--cold-distill-anchor-local-scale-floor=0.05`). The level residual is normalized by the
reference's own peak-to-peak range within each window, floored at 5 % of `NORM_SCALE`,
instead of by `NORM_SCALE`. Verified: the geometric-mean visibility of fasting vs meal
physiology to a flat predictor goes **0.023 → 0.361** (43x under-weighted → 2.8x).

**B2 — long anchor windows** (`--cold-distill-anchor-long-window=480
--cold-distill-anchor-long-samples=2`, short windows 60/6). A slow arc is now scored as
an arc, not as ~60 local slopes a flat predictor gets nearly right. Costs ~2.75x the
distillation rollout budget; measured in a smoke run and folded into the epoch plan.

**B3 — knowledge signals get 25 epochs instead of 10.** `enable_at` is the phase
boundary, so `--phase1-epochs=30 --phase2-epochs=25`. In iter 93 every
literature-grounded signal ran only in epochs 50–59 of 60, at an LR decaying 3e-4 → 1e-5.

**B4 — glycogen is a storage pool, not a species in equilibrium**
(`pulse/modules/metabolic.py`). This is the iteration's biggest finding and it is
**not** what the iter-93 notes predicted.

> The mass-action assembly is `prod·prod_scale − cons·cons_scale·norm_state`, and
> `norm_state` is **zero at the pool's typical value**. `GlycogenFluxHead` emits
> `prod = synthesis ≥ 0`, so at typical the entire breakdown term vanishes.
> **`typical` was an absorbing floor.** Measured on the iter-93 artifact, both pools hit
> exactly `min − typical = 0.0000` in *every* protocol — fasted, fed, and a 2 h hard
> bout — while the teacher falls 58 g (liver) and 206 g (muscle) below start on the
> same input.

So liver glycogen was never a supervision problem and muscle glycogen was never a
timescale problem; iters 55–57 and the iter-93 notes both read it wrong, and **B1 alone
would not have fixed either**. Both pools now use an explicit flux balance in raw g/min,
following the precedent glucose already sets in the same `forward`:

```
d(pool)/dt = flux_scale · ( synth·headroom − breakdown·fullness )
  headroom = relu(1 − pool/(1.3·typical))   synthesis stops at a full store
  fullness = clamp(pool/typical, 0, 1)      breakdown has full authority at a normal
                                            store, fades to 0 as the pool empties
```

The head's learned gates and outputs are reused unchanged; only the rate assembly
changes. Verified: the floor is gone in every protocol, and a fresh-init model stays
finite and bounded over 24 h with a hard bout.

### Part C — circadian amplitudes

Teacher constants brought to the literature the repo already cites, and the rule that
let them drift made two-sided.

| statistic | before | after | target |
|---|---|---|---|
| `hrv_sleep_rise` | +45.2 (z=+4.43) | **+19.4 (z=+0.99)** | 12 ± 7.5 |
| `sleep_hr_dip` | −19.5 (z=−2.87) | **−11.9 (z=−0.99)** | −8 ± 4 |
| `temp_circadian_nadir` | −1.02 (z=−1.75) | **−0.61 (z=−0.37)** | −0.5 ± 0.3 |
| `sbp_sleep_dip` | −19.3 (z=−1.87) | −19.3 (z=**−0.87**) | −10 → **−15** ± 5 |
| `dbp_sleep_dip` | −11.1 (z=−1.03) | −11.1 (z=**−0.41**) | −7 → **−9.5** ± 4 |

Teacher-vs-literature contradictions: **6/31 → 4/31**, with every metabolic and meal
statistic byte-identical. Daily core-temp swing 1.04 °C → **0.61 °C**.

`TEMP_CIRCADIAN_AMPLITUDE` was `relu(0.5 − amplitude)` — a one-sided floor, so a 2x
overshoot was free. It now bands [0.5, 0.8]. The SBP/DBP changes are to the **targets**,
not the teacher: each encoded a value at or below the shallow edge of the range its own
comment cites. Full reasoning, including the disclosure that this also moves the
teacher's z, is in `cohorts/breadth_floor.py`.

## Rejected, with reasons — do not re-propose without new evidence

**`hr_circ_amp` 5.0 → 3.5.** Reverted after measurement. No cohort statistic constrains
it (it cancels in `sleep_hr_dip`, whose arms share a clock window), so the only
justification available was a whole-day sleep-vs-wake contrast — not an encoded, cited
quantity. And it does not achieve what it was invoked for: the contrast is 26.7 % at 5.0
and 24.6 % at 3.5, both above the 10–20 % ambulatory band. The protocol also matters in
the opposite direction to the intuition — adding realistic daytime activity *widens* the
contrast (24.1 % → 26.7 %) by lifting the daytime mean.

**Justifying the circadian work with `temp skill = −50.05`.** That number is an artifact
of a ruler whose temperature ground truth has two distinct values (36.6, 36.7) and a
persistence error of 0.0004. The iter-77/87 reading — structural artifact, do not chase
— was right. Part C stands on the literature contradiction alone.

## Two risks this iteration deliberately takes

**B1 tightens agreement with the teacher exactly where the teacher knows least.**
Normalizing by the reference's own within-window range amplifies the residual (up to
20x at the floor) precisely in windows where the teacher barely moves — and a teacher
that is flat because it is *wrong* now costs the student as much as one that is flat
because the physiology is. That runs against the PRD's "teachers as fences, not
mandatory pointwise truth", and it is a real trade for making slow physiology visible
at all. The floor and the Huber delta bound it, but the honest mitigation is the ruler:
`cgm_real` is **real measured data, not the teacher**, so the signature of this failure
mode is specific and checkable — teacher-source agreement improving while `cgm_real`
skill degrades. Watch that pair.

**`teacher_dynamic` scores agreement with the teacher, not with reality.** These
episodes are not circular in the training sense (they are excluded from the distillation
pool, so the model has not been fitted to them), which is what makes them a fair test of
whether the student can express meal dynamics out-of-sample. They are still circular in
the deeper sense that the teacher is our current approximate law. `cgm_real` is the only
source in the ruler whose ground truth is independent of our own modelling.

## Open, measured, deliberately not fixed

1. **The `sleep_hr_dip` decomposition is undetermined.** With `sleep_hr_frac = 0` the
   dip is still −6.9 bpm, so only ~38 % of it comes from the term named for it; the rest
   arrives via sleep → cortisol → `cort_hr`. Both channels are real and **nothing
   encoded determines the split** — lowering `cort_hr` would satisfy the same target
   equally well. 0.06 fixes the total, which is all the literature here constrains, and
   must not be read as "sleep bradycardia is 6 % of HR0". Separating them needs a
   contribution that isolates the channels (beta-blockade or cortisol-suppression arm).
2. **Whole-day HR contrast ~25 % vs the 10–20 % ambulatory band.** Real, unexplained,
   and not fixable by the constant that looked responsible (see Rejected). Needs a cited
   contribution encoding the whole-day quantity before anything is tuned for it.
3. **The absorbing-floor bug is generic to the mass-action shape.** Any species whose
   head emits `prod ≥ 0` with no setpoint term has the same trap (it is the documented
   iter-51 dead-pathway wall). Glycogen is fixed; `mitochondrial_capacity`, `lactate`
   and `hepatic_output` use `SpeciesHead` and should be audited the same way. Not done
   here: no encoded target constrains them, and widening the blast radius on a run that
   already changes the ruler would make a regression unattributable.
4. **`extended_fast_liver_glycogen_overnight` (z=+2.09).** The **teacher** realizes
   −11.97 g against its own −60 ± 25 target on those arms. Either the target or the arms
   are wrong; settle it against the literature before pointing loss at it.
5. **`leptin_fed_vs_fasted` (z=−2.00).** Leptin realizes 0.00 with sd 0.00 — completely
   inert to feeding. A genuine flat-marker defect, untouched here.
6. **`fasting_breakfast_glucose_morning` (z=−2.85)** and
   **`extended_fast_insulin_basal` (z=+2.07)** remain, both previously diagnosed as
   artifacts of the arms/population rather than teacher defects. Not re-litigated.

## Acceptance criteria, in priority order

1. **Primary.** 24 h fast cascade at the prior-mean embedding moves toward the teacher:
   `liver_glycogen` ratio ≥ 0.5 (iter 93 shipped −0.00), `bhb` ≥ 0.5 (0.02), insulin
   moving in the **correct direction** (−0.17). Run
   `scripts/iter94_student_fast_probe.py <ckpt>`.
2. **Secondary.** `skill_vs_persistence` on `cgm_real` and `teacher_dynamic` beats the
   A5 baseline measured on the same ruler with the iter-93 artifact.
3. **Circadian.** Student temp peak-to-trough ≤ 0.7 °C; sleep HR dip and HRV rise
   within their cohort bands.
4. **Guard.** iter-92 meal kinetics hold (glucose peak 45–60 min, ghrelin nadir
   −30…−50 % at 60–90 min).
5. **Guard.** `legacy_static` absolute MAPEs do not move materially.

**`gate.passed` is not an acceptance criterion for this iteration.** The skill
thresholds are set at the natural zero — "as good as predicting no change" — and the
model was 8.2x worse than persistence on glucose under the old ruler. Expect failures;
they are the instrument working.

## Run plan

Training and benchmarking are **separate jobs**. iter-93 spent 12.2 h of its 20 h in the
benchmark; the 48-episode ruler costs ~22.5 h at ~1690 s/episode, and 18 h of training
plus that would exceed the 32 h task timeout.

- **A5 baseline** — `--benchmark-only` on `gs://…/jobs/iter93/model.pt` against the new
  ruler. Establishes what iter-94 is compared against. No training cost.
- **iter-94 training** — no `--benchmark-dataset-uri`; the artifact still uploads.
  Estimated 30×265 s + 25×~2270 s ≈ 18 h.
- **iter-94 benchmark** — `--benchmark-only` on the resulting artifact.
