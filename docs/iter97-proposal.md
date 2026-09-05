# iter 97 — the review iteration

*Written 2026-09-04/05 while the work landed. Companion to `docs/review-2026-09-04.md`,
which is the spec; this document records what was actually built, what it measured,
and what iter 97 must show on the artifact.*

## Thesis

iter 96 improved both gate blockers and still failed, and the review found why the
number could not be trusted either way: the same three diseases in every layer.
Frame and convention bugs (an HPA cascade that never trained, an inverted sleep arm,
a calibration objective that was a different forward map from the scorer), conservation
declared as architecture and enforced nowhere carbon moves (a gut kernel delivering
3.5x the ingested mass with a hard step inside the scored window, hepatic output
booked three times in the teacher), and supervision mis-sized both ways (25 literature
optimizer steps per run against ~5,300 pointwise imitation steps of a teacher we knew
was wrong). iter 97 fixes **all** of them — no cherry-picking — so that the next
number means what it says. It is therefore a *structural* iteration whose primary
claims are true by construction and pinned by tests, with the gate as the floor.

## What changed, by layer (all measured; commits on main)

### Ruler and calibration (`b41acde..b17bb14`, `eb8a499`)
- Calibration integrates ONE continuous trajectory over the observation window and
  reads predictions at the check-ins — the same forward map the scorer runs.
  Windowed-vs-scored prediction mismatch at the fitted points: legacy up to 30 mg/dL,
  cgm_real mean 5.9 / max 15 mg/dL → **0.0000**.
- One shared `pulse/calibration.py` used by the benchmark and the server: chronological
  20 % hold-out, early stop with patience, acceptance only on held-out improvement,
  prior toward the trained prior mean (weight 0.25, to be swept), soft norm penalty
  instead of the hard 3.0 clamp, Huber, sleep/activity masks and the real start time in
  both paths. On three real nights the iter-96 artifact's eval MAE went from 11.3 / 10.5
  (prior-mean) to 4.2 / 4.6 mg/dL / bpm at ‖emb‖ 0.36 — the old path reached 4.1 / 7.5
  at ‖emb‖ 1.4–2.3 and 3x the time.
- Skill is `1 − MAE / max(persistence_MAE, σ_obs)` in physical units, per episode first.
  The old ratio is kept under `skill_vs_persistence_mape`. Recomputed on the iter-96
  report: sbp/dbp on teacher_dynamic pass (+0.62 / +0.76), glucose and hr still fail on
  both dynamic sources — the real failures survive, the artifact does not.
- `ruler_fingerprint` (git SHA, per-source truth sha256, dataset md5, thresholds md5,
  calibration settings) and a `--frozen-ruler` file so any artifact can be re-scored on
  exactly the ruler a number was produced with. `scripts/rescore_artifact.py`.
- Headline `headline.normalized_mae` with equal source weight (cgm_real, teacher_dynamic,
  teacher); `overall_weighted_mape` kept as a labelled continuity line.
- Teacher episodes no longer score their own check-ins; verifier skips checks whose
  time mask is empty; meal fences raised to the textbook slopes; textbook checks report
  soft margins and a hairline flag; every "resting" minute in the ruler is activity 0.
- The sweep passes every knob explicitly (its `max_norm` rows had been no-ops).

### Teacher (`91ecfc7..372388e`)
- **One carbon budget.** `glucose_fluxes()` writes every glucose-carbon flux once in one
  unit system (V_G 1.85 dL/kg): `dG = Ra − syn_L − syn_M + EGP − k_ii·G − X·G − U_ex`,
  `dLGly = syn_L + gng_divert − glycogenolysis`. **Gb is a derived fixed point**
  (`Hep_b = uptake_ii·Gb·V_G`, the `lip_max` pattern). Eucaloric-day ledger residual
  **+0.20 g/day** (was 48 g synthesized from nowhere and 48 g destroyed). The 48-h fast
  floor is absolute: Gb 70/95/130 patients land at 66.5/72.0/67.7 mg/dL (were
  56.5/73.8/97.1). `hepatic_output` rests at its typical.
- Insulin action can no longer be a glucose source (Bergman-signed); rectifiers at
  basal replaced by saturating functions centred at basal — ghrelin now **rises 15 % /
  35 %** at 24 / 48 h of fasting (was 100.000 flat); the incretin share of insulin AUC
  is **0.66** (was 0.07; Nauck 0.5–0.7); HPA suppression applied once, cort/ACTH the
  same asleep and awake (0.43 vs 0.47; was 0.26 vs 0.47), cortisol peak 08:36 (was
  09:30); pools rest at their declared typicals (muscle glycogen 400, not the 450 cap);
  kernel truncation loss 0.3 % (was 3.8 % mean / 14.4 % max); carbohydrate appearance
  mass-conserving (gain 2.55 → 7.72 = 1000/V_G); rest activity 0; `hr_circ_amp` 5 → 2.5
  so the asleep 03–06 HR rise is **+1.48 bpm** (real +1.6; was +2.64).
- The three standalone generators (55 % of trajectory-distillation weight) are now
  views of `simulate_full_body` — they had still been teaching the double-driven
  cortisol (nadir 16.5), absolute-insulin ghrelin, and the 6.7x cortisol→HR gain.
- Anchor audit (N=12): contradictions at |z| ≥ 2 **2/40 → 0/40**, mean |z| 0.61 → 0.51;
  textbook-on-teacher **40/40 → 40/40**. Section-8 numbers survive (24-h fast insulin
  5.7 / BHB 0.79 / glucose 84.7 / liver glycogen 50.5 g; 75 g meal +56.6 at 55 min).

### Student (branch `worktree-agent-adc9e04899aad6805`, merged; `fec452e` integration)
- **Gut kernel as a normalized density**: a learned mixture over a bank of gamma shapes
  that are zero at t = 0, integrate to 1 and decay smoothly, times a per-patient
  bioavailability. K(0) = 0 and ∫K = f_bio·mass **by construction**. For 60/20/25 g:
  appearance at dt = 0 was 7.98 (its maximum) → 0.000; AUC 3.5x the teacher → within
  3 %; the −8 mg/dL/h glucose cliff at dinner + 8 h is gone (+0.12 / −0.17 / −0.21
  mg/dL/h across 02–05 h). The untrained mixture prior is fitted to the teacher's real
  two-component profile; the active window is 720 min because the teacher's slow
  component carries mass past 480 and the window doubles as the meal lookback.
- Stress cascade reads `raw_state` (it was reading normalized state as raw, so the
  feedback parameter never trained); time features are `[sin, cos, sin 2θ, cos 2θ]`
  (the midnight sawtooth injected −116 bpm/h into the HR vector field); SBP = DBP +
  pulse pressure and HRV in log-space **by construction**; muscle glycogen breakdown
  `∝ relu(activity − rest)` (was spending 216 g in a 36-h rest); per-patient glucose /
  insulin thresholds and an `Ib` head (a Gb=75 patient fasts to 78 mg/dL, not 54;
  Gb=120 fasts with insulin 12, not 24); hepatobiliary as true mass action with a closed
  loop; mitochondrial capacity has one structural role; BHB has a substrate term;
  insulin action is a sink; no structurally dead parameters (12,555 of 75,914 → 0 of
  68,293); glucose ledger closed (residual 3.8e−4 g/day).
- Glucose balance mirrors the teacher's fixed point (k_ii population constant, EGP_b =
  k_ii·Gb, glycogenolysis and GNG normalized to 1 at basal) — *pending at the time of
  writing; see the build-state note.*

### Training signals and loop (`108bcc9..3cff7ff`)
- **Aux signals step every 8 trajectory windows** (12 steps per epoch instead of 1),
  phase-2 LR floor 1e-3, per-signal gradient clip 5 with `[GRADNORM]` logged each epoch,
  cumulative aux steps per signal in the checkpoint. The rules' sleep arm is the right
  way round; meal lookback is the kernel window; undeclared arms run awake at rest 0.
- Trajectory bands per marker (0.15 σ observed, 0.30 σ unobserved), ghrelin and
  glucagon supervised by window-mean + trend sign; cohort losses score the batch mean
  against the SEM with POINT/BAND/AT_MOST/AT_LEAST shapes, embeddings detached,
  adaptive share capped at 25 % multiplying the hand-set weight; coupling prior is a
  two-sided band hinge on the declared range in normalized units; soft extrema
  everywhere (a 190 peak reads 187.5, not 167); distillation level windows carry
  meals and duodenal drive with a 0.25 σ dead-zone; the verifier surrogate returns
  no-op where a window cannot contain the checked phase.
- Phase 3 (6 epochs) with input dropout on all inputs and meal-macro dropout; the
  teach-to-test protocols are perturbed per epoch (dose ±20 %, timing ±30 min, start
  ±1 h); `insulin` dropped from distillation (double pull vs dose-response); HR level
  supervised by the Holter anchors, not a hand-set 70.
- **Teacher rule violations: 22/61 → 2/61**, both flagged as deliberate corrections
  (and handed back to the teacher). `scripts/rules_teacher_audit.py` gates dispatch.
- argparse defaults can no longer shadow the spec; CLI flags diverging from `--spec`
  are an error; every load-bearing flag is explicit in `train/spec.json`.

## Acceptance (measured on the artifact, in this order)

**A. By construction (the tests, plus a probe on the artifact):** carbon ledger closed
(student and teacher); K(0) = 0 and ∫K = f_bio·mass; SBP > DBP and HRV > 0 for every
calibrated embedding; muscle glycogen flat at rest; `stress._beta_raw` ≠ its init
(the cascade trained); no state at a catastrophe clamp on any benchmark episode.

**B. Pre-dawn window (scripts/iter96_dawn_probe.py, prior-mean embedding, 14 cgm_real
nights):** HR slope 03–06 ≤ the teacher's new **+1.48 bpm/h** (iter 96: +3.06; real
+1.06); glucose slope ≤ **−0.5 mg/dL/h** (iter 96: −0.09 — contaminated by the kernel
cliff; real −1.75); cortisol nadir < 7 µg/dL (iter 96: 7.16).

**C. Gate on the corrected ruler:** noise-floored skill > 0 for glucose and hr on
cgm_real and teacher_dynamic; textbook mean ≥ 0.85 with no hairline misses counted as
regressions; verifier categories pass. The comparison baseline is iter 96 **re-scored on
the same ruler** (`ruler_fingerprint` must match), not the iter-96 report.

**D. Guards:** iter-92 meal kinetics (glucose peak 45–70 min, ghrelin nadir −30..−50 %);
legacy_static absolute MAPE within 20 % of iter-96 re-scored; the eucaloric 5-day probe
(liver glycogen > 70 g, mito > 0.9).

**Falsifier.** The literature stack now gets 12x the optimizer steps and losses shaped as
the PRD prescribes. If the student's cohort-anchor z distribution does not tighten
against iter 96 (mean |z| on the 40 anchors at the prior-mean embedding), then
"a parameter no strong gradient reaches" was not the disease and the next iteration is
about representational capacity, not supervision.

## Baselines measured before dispatch

**iter 96 re-scored on the corrected ruler** (tree `77e02d5`: frame-fixed calibration
with the OLD 512-step / prior-0 algorithm, noise-floored skill, frozen old-teacher truth;
grovina-mini, 2026-09-05):

| | iter-96 report | iter-96 re-scored |
|---|---|---|
| gate failures | 6 | **2** (cgm_real glucose −0.05, hr −0.14) |
| skill cgm_real glucose / hr | −0.58 / −0.39 | **−0.05 / −0.14** |
| skill teacher_dynamic glucose / hr / sbp / dbp | −0.28 / −0.23 / −1.82 / −2.26 | +0.67 / +0.53 / +0.66 / +0.83 |
| cgm_real MAE glucose / hr | — | 8.37 mg/dL / 3.42 bpm (σ_obs 8 / 3) |
| headline normalized MAE | — | 0.608 |
| textbook / verifier | 0.8625 / 0.906 | 0.8625 / 0.852 |

**iter 96 re-scored on the FINAL iter-97 teacher's frozen truth** (same tree and old
calibration; `ruler.frozen.iter97-final.json`, git 4495d75): the same two failures
(cgm_real glucose −0.05, hr −0.14); teacher_dynamic glucose +0.54, hr +0.17, sbp +0.66,
dbp +0.83, temp +0.63; headline normalized MAE 0.701; textbook 0.8625; verifier 0.847.
The teacher_dynamic numbers are lower than on the old-teacher ruler because the truth
moved (the new teacher's hr truth sd 1.87 vs 3.77) — which is exactly what the
fingerprint is for. A third re-score with the NEW calibration (prior 1.0, the setting the
iter-97 job's benchmark uses) on this same ruler is the calibration-matched baseline.

Reading: fixing the calibration frame alone (the embedding was being fitted to a
trajectory the scorer never ran) took the real-data blockers from clearly negative to
within noise of zero, and the teacher_dynamic failures were the ruler. The iter-97
artifact is compared against THIS row (and against the second re-score on the final
teacher's frozen truth, `ruler.frozen.iter97-final.json`, queued behind the sweep).

## Run plan

1. iter-96 re-score on the frozen old-teacher ruler at `77e02d5` (grovina-mini): the
   ruler-fix-only delta. Then freeze the FINAL iter-97 teacher's truth and re-score
   iter 96 on that file in the `77e02d5` tree: the exact baseline for C.
2. Calibration sweep of the new knobs on the iter-96 artifact (grovina-mini, chained).
3. Local validation on grovina-mini (short run) and a Cloud Run smoke in parallel.
4. Dispatch training **with** the benchmark args (iter 96 was dispatched without them
   and sat ungated for two days); report path `training/jobs/iter97/benchmark-report.json`.
   Expected wall-clock ~24 h (44 h timeout). Watch the first phase-2 `[GRADNORM]` and
   epoch-time lines; cut `--aux-every-k-windows` to 12 if an epoch exceeds 45 min.
