# Pulse Iter 23 — Resume Handoff

Self-contained resume point. **Read only this file to pick up the work
after a reboot.** Iters 12-22 are documented in their own
`iter<N>-handoff.md` files but you do not need to read them to proceed.

> **Resume action:** iter 23 was submitted at the time this handoff was
> written. Job ID and Build ID are in the *In-flight* section at the
> bottom. Skip straight there to check whether the build finished
> before doing anything else — do **not** re-submit before verifying.

## Headline status

**Iter 22 falsified the gradient-budget hypothesis.** Bumping global
cohort weight 0.15 → 0.40 and per-spec weight 1.0 → 3.0 on the 9
collapsed metabolic specs failed all four collapsed-species hard gates
and regressed glucose MAPE further:

| metric | iter 20 | iter 21 | **iter 22** | gate | result |
| --- | --- | --- | --- | --- | --- |
| `overall_weighted_mape` | 0.1859 | 0.1896 | **0.1934** | ≤ 0.18 | ✗ |
| glucose `mean_mape` | 0.5141 | 0.5432 | **0.5657** | ≤ 0.50 | ✗ |
| `trajectory_rollout` raw_loss | 9.61 | 10.87 | **10.80** | drop | ✗ flat |
| verifier `overall_score` Δ vs iter 20 | — | +0.086 | **+0.090** | ≥ +0.066 | ✓ held |
| verifier meal Δ vs iter 20 | — | +0.163 | **+0.167** | hold | ✓ held |

Cohort-ablation on iter 22 trained checkpoint vs iter 21:

| spec | target | iter 21 | **iter 22** | gate | result |
| --- | --- | --- | --- | --- | --- |
| `large_meal_glp1_peak` | 12 | 0.030 | **0.141** | ≥ 5 | ✗ |
| `ogtt_glucagon_suppression` | -25 | -0.012 | **-0.016** | ≤ -10 | ✗ flat |
| `mixed_meal_glucagon_suppression` | -10 | -0.011 | **-0.016** | — | ✗ flat |
| `meal_hgo_suppression` | -1.0 | 0.008 | **0.007** | — | ✗ flat |
| `extended_fast_bhb_overnight` | 0.20 | -0.003 | **-0.003** | ≥ 0.10 | ✗ flat |
| `meal_ffa_suppression` | -0.30 | -0.000 | **-0.000** | — | ✗ flat |
| `extended_fast_ffa_overnight` | 0.40 | 0.000 | **-0.000** | — | ✗ flat |
| `meal_ghrelin_suppression` | -30 | -0.042 | **-0.094** | — | ✗ tiny move |
| `extended_fast_insulin_basal` | 7 | 20.28 | **18.08** | ≤ 14 | ✗ moved 11% |
| `ogtt_75g_insulin_peak` z | 0 | -0.87 | **-0.84** | ≤ \|1.0\| | ✓ held |
| `ogtt_75g_insulin_mean_3h` z | 0 | -0.21 | **-0.19** | held | ✓ held |
| cold-calibration misleading specs | ≤ 2 | 2 | **2** | ≤ 2 | ✓ |
| gut probe ratio (≥30g) | ≤ 1.20 | ~0.99 | **~0.97** | hold | ✓ |

Signal-balance on iter 22 trained checkpoint:

| signal | iter 21 ‖g_all‖ | **iter 22 ‖g_all‖** | iter 22 raw_loss |
| --- | --- | --- | --- |
| `cohort_statistic` | 5.9 | **5.0** | 3.47 |
| `trajectory_rollout` | 26.0 | **26.1** | 10.80 |
| ratio (cohort vs trajectory) | 4.4× weaker | **5.2× weaker** | — |

Per-spec `‖g_met‖` from cohort term on iter 22:

| spec group | ‖g_met‖ |
| --- | --- |
| insulin specs (working) | 17.7 – 55.5 |
| glucose specs (working) | 14.4 – 48.9 |
| **collapsed metabolic specs** (`glp1`, `glucagon`, `ffa`, `bhb`, `hgo`, `ghrelin`) | **6e-3 – 1.2** |

The 4-decade gap in `‖g_met‖` between working and collapsed species
lives in `∂pred/∂params`, not `∂loss/∂pred` — multiplying the cohort
loss weight scales the gradient by the same factor regardless of
direction, so amplification on a near-zero gradient yields a
near-zero gradient. The iter-21 risks table called this exact
outcome row: *"Nothing moves: collapsed specs stay at 0 AND glucose
stays bad → gradient-budget contest is not the actual blocker →
representation collapse is structural → iter 23 factorizes the
metabolic head per-species (the iter-16 gut-fix analog)."*

## What's actually wrong

The metabolic module's `MassActionModule` had one shared 48-dim MLP
emitting 14 rates (production + consumption for 7 species) from a
single trunk. Every species fed gradients into that trunk. Insulin's
trajectory supervision is dense (every minute, 240-min windows) and
high-amplitude (5-60 µU/mL), so insulin's gradient dominated trunk
updates. The trunk reorganized to encode insulin/glucose dynamics
features, and the small-magnitude species (GLP-1, glucagon, BHB, FFA,
HGO, ghrelin) lost any directions in `h`-space they could exploit.

The optimizer's path of least resistance for those species: keep
`W_species ≈ 0`, output near-zero rates everywhere, accept a constant
`z² ≈ 4` cohort loss. Local minimum. Any perturbation increases loss
as much as it decreases it (the cohort signal can't tell the network
"output 12 at meal+30min and 0 elsewhere" because there's no trunk
feature shaped like a GLP-1 pulse). The "wide manifold of internal
states all producing near-zero predictions" the iter-22 cohort-ablation
found is precisely this basin.

The same module also showed the basal-vs-peak insulin tension:
trained basal=18 µU/mL (target 7) and peak=39 µU/mL (target 60) — both
wrong, **in opposite directions**. A single learned scale on the
insulin head cannot satisfy basal-low and peak-high simultaneously.

## What iter 23 changes

Two architectural surgeries on the metabolic + appetite modules.
Iter-22's two cohort knobs are reverted to iter-20 baseline so iter
23 is a pure architecture-vs-architecture test.

### Intervention A — per-species heads (the iter-16 gut-fix analog)

`apps/pulse/engine/pulse/modules/base.py`:

* `SpeciesHead` (new): a small MLP (Linear-Tanh-Linear-Tanh-Linear)
  taking the full module input and emitting `(raw_prod, raw_cons)`
  for one species. The parent module applies softplus + per-species
  scale.
* `MassActionModule.__init__` now takes `head_factories: dict[int,
  HeadFactory] | None` and instantiates `n_species` independent heads
  (one per species), with an optional override per index.
* `MassActionModule.forward` runs each head and stacks their outputs.

This isolates each species' representation. Insulin's gradient
pressure no longer reshapes the GLP-1 head; gradient pressure on a
small species lands on parameters that actually move that species'
prediction. The shared trunk is gone — each species owns its own
hidden representation.

Side-effect: the same change applies to `AppetiteModule` (3 species:
ghrelin, leptin, glp1) and `StressModule` (2 species: cortisol, acth)
which both inherit from `MassActionModule`. This is the right scope
— ghrelin and glp1 are in the appetite module, not metabolic, and
their iter-22 collapse is the same mechanism.

Param count: total model 14,345 → 43,412 (3.0×). Metabolic alone:
~4.4k → 26.9k (6×). Heads scale linearly with `n_species`; the
budget is dominated by metabolic since it has the most species (7).

### Intervention C — glucose-gated insulin head

`apps/pulse/engine/pulse/modules/metabolic.py`:

* `GlucoseGatedInsulinHead` (new): replaces the standard `SpeciesHead`
  for the insulin index only (installed via the new `head_factories`
  dict). The MLP emits **three** logits — `raw_basal`, `raw_peak`,
  `raw_cons` — and combines them as

      prod = softplus(raw_basal) + softplus(raw_peak) · σ((g - g_thresh) / g_temp)

  where `g` is the module's normalized glucose state at the current
  time-step, and `g_thresh` / `log_g_temp` are learnable scalars
  initialized so the gate is ≈0.27 at fasting (g=95 mg/dL) and ≈0.94
  at OGTT-peak (g=150 mg/dL). The gate transitions over a normalized
  width of ≈0.5 (~15 mg/dL).

This structurally decouples basal floor from peak amplitude.
`raw_basal` controls fasting insulin via the always-on softplus
term. `raw_peak` controls postprandial amplitude via the
glucose-gated term, which vanishes at low glucose. Optimizer can
satisfy "basal=7" and "peak=60" with separate parameters instead of
a single shared scale that has to interpolate between them.

### Reverted from iter 22

* `apps/pulse/engine/pulse/knowledge/cohorts/nutrition.py` — removed
  `weight=3.0` from 8 specs (back to default 1.0).
* `apps/pulse/engine/pulse/knowledge/cohorts/glucose_handling.py`
  — removed `weight=3.0` from `MEAL_HGO_SUPPRESSION`.
* `apps/pulse/train/spec.json` — `--cohort-statistic-weight=0.40` →
  `0.15`; iter 23 hypothesis + expectedEffect.

So iter 23 is a clean architecture-vs-architecture test against the
iter-21 baseline: same cohort weighting, same trajectory pipeline,
same PatientParams — only the metabolic / appetite / stress module
heads change.

### Training pipeline: nothing else changes

Trajectory windows, bands, contributions, default patients, landmark,
gut sweep, insulin sweep, all weights — identical to iter 21 and 22.
Phase-1/2 epochs, LRs, hidden_dim, sample sizes — identical.

## What iter 23 does NOT change

* `PatientParams` — the iter 21 cold-model recalibration is locked
  in. Cold-calibration diagnostic still shows misleading-spec count
  = 2 (ghrelin, fasting_breakfast_glucose).
* Trajectory pipeline (windows, bands, contributions, default
  patients, landmark, gut sweep, insulin sweep) — all identical.
* Gut module — still the iter-16 factorized kernel.
* Cardiovascular / Thermoreg / Respiratory — all `LearnedDynamicsModule`,
  not affected by the `MassActionModule` rewrite.
* All training-pipeline knobs.
* Cohort spec list and per-spec sigmas (only the iter-22 weight bumps
  are reverted).

## Code state (iter 23 changes ready to commit)

Modified:

* `apps/pulse/engine/pulse/modules/base.py` — added `SpeciesHead`,
  `HeadFactory`, rewrote `MassActionModule` to instantiate per-species
  heads via `nn.ModuleList`. Buffer type annotations added so mypy
  can resolve the `* prod_scale` / `* cons_scale` operators against
  the new `nn.ModuleList`.
* `apps/pulse/engine/pulse/modules/metabolic.py` — added
  `GlucoseGatedInsulinHead`; `MetabolicModule` installs it at the
  insulin index via `head_factories`.
* `apps/pulse/engine/pulse/knowledge/cohorts/nutrition.py` — removed
  `weight=3.0` from 8 specs (and the inline iter-22 rationale comments).
* `apps/pulse/engine/pulse/knowledge/cohorts/glucose_handling.py`
  — removed `weight=3.0` from `MEAL_HGO_SUPPRESSION`.
* `apps/pulse/train/spec.json` — `--cohort-statistic-weight=0.40` →
  `0.15`; iter 23 hypothesis + expectedEffect rewritten.

New file: this handoff. No new engine tests (the per-species head
topology is exercised by every existing model-forward test). All
130 engine tests pass. See `training-runs.md` for the **commit before
submit** workflow; that doc was added when formalizing the policy
after this run.

## In-flight: iter 23 was submitted

* Job ID:   `train-20260424T163240Z`
* Build ID: `ab7bd13d-af70-45c4-8be3-7d19aadeadb6`
* Submitted: 2026-04-24T16:32:40Z
* `meta.json` for this job shows a dirty working tree (iter 23 was
  launched before the commit recorded in the same change set as
  `training-runs.md`). Reconcile by treating the first commit on the
  iter-23 branch that contains the files below as the canonical source
  for the trained architecture. **Later iters: commit first** — see
  `apps/pulse/docs/training-runs.md`.
* Expected finish: ~6h after submit (matches iter 19-22 wall clock,
  param count 3× higher but most additions are small per-species
  MLPs that vectorize fine on the same CPU pool).

The Cloud Build *will* fail at the final `EnforceBenchmarkGate` step
because we haven't cleared `overall_mape ≤ 0.16` yet — same expected
failure pattern as iters 17-22. Training, benchmark, comparison, and
artifact upload all complete normally; the gate step is a separate
post-training check. Treat the build's FAILURE status as *expected*
and look at the artifacts directly.

## How to resume after reboot

```bash
cd /Users/grovina/Projects/grovina/platform
gswitch grovina  # gcloud config configurations activate grovina
gcloud auth list
ls apps/pulse/engine/.venv/bin/python

# Check whether iter 23 has finished (artifacts land last)
gsutil ls -l gs://grovina-pulse/training/jobs/train-20260424T163240Z/

# When checkpoint.pt + benchmark-report.json + delta-vs-baseline.json
# are all present, pull them. Use parallel_process_count=1 because
# gsutil multiprocessing on macOS is broken (see iter 21 handoff).
mkdir -p /tmp/iter23 && \
  gsutil -o "GSUtil:parallel_process_count=1" cp \
    gs://grovina-pulse/training/jobs/train-20260424T163240Z/{benchmark-report.json,delta-vs-baseline.json,checkpoint.pt,spec.json} \
    /tmp/iter23/

# And iter 22 for compare baselines (the delta-vs-baseline.json that
# Cloud Build produces compares iter 23 vs iter 22 automatically; keep
# iter 22 around for any side-by-side diagnostic)
mkdir -p /tmp/iter22 && \
  gsutil -o "GSUtil:parallel_process_count=1" cp \
    gs://grovina-pulse/training/jobs/train-20260424T071900Z/{benchmark-report.json,delta-vs-baseline.json,checkpoint.pt} \
    /tmp/iter22/

# Tests should still pass identically
cd apps/pulse/engine && .venv/bin/python -m pytest tests/ -x -q

# Diagnostic suite on iter 23 checkpoint
cd apps/pulse/engine && .venv/bin/python -m pulse.diagnostics cold-calibration
cd apps/pulse/engine && .venv/bin/python -m pulse.diagnostics signal-balance \
  --checkpoint /tmp/iter23/checkpoint.pt --n-patients 20
cd apps/pulse/engine && .venv/bin/python -m pulse.diagnostics cohort-ablation \
  --checkpoint /tmp/iter23/checkpoint.pt --sample-patients 4

# Headline check: did the gates move?
.venv/bin/python -c "
import json
r22 = json.load(open('/tmp/iter22/benchmark-report.json'))
r23 = json.load(open('/tmp/iter23/benchmark-report.json'))
print(f'overall:  iter22={r22[\"overall_weighted_mape\"]:.4f}  iter23={r23[\"overall_weighted_mape\"]:.4f}')
print(f'glucose:  iter22={r22[\"per_marker\"][\"glucose\"][\"mean_mape\"]:.4f}  iter23={r23[\"per_marker\"][\"glucose\"][\"mean_mape\"]:.4f}')
print(f'verifier: iter22={r22[\"verifier\"][\"overall_score\"]:.4f}  iter23={r23[\"verifier\"][\"overall_score\"]:.4f}')
"
```

Note: iter-22 (and earlier) checkpoints are **not loadable** by the
iter-23 model — the parameter names and shapes changed when the
shared MLP was replaced with per-species `nn.ModuleList` heads. This
is intentional (clean architectural break). Diagnostics that load
iter-23 checkpoints work fine; diagnostics that try to load iter-22
checkpoints into the iter-23 model will fail with a state-dict
mismatch error. Use the existing iter-22 artifacts via the saved
benchmark / delta-vs-baseline JSON only.

## Cloud Build / job IDs

| iter | job ID | build status when noted | benchmark-report key metrics |
| ---- | ------ | ----------------------- | ---------------------------- |
| 17   | train-20260422T065354Z | FAILURE (gate) | overall=0.198, glu_mean=0.579 |
| 18   | train-20260422T140536Z | FAILURE (gate) | overall=0.197, glu_mean=0.575 |
| 19   | train-20260422T180605Z | FAILURE (gate) | overall=0.198, glu_mean=0.578 |
| 20   | train-20260423T122419Z | FAILURE (gate) | overall=0.186, glu_mean=0.514 |
| 21   | train-20260423T175447Z | FAILURE (gate) | overall=0.190, glu_mean=0.543 (insulin_peak +18 µU/mL structural win) |
| 22   | train-20260424T071900Z | FAILURE (gate) | overall=0.193, glu_mean=0.566 (cohort-amplification falsified) |
| 23   | train-20260424T163240Z | WORKING at handoff time | (pending) |

## Success criteria for iter 23

Hard gates — intervention A (per-species heads):

1. **`large_meal_glp1_peak` predicted Δ ≥ 6 pmol/L** (iter 22: 0.14;
   cold 12.16; target 12).
2. **`ogtt_glucagon_suppression` predicted Δ ≤ -10 pg/mL** (iter 22:
   -0.016; cold -23.23; target -25).
3. **`extended_fast_bhb_overnight` predicted ≥ 0.10 mmol/L** (iter
   22: -0.003; cold 0.06 in 240-min window, 0.27 at 16h; target 0.20).
4. **`meal_hgo_suppression` predicted ≤ -0.5 mg/min** (iter 22: 0.007;
   cold -0.61; target -1.0).
5. **`meal_ffa_suppression` predicted ≤ -0.10 mmol/L** (iter 22:
   -0.000; cold -0.165; target -0.30).
6. **`meal_ghrelin_suppression` predicted ≤ -10 pg/mL** (iter 22:
   -0.094; cold -67.65; target -30).

Hard gates — intervention C (glucose-gated insulin):

7. **`extended_fast_insulin_basal` predicted ≤ 12 µU/mL** (iter 22:
   18.1; cold 11.36; target 7).
8. **`ogtt_75g_insulin_peak` predicted ≥ 45 µU/mL** (iter 22: 39.1;
   cold 53.6; target 60). Both 7 *and* 8 must hold simultaneously —
   that is the falsifiable test for whether the glucose-gated
   parameterization actually decouples basal from peak.

Sanity gates (unchanged from iter 22 baseline):

9. **No regression in gut probe metrics:** ratio stays ≤ 1.20× on
   every dose ≥ 30g; per-gram slope ≤ 3.5; AUC sub-loss < 0.10.
10. **Cold-calibration misleading-spec count ≤ 2** (PatientParams
    unchanged).

Soft gates (recovery):

11. **glucose `mean_mape` ≤ 0.50** (iter 22: 0.566).
12. **`overall_weighted_mape` ≤ 0.18** (iter 22: 0.193).
13. **verifier `overall_score` within -0.02 of 0.862** (iter 22's
    overall — the +0.090 gain over iter 20 must hold).
14. **`trajectory_rollout` raw_loss ≤ 10.5** (iter 22: 10.80) — per-
    species heads should let trajectory tracking improve once cohort
    is no longer pulling the trunk in incompatible directions.

## Risks / what to look for if iter 23 doesn't recover

| symptom | likely cause | next move |
| ------- | ------------ | --------- |
| Some collapsed specs recover (e.g. GLP-1, glucagon, HGO, ghrelin) but BHB and FFA stay near zero | recovered ones live in time scales the 240-min training windows can resolve; BHB and FFA need the 16-22h fasted dynamics that windows never visit (cold-model BHB only reaches 0.27 at 16h, while training windows see 0.10-0.20 with ~0.05 range) | **Iter 24**: extend `TRAIN_WINDOW` for a fraction of windows (e.g. 30% at 720 min, 70% at 240 min), OR add a long-fast knowledge contribution that simulates 16-22h fasted episodes. Both fixes give BHB / FFA dynamics target trajectories to learn from. |
| Insulin basal drops too far (<5 µU/mL) | glucose-gate `g_thresh` initialized too high or `g_temp` too narrow → at fasting the gate is ≈0 and `raw_basal` had to absorb all the amplitude → optimizer may overshoot when amplitude lifts | **Iter 24**: retune `g_thresh` init from 0.5 → 0.3 (~104 mg/dL transition midpoint) and add a soft regularizer pulling `raw_basal` toward typical fasting basal. |
| `ogtt_75g_insulin_peak` recovers (≥45) but `extended_fast_insulin_basal` regresses upward | the gate isn't decoupling — `raw_peak` is too entangled with `raw_basal` because both pass through the same MLP trunk before the final 3-output linear | **Iter 24**: split `raw_basal` and `raw_peak` into two separate sub-MLPs inside the head (basal-only branch consumes glucose-free state; peak branch consumes full state). |
| Verifier meal coherence regresses below iter 22 (+0.167) | per-species heads broke joint dynamics that gave iter 21/22 their meal coherence — the trunk was implicitly enforcing some species-correlation that we now have to learn separately | **Iter 24**: re-tune `coupling_priors/metabolism.py` magnitude bands on the species we just isolated (looser bands so per-species heads can find the joint dynamics without coupling-prior pressure). |
| Glucose `mean_mape` regresses while collapsed species recover | per-species heads gave each species too much capacity → cohort signals over-fit, glucose trajectory tracking degraded | **Iter 24**: tighten `--trajectory-band-default` from 0.05 → 0.03 on metabolic markers, OR reduce per-species hidden_dim from 48 → 24 to constrain head capacity. |
| `trajectory_rollout` raw_loss climbs above iter 22's 10.80 | same as above — heads have more freedom to over-fit cohort against trajectory; trunk no longer regularizes | **Iter 24**: tighten trajectory_band, OR add a soft penalty on per-species head L2 weight. |
| Nothing moves: collapsed specs stay at 0 AND glucose stays bad | failure is not in metabolic head topology — it's either in the upstream coupling (so collapsed species have no signal to react to) or in the supervision data itself | **Iter 24**: instrument signal-balance per-species (not just per-signal) to confirm `‖g_met_per_species‖` actually picked up after factorization. If it did but predictions still didn't move, the issue is data visibility (BHB/FFA case above). If it didn't pick up, the head is structurally OK but the *input* features it sees don't carry the relevant information — investigate coupling inputs. |

## Pointer references (only read if needed)

* `apps/pulse/docs/iter22-handoff.md` — full diagnosis of the
  gradient-budget vs representation-collapse hypothesis test, the
  per-spec ‖g_met‖ table, and the iter-22 in-flight artifact paths.
  Read for the iter-22 baseline state.
* `apps/pulse/docs/iter21-handoff.md` — five-PatientParams cold-model
  recalibration details, the cold-calibration diagnostic spec, and
  the risks table whose "representation collapse is structural" row
  iter 23 implements.
* `apps/pulse/docs/iter16-handoff.md` — factorized kernel structural
  guarantees on the gut module. Iter 23's `SpeciesHead` /
  `GlucoseGatedInsulinHead` are the metabolic-axis analog of the same
  surgical principle: replace a shared multi-output MLP with
  per-output heads so dominant signals stop monopolizing
  representation capacity.
* `apps/pulse/engine/pulse/knowledge/AUTHORING.md` — guide for
  encoding new medical knowledge (cohort specs, coupling priors,
  scenarios, sweep signals).
* `apps/pulse/docs/prd.md` — top-level vision. Read for context
  but not required for iter 23 mechanics.
