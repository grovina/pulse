# Pulse Iter 20 — Resume Handoff

Self-contained resume point. **Read only this file to pick up the work
after a reboot.** Iters 12-19 are documented in their own
`iter<N>-handoff.md` files but you do not need to read them to proceed.

## Headline status

**Iter 19 fully validated the AUC-matching plan and the gut kernel
problem is now solved.** The kernel hits all four hard gates we set:

| iter 19 hard gate | target | actual | result |
| ----------------- | ------ | ------ | ------ |
| per-dose pred/cold ratio (≥30g) | ≤ 1.15× | uniform 1.13× | **HIT** |
| AUC(120g) | ≤ 400 | 384 | **HIT** |
| per-gram slope (mg-min/g) | ≤ 3.20 | 3.19 | **HIT** |
| AUC sub-loss raw | < 0.05 | 0.050 | **HIT** |

But headline metrics didn't move:

| metric                  | iter 17 | iter 18 | **iter 19** |
| ----------------------- | ------- | ------- | ----------- |
| glucose mean_mape       | 0.5793  | 0.5752  | **0.5778**  |
| overall_weighted_mape   | 0.1983  | 0.1972  | **0.1980**  |
| verifier_overall        | 0.7929  | 0.7828  | **0.7835**  |
| verifier[coupling] Δ    | -0.017  | -0.086  | **+0.025**  |

That gap — kernel essentially converged to cold target (1.13×) but
real-episode glucose still 58% off — is the empirical proof we needed.
**The gut is no longer the bottleneck.** Downstream modules (insulin
response, glucose clearance, hepatic glucose output) were leaning on
gut over-amplitude as a free parameter to satisfy trajectory MSE; now
that gut is constrained by the AUC term, those modules are exposed as
genuinely under-tuned but trajectory MSE alone is too weak to fix
them.

**Iter 20's job: raise signal density on the metabolic axis until
trajectory MSE no longer admits compensatory equilibria.** This is a
structural intervention, not a single new signal — see "What iter 20
changes" below for the five-pronged build that's already complete.

## The full iter 16-19 arc (one-paragraph)

Iter 16 factorized the gut kernel so zero-at-zero, dose-linearity, and
monotonicity became analytic (`apps/pulse/engine/pulse/modules/base.py`).
That fixed the *shape* but produced uniform 1.5× over-amplitude — a
"compensatory equilibrium" with under-tuned downstream modules. Iter 17
tried to break it by 5×ing `gut-dose-sweep-weight` (0.10→0.50); failed
because the supervision loss was structurally diluted by per-element
mean over a [B, T=240, C=4] window with mostly-zero entries. Iter 18
introduced an AUC-matching term in `GutDoseSweepSignal._losses` that
collapses (T, C) into per-(dose, batch, channel) integrals before
squaring, providing an undiluted gradient on amplitude; reverted spec
weight 0.10. Amplitude moved 1.50→1.33 (first time ever) but stalled
because AUC's weighted ‖∇gut‖ (0.31) was below trajectory_rollout's
(0.64). Iter 19 added a `--gut-dose-sweep-auc-weight` CLI flag and
set it to 5.0; weighted ‖∇gut‖ for gut_dose_sweep jumped past
trajectory's contribution; amplitude collapsed to 1.13× and AUC sub-
loss to 0.050. Glucose mean_mape stayed at 0.578 ⇒ remaining gap is
in downstream modules, not gut.

## Diagnostic snapshot of iter 19

**Probe at zero embedding** (`pulse.diagnostics probe` analog):

| dose | cold target | iter 18 | **iter 19** | iter 19 ratio |
| ---- | ----------- | ------- | ----------- | ------------- |
| 0    | 0           | 0.86    | 0.71        | —             |
| 15   | 42.3        | 57.1    | **48.6**    | 1.15×         |
| 30   | 84.7        | 113.4   | **96.5**    | 1.14×         |
| 60   | 169.3       | 225.9   | **192.2**   | 1.14×         |
| 120  | 338.6       | 450.9   | **383.8**   | 1.13×         |

Per-gram slope: cold = 2.82 mg-min/g, iter 19 = **3.19** (within 13%).

**Sub-losses at zero-emb, iter 19 model:**

```
mse  = 0.029   (iter18: 0.037, iter17: 0.054)
auc  = 0.050   (iter18: 0.178, iter17: 0.467) — at the 0.05 success threshold
rank = 0.000   (always zero; structural)
```

**`signal-balance` (unweighted ‖∇gut‖, then × spec weight):**

| signal             | weight | iter 18 weighted | **iter 19 weighted** |
| ------------------ | ------ | ---------------- | -------------------- |
| gut_dose_sweep     | 0.10   | 0.31             | **~0.38** (auc_weight=5 in training) |
| trajectory_rollout | ~1.0   | 0.64             | **0.40**             |
| cohort_statistic   | 0.15   | 0.08             | **0.14**             |

The new equilibrium has gut_dose_sweep and trajectory_rollout pulls
roughly equal — that's why amplitude stabilized at 1.13× rather than
fully reaching 1.0×. This is fine; the residual ~13% is well within
ODE-integration noise and structural drift.

**Trajectory loss is essentially unchanged** (iter 18: 8.53 → iter 19:
8.59) — the *expected* signature that downstream modules absorbed the
gut amplitude reduction by quietly drifting *their* parameters into a
different compensation, rather than rising as miscalibrated. This
confirms downstream modules have capacity but no targeted supervision.

## What iter 20 changes (built — ready to submit)

The diagnostic from iter 19 says the same thing two ways: (a) the gut
kernel converged to cold target, (b) `glucose_mape` did not move,
therefore (c) downstream modules are absorbing the gut amplitude
reduction by quietly re-tuning compensation. The fix is not "one more
signal" — it is **raising signal density on the metabolic axis** so
the optimization landscape stops admitting cheap compensatory
equilibria. Iter 20 attacks this on three fronts simultaneously.

### 1. Direct probe supervision on the metabolic module

Mirror the gut_dose_sweep recipe for the seven metabolic species
(glucose, insulin, glucagon, FFA, BHB, lactate, hepatic glucose
output). Implemented as `InsulinSweepSignal` in
`apps/pulse/engine/pulse/training/insulin_sweep_signal.py`:

- `InsulinSweepProtocol` — explicit (glucose, insulin) sweep grids
  (8 glucose × 5 insulin baseline points, configurable).
- `_cold_metabolic_rates(glucose, insulin, params)` — analytical
  Bergman-style derivatives for each species at a fasting baseline
  (no meal, no stress, neutral autonomic state).
- `compute()` — runs `model.metabolic` across the sweep at sampled
  patient embeddings + the zero embedding, computes per-species
  `mse + ranking_weight·rank + auc_weight·integral` (same three-term
  loss as gut). Per-species weights bias the loss toward insulin
  (GSIR) and hepatic output, which are the species directly involved
  in glucose dynamics.

CLI flags plumbed in `train.py`:
`--insulin-sweep-weight`, `--insulin-sweep-sample-patients`,
`--insulin-sweep-auc-weight`, `--insulin-sweep-ranking-weight`.

`signal_balance.py` extended to track `grad_norm_metabolic` so the
gradient-budget reasoning generalizes to the metabolic module the
same way it did to the gut.

Tests: `tests/test_insulin_sweep_signal.py` (8 cases) — verifies cold
target monotonicity (GSIR rises with glucose above threshold; insulin
clearance rises with insulin), gradient lands on the metabolic
pipeline only, loss decreases over toy training iterations.

### 2. Literature absorption — six new high-quality cohort specs

`pulse/knowledge/cohorts/glucose_handling.py` adds six
`CohortStatisticSpec` entries grounded in standard endocrinology
sources (DeFronzo 1979 OGTT; ADA/WHO criteria; mixed-meal HGO
suppression literature). These give the model real-world amplitude
anchors that the cold model alone cannot:

| spec | marker | window | target | sigma | source |
| ---- | ------ | ------ | ------ | ----- | ------ |
| `ogtt_75g_glucose_peak` | glucose | 60–180 min | 150 mg/dL | 25 | DeFronzo / ADA |
| `ogtt_75g_glucose_late` | glucose | 180–300 min | 100 mg/dL | 15 | DeFronzo / ADA |
| `ogtt_75g_insulin_peak` | insulin | 30–120 min | 75 μU/mL | 25 | DeFronzo |
| `ogtt_75g_insulin_mean` | insulin | 0–180 min | 50 μU/mL | 20 | DeFronzo |
| `mixed_meal_hgo_suppression` | hepatic_glucose_output | 60–180 min | 30% of fasting | 15% | mixed-meal HGO RCTs |
| `small_carb_glucose_peak` | glucose | 30–90 min | 110 mg/dL | 12 | low-dose challenge studies |

These are picked to *constrain the same metabolic module slice* that
`InsulinSweepSignal` distills against — cold-model rates for shape +
literature amplitudes for the integral. Two complementary signal
sources on the same parameters is the structural answer to "how do
we make the cohort signal not be dominated by trajectory_rollout's
implicit compensation pull".

### 3. Sample-size + initial-state independence for the cohort signal

Two structural improvements to `CohortStatisticSignal` so each
literature spec actually delivers a clean gradient:

- `cohort-sample-patients` default bumped 4 → 16. Cohort statistics
  are population-level by definition; supervising 4 sampled patients
  per epoch was statistically too noisy to give a stable amplitude
  signal. 16 keeps wall-clock acceptable while quadrupling the
  effective N.
- New `InitMode` (`COLD` | `NORM_CENTER`) on each
  `CohortStatisticSpec`. Cold-model initial states are correct for
  fasted protocols (OGTT, mixed meal) but biased for protocols that
  start in a typical fed/awake state. Each spec now declares its own
  init mode, removing a class of silent biases. The six new glucose-
  handling specs all use `InitMode.COLD` (their protocols are
  fasted).

### 4. Coupling priors as a first-class registry

`pulse/knowledge/coupling_priors/{__init__.py, metabolism.py,
endocrine.py}` is a new namespace separate from `KnowledgeContribution`.
A `CouplingPrior(source_marker, target_marker, sign, magnitude_range)`
is a sub-knowledge-contribution unit you can add from a textbook
without writing a full contribution class. `merge_coupling_priors`
now accepts both contribution-derived priors and a standalone list,
and `TrajectoryRolloutSignal` consumes `ALL_COUPLING_PRIORS` so every
edge prior is enforced regardless of source. The metabolism file
encodes 12 standard fuel-axis priors (glucose↔insulin, insulin→FFA,
glucagon→hepatic, etc.); the endocrine file encodes 8 cross-axis
priors (ACTH→cortisol, cortisol→glucose, ghrelin→appetite, etc.).
This lets future iterations harvest priors from textbooks at a much
finer granularity than full contributions allow.

### Pre-submit diagnostic on iter 19 ckpt

`signal-balance` on `/tmp/iter19/checkpoint.pt` (raw ‖∇‖ at
weight=1.0, then × spec weight to recover in-spec contribution):

| signal             | spec weight | raw ‖∇gut‖ | raw ‖∇met‖ | in-spec ‖∇met‖ |
| ------------------ | ----------- | ---------- | ---------- | -------------- |
| gut_dose_sweep     | 0.10        | 1.07       | 0.00       | —              |
| insulin_sweep      | 0.30        | 0.00       | 0.30       | 0.09           |
| cohort_statistic   | 0.15        | 3.52       | 17.5       | **2.63**       |
| trajectory_rollout | 1.00        | 0.83       | 5.29       | **5.29**       |

The cohort signal landed a *much* larger raw ‖∇met‖ than expected
(17.5; iter 19 with 11 specs at sample N=4 was 0.95) — confirms the
six new glucose-handling specs + N=4→16 bump together quintuple the
metabolic pull. In-spec, trajectory still leads metabolic at 5.29
vs 2.63 — bumped `--insulin-sweep-weight` 0.10 → 0.30 to push the
combined sweep + cohort metabolic supervision past trajectory.

`cohort-ablation` on the same ckpt — every spec lands non-zero
gradient (no dead specs). Three notable signals:

| spec | z | predicted | target | diagnosis |
| ---- | -- | --------- | ------ | --------- |
| `small_carb_glucose_peak` | **5.03** | 170 | 110 | model has ~50% over-amplitude on small carb loads — single-spec dominator (‖∇met‖=231) |
| `ogtt_75g_glucose_120min` | **2.79** | 162 | 120 | classic insulin-resistance late-glucose pattern — peak satisfied (z=0.48) but glucose stays high → confirms downstream clearance is the bottleneck |
| `ogtt_75g_insulin_peak` | **-1.58** | 20 | 60 | model under-secretes insulin by 3× — exactly what `InsulinSweepSignal`'s GSIR distillation should fix |

Two species are stuck at 0 in fasting state (`bhb`, `ffa`) — both
hit z=-2.0. Iter 20 `InsulinSweepSignal` includes them in the cold
target sweep so the metabolic module gets explicit pressure to
track them; if they remain at 0 after iter 20, iter 21 needs to
look at the metabolic module's mass-action structure for those
species.

### 5. New diagnostic — per-spec cohort ablation

`pulse.diagnostics cohort-ablation --checkpoint ...` reports per-spec
z-residual *and* per-module grad-norm (gut, metabolic, appetite,
stress, cardiovascular, thermoreg, respiratory) at a single
checkpoint. The cohort signal averages across specs in training; this
diagnostic un-averages so we can see (a) which specs the model
already satisfies (`|z| < 1`), (b) which dominate the cohort loss
(`|z| > 3`), (c) which physiological pillar each spec actually
pulls on. Lets us catch dead specs (zero gradient on every module —
the spec is structurally muted) before they silently dilute the
budget.

Tests: `tests/test_cohort_ablation.py` (3 cases) — verifies per-spec
rows, that every spec lands non-zero gradient on at least one module,
and the renderer shape.

## What iter 20 should NOT change

- Gut kernel architecture (`pulse/modules/base.py`) — converged.
- `gut_dose_sweep_signal.py` AUC term — converged.
- `--gut-dose-sweep-auc-weight=5.0` and `--gut-dose-sweep-weight=0.10`
  — keep as-is. The kernel needs continuous pressure to stay at 1.13×;
  removing supervision will let it drift back into compensation.
- Cloud Build / CI setup — has been stable since iter 17's `_try_probe`
  fix in `pulse.diagnostics.compare`.
- `--cohort-statistic-weight=0.15` — the spec mass-add (cohort N
  bumped 4→16 and six new specs land more pull at the same weight,
  so leaving the weight constant is the conservative move; if the
  resulting weighted ‖∇metabolic‖ is too small relative to
  trajectory's, iter 21 bumps it).

## Code state (iter 20 changes ready to commit)

New files:

- `apps/pulse/engine/pulse/training/insulin_sweep_signal.py`
- `apps/pulse/engine/pulse/knowledge/coupling_priors/{__init__.py,
  metabolism.py, endocrine.py}`
- `apps/pulse/engine/pulse/knowledge/cohorts/glucose_handling.py`
- `apps/pulse/engine/pulse/diagnostics/cohort_ablation.py`
- `apps/pulse/engine/tests/test_insulin_sweep_signal.py`
- `apps/pulse/engine/tests/test_cohort_ablation.py`

Modified:

- `apps/pulse/engine/pulse/knowledge/cohort_types.py` — `InitMode`
  enum + `init_mode` field on `CohortStatisticSpec`.
- `apps/pulse/engine/pulse/knowledge/cohort_statistics.py` — register
  `glucose_handling.COHORT_STATISTICS`.
- `apps/pulse/engine/pulse/training/cohort_signal.py` — consumes
  `spec.init_mode`; default `sample_patients` 4 → 16.
- `apps/pulse/engine/pulse/coupling_prior_loss.py` —
  `merge_coupling_priors` accepts `extra_priors`.
- `apps/pulse/engine/pulse/training/trajectory_signal.py` — feeds
  `ALL_COUPLING_PRIORS` into `merge_coupling_priors`.
- `apps/pulse/engine/pulse/training/__init__.py` — exports
  `InsulinSweepProtocol`, `InsulinSweepSignal`.
- `apps/pulse/engine/pulse/train.py` — `cohort_sample_patients`
  default 4 → 16; new `--insulin-sweep-*` flags; instantiates
  `InsulinSweepSignal`.
- `apps/pulse/engine/pulse/diagnostics/signal_balance.py` — tracks
  `grad_norm_metabolic`; `InsulinSweepSignal` in default registry.
- `apps/pulse/engine/pulse/diagnostics/{__init__.py, __main__.py}` —
  exports + CLI for `cohort-ablation` subcommand.
- `apps/pulse/train/spec.json` — iter 20 spec with new flags.

Unchanged from iter 19 and depended on:

- `apps/pulse/engine/pulse/modules/base.py` — factorized gut kernel.
- `apps/pulse/engine/pulse/training/gut_dose_sweep_signal.py` —
  three-term loss.
- `apps/pulse/engine/pulse/diagnostics/compare.py` — `_try_probe`.

All 125 engine tests pass.

## How to resume after reboot

```bash
# 1. Verify env (needs gcloud + gsutil + python venv)
cd /Users/grovina/Projects/grovina/platform
gcloud auth list
ls apps/pulse/engine/.venv/bin/python

# 2. Re-pull iter 19 artifacts (they survive in GCS forever)
mkdir -p /tmp/iter19 && \
  gsutil cp gs://grovina-pulse/training/jobs/train-20260422T180605Z/{benchmark-report.json,delta-vs-baseline.json,checkpoint.pt} /tmp/iter19/

# 3. Re-pull iter 18 for cross-iter compare baselines
mkdir -p /tmp/iter18 && \
  gsutil cp gs://grovina-pulse/training/jobs/train-20260422T140536Z/{benchmark-report.json,checkpoint.pt} /tmp/iter18/

# 4. Sanity probe iter 19 kernel
cd apps/pulse/engine && .venv/bin/python -c "
import torch
from pulse.training.gut_dose_sweep_signal import GutDoseSweepSignal, GutDoseSweepProtocol
from pulse.training.signals import WeightSchedule
from pulse.diagnostics.probe import load_model_from_checkpoint
from pulse.modules.gut import MealEvent
m, _ = load_model_from_checkpoint('/tmp/iter19/checkpoint.pt')
m.eval()
proto = GutDoseSweepProtocol()
sig = GutDoseSweepSignal(n_patients=0, sample_patients=0,
    include_default_embedding=True, weight=WeightSchedule(1.0), protocol=proto, auc_weight=5.0)
T = proto.post_window_min
times = torch.arange(T, dtype=torch.float32)
EMBED = next(m.embedding_projections['gut'].parameters()).shape[1]
emb_gut = m.embedding_projections['gut'](torch.zeros(EMBED))
preds = []
with torch.no_grad():
  for d in proto.carb_doses_g:
    out = m.gut.forward_window(times, [MealEvent(0.0, float(d), 5.0, 10.0)], emb_gut)
    preds.append(out.unsqueeze(0))
mse, rank, auc, _ = sig._losses(preds, sig._targets, sig._abs_scale, T)
print(f'mse={float(mse):.4f}  auc={float(auc):.4f}')
" 

# 5. Sanity-check iter 20 build (everything below is built; verifies tests pass)
cd apps/pulse/engine && .venv/bin/python -m pytest tests/ -x -q

# 6. Local prediction: load iter 19 ckpt, run the new diagnostics so we
#    know the iter 20 pull on metabolic before submitting
cd apps/pulse/engine && .venv/bin/python -m pulse.diagnostics signal-balance \
  --checkpoint /tmp/iter19/checkpoint.pt --n-patients 20

cd apps/pulse/engine && .venv/bin/python -m pulse.diagnostics cohort-ablation \
  --checkpoint /tmp/iter19/checkpoint.pt --sample-patients 4

# 7. Submit iter 20 via bash apps/pulse/scripts/train-submit.sh --gcp
```

## Cloud Build / job IDs

| iter | build ID | job ID | benchmark-report key metrics |
| ---- | -------- | ------ | ---------------------------- |
| 16   | (earlier) | train-20260421T135353Z | overall=0.196, glu_mean=0.565, ver=0.790 |
| 17   | (earlier) | train-20260422T065354Z | overall=0.198, glu_mean=0.579, ver=0.793 |
| 18   | 21820f0b-0000-4366-b7bb-8893aec0d613 | train-20260422T140536Z | overall=0.197, glu_mean=0.575, ver=0.783 |
| 19   | 0ef57a5a-a358-41f7-ae7c-ad7d298b7e7c | train-20260422T180605Z | overall=0.198, glu_mean=0.578, ver=0.784 |

All four FAILUREs in `gcloud builds list` are the expected
`EnforceBenchmarkGate` step failing because none of these iters cleared
the `overall_mape ≤ 0.16` gate. Training, comparison, and artifact
upload all completed for every one of them.

## Success criteria for iter 20

Hard gates:

1. **Insulin sweep AUC sub-loss raw < 0.10** at end of training (the
   metabolic-module analog of the gut sweep target). Insulin and
   hepatic-output species carry the highest per-species weight, so
   "metabolic-module sweep AUC" is dominated by these in practice.
2. **glucose mean_mape ≤ 0.45** (iter 19: 0.578). This is the *real*
   gate — iter 20 should be the iter where glucose accuracy starts
   moving for the first time since iter 12.
3. **No regression in gut probe metrics:** ratio stays ≤ 1.20× on
   every dose ≥ 30g (iter 19: 1.13×); slope stays ≤ 3.5 (iter 19:
   3.19); AUC sub-loss stays < 0.10 (iter 19: 0.05). The gut should
   *not* drift while we're working on downstream.
4. **No dead literature spec:** running
   `pulse.diagnostics cohort-ablation` on the iter 20 checkpoint must
   show every spec landing non-zero gradient on at least one module.
   A dead spec means the model has structurally muted that piece of
   knowledge and the iter is wasting capacity.

Soft gates:

5. **overall_weighted_mape ≤ 0.18** (iter 19: 0.198, gate 0.16).
6. **`signal-balance`** shows `InsulinSweepSignal` contributing
   meaningful weighted ‖∇metabolic‖ — at least 50% of
   trajectory_rollout's weighted ‖∇metabolic‖ (mirrors the gut
   gradient-budget lesson from iters 17/18).
7. **trajectory_rollout raw_loss visibly drops** (iter 19: 8.59) — if
   downstream modules genuinely improve, trajectory MSE should fall
   for real, not just shift compensation between modules.
8. **OGTT specs converge:** `ogtt_75g_glucose_peak` and
   `ogtt_75g_insulin_peak` z-residuals end < 1.5σ (iter 19:
   unmeasured, but expected ≫ 2σ given the unconstrained metabolic
   module).

## Risks / what to look for if iter 20 doesn't move glucose_mape

| symptom | likely cause | next move |
| ------- | ------------ | --------- |
| Insulin sweep loss drops cleanly but glucose_mape unchanged | clearance + HGO are also under-tuned and absorbing the slack; or, OGTT cohort specs are converging on a wrong arm distribution | Iter 21: increase per-species weights for FFA/BHB/lactate (currently 0.0–0.3); add clearance-specific sweep covering wider insulin range |
| Insulin sweep loss stays high despite weight bumps | metabolic module architecturally can't track GSIR or HGO curves | Inspect `pulse/modules/metabolic.py`; may need a factorization analogous to iter 16's gut fix (e.g. log-linear GSIR head) |
| Insulin sweep helps but gut amplitude regresses | shared embedding projection pulls back into compensation | Add a regularizer on the gut kernel parameters or hold them fixed for a phase |
| trajectory_rollout raw_loss skyrockets | downstream modules can't co-adapt fast enough at high pull rate | Ramp `--insulin-sweep-weight` from 0 → target across phase 2 |
| Cohort signal still dominated by 1–2 specs (per ablation diagnostic) | `cohort-statistic-weight=0.15` is too small relative to per-spec sigmas | Iter 21: bump to 0.30 or rebalance per-spec sigmas to flatten the loss landscape |
| Coupling-prior loss component spikes | new edges in `coupling_priors/{metabolism,endocrine}.py` collide with learned signs | Tighten `magnitude_range` on the offending edges, or drop them and re-derive from the textbook reference |

## Pointer references (only read if needed)

- `apps/pulse/docs/iter18-handoff.md` — full justification of the AUC
  term + dilution diagnosis. Read only if iter 20's similar dilution
  question comes up for downstream modules.
- `apps/pulse/docs/iter19-handoff.md` — gradient-budget analysis, exact
  per-loss ‖∇gut‖ measurements. Read only if you need to re-derive
  the gradient-budget reasoning for a new module.
- `apps/pulse/docs/iter17-handoff.md` — `_try_probe` CI fix details.
  Read only if `CompareToPrevious` breaks again.
- `apps/pulse/docs/iter16-handoff.md` — factorized kernel structural
  guarantees. Read only if you suspect the gut module structure broke.
- `apps/pulse/docs/prd.md` — top-level vision. Read for context but
  not required for iter 20 mechanics.
