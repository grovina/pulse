# Pulse Iter 16 — Resume Handoff

Self-contained resume point. Read only this file to pick up the work.

## Where we are

**Iter 16 is staged.** Code changes are committed; the Cloud Build
submission is the next action.

- Spec: `apps/pulse/train/spec.json`.
- Cloud Build config: `apps/pulse/train/cloudbuild.yaml`.
- Submission: `bash apps/pulse/scripts/train-submit.sh --gcp` from repo root.
- Job ID assigned at submit time. Artifact root:
  `gs://grovina-pulse/training/jobs/<JOB_ID>/`.

## What iter 15 did and why it wasn't enough

Iter 15 added a dose-ranking term to `GutDoseSweepSignal` and restored
the gut weights (0.02 → 0.10 for both `gut-dose-sweep-weight` and
`gut-loss-weight`) after iter 14's collapse. The ranking term penalizes
`relu(rank_margin · cold_AUC_gap − pred_AUC_gap)` for every dose pair.
Result was substantial-but-partial:

| metric                  | iter 14 | **iter 15** | Δ           |
| ----------------------- | ------- | ----------- | ----------- |
| glucose_mape            | 0.6246  | **0.5220**  | −0.103 ✓    |
| overall_mape            | 0.2150  | **0.1952**  | −0.020 ✓    |
| fasting drift / 4h      | +60.6   | **+53.0**   | −7.6 ✓      |
| verifier_overall        | 0.6907  | 0.6742      | −0.017 ✗    |
| verifier_cat[meal]      | 0.6428  | 0.6030      | −0.040 ✗    |

Probe at zero embedding showed the inversion was *almost* gone (single
1% wobble at 30→45 g vs iter 14's 102→38 collapse), and signal-balance
showed the kernel was responsive again (∇gut from gut_dose_sweep
0.27 → 0.72; from trajectory 0.97 → 4.14). Glucose_mape made the
biggest single-iter improvement since iter 12.

But iter 15's probe revealed two persistent pathologies the ranking
term cannot reach:

| dose | cold target | iter 15 actual | shortfall |
| ---- | ----------- | -------------- | --------- |
| 0    | 0           | **40.5**       | +40 (chronic baseline offset) |
| 30   | 85          | 97.2           | +12       |
| 45   | 127         | 96.1           | −31       |
| 60   | 169         | 101.8          | −67       |
| 90   | 254         | 125.0          | −129      |
| 120  | 339         | 130.5          | −209 (saturated) |

1. **Chronic nonzero baseline.** AUC at zero macros has been ~40-65 mg-min
   across iters 13/14/15. Cold target says 0 (no carbs ⇒ no glucose
   appearance). MSE has been bleeding loss on this row for a dozen iters.
2. **Severe saturation past 30 g.** The kernel can produce ~130 AUC at
   any dose ≥ 30 g and physically cannot grow further.

## Root cause: the kernel's MLP architecture

`pulse/modules/base.py::GutModuleBase` (pre-iter-16) was structured as

```python
self.kernel = nn.Sequential(
    nn.Linear(4 + emb_dim, 32),  # input = (carbs, fats, proteins, t/60, emb)
    nn.Tanh(),
    nn.Linear(32, 32),
    nn.Tanh(),
    nn.Linear(32, 4),
)
appearance = softplus(raw[..., :3])
nutrient_flag = sigmoid(raw[..., 3:4])
```

Two architectural defects pop out:

- **Saturation.** Carbs enter raw and unnormalized (range 0–120 g). The
  first `Tanh()` clips them: with weights ~0.02 a 60 g meal already
  pushes Tanh to ~0.77; a 120 g meal sits at ~0.96. Past ~30 g the
  kernel cannot tell doses apart.
- **Baseline offset.** First `Linear` has a bias term. At
  macros=(0,0,0) the kernel still produces softplus(...) > 0 because
  the bias drives non-zero pre-activations. MSE can only push the
  baseline down — never to zero — because doing so harms non-zero-dose
  responses (they share weights and biases).

Both are *structural*, not weight-tuning, problems. No combination of
loss weights or training schedule can produce
`appearance(macros=0) ≡ 0` from a biased MLP.

## What iter 16 changes

Single fundamental change to `GutModuleBase`: factorize the kernel.

```python
# MLP conditions ONLY on (t, embedding). Macros enter as a multiplier.
input_dim = 1 + embedding_dim
output_dim = N_MACROS * N_APPEARANCE + 1  # 3*3 response matrix + 1 flag logit

self.kernel = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.Tanh(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.Tanh(),
    nn.Linear(hidden_dim, output_dim),
)

K = softplus(raw[..., :9].reshape(..., 3, 3))      # per-gram response matrix
appearance = einsum("...m,...mn->...n", macros, K)  # 3 channels

# nutrient_flag: sigmoid head, gated by analytic presence factor
total_macro = macros.sum(dim=-1)
presence = 1 - exp(-total_macro / 10g)              # 0 at zero, ~1 at >30g
nutrient_flag = sigmoid(raw[..., 9]) * presence
```

This guarantees, **analytically (not via SGD)**:

| property                              | before | after |
| ------------------------------------- | ------ | ----- |
| `appearance(macros=0) == 0`           | learned (broken) | **analytic ✓** |
| `appearance(α·macros) == α·appearance(macros)` | learned (broken) | **analytic ✓** |
| dose-monotonicity at every (t, emb)   | learned + ranking penalty | **analytic ✓** |
| `nutrient_flag(macros=0) == 0`        | learned | **analytic ✓** |

Param count is essentially unchanged (input shrinks by 3, output grows
by 6). The ranking term in `GutDoseSweepSignal` becomes redundant for
the appearance channels but stays as defense-in-depth — its loss
component should hover near 0 throughout training (a healthy
diagnostic).

The kernel's job collapses to its only legitimate one: learn a per-gram,
time-and-embedding-conditional response shape. Linearity, zero-at-zero,
and monotonicity are free properties.

## Iter 16 spec changes

`apps/pulse/train/spec.json`:

- Hypothesis & expectedEffect rewritten to describe the structural
  change.
- Hyperparams unchanged from iter 15. `gut-dose-sweep-weight=0.10`,
  `gut-loss-weight=0.10`, `ranking_weight=1.0`, `rank_margin=0.3`.

## Code changes (commit-level)

1. **`apps/pulse/engine/pulse/modules/base.py`** —
   `GutModuleBase` rewritten in factorized form. Public signature
   `forward_single_meal(macros, time_since_meal, embedding) -> [..., 4]`
   unchanged; both call sites
   (`apps/pulse/engine/pulse/modules/gut.py:102, :165`) work as before.
2. **`apps/pulse/engine/tests/test_gut_module.py`** — new
   `TestGutKernelStructuralProperties` class with six tests pinning
   down zero-at-zero, dose-linearity (1×/2× and arbitrary factor),
   nutrient_flag presence-gating, and end-to-end forward_window dose
   proportionality.

No call site, no other test, no other signal needed any change. The
factorized kernel is a strict drop-in.

## Local sanity (fresh-init kernel, before training)

```
=== Fresh-init dose sweep at zero embedding ===
  carbs   AUC_glucose   monotone
      0        0.0000   ✓
     15     7839.21     ✓
     30    15678.42     ✓
     45    23517.63     ✓
     60    31356.84     ✓
     90    47035.27     ✓
    120    62713.69     ✓
    240   125427.38     ✓
=== Linearity check ===
  AUC(120g) / AUC(30g) = 4.000000  (expected: 4.000000)
=== Zero-dose check at random embedding ===
  empty meal list: max|out| = 0.0000000000
```

Magnitudes are large at fresh init (random weights), but every
structural property holds *exactly*. SGD now only has to learn the
per-gram time-shape; it cannot break dose-linearity or zero-at-zero.

## Success criteria for iter 16

Hard gates (these must hold or iter 16 has missed):

1. **Probe at zero embedding** (`pulse.diagnostics probe`):
   - AUC(0g) ≤ 1.0 (was 40.5 at iter 15; should be near machine zero
     since masking outside the meal-active window is exact).
   - Strictly monotone across all probed doses.
   - AUC(120g) / AUC(30g) ratio ≈ 4.0 ± 0.05.
2. **glucose_mape ≤ 0.30** (was 0.52 at iter 15; iter 12 was ~0.30).
3. **fasting drift ≤ +30 mg/dL** (was +53 at iter 15).
4. **`gut_dose_sweep` rank-loss component ≈ 0** throughout training
   (visible in epoch logs as `gut_sweep=...(rank=...)` if the signal
   logs sub-metrics; otherwise pull from the per-epoch report).

Soft gates (would be nice):

- **overall_mape ≤ 0.16** to clear the benchmark gate.
- **verifier_cat[meal] ≥ 0.65** (was 0.603 at iter 15).
- Wall-clock similar to iter 15 (~3.5 h).

## What to look at if iter 16 fails

| symptom                              | likely cause                       | next move |
| ------------------------------------ | ---------------------------------- | --------- |
| AUC(0g) > 1                          | Window-mask bug elsewhere; check `forward_window` mask | Read `pulse/modules/gut.py:148-167` |
| Linearity violated in probe          | Bug in einsum or softplus reshape  | Read `pulse/modules/base.py::forward_single_meal` |
| glucose_mape stays high (>0.4)       | Per-gram K matrix isn't reaching cold-target slope | Read training logs; check `gut_sweep` MSE component |
| verifier_meal regresses              | Cross-macro coupling matters more than expected | Add a small coupling MLP: `K += K_couple(t, emb, macros)` |

The factorized form removes the kernel's *capacity* to encode
cross-macro non-linear effects (e.g. fat slowing carb absorption). Cold
targets assume linear superposition, so this matches our supervision.
If iter 16 reveals a residual MSE that the linear form provably can't
reach, iter 17 would add a small additive coupling head.

## Diagnostic commands

```bash
# Pull iter 16 artifacts
gsutil cp gs://grovina-pulse/training/jobs/<JOB_ID>/{benchmark-report.json,delta-vs-baseline.json,checkpoint.pt} /tmp/iter16/

# Probe the trained kernel
cd apps/pulse/engine && .venv/bin/python -m pulse.diagnostics probe \
  --checkpoint /tmp/iter16/checkpoint.pt

# Signal-balance (gradient norms per signal)
.venv/bin/python -m pulse.diagnostics signal-balance \
  --checkpoint /tmp/iter16/checkpoint.pt

# Cloud Build status / logs
gcloud builds list --project=grovina --limit=5
gcloud builds log <BUILD_ID> --project=grovina
```
