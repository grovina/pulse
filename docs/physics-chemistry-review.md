# Physics & chemistry review (2026-06-12)

A first-principles pass over the encoded physics/chemistry, in the Feynman
sense: does the math actually do what it claims, and do the resting states the
generators produce match the values we say are typical? **No code was changed**
— this is a findings list for the maintainer to act on with the benchmark in
the loop. Measurements below are from the `pulse-model.pt.last-good.pt` model and
the four active knowledge generators.

## What holds up

- **Mass action on normalized state is honest.** `MassActionModule` computes
  `prod − cons·state` on the *z-scored* state, which at first looks like it
  breaks the "consumption ∝ concentration" claim (the term goes negative below
  typical). Worked through, it is algebraically identical to genuine mass action
  `rate = P_eff − k_eff·[X]` with `P_eff, k_eff ≥ 0` — a reparameterization, not a
  violation. The docstring is truthful.
- The gut kernel enforces its invariants structurally (zero appearance at zero
  dose; dose-linearity) rather than hoping SGD learns them — the right call.
- HPA cascade ordering (ACTH leads cortisol), insulin-suppressed lipolysis and
  ketogenesis as Hill terms, and the tissue-split glycogen kinetics are sound.

## Findings, by confidence

### 1. Standalone Bergman generator is stale vs `full_body` (clear, verified)

`knowledge/bergman_glucose_insulin.py` still carries pre-iter-21 constants that
`full_body` was explicitly recalibrated away from. Both feed the same marker pool,
so they now contradict each other on glucose/insulin/BHB.

| const | bergman (stale) | full_body (iter-21) | consequence of the stale value |
|---|---|---|---|
| `h` (secretion threshold) | 80 | 95 | `h < Gb=95` keeps glucose-stimulated secretion **on at fasting**, so insulin can't fall below `Ib` during a fast |
| `gamma` (β-cell gain) | 0.015 | 0.07 | OGTT insulin peak ~5× too weak vs DeFronzo (~60 µU/mL) |
| `Si` | 0.0002 | 0.0004 | looser late-glucose clearance |
| `k_bhb` | 0.03 | 0.005 | BHB clears ~6× too fast |
| `IC50_keto` | 10 | 15 | over-suppressed ketogenesis |

Verified: a pure fast at normal `Gb=95` pins insulin at **11.5 µU/mL** with the
stale params vs **9.99** (relaxes to `Ib`) with the iter-21 params. Overnight BHB
settles ~0.04 vs the ~0.2 mmol/L the cold model targets.

This is the one finding that's a bug rather than a taste call — you wouldn't
*deliberately* keep ketone clearance 6× off. Looks like the standalone simply
wasn't updated when the cold model was. Fix: port the five constants. Low risk
(no test pins these; both generators converge rather than diverge afterward).

### 2. Resting baselines sit below their stated typicals (mild, both generators agree)

A recurring shape: a hormone is suppressed by the **absolute** level of another
hormone, so even *basal* levels suppress it and the resting pool lands below its
typical.

- **Glucagon**: `glucagon_supp = 0.5·I/(Ib+10)` (in both `full_body` and bergman).
  Basal insulin already suppresses, so glucagon rests at **~55 pg/mL** (measured,
  overnight) vs typical **70** — and an overnight fast, where glucagon should be
  at or above baseline, reads ~20% low.
- **Ghrelin**: rests at **~80 pg/mL** (measured, both generators) vs typical **100**.

The honest form is suppression by *above-basal* drive, `max(I − Ib, 0)`, so the
resting pool sits at its typical with zero exogenous drive. Note: this also
answers the "don't reference a baseline" objection from the side of the *rate
law* — `max(I−Ib,0)` keeps insulin suppressing monotonically at all levels; it's
the choice of stimulus (absolute vs above-basal), not a switch-off threshold.
The cohort targets are *deltas*, so they tolerate the offset; whether to fix
depends on whether absolute baselines matter to you. Shifts tuned data → wants a
benchmark check.

**(Correction to an earlier claim:** I initially reported ghrelin diverging
*between* generators, 71 vs 100. That was a measurement error — I sampled a
post-breakfast window, not the fasted one. Measured properly, both generators
agree at ~80. The real issue is the uniform ~20% offset from typical, not a
cross-generator conflict.)**

### 3. Cortisol circadian shape (worth a look)

Measured overnight (02:00–05:00, near the physiological *nadir*) cortisol is
**~17 µg/dL** — should be ~3–5. Two causes, both shared by `full_body` and
`cortisol_circadian`:

- A single cosine peaked at 08:00 has its minimum at 20:00, so it can't represent
  the real asymmetric rhythm (sharp morning rise, slow decline, **midnight**
  trough). The nadir is misplaced by ~12 h.
- `dCort = −k_cort·(Cort − cort_target) + k_acth_to_cort·ACTH` relaxes toward the
  *full* circadian `cort_target` **and** adds an ACTH production term on top —
  double-counting the drive (~+4 µg/dL). Either `cort_target` should be the
  ACTH-independent component (lower), or relaxation should be to a flat floor.

### 4. Cyclic-time discontinuity (structural, model-level)

`modules/base.py:compute_time_features` emits `(t_hours − 12)/12` as a linear
ramp. Since time is taken mod 1440, that feature jumps **+1 → −1 at midnight** — a
non-physical kink injected into every module's dynamics at 00:00. The sin/cos pair
already identifies the phase smoothly and unambiguously, so the ramp is redundant
*and* discontinuous. Changing it needs a retrain (it's a model input), so this is
a flag, not a quick fix.

### 5. Glycogen↔glucose non-conservation (already acknowledged in-code)

`full_body.py` (~line 457) documents that the glycogenolytic glucose flux is
lumped in `Hep` while the glycogen pools deplete independently — carbon isn't
conserved between the pools and blood glucose yet. Confirming it's real and a
known next-iter gap, not an oversight.

## Suggested order of operations

1. **Bergman staleness** — clear win, low risk. Port the five constants.
2. **Above-basal suppression for glucagon/ghrelin** — clean structural fix, but
   shifts tuned baselines → run the gate before/after.
3. **Cortisol nadir + time discontinuity** — real but need a retrain to evaluate.
