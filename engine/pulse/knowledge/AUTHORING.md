# Encoding medical knowledge as gradient signals

This guide is for the human (or agent) sitting down to absorb a new
piece of literature into Pulse. The goal is to turn a paper, textbook
chapter, or RCT result into something the model can be supervised
against — i.e. a differentiable loss landed on the right module.

The core principle: **every form of medical knowledge is a gradient
signal**. Architecture, priors, training data, and cohort statistics
are all the same thing on different time-scales — they shape the
optimization landscape so the model converges to a physiology-
respecting fixed point. Each of the surfaces below is the place to
put a particular *granularity* of knowledge.

## Decision tree: where does this piece of knowledge go?

Start with the smallest, cheapest unit that captures the finding.
Bigger units cost more to author and to train against; smaller units
compose better.

```
Is the finding "X causes Y to go up/down by some plausible amount"?
└─ Yes → CouplingPrior (knowledge/coupling_priors/*.py)

Is the finding "in cohort C, marker M had statistic S over window W
(with reported sigma)"?
└─ Yes → CohortStatisticSpec (knowledge/cohorts/*.py)

Is the finding "in this scenario, marker M follows roughly this
trajectory shape"?
└─ Yes → TextbookScenario / FlowStory
   (knowledge/textbook_scenarios/*.py)

Is the finding "module M's local input→output mapping should match
this analytical reference at these grid points"?
└─ Yes → A new *SweepSignal* (training/<module>_sweep_signal.py)
   modeled on gut_dose_sweep_signal.py and insulin_sweep_signal.py.

Is the finding "this physiological subsystem needs a structural
constraint the network can't learn from data alone"
(zero-at-zero, monotonicity, mass conservation, etc.)?
└─ Yes → A factorized module head in pulse/modules/<module>.py.
   See modules/base.py for the gut kernel — it makes
   zero-at-zero / dose-linearity / monotonicity *analytic*.
```

## Surface 1 — CouplingPrior

**Use for:** any "X up → Y up" or "X up → Y down" relationship from a
textbook, where you can give a plausible magnitude range. One line.

**Where:** `pulse/knowledge/coupling_priors/<system>.py`. Add to the
right system file (`metabolism.py`, `endocrine.py`, …) or create a
new one and register it in `coupling_priors/__init__.py`.

**Template:**

```python
from ..base import CouplingPrior

def _p(src: str, tgt: str, sign: int, mag: tuple[float, float]) -> CouplingPrior:
    return CouplingPrior(source_marker=src, target_marker=tgt,
                         sign=sign, magnitude_range=mag)

COUPLING_PRIORS: list[CouplingPrior] = [
    # cite source in a comment
    _p("glucose", "insulin", +1, (0.001, 0.02)),  # GSIR
    _p("insulin", "glucose", -1, (0.005, 0.05)),  # peripheral disposal
    # ...
]
```

**Sign conventions:** `+1` means "increase in source ⇒ increase in
target dynamics rate". The `magnitude_range` is the absolute value
the learned coupling weight is allowed to occupy; pick conservative
bounds from the textbook. Tighter ranges help; absurd ranges
mute the prior.

**Cost:** ~1 minute. **Power:** weak; just shapes the slope of an
edge in the coupling graph. Use lots of these.

## Surface 2 — CohortStatisticSpec

**Use for:** quantitative findings of the form "in N people doing
protocol P, marker M had statistic S (mean / peak / AUC / late) over
window W, with sigma σ". This is the primary surface for absorbing
RCTs, observational cohorts, and standard textbook tables.

**Where:** `pulse/knowledge/cohorts/<topic>.py`. Add to an existing
topic or create a new file and register it in `cohort_statistics.py`.

**Template:** copy `cohorts/glucose_handling.py` and adapt.

```python
from ..cohort_types import (
    CohortArmSpec, CohortStatisticSpec, InitMode,
    StatisticKind, StatisticWindow,
)

_OGTT_75G_MEALS = ((0.0, 75.0, 0.0, 0.0),)  # (t_min, carbs_g, protein_g, fat_g)

OGTT_GLUCOSE_PEAK = CohortStatisticSpec(
    name="ogtt_75g_glucose_peak",
    source="DeFronzo (1979); ADA / WHO OGTT criteria",
    description="75 g OGTT → glucose peak ≈ 150 mg/dL at 30–60 min post-load",
    arms=(
        CohortArmSpec(label="ogtt_75g", duration_min=300,
                      start_hour=8.0, meals=_OGTT_75G_MEALS),
    ),
    marker_id="glucose",
    kind=StatisticKind.PEAK_VALUE,
    window=StatisticWindow(start_min=60, end_min=180),
    target=150.0,
    sigma=25.0,
    init_mode=InitMode.COLD,  # OGTT is fasted
)

COHORT_STATISTICS: list[CohortStatisticSpec] = [OGTT_GLUCOSE_PEAK, ...]
```

**Choosing `sigma`:** the literature-reported SD if you have it,
otherwise estimate. `sigma` controls how hard the gradient pulls
when the model is off — too small and one spec dominates the cohort
signal; too large and the spec contributes essentially nothing.
Aim for `sigma` ≈ 10–20% of `target` as a first cut, then look at
per-spec z-residual via `pulse.diagnostics cohort-ablation` after
training and adjust.

**Choosing `init_mode`:** `COLD` for fasted protocols (OGTT, mixed-
meal, overnight fast → challenge); `NORM_CENTER` for protocols that
should start at a typical fed/awake state.

**Choosing `window` and `kind`:** `kind` is one of the
`StatisticKind` enum values (`PEAK_VALUE`, `MEAN_VALUE`,
`AUC_OVER_BASELINE`, etc.); `window` is the time range over which to
compute it, both relative to start of the arm.

**Cost:** ~5 minutes per spec. **Power:** strong amplitude pull on
whichever module owns the marker. Use *highly-cited, quantitative*
findings — one good spec beats ten low-quality ones.

**After authoring:** run
`pulse.diagnostics cohort-ablation --checkpoint <ckpt>` to verify the
new spec lands non-zero gradient on at least one module. A dead spec
(zero on every module) means the model has muted that knowledge —
fix it now, not three iterations later.

## Surface 3 — TextbookScenario / FlowStory

**Use for:** qualitative or semi-quantitative trajectory shapes —
"after a meal, glucose rises, then falls; insulin lags glucose by ~10
min; ketones rise after 12h fasting". Verifier checks (passed/failed)
are the natural target. Encoded as Python checks against simulated
trajectories.

**Where:** `pulse/knowledge/textbook_scenarios/*.py`. Existing files
are organized by domain (`metabolic.py`, `sleep_circadian.py`,
`exercise.py`, …). The flow-story format
(`textbook_scenarios/flow_story_protocol.py`) is the more structured
variant where each phase has named landmarks the verifier scores.

**Cost:** ~30 minutes per scenario. **Power:** medium — pulls on
shape rather than amplitude. Best paired with cohort specs that pin
the amplitude.

## Surface 4 — Probe-style sweep signals (`*SweepSignal`)

**Use for:** when you can write down an analytical reference for a
module's *local* input→output mapping, and the existing
trajectory/cohort signals don't pull strongly enough on that mapping.
Two examples in the codebase:

- `gut_dose_sweep_signal.py` — gut kernel vs analytical absorption
  curve at an explicit dose grid.
- `insulin_sweep_signal.py` — metabolic module rates vs Bergman-style
  cold-model curves at an explicit (glucose, insulin) grid.

**When to add a new one:** when the diagnostic pattern from iters
17–19 repeats — a converged-looking module that the headline metric
says is still wrong, with `signal-balance` showing trajectory pull
out-massing every other signal on that module. That's the signature
of compensatory equilibrium and the targeted fix is a sweep signal.

**Template:** copy `insulin_sweep_signal.py` (it's the most general
example; gut is simpler). The three-term loss
(`mse + ranking_weight·rank + auc_weight·integral`) is the recipe
that survived iters 17→19 on the gut.

**Cost:** ~half a day per module. **Power:** very strong, surgical.
Use sparingly and only after the diagnostic says you need it.

## Surface 5 — Module factorization

**Use for:** structural constraints the network can't learn from data
alone (zero-at-zero, mass conservation, sign restrictions,
monotonicity in a specific argument). Iter 16's gut kernel is the
canonical example: instead of penalizing the network for violating
zero-at-zero, the kernel is *factorized* so zero-at-zero is analytic.

**Where:** `pulse/modules/<module>.py`. Look at `modules/base.py`
(GutKernel) and `modules/metabolic.py` (mass-action structure) for
the patterns.

**Cost:** ~1–2 days per factorization. **Power:** absolute (the
property becomes mathematically guaranteed, not learned). Use when a
property is non-negotiable and the optimizer keeps fighting it.

## After authoring: how to know it worked

For all surfaces:

1. **Tests:** add a unit test that exercises the new contribution.
   For sweep signals and factorizations the bar is "gradient flows;
   loss decreases on a toy training loop". For cohort specs the bar
   is "the spec is registered in `ALL_COHORT_STATISTICS` and the
   signal can compute its z-residual".
2. **Local diagnostic before submit:**
   - For new sweep signals: run `signal-balance` on the previous
     iter's checkpoint; verify the new signal's weighted ‖∇module‖
     is comparable to or larger than trajectory_rollout's.
   - For new cohort specs: run `cohort-ablation` on the previous
     iter's checkpoint; verify the new spec lands non-zero gradient.
   - For new coupling priors: run a short training loop (≥5 epochs)
     and verify the coupling-prior loss component doesn't spike.
3. **After training:** check the same diagnostics on the new
   checkpoint. The signal should have moved its target toward
   convergence; the per-module gradient should have shifted in the
   intended direction.

## What *not* to do

- **Don't** add five low-quality cohort specs to "increase coverage".
  One high-σ spec on a contested marker pulls the model in random
  directions and dominates the cohort loss budget. Quality > quantity.
- **Don't** copy a `magnitude_range` from one coupling prior to
  another without checking units. The learned coupling lives in a
  rate-of-change space; a magnitude that's right for `glucose →
  insulin` is usually wrong for `cortisol → glucose`.
- **Don't** add a new sweep signal "just in case". They're expensive
  and only help when the trajectory loss is structurally too weak on
  a module. Use the diagnostics first.
- **Don't** factorize a module without first trying a sweep signal
  + coupling priors. Factorization is irreversible and constrains
  future iterations.
