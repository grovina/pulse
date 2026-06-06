# Pulse modeling state — completeness map

*Written iter 74 (2026-06-02). Consolidates what was scattered across
`prd.md`, `architecture-roadmap.md`, `physiology-coverage.md`,
`dead-pathways.md`, `multi-timescale-plan.md`. Read this first; it is the
map. The per-iter chronicle lives in `train/spec.json` (the `hypothesis`
field of each committed iter).*

## The two-layer ceiling

"How complete is the modeling?" has two answers, because there are two
models:

1. **The cold knowledge model** (`engine/pulse/knowledge/full_body.py`) —
   the mechanistic ODE simulator that is the *supervision target*. This is
   **fairly complete**: 19 of 23 markers have real differential equations
   with ~30+ coupled cross-effects (Bergman glucose-insulin, HPA cascade,
   lipolysis→ketogenesis, activity→lactate→RR, sleep→all vitals),
   literature-calibrated (DeFronzo OGTT, Wolever, cosinor cortisol). Four
   states are *padded with constants*, not simulated: `liver_glycogen`,
   `muscle_glycogen`, `mitochondrial_capacity`, `crh`. The cold model is the
   ceiling — the learned model cannot be more mechanistically coherent than
   its teacher.

2. **The learned model** (`engine/pulse/model.py`, the modular neural ODE) —
   distilled from the cold model + textbook scenarios + cohort statistics +
   physiology rules. Its completeness is gated not by the teacher but by
   **which markers the supervision actually reaches with gradient.** This is
   the real frontier, and it is uneven.

## Where the eval actually looks (corrected)

A recurring misconception (caught iter 74): the benchmark does **not**
evaluate at the zero embedding. `benchmark.py` calibrates a per-episode
embedding (512 Adam steps) or falls back to a non-zero seeded vector. The
**gate** scores only `glucose` and `hr` (≤0.20, ≤0.15); `per_marker` /
`overall_weighted_mape` score ~14 markers including the hormones;
`hrv/rr/spo2` are **not scored at all**. The zero embedding still matters
because *textbook scenarios* query it (so it drives `textbook_mean_pass_rate`).

Implication: substrate work on the hormones / lactate / glycogen shows up in
`overall`, `per_marker`, and the textbook/verifier scores — **not** in the
narrow glucose/hr gate. The gate is the acute-imputer surface; it is not the
comprehensiveness frontier.

## Marker completeness tiers (supervision view, iter 73 numbers)

| Tier | Markers | Diagnosis |
|---|---|---|
| Strong (absolute level pinned) | glucose, insulin*, hr, cortisol, acth, temp, sbp | multiple focused signals |
| Shape/rate only (amplitude floats) | glucagon, ffa, bhb, ghrelin, leptin, glp1, dbp | cold-distill + dilute cohort |
| Rate-only, level free | lactate, hepatic_output | only the insulin-sweep rate |
| Not scored / weakly supervised | hrv, rr, spo2 | reached only via per-patient embeddings |
| Gradient-starved free params | liver_glycogen, muscle_glycogen, mito_capacity, crh | padded constants in the teacher |

\* insulin is "strong" structurally but was the worst marker (0.90) at iter
73 — the standing amplitude gap, a supervision-strength issue not an
architecture one.

## The one meta-pattern

Every stalled iteration for 30+ rounds is the same disease in different
clothes: **a parameter no strong gradient reaches.** Dead pathways
(iters 38–52), inert glycogen (55–57), CRH drift (67–73), the meal-amplitude
gap (68–73) — all under-supervision, not architecture. The CRH cascade
failed three times because the teacher doesn't simulate CRH; glycogen flux
regressed because cohort window-means are too weak; breadth (5→60 rules)
didn't translate to accuracy because per-constraint gradient sits below the
noise floor.

## The supervision substrate — confirmed defects (iter 74)

Verified against the code, not assumed:

- **Dilution (confirmed).** Both constraint signals aggregate as a
  fixed-budget weighted mean (`w·itemᵢ/Σweight`). Physiology: 0.05 ÷ ~60
  rules ≈ 8.4e-4 per rule; `adaptive=True` (iter 67) concentrates that onto
  violated rules. Cohort: 0.15 ÷ Σweight≈57 ≈ 2.6e-3 per default spec, with
  **no adaptive** — only hand-tuned 12×/3× bumps. (iter 74 ports adaptive to
  cohort.)
- **Missing constraint classes (all confirmed absent).** No
  mass-balance/conservation loss; no homeostatic-setpoint constraint for any
  non-glucose marker (`fasting_stability` is glucose-only and was disabled);
  no inter-marker ratio loss (the one "ratio" rule was two independent
  hinges). (iter 74 adds all three, scoped for physiological safety.)
- **Refuted:** the "zero-embedding blind spot" — see eval section above.

## The gap that defines the product

The PRD north star is **counterfactual reasoning over weeks-to-months**
(sleep restriction, semaglutide, training blocks). The model today is an
**acute-state imputer, τ ≤ hours.** The chronic machinery (slow
glycogen/mito states) exists architecturally but is inert — gradient-starved,
and the teacher pads those states with constants. The single largest distance
between "what we built" and "what it's for" is the multi-timescale axis,
blocked by the same supervision disease plus a teacher that doesn't simulate
slow dynamics.

## Guiding principle

**Always target the cleanest, most fundamentally correct modeling — least
hacky/patchy, most accurate.** Prefer fixing structure (which signals reach
which parameters, the coupling graph, conservation laws) over tuning knobs.
A clean break beats a band-aid even when both pass the gate. When a
constraint risks fighting real physiology (a flat pin on a marker that
legitimately moves, a mass budget with wrong units), it is the wrong
constraint — find the formulation that is *true by construction*
(inequalities that only fire on physical violations; targets calibrated from
the teacher to keep units consistent).

## Open frontiers (structural, not tweaks)

1. **Multi-timescale / chronic** — extend the teacher to simulate
   glycogen/mito flux, build chronic cohorts, get the slow states moving.
   The north-star move; depends on a working substrate.
2. **Exercise coverage** — the largest single hole: no
   `activity→{hr,lactate,ffa,...}` cohorts.
3. **Incretin edge** `glp1→insulin`, **autonomic/baroreflex**,
   **leptin↔ghrelin energy-balance loop** — absent couplings.
4. **Teacher extension for CRH** — prerequisite for ACTH pulsatility / CAR.
