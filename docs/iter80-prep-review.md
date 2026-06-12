# Modeling & abstraction review — pre-read for iter 80

*Written 2026-06-12, while the iter-79 honest-baseline run is in flight. Purpose:
have the abstraction-level review done **before** the numbers land, so the moment
iter 79 writes its `benchmark-report.json` we can read the signal into a decision
that is already framed, and make the next move assertively rather than starting
the analysis cold. Read alongside `docs/modeling-state.md` (the map),
`docs/architecture-roadmap.md` (the chronicle), and the new
*Correctness in the Feynman sense* section now at the top of `prd.md`'s
Philosophy.*

---

## TL;DR

The abstractions are sound and the planned next move — **close the
glycogen↔glucose mass-balance loop in the teacher** (split `Hep` into
glycogenolytic + gluconeogenic) — is the *right* move for the right reason: it is
the first structural change that supervises a slow pool by making it
*load-bearing under a conservation law*, instead of adding capacity and hoping a
gradient finds it. That is the exact thing that failed for CRH three times and
for glycogen in iters 55–57.

Two things must be true for iter 80 to be impeccable rather than merely good, and
both are decidable from the iter-79 baseline plus a few hours of teacher work:

1. **The split must be conservation-exact at the point of introduction** — total
   hepatic glucose output unchanged at the calibration state — so iter 80
   measures "mass-balance structure," not "accidentally recalibrated fasting
   glucose." (§3)
2. **Glucose is now downstream of glycogen.** Coupling the slow pool into glucose
   is what finally gives glycogen a strong gradient (it inherits glucose's), but
   it also puts the gate-critical marker at risk of the pool's errors. iter 80's
   whole bet lives in that trade. (§3)

The rest of this doc: what the abstractions get right and must not be broken
(§1), the one place doc and code have drifted (§2), the iter-80 risk in detail
(§3), the deeper Feynman tension the run should make us confront — *is the teacher
actually overrulable by literature, or is it the de facto source of truth?* (§4),
the template every future latent state must follow (§5), and the concrete
read-off + decision tree for when iter 79 lands (§6).

---

## 1. What the abstractions get right (do not break these)

- **The four-layer hierarchy** (physics/chemistry → anatomy → medical knowledge →
  individual variation) is the spine, and the code honors it: existence of a
  coupling edge is structural, sign is a regularized prior, strength is learned;
  calibration moves only the embedding, never weights. This is the load-bearing
  separation and every move should preserve it.

- **The under-supervision diagnosis is correct and hard-won.** The single
  meta-pattern — *every multi-iter stall is a parameter no strong gradient
  reaches* — is established across dead pathways (38–52), inert glycogen (55–57),
  CRH drift (67–73), the amplitude gap (68–73). The corollary is also correct:
  the fixes that worked were all changes to the *gradient surface*
  (SetpointHead's saturation-free coordinates, rate-matching's undiluted
  gradient, the compact HPA cascade that mirrors the teacher's own structure) —
  never knob-tuning. Keep treating "structural vs parametric" as the first fork.

- **Distillation must happen at the embedding the eval lands on.** The five-iter
  dead-pathway wall (47–51) taught that distilling at the wrong embedding moves a
  point the bench never visits at the cost of the point it does. Any new
  distillation target (including a glycogen trajectory) inherits this: supervise
  at the calibrated operating point, not at zero.

- **Calibrated supervision** (size the constraint to the evidence) and the
  **fence-not-pin** stance on approximate teachers are exactly the right
  epistemics and now sit under the same Feynman frame.

## 2. The one doc/code drift worth fixing

`prd.md`'s Architecture section still states mass-action modules **enforce**
`rate = production − consumption × concentration` with production/consumption
**non-negative** as "fundamental chemistry." That is no longer literally true:
since iter 51, glucagon/FFA/ghrelin use `SetpointHead`, which emits a *signed*
`prod = k·target_z/typical` — negative production is permitted, deliberately,
because insulin-suppressed glucagon / antilipolysis / nutrient-suppressed ghrelin
are not mass-action in the chemical sense. This was the correct call (the
mass-action prior was the wrong *shape* for an equilibrium that has to leave
typical), and the new Feynman section already names this pattern ("even the
constraints we call enforced in architecture are approximations we hold
provisionally"). **Action:** when convenient, soften the Architecture prose to
match — mass-action is a *prior we relax where physiology demands*, not an
inviolable enforcement. Small, but the PRD is the spec other reasoning rests on,
and a reader who believes prod ≥ 0 is enforced will mis-model.

## 3. iter 80 in detail — the mass-balance move and its blast radius

The teacher today (verified in `knowledge/full_body.py`) integrates
`liver_glycogen`/`muscle_glycogen` as flux integrators but **deliberately does not
let them feed back** into glucose/hepatic-output/ketone equations — the comment
at the glycogen block is explicit that the 19-marker observed ODE is
byte-identical to iter 75 and that the `Hep` split "is the next iter." So iter 80
is pre-scoped and the attribution is clean. Two disciplines make it impeccable:

**(a) Conservation-exact at the split point.** `Hep` is currently a single
literature-calibrated lumped term that produces correct fasting glucose. When you
split it into `Hep_glycogenolytic(LGly, insulin) + Hep_gluconeogenic(...)`, the
two parts **must sum to the old `Hep` at the calibration state** (fed/early-fast,
default patient). If they don't, fasting glucose shifts for a reason that has
nothing to do with the new mechanism, and iter 80 confounds "I added
mass-balance" with "I retuned the fasting setpoint." Make the split a
reparameterization of the existing term (partition, don't replace), let the two
components *diverge only as `LGly` depletes* — which is precisely where the new
physics earns its keep (liver empties → glycogenolytic component falls →
gluconeogenesis + ketosis rise). Verify by diffing teacher glucose/BHB
trajectories before/after on a fed day: they should be ≈identical until the pool
starts depleting (~12h+ fast), then diverge in the physiologically correct
direction. That diff *is* the unit test for "true by construction."

**(b) Glucose becomes downstream of the slow pool — that is the point, and the
risk.** Today glycogen is gradient-starved because nothing strong reaches it
(cold-distill on a flat-ish trajectory ≈ the iters 55–57 failure). The moment
`LGly` feeds glucose, the *strong* glucose gradient (real-user data + the gate +
dose-response) flows backward through the conservation coupling into the glycogen
pool. **This is the first real supervision glycogen will ever get** — and it is
structural, not a new loss term. But the same edge means a wrong glycogen
trajectory now corrupts the gate-critical glucose marker. So the iter-80 read is
binary and clear: either mass-balance coupling pulls glycogen into shape *and*
holds glucose (the win), or glycogen's freed capacity drags glucose off the gate
(the iter-69-style collateral). The iter-79 baseline is what tells us how much
glucose headroom we have to spend on that bet (see §6).

**(c) Keep it one mechanism.** The roadmap's own scar tissue (iter 69 bundled
dose-response + CRH and couldn't attribute the regression) says: ship the `Hep`
split *alone*, every other knob byte-identical to the iter-79 config, exactly as
iter 79 held config identical to iter 78. mitochondrial_capacity stays padded
(its τ ≈ weeks can't be exercised by a ≤1-day protocol anyway — adding it now
would just be CRH again).

## 4. The Feynman tension the baseline should force us to confront

The PRD's aspiration: literature *refines the teacher* — "when the cold model is
wrong or incomplete, the learned model can diverge from it, guided by other
training signals." The training reality, read off the current spec: `cold-distill
weight 0.3` delivers a **direct** trajectory/anchor gradient on 12 markers, while
`cohort 0.15` and `physiology-rules 0.05` are diluted across ~18 protocols and 61
rules (per-item ≈ 1e-3 and ≈ 1e-3, even with adaptive concentration). For the
unobserved markers, the teacher is not *a* prior among the literature — it is, by
gradient mass, **the de facto source of truth**, and the literature is a faint
correction on top.

This is not wrong — bootstrapping from the teacher is deliberate and the PRD
endorses it. But it means the Feynman commitment ("refine the law by overturning
it") is currently **aspirational, not operative**: the literature cannot
out-vote the teacher anywhere the two disagree. The honest baseline is the moment
to make this explicit and decide whether that is acceptable for now. Concretely,
the iter-79 report lets us check it: where `per_marker_by_source['real']`
(literature/real-data-facing) and the teacher-facing numbers diverge for the same
marker, that gap is the teacher imposing a structure the evidence doesn't
support. If those gaps are small, teacher-dominance is fine and the mass-balance
work is the priority. If they are large, the next frontier after mass-balance is
*supervision rebalancing* — giving literature enough gradient to overrule the
teacher where they conflict — which is the operative form of the Feynman
principle, not a slogan.

## 5. The template every future latent/slow state must follow

The CRH saga (69–72, three failures) and the glycogen revival (55–57 inert →
76 simulated → 80 load-bearing) reduce to one rule, and it should be stated as
doctrine:

> **Never add a latent state without simultaneously wiring the conservation law
> or coupling that gives it a strong gradient path to an observed marker.**

CRH failed because it was a free latent with no ground truth — the teacher pads
it, so `λ·CRH → ACTH` let ACTH's rate be explained by a floating variable, and
gradient dragged it off the proven direct drive every time. Glycogen iters 55–57
failed identically (SetpointHead pool, no feedback, nothing to supervise it).
iter 80 *works* — if it works — precisely because the mass-balance edge is both
the mechanism *and* the supervision: glucose's gradient is what shapes the pool.
This is the difference between "capacity" and "a law." Add laws, not capacity.
(Corollary for mito_capacity and any `cortisol_baseline_drift`: deferred until a
conservation/coupling edge can supervise them, exactly as glycogen waited for the
`Hep` split.)

## 6. When iter 79 lands — read-off and decision tree

The iter-79 run is gradient-identical to iter 78 (test-guaranteed), so its report
**is** the first honest iter-78 baseline. Read, in order:

1. **Did it complete?** `gsutil ls gs://grovina-pulse-data/training/jobs/iter79/`
   — `benchmark-report.json` present = real success (`succeededCount` lies; see
   `[[pulse-cloud-run-training-ops]]`). If it timed out, the remaining lever is
   `cold_model_distillation` (the un-batched inner Adam calibration, ~1339s) —
   R1 in the handoff.

2. **Glucose headroom (decides how much we can spend on the iter-80 bet).**
   `per_marker_by_source['real']` skill_vs_persistence for glucose, and
   glucose's gate value. Glucose persistence on real data is ~0.014 — a brutal
   bar — so near-zero/negative skill is *honest*, not a regression. What matters
   for iter 80: how close is glucose to the gate, and does dose-response `peak`
   mode (now `glucose:peak:0.7:0.25` in the spec) finally show amplitude movement
   (the standing 0/3 `meal_dose_response` was the amplitude gap)? **If glucose has
   margin → iter 80's glycogen→glucose coupling is safe to dispatch. If glucose
   is on the knife-edge → land the conservation-exact split first with the
   coupling gain initialized near zero, and ramp the feedback in a follow-up, so
   the slow pool can't yank the gate on introduction.**

3. **Are the slow pools moving at all?** `dist_liver_glycogen` /
   `dist_muscle_glycogen` in cold-distill telemetry (iter 76 made them simulated;
   this is the first *completed* phase-2 read on whether they track depletion).
   If they're already tracking the teacher's open-loop trajectory, the iter-80
   closed-loop coupling has a stable base to build on. If they're still inert,
   that's a flag that even the iter-76 open-loop distillation is too weak — and
   the §4 supervision-rebalancing question moves up the queue.

4. **Teacher-vs-real divergence (the §4 check).**
   `overall_weighted_mape_by_source` real vs teacher, and per-marker where they
   split. Small gaps → mass-balance is the priority. Large gaps → the literature
   can't overrule the teacher and supervision rebalancing is the real frontier.

5. **Textbook + semantic gates.** `textbook_mean_pass_rate` (gate 0.45) and the
   meal verifier — the load-bearing "counter-regulation actually fires" check.

**Decision tree:**

- **Baseline healthy, glucose has margin, pools tracking** → dispatch iter 80 as
  the conservation-exact `Hep` split, single mechanism, config-identical. This is
  the assertive default and the most likely path.
- **Baseline healthy, glucose knife-edge** → iter 80 = split + near-zero coupling
  gain (structure in, feedback ramped later). Protect the gate; still advance the
  north star.
- **Pools still inert / large teacher-vs-real gaps** → iter 80 pivots to
  supervision rebalancing (the operative Feynman move) *before* adding the
  mass-balance edge, because coupling a still-unsupervised pool into glucose would
  just route noise into the gate.
- **Run timed out** → measurement-first again (batch the cold-distill anchor
  windows, R1), no modeling change, before anything else.

The point of having this written now: whichever branch the numbers select, the
move is already designed and its risk already understood. No cold-start analysis,
no confounded bundle — one clean, attributable, structurally-correct step.

---

## Secondary observations (not iter-80-blocking, logged so they aren't lost)

- **The coupling graph is still thin** (roadmap gap #4) and the largest
  *structural* hole is the absence of a **catecholamine / sympathetic** pathway.
  HR, BP, FFA mobilization, and hepatic glucose output under stress and exercise
  all physiologically route through sympathetic drive that the model does not
  represent — cortisol is silently doing that job as a proxy (it's wired into
  cardiovascular, thermoreg, and metabolic). This is a candidate for the frontier
  *after* mass-balance, and it is the cleanest explanation for why exercise
  coverage (gap #2) and the HR gate have been persistently hard: the driver isn't
  in the graph.
- **Learned-model vs teacher coupling mismatch:** the learned cardiovascular
  module imports cortisol+temp+glucose+insulin (4 edges) while the teacher's CV
  ODE uses only cortisol+activity+sleep. Not a bug (the learned model is allowed
  richer inputs), but it means some learned CV couplings have *no* teacher signal
  behind them — they're shaped only by real-user vitals and physiology rules.
  Worth a deliberate audit once mass-balance lands: every learned edge should have
  *some* supervision, or it's another dead pathway waiting to happen.
- **Incretin and energy-balance edges still absent:** `glp1→insulin` (the
  incretin effect is ~50% of post-meal insulin), `leptin↔ghrelin`. These are
  canonical and currently unrepresentable — low-risk additive moves whenever the
  cohort arms exist to supervise them.
- **The gate (glucose+hr, acute) and the ambition (weeks-scale counterfactuals)
  remain far apart.** The honest baseline's real value is that it's *honest*
  (skill-vs-persistence on real data), which is the first metric that can't be
  gamed by teacher-matching. As the model moves onto the multi-timescale axis,
  success criteria should migrate toward held-out counterfactual / chronic-block
  checks — passing the acute gate is necessary, never sufficient, and per the new
  Feynman section, never to be mistaken for correctness.
