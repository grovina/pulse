# iter-95 proposal — one line of frame error, and the first liver axis

Two halves in one run. Iterations are expensive (iter-94 trained for 28.5 h), so the
structural correction that unblocks the fasted state and the first hepatobiliary states
ship together. §5 is honest about what that costs in attribution.

Predecessor: `docs/iter94-spec.md`. Every measurement below was taken on this tree with
`scripts/iter94_student_fast_probe.py` and `scripts/iter95_head_shape_audit.py`, against
`gs://grovina-pulse-data/training/jobs/iter94/model.pt`.

> **This document was revised mid-build.** The first version diagnosed three separate
> per-species shape errors. Implementing the first fix showed they are one frame error
> with three faces. §1.5 records what the first version got wrong, because the wrong
> version is the more instructive one.

---

## 0. What iter-94 actually returned

Training (`trainer-j87zm`) completed genuinely — 28.5 h, artifact uploaded 2026-08-15
20:08. **The benchmark job it required was never dispatched**, and neither was the
primary criterion; both were finally run 2026-08-20, six days later. The benchmark is
running now as `trainer-rg58t` (§6).

Primary criterion, **1 of 3**:

```
liver_glycogen  ratio  1.01   (iter-93: -0.00)   PASS   student 41.0 g vs teacher 41.7 g @24h
bhb             ratio  0.00   (iter-93:  0.02)   FAIL   frozen: +0.001 mmol/L over 24 h
insulin         ratio -0.40   (iter-93: -0.17)   FAIL   rises 10 -> 12.5; teacher falls 10 -> 3.8
```

**B4 — the glycogen absorbing-floor fix — worked outright.** **B1/B2/B3 — the
supervision-weighting work — moved neither bhb nor insulin.** The second consecutive
iteration in which replacing a wrong shape succeeded immediately where supervision-side
work had failed for several iterations running.

---

## 1. The finding: `typical` was an absorbing floor for every species

iter 94 found that the glycogen pools could not fall below `typical`, and read it as
specific to `GlycogenFluxHead`. It is not specific to anything. It is one line.

Modules receive the NORMALIZED state but their rates are applied to the RAW state
(`model.py`: `norm_state = (state - norm_center)/norm_scale`, then
`rates[:, idx] = met_rates[:, i]`). The mass-action assembly was

```
rate = prod·prod_scale − cons·cons_scale·norm_state        prod, cons ≥ 0
```

so consumption was proportional to the **normalized deviation from typical**, not to the
**concentration** — even though `MassActionModule`'s own docstring has always read
*"rate = production − consumption × concentration"*. This is the same frame-error family
as the iter-90 Sg bug.

### 1.1 Consequence one — nothing could go below `typical`

At `raw == typical` the consumption term is exactly zero, so `rate = prod ≥ 0`. Below
typical, `norm_state < 0` turns `−cons·norm_state` into a positive **source**. Measured
on the iter-94 artifact across fast / fed / big-meal / hard-bout protocols:

```
marker           typical   student min   min-typ    teacher min
insulin            10.00       10.0000    0.0000         2.8143
glucagon           70.00       70.0000    0.0000        69.3306
ffa                 0.50        0.5000    0.0000         0.4720
bhb                 0.10        0.1000    0.0000         0.1010
lactate             1.00        1.0000    0.0000         1.0000
hepatic_output      2.00        2.0000    0.0000         1.1233
```

All six, exactly `0.0000` — the same signature iter 94 found for glycogen. Seven markers
were affected in total (`insulin`, `glucagon`, `ffa`, `lactate`, `hepatic_output`,
`leptin`, `glp1`); `SetpointHead` species escaped it because their production is signed.

**Insulin is a gate-scored observed marker and has been unable to fall below 10 µU/mL for
the entire history of the project.**

### 1.2 Consequence two — it manufactured the iter-51 dead-pathway trap

Sitting at `typical` required `prod = 0`, i.e. `prod_raw → −∞`, where
`d(softplus)/d(prod_raw) = sigmoid(prod_raw)` vanishes and the head's gradient dies. That
is precisely the "dead pathway" wall of iters 47-52. Measured on the isolated assembly
(one species, insulin's typical=10 and NORM_SCALE=10, optimizer asked to place the
equilibrium at a target):

```
assembly          target  achieved    error  prod_logit  grad frac
deviation           3.80    10.002    6.202      -9.049   1.18e-04   floored, and saturating
concentration       3.80     3.800   -0.000      -0.227   4.43e-01   ok
deviation          10.00    10.081    0.081      -5.676   3.42e-03   reached, gradient 150x weaker
concentration      10.00    10.000    0.000       0.010   5.02e-01   ok
deviation          50.00    50.000    0.000      -0.205   4.49e-01   ok
concentration      50.00    50.000    0.000       0.948   7.21e-01   ok
```

Note the last pair: **both frames handle a target ABOVE typical perfectly.** That is the
whole history of this project in one row — meal peaks always worked, fasting troughs
never did.

### 1.3 The fix adds nothing; it deletes

```
rate = prod·prod_scale − cons·cons_scale·raw_state
```

Equilibrium becomes `raw* = typical·prod/cons` — reachable anywhere in (0, ∞) — and
sitting at typical requires `prod = cons`, two moderate positive values, so there is no
saturation to escape. This is the law the docstring always claimed. It is a frame
correction: it *removes* an artificial floor rather than adding a constraint, which is
what `Correctness in the Feynman sense` asks for.

**`SetpointHead` is therefore removed** (109 lines). It was invented in iter 51 to escape
a trap that was an artifact of this frame, and it carried a failure mode of its own — its
`k_factor = softplus(raw)` could collapse to zero, freezing the species *and* zeroing the
gradient onto `target_z`. Measured on the iter-94 artifact, bhb sat at `k_factor = 1e-5`
with `target_z = +2.64`: the head asking for 0.23 mmol/L with no authority to get there.
Nothing replaces it — `SpeciesHead` now has live gradients everywhere.

**Iters 51-57's architecture of per-species workarounds was compensating for this one
line.**

### 1.4 Verified after the change

On fresh-init models across three seeds, 8-9 of 11 species now go below `typical`
(`insulin` reaches 6.70; the two that don't vary by seed and are simply slow, not
floored). An unrelated benefit: the untrained insulin runaway to its 210 µU/mL clamp —
the divergence `_EGP_MAX` was added to guard against — is gone, with insulin peaking at
21.5 instead.

### 1.5 What the first version of this document got wrong

- It diagnosed **three separate category errors** (storage pool / flux / flux product)
  needing three separate per-species forms. There is one frame error. The per-species
  fixes would each have worked, and would each have been a workaround.
- It reported **τ as `1/C`**, omitting the NORM_SCALE factor. The true relaxation
  constant is `NORM_SCALE/C`. Corrected: bhb ≈ 265 days (not "12 years"),
  `hepatic_output` ≈ 6.4 days (not 11.8). Conclusions unchanged, figures were off by
  `NORM_SCALE` each. Fixed in the committed diagnostic — on the nose, for a finding about
  frames.
- It listed **`lactate` as inert**. Wrong: the teacher's
  `dLac = −k_lac·(Lac − Lac_b) + lac_act_gain·act²` is a genuine setpoint species with an
  activity drive, and it read flat only because the audit protocol runs at zero activity.

---

## 2. Half A — the frame fix and what it unblocks

**A0 — the frame fix.** `modules/base.py` (`MassActionModule.forward` +
`raw_state`), plus `modules/metabolic.py`, which assembles its own rates rather than
calling `super().forward()` and so needed the same correction applied separately.
`SetpointHead` removed; `bhb`, `mitochondrial_capacity`, `ghrelin`, `cortisol`, `acth`,
`crh` move to `SpeciesHead`. Trained `cons` values do not transfer — the effective
relaxation rate changes from `cons·cons_scale/NORM_SCALE` to `cons·cons_scale`.

**A1 — the defended glucose level falls as liver glycogen empties.** ✅ built and
measured. The student's fasting equilibrium was `b_emb`, **constant in time**, so no fast
of any length could lower it. The teacher gained this in iter 93 (`full_body.py:733-748`,
Cahill 2006); the student never did. Ported reading iter-94's now-working glycogen pool,
with the drop fraction learned and bounded, init at the teacher's value, and exactly the
identity at the fed calibration state. Applied to iter-94's trained weights, a 24 h fast
goes **95.0 → 76.4 mg/dL against the teacher's 78.2** — a delta ratio of 1.11, from
+5.2 in the wrong direction. The fed day is unchanged.

**A5 — fasting hypoinsulinemia.** The *other half* of the same iter-93 teacher fix, also
never ported. The teacher's iter-93 comment names the student's exact failure chain:
*"insulin never falls; and lipolysis is insulin-gated, so FFA never rises."*
`effective_Ib = Ib · min(G/Gb,1)^fast_ins_exp`, referenced to the FED setpoint — the
exponent matters because basal insulin roughly halves while glucose drops ~15%
(Polonsky 1988). Applied to **total** insulin production rather than a basal term alone,
on evidence: measured, the student's soft gate still supplies 18-47% of production across
a fast where the teacher's hard `max(G−h,0)` threshold gives exactly zero.

**A3 — `NORM_SCALE[bhb]` 0.05 → ~0.5.** Independent of the frame. At 0.05 the
catastrophe clamp `center + 20·NORM_SCALE` lands at **1.10 mmol/L, below the teacher's
own 1.315 at 24 h** (3.5 at 48 h). A unit choice, not a constraint — fixing it removes an
artificial bound. Audit every consumer of that scale before landing.

**A2 / A4 — now conditional, and to be measured, not assumed.** The first version of this
plan gave `bhb` the teacher's ketogenesis form and `hepatic_output` an explicit flux form.
In the corrected frame both can reach any positive equilibrium with live gradients, so
**neither may be necessary**. The disciplined order is: smoke-train on A0+A1+A5+A3, re-run
the audit, and add a structural form only where the measurement still demands one. Adding
both now would be exactly the reflex this iteration is about.

---

## 3. Half B — the biliary axis, built so the enzymes drop in later

The eventual target is cholestatic coverage: ALP, GGT, ALT, bilirubin. Starting there
would fail, for a reason iter-94 already documented: **nothing in the state vector could
drive them.** They are damage and obstruction readouts; with no injury, steatosis, drug or
obstruction represented they would ship as constants, and the gate cannot tell a constant
from a simulator. Their timescales (ALP, GGT τ ≈ 7-10 days) also land where every slow
state is currently weakest.

So the entry point is the **mediator**, not the readouts.

### 3.1 The design decision that matters now

**Model the canalicular export step explicitly, not just a serum pool.** Cholestasis *is*
a failure of that transport step, and ALP/GGT are induced by cholangiocyte exposure to
retained bile acids. If only serum bile acids exist as a species, adding ALP later means
bolting on a driver again. If the export step exists as a step, the enzymes hang off its
impairment naturally.

### 3.2 Three states, all with real drivers on day one

| state | driver | why it is not inert |
|---|---|---|
| `cck` | duodenal lipid + protein appearance | the gut module **already exports** lipid and amino channels; no new input needed |
| `gallbladder_bile` | CCK-gated emptying, inter-meal refill | fast, meal-locked, in the regime iter-92 already tuned |
| `bile_acids` | release → ileal reabsorption → portal → **hepatic export** → systemic spillover | postprandial serum excursion is a scorable dynamic event |

### 3.2.1 The gallbladder is a POOL, and contraction is a gate — not the other way round

Considered and rejected: making `gallbladder_contraction ∈ [0, 1]` the state and holding
volume constant. It is the more bounded and tempting parameterisation, and it is the same
category error as §1 — contraction has no conservation law of its own. It is a
dimensionless fraction fully determined at each instant by CCK, i.e. a **gate**, while
volume/content is the one genuine **pool** in the axis.

The concrete failure it would produce: **you cannot empty a gallbladder twice.** Two meals
90 min apart give a large bile-acid excursion and then a much smaller one, because the
reservoir is depleted. With a constant volume the second meal reproduces the first
exactly. That is the same class of error as the glycogen absorbing floor and the iter-91
postprandial drift. It would also erase the interdigestive dynamics — the gallbladder
fills and concentrates through an overnight fast, which is *why* the first meal of the day
gives the largest excursion.

The resolution keeps both:

```
state    gallbladder_bile        a pool — floor at empty, ceiling at capacity
gate     contraction = f(CCK)    computed, not stored; multiplies the emptying flux
derived  ejection fraction = dV/V0 over 60 min    <-- score THIS against HIDA literature
```

Structurally identical to `GlycogenFluxHead` — the pool is the state, the learned gates
sit on the flux — which is the one part of iter 94 that demonstrably worked. Named for
**bile-acid content (µmol)** rather than volume (mL): the gallbladder concentrates bile
~10x, so content is what conserves through the enterohepatic loop.

All three live in the **fast meal-response regime the model is already good at**, so they
are testable on the existing ruler this iteration.

### 3.3 What must be authored

- **teacher physics** — the enterohepatic loop in `knowledge/full_body.py`
- **coupling priors** — a new `knowledge/coupling_priors/hepatobiliary.py`, registered in
  `coupling_priors/__init__.py`; `bile_acids → glp1` (TGR5) gives the axis an observable
  consequence rather than leaving it a closed loop
- **cohort statistics** — literature anchors in `knowledge/cohorts/`
- **module + wiring** — `System.HEPATOBILIARY`, `modules/hepatobiliary.py`, an `_emb_dims`
  entry, coupling in/out in `model.py:forward`
- markers are `internal`/unobserved in the ruler — same regime as glucagon and FFA

Anchors to encode: fasting serum total bile acids; postprandial fold-rise and
time-to-peak; gallbladder ejection fraction and emptying half-time; CCK basal and
postprandial peak. **Each to be verified against primary sources before it is written
in** — none from recall. The iter-94 circadian work is the precedent for what happens
when a cited range and an encoded constant drift apart.

### 3.4 Blast-radius control — initialization, not constraint

**Zero-init the outgoing coupling** (`bile_acids → glp1`) so at step 0 the new axis is an
exact no-op for every pre-existing marker and any Half-A regression stays attributable.
Initialization, not a constraint on the learned solution; the repo already uses the
pattern in `glucose_baseline_net` and `ra_baseline_net`.

---

## 4. Acceptance criteria, in priority order

1. **Primary — the fasted state engages.** On a 24 h fast at the prior-mean embedding:
   `bhb` ratio ≥ 0.5 (iter-94: 0.00), insulin moving in the **correct direction**
   (iter-94: −0.40), `liver_glycogen` ratio holds ≥ 0.5 (iter-94: 1.01).
   `scripts/iter94_student_fast_probe.py`.
2. **Primary — the floor is gone in the TRAINED model.** Every species reaches below
   `typical` where the teacher does, and no rate constant has collapsed.
   `scripts/iter95_head_shape_audit.py`. Structurally verified at fresh init already
   (§1.4); this confirms training does not re-create it.
3. **Biliary — the axis is dynamic, not decorative.** Postprandial `bile_acids` excursion
   and derived gallbladder ejection fraction within their encoded bands, `cck` peaking
   ahead of the bile-acid rise.
4. **Guard — iter-92 meal kinetics hold** (glucose peak 45-60 min, ghrelin nadir
   −30…−50 % at 60-90 min). A0 changes ghrelin's head, so this guard is load-bearing.
5. **Guard — `legacy_static` absolute MAPEs do not move materially**, and `cgm_real` skill
   does not degrade against the iter-94 benchmark once `trainer-rg58t` lands.

**`gate.passed` is not an acceptance criterion**, as in iter 94.

---

## 5. Risks this iteration takes, stated plainly

**A0 touches every mass-action species in three modules.** That is the point — it is one
root cause — but it means no iter-94 weights carry meaning, `cons` is re-learned
throughout, and a regression anywhere in metabolic/appetite/stress is attributable to it.
The mitigation is that it is verifiable at fresh init and in a smoke train before any paid
dispatch, and §1.2's isolated experiment already establishes the mechanism independently.

**Euler headroom shrinks by NORM_SCALE.** The relaxation coefficient goes from
`cons·cons_scale/NORM_SCALE` to `cons·cons_scale` — 10x stiffer for insulin, 20x for
glucagon. At dt=1 the stability limit is 2/min and insulin sits at ~0.24/min, so there is
margin, but `cons` is an unbounded softplus and this must be watched in the smoke train
rather than assumed.

**Removing `SetpointHead` changes six markers that were not broken** — `ghrelin`,
`cortisol`, `acth`, `crh` among them. It is not optional (its derivation is invalid in the
corrected frame), but ghrelin and cortisol currently pass and could regress. Criterion 4
is the guard.

**Merging two halves costs attribution.** iter-94's spec explicitly warned against
widening blast radius. This run does it deliberately, because a paid iteration is ~28 h.

**State dim 24 → 27 changes `input_dim` for every module.** "No-op at init" does not
extend to "no-op at convergence".

**The biliary teacher is new physics with no real ground truth.** Every bile-acid number
the student learns comes from our own teacher; `cgm_real` cannot check it. Same
circularity iter-94 disclosed for `teacher_dynamic`, disclosed again here rather than
discovered later.

---

## 6. Ops

- `trainer-rg58t` — `--benchmark-only` on `training/jobs/iter94/model.pt` against
  `benchmark.dataset.iter94.json`, report to
  `gs://grovina-pulse-data/training/jobs/iter94/benchmark-report.json`. Dispatched
  2026-08-20 10:07 UTC on image `trainer:40ca3b9`, **same image and args as the A5
  baseline** (`trainer-v2w6z`, 11h43m), so the comparison is apples-to-apples. ETA
  ~21:50 UTC. Purpose: the iter-94 B1 risk check — teacher-source agreement improving
  while `cgm_real` skill degrades. Read the result from the **artifact**, never
  `succeededCount`.
- Half A is verifiable locally before any paid dispatch. Do that first.
