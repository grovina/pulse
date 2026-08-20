# iter-95 proposal — the shape of a state is not a detail, and the first liver axis

Two halves in one run. Iterations are expensive (iter-94 trained for 28.5 h), so the
structural corrections that unblock the fasted state and the first hepatobiliary states
ship together. §5 is honest about what that costs in attribution.

Predecessor: `docs/iter94-spec.md`. Measurements below were taken on
`gs://grovina-pulse-data/training/jobs/iter94/model.pt` with
`scripts/iter94_student_fast_probe.py` and `scripts/iter95_head_shape_audit.py`.

---

## 0. What iter-94 actually returned

The training job (`trainer-j87zm`) completed genuinely — 28.5 h, artifact uploaded
2026-08-15 20:08. **The benchmark job it required was never dispatched**, and neither
was the primary criterion; both were run on 2026-08-20, six days after the artifact
landed. The benchmark is now running as `trainer-rg58t` (see §6).

Primary criterion (`iter94_student_fast_probe.py`), **1 of 3**:

```
liver_glycogen  ratio  1.01   (iter-93: -0.00)   PASS   student 41.0 g vs teacher 41.7 g @24h
bhb             ratio  0.00   (iter-93:  0.02)   FAIL   frozen: +0.001 mmol/L over 24 h
insulin         ratio -0.40   (iter-93: -0.17)   FAIL   rises 10 -> 12.5; teacher falls 10 -> 3.8
```

**B4 — the glycogen absorbing-floor fix — worked outright.** The pool now tracks the
teacher almost exactly. That is the second consecutive iteration in which replacing a
wrong dynamical shape produced an immediate, decisive result where supervision-side
work had failed for several iterations running.

**B1/B2/B3 — the supervision-weighting work — moved neither bhb nor insulin.** The
reason is not supervision, and §1 is the finding.

---

## 1. The finding: three categories of state, one shape, and it fits only one of them

iter 94 diagnosed the glycogen pools as follows:

> The mass-action assembly is `prod·prod_scale − cons·cons_scale·norm_state`, and
> `norm_state` is zero at the pool's typical value … `typical` was an absorbing floor.

That is correct and it is narrower than the truth. The mass-action / setpoint shape is
being applied to **three different categories of state**, and it is the right shape for
one of them:

| category | example | is it a species in equilibrium? | measured symptom |
|---|---|---|---|
| storage pool | `liver_glycogen`, `muscle_glycogen` | no — a pool with a floor and a ceiling | absorbing floor at `typical` (**fixed in 94**) |
| **flux** | `hepatic_output` | no — a *rate*, with no concentration to be consumed | **τ = 11 794 min ≈ 11.8 days** |
| **flux product** | `bhb` | no — set by production minus oxidation, not by relaxation to a defended level | **rate constant collapsed: τ ≈ 12 years** |
| true species | `glucagon`, `lactate`, `ffa` | yes | shape is right (see §1.3) |

Each inertness has been read, across several iterations, as a supervision problem or a
timescale problem. Neither reading was ever correct.

### 1.1 `bhb` — the rate constant collapsed, and that is the shape reporting itself

`SetpointHead` emits `k_factor = softplus(raw)` and assembles
`rate = cons_scale·k_factor·(target_z − norm_state)`. On the iter-94 artifact, at 24 h
of fasting:

```
k_factor  = 0.00001        tau = 6 404 902 min  (~12 years)
target_z  = +2.64          i.e. the head "wants" bhb = 0.232 mmol/L
bhb       = 0.1006         i.e. it has no authority to get there
```

Because the gradient onto `target_z` is itself multiplied by `k_factor`, the collapse is
**self-locking**: once the rate constant dies, the setpoint can never be corrected.

The tempting fix is to floor `k_factor` — `k = k_min + range·sigmoid(...)`, the band
pattern already used for `Sg`, `Si` and `p2` in this same file. **That fix is wrong**,
and it is wrong in the specific way the PRD's *Correctness in the Feynman sense* names:
it would force a wrong shape to move rather than ask why the optimizer wants it dead.

Asked properly, the collapse is not a bug at all. BHB is given a form that says "relax
toward a defended setpoint at rate k". BHB has no defended setpoint — it is the product
of hepatic ketogenesis minus peripheral oxidation. Given that form, and training data in
which BHB sits at 0.1 in every fed episode, **zeroing the rate constant is the correct
solution to the problem as posed.** The optimizer is reporting that the shape is wrong.

The teacher already has the right form (`knowledge/full_body.py:170-192`): ketogenesis
driven by FFA substrate supply, gated on liver-glycogen depletion, suppressed by
insulin, against first-order clearance `k_bhb`.

### 1.2 `hepatic_output` — a flux modelled as a concentration

Hepatic glucose output is a rate in mg/min. `prod − cons·concentration` presumes a
concentration for consumption to be proportional to; a flux has none. The measured
consequence is τ = 11.8 days — inert on every protocol the model is trained or scored
on. The teacher (`full_body.py:816-826`) instead splits it into a glycogenolytic share
scaled by glycogen availability plus a gluconeogenic share, and its hepatic output falls
2.00 → 1.51 → 1.31 across the 48 h fast while the student's does not move.

### 1.3 What is NOT broken — a retraction

An earlier reading of this audit listed `lactate` as inert on the strength of its
τ = 1210 min and a flat teacher trace. **That was wrong.** The teacher's
`dLac = −k_lac·(Lac − Lac_b) + lac_act_gain·act²` is a genuine setpoint species with an
activity drive; it read flat only because the audit protocol runs at zero activity. Its
shape is right. It needs an exercise bout to test, not a fast, and nothing here should
touch it.

`ffa` (τ = 4084 min) is a genuine watch item but is **not** proposed for change. Its
`BasalPlusGatedPeakHead` is arguably the right shape for stimulus-gated lipolysis, and
its under-response (0.29 of the teacher's 24 h delta) is downstream of insulin never
falling. Re-measure after A1 lands before touching it.

### 1.4 `NORM_SCALE[bhb]` puts the catastrophe clamp inside the physiological range

`PHYSIOLOGICAL_MAX = center + 20·NORM_SCALE` is documented as "deliberately large so the
clamp is INACTIVE in-distribution". For bhb, `NORM_SCALE = 0.05` makes it
**1.10 mmol/L — below the teacher's own 1.315 at 24 h**, and far below 3.505 at 48 h.
Ketosis needs `target_z ≈ +24` where every other species operates in z ∈ [−2, +2].

This is a unit choice, not a constraint: correcting it *removes* an artificial bound
rather than adding one.

---

## 2. Half A — give three states the shape their physics has

No new constraints, bands, floors or penalties. Two of the four are ports of forms the
teacher already derives from cited physiology; the student is structurally unable to
express them today.

**A1 — the defended glucose level falls as liver glycogen empties.**
`modules/metabolic.py`. The student's fasting equilibrium is `b_emb`, a per-patient
constant read from the embedding — **constant in time**, so no fast of any length can
lower it. The teacher gained exactly this in iter 93
(`full_body.py:733-748`, Cahill 2006): `Gb_fasted = Gb·(1 − fast_gb_drop·glyco_depleted)`,
keyed to linear pool depletion and exactly the identity at the fed calibration state.
Port it, reading the now-working `liver_glycogen` state. This is the single highest-value
change in the iteration: it is what turns iter-94's working glycogen pool from a
correctly-moving number into a number that *drives something*.

`docs/physiology-coverage.md:90` has carried `liver_glycogen → hepatic_output` as a
known GAP since before iter 90.

**A2 — `bhb` gets the ketogenesis form, replacing `SetpointHead`.** Production driven by
FFA substrate × glycogen depletion × insulin suppression, against first-order clearance,
mirroring `full_body.py:170-192`. The head's learned outputs are reused; only the rate
assembly changes — the same move B4 made for glycogen, for the same reason.

> **Falsifiable prediction, and the acceptance test for the diagnosis itself:** with the
> ketogenesis form and **no floor on any rate constant**, `k` should stay alive on its
> own. If it still collapses, §1.1 is wrong and the shape is not the explanation.

**A3 — `NORM_SCALE[bhb]` 0.05 → ~0.5.** Sized so the clamp clears the teacher's 48 h
value with margin. Touches `PHYSIOLOGICAL_MIN/MAX` by derivation; audit every other
consumer of that scale before landing.

**A4 — `hepatic_output` gets the flux form, replacing `SpeciesHead`.** Glycogenolytic
share scaled by glycogen availability plus a gluconeogenic share, mirroring
`full_body.py:816-826`.

A1 and A4 are **complementary, not redundant** — the teacher deliberately carries both.
iter 80 added the hepatic-output split; iter 93 added `Gb_fasted` *because the split
alone was not enough*, the minimal model's defended level itself has to fall.

---

## 3. Half B — the biliary axis, built so the enzymes drop in later

The eventual target is cholestatic coverage: ALP, GGT, ALT, bilirubin. Starting there
would fail, for a reason iter-94 already documented: **nothing in the state vector could
drive them.** They are damage and obstruction readouts. With no hepatocyte injury,
steatosis, drug or biliary obstruction represented, they would ship as constants — and
the iter-94 ruler finding is precisely that the gate cannot tell a constant from a
simulator. Their timescales (ALP, GGT τ ≈ 7-10 days) also land exactly where §1 shows
every existing slow state is currently inert.

So the entry point is the **mediator**, not the readouts.

### 3.1 The design decision that matters now

**Model the canalicular export step explicitly, not just a serum pool.** Cholestasis
*is* a failure of that transport step, and ALP/GGT are induced by cholangiocyte exposure
to retained bile acids. If only serum bile acids exist as a species, adding ALP later
means bolting on a driver again — the inert-marker failure mode. If the export step
exists as a step, the enzymes hang off its impairment naturally, and "eventually easy"
is actually true.

### 3.2 Three states, all with real drivers on day one

| state | driver | why it is not inert |
|---|---|---|
| `cck` | duodenal lipid + protein appearance | the gut module **already exports** lipid and amino channels (`GUT_OUTPUT_SCALE`); no new input needed |
| `gallbladder_bile` | CCK-gated emptying, inter-meal refill | fast, meal-locked, in the regime iter-92 already tuned |
| `bile_acids` | release → ileal reabsorption → portal → **hepatic export** → systemic spillover | postprandial serum excursion is a scorable dynamic event |

### 3.2.1 The gallbladder is a POOL, and contraction is a gate — not the other way round

Considered and rejected: making `gallbladder_contraction ∈ [0, 1]` the state and holding
volume constant. It is the more bounded and tempting parameterisation, and it is the same
category error §1 is about — contraction has no conservation law of its own. It is a
dimensionless fraction fully determined at each instant by CCK, i.e. a **gate**, while
volume/content is the one genuine **pool** in the axis.

The concrete failure it would produce: **you cannot empty a gallbladder twice.** Two meals
90 min apart give a large bile-acid excursion and then a much smaller one, because the
reservoir is depleted. With a constant volume the second meal reproduces the first exactly.
That is the same class of error as the glycogen absorbing floor (a pool that could not
deplete) and the iter-91 postprandial drift (meals that accumulated without clearing). It
would also erase the interdigestive dynamics — the gallbladder fills and concentrates
through an overnight fast, which is *why* the first meal of the day gives the largest
excursion.

The resolution keeps both:

```
state    gallbladder_bile    a pool — floor at empty, ceiling at capacity
gate     contraction = f(CCK)    computed, not stored; multiplies the emptying flux
derived  ejection fraction = dV/V0 over 60 min    <-- score THIS against HIDA literature
```

Structurally identical to `GlycogenFluxHead` — the pool is the state, the learned gates sit
on the flux — which is the one part of iter 94 that demonstrably worked. Conservation is
kept, the clinical anchor (ejection fraction is what HIDA actually measures) survives as a
scorable derived quantity, and it costs one state rather than two.

Named for **bile-acid content (µmol)** rather than volume (mL): the gallbladder concentrates
bile ~10x, so content is what conserves through the enterohepatic loop. The two are
interchangeable up to a concentration constant, and volume only matters for matching
ultrasound directly.

All three live in the **fast meal-response regime the model is already good at**, which
is the point: they are testable on the existing ruler this iteration, not two iterations
from now.

The outgoing coupling `bile_acids → glp1` (TGR5) is real and should be declared, which
also gives the new axis an observable consequence rather than leaving it a closed loop.

### 3.3 What must be authored

Following `knowledge/AUTHORING.md`'s decision tree:

- **teacher physics** — the enterohepatic loop in `knowledge/full_body.py`
- **coupling priors** — a new `knowledge/coupling_priors/hepatobiliary.py`, registered in
  `coupling_priors/__init__.py`
- **cohort statistics** — literature anchors in `knowledge/cohorts/`, so the axis is
  supervised against magnitudes rather than shapes alone
- **module + wiring** — a new `System.HEPATOBILIARY`, `modules/hepatobiliary.py`, an
  `_emb_dims` entry, and coupling in/out in `model.py:forward`
- markers are `internal`/unobserved in the ruler — same supervision regime as glucagon
  and FFA (teacher distillation + cohort specs + textbook scenarios)

Anchors to encode: fasting serum total bile acids; postprandial fold-rise and
time-to-peak; gallbladder ejection fraction and emptying half-time; CCK basal and
postprandial peak. **Each to be verified against primary sources before it is written
in** — none of these should be encoded from recall, and the iter-94 circadian work is
the precedent for what happens when a cited range and the encoded constant drift apart.

### 3.4 Blast-radius control — initialization, not constraint

**Zero-init the outgoing coupling** (`bile_acids → glp1`) so that at step 0 the new axis
is an exact no-op for every pre-existing marker, and any Half-A regression stays
attributable. This is initialization, not a constraint on the learned solution — the
repo already uses the pattern in `glucose_baseline_net`, `ra_baseline_net` and
`SetpointHead`'s final layer.

---

## 4. Acceptance criteria, in priority order

1. **Primary — the shape diagnosis is correct.** On a 24 h fast at the prior-mean
   embedding: `bhb` ratio ≥ 0.5 (iter-94: 0.00), insulin moving in the **correct
   direction** (iter-94: −0.40), `liver_glycogen` ratio holds ≥ 0.5 (iter-94: 1.01).
   `scripts/iter94_student_fast_probe.py`.
2. **Primary — no rate constant collapsed, with no floor added.** Every non-overridden
   species shows a live `cons` and a τ within its physiological response time.
   `scripts/iter95_head_shape_audit.py`. This is the falsification test for §1.1.
3. **Biliary — the axis is dynamic, not decorative.** Postprandial `bile_acids`
   excursion and gallbladder ejection fraction (derived, §3.2.1) within their encoded bands, and
   `cck` peaking ahead of the bile-acid rise.
4. **Guard — iter-92 meal kinetics hold** (glucose peak 45-60 min, ghrelin nadir
   −30…−50 % at 60-90 min).
5. **Guard — `legacy_static` absolute MAPEs do not move materially**, and `cgm_real`
   skill does not degrade against the iter-94 benchmark once `trainer-rg58t` lands.

As in iter 94, **`gate.passed` is not an acceptance criterion.** The skill thresholds
sit at the natural zero and the model is still worse than persistence on glucose.

---

## 5. Risks this iteration takes, stated plainly

**Merging two halves costs attribution.** iter-94's spec explicitly warned that
"widening the blast radius on a run that already changes the ruler would make a
regression unattributable". This run does exactly that, deliberately, because a paid
iteration is ~28 h. The mitigations are real but not airtight: the zero-init in §3.4,
disjoint acceptance probes for each half (§4.1-2 vs §4.3), and the fact that Half A is
verifiable **locally, before dispatch** on a smoke-trained model.

**State dim 24 → 27 changes `input_dim` for every module.** Adding markers is not
additive — every module's first layer changes shape, so the whole learned solution
shifts and no iter-94 weights transfer. This is a full retrain regardless, but it means
"Half B was a no-op at init" does not extend to "Half B was a no-op at convergence".

**A3 changes a scale that other code reads.** `NORM_SCALE[bhb]` feeds the clamp, the
distillation level terms, and any cohort spec expressed in z. Audit every consumer;
a silent double-count here would look exactly like a physiology result.

**The biliary teacher is new physics with no real ground truth.** Every bile-acid number
the student learns will come from our own teacher. `cgm_real` cannot check it. That is
the same circularity the iter-94 spec disclosed for `teacher_dynamic`, and it should be
disclosed again rather than discovered later.

---

## 6. Ops

- `trainer-rg58t` — `--benchmark-only` on `training/jobs/iter94/model.pt` against
  `benchmark.dataset.iter94.json`, report to
  `gs://grovina-pulse-data/training/jobs/iter94/benchmark-report.json`. Dispatched
  2026-08-20 on image `trainer:40ca3b9` — **the same image and args as the A5 baseline**
  (`trainer-v2w6z`, 11h43m), so the comparison is apples-to-apples. Its purpose is the
  iter-94 B1 risk check: teacher-source agreement improving while `cgm_real` skill
  degrades. Read the result from the **artifact**, never `succeededCount`.
- Half A is verifiable locally before any paid dispatch. Do that first.
