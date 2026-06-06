# Pulse PRD

## Vision

Pulse is a personalized physiology simulator. It models the human body as a network of interconnected physiological subsystems, each informed by medical science and refined by individual data. The model integrates knowledge from published research with sparse, cheap signals from the user — meals, feelings, occasional measurements — to produce a coherent picture of what's happening inside a specific person's body.

The long-term aspiration: a living synthesis of medical knowledge. Each published finding, each clinical insight, each established physiological relationship becomes a contribution that the model absorbs and integrates. Over time, the system accumulates the breadth of medical science into a single coherent simulation — personalized to each individual through their own data.

### Prior art and what's different

Whole-body physiological simulators have existed for decades. HumMod (10,000+ variables, 5,000+ papers, XML-based ODEs) grew out of Guyton's circulatory model from the 1970s. BioGears (open-source C++, real-time, patient parameterization) brought it into modern software engineering. AIBODY (132,000 parameters, sub-cellular resolution) pushed fidelity further. These are impressive engineering efforts, and they share a common limitation: they are forward-only simulators. They can answer "given this patient configuration, what happens?" but they cannot answer "given this published finding, how should the model change?"

Pulse is differentiable end-to-end. This means every published finding — an RCT reporting a mean glucose reduction, a cohort study linking sleep deprivation to insulin resistance, a dose-response curve for a dietary intervention — becomes a gradient signal that directly updates model parameters. The model doesn't just *contain* medical knowledge; it *optimizes agreement* with it. This is the qualitative difference from prior art, and it is the reason the model uses learned dynamics (neural modules) rather than hand-tuned equation systems.

## Philosophy

### What we impose vs. what we learn

Physiology is governed by physics and chemistry — conservation of mass, reaction kinetics, thermodynamics. These are nature's constraints. Everything else — parameter values, regulatory gains, coupling strengths, circadian amplitudes — varies between individuals and changes over time.

Beyond physics and chemistry, the human body has anatomical structure. The pancreas connects to the bloodstream. The gut absorbs nutrients and delivers them to the portal circulation. The HPA axis forms a specific feedback loop. This structure is not a modeling choice — it is a physical fact about how bodies are built.

Medical science gives us a third layer of knowledge: not physical laws, but well-established patterns. Insulin suppresses blood glucose. Cortisol follows a circadian rhythm. Heart rate responds to autonomic tone. These are observations, not laws — they describe what typically happens, not what must happen. They are priors, not constraints.

This gives us a clear hierarchy:

| Layer | Nature | Role in the model | Example |
|-------|--------|-------------------|---------|
| Physics / chemistry | Universal laws | Enforced in architecture | Mass-action kinetics, conservation of mass |
| Anatomy / structure | How the body is built | Module boundaries, coupling graph | Gut → blood, HPA feedback loop, SBP > DBP |
| Medical knowledge | What typically happens | Training data, coupling sign priors | Insulin → glucose is suppressive, cortisol peaks in morning |
| Individual variation | What happens in this person | Learned from user data via embedding | Exact insulin sensitivity, resting heart rate |

We enforce the first two layers in the architecture. We use the third as prior knowledge that guides learning. We learn the fourth from each person's data.

This hierarchy is the boundary between using medical insight and limiting the model. The sign of a coupling (insulin suppresses glucose) is anatomical — the receptor pathway exists and is suppressive. The strength of that coupling in a specific person is individual variation. We encode the sign; we learn the strength.

### Why modularity

A monolithic neural network that takes in the full marker state vector and produces a matching rate for each component can, in principle, learn any dynamics. But it has no structural guidance. A glucose reading enters a shared 128-dim encoder and must somehow influence heart rate, cortisol, and ghrelin — all through entangled hidden state. The model has to discover which connections exist from scratch, which requires far more data than sparse user check-ins can provide.

The human body is itself modular — it is organized into organs and organ systems with well-defined interfaces. Encoding this modularity is not a modeling preference; it reflects the physical structure of the system being modeled.

Modularity provides three concrete advantages:

**Information amplification.** When the model knows from anatomy that insulin suppresses ghrelin, a single glucose reading doesn't just constrain glucose — it propagates through the coupling graph. Glucose constrains insulin (they're in the same module with known coupling), insulin constrains ghrelin (cross-module coupling), ghrelin constrains the interpretation of hunger reports. Medical knowledge multiplies the information content of each observation.

**Reduced search space.** Each module has a small network with targeted inputs rather than one large network seeing everything. The model doesn't have to discover that temperature doesn't affect glucagon — that non-connection is given structurally. Learning is focused on the dynamics within and between known connections.

**Incremental knowledge accumulation.** A new paper about GLP-1's effect on cholesterol might add or sharpen a coupling edge, adjust priors on an existing pathway, contribute cohort-level supervision (mean difference between arms), and supply training episodes that exercise the mechanism. The modular structure makes it natural to add, update, or remove individual pieces of medical knowledge without disrupting the rest of the model.

### Why differentiability

A hand-tuned ODE system (like the knowledge contribution generators we use for bootstrapping) can produce plausible forward trajectories. But it cannot learn from the literature at scale. When a paper reports "16-week Mediterranean diet intervention reduced fasting glucose by 7.2 mg/dL vs control," incorporating that into a hand-tuned system requires a human to figure out which parameters to adjust and by how much — an intractable task in a coupled multi-system model.

In a differentiable model, the same finding becomes a loss function: simulate a control cohort and an intervention cohort, compute the mean fasting glucose difference, penalize deviation from the reported effect size, and backpropagate. The optimizer discovers which internal dynamics need to shift to produce the observed population-level effect. This works for any form of published evidence — RCTs, cohort studies, dose-response curves, case-control comparisons — as long as the claim can be expressed as a differentiable function of simulated outputs.

This is why the model uses learned dynamics (neural modules constrained by physics and anatomy) rather than a fixed equation system. The equations in our knowledge contributions are valuable as bootstrapping signal and as interpretable references, but they are not the runtime model. The runtime model is the differentiable network that has been trained against those equations, against coupling priors, against verifier checks, and — increasingly — against the published literature directly.

### Many weak constraints

A single glucose reading is nearly useless. A single "I feel hungry" report tells you almost nothing. But a glucose reading plus a hunger report plus a heart rate measurement plus a logged meal, accumulated over days and weeks, collectively constrain the model enough to personalize it.

This is the product bet: that many cheap, sparse, imperfect signals can determine a useful personalized model. The key requirement is that signals must span enough independent physiological axes. The modular architecture helps: coupling propagation means each signal constrains more axes than it would in isolation.

### Calibrated supervision

Match supervision to the evidence we have — no more, no less.

**Don't enforce more than we know.** Pinning a marker to a specific value we never measured, or fitting a rough reference trajectory point-for-point, encodes invented specificity as truth. Relaxed losses — inequality, band, sign, trend, robust — keep supervision inside the boundary of evidence. This is why teacher trajectories act as fences rather than pointwise targets when the reference is approximate.

**Don't enforce less than we know either.** When the literature establishes that glucagon falls after meals, that FFA tracks inversely with insulin postprandially, or that cortisol peaks in the early morning, declining to encode those facts because they aren't precise numbers is a failure to use available evidence. The model has no other way to learn them. Markers that stay flat after training are usually a symptom of this — not a representational limit, but supervision the literature could have provided that we failed to formalize.

The two failure modes are symmetric. Over-enforcement bakes invented specifics into the model. Under-enforcement leaves entire pathways unsupervised — visible as flat markers, missed dose-responses, or violated qualitative relationships at evaluation time.

The operational consequence: physiology knowledge enters as a registry of constraints in many shapes — point values where we measure them, bands where we know ranges, signs where we know directions, timing windows where we know phases, hinge-shaped relations where we know how variables move together. Each constraint is sized to the strength of evidence behind it. Strong claims become tight constraints, weak claims loose ones, unknowns remain unconstrained.

### Epistemological humility

Not all signals carry equal certainty, not all signals are present, and real-world data is inherently messy.

**Signal uncertainty.** A glucose meter reading is a direct observation with known measurement error. "I feel hungry" is a subjective report that might reflect ghrelin, habit, boredom, or a skipped meal. The system treats every datapoint as evidence that shifts a probability distribution, not as a deterministic assertion. Specificity — how strongly a signal constrains the physiological state — varies by signal type and context.

**Missing data.** Users won't log every meal, every exercise session, or every sleep event. External inputs are helpful but never required. When an input is missing, the model falls back to learned defaults informed by the person embedding and time of day. The model is trained with systematic input dropout so it handles missing data as a normal condition, not an error.

**Context attenuation.** Signals can modulate each other's informativeness. Frequent urination with high water intake is less informative about glucose than frequent urination alone. The system accounts for alternative explanations co-present in the same observation.

**Gradual convergence.** No single observation should cause a large model update. The prior penalty keeps calibration conservative. Confidence builds only when multiple independent signals align over time.

## Architecture

### Person embedding

A learned latent vector that encodes who this person is — their constitutional characteristics, metabolic tendencies, autonomic tone, and other individual variation. It is not a snapshot of current physiology but a representation of the person that, combined with context (time, age, inputs), determines physiological dynamics.

The embedding is projected per module via learned transformations. The Metabolic module gets a metabolic-relevant view of the embedding; the Cardiovascular module gets a cardiovascular-relevant view. Same person, different aspects.

Calibration adjusts only the embedding, not the model weights. The population-level model (weights) captures how physiology works in general; the embedding captures how it works for this person.

### Modular physiology

The body is modeled as a network of seven physiological subsystems, each implemented as a module with its own learned dynamics.

Each module:
- Owns a set of state variables (markers)
- Receives coupling inputs from other modules through named interfaces
- Receives relevant external inputs (meals, time, sleep, activity)
- Receives a module-specific projection of the person embedding
- Computes rates of change for its state variables

Two module architecture types, chosen by the nature of the state:

**Mass-action modules** (for chemical species): enforce rate = production − consumption × concentration, where production and consumption are non-negative learned functions. This is fundamental chemistry — concentrations have non-negative production, and clearance is proportional to concentration.

**Learned-dynamics modules** (for vital signs): the rate of change is a fully learned function of inputs. No single correct equation governs heart rate or temperature — the model discovers the regulatory feedback.

### Coupling graph

Modules interact through explicit coupling variables — named, typed values exported by one module and consumed by another. The Metabolic module exports glucose and insulin; the Appetite module imports insulin. This reflects anatomy: insulin travels through the bloodstream to reach the cells that regulate ghrelin.

The coupling graph is sparse. Each module connects to a few others, not all. This sparsity is itself medical knowledge — the absence of an edge is as informative as its presence. Temperature doesn't directly affect glucagon production; there is no edge there.

For each coupling edge:
- **Existence** is an anatomical fact encoded in the architecture
- **Sign** (direction of effect) is a medical prior — regularized but learnable
- **Strength** (magnitude of effect) is fully learned from data

```
                  Meals ─────→ [GUT] ─── nutrients ──→ [METABOLIC] ←── cortisol ─── [STRESS]
                                  │                        │  │                       ↑
                            nutrient flag            insulin│  │glucose          glucose│
                                  │                        │  │                       │
                                  └───────→ [APPETITE] ←───┘  │         ┌─────────────┘
                                                               │         │
          Sleep/Wake ──→ all modules                    cortisol│    met_rate
          Activity ────→ METABOLIC, CARDIO, THERMO, RESP       │         │
          Time of day ─→ STRESS, APPETITE, THERMO              │         │
                                                               ▼         ▼
                                      [CARDIOVASCULAR] ←─ [THERMOREG] ←─ [METABOLIC]
                                            ↑
                                        temperature
                                            │
                                      [THERMOREG]

                                      [RESPIRATORY] ←── lactate ── [METABOLIC]
```

When a coupling source is unavailable (module not yet implemented, or during module-specific pre-training), the coupling input falls back to a learned default value. This ensures modules can be developed and tested incrementally.

### External inputs

The model receives several categories of external context:

**Time of day** — always available. Encoded as cyclic features that let the model learn time-dependent dynamics without imposing specific temporal patterns.

**Meals** — composition and timing, processed by the Gut module into nutrient appearance signals. Optional: when meals are not logged, the model uses embedding and time to estimate likely meal effects.

**Sleep/wake state** — whether the person is sleeping or awake. Profoundly affects most physiological systems. Optional: when not reported, the model infers a probabilistic sleep state from time and embedding.

**Activity level** — crude measure of physical exertion. Affects glucose, lactate, heart rate, temperature, cortisol. Optional: when not reported, the model assumes a default activity state.

All external inputs except time are optional. The model degrades gracefully with missing inputs rather than breaking — it simply falls back to less-personalized predictions. This is a core design requirement: the model must produce reasonable output with zero external inputs (just time and embedding) and get progressively better as more context is provided.

## Modules

### Gut & Absorption

Models how food transitions from mouth to bloodstream.

- **Architecture**: learned absorption function, not an ODE. Maps (meal composition, time since eating, embedding) to nutrient appearance rates. Multiple meals superimpose linearly — a physical simplification that holds for normal eating patterns.
- **Inputs**: meals (macros + timing), embedding
- **Exports**: glucose appearance rate, lipid appearance rate, amino acid appearance rate, nutrient sensing flag
- **Key medical knowledge**: gastric emptying is roughly exponential. Carbs absorb faster than protein, protein faster than fat. Absorption rates vary by individual and meal composition.

### Metabolic / Energy

Blood chemistry homeostasis — the core energy regulation system.

- **State**: glucose, insulin, glucagon, free fatty acids, β-hydroxybutyrate, lactate
- **Architecture**: mass-action kinetics
- **Coupling inputs**: nutrient appearance (Gut), cortisol (Stress)
- **External inputs**: activity level, sleep/wake
- **Exports**: glucose, insulin, lactate, metabolic rate
- **Key medical knowledge**: insulin responds to glucose and clears it. Glucagon counters insulin during fasting. Low insulin permits lipolysis (FFA rise) and ketogenesis (BHB rise). Cortisol drives hepatic glucose output. Activity increases glucose uptake and lactate production.

### Appetite & Satiety

Hunger and fullness signaling.

- **State**: ghrelin, leptin, GLP-1
- **Architecture**: mass-action kinetics
- **Coupling inputs**: insulin (Metabolic), nutrient sensing (Gut)
- **External inputs**: sleep/wake, time of day
- **Exports**: ghrelin, GLP-1
- **Key medical knowledge**: ghrelin rises during fasting, drops after meals. GLP-1 spikes with nutrient sensing in the gut. Leptin changes slowly and tracks long-term energy status.

### Stress / HPA Axis

Stress response and cortisol regulation.

- **State**: cortisol (with optional hidden state for HPA feedback dynamics)
- **Architecture**: mass-action kinetics
- **Coupling inputs**: glucose (Metabolic — hypoglycemia triggers cortisol release)
- **External inputs**: stress level (user-reported), sleep/wake, time of day
- **Exports**: cortisol
- **Key medical knowledge**: the HPA axis has negative feedback — cortisol suppresses its own upstream production. Stress activates the axis. The circadian cortisol pattern is a learned consequence of SCN input (represented through time features), not an imposed waveform.

### Cardiovascular

Heart and circulation.

- **State**: heart rate, HRV (RMSSD), systolic BP, diastolic BP
- **Architecture**: fully learned dynamics
- **Coupling inputs**: cortisol (Stress), temperature (Thermoregulation)
- **External inputs**: activity level, sleep/wake
- **Architectural constraint**: SBP > DBP (physical — systolic is during contraction, diastolic during relaxation)
- **Key medical knowledge**: baroreflex couples HR and BP via negative feedback. Autonomic balance modulates HR and HRV. Exercise and sleep produce large, predictable shifts.

### Thermoregulation

Body temperature control.

- **State**: core temperature
- **Architecture**: fully learned dynamics
- **Coupling inputs**: metabolic rate (Metabolic), cortisol (Stress)
- **External inputs**: activity level, sleep/wake, time of day
- **Key medical knowledge**: temperature has a circadian pattern (learned, not imposed). Diet-induced thermogenesis raises temperature after eating. Exercise raises it. Sleep lowers it.

### Respiratory

Breathing and oxygenation.

- **State**: respiratory rate, SpO₂
- **Architecture**: fully learned dynamics
- **Coupling inputs**: lactate (Metabolic)
- **External inputs**: activity level, sleep/wake
- **Key medical knowledge**: respiratory rate tracks metabolic CO₂ production. SpO₂ is normally stable and clinically relevant mainly in pathological states.

## Knowledge system

### Knowledge as data

The model's understanding of physiology comes from two sources: the architecture (which imposes physics and anatomy) and training data (which encodes medical knowledge). Training data is the vehicle through which all medical knowledge beyond physics and anatomy enters the model.

Each piece of medical knowledge — a published paper, a textbook chapter, a clinical guideline — is encoded as a knowledge contribution: a self-contained unit that defines what the learner should respect (episodes, priors, population-level targets, or a mix).

Over time, as contributions accumulate, the model absorbs more of medical science into a single coherent simulation. Adding a new contribution is like patching new knowledge into the model's understanding of the body.

**Episodes and targets as source of truth.** The artifacts we actually optimize against — trajectories, intervals, scalar summaries, inequality targets — are the operational source of truth. Mechanistic simulators, ODE templates, and procedural generators are **shortcuts** to produce those artifacts at scale; they are not authoritative merely because they ran. When a generator is too simplistic or out of date with the evidence, we **revise the episodes or targets** (by hand, by editor rules, or by replacing the generator). Manual curation and hand-adjusted tables are first-class, not a temporary workaround.

### Contributions

Each knowledge contribution is a file (or registered module) in the knowledge system. A contribution:
- Documents its source (citation, rationale)
- Supplies **training signal** appropriate to the claim: e.g. synthetic time series, hand-entered points, cohort statistics, or constraints derived from literature
- Optionally declares coupling priors: sign and magnitude range for coupling edges relevant to its domain

Contributions can range from complex dynamic models (Bergman glucose-insulin kinetics generating multi-hour trajectories) to simple manually curated datapoints (a table of expected cortisol values at different times of day). The training signal needs to be **plausible and well-scoped**; how it was produced does not matter for legitimacy.

**Teachers as fences, not mandatory pointwise truth.** When a contribution uses another model to propose trajectories, we need not require **dense, exact** agreement everywhere. The reference can act as a **fence**: match a qualitative trend, stay within a band around the reference values, enforce the correct sign over an interval, or use a robust loss that down-weights spurious detail. That preserves weak supervision where the reference is approximate and avoids baking in a toy model's precise errors.

### Mechanism-first ingestion

The highest-leverage way to absorb papers and textbooks is usually **mechanistic**: what influences what, through which interface, in which direction, on which timescale, and under which boundary conditions. That maps naturally to **coupling edges**, module boundaries, sign and range priors, and multi-step episodes that **exercise a pathway** rather than a single scalar headline.

Abstract-level claims (for example, a group mean difference after an intervention) are still valuable but **underdetermine** internal dynamics. They work best as **additional** weak constraints alongside mechanism-grounded structure and trajectory-level data, not as the sole source of supervision.

### Cohort and summary-statistic supervision

Most medical publications report population-level results: group means, effect sizes, confidence intervals, dose-response curves. This is the dominant form of medical evidence, and it is the primary vehicle through which published literature enters the model at scale.

A cohort contribution defines a **simulation protocol** (sample N virtual patients from the population prior, apply specified conditions to each arm, integrate forward) and a **target statistic** (the reported outcome). The training loop executes the protocol, computes the statistic from simulated outputs, and backpropagates the discrepancy against the target. Because the entire pipeline — population sampling, forward integration, statistic computation — is differentiable, gradients flow from the published finding directly into model weights.

This is the mechanism through which Pulse can absorb the medical literature systematically. Each encodable finding becomes a weak constraint, and the model gradient-descents agreement with all of them simultaneously. The breadth and diversity of constraints is what makes individual findings identifiable — any single cohort average is underdetermined, but hundreds of independent findings from different domains, combined with trajectory-level episodes, coupling priors, and verifier checks, collectively constrain the model toward accurate physiology.

Risks to manage: **identifiability** (summary losses are informative only together with other independent constraints), **confounding** (synthetic cohorts should match the stratification of real trials), and **heterogeneity** (effects apply to specific populations; contributions should state scope or carry uncertainty).

### Open knowledge accumulation

The knowledge contribution system is designed to be open and cumulative. The long-term aspiration is that researchers can contribute their findings directly: a team publishing a paper on the relationship between sleep deprivation and insulin sensitivity can encode that finding as a cohort contribution — simulation protocol plus target statistic — alongside the paper itself. A physiologist discovering a new coupling pathway can add an edge to the coupling graph with sign and magnitude priors. A clinical study can contribute validation scenarios as benchmark episodes.

Contributions take many forms, from the simple to the complex:
- **Coupling prior**: "GLP-1 suppresses glucagon" — a sign constraint on a coupling edge
- **Population statistic**: "median fasting glucose in healthy adults is 92 mg/dL" — a scalar target on the population mean
- **RCT result**: "intervention X reduced HbA1c by 0.5% vs placebo (n=200/arm)" — a cohort simulation protocol with a differential target
- **Mechanistic pathway**: Bergman minimal model equations generating glucose-insulin trajectories — an episode generator producing multi-hour time series
- **Validation scenario**: "after 75g OGTT, glucose peaks at 30-60 min and returns to baseline by 120 min" — a verifier check with specific boundary conditions

Each contribution is traceable to its source, can be updated or removed as understanding evolves, and the model can be retrained to reflect changes. The differentiable architecture means contributions compose naturally — they are all gradient signals into the same parameter space.

### Traceability

Every training signal traces to a knowledge source (episode set, prior, cohort objective, or verifier). This makes the model auditable — unexpected behavior can be traced to which contributions influence it. Contributions can be updated, corrected, or removed as medical understanding evolves, and the model retrained to reflect the change.

## Calibration

Calibration personalizes the model by adjusting the person embedding to best explain observed signals.

The process: integrate the modular ODE forward using the current embedding, compare predictions against check-in observations (measurements as direct fit, feelings and body signals as soft evidence), and optimize the embedding to maximize the joint posterior. Accept updates only when validation loss improves meaningfully. Update baseline state anchors on acceptance.

The modular architecture makes calibration more powerful: each observation propagates through the coupling graph, constraining multiple modules simultaneously. A glucose reading constrains not just glucose but — through learned coupling — insulin, ghrelin, and any other state connected via the graph.

Calibration adjusts only the embedding, never the model weights. The population-level model represents general physiology; the embedding represents this person.

### Future: amortized inference

The current calibration approach optimizes the embedding per request via gradient descent. An alternative is amortized inference: train a separate neural network (on the simulator itself) that maps directly from a set of sparse observations to a posterior distribution over embeddings. This would be faster at inference time, provide uncertainty estimates, and handle the sparse-data regime more gracefully. Simulation-based inference techniques (neural posterior estimation) are a natural fit — the differentiable forward model can generate unlimited training pairs of (embedding, observations) to train the inference network.

## Training doctrine

### Principles

1. Impose only physics, chemistry, and anatomy in the architecture. All dynamics are learned.
2. Use medical knowledge as training signal and coupling priors — not as hard-coded runtime dynamics that replace learned flows (except where physics or anatomy demands a structural fact).
3. **Optimize against contributed signal**, not against the internal details of whatever generator produced it. Episodes, bands, trends, and population summaries defined by contributions are what matter; generators are expendable implementation tools.
4. **Match supervision strength to the breadth of evidence behind it** (see *Calibrated supervision*). Use relaxed losses — inequality, band, sign, trend, robust — where the reference is rough or we know shape but not magnitude; never copy invented specificity. Equally: encode every plausibility constraint the literature supports — direction-of-change after events, cross-marker relations, timing bands, variability floors — even without a precise value. Failing to enforce knowable structure is as wrong as enforcing what we don't know.
5. Use **verifiers** as quality gates on properties that should hold globally or across scenarios. Use **cohort- and summary-statistic** losses where literature or trial summaries add **independent** population-level constraints. Both complement trajectory-level fit; neither replaces diverse episode data or mechanism-grounded structure.
6. Train with systematic input dropout so the model handles missing external inputs as a normal condition.
7. Modules can be pre-trained on module-specific data, then fine-tuned end-to-end with coupling active.
8. Iterate by adding knowledge contributions (mechanism-first where possible), improving data diversity, refining coupling priors, and tightening verifiers.

### Cold model bootstrapping

Mechanistic reference models — hand-tuned ODE systems grounded in published physiology — serve as the initial teacher for the differentiable model. They provide the starting point: plausible trajectories that the learned model reproduces before it encounters any other training signal. This is not a workaround or a temporary measure; it is a deliberate training phase. The cold model captures well-established average dynamics (Bergman glucose-insulin kinetics, circadian cortisol, baroreflex coupling) and gives the learned model a stable initialization from which to absorb more nuanced evidence.

The cold model is a teacher, not the runtime. Its job is to get the learned model into a regime where trajectories are physiologically plausible, after which cohort losses, coupling priors, verifiers, and eventually real-world data can refine it further. When the cold model is wrong or incomplete, the learned model can diverge from it — guided by other training signals — without structural penalty.

### Phased training

Training applies signal sources in stages of increasing subtlety:

1. **Episode distillation**: the learned model reproduces trajectories from mechanistic reference models. This establishes baseline dynamics and stable integration behavior.
2. **Coupling and structure**: coupling priors enforce sign constraints on cross-module sensitivities. Verifier checks assert global properties (circadian patterns, meal responses, range plausibility).
3. **Literature absorption**: cohort and summary-statistic losses from published findings refine population-level dynamics beyond what the reference models capture.
4. **Diversity and stress-testing**: varied scenarios, edge cases, input dropout, and adversarial conditions harden the model against distribution shift.

Each stage builds on the stability established by the previous one. Applying all pressures simultaneously from the start tends to cause instability; staging them gives the optimizer a tractable path.

### Training entry point

Comparable checkpoints should use **`apps/pulse/scripts/train-submit.sh`** and **`apps/pulse/train/spec.json`** so hyperparameters stay aligned.

- **Single script:** **`train-submit.sh`** — uploads **`spec.json`** + **`meta.json`** to GCS, runs **`./m app deploy pulse-train`** to refresh the trainer image + job spec (idempotent), then **`gcloud run jobs execute pulse-trainer --args=...`** (the image's `ENTRYPOINT` is `python -m pulse.train`; per-run inputs are flag overrides). Tails Cloud Logging while polling for completion; on success runs **`pulse.diagnostics compare`** vs the most recent prior job that has both a checkpoint and a benchmark report, and enforces the benchmark gate locally. Override the spec path with **`--spec`** (accepts **`gs://`**); skip the deploy step with **`--skip-deploy`** when iterating on the orchestrator and the image is already current. Benchmark dataset and thresholds URIs come from the spec (**`benchmarkDatasetUri`** / **`benchmarkThresholdsUri`** or **`infra.*`**) when set; otherwise **`gs://<bucket>/benchmarks/benchmark.dataset.generated.json`** and **`…/benchmark.thresholds.json`** with **`GCS_BUCKET`** (default **`grovina-pulse`**). Needs **`jq`**, **`gcloud`**, **`gsutil`**.
- **Spec file:** **`apps/pulse/train/spec.json`** — the repo's default recipe (`hypothesis`, `params.trainArgs`, optional **`infra`** / top-level URIs for project, bucket, benchmarks). Each job stores **`spec.json`** and **`meta.json`** under **`gs://<bucket>/training/jobs/<jobId>/`**; **`meta.json`** records **`gitSha`**, **`gitClean`**, and **`specPath`**. **`spec.template.json`** is only a blank starter for local copies. **Reproducing a recipe:** check out the **`gitSha`** from meta (or use **`HEAD`** on a branch you trust) and run **`train-submit.sh`** with that tree's **`spec.json`** (or pass **`--spec`** to the exact file, including **`gs://…/training/jobs/<jobId>/spec.json`** to re-run the frozen JSON from a past job). **Reproducing a full run artifact-for-artifact** still needs the same **`jobId`** prefix (checkpoint, reports) and the same infra inputs (benchmark URIs, engine image); random seeds and data drift can differ unless you pin those explicitly.
- **GCP (CI):** **`.github/workflows/pulse-train.yml`** — runs **`train-submit.sh`** when watched paths change and the commit message contains **`[pulse-train]`**. Secrets: **`PULSE_GCP_WORKLOAD_IDENTITY_PROVIDER`**, **`PULSE_GCP_SERVICE_ACCOUNT`**. The checked-out **`apps/pulse/train/spec.json`** is the spec; no repo-variable override.
- **Engine + trainer deploys:** **`apps/pulse/engine/cloudbuild.yaml`** builds the engine image (server target) and updates the **`pulse-engine`** Cloud Run service. **`apps/pulse/train/cloudbuild.yaml`** builds the trainer image (train target) and updates the **`pulse-trainer`** Cloud Run job spec. Both are pure deploys (no execute step) — invoked via **`./m app deploy pulse`** or **`./m app deploy pulse-engine`** / **`pulse-train`**.

## Check-in signals

| Signal type | How it constrains the model | Specificity |
|-------------|---------------------------|-------------|
| Measurements (glucose, HR, BP, temp) | Direct observation fit on corresponding module state | High |
| Feelings (hungry, full, stressed, tired, shaky) | Soft evidence on associated markers across modules | Low to moderate |
| Body signals (urination frequency, stool quality) | Soft evidence with context attenuation | Low |
| Meal composition (carbs, fats, proteins) | External input to Gut module | Input |
| Sleep/wake state | External input to all modules | Input |
| Activity level | External input to relevant modules | Input |
| Time of day | Cyclic features available to all modules | Input |

## UX principles

- Immediate value from the prior model — no onboarding barrier.
- Every datapoint refines personalization, but no datapoint is required.
- Lightweight input: tap feelings, optionally log a measurement or meal.
- Missing data is normal. The model degrades gracefully, never fails.
- The model improves silently in the background.

## Success criteria

- Modules produce physiologically plausible trajectories individually and coupled.
- Coupling amplifies information: observations on one module improve predictions for coupled modules.
- Calibration converges with realistic data sparsity (a few check-ins per day).
- Knowledge contributions are additive: new sources improve quality without regressing existing behavior.
- Cohort- and summary-statistic contributions, when used, move group-level behavior toward documented patterns without breaking trajectory-level plausibility or verifier expectations.
- The model handles missing external inputs without catastrophic degradation.
- Model behavior is traceable to knowledge sources.
