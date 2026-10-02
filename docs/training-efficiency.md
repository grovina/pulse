# Training efficiency

*Written 2026-10-01 against the iter-99 recipe (`train/spec.json`). Measured on a
4-vCPU sandbox, one core per process (`taskset`), PyTorch 2.12 CPU. Absolute times
are this box's; the ratios are what carry to the 8-vCPU Cloud Run trainer.*

## The one cost everything pays

Every training signal — trajectory imitation, cohort statistics, rule hinges,
dose-response, mass balance, distillation — pays for the same thing: an Euler
rollout of the student at dt = 1 min, forward and backward. A phase-2 epoch of the
iter-99 recipe runs on the order of 10⁵ such steps.

At HEAD (`843b908`) one step of the 56 k-parameter student cost **6.4 ms forward +
10.6 ms backward at batch 1, and 19 ms at batch 32** (one idle core; ~2x that on a
busy one): the step is pure dispatch overhead — ~1,300 top-level aten ops forward (~4,000 counting nested ones), ~600
autograd nodes — on tensors of a few dozen floats. Three consequences:

1. **About half of every step was state-independent work.** 47 of the 89 `Linear`
   calls per step depended only on the embedding or the clock: eight embedding
   projections, ten per-patient setpoint heads (FFA/glucagon/body-mass recomputed
   twice), the metabolic module's fasted-reference head evaluations, and the
   external/embedding/time columns of every first-layer matmul. Circadian drives
   (`atan2` hour, HPA curve, meal anticipation, nocturnal leptin), rate constants
   (`softplus`/`sigmoid` of scalars) and learned input defaults likewise.
2. **Batch width is free, so batching is the lever** — and the aux signals rolled
   their protocols out one at a time: one rollout per cohort spec group per arm,
   per rule arm group, per carb dose, per (embedding, dose) pair, per distillation
   anchor window; the coupling prior made ~190 batch-1 forwards per window.
3. **The machine is mostly idle.** Intra-op threads cannot help ops this small
   (4 threads vs 1: 9.3 s vs 9.9 s per window), so the 8-vCPU trainer runs on ~1.

## What changed

### 1. The vector field in two phases (numerically identical)

`modules/base.py`: every module splits into `prepare(external, embedding,
time_features)` — everything that does not read the ODE state, returned as
`(const, seq)` — and `step(x, p)` on `x = [own state ‖ coupling]`.
`model.integrate` calls `prepare` once per rollout with a leading time axis and
slices `seq` per step; `forward`/`fluxes` (the APIs tests and probes use) are
`prepare` without a time axis followed by `step`, so there is one code path.

Inside the step:

- each `MassActionModule` evaluates all of its MLP heads as one **head bank**
  (stacked weights, three batched matmuls) — every head keeps its own weights, so
  the iter-23 per-species isolation is untouched; only the arithmetic is batched;
- a module's input is ONE gather from `[normalized state ‖ gut ‖ duodenal]`, and the
  rates go back to `MARKERS` order with ONE gather (was 28 `index_put`s);
- the excluded `mitochondrial_capacity` column is a zero weight column rather than
  a per-step slice;
- `euler_step` handles HRV, pulse pressure and SpO₂ in one vectorized pass.

Checked against HEAD on a randomized model (every weight perturbed so zero-init
layers are live): rates, 200-step rollouts with meals/sleep/learned defaults/
checkpointing, `fluxes`, and the gradient of every parameter agree to **9e-6
relative** (float reassociation). A phase-1 epoch reproduces HEAD's loss and
gradient-norm log line digit for digit.

### 2. Many protocols, one rollout (gradient-identical)

`pulse/rollouts.py`: `rollout_many` stacks any number of protocols into one batched
`integrate` — each row with its own clock, meals (its own gut and duodenal tapes),
sleep/activity tape (`NaN` = withheld → learned default) and **length**
(`integrate(active_steps=)`: a finished row holds its last state while the longest
one runs on). Rows never interact, so summing the requests' losses into one
backward is gradient-identical to a backward per request.
`isolate_nonfinite=True` re-rolls the others without a request that diverged, so a
NaN still costs only its own group (in a shared batch its NaN activations would
otherwise reach every weight gradient through `0 · NaN`).

Used by: all cohort groups of an aux step; all rule arm groups; all dose-response
doses; all carb-mass-balance (embedding, dose) pairs; all distillation anchor
windows. Checked against HEAD on the same randomized model and RNG seed — worst
relative gradient error (vs each tensor's max): cohort 2.4e-4 (2,880-step
rollouts), dose + carb 4.7e-5, distillation anchors 1.7e-5, rules at float
noise.

The coupling prior evaluates every (sampled step × prior) finite difference as a
row of one batched forward: ~140x cheaper per window, loss equal to 6 digits, and
gradients equal to the float32 finite-difference noise floor (batching the same
`(r0, r1)` pair in the old loop moves them by the same 5e-3).

### 3. Bugs found on the way

- **Distillation calibration leaked its gradient into training.** `_calibrate`
  ran `loss.backward()` with the model's weights still requiring grad, so every
  calibration step added the calibration objective's gradient — unweighted,
  unclipped — to the shared `.grad` buffers the next joint aux step applies. Two
  calibration steps left a norm-53 gradient on 118 of 203 parameters (the joint
  clip is 10). Now inside `model.frozen_parameters`, as is
  `pulse.calibration.calibrate_embedding`; both also stop paying for weight
  gradients nobody wanted.
- **The coupling prior evaluated the gut on the absolute clock** (the iter-87
  frame) with zero duodenal delivery, justified as "cancels in the difference".
  It does not for edges whose target rate couples appearance with state (glucose:
  the plasma share of appearance depends on the glycogen pools). It now uses the
  window's own gut and duodenal tapes at the sampled step.
- A diverged cohort group could write `NaN` into the adaptive-weight EMA. Groups
  whose rollout diverges are now skipped before scoring.
- `train.py` used a nested same-quote f-string: a syntax error on Python < 3.12
  although `pyproject.toml` says `>=3.10`.

### 4. Exact shortcuts

- The distillation calibration integrates to the last check-in, not the end of
  the protocol (states after it cannot reach the objective) — the horizon
  `pulse.calibration` already uses. The 48 h sleep cohort calibrates on 1,441
  steps instead of 2,880; the post-prandial cohort on 181 instead of 480.

## Measured

Per Euler step, forward + backward, one idle core (`scripts/bench_training_step.py`):

| batch | HEAD | now (eager) | |
|---:|---:|---:|---:|
| 1 | 16.9 ms | 8.5 ms | 2.0x |
| 8 | 17.4 ms | 9.0 ms | 1.9x |
| 32 | 19.0 ms | 10.2 ms | 1.9x |

The mini recipe (`train/spec.json` with 6+2 patients, 1 window each, 1 epoch per
phase), per aux step at the start of phase 2, one core each on a busy box — the
per-step 2x times the batching:

| signal | HEAD | now | |
|---|---:|---:|---:|
| trajectory, per window (phase 1) | 9.1 s | 3.4 s | 2.7x |
| cohort statistics (3 groups) | 301 s | 27 s | 11x |
| carb mass balance | 70 s | 4.0 s | 17x |
| dose response | 26.5 s | 4.5 s | 5.9x |
| meal response | 14.0 s | 6.2 s | 2.3x |
| coupling prior, per window | ~5 s | 0.04 s | ~140x |
| peak RSS | 7.1 GB (OOM-killed in distillation) | 1.9-3.1 GB | |

## Opt-in levers (change the optimization or the runtime, not the math)

### `--trajectory-batch-windows B`

B trajectory windows per optimizer step, one batched rollout, loss = mean over
windows. At B = 1 the trajectory signal is bit-identical to before (same RNG draws
in the same order, same per-window loss). B = 8 costs ~1.1-1.2x a single window,
so phase 1 runs ~6-7x faster — but it is B-fold fewer, less noisy optimizer steps,
which is a different optimization (consider a higher LR). The aux cadence stays
"one aux step per k windows consumed", so the literature-to-imitation data ratio
does not move with B.

### `--compile-step` (or `PULSE_COMPILE_STEP=1`)

`torch.compile` of the integrator's Euler step (`model._model_step`) — the unit
that is ~600 small ops forward and as many autograd nodes. Per step, forward +
backward, one idle core, after warm-up:

| batch | HEAD | eager now | compiled |
|---:|---:|---:|---:|
| 1 | 16.9 ms | 8.5 ms | 2.98 ms |
| 8 | 17.4 ms | 9.0 ms | 3.46 ms |
| 32 | 19.0 ms | 10.2 ms | 4.26 ms |

i.e. 2.4-2.9x over the eager step and 4.5-5.7x over HEAD. Same math, so it is run
plumbing, not part of the recipe. Costs a C++ compiler at runtime — the trainer
image now installs `g++`; any compile failure falls back to eager, once, with a
warning — and a one-off compile of 1.5-7 min (one core) per graph: two graphs per
step signature (a rollout's first step, whose state usually carries no gradient, and
the rest), a signature being grad mode × frozen rows or not × checkpointed or not.
The batch is dynamic: traced with `TORCH_LOGS=recompiles`, widths 3, 5, 1 and 9
share the same two graphs once dynamo's duck sizing is off (left on, it tied the
batch to any equal-sized dim — 3 to the duodenal channels, 2 to the external inputs —
and recompiled per width) and per-step inputs have their own storage (views at offset
t·B made it guard on offsets).

Correctness, against eager on a randomized model: trajectories to 8e-7 (normalized),
and every parameter's and the embedding's gradient to float rounding — for a single
rollout, a heterogeneous batch with per-row lengths and inputs under gradient
checkpointing, and a frozen-weight calibration rollout. One trap found on the way:
**inductor's batch-1 specialization of the step miscomputes some weight gradients**
(torch 2.12; the metabolic heads' gut-appearance column, up to 45 % off, while the
trajectory and the embedding gradient are exact; `aot_eager` and every batch ≥ 2
graph are correct). So under compilation `integrate` runs a single rollout as two
identical rows and keeps the first — the twin row carries no gradient and costs
nothing.

## Where the next order of magnitude is

1. **Use the other 7 cores.** The aux signals of one joint step are independent
   given the weights; computing them in worker processes (each holding its own
   signal state and RNG stream) and summing gradients would cut an aux step to its
   slowest signal.
2. **The step count.** dt = 1 min forward Euler over 1-2-day protocols is ~10⁵
   steps an epoch. The fastest learned relaxation is ~0.5/min (CCK); an
   exponential (integrating-factor) step for the mass-action species is stable at
   any dt and would allow dt = 5 min — a 5x on everything, but it changes the
   discrete model, so it needs a retrain and the ruler.
3. **Distillation calibration** (8 sequential Adam steps over day-long rollouts per
   newly seen protocol) is now the largest single aux cost. Calibrating the whole
   protocol pool at once — Adam is elementwise, so one Adam over stacked
   embeddings IS one Adam per protocol — would amortize it across protocols.
