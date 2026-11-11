# Pulse training runs

`pulse.train` runs either **locally** (`uv run python -m pulse.train …`, fine
for small CPU runs) or as the **`trainer` Cloud Run job** for long runs. Both
use the same entry point and the same flags — the job is just the same image
(`Dockerfile` `train` target) running in the cloud.

## Local

```bash
uv run python -m pulse.train --n-patients 20 --n-epochs 80   # + other flags
uv run python -m pulse.benchmark                             # evaluate the gate
```

Pass `--gcs-bucket` + `--gcs-object` to upload the checkpoint and benchmark
report to `gs://<bucket>/training/jobs/<id>/`; omit them to keep a run fully
local.

## Cloud Run job

1. **Deploy** (idempotent) — build the images and create/update the `trainer`
   job: `bash deploy/deploy.sh`. Config via env (`PROJECT` / `REGION` /
   `BUCKET`); see the script header. The job is 8 vCPU / 32 Gi / 44 h, which
   is what the current `train/spec.json` recipe needs (iter 98 ran 34 h on
   that size; the previous 4 CPU / 16 Gi / 24 h defaults killed it).
2. **Run** — execute with per-run flag overrides:
   ```bash
   gcloud run jobs execute trainer --region europe-west1 \
     --args=--spec=train/spec.json,--gcs-bucket=<bucket>,--gcs-object=training/jobs/<id>/model.pt
   ```
   Results land in `gs://<bucket>/training/jobs/<id>/`. List/inspect executions
   with `gcloud run jobs executions list --job=trainer --region=europe-west1`;
   logs stream to Cloud Logging. After a run, compare against a prior job with
   `pulse.diagnostics compare` (download both jobs' artifacts from GCS first).

## Where the time goes

Every signal is a rollout of the minute-step ODE, and a step is **dispatch-bound**:
the model is ~38 K parameters, so the cost of a simulated minute is the number of
tensor ops, not their size. Iter 107's `model.forward` issued ~1,465 ops per minute
(67 separate small matmuls, per-step setpoint nets, time features, coupling
assembly and rate scatter); a 240-min window cost ~5.5 s forward + backward on one
core, and a batch of 32 windows cost about the same as one.

What changed, and what each part bought (measured on one core, same weights, a
scaled-down phase-2 epoch of `train/spec.json` with one full aux step; the last
column adds `--compile-steps`, steady state):

| signal | before | planned step | + compiled steps |
|---|---|---|---|
| trajectory window (phase 2, incl. coupling prior) | 12 s | 2.2 s (5.5x) | 0.6 s (20x) |
| cohort aux step (3 groups) | 228 s | 15.6 s (15x) | 6 s (38x) |
| cold-model distillation step | 238 s | 43 s (5.5x) | 12 s (20x) |
| carb mass balance | 47 s | 2.8 s (17x) | 1 s (47x) |
| physiology rules (2 arm groups) | 34 s | 2.9 s (12x) | 1 s (34x) |
| dose response | 16.5 s | 2.9 s (5.7x) | 1 s (16x) |
| **epoch** | **633 s** | **82.5 s (7.7x)** | **26.7 s (24x)** |

Peak RSS fell from 6.7 GB to 1.9 GB. For the full iter-107 recipe (30 trajectory-only
epochs, 28 with the literature stack) the same per-signal costs put a run at about
1/7 of its old wall-clock eager and about 1/20 with compiled steps.

- **Planned integration** (`model._RolloutPlan`). A module's rate is split by what it
  depends on (`modules/base.PhysiologyModule`): `constants(embedding)` — the patient;
  `drives(external, coupling, time)` — the protocol; `step(state, …)` — the ODE state.
  `integrate` evaluates the first two once per rollout over the whole `[B, T]` window
  and runs only `step` per minute; `model.forward` composes the same three for one
  time point, so there is one physiology implementation. ~530 ops per minute then; 979
  aten dispatches per minute today, counted by a `TorchDispatchMode`, which counts more
  than the earlier figure did.
- **Fused heads** (`model._HeadBank`). The eight MLP heads run as one block-diagonal
  three-layer network per step; the protocol and embedding halves of every first
  layer are computed once per rollout.
- **Batching independent rollouts.** A step costs about the same at batch 1 and 64,
  so everything that is not a sequential optimizer step rolls together: a cohort
  aux step's groups and arms (`cohort_loss.rollout_arms`, arms of different lengths
  share a rollout via `integrate(member_steps=…)`), the rules' arm groups, every dose
  of the dose response, every (dose, patient) pair of the carb balance, the
  distillation's level windows, the insulin-sweep points, and the coupling prior —
  which used to make ~210 single-row forwards per window (3 samples x 35 edges x 2)
  and now makes one (280x).
- **No gradient checkpointing.** With the lean step it measured 2.5x slower AND used
  more memory (its saved-tensor hooks now cost more than the graph).
- **One intra-op thread** (`--torch-threads`, default 1). Threads have nothing to
  split in a dispatch-bound step; 4 threads next to one other busy process made a
  window 8x slower.

### The evidence step

A phase-2 optimizer step samples `--evidence-batch-size` cohort items and one rule and
rolls them together (`training/evidence_step.accumulate_evidence_step`; the trainer only
calls it).

1. **Draw people** term by term, in order, exactly as `item_nll` / `rule_nll` do.
2. **Spin up once.** Every row's fed day (plus its arm's pre-fast) is one no-grad
   `integrate` call per distinct spin-up length, each member on its own clock, meals and
   sleep (`arm_init.student_initial_states_for_rows`). Before, each arm of each item
   redid its own 1,440-minute day, 63% of all simulated minutes in the corpus.
3. **Roll in groups.** Items are grouped by horizon (`_group_units`): a group stops growing
   when the next item is shorter than half its longest, or when rows x horizon passes
   120,000 row-minutes (about 80 day-long rows), and observing items group apart from plain
   ones. A group is ONE `rollout_arms` call with every arm of every member item as rows, so
   all the arms of an item share one graph, and short arms are held (`member_steps`), not
   padded out to a day.
4. **Score, then backward per group.** Each item is scored on its own rows, the group's loss
   is the weighted sum of its items' NLLs, and `accumulate_grad` backpropagates it before the
   next group is rolled, so one graph is alive at a time. The step's loss is the same weighted
   sum of per-item NLLs as before; `tests/test_evidence_step_batched.py` pins the per-item NLL
   (1e-4 relative) and the accumulated gradient (cosine > 0.9999, norm within 1e-3) against
   the item-by-item path.

**Clipping is per group.** `--aux-signal-clip` used to clip each item's gradient before it
joined the step; one backward per group clips the group's gradient as a whole. A
non-finite loss or gradient likewise drops the group rather than the item. Each epoch logs
`[CLIP] ... evidence_backwards=N clipped=K` (also `evidence_backwards` / `evidence_clipped`
in the epoch metrics) so a run shows whether the clip binds.

**Graph memory.** A rollout with gradient holds about 1.2 MB per simulated minute however
few rows it has, plus about 16 KB per row-minute (peak RSS over the model, 240 min:
287 MB at 1 row, 354 MB at 32, 554 MB at 128, 773 MB at 256). Up to ~75 rows the fixed part
dominates, so adding rows is nearly free; past that rows dominate. The item-by-item path held
every arm of an item at once (an item with two 2,521-minute arms is two graphs of
~3 GB each, before the backward's own buffers); the batched step holds one.
Across eight consecutive steps resident memory was flat (no growth) and `malloc_trim`
returned the allocator's retained pages, so there is no cache to clear: the 10.7 GB peak of
iter 115 is the graph of its longest items plus allocator retention.

**Measured** (`python -m pulse.diagnostics step-time`, CPU seconds per step, one torch thread,
untrained default model, fed day 1,440 min, arms capped at 300 min, 4 draws per arm, one
2-draw rule per step, on a 2-CPU box shared with other jobs):

| `--evidence-batch-size` | item by item | batched | spin-up / rollout+score / backward | groups, rows |
|---|---|---|---|---|
| 4 (mean of 2 steps) | 47.5 s | 15.7 s (3.0x) | 8.0 / 2.8 / 4.9 s | 4, 22 |
| 16 (1 step) | 228 s | 37.3 s (6.1x) | 24.4 / 4.3 / 8.6 s | 5, 98 |

Those two rows were measured while `integrate` still built the gut and duodenal windows of
members with different meals one member at a time, which is what the spin-up of a step is.
With one stomach run for the whole batch (`model._meal_outputs`) the batch-size-4 step
measures 39.8 s item by item and 9.9 s batched (spin-up 3.4 s, rollout and score 2.3 s,
backward 4.2 s; peak RSS 1.3 GB). Uncapped steps are dominated by the long rollouts, which
this box cannot hold with gradient: run the same command with `--max-arm-minutes 0` in the
cloud.

**A run that dies leaves a board.** At the end of phase 1 (and of phase 2 when phase 3
follows) the trainer scores the model on the literature board without the identifiability
pass and without the individual ruler and writes
`<stem>.phase<N>.scoreboard.json` beside the checkpoint (uploaded next to `last_good.pt`).
`--deadline-hours H` stops training after the epoch whose successor, at that epoch's
duration, would end past `H` hours; the normal end-of-run checkpoint, board and upload
follow, so set `H` below the task timeout by the time that end-of-run scoring takes.

**Across processes.** The groups of a step are independent given the weights, so the way
to use more cores is one worker per group: the parent draws the people (keeping the rng
order), runs the spin-up, ships each worker the weights, its rows and its terms, and each
worker returns its group's gradient and NLLs; the parent applies the per-group clip and sums.
The tape, dose-response and carb-balance signals are separate tasks of the same shape.

`tests/test_planned_integrator.py` pins the equivalence: the planned rollout matches
the per-minute `model.forward` reference (`integrate(..., planned=False)` — also the
path to use when module or head forward hooks must fire), per-member protocols match
separate rollouts, and the batched helpers match their serial forms. Trajectories
agree to ~1e-7 relative over a window (float re-association in the fused heads),
growing to ~1e-4 NORM_SCALE units over a 48 h rollout.

### Writing a module so it stays fast

- Per-patient quantities (setpoint heads, transforms of population scalars) go in
  `constants`; anything that reads only time, sleep/activity or the gut / duodenal
  channels goes in `drives`; only state-dependent terms go in `step`. A drive that
  reads an ODE-marker coupling channel gets NaN (the plan passes the state as NaN)
  and the equivalence test fails loudly.
- Keep MLP heads `Linear-Tanh-Linear-Tanh-Linear` (the bank checks), declare them in
  `mlp_heads()`, and turn raw outputs into fluxes in the head's `post`.

### More speed, as choices rather than defaults

- `--compile-steps` hands the planned minute step to `torch.compile` (the trainer
  image carries g++ for it): forward and backward become a few fused kernels each —
  a batch-1 window measured 0.41 s against 2.0 s eager, with the same trajectories
  and gradients. Its four variants (batch 1 vs N, with and without grad) compile at
  startup, in about a minute, and are checked against the eager step there; if that
  fails, compiled steps stay off and the log says why.
- `--trajectory-windows-per-step k` rolls k windows per optimizer step as one batch
  and averages their losses: roughly k x the window throughput, for k x fewer
  imitation steps per epoch. It changes the optimization schedule, so it belongs in a
  recipe with its own hypothesis (e.g. k = 4 with the LR re-tuned), not in a silent
  default. `--aux-every-k-windows` keeps counting windows.
- The aux cadence in `train/spec.json` (aux every 12 windows, distillation and carb
  balance every 4th aux step) was set by what Cloud Run could afford at iter 97. It
  can now afford several times more literature steps per epoch.

## Commit before you launch

Commit the code you intend to train before deploying — the trainer image is
built from your working tree, so a clean commit ties each `jobId` to an exact
tree and avoids "which uncommitted edit was in that run?" confusion when
comparing benchmarks or handoffs. Run the tests first:
`uv run --group dev pytest tests/ -n auto` (the suite is rollout-heavy, so it
runs across cores).

## Choosing the next iteration: structural over parametric

Prefer fixing the *structure* — essential constraints, the coupling
graph, which signals reach which parameters, the architecture — over
tuning hyperparameters. Param-tuning chains feel productive but they
overfit to the bench's idiosyncrasies and they cannot move a
structural ceiling: when a benchmark number stays *byte-identical*
across two or more iters of sweeps in its supposed control variable,
the bottleneck is upstream of any knob (the canonical case: dead-
pathway MAPE flat across iters 38-46 regardless of physiology-rule
weight, sampling, or multi-arm — see `dead-pathways.md` — because *no
training signal reached those parameters at all*). If a robustly
architected model is doing its job, params should be *resilient* —
they should matter less, not more.

In practice:

- When a metric is flat across ≥2 param-sweep iters of its control
  variable, stop sweeping. Do a diagnostic dive: which gradient /
  signal / connection is supposed to move this, and does it reach the
  relevant part of state / parameter space at all?
- Prefer one structural change (new signal, new constraint, coupling-
  graph edit, architecture tweak) over N parameter sweeps when both
  are on the table for the next iter.
- Single-variable param swaps are still the right tool for *isolating
  a cause* once a structural hypothesis is in play (e.g. iter-45
  cleanly isolated `sample_patients=10` as iter-44's culprit). The
  anti-pattern is chained sweeps in search of marginal gains.
- Each iter's `spec.json` hypothesis should state what *fundamental
  property of the system* the change tests. If the honest answer is
  "none, it's a tuning sweep," reconsider — an iter that doesn't
  generate a learning is usually overfitting.
- Bias toward *cleaner* over *faster/hackier*: a clean break (e.g.
  moving dead test fixtures out of production code, replacing a
  band-aid signal rather than stacking another on top) beats a quick
  patch even when both pass the gate.

## Further reading

- Active structural thread: `dead-pathways.md`.
- Latest iteration handoff: `iter<N>-handoff.md` (historical lab notes).
- Deploy mechanics: `deploy/deploy.sh` + `deploy/cloudbuild.yaml`.
