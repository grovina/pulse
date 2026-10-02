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

## Where the time goes (iter 108)

Every signal is a rollout of the minute-step ODE, and a step is **dispatch-bound**:
the model is ~38 K parameters, so the cost of a simulated minute is the number of
tensor ops, not their size. Iter 107's `model.forward` issued ~1,465 ops per minute
(67 separate small matmuls, per-step setpoint nets, time features, coupling
assembly and rate scatter); a 240-min window cost ~5.5 s forward + backward on one
core, and a batch of 32 windows cost about the same as one.

What changed, and what each part bought (measured on one core, same weights, a
scaled-down phase-2 epoch of `train/spec.json` with one full aux step; the last
column adds `--compile-steps`, steady state):

| signal | iter 107 | iter 108 | + compiled steps |
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
  time point, so there is one physiology implementation. ~530 ops per minute.
- **Fused heads** (`model._HeadBank`). The ten MLP heads run as one block-diagonal
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
`uv run --group dev pytest tests/`.

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
