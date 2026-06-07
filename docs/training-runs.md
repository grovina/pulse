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
   `BUCKET`); see the script header.
2. **Run** — execute with per-run flag overrides:
   ```bash
   gcloud run jobs execute trainer --region europe-west1 \
     --args=--gcs-bucket=<bucket>,--gcs-object=training/jobs/<id>/model.pt
   ```
   Results land in `gs://<bucket>/training/jobs/<id>/`. List/inspect executions
   with `gcloud run jobs executions list --job=trainer --region=europe-west1`;
   logs stream to Cloud Logging. After a run, compare against a prior job with
   `pulse.diagnostics compare` (download both jobs' artifacts from GCS first).

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
