# Pulse training runs

How we launch GCP training jobs. The entrypoint is
`apps/pulse/scripts/train-submit.sh`. There is no local-training mode —
training always runs on Cloud Run, against the spec at
`apps/pulse/train/spec.json`.

The script splits cleanly into two phases:

1. **Deploy** (idempotent): `./m app deploy pulse-train` builds the
   trainer image (`apps/pulse/train/cloudbuild.yaml`, `--target=train`)
   and refreshes the `pulse-trainer` Cloud Run job spec. Fast no-op
   when the image is already current.
2. **Run** (ephemeral): `gcloud run jobs execute pulse-trainer --args=...`
   invokes the deployed image with per-run flag overrides
   (`--gcs-bucket`, `--gcs-object`, `--benchmark-*`, etc.) — same shape
   as `python -m pulse.train --gcs-bucket=... --gcs-object=...` locally.
   Cloud Logging is tailed live; on completion, `pulse.diagnostics
   compare` runs against the most recent prior job that has both a
   checkpoint and a benchmark report, and the local script enforces
   the benchmark gate.

`--skip-deploy` short-circuits step 1 when iterating on the orchestrator
itself.

## Commit before submit

**Commit the code and spec you intend to train before running
`train-submit.sh`.** `meta.json` records `gitSha` and `gitClean`. A
clean commit ties each job ID to an exact tree, makes diffs and rollback
obvious, and avoids "which uncommitted edit was in that run?" confusion
when comparing benchmarks or handoffs.

Typical sequence:

1. Implement the change and update `apps/pulse/train/spec.json` (and
   the new `iter<N>-handoff.md` when applicable).
2. Run tests: `cd apps/pulse/engine && .venv/bin/python -m pytest tests/`.
3. `git add` / `git commit` with a message that states the iteration
   and the main code change.
4. `bash apps/pulse/scripts/train-submit.sh` from the repo root.

CI-driven submits can pass `--git-sha` explicitly; otherwise commit-first
keeps `meta.json` honest.

The script blocks for the whole run (it does the post-run benchmark diff
once training completes), so an agent calling it through a tool with a
short timeout will see the wrapper killed long before the job finishes —
even though the job *did* launch (`gcloud run jobs execute --async`). To
stop that from leaking duplicate concurrent runs (iter 72 got
triple-dispatched this way), `train-submit.sh` refuses to launch when a
`pulse-trainer` execution is already running. Pass `--force` for the rare
intentional concurrent run. To check / clean up runaway duplicates:
`gcloud run jobs executions list --job=pulse-trainer --region=europe-west1`
then `... executions cancel <name>`.

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

- Active structural thread: `apps/pulse/docs/dead-pathways.md`.
- Latest iteration handoff: `apps/pulse/docs/iter<N>-handoff.md`.
- Script usage: header comment in `apps/pulse/scripts/train-submit.sh`.
- Trainer deploy yaml: `apps/pulse/train/cloudbuild.yaml` (pure deploy —
  no execute step).
