# Pulse

> Training a model of the body against the medical literature.

## The idea

Most whole-body physiology simulators are **forward-only**: give them a patient and they predict
what happens. They can't *learn* from new evidence — incorporating a finding means a human
hand-tuning parameters inside a coupled multi-system model.

Pulse is **differentiable end to end**. A published result — an RCT effect size, a dose-response
curve — can be written as a loss term, and the optimiser discovers which internal dynamics
reproduce it. The model doesn't just *contain* medical knowledge; it *optimises agreement* with it.

The thesis, in one line: **a trainable synthesis of physiology, updated by the literature itself.**

## How it works

The body is modelled as a network of coupled physiological subsystems. What is **imposed** and what
is **learned** follow a deliberate hierarchy:

| Layer | Nature | In the model |
|---|---|---|
| Physics / chemistry | Universal laws | Enforced in the architecture (mass-action, conservation) |
| Anatomy | How the body is built | Module boundaries and the coupling graph |
| Medical knowledge | What typically happens | Training signal and coupling-sign priors |
| Individual variation | This person | Learned from their own data |

Seven subsystems — metabolic, appetite, stress, cardiovascular, thermoregulation, respiratory, gut —
are wired along a sparse, anatomically faithful coupling graph. Chemical species follow mass-action
rate laws; vital signs use learned dynamics constrained by the known couplings. A single glucose
reading propagates through real pathways instead of an entangled hidden state.

## Running it

Uses [uv](https://docs.astral.sh/uv/):

```bash
uv sync                            # CPU PyTorch by default (macOS, CI, Cloud Run)
uv run python -m pulse.train       # train against synthetic episodes + knowledge losses
uv run python -m pulse.benchmark   # evaluate against the benchmark/gate
uv run uvicorn pulse.server:app    # serve inference
```

On an NVIDIA host, opt into the CUDA build instead:

```bash
uv sync --no-default-groups --group gpu
```

## Layout

- `pulse/` — the model: `model.py`, the subsystem `modules/`, the encoded `knowledge/`, the
  differentiable losses (`*_loss.py`), the trainer (`train.py`), `benchmark.py`, `verifier.py`, and
  a FastAPI `server.py`.
- `scripts/` — standalone analysis + ingestion tools (run with `uv run`).
- `deploy/` — Cloud Run deploy (`engine` service + `trainer` job): `setup-gcp.sh`,
  `cloudbuild.yaml`, `deploy.sh`. Hardware/cloud-agnostic, parameterized via env.
- `docs/` — design and history (below).

## Documents

- [docs/prd.md](docs/prd.md) — the vision, prior art, and the philosophy of many weak constraints.
- [docs/roadmap.md](docs/roadmap.md) — where it stands and where it's headed.
- [docs/training-runs.md](docs/training-runs.md) — how to launch local + Cloud Run training.
- [docs/](docs/) — design notes plus the iteration history. The `iter<N>-handoff.md` files are a
  running lab notebook; they predate this standalone repo, so their paths/commands reflect the
  earlier monorepo layout.

## Prior art

The whole-body simulators Pulse learns from — HumMod, BioGears, and the lineage back to Guyton's
1970s circulatory model. Remarkable forward-only engineering; Pulse's one departure is making the
whole thing differentiable.

## License

[MIT](LICENSE) — © 2026 Gabriel Rovina.
