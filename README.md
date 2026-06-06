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
uv sync
uv run python -m pulse.train       # train against synthetic episodes + knowledge losses
uv run python -m pulse.benchmark   # evaluate against the benchmark/gate
uv run uvicorn pulse.server:app    # serve inference
```

## Layout

- `pulse/` — the model: `model.py`, the subsystem `modules/`, the encoded `knowledge/`, the
  differentiable losses (`*_loss.py`), the trainer (`train.py`), `benchmark.py`, `verifier.py`, and
  a FastAPI `server.py`.
- `docs/` — design and history (below).

## Documents

- [docs/prd.md](docs/prd.md) — the vision, prior art, and the philosophy of many weak constraints.
- [docs/roadmap.md](docs/roadmap.md) — where it stands and where it's headed.
- [docs/](docs/) — the iteration history, a running lab notebook.

## Prior art

The whole-body simulators Pulse learns from — HumMod, BioGears, and the lineage back to Guyton's
1970s circulatory model. Remarkable forward-only engineering; Pulse's one departure is making the
whole thing differentiable.

## License

[MIT](LICENSE) — © 2026 Gabriel Rovina.
