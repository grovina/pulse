# Pulse — a differentiable model of human physiology

> Training a model of the body against the medical literature.

Most whole-body physiology simulators (HumMod, BioGears, and the lineage going back to
Guyton's 1970s circulatory model) are **forward-only**: you give them a patient and they
predict what happens. They are impressive engineering, but they share one limitation — they
cannot *learn* from new evidence. Incorporating a finding means a human deciding which
parameters to hand-tune, and by how much, inside a coupled multi-system model.

**Pulse is differentiable end to end.** That single change flips the relationship between
a model and the literature: a published result — an RCT effect size, a cohort association,
a dose-response curve — can be written as a loss term, and the optimiser discovers which
internal dynamics have to shift to reproduce it. The model doesn't just *contain* medical
knowledge; it *optimises agreement* with it.

## The idea

The body is modelled as a network of coupled physiological subsystems. What we **impose**
and what we **learn** follow a deliberate hierarchy:

| Layer | Nature | How it enters the model |
|---|---|---|
| Physics / chemistry | Universal laws | Enforced in the architecture (mass-action kinetics, conservation of mass) |
| Anatomy / structure | How the body is built | The module boundaries and the coupling graph |
| Medical knowledge | What typically happens | Training signal and coupling-sign priors |
| Individual variation | What happens in *this* person | Learned from that person's own data |

So the *sign* of a coupling (insulin suppresses glucose) is structural; its *strength* in a
given person is learned. Chemical species use mass-action rate laws; vital signs use learned
dynamics constrained by the known couplings. Seven subsystems — metabolic, appetite, stress,
cardiovascular, thermoregulation, respiratory, gut — are wired along an anatomically faithful
(sparse) coupling graph rather than a fully-connected blob, so a single glucose reading
propagates through known pathways instead of having to discover them from scratch.

The thesis, in one line: **a trainable synthesis of physiology, updated by the literature itself.**

## Roadmap

The differentiable engine, the seven coupled subsystems, the knowledge-loss training loop,
and the benchmark/verifier harness are in place. Current work moves along three lines:

- **Benchmark.** Tighten predictive accuracy across markers and broaden coverage of the gate.
- **Grounding.** Move from synthetic reference trajectories toward measured data, and improve
  per-person calibration from sparse, real-world inputs.
- **Literature.** Widen the set of published findings the model trains against, and the
  machinery for turning a paper into a loss term.

The iteration history — design decisions, experiments, and what each round changed — lives in
[`docs/`](docs/), a running lab notebook.

## Layout

- [`pulse/`](pulse/) — the model and everything around it: `model.py`, the subsystem
  `modules/`, the encoded `knowledge/`, the differentiable knowledge losses (`*_loss.py`),
  the trainer (`train.py`), the `benchmark.py` and `verifier.py`, and a FastAPI inference
  server (`server.py`).
- [`docs/`](docs/) — the vision ([`docs/prd.md`](docs/prd.md)), design notes, and the full
  iteration history (a running lab notebook).
- [`tests/`](tests/), [`scripts/`](scripts/) — checks on the physiological signals and
  utilities.

## Running it

Uses [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run python -m pulse.train       # train (synthetic episodes + knowledge losses)
uv run python -m pulse.benchmark   # evaluate against the benchmark/gate
uv run uvicorn pulse.server:app    # serve inference
```

## Background

The longer argument — prior art, the philosophy of "many weak constraints", and why
differentiability is the crux — is in [`docs/prd.md`](docs/prd.md).

---

Built by [Gabriel Rovina](https://grovina.com). Released under the MIT License.
