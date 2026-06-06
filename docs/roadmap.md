# Roadmap

Pulse is an active research project. The differentiable engine, the seven coupled subsystems, the
knowledge-loss training loop, and the benchmark/verifier harness are in place and run end to end.

## Where it stands

- The predictive benchmark is not yet closed across all markers.
- Some reference trajectories are synthetic: for markers no one self-measures (glucagon, FFA,
  ghrelin, …), targets come from a hand-built reference ODE. There is no real patient data yet.

## Where it's headed

- **Benchmark** — tighten predictive accuracy across markers and broaden coverage of the gate.
- **Grounding** — move from synthetic reference trajectories toward measured data, and improve
  per-person calibration from sparse, real-world inputs.
- **Literature** — widen the set of published findings the model trains against, and the machinery
  for turning a paper into a loss term.

The round-by-round detail — hypotheses, experiments, and what each iteration changed — lives in the
iteration handoffs in this folder.
