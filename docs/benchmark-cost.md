# Benchmark cost: where the time goes (2026-06-12)

Quick anatomy of the gate's runtime, and the cheap levers to cut it. Measured
against `pulse-model.pt.last-good.pt` + `benchmark.dataset.generated.json`.

## Cost structure

- **24 episodes**, 720 min each, up to **8 workers** (`PULSE_BENCHMARK_PARALLEL`),
  so ~3 episodes per worker, serial within a worker.
- **Per episode the dominant cost is calibration:** `BENCHMARK_GATE_CALIBRATE_STEPS
  = 512` Adam steps, each integrating *this episode's* calibration windows
  forward **and** backward. Episode 1 has 80 calibration obs grouped into **3
  windows × 240 min** → 512 × 3 × 240 ≈ 370k single-minute ODE forwards **with
  autograd**, per episode. That graph is large enough to OOM a small box.
- After calibration: one no-grad 720-min integration + verifier scoring — cheap
  by comparison.
- **The 64×64 Hessian / Bayesian path (`bayesian_calibrate`) is NOT in the gate.**
  It's `O(64)` backward passes and would dominate — but it's only used by
  `scripts/bayesian_demo.py` and `scripts/check_hr_drift.py`, not the gate. Good.

So ~all the gate time is `24 × 512` windowed calibration backprops / 8 workers.
One knob (`512`) sets it.

## Levers (cheap, low risk)

1. **Attach the embedding prior to the checkpoint.** This checkpoint reports
   `prior=isoL2` — no `_embedding_prior_mean` — so calibration starts from
   **zeros** (step-0 loss ≈ 2891) and must travel the whole way to the patient
   code. `train.py` already computes the trained-table per-dim mean/std for the
   iter-61 diag-Gauss path; the last-good checkpoint just doesn't carry it.
   Initialising at the population mean → fewer steps to converge **and** better
   calibration. Free win.
2. **Cut `PULSE_BENCHMARK_CALIBRATE_STEPS`.** It's already an env override (no
   code change). 512 Adam steps to fit a 64-dim embedding is generous; with a
   warm init it's typically 150–250. Sweep 128/192/256 on the real bench host
   (E2_HIGHCPU_8, where this is cheap), confirm `overall_weighted_mape` is
   unchanged, adopt the smallest. Plausibly 2–3× faster gate.
3. **Early-stop calibration on plateau** (relative loss-change < tol) instead of a
   fixed step count — adaptive, self-tuning across episodes with different obs
   density.

The convergence sweep that picks the exact step count is cheap on the bench's
own hardware; it crawls/OOMs in a small sandbox, so run it there.
