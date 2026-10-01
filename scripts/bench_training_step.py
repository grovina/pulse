"""Throughput of the student's rollout — the unit every training signal pays for.

Times ``integrate`` forward + backward per Euler step at a few batch widths, so a
change to the hot path (``model.prepare`` / ``model.step``, the modules' ``step``,
``euler_step``) can be checked for a regression before it costs a 30-hour run.
Single-threaded by default: the per-step work is ~600 tiny ops, so intra-op
threads only add synchronisation (see docs/training-efficiency.md).

    uv run python scripts/bench_training_step.py                 # eager
    uv run python scripts/bench_training_step.py --compile-step  # torch.compile'd step
    uv run python scripts/bench_training_step.py --batch 1 8 32 --steps 480
"""

from __future__ import annotations

import argparse
import time

import torch

from pulse.model import ModularPhysiologyNetwork, integrate, set_step_compilation
from pulse.modules.gut import MealEvent
from pulse.types import EMBEDDING_DIM, NORM_CENTER


def _time_rollout(model: ModularPhysiologyNetwork, batch: int, steps: int) -> tuple[float, float]:
    emb = torch.nn.Parameter(0.1 * torch.randn(batch, EMBEDDING_DIM))
    state = torch.tensor(NORM_CENTER, dtype=torch.float32).expand(batch, -1).clone()
    meals = [MealEvent(time=30.0, carbs=60.0, fats=20.0, proteins=25.0)]
    t0 = time.perf_counter()
    traj = integrate(
        model, state, emb, steps, start_time_minutes=480.0, meals=meals,
        sleep_wake=torch.ones(steps), activity=torch.zeros(steps),
    )
    t1 = time.perf_counter()
    traj.pow(2).mean().backward()
    t2 = time.perf_counter()
    return (t1 - t0) / steps, (t2 - t1) / steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--compile-step", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    h = args.hidden_dim
    model = ModularPhysiologyNetwork(
        metabolic_hidden=h, appetite_hidden=max(24, h // 2), stress_hidden=max(24, h // 2),
        cardiovascular_hidden=h, thermoreg_hidden=max(16, h // 3), respiratory_hidden=max(16, h // 3),
    )
    set_step_compilation(args.compile_step)
    print(f"torch {torch.__version__}  threads={torch.get_num_threads()}  "
          f"compile_step={args.compile_step}  steps={args.steps}")
    for batch in args.batch:
        warm = time.perf_counter()
        _time_rollout(model, batch, min(args.steps, 20))  # warm-up (and compile)
        warm = time.perf_counter() - warm
        runs = [_time_rollout(model, batch, args.steps) for _ in range(args.repeats)]
        fwd = min(r[0] for r in runs) * 1e3
        bwd = min(r[1] for r in runs) * 1e3
        print(f"  B={batch:3d}: forward {fwd:6.2f} ms/step  backward {bwd:6.2f} ms/step  "
              f"total {fwd + bwd:6.2f} ms/step  ({(fwd + bwd) / batch:6.3f} ms per row-step; "
              f"warm-up {warm:.1f}s)")


if __name__ == "__main__":
    main()
