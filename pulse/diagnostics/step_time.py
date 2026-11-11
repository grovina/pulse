"""CPU seconds per evidence step: items rolled one by one against the batched step.

Samples the same items, rules and people for both paths and times each in CPU seconds
(``time.process_time``, so a busy machine does not inflate it) on one torch thread.
``--max-arm-minutes`` keeps only the items and rules whose arms are all at most that
long, which is how a small machine times a step it could not hold uncapped (0 keeps
everything). The serial path is ``item_nll`` / ``rule_nll`` with a backward per item,
as the trainer did before the batched step; the batched path is
``accumulate_evidence_step``.
"""

from __future__ import annotations

import resource
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from ..knowledge.cohort_statistics import ALL_COHORT_STATISTICS
from ..knowledge.physiology_rules import PHYSIOLOGY_RULES, PhysiologyRule
from ..model import ModularPhysiologyNetwork
from ..training.evidence_step import (
    EvidenceTerm,
    accumulate_evidence_step,
    item_nll,
    rule_nll,
    sample_evidence,
)
from ..training.factors import FED_DAY_MINUTES
from ..training.safe_step import accumulate_grad
from ..training.signals import SignalContext


@dataclass
class StepTime:
    batch_size: int
    max_arm_minutes: int
    n_steps: int
    serial_s: float
    batched_s: float
    spinup_s: float
    rollout_s: float
    backward_s: float
    n_backwards: float
    n_rows: float
    peak_rss_mb: float
    pool_items: int
    pool_rules: int
    groups: list[list[tuple[int, int]]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _within(arms, cap: int) -> bool:
    return cap <= 0 or all(int(a.duration_min) <= cap for a in arms)


def _context(model: torch.nn.Module) -> SignalContext:
    params = list(model.parameters())
    return SignalContext(
        epoch=0, total_epochs=1, rng=np.random.default_rng(0), device=torch.device("cpu"),
        optimizer=torch.optim.SGD(params, lr=0.0), params=params, grad_clip=1.0,
    )


def step_time(
    model: ModularPhysiologyNetwork,
    *,
    batch_size: int = 4,
    n_draws: int = 4,
    rule_draws: int = 2,
    max_arm_minutes: int = 0,
    n_steps: int = 2,
    fed_day_minutes: int = FED_DAY_MINUTES,
    seed: int = 0,
) -> StepTime:
    """Mean CPU seconds per step, serial and batched, over ``n_steps`` sampled steps."""
    torch.set_num_threads(1)
    items = [s for s in ALL_COHORT_STATISTICS if _within(s.arms, max_arm_minutes)]
    rules = [r for r in PHYSIOLOGY_RULES if _within(r.arms, max_arm_minutes)]
    ctx = _context(model)
    device = torch.device("cpu")
    steps = []
    rng = np.random.default_rng(seed)
    for _ in range(n_steps):
        terms = [EvidenceTerm(s, float(s.weight), n_draws) for s in sample_evidence(items, batch_size, rng)]
        terms += [EvidenceTerm(r, float(r.weight), rule_draws) for r in sample_evidence(rules, 1, rng)]
        steps.append(terms)

    def serial(terms, rng_):
        for t in terms:
            fn = rule_nll if isinstance(t.unit, PhysiologyRule) else item_nll
            nll = fn(model, t.unit, n_draws=t.n_draws, rng=rng_, device=device,
                     fed_day_minutes=fed_day_minutes)
            accumulate_grad(t.weight * nll, ctx, signal="evidence")

    serial_s = batched_s = spinup = rollout = backward = backwards = rows = 0.0
    groups: list[list[tuple[int, int]]] = []
    for i, terms in enumerate(steps):
        t0 = time.process_time()
        serial(terms, np.random.default_rng(seed + 1 + i))
        serial_s += time.process_time() - t0
        model.zero_grad()
        t0 = time.process_time()
        report = accumulate_evidence_step(
            model, ctx, terms, rng=np.random.default_rng(seed + 1 + i), device=device,
            fed_day_minutes=fed_day_minutes,
        )
        batched_s += time.process_time() - t0
        model.zero_grad()
        spinup += report.spinup_s
        rollout += report.rollout_s
        backward += report.backward_s
        backwards += report.n_backwards
        rows += sum(r for r, _h in report.group_sizes)
        groups.append(report.group_sizes)
    n = float(max(n_steps, 1))
    return StepTime(
        batch_size, max_arm_minutes, n_steps, serial_s / n, batched_s / n, spinup / n,
        rollout / n, backward / n, backwards / n, rows / n,
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        len(items), len(rules), groups,
    )


def render(r: StepTime) -> str:
    cap = f"arms <= {r.max_arm_minutes} min" if r.max_arm_minutes > 0 else "uncapped arms"
    return "\n".join([
        f"step-time: {r.batch_size} items + 1 rule, {cap} "
        f"({r.pool_items} items / {r.pool_rules} rules eligible), mean of {r.n_steps} step(s), CPU seconds",
        f"  serial   (item by item)  {r.serial_s:8.1f} s",
        f"  batched                  {r.batched_s:8.1f} s   speed-up {r.serial_s / max(r.batched_s, 1e-9):.2f}x",
        f"    spin-up {r.spinup_s:.1f} s   rollout+score {r.rollout_s:.1f} s   backward {r.backward_s:.1f} s",
        f"    {r.n_backwards:.1f} rollout group(s), {r.n_rows:.0f} rows; groups of the last step "
        f"(rows, horizon): {r.groups[-1] if r.groups else []}",
        f"  peak RSS {r.peak_rss_mb:.0f} MB",
    ])


__all__ = ["StepTime", "render", "step_time"]
