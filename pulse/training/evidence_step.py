"""One training step: a batch of evidence, fresh people, summed likelihoods.

The step samples cohort items, draws a population for each arm, spins the
student up to the arm's start, and adds the Gaussian negative log-likelihood
of that item. The sum is not divided by the batch. Adding an item adds a
term.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from ..cohort_loss import (
    _rollout_arm_states,
    _score_arm_trajs,
    statistic_nll,
)
from ..knowledge.cohort_types import CohortStatisticSpec, InitMode, Population, StatisticKind
from ..knowledge.physiology_rules import PhysiologyRule
from ..physiology_rules_loss import rule_context_for_arm
from .arm_init import student_initial_states
from .embedding_sampler import draw_people
from .factors import FED_DAY_MINUTES
from ..types import NORM_CENTER, is_observable


def sample_evidence(pool: Sequence, batch_size: int, rng: np.random.Generator) -> list:
    """``batch_size`` items from ``pool``. A batch larger than the pool
    samples with replacement, so every slot still contributes a term."""
    if batch_size <= 0 or not pool:
        return []
    replace = batch_size > len(pool)
    chosen = rng.choice(len(pool), size=batch_size, replace=replace)
    return [pool[int(i)] for i in chosen]


def _sigma(spec: CohortStatisticSpec, n_people: int) -> float:
    if spec.kind in (
        StatisticKind.SPREAD, StatisticKind.CORRELATION, StatisticKind.RATIO_INCREMENTS,
    ):
        return float(spec.sigma)
    n = int(spec.n_arm) if spec.n_arm is not None else max(n_people, 1)
    return float(spec.sigma) / (float(n) ** 0.5)


def _codes_for_arms(
    spec: CohortStatisticSpec,
    n: int,
    rng: np.random.Generator,
    device: torch.device | str,
) -> list[torch.Tensor]:
    """One draw per arm. Arms that name the same population share the draw,
    so a two-arm difference is the same people under both protocols."""
    populations: list[Population | None] = [arm.population for arm in spec.arms]
    if all(p == populations[0] for p in populations):
        shared = draw_people(n, rng, device, populations[0])
        return [shared for _ in spec.arms]
    return [draw_people(n, rng, device, population) for population in populations]


def _initial_states(
    model: torch.nn.Module,
    spec: CohortStatisticSpec,
    codes: Sequence[torch.Tensor],
    *,
    fed_day_minutes: int,
) -> list[torch.Tensor]:
    if spec.init_mode is InitMode.NORM_CENTER:
        basal = torch.tensor(NORM_CENTER, dtype=torch.float32, device=codes[0].device)
        return [basal.unsqueeze(0).expand(int(c.shape[0]), -1).contiguous() for c in codes]
    return [
        student_initial_states(model, c, arm, fed_day_minutes=fed_day_minutes)
        for c, arm in zip(codes, spec.arms)
    ]


def item_nll(
    model: torch.nn.Module,
    spec: CohortStatisticSpec,
    *,
    n_draws: int,
    rng: np.random.Generator,
    device: torch.device | str,
    fed_day_minutes: int = FED_DAY_MINUTES,
) -> torch.Tensor:
    """Gaussian negative log-likelihood of one evidence item."""
    if n_draws <= 0:
        return torch.zeros((), device=device)
    codes = _codes_for_arms(spec, n_draws, rng, device)
    states = _initial_states(model, spec, codes, fed_day_minutes=fed_day_minutes)
    observe = is_observable(spec.marker_id) or (
        spec.partner is not None and is_observable(spec.partner)
    )
    trajs = [
        _rollout_arm_states(model, c, arm, state, observe=observe)
        for c, arm, state in zip(codes, spec.arms, states)
    ]
    z2, _pred, _z = _score_arm_trajs(spec, trajs, codes[0], n_draws)
    return statistic_nll(z2, _sigma(spec, n_draws))


def rule_nll(
    model: torch.nn.Module,
    rule: PhysiologyRule,
    *,
    n_draws: int,
    rng: np.random.Generator,
    device: torch.device | str,
    fed_day_minutes: int = FED_DAY_MINUTES,
) -> torch.Tensor:
    """Sum of squared hinges for one rule, over its arms and the people drawn.

    Not a mean. The same draw faces every arm, and a satisfied hinge adds
    nothing. An empty draw is zero.
    """
    if n_draws <= 0 or not rule.arms:
        return torch.zeros((), device=device)
    codes = draw_people(n_draws, rng, device, None)
    if rule.init_mode is InitMode.NORM_CENTER:
        basal = torch.tensor(NORM_CENTER, dtype=torch.float32, device=device)
        states = [basal.unsqueeze(0).expand(n_draws, -1).contiguous() for _ in rule.arms]
    else:
        states = [
            student_initial_states(model, codes, arm, fed_day_minutes=fed_day_minutes)
            for arm in rule.arms
        ]
    total = torch.zeros((), device=device)
    scale = float(rule.scale)
    for arm, state in zip(rule.arms, states):
        traj = _rollout_arm_states(model, codes, arm, state)
        ctx = rule_context_for_arm(arm)
        for b in range(n_draws):
            total = total + (rule.predicate(traj[b], ctx) / scale).pow(2)
    return total


def evidence_batch_nll(
    model: torch.nn.Module,
    specs: Sequence[CohortStatisticSpec],
    *,
    n_draws: int,
    rng: np.random.Generator,
    device: torch.device | str,
    fed_day_minutes: int = FED_DAY_MINUTES,
) -> torch.Tensor:
    """Sum of per-item likelihoods, each multiplied by the item's own weight.

    Not a mean. An empty batch is zero.
    """
    total = torch.zeros((), device=device)
    for spec in specs:
        total = total + float(spec.weight) * item_nll(
            model, spec, n_draws=n_draws, rng=rng, device=device,
            fed_day_minutes=fed_day_minutes,
        )
    return total


__all__ = [
    "evidence_batch_nll",
    "item_nll",
    "rule_nll",
    "sample_evidence",
]
