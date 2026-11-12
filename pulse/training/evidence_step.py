"""One training step: a batch of evidence, fresh people, summed likelihoods.

A step samples cohort items and rules, draws a population for each arm, spins the
student up to the arm's start, and adds the Gaussian negative log-likelihood of each
item (the squared hinge of each rule). The sum is not divided by the batch. Adding an
item adds a term.

``accumulate_evidence_step`` is the trainer's path. It rolls every row of the step
together: ONE no-grad spin-up per distinct spin-up length, then one rollout with
gradient per group of items whose horizons are close, each group backpropagated alone.
``item_nll`` and ``rule_nll`` are the same computation for one item or rule, rolled
arm by arm; the batched path is tested against them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

from ..cohort_loss import (
    _rollout_arm_states,
    _score_arm_trajs,
    rollout_arms,
    statistic_nll,
)
from ..knowledge.cohort_types import CohortArmSpec, CohortStatisticSpec, InitMode, Population, StatisticKind
from ..knowledge.physiology_rules import PhysiologyRule
from ..physiology_rules_loss import rule_context_for_arm
from .arm_init import student_initial_states, student_initial_states_for_rows
from .embedding_sampler import draw_people
from .factors import FED_DAY_MINUTES
from .safe_step import accumulate_grad
from .signals import SignalContext
from ..types import NORM_CENTER, is_observable


# How a step's items are drawn from the pool, as weights over papers.
#   item:  every item equally, so a paper weighs as many items as it printed.
#   sqrt:  an item of a paper with k items is drawn 1/sqrt(k) as often, so the
#          paper weighs sqrt(k): a paper's points are mostly one cohort, one
#          protocol and one assay, closer to one finding than to k independent ones.
#   paper: every paper equally, whatever it printed.
PAPER_WEIGHTINGS = ("item", "sqrt", "paper")


def evidence_probabilities(pool: Sequence, weighting: str = "item") -> np.ndarray | None:
    """Draw probabilities over ``pool`` for ``weighting`` (None: uniform).

    An item's paper is its ``paper`` slug; an item without one counts as a paper of
    its own.
    """
    if weighting not in PAPER_WEIGHTINGS:
        raise ValueError(f"unknown evidence weighting {weighting!r}; expected one of {PAPER_WEIGHTINGS}")
    if weighting == "item" or not pool:
        return None
    papers = [getattr(item, "paper", "") or f"#{i}" for i, item in enumerate(pool)]
    counts: dict[str, int] = {}
    for p in papers:
        counts[p] = counts.get(p, 0) + 1
    power = 0.5 if weighting == "sqrt" else 1.0
    w = np.array([counts[p] ** -power for p in papers], dtype=np.float64)
    return w / w.sum()


def sample_evidence(
    pool: Sequence,
    batch_size: int,
    rng: np.random.Generator,
    probabilities: np.ndarray | None = None,
) -> list:
    """``batch_size`` items from ``pool``, drawn by ``probabilities`` (uniform when
    None; see ``evidence_probabilities``). A batch larger than the pool samples with
    replacement, so every slot still contributes a term."""
    if batch_size <= 0 or not pool:
        return []
    replace = batch_size > len(pool)
    if probabilities is not None and not replace:
        replace = batch_size > int(np.count_nonzero(probabilities))
    chosen = rng.choice(len(pool), size=batch_size, replace=replace, p=probabilities)
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


def _observes(spec: CohortStatisticSpec) -> bool:
    return is_observable(spec.marker_id) or (
        spec.partner is not None and is_observable(spec.partner)
    )


def _item_nll_of(
    spec: CohortStatisticSpec,
    trajs: Sequence[torch.Tensor],
    codes: torch.Tensor,
    n_draws: int,
) -> torch.Tensor:
    z2, _pred, _z = _score_arm_trajs(spec, trajs, codes, n_draws)
    return statistic_nll(z2, _sigma(spec, n_draws))


def _rule_hinge_of(
    rule: PhysiologyRule, trajs: Sequence[torch.Tensor], n_draws: int, device: torch.device | str,
) -> torch.Tensor:
    total = torch.zeros((), device=device)
    scale = float(rule.scale)
    for arm, traj in zip(rule.arms, trajs):
        ctx = rule_context_for_arm(arm)
        for b in range(n_draws):
            total = total + (rule.predicate(traj[b], ctx) / scale).pow(2)
    return total


def item_nll(
    model: torch.nn.Module,
    spec: CohortStatisticSpec,
    *,
    n_draws: int,
    rng: np.random.Generator,
    device: torch.device | str,
    fed_day_minutes: int = FED_DAY_MINUTES,
) -> torch.Tensor:
    """Gaussian negative log-likelihood of one evidence item, rolled arm by arm."""
    if n_draws <= 0:
        return torch.zeros((), device=device)
    codes = _codes_for_arms(spec, n_draws, rng, device)
    states = _initial_states(model, spec, codes, fed_day_minutes=fed_day_minutes)
    observe = _observes(spec)
    trajs = [
        _rollout_arm_states(model, c, arm, state, observe=observe)
        for c, arm, state in zip(codes, spec.arms, states)
    ]
    return _item_nll_of(spec, trajs, codes[0], n_draws)


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
    trajs = [_rollout_arm_states(model, codes, arm, state) for arm, state in zip(rule.arms, states)]
    return _rule_hinge_of(rule, trajs, n_draws, device)


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


# --- the batched step ----------------------------------------------------------------

# Rollout groups. A rollout costs about the same at batch 1 and at batch 100, so
# items are merged until padding or memory says stop. Padding: an item whose horizon
# is under 1/PAD_RATIO of its group's is rolled alone, so no row is held for more than
# PAD_RATIO times its own length. Memory: a graph costs about 1.2 MB per simulated
# minute plus 16 KB per row-minute (measured, 4 to 256 rows, 240 min), so a group stops
# growing at MAX_ROW_MINUTES (about 80 day-long rows, +1.3 GB over the fixed part).
PAD_RATIO = 2.0
MAX_ROW_MINUTES = 120_000


@dataclass(frozen=True)
class EvidenceTerm:
    """One cohort item or rule in a step: ``weight`` multiplies its NLL in the loss
    and ``n_draws`` people face each of its arms."""

    unit: CohortStatisticSpec | PhysiologyRule
    weight: float
    n_draws: int


@dataclass
class StepReport:
    """What a step did. ``nll`` is each term's unweighted NLL, in the order given;
    a term with no draws is 0."""

    nll: list[float]
    n_backwards: int = 0
    n_clipped: int = 0
    spinup_s: float = 0.0
    rollout_s: float = 0.0
    backward_s: float = 0.0
    group_sizes: list[tuple[int, int]] = field(default_factory=list)  # (rows, horizon)


@dataclass
class _Unit:
    index: int
    term: EvidenceTerm
    arms: tuple[CohortArmSpec, ...]
    codes: list[torch.Tensor]
    observe: bool
    horizon: int
    states: list[torch.Tensor] | None = None

    @property
    def rows(self) -> int:
        return sum(int(c.shape[0]) for c in self.codes)


def _unit_of(
    index: int, term: EvidenceTerm, rng: np.random.Generator, device: torch.device | str,
) -> _Unit | None:
    unit = term.unit
    if term.n_draws <= 0 or not unit.arms:
        return None
    if isinstance(unit, PhysiologyRule):
        shared = draw_people(term.n_draws, rng, device, None)
        codes, observe = [shared for _ in unit.arms], False
    else:
        codes, observe = _codes_for_arms(unit, term.n_draws, rng, device), _observes(unit)
    return _Unit(
        index, term, tuple(unit.arms), codes, observe,
        max(int(arm.duration_min) for arm in unit.arms),
    )


def _group_units(
    units: Sequence[_Unit], *, pad_ratio: float = PAD_RATIO, max_row_minutes: int = MAX_ROW_MINUTES,
) -> list[list[_Unit]]:
    """Units grouped for rollout: non-observing before observing, longest first within
    each. A unit joins the open group while it is at least ``1/pad_ratio`` as long as
    the group's longest and keeps the group's rows x horizon within ``max_row_minutes``;
    otherwise it opens a new group."""
    groups: list[list[_Unit]] = []
    rows = 0
    for unit in sorted(units, key=lambda u: (u.observe, -u.horizon)):
        open_group = groups[-1] if groups else None
        if (
            open_group is not None
            and open_group[0].observe == unit.observe
            and unit.horizon * pad_ratio >= open_group[0].horizon
            and (rows + unit.rows) * open_group[0].horizon <= max_row_minutes
        ):
            open_group.append(unit)
            rows += unit.rows
        else:
            groups.append([unit])
            rows = unit.rows
    return groups


def accumulate_evidence_step(
    model: torch.nn.Module,
    ctx: SignalContext,
    terms: Sequence[EvidenceTerm],
    *,
    rng: np.random.Generator,
    device: torch.device | str,
    fed_day_minutes: int = FED_DAY_MINUTES,
    pad_ratio: float = PAD_RATIO,
    max_row_minutes: int = MAX_ROW_MINUTES,
) -> StepReport:
    """Backward the weighted sum of ``terms`` into the shared gradient buffers.

    People are drawn term by term in the order given (``item_nll`` / ``rule_nll`` draw
    the same way), so the same rng gives the same people. Every row's spin-up is one
    no-grad batched rollout per spin-up length. Terms are then rolled in groups
    (``_group_units``), each group as ONE ``rollout_arms`` call with the arms of all its
    terms as rows, scored term by term, and backpropagated before the next group is
    rolled, so only one graph is alive at a time. The step's loss is the same weighted
    sum of per-term NLLs as ``weight * item_nll`` summed over the terms.

    Gradient handling is ``accumulate_grad``'s, applied once per GROUP: the group's
    gradient is clipped to ``ctx.aux_signal_clip`` as a whole, and a non-finite loss or
    gradient drops the whole group's contribution. Clipping each term alone would need
    a backward per term. ``StepReport.n_clipped`` counts the groups whose clip bound.
    """
    report = StepReport(nll=[0.0] * len(terms))
    units = [u for i, t in enumerate(terms) if (u := _unit_of(i, t, rng, device)) is not None]
    if not units:
        return report

    t0 = time.process_time()
    needs_spinup = [u for u in units if u.term.unit.init_mode is not InitMode.NORM_CENTER]
    basal = torch.tensor(NORM_CENTER, dtype=torch.float32, device=device)
    for u in units:
        if u.term.unit.init_mode is InitMode.NORM_CENTER:
            u.states = [basal.unsqueeze(0).expand(int(c.shape[0]), -1).contiguous() for c in u.codes]
    if needs_spinup:
        flat = [(c, arm) for u in needs_spinup for c, arm in zip(u.codes, u.arms)]
        spun = iter(student_initial_states_for_rows(model, flat, fed_day_minutes=fed_day_minutes))
        for u in needs_spinup:
            u.states = [next(spun) for _ in u.arms]
    report.spinup_s = time.process_time() - t0

    for group in _group_units(units, pad_ratio=pad_ratio, max_row_minutes=max_row_minutes):
        t0 = time.process_time()
        rows = [
            (arm, c, state)
            for u in group for arm, c, state in zip(u.arms, u.codes, u.states)
        ]
        trajs = iter(rollout_arms(model, rows, observe=group[0].observe))
        loss = torch.zeros((), device=device)
        for u in group:
            u_trajs = [next(trajs) for _ in u.arms]
            n = u.term.n_draws
            if isinstance(u.term.unit, PhysiologyRule):
                nll = _rule_hinge_of(u.term.unit, u_trajs, n, device)
            else:
                nll = _item_nll_of(u.term.unit, u_trajs, u.codes[0], n)
            report.nll[u.index] = float(nll.detach())
            loss = loss + float(u.term.weight) * nll
        report.rollout_s += time.process_time() - t0

        t0 = time.process_time()
        seen = len(ctx.aux_grad_norms.get("evidence", ()))
        accumulate_grad(loss, ctx, signal="evidence")
        norms = ctx.aux_grad_norms.get("evidence", ())
        if len(norms) > seen and ctx.aux_signal_clip > 0.0 and norms[-1] > ctx.aux_signal_clip:
            report.n_clipped += 1
        report.n_backwards += 1
        report.group_sizes.append((sum(u.rows for u in group), group[0].horizon))
        report.backward_s += time.process_time() - t0
        del trajs, loss
    return report


__all__ = [
    "EvidenceTerm",
    "StepReport",
    "accumulate_evidence_step",
    "PAPER_WEIGHTINGS",
    "evidence_batch_nll",
    "evidence_probabilities",
    "item_nll",
    "rule_nll",
    "sample_evidence",
]
