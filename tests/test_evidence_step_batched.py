"""The batched evidence step: one spin-up, shared rollouts, the same loss and gradient.

``accumulate_evidence_step`` must give each term the NLL ``item_nll`` / ``rule_nll`` give it
for the same people, and the same accumulated gradient. The arms here are 30-50 min and
the fed day is 30 min, so every case is a short rollout.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pulse.knowledge.cohort_types import (
    CohortArmSpec,
    CohortStatisticSpec,
    InitMode,
    Population,
    StatisticKind,
    StatisticWindow,
    held,
)
from pulse.knowledge.physiology_rules import PhysiologyRule
from pulse.model import ModularPhysiologyNetwork
from pulse.training import SignalContext, arm_init, evidence_step
from pulse.training.arm_init import student_initial_states, student_initial_states_for_rows
from pulse.training.embedding_sampler import draw_people
from pulse.training.evidence_step import (
    EvidenceTerm,
    _Unit,
    _group_units,
    accumulate_evidence_step,
    item_nll,
    rule_nll,
)
from pulse.training.factors import FED_DAY_MEALS
from pulse.types import EMBEDDING_DIM

FED = 30
MEAL = ((10.0, 60.0, 20.0, 20.0),)


def _model() -> ModularPhysiologyNetwork:
    """Small widths with every zero-init output layer woken, so all heads carry gradient."""
    torch.manual_seed(3)
    model = ModularPhysiologyNetwork(
        metabolic_hidden=12, appetite_hidden=8, stress_hidden=8,
        cardiovascular_hidden=12, thermoreg_hidden=8, respiratory_hidden=8,
        gut_hidden=8, hepatobiliary_hidden=8,
    )
    g = torch.Generator().manual_seed(4)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    return model


def _spec(name, arms, kind=StatisticKind.MEAN_IN_WINDOW, marker="glucose", target=100.0, **kw):
    window = StatisticWindow(10, arms[0].duration_min)
    return CohortStatisticSpec(
        name=name, source="t", description="", arms=tuple(arms), marker_id=marker,
        kind=kind, window=window, target=target, sigma=5.0, **kw,
    )


def _arm(label, minutes, **kw):
    return CohortArmSpec(label=label, duration_min=minutes, start_hour=kw.pop("start_hour", 8.0),
                         meals=kw.pop("meals", MEAL), **kw)


def _rule() -> PhysiologyRule:
    return PhysiologyRule(
        name="peak", source="t", description="",
        arms=(_arm("r40", 40), _arm("r30", 30, start_hour=20.0)),
        predicate=lambda traj, ctx: torch.relu(traj[:, ctx.marker_index["glucose"]].max() - 60.0),
        scale=10.0,
    )


def _terms() -> list[EvidenceTerm]:
    one = _spec("one", [_arm("a", 40)])
    diff = _spec("diff", [_arm("fed", 40), _arm("fast", 40, meals=(), start_hour=9.0)],
                 kind=StatisticKind.DELTA_MEANS, target=5.0)
    pops = _spec("pops", [_arm("f", 30, population=Population(name="f", sex="female")),
                          _arm("m", 30, population=Population(name="m", sex="male"))],
                 kind=StatisticKind.DELTA_MEANS, target=0.0)
    infusion = _spec("infusion", [_arm("iv", 35, meals=(), glucose_infusion=held(2.0, 35))])
    clamp = _spec("clamp", [_arm("clamp", 50, meals=(), glucose_clamp=held(110.0, 50))],
                  marker="glucose_infusion_rate", target=3.0)
    centred = _spec("centred", [_arm("c", 30)], init_mode=InitMode.NORM_CENTER, target=95.0)
    prefast = _spec("prefast", [_arm("p", 30, meals=(), prefast_hours=1.0)], target=90.0)
    return [
        EvidenceTerm(one, 0.5, 2),
        EvidenceTerm(diff, 2.0, 2),
        EvidenceTerm(pops, 1.0, 2),
        EvidenceTerm(infusion, 1.5, 2),
        EvidenceTerm(clamp, 1.0, 2),
        EvidenceTerm(centred, 0.7, 2),
        EvidenceTerm(prefast, 1.0, 2),
        EvidenceTerm(_rule(), 0.3, 2),
    ]


def _ctx(model, clip: float = 0.0) -> SignalContext:
    params = list(model.parameters())
    return SignalContext(
        epoch=0, total_epochs=1, rng=np.random.default_rng(0), device=torch.device("cpu"),
        optimizer=torch.optim.SGD(params, lr=1e-3), params=params, grad_clip=10.0,
        aux_signal_clip=clip,
    )


def _flat_grad(model) -> torch.Tensor:
    return torch.cat([
        (p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
        for p in model.parameters()
    ])


def _serial(model, terms, seed):
    rng = np.random.default_rng(seed)
    model.zero_grad()
    nlls, total = [], torch.zeros(())
    for t in terms:
        fn = rule_nll if isinstance(t.unit, PhysiologyRule) else item_nll
        nll = fn(model, t.unit, n_draws=t.n_draws, rng=rng, device="cpu", fed_day_minutes=FED)
        nlls.append(float(nll.detach()))
        total = total + t.weight * nll
    total.backward()
    return nlls, _flat_grad(model).clone()


def _batched(model, terms, seed, clip=0.0, **kw):
    model.zero_grad()
    report = accumulate_evidence_step(
        model, _ctx(model, clip), terms, rng=np.random.default_rng(seed), device="cpu",
        fed_day_minutes=FED, **kw,
    )
    return report, _flat_grad(model).clone()


_RUNS: dict = {}


def _shared_run():
    """The serial reference and the batched step for the same people, computed once."""
    if not _RUNS:
        model = _model()
        terms = _terms()
        _RUNS["serial"] = _serial(model, terms, seed=11)
        _RUNS["batched"] = _batched(model, terms, seed=11)
    return _RUNS["serial"], _RUNS["batched"]


def test_batched_step_matches_each_term_alone() -> None:
    (want_nll, want_grad), (report, got_grad) = _shared_run()

    assert report.nll == pytest.approx(want_nll, rel=1e-4)
    cosine = torch.dot(got_grad, want_grad) / (got_grad.norm() * want_grad.norm())
    assert float(cosine) > 0.9999
    assert float(got_grad.norm() / want_grad.norm()) == pytest.approx(1.0, abs=1e-3)
    assert all(v > 0.0 for v in want_nll)


def test_mixed_horizons_share_a_group_and_observing_terms_do_not() -> None:
    _, (report, _) = _shared_run()
    # The observed 50-min clamp is alone; the other seven terms (30-40 min) roll together.
    assert dict((h, r) for r, h in report.group_sizes) == {50: 2, 40: 20}
    assert report.n_backwards == 2


def test_a_group_never_pads_beyond_the_ratio_or_the_row_budget() -> None:
    def unit(i, horizon, rows=4, observe=False):
        arm = _arm("a", horizon)
        return _Unit(i, EvidenceTerm(_spec("s", [arm]), 1.0, rows), (arm,),
                     [torch.zeros(rows, EMBEDDING_DIM)], observe, horizon)

    units = [unit(0, 1440), unit(1, 900), unit(2, 100), unit(3, 60), unit(4, 700, observe=True)]
    groups = _group_units(units, pad_ratio=2.0, max_row_minutes=10**9)
    assert [[u.index for u in g] for g in groups] == [[0, 1], [2, 3], [4]]
    narrow = _group_units(units[:2], pad_ratio=2.0, max_row_minutes=1440 * 7)
    assert [[u.index for u in g] for g in narrow] == [[0], [1]]


def test_one_spinup_per_length_and_one_backward_per_group(monkeypatch) -> None:
    model = _model()
    spin, backward = [], []
    real_integrate, real_accumulate = arm_init.integrate, evidence_step.accumulate_grad

    def spy_integrate(model, state, emb, n_steps, **kw):
        spin.append((n_steps, torch.is_grad_enabled(), int(state.shape[0])))
        return real_integrate(model, state, emb, n_steps, **kw)

    def spy_accumulate(loss, ctx, **kw):
        backward.append(loss.grad_fn is not None)
        return real_accumulate(loss, ctx, **kw)

    monkeypatch.setattr(arm_init, "integrate", spy_integrate)
    monkeypatch.setattr(evidence_step, "accumulate_grad", spy_accumulate)
    report, _ = _batched(model, _terms(), seed=2)

    # Fed day 30 min for most arms, 30 + 60 for the pre-fasted one; the NORM_CENTER item
    # rolls none. Every row of a length goes in one call, without a graph.
    assert sorted(spin) == [(30, False, 18), (90, False, 2)]
    assert len(backward) == report.n_backwards == len(report.group_sizes) and all(backward)


def test_clip_is_per_group_and_counted() -> None:
    model = _model()
    _, (free, free_grad) = _shared_run()
    assert free.n_clipped == 0
    tight = 1e-3
    report, grad = _batched(model, _terms(), seed=11, clip=tight)
    assert report.n_clipped == report.n_backwards == 2
    # Two groups, each rescaled to the clip: the sum cannot exceed twice the clip.
    assert float(grad.norm()) <= 2 * tight * (1 + 1e-4)
    assert float(grad.norm()) < float(free_grad.norm())


def test_a_term_with_no_draws_is_zero_and_an_empty_step_is_a_no_op() -> None:
    model = _model()
    spec = _spec("one", [_arm("a", 40)])
    report, grad = _batched(model, [EvidenceTerm(spec, 1.0, 0)], seed=0)
    assert report.nll == [0.0] and report.n_backwards == 0 and float(grad.abs().sum()) == 0.0


def test_spinup_rows_match_arm_by_arm() -> None:
    model = _model()
    rng = np.random.default_rng(0)
    arms = [_arm("a", 60, start_hour=7.0), _arm("b", 60, start_hour=22.5, prefast_hours=1.0),
            _arm("c", 60, start_hour=13.0)]
    rows = [(draw_people(n, rng, "cpu", None), arm) for n, arm in zip((2, 3, 1), arms)]
    together = student_initial_states_for_rows(model, rows, fed_day_minutes=FED)
    for (codes, arm), got in zip(rows, together):
        want = student_initial_states(model, codes, arm, fed_day_minutes=FED)
        assert got.shape == want.shape and got.grad_fn is None
        assert torch.allclose(got, want, rtol=1e-4, atol=1e-5)
    # The fed day moved the state, and a different clock moved it differently.
    assert not torch.allclose(together[0][0], together[2][0])


def test_the_spinup_sleeps_from_23_to_07_through_the_prefast() -> None:
    arm = _arm("a", 60, start_hour=8.0, prefast_hours=10.0)
    total, start_min, _meals, sleep = arm_init._spinup_protocol(arm, 1440, torch.device("cpu"))
    assert total == 1440 + 600 and start_min == pytest.approx(22.0 * 60.0)
    want = []
    for minute in range(total):
        clock = (22.0 + minute / 60.0) % 24.0
        want.append(0.0 if (clock >= 23.0 or clock < 7.0) else 1.0)
    assert sleep.tolist() == want
    # The night before the arm is slept: 23:00 to 07:00 lie inside the pre-fast.
    assert float(sleep[1440 + 60:1440 + 540].sum()) == 0.0 and float(sleep[-60:].min()) == 1.0


def _spinup_meals(arm, fed: int = 1440) -> list[tuple[float, float, float]]:
    """``(clock hour, spin-up minute, carbohydrate)`` of each meal, in order."""
    _total, start_min, meals, _sleep = arm_init._spinup_protocol(arm, fed, torch.device("cpu"))
    return [(round(((start_min + m.time) / 60.0) % 24.0, 3), m.time, m.carbs) for m in meals]


def test_a_prefast_begins_with_the_meal_nearest_its_start() -> None:
    breakfast, _lunch, dinner = FED_DAY_MEALS
    # Ten hours before 08:00 is 22:00: the 19:00 dinner is eaten then instead.
    got = _spinup_meals(_arm("a", 60, start_hour=8.0, prefast_hours=10.0))
    assert [clock for clock, _t, _c in got] == [8.0, 13.0, 22.0]
    assert got[-1][1:] == (1440.0, dinner[1])
    # Three hours before 08:00 is 05:00: the breakfast still to come is eaten early,
    # and the day before it keeps its three meals.
    got = _spinup_meals(_arm("a", 60, start_hour=8.0, prefast_hours=3.0))
    assert [clock for clock, _t, _c in got] == [8.0, 13.0, 19.0, 5.0]
    assert got[-1][1:] == (1440.0, breakfast[1])
    # Thirteen hours before 08:00 is 19:00, the dinner's own hour: yesterday's stays.
    got = _spinup_meals(_arm("a", 60, start_hour=8.0, prefast_hours=13.0))
    assert [(clock, t) for clock, t, _c in got] == [
        (19.0, 0.0), (8.0, 780.0), (13.0, 1080.0), (19.0, 1440.0),
    ]
    # Whatever the clock, the last meal is prefast_hours before the arm.
    for start_hour, pre in ((8.0, 10.0), (23.0, 1.5), (8.0, 72.0), (15.0, 8.0), (12.0, 4.0)):
        arm = _arm("a", 60, start_hour=start_hour, prefast_hours=pre)
        total, _start, meals, _sleep = arm_init._spinup_protocol(arm, 1440, torch.device("cpu"))
        assert total - max(m.time for m in meals) == pytest.approx(pre * 60.0)


def test_an_arm_with_no_prefast_starts_from_the_day_on_its_clock() -> None:
    got = _spinup_meals(_arm("a", 60, start_hour=8.0))
    assert sorted(clock for clock, _t, _c in got) == sorted(h for h, *_ in FED_DAY_MEALS)
    assert max(t for _clock, t, _c in got) < 1440
    # A fed day of no length has no meal in it, pre-fast or not.
    assert _spinup_meals(_arm("a", 60, start_hour=8.0, prefast_hours=10.0), fed=0) == []
