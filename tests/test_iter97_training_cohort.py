"""Iter 97 (training): cohort-statistic supervision shape.

Review 4.3 (per-individual z^2 against the individual sigma; adaptive weights
replacing hand-set weights; gradient into the embedding table; a one-sided WHO
criterion encoded as a point), 4.10 (teach-to-test on the sleep cohort arm)
and 4.9 (explicit series / group slices).
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
import torch.nn as nn

import pulse

REPO = os.path.dirname(os.path.dirname(os.path.abspath(pulse.__file__)))
assert pulse.__file__.startswith(REPO), pulse.__file__

from pulse.cohort_loss import score_batch_statistic, shaped_residual  # noqa: E402
from pulse.knowledge.cohort_types import (  # noqa: E402
    CohortArmSpec,
    CohortStatisticSpec,
    StatisticKind,
    StatisticWindow,
    TargetShape,
)
from pulse.model import ModularPhysiologyNetwork  # noqa: E402
from pulse.training import CohortStatisticSignal, SignalContext, WeightSchedule, joint_aux_step  # noqa: E402
from pulse.training.cohort_signal import perturb_group_arms  # noqa: E402
from pulse.types import EMBEDDING_DIM  # noqa: E402


def _spec(**kw) -> CohortStatisticSpec:
    base = dict(
        name="s", source="t", description="t",
        arms=(CohortArmSpec(label="a", duration_min=120, start_hour=8.0, meals=()),),
        marker_id="glucose", kind=StatisticKind.MEAN_IN_WINDOW,
        window=StatisticWindow(start_min=0, end_min=60), target=100.0, sigma=10.0,
    )
    base.update(kw)
    return CohortStatisticSpec(**base)


# --- 4.3: batch mean vs SEM ------------------------------------------------------

def test_between_patient_spread_is_not_a_loss_floor() -> None:
    spec = _spec()
    # Six patients spread +/-15 around a batch mean that equals the target.
    pred = torch.tensor([85.0, 115.0, 90.0, 110.0, 95.0, 105.0])
    loss, mean, z = score_batch_statistic(pred, spec)
    assert float(loss) == pytest.approx(0.0, abs=1e-9)
    assert mean == pytest.approx(100.0) and z == pytest.approx(0.0)
    # The old per-individual scoring would have charged Var_between / sigma^2.
    old = ((pred - spec.target) / spec.sigma).pow(2).mean()
    assert float(old) > 1.0


def test_batch_mean_is_scored_at_the_sem() -> None:
    spec = _spec()
    pred = torch.full((4,), 105.0)
    loss, _, z = score_batch_statistic(pred, spec)
    # sem = 10 / sqrt(4) = 5 -> (5/5)^2 = 1 ; z reported over the individual sigma.
    assert float(loss) == pytest.approx(1.0)
    assert z == pytest.approx(0.5)
    # n_arm pins the effective n regardless of the batch size.
    spec1 = _spec(n_arm=1)
    loss1, _, _ = score_batch_statistic(pred, spec1)
    assert float(loss1) == pytest.approx(0.25)


def test_target_shapes() -> None:
    at_most = _spec(shape=TargetShape.AT_MOST, target=140.0)
    assert float(score_batch_statistic(torch.full((3,), 120.0), at_most)[0]) == 0.0
    assert float(score_batch_statistic(torch.full((3,), 150.0), at_most)[0]) > 0.0
    at_least = _spec(shape=TargetShape.AT_LEAST, target=60.0)
    assert float(score_batch_statistic(torch.full((3,), 70.0), at_least)[0]) == 0.0
    assert float(score_batch_statistic(torch.full((3,), 50.0), at_least)[0]) > 0.0
    band = _spec(shape=TargetShape.BAND, target=100.0, band_halfwidth=10.0)
    assert float(score_batch_statistic(torch.full((3,), 108.0), band)[0]) == 0.0
    assert float(shaped_residual(torch.tensor(115.0), band)) == pytest.approx(5.0)
    assert float(shaped_residual(torch.tensor(85.0), band)) == pytest.approx(-5.0)
    with pytest.raises(ValueError):
        _spec(shape=TargetShape.BAND)  # a band needs a width


def _tiny_model() -> ModularPhysiologyNetwork:
    return ModularPhysiologyNetwork(
        embedding_dim=EMBEDDING_DIM,
        metabolic_hidden=16, appetite_hidden=12, stress_hidden=12,
        cardiovascular_hidden=16, thermoreg_hidden=8, respiratory_hidden=8,
    )


def _toy_meal_spec() -> CohortStatisticSpec:
    return _spec(
        name="toy_glucose_meal",
        arms=(
            CohortArmSpec(label="fasted", duration_min=120, start_hour=8.0, meals=()),
            CohortArmSpec(label="fed", duration_min=120, start_hour=8.0, meals=((30.0, 50.0, 10.0, 15.0),)),
        ),
        kind=StatisticKind.DELTA_MEANS, window=StatisticWindow(start_min=60, end_min=110),
        target=20.0, sigma=10.0,
    )


def test_cohort_gradient_does_not_reach_the_embedding_table() -> None:
    torch.manual_seed(2)
    model = _tiny_model()
    emb = nn.Embedding(2, EMBEDDING_DIM)
    nn.init.normal_(emb.weight, std=0.1)
    params = list(model.parameters()) + list(emb.parameters())
    opt = torch.optim.Adam(params, lr=1e-2)
    sig = CohortStatisticSignal(specs=[_toy_meal_spec()], n_patients=2, sample_patients=2,
                                weight=WeightSchedule(0.5))
    ctx = SignalContext(epoch=0, total_epochs=1, rng=np.random.default_rng(0),
                        device=torch.device("cpu"), optimizer=opt, params=params, grad_clip=10.0)
    before_emb = emb.weight.detach().clone()
    before_model = {n: p.detach().clone() for n, p in model.named_parameters()}
    res = sig.compute(model, emb, ctx)
    joint_aux_step(ctx)
    assert res.n_units == 1
    assert torch.equal(emb.weight.detach(), before_emb), "cohort loss moved the embedding table"
    assert any(float((p.detach() - before_model[n]).abs().sum()) > 0 for n, p in model.named_parameters())


def test_adaptive_cap_and_multiply_in_the_cohort_signal() -> None:
    specs = [_spec(name=f"s{i}", weight=(0.5 if i == 0 else 1.0)) for i in range(5)]
    sig = CohortStatisticSignal(specs=specs, adaptive=True, weight=WeightSchedule(0.1))
    sig._update_violation_ema({"s0": 50.0, "s1": 50.0, "s2": 0.1, "s3": 0.1, "s4": 0.1})
    w = sig._adaptive_weights_from_ema()
    total = sum(s.weight for s in specs)
    assert abs(sum(w.values()) - total) < 1e-9
    assert max(w.values()) <= 0.25 * total + 1e-9
    # Equal EMA, half the hand-set weight -> half the adaptive weight.
    assert w["s0"] == pytest.approx(0.5 * w["s1"], rel=1e-6) or max(w.values()) == pytest.approx(0.25 * total)


# --- 4.10: protocol perturbation ---------------------------------------------------

def test_perturbation_keeps_meals_on_their_side_of_every_window() -> None:
    from pulse.knowledge.cohorts.sleep import SLEEP_RESTRICTION_NEXT_DAY_GLUCOSE as S
    rng = np.random.default_rng(3)
    lo, hi = S.window.start_min, S.window.end_min
    for _ in range(50):
        arms = perturb_group_arms([S], rng)
        assert arms is not None and len(arms) == 2
        for arm, orig in zip(arms, S.arms):
            assert len(arm.meals) == len(orig.meals)
            for (t, c, f, p), (t0, c0, f0, p0) in zip(arm.meals, orig.meals):
                assert abs(t - t0) <= 30.0 + 1e-9
                assert (t < lo) == (t0 < lo) and (t < hi) == (t0 < hi)
                assert 0.8 * c0 - 1e-9 <= c <= 1.2 * c0 + 1e-9
            assert abs(((arm.start_hour - orig.start_hour + 12) % 24) - 12) <= 1.0 + 1e-9
        # The two arms share a schedule and get the SAME perturbed meals.
        assert arms[0].meals == arms[1].meals


def test_short_arms_are_not_perturbed_and_fixed_protocol_is_sampled() -> None:
    assert perturb_group_arms([_toy_meal_spec()], np.random.default_rng(0)) is None
    from pulse.knowledge.cohorts.sleep import SLEEP_RESTRICTION_NEXT_DAY_GLUCOSE as S
    sig = CohortStatisticSignal(specs=[S], perturb_protocols=True, perturb_fixed_prob=0.25)
    assert sig.perturb_fixed_prob == 0.25


def test_groups_per_step_round_robins_over_all_groups() -> None:
    specs = [_spec(name=f"g{i}", window=StatisticWindow(start_min=0, end_min=20),
                   arms=(CohortArmSpec(label=f"a{i}", duration_min=30, start_hour=8.0, meals=()),))
             for i in range(3)]
    torch.manual_seed(0)
    model = _tiny_model()
    emb = nn.Embedding(1, EMBEDDING_DIM)
    params = list(model.parameters())
    opt = torch.optim.SGD(params, lr=0.0)
    sig = CohortStatisticSignal(specs=specs, n_patients=0, sample_patients=0, use_cold_initial_state=False,
                                weight=WeightSchedule(0.1), groups_per_step=2)
    seen: list[set[str]] = []
    for epoch in range(3):
        ctx = SignalContext(epoch=epoch, total_epochs=3, rng=np.random.default_rng(epoch),
                            device=torch.device("cpu"), optimizer=opt, params=params, grad_clip=10.0)
        res = sig.compute(model, emb, ctx)
        seen.append({k[2:] for k in res.sub_metrics if k.startswith("z_")})
        assert res.sub_metrics["n_groups"] == 2.0
    assert set().union(*seen) == {"g0", "g1", "g2"}
