"""The planned integrator computes the model's own rates, faster.

``integrate`` evaluates everything that does not depend on the ODE state once per
rollout (``_RolloutPlan``) and runs every MLP head as one fused network
(``_HeadBank``). These tests pin that this is a re-ordering of the same arithmetic,
not a second model: the planned rollout matches the per-minute ``model.forward``
reference, per-member protocols match separate rollouts, and the batched training
helpers built on it match their serial forms.
"""

from __future__ import annotations

import numpy as np
import torch

from pulse.cohort_loss import _rollout_arm_states, rollout_arms
from pulse.coupling_prior_loss import (
    coupling_band_hinge,
    coupling_prior_loss_on_window,
    normalized_sensitivity,
)
from pulse.knowledge.base import CouplingPrior
from pulse.knowledge.cohort_types import CohortArmSpec
from pulse.model import (
    ModularPhysiologyNetwork,
    _HeadBank,
    integrate,
    precompute_duodenal_outputs,
    precompute_gut_outputs,
)
from pulse.modules.gut import MealEvent
from pulse.types import EMBEDDING_DIM, MARKER_INDEX, NORM_CENTER, NORM_SCALE, STATE_DIM

MEALS = [
    MealEvent(time=20.0, carbs=60.0, fats=20.0, proteins=25.0),
    MealEvent(time=-150.0, carbs=40.0, fats=10.0, proteins=15.0),
]


def _live_model(seed: int = 0) -> ModularPhysiologyNetwork:
    """Small widths, and every zero-init output layer woken so all heads matter."""
    torch.manual_seed(seed)
    model = ModularPhysiologyNetwork(
        metabolic_hidden=12, appetite_hidden=8, stress_hidden=8,
        cardiovascular_hidden=12, thermoreg_hidden=8, respiratory_hidden=8,
        gut_hidden=8, hepatobiliary_hidden=8,
    )
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    return model


def _states(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    scale = torch.tensor(NORM_SCALE)
    s = torch.tensor(NORM_CENTER) + 0.3 * torch.randn(n, STATE_DIM, generator=g) * scale
    s = s.clamp(min=1e-3)
    s[:, MARKER_INDEX["spo2"]] = s[:, MARKER_INDEX["spo2"]].clamp(90.0, 99.5)
    s[:, MARKER_INDEX["sbp"]] = s[:, MARKER_INDEX["dbp"]] + 40.0
    s[:, MARKER_INDEX["insulin_action"]] = 0.1 * torch.randn(n, generator=g)
    return s


def _close(a: torch.Tensor, b: torch.Tensor, tol: float) -> None:
    a, b = a.detach(), b.detach()
    scale = b.abs().max().clamp(min=1e-6)
    err = float((a - b).abs().max() / scale)
    assert err < tol, f"normalized max error {err:.3e} >= {tol}"


def _roll(model, planned: bool, **kw) -> tuple[torch.Tensor, torch.Tensor, list]:
    emb = kw.pop("emb").clone().requires_grad_(True)
    traj = integrate(model, kw.pop("state"), emb, kw.pop("T"), planned=planned, **kw)
    w = torch.linspace(0.5, 1.5, traj.shape[-2]).unsqueeze(-1) / torch.tensor(NORM_SCALE)
    model.zero_grad()
    (traj * w).sum().backward()
    return traj.detach(), emb.grad.detach(), [
        None if p.grad is None else p.grad.detach().clone() for p in model.parameters()
    ]


def test_planned_rollout_matches_the_pointwise_reference() -> None:
    model = _live_model()
    T = 90
    t = torch.arange(T, dtype=torch.float32)
    sw = (torch.sin(t / 15.0) > -0.3).float()
    act = 0.4 * (torch.cos(t / 9.0) > 0.5).float()
    g = torch.Generator().manual_seed(3)
    cases = [
        dict(state=_states(1, 1)[0], emb=0.4 * torch.randn(EMBEDDING_DIM, generator=g),
             meals=MEALS, sleep_wake=sw, activity=act, start_time_minutes=480.0),
        dict(state=_states(3, 2), emb=0.4 * torch.randn(3, EMBEDDING_DIM, generator=g),
             meals=MEALS, sleep_wake=sw, activity=None, start_time_minutes=1400.0),
        dict(state=_states(2, 3), emb=0.4 * torch.randn(2, EMBEDDING_DIM, generator=g),
             meals=[], sleep_wake=None, activity=act, start_time_minutes=30.0),
    ]
    for kw in cases:
        kw["T"] = T
        if kw["state"].dim() == 2 and not kw["meals"]:
            # The pointwise path needs precomputed gut outputs for a batch.
            kw["gut_outputs"] = precompute_gut_outputs(model, kw["emb"], T, meals=[])
        ref = _roll(model, False, **dict(kw))
        new = _roll(model, True, **dict(kw))
        _close(new[0], ref[0], 1e-5)
        _close(new[1], ref[1], 1e-4)
        for gp, gr in zip(new[2], ref[2]):
            assert (gp is None) == (gr is None)
            if gr is not None and float(gr.abs().max()) > 1e-6:
                _close(gp, gr, 1e-3)


def test_per_member_protocols_match_separate_rollouts() -> None:
    model = _live_model(1)
    T = 60
    g = torch.Generator().manual_seed(5)
    emb = 0.4 * torch.randn(2, EMBEDDING_DIM, generator=g)
    state = _states(2, 7)
    meals = [MEALS, [MealEvent(time=5.0, carbs=75.0, fats=0.0, proteins=0.0)]]
    starts = [300.0, 1200.0]
    sw = torch.stack([torch.ones(T), torch.full((T,), float("nan"))])  # member 1: missing
    act = torch.stack([torch.full((T,), 0.3), torch.zeros(T)])
    with torch.no_grad():
        batched = integrate(
            model, state, emb, T, meals=meals,
            start_time_minutes=torch.tensor(starts, dtype=torch.float64),
            sleep_wake=sw, activity=act,
        )
        for b in range(2):
            alone = integrate(
                model, state[b], emb[b], T, meals=meals[b], start_time_minutes=starts[b],
                sleep_wake=None if b == 1 else sw[b], activity=act[b],
            )
            _close(batched[b], alone, 1e-6)


def test_member_steps_hold_a_finished_member() -> None:
    model = _live_model(2)
    emb = torch.zeros(2, EMBEDDING_DIM)
    state = _states(2, 9)
    with torch.no_grad():
        out = integrate(model, state, emb, 50, meals=MEALS, member_steps=torch.tensor([50, 20]))
        short = integrate(model, state[1], emb[1], 20, meals=MEALS)
    _close(out[1, :20], short, 1e-6)
    # Held, not integrated: every row after the horizon repeats the last one.
    assert torch.equal(out[1, 20:], out[1, 19:20].expand(30, -1))


def test_head_bank_matches_every_head_network() -> None:
    model = _live_model(3)
    B, T = 3, 4
    g = torch.Generator().manual_seed(11)
    emb = 0.4 * torch.randn(B, EMBEDDING_DIM, generator=g)
    state = _states(B, 12)
    gut = torch.rand(B, T, 4, generator=g)
    duo = torch.rand(B, T, 3, generator=g)
    sw = torch.rand(B, T, generator=g)
    act = torch.rand(B, T, generator=g)
    tf = torch.randn(B, T, 4, generator=g)
    proj = {n: p(emb) for n, p in model.embedding_projections.items()}
    z = torch.cat([gut, duo, sw.unsqueeze(-1), act.unsqueeze(-1), tf], dim=-1)
    bank = _HeadBank(model, proj, z)
    norm = (state - model.norm_center) / model.norm_scale
    t = 2
    fused = bank(norm, bank.exo_steps[t])
    ext = {"sleep_wake": sw[:, t], "activity": act[:, t]}
    for name, slots in bank.slots.items():
        mod = model._modules_by_name[name]
        x = mod.head_input(
            norm[:, model._step_gather_indices(name, torch.device("cpu"))[0]],
            model.coupling_for(name, norm, gut[:, t], duo[:, t]),
            torch.stack([ext[e] for e in mod.external_inputs], dim=-1),
            proj[name], tf[:, t],
        )
        nets = mod.mlp_heads()
        for key, i in slots:
            _close(fused[i], nets[key](x), 1e-5)


def test_batched_coupling_prior_matches_the_per_prior_difference() -> None:
    model = _live_model(4)
    T = 40
    traj = _states(T, 13).requires_grad_(True)
    emb = 0.3 * torch.randn(EMBEDDING_DIM, generator=torch.Generator().manual_seed(2))
    priors = [
        CouplingPrior("glucose", "insulin", +1, (0.05, 2.0)),
        CouplingPrior("cortisol", "hr", +1, (0.0, 0.5)),
        CouplingPrior("insulin", "ghrelin", -1, (0.01, 1.0)),
    ]
    gut = precompute_gut_outputs(model, emb, T, meals=MEALS)
    duo = precompute_duodenal_outputs(model, T, meals=MEALS)
    sw = torch.ones(T)
    act = torch.zeros(T)
    got = coupling_prior_loss_on_window(
        model, traj, emb, 400.0, MEALS, sw, act, priors, n_samples=3,
        gut_window=gut, duodenal_window=duo,
    )
    # The definition: one forward pair per (step, prior) at the window's own inputs.
    want = []
    for idx in (9, 19, 29):
        t = torch.tensor([(400.0 + idx) % 1440.0])
        for p in priors:
            si, ti = MARKER_INDEX[p.source_marker], MARKER_INDEX[p.target_marker]
            eps = max(NORM_SCALE[si] * 0.04, 1e-4)
            s1 = traj[idx].clone()
            s1[si] = s1[si] + eps

            def rate(s: torch.Tensor) -> torch.Tensor:
                return model(
                    s.unsqueeze(0), emb.unsqueeze(0), t, MEALS,
                    sleep_wake=sw[idx:idx + 1], activity=act[idx:idx + 1],
                    gut_override=gut[idx:idx + 1], duodenal_override=duo[idx:idx + 1],
                )[0, ti]

            sens = (rate(s1) - rate(traj[idx])) / eps
            want.append(coupling_band_hinge(normalized_sensitivity(sens, p.source_marker, p.target_marker), p))
    _close(got, torch.stack(want).mean(), 1e-5)


def test_rollout_arms_matches_each_arm_alone() -> None:
    model = _live_model(5)
    arms = [
        CohortArmSpec(label="a", duration_min=40, start_hour=7.0,
                      meals=((10.0, 50.0, 10.0, 10.0),)),
        CohortArmSpec(label="b", duration_min=25, start_hour=22.0, meals=(),
                      sleep_wake=tuple([0.0] * 25)),
    ]
    g = torch.Generator().manual_seed(4)
    rows = [(arm, 0.3 * torch.randn(2, EMBEDDING_DIM, generator=g), _states(2, i + 20))
            for i, arm in enumerate(arms)]
    with torch.no_grad():
        together = rollout_arms(model, rows)
        for (arm, emb, state), traj in zip(rows, together):
            assert traj.shape == (2, arm.duration_min, STATE_DIM)
            _close(traj, _rollout_arm_states(model, emb, arm, state), 1e-6)


def test_drives_never_read_the_state() -> None:
    """The plan hands drives NaN in every marker coupling channel; a drive that read
    one would make the rollout non-finite."""
    model = _live_model(6)
    with torch.no_grad():
        out = integrate(model, _states(2, 30), torch.zeros(2, EMBEDDING_DIM), 30, meals=MEALS,
                        sleep_wake=torch.ones(30), activity=torch.zeros(30))
    assert bool(torch.isfinite(out).all())


def test_planned_rollout_is_nearly_flat_in_batch_size() -> None:
    """The reason batching pays: a step is dispatch-bound, so 16 members cost far
    less than 16 rollouts. Loose bound — it guards the design, not the machine."""
    import time

    model = _live_model(7)
    T = 60

    def run(B: int) -> float:
        best = float("inf")
        for _ in range(2):
            t0 = time.perf_counter()
            with torch.no_grad():
                integrate(model, _states(B, 1), torch.zeros(B, EMBEDDING_DIM), T, meals=MEALS)
            best = min(best, time.perf_counter() - t0)
        return best

    one, sixteen = run(1), run(16)
    assert sixteen < 4.0 * one, (one, sixteen)


def test_windows_per_step_batches_trajectory_windows() -> None:
    from pulse.training import SignalContext, TrajectoryRolloutSignal, WeightSchedule

    model = _live_model(8)
    emb = torch.nn.Embedding(2, EMBEDDING_DIM)
    params = list(model.parameters()) + list(emb.parameters())

    def run(k: int):
        sig = TrajectoryRolloutSignal(
            n_patients=2, n_days=1, seed=3, contribution_weights=None, windows_per_patient=2,
            meal_window_bias=0.5, input_dropout=0.3, huber_delta=1.0, gut_loss_weight=0.0,
            coupling_weight=WeightSchedule(0.0), verifier_weight=WeightSchedule(0.0),
            windows_per_step=k,
        )
        ctx = SignalContext(epoch=0, total_epochs=1, rng=np.random.default_rng(0),
                            device=torch.device("cpu"),
                            optimizer=torch.optim.SGD(params, lr=0.0), params=params, grad_clip=10.0)
        counts = list(sig.iter_windows(model, emb, ctx))
        return counts, sig.last_result

    counts1, res1 = run(1)
    counts2, res2 = run(2)
    assert counts1 == [1, 2, 3, 4] and counts2 == [2, 4]
    assert res1.n_units == res2.n_units == 4
    # lr = 0: the same four windows (same rng order) score the same per-window losses.
    np.testing.assert_allclose(res1.loss_sum, res2.loss_sum, rtol=1e-4)


def test_compiled_steps_match_the_eager_step() -> None:
    """``--compile-steps``: opt-in, needs a C++ compiler and minutes of compilation,
    so this runs only with PULSE_TEST_COMPILE=1."""
    import os

    import pytest

    if os.environ.get("PULSE_TEST_COMPILE") != "1":
        pytest.skip("set PULSE_TEST_COMPILE=1 to compile the planned step")
    from pulse.model import enable_compiled_steps

    model = _live_model(9)
    try:
        assert enable_compiled_steps(model), "warm-up failed (see the printed reason)"
        state = _states(3, 40)
        emb = 0.3 * torch.randn(3, EMBEDDING_DIM, generator=torch.Generator().manual_seed(9))
        compiled = _roll(model, True, state=state, emb=emb, T=30, meals=MEALS)
    finally:
        enable_compiled_steps(enabled=False)
    eager = _roll(model, True, state=state, emb=emb, T=30, meals=MEALS)
    _close(compiled[0], eager[0], 1e-5)
    _close(compiled[1], eager[1], 1e-4)
