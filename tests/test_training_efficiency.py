"""The efficiency machinery is exact: batching, the prepare/step split and the frozen
calibration change the cost of training, not what it computes
(docs/training-efficiency.md)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from pulse.coupling_prior_loss import (
    _eps_for_marker,
    coupling_band_hinge,
    coupling_prior_loss_on_window,
    normalized_sensitivity,
)
from pulse.knowledge.base import CouplingPrior
from pulse.model import (
    ModularPhysiologyNetwork,
    euler_step,
    frozen_parameters,
    integrate,
    precompute_duodenal_outputs,
    precompute_gut_outputs,
)
from pulse.modules.gut import MealEvent
from pulse.rollouts import RolloutRequest, rollout_many
from pulse.training import SignalContext, TrajectoryRolloutSignal, WeightSchedule
from pulse.types import EMBEDDING_DIM, MARKER_INDEX, NORM_CENTER, STATE_DIM

_CENTER = torch.tensor(NORM_CENTER, dtype=torch.float32)


def _model(seed: int = 0) -> ModularPhysiologyNetwork:
    """A small model with every weight perturbed, so zero-init layers are live."""
    torch.manual_seed(seed)
    m = ModularPhysiologyNetwork(
        metabolic_hidden=8, appetite_hidden=8, stress_hidden=8,
        cardiovascular_hidden=8, thermoreg_hidden=8, respiratory_hidden=8,
        gut_hidden=8, hepatobiliary_hidden=8,
    )
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.02 * torch.randn(p.shape, generator=g))
    return m


def _close(a: torch.Tensor, b: torch.Tensor, rtol: float = 1e-4) -> bool:
    return bool(((a - b).abs() <= rtol * (b.abs() + 1e-3)).all())


# --- the prepare / step split ------------------------------------------------------

def test_rollout_matches_stepping_forward_by_hand() -> None:
    """``integrate`` (prepare once, step per minute) equals calling ``forward`` per step."""
    m = _model()
    emb = 0.3 * torch.randn(2, EMBEDDING_DIM)
    meals = [MealEvent(time=5, carbs=50, fats=10, proteins=20)]
    n = 30
    sw = (torch.arange(n) > 12).float()
    gut = precompute_gut_outputs(m, emb, n, meals=meals)
    duo = precompute_duodenal_outputs(m, n, meals=meals)
    with torch.no_grad():
        traj = integrate(m, _CENTER.expand(2, -1), emb, n, start_time_minutes=1430.0,
                         meals=meals, sleep_wake=sw, gut_outputs=gut, duodenal_outputs=duo)
        state = _CENTER.expand(2, -1)
        for k in range(5):
            assert _close(traj[:, k], state)
            t = torch.full((2,), (1430.0 + k) % 1440.0)
            rates = m(state, emb, t, meals, sleep_wake=sw[k].expand(2), gut_override=gut[:, k],
                      duodenal_override=duo[k])
            state = euler_step(state, rates, 1.0)


# --- heterogeneous batches ---------------------------------------------------------

def _requests(m: ModularPhysiologyNetwork) -> list[RolloutRequest]:
    g = torch.Generator().manual_seed(5)
    return [
        RolloutRequest(
            duration_min=40, start_minutes=480.0,
            meals=[MealEvent(time=3, carbs=60, fats=20, proteins=25)],
            embeddings=0.2 * torch.randn(2, EMBEDDING_DIM, generator=g),
            states=_CENTER.expand(2, -1),
            sleep_wake=torch.ones(40), activity=torch.zeros(40),
        ),
        RolloutRequest(  # shorter, other clock, no meal, learned input defaults
            duration_min=25, start_minutes=1420.0, meals=[],
            embeddings=0.2 * torch.randn(1, EMBEDDING_DIM, generator=g),
            states=_CENTER.unsqueeze(0) * 1.02,
        ),
        RolloutRequest(  # sleep logged, activity withheld (NaN row in the batch)
            duration_min=33, start_minutes=60.0,
            meals=[MealEvent(time=10, carbs=30, fats=30, proteins=10)],
            embeddings=0.2 * torch.randn(3, EMBEDDING_DIM, generator=g),
            states=_CENTER.expand(3, -1),
            sleep_wake=torch.zeros(33),
        ),
    ]


def test_rollout_many_equals_rolling_each_protocol_alone() -> None:
    m = _model()
    reqs = _requests(m)
    with torch.no_grad():
        together = rollout_many(m, reqs)
        alone = [rollout_many(m, [r])[0] for r in reqs]
    for r, a, b in zip(reqs, together, alone):
        assert a.shape == (r.states.shape[0], r.duration_min, STATE_DIM)
        assert _close(a, b), float((a - b).abs().max())


def test_rollout_many_is_gradient_identical() -> None:
    m = _model()

    def grads(batched: bool) -> list[torch.Tensor]:
        reqs = _requests(m)
        for r in reqs:
            r.embeddings.requires_grad_(True)
        m.zero_grad()
        trajs = rollout_many(m, reqs) if batched else [rollout_many(m, [r])[0] for r in reqs]
        sum(t.pow(2).mean() for t in trajs).backward()
        return [p.grad.clone() for p in m.parameters() if p.grad is not None] + [
            r.embeddings.grad.clone() for r in reqs]

    for a, b in zip(grads(True), grads(False)):
        assert float((a - b).abs().max()) <= 1e-3 * float(b.abs().max()) + 1e-9


def test_rollout_many_isolates_a_diverged_request() -> None:
    m = _model()
    reqs = _requests(m)
    bad = RolloutRequest(
        duration_min=10, start_minutes=0.0, meals=[],
        embeddings=torch.zeros(1, EMBEDDING_DIM),
        states=torch.full((1, STATE_DIM), float("nan")),
    )
    with torch.no_grad():
        out = rollout_many(m, [reqs[0], bad, reqs[1]], isolate_nonfinite=True)
        ref = rollout_many(m, [reqs[0], reqs[1]])
    assert out[1] is None
    assert _close(out[0], ref[0]) and _close(out[2], ref[1])


def test_withheld_input_rows_take_the_learned_default() -> None:
    """A NaN sleep row in a batch behaves exactly like ``sleep_wake=None`` alone."""
    m = _model()
    emb = 0.2 * torch.randn(2, EMBEDDING_DIM)
    n = 20
    sw = torch.stack([torch.ones(n), torch.full((n,), float("nan"))])
    with torch.no_grad():
        mixed = integrate(m, _CENTER.expand(2, -1), emb, n, sleep_wake=sw)
        logged = integrate(m, _CENTER.unsqueeze(0), emb[:1], n, sleep_wake=torch.ones(n))
        defaulted = integrate(m, _CENTER.unsqueeze(0), emb[1:], n)
    assert _close(mixed[:1], logged) and _close(mixed[1:], defaulted)


def test_active_steps_holds_finished_rows() -> None:
    m = _model()
    emb = 0.2 * torch.randn(2, EMBEDDING_DIM)
    with torch.no_grad():
        traj = integrate(m, _CENTER.expand(2, -1), emb, 30, active_steps=torch.tensor([30, 12]))
        short = integrate(m, _CENTER.unsqueeze(0), emb[1:], 12)
    assert _close(traj[1:, :12], short)
    assert torch.equal(traj[1, 12:], traj[1, 11].expand(18, -1))


# --- calibration does not train the model ---------------------------------------------

def test_frozen_parameters_restores_flags_and_blocks_weight_grads() -> None:
    m = _model()
    next(m.parameters()).requires_grad_(False)  # a pre-existing frozen flag survives
    flags = [p.requires_grad for p in m.parameters()]
    emb = torch.zeros(EMBEDDING_DIM, requires_grad=True)
    with frozen_parameters(m):
        integrate(m, _CENTER, emb, 15).pow(2).mean().backward()
    assert emb.grad is not None
    assert all(p.grad is None for p in m.parameters())
    assert [p.requires_grad for p in m.parameters()] == flags


def test_distillation_calibration_leaves_no_weight_gradient() -> None:
    from pulse.training import ColdModelDistillationSignal

    m = _model()
    sig = ColdModelDistillationSignal(weight=WeightSchedule(0.3), pool="synthetic", mode="anchored")
    proto = min(sig._protocols, key=lambda p: p.duration_min)
    sig._calibrate(m, proto, torch.zeros(EMBEDDING_DIM), 1, torch.device("cpu"))
    assert all(p.grad is None for p in m.parameters())


# --- the coupling prior --------------------------------------------------------------

def test_batched_coupling_prior_equals_the_per_prior_loop() -> None:
    m = _model()
    priors = [
        CouplingPrior("glucose", "insulin", +1, (0.01, 0.5)),
        CouplingPrior("insulin", "glucose", -1, (0.0001, 0.01)),
        CouplingPrior("cortisol", "hr", +1, (0.01, 0.3)),
    ]
    n = 40
    emb = 0.2 * torch.randn(EMBEDDING_DIM)
    meals = [MealEvent(time=2, carbs=60, fats=20, proteins=25)]
    gut = precompute_gut_outputs(m, emb, n, meals=meals).detach()
    duo = precompute_duodenal_outputs(m, n, meals=meals).detach()
    sw = torch.ones(n)
    with torch.no_grad():
        traj = integrate(m, _CENTER, emb, n, start_time_minutes=500.0, meals=meals,
                         sleep_wake=sw, gut_outputs=gut, duodenal_outputs=duo)
    got = coupling_prior_loss_on_window(m, traj, emb, 500.0, meals, sw, None, priors,
                                        n_samples=3, gut_outputs=gut, duodenal_outputs=duo)
    idxs = sorted({int((k + 1) * (n - 1) / 4) for k in range(3)})
    ref = 0.0
    for idx in idxs:
        kw = dict(sleep_wake=sw[idx:idx + 1], gut_override=gut[idx:idx + 1],
                  duodenal_override=duo[idx:idx + 1])
        t = torch.tensor([500.0 + idx])
        total = 0.0
        for p in priors:
            si, ti = MARKER_INDEX[p.source_marker], MARKER_INDEX[p.target_marker]
            eps = _eps_for_marker(p.source_marker)
            s1 = traj[idx].clone()
            s1[si] += eps
            r0 = m(traj[idx:idx + 1], emb.unsqueeze(0), t, meals, **kw)[0]
            r1 = m(s1.unsqueeze(0), emb.unsqueeze(0), t, meals, **kw)[0]
            sens = normalized_sensitivity((r1[ti] - r0[ti]) / eps, p.source_marker, p.target_marker)
            total = total + coupling_band_hinge(sens, p)
        ref = ref + total / len(priors)
    ref = ref / len(idxs)
    assert abs(float(got) - float(ref)) <= 1e-4 * abs(float(ref)) + 1e-7


# --- trajectory windows per step -----------------------------------------------------

def _traj_signal(batch: int) -> TrajectoryRolloutSignal:
    return TrajectoryRolloutSignal(
        n_patients=2, n_days=1, seed=0, contribution_weights={"full_body": 1.0},
        windows_per_patient=1, meal_window_bias=0.5, input_dropout=0.5, huber_delta=1.0,
        gut_loss_weight=0.0, coupling_weight=WeightSchedule(0.0), verifier_weight=WeightSchedule(0.0),
        n_default_patients=1, batch_windows=batch,
    )


def test_batched_windows_score_each_window_as_alone() -> None:
    """One B=3 rollout gives the per-window losses three single-window rollouts give."""
    m = _model()
    emb = nn.Embedding(2, EMBEDDING_DIM)
    sig = _traj_signal(3)
    rng = np.random.default_rng(0)
    windows = [sig._sample_window(i, d, 0, rng) for i, d in enumerate(sig.dataset)]
    dev = torch.device("cpu")

    def losses(ws):
        pred, e, gut, duo = sig._rollout_windows(m, emb, ws, dev)
        return [float(sig._window_loss(m, w, pred[i], e[i], gut[i],
                                       None if duo is None else duo[i], 0.0, 0.0, dev)[0])
                for i, w in enumerate(ws)]

    with torch.no_grad():
        together = losses(windows)
        alone = [losses([w])[0] for w in windows]
    np.testing.assert_allclose(together, alone, rtol=1e-4)


@pytest.mark.parametrize("batch", [1, 2, 3])
def test_window_batches_yield_cumulative_window_counts(batch: int) -> None:
    m = _model()
    emb = nn.Embedding(2, EMBEDDING_DIM)
    params = list(m.parameters()) + list(emb.parameters())
    ctx = SignalContext(epoch=0, total_epochs=1, rng=np.random.default_rng(0),
                        device=torch.device("cpu"), optimizer=torch.optim.Adam(params, lr=1e-4),
                        params=params, grad_clip=10.0)
    sig = _traj_signal(batch)
    counts = list(sig.iter_windows(m, emb, ctx))
    assert counts[-1] == 3 and sig.last_result.n_units == 3
    assert len(counts) == -(-3 // batch)
