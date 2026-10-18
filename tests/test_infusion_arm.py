"""IV glucose infusion and the glucose clamp are exogenous inputs, like ``sglt2``.

``glucose_infusion`` is a per-minute appearance rate in mg/dL/min of the patient's
glucose space, added to the metabolic glucose rate. ``glucose_clamp`` is a target
glucose: the step infuses ``relu(Kp · (target − G))``. Neither reaches the learned heads,
and with both absent every trajectory is the one the model always produced.
"""

from __future__ import annotations

import torch

from pulse.cohort_loss import _rollout_arm_states, rollout_arms
from pulse.knowledge.cohort_types import CohortArmSpec
from pulse.model import ModularPhysiologyNetwork, integrate
from pulse.modules.metabolic import GLUCOSE_CLAMP_KP, clamp_infusion_t
from pulse.types import EMBEDDING_DIM, MARKER_INDEX, NORM_CENTER, STATE_DIM

_G = MARKER_INDEX["glucose"]
_T = 60
_INFUSION = 2.0  # mg/dL/min


def _model() -> ModularPhysiologyNetwork:
    torch.manual_seed(0)
    return ModularPhysiologyNetwork()


def _state() -> torch.Tensor:
    return torch.tensor(NORM_CENTER, dtype=torch.float32).unsqueeze(0)


def _emb() -> torch.Tensor:
    return torch.zeros(1, EMBEDDING_DIM)


def _roll(model, planned: bool = True, **kw) -> torch.Tensor:
    return integrate(model, _state(), _emb(), _T, planned=planned, **kw)


def test_infusion_raises_glucose() -> None:
    model = _model()
    with torch.no_grad():
        base = _roll(model)
        infused = _roll(model, glucose_infusion=torch.full((_T,), _INFUSION))
    assert infused[0, -1, _G] > base[0, -1, _G]
    assert torch.all(infused[0, 1:, _G] > base[0, 1:, _G])


def test_no_infusion_changes_nothing() -> None:
    model = _model()
    with torch.no_grad():
        for planned in (True, False):
            none = _roll(model, planned=planned)
            zeros = _roll(model, planned=planned, glucose_infusion=torch.zeros(_T))
            nans = _roll(model, planned=planned, glucose_infusion=torch.full((_T,), float("nan")))
            no_clamp = _roll(model, planned=planned, glucose_clamp=torch.zeros(_T))
            assert torch.equal(none, zeros)
            assert torch.equal(none, nans)
            assert torch.equal(none, no_clamp)


def test_planned_and_pointwise_paths_agree_with_infusion() -> None:
    model = _model()
    series = torch.full((_T,), _INFUSION)
    with torch.no_grad():
        planned = _roll(model, glucose_infusion=series)
        pointwise = _roll(model, planned=False, glucose_infusion=series)
    assert torch.allclose(planned, pointwise, rtol=1e-4, atol=1e-3)


def test_gradient_flows_through_the_infusion() -> None:
    model = _model()
    series = torch.full((_T,), _INFUSION)
    traj = _roll(model, glucose_infusion=series)
    model.zero_grad()
    traj[0, :, _G].sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert any(float(g.abs().max()) > 0.0 for g in grads)


def test_infusion_series_gets_gradient() -> None:
    model = _model()
    series = torch.full((_T,), _INFUSION, requires_grad=True)
    _roll(model, glucose_infusion=series)[0, -1, _G].backward()
    assert series.grad is not None and float(series.grad[0]) > 0.0


def _arm(**kw) -> CohortArmSpec:
    return CohortArmSpec(label="iv", duration_min=_T, start_hour=8.0, meals=(), **kw)


def test_arm_infusion_raises_glucose_through_cohort_loss() -> None:
    model = _model()
    plain, infused = _arm(), _arm(glucose_infusion=tuple([_INFUSION] * _T))
    state = _state()
    with torch.no_grad():
        base = _rollout_arm_states(model, _emb(), plain, state)
        got = _rollout_arm_states(model, _emb(), infused, state)
        rows = [(plain, _emb(), state), (infused, _emb(), state)]
        batched = rollout_arms(model, rows)
    assert got[0, -1, _G] > base[0, -1, _G]
    assert torch.allclose(batched[0], base, atol=1e-4)
    assert torch.allclose(batched[1], got, atol=1e-4)
    assert batched[1][0, -1, _G] > batched[0][0, -1, _G]


def test_arm_without_infusion_matches_plain_integrate() -> None:
    model = _model()
    state = _state()
    with torch.no_grad():
        arm = _arm()
        via_arm = _rollout_arm_states(model, _emb(), arm, state)
        assert via_arm.shape == (1, _T, STATE_DIM)
        direct = integrate(
            model, state, _emb(), _T, dt=1.0, start_time_minutes=8.0 * 60.0,
            sleep_wake=torch.ones(_T), activity=torch.zeros(_T),
        )
    assert torch.equal(via_arm, direct)


def test_clamp_holds_glucose_at_the_target() -> None:
    model = _model()
    target = 130.0
    series = torch.full((_T,), target)
    with torch.no_grad():
        base = _roll(model)
        clamped = _roll(model, glucose_clamp=series)
        for planned_off in (clamped, _roll(model, planned=False, glucose_clamp=series)):
            settled = planned_off[0, 20:, _G]
            assert float((settled - target).abs().max()) < 3.0
    assert base[0, -1, _G] < target - 20.0
    glucose = clamped[0, :, _G]
    gir = clamp_infusion_t(glucose, series)
    assert float(gir[30:].min()) > 0.0
    assert torch.allclose(gir, torch.relu(GLUCOSE_CLAMP_KP * (series - glucose)))


def test_clamp_arm_through_cohort_loss() -> None:
    model = _model()
    target = 130.0
    with torch.no_grad():
        traj = _rollout_arm_states(model, _emb(), _arm(glucose_clamp_mg_dl=target), _state())
        batched = rollout_arms(model, [(_arm(glucose_clamp_mg_dl=target), _emb(), _state())])
    assert float((traj[0, 20:, _G] - target).abs().max()) < 3.0
    assert torch.allclose(batched[0], traj, atol=1e-4)
