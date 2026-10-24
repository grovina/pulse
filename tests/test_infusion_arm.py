"""IV glucose infusion and the clamps are exogenous inputs, like ``sglt2``.

``glucose_infusion`` is a per-minute appearance rate in mg/dL/min of the patient's
glucose space, added to the metabolic glucose rate. ``glucose_clamp`` is a target
glucose: the step infuses ``relu(Kp · (target − G))``. ``insulin_clamp`` is an insulin
plateau held by a constant infusion. None reaches the learned heads, and with all
absent every trajectory is the one the model always produced. What a clamp measures,
the infusion rate, is an observable the metabolic law publishes in the same minute.
"""

from __future__ import annotations

import torch

from pulse.cohort_loss import _rollout_arm_states, cohort_statistic_loss_one_spec, rollout_arms
from pulse.knowledge.cohort_types import (
    CohortArmSpec, CohortStatisticSpec, StatisticKind, StatisticWindow,
)
from pulse.model import ModularPhysiologyNetwork, integrate
from pulse.modules.metabolic import GLUCOSE_CLAMP_KP, VG_DL_PER_KG, clamp_infusion_t
from pulse.types import (
    EMBEDDING_DIM, MARKER_INDEX, NORM_CENTER, OBSERVABLE_DIM, SERIES_INDEX, STATE_DIM,
)

_G = MARKER_INDEX["glucose"]
_I = MARKER_INDEX["insulin"]
_GIR = SERIES_INDEX["glucose_infusion_rate"]
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


def test_insulin_clamp_holds_the_plateau() -> None:
    model = _model()
    plateau = 60.0
    series = torch.full((_T,), plateau)
    with torch.no_grad():
        base = _roll(model)
        for planned in (True, False):
            held = _roll(model, planned=planned, insulin_clamp=series, glucose_clamp=torch.full((_T,), 90.0))
            assert float((held[0, 40:, _I] - plateau).abs().max()) < 0.5
        no_clamp = _roll(model, insulin_clamp=torch.zeros(_T))
    assert torch.equal(base, no_clamp)
    assert float(base[0, -1, _I]) < plateau - 20.0


def test_observing_changes_no_state() -> None:
    model = _model()
    kw = {"glucose_clamp": torch.full((_T,), 120.0), "insulin_clamp": torch.full((_T,), 60.0)}
    with torch.no_grad():
        plain = _roll(model, **kw)
        observed = _roll(model, observe=True, **kw)
    assert observed.shape == (1, _T, STATE_DIM + OBSERVABLE_DIM)
    assert torch.equal(observed[..., :STATE_DIM], plain)


def test_the_gir_observable_is_the_clamp_infusion_of_its_own_minute() -> None:
    model = _model()
    target = torch.full((_T,), 120.0)
    with torch.no_grad():
        traj = _roll(model, observe=True, glucose_clamp=target)
    expected = clamp_infusion_t(traj[0, :, _G], target) * VG_DL_PER_KG
    torch.testing.assert_close(traj[0, :, _GIR], expected)


def test_a_clamp_gir_spec_scores_and_trains() -> None:
    model = _model()
    arm = _arm(glucose_clamp_mg_dl=90.0, insulin_clamp_uU_ml=60.0)
    spec = CohortStatisticSpec(
        name="clamp_gir", source="test", description="steady-state GIR",
        arms=(arm,), marker_id="glucose_infusion_rate", kind=StatisticKind.MEAN_IN_WINDOW,
        window=StatisticWindow(40, _T), target=8.0, sigma=1.0,
    )
    loss, pred, _z = cohort_statistic_loss_one_spec(model, [_emb()[0]], spec, _state()[0])
    assert pred > 0.0
    model.zero_grad()
    loss.backward()
    assert any(p.grad is not None and float(p.grad.abs().max()) > 0.0 for p in model.parameters())
