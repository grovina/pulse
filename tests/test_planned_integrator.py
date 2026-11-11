"""The planned integrator computes the model's own rates, faster.

``integrate`` evaluates everything that does not depend on the ODE state once per
rollout (``_RolloutPlan``) and runs every MLP head as one fused network
(``_HeadBank``). These tests pin that this is a re-ordering of the same arithmetic,
not a second model: the planned rollout matches the per-minute ``model.forward``
reference, per-member protocols match separate rollouts, and the batched training
helpers built on it match their serial forms. The optional per-marker decay channel
(PLAN B2, ``euler_step``) goes through the same plan and the same pointwise path, so it
is pinned here the same way; its numerics are in ``tests/test_etd_step.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
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
    _meal_outputs,
    _swallow_window,
    integrate,
    precompute_duodenal_outputs,
    precompute_gut_outputs,
)
from pulse.modules import (
    AppetiteModule,
    CardiovascularModule,
    GastricModule, CircadianModule,
    HepatobiliaryModule,
    MetabolicModule,
    RespiratoryModule,
    StressModule,
    ThermoregModule,
)
from pulse.modules.gut import MealEvent
from pulse.types import (
    EMBEDDING_DIM,
    MARKER_INDEX,
    MODULE_MARKER_INDICES,
    NORM_CENTER,
    NORM_SCALE,
    STATE_DIM,
)

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


_ALL_MODULES = (
    MetabolicModule, AppetiteModule, StressModule, CardiovascularModule,
    ThermoregModule, RespiratoryModule, HepatobiliaryModule, GastricModule,
    CircadianModule,
)


def _report_decay(monkeypatch, fn, modules=_ALL_MODULES) -> None:
    """Make exactly ``modules`` report ``fn(self, state, coupling, const, drv, raw)`` as their
    decay (a module opts in by defining ``decay_rates`` on its class), whatever the real modules
    define by the time this runs."""
    for cls in modules:
        monkeypatch.setattr(cls, "decay_rates", fn, raising=False)
    _only_reporting(monkeypatch, modules)


def _only_reporting(monkeypatch, modules) -> None:
    monkeypatch.setattr("pulse.model._reports_decay", lambda mod: type(mod) in modules)


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


def test_planned_rollout_matches_the_pointwise_reference_with_decay(monkeypatch) -> None:
    """The decay channel is assembled twice — per module in ``_RolloutPlan`` and in
    ``forward(return_decay=True)`` — and both must hand ``euler_step`` the same thing: states,
    embedding gradients, parameter gradients and the gradient of a learnable decay scale."""
    k_scale = torch.nn.Parameter(torch.tensor(0.2))

    def state_dependent(self, state, coupling, const, drv, raw):
        return k_scale * (0.25 + torch.sigmoid(2.0 * state))

    T = 90
    t = torch.arange(T, dtype=torch.float32)
    sw = (torch.sin(t / 15.0) > -0.3).float()
    act = 0.4 * (torch.cos(t / 9.0) > 0.5).float()
    g = torch.Generator().manual_seed(7)
    for modules in (_ALL_MODULES, (CardiovascularModule, MetabolicModule, RespiratoryModule)):
        model = _live_model(13)
        _report_decay(monkeypatch, state_dependent, modules)
        kw = dict(state=_states(3, 8), emb=0.4 * torch.randn(3, EMBEDDING_DIM, generator=g), T=T,
                  meals=MEALS, sleep_wake=sw, activity=act, start_time_minutes=480.0)
        out = {}
        for planned in (False, True):
            k_scale.grad = None
            rolled = _roll(model, planned, **dict(kw))
            assert k_scale.grad is not None, f"planned={planned}: the decay never reached the step"
            out[planned] = rolled + (k_scale.grad.clone(),)
        ref, new = out[False], out[True]
        _close(new[0], ref[0], 1e-5)
        _close(new[1], ref[1], 1e-4)
        for gp, gr in zip(new[2], ref[2]):
            assert (gp is None) == (gr is None)
            if gr is not None and float(gr.abs().max()) > 1e-6:
                _close(gp, gr, 1e-3)
        assert float(ref[3]) != 0.0 and bool(torch.isfinite(new[3]))
        _close(new[3], ref[3], 1e-3)
        monkeypatch.undo()

    # ... and the channel is live: the same model without it integrates a different trajectory.
    _only_reporting(monkeypatch, ())
    plain = _roll(_live_model(13), True, **dict(kw))
    assert float((plain[0] - new[0]).abs().max()) > 1e-4


def test_decay_is_collected_in_marker_order_like_the_rates(monkeypatch) -> None:
    """``forward(return_decay=True)`` concatenates the modules' decays in ``_MODULE_ORDER`` and
    gathers them into marker order exactly as it does the rates: every module's markers get
    their own values, whichever module they belong to."""
    model = _live_model(14)
    state = _states(2, 15)
    emb = 0.3 * torch.randn(2, EMBEDDING_DIM, generator=torch.Generator().manual_seed(16))
    gut, duo = torch.zeros(2, 4), torch.zeros(2, 3)
    t = torch.tensor([400.0, 400.0])
    want = torch.zeros(STATE_DIM)
    _only_reporting(monkeypatch, _ALL_MODULES)
    for name, mod in model._modules_by_name.items():
        idx = MODULE_MARKER_INDICES[name]
        vals = 0.01 * (1 + len(want.nonzero())) + 0.001 * torch.arange(len(idx), dtype=torch.float32)
        want[idx] = vals
        monkeypatch.setattr(type(mod), "decay_rates", (lambda v: lambda self, state, coupling, const, drv, raw:
                                                       v.expand(state.shape).clone())(vals), raising=False)
    rates, decay = model(state, emb, t, [], gut_override=gut, duodenal_override=duo, return_decay=True)
    assert decay.shape == rates.shape == (2, STATE_DIM)
    assert torch.equal(decay, want.expand(2, STATE_DIM))
    # The rates are those of the call that does not ask for the decay.
    assert torch.equal(rates, model(state, emb, t, [], gut_override=gut, duodenal_override=duo))
    # Unbatched, the decay is unbatched too.
    r1, d1 = model(state[0], emb[0], t[0], [], gut_override=gut[0], duodenal_override=duo[0],
                   return_decay=True)
    assert d1.shape == r1.shape == (STATE_DIM,) and torch.equal(d1, want)


def test_a_module_that_reports_no_decay_contributes_zeros_and_none_reports_none(monkeypatch) -> None:
    model = _live_model(15)
    state, emb = _states(2, 17), torch.zeros(2, EMBEDDING_DIM)
    kw = dict(gut_override=torch.zeros(2, 4), duodenal_override=torch.zeros(2, 3), return_decay=True)
    t = torch.tensor([400.0, 400.0])
    _only_reporting(monkeypatch, ())
    rates, decay = model(state, emb, t, [], **kw)
    assert decay is None and rates.shape == (2, STATE_DIM)
    monkeypatch.undo()
    _report_decay(monkeypatch, lambda self, state, coupling, const, drv, raw: torch.full_like(state, 0.3),
                  (CardiovascularModule,))
    _, decay = model(state, emb, t, [], **kw)
    cvs = MODULE_MARKER_INDICES["cardiovascular"]
    assert bool((decay[:, cvs] == 0.3).all())
    assert float(decay.abs().sum() - 0.3 * 2 * len(cvs)) == pytest.approx(0.0, abs=1e-6)


def test_a_parameter_only_decay_broadcasts_across_the_batch(monkeypatch) -> None:
    """Cardiovascular, thermoreg and respiratory keep their k in ``constants``: a decay with no
    batch dimension must be broadcast to the rate's shape in the plan and in ``forward``."""
    model = _live_model(16)
    row = {CardiovascularModule: [0.3, 0.3, 0.3, 0.3], ThermoregModule: [0.025], RespiratoryModule: [0.08, 0.08]}
    for cls, vals in row.items():
        monkeypatch.setattr(cls, "decay_rates", (lambda v: lambda self, state, coupling, const, drv, raw:
                                                 torch.tensor(v))(vals), raising=False)
    _only_reporting(monkeypatch, tuple(row))
    emb = 0.3 * torch.randn(3, EMBEDDING_DIM, generator=torch.Generator().manual_seed(18))
    state = _states(3, 19)
    with torch.no_grad():
        a = integrate(model, state, emb, 30, meals=MEALS, planned=True)
        b = integrate(model, state, emb, 30, meals=MEALS, planned=False)
    _close(a, b, 1e-5)


def test_a_decay_of_the_wrong_width_is_an_error_that_names_the_module(monkeypatch) -> None:
    """A module that reports a rate for the wrong number of markers would shift every later
    module's decay in the concatenation; it must fail where it is made, in the plan and in
    ``forward``."""
    model = _live_model(18)
    _report_decay(monkeypatch, lambda self, state, coupling, const, drv, raw: torch.zeros(state.shape[0], 3),
                  (CardiovascularModule,))  # it has four markers
    state, emb = _states(2, 20), torch.zeros(2, EMBEDDING_DIM)
    for planned in (True, False):
        with pytest.raises(ValueError, match="cardiovascular.decay_rates"):
            integrate(model, state, emb, 3, meals=MEALS, planned=planned)


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


def _member_meals(n: int) -> list[list[MealEvent]]:
    """Different meals per member, including a member with none and one past the window."""
    out = [
        [MealEvent(time=5.0 + 7 * b, carbs=30.0 + 9 * b, fats=4.0 + b, proteins=10.0 + 2 * b),
         MealEvent(time=40.4 + b, carbs=20.0, fats=8.0, proteins=12.0),
         MealEvent(time=-30.0, carbs=15.0, fats=3.0, proteins=2.0)]
        for b in range(n)
    ]
    out[1] = []
    out[2] = [MealEvent(time=500.0, carbs=50.0, fats=5.0, proteins=5.0)]
    return out


def test_one_stomach_run_gives_each_members_gut_and_duodenal_windows(monkeypatch) -> None:
    """Per-member meals cost one compartment simulation, not two per member, and each member's
    rows are what ``precompute_gut_outputs`` / ``precompute_duodenal_outputs`` give it alone —
    values and gradients to the embedding and to the emptying rate."""
    import pulse.model as model_module

    model = _live_model(12)
    B, T = 5, 90
    g = torch.Generator().manual_seed(3)
    emb = (0.4 * torch.randn(B, EMBEDDING_DIM, generator=g)).requires_grad_(True)
    member_meals = _member_meals(B)
    weights = torch.rand(B, T, 4, generator=g)

    runs = []
    real = model_module.simulate
    monkeypatch.setattr(model_module, "simulate", lambda *a, **k: runs.append(1) or real(*a, **k))
    swallow = _swallow_window(T, member_meals, B, member_meals, torch.device("cpu"))
    gut, duo = _meal_outputs(model, emb, swallow, has_meals=True)
    assert len(runs) == 1
    loss = (gut * weights).sum() + (duo * weights[..., :3]).sum()
    model.zero_grad()
    emb_grad, = torch.autograd.grad(loss, emb, retain_graph=True)
    loss.backward()
    rate_grad = model.gastric.log_empty_kcal.grad.clone()

    monkeypatch.undo()
    emb2 = emb.detach().clone().requires_grad_(True)
    gut_ref = torch.stack([
        precompute_gut_outputs(model, emb2[b], T, meals=member_meals[b]) for b in range(B)])
    duo_ref = torch.stack([
        precompute_duodenal_outputs(model, T, meals=member_meals[b]) for b in range(B)])
    loss_ref = (gut_ref * weights).sum() + (duo_ref * weights[..., :3]).sum()
    model.zero_grad()
    emb2_grad, = torch.autograd.grad(loss_ref, emb2, retain_graph=True)
    loss_ref.backward()

    torch.testing.assert_close(gut, gut_ref, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(duo, duo_ref, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(emb_grad, emb2_grad, rtol=1e-5, atol=1e-8)
    torch.testing.assert_close(rate_grad, model.gastric.log_empty_kcal.grad, rtol=1e-5, atol=1e-8)
    assert float(rate_grad.abs()) > 0.0


def test_a_shared_meal_list_is_one_stomach_run_for_every_member() -> None:
    model = _live_model(13)
    B, T = 3, 70
    emb = 0.3 * torch.randn(B, EMBEDDING_DIM, generator=torch.Generator().manual_seed(4))
    swallow = _swallow_window(T, MEALS, B, None, torch.device("cpu"))
    gut, duo = _meal_outputs(model, emb, swallow[:1], has_meals=True)
    assert gut.shape == (B, T, 4) and duo.shape == (1, T, 3)
    torch.testing.assert_close(gut, precompute_gut_outputs(model, emb, T, meals=MEALS), rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(duo[0], precompute_duodenal_outputs(model, T, meals=MEALS), rtol=1e-6, atol=1e-7)
    no_gut, no_duo = _meal_outputs(model, emb, swallow[:1], has_meals=False)
    assert not no_gut.any() and not no_duo.any() and no_gut.shape == (B, T, 4)


def test_swallow_scatter_matches_adding_each_meal_in_turn() -> None:
    from pulse.gastric import swallow_from_member_meals

    member_meals = _member_meals(4) + [[MEALS[0], MEALS[0], MealEvent(time=20.2, carbs=1.0, fats=2.0, proteins=3.0)]]
    got = swallow_from_member_meals(80, member_meals, torch.device("cpu"))
    want = torch.zeros(len(member_meals), 80, 3)
    for b, meals in enumerate(member_meals):
        for m in meals:
            minute = int(np.ceil(m.time - 1e-9))
            if 0 <= minute < 80:
                want[b, minute] += torch.tensor([m.carbs, m.proteins, m.fats])
    torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_per_member_meals_roll_like_each_member_alone_with_gradients() -> None:
    model = _live_model(14)
    B, T = 4, 80
    emb = 0.3 * torch.randn(B, EMBEDDING_DIM, generator=torch.Generator().manual_seed(6))
    state = _states(B, 12)
    member_meals = _member_meals(B)
    traj, emb_grad, _ = _roll(model, True, state=state, emb=emb, T=T, meals=member_meals)
    for b in range(B):
        alone, alone_grad, _ = _roll(
            model, True, state=state[b:b + 1], emb=emb[b:b + 1], T=T, meals=member_meals[b])
        _close(traj[b], alone[0], 1e-6)
        _close(emb_grad[b], alone_grad[0], 1e-5)


def test_integrate_steps_whole_minutes_only() -> None:
    model = _live_model(15)
    with pytest.raises(ValueError, match="whole minutes"):
        integrate(model, _states(1, 1), torch.zeros(1, EMBEDDING_DIM), 5, dt=0.5)


# Recorded from the planned integrator before the per-minute step dropped the terms nothing
# read and moved person- and protocol-constant terms into ``constants`` / ``drives``: a
# 240-minute rollout of ``_live_model(21)`` with three members (two meals, none, one meal),
# sleep, activity, an SGLT2 arm and per-member clocks. States are z-scores of the markers
# below at the last minute; the gradient is ``|d loss / d embedding|`` per member.
_GOLDEN_MARKERS = (
    "glucose", "insulin", "cortisol", "hr", "temp", "gastric_carb", "intestinal_bile", "bile_acids")
_GOLDEN_FINAL = [
    [0.264008, 0.768351, 0.476284, 0.563051, -14.7389, 1.31717, 1.43054, 2.72093],
    [-0.10647, -0.233391, -0.585703, -0.112075, -20.8669, 0.0, 0.553161, 1.19113],
    [0.0491768, 0.125701, -0.830219, 0.541172, -17.9501, 0.939248, 1.37881, 2.59091],
]
_GOLDEN_EMBEDDING_GRAD = [625.632, 261.89, 716.891]
_GOLDEN_OBSERVABLES_T89 = [
    [0.0, 2.72325, 0.0, 310.035, 237.348, 0.765552, 1.48435, 0.0615583, 0.130303, 1.0],
    [0.0, 1.8566, 0.0, 276.365, 208.278, 0.753632, 1.31951, 0.113586, 0.485541, 1.0],
    [0.0, 1.97396, 85.7782, 303.006, 231.933, 0.765442, 1.45066, 0.0844609, 0.227545, 1.0],
]


def test_a_rollout_keeps_the_trajectories_and_gradients_it_always_had() -> None:
    model = _live_model(21)
    steps = 240
    emb = (0.3 * torch.randn(3, EMBEDDING_DIM, generator=torch.Generator().manual_seed(2))).requires_grad_(True)
    state = _states(3, 4)
    meals = [
        [MealEvent(time=10.0, carbs=50.0, fats=15.0, proteins=20.0),
         MealEvent(time=150.0, carbs=30.0, fats=10.0, proteins=12.0)],
        [],
        [MealEvent(time=40.0, carbs=80.0, fats=25.0, proteins=10.0)],
    ]
    sleep = torch.ones(3, steps)
    sleep[:, :50] = 0.0
    activity = torch.zeros(3, steps)
    activity[1, 90:130] = 0.5
    sglt2 = torch.zeros(3, steps)
    sglt2[2] = 1.0
    kw = dict(meals=meals, sleep_wake=sleep, activity=activity, sglt2=sglt2,
              start_time_minutes=torch.tensor([300.0, 900.0, 1300.0], dtype=torch.float64))
    traj = integrate(model, state, emb, steps, **kw)
    scale = torch.tensor(NORM_SCALE)
    (traj * torch.linspace(0.5, 1.5, steps).view(1, steps, 1) / scale).sum().backward()

    cols = [MARKER_INDEX[m] for m in _GOLDEN_MARKERS]
    z = (traj[:, -1, cols].detach() - torch.tensor(NORM_CENTER)[cols]) / scale[cols]
    torch.testing.assert_close(z, torch.tensor(_GOLDEN_FINAL), rtol=2e-4, atol=1e-5)
    torch.testing.assert_close(
        emb.grad.norm(dim=-1), torch.tensor(_GOLDEN_EMBEDDING_GRAD), rtol=2e-4, atol=0.0)

    with torch.no_grad():
        observed = integrate(model, state, emb.detach(), 90, **{**kw, "sleep_wake": sleep[:, :90],
                             "activity": activity[:, :90], "sglt2": sglt2[:, :90]}, observe=True)
        plain = integrate(model, state, emb.detach(), 90, **{**kw, "sleep_wake": sleep[:, :90],
                          "activity": activity[:, :90], "sglt2": sglt2[:, :90]})
    # Observing computes the study quantities; it does not change what is integrated.
    assert torch.equal(observed[..., :STATE_DIM], plain)
    torch.testing.assert_close(
        observed[:, -1, STATE_DIM:], torch.tensor(_GOLDEN_OBSERVABLES_T89), rtol=2e-4, atol=1e-5)


def test_the_checkpoint_layout_is_untouched() -> None:
    """Moving terms into ``constants`` and ``drives`` adds no parameter or buffer."""
    model = _live_model(21)
    assert sum(p.numel() for p in model.parameters()) == 11079
    assert len(model.state_dict()) == 252


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
    ext = {"sleep_wake": sw[:, t], "activity": act[:, t], "sglt2": torch.zeros(B),
               "glucose_infusion": torch.zeros(B), "glucose_clamp": torch.zeros(B),
               "insulin_clamp": torch.zeros(B), "bile_fistula": torch.zeros(B)}
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
    # The mean hinge on this window is ~1e-6, so a relative tolerance treats a
    # last-bit difference between the fused heads and the pointwise heads as a miss.
    assert float((got - torch.stack(want).mean()).abs().detach()) < 1e-6


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
    less than 16 rollouts. Loose bound — it guards the design, not the machine.

    Timed on one intra-op thread, as the trainer runs (``--torch-threads 1``): the 16-member
    step is multi-threaded where the 1-member step is not, and on four threads next to other
    busy processes the 16-member / 1-member time ratio measured 10.9, against 1.07 on one
    thread — a scheduler effect, not the design this test guards."""
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

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        one, sixteen = run(1), run(16)
    finally:
        torch.set_num_threads(threads)
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


def test_compiled_steps_match_the_eager_step_with_decay(monkeypatch) -> None:
    """The same guard with the decay channel on: the compiled step carries the ETD1 gain (the
    ``where``-guarded series and closed form, the log and logit coordinates) and must agree with
    eager, forward and backward. Opt-in for the same reason as the test above."""
    import os

    if os.environ.get("PULSE_TEST_COMPILE") != "1":
        pytest.skip("set PULSE_TEST_COMPILE=1 to compile the planned step")
    from pulse.model import enable_compiled_steps

    def decay(self, state, coupling, const, drv, raw):
        return 0.05 + 0.75 * torch.sigmoid(3.0 * state)

    _report_decay(monkeypatch, decay)
    model = _live_model(17)
    try:
        assert enable_compiled_steps(model), "warm-up failed (see the printed reason)"
        state = _states(3, 41)
        emb = 0.3 * torch.randn(3, EMBEDDING_DIM, generator=torch.Generator().manual_seed(10))
        compiled = _roll(model, True, state=state, emb=emb, T=30, meals=MEALS)
    finally:
        enable_compiled_steps(enabled=False)
    eager = _roll(model, True, state=state, emb=emb, T=30, meals=MEALS)
    _close(compiled[0], eager[0], 1e-5)
    _close(compiled[1], eager[1], 1e-4)
