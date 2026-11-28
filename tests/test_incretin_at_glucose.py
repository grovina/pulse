"""The soma-shaped incretin term (``--incretin-at-glucose``): off by default, built at zero gain,
measured against the person's own fasted incretin factor, and a run setting that the checkpoint
carries to every loader. See ``pulse.incretin`` and STATE.md."""
from __future__ import annotations

import pytest
import torch

from pulse import incretin
from pulse.model import MealEvent, ModularPhysiologyNetwork, integrate
from pulse.types import MARKER_INDEX, NORM_CENTER

INS = MARKER_INDEX["insulin"]
N = 4
STEPS = 480


@pytest.fixture(autouse=True)
def _switch_restored(monkeypatch):
    monkeypatch.setattr(incretin, "INCRETIN_AT_GLUCOSE", False)


def _model() -> ModularPhysiologyNetwork:
    torch.manual_seed(3)
    m = ModularPhysiologyNetwork()
    m.eval()
    return m


def _people():
    torch.manual_seed(5)
    return torch.randn(N, 32) * 0.5


def _roll(m, on: bool, gain: float, meal: bool) -> torch.Tensor:
    incretin.set_incretin_at_glucose(on)
    with torch.no_grad():
        m.metabolic.incretin_glu_gain.fill_(gain)
        init = torch.tensor(NORM_CENTER, dtype=torch.float32).expand(N, -1).contiguous()
        meals = [[MealEvent(time=30.0, carbs=60.0, fats=30.0, proteins=25.0)]] * N if meal else None
        return integrate(m, init, _people(), STEPS, meals=meals, start_time_minutes=420.0)


def test_the_term_is_built_at_zero_and_off_by_default():
    m = _model()
    assert incretin.INCRETIN_AT_GLUCOSE is False
    assert float(m.metabolic.incretin_glu_gain.detach()) == 0.0
    assert float(m.metabolic.log_incretin_glu_thr.exp()) == pytest.approx(90.0, rel=1e-5)
    assert float(m.metabolic.log_incretin_glu_width.exp()) == pytest.approx(10.0, rel=1e-5)


@pytest.mark.parametrize("meal", [False, True])
def test_off_and_on_at_zero_gain_give_identical_rollouts(meal):
    m = _model()
    off = _roll(m, False, 0.0, meal)
    assert torch.equal(off, _roll(m, False, 3.0, meal))   # off ignores a gain that is set
    assert torch.equal(off, _roll(m, True, 0.0, meal))    # on with zero gain: nothing yet


def test_a_positive_gain_raises_late_meal_insulin_and_leaves_a_fast_alone():
    m = _model()
    off_meal, on_meal = _roll(m, False, 0.0, True), _roll(m, True, 5.0, True)
    off_fast, on_fast = _roll(m, False, 0.0, False), _roll(m, True, 5.0, False)
    late = slice(150, STEPS)
    meal_gain = float(on_meal[:, late, INS].mean() / off_meal[:, late, INS].mean()) - 1.0
    fast_gain = float(on_fast[:, :, INS].mean() / off_fast[:, :, INS].mean()) - 1.0
    assert meal_gain > 0.01
    assert abs(fast_gain) < 0.25 * meal_gain
    assert abs(fast_gain) < 0.01


def test_a_fasted_person_has_no_excess_over_their_own_fasted_factor():
    m = _model()
    glp1, gip = m.appetite.fasted_incretin_levels(
        m.embedding_projections["appetite"](_people()))
    assert glp1.shape == gip.shape == (N,) and not glp1.requires_grad
    isl = m.metabolic.islet()
    f = incretin.incretin_factor_t(
        torch.relu(glp1 - 10.0), torch.relu(gip - 10.0), isl)
    zero = incretin.incretin_at_glucose_t(
        f, f, torch.full((N,), 120.0), torch.tensor(1.0), torch.tensor(90.0), torch.tensor(10.0))
    assert torch.equal(zero, torch.zeros(N))


def test_the_cli_flag_is_off_by_default_and_sets_the_run():
    from pulse.train import build_arg_parser, resolved_train_config
    parser = build_arg_parser()
    assert parser.parse_args([]).incretin_at_glucose is False
    args = parser.parse_args(["--incretin-at-glucose"])
    assert args.incretin_at_glucose is True
    assert resolved_train_config(args)["incretin_at_glucose"] is True


def test_the_parameters_load_from_an_older_checkpoint_at_their_initial_values():
    m = _model()
    state = {k: v for k, v in m.state_dict().items() if "incretin_glu" not in k}
    fresh = ModularPhysiologyNetwork(**m.constructor_kwargs)
    fresh.load_model_state(state)                    # the three keys are ADDED_STATE_KEYS
    assert float(fresh.metabolic.incretin_glu_gain) == 0.0


def test_checkpoint_round_trip(monkeypatch, tmp_path):
    import pulse.train as train_mod
    from pulse.diagnostics.probe import load_model_from_checkpoint
    monkeypatch.setattr(train_mod, "_WATCHDOG_TIMEOUT_S", 0)
    prev = torch.get_num_threads()
    outs = {}
    try:
        for on in (False, True):
            out = tmp_path / f"m{int(on)}.pt"
            code = train_mod.train(
                n_patients=2, n_epochs=1, hidden_dim=8, windows_per_patient=1, n_days=1, seed=42,
                coupling_prior_weight=0.0, verifier_loss_weight=0.0, cohort_statistic_weight=0.0,
                setpoint_supervision_weight=0.5, embedding_prior_weight=0.001,
                contribution_weights={"full_body": 1.0}, output_path=str(out), scoreboard=False,
                incretin_at_glucose=on, train_only="metabolic.incretin_glu_*" if on else None,
            )
            assert code == 0
            outs[on] = out
    finally:
        torch.set_num_threads(prev)
    for on, path in outs.items():
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        assert ckpt["incretin_at_glucose"] is on
        assert "metabolic.incretin_glu_gain" in ckpt["model_state"]
        incretin.set_incretin_at_glucose(not on)               # whatever a previous load left
        model, _ = load_model_from_checkpoint(path)
        assert incretin.INCRETIN_AT_GLUCOSE is on
        assert torch.equal(model.metabolic.incretin_glu_gain, ckpt["model_state"]["metabolic.incretin_glu_gain"])
    incretin.set_incretin_at_glucose(True)                     # an older checkpoint: off
    old = {k: v for k, v in torch.load(outs[False], weights_only=False).items() if k != "incretin_at_glucose"}
    torch.save(old, tmp_path / "old.pt")
    load_model_from_checkpoint(tmp_path / "old.pt")
    assert incretin.INCRETIN_AT_GLUCOSE is False


def test_gain_nonneg_clamps_after_every_step_and_zero_init_is_reachable():
    gain = torch.nn.Parameter(torch.zeros(()))
    other = torch.nn.Parameter(torch.ones(()))
    opt = torch.optim.Adam([gain, other], lr=0.1)
    incretin.hold_incretin_gain_nonneg(opt, gain)
    assert float(gain) == 0.0
    for _ in range(3):                                   # loss pushes the gain down
        opt.zero_grad()
        (gain * 5.0 + other).backward()
        opt.step()
        assert float(gain) == 0.0
    assert float(other) < 1.0                            # the hook leaves other parameters alone
    for _ in range(3):                                   # pushing up leaves 0 at once
        opt.zero_grad()
        (-gain * 5.0).backward()
        opt.step()
    assert float(gain) > 0.0
    with torch.no_grad():
        gain.fill_(-0.5)
    opt.zero_grad(); (gain * 0.0).backward(); opt.step()
    assert float(gain) >= 0.0


def test_gain_nonneg_flag_and_checkpoint(monkeypatch, tmp_path):
    import pulse.train as train_mod
    parser = train_mod.build_arg_parser()
    assert parser.parse_args([]).incretin_gain_nonneg is False
    assert parser.parse_args(["--incretin-gain-nonneg"]).incretin_gain_nonneg is True
    monkeypatch.setattr(train_mod, "_WATCHDOG_TIMEOUT_S", 0)
    prev = torch.get_num_threads()
    out = tmp_path / "m.pt"
    try:
        code = train_mod.train(
            n_patients=2, n_epochs=1, hidden_dim=8, windows_per_patient=1, n_days=1, seed=42,
            coupling_prior_weight=0.0, verifier_loss_weight=0.0, cohort_statistic_weight=0.0,
            setpoint_supervision_weight=0.5, embedding_prior_weight=0.001,
            contribution_weights={"full_body": 1.0}, output_path=str(out), scoreboard=False,
            incretin_at_glucose=True, incretin_gain_nonneg=True,
            train_only="metabolic.*incretin_glu_*",
        )
    finally:
        torch.set_num_threads(prev)
    assert code == 0
    ckpt = torch.load(out, map_location="cpu", weights_only=False)
    assert ckpt["incretin_gain_nonneg"] is True
    assert float(ckpt["model_state"]["metabolic.incretin_glu_gain"]) >= 0.0
