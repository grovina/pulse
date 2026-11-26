"""A fitness-dependent gain and a slow post-exercise state on the heart-rate path, both built
at zero: the model of before until a run trains them. Plus the corpus `vo2max` field."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pulse.knowledge.cohort_types import Population, factor_index
from pulse.knowledge.corpus import CorpusError, parse_finding
from pulse.knowledge.full_body import PatientParams, simulate_full_body
from pulse.model import ModularPhysiologyNetwork, integrate
from pulse.training import factors
from pulse.types import MARKER_INDEX, NORM_CENTER, NORM_SCALE

HR = MARKER_INDEX["hr"]
SLOW = MARKER_INDEX["hr_slow"]
FIXTURE = Path(__file__).parent / "fixtures" / "corpus" / "findings" / "synthetic-test-2099.json"

# Recorded from the code before the fitness gain and the slow state existed (origin/main at
# 5cd1fb0): the model below, seed 7, hr every 15 min and sbp every 30 min of a 180 min roll
# with activity 0.9 in minutes 20 to 69.
GOLDEN_HR = [70.0, 76.409, 75.683, 75.369, 75.977, 76.92, 78.044, 78.435, 78.306, 78.023, 77.731, 77.432]
GOLDEN_SBP = [120.0, 97.436, 96.168, 94.009, 93.868, 94.564]


def _model() -> ModularPhysiologyNetwork:
    torch.manual_seed(7)
    m = ModularPhysiologyNetwork()
    m.eval()
    with torch.no_grad():
        m.cardiovascular.network[-1].weight.normal_(0, 1.5)
        m.cardiovascular.network[-1].bias.normal_(0, 0.1)
    return m


def _golden_person() -> torch.Tensor:
    """The person the golden was recorded on: drawn right after the model's own draws."""
    torch.manual_seed(7)
    _model()
    return torch.randn(32) * 0.6


def _roll(m, emb, act, steps=180):
    init = torch.tensor(NORM_CENTER, dtype=torch.float32)
    with torch.no_grad():
        return integrate(m, init, emb, steps, activity=act, sleep_wake=torch.ones(steps))


def _bout(level=0.9, start=20, end=70, steps=180):
    act = torch.zeros(steps)
    act[start:end] = level
    return act


def _person(fitness_z=0.0) -> torch.Tensor:
    torch.manual_seed(7)
    emb = torch.randn(32) * 0.6
    emb[factor_index("fitness")] = fitness_z
    return emb


def test_zero_gains_reproduce_the_model_of_before():
    m = _model()
    assert float(m.cardiovascular.hr_fit_gain.detach()) == 0.0
    assert float(m.cardiovascular.hr_slow_gain.detach()) == 0.0
    emb = _golden_person()
    out = _roll(m, emb, _bout())
    assert out[::15, HR].tolist() == pytest.approx(GOLDEN_HR, abs=1e-3)
    assert out[::30, MARKER_INDEX["sbp"]].tolist() == pytest.approx(GOLDEN_SBP, abs=1e-3)


def test_the_slow_state_is_integrated_whether_or_not_anything_reads_it():
    m = _model()
    assert _roll(m, _person(), _bout())[:, SLOW].max() > 0.2


def test_the_slow_state_charges_in_a_bout_and_leaks_with_its_tau():
    m = _model()
    s = _roll(m, _person(), _bout(0.9, 20, 120, 400), 400)[:, SLOW]
    assert s[19] == 0.0
    assert 0.7 < float(s[119]) < 0.8            # (0.9 - 0.1) * (1 - exp(-100/30)) = 0.77
    # 30 min after the bout it has fallen to 1/e of where it was
    assert float(s[149]) == pytest.approx(float(s[119]) * np.exp(-1.0), rel=0.05)
    assert float(s[399]) < 0.001


def test_a_rest_level_activity_does_not_charge_it():
    m = _model()
    s = _roll(m, _person(), torch.full((180,), 0.05))[:, SLOW]
    assert float(s.max()) == 0.0


def test_a_positive_slow_gain_holds_heart_rate_up_after_the_bout():
    m = _model()
    base = _roll(m, _person(), _bout())[:, HR]
    with torch.no_grad():
        m.cardiovascular.hr_slow_gain.fill_(30.0)
    hr = _roll(m, _person(), _bout())[:, HR]
    assert float(hr[10]) == pytest.approx(float(base[10]), abs=1e-3)   # before the bout
    assert float(hr[69] - base[69]) > 8.0                                 # building during it
    tail = hr - base
    assert float(tail[100]) > 3.0 > 0.0                                   # 30 min after: still up
    assert float(tail[100]) < float(tail[69])
    assert float(tail[179]) < 1.0                                         # and gone by 110 min


def test_a_positive_fit_gain_lowers_heart_rate_of_the_fitter_person_at_the_same_work():
    m = _model()
    with torch.no_grad():
        m.cardiovascular.hr_fit_gain.fill_(12.0)
    low, mid, high = (_roll(m, _person(z), _bout())[:, HR] for z in (-1.5, 0.0, 1.5))
    assert float(high[69]) < float(mid[69]) < float(low[69])
    assert float(low[69] - high[69]) > 15.0       # 12 bpm per SD per unit activity, x 3 SD x 0.9, relaxed
    # at rest the gain is silent (fitness shows in the resting level, which the person's code carries)
    zero_gain = _model()
    for z in (-1.5, 1.5):
        r_gain = _roll(m, _person(z), torch.zeros(180))[:, HR]
        r_zero = _roll(zero_gain, _person(z), torch.zeros(180))[:, HR]
        assert torch.allclose(r_gain, r_zero, atol=1e-4)


def test_the_planned_and_the_stepwise_rollouts_agree_with_the_new_gains_set():
    m = _model()
    with torch.no_grad():
        m.cardiovascular.hr_fit_gain.fill_(12.0)
        m.cardiovascular.hr_slow_gain.fill_(30.0)
    init = torch.tensor(NORM_CENTER, dtype=torch.float32)
    emb = _person(-1.0)
    with torch.no_grad():
        a = integrate(m, init, emb, 120, activity=_bout(0.9, 20, 70, 120), sleep_wake=torch.ones(120), planned=True)
        b = integrate(m, init, emb, 120, activity=_bout(0.9, 20, 70, 120), sleep_wake=torch.ones(120), planned=False)
    assert torch.allclose(a, b, atol=1e-3)


# -- warm start from a checkpoint that predates them --------------------------------------

def _old_state(m) -> dict:
    """What a checkpoint written before these parameters holds: no gains and no tau, and the
    marker tables one marker shorter."""
    state = {k: v.clone() for k, v in m.state_dict().items()
             if not (k.startswith("cardiovascular.") and k.split(".")[1] in ("hr_fit_gain", "hr_slow_gain", "slow_tau_raw"))}
    state["norm_center"] = state["norm_center"][:-1]
    state["norm_scale"] = state["norm_scale"][:-1]
    return state


def test_a_checkpoint_without_the_new_parameters_loads_strictly_and_they_start_at_zero():
    trained = _model()
    fresh = ModularPhysiologyNetwork()
    fresh.load_model_state(_old_state(trained))
    cv = fresh.cardiovascular
    assert float(cv.hr_fit_gain) == 0.0 and float(cv.hr_slow_gain) == 0.0
    assert torch.equal(fresh.norm_center, torch.tensor(NORM_CENTER))
    assert torch.equal(cv.network[-1].weight, trained.cardiovascular.network[-1].weight)
    out = _roll(fresh, _golden_person(), _bout())
    assert out[::15, HR].tolist() == pytest.approx(GOLDEN_HR, abs=1e-3)


def test_the_warm_start_loader_keeps_every_trained_tensor_and_reports_only_the_new_ones_missing():
    from pulse.train import load_compatible_state_dict

    trained = _model()
    fresh = ModularPhysiologyNetwork()
    missing, unexpected, mismatched = load_compatible_state_dict(fresh, _old_state(trained))
    assert sorted(missing) == ["cardiovascular.hr_fit_gain", "cardiovascular.hr_slow_gain",
                               "cardiovascular.slow_tau_raw"]
    assert not unexpected and not mismatched


def test_a_checkpoint_with_them_round_trips():
    m = _model()
    with torch.no_grad():
        m.cardiovascular.hr_slow_gain.fill_(5.0)
    other = ModularPhysiologyNetwork()
    other.load_model_state(m.state_dict())
    assert float(other.cardiovascular.hr_slow_gain) == 5.0


# -- the corpus field and the fitness code -------------------------------------------------

def _data() -> dict:
    return copy.deepcopy(json.loads(FIXTURE.read_text(encoding="utf-8")))


def test_vo2max_is_read_like_age_and_reaches_the_arms_population():
    data = _data()
    data["population"]["vo2max"] = {"mean": 50.0, "sd": 5.0, "min": None, "max": None}
    pop = parse_finding(data).protocols["ogtt"].population
    assert pop.vo2max == (40.0, 60.0) and pop.name.endswith("_vo2" + "40-60")
    data["population"]["vo2max"] = {"mean": 50.0, "sd": 5.0, "min": 41.0, "max": 63.0}
    assert parse_finding(data).protocols["ogtt"].population.vo2max == (41.0, 63.0)
    data["protocols"][1]["population"] = {"vo2max": {"mean": 30.0, "sd": 3.0, "min": None, "max": None}}
    assert parse_finding(data).protocols["fasting"].population.vo2max == (24.0, 36.0)


def test_vo2max_in_l_per_min_or_with_nothing_stated_is_refused():
    data = _data()
    data["population"]["vo2max"] = {"mean": 3.4, "sd": 0.4, "min": None, "max": None}
    with pytest.raises(CorpusError, match="mL/kg/min"):
        parse_finding(data)
    data["population"]["vo2max"] = {"mean": None, "sd": None, "min": None, "max": None}
    with pytest.raises(CorpusError):
        parse_finding(data)


def test_a_population_without_vo2max_is_what_it_was():
    pop = parse_finding(_data()).protocols["ogtt"].population
    assert pop.vo2max is None and "_vo2" not in pop.name


def _draw(pop, seed=3, n=6):
    return factors.draw_factor_codes(n, np.random.default_rng(seed), "cpu", pop)


def test_the_fitness_code_ignores_vo2max_while_the_switch_is_off():
    assert factors.FITNESS_FROM_VO2MAX is False
    plain = Population(name="a", sex="male", age=(20.0, 30.0))
    fit = Population(name="b", sex="male", age=(20.0, 30.0), vo2max=(50.0, 56.0))
    assert torch.equal(_draw(plain), _draw(fit))


def test_the_fitness_code_follows_vo2max_when_on_and_moves_no_other_coordinate(monkeypatch):
    monkeypatch.setattr(factors, "FITNESS_FROM_VO2MAX", True)
    plain = Population(name="a", sex="male", age=(20.0, 30.0))
    fit = Population(name="b", sex="male", age=(20.0, 30.0), vo2max=(47.0, 56.0))
    a, b = _draw(plain), _draw(fit)
    col = factor_index("fitness")
    others = [i for i in range(a.shape[1]) if i != col]
    assert torch.equal(a[:, others], b[:, others])
    z = b[:, col]
    assert float(z.min()) >= (47.0 - factors.VO2MAX_CENTER) / factors.VO2MAX_SCALE - 1e-6
    assert float(z.max()) <= (56.0 - factors.VO2MAX_CENTER) / factors.VO2MAX_SCALE + 1e-6
    point = Population(name="c", vo2max=(38.0, 38.0))
    assert torch.allclose(_draw(point)[:, col], torch.zeros(6))


# -- the teacher ----------------------------------------------------------------------------

def _teacher_hr(**kw):
    p = PatientParams(**kw)
    act = np.zeros(240, dtype=np.float32)
    act[20:80] = 0.8
    traj, _ = simulate_full_body(p, [], np.ones(240, dtype=np.float32), act, 240, 8.0)
    return traj[:, MARKER_INDEX["hr"]], traj[:, SLOW]


def test_the_teacher_holds_heart_rate_up_for_tens_of_minutes_and_a_fitter_patient_runs_lower():
    hr, slow = _teacher_hr()
    hr0, _ = _teacher_hr(hr_slow_gain=0.0, hr_fit_gain=0.0)
    assert slow[19] == 0.0 and slow[79] > 0.5 and slow[159] < slow[79] * 0.25
    assert hr[110] - hr0[110] > 5.0           # 30 min after the bout
    assert abs(hr[239] - hr0[239]) < 2.0
    fit, _ = _teacher_hr(fitness_z=1.5)
    unfit, _ = _teacher_hr(fitness_z=-1.5)
    assert fit[79] < hr[79] < unfit[79]
    assert 15.0 < unfit[79] - fit[79] < 40.0   # papers at the same activity differ by 15 to 25 bpm


# -- the switch is a run setting: CLI -> checkpoint -> every loader ----------------------------

def _cohort_codes(pop, n=5):
    from pulse.scoreboard import cohort_people
    return cohort_people(n, 11)(pop)


def test_the_cli_flag_is_off_by_default_and_sets_the_run():
    from pulse.train import build_arg_parser, resolved_train_config
    parser = build_arg_parser()
    assert parser.parse_args([]).fitness_from_vo2max is False
    args = parser.parse_args(["--fitness-from-vo2max"])
    assert args.fitness_from_vo2max is True
    assert resolved_train_config(args)["fitness_from_vo2max"] is True


def test_a_checkpoint_carries_the_setting_and_loaders_apply_it(monkeypatch, tmp_path):
    from pulse.diagnostics.probe import load_model_from_checkpoint
    monkeypatch.setattr(factors, "FITNESS_FROM_VO2MAX", False)   # restored after the test
    m = _model()
    base = {"model_state": m.state_dict(), "model_config": m.constructor_kwargs}
    for recorded, want in ((True, True), (False, False)):
        torch.save({**base, "fitness_from_vo2max": recorded}, tmp_path / "c.pt")
        factors.set_fitness_from_vo2max(not want)               # whatever a previous load left
        load_model_from_checkpoint(tmp_path / "c.pt")
        assert factors.FITNESS_FROM_VO2MAX is want
    factors.set_fitness_from_vo2max(True)                       # an older checkpoint: off
    torch.save(base, tmp_path / "old.pt")
    load_model_from_checkpoint(tmp_path / "old.pt")
    assert factors.FITNESS_FROM_VO2MAX is False


def test_a_trained_checkpoint_records_the_flag_and_the_board_people_follow_it(monkeypatch, tmp_path):
    import pulse.train as train_mod
    from pulse.scoreboard import scoreboard_for_checkpoint  # noqa: F401  (the loader path)
    from pulse.diagnostics.probe import load_model_from_checkpoint
    monkeypatch.setattr(factors, "FITNESS_FROM_VO2MAX", False)
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
                fitness_from_vo2max=on,
            )
            assert code == 0
            outs[on] = out
    finally:
        torch.set_num_threads(prev)
    fit = Population(name="b", sex="male", age=(20.0, 30.0), vo2max=(47.0, 56.0))
    col = factor_index("fitness")
    people = {}
    for on, path in outs.items():
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        assert ckpt["fitness_from_vo2max"] is on
        factors.set_fitness_from_vo2max(not on)
        load_model_from_checkpoint(path)                        # what scoreboard_for_checkpoint does
        people[on] = _cohort_codes(fit)
    others = [i for i in range(people[True].shape[1]) if i != col]
    assert torch.equal(people[False][:, others], people[True][:, others])
    assert not torch.equal(people[False][:, col], people[True][:, col])
    z = people[True][:, col]
    assert float(z.min()) >= (47.0 - 38.0) / 9.0 - 1e-6 and float(z.max()) <= (56.0 - 38.0) / 9.0 + 1e-6


def test_default_off_people_are_unchanged_for_a_population_with_vo2max(monkeypatch):
    monkeypatch.setattr(factors, "FITNESS_FROM_VO2MAX", False)
    fit = Population(name="b", sex="male", age=(20.0, 30.0), vo2max=(47.0, 56.0))
    plain = Population(name="b", sex="male", age=(20.0, 30.0))
    assert torch.equal(_cohort_codes(fit), _cohort_codes(plain))
