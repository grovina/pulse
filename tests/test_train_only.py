"""--train-only: train a named subset of parameters, keep every other one bit-identical."""

from __future__ import annotations

import pytest
import torch

import pulse.train as train_mod
from pulse.model import ModularPhysiologyNetwork
from pulse.train import _freeze_all_but, _parse_train_only, build_arg_parser, resolved_train_config


def _tiny(**kw):
    return ModularPhysiologyNetwork(
        metabolic_hidden=8, appetite_hidden=24, stress_hidden=24,
        cardiovascular_hidden=8, thermoreg_hidden=16, respiratory_hidden=16, **kw,
    )


def _run(tmp_path, name, init_from, **kw):
    out = tmp_path / name
    code = train_mod.train(
        n_patients=2, n_epochs=2, hidden_dim=8, windows_per_patient=1, n_days=1, seed=42,
        coupling_prior_weight=0.0, verifier_loss_weight=0.0, cohort_statistic_weight=0.0,
        setpoint_supervision_weight=0.5, embedding_prior_weight=0.001,
        contribution_weights={"full_body": 1.0}, output_path=str(out), scoreboard=False,
        init_from=str(init_from), **kw,
    )
    assert code == 0
    return torch.load(out, map_location="cpu", weights_only=False)


@pytest.fixture
def warm(tmp_path, monkeypatch):
    monkeypatch.setattr(train_mod, "_WATCHDOG_TIMEOUT_S", 0)
    prev = torch.get_num_threads()
    torch.manual_seed(0)
    m = _tiny()
    for p in m.parameters():  # a non-trivial warm start, so "unchanged" is not "still zero"
        p.data.add_(0.01 * torch.randn_like(p))
    path = tmp_path / "warm.pt"
    torch.save({"model_state": m.state_dict()}, path)
    yield path
    torch.set_num_threads(prev)


def test_patterns_parse_and_default_is_everything() -> None:
    assert _parse_train_only(None) == () and _parse_train_only(" a.* , b ") == ("a.*", "b")
    m = _tiny()
    assert len(_freeze_all_but(m, ())) == len(list(m.parameters()))
    assert all(p.requires_grad for p in m.parameters())
    assert build_arg_parser().parse_args([]).train_only is None
    assert resolved_train_config(build_arg_parser().parse_args(["--train-only", "x"]))["train_only"] == "x"


def test_freeze_keeps_only_the_matches() -> None:
    m = _tiny()
    got = _freeze_all_but(m, ("cardiovascular.hr_slow_gain", "cardiovascular.slow_tau_raw"))
    assert len(got) == 2
    on = {n for n, p in m.named_parameters() if p.requires_grad}
    assert on == {"cardiovascular.hr_slow_gain", "cardiovascular.slow_tau_raw"}


def test_a_pattern_that_matches_nothing_fails_fast() -> None:
    with pytest.raises(ValueError, match="no_such"):
        _freeze_all_but(_tiny(), ("cardiovascular.hr_slow_gain", "no_such.*"))


def test_train_only_leaves_the_rest_bit_identical(tmp_path, warm) -> None:
    pats = "metabolic.log_k_*,metabolic.log_si"  # the teacher tape reaches these (not the heart-rate gains)
    before = torch.load(warm, map_location="cpu", weights_only=False)["model_state"]
    ck = _run(tmp_path, "only.pt", warm, train_only=pats)
    after = ck["model_state"]
    assert ck["train_only"] == ["metabolic.log_k_*", "metabolic.log_si"]
    moved = {k for k in before if not torch.equal(before[k], after[k])}
    assert moved, "nothing trained moved"
    assert all(k.startswith("metabolic.log_k_") or k == "metabolic.log_si" for k in moved), moved


def test_default_trains_everything(tmp_path, warm) -> None:
    before = torch.load(warm, map_location="cpu", weights_only=False)["model_state"]
    ck = _run(tmp_path, "all.pt", warm)
    assert ck["train_only"] is None
    moved = {k.split(".")[0] for k in before if not torch.equal(before[k], ck["model_state"][k])}
    assert len(moved) > 3


def test_a_loss_with_no_grad_path_is_skipped_not_an_error(tmp_path, warm) -> None:
    before = torch.load(warm, map_location="cpu", weights_only=False)["model_state"]
    ck = _run(tmp_path, "nograd.pt", warm, train_only="cardiovascular.hr_*")  # the tape never reaches them
    assert all(torch.equal(before[k], v) for k, v in ck["model_state"].items())


def test_lr_multipliers_parse_and_validate() -> None:
    from pulse.train import _lr_multipliers, _parse_train_only_lr

    assert _parse_train_only_lr(None) == ()
    assert _parse_train_only_lr(" a.*=3 , b=0.5 ") == (("a.*", 3.0), ("b", 0.5))
    for bad in ("a", "=3", "a=0", "a=-1", "a=x", "a=inf"):
        with pytest.raises(ValueError, match="train-only-lr"):
            _parse_train_only_lr(bad)
    m = _tiny()
    params = _freeze_all_but(m, ("metabolic.log_k_*", "metabolic.log_si"))
    got = _lr_multipliers(m, params, (("metabolic.log_si", 3.0),))
    assert sorted(set(got)) == [1.0, 3.0] and got.count(3.0) == 1
    with pytest.raises(ValueError, match="no trained parameter"):
        _lr_multipliers(m, params, (("cardiovascular.*", 2.0),))
    with pytest.raises(ValueError, match="keep the phase lr"):
        _lr_multipliers(m, params, (("metabolic.*", 2.0),))


def test_lr_multipliers_ride_the_cosine_floor_included() -> None:
    from pulse.train import _apply_lr_mults, _make_adam

    a, b, c = (torch.nn.Parameter(torch.zeros(())) for _ in range(3))
    opt = _make_adam([a, b, c], 0.01, [1.0, 3.0, 0.5])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=4, eta_min=0.002)
    _apply_lr_mults(opt)
    seen = []
    for _ in range(5):
        seen.append({g["lr_mult"]: g["lr"] for g in opt.param_groups})
        sched.step()
        _apply_lr_mults(opt)
    for lrs in seen:
        assert lrs[3.0] == pytest.approx(3.0 * lrs[1.0]) and lrs[0.5] == pytest.approx(0.5 * lrs[1.0])
    assert seen[0][1.0] == pytest.approx(0.01) and seen[0][3.0] == pytest.approx(0.03)
    assert seen[4][1.0] == pytest.approx(0.002, abs=1e-9) and seen[4][3.0] == pytest.approx(0.006, abs=1e-8)
    # no multipliers: the plain optimizer, one group
    assert len(_make_adam([a], 0.01, None).param_groups) == 1


def test_train_only_lr_end_to_end(tmp_path, warm) -> None:
    pats = "metabolic.log_k_*,metabolic.log_si"
    before = torch.load(warm, map_location="cpu", weights_only=False)["model_state"]
    ck = _run(tmp_path, "mult.pt", warm, train_only=pats, train_only_lr="metabolic.log_si=3")
    assert ck["train_only_lr"] == "metabolic.log_si=3"
    moved = {k for k in before if not torch.equal(before[k], ck["model_state"][k])}
    assert moved and all(k.startswith("metabolic.log_k_") or k == "metabolic.log_si" for k in moved)
    assert build_arg_parser().parse_args([]).train_only_lr is None
    with pytest.raises(ValueError, match="needs --train-only"):
        _run(tmp_path, "bad.pt", warm, train_only_lr="metabolic.log_si=3")
