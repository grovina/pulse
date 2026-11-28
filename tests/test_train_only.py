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
