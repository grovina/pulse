"""Plan A8: one default person, and checkpoints that can be re-scored and warm-started.

Three small things that were each a different person or a different ruler:

* the server predicted a brand-new user as a RANDOM person (N(0, 0.1^2) seeded from a hash of the
  user id) while ``benchmark.py`` had used the trained prior mean since iter 97 -- and once that
  was fixed the two STILL disagreed on the no-prior fallback, the server returning zeros and the
  benchmark its seeded draw, so a checkpoint carrying no prior was scored against one person and
  served as another. Both now call ``model.population_prior_embedding``;
* ``last_good.pt`` held the patient table but not ``embedding_prior_{mean,std}``, so a rolling
  checkpoint was re-scored on the isotropic-L2 fallback, not the prior the final artifact is judged on;
* ``--init-from`` loaded the decoders and left the 40 deterministic seed-42 patients on a fresh
  N(0, 0.1) table, so the heads were warm and the codes cold.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from pulse import server
from pulse import train as train_mod
from pulse.model import ModularPhysiologyNetwork, population_prior_embedding
from pulse.train import (
    _embedding_prior_stats,
    _last_good_checkpoint,
    _warm_start_patient_table,
    train,
)
from pulse.types import EMBEDDING_DIM, MARKER_IDS


def _tiny_model() -> ModularPhysiologyNetwork:
    torch.manual_seed(0)
    m = ModularPhysiologyNetwork(
        metabolic_hidden=8, appetite_hidden=8, stress_hidden=8,
        cardiovascular_hidden=8, thermoreg_hidden=8, respiratory_hidden=8,
    )
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _with_prior(model: ModularPhysiologyNetwork) -> torch.Tensor:
    g = torch.Generator().manual_seed(5)
    prior = torch.randn(model.embedding_dim, generator=g) * 0.1
    model._embedding_prior_mean = prior
    model._embedding_prior_std = torch.full((model.embedding_dim,), 0.15)
    return prior


# --- 1. the server's default person -----------------------------------------------------------

def test_new_user_gets_the_prior_mean_not_a_random_vector() -> None:
    model = _tiny_model()
    prior = _with_prior(model)
    out = server.get_initial_embedding(model, None)
    assert torch.equal(out, prior)
    # A copy: calibration and the response serialiser must not be able to edit the model's prior.
    out.add_(1.0)
    assert not torch.equal(model._embedding_prior_mean, out)
    assert torch.equal(model._embedding_prior_mean, prior)


def test_no_prior_means_zeros_the_default_patient() -> None:
    model = _tiny_model()
    assert getattr(model, "_embedding_prior_mean", None) is None
    out = server.get_initial_embedding(model, None)
    assert out.shape == (model.embedding_dim,) and out.dtype == torch.float32
    assert torch.equal(out, torch.zeros(model.embedding_dim))


def test_default_person_draws_nothing_from_any_rng() -> None:
    # "Never a random vector": no RNG is consumed, and the answer cannot depend on one.
    for with_prior in (True, False):
        model = _tiny_model()
        if with_prior:
            _with_prior(model)
        torch.manual_seed(1)
        np.random.seed(1)
        t_state, n_state = torch.get_rng_state(), np.random.get_state()[1].copy()
        a = server.get_initial_embedding(model, None)
        assert torch.equal(torch.get_rng_state(), t_state)
        assert np.array_equal(np.random.get_state()[1], n_state)
        torch.manual_seed(2)
        assert torch.equal(a, server.get_initial_embedding(model, None))


def test_client_embedding_wins_only_at_the_models_width() -> None:
    model = _tiny_model()
    prior = _with_prior(model)
    mine = [0.01 * i for i in range(model.embedding_dim)]
    assert torch.allclose(server.get_initial_embedding(model, mine), torch.tensor(mine))
    # A stale-width embedding (e.g. from before EMBEDDING_DIM 64 -> 32) is not a person here.
    assert torch.equal(server.get_initial_embedding(model, mine[:-1]), prior)
    assert torch.equal(server.get_initial_embedding(model, []), prior)


def test_the_random_seeding_is_gone() -> None:
    assert not hasattr(server, "seeded_embedding") and not hasattr(server, "seed_from_user_id")
    # The benchmark's own seeded fallback, which outlived iter 97's fix to its main path.
    from pulse import benchmark
    assert not hasattr(benchmark, "deterministic_user_embedding")


def test_server_and_benchmark_cannot_disagree_about_the_default_person() -> None:
    """The invariant the shared helper exists for, checked on BOTH branches.

    A8's claim is "one default person, consistently". That held for a checkpoint with a
    prior and failed without one: the server returned zeros and the benchmark a
    user-id-seeded draw. A per-call-site reimplementation is how that happened, so this
    asserts agreement rather than asserting either value.
    """
    for with_prior in (True, False):
        model = _tiny_model()
        if with_prior:
            _with_prior(model)
        shared = population_prior_embedding(model)
        assert torch.equal(server.get_initial_embedding(model, None), shared)
        assert shared.shape == (model.embedding_dim,) and shared.dtype == torch.float32
        # It is ONE person, not a draw: same answer every call, and a copy each time.
        again = population_prior_embedding(model)
        assert torch.equal(again, shared) and again is not shared
        again.add_(1.0)
        assert torch.equal(population_prior_embedding(model), shared)


def test_no_call_site_rebuilds_the_default_person_itself() -> None:
    """A LONE read of the prior mean is a hand-rolled default person.

    Both modules legitimately read ``_embedding_prior_mean``, but always TOGETHER with
    ``_embedding_prior_std``: that pair is the calibration regulariser, a different thing
    from the person a prediction starts at. Reading the mean on its own is what the two
    deleted implementations did, so that is what this forbids -- the duplication, not the
    attribute. (Checked as source text because the drift was two code paths agreeing on
    the prior branch and differing on the fallback, which no single call can observe.)
    """
    import pulse.benchmark
    for mod in (server, pulse.benchmark):
        lines = Path(mod.__file__).read_text().splitlines()
        for i, line in enumerate(lines):
            if "_embedding_prior_mean" not in line:
                continue
            near = "\n".join(lines[max(0, i - 2):i + 3])
            assert "_embedding_prior_std" in near, (
                f"{Path(mod.__file__).name}:{i + 1} reads the prior MEAN alone:\n"
                f"{line.strip()}\ncall population_prior_embedding() instead of rebuilding "
                "the default person -- that duplication is what A8 removed"
            )


def test_simulate_predicts_every_new_user_as_the_same_median_person(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _tiny_model()
    prior = _with_prior(model)
    monkeypatch.setattr(server, "_MODEL", model)
    monkeypatch.setattr(
        server, "_MODEL_META",
        server.LoadedModel(hidden_dim=8, marker_ids=list(MARKER_IDS), model_version="test"),
    )

    def run(user_id: str, **kw) -> dict:
        return server.simulate(server.SimulateRequest(
            user_id=user_id, duration_min=60, sample_interval=30, **kw))

    a, b = run("alice"), run("bob")
    assert a["embedding"] == b["embedding"] == prior.tolist()
    assert a["series"] == b["series"]                       # same person, same prediction
    assert a["calibration"]["reason"] == "disabled"
    mine = [0.02] * model.embedding_dim
    assert run("carol", embedding=mine)["embedding"] == pytest.approx(mine)


# --- 2. last_good.pt carries the prior --------------------------------------------------------

def _table(n: int = 5, offset: float = 0.3, seed: int = 1) -> nn.Embedding:
    emb = nn.Embedding(n, EMBEDDING_DIM)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        emb.weight.copy_(torch.randn(n, EMBEDDING_DIM, generator=g) * 0.2 + offset)
    return emb


def test_prior_stats_are_the_tables_mean_and_population_std() -> None:
    emb = _table()
    mean, std = _embedding_prior_stats(emb)
    w = emb.weight.detach().numpy()
    assert len(mean) == len(std) == EMBEDDING_DIM
    np.testing.assert_allclose(mean, w.mean(axis=0), atol=1e-6)
    np.testing.assert_allclose(std, w.std(axis=0, ddof=0), atol=1e-6)


def test_last_good_round_trips_with_a_usable_prior(tmp_path: Path) -> None:
    model, emb = _tiny_model(), _table()
    ck = _last_good_checkpoint(
        model, emb, hidden_dim=8, epoch=3, phase=2, metrics={"epoch": 3.0}, seed=42,
    )
    path = tmp_path / "pulse-model.pt.last-good.pt"
    torch.save(ck, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)

    rebuilt = ModularPhysiologyNetwork.from_checkpoint(loaded)
    assert getattr(rebuilt, "_embedding_prior_mean", None) is not None
    assert getattr(rebuilt, "_embedding_prior_std", None) is not None
    w = emb.weight.detach()
    assert torch.allclose(rebuilt._embedding_prior_mean, w.mean(dim=0), atol=1e-6)
    assert torch.allclose(rebuilt._embedding_prior_std, w.std(dim=0, unbiased=False), atol=1e-6)
    # ...and it is still a --resume-from checkpoint: table, epoch, phase, metrics untouched.
    assert torch.equal(loaded["embeddings_state"]["weight"], w)
    assert (loaded["epoch"], loaded["phase"], loaded["metrics"]) == (3, 2, {"epoch": 3.0})
    assert loaded["seed"] == 42                         # which people the rows are, for --init-from
    assert loaded["embedding_dim"] == EMBEDDING_DIM and loaded["hidden_dim"] == 8


def test_prior_in_last_good_is_the_current_table_not_a_stale_one() -> None:
    model, emb = _tiny_model(), _table()
    first = _last_good_checkpoint(model, emb, hidden_dim=8, epoch=0, phase=1, metrics={}, seed=42)
    with torch.no_grad():
        emb.weight.add_(1.0)
    second = _last_good_checkpoint(model, emb, hidden_dim=8, epoch=1, phase=1, metrics={}, seed=42)
    delta = np.array(second["embedding_prior_mean"]) - np.array(first["embedding_prior_mean"])
    np.testing.assert_allclose(delta, 1.0, atol=1e-6)
    np.testing.assert_allclose(second["embedding_prior_std"], first["embedding_prior_std"], atol=1e-6)


# --- 3. --init-from restores the patient table ------------------------------------------------

def _fresh(n: int = 4) -> nn.Embedding:
    emb = nn.Embedding(n, EMBEDDING_DIM)
    nn.init.normal_(emb.weight, std=0.1)
    return emb


def test_init_from_restores_a_saved_table(capsys: pytest.CaptureFixture[str]) -> None:
    saved = _table(n=4)
    target = _fresh(4)
    assert not torch.allclose(target.weight, saved.weight)
    ok = _warm_start_patient_table(
        target, {"embeddings_state": saved.state_dict(), "seed": 42}, "init.pt", seed=42)
    assert ok is True
    assert torch.equal(target.weight, saved.weight)
    out = capsys.readouterr().out
    assert "[INIT]" in out and "restored" in out and "init.pt" in out


@pytest.mark.parametrize("label,ckpt,needle", [
    ("absent", {}, "no embeddings_state"),
    ("not a table", {"embeddings_state": {"weight": None}}, "no embeddings_state"),
    ("rows differ", {"embeddings_state": _table(n=6).state_dict()}, "(6, 32)"),
    ("width differs", {"embeddings_state": nn.Embedding(4, EMBEDDING_DIM // 2).state_dict()}, "(4, 16)"),
    ("other seed", {"embeddings_state": _table(n=4).state_dict(), "seed": 7}, "seed 7"),
])
def test_init_from_falls_back_to_the_fresh_table_and_says_why(
    capsys: pytest.CaptureFixture[str], label: str, ckpt: dict, needle: str,
) -> None:
    target = _fresh(4)
    before = target.weight.detach().clone()
    ok = _warm_start_patient_table(target, ckpt, "init.pt", seed=42)
    assert ok is False, label
    assert torch.equal(target.weight, before), label
    out = capsys.readouterr().out
    assert "[INIT]" in out and "fresh N(0, 0.1) init" in out and needle in out, (label, out)


def test_init_from_without_a_recorded_seed_still_restores() -> None:
    # A hand-built init.pt (or a last_good.pt from before A8) records no seed: the shape is all
    # there is to check, and the table is loaded.
    saved, target = _table(n=4), _fresh(4)
    assert _warm_start_patient_table(
        target, {"embeddings_state": saved.state_dict()}, "last_good.pt", seed=42)
    assert torch.equal(target.weight, saved.weight)


# --- the three, through the real train() ------------------------------------------------------

def _tiny_train(out: Path, **kw) -> int:
    prev_threads = torch.get_num_threads()      # train() pins the intra-op pool to 1
    try:
        return train(
            n_patients=2, n_epochs=1, hidden_dim=8, windows_per_patient=1, n_days=1, seed=42,
            coupling_prior_weight=0.0, verifier_loss_weight=0.0, cohort_statistic_weight=0.0,
            setpoint_supervision_weight=0.5, embedding_prior_weight=0.001,
            contribution_weights={"full_body": 1.0}, output_path=str(out), **kw,
        )
    finally:
        torch.set_num_threads(prev_threads)


def test_train_writes_the_prior_in_last_good_and_warm_starts_from_the_artifact(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The epoch watchdog arms faulthandler on sys.stderr, which capsys replaces with a stream that
    # has no fileno; it is a hang detector, not part of what is under test.
    monkeypatch.setattr(train_mod, "_WATCHDOG_TIMEOUT_S", 0)
    run1 = tmp_path / "run1.pt"
    assert _tiny_train(run1) == 0
    out1 = capsys.readouterr().out
    assert "[EMBPRIOR]" in out1 and "center_weight=1.0" in out1     # the pin is visible in the log
    last_good = torch.load(f"{run1}.last-good.pt", map_location="cpu", weights_only=False)
    final = torch.load(run1, map_location="cpu", weights_only=False)

    # The rolling checkpoint re-scores on its own prior: stats of ITS table, same as the artifact's.
    table = last_good["embeddings_state"]["weight"].numpy()
    np.testing.assert_allclose(last_good["embedding_prior_mean"], table.mean(axis=0), atol=1e-6)
    np.testing.assert_allclose(last_good["embedding_prior_std"], table.std(axis=0), atol=1e-6)
    assert last_good["seed"] == final["seed"] == 42
    assert last_good["embedding_prior_mean"] == final["embedding_prior_mean"]
    assert last_good["embedding_prior_std"] == final["embedding_prior_std"]
    rebuilt = ModularPhysiologyNetwork.from_checkpoint(last_good)
    assert rebuilt._embedding_prior_mean.shape == (EMBEDDING_DIM,)

    # The final artifact carries the table too, or --init-from has nothing to read.
    assert torch.equal(final["embeddings_state"]["weight"], last_good["embeddings_state"]["weight"])
    assert final["embedding_prior_center_weight"] == 1.0

    # --init-from: plant a table a fresh N(0, 0.1) draw could not be mistaken for, and check it comes back.
    planted = torch.randn(2, EMBEDDING_DIM, generator=torch.Generator().manual_seed(9)) * 0.4
    final["embeddings_state"] = {"weight": planted}
    init = tmp_path / "init.pt"
    torch.save(final, init)
    assert _tiny_train(tmp_path / "run2.pt", init_from=str(init), lr=1e-6) == 0
    assert "patient table restored" in capsys.readouterr().out
    run2 = torch.load(tmp_path / "run2.pt", map_location="cpu", weights_only=False)
    assert torch.allclose(run2["embeddings_state"]["weight"], planted, atol=1e-3)
