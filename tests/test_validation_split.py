"""The run's own validation split (``train --val-fraction``): drawn from the trained
papers only, deterministic, off by default, and the best-scored epoch is the one kept."""

from __future__ import annotations

import torch
import torch.nn as nn

from pulse.knowledge import ALL_COHORT_STATISTICS
from pulse.knowledge.physiology_rules import PHYSIOLOGY_RULES
from pulse.knowledge.sealing import paper_is_sealed, paper_is_validation
from pulse.train import BestOnValidation, build_arg_parser, split_validation


def test_the_split_never_includes_a_sealed_paper():
    papers = {s.paper for s in ALL_COHORT_STATISTICS} | {r.paper for r in PHYSIOLOGY_RULES}
    for fraction in (0.1, 0.5, 1.0):
        assert not any(paper_is_sealed(p) and paper_is_validation(p, fraction) for p in papers)
    # slugs the seal takes are refused however large the fraction
    sealed = [f"sealed-probe-{i}" for i in range(200) if paper_is_sealed(f"sealed-probe-{i}")]
    assert sealed and not any(paper_is_validation(p, 1.0) for p in sealed)
    _train, val, _rules = split_validation(ALL_COHORT_STATISTICS, PHYSIOLOGY_RULES, 0.1)
    assert val and not any(paper_is_sealed(s.paper) for s in val)


def test_the_split_is_deterministic_and_about_the_fraction():
    papers = sorted({s.paper for s in ALL_COHORT_STATISTICS})
    a = [p for p in papers if paper_is_validation(p, 0.1)]
    assert a == [p for p in papers if paper_is_validation(p, 0.1)]
    assert 0.03 < len(a) / len(papers) < 0.2
    assert set(a) <= {p for p in papers if paper_is_validation(p, 0.3)}  # more only adds


def test_it_is_not_the_seal_draw():
    slugs = [f"probe-{i}" for i in range(400)]
    unsealed = [p for p in slugs if not paper_is_sealed(p)]
    share = sum(paper_is_validation(p, 0.1) for p in unsealed) / len(unsealed)
    assert 0.05 < share < 0.16


def test_the_validation_papers_are_out_of_the_training_pool():
    train, val, rules = split_validation(ALL_COHORT_STATISTICS, PHYSIOLOGY_RULES, 0.1)
    assert len(train) + len(val) == len(ALL_COHORT_STATISTICS)
    assert not {s.paper for s in train} & {s.paper for s in val}
    assert not any(paper_is_validation(r.paper, 0.1) for r in rules)


def test_fraction_zero_changes_nothing():
    train, val, rules = split_validation(ALL_COHORT_STATISTICS, PHYSIOLOGY_RULES, 0.0)
    assert train == list(ALL_COHORT_STATISTICS) and val == [] and rules == list(PHYSIOLOGY_RULES)
    args = build_arg_parser().parse_args([])
    assert args.val_fraction == 0.0 and args.val_every == 1


def test_selection_keeps_the_best_scored_checkpoint():
    model = nn.Linear(1, 1)
    keeper = BestOnValidation()
    scores = [0.9, 0.7, 0.6, 0.8, 0.65]  # the mocked validation scores, one per epoch
    for epoch, score in enumerate(scores):
        with torch.no_grad():
            model.weight.fill_(float(epoch))
        keeper.consider(epoch, score, model)
    assert keeper.best_epoch == 2 and keeper.best_score == 0.6
    assert float(keeper.best_state["weight"]) == 2.0  # a copy, not the live last weights
    assert [c["median_abs_z"] for c in keeper.curve] == scores
    report = keeper.report(0.1, 1, ["a-1"], 3, last_epoch=4)
    assert report["kept"] == "best" and report["best_epoch"] == 2


def test_a_tie_keeps_the_earlier_epoch():
    model = nn.Linear(1, 1)
    keeper = BestOnValidation()
    assert keeper.consider(0, 0.5, model) and not keeper.consider(1, 0.5, model)
    assert keeper.best_epoch == 0


def test_train_keeps_the_best_epoch_as_model_and_the_last_as_model_last(tmp_path, monkeypatch):
    import pulse.train as train_mod

    monkeypatch.setattr(train_mod, "_WATCHDOG_TIMEOUT_S", 0)
    scores = iter([0.5, 0.9])  # epoch 0 validates best; epoch 1 is worse
    monkeypatch.setattr(train_mod, "validation_median_abs_z", lambda *a, **k: next(scores))
    out = tmp_path / "model.pt"
    prev = torch.get_num_threads()
    try:
        code = train_mod.train(
            n_patients=2, n_epochs=2, hidden_dim=8, windows_per_patient=1, n_days=1, seed=42,
            coupling_prior_weight=0.0, verifier_loss_weight=0.0, cohort_statistic_weight=0.0,
            setpoint_supervision_weight=0.5, embedding_prior_weight=0.001,
            contribution_weights={"full_body": 1.0}, output_path=str(out), scoreboard=False,
            val_fraction=0.1,
        )
    finally:
        torch.set_num_threads(prev)
    assert code == 0
    best = torch.load(out, map_location="cpu", weights_only=False)
    last = torch.load(tmp_path / "model_last.pt", map_location="cpu", weights_only=False)
    assert best["validation"]["best_epoch"] == 0 and best["validation"]["kept"] == "best"
    assert [c["median_abs_z"] for c in best["validation"]["curve"]] == [0.5, 0.9]
    assert any(not torch.equal(best["model_state"][k], last["model_state"][k]) for k in best["model_state"])
