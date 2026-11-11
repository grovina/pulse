"""A run that dies leaves a board, and a run near its deadline stops cleanly."""

from __future__ import annotations

import json
import sys

import pytest
import torch

from pulse import train

_TINY = [
    "--n-patients=2", "--n-days=1", "--windows-per-patient=1", "--hidden-dim=8",
    "--steps-per-epoch=1", "--cohort-statistic-weight=0", "--physiology-rules-weight=0",
    "--dose-response-weight=0", "--carb-mass-balance-weight=0",
]


def _run(monkeypatch, tmp_path, *flags):
    """``main()`` on a tiny recipe with a stub board and upload; returns (board calls, uploads)."""
    calls, uploads = [], []

    def stub_board(model, *, previous, identifiability):
        calls.append(identifiability)
        return {"identifiability": identifiability}

    monkeypatch.setattr(train, "score_on_board", stub_board)
    monkeypatch.setattr(train, "_upload_to_gcs", lambda local, bucket, obj: uploads.append(obj))
    out = tmp_path / "m.pt"
    monkeypatch.setattr(sys, "argv", [
        "train", *_TINY, f"--output-path={out}", "--gcs-bucket=b", "--gcs-object=run/m.pt", *flags,
    ])
    train.main()
    return calls, uploads, out


def test_each_phase_end_leaves_a_board_without_the_identifiability_pass(monkeypatch, tmp_path) -> None:
    calls, uploads, out = _run(
        monkeypatch, tmp_path,
        "--phase1-epochs=1", "--phase2-epochs=1", "--phase3-epochs=1", "--n-epochs=3",
    )
    # Phases 1 and 2 end mid-run; the end-of-run board is the full one.
    assert calls == [False, False, True]
    for phase in (1, 2):
        board = json.loads((tmp_path / f"m.phase{phase}.scoreboard.json").read_text())
        assert board == {"identifiability": False}
        assert f"run/m.phase{phase}.scoreboard.json" in uploads
    assert (tmp_path / "m.scoreboard.json").exists() and "run/m.scoreboard.json" in uploads


def test_the_deadline_stops_training_and_still_scores_and_uploads(monkeypatch, tmp_path) -> None:
    calls, uploads, out = _run(
        monkeypatch, tmp_path,
        "--phase1-epochs=2", "--phase2-epochs=2", "--n-epochs=4", "--deadline-hours=0.000001",
    )
    checkpoint = torch.load(out, weights_only=False)
    assert checkpoint["stopped_for_deadline"] is True
    # Stopped inside phase 1 after one epoch: no phase board, then the normal end of run.
    assert calls == [True]
    assert "run/m.pt" in uploads and "run/m.scoreboard.json" in uploads
    assert not list(tmp_path.glob("m.phase*.scoreboard.json"))


def test_no_deadline_runs_every_epoch(monkeypatch, tmp_path) -> None:
    _calls, _uploads, out = _run(
        monkeypatch, tmp_path, "--phase1-epochs=1", "--phase2-epochs=1", "--n-epochs=2",
    )
    assert torch.load(out, weights_only=False)["stopped_for_deadline"] is False
