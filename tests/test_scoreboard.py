"""The PLAN section 4 board: item residuals, the three lines, conflicts and breaks."""

from __future__ import annotations

import json

import pytest
import torch

from pulse.knowledge.cohort_types import (
    CohortArmSpec, CohortStatisticSpec, InitMode, StatisticKind, StatisticWindow, TargetShape,
)
from pulse.knowledge.physiology_rules import PHYSIOLOGY_RULES
from pulse.model import ModularPhysiologyNetwork
from pulse.scoreboard import (
    ItemResidual,
    attach_individual_skill,
    breaks,
    fit_summary,
    population_members,
    render,
    scoreboard,
    scoreboard_path,
    target_conflicts,
)
from pulse.types import EMBEDDING_DIM

_ARM = CohortArmSpec(label="rest", duration_min=60, start_hour=8.0, meals=())
_MEAL = CohortArmSpec(label="meal", duration_min=90, start_hour=8.0, meals=((0.0, 50.0, 0.0, 0.0),))


def _spec(name: str, *, target: float = 95.0, sigma: float = 10.0, arm=_ARM, sealed=False, **kw):
    return CohortStatisticSpec(
        name=name, source="test", description="", arms=(arm,), marker_id=kw.pop("marker_id", "glucose"),
        kind=StatisticKind.MEAN_IN_WINDOW, window=StatisticWindow(30, 60), target=target,
        sigma=sigma, init_mode=InitMode.NORM_CENTER, sealed=sealed, **kw,
    )


def _row(name: str, z: float, family: str = "resting", sealed: bool = True) -> ItemResidual:
    return ItemResidual(
        name=name, source="", family=family, operator="window_mean", marker_id="glucose",
        sealed=sealed, predicted=0.0, target=0.0, sigma=1.0, z=z,
    )


def test_the_population_is_the_draws_then_the_median_person():
    members, n = population_members(0, seed=0)
    assert n == 0 and len(members) == 1 and torch.equal(members[0], torch.zeros(EMBEDDING_DIM))
    members, n = population_members(4, seed=0)
    assert n == 4 and len(members) == 5
    assert torch.equal(members[-1], torch.zeros(EMBEDDING_DIM))
    again, _ = population_members(4, seed=0)
    assert all(torch.equal(a, b) for a, b in zip(members, again))
    other, _ = population_members(4, seed=1)
    assert not all(torch.equal(a, b) for a, b in zip(members, other))


def test_fit_summary_reads_in_units_of_each_items_sigma():
    rows = [_row("a", 0.5), _row("b", -1.5), _row("c", 3.0, family="clamp"), _row("d", 0.0)]
    s = fit_summary(rows)
    assert s["n"] == 4
    assert s["median_abs_z"] == pytest.approx(1.0)
    assert s["within_1sigma"] == pytest.approx(0.5)
    assert s["within_2sigma"] == pytest.approx(0.75)
    assert set(s["by_family"]) == {"clamp", "resting"}
    assert s["by_family"]["clamp"]["n"] == 1
    assert fit_summary([])["n"] == 0


def test_targets_that_disagree_beyond_their_sigmas_are_a_conflict():
    a = _spec("a", target=90.0, sigma=3.0)
    b = _spec("b", target=110.0, sigma=4.0)
    near = _spec("near", target=95.0, sigma=3.0)
    elsewhere = _spec("elsewhere", target=200.0, arm=_MEAL)
    out = target_conflicts([a, b, near, elsewhere])
    assert [c["items"] for c in out] == [["a", "b"], ["b", "near"]]
    assert out[0]["gap_sigma"] == pytest.approx((110.0 - 4.0 - 93.0) / 5.0)
    ceiling = _spec("ceiling", target=100.0, sigma=2.0, shape=TargetShape.AT_MOST)
    floor = _spec("floor", target=90.0, sigma=2.0, shape=TargetShape.AT_LEAST)
    assert target_conflicts([ceiling, floor]) == []


def test_a_break_is_inside_one_sigma_then_beyond_two():
    previous = [{"name": "held", "z": 0.4}, {"name": "drifted", "z": 0.8}, {"name": "was_bad", "z": 3.0}]
    now = [_row("held", 0.9), _row("drifted", -2.5), _row("was_bad", 4.0), _row("new", 9.0)]
    assert [b["name"] for b in breaks(previous, now)] == ["drifted"]


def test_individual_skill_is_read_against_both_baselines_on_measured_people():
    from pulse.benchmark import _episode_marker_stats, individual_skill

    def episode(source: str, pred: float, median: float, running: float) -> dict:
        pts = [
            {"marker_id": "glucose", "time": t, "truth": 100.0, "pred": pred, "baseline": 100.0,
             "median_person": median, "running_mean": running}
            for t in range(3)
        ]
        return {"source": source, "markers": _episode_marker_stats(pts)}

    results = {"per_episode": [
        episode("cgmacros", pred=104.0, median=130.0, running=110.0),
        episode("cgmacros", pred=96.0, median=70.0, running=90.0),
        episode("teacher_dynamic", pred=0.0, median=0.0, running=0.0),
    ]}
    skill = individual_skill(results)["by_source"]
    assert set(skill) == {"cgmacros"}
    g = skill["cgmacros"]["glucose"]
    assert g["n_episodes"] == 2
    assert g["mae"] == pytest.approx(4.0)
    assert g["vs_median_person"] == pytest.approx(1.0 - 4.0 / 30.0)
    assert g["vs_running_mean"] == pytest.approx(1.0 - 4.0 / 10.0)


def test_scoreboard_path_sits_beside_the_checkpoint(tmp_path):
    assert scoreboard_path(tmp_path / "run.pt") == tmp_path / "run.scoreboard.json"


@pytest.fixture(scope="module")
def board():
    torch.manual_seed(0)
    model = ModularPhysiologyNetwork()
    rules = [r for r in PHYSIOLOGY_RULES if r.init_mode is InitMode.NORM_CENTER][:1]
    return scoreboard(
        model,
        n_draws=2,
        sealed_specs=[_spec("held_out", sealed=True)],
        sealed_rules=[],
        training_specs=[_spec("fasting_glucose"), _spec("a_far_target", target=300.0)],
        training_rules=rules,
        identifiability=True,
    )


def test_the_board_scores_the_sealed_set_apart_from_training(board):
    assert board["n_population"] == 2
    assert board["literature_fit"]["sealed"]["n"] == 1
    assert board["literature_fit"]["training"]["n"] == 2 + sum(
        1 for i in board["items"] if not i["sealed"] and i["operator"] == "hinge"
    )
    items = {i["name"]: i for i in board["items"]}
    assert items["held_out"]["sealed"] and not items["fasting_glucose"]["sealed"]
    assert items["a_far_target"]["z"] < -5.0
    assert board["individual_skill"] is None
    json.dumps(board)


def test_the_board_counts_open_directions(board):
    openness = board["openness"]
    assert openness["n_params"] > 0
    assert 0 < openness["n_open"] <= openness["n_params"]
    assert openness["person_axis"]["dim"] == EMBEDDING_DIM


def test_a_second_board_finds_breaks_against_the_first(board):
    worse = [
        {**i, "z": 5.0} if i["name"] == "fasting_glucose" else i for i in board["items"]
    ]
    rows = [ItemResidual(**i) for i in worse]
    found = breaks(board["items"], rows)
    held = abs(next(i["z"] for i in board["items"] if i["name"] == "fasting_glucose")) <= 1.0
    assert [b["name"] for b in found] == (["fasting_glucose"] if held else [])


def test_the_board_renders_every_line(board):
    skill = {"by_source": {"cgmacros": {"glucose": {
        "n_episodes": 3, "mae": 9.0, "median_person_mae": 12.0, "running_mean_mae": 10.0,
        "vs_median_person": 0.25, "vs_running_mean": 0.1,
    }}}}
    text = render(attach_individual_skill(board, skill))
    for line in ("literature fit (sealed)", "individual skill", "open directions", "conflicts", "breaks"):
        assert line in text
    assert "cgmacros" in text
