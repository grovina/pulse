"""Lab viewer graph is the coupling layout `forward()` reads."""

from __future__ import annotations

from pulse.lab_export import build_graph, build_run, write_lab
from pulse.lab_sim import LabSession
from pulse.types import (
    COUPLING_GRAPH,
    DUODENAL_CHANNEL_IDS,
    GUT_CHANNEL_IDS,
    MARKERS,
    MODULE_COUPLING_CHANNELS,
    STATE_DIM,
    System,
)


def test_graph_covers_every_runtime_edge() -> None:
    graph = build_graph()
    assert graph["schema"] == "pulse.lab.graph.v2"
    assert graph["engine"] == "student"
    module_ids = [m["id"] for m in graph["modules"]]
    assert module_ids == ["gut", *[s.value for s in System]]
    assert {m["id"] for m in graph["markers"]} == {m.id for m in MARKERS}
    assert len(graph["markers"]) == STATE_DIM
    assert {c["id"] for c in graph["channels"]} == set(GUT_CHANNEL_IDS) | set(DUODENAL_CHANNEL_IDS)

    exported = {(e["source"], e["target_module"], e["sign"]) for e in graph["edges"]}
    expected = set()
    for target, channels in MODULE_COUPLING_CHANNELS.items():
        for ch in channels:
            if ch in {m.id for m in MARKERS}:
                sign = next(
                    e.sign_prior for e in COUPLING_GRAPH
                    if e.source_marker == ch and e.target_module == target
                )
            else:
                sign = 1
            expected.add((ch, target, sign))
    assert exported == expected
    assert {lp["id"] for lp in graph["loops"]} == {"carbon", "bile"}
    assert "sources" not in graph
    assert all("label" in m and "_" not in m["label"] for m in graph["markers"])
    assert all("label" in c and "_" not in c["label"] for c in graph["channels"])
    labels = {m["id"]: m["label"] for m in graph["markers"]}
    assert labels["fat_mass"] == "Fat mass"
    assert labels["insulin_action"] == "Insulin action"
    assert labels["glucose"] == "Glucose"
    carbon = next(lp for lp in graph["loops"] if lp["id"] == "carbon")
    assert {f["id"] for f in carbon["flows"]} >= {"ra", "uptake", "brk_M"}
    assert any(w["kind"] == "hole" and w["label"] == "Oxidized" for w in carbon["waypoints"])
    glucose_wp = next(w for w in carbon["waypoints"] if w["id"] == "glucose")
    assert glucose_wp["label"] == "Glucose"
    bile = next(lp for lp in graph["loops"] if lp["id"] == "bile")
    assert any(w["id"] == "faecal" and w["kind"] == "hole" for w in bile["waypoints"])
    assert all("label" in w for w in carbon["waypoints"] + bile["waypoints"])


def test_gut_is_a_kernel_node_not_an_ode_module() -> None:
    gut = next(m for m in build_graph()["modules"] if m["id"] == "gut")
    assert gut["kind"] == "kernel"
    assert gut["markers"] == []
    assert "gut.glucose_appearance" in gut["channels"]
    assert "duodenal.carb" in gut["channels"]


def test_meal_protocol_glucose_rises() -> None:
    run = build_run("meal", sample_every=5)
    gi = run["marker_ids"].index("glucose")
    t0 = run["frames"][0]["state"][gi]
    peak = max(f["state"][gi] for f in run["frames"])
    assert peak - t0 > 20.0
    assert run["phases"]
    assert run["meals"][0]["carbs"] == 50.0
    assert run["schema"] == "pulse.lab.run.v2"
    appearing = [f for f in run["frames"] if f["appearing"]]
    assert appearing
    assert max(f["flux"]["carbon"]["ra"] for f in appearing) > 0.05
    assert max(f["duo"][2] for f in appearing) > 0.01
    for f in run["frames"]:
        assert abs(f["flux"]["carbon"]["residual"]) < 1e-5
        assert abs(f["flux"]["bile"]["residual"]) < 1e-5
        assert "hunger" in f["feelings"]


def test_fast_raises_ketones() -> None:
    run = build_run("fast", sample_every=10)
    bi = run["marker_ids"].index("bhb")
    fi = run["marker_ids"].index("fat_mass")
    bhb0 = run["frames"][0]["state"][bi]
    bhb1 = run["frames"][-1]["state"][bi]
    fat0 = run["frames"][0]["state"][fi]
    fat1 = run["frames"][-1]["state"][fi]
    assert bhb1 > bhb0
    assert fat1 < fat0
    assert max(f["flux"]["carbon"]["ra"] for f in run["frames"]) < 0.01
    assert run["frames"][-1]["flux"]["carbon"]["oxidized"] > 0.0


def test_write_lab_is_graph_only(tmp_path) -> None:
    write_lab(tmp_path)
    assert (tmp_path / "graph.json").exists()
    assert not (tmp_path / "runs.json").exists()
    stale = tmp_path / "runs"
    stale.mkdir()
    (stale / "day.teacher.json").write_text("{}")
    write_lab(tmp_path)
    assert not stale.exists()


def test_live_session_does_not_simulate_the_future() -> None:
    session = LabSession(sample_every=5)
    session.reset("morning")
    assert session.run["engine"] == "student"
    assert session.t == 0
    assert session.run["frames"][-1]["t"] == 0
    assert len(session.run["frames"]) == 1
    hungry0 = session.run["frames"][0]["feelings"]["hunger"]["label"]
    assert hungry0 == "Hungry"
    session.eat_plate("plate")
    assert session.t == 0
    assert session.run["frames"][-1]["t"] == 0
    assert len(session.run["frames"]) == 1
    assert session.run["meals"][0]["carbs"] == 50.0
    assert session.run["frames"][0]["feelings"]["hours_since_meal"] < 0.2
    session.advance(60)
    assert session.t == 60
    assert session.run["frames"][-1]["t"] == 60
    assert session.run["frames"][-1]["t"] < session.run["duration_min"]
    session.walk(minutes=30)
    assert session.run["frames"][-1]["activity"] > 0.2
    session.lie_down()
    asleep = session.run["frames"][-1]
    assert asleep["sleep_wake"] < 0.5
    assert asleep["feelings"]["sleep"]["label"] == "Asleep"
    try:
        session.eat_plate("snack")
        raise AssertionError("eat while asleep should fail")
    except ValueError as exc:
        assert str(exc) == "asleep"


def test_dawn_starts_asleep() -> None:
    session = LabSession(sample_every=5)
    session.reset("dawn")
    assert session.snapshot()["asleep"] is True
    try:
        session.eat_plate("snack")
        raise AssertionError("eat while asleep should fail")
    except ValueError as exc:
        assert str(exc) == "asleep"
