"""Lab viewer graph is the coupling layout `forward()` reads."""

from __future__ import annotations

from pulse.lab_export import build_graph, build_run
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
