"""What a meal says beyond its three grams: parsed strictly, carried beside the
meal tuple, never seen by the model, and enough to tell two protocols apart."""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path

import pytest

import pulse.admit as gate
from pulse.knowledge.cohort_types import PROTEIN_SOURCES, MealDetails
from pulse.knowledge.corpus import CorpusError, parse_finding

ROOT = Path(__file__).resolve().parents[1]
FINDING = Path(__file__).parent / "fixtures" / "corpus" / "findings" / "synthetic-test-2099.json"
DETAIL_KEYS = ("fibre_g", "sugars_g", "fat_g_by_type", "fat_source", "protein_source", "form", "food")

EVERY_KEY = {
    "fibre_g": 4.0, "sugars_g": 12.5,
    "fat_g_by_type": {"saturated": 18.0, "monounsaturated": 6.0, "polyunsaturated": 2.0},
    "fat_source": "butter", "protein_source": "dairy", "form": "solid",
    "food": "white bread, butter and milk",
}


def _data() -> dict:
    return copy.deepcopy(json.loads(FINDING.read_text(encoding="utf-8")))


def _mixed(**details) -> dict:
    """The fixture with its OGTT turned into a mixed meal carrying ``details``."""
    data = _data()
    data["protocols"][0]["meals"] = [
        {"t_min": 0, "carbs_g": 60.0, "fat_g": 30.0, "protein_g": 20.0, **details},
    ]
    return data


def _arm(data: dict, name: str = "ogtt"):
    return next(a for s in parse_finding(data).specs for a in s.arms if a.label == name)


def _problems(data: dict) -> str:
    with pytest.raises(CorpusError) as err:
        parse_finding(data)
    return "\n".join(err.value.reasons)


def _stripped(data: dict) -> dict:
    data = copy.deepcopy(data)
    for p in data["protocols"]:
        for m in p["meals"]:
            for k in DETAIL_KEYS:
                m.pop(k, None)
    return data


# --- parsing --------------------------------------------------------------------------

def test_every_key_parses_into_the_arms_meal_details():
    arm = _arm(_mixed(**EVERY_KEY))
    assert arm.meals == ((0.0, 60.0, 30.0, 20.0),)
    assert arm.meal_details == (MealDetails(
        fibre_g=4.0, sugars_g=12.5,
        fat_g_by_type=(("saturated", 18.0), ("monounsaturated", 6.0), ("polyunsaturated", 2.0)),
        fat_source="butter", protein_source="dairy", form="solid", food="white bread, butter and milk",
    ),)


@pytest.mark.parametrize("key,value", [(k, v) for k, v in EVERY_KEY.items()])
def test_each_key_parses_alone(key, value):
    details = _arm(_mixed(**{key: value})).meal_details
    assert len(details) == 1 and not details[0].is_empty()


def test_a_file_with_every_key_gives_the_model_what_the_file_without_them_gives():
    full, bare = parse_finding(_mixed(**EVERY_KEY)), parse_finding(_stripped(_mixed(**EVERY_KEY)))
    assert [s.name for s in full.specs] == [s.name for s in bare.specs]
    for a, b in zip(full.specs, bare.specs):
        assert (a.target, a.sigma, a.n_arm) == (b.target, b.sigma, b.n_arm)
        assert [x.meals for x in a.arms] == [x.meals for x in b.arms]
        assert [dataclasses.replace(x, meal_details=()) for x in a.arms] == list(b.arms)
        assert dataclasses.replace(a, arms=b.arms) == b
    assert all(x.meal_details == () for s in bare.specs for x in s.arms)


def test_details_follow_their_meal_when_the_meals_are_sorted():
    data = _data()
    data["protocols"][0]["meals"] = [
        {"t_min": 60, "carbs_g": 10.0, "fat_g": 5.0, "protein_g": 0.0, "fat_source": "coconut oil"},
        {"t_min": 0, "carbs_g": 50.0, "fat_g": 0.0, "protein_g": 0.0},
    ]
    data["protocols"][0]["duration_min"] = 121
    arm = _arm(data)
    assert [m[0] for m in arm.meals] == [0.0, 60.0]
    assert arm.meal_details == (MealDetails(), MealDetails(fat_source="coconut oil"))


@pytest.mark.parametrize("details,needle", [
    ({"fibre_g": -1.0}, "fibre_g must not be negative"),
    ({"sugars_g": -0.5}, "sugars_g must not be negative"),
    ({"sugars_g": 70.0}, "more than carbs_g"),
    ({"fat_g_by_type": {"saturated": 25.0, "monounsaturated": 10.0}}, "more than fat_g"),
    ({"fat_g_by_type": {"saturated": -1.0}}, "must not be negative"),
    ({"fat_g_by_type": {"trans": 1.0}}, "unknown fields ['trans']"),
    ({"fat_g_by_type": {}}, "fat_g_by_type is empty"),
    ({"protein_source": "beef"}, "protein_source 'beef' is not one of"),
    ({"form": "gel"}, "form 'gel' is not one of"),
    ({"fat_source": ""}, "fat_source is empty"),
    ({"food": "x" * 81}, "longer than 80"),
    ({"fibre": 3.0}, "unknown fields ['fibre']"),
])
def test_a_bad_detail_is_refused(details, needle):
    assert needle in _problems(_mixed(**details))


def test_a_source_needs_the_nutrient():
    data = _data()      # the fixture's OGTT: 75 g carbohydrate, no fat, no protein
    data["protocols"][0]["meals"][0].update(fat_source="butter", protein_source="whey")
    text = _problems(data)
    assert "fat_source and no fat" in text and "protein_source and no protein" in text


def test_rounded_parts_may_pass_the_total_by_half_a_gram_each():
    ok = _mixed(fat_g_by_type={"saturated": 15.4, "monounsaturated": 10.4, "polyunsaturated": 5.0}, sugars_g=60.4)
    assert _arm(ok).meal_details[0].fat_g_by_type[0] == ("saturated", 15.4)


def test_the_controlled_lists_are_enforced_on_the_runtime_type_too():
    with pytest.raises(ValueError):
        MealDetails(protein_source="beef")
    with pytest.raises(ValueError):
        MealDetails(form="gel")
    assert "whey" in PROTEIN_SOURCES and "other" in PROTEIN_SOURCES


# --- the committed corpus -------------------------------------------------------------

def _committed() -> list[Path]:
    return sorted((ROOT / "corpus" / "findings").glob("*.json"))


@pytest.mark.parametrize("path", _committed(), ids=lambda p: p.stem)
def test_a_committed_file_loads_as_its_grams_alone_would(path):
    """The details never reach a name, target, sigma or meal tuple: each committed
    file gives the specs it gives with every detail key taken out."""
    data = json.loads(path.read_text(encoding="utf-8"))
    full, bare = parse_finding(data, path=path), parse_finding(_stripped(data), path=path)
    assert [s.name for s in full.specs] == [s.name for s in bare.specs]
    for a, b in zip(full.specs, bare.specs):
        assert (a.target, a.sigma, a.n_arm, a.kind) == (b.target, b.sigma, b.n_arm, b.kind)
        assert [x.meals for x in a.arms] == [x.meals for x in b.arms]
        assert dataclasses.replace(a, arms=tuple(dataclasses.replace(x, meal_details=()) for x in a.arms)) == b
    for p, spec_arms in ((p, [x for s in full.specs for x in s.arms if x.label == p["name"]]) for p in data["protocols"]):
        grams = tuple(sorted((float(m["t_min"]), float(m["carbs_g"]), float(m["fat_g"]), float(m["protein_g"]))
                             for m in p["meals"]))
        assert all(x.meals == grams for x in spec_arms), p["name"]


# --- the conflict check ---------------------------------------------------------------

def _rival(peak_mmol: float, **details) -> dict:
    """Another paper, the same mixed meal in grams, a different peak."""
    data = _mixed(**details)
    data["paper"], data["citation"]["year"], data["citation"]["doi"] = "synthetic-test-2101", 2101, "10.0000/s.2101"
    data["protocols"] = [dict(data["protocols"][0], name="oral_load")]
    data["findings"] = [dict(
        data["findings"][0], id="peak", protocols=["oral_load"], value=peak_mmol,
        uncertainty={"kind": "sd", "value": 0.8, "n": 10},
    )]
    return data


def _admitted(monkeypatch, data: dict) -> None:
    import pulse.knowledge.cohort_statistics as registry

    specs = parse_finding(data).specs
    monkeypatch.setattr(registry, "ALL_COHORT_STATISTICS", [*registry.ALL_COHORT_STATISTICS, *specs])


def test_equal_grams_and_another_fat_source_are_another_protocol(monkeypatch):
    _admitted(monkeypatch, _mixed(fat_source="butter"))
    assert gate._conflict_check(parse_finding(_rival(12.0, fat_source="coconut oil"))).ok


def test_equal_grams_without_details_still_conflict(monkeypatch):
    _admitted(monkeypatch, _mixed())
    check = gate._conflict_check(parse_finding(_rival(12.0)))
    assert not check.ok and "peak contradicts synthetic-test-2099.ogtt_glucose_peak" in check.reasons[0]


def test_equal_details_still_conflict(monkeypatch):
    _admitted(monkeypatch, _mixed(fat_source="Butter", form="solid", food="toast and butter"))
    # Case and spacing of a source, and the wording of the food, do not make a new protocol.
    assert not gate._conflict_check(parse_finding(_rival(12.0, fat_source="butter ", food="buttered toast"))).ok


def test_a_detail_only_one_side_states_does_not_separate_them(monkeypatch):
    _admitted(monkeypatch, _mixed())
    assert not gate._conflict_check(parse_finding(_rival(12.0, fat_source="coconut oil"))).ok


@pytest.mark.parametrize("mine,theirs", [
    ({"fibre_g": 2.0}, {"fibre_g": 9.0}),
    ({"protein_source": "whey"}, {"protein_source": "soy"}),
    ({"form": "liquid"}, {"form": "solid"}),
    ({"fat_g_by_type": {"saturated": 20.0}}, {"fat_g_by_type": {"saturated": 5.0, "polyunsaturated": 15.0}}),
    ({"sugars_g": 5.0}, {"sugars_g": 40.0}),
])
def test_each_compared_detail_separates_protocols(monkeypatch, mine, theirs):
    _admitted(monkeypatch, _mixed(**mine))
    assert gate._conflict_check(parse_finding(_rival(12.0, **theirs))).ok


def test_an_acknowledged_conflict_across_fat_sources_is_refused_as_absent(monkeypatch):
    _admitted(monkeypatch, _mixed(fat_source="butter"))
    named = _rival(12.0, fat_source="coconut oil")
    named["findings"][0]["acknowledged_conflicts"] = ["synthetic-test-2099.ogtt_glucose_peak"]
    check = gate._conflict_check(parse_finding(named))
    assert not check.ok and "do not conflict" in check.reasons[0]


# --- the guide ------------------------------------------------------------------------

def test_the_guides_fat_source_example_parses_into_two_arms_the_model_cannot_tell_apart():
    import re

    text = (ROOT / "corpus" / "README.md").read_text(encoding="utf-8")
    block = next(b for b in re.findall(r"```json\n(.*?)\n```", text, re.S) if '"fat_source": "butter"' in b)
    data = _data()
    data["protocols"] = json.loads(block)["protocols"]
    base = data["findings"][0]
    data["findings"] = [dict(base, id=f"peak_{p['name']}", protocols=[p["name"]]) for p in data["protocols"]]
    butter, coconut = (_arm(data, n) for n in ("butter_meal", "coconut_meal"))
    assert dataclasses.replace(butter, label="", meal_details=()) == dataclasses.replace(coconut, label="", meal_details=())
    assert butter.meal_details[0].differs_from(coconut.meal_details[0])
