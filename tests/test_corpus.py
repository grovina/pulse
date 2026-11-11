"""The corpus as files: strict parsing, the loader's output, sealing, and the one-source rule."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import pulse.knowledge.cohort_statistics as registry
from pulse.knowledge.cohort_statistics import ALL_COHORT_STATISTICS, expand_curves, join_file_findings
from pulse.knowledge.cohort_types import InitMode, StatisticKind
from pulse.knowledge.corpus import (
    REFUSAL_KINDS, CorpusError, count_missing, count_refusals, load_corpus, parse_finding, parse_refusal, read_finding,
)
from pulse.knowledge.physiology_rules import _AUTHORED_RULES
from pulse.knowledge.sealing import apply_seal, paper_is_sealed
from pulse.types import MG_DL_PER_G

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures" / "corpus"
FINDING = FIXTURES / "findings" / "synthetic-test-2099.json"
REFUSAL = FIXTURES / "refusals" / "synthetic-test-2098.json"
SEALED_SLUG = "synthetic-test-2104"


def _data() -> dict:
    return copy.deepcopy(json.loads(FINDING.read_text(encoding="utf-8")))


def _renamed(slug: str, year: int, doi: str) -> dict:
    data = _data()
    data["paper"] = slug
    data["citation"]["year"] = year
    data["citation"]["doi"] = doi
    return data


def _finding(data: dict, fid: str) -> dict:
    return next(f for f in data["findings"] if f["id"] == fid)


def _problems(data: dict) -> list[str]:
    with pytest.raises(CorpusError) as err:
        parse_finding(data)
    return err.value.reasons


def _spec(finding, fid: str):
    return next(s for s in finding.specs if s.name.endswith(f".{fid}"))


# --- the loader's output for the fixture paper ------------------------------------

def test_the_fixture_paper_parses_into_one_spec_per_finding():
    f = read_finding(FINDING)
    assert f.paper == "synthetic-test-2099" and f.n == 12
    assert [s.name for s in f.specs] == [f"synthetic-test-2099.{i.id}" for i in f.items]
    assert {s.kind for s in f.specs} == set(StatisticKind)
    assert all(s.paper == "synthetic-test-2099" and not s.sealed for s in f.specs)
    assert all(s.init_mode is InitMode.COLD for s in f.specs)


def test_values_arrive_in_the_markers_unit():
    f = read_finding(FINDING)
    peak = _spec(f, "ogtt_glucose_peak")
    assert peak.target == pytest.approx(8.0 * 18.016)          # mmol/L -> mg/dL
    curve = _spec(f, "ogtt_insulin_curve")
    assert [c.target for c in curve.curve] == pytest.approx([10.0, 70.0, 80.0, 50.0])   # pmol/L -> µU/mL
    assert [c.sigma for c in curve.curve] == pytest.approx([1.0, 10.0, 12.0, 8.0])
    assert _spec(f, "fasting_insulin_spread").target == pytest.approx(3.0)
    assert _spec(f, "ogtt_minus_fasting_glucose").target == pytest.approx(1.1 * 18.016)


def test_an_sd_keeps_the_papers_n_and_never_leaves_n_arm_unset():
    f = read_finding(FINDING)
    peak = _spec(f, "ogtt_glucose_peak")
    assert peak.sigma == pytest.approx(1.2 * 18.016) and peak.n_arm == 12
    for s in f.specs:
        assert s.n_arm is not None, s.name
    assert _spec(f, "insulin_increment_ogtt_over_bolus").n_arm == 1


def test_a_peak_time_is_the_papers_clock_not_the_windows():
    data = _data()
    item = _finding(data, "ogtt_glucose_time_to_peak")
    item["window"] = {"start_min": 15, "end_min": 120}
    spec = _spec(parse_finding(data), "ogtt_glucose_time_to_peak")
    assert spec.target == 15.0              # 30 min on the paper's clock, 15 min into the window
    item["value"] = 0.1                     # 6 min: before the window opens
    item["uncertainty"] = {"kind": "sd", "value": 0.1, "n": 12}
    assert any("outside its window" in p for p in _problems(data))


def test_the_curve_window_spans_its_points_and_expands_to_point_specs():
    f = read_finding(FINDING)
    curve = _spec(f, "ogtt_insulin_curve")
    assert (curve.window.start_min, curve.window.end_min) == (0, 91)
    points = expand_curves([curve])
    assert [p.window.start_min for p in points] == [0, 30, 60, 90]
    assert all(p.n_arm == 1 for p in points)


def test_protocols_become_arms_in_the_runtime_meal_order():
    data = _data()
    data["protocols"][0]["meals"] = [{"t_min": 0, "carbs_g": 75.0, "fat_g": 11.0, "protein_g": 22.0}]
    arm = parse_finding(data).protocols["ogtt"]
    assert arm.meals == ((0.0, 75.0, 11.0, 22.0),)    # (t, carbs, fat, protein): cohort_loss._meals_from_spec
    assert arm.label == "ogtt" and arm.duration_min == 120 and arm.start_hour == 8.0


def test_an_iv_bolus_is_the_ivgtt_infusion():
    arm = read_finding(FINDING).protocols["iv_bolus"]
    rate = 0.3 * 70.0 / 2 * MG_DL_PER_G
    assert arm.glucose_infusion[:2] == pytest.approx((rate, rate)) and arm.glucose_infusion[2:] == (0.0,) * 58
    assert arm.sleep_wake is None and arm.glucose_clamp is None


def test_input_segments_fill_a_series_with_their_baseline_elsewhere():
    data = _data()
    proto = data["protocols"][1]
    proto["sleep_wake"] = {"segments": [{"from_min": 0, "to_min": 30, "level": 0.0}]}
    proto["activity"] = {"segments": [{"from_min": 60, "to_min": 90, "level": 0.7}]}
    proto["glucose_clamp"] = {"unit": "mmol/L", "segments": [{"from_min": 30, "to_min": 120, "level": 5.0}]}
    proto["glucose_infusion"] = {"unit": "mg/kg/min", "segments": [{"from_min": 0, "to_min": 10, "level": 1.85}]}
    arm = parse_finding(data).protocols["fasting"]
    assert arm.sleep_wake[:31] == (0.0,) * 30 + (1.0,) and arm.sleep_wake[119] == 1.0   # awake is the baseline
    assert arm.activity[59] == 0.0 and arm.activity[60] == 0.7 and arm.activity[90] == 0.0
    assert arm.glucose_clamp[29] == 0.0 and arm.glucose_clamp[30] == pytest.approx(5.0 * 18.016)
    assert arm.glucose_infusion[0] == pytest.approx(1.0)       # 1.85 mg/kg/min is 1 mg/dL/min


def test_the_population_of_the_paper_becomes_the_arms_population():
    arm = read_finding(FINDING).protocols["ogtt"]
    assert arm.population.sex == "male" and arm.population.age == (20.0, 31.0)
    assert arm.population.bmi is None and arm.population.healthy
    data = _data()
    data["protocols"][1]["population"] = {"sex": "female", "bmi": [30, 40], "status": "unselected"}
    arms = parse_finding(data).protocols
    assert arms["fasting"].population.sex == "female" and arms["fasting"].population.bmi == (30.0, 40.0)
    assert not arms["fasting"].population.healthy
    assert arms["ogtt"].population == read_finding(FINDING).protocols["ogtt"].population


def test_a_population_with_nothing_set_is_the_reference_prior():
    data = _data()
    data["population"].update(sex="mixed", age_years=None, bmi=None, status="unselected")
    assert parse_finding(data).protocols["ogtt"].population is None


def test_an_age_without_a_range_is_mean_plus_or_minus_two_sd():
    data = _data()
    data["population"]["age_years"] = {"mean": 40.0, "sd": 5.0, "min": None, "max": None}
    assert parse_finding(data).protocols["ogtt"].population.age == (30.0, 50.0)


# --- strictness ---------------------------------------------------------------------

@pytest.mark.parametrize("where, path", [
    ("file", []), ("citation", ["citation"]), ("population", ["population"]),
    ("protocol", ["protocols", 0]), ("meal", ["protocols", 0, "meals", 0]),
    ("finding", ["findings", 0]), ("location", ["findings", 0, "location"]),
    ("uncertainty", ["findings", 0, "uncertainty"]), ("window", ["findings", 0, "window"]),
    ("curve point", ["findings", 2, "points", 0]),
])
def test_an_unknown_key_is_an_error_anywhere(where, path):
    data = _data()
    node = data
    for step in path:
        node = node[step]
    node["sigma"] = 1.0
    assert any("unknown fields ['sigma']" in p for p in _problems(data)), where


def test_a_file_never_states_sigma_n_arm_weight_shape_or_sealed():
    for key in ("sigma", "n_arm", "weight", "shape", "sealed", "band_halfwidth"):
        data = _data()
        data["findings"][0][key] = 1
        assert any(key in p for p in _problems(data))


@pytest.mark.parametrize("path, key", [
    ([], "citation"), ([], "population"), ([], "protocols"), ([], "findings"), ([], "left_out"),
    (["citation"], "title"), (["population"], "n"), (["population"], "age_years"),
    (["protocols", 0], "meals"), (["protocols", 0], "start_hour"), (["protocols", 0], "assumed"),
    (["findings", 0], "location"), (["findings", 0], "uncertainty"), (["findings", 0], "unit"),
    (["findings", 0], "window"), (["findings", 0, "uncertainty"], "n"),
    (["findings", 0, "location"], "quote"),
])
def test_a_missing_field_is_an_error(path, key):
    data = _data()
    node = data
    for step in path:
        node = node[step]
    del node[key]
    assert any(key in p and "missing" in p for p in _problems(data))


def test_an_unknown_marker_is_an_error_with_a_suggestion():
    data = _data()
    data["findings"][0]["marker"] = "glucoze"
    assert any("glucoze" in p and "glucose" in p for p in _problems(data))
    data["findings"][0]["marker"] = "adiponectin"
    assert any("adiponectin" in p for p in _problems(data))


def test_an_unknown_or_unlisted_unit_is_an_error():
    data = _data()
    _finding(data, "ogtt_glucose_peak")["unit"] = "mmol/dL"
    assert any("mmol/dL" in p and "not a listed unit" in p for p in _problems(data))
    data = _data()
    _finding(data, "fasting_insulin_spread")["unit"] = "ng/mL"
    assert any("ng/mL" in p for p in _problems(data))


def test_ratios_and_correlations_take_only_their_dimensionless_unit():
    data = _data()
    _finding(data, "insulin_ogtt_over_fasting")["unit"] = "mmol/L"
    assert any("must be 'ratio'" in p for p in _problems(data))
    data = _data()
    _finding(data, "glucose_insulin_resistance_r")["unit"] = "ratio"
    assert any("must be 'r'" in p for p in _problems(data))


def test_a_unit_slip_is_caught_by_the_plausible_range():
    data = _data()
    _finding(data, "ogtt_glucose_peak")["unit"] = "mg/dL"          # 8.0 read as mg/dL
    msgs = _problems(data)
    assert any("outside [40, 400]" in p and "check the unit" in p for p in msgs)
    _finding(data, "ogtt_glucose_peak")["unusual_value_ok"] = "the paper reports glycaemia in a hypoglycaemic arm"
    assert parse_finding(data)       # stated, so allowed
    data = _data()
    _finding(data, "ogtt_glucose_peak")["value"] = 144.0             # a mg/dL value left in mmol/L
    assert any("outside [40, 400]" in p for p in _problems(data))


def test_a_delta_wider_than_the_marker_range_is_a_slip():
    data = _data()
    _finding(data, "ogtt_minus_fasting_glucose").update(
        value=1100.0, uncertainty={"kind": "sem", "value": 1.0, "n": 12})
    assert any("wider than its whole range" in p for p in _problems(data))


def test_all_problems_of_a_file_are_reported_together():
    data = _data()
    data["citation"]["extra"] = 1
    data["findings"][0]["marker"] = "nope"
    del _finding(data, "ogtt_minus_fasting_glucose")["location"]
    _finding(data, "fasting_glucose_mean")["uncertainty"]["kind"] = "variance"
    data["protocols"][0]["start_hour"] = 30.0
    problems = _problems(data)
    assert len(problems) >= 5
    for fragment in ("extra", "nope", "location", "variance", "start_hour"):
        assert any(fragment in p for p in problems), fragment


def test_the_problems_carry_their_check():
    data = _data()
    del _finding(data, "ogtt_glucose_peak")["location"]["quote"]
    _finding(data, "ogtt_insulin_curve")["points"][0]["uncertainty"]["kind"] = "variance"
    _finding(data, "ogtt_glucose_time_to_peak")["unit"] = "years"
    data["protocols"][2]["duration_min"] = "long"
    with pytest.raises(CorpusError) as err:
        parse_finding(data)
    assert {p.check for p in err.value.problems} >= {"place", "uncertainty", "units", "protocol"}


def test_a_number_needs_a_place_in_the_paper():
    for bad in ({"where": "T", "quote": "8 mmol/L"}, {"where": "Table 2", "quote": "peak glucose was high"},
                {"where": "Table 2", "quote": "  "}):
        data = _data()
        _finding(data, "ogtt_glucose_peak")["location"] = bad
        assert any("location" in p for p in _problems(data)), bad


def test_an_uncertainty_needs_a_stated_kind_and_its_n():
    data = _data()
    _finding(data, "ogtt_glucose_peak")["uncertainty"] = {"value": 1.2, "n": 12}
    assert any("kind" in p for p in _problems(data))
    data = _data()
    del _finding(data, "ogtt_glucose_peak")["uncertainty"]["n"]
    assert any("n" in p and "missing" in p for p in _problems(data))
    data = _data()
    _finding(data, "ogtt_glucose_peak")["uncertainty"] = {"kind": "sd", "value": 1.2, "n": 1}
    assert any("n of at least 2" in p for p in _problems(data))


@pytest.mark.parametrize("fid, uncertainty", [
    ("insulin_ogtt_over_fasting", {"kind": "sd", "value": 1.0, "n": 12}),
    ("insulin_increment_ogtt_over_bolus", {"kind": "range", "low": 1.0, "high": 2.0, "n": 12}),
    ("fasting_insulin_spread", {"kind": "sd", "value": 3.0, "n": 12}),
    ("glucose_insulin_resistance_r", {"kind": "range", "low": 0.1, "high": 0.7, "n": 12}),
    ("ogtt_glucose_peak", {"kind": "n_only", "n": 12}),
])
def test_an_uncertainty_the_statistic_cannot_use_is_an_error(fid, uncertainty):
    data = _data()
    _finding(data, fid)["uncertainty"] = uncertainty
    assert any("takes an uncertainty of kind" in p for p in _problems(data))


def test_the_value_has_to_lie_inside_its_own_interval():
    data = _data()
    _finding(data, "ogtt_minus_fasting_glucose")["value"] = 3.0
    assert any("outside its ci" in p for p in _problems(data))


def test_protocol_references_are_checked():
    data = _data()
    _finding(data, "ogtt_glucose_peak")["protocols"] = ["nowhere"]
    assert any("nowhere" in p for p in _problems(data))
    data = _data()
    _finding(data, "ogtt_minus_fasting_glucose")["protocols"] = ["ogtt"]
    assert any("runs 2 protocol" in p for p in _problems(data))
    data = _data()
    data["protocols"].append({**copy.deepcopy(data["protocols"][1]), "name": "unused"})
    assert any("unused" in p and "no finding uses it" in p for p in _problems(data))
    data = _data()
    data["protocols"].append(copy.deepcopy(data["protocols"][0]))
    assert any("declared twice" in p for p in _problems(data))


def test_a_file_can_say_an_error_was_computed_over_more_tests_than_people():
    data = _data()
    u = _finding(data, "fasting_glucose_mean")["uncertainty"]
    plain = next(i.spec for i in parse_finding(data).items if i.id == "fasting_glucose_mean")
    if u["kind"] != "sem":
        u.clear()
        u.update({"kind": "sem", "value": 0.1, "n": 12})
        plain = next(i.spec for i in parse_finding(data).items if i.id == "fasting_glucose_mean")
    u["observations"] = 3 * u["n"]
    spec = next(i.spec for i in parse_finding(data).items if i.id == "fasting_glucose_mean")
    assert spec.n_arm == plain.n_arm
    assert spec.sigma == pytest.approx(plain.sigma * 3 ** 0.5)
    u["observations"] = u["n"]
    assert any("must exceed n" in p for p in _problems(data))


def test_a_finding_file_records_what_it_left_out_and_why():
    data = _data()
    data["left_out"] = [
        {"what": "C-peptide at each time (Table 2)", "missing": {"kind": "marker", "name": "c_peptide"}},
        {"what": "time courses (Figure 1)", "missing": {"kind": "source", "name": "figure_only"}},
    ]
    left = parse_finding(data).left_out
    assert [(x.kind, x.name) for x in left] == [("marker", "c_peptide"), ("source", "figure_only")]


@pytest.mark.parametrize("entry, fragment", [
    ({"what": "x", "missing": {"kind": "marker", "name": "glucose"}}, "already carries"),
    ({"what": "x", "missing": {"kind": "figure", "name": "figure_only"}}, "kind"),
    ({"what": "x", "missing": {"kind": "source", "name": "Figure only"}}, "lowercase"),
    ({"what": "x"}, "missing"),
    ({"what": "x", "missing": {"kind": "source", "name": "figure_only"}, "value": 3}, "value"),
])
def test_a_left_out_entry_names_a_countable_missing_piece(entry, fragment):
    data = _data()
    data["left_out"] = [entry]
    assert any("left_out 1" in p and fragment in p for p in _problems(data))


def test_what_is_missing_is_counted_over_refusals_and_left_outs(tmp_path):
    data = _data()
    data["left_out"] = [
        {"what": "a", "missing": {"kind": "operator", "name": "single_arm_auc"}},
        {"what": "b", "missing": {"kind": "operator", "name": "single_arm_auc"}},
        {"what": "c", "missing": {"kind": "source", "name": "figure_only"}},
    ]
    other = _renamed(SEALED_SLUG, 2104, "10.0000/synthetic.2104")
    other["left_out"] = [{"what": "d", "missing": {"kind": "source", "name": "figure_only"}}]
    corpus = load_corpus(_corpus(tmp_path, {data["paper"]: data, SEALED_SLUG: other}))
    counts = count_missing(corpus)
    assert counts["operator"] == {"single_arm_auc": (0, 1)}  # one paper, however many entries
    assert counts["source"] == {"figure_only": (0, 2)}


def test_a_protocol_names_the_keys_the_paper_does_not_state():
    data = _data()
    data["protocols"][0]["assumed"] = ["start_hour", "duration_min"]
    finding = parse_finding(data)
    name = data["protocols"][0]["name"]
    assert finding.assumed[name] == ("start_hour", "duration_min")
    assert all(not keys for other, keys in finding.assumed.items() if other != name)


@pytest.mark.parametrize("assumed, fragment", [
    ("start_hour", "not a list"),
    (["start_hour", 3], "not a list"),
    (["prefast_hours"], "not one of its keys"),   # the fixture's first protocol does not state it
    (["name"], "not one of its keys"),
    (["clock"], "not one of its keys"),
    (["start_hour", "start_hour"], "repeats"),
])
def test_assumed_names_only_keys_the_protocol_states(assumed, fragment):
    data = _data()
    assert "prefast_hours" not in data["protocols"][0]
    data["protocols"][0]["assumed"] = assumed
    assert any("assumed" in p and fragment in p for p in _problems(data))


def test_windows_must_fit_the_arm():
    data = _data()
    _finding(data, "ogtt_glucose_peak")["window"] = {"start_min": 0, "end_min": 121}
    assert any("duration_min" in p for p in _problems(data))
    data = _data()
    _finding(data, "ogtt_insulin_curve")["points"][3]["t_min"] = 120
    assert any("needs duration_min 121" in p for p in _problems(data))
    data = _data()
    _finding(data, "insulin_increment_ogtt_over_bolus")["baseline_window"] = {"start_min": 0, "end_min": 30}
    assert any("baseline" in p for p in _problems(data))


def test_conditional_fields_belong_to_their_statistic():
    data = _data()
    _finding(data, "ogtt_glucose_peak")["partner"] = "insulin"
    assert any("unknown fields ['partner']" in p for p in _problems(data))
    data = _data()
    del _finding(data, "glucose_insulin_resistance_r")["partner"]
    assert any("partner" in p and "missing" in p for p in _problems(data))
    data = _data()
    _finding(data, "glucose_insulin_resistance_r")["partner"] = "mood"
    assert any("neither a marker nor a person factor" in p for p in _problems(data))
    data = _data()
    _finding(data, "ogtt_glucose_peak")["statistic"] = "median"
    assert any("median" in p for p in _problems(data))


def test_a_paper_is_a_condition_free_slug_filed_under_its_own_name(tmp_path):
    data = _data()
    data["paper"] = "Synthetic Test"
    assert any("slug" in p for p in _problems(data))
    path = tmp_path / "other-name.json"
    with pytest.raises(CorpusError, match="other-name.json"):
        parse_finding(_data(), path=path)


def test_a_citation_needs_an_identifier_and_a_year_the_slug_agrees_with():
    data = _data()
    del data["citation"]["doi"]
    assert any("identifier" in p for p in _problems(data))
    data = _data()
    data["citation"]["doi"] = "not-a-doi"
    assert any("DOI" in p for p in _problems(data))
    data = _data()
    data["citation"]["year"] = 2001
    assert any("ends in a year" in p for p in _problems(data))
    data = _data()
    del data["citation"]["doi"]
    data["citation"]["pmid"] = "12345"
    data["citation"]["pmcid"] = "PMC999"
    assert parse_finding(data)


def test_a_condition_the_model_cannot_express_is_not_a_population():
    data = _data()
    data["population"]["status"] = "type 2 diabetes"
    assert any("status" in p for p in _problems(data))


def test_a_pre_fasted_protocol_and_init_mode():
    data = _data()
    for proto in data["protocols"]:
        proto["init"] = "norm_center"
    data["protocols"][0]["prefast_hours"] = 10.0
    f = parse_finding(data)
    assert f.protocols["ogtt"].prefast_hours == 10.0
    assert _spec(f, "ogtt_glucose_peak").init_mode is InitMode.NORM_CENTER
    data["protocols"][1]["init"] = "cold"
    assert any("disagree on init" in p for p in _problems(data))


def test_invalid_json_is_an_error(tmp_path):
    p = tmp_path / "x.json"
    p.write_text("{", encoding="utf-8")
    with pytest.raises(CorpusError, match="invalid JSON"):
        read_finding(p)


# --- refusals ----------------------------------------------------------------------

def test_the_fixture_refusal_parses_and_counts_by_kind():
    r = parse_refusal(json.loads(REFUSAL.read_text(encoding="utf-8")), path=REFUSAL)
    assert (r.kind, r.name) == ("marker", "adiponectin")
    assert count_refusals([r, r]) == {"marker": {"adiponectin": 2}}
    assert REFUSAL_KINDS == ("marker", "input", "population", "timescale", "operator", "source")


def test_a_refusal_is_strict_too():
    base = json.loads(REFUSAL.read_text(encoding="utf-8"))

    def problems(mutate):
        data = copy.deepcopy(base)
        mutate(data)
        with pytest.raises(CorpusError) as err:
            parse_refusal(data)
        return err.value.reasons

    assert any("unknown fields" in p for p in problems(lambda d: d.update(sigma=1)))
    assert any("missing" in p for p in problems(lambda d: d.pop("detail")))
    assert any("not one of" in p for p in problems(lambda d: d["missing"].update(kind="assay")))
    assert any("already carries" in p for p in problems(lambda d: d["missing"].update(name="glucose")))
    assert any("non-empty list" in p for p in problems(lambda d: d.update(reported=[])))
    assert any("quote" in p for p in problems(lambda d: d["reported"][0]["location"].update(quote="fell")))


def test_the_refusal_pile_counts_by_kind_then_by_name():
    from pulse.knowledge.corpus import Refusal

    def r(kind, name):
        base = parse_refusal(json.loads(REFUSAL.read_text(encoding="utf-8")))
        return Refusal(base.paper, base.citation, base.reported, kind, name, "d")

    counts = count_refusals([r("input", "hypoxia"), r("input", "hypoxia"), r("input", "altitude"),
                             r("operator", "AUC of a derivative")])
    assert counts == {"input": {"hypoxia": 2, "altitude": 1}, "operator": {"AUC of a derivative": 1}}


# --- the directory ------------------------------------------------------------------

def _corpus(tmp_path: Path, findings: dict[str, dict], refusals: dict[str, dict] | None = None) -> Path:
    for sub, files in (("findings", findings), ("refusals", refusals or {})):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
        for slug, data in files.items():
            (tmp_path / sub / f"{slug}.json").write_text(json.dumps(data), encoding="utf-8")
    return tmp_path


def test_a_directory_loads_every_paper(tmp_path):
    root = _corpus(
        tmp_path, {"synthetic-test-2099": _data(), SEALED_SLUG: _renamed(SEALED_SLUG, 2104, "10.0000/synthetic.2104")},
        {"synthetic-test-2098": json.loads(REFUSAL.read_text(encoding="utf-8"))},
    )
    corpus = load_corpus(root)
    assert [f.paper for f in corpus.findings] == ["synthetic-test-2099", SEALED_SLUG]
    assert len(corpus.specs) == 20 and len(corpus.refusals) == 1


def test_a_paper_cannot_be_both_a_finding_and_a_refusal(tmp_path):
    refusal = json.loads(REFUSAL.read_text(encoding="utf-8"))
    refusal["paper"] = "synthetic-test-2099"
    refusal["citation"]["year"] = 2099
    refusal["citation"]["pmid"] = "99999997"
    root = _corpus(tmp_path, {"synthetic-test-2099": _data()}, {"synthetic-test-2099": refusal})
    with pytest.raises(CorpusError, match="both a finding and a refusal"):
        load_corpus(root)


def test_one_paper_cannot_be_filed_under_two_slugs(tmp_path):
    twin = _renamed(SEALED_SLUG, 2104, _data()["citation"]["doi"].upper())    # same DOI, other case
    root = _corpus(tmp_path, {"synthetic-test-2099": _data(), SEALED_SLUG: twin})
    with pytest.raises(CorpusError, match="are one paper"):
        load_corpus(root)


def test_every_problem_of_every_file_is_reported_with_its_file(tmp_path):
    a, b = _data(), _renamed(SEALED_SLUG, 2104, "10.0000/synthetic.2104")
    a["findings"][0]["marker"] = "nope"
    b["findings"][0]["unit"] = "nope"
    (tmp_path / "findings").mkdir()
    (tmp_path / "findings" / "notes.txt").write_text("x")
    root = _corpus(tmp_path, {"synthetic-test-2099": a, SEALED_SLUG: b})
    with pytest.raises(CorpusError) as err:
        load_corpus(root)
    text = "\n".join(err.value.reasons)
    assert "synthetic-test-2099.json: " in text and f"{SEALED_SLUG}.json: " in text and "notes.txt" in text


def test_the_real_corpus_directory_is_clean():
    corpus = load_corpus()
    assert isinstance(corpus.specs, tuple)
    for path in (ROOT / "corpus" / "findings").glob("*.json"):
        assert read_finding(path).paper == path.stem
    for path in (ROOT / "corpus" / "refusals").glob("*.json"):
        assert parse_refusal(json.loads(path.read_text(encoding="utf-8")), path=path).paper == path.stem
    both = {f.paper for f in corpus.findings} & {r.paper for r in corpus.refusals}
    assert not both


def test_the_real_corpus_loads_and_every_paper_is_in_one_pile():
    """Every file under ``corpus/`` parses, and no paper is in two piles or under two slugs
    (``load_corpus`` raises otherwise). Whether a number is the paper's is the second
    reader's check, not a test's."""
    corpus = load_corpus(ROOT / "corpus")
    papers = [x.paper for pile in (corpus.findings, corpus.refusals, corpus.queue) for x in pile]
    assert len(papers) == len(set(papers))
    assert papers


# --- the registry -------------------------------------------------------------------

def test_file_papers_are_added_after_the_python_registry_and_change_none_of_it():
    python = registry._PYTHON_AUTHORED
    assert registry._AUTHORED[: len(python)] == python
    added = registry._AUTHORED[len(python):]
    file_papers = {f.paper for f in load_corpus(ROOT / "corpus").findings}
    assert {s.paper for s in added} == file_papers
    assert file_papers.isdisjoint(s.paper for s in python)

    from pulse.knowledge.physiology_rules import PHYSIOLOGY_RULES
    from pulse.scoreboard import _fingerprint

    old = expand_curves([apply_seal(s) for s in python])
    kept = [s for s in ALL_COHORT_STATISTICS if s.paper not in file_papers]
    assert _fingerprint([s for s in old if not s.sealed], PHYSIOLOGY_RULES) == _fingerprint(kept, PHYSIOLOGY_RULES)
    everything = (*ALL_COHORT_STATISTICS, *registry.SEALED_COHORT_STATISTICS)
    assert sorted(s.name for s in old) == sorted(s.name for s in everything if s.paper not in file_papers)


def test_file_papers_join_the_registry_and_are_sealed_by_the_draw(tmp_path):
    unsealed, sealed = "synthetic-test-2099", SEALED_SLUG
    assert not paper_is_sealed(unsealed) and paper_is_sealed(sealed)
    root = _corpus(tmp_path, {unsealed: _data(), sealed: _renamed(sealed, 2104, "10.0000/synthetic.2104")})
    joined = join_file_findings(registry._PYTHON_AUTHORED, load_corpus(root).specs, frozenset())
    assert joined[: len(registry._PYTHON_AUTHORED)] == registry._PYTHON_AUTHORED
    everything = expand_curves([apply_seal(s) for s in joined])
    mine = {p: [s for s in everything if s.paper == p] for p in (unsealed, sealed)}
    assert mine[unsealed] and not any(s.sealed for s in mine[unsealed])
    assert mine[sealed] and all(s.sealed for s in mine[sealed])      # every item and curve point together
    assert any("__t" in s.name for s in mine[sealed])


def test_a_file_cannot_author_its_own_seal():
    f = read_finding(FINDING)
    assert not any(s.sealed for s in f.specs)
    from dataclasses import replace
    with pytest.raises(ValueError, match="sealing is the registry"):
        apply_seal(replace(f.specs[0], sealed=True))


def test_a_paper_in_a_python_registry_cannot_also_be_a_file():
    f = read_finding(FINDING)
    python_paper = registry._PYTHON_AUTHORED[0].paper
    from dataclasses import replace
    clash = [replace(s, paper=python_paper) for s in f.specs]
    with pytest.raises(ValueError, match=f"both as corpus files and in a Python registry.*{python_paper}"):
        join_file_findings(registry._PYTHON_AUTHORED, clash, frozenset())


def test_a_paper_in_the_physiology_rules_cannot_also_be_a_file():
    f = read_finding(FINDING)
    from dataclasses import replace
    rule_paper = _AUTHORED_RULES[0].paper
    clash = [replace(s, paper=rule_paper) for s in f.specs]
    with pytest.raises(ValueError, match=rule_paper):
        join_file_findings([], clash, frozenset(r.paper for r in _AUTHORED_RULES))


def test_item_names_cannot_repeat_across_the_registry():
    f = read_finding(FINDING)
    with pytest.raises(ValueError, match="used twice"):
        join_file_findings([], [*f.specs, *f.specs], frozenset())


def test_the_registry_uses_the_real_corpus_directory():
    from pulse.knowledge.corpus import corpus_root
    assert corpus_root() == ROOT / "corpus"


# --- the queue: the third pile ------------------------------------------------------

QUEUE = FIXTURES / "queue" / "synthetic-test-2097.json"


def _queue() -> dict:
    return copy.deepcopy(json.loads(QUEUE.read_text(encoding="utf-8")))


def _queue_problems(data: dict) -> list[str]:
    from pulse.knowledge.corpus import parse_queue_item

    with pytest.raises(CorpusError) as err:
        parse_queue_item(data)
    return err.value.reasons


def test_a_queue_item_parses_and_carries_no_numbers_and_no_seal():
    from pulse.knowledge.corpus import QueueItem, read_queue_item

    q = read_queue_item(QUEUE)
    assert isinstance(q, QueueItem) and q.paper == "synthetic-test-2097"
    assert q.identifiers() == {"doi:10.0000/synthetic.2097.001", "pmcid:PMC9999997"}
    assert [r.form for r in q.reports] == ["table", "figure"]
    assert not hasattr(q, "sealed") and not hasattr(q, "value")
    for key in ("sealed", "value", "findings", "target"):
        data = _queue()
        data[key] = 1
        assert any(f"unknown fields ['{key}']" in p for p in _queue_problems(data))


@pytest.mark.parametrize("mutate, fragment", [
    (lambda d: d.pop("licence"), "licence"),
    (lambda d: d.pop("note"), "note"),
    (lambda d: d.update(doi=None, pmcid=None), "at least one identifier"),
    (lambda d: d.update(doi="nope"), "DOI"),
    (lambda d: d.update(protocol_kinds=["OGTT", "OGTT"]), "protocol_kinds"),
    (lambda d: d.update(protocol_kinds=["spa day"]), "protocol kind"),
    (lambda d: d.update(population_class="sick"), "population_class"),
    (lambda d: d.update(uncertainty="se"), "uncertainty"),
    (lambda d: d.update(reports=[]), "reports"),
    (lambda d: d["reports"][0].update(form="slide"), "form"),
    (lambda d: d.update(figure_only="no"), "figure_only"),
    (lambda d: d.update(licence_url="creativecommons"), "URL"),
    (lambda d: d.update(paper="synthetic-test-2096"), "does not contain"),
])
def test_a_queue_item_is_strict(mutate, fragment):
    data = _queue()
    mutate(data)
    assert any(fragment in p for p in _queue_problems(data))


def test_a_paper_is_in_exactly_one_pile(tmp_path):
    root = _corpus(tmp_path, {"synthetic-test-2099": _data()})
    (root / "queue").mkdir()
    queued = _queue()
    queued["paper"] = "synthetic-test-2099"
    queued["citation"] = queued["citation"].replace("2097", "2099")
    queued["doi"], queued["pmcid"] = "10.0000/other.1", None
    (root / "queue" / "synthetic-test-2099.json").write_text(json.dumps(queued), encoding="utf-8")
    with pytest.raises(CorpusError, match="both a finding and a queue item.*delete its queue file"):
        load_corpus(root)
    (root / "queue" / "synthetic-test-2099.json").unlink()
    (root / "queue" / "synthetic-test-2097.json").write_text(json.dumps(_queue()), encoding="utf-8")
    assert load_corpus(root).counts == {"queue": 1, "findings": 1, "refusals": 0}


def test_an_identifier_already_anywhere_in_the_corpus_is_refused_in_the_queue(tmp_path):
    root = _corpus(tmp_path, {"synthetic-test-2099": _data()})
    (root / "queue").mkdir()
    twin = _queue()
    twin["doi"] = _data()["citation"]["doi"]
    (root / "queue" / "synthetic-test-2097.json").write_text(json.dumps(twin), encoding="utf-8")
    with pytest.raises(CorpusError, match="are one paper"):
        load_corpus(root)


def test_the_real_queue_is_valid_and_disjoint_from_every_python_registry():
    from pulse.knowledge.cohort_statistics import _PYTHON_AUTHORED
    from pulse.knowledge.corpus import count_piles

    corpus = load_corpus()
    python_papers = {s.paper for s in _PYTHON_AUTHORED} | {r.paper for r in _AUTHORED_RULES}
    assert not {q.paper for q in corpus.queue} & python_papers
    piles = [{q.paper for q in corpus.queue}, {f.paper for f in corpus.findings}, {r.paper for r in corpus.refusals}]
    assert sum(len(p) for p in piles) == len(set().union(*piles))
    assert count_piles() == corpus.counts
    for q in corpus.queue:
        assert q.licence and q.reports and q.identifiers()
        assert (ROOT / "corpus" / "queue" / f"{q.paper}.json").is_file()
        # the seal follows from the slug and is not stored
        assert "sealed" not in (ROOT / "corpus" / "queue" / f"{q.paper}.json").read_text(encoding="utf-8")


def _strings(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from _strings(value)


def test_no_paper_full_text_is_committed():
    """A file holds numbers and the sentence or table row each came from, never a paper's
    text: no file but JSON and the guide, and no string longer than a paragraph."""
    for path in (ROOT / "corpus").rglob("*"):
        assert path.is_dir() or path.suffix in {".json", ".md", ""} or path.name == ".gitkeep", path
        if path.suffix == ".json":
            longest = max(map(len, _strings(json.loads(path.read_text(encoding="utf-8")))), default=0)
            assert longest < 1_500, (path, longest)