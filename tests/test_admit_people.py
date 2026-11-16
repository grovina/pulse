"""The gate's people check: a paper that prints a sealed paper's cohort is refused."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import pulse.admit as gate

FIXTURE = Path(__file__).parent / "fixtures" / "corpus" / "findings" / "synthetic-test-2099.json"


def _item(i: str, value: float, sd: float, quote: str = "") -> dict:
    return {"id": i, "value": value, "uncertainty": {"kind": "sd", "value": sd, "n": 30},
            "location": {"where": "Table 1", "quote": quote}}


def _record(slug: str, *items: dict, **extra) -> dict:
    return {"paper": slug, "citation": {"authors": "A et al."}, "findings": list(items), **extra}


@pytest.fixture
def root(tmp_path, monkeypatch):
    (tmp_path / "findings").mkdir()
    # slugs starting "seal-" are sealed, the rest train: the draw is not what is tested here
    monkeypatch.setattr(gate, "_sealed", lambda slug: (slug.startswith("seal-"), ""))
    return tmp_path


def _put(root: Path, data: dict) -> Path:
    path = root / "findings" / f"{data['paper']}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _check(root: Path, data: dict, paper_file: Path | None = None) -> gate.Check:
    path = root / "findings" / f"{data['paper']}.json"
    return gate._people_check(path, data, data["paper"], data["paper"].startswith("seal-"), root, paper_file)


SBP, DBP = _item("sbp", 129.43, 12.12), _item("dbp", 85.03, 10.45)


def test_two_distinctive_pairs_in_common_fail_and_name_the_paper_and_the_pairs(root):
    _put(root, _record("seal-2016", SBP, DBP))
    check = _check(root, _record("new-2020", SBP, DBP))
    assert not check.ok
    text = " ".join(check.reasons)
    assert "seal-2016" in text and "129.43 +/- 12.12" in text and "85.03 +/- 10.45" in text
    assert "shares_people_with_sealed_paper" in text


def test_one_pair_passes_with_a_note(root):
    _put(root, _record("seal-2016", SBP, _item("x", 1.234, 0.567)))
    check = _check(root, _record("new-2020", SBP, _item("y", 9.876, 0.543)))
    assert check.ok and "129.43 +/- 12.12" in check.note and "seal-2016" in check.note
    assert "one pair also in" in check.note


def test_short_pairs_never_match(root):
    shorts = [_item("a", 5.0, 0.4), _item("b", 120.0, 15.0), _item("c", 7.5, 1.2), _item("d", 0.5, 0.25)]
    _put(root, _record("seal-2016", *shorts))
    check = _check(root, _record("new-2020", *shorts))
    assert check.ok and not check.note


@pytest.mark.parametrize("value,spread,counts", [
    (129.43, 12.12, True), (85.03, 10.45, True), (5.0, 0.4, False), (120.0, 15.0, False),
    (7.93, 3.08, True), (5.25, 0.4, False), (12.5, 2.5, True),
])
def test_what_counts_as_distinctive(value, spread, counts):
    assert gate._distinctive(value, spread) is counts


def test_the_papers_text_is_searched_for_pairs_in_both_table_forms(root, tmp_path):
    _put(root, _record("seal-2016", SBP, DBP))
    paper = tmp_path / "paper.xml"
    paper.write_text("<td>Systolic</td><td>129.43 ± 12.12</td><td>Diastolic</td><td>85.03 (±10.45)</td>", encoding="utf-8")
    check = _check(root, _record("new-2020", _item("other", 1.0, 0.1)), paper)
    assert not check.ok and "129.43 +/- 12.12" in " ".join(check.reasons)


def test_a_shared_trial_registration_fails(root, tmp_path):
    _put(root, _record("seal-2016", _item("a", 1.0, 0.1, "registered as NCT01234567")))
    check = _check(root, _record("new-2020", _item("b", 2.0, 0.2, "ClinicalTrials.gov nct01234567")))
    assert not check.ok and "NCT01234567" in " ".join(check.reasons)
    paper = tmp_path / "paper.txt"
    paper.write_text("Trial registration: ISRCTN12345678.", encoding="utf-8")
    _put(root, _record("seal-2017", _item("a", 1.0, 0.1, "ISRCTN12345678")))
    assert not _check(root, _record("new-2021", _item("c", 3.0, 0.3)), paper).ok


def test_different_registrations_pass(root):
    _put(root, _record("seal-2016", _item("a", 1.0, 0.1, "NCT01234567")))
    assert _check(root, _record("new-2020", _item("b", 2.0, 0.2, "NCT07654321"))).ok


def test_a_paper_never_matches_itself(root):
    data = _record("seal-2016", SBP, DBP)
    _put(root, data)
    assert _check(root, data).ok
    _put(root, _record("new-2020", SBP, DBP))
    assert _check(root, _record("new-2020", SBP, DBP)).reasons  # but its twin does match


def test_two_training_papers_sharing_people_pass_with_a_warning(root):
    _put(root, _record("train-2016", SBP, DBP))
    check = _check(root, _record("new-2020", SBP, DBP))
    assert check.ok and "WARNING train-2016 may be the same people" in check.note


def test_two_sealed_papers_sharing_people_pass_with_a_warning(root):
    _put(root, _record("seal-2018", SBP, DBP))
    check = _check(root, _record("seal-2020", SBP, DBP))
    assert check.ok and "WARNING seal-2018" in check.note


def test_a_new_sealed_paper_is_compared_with_the_training_files(root):
    _put(root, _record("train-2016", SBP, DBP))
    check = _check(root, _record("seal-2020", SBP, DBP))
    assert not check.ok and "train-2016" in " ".join(check.reasons)
    # a sealed twin only warns
    _put(root, _record("seal-2018", SBP, DBP))
    assert "seal-2018" not in " ".join(_check(root, _record("seal-2020", SBP, DBP)).reasons)


def test_population_mean_and_sd_and_curve_points_are_pairs(root):
    pop = {"age_years": {"mean": 24.12, "sd": 3.96}, "bmi": {"mean": 37.83, "sd": 4.92}}
    curve = {"id": "c", "points": [{"t_min": 60, "value": 86.87, "uncertainty": {"kind": "sd", "value": 19.36}}]}
    _put(root, _record("seal-2016", curve, protocols=[{"population": pop}]))
    check = _check(root, _record("new-2020", curve, population=pop))
    assert not check.ok and "37.83 +/- 4.92" in " ".join(check.reasons)


def test_the_gate_fails_the_whole_file_and_skips_the_model(root, monkeypatch):
    monkeypatch.setattr(gate, "_run_checks", lambda *a: pytest.fail("the model must not run"))
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    data["protocols"][0]["description"] += " Men: systolic 129.43 \u00b1 12.12, diastolic 85.03 (\u00b110.45) mmHg."
    sealed = dict(data, paper="seal-2016", citation={**data["citation"], "doi": "10.0000/other"})
    _put(root, sealed)
    new = dict(data, paper="train-2099")
    new["citation"] = {**data["citation"], "doi": "10.0000/new"}
    (root / "refusals").mkdir()
    result = gate.admit(_put(root, new), root=root)
    assert "people" in {c.name for c in result.failed}
    assert "shares no people with a sealed paper" in gate.format_admission(result)
    # a refusal does not run the check
    refusal = {"paper": "train-2098"}
    assert "people" not in {c.name for c in gate.admit(_put(root, refusal).rename(root / "refusals" / "train-2098.json"), root=root).checks}
