"""The gate: which check fails, what it never prints, and how a conflict is recorded."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

import pulse.admit as gate
from pulse.admit import admit, format_admission, main
from pulse.knowledge.corpus import read_finding

FIXTURES = Path(__file__).parent / "fixtures" / "corpus"
FINDING = FIXTURES / "findings" / "synthetic-test-2099.json"
REFUSAL = FIXTURES / "refusals" / "synthetic-test-2098.json"


def _data() -> dict:
    return copy.deepcopy(json.loads(FINDING.read_text(encoding="utf-8")))


@pytest.fixture
def root(tmp_path):
    (tmp_path / "findings").mkdir()
    (tmp_path / "refusals").mkdir()
    return tmp_path


def _put(root: Path, data: dict, sub: str = "findings") -> Path:
    path = root / sub / f"{data['paper']}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _small() -> dict:
    """The fixture cut to a level and a spread on two protocols: the cheapest file that rolls."""
    data = _data()
    data["findings"] = [f for f in data["findings"] if f["id"] in ("ogtt_glucose_peak", "fasting_insulin_spread")]
    data["protocols"] = [p for p in data["protocols"] if p["name"] in ("ogtt", "fasting")]
    return data


def _failed(result) -> dict[str, tuple[str, ...]]:
    return {c.name: c.reasons for c in result.failed}


_STUB = (gate.Check("runs", True), gate.Check("gradient", True))


@pytest.fixture
def stub_rollouts(monkeypatch):
    """Skip the rollouts: for tests about filing, sealing and output, not about the model."""
    monkeypatch.setattr(gate, "_run_checks", lambda finding, device, fed: _STUB)


@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    """One real gate run on the fixture, on the untrained student, shared by the tests below."""
    root = tmp_path_factory.mktemp("corpus")
    (root / "findings").mkdir()
    (root / "refusals").mkdir()
    data = _small()
    path = root / "findings" / "synthetic-test-2099.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return admit(path, root=root)


def test_the_fixture_is_admitted_on_the_untrained_student(full_run):
    assert full_run.ok, format_admission(full_run)
    assert [c.name for c in full_run.checks] == [
        "parse", "protocol", "units", "place", "uncertainty", "filing", "runs", "gradient", "conflicts",
    ]
    assert full_run.items == 2 and full_run.sealed is False


def test_it_lists_what_the_reader_assumed(root, stub_rollouts):
    data = _data()
    assert "assumed by the reader" not in format_admission(admit(_put(root, data), root=root))
    first = data["protocols"][0]
    first["assumed"] = ["start_hour"]
    text = format_admission(admit(_put(root, data), root=root))
    assert f"assumed by the reader, not stated by the paper:\n  {first['name']}: start_hour" in text


def test_it_reports_whether_the_paper_is_sealed(root, stub_rollouts):
    data = _data()
    data["paper"], data["citation"]["year"], data["citation"]["doi"] = "synthetic-test-2104", 2104, "10.0000/s.2104"
    result = admit(_put(root, data), root=root)
    assert result.sealed is True
    assert "sealed: yes" in format_admission(result)
    assert "sealed: no" in format_admission(admit(_put(root, _data()), root=root))


def test_it_says_which_check_failed_and_skips_the_rest(root):
    data = _data()
    data["findings"][0]["marker"] = "nope"
    del next(f for f in data["findings"] if f["id"] == "ogtt_insulin_curve")["location"]["quote"]
    next(f for f in data["findings"] if f["id"] == "ogtt_minus_fasting_glucose")["uncertainty"]["kind"] = "variance"
    result = admit(_put(root, data), root=root, fed_day_minutes=0)
    assert not result.ok
    assert set(_failed(result)) == {"parse", "place", "uncertainty"}
    assert [c.name for c in result.checks if c.skipped] == ["filing", "runs", "gradient", "conflicts"]
    text = format_admission(result)
    assert "FAIL  each number has a place in the paper" in text and "not admitted" in text


def test_a_unit_slip_fails_the_units_check(root):
    data = _data()
    data["findings"][0]["unit"] = "mg/dL"
    assert set(_failed(admit(_put(root, data), root=root, fed_day_minutes=0))) == {"units"}


def test_a_file_that_is_not_json_fails_to_parse(root):
    path = root / "findings" / "synthetic-test-2099.json"
    path.write_text("[", encoding="utf-8")
    assert "invalid JSON" in _failed(admit(path, root=root))["parse"][0]


def test_the_kind_comes_from_the_directory_or_the_flag(tmp_path):
    path = tmp_path / "x.json"
    path.write_text("{}", encoding="utf-8")
    result = admit(path, root=tmp_path)
    assert not result.ok and "finding, a refusal or a queue" in result.failed[0].reasons[0]
    assert admit(path, kind="refusal", root=tmp_path).failed[0].name == "parse"


def test_the_file_must_be_named_for_its_paper(root):
    path = root / "findings" / "wrong-name.json"
    path.write_text(FINDING.read_text(encoding="utf-8"), encoding="utf-8")
    assert "wrong-name.json" in _failed(admit(path, root=root, fed_day_minutes=0))["parse"][0]


def test_a_paper_already_filed_is_refused(root):
    first = _put(root, _data())
    second = root / "refusals" / "synthetic-test-2099.json"
    refusal = json.loads(REFUSAL.read_text(encoding="utf-8"))
    refusal["paper"], refusal["citation"]["year"] = "synthetic-test-2099", 2099
    second.write_text(json.dumps(refusal), encoding="utf-8")
    result = admit(first, root=root, fed_day_minutes=0)
    assert "already filed as refusals/synthetic-test-2099.json" in _failed(result)["filing"][0]
    assert [c.name for c in result.checks if c.skipped] == ["runs", "gradient", "conflicts"]
    assert not admit(second, root=root).ok     # and the other way round


def test_the_same_paper_under_another_slug_is_refused(root):
    _put(root, _data())
    twin = _data()
    twin["paper"], twin["citation"]["year"] = "synthetic-test-2104", 2104      # same DOI
    result = admit(_put(root, twin), root=root, fed_day_minutes=0)
    assert "another slug, synthetic-test-2099" in _failed(result)["filing"][0]
    twin["citation"].pop("doi")
    twin["citation"]["pmid"] = "42"
    other = _data()
    other["citation"].pop("doi")
    other["citation"]["pmid"] = "42"
    (root / "findings" / "synthetic-test-2099.json").write_text(json.dumps(other), encoding="utf-8")
    assert "pmid:42" in _failed(admit(_put(root, twin), root=root, fed_day_minutes=0))["filing"][0]


def test_a_paper_in_a_python_registry_is_refused(root):
    data = _data()
    data["paper"] = "wahren-1971"
    data["citation"]["year"] = 1971
    result = admit(_put(root, data), root=root, fed_day_minutes=0)
    assert "Python registry" in _failed(result)["filing"][0]


def test_readmitting_the_same_file_is_not_a_duplicate(root, stub_rollouts):
    path = _put(root, _data())
    assert admit(path, root=root).ok
    assert admit(path, root=root).ok


def test_a_protocol_that_does_not_run_fails_the_runs_check(root, monkeypatch):
    def boom(self, keys=None):
        raise RuntimeError("no such input")

    monkeypatch.setattr(gate._Probe, "roll_all", boom)
    result = admit(_put(root, _small()), root=root, fed_day_minutes=0)
    failed = _failed(result)
    assert "protocol ogtt does not run: RuntimeError: no such input" in failed["runs"][0]
    assert "protocol fasting does not run" in failed["runs"][1]


def test_a_non_finite_trajectory_fails_the_runs_check(root, monkeypatch):
    monkeypatch.setattr(gate._Probe, "finite", staticmethod(lambda trajs: False))
    assert "non-finite" in _failed(admit(_put(root, _small()), root=root, fed_day_minutes=0))["runs"][0]


def test_a_finding_the_parameters_cannot_move_fails_the_gradient_check(root, monkeypatch):
    monkeypatch.setattr(gate._Probe, "moves", lambda self, specs, base: {i: 0.0 for i in specs})
    failed = _failed(admit(_put(root, _small()), root=root, fed_day_minutes=0))
    assert set(failed) == {"gradient"} and len(failed["gradient"]) == 2
    assert "does not move when the parameters do" in failed["gradient"][0]
    monkeypatch.setattr(gate._Probe, "moves", lambda self, specs, base: {i: float("nan") for i in specs})
    assert "not finite" in _failed(admit(_put(root, _small()), root=root, fed_day_minutes=0))["gradient"][0]


def _with_windows(**windows) -> dict:
    """The small fixture plus fasting-glucose means over the given (start, end) windows."""
    data = _small()
    template = next(f for f in _data()["findings"] if f["id"] == "fasting_glucose_mean")
    for name, (lo, hi) in windows.items():
        data["findings"].append(dict(copy.deepcopy(template), id=name, window={"start_min": lo, "end_min": hi}))
    return data


@pytest.fixture(scope="module")
def probed():
    """The probe's measured moves for a live set, a dead finding and a weakly coupled one."""
    from pulse.knowledge.corpus import parse_finding

    finding = parse_finding(_with_windows(dead_start=(0, 1), weak_one_step=(0, 2)))
    specs = {i.id: i.spec for i in finding.items}
    probe = gate._Probe("cpu", 0)
    probe.prepare(list(specs.values()))
    base = probe.roll_all()
    return probe, specs, base, probe.moves(specs, base)


def test_a_finding_that_only_reads_the_start_state_is_dead(probed):
    _probe, _specs, _base, moved = probed
    assert moved["dead_start"] == 0.0           # exactly: the same ops, the same inputs
    assert moved["dead_start"] <= gate._MOVED


def test_a_weakly_coupled_finding_passes_and_the_live_ones_pass_easily(probed):
    _probe, _specs, _base, moved = probed
    assert moved["weak_one_step"] > 100 * gate._MOVED   # one minute of dynamics: ~4e-4 at the nudge
    for i in ("ogtt_glucose_peak", "fasting_insulin_spread"):
        assert moved[i] > moved["weak_one_step"]


def test_the_move_is_a_signal_not_rounding_noise(probed, monkeypatch):
    """Halving the nudge halves the move of a live finding (linear response) and leaves a dead one at 0."""
    probe, specs, base, moved = probed
    monkeypatch.setattr(gate, "_NUDGE", gate._NUDGE / 10)
    smaller = probe.moves({k: specs[k] for k in ("weak_one_step", "ogtt_glucose_peak", "dead_start")}, base)
    assert smaller["dead_start"] == 0.0
    for i in ("weak_one_step", "ogtt_glucose_peak"):
        assert smaller[i] == pytest.approx(moved[i] / 10, rel=0.35)


def test_a_dead_finding_fails_the_real_gate(root):
    result = admit(_put(root, _with_windows(dead_start=(0, 1))), root=root, fed_day_minutes=0)
    failed = _failed(result)
    assert set(failed) == {"gradient"} and "dead_start" in failed["gradient"][0]
    assert "start state" in failed["gradient"][0]
    assert admit(_put(root, _with_windows(weak_one_step=(0, 2))), root=root, fed_day_minutes=0).ok


# --- conflicts ------------------------------------------------------------------------

def _admitted(monkeypatch, data: dict):
    import pulse.knowledge.cohort_statistics as registry

    specs = read_finding_from(data).specs
    monkeypatch.setattr(registry, "ALL_COHORT_STATISTICS", [*registry.ALL_COHORT_STATISTICS, *specs])


def read_finding_from(data: dict):
    from pulse.knowledge.corpus import parse_finding
    return parse_finding(data)


def _rival(peak_mmol: float, **extra) -> dict:
    data = _data()
    data["paper"], data["citation"]["year"], data["citation"]["doi"] = "synthetic-test-2101", 2101, "10.0000/s.2101"
    data["protocols"] = [dict(data["protocols"][0], name="oral_load")]       # the same protocol, another name
    data["findings"] = [dict(
        data["findings"][0], id="peak", protocols=["oral_load"], value=peak_mmol,
        uncertainty={"kind": "sd", "value": 0.8, "n": 10}, **extra,
    )]
    return data


def test_a_finding_that_contradicts_an_admitted_one_is_refused(monkeypatch):
    _admitted(monkeypatch, _data())
    ok = gate._conflict_check(read_finding_from(_rival(8.4)))
    assert ok.ok
    bad = gate._conflict_check(read_finding_from(_rival(12.0)))
    assert not bad.ok
    assert "peak contradicts synthetic-test-2099.ogtt_glucose_peak" in bad.reasons[0]
    assert "acknowledged_conflicts" in bad.reasons[0]


def test_a_conflict_is_judged_on_the_standard_error_not_the_between_person_sd(monkeypatch):
    # Means 144 and 181 mg/dL with SDs of 22 and 14: the SD intervals overlap, the
    # standard errors (6.2, 4.4) do not.
    _admitted(monkeypatch, _data())
    assert not gate._conflict_check(read_finding_from(_rival(10.0))).ok


def test_an_acknowledged_conflict_is_recorded_not_silent(monkeypatch):
    _admitted(monkeypatch, _data())
    named = _rival(12.0, acknowledged_conflicts=["synthetic-test-2099.ogtt_glucose_peak"])
    assert gate._conflict_check(read_finding_from(named)).ok


def test_an_acknowledgement_of_a_conflict_that_is_not_there_is_refused(monkeypatch):
    _admitted(monkeypatch, _data())
    named = _rival(8.4, acknowledged_conflicts=["synthetic-test-2099.ogtt_glucose_peak"])
    check = gate._conflict_check(read_finding_from(named))
    assert not check.ok and "do not conflict" in check.reasons[0]


def test_other_protocols_do_not_conflict(monkeypatch):
    _admitted(monkeypatch, _data())
    other = _rival(12.0)
    other["protocols"][0]["meals"][0]["carbs_g"] = 50.0
    assert gate._conflict_check(read_finding_from(other)).ok


def test_a_papers_own_earlier_file_is_not_a_conflict_with_itself(monkeypatch):
    _admitted(monkeypatch, _data())
    assert gate._conflict_check(read_finding_from(_data())).ok


# --- refusals -------------------------------------------------------------------------

def test_a_refusal_is_validated_and_admitted(root):
    result = admit(_put(root, json.loads(REFUSAL.read_text(encoding="utf-8")), "refusals"), root=root)
    assert result.ok and result.kind == "refusal"
    assert [c.name for c in result.checks] == ["parse", "filing"]


def test_a_bad_refusal_is_refused(root):
    data = json.loads(REFUSAL.read_text(encoding="utf-8"))
    data["missing"]["kind"] = "assay"
    result = admit(_put(root, data, "refusals"), root=root)
    assert "not one of marker, input, population, timescale, operator" in _failed(result)["parse"][0]


def test_the_cli_exit_codes_and_refusal_counts(root, capsys):
    good = _put(root, json.loads(REFUSAL.read_text(encoding="utf-8")), "refusals")
    assert main([str(good)]) == 0
    assert "admitted" in capsys.readouterr().out
    bad = root / "refusals" / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    assert main([str(bad)]) == 1


def test_refusals_count_by_kind(root, monkeypatch, capsys):
    monkeypatch.setenv("PULSE_CORPUS_DIR", str(root))
    _put(root, json.loads(REFUSAL.read_text(encoding="utf-8")), "refusals")
    assert main(["--refusals"]) == 0
    out = capsys.readouterr().out
    assert "1 refusals" in out and "marker: 1" in out and "adiponectin" in out


# --- what the gate never does -----------------------------------------------------------

def test_the_gate_prints_nothing_about_fit(root, capsys, full_run, stub_rollouts):
    path = _put(root, _data())
    assert main([str(path)]) == 0
    out = capsys.readouterr().out + format_admission(full_run)
    assert not re.search(r"\b(loss|residual|predicted|prediction|z-score|fit|score)\b", out, re.I)
    assert "144" not in out and "21.6" not in out        # no target or sigma either


def test_the_gate_never_reads_a_checkpoint():
    text = Path(gate.__file__).read_text(encoding="utf-8")
    for forbidden in ("torch.load", "load_state_dict", "checkpoint", "gcs", "storage."):
        assert forbidden not in text.replace("never a checkpoint", ""), forbidden


# --- the queue ------------------------------------------------------------------------

QUEUE = FIXTURES / "queue" / "synthetic-test-2097.json"


def test_a_queue_file_is_validated_and_admitted(root):
    (root / "queue").mkdir(exist_ok=True)
    path = root / "queue" / "synthetic-test-2097.json"
    path.write_text(QUEUE.read_text(encoding="utf-8"), encoding="utf-8")
    result = admit(path, root=root)
    assert result.ok and result.kind == "queue" and [c.name for c in result.checks] == ["parse", "filing"]


def test_a_queue_file_with_a_duplicate_identifier_or_registry_slug_is_refused(root):
    (root / "queue").mkdir(exist_ok=True)
    _put(root, _data())
    twin = json.loads(QUEUE.read_text(encoding="utf-8"))
    twin["doi"] = _data()["citation"]["doi"]
    path = root / "queue" / "synthetic-test-2097.json"
    path.write_text(json.dumps(twin), encoding="utf-8")
    assert "another slug" in _failed(admit(path, root=root))["filing"][0]
    twin = json.loads(QUEUE.read_text(encoding="utf-8"))
    twin["paper"], twin["citation"] = "wahren-1971", "Wahren J et al. 1971"
    twin["doi"], twin["pmcid"] = "10.0000/w.1", None
    path = root / "queue" / "wahren-1971.json"
    path.write_text(json.dumps(twin), encoding="utf-8")
    assert "Python registry" in _failed(admit(path, root=root))["filing"][0]


def test_filing_a_finding_for_a_queued_paper_says_to_delete_the_queue_file(root):
    (root / "queue").mkdir(exist_ok=True)
    queued = json.loads(QUEUE.read_text(encoding="utf-8"))
    queued["paper"], queued["citation"] = "synthetic-test-2099", queued["citation"].replace("2097", "2099")
    queued["doi"], queued["pmcid"] = "10.0000/q.1", None
    (root / "queue" / "synthetic-test-2099.json").write_text(json.dumps(queued), encoding="utf-8")
    result = admit(_put(root, _data()), root=root, fed_day_minutes=0)
    assert "delete it" in _failed(result)["filing"][0]


def test_the_piles_are_countable_from_the_command_line(root, monkeypatch, capsys):
    monkeypatch.setenv("PULSE_CORPUS_DIR", str(root))
    (root / "queue").mkdir(exist_ok=True)
    (root / "queue" / "synthetic-test-2097.json").write_text(QUEUE.read_text(encoding="utf-8"), encoding="utf-8")
    assert main(["--piles"]) == 0
    assert capsys.readouterr().out.split() == ["queue:", "1", "findings:", "0", "refusals:", "0"]
