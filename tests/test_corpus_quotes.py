"""The quote check: what counts as a quote being in the paper, and how the gate uses it."""

from __future__ import annotations

import copy
import html
import json
from pathlib import Path

import pytest

import pulse.admit as gate
from pulse.admit import admit, format_admission, main
from pulse.knowledge.corpus_quotes import FOUND, MISSING, NUMBERS, paper_text, quote_verdict

FIXTURES = Path(__file__).parent / "fixtures" / "corpus"
FINDING = FIXTURES / "findings" / "synthetic-test-2099.json"
REFUSAL = FIXTURES / "refusals" / "synthetic-test-2098.json"
QUEUE = FIXTURES / "queue" / "synthetic-test-2097.json"

PAPER = paper_text(
    "<article><p>Peak glucose was 8.0 (SD&#160;1.2) mmol/L, a change of "
    "−1.4 ± 0.2 from <italic>baseline</italic>.</p>"
    "<table><tr><th>AUC</th><th>Lunch</th><th>No lunch</th></tr>"
    "<tr><td>Early</td><td>37.8 ± 4.9</td><td>30.2 ± 2.9</td></tr>"
    "<tr><td>12</td><td>5</td></tr></table>"
    "<p>Insulin was 61.5 pmol/L." + " filler" * 200 + " Leptin was 9.75 ng/mL.</p></article>"
)


# --- one quote against a paper ---------------------------------------------------------

def test_a_sentence_is_found_through_markup_and_typography():
    quote = "Peak glucose was 8.0 (SD 1.2) mmol/L, a change of -1.4 ± 0.2 from baseline"
    assert quote_verdict(quote, PAPER) == (FOUND, "")


def test_a_table_row_is_recognised_by_its_numbers():
    assert quote_verdict("Early | 37.8 ± 4.9 | 30.2 ± 2.9", PAPER) == (NUMBERS, "")


def test_a_number_the_paper_does_not_print_is_missing_and_named():
    verdict, reason = quote_verdict("Early | 37.8 ± 4.9 | 31.2 ± 2.9", PAPER)
    assert verdict == MISSING and "31.2" in reason and "37.8" not in reason


def test_neighbouring_cells_do_not_run_together():
    verdict, reason = quote_verdict("the count was 125", PAPER)
    assert verdict == MISSING and "125" in reason


def test_numbers_the_paper_prints_far_apart_are_not_one_quote():
    verdict, reason = quote_verdict("insulin 61.5 and leptin 9.75", PAPER)
    assert verdict == MISSING and "nowhere together" in reason


def test_a_number_is_matched_whole_not_inside_a_longer_one():
    verdict, reason = quote_verdict("Early | 7.8 ± 4.9", PAPER)     # the paper prints 37.8
    assert verdict == MISSING and "7.8" in reason


def test_a_hyphen_before_a_number_may_be_a_range_or_a_minus_sign():
    paper = paper_text("<p>The area over 0–30 min was −3.3 ± 7.7 in the obese.</p>")
    assert quote_verdict("iAUC 30 min | -3.3 ± 7.7", paper)[0] == NUMBERS   # "0-30": the 30 is printed
    assert quote_verdict("iAUC | 3.3 ± 7.7", paper)[0] == NUMBERS            # the sign is the second reader's
    assert quote_verdict("iAUC | -7.7", paper)[0] == MISSING                 # but a minus the paper lacks is not


# --- the gate, given the paper ---------------------------------------------------------

@pytest.fixture
def root(tmp_path):
    for sub in ("findings", "refusals", "queue"):
        (tmp_path / sub).mkdir()
    return tmp_path


@pytest.fixture
def stub_rollouts(monkeypatch):
    monkeypatch.setattr(gate, "_run_checks", lambda finding, device, fed: (gate.Check("runs", True), gate.Check("gradient", True)))


def _finding() -> dict:
    data = copy.deepcopy(json.loads(FINDING.read_text(encoding="utf-8")))
    data["findings"] = [f for f in data["findings"] if f["id"] in ("ogtt_glucose_peak", "fasting_insulin_spread")]
    data["protocols"] = [p for p in data["protocols"] if p["name"] in ("ogtt", "fasting")]
    return data


def _put(root: Path, data: dict, sub: str) -> Path:
    path = root / sub / f"{data['paper']}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _paper(root: Path, quotes: list[str]) -> Path:
    path = root / "fulltext.xml"
    path.write_text("<article>" + "".join(f"<p>{html.escape(q)}</p>" for q in quotes) + "</article>", encoding="utf-8")
    return path


def test_a_file_whose_quotes_are_in_the_paper_is_admitted(root, stub_rollouts):
    data = _finding()
    paper = _paper(root, [f["location"]["quote"] for f in data["findings"]])
    result = admit(_put(root, data, "findings"), root=root, paper_file=paper)
    assert result.ok, format_admission(result)
    assert [c.name for c in result.checks] == [
        "parse", "protocol", "units", "place", "uncertainty", "quotes", "filing", "runs", "gradient", "conflicts",
    ]
    assert "each quote is in the paper (2 word for word, 0 by their numbers)" in format_admission(result)


def test_without_the_paper_there_is_no_quote_check(root, stub_rollouts):
    result = admit(_put(root, _finding(), "findings"), root=root)
    assert result.ok and "quotes" not in [c.name for c in result.checks]


def test_a_quote_that_is_not_in_the_paper_stops_the_gate_before_the_model_runs(root, monkeypatch):
    def never(*_args):
        raise AssertionError("the model was rolled for a file whose quotes are not in the paper")

    monkeypatch.setattr(gate, "_run_checks", never)
    data = _finding()
    paper = _paper(root, [f["location"]["quote"] for f in data["findings"]])
    data["findings"][0]["location"]["quote"] = "Peak glucose was 8.3 (SD 1.2) mmol/L (n = 12)."
    result = admit(_put(root, data, "findings"), root=root, paper_file=paper)
    failed = {c.name: c.reasons for c in result.failed}
    assert set(failed) == {"quotes"}
    assert failed["quotes"] == ("ogtt_glucose_peak: the paper nowhere prints 8.3",)
    assert [c.name for c in result.checks if c.skipped] == ["runs", "gradient", "conflicts"]


def test_a_paper_that_cannot_be_read_fails_the_quote_check(root, stub_rollouts):
    result = admit(_put(root, _finding(), "findings"), root=root, paper_file=root / "absent.xml")
    assert not result.ok and "cannot read the paper" in result.failed[0].reasons[0]


def test_a_refusals_quotes_are_checked_too(root):
    data = json.loads(REFUSAL.read_text(encoding="utf-8"))
    quotes = [r["location"]["quote"] for r in data["reported"]]
    path = _put(root, data, "refusals")
    assert admit(path, root=root, paper_file=_paper(root, quotes)).ok
    result = admit(path, root=root, paper_file=_paper(root, ["Nothing of the kind was measured in 2099."]))
    assert [c.name for c in result.failed] == ["quotes"]


def test_a_queue_item_has_no_quotes_to_check(root):
    path = _put(root, json.loads(QUEUE.read_text(encoding="utf-8")), "queue")
    result = admit(path, root=root, paper_file=root / "absent.xml")
    assert result.ok and "quotes" not in [c.name for c in result.checks]


def test_the_command_line_takes_the_paper(root, capsys, stub_rollouts, monkeypatch):
    monkeypatch.setenv("PULSE_CORPUS_DIR", str(root))
    data = _finding()
    paper = _paper(root, [f["location"]["quote"] for f in data["findings"]])
    assert main([str(_put(root, data, "findings")), "--paper", str(paper)]) == 0
    assert "each quote is in the paper" in capsys.readouterr().out
