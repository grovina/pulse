"""The guide a reader is handed: it lists every unit the loader converts, and its examples parse."""

from __future__ import annotations

import json
import re
from pathlib import Path

from pulse.knowledge.corpus import parse_finding, parse_refusal
from pulse.knowledge.corpus_units import CONVERSIONS

ROOT = Path(__file__).resolve().parents[1]


def test_the_readme_lists_every_unit_it_converts():
    text = (ROOT / "corpus" / "README.md").read_text(encoding="utf-8").lower().replace("µ", "u").replace("μ", "u")
    text = re.sub(r"\s+", "", text)
    for series, table in CONVERSIONS.items():
        assert f"`{series}`" in (ROOT / "corpus" / "README.md").read_text(encoding="utf-8"), series
        for unit in table:
            assert unit.replace(" ", "") in text, (series, unit)


def test_the_readme_examples_parse():
    text = (ROOT / "corpus" / "README.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```json\n(.*?)\n```", text, re.S)
    worked = [json.loads(b) for b in blocks if '"paper": "example-2099"' in b]
    refusal = [json.loads(b) for b in blocks if '"paper": "example-2098"' in b]
    assert len(worked) == 1 and len(refusal) == 1
    assert len(parse_finding(worked[0]).items) == 3
    assert parse_refusal(refusal[0]).kind == "marker"
