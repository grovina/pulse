"""Every corpus file against the gate it was admitted through.

A file is gated once, when it is filed. Code changes after that can make a marker or a
statistic stop moving with the parameters, and nothing says so: hedrington-2015's
``gir_final`` went dead when exercise began to lower basal insulin (dcf7ec2), and was
found by accident 40 commits later. The full sweep takes about 25 CPU-minutes (7 s a
finding file, a minute for the largest), so it is behind ``PULSE_GATE_SWEEP=1`` and is run
after a change to the metabolic, cardiovascular or hormone modules:

    PULSE_GATE_SWEEP=1 uv run pytest -q tests/test_corpus_gate_sweep.py

The default run gates three small files that cover a hormone, a clamp and a heart-rate
finding, and the one file known to fail, so that it is recorded here and the list shrinks
when the model recovers.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from pulse.admit import admit

ROOT = Path(__file__).resolve().parents[1] / "corpus"

# {paper: {finding id: why the model cannot move it}}. A finding listed here must fail
# the movement check on exactly these ids; one that starts moving again must be removed.
KNOWN_DEAD = {
    "hedrington-2015": {
        "gir_final": "a clamp infuses only to lift glucose to its target; since exercise lowers basal "
                     "insulin (dcf7ec2) glucose runs above the target through the exercise "
                     "(95 -> 103 mg/dL against 95.5), so the infusion is exactly 0 and its relu has no gradient",
    },
}
FAST = ("ali-2026", "anderwald-2011", "decker-2024")


def _gate(pile: str, slug: str):
    return admit(ROOT / pile / f"{slug}.json")


def _dead_ids(result) -> set[str]:
    gradient = next(c for c in result.checks if c.name == "gradient")
    return {line.split(":")[0] for line in gradient.reasons}


@pytest.mark.parametrize("slug", FAST)
def test_a_small_file_still_passes_the_gate(slug):
    result = _gate("findings", slug)
    assert result.ok, [(c.name, c.reasons) for c in result.failed]


@pytest.mark.parametrize("slug", sorted(KNOWN_DEAD))
def test_a_known_dead_finding_is_dead_for_the_reason_recorded(slug):
    result = _gate("findings", slug)
    assert [c.name for c in result.failed] == ["gradient"]
    assert _dead_ids(result) == set(KNOWN_DEAD[slug]), "a finding moves again: drop it from KNOWN_DEAD"


@pytest.mark.skipif(os.environ.get("PULSE_GATE_SWEEP") != "1", reason="25 CPU-minutes; set PULSE_GATE_SWEEP=1")
@pytest.mark.parametrize("pile", ("findings", "refusals"))
def test_every_file_passes_the_gate_but_the_known_dead(pile):
    failures = {}
    for path in sorted((ROOT / pile).glob("*.json")):
        result = _gate(pile, path.stem)
        if result.ok:
            assert path.stem not in KNOWN_DEAD, f"{path.stem} passes again: drop it from KNOWN_DEAD"
            continue
        if path.stem in KNOWN_DEAD and [c.name for c in result.failed] == ["gradient"] \
                and _dead_ids(result) == set(KNOWN_DEAD[path.stem]):
            continue
        failures[path.stem] = [(c.name, c.reasons) for c in result.failed]
    assert not failures
