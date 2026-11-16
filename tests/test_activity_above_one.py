"""The activity scale runs to 2.0: a hard bout above the old cap of 1.0 can be given.

Level 1.0 is VO2 35.4 mL/kg/min, below a fit person exercising hard (li-2024's run at
65% of VO2max is 1.025). The loader accepts levels to 2.0 (VO2 67 mL/kg/min) and
refuses above. Above 1.0 the model is extrapolating: these tests ask only that every
state stays finite and that oxygen uptake, which is mechanistic, keeps rising.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from pulse.knowledge.corpus import CorpusError, parse_finding
from pulse.model import ModularPhysiologyNetwork, integrate
from pulse.types import EMBEDDING_DIM, NORM_CENTER, SERIES_INDEX


FINDING = Path(__file__).parent / "fixtures" / "corpus" / "findings" / "synthetic-test-2099.json"
_T = 40
_VO2 = SERIES_INDEX["vo2"]


def _data() -> dict:
    return copy.deepcopy(json.loads(FINDING.read_text(encoding="utf-8")))


def _with_activity(level: float) -> dict:
    data = _data()
    data["protocols"][1]["activity"] = {"segments": [{"from_min": 60, "to_min": 90, "level": level}]}
    return data


def test_the_loader_takes_a_level_up_to_two():
    arm = parse_finding(_with_activity(2.0)).protocols["fasting"]
    assert arm.activity[60] == 2.0 and arm.activity[59] == 0.0


def test_the_loader_refuses_a_level_above_two():
    with pytest.raises(CorpusError) as err:
        parse_finding(_with_activity(2.01))
    assert any("is above 2" in r for r in err.value.reasons)


def _roll(level: float) -> torch.Tensor:
    torch.manual_seed(0)
    model = ModularPhysiologyNetwork()
    state = torch.tensor(NORM_CENTER, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        return integrate(model, state, torch.zeros(1, EMBEDDING_DIM), _T,
                         activity=torch.full((_T,), level), observe=True)


def test_every_state_stays_finite_up_to_two():
    for level in (1.0, 1.5, 2.0):
        assert torch.isfinite(_roll(level)).all(), level


def test_oxygen_uptake_keeps_rising_above_one():
    vo2 = [_roll(level)[0, -1, _VO2].item() for level in (0.5, 1.0, 1.5, 2.0)]
    assert vo2 == sorted(vo2) and vo2[-1] > vo2[1] * 1.3, vo2
