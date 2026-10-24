"""scripts/ingest_cgmacros.py: one person's morning becomes a calibrate-then-predict episode."""

from __future__ import annotations

import importlib.util
import json
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from pulse.benchmark import load_benchmark_dataset, subject_of
from pulse.knowledge.evidence import REAL_EVIDENCE, Authority

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ingest_cgmacros.py"
_spec = importlib.util.spec_from_file_location("ingest_cgmacros", _SCRIPT)
ingest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ingest)

_HEADER = (
    "Unnamed: 0,Timestamp,Libre GL,Dexcom GL,HR,Calories (Activity),METs,Meal Type,"
    "Calories,Carbs,Protein,Fat,Fiber,Amount Consumed ,Image path"
)
_BIO = "subject,Age,Gender,BMI,Body weight ,Height ,Self-identify ,A1c PDL (Lab)\n1,30,F,22,130,64,White,5.2\n"


def _day(start: datetime, minutes: int, meals: dict[int, tuple[str, float, float, float, float]], gap_at=None) -> list[str]:
    lines = []
    for i in range(minutes):
        if gap_at is not None and i == gap_at:
            continue
        ts = start + timedelta(minutes=i)
        kind, carbs, prot, fat, share = meals.get(i, ("", "", "", "", ""))
        mets = 40.0 if 100 <= i < 110 else 10.0
        lines.append(
            f"{i},{ts:%Y-%m-%d %H:%M:%S},{90 + (i % 30) / 3:.2f},,{60 + i % 5},1.0,{mets},"
            f"{kind},,{carbs},{prot},{fat},,{share},"
        )
    return lines


def _zip(tmp_path: Path, rows: list[str]) -> Path:
    path = tmp_path / "CGMacros.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("CGMacros/bio.csv", _BIO)
        zf.writestr("CGMacros/CGMacros-001/CGMacros-001.csv", "\n".join([_HEADER, *rows]) + "\n")
    return path


_MORNING = {
    60: ("Breakfast", 40.0, 20.0, 10.0, 100.0),    # 08:00
    150: ("snack", 10.0, 0.0, 0.0, 50.0),
    300: ("Lunch", 80.0, 30.0, 30.0, 100.0),       # 12:00, four hours later
}


def test_a_morning_calibrates_on_breakfast_and_predicts_lunch(tmp_path):
    rows = _day(datetime(2020, 5, 2, 7, 0), 600, _MORNING)
    episodes, meta = ingest.build_cgmacros_episodes(_zip(tmp_path, rows), per_person=1)
    assert len(episodes) == 1 and meta["episodes_by_group"] == {"healthy": 1}
    ep = episodes[0]
    assert subject_of(ep["user_id"]) == "cgmacros-001"
    assert ep["start_time_minutes"] == 7 * 60 + 30
    lunch_t = 300 - 60 + ingest.LEAD_IN_MIN
    assert ep["duration_min"] == lunch_t + ingest.EVAL_HORIZON_MIN
    assert [m["time"] for m in ep["meals"]] == [30.0, 120.0, float(lunch_t)]
    assert ep["meals"][1]["carbs"] == pytest.approx(5.0)  # half the snack was eaten
    assert all(c["time"] < lunch_t for c in ep["calibration_check_ins"])
    assert all(m["time"] > lunch_t for m in ep["eval_measurements"])
    assert {m["marker_id"] for m in ep["eval_measurements"]} == {"glucose", "hr"}
    assert ep["activity"][100 - 30] == pytest.approx(3.0 / 8.0)
    assert ep["activity"][0] == 0.0


def test_a_gap_in_the_minutes_or_lunch_too_soon_is_no_episode(tmp_path):
    gapped = _day(datetime(2020, 5, 2, 7, 0), 600, _MORNING, gap_at=200)
    assert ingest.build_cgmacros_episodes(_zip(tmp_path, gapped), per_person=1)[0] == []
    soon = {**_MORNING, 300: ("", "", "", "", ""), 150: ("Lunch", 80.0, 30.0, 30.0, 100.0)}
    rows = _day(datetime(2020, 5, 2, 7, 0), 600, soon)
    assert ingest.build_cgmacros_episodes(_zip(tmp_path, rows), per_person=1)[0] == []


def test_the_episodes_load_as_a_real_source(tmp_path):
    rows = _day(datetime(2020, 5, 2, 7, 0), 600, _MORNING)
    episodes, meta = ingest.build_cgmacros_episodes(_zip(tmp_path, rows), per_person=1)
    path = tmp_path / "bench.json"
    path.write_text(json.dumps({"meta": meta, "episodes": episodes}))
    (loaded,) = load_benchmark_dataset(str(path))
    assert loaded.source == ingest.SOURCE
    assert ingest.SOURCE in {item.name for item in REAL_EVIDENCE}
    assert all(item.authority is Authority.REAL for item in REAL_EVIDENCE)
