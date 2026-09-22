"""The sealed chain judges a course. It does not train, and it does not gate."""

from pulse.knowledge.emergence_chain import judge_chain
from pulse.train import resume_start_epoch


def _course(**hours: dict[str, float]) -> dict[str, dict[str, float]]:
    return {f"{h}h": values for h, values in hours.items()}


_FED = {
    "glucose": 95.0, "insulin": 10.0, "glucagon": 70.0,
    "ffa": 0.5, "bhb": 0.1, "liver_glycogen": 100.0, "cortisol": 12.0,
}


def _fasting_like() -> dict[str, dict[str, float]]:
    h16 = {**_FED, "glucose": 85.0, "insulin": 6.0, "glucagon": 84.0, "ffa": 0.69, "bhb": 1.2, "liver_glycogen": 47.0}
    h24 = {**h16, "glucose": 86.0, "insulin": 6.0, "glucagon": 84.0, "bhb": 1.6, "liver_glycogen": 30.0}
    h48 = {**h16, "glucose": 78.0, "insulin": 3.8, "glucagon": 95.0, "ffa": 0.85, "bhb": 3.1, "liver_glycogen": 7.0}
    return _course(**{"16": h16, "24": h24, "48": h48})


def test_teacher_shaped_course_passes():
    judged = judge_chain(_fasting_like(), gb=95.0, ib=10.0, ffa_b=0.5, gn_b=70.0)
    assert judged["passed"]
    assert judged["first_failure"] is None
    assert [lnk["name"] for lnk in judged["links"]][2] == "glucagon_up_16h"


def test_a_return_to_the_setpoint_is_not_still_down():
    course = _fasting_like()
    course["24h"] = {**course["24h"], "glucose": 94.9}
    judged = judge_chain(course, gb=95.0, ib=10.0, ffa_b=0.5, gn_b=70.0)
    assert judged["first_failure"] == "glucose_still_down_24h"


def test_rebound_fails_at_the_next_morning_before_bhb():
    course = _fasting_like()
    course["24h"] = {**course["24h"], "glucose": 95.0}
    course["48h"] = {**course["48h"], "glucose": 93.0, "insulin": 10.2, "bhb": 1.1, "liver_glycogen": 38.0}
    judged = judge_chain(course, gb=95.0, ib=10.0, ffa_b=0.5, gn_b=70.0)
    assert judged["first_failure"] == "glucose_still_down_24h"
    assert not judged["passed"]


def test_glucagon_that_never_rises_is_the_first_failure_when_glucose_and_insulin_did():
    course = _fasting_like()
    course["16h"] = {**course["16h"], "glucagon": 54.0}
    judged = judge_chain(course, gb=95.0, ib=10.0, ffa_b=0.5, gn_b=70.0)
    assert judged["first_failure"] == "glucagon_up_16h"


def test_resume_starts_at_the_epoch_after_the_one_that_finished():
    assert resume_start_epoch({"epoch": 55}) == 56
    try:
        resume_start_epoch({})
    except ValueError:
        return
    raise AssertionError("a checkpoint with no epoch should not resume")


def test_bhb_below_the_cahill_floor_fails_last():
    course = _fasting_like()
    course["48h"] = {**course["48h"], "bhb": 1.24}
    judged = judge_chain(course, gb=95.0, ib=10.0, ffa_b=0.5, gn_b=70.0)
    assert judged["first_failure"] == "bhb_cahill_48h"
