"""Leptin falls by most of its level in a long fast (deru-2023, Boden 1996), and the fed side
keeps its old size: the insulin term scales the fat-set level instead of adding 2.5 ng/mL."""
from pulse.modules.appetite import (
    _LEP_INS_GAIN_FAST, _LEP_INS_GAIN_FED,
)


def _target_factor(r: float) -> float:
    gain = _LEP_INS_GAIN_FED if r >= 0 else _LEP_INS_GAIN_FAST
    return max(0.05, 1.0 + gain * r)


def test_fed_side_is_the_old_additive_size_at_typical_leptin():
    # +2.5 at the 10 ng/mL typical level, per unit of (insulin_slow/ib - 1)
    assert abs(_target_factor(1.0) - 1.25) < 1e-9


def test_a_16_to_24_h_fast_takes_30_to_50_percent_and_a_long_one_most():
    assert 0.5 <= _target_factor(-0.5) <= 0.7      # insulin_slow at 0.5 ib
    assert _target_factor(-1.0) < 0.25             # insulin_slow at zero


def test_teacher_matches_the_student_factor():
    from pulse.knowledge.full_body import PatientParams
    p = PatientParams()
    assert p.lep_ins_gain == _LEP_INS_GAIN_FED and p.lep_ins_gain_fast == _LEP_INS_GAIN_FAST
