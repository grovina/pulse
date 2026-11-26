"""RQ follows the fuel mix: lower after a long fast, higher under high insulin, unchanged at the
12 h fasted level (the old constant 0.807 stood for that)."""
import torch

from pulse.modules.metabolic import _RQ_INS_FASTED, rq_fuel_shift_t


def _s(x):
    return float(rq_fuel_shift_t(torch.tensor(x)))


def test_zero_at_the_fasted_insulin():
    assert abs(_s(_RQ_INS_FASTED)) < 1e-9


def test_a_72_h_fast_takes_rq_to_about_0_74():
    assert -0.08 < _s(0.2) < -0.05            # 0.807 -> ~0.74


def test_high_insulin_raises_rq_and_is_capped():
    assert 0.07 < _s(6.0) < 0.12
    assert _s(1000.0) <= 0.12 + 1e-9
