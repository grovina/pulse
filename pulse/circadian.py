"""The clock and one night's sleep debt.

Donga et al., J Clin Endocrinol Metab 2010;95:2963-2968, Table 1. Nine
healthy subjects, one night. Sleep allowed 23:00–07:30 or, on the other
occasion, 01:00–05:00. Both nights are centered at 03:00, so the comparison
is duration, not phase. The next afternoon's clamp (insulin 22.0 vs 21.4
mU/L, basal insulin 3.1 vs 3.6):

    glucose disposal     40.7 ± 5.1   vs  32.5 ± 3.6 µmol/kg LBM/min
    endogenous production 3.6 ± 0.2   vs   4.4 ± 0.3
    NEFA                  57 ± 4      vs    68 ± 5 µmol/L
    glucose infusion      36.9 ± 5.1  vs   27.8 ± 3.7

Basal glucose, insulin, NEFA, endogenous production, glucagon and cortisol
were unchanged. Debt therefore scales insulin-dependent disposal, the
insulin suppression of hepatic output, and the insulin suppression of
lipolysis. Each factor is 1 at no debt and at basal insulin, which is the
basal row of the table. Cortisol is not given a debt gain: the table's
cortisol means are 349 ± 59 and 395 ± 59 nmol/L (P = 0.31).

The phase state is hours, not an angle. A night centered on 03:00 leaves it
alone. A later night pulls it later, and the modules that read the hour
read hour minus that offset.
"""

from __future__ import annotations

import math

import torch

# An 8 h night against this need commits no debt. The restricted arm of the
# table is 4 h of allowed sleep, so the debt that carries the clamp ratios
# is 4 h.
NEED_H = 8.0
DEBT_OF_RESTRICTED_NIGHT_H = 4.0
NIGHT_OPEN_H = 21.0
NIGHT_COMMIT_H = 9.0
PHASE_REF_H = 3.0
PHASE_BOUND_H = 6.0
DEBT_BOUND_H = 16.0
# Marks a night as open with none of it slept yet. The committed value is
# never negative: the shortfall replaces this count at 09:00.
NIGHT_OPEN_DEBT = -1e-6
# Hours of phase per hour asleep when the timing error is a quarter-cycle.
# Not a published rate. One night centered 2 h late moves the offset about
# half an hour; a night centered at 03:00 does not.
PHASE_PULL_PER_H = 0.125

# Table 1, the hyperinsulinaemic steady state.
RD_NORMAL = 40.7
RD_SHORT = 32.5
CLAMP_INSULIN_FOLD = 22.0 / 3.1

# Solved so that, at DEBT_OF_RESTRICTED_NIGHT_H and at CLAMP_INSULIN_FOLD
# times basal insulin, the teacher's hepatic target rises by 4.4/3.6 and
# the lipolysis equilibrium rises by 68/57. tests/test_circadian.py
# recomputes both ratios from the live gates.
HEPATIC_K = 0.009593854925064564
LIPOLYSIS_K = 0.012160994006257557

PERIPHERAL_K = -math.log(RD_SHORT / RD_NORMAL) / DEBT_OF_RESTRICTED_NIGHT_H


def _owed(debt: float) -> float:
    return debt if debt > 0.0 else 0.0


def peripheral_factor(debt: float) -> float:
    """Insulin-dependent disposal, 1 at no debt. 32.5/40.7 at 4 h."""
    return math.exp(-PERIPHERAL_K * _owed(debt))


def peripheral_factor_t(debt: torch.Tensor) -> torch.Tensor:
    return torch.exp(-PERIPHERAL_K * torch.relu(debt))


def retain_basal(value: float, debt: float, k: float) -> float:
    """Move ``value`` toward 1 as debt rises. Identity at debt 0 and at value 1."""
    f = math.exp(-k * _owed(debt))
    return f * value + (1.0 - f)


def retain_basal_t(value: torch.Tensor, debt: torch.Tensor, k: float) -> torch.Tensor:
    f = torch.exp(-k * torch.relu(debt))
    return f * value + (1.0 - f)


def lipolysis_scale(insulin: float, basal_insulin: float, ic50: float, debt: float) -> float:
    """Multiplier on the basal-normalised lipolysis rate. 1 at basal insulin."""
    s = (1.0 + basal_insulin / ic50) / (1.0 + insulin / ic50)
    return retain_basal(s, debt, LIPOLYSIS_K) / s


def lipolysis_scale_t(
    insulin: torch.Tensor,
    basal_insulin: torch.Tensor,
    ic50: torch.Tensor,
    debt: torch.Tensor,
) -> torch.Tensor:
    s = (1.0 + basal_insulin / ic50) / (1.0 + insulin / ic50)
    return retain_basal_t(s, debt, LIPOLYSIS_K) / s


def _wrap24(hour: float) -> float:
    return hour % 24.0


def _on_mark(hour: float, mark_h: float) -> bool:
    """True on the minute whose clock label is ``mark_h``. A decode error of
    half a minute still lands on the same minute and not the next one."""
    minute = (hour * 60.0) % 1440.0
    delta = (minute - mark_h * 60.0 + 720.0) % 1440.0 - 720.0
    return abs(delta) <= 0.5


def _in_night(bio: float) -> bool:
    return bio >= NIGHT_OPEN_H or bio < NIGHT_COMMIT_H


def step_clock(offset: float, debt: float, hour: float, asleep: float) -> tuple[float, float]:
    """One minute. ``hour`` is the wall clock in [0, 24). ``asleep`` is in [0, 1].

    During the biological night the debt state counts hours slept, as a
    negative number. At 09:00 that count becomes the shortfall against 8 h,
    which is what the sensitivity factors read. A night that has not opened
    does not commit, so a morning that starts the simulation is not charged
    for sleep that never had a bedtime.
    """
    hour = _wrap24(hour)
    asleep = min(max(asleep, 0.0), 1.0)
    bio = _wrap24(hour - offset)
    dev = _wrap24(bio - PHASE_REF_H + 12.0) - 12.0
    pull = math.sin(2.0 * math.pi * dev / 24.0)
    new_offset = offset + asleep * (PHASE_PULL_PER_H / 60.0) * pull
    new_offset = min(max(new_offset, -PHASE_BOUND_H), PHASE_BOUND_H)

    new_debt = debt
    if _on_mark(bio, NIGHT_OPEN_H):
        new_debt = NIGHT_OPEN_DEBT
    elif _on_mark(bio, NIGHT_COMMIT_H) and new_debt < 0.0:
        new_debt = max(0.0, NEED_H + new_debt)
    elif _in_night(bio) and new_debt < 0.0:
        new_debt -= asleep / 60.0
    new_debt = min(max(new_debt, -DEBT_BOUND_H), DEBT_BOUND_H)
    return new_offset, new_debt


def _on_mark_t(hour: torch.Tensor, mark_h: float) -> torch.Tensor:
    minute = torch.remainder(hour * 60.0, 1440.0)
    delta = torch.remainder(minute - mark_h * 60.0 + 720.0, 1440.0) - 720.0
    return delta.abs() <= 0.5


def step_clock_t(
    offset: torch.Tensor,
    debt: torch.Tensor,
    hour: torch.Tensor,
    asleep: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tensor form of ``step_clock``. Same branches, same constants."""
    hour = torch.remainder(hour, 24.0)
    asleep = asleep.clamp(0.0, 1.0)
    bio = torch.remainder(hour - offset, 24.0)
    dev = torch.remainder(bio - PHASE_REF_H + 12.0, 24.0) - 12.0
    pull = torch.sin(2.0 * math.pi * dev / 24.0)
    new_offset = (offset + asleep * (PHASE_PULL_PER_H / 60.0) * pull).clamp(
        -PHASE_BOUND_H, PHASE_BOUND_H)

    opened = _on_mark_t(bio, NIGHT_OPEN_H)
    commit = _on_mark_t(bio, NIGHT_COMMIT_H)
    counting = (bio >= NIGHT_OPEN_H) | (bio < NIGHT_COMMIT_H)
    counted = debt - asleep / 60.0
    committed = torch.clamp(NEED_H + debt, min=0.0)
    new_debt = torch.where(counting & (debt < 0.0), counted, debt)
    new_debt = torch.where(commit & (debt < 0.0), committed, new_debt)
    new_debt = torch.where(opened, torch.full_like(debt, NIGHT_OPEN_DEBT), new_debt)
    new_debt = new_debt.clamp(-DEBT_BOUND_H, DEBT_BOUND_H)
    return new_offset, new_debt
