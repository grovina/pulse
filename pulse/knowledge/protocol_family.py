"""Which protocol a finding was measured in, read off its arms.

The scoreboard reports the held-out fit per protocol family (PLAN section 4)
and sealing is drawn within a family (Phase 0), so a family has to be a
property of the finding and not a label its author chose: a label drifts from
the arms it describes the first time an arm is edited. It is derived.

A protocol belongs to one family. Where two apply, the stronger intervention
wins, because that is the thing the arm was built to probe: a drug over the
clamp it is measured under, an IV input over a sleep or exercise manipulation,
a manipulation over the meals of the day around it, a load over the fast it
interrupts.
"""

from __future__ import annotations

from enum import Enum
from typing import Sequence

from .cohort_types import CohortArmSpec, is_on

__all__ = ["ProtocolFamily", "protocol_family", "families_of"]

# A fast this long is the finding, not the run-in before one.
_FAST_HOURS = 12.0


class ProtocolFamily(str, Enum):
    """Ordered weakest to strongest; ``protocol_family`` takes the strongest arm."""

    RESTING = "resting"
    FASTING = "fasting"
    MIXED_MEAL = "mixed_meal"
    ORAL_GLUCOSE = "oral_glucose"
    SLEEP = "sleep"
    EXERCISE = "exercise"
    IV_GLUCOSE = "iv_glucose"
    CLAMP = "clamp"
    DRUG = "drug"


_RANK: dict[ProtocolFamily, int] = {
    f: i for i, f in enumerate(ProtocolFamily)
}


def _sleep_is_short(arm: CohortArmSpec) -> bool:
    """Does this arm's own sleep log fall short of the need over the nights it spans?

    The need is ``circadian.NEED_H`` with an hour of slack, so an ordinary night
    logged at 20-minute smoothing is not a restriction. An arm that logs no sleep
    at all declares nothing about sleep and is not short.
    """
    if arm.sleep_wake is None:
        return False
    from ..circadian import NEED_H

    asleep_h = sum(1.0 - float(x) for x in arm.sleep_wake) / 60.0
    nights = max(1, round(arm.duration_min / 1440.0))
    return asleep_h < NEED_H * nights - 1.0


def _family_of_arm(arm: CohortArmSpec, sleep_differs: bool) -> ProtocolFamily:
    if is_on(arm.sglt2):
        return ProtocolFamily.DRUG
    if is_on(arm.glucose_clamp) or is_on(arm.insulin_clamp):
        return ProtocolFamily.CLAMP
    if is_on(arm.glucose_infusion):
        return ProtocolFamily.IV_GLUCOSE
    if arm.activity is not None and any(a > 0.0 for a in arm.activity):
        return ProtocolFamily.EXERCISE
    if sleep_differs or _sleep_is_short(arm):
        return ProtocolFamily.SLEEP
    if arm.meals:
        carb_only = all(fat == 0.0 and protein == 0.0 for _t, _c, fat, protein in arm.meals)
        single_load = len(arm.meals) == 1
        if carb_only and single_load:
            return ProtocolFamily.ORAL_GLUCOSE
        return ProtocolFamily.MIXED_MEAL
    if arm.duration_min + arm.prefast_hours * 60.0 >= _FAST_HOURS * 60.0:
        return ProtocolFamily.FASTING
    return ProtocolFamily.RESTING


def protocol_family(arms: Sequence[CohortArmSpec]) -> ProtocolFamily:
    """The family of a finding measured over ``arms``.

    An item with no arm of its own — the dose canon, which carries its protocol
    in its own signal — is ``RESTING``.
    """
    if not arms:
        return ProtocolFamily.RESTING
    logs = {arm.sleep_wake for arm in arms}
    sleep_differs = len(logs) > 1
    return max(
        (_family_of_arm(arm, sleep_differs) for arm in arms),
        key=lambda f: _RANK[f],
    )


def families_of(items: Sequence[object]) -> dict[str, list[object]]:
    """``items`` grouped by the family of their ``arms``, in declaration order."""
    out: dict[str, list[object]] = {}
    for item in items:
        family = protocol_family(getattr(item, "arms", ()))
        out.setdefault(family.value, []).append(item)
    return out
