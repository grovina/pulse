"""A finding's protocol family is read off its arms; new items are sealed at random."""

from __future__ import annotations

import pytest

from pulse.knowledge.cohort_statistics import ALL_COHORT_STATISTICS, SEALED_COHORT_STATISTICS
from pulse.knowledge.cohort_types import CohortArmSpec, held
from pulse.knowledge.physiology_rules import PHYSIOLOGY_RULES
from pulse.knowledge.protocol_family import ProtocolFamily, families_of, protocol_family
from pulse.knowledge.sealing import SEAL_SHARE, sealed_at_random


def _arm(**kw) -> CohortArmSpec:
    base = {"label": "a", "duration_min": 120, "start_hour": 8.0, "meals": ()}
    return CohortArmSpec(**{**base, **kw})


def _awake_nights(hours_asleep: float, nights: int = 1) -> tuple[float, ...]:
    day = [1.0] * 1440
    for m in range(int(hours_asleep * 60)):
        day[m] = 0.0
    return tuple(day * nights)


@pytest.mark.parametrize(("arm", "family"), [
    (_arm(), ProtocolFamily.RESTING),
    (_arm(duration_min=60, prefast_hours=24.0), ProtocolFamily.FASTING),
    (_arm(meals=((0.0, 75.0, 0.0, 0.0),)), ProtocolFamily.ORAL_GLUCOSE),
    (_arm(meals=((0.0, 50.0, 15.0, 20.0),)), ProtocolFamily.MIXED_MEAL),
    (_arm(activity=(0.0,) * 60 + (0.5,) * 60), ProtocolFamily.EXERCISE),
    (_arm(glucose_infusion=(2.0,) * 120), ProtocolFamily.IV_GLUCOSE),
    (_arm(glucose_clamp=held(90.0, 120)), ProtocolFamily.CLAMP),
    (_arm(insulin_clamp=held(60.0, 120), meals=((0.0, 75.0, 0.0, 0.0),)), ProtocolFamily.CLAMP),
    (_arm(glucose_clamp=held(150.0, 120), sglt2=held(1.0, 120)), ProtocolFamily.DRUG),
    (_arm(glucose_clamp=held(150.0, 120), sglt2=held(0.0, 120)), ProtocolFamily.CLAMP),
])
def test_the_strongest_intervention_names_the_family(arm, family):
    assert protocol_family((arm,)) is family


def test_sleep_is_short_only_against_the_need():
    full = _arm(duration_min=1440, sleep_wake=_awake_nights(8.0))
    short = _arm(duration_min=1440, sleep_wake=_awake_nights(4.0))
    two_short = _arm(duration_min=2880, sleep_wake=_awake_nights(6.0, nights=2))
    assert protocol_family((full,)) is ProtocolFamily.FASTING  # a meal-free day, slept well
    assert protocol_family((short,)) is ProtocolFamily.SLEEP
    assert protocol_family((two_short,)) is ProtocolFamily.SLEEP


def test_arms_that_differ_only_in_sleep_are_a_sleep_protocol():
    a = _arm(duration_min=1440, sleep_wake=_awake_nights(8.0), label="a")
    b = _arm(duration_min=1440, sleep_wake=_awake_nights(7.5), label="b")
    assert protocol_family((a, b)) is ProtocolFamily.SLEEP


def test_every_filed_item_has_a_family():
    items = [*ALL_COHORT_STATISTICS, *SEALED_COHORT_STATISTICS, *PHYSIOLOGY_RULES]
    grouped = families_of(items)
    assert sum(len(v) for v in grouped.values()) == len(items)
    assert set(grouped) <= {f.value for f in ProtocolFamily}
    assert len(grouped) >= 4


def test_the_seal_draw_is_reproducible_and_near_its_share():
    names = [f"item_{i}" for i in range(4000)]
    draws = [sealed_at_random(n) for n in names]
    assert draws == [sealed_at_random(n) for n in names]
    assert abs(sum(draws) / len(draws) - SEAL_SHARE) < 0.02
    assert not any(sealed_at_random(n, share=0.0) for n in names[:50])
    assert all(sealed_at_random(n, share=1.0) for n in names[:50])
    with pytest.raises(ValueError):
        sealed_at_random("x", share=1.5)
