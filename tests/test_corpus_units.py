"""The unit table: every marker has a row, every conversion is right, and an unlisted unit is refused."""

from __future__ import annotations

import pytest

from pulse.knowledge.corpus_units import (
    CONVERSIONS, UnitError, listed_units, native_unit, normalise, plausible_range, time_to_minutes, to_native,
)
from pulse.types import MARKERS, OBSERVABLES, SERIES_INDEX


def test_every_series_the_model_carries_has_a_unit_row():
    assert set(CONVERSIONS) == set(SERIES_INDEX)


def test_each_series_native_unit_is_listed_and_is_the_identity():
    for sid in SERIES_INDEX:
        entry = CONVERSIONS[sid].get(normalise(native_unit(sid)))
        if entry is None:  # the native unit written another way (types.py says '× pop. mean')
            assert sid == "mitochondrial_capacity", sid
            entry = CONVERSIONS[sid]["x_pop_mean"]
        assert entry == (1.0, 0.0), sid


def test_every_factor_is_positive_and_only_temperature_has_an_offset():
    for sid, table in CONVERSIONS.items():
        for unit, (factor, offset) in table.items():
            assert factor > 0, (sid, unit)
            if offset:
                assert sid == "temp", (sid, unit)


def test_every_series_has_a_plausible_range():
    for sid in SERIES_INDEX:
        lo, hi = plausible_range(sid)
        assert lo < hi, sid


def test_micro_signs_and_case_are_one_unit():
    for spelling in ("µU/mL", "μU/mL", "uU/mL", "UU/ML", " uU / mL "):
        assert to_native("insulin", 10.0, spelling) == pytest.approx(10.0)


@pytest.mark.parametrize("series, value, unit, expected", [
    ("glucose", 5.5, "mmol/L", 5.5 * 18.016),
    ("glucose", 5.5, "mM", 5.5 * 18.016),
    ("glucose", 90.0, "mg/dL", 90.0),
    ("glucose", 0.9, "g/L", 90.0),
    ("insulin", 60.0, "pmol/L", 10.0),
    ("insulin", 10.0, "mU/L", 10.0),
    ("insulin", 10.0, "µIU/mL", 10.0),
    ("insulin_slow", 120.0, "pmol/L", 20.0),
    ("glucagon", 3.4828, "pmol/L", 12.13, ),
    ("glucagon", 70.0, "ng/L", 70.0),
    ("ffa", 500.0, "µmol/L", 0.5),
    ("ffa", 500.0, "µEq/L", 0.5),
    ("ffa", 500.0, "uEq/L", 0.5),
    ("bhb", 10.41, "mg/dL", 1.0),
    ("bhb", 1000.0, "µmol/L", 1.0),
    ("lactate", 9.008, "mg/dL", 1.0),
    ("ghrelin", 2.967, "pmol/L", 10.0),
    ("leptin", 625.0, "pmol/L", 10.0),
    ("glp1", 33.0, "pg/mL", 10.0),
    ("cortisol", 400.0, "nmol/L", 14.498),
    ("cortisol", 120.0, "ng/mL", 12.0),
    ("acth", 6.6, "pmol/L", 29.97),
    ("hr", 70.0, "beats/min", 70.0),
    ("hrv", 0.05, "s", 50.0),
    ("sbp", 16.0, "kPa", 120.0),
    ("temp", 98.6, "°F", 37.0),
    ("temp", 310.15, "K", 37.0),
    ("temp", 37.0, "°C", 37.0),
    ("spo2", 0.97, "fraction", 97.0),
    ("sns_tone", 20.0, "%", 0.2),
    ("epinephrine", 100.0, "pmol/L", 18.32),
    ("epinephrine", 0.05, "ng/mL", 50.0),
    ("gip", 100.0, "pg/mL", 20.066),
    ("fat_mass", 20000.0, "g", 20.0),
    ("fat_mass", 44.0924, "lb", 20.0),
    ("circ_phase_offset", 90.0, "min", 1.5),
    ("hepatic_output", 11.1, "µmol/kg/min", 2.0),
    ("glucose_infusion_rate", 10.0, "mg/kg/min", 10.0),
    ("glucose_disposal", 0.0555, "mmol/kg/min", 10.0),
    ("urinary_glucose", 72.0, "g/day", 50.0),
    ("urinary_glucose", 0.5, "mmol/min", 90.08),
    ("vo2", 0.25, "L/min", 250.0),
    ("energy_expenditure", 1440.0, "kcal/day", 1.0),
    ("energy_expenditure", 4.184, "kJ/min", 1.0),
    ("energy_expenditure", 69.7, "W", 1.0),
    ("energy_expenditure", 6.0, "MJ/day", 995.9 / 1000 * 1.0),
    ("lipolysis", 5.0, "µmol/L/min", 0.005),
    ("gng_fraction", 47.0, "%", 0.47),
    ("bile_acids", 5000.0, "nmol/L", 5.0),
])
def test_conversion(series, value, unit, expected):
    assert to_native(series, value, unit) == pytest.approx(expected, rel=2e-3)


def test_a_difference_ignores_the_offset():
    assert to_native("temp", 1.8, "°F", difference=True) == pytest.approx(1.0)
    assert to_native("temp", 1.0, "K", difference=True) == pytest.approx(1.0)
    assert to_native("glucose", 1.0, "mmol/L", difference=True) == pytest.approx(18.016)


def test_the_fahrenheit_zero_is_not_the_celsius_zero():
    assert to_native("temp", 32.0, "°F") == pytest.approx(0.0)
    assert to_native("temp", 32.0, "°F", difference=True) == pytest.approx(32.0 * 5.0 / 9.0)


@pytest.mark.parametrize("series, unit", [
    ("insulin", "ng/mL"),         # needs a potency the paper must state
    ("insulin", "mmol/L"),
    ("glucose", "mg/kg"),
    ("glucose", ""),
    ("glucose_infusion_rate", "mg/kg FFM/min"),
    ("vo2", "mL/kg/min"),        # needs a body mass
    ("cck", "pg/mL"),            # CCK-8 and CCK-33 differ four-fold
    ("temp", "C/F"),
])
def test_an_unlisted_conversion_is_refused_not_guessed(series, unit):
    with pytest.raises(UnitError, match="not a listed unit"):
        to_native(series, 1.0, unit)


def test_refusal_names_the_listed_units():
    with pytest.raises(UnitError) as err:
        to_native("insulin", 1.0, "ng/mL")
    assert "pmol/l" in str(err.value) and "ng/mL" in str(err.value)


def test_an_unknown_series_is_refused():
    with pytest.raises(UnitError):
        to_native("adiponectin", 1.0, "ng/mL")
    with pytest.raises(UnitError):
        listed_units("adiponectin")


def test_time_units():
    assert time_to_minutes(1.5, "h") == 90.0
    assert time_to_minutes(30.0, "min") == 30.0
    assert time_to_minutes(90.0, "s") == 1.5
    with pytest.raises(UnitError):
        time_to_minutes(1.0, "days")


def test_the_two_insulin_pairs_agree():
    # pmol/L -> µU/mL and back through the glucose-free route: 6 pmol/L per µU/mL.
    assert to_native("insulin", 6.0, "pmol/L") == pytest.approx(1.0)
    assert len(MARKERS) + len(OBSERVABLES) == len(SERIES_INDEX)
