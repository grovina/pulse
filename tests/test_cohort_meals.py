"""An arm's meal is (minute, carbohydrate g, fat g, protein g), fat before protein."""

from __future__ import annotations

import pytest

from pulse.cohort_loss import _meals_from_spec
from pulse.knowledge.cohorts.glucose_handling import _LANDAU_ARM


def test_the_landau_meal_has_the_energy_shares_the_paper_reports():
    (meal,) = _meals_from_spec(_LANDAU_ARM.meals)
    kcal = 4.0 * meal.carbs + 9.0 * meal.fats + 4.0 * meal.proteins
    assert 4.0 * meal.carbs / kcal == pytest.approx(0.48)
    assert 9.0 * meal.fats / kcal == pytest.approx(0.33)
    assert 4.0 * meal.proteins / kcal == pytest.approx(0.19)
