"""
Registry of all quantitative cohort statistic specs.

Each domain file under ``knowledge.cohorts`` defines ``COHORT_STATISTICS``;
this module merges them in training order. Add a new slice by creating or
extending a file there and appending its list below.
"""

from __future__ import annotations

from .cohort_types import CohortStatisticSpec
from .cohorts import breadth_floor, cardiovascular, glucose_handling, nutrition, sleep

ALL_COHORT_STATISTICS: list[CohortStatisticSpec] = [
    *nutrition.COHORT_STATISTICS,
    *sleep.COHORT_STATISTICS,
    *glucose_handling.COHORT_STATISTICS,
    *cardiovascular.COHORT_STATISTICS,
    *breadth_floor.COHORT_STATISTICS,
]

__all__ = ["ALL_COHORT_STATISTICS"]
