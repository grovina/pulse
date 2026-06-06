"""
Knowledge contribution registry.

Each contribution is a traceable source of training data that encodes
medical knowledge. Add new contributions by creating a file in this
directory and registering it here.
"""

from .base import Episode, KnowledgeContribution, CouplingPrior
from .bergman_glucose_insulin import BergmanGlucoseInsulin
from .cardiovascular_dynamics import CardiovascularDynamics
from .cohort_statistics import ALL_COHORT_STATISTICS
from .cohort_types import (
    CohortArmSpec,
    CohortStatisticSpec,
    StatisticKind,
    StatisticWindow,
)
from .cortisol_circadian import CortisolCircadian
from .full_body import FullBody

ALL_CONTRIBUTIONS: list[KnowledgeContribution] = [
    FullBody(),
    BergmanGlucoseInsulin(),
    CortisolCircadian(),
    CardiovascularDynamics(),
]

__all__ = [
    "Episode",
    "KnowledgeContribution",
    "CouplingPrior",
    "CohortArmSpec",
    "CohortStatisticSpec",
    "StatisticKind",
    "StatisticWindow",
    "FullBody",
    "ALL_CONTRIBUTIONS",
    "ALL_COHORT_STATISTICS",
]
