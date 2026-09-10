"""
Knowledge registry.

Training episodes come from one generator (FullBody). Sparse evidence —
cohort statistics, physiology rules, the wearable tape, teacher internals —
lives on ALL_EVIDENCE. Views of the generator still exist for diagnostics;
they are not training contributions.
"""

from .base import Episode, KnowledgeContribution, CouplingPrior
from .cohort_statistics import ALL_COHORT_STATISTICS
from .cohort_types import (
    CohortArmSpec,
    CohortStatisticSpec,
    StatisticKind,
    StatisticWindow,
)
from .evidence import (
    ALL_EVIDENCE,
    Authority,
    EvidenceItem,
    TEACHER_DISTILL_LONG_ONLY,
    TEACHER_DISTILL_MARKERS,
    TEACHER_TAPE_MARKERS,
)
from .full_body import FullBody

# Views (Bergman / cortisol / cardio) still exist as diagnostic masks of the
# generator. They are not training contributions: their dense hormone tapes
# are not evidence.
ALL_CONTRIBUTIONS: list[KnowledgeContribution] = [
    FullBody(),
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
    "ALL_EVIDENCE",
    "Authority",
    "EvidenceItem",
    "TEACHER_TAPE_MARKERS",
    "TEACHER_DISTILL_MARKERS",
    "TEACHER_DISTILL_LONG_ONLY",
]
