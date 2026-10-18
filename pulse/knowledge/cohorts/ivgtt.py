"""
Cohort statistics: the intravenous glucose tolerance test (IVGTT).

An IV glucose bolus is the first protocol the glucose-infusion input
(``CohortArmSpec.glucose_infusion``, PLAN Phase 4) unblocks: 0.3 g/kg body
weight delivered over two minutes, straight into the plasma glucose space, with
no meal and no gut absorption in the loop. The bolus is the gold-standard probe
of β-cell function because the acute (first-phase) insulin response it triggers
is carried by glucose alone, with the incretin axis silent.

Two findings ride the same arm: the glucose excursion the bolus produces, and
the acute insulin response it triggers. Both are wide-sigma soft slices — the
population envelope is broad and the targets are digitized from the canonical
IVGTT descriptions, not transcribed from one table.
"""

from __future__ import annotations

from ..cohort_types import (
    CohortArmSpec,
    CohortStatisticSpec,
    StatisticKind,
    StatisticWindow,
)
from ...types import BODY_MASS_KG, MG_DL_PER_G


# 0.3 g/kg glucose over two minutes, as a per-minute appearance rate in mg/dL/min
# of the patient's glucose space (the unit ``glucose_infusion`` expects).
_IVGTT_DOSE_G_PER_KG = 0.3
_IVGTT_INJECT_MIN = 2
_IVGTT_DURATION_MIN = 60
_IVGTT_GRAMS = _IVGTT_DOSE_G_PER_KG * BODY_MASS_KG
_IVGTT_RATE = _IVGTT_GRAMS / _IVGTT_INJECT_MIN * MG_DL_PER_G
_IVGTT_INFUSION = tuple(
    _IVGTT_RATE if t < _IVGTT_INJECT_MIN else 0.0 for t in range(_IVGTT_DURATION_MIN)
)
_IVGTT_ARM = CohortArmSpec(
    label="ivgtt_0_3g_kg",
    duration_min=_IVGTT_DURATION_MIN,
    start_hour=8.0,
    meals=(),
    glucose_infusion=_IVGTT_INFUSION,
)


# ---------------------------------------------------------------------------
# IVGTT → peak plasma glucose
# ---------------------------------------------------------------------------
# A 0.3 g/kg bolus over ~1-2 min drives plasma glucose to a peak of roughly
# 250-350 mg/dL within the first few minutes, set by the dose over the glucose
# distribution volume. Target 300 ± 70 spans the injection-speed and volume
# spread across the classic IVGTT / Bergman minimal-model literature.
IVGTT_GLUCOSE_PEAK = CohortStatisticSpec(
    name="ivgtt_0_3g_kg_glucose_peak",
    source="Bergman et al. (1979) minimal model; standard 0.3 g/kg IVGTT",
    description="0.3 g/kg IV glucose → plasma glucose peak ≈ 300 mg/dL in the first minutes",
    arms=(_IVGTT_ARM,),
    marker_id="glucose",
    kind=StatisticKind.PEAK_VALUE,
    window=StatisticWindow(start_min=0, end_min=10),
    target=300.0,
    sigma=70.0,
)


# ---------------------------------------------------------------------------
# IVGTT → acute (first-phase) insulin response
# ---------------------------------------------------------------------------
# The hallmark IVGTT finding: a sharp first-phase insulin peak within the first
# ~10 min, carried by glucose with the incretin axis silent, and absent in type 2
# diabetes. In healthy adults the acute insulin response peaks near 50-70 µU/mL.
# Target 55 ± 25 reflects the wide normal envelope.
IVGTT_INSULIN_FIRST_PHASE = CohortStatisticSpec(
    name="ivgtt_0_3g_kg_insulin_first_phase",
    source="Bergman et al. (1979); Kahn et al. (1993) — acute insulin response to IV glucose",
    description="0.3 g/kg IV glucose → first-phase insulin peak ≈ 55 µU/mL within 10 min (healthy adults)",
    arms=(_IVGTT_ARM,),
    marker_id="insulin",
    kind=StatisticKind.PEAK_VALUE,
    window=StatisticWindow(start_min=0, end_min=10),
    target=55.0,
    sigma=25.0,
)


COHORT_STATISTICS: list[CohortStatisticSpec] = [
    IVGTT_GLUCOSE_PEAK,
    IVGTT_INSULIN_FIRST_PHASE,
]
