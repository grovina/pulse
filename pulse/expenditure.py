"""Pulmonary gas exchange and the energy it costs.

Wahren, Felig, Ahlborg and Jorfeldt, J Clin Invest 1971;50:2715, Table I.
Healthy postabsorptive men, upright bicycle. Pulmonary oxygen uptake, mean
± SE: 256 ± 19 ml/min at rest, 1811 ± 69 ml/min after 40 min at 800
kg-m/min. Respiratory quotient 0.77 ± 0.01 at rest, 0.82 ± 0.01 at that
40-minute point. The model's moderate bout is activity 0.7, the same map
the splanchnic-glucose cohort uses (1 kg-m/min = 0.163 W).

Weir, J Physiol 1949;109:1-9, the protein-free equation:
kcal/min = 3.941·VO₂ + 1.106·VCO₂, liters. Diet-induced thermogenesis is
the thermic fractions already on the temperature law (protein 25%,
carbohydrate 8%, fat 3%; Westerterp, Nutr Metab 2004), oxidized at the
respiratory quotients of those fuels.

Respiratory frequency keeps the saturated Hill form. Its input is CO₂
production, scaled so that the 800 kg-m/min point is the same place on
the Hill that activity 0.7 used to be. Lactate and temperature do not
move it. At rest, with no meal, the offset is 0, so the rate equilibrates
at the person's RR₀.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from .types import BODY_MASS_KG

# Table I. The rest SE is the same column alignment as the explicit
# "1811 ± 69": where the ± glyph did not survive, it was read as a 4.
VO2_REST_ML = 256.0
VO2_REST_SE_ML = 19.0
VO2_800_ML = 1811.0
VO2_800_SE_ML = 69.0
RQ_REST = 0.77
RQ_800 = 0.82
ACTIVITY_800 = 0.7

# Weir 1949, neglecting urinary nitrogen. kcal per liter.
WEIR_VO2 = 3.941
WEIR_VCO2 = 1.106

# Atwater factors, kcal per gram of the fuel.
KCAL_CARB = 4.0
KCAL_FAT = 9.0
KCAL_PROT = 4.0
# Glycogen is booked as its glucose. A gram that entered the pool is not
# also a gram of adipose.
KCAL_PER_G_GLYCOGEN = KCAL_CARB
# kcal per kg of adipose tissue, not per kg of pure triglyceride.
KCAL_PER_KG_FAT = 7700.0

# Westerterp 2004, the fraction of the fuel's energy that reappears as heat.
DIT_CARB = 0.08
DIT_FAT = 0.03
DIT_PROT = 0.25
# Lusk's non-protein quotients; protein from the same tables.
RQ_CARB = 1.0
RQ_FAT = 0.707
RQ_PROT = 0.801

# Hill n=2 on the CO₂-equivalent of activity. 20 /min is the ceiling.
# Half-maximal at 0.5. These are the constants the linear drive replaced:
# that one equilibrated at 63 /min when activity was 0.9.
ACT_RR_GAIN = 20.0
RR_ACT_K = 0.5


class Gas(NamedTuple):
    kcal_per_min: float
    vo2_ml_per_min: float
    vco2_ml_per_min: float
    rq: float


def _clamp_rq(rq: float) -> float:
    return min(1.0, max(0.70, rq))


def work_vo2_ml(mass_kg: float, activity: float) -> float:
    """Pulmonary O₂ at this mass and activity, before the meal."""
    act = activity if activity > 0.0 else 0.0
    scale = mass_kg / BODY_MASS_KG
    slope = (VO2_800_ML - VO2_REST_ML) / ACTIVITY_800
    return (VO2_REST_ML + slope * act) * scale


def work_rq(activity: float) -> float:
    act = activity if activity > 0.0 else 0.0
    return _clamp_rq(RQ_REST + (RQ_800 - RQ_REST) * (act / ACTIVITY_800))


def dit_kcal(carb_g: float, fat_g: float, prot_g: float) -> float:
    return (DIT_CARB * KCAL_CARB * carb_g
            + DIT_FAT * KCAL_FAT * fat_g
            + DIT_PROT * KCAL_PROT * prot_g)


def _vo2_ml_of(kcal: float, rq: float) -> float:
    return kcal / (WEIR_VO2 + WEIR_VCO2 * rq) * 1000.0


def gas_exchange(
    mass_kg: float, activity: float, carb_g: float, fat_g: float, prot_g: float,
) -> Gas:
    vo2 = work_vo2_ml(mass_kg, activity)
    rq = work_rq(activity)
    vco2 = rq * vo2
    parts = (
        (DIT_CARB * KCAL_CARB * carb_g, RQ_CARB),
        (DIT_FAT * KCAL_FAT * fat_g, RQ_FAT),
        (DIT_PROT * KCAL_PROT * prot_g, RQ_PROT),
    )
    for kcal, fuel_rq in parts:
        d_vo2 = _vo2_ml_of(kcal, fuel_rq)
        vo2 += d_vo2
        vco2 += fuel_rq * d_vo2
    kcal_min = (WEIR_VO2 * vo2 + WEIR_VCO2 * vco2) / 1000.0
    return Gas(kcal_min, vo2, vco2, vco2 / vo2)


def _span_ml(mass_kg: float) -> tuple[float, float]:
    rest = work_rq(0.0) * work_vo2_ml(mass_kg, 0.0)
    anchor = work_rq(ACTIVITY_800) * work_vo2_ml(mass_kg, ACTIVITY_800)
    return rest, anchor - rest


def frequency_offset(vco2_ml: float, mass_kg: float) -> float:
    """Breaths/min above RR₀. Zero when CO₂ is the resting output."""
    rest, span = _span_ml(mass_kg)
    act = ACTIVITY_800 * max(vco2_ml - rest, 0.0) / span
    return ACT_RR_GAIN * act * act / (act * act + RR_ACT_K * RR_ACT_K)


def work_vo2_ml_t(mass_kg: torch.Tensor, activity: torch.Tensor) -> torch.Tensor:
    act = torch.relu(activity)
    scale = mass_kg / BODY_MASS_KG
    slope = (VO2_800_ML - VO2_REST_ML) / ACTIVITY_800
    return (VO2_REST_ML + slope * act) * scale


def work_rq_t(activity: torch.Tensor) -> torch.Tensor:
    act = torch.relu(activity)
    rq = RQ_REST + (RQ_800 - RQ_REST) * (act / ACTIVITY_800)
    return rq.clamp(0.70, 1.0)


def dit_kcal_t(
    carb_g: torch.Tensor, fat_g: torch.Tensor, prot_g: torch.Tensor,
) -> torch.Tensor:
    return (DIT_CARB * KCAL_CARB * carb_g
            + DIT_FAT * KCAL_FAT * fat_g
            + DIT_PROT * KCAL_PROT * prot_g)


def _vo2_ml_of_t(kcal: torch.Tensor, rq: float) -> torch.Tensor:
    return kcal / (WEIR_VO2 + WEIR_VCO2 * rq) * 1000.0


def gas_exchange_t(
    mass_kg: torch.Tensor,
    activity: torch.Tensor,
    carb_g: torch.Tensor,
    fat_g: torch.Tensor,
    prot_g: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(kcal/min, VO₂ ml/min, VCO₂ ml/min, RQ)``."""
    vo2 = work_vo2_ml_t(mass_kg, activity)
    rq = work_rq_t(activity)
    vco2 = rq * vo2
    parts = (
        (DIT_CARB * KCAL_CARB * carb_g, RQ_CARB),
        (DIT_FAT * KCAL_FAT * fat_g, RQ_FAT),
        (DIT_PROT * KCAL_PROT * prot_g, RQ_PROT),
    )
    for kcal, fuel_rq in parts:
        d_vo2 = _vo2_ml_of_t(kcal, fuel_rq)
        vo2 = vo2 + d_vo2
        vco2 = vco2 + fuel_rq * d_vo2
    kcal_min = (WEIR_VO2 * vo2 + WEIR_VCO2 * vco2) / 1000.0
    return kcal_min, vo2, vco2, vco2 / vo2


def frequency_offset_t(vco2_ml: torch.Tensor, mass_kg: torch.Tensor) -> torch.Tensor:
    rest = work_rq_t(vco2_ml.new_zeros(())) * work_vo2_ml_t(mass_kg, vco2_ml.new_zeros(()))
    anchor_act = vco2_ml.new_tensor(ACTIVITY_800)
    anchor = work_rq_t(anchor_act) * work_vo2_ml_t(mass_kg, anchor_act)
    span = anchor - rest
    act = ACTIVITY_800 * torch.relu(vco2_ml - rest) / span
    return ACT_RR_GAIN * act * act / (act * act + RR_ACT_K * RR_ACT_K)
