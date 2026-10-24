"""GIP, the amino-acid path into insulin, and GLP-1's hold on glucagon.

Shared by the teacher and the student. Vilsbøll, Krarup, Madsbad and Holst,
Diabetologia 2003;46:183: at ordinary glucose the two incretins contribute
about equally, and GLP-1, not GIP, suppresses glucagon further. Vollmer et
al., Diabetes 2008;57:678: after 240 min, integrated GIP was 186 ± 17% higher
following an 820 kcal breakfast than following 75 g of glucose. Salehi, Vahl
and D'Alessio, J Clin Endocrinol Metab 2008;93:4909, Table: with glucose
clamped, exendin-(9-39) raised the incremental glucagon area from 221 ± 353
to 1657 ± 699 pg/mL·min over the 175 min after 75 g of oral glucose.
Westphal, Gannon and Nuttall, Am J Clin Nutr 1990;52:267: 50 g of protein
alone did not change the serum glucose.

The student learns every number below from these values — GIP's kinetics on
the appetite module that holds GIP (``GIP_KINETICS``), the islet gains on the
metabolic module (``ISLET``). The teacher keeps them.
"""

from __future__ import annotations

from typing import Mapping

import torch

from .gastric import EMPTY_KCAL_PER_MIN
from .published import Published
from .types import MG_DL_PER_G

GIP_B = 10.0
K_GIP = 0.2  # same clearance as GLP-1; both are gone in minutes
K_INCRETIN = 10.0
# Half of the factor that used to sit on GLP-1 alone. When the two excesses
# are equal the sum is the old factor, so a meal that raises them together
# does not grow a second copy of the incretin effect.
INCRETIN_HALF = 2.0

# Vollmer breakfast: 360 kcal carbohydrate, 353 fat, 107 protein.
_MEAL_KCAL = 820.0
_CARB_G = 360.0 / 4.0
_FAT_G = 353.0 / 9.0
_WINDOW_MIN = 240.0
_OGTT_G = 75.0
VOLLMER_RATIO = 1.0 + 1.86
VOLLMER_RATIO_SE = 0.17

# One compartment empties at 2.13 kcal/min, mixed in the proportion swallowed.
# The breakfast is still emptying at 240 min, so the ratio is of what has
# left by then, not of the plate.
_DELIVERED = min(1.0, _WINDOW_MIN * EMPTY_KCAL_PER_MIN / _MEAL_KCAL)
_CARB_DELIVERED = _DELIVERED * _CARB_G
_FAT_DELIVERED = _DELIVERED * _FAT_G
# (carb + weight * fat) / 75 g = 2.86, and the glucose load is all carb.
GIP_FAT_WEIGHT = (VOLLMER_RATIO * _OGTT_G - _CARB_DELIVERED) / _FAT_DELIVERED

# Teacher GLP-1 secretion (PatientParams.glp1_meal_gain / k_glp1). While the
# Vollmer breakfast is emptying, GIP's excess is set equal to GLP-1's, which
# is the "nearly equally" operating point.
_GLP1_MEAL_GAIN = 0.7
_K_GLP1 = 0.2
_D_CARB = EMPTY_KCAL_PER_MIN * (360.0 / _MEAL_KCAL) / 4.0
_D_FAT = EMPTY_KCAL_PER_MIN * (353.0 / _MEAL_KCAL) / 9.0
_STIM = _D_CARB + GIP_FAT_WEIGHT * _D_FAT
_GLP1_EXCESS = (_GLP1_MEAL_GAIN / _K_GLP1) * _D_CARB * MG_DL_PER_G
GIP_GAIN = _GLP1_EXCESS * K_GIP / _STIM

# Salehi table, (1657 − 221) / 175 min.
GLP1_GN_PG = (1657.0 - 221.0) / 175.0
# Excess while 75 g of glucose (300 kcal) is leaving at 2.13 kcal/min.
SALEHI_GLP1_EXCESS = (
    (_GLP1_MEAL_GAIN / _K_GLP1) * (EMPTY_KCAL_PER_MIN / 4.0) * MG_DL_PER_G
)
_SAT_REF = SALEHI_GLP1_EXCESS / (SALEHI_GLP1_EXCESS + K_INCRETIN)

# µU/mL per minute, and pg/mL per minute, per gram of protein leaving the
# stomach. Both are zero when no protein is leaving, so fasting levels stay
# put. The pair is sized on Westphal's 50 g protein meal: insulin rises, and
# the 4 h glucose difference stays inside the fasting uncertainty of
# 0.1 mmol/L.
# Per gram per minute leaving the stomach. 1.0 µU/mL/min raises insulin by
# about 1 µU/mL while 50 g of protein is emptying. 3.5 pg/mL/min is the
# glucagon rise that puts the mixed meal's suppression back on its published
# −10 pg/mL once GLP-1 is holding glucagon down; a smaller rise lets that
# meal's glucose-and-insulin suppression win, and the protein meal's glucose
# then falls.
AMINO_INS_GAIN = 1.0
PROTEIN_GN_GAIN = 3.5

GIP_KINETICS: dict[str, Published] = {
    "gip_b": Published(GIP_B, "Fasting plasma GIP in healthy adults"),
    "k_gip": Published(K_GIP, "Plasma GIP decay after an intravenous bolus"),
    "gip_gain": Published(
        GIP_GAIN, "Vilsbøll 2003: GIP beside GLP-1 after a meal at ordinary glucose"),
    "gip_fat_weight": Published(
        GIP_FAT_WEIGHT, "Vollmer 2008: integrated GIP after an 820 kcal breakfast against 75 g glucose"),
}
ISLET: dict[str, Published] = {
    "k_incretin": Published(
        K_INCRETIN, "Vilsbøll 2003: the insulin response to each incretin as its level rises"),
    "incretin_half": Published(
        INCRETIN_HALF, "Vilsbøll 2003: the incretins' share of the meal insulin response"),
    "glp1_gn_pg": Published(
        GLP1_GN_PG, "Salehi 2008: glucagon area with exendin-(9-39) after 75 g, glucose clamped"),
    "amino_ins_gain": Published(AMINO_INS_GAIN, "Westphal 1990: insulin after 50 g of protein alone"),
    "protein_gn_gain": Published(
        PROTEIN_GN_GAIN, "Westphal 1990: glucagon after 50 g of protein alone, glucose unchanged"),
}


def glucagon_from_protein(protein_g: float) -> float:
    return PROTEIN_GN_GAIN * max(protein_g, 0.0)


def glucagon_from_protein_t(protein_g: torch.Tensor, islet: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return islet["protein_gn_gain"] * torch.clamp(protein_g, min=0.0)


def _sat(excess: float) -> float:
    e = excess if excess > 0.0 else 0.0
    return e / (e + K_INCRETIN)


def incretin_factor(glp1_excess: float, gip_excess: float) -> float:
    """1 at basal GLP-1 and basal GIP."""
    return 1.0 + INCRETIN_HALF * (_sat(glp1_excess) + _sat(gip_excess))


def gip_rate(gip: float, carb_g: float, fat_g: float) -> float:
    stim = max(carb_g, 0.0) + GIP_FAT_WEIGHT * max(fat_g, 0.0)
    return -K_GIP * (gip - GIP_B) + GIP_GAIN * stim


def glp1_glucagon_rate(glp1_excess: float, k_gn: float) -> float:
    """pg/mL per minute. Zero at basal GLP-1.

    At ``SALEHI_GLP1_EXCESS`` the glucagon equilibrium is ``GLP1_GN_PG`` lower
    than it is with GLP-1 at basal. ``k_gn`` cancels in that equilibrium.
    """
    drop = GLP1_GN_PG * _sat(glp1_excess) / _SAT_REF
    return drop * k_gn


def amino_insulin_rate(protein_g: float) -> float:
    """µU/mL per minute. Zero when no protein is leaving the stomach."""
    return AMINO_INS_GAIN * max(protein_g, 0.0)


def _sat_t(excess: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    e = torch.clamp(excess, min=0.0)
    return e / (e + k)


def incretin_factor_t(
    glp1_excess: torch.Tensor, gip_excess: torch.Tensor, islet: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    k = islet["k_incretin"]
    return 1.0 + islet["incretin_half"] * (_sat_t(glp1_excess, k) + _sat_t(gip_excess, k))


def gip_rate_t(
    gip: torch.Tensor, carb_g: torch.Tensor, fat_g: torch.Tensor,
    kinetics: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    stim = torch.clamp(carb_g, min=0.0) + kinetics["gip_fat_weight"] * torch.clamp(fat_g, min=0.0)
    return -kinetics["k_gip"] * (gip - kinetics["gip_b"]) + kinetics["gip_gain"] * stim


def glp1_glucagon_rate_t(
    glp1_excess: torch.Tensor, k_gn: torch.Tensor, islet: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """At ``SALEHI_GLP1_EXCESS`` the drop is ``glp1_gn_pg``, whatever the saturation."""
    k = islet["k_incretin"]
    salehi = glp1_excess.new_tensor(SALEHI_GLP1_EXCESS)
    drop = islet["glp1_gn_pg"] * _sat_t(glp1_excess, k) / _sat_t(salehi, k)
    return drop * k_gn


def amino_insulin_rate_t(protein_g: torch.Tensor, islet: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return islet["amino_ins_gain"] * torch.clamp(protein_g, min=0.0)
