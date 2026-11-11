"""Glucose the kidney does not take back.

Diabetes 2017;66:1999–2006 (doi 10.2337/db17-0100). People without
diabetes, n = 15. The plasma concentration where glucose appears in
the urine is 190 ± 2 mg/dL. Creatinine clearance in that group is
139 ± 7 mL/min. Above the threshold the excretion rises with the
filtered load of the excess, which is that clearance times the
concentration above 190 mg/dL. Below the threshold the excretion is
zero. Empagliflozin 25 mg/day moves that threshold and the clearance
to the day-13 values in the same paper. The input is off unless a
protocol turns it on.

The student learns all four numbers (``KIDNEY``) from these values; the
teacher keeps them.
"""

from __future__ import annotations

from typing import Mapping

import torch

from .published import Published

RTG_MG_DL = 190.0
RTG_SE_MG_DL = 2.0
GFR_ML_PER_MIN = 139.0
GFR_SE_ML_PER_MIN = 7.0

# Empagliflozin 25 mg/day, people without diabetes, n = 15. The regression
# of urinary glucose on plasma glucose crosses zero at 34 mg/dL on day 14
# (32 mg/dL at 48 h). Creatinine clearance on day 13 is 126 ± 10 mL/min.
# The 24 h excretion on that day is 42 ± 5 g. Exposure 0 is the untreated
# kidney. Exposure 1 is that steady dose. A value between them is the
# input's mix of the two measured kidneys, not a dose titration.
EMPA_RTG_MG_DL = 34.0
# The paper reports no SE on the treated intercept. The spread between its
# two measured days, 34 on day 14 and 32 at 48 h, is what it does report.
EMPA_RTG_SE_MG_DL = 2.0
EMPA_GFR_ML_PER_MIN = 126.0
EMPA_GFR_SE_ML_PER_MIN = 10.0
EMPA_UGE_G_PER_DAY = 42.0
EMPA_UGE_SE_G_PER_DAY = 5.0

_STUDY = "Diabetes 2017;66:1999, urinary glucose regressed on plasma glucose"
KIDNEY: dict[str, Published] = {
    "rtg": Published(RTG_MG_DL, f"{_STUDY}: where it leaves zero"),
    "gfr": Published(GFR_ML_PER_MIN, f"{_STUDY}: its slope above the threshold"),
    "empa_rtg": Published(EMPA_RTG_MG_DL, f"{_STUDY}, on empagliflozin 25 mg/day: where it leaves zero"),
    "empa_gfr": Published(EMPA_GFR_ML_PER_MIN, f"{_STUDY}, on empagliflozin 25 mg/day: its slope"),
}


def _clearance(exposure: float) -> tuple[float, float]:
    e = min(max(float(exposure), 0.0), 1.0)
    rtg = RTG_MG_DL + (EMPA_RTG_MG_DL - RTG_MG_DL) * e
    gfr = GFR_ML_PER_MIN + (EMPA_GFR_ML_PER_MIN - GFR_ML_PER_MIN) * e
    return rtg, gfr


def clearance_t(
    exposure: torch.Tensor, kidney: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(threshold mg/dL, GFR ml/min)`` at this drug exposure: the protocol's and the
    person's alone, so a rollout computes them once."""
    e = exposure.clamp(0.0, 1.0)
    rtg = kidney["rtg"] + (kidney["empa_rtg"] - kidney["rtg"]) * e
    gfr = kidney["gfr"] + (kidney["empa_gfr"] - kidney["gfr"]) * e
    return rtg, gfr


def urinary_mg_per_min(glucose_mg_dl: float, exposure: float = 0.0) -> float:
    rtg, gfr = _clearance(exposure)
    excess = max(float(glucose_mg_dl) - rtg, 0.0)
    return gfr * excess / 100.0


def urinary_mg_per_min_t(
    glucose_mg_dl: torch.Tensor,
    exposure: torch.Tensor | float,
    kidney: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    if not isinstance(exposure, torch.Tensor):
        exposure = glucose_mg_dl.new_tensor(float(exposure))
    rtg, gfr = clearance_t(exposure, kidney)
    return excretion_mg_per_min_t(glucose_mg_dl, rtg, gfr)


def excretion_mg_per_min_t(
    glucose_mg_dl: torch.Tensor, rtg: torch.Tensor, gfr: torch.Tensor,
) -> torch.Tensor:
    """Glucose in the urine (mg/min) given the clearance ``clearance_t`` returned."""
    excess = (glucose_mg_dl - rtg).clamp(min=0.0)
    return gfr * excess / 100.0


def urinary_rate(glucose_mg_dl: float, mg_dl_per_g: float, exposure: float = 0.0) -> float:
    """mg/dL/min leaving the glucose distribution volume."""
    return urinary_mg_per_min(glucose_mg_dl, exposure) * float(mg_dl_per_g) / 1000.0


def urinary_rate_t(
    glucose_mg_dl: torch.Tensor,
    mg_dl_per_g: torch.Tensor,
    rtg: torch.Tensor,
    gfr: torch.Tensor,
) -> torch.Tensor:
    return excretion_mg_per_min_t(glucose_mg_dl, rtg, gfr) * mg_dl_per_g / 1000.0
