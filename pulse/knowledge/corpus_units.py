"""The units a paper may report in, and the one table that converts them.

A finding file states every number in **the paper's own unit**. The loader turns
it into the marker's unit (``pulse.types``: mg/dL for glucose, µU/mL for insulin,
…) through ``CONVERSIONS`` below and nowhere else. A unit that is not listed for a
marker is refused (``UnitError``), never guessed: a reading agent who meets one
either converts it by hand and says so in the quote, or writes a refusal.

Each entry is ``unit -> (factor, offset)`` with ``native = paper * factor + offset``.
The offset is non-zero only for temperature. A **level** (a mean, a peak, a curve
point) converts with factor and offset; a **difference** (a delta, a spread, an
uncertainty) converts with the factor alone, because an interval of 1 °F is 5/9 of
an interval of 1 °C whatever the zero.

Molar conversions use the molecular weight named beside them. Insulin uses the
SI/IUPAC 6.0 pmol/L per µU/mL; some older papers use 6.945 or 7.175, and a paper
that says so is converted by hand, not listed here.

Unit strings are compared after ``normalise``, which gives each unit one spelling:
``µU/mL``, ``μU/mL`` and ``uU/mL`` are one unit, and so are ``mmol L-1``, ``mM`` and
``mmol/L``, or ``µmol min-1 kg-1`` and ``umol/kg/min``. The tables hold that spelling
and no other.
"""

from __future__ import annotations

import re

from ..types import MARKERS, OBSERVABLES, SERIES_INDEX

__all__ = [
    "CONVERSIONS",
    "TIME_UNITS",
    "UnitError",
    "listed_units",
    "native_unit",
    "normalise",
    "plausible_range",
    "time_to_minutes",
    "to_native",
]


class UnitError(ValueError):
    """A unit the table does not list, or a number outside any sane range for it."""


# A molar concentration written with M: pM is pmol/L.
_MOLAR = {"mm": "mmol/l", "um": "umol/l", "nm": "nmol/l", "pm": "pmol/l"}
_PER = re.compile(r"^(.+)-1$")


def normalise(unit: str) -> str:
    """The one spelling of ``unit`` the tables use.

    Case, spaces and the micro sign do not count. A factor raised to minus one is
    a division (``mmol L-1`` is ``mmol/l``), ``M`` is mol/L, and the divisors are
    put in alphabetical order (``umol/min/kg`` is ``umol/kg/min``).
    """
    text = unit.strip().lower().replace("µ", "u").replace("μ", "u")
    text = text.replace("⁻¹", "-1").replace("−", "-").replace("–", "-")
    text = text.replace("·", " ").replace("⋅", " ")
    parts = []
    for token in text.split():
        per = _PER.match(token)
        parts.append("/" + per.group(1) if per else token)
    text = "".join(parts)
    text = _MOLAR.get(text, text)
    head, *divisors = text.split("/")
    return "/".join([head, *sorted(divisors)])


# Molecular weights (g/mol) behind the molar factors.
_GLUCOSE_MW = 180.16
_MG_DL_PER_MMOL_L = _GLUCOSE_MW / 10.0          # 18.016
_PMOL_PER_UU_ML = 6.0                           # insulin, SI
_INSULIN_PMOL_L = 1.0 / _PMOL_PER_UU_ML
_KJ_PER_KCAL = 4.184
_KCAL_PER_KJ = 1.0 / _KJ_PER_KCAL


def _pg_ml_per_pmol_l(mw: float) -> float:
    """pg/mL carried by 1 pmol/L of a peptide of molecular weight ``mw``."""
    return mw / 1000.0


_INSULIN: dict[str, tuple[float, float]] = {
    "uu/ml": (1.0, 0.0),
    "uiu/ml": (1.0, 0.0),
    "miu/l": (1.0, 0.0),
    "mu/l": (1.0, 0.0),
    "pmol/l": (_INSULIN_PMOL_L, 0.0),
    "nmol/l": (_INSULIN_PMOL_L * 1000.0, 0.0),
}
_GLUCOSE_INFUSION: dict[str, tuple[float, float]] = {
    "mg/kg/min": (1.0, 0.0),
    "umol/kg/min": (_GLUCOSE_MW / 1000.0, 0.0),
    "mmol/kg/min": (_GLUCOSE_MW, 0.0),
}
_FRACTION: dict[str, tuple[float, float]] = {"fraction": (1.0, 0.0), "%": (0.01, 0.0)}
_HOURS: dict[str, tuple[float, float]] = {"h": (1.0, 0.0), "min": (1.0 / 60.0, 0.0)}
_GRAMS: dict[str, tuple[float, float]] = {"g": (1.0, 0.0)}

# marker or observable id -> {normalised paper unit: (factor, offset)}
CONVERSIONS: dict[str, dict[str, tuple[float, float]]] = {
    "glucose": {
        "mg/dl": (1.0, 0.0), "mg/100ml": (1.0, 0.0),
        "mmol/l": (_MG_DL_PER_MMOL_L, 0.0),
        "g/l": (100.0, 0.0), "mg/l": (0.1, 0.0),
    },
    "insulin": dict(_INSULIN),
    "glucagon": {
        "pg/ml": (1.0, 0.0), "ng/l": (1.0, 0.0), "pmol/l": (_pg_ml_per_pmol_l(3482.8), 0.0),
    },
    "ffa": {
        "mmol/l": (1.0, 0.0), "meq/l": (1.0, 0.0), "umol/l": (0.001, 0.0),
        "ueq/l": (0.001, 0.0),  # monocarboxylic acids: 1 uEq = 1 umol
    },
    "bhb": {  # beta-hydroxybutyric acid, 104.1 g/mol
        "mmol/l": (1.0, 0.0), "umol/l": (0.001, 0.0), "mg/dl": (10.0 / 104.1, 0.0),
    },
    "lactate": {  # lactic acid, 90.08 g/mol
        "mmol/l": (1.0, 0.0), "umol/l": (0.001, 0.0), "mg/dl": (10.0 / 90.08, 0.0),
    },
    "hepatic_output": dict(_GLUCOSE_INFUSION),
    "ghrelin": {
        "pg/ml": (1.0, 0.0), "ng/l": (1.0, 0.0), "ng/ml": (1000.0, 0.0),
        "pmol/l": (_pg_ml_per_pmol_l(3370.0), 0.0),
    },
    "leptin": {  # 16 kDa
        "ng/ml": (1.0, 0.0), "ug/l": (1.0, 0.0), "pg/ml": (0.001, 0.0), "pmol/l": (0.016, 0.0),
    },
    "glp1": {  # GLP-1(7-36) amide, 3297.7 g/mol; pg/mL = ng/L
        "pmol/l": (1.0, 0.0), "pg/ml": (1.0 / 3.2977, 0.0), "ng/l": (1.0 / 3.2977, 0.0),
        "nmol/l": (1000.0, 0.0),  # how a paper prints an area of it (nmol/L x min)
    },
    "cortisol": {  # 362.46 g/mol
        "ug/dl": (1.0, 0.0), "ug/l": (0.1, 0.0), "ng/ml": (0.1, 0.0),
        "nmol/l": (362.46 / 10000.0, 0.0),
    },
    "acth": {"pg/ml": (1.0, 0.0), "ng/l": (1.0, 0.0), "pmol/l": (_pg_ml_per_pmol_l(4541.0), 0.0)},
    "hr": {"bpm": (1.0, 0.0), "beats/min": (1.0, 0.0), "/min": (1.0, 0.0), "1/min": (1.0, 0.0)},
    "hrv": {"ms": (1.0, 0.0), "s": (1000.0, 0.0)},
    "sbp": {"mmhg": (1.0, 0.0), "mm/hg": (1.0, 0.0), "kpa": (7.50062, 0.0)},  # some papers print mm/Hg
    "dbp": {"mmhg": (1.0, 0.0), "mm/hg": (1.0, 0.0), "kpa": (7.50062, 0.0)},  # some papers print mm/Hg
    "temp": {
        "°c": (1.0, 0.0), "degc": (1.0, 0.0), "°f": (5.0 / 9.0, -160.0 / 9.0), "degf": (5.0 / 9.0, -160.0 / 9.0),
        "k": (1.0, -273.15),
    },
    "rr": {"/min": (1.0, 0.0), "1/min": (1.0, 0.0), "breaths/min": (1.0, 0.0)},
    "spo2": {"%": (1.0, 0.0), "fraction": (100.0, 0.0)},
    "liver_glycogen": dict(_GRAMS),
    "muscle_glycogen": dict(_GRAMS),
    "mitochondrial_capacity": {"×pop.mean": (1.0, 0.0), "x_pop_mean": (1.0, 0.0)},
    "crh": {"pg/ml": (1.0, 0.0), "ng/l": (1.0, 0.0), "pmol/l": (_pg_ml_per_pmol_l(4758.0), 0.0)},
    "insulin_action": {"a.u.": (1.0, 0.0)},
    "cck": {"pmol/l": (1.0, 0.0)},
    "gallbladder_bile": {"mmol": (1.0, 0.0), "umol": (0.001, 0.0)},
    "intestinal_bile": {"mmol": (1.0, 0.0), "umol": (0.001, 0.0)},
    "bile_acids": {"umol/l": (1.0, 0.0), "nmol/ml": (1.0, 0.0),
                   "nmol/l": (0.001, 0.0), "mmol/l": (1000.0, 0.0)},
    "fat_mass": {"kg": (1.0, 0.0), "g": (0.001, 0.0), "lb": (0.45359237, 0.0)},
    "insulin_slow": dict(_INSULIN),
    "insulin_potentiator": {"mg/dl": (1.0, 0.0)},
    "sns_tone": dict(_FRACTION),
    "epinephrine": {  # 183.2 g/mol
        "pg/ml": (1.0, 0.0), "ng/l": (1.0, 0.0), "ng/ml": (1000.0, 0.0),
        "pmol/l": (_pg_ml_per_pmol_l(183.2), 0.0), "nmol/l": (183.2, 0.0),
    },
    "gastric_carb": dict(_GRAMS),
    "gastric_protein": dict(_GRAMS),
    "gastric_fat": dict(_GRAMS),
    "circ_phase_offset": dict(_HOURS),
    "sleep_debt": dict(_HOURS),
    "gip": {  # GIP(1-42), 4983.5 g/mol
        "pmol/l": (1.0, 0.0), "pg/ml": (1000.0 / 4983.5, 0.0),
        "nmol/l": (1000.0, 0.0),  # how a paper prints an area of it (nmol/L x min)
    },
    # observables
    "glucose_infusion_rate": dict(_GLUCOSE_INFUSION),
    "glucose_disposal": dict(_GLUCOSE_INFUSION),
    "urinary_glucose": {
        "mg/min": (1.0, 0.0), "mg/h": (1.0 / 60.0, 0.0), "g/h": (1000.0 / 60.0, 0.0),
        "g/day": (1000.0 / 1440.0, 0.0), "g/24h": (1000.0 / 1440.0, 0.0),
        "mmol/min": (_GLUCOSE_MW, 0.0),
        "mmol/day": (_GLUCOSE_MW / 1440.0, 0.0), "mmol/24h": (_GLUCOSE_MW / 1440.0, 0.0),
    },
    "vo2": {"ml/min": (1.0, 0.0), "l/min": (1000.0, 0.0)},
    "vco2": {"ml/min": (1.0, 0.0), "l/min": (1000.0, 0.0)},
    "rq": {"ratio": (1.0, 0.0), "rq": (1.0, 0.0)},
    "energy_expenditure": {
        "kcal/min": (1.0, 0.0), "kcal/h": (1.0 / 60.0, 0.0),
        "kcal/day": (1.0 / 1440.0, 0.0), "kcal/24h": (1.0 / 1440.0, 0.0),
        "kj/min": (_KCAL_PER_KJ, 0.0), "w": (60.0 * _KCAL_PER_KJ / 1000.0, 0.0),
        "kj/day": (_KCAL_PER_KJ / 1440.0, 0.0), "mj/day": (1000.0 * _KCAL_PER_KJ / 1440.0, 0.0),
    },
    "lipolysis": {"mmol/l/min": (1.0, 0.0), "umol/l/min": (0.001, 0.0)},
    "gng_fraction": dict(_FRACTION),
    "fat_absorption": dict(_FRACTION),
}

TIME_UNITS: dict[str, float] = {"min": 1.0, "h": 60.0, "s": 1.0 / 60.0}

# A level outside these bounds, after conversion, is almost always a unit slip
# (5 read as mg/dL when it was mmol/L). The marker bounds are ``pulse.types``'
# display range; observables carry their own. A real value outside them is stated
# with ``unusual_value_ok`` in the finding.
_OBSERVABLE_RANGE: dict[str, tuple[float, float]] = {
    "glucose_infusion_rate": (0.0, 20.0),
    "glucose_disposal": (0.0, 30.0),
    "urinary_glucose": (0.0, 1000.0),
    "vo2": (50.0, 6000.0),
    "vco2": (50.0, 6000.0),
    "rq": (0.5, 1.5),
    "energy_expenditure": (0.2, 40.0),
    "lipolysis": (0.0, 5.0),
    "gng_fraction": (0.0, 1.0),
    "fat_absorption": (0.0, 1.0),
}


def native_unit(series: str) -> str:
    """The unit the model carries ``series`` in (the string ``pulse.types`` states)."""
    for m in MARKERS:
        if m.id == series:
            return m.unit
    for o in OBSERVABLES:
        if o.id == series:
            return o.unit
    raise UnitError(f"{series!r} is not an observed series")


def plausible_range(series: str) -> tuple[float, float]:
    for m in MARKERS:
        if m.id == series:
            return float(m.min), float(m.max)
    return _OBSERVABLE_RANGE[series]


def listed_units(series: str) -> tuple[str, ...]:
    if series not in SERIES_INDEX:
        raise UnitError(f"{series!r} is not an observed series")
    return tuple(CONVERSIONS[series])


def to_native(series: str, value: float, unit: str, *, difference: bool = False) -> float:
    """``value`` (a level, or with ``difference=True`` an interval) from ``unit`` to ``series``' unit."""
    table = CONVERSIONS.get(series)
    if table is None:
        raise UnitError(f"{series!r} has no unit table")
    entry = table.get(normalise(unit))
    if entry is None:
        raise UnitError(
            f"{unit!r} is not a listed unit for {series} (listed: {', '.join(table)}); "
            f"convert by hand and say so in the quote, or write a refusal",
        )
    factor, offset = entry
    return float(value) * factor if difference else float(value) * factor + offset


def time_to_minutes(value: float, unit: str) -> float:
    entry = TIME_UNITS.get(normalise(unit))
    if entry is None:
        raise UnitError(f"{unit!r} is not a listed time unit (listed: {', '.join(TIME_UNITS)})")
    return float(value) * entry
