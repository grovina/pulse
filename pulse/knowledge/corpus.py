"""The literature corpus as files: one paper, one file, behind ``pulse.admit``.

``corpus/findings/<paper>.json`` holds a paper's citation, its population, the
protocols it ran and its findings; ``corpus/refusals/<paper>.json`` records a
paper the model cannot yet express. ``corpus/README.md`` is the guide a reader is
handed with one paper. This module parses both strictly, the way soma's catalog
does: an unknown key, a missing field, an unknown marker or unit is an error, and
**all** the problems of a file are reported together (``CorpusError``).

A finding file becomes ``CohortStatisticSpec`` s that are joined into the registry
in ``cohort_statistics`` and pass through ``apply_seal`` like every authored item.
Numbers are stated in the paper's unit and converted by ``corpus_units``;
uncertainty is stated as a kind and turned into ``sigma`` / ``n_arm`` by
``corpus_uncertainty``. A file never states ``sigma``, ``n_arm``, ``weight``,
``shape`` or ``sealed``.

The meal tuple of a ``CohortArmSpec`` is ``(t_min, carbs_g, fat_g, protein_g)``
(``cohort_loss._meals_from_spec`` unpacks it so), not the protein-then-fat order
``AUTHORING.md`` used to say. A file names the three fields; this module writes
the tuple.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..types import (
    BODY_MASS_KG, MG_DL_PER_G, SERIES_INDEX, VG_DL_PER_KG,
)
from .cohort_types import (
    FACTORS,
    SAMPLED_KINDS,
    TWO_ARM_KINDS,
    BelowBaseline,
    CohortArmSpec,
    CohortStatisticSpec,
    CurvePoint,
    InitMode,
    Population,
    StatisticKind,
    StatisticWindow,
)
from .corpus_uncertainty import (
    ALLOWED_KINDS,
    UNCERTAINTY_KINDS,
    Uncertainty,
    UncertaintyError,
    curve_point_sigma,
    to_sigma,
)
from .corpus_units import (
    UnitError,
    listed_units,
    normalise,
    plausible_range,
    time_to_minutes,
    to_native,
)

__all__ = [
    "CORPUS_ENV",
    "QUEUE_FORMS",
    "QUEUE_PROTOCOL_KINDS",
    "QUEUE_UNCERTAINTY",
    "REFUSAL_KINDS",
    "Corpus",
    "CorpusError",
    "Finding",
    "FindingItem",
    "Problem",
    "QueueItem",
    "Refusal",
    "corpus_root",
    "count_refusals",
    "load_corpus",
    "load_file_specs",
    "count_piles",
    "parse_finding",
    "parse_queue_item",
    "parse_refusal",
    "read_finding",
    "read_queue_item",
    "read_refusal",
]

CORPUS_ENV = "PULSE_CORPUS_DIR"

# Why a paper, or part of one, could not be entered: a piece the model lacks, or
# (``source``) something the paper itself does not give (a number only in a figure,
# an error bar nobody labelled, an exercise intensity it never states).
REFUSAL_KINDS = ("marker", "input", "population", "timescale", "operator", "source")
LEFT_OUT_KINDS = REFUSAL_KINDS

# A queued paper: what a reader needs to start, and nothing the reader must derive.
QUEUE_PROTOCOL_KINDS = (
    "OGTT", "mixed meal", "clamp", "24 h profile", "IVGTT", "exercise", "prolonged fast", "other",
)
QUEUE_FORMS = ("table", "text", "figure")
QUEUE_UNCERTAINTY = ("sd", "sem", "ci", "range", "unlabelled")
QUEUE_POPULATIONS = ("healthy", "nonhealthy", "mixed")

_SLUG = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")
_NAME = re.compile(r"[a-z][a-z0-9_]*")
_DOI = re.compile(r"10\.\d{4,9}/\S+")
_PMID = re.compile(r"\d{1,9}")
_PMCID = re.compile(r"PMC\d{1,9}")

# Checks a problem can belong to; ``pulse.admit`` reports each as one line.
CHECKS = ("parse", "protocol", "units", "place", "uncertainty")


@dataclass(frozen=True)
class Problem:
    check: str
    message: str

    def __str__(self) -> str:
        return self.message


class CorpusError(ValueError):
    """Every problem found in one file (or across the corpus)."""

    def __init__(self, problems: list[Problem], paper: str = "") -> None:
        self.problems = list(problems)
        self.paper = paper
        super().__init__("; ".join(p.message for p in self.problems))

    @property
    def reasons(self) -> list[str]:
        return [p.message for p in self.problems]


def corpus_root() -> Path:
    """``corpus/`` of this checkout, or ``$PULSE_CORPUS_DIR``.

    The trainer image copies ``corpus/`` next to ``pulse/`` and sets the variable;
    an installed (non-editable) package has no repo beside it and needs the
    variable. A missing directory is an error, not an empty corpus: a trainer
    that silently lost its corpus would train on less than it reports.
    """
    env = os.environ.get(CORPUS_ENV)
    root = Path(env) if env else Path(__file__).resolve().parents[2] / "corpus"
    if not root.is_dir():
        raise FileNotFoundError(
            f"no corpus directory at {root}; set {CORPUS_ENV} or ship corpus/ beside pulse/",
        )
    return root


# --------------------------------------------------------------------------------
# A strict reader
# --------------------------------------------------------------------------------

class _Reader:
    """Collects problems while reading JSON objects field by field."""

    def __init__(self) -> None:
        self.problems: list[Problem] = []

    def add(self, check: str, message: str) -> None:
        self.problems.append(Problem(check, message))

    def obj(
        self, data: Any, where: str, required: set[str] | frozenset[str],
        optional: set[str] | frozenset[str] = frozenset(), check: str = "parse",
    ) -> dict | None:
        if not isinstance(data, dict):
            self.add(check, f"{where} is not an object")
            return None
        extra = set(data) - set(required) - set(optional)
        if extra:
            self.add(check, f"{where} has unknown fields {sorted(extra)}")
        missing = set(required) - set(data)
        if missing:
            self.add(check, f"{where} is missing {sorted(missing)}")
        return data

    def string(self, d: dict, key: str, where: str, check: str = "parse") -> str | None:
        if key not in d:
            return None
        v = d[key]
        if not isinstance(v, str) or not v.strip():
            self.add(check, f"{where} {key} is empty or not text")
            return None
        return v.strip()

    def number(
        self, d: dict, key: str, where: str, check: str = "parse", *, positive: bool = False,
        nonneg: bool = False, null_ok: bool = False,
    ) -> float | None:
        if key not in d:
            return None
        v = d[key]
        if v is None and null_ok:
            return None
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            self.add(check, f"{where} {key} is not a number")
            return None
        x = float(v)
        if x != x or x in (float("inf"), float("-inf")):
            self.add(check, f"{where} {key} is not finite")
            return None
        if positive and not x > 0:
            self.add(check, f"{where} {key} must be positive")
            return None
        if nonneg and x < 0:
            self.add(check, f"{where} {key} must not be negative")
            return None
        return x

    def integer(
        self, d: dict, key: str, where: str, check: str = "parse", *, minimum: int | None = None,
    ) -> int | None:
        if key not in d:
            return None
        v = d[key]
        if isinstance(v, bool) or not isinstance(v, int):
            self.add(check, f"{where} {key} is not a whole number")
            return None
        if minimum is not None and v < minimum:
            self.add(check, f"{where} {key} must be at least {minimum}")
            return None
        return int(v)

    def choice(
        self, d: dict, key: str, where: str, options: tuple[str, ...], check: str = "parse",
    ) -> str | None:
        v = self.string(d, key, where, check)
        if v is None:
            return None
        if v not in options:
            self.add(check, f"{where} {key} {v!r} is not one of {', '.join(options)}")
            return None
        return v


# --------------------------------------------------------------------------------
# Parsed forms
# --------------------------------------------------------------------------------

@dataclass(frozen=True)
class Citation:
    authors: str
    title: str
    journal: str
    year: int
    doi: str | None = None
    pmid: str | None = None
    pmcid: str | None = None

    def identifiers(self) -> frozenset[str]:
        out = set()
        if self.doi:
            out.add(f"doi:{self.doi.lower()}")
        if self.pmid:
            out.add(f"pmid:{self.pmid}")
        if self.pmcid:
            out.add(f"pmcid:{self.pmcid.upper()}")
        return frozenset(out)

    def short(self) -> str:
        first = self.authors.split(",")[0].split(" and ")[0].strip()
        return f"{first} ({self.year}) {self.journal}"


@dataclass(frozen=True)
class Location:
    where: str
    quote: str


@dataclass(frozen=True)
class FindingItem:
    id: str
    spec: CohortStatisticSpec
    location: Location
    acknowledged_conflicts: tuple[str, ...] = ()


@dataclass(frozen=True)
class LeftOut:
    """Something the paper reports that its finding file could not carry, and why."""

    what: str
    kind: str  # one of LEFT_OUT_KINDS
    name: str


@dataclass(frozen=True)
class Finding:
    paper: str
    citation: Citation
    n: int
    protocols: dict[str, CohortArmSpec]
    items: tuple[FindingItem, ...]
    path: Path | None = None
    # protocol name -> the keys whose values the reader assumed (the paper does not state them)
    assumed: dict[str, tuple[str, ...]] = field(default_factory=dict)
    left_out: tuple[LeftOut, ...] = ()

    @property
    def specs(self) -> list[CohortStatisticSpec]:
        return [item.spec for item in self.items]


@dataclass(frozen=True)
class Reported:
    what: str
    location: Location


@dataclass(frozen=True)
class Refusal:
    paper: str
    citation: Citation
    reported: tuple[Reported, ...]
    kind: str
    name: str
    detail: str
    path: Path | None = None


@dataclass(frozen=True)
class QueueReport:
    marker: str  # free text: a queued paper may report what the model has no series for
    where: str
    form: str


@dataclass(frozen=True)
class QueueItem:
    """A paper waiting to be read. Carries no numbers and no seal: the seal follows from
    the slug (``sealing.paper_is_sealed``) and would go stale if it were written down."""

    paper: str
    citation: str
    doi: str | None
    pmid: str | None
    pmcid: str | None
    licence: str
    licence_url: str | None
    protocol: str
    protocol_kinds: tuple[str, ...]
    population: str
    population_class: str
    reports: tuple[QueueReport, ...]
    uncertainty: str
    figure_only: bool
    note: str
    path: Path | None = None

    def identifiers(self) -> frozenset[str]:
        out = set()
        if self.doi:
            out.add(f"doi:{self.doi.lower()}")
        if self.pmid:
            out.add(f"pmid:{self.pmid}")
        if self.pmcid:
            out.add(f"pmcid:{self.pmcid.upper()}")
        return frozenset(out)


@dataclass(frozen=True)
class Corpus:
    findings: tuple[Finding, ...] = ()
    refusals: tuple[Refusal, ...] = ()
    specs: tuple[CohortStatisticSpec, ...] = field(default=())
    queue: tuple[QueueItem, ...] = ()

    @property
    def counts(self) -> dict[str, int]:
        """Papers in each of the three piles. A paper is in exactly one."""
        return {"queue": len(self.queue), "findings": len(self.findings), "refusals": len(self.refusals)}


# --------------------------------------------------------------------------------
# Citation, location, population
# --------------------------------------------------------------------------------

def _citation(r: _Reader, data: Any, paper: str) -> Citation | None:
    d = r.obj(
        data, "citation", {"authors", "title", "journal", "year"}, {"doi", "pmid", "pmcid"},
    )
    if d is None:
        return None
    authors = r.string(d, "authors", "citation")
    title = r.string(d, "title", "citation")
    journal = r.string(d, "journal", "citation")
    year = r.integer(d, "year", "citation", minimum=1800)
    ids: dict[str, str | None] = {}
    for key, pattern in (("doi", _DOI), ("pmid", _PMID), ("pmcid", _PMCID)):
        value = r.string(d, key, "citation") if key in d else None
        if value is not None and not pattern.fullmatch(value):
            r.add("parse", f"citation {key} {value!r} is not a {key.upper()}")
            value = None
        ids[key] = value
    if not any(d.get(k) for k in ("doi", "pmid", "pmcid")):
        r.add("parse", "citation needs at least one identifier: doi, pmid or pmcid")
    year_in_slug = re.search(r"-(\d{4})$", paper)
    if year_in_slug and year is not None and int(year_in_slug.group(1)) != year:
        r.add("parse", f"paper {paper!r} ends in a year but citation year is {year}")
    if None in (authors, title, journal, year):
        return None
    assert authors and title and journal and year is not None
    return Citation(authors, title, journal, year, ids["doi"], ids["pmid"], ids["pmcid"])


def _location(r: _Reader, data: Any, where: str) -> Location | None:
    d = r.obj(data, f"{where} location", {"where", "quote"}, check="place")
    if d is None:
        return None
    place = r.string(d, "where", f"{where} location", "place")
    quote = r.string(d, "quote", f"{where} location", "place")
    if place is not None and len(place) < 3:
        r.add("place", f"{where} location where {place!r} does not name a table, figure or section")
        place = None
    if quote is not None and not re.search(r"\d", quote):
        r.add("place", f"{where} location quote {quote!r} contains no number: quote the reported text")
        quote = None
    if place is None or quote is None:
        return None
    return Location(place, quote)


@dataclass(frozen=True)
class _Summary:
    mean: float | None
    sd: float | None
    low: float | None
    high: float | None

    def span(self) -> tuple[float, float] | None:
        if self.low is not None and self.high is not None:
            return (self.low, self.high)
        if self.mean is not None and self.sd is not None:
            return (self.mean - 2.0 * self.sd, self.mean + 2.0 * self.sd)
        return None


@dataclass(frozen=True)
class _Who:
    n: int
    sex: str | None
    age: tuple[float, float] | None
    bmi: tuple[float, float] | None
    healthy: bool


def _summary(r: _Reader, data: Any, where: str) -> _Summary | None:
    if data is None:
        return _Summary(None, None, None, None)
    d = r.obj(data, where, {"mean", "sd", "min", "max"})
    if d is None:
        return None
    vals = [r.number(d, k, where, null_ok=True) for k in ("mean", "sd", "min", "max")]
    mean, sd, lo, hi = vals
    if all(v is None for v in vals):
        r.add("parse", f"{where} states nothing; write null if the paper does not report it")
    if lo is not None and hi is not None and lo > hi:
        r.add("parse", f"{where} min is above max")
    if (lo is None) != (hi is None):
        r.add("parse", f"{where} gives only one end of the range")
    return _Summary(mean, sd, lo, hi)


def _population(r: _Reader, data: Any) -> _Who | None:
    d = r.obj(
        data, "population", {"n", "sex", "age_years", "bmi", "status", "description"},
    )
    if d is None:
        return None
    n = r.integer(d, "n", "population", minimum=1)
    sex = r.choice(d, "sex", "population", ("female", "male", "mixed"))
    status = r.choice(d, "status", "population", ("healthy", "unselected"))
    r.string(d, "description", "population")
    age = _summary(r, d.get("age_years"), "population age_years") if "age_years" in d else None
    bmi = _summary(r, d.get("bmi"), "population bmi") if "bmi" in d else None
    if None in (n, sex, status) or age is None and "age_years" in d or bmi is None and "bmi" in d:
        return None
    assert n is not None
    return _Who(
        n, None if sex == "mixed" else sex,
        age.span() if age else None, bmi.span() if bmi else None, status == "healthy",
    )


def _arm_population(who: _Who, override: dict | None) -> Population | None:
    sex, age, bmi, healthy = who.sex, who.age, who.bmi, who.healthy
    if override:
        sex = override["sex"] if "sex" in override else sex
        age = override["age"] if "age" in override else age
        bmi = override["bmi"] if "bmi" in override else bmi
        healthy = override["healthy"] if "healthy" in override else healthy
    if sex is None and age is None and bmi is None and not healthy:
        return None

    def span(s: tuple[float, float] | None) -> str:
        return "any" if s is None else f"{s[0]:g}-{s[1]:g}"

    name = f"{sex or 'mixed'}_age{span(age)}_bmi{span(bmi)}_{'healthy' if healthy else 'unselected'}"
    return Population(name=name, sex=sex, age=age, bmi=bmi, healthy=healthy)


# --------------------------------------------------------------------------------
# Protocols
# --------------------------------------------------------------------------------

_SERIES_INPUTS = {
    # name: (baseline, top, unit table or None)
    "sleep_wake": (1.0, 1.0),
    "activity": (0.0, 1.0),
    "glucose_infusion": (0.0, None),
    "glucose_clamp": (0.0, None),
    "insulin_clamp": (0.0, None),
    "sglt2": (0.0, 1.0),
    "bile_fistula": (0.0, 1.0),
}
_INPUT_UNITS = {"glucose_infusion", "glucose_clamp", "insulin_clamp"}


def _series_input(
    r: _Reader, data: Any, name: str, where: str, duration: int,
) -> tuple[float, ...] | None:
    required = {"segments", "unit"} if name in _INPUT_UNITS else {"segments"}
    d = r.obj(data, f"{where} {name}", required, check="protocol")
    if d is None:
        return None
    baseline, top = _SERIES_INPUTS[name]
    scale = 1.0
    if name in _INPUT_UNITS:
        unit = r.string(d, "unit", f"{where} {name}", "protocol")
        if unit is not None:
            try:
                if name == "glucose_infusion":
                    scale = {"mg/dl/min": 1.0, "mg/kg/min": 1.0 / VG_DL_PER_KG}[normalise(unit)]
                elif name == "glucose_clamp":
                    scale = to_native("glucose", 1.0, unit, difference=True)
                else:
                    scale = to_native("insulin", 1.0, unit, difference=True)
            except (KeyError, UnitError):
                units = {
                    "glucose_infusion": "mg/dL/min, mg/kg/min",
                    "glucose_clamp": ", ".join(listed_units("glucose")),
                    "insulin_clamp": ", ".join(listed_units("insulin")),
                }[name]
                r.add("units", f"{where} {name} unit {unit!r} is not listed (listed: {units})")
    segments = d.get("segments")
    if not isinstance(segments, list) or not segments:
        r.add("protocol", f"{where} {name} segments is not a non-empty list")
        return None
    series = [baseline] * duration
    taken = [False] * duration
    ok = True
    for i, seg in enumerate(segments):
        w = f"{where} {name} segment {i + 1}"
        sd = r.obj(seg, w, {"from_min", "to_min", "level"}, check="protocol")
        if sd is None:
            ok = False
            continue
        lo = r.integer(sd, "from_min", w, "protocol", minimum=0)
        hi = r.integer(sd, "to_min", w, "protocol")
        level = r.number(sd, "level", w, "protocol", nonneg=True)
        if lo is None or hi is None or level is None:
            ok = False
            continue
        if not lo < hi <= duration:
            r.add("protocol", f"{w} must satisfy from_min < to_min <= duration_min ({duration})")
            ok = False
            continue
        if top is not None and level > top:
            r.add("protocol", f"{w} level {level:g} is above {top:g}")
            ok = False
            continue
        if any(taken[lo:hi]):
            r.add("protocol", f"{w} overlaps another segment")
            ok = False
            continue
        for t in range(lo, hi):
            series[t] = level * scale
            taken[t] = True
    return tuple(series) if ok else None


# Keys of a protocol that say nothing about what the subjects did, so cannot be assumed.
_NOT_ASSUMABLE = frozenset({"name", "description", "population", "assumed"})


def _assumed(r: _Reader, d: dict, where: str) -> tuple[str, ...] | None:
    """``assumed``: the protocol's own keys whose values the paper does not state."""
    raw = d.get("assumed")
    if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
        r.add("protocol", f"{where} assumed is not a list of this protocol's keys ([] if the paper states them all)")
        return None
    stated = sorted(set(d) - _NOT_ASSUMABLE)
    ok = True
    for key in raw:
        if key not in stated:
            r.add("protocol", f"{where} assumed names {key!r}, which is not one of its keys ({', '.join(stated)})")
            ok = False
    if len(set(raw)) != len(raw):
        r.add("protocol", f"{where} assumed repeats a key")
        ok = False
    return tuple(raw) if ok else None


def _protocol(
    r: _Reader, data: Any, index: int, who: _Who,
) -> tuple[str, CohortArmSpec, InitMode, tuple[str, ...]] | None:
    where = f"protocol {index + 1}"
    optional = set(_SERIES_INPUTS) | {"glucose_bolus", "population", "prefast_hours", "init"}
    d = r.obj(
        data, where, {"name", "description", "duration_min", "start_hour", "meals", "assumed"}, optional,
        check="protocol",
    )
    if d is None:
        return None
    name = r.string(d, "name", where, "protocol")
    if name is not None:
        if not _NAME.fullmatch(name):
            r.add("protocol", f"{where} name {name!r} is not lowercase letters, digits and underscores")
            name = None
        else:
            where = f"protocol {name}"
    r.string(d, "description", where, "protocol")
    duration = r.integer(d, "duration_min", where, "protocol", minimum=1)
    start = r.number(d, "start_hour", where, "protocol")
    if start is not None and not 0.0 <= start < 24.0:
        r.add("protocol", f"{where} start_hour must be in [0, 24)")
        start = None
    prefast = r.number(d, "prefast_hours", where, "protocol", nonneg=True) if "prefast_hours" in d else 0.0
    init = InitMode.COLD
    if "init" in d:
        mode = r.choice(d, "init", where, tuple(m.value for m in InitMode), "protocol")
        init = InitMode(mode) if mode else init
    assumed = _assumed(r, d, where) if "assumed" in d else None

    meals: list[tuple[float, float, float, float]] = []
    raw_meals = d.get("meals")
    if "meals" in d and not isinstance(raw_meals, list):
        r.add("protocol", f"{where} meals is not a list")
    for i, m in enumerate(raw_meals if isinstance(raw_meals, list) else []):
        w = f"{where} meal {i + 1}"
        md = r.obj(m, w, {"t_min", "carbs_g", "fat_g", "protein_g"}, check="protocol")
        if md is None:
            continue
        t = r.number(md, "t_min", w, "protocol", nonneg=True)
        c = r.number(md, "carbs_g", w, "protocol", nonneg=True)
        f = r.number(md, "fat_g", w, "protocol", nonneg=True)
        p = r.number(md, "protein_g", w, "protocol", nonneg=True)
        if None in (t, c, f, p):
            continue
        assert t is not None and c is not None and f is not None and p is not None
        if duration is not None and t >= duration:
            r.add("protocol", f"{w} t_min {t:g} is not before the arm ends (duration_min {duration})")
        meals.append((t, c, f, p))

    series: dict[str, tuple[float, ...] | None] = {}
    if duration is not None:
        for key in _SERIES_INPUTS:
            if key in d:
                series[key] = _series_input(r, d[key], key, where, duration)
        if "glucose_bolus" in d:
            bd = r.obj(
                d["glucose_bolus"], f"{where} glucose_bolus",
                {"dose_g_per_kg", "start_min", "over_min"}, check="protocol",
            )
            if bd is not None:
                dose = r.number(bd, "dose_g_per_kg", f"{where} glucose_bolus", "protocol", positive=True)
                t0 = r.integer(bd, "start_min", f"{where} glucose_bolus", "protocol", minimum=0)
                over = r.integer(bd, "over_min", f"{where} glucose_bolus", "protocol", minimum=1)
                if dose is not None and t0 is not None and over is not None:
                    if t0 + over > duration:
                        r.add("protocol", f"{where} glucose_bolus runs past the arm")
                    else:
                        rate = dose * BODY_MASS_KG / over * MG_DL_PER_G
                        base = list(series.get("glucose_infusion") or (0.0,) * duration)
                        for t in range(t0, t0 + over):
                            base[t] += rate
                        series["glucose_infusion"] = tuple(base)

    override: dict | None = None
    if "population" in d:
        pd = r.obj(
            d["population"], f"{where} population", set(),
            {"sex", "age_years", "bmi", "status"}, check="protocol",
        )
        override = {}
        if pd is not None:
            if "sex" in pd:
                s = r.choice(pd, "sex", f"{where} population", ("female", "male", "mixed"), "protocol")
                if s:
                    override["sex"] = None if s == "mixed" else s
            for key, field_name in (("age_years", "age"), ("bmi", "bmi")):
                if key in pd:
                    if pd[key] is None:
                        override[field_name] = None
                        continue
                    summary = _summary(r, pd[key], f"{where} population {key}")
                    if summary is None:
                        continue
                    if summary.span() is None:
                        r.add(
                            "protocol",
                            f"{where} population {key} gives no range: "
                            f"it needs min and max, or mean and sd",
                        )
                    else:
                        override[field_name] = summary.span()
            if "status" in pd:
                s = r.choice(pd, "status", f"{where} population", ("healthy", "unselected"), "protocol")
                if s:
                    override["healthy"] = s == "healthy"

    if name is None or duration is None or start is None or prefast is None or assumed is None:
        return None
    if any(series.get(k, 0) is None for k in series):
        return None
    arm = CohortArmSpec(
        label=name, duration_min=duration, start_hour=start, meals=tuple(sorted(meals)),
        sleep_wake=series.get("sleep_wake"), activity=series.get("activity"),
        glucose_infusion=series.get("glucose_infusion"), glucose_clamp=series.get("glucose_clamp"),
        insulin_clamp=series.get("insulin_clamp"), sglt2=series.get("sglt2"),
        bile_fistula=series.get("bile_fistula"), prefast_hours=prefast,
        population=_arm_population(who, override),
    )
    return name, arm, init, assumed


# --------------------------------------------------------------------------------
# Findings
# --------------------------------------------------------------------------------

# An area is carried as a time-weighted mean, so a total area is a level and an
# incremental area (a mean of a difference) is a difference.
_LEVEL_KINDS = frozenset({
    StatisticKind.MEAN_IN_WINDOW, StatisticKind.PEAK_VALUE, StatisticKind.AREA_UNDER_CURVE,
})
_DIFF_KINDS = frozenset({
    StatisticKind.DELTA_MEANS, StatisticKind.DELTA_PEAKS, StatisticKind.SPREAD,
    StatisticKind.CHANGE_FROM_BASELINE, StatisticKind.INCREMENTAL_AREA,
})
# The paper's time unit for an area, in minutes.
_TIME_UNITS = {"min": 1.0, "h": 60.0}
_RATIO_KINDS = frozenset({StatisticKind.RATIO_MEANS, StatisticKind.RATIO_INCREMENTS})


@dataclass(frozen=True)
class _Units:
    """Convert a finding's numbers from its stated unit.

    ``scale`` turns a printed area into the time-weighted mean the spec carries:
    minutes per printed time unit over the span of the sample times. It is 1 for
    every other statistic, and applies to a value and to every width alike.
    """

    kind: StatisticKind
    marker: str
    unit: str
    scale: float = 1.0

    def check_unit(self) -> None:
        if self.kind in _RATIO_KINDS:
            if normalise(self.unit) != "ratio":
                raise UnitError(f"a {self.kind.value} is dimensionless: unit must be 'ratio'")
        elif self.kind is StatisticKind.CORRELATION:
            if normalise(self.unit) != "r":
                raise UnitError("a correlation is dimensionless: unit must be 'r'")
        elif self.kind is StatisticKind.TIME_TO_PEAK:
            time_to_minutes(1.0, self.unit)
        else:
            to_native(self.marker, 0.0, self.unit)

    def level(self, v: float) -> float:
        if self.kind in _RATIO_KINDS | {StatisticKind.CORRELATION}:
            return v
        if self.kind is StatisticKind.TIME_TO_PEAK:
            return time_to_minutes(v, self.unit)
        if self.kind in _DIFF_KINDS:
            return to_native(self.marker, v * self.scale, self.unit, difference=True)
        return to_native(self.marker, v * self.scale, self.unit)

    def width(self, v: float) -> float:
        if self.kind in _RATIO_KINDS | {StatisticKind.CORRELATION}:
            return v
        if self.kind is StatisticKind.TIME_TO_PEAK:
            return time_to_minutes(v, self.unit)
        return to_native(self.marker, v * self.scale, self.unit, difference=True)


_UNCERTAINTY_KEYS = {
    "sd": ({"kind", "n", "value"}, set()),
    "sem": ({"kind", "n", "value"}, {"observations"}),
    "ci": ({"kind", "n", "low", "high", "level"}, {"observations"}),
    "range": ({"kind", "n", "low", "high"}, set()),
    "n_only": ({"kind", "n"}, set()),
}


def _uncertainty(
    r: _Reader, data: Any, where: str, kind: StatisticKind, units: _Units, reported: float | None,
) -> Uncertainty | None:
    """The stated uncertainty, with its widths converted to the model's unit.

    A confidence interval or range matters only by its width, so it is stored as
    ``low=0, high=width``. ``reported`` (the statistic in the paper's unit) has to
    lie inside its own interval, which catches a value and an interval in two units.
    """
    if not isinstance(data, dict):
        r.add("uncertainty", f"{where} uncertainty is not an object")
        return None
    k = data.get("kind")
    if k not in UNCERTAINTY_KINDS:
        r.add(
            "uncertainty",
            f"{where} uncertainty kind {k!r} is not one of {', '.join(UNCERTAINTY_KINDS)}",
        )
        return None
    required, optional = _UNCERTAINTY_KEYS[k]
    d = r.obj(data, f"{where} uncertainty", required, optional, check="uncertainty")
    assert d is not None
    w = f"{where} uncertainty"
    n = r.integer(d, "n", w, "uncertainty", minimum=1)
    value = low = high = None
    level = None
    if k in ("sd", "sem"):
        value = r.number(d, "value", w, "uncertainty", positive=True)
    elif k in ("ci", "range"):
        low = r.number(d, "low", w, "uncertainty")
        high = r.number(d, "high", w, "uncertainty")
        if k == "ci":
            level = r.integer(d, "level", w, "uncertainty")
    observations = None
    if "observations" in d:
        observations = r.integer(d, "observations", w, "uncertainty", minimum=1)
        if observations is None:
            return None
    if n is None or (k in ("sd", "sem") and value is None) or (
        k in ("ci", "range") and (low is None or high is None)
    ):
        return None
    if low is not None and high is not None and reported is not None:
        if not min(low, high) <= reported <= max(low, high):
            r.add("uncertainty", f"{where}: the value {reported:g} is outside its {k} [{low:g}, {high:g}]")
            return None
    try:
        u = Uncertainty(
            k, n,
            None if value is None else units.width(value),
            None if low is None or high is None else 0.0,
            None if low is None or high is None else units.width(high - low),
            level,
            observations,
        )
    except UnitError as err:
        r.add("units", f"{w}: {err}")
        return None
    problems = u.problems()
    if problems:
        for p in problems:
            r.add("uncertainty", f"{where} {p}")
        return None
    if u.kind not in ALLOWED_KINDS[kind]:
        r.add(
            "uncertainty",
            f"{where}: a {kind.value} takes an uncertainty of kind "
            f"{', '.join(sorted(ALLOWED_KINDS[kind]))}, not {u.kind}",
        )
        return None
    return u


_FINDING_REQUIRED = {"id", "description", "marker", "statistic", "protocols", "unit", "location"}
_FINDING_OPTIONAL = {"unusual_value_ok", "acknowledged_conflicts"}


def _window(r: _Reader, data: Any, where: str, duration: int) -> StatisticWindow | None:
    d = r.obj(data, where, {"start_min", "end_min"}, check="protocol")
    if d is None:
        return None
    lo = r.integer(d, "start_min", where, "protocol", minimum=0)
    hi = r.integer(d, "end_min", where, "protocol")
    if lo is None or hi is None:
        return None
    if not lo < hi <= duration:
        r.add("protocol", f"{where} must satisfy start_min < end_min <= duration_min ({duration})")
        return None
    return StatisticWindow(lo, hi)


def _sample_times(r: _Reader, data: Any, where: str, duration: int) -> tuple[int, ...] | None:
    """The paper's sampling minutes: whole minutes, at least two, strictly increasing,
    inside the arm (an arm has a state at minutes ``0 .. duration - 1``)."""
    if not isinstance(data, list) or len(data) < 2:
        r.add("parse", f"{where} times is not a list of at least two minutes")
        return None
    if any(isinstance(t, bool) or not isinstance(t, int) for t in data):
        r.add("parse", f"{where} times are not whole minutes")
        return None
    if any(b <= a for a, b in zip(data, data[1:])):
        r.add("parse", f"{where} times are not strictly increasing")
        return None
    if data[0] < 0 or data[-1] >= duration:
        r.add("protocol", (
            f"{where} times run {data[0]}..{data[-1]}, outside the arm's minutes 0..{duration - 1}: "
            f"an arm sampled at minute {data[-1]} needs duration_min {data[-1] + 1}"
        ))
        return None
    return tuple(data)


def _finding_item(
    r: _Reader, data: Any, index: int, paper: str, citation: Citation | None,
    protocols: dict[str, CohortArmSpec], inits_by_name: dict[str, InitMode],
) -> tuple[FindingItem | None, set[str]]:
    where = f"finding {index + 1}"
    used: set[str] = set()
    if not isinstance(data, dict):
        r.add("parse", f"{where} is not an object")
        return None, used
    fid = data.get("id")
    if isinstance(fid, str) and _NAME.fullmatch(fid):
        where = f"finding {fid}"
    stat = data.get("statistic")
    try:
        kind = StatisticKind(stat)
    except ValueError:
        r.add("parse", f"{where} statistic {stat!r} is not one of {', '.join(k.value for k in StatisticKind)}")
        return None, used
    required = set(_FINDING_REQUIRED)
    optional = set(_FINDING_OPTIONAL)
    if kind is StatisticKind.CURVE:
        required |= {"points"}
    elif kind in SAMPLED_KINDS:
        # The window is the sample times', so it is not stated.
        required |= {"times", "time_unit", "value", "uncertainty"}
    else:
        required |= {"window", "value", "uncertainty"}
    if kind in TWO_ARM_KINDS:
        optional |= {"per_arm_windows"}
    if kind in (StatisticKind.RATIO_INCREMENTS, StatisticKind.CHANGE_FROM_BASELINE):
        required |= {"baseline_window"}
    if kind is StatisticKind.INCREMENTAL_AREA:
        required |= {"below_baseline"}
        optional |= {"baseline_window"}
    if kind is StatisticKind.CORRELATION:
        required |= {"partner"}
    d = r.obj(data, where, required, optional)
    if d is None:
        return None, used
    fid = r.string(d, "id", where)
    if fid is not None and not _NAME.fullmatch(fid):
        r.add("parse", f"{where} id {fid!r} is not lowercase letters, digits and underscores")
        fid = None
    description = r.string(d, "description", where)
    marker = r.string(d, "marker", where)
    if marker is not None and marker not in SERIES_INDEX:
        import difflib
        near = difflib.get_close_matches(marker, list(SERIES_INDEX), n=3)
        r.add("parse", f"{where} marker {marker!r} is not a series the model carries"
              + (f" (did you mean {', '.join(near)}?)" if near else ""))
        marker = None
    unit = r.string(d, "unit", where, "units")
    location = _location(r, d.get("location"), where) if "location" in d else None

    # Protocols
    names = d.get("protocols")
    arms: list[CohortArmSpec] = []
    arm_inits: list[InitMode] = []
    if "protocols" in d:
        want = 2 if kind in TWO_ARM_KINDS else 1
        if not isinstance(names, list) or not all(isinstance(x, str) for x in names):
            r.add("protocol", f"{where} protocols is not a list of protocol names")
        elif len(names) != want:
            r.add("protocol", f"{where}: a {kind.value} runs {want} protocol(s), got {len(names)}")
        else:
            for nm in names:
                if nm not in protocols:
                    r.add("protocol", f"{where} names protocol {nm!r}, which the file does not declare")
                else:
                    arms.append(protocols[nm])
                    arm_inits.append(inits_by_name[nm])
                    used.add(nm)
    ok_protocols = len(arms) == (2 if kind in TWO_ARM_KINDS else 1)
    inits = set(arm_inits)
    if len(inits) > 1:
        r.add("protocol", f"{where}: its protocols disagree on init")
    init = next(iter(inits)) if len(inits) == 1 else InitMode.COLD

    duration = min((a.duration_min for a in arms), default=0)

    # The sampling minutes of an area
    sample_times: tuple[int, ...] | None = None
    below_baseline: BelowBaseline | None = None
    scale = 1.0
    if kind in SAMPLED_KINDS:
        sample_times = _sample_times(r, d.get("times"), where, duration) if "times" in d and duration else None
        time_unit = r.choice(d, "time_unit", where, tuple(_TIME_UNITS), "units") if "time_unit" in d else None
        if sample_times is not None and time_unit is not None:
            scale = _TIME_UNITS[time_unit] / (sample_times[-1] - sample_times[0])
        else:
            sample_times = None
    if kind is StatisticKind.INCREMENTAL_AREA and "below_baseline" in d:
        raw = r.choice(d, "below_baseline", where, tuple(b.value for b in BelowBaseline))
        below_baseline = BelowBaseline(raw) if raw is not None else None

    # Units
    units: _Units | None = None
    if marker is not None and unit is not None:
        units = _Units(kind, marker, unit, scale)
        try:
            units.check_unit()
        except UnitError as err:
            r.add("units", f"{where}: {err}")
            units = None
    unusual = r.string(d, "unusual_value_ok", where) if "unusual_value_ok" in d else None

    def plausible(value: float, label: str) -> None:
        if marker is None or unusual:
            return
        lo, hi = plausible_range(marker)
        if kind in _LEVEL_KINDS | {StatisticKind.CURVE}:
            if not lo <= value <= hi:
                r.add(
                    "units",
                    f"{where}: {label} is {value:g} in the model's unit for {marker}, outside "
                    f"[{lo:g}, {hi:g}]; check the unit, or state why in unusual_value_ok",
                )
        elif kind in _DIFF_KINDS and abs(value) > (hi - lo):
            r.add(
                "units",
                f"{where}: {label} is {value:g} in the model's unit for {marker}, wider than its whole "
                f"range [{lo:g}, {hi:g}]; check the unit, or state why in unusual_value_ok",
            )

    acknowledged: tuple[str, ...] = ()
    if "acknowledged_conflicts" in d:
        ack = d["acknowledged_conflicts"]
        if isinstance(ack, list) and all(isinstance(x, str) and x for x in ack):
            acknowledged = tuple(ack)
        else:
            r.add("parse", f"{where} acknowledged_conflicts is not a list of item names")

    # Windows
    window = None
    per_arm = None
    baseline = None
    partner = None
    if kind in SAMPLED_KINDS:
        if sample_times is not None:
            window = StatisticWindow(sample_times[0], sample_times[-1] + 1)
        if "baseline_window" in d and ok_protocols and duration:
            baseline = _window(r, d["baseline_window"], f"{where} baseline_window", duration)
    elif kind is not StatisticKind.CURVE and ok_protocols and duration:
        window = _window(r, d.get("window"), f"{where} window", duration) if "window" in d else None
        if "per_arm_windows" in d:
            raw = d["per_arm_windows"]
            if not isinstance(raw, list) or len(raw) != 2:
                r.add("protocol", f"{where} per_arm_windows is not a list of two windows")
            else:
                got = [
                    _window(r, w, f"{where} per_arm_windows[{i}]", arms[i].duration_min)
                    for i, w in enumerate(raw)
                ]
                if None not in got:
                    per_arm = tuple(w for w in got if w is not None)
        if "baseline_window" in d:
            baseline = _window(r, d["baseline_window"], f"{where} baseline_window", duration)
    if "partner" in d:
        partner = r.string(d, "partner", where)
        if partner is not None and partner not in SERIES_INDEX and partner not in FACTORS:
            r.add("parse", f"{where} partner {partner!r} is neither a marker nor a person factor")
            partner = None

    # Numbers and uncertainty
    target = 0.0
    sigma = 0.0
    n_arm: int | None = None
    curve: tuple[CurvePoint, ...] | None = None
    complete = units is not None and ok_protocols and fid is not None
    if kind is StatisticKind.CURVE:
        points = d.get("points")
        if not isinstance(points, list) or len(points) < 2:
            r.add("parse", f"{where} points is not a list of at least two points")
            complete = False
        elif units is not None and ok_protocols:
            cps: list[CurvePoint] = []
            seen: set[int] = set()
            for i, pt in enumerate(points):
                w = f"{where} point {i + 1}"
                pd = r.obj(pt, w, {"t_min", "value", "uncertainty"})
                if pd is None:
                    complete = False
                    continue
                t = r.integer(pd, "t_min", w, minimum=0)
                v = r.number(pd, "value", w, "units")
                u = _uncertainty(r, pd.get("uncertainty"), w, kind, units, v) if "uncertainty" in pd else None
                if t is not None and t in seen:
                    r.add("parse", f"{w} repeats t_min {t}")
                    t = None
                if t is not None:
                    seen.add(t)
                    if t >= arms[0].duration_min:
                        r.add("protocol", (
                            f"{w} t_min {t} is past the arm's last minute ({arms[0].duration_min - 1}): "
                            f"an arm sampled at minute {t} needs duration_min {t + 1}"
                        ))
                        t = None
                if t is None or v is None or u is None:
                    complete = False
                    continue
                try:
                    native = units.level(v)
                    plausible(native, f"{w} value")
                    cps.append(CurvePoint(t, native, curve_point_sigma(u)))
                except (UnitError, UncertaintyError) as err:
                    r.add("units" if isinstance(err, UnitError) else "uncertainty", f"{w}: {err}")
                    complete = False
            curve = tuple(sorted(cps, key=lambda c: c.t_min))
            if not complete:
                curve = None
            else:
                window = StatisticWindow(curve[0].t_min, curve[-1].t_min + 1)
                target, sigma, n_arm = curve[0].target, curve[0].sigma, 1
    else:
        value = r.number(d, "value", where, "units") if "value" in d else None
        u = (
            _uncertainty(r, d.get("uncertainty"), where, kind, units, value)
            if units is not None and "uncertainty" in d else None
        )
        if value is None or u is None or units is None:
            complete = False
        else:
            try:
                target = units.level(value)
                if kind is StatisticKind.TIME_TO_PEAK and window is not None:
                    target -= window.start_min  # the paper's clock -> minutes from window start
                    if not 0.0 <= target <= window.end_min - window.start_min:
                        r.add(
                            "units",
                            f"{where}: the peak time {value:g} {unit} is outside its window "
                            f"[{window.start_min}, {window.end_min}) min",
                        )
                        complete = False
                if kind is StatisticKind.INCREMENTAL_AREA and below_baseline is BelowBaseline.IGNORED \
                        and target < 0:
                    r.add(
                        "parse",
                        f"{where}: an incremental area with below_baseline 'ignored' counts only "
                        f"what is above baseline, so it cannot be negative",
                    )
                    complete = False
                if kind in _RATIO_KINDS and not target > 0:
                    r.add("parse", f"{where}: a ratio must be positive")
                    complete = False
                if kind is StatisticKind.CORRELATION and not -1.0 < target < 1.0:
                    r.add("parse", f"{where}: a correlation lies strictly between -1 and 1")
                    complete = False
                if kind is StatisticKind.SPREAD and not target > 0:
                    r.add("parse", f"{where}: a spread must be positive")
                    complete = False
                plausible(target, "value")
                if complete:
                    sigma, n_arm = to_sigma(kind, target, u)
            except (UnitError, UncertaintyError) as err:
                r.add("units" if isinstance(err, UnitError) else "uncertainty", f"{where}: {err}")
                complete = False
        if window is None and "window" in d and ok_protocols:
            complete = False
        if kind in (StatisticKind.RATIO_INCREMENTS, StatisticKind.CHANGE_FROM_BASELINE) \
                and baseline is None:
            complete = False
        if kind in SAMPLED_KINDS and sample_times is None:
            complete = False
        if kind is StatisticKind.INCREMENTAL_AREA and below_baseline is None:
            complete = False
        if "baseline_window" in d and baseline is None:
            complete = False

    if (
        not complete or fid is None or description is None or marker is None or location is None
        or citation is None or window is None or not ok_protocols
    ):
        return None, used
    name = f"{paper}.{fid}"
    try:
        spec = CohortStatisticSpec(
            name=name,
            source=f"{citation.short()}; {location.where}",
            description=description,
            arms=tuple(arms),
            marker_id=marker,
            kind=kind,
            window=window,
            target=target,
            sigma=sigma,
            n_arm=n_arm,
            per_arm_windows=per_arm,
            init_mode=init,
            curve=curve,
            partner=partner,
            baseline=baseline,
            sample_times=sample_times,
            below_baseline=below_baseline,
            paper=paper,
        )
    except ValueError as err:
        r.add("parse", str(err))
        return None, used
    return FindingItem(fid, spec, location, acknowledged), used


def _missing(r: _Reader, data: Any, where: str, kinds: tuple[str, ...]) -> tuple[str, str] | None:
    """``{"kind", "name"}``: the piece that is missing, countable by both."""
    md = r.obj(data, where, {"kind", "name"})
    if md is None:
        return None
    kind = r.choice(md, "kind", where, kinds)
    name = r.string(md, "name", where)
    if name is not None and not _NAME.fullmatch(name):
        r.add("parse", f"{where} name {name!r} is not lowercase letters, digits and underscores")
        name = None
    if kind == "marker" and name in SERIES_INDEX:
        r.add("parse", f"{where}: marker {name!r} is a series the model already carries")
        name = None
    return (kind, name) if kind and name else None


def _left_out(r: _Reader, raw: Any) -> tuple[LeftOut, ...]:
    if not isinstance(raw, list):
        r.add("parse", "left_out is not a list ([] if the file carries everything the paper reports)")
        return ()
    out: list[LeftOut] = []
    for i, item in enumerate(raw):
        w = f"left_out {i + 1}"
        d = r.obj(item, w, {"what", "missing"})
        if d is None:
            continue
        what = r.string(d, "what", w)
        missing = _missing(r, d.get("missing"), f"{w} missing", LEFT_OUT_KINDS) if "missing" in d else None
        if what and missing:
            out.append(LeftOut(what, *missing))
    return tuple(out)


def parse_finding(data: Any, *, path: Path | None = None) -> Finding:
    """One paper's file as a ``Finding``. Raises ``CorpusError`` with every problem."""
    r = _Reader()
    d = r.obj(
        data, "file", {"paper", "citation", "population", "protocols", "findings", "left_out"},
    )
    paper = ""
    if d is not None:
        raw = d.get("paper")
        if isinstance(raw, str) and _SLUG.fullmatch(raw):
            paper = raw
        elif "paper" in d:
            r.add("parse", f"paper {raw!r} is not a lowercase slug like 'schulla-2003'")
    if d is None:
        raise CorpusError(r.problems, paper)
    if path is not None and paper and path.stem != paper:
        r.add("parse", f"the file is named {path.name} but its paper is {paper!r}")
    citation = _citation(r, d.get("citation"), paper) if "citation" in d else None
    who = _population(r, d.get("population")) if "population" in d else None

    protocols: dict[str, CohortArmSpec] = {}
    inits: dict[str, InitMode] = {}
    assumed: dict[str, tuple[str, ...]] = {}
    raw_protocols = d.get("protocols")
    if "protocols" in d and (not isinstance(raw_protocols, list) or not raw_protocols):
        r.add("protocol", "protocols is not a non-empty list")
    elif isinstance(raw_protocols, list):
        # A bad population block must not hide the protocols' own problems.
        stand_in = who or _Who(1, None, None, None, False)
        for i, p in enumerate(raw_protocols):
            got = _protocol(r, p, i, stand_in)
            if got is None:
                continue
            nm, arm, init, unstated = got
            if nm in protocols:
                r.add("protocol", f"protocol {nm} is declared twice")
            else:
                protocols[nm], inits[nm], assumed[nm] = arm, init, unstated

    items: list[FindingItem] = []
    used: set[str] = set()
    raw_findings = d.get("findings")
    if "findings" in d and (not isinstance(raw_findings, list) or not raw_findings):
        r.add("parse", "findings is not a non-empty list")
    elif isinstance(raw_findings, list):
        ids: set[str] = set()
        for i, f in enumerate(raw_findings):
            item, u = _finding_item(r, f, i, paper, citation, protocols, inits)
            used |= u
            if item is not None:
                if item.id in ids:
                    r.add("parse", f"finding id {item.id} is used twice")
                ids.add(item.id)
                items.append(item)
        if isinstance(raw_protocols, list) and len(items) == len(raw_findings):
            for nm in sorted(set(protocols) - used):
                r.add("protocol", f"protocol {nm} is declared but no finding uses it")
    left_out = _left_out(r, d.get("left_out")) if "left_out" in d else ()
    if r.problems:
        raise CorpusError(r.problems, paper)
    assert citation is not None and who is not None
    return Finding(paper, citation, who.n, protocols, tuple(items), path, assumed, left_out)


def parse_refusal(data: Any, *, path: Path | None = None) -> Refusal:
    """A refusal file: a paper the model cannot express, and the missing piece."""
    r = _Reader()
    d = r.obj(data, "file", {"paper", "citation", "reported", "missing", "detail"})
    paper = ""
    if d is not None:
        raw = d.get("paper")
        if isinstance(raw, str) and _SLUG.fullmatch(raw):
            paper = raw
        elif "paper" in d:
            r.add("parse", f"paper {raw!r} is not a lowercase slug like 'schulla-2003'")
    if d is None:
        raise CorpusError(r.problems, paper)
    if path is not None and paper and path.stem != paper:
        r.add("parse", f"the file is named {path.name} but its paper is {paper!r}")
    citation = _citation(r, d.get("citation"), paper) if "citation" in d else None
    reported: list[Reported] = []
    raw = d.get("reported")
    if "reported" in d and (not isinstance(raw, list) or not raw):
        r.add("parse", "reported is not a non-empty list")
    elif isinstance(raw, list):
        for i, item in enumerate(raw):
            w = f"reported {i + 1}"
            rd = r.obj(item, w, {"what", "location"})
            if rd is None:
                continue
            what = r.string(rd, "what", w)
            loc = _location(r, rd.get("location"), w) if "location" in rd else None
            if what and loc:
                reported.append(Reported(what, loc))
    kind = name = None
    missing = _missing(r, d.get("missing"), "missing", REFUSAL_KINDS) if "missing" in d else None
    if missing is not None:
        kind, name = missing
    detail = r.string(d, "detail", "refusal")
    if r.problems:
        raise CorpusError(r.problems, paper)
    assert citation and kind and name and detail
    return Refusal(paper, citation, tuple(reported), kind, name, detail, path)


def parse_queue_item(data: Any, *, path: Path | None = None) -> QueueItem:
    """A queue file: a paper to be read, with where its numbers are."""
    r = _Reader()
    keys = {
        "paper", "citation", "doi", "pmid", "pmcid", "licence", "licence_url", "protocol",
        "protocol_kinds", "population", "population_class", "reports", "uncertainty",
        "figure_only", "note",
    }
    d = r.obj(data, "file", keys)
    paper = ""
    if d is not None:
        raw = d.get("paper")
        if isinstance(raw, str) and _SLUG.fullmatch(raw):
            paper = raw
        elif "paper" in d:
            r.add("parse", f"paper {raw!r} is not a lowercase slug like 'schulla-2003'")
    if d is None:
        raise CorpusError(r.problems, paper)
    if path is not None and paper and path.stem != paper:
        r.add("parse", f"the file is named {path.name} but its paper is {paper!r}")
    citation = r.string(d, "citation", "queue item")
    year = re.search(r"-(\d{4})$", paper)
    if citation and year and year.group(1) not in citation:
        r.add("parse", f"paper {paper!r} ends in a year that the citation does not contain")
    ids: dict[str, str | None] = {}
    for key, pattern in (("doi", _DOI), ("pmid", _PMID), ("pmcid", _PMCID)):
        if key in d and d[key] is None:
            ids[key] = None
            continue
        value = r.string(d, key, "queue item")
        if value is not None and not pattern.fullmatch(value):
            r.add("parse", f"queue item {key} {value!r} is not a {key.upper()}")
            value = None
        ids[key] = value
    if "doi" in d and "pmid" in d and "pmcid" in d and not any(d.get(k) for k in ("doi", "pmid", "pmcid")):
        r.add("parse", "a queue item needs at least one identifier: doi, pmid or pmcid")
    licence = r.string(d, "licence", "queue item")
    licence_url = None
    if "licence_url" in d and d["licence_url"] is not None:
        licence_url = r.string(d, "licence_url", "queue item")
        if licence_url is not None and not licence_url.startswith(("http://", "https://")):
            r.add("parse", f"queue item licence_url {licence_url!r} is not a URL")
    protocol = r.string(d, "protocol", "queue item")
    population = r.string(d, "population", "queue item")
    pclass = r.choice(d, "population_class", "queue item", QUEUE_POPULATIONS)
    unc = r.choice(d, "uncertainty", "queue item", QUEUE_UNCERTAINTY)
    figure_only = d.get("figure_only")
    if "figure_only" in d and not isinstance(figure_only, bool):
        r.add("parse", "queue item figure_only is not true or false")
    note = d.get("note")
    if "note" in d and not isinstance(note, str):
        r.add("parse", "queue item note is not text")
    kinds = d.get("protocol_kinds")
    kind_list: list[str] = []
    if "protocol_kinds" in d:
        if not isinstance(kinds, list) or not kinds or len(set(map(str, kinds))) != len(kinds):
            r.add("parse", "queue item protocol_kinds is not a non-empty list without repeats")
        else:
            for k in kinds:
                if k not in QUEUE_PROTOCOL_KINDS:
                    r.add("parse", f"queue item protocol kind {k!r} is not one of {', '.join(QUEUE_PROTOCOL_KINDS)}")
                else:
                    kind_list.append(k)
    reports: list[QueueReport] = []
    raw = d.get("reports")
    if "reports" in d and (not isinstance(raw, list) or not raw):
        r.add("parse", "queue item reports is not a non-empty list")
    elif isinstance(raw, list):
        for i, item in enumerate(raw):
            w = f"report {i + 1}"
            rd = r.obj(item, w, {"marker", "where", "form"})
            if rd is None:
                continue
            marker = r.string(rd, "marker", w)
            where = r.string(rd, "where", w)
            form = r.choice(rd, "form", w, QUEUE_FORMS)
            if marker and where and form:
                reports.append(QueueReport(marker, where, form))
    if r.problems:
        raise CorpusError(r.problems, paper)
    assert citation and licence and protocol and population and pclass and unc
    return QueueItem(
        paper, citation, ids["doi"], ids["pmid"], ids["pmcid"], licence, licence_url, protocol,
        tuple(kind_list), population, pclass, tuple(reports), unc, bool(figure_only), str(note), path,
    )


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        raise CorpusError([Problem("parse", f"invalid JSON: {err}")]) from err


def read_finding(path: Path) -> Finding:
    return parse_finding(_load_json(path), path=path)


def read_refusal(path: Path) -> Refusal:
    return parse_refusal(_load_json(path), path=path)


def read_queue_item(path: Path) -> QueueItem:
    return parse_queue_item(_load_json(path), path=path)


# --------------------------------------------------------------------------------
# The corpus directory
# --------------------------------------------------------------------------------

def _files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.name != ".gitkeep")


def load_corpus(root: Path | None = None) -> Corpus:
    """Every file under ``root/findings``, ``root/refusals`` and ``root/queue``, or
    ``CorpusError`` with every problem of every file, prefixed by the file."""
    root = root or corpus_root()
    problems: list[Problem] = []
    findings: list[Finding] = []
    refusals: list[Refusal] = []
    queue: list[QueueItem] = []
    for directory, reader, sink in (
        (root / "findings", read_finding, findings),
        (root / "refusals", read_refusal, refusals),
        (root / "queue", read_queue_item, queue),
    ):
        for path in _files(directory):
            if path.suffix != ".json":
                problems.append(Problem("parse", f"{path.parent.name}/{path.name}: not a .json file"))
                continue
            try:
                sink.append(reader(path))  # type: ignore[arg-type]
            except CorpusError as err:
                problems.extend(
                    Problem(p.check, f"{path.parent.name}/{path.name}: {p.message}") for p in err.problems
                )
    problems.extend(_cross_problems(findings, refusals, queue))
    if problems:
        raise CorpusError(problems)
    specs = tuple(s for f in findings for s in f.specs)
    return Corpus(tuple(findings), tuple(refusals), specs, tuple(queue))


def count_piles(root: Path | None = None) -> dict[str, int]:
    """``{"queue": n, "findings": n, "refusals": n}``: papers in each pile."""
    return load_corpus(root).counts


def _cross_problems(
    findings: list[Finding], refusals: list[Refusal], queue: list[QueueItem] = (),  # type: ignore[assignment]
) -> list[Problem]:
    out: list[Problem] = []
    piles = {"finding": {f.paper for f in findings}, "refusal": {x.paper for x in refusals},
             "queue item": {q.paper for q in queue}}
    for (na, a), (nb, b) in (
        (("finding", piles["finding"]), ("refusal", piles["refusal"])),
        (("finding", piles["finding"]), ("queue item", piles["queue item"])),
        (("refusal", piles["refusal"]), ("queue item", piles["queue item"])),
    ):
        for paper in sorted(a & b):
            out.append(Problem("parse", f"{paper} is both a {na} and a {nb}; a paper is in one pile"
                               + (" (delete its queue file when you file it)" if "queue item" in (na, nb) else "")))
    seen: dict[str, str] = {}
    for item in [*findings, *refusals, *queue]:
        ids = item.citation.identifiers() if hasattr(item.citation, "identifiers") else item.identifiers()
        for ident in sorted(ids):
            other = seen.setdefault(ident, item.paper)
            if other != item.paper:
                out.append(Problem("parse", f"{item.paper} and {other} are one paper ({ident})"))
    names: Counter[str] = Counter(s.name for f in findings for s in f.specs)
    for nm, count in names.items():
        if count > 1:
            out.append(Problem("parse", f"item name {nm} is used {count} times"))
    return out


def load_file_specs(root: Path | None = None) -> list[CohortStatisticSpec]:
    """The unsealed specs of every finding file: what ``cohort_statistics`` joins."""
    return list(load_corpus(root).specs)


def count_missing(corpus: "Corpus") -> dict[str, dict[str, tuple[int, int]]]:
    """``{kind: {name: (papers refused for it, papers that left something out for it)}}``.

    Everything the papers report that the corpus could not take, by the piece that
    was missing: the refusal pile and every finding file's ``left_out``. A paper
    counts once per name.
    """
    refused: dict[tuple[str, str], set[str]] = {}
    left: dict[tuple[str, str], set[str]] = {}
    for x in corpus.refusals:
        refused.setdefault((x.kind, x.name), set()).add(x.paper)
    for f in corpus.findings:
        for item in f.left_out:
            left.setdefault((item.kind, item.name), set()).add(f.paper)
    out: dict[str, dict[str, tuple[int, int]]] = {}
    for kind in LEFT_OUT_KINDS:
        names = {n for k, n in (*refused, *left) if k == kind}
        rows = {n: (len(refused.get((kind, n), ())), len(left.get((kind, n), ()))) for n in names}
        if rows:
            out[kind] = dict(sorted(rows.items(), key=lambda kv: (-sum(kv[1]), kv[0])))
    return out


def count_refusals(refusals: list[Refusal] | tuple[Refusal, ...]) -> dict[str, dict[str, int]]:
    """``{kind: {name: papers}}``: the refusal pile, countable by kind and by missing piece."""
    out: dict[str, Counter[str]] = {k: Counter() for k in REFUSAL_KINDS}
    for x in refusals:
        out[x.kind][x.name] += 1
    return {k: dict(sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))) for k, c in out.items() if c}
