"""How a paper's stated uncertainty becomes the ``sigma`` and ``n_arm`` of a spec.

A finding file never states ``sigma`` or ``n_arm``. It states what the paper
reported (a between-person SD, a standard error, a confidence interval, a range,
or only its n) and this module derives both numbers, so the likelihood training
applies is the one the paper supports. Why it has to be derived:

* ``cohort_loss.score_batch_statistic`` scores the **batch mean** of the sampled
  people against the target with ``sem = sigma / sqrt(n)``, ``n = spec.n_arm``,
  or the number of people drawn if ``n_arm`` is ``None``. So ``sigma`` there is a
  *between-person* SD and ``n_arm`` the paper's n.
* ``evidence_step._sigma`` does the same division, by ``sqrt(draws)`` when
  ``n_arm`` is ``None``. A sigma that is already a standard error, left with
  ``n_arm=None``, is tightened a second time by the number of draws: the commonest
  way to get a likelihood far sharper than the study.
* ``SPREAD``, ``CORRELATION`` and ``RATIO_INCREMENTS`` are population summaries.
  Their ``sigma`` is used as is: it must be the standard error of the published
  number.
* A ``CURVE`` point is compiled to a one-person-equivalent ``MEAN_IN_WINDOW`` with
  ``n_arm=1`` and ``sigma`` inflated by ``sqrt(n_points)``; its ``sigma`` must be
  the point's standard error.

The rules (``to_sigma``), with ``N_CAP = 25``:

=============================  ==========================================  ==========
statistic                      sigma                                       n_arm
=============================  ==========================================  ==========
mean / peak / time-to-peak /   the between-person SD: ``sd`` as reported;   min(n, 25)
delta / ratio of means         ``sem * sqrt(n)``; ``(hi - lo) / (2 t) *
                               sqrt(n)`` for a CI; ``(hi - lo) / xi(n)``
                               for a range (Wan et al. 2014)
ratio of increments            the reported standard error (``sem`` or     1
                               CI half-width / quantile)
spread                         ``sd / sqrt(2 (n_eff - 1))`` from ``n``     1
                               alone; or the reported standard error /
                               CI of the SD
correlation                    ``(1 - r^2) / sqrt(n_eff - 2)`` from ``n``  1
                               alone; or the reported SE / CI; never
                               below ``CORRELATION_SIGMA_FLOOR``
curve point                    the point's standard error                  1
=============================  ==========================================  ==========

``n_eff = min(n, N_CAP)``, and a reported standard error from a study with
``n > N_CAP`` is widened by ``sqrt(n / N_CAP)``. PLAN section 3: a sigma tighter
than the study supports can wrongly exclude the truth. One cohort's mean is
an estimate of the population mean no better than the difference between
cohorts allows, and a large n makes a paper's own standard error much smaller
than that: 200 people measured once do not pin a population mean to 7% of its
spread. 25 is a judgement, not a measurement; it holds the sharpest finding to
one fifth of its between-person SD. It lives here, in one place.

A one-sided or banded claim (a ceiling, a diagnostic threshold) is not a
reported number with an uncertainty and has no place in this format; it is a
``PhysiologyRule``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

from .cohort_types import StatisticKind

__all__ = [
    "ALLOWED_KINDS",
    "CORRELATION_SIGMA_FLOOR",
    "N_CAP",
    "UNCERTAINTY_KINDS",
    "Uncertainty",
    "UncertaintyError",
    "curve_point_sigma",
    "to_sigma",
]

N_CAP = 25
CORRELATION_SIGMA_FLOOR = 0.05

UNCERTAINTY_KINDS = ("sd", "sem", "ci", "range", "n_only")

_SCALAR = frozenset({"sd", "sem", "ci", "range"})
ALLOWED_KINDS: dict[StatisticKind, frozenset[str]] = {
    StatisticKind.MEAN_IN_WINDOW: _SCALAR,
    StatisticKind.PEAK_VALUE: _SCALAR,
    StatisticKind.TIME_TO_PEAK: _SCALAR,
    StatisticKind.DELTA_MEANS: _SCALAR,
    StatisticKind.DELTA_PEAKS: _SCALAR,
    StatisticKind.CURVE: _SCALAR,
    StatisticKind.RATIO_MEANS: frozenset({"sem", "ci"}),
    StatisticKind.RATIO_INCREMENTS: frozenset({"sem", "ci"}),
    StatisticKind.SPREAD: frozenset({"n_only", "sem", "ci"}),
    StatisticKind.CORRELATION: frozenset({"n_only", "sem", "ci"}),
}

# Two-sided t quantiles for df = 1..30, by confidence level.
_T = {
    95: (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179,
         2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064,
         2.060, 2.056, 2.052, 2.048, 2.045, 2.042),
    90: (6.314, 2.920, 2.353, 2.132, 2.015, 1.943, 1.895, 1.860, 1.833, 1.812, 1.796, 1.782,
         1.771, 1.761, 1.753, 1.746, 1.740, 1.734, 1.729, 1.725, 1.721, 1.717, 1.714, 1.711,
         1.708, 1.706, 1.703, 1.701, 1.699, 1.697),
    99: (63.657, 9.925, 5.841, 4.604, 4.032, 3.707, 3.499, 3.355, 3.250, 3.169, 3.106, 3.055,
         3.012, 2.977, 2.947, 2.921, 2.898, 2.878, 2.861, 2.845, 2.831, 2.819, 2.807, 2.797,
         2.787, 2.779, 2.771, 2.763, 2.756, 2.750),
}
CI_LEVELS = tuple(_T)


class UncertaintyError(ValueError):
    """An uncertainty that cannot be turned into a likelihood."""


@dataclass(frozen=True)
class Uncertainty:
    """What a paper reported about one number, already in the marker's unit.

    ``sd`` / ``sem``: ``value``. ``ci``: ``low`` / ``high`` / ``level`` (percent).
    ``range``: ``low`` / ``high`` (the smallest and largest individual). ``n_only``:
    nothing but ``n``. ``n`` is the number of people behind the number.
    """

    kind: str
    n: int
    value: float | None = None
    low: float | None = None
    high: float | None = None
    level: int | None = None

    def problems(self) -> list[str]:
        """Everything wrong with this uncertainty on its own, as sentences."""
        out: list[str] = []
        if self.kind not in UNCERTAINTY_KINDS:
            return [f"uncertainty kind {self.kind!r} is not one of {', '.join(UNCERTAINTY_KINDS)}"]
        if self.n < 1:
            out.append("uncertainty n must be at least 1")
        if self.kind in ("sd", "sem"):
            if self.value is None or not self.value > 0:
                out.append(f"uncertainty {self.kind} needs a positive value")
            if self.kind == "sd" and self.n < 2:
                out.append("a between-person sd needs n of at least 2")
        elif self.kind in ("ci", "range"):
            if self.low is None or self.high is None or not self.high > self.low:
                out.append(f"uncertainty {self.kind} needs low < high")
            if self.n < 2:
                out.append(f"a {self.kind} needs n of at least 2")
            if self.kind == "ci" and self.level not in CI_LEVELS:
                out.append(f"a ci level is one of {CI_LEVELS}")
        return out


def _t_quantile(level: int, df: int) -> float:
    """Two-sided ``level``% t quantile with ``df`` degrees of freedom."""
    if df < 1:
        raise UncertaintyError("a confidence interval needs n of at least 2")
    table = _T[level]
    if df <= len(table):
        return table[df - 1]
    z = NormalDist().inv_cdf(0.5 + level / 200.0)
    # Cornish-Fisher expansion of the t quantile in 1/df (error < 1e-3 beyond df 30).
    return (
        z + (z**3 + z) / (4 * df) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)
    )


def _range_xi(n: int) -> float:
    """Wan et al. (2014): the range of ``n`` normal draws is about ``xi(n)`` SDs."""
    return 2.0 * NormalDist().inv_cdf((n - 0.375) / (n + 0.25))


def standard_error(u: Uncertainty) -> float:
    """The standard error of the reported statistic, as the paper has it."""
    if u.kind == "sem":
        assert u.value is not None
        return u.value
    if u.kind == "sd":
        assert u.value is not None
        return u.value / math.sqrt(u.n)
    assert u.low is not None and u.high is not None
    if u.kind == "ci":
        assert u.level is not None
        return (u.high - u.low) / (2.0 * _t_quantile(u.level, u.n - 1))
    if u.kind == "range":
        return (u.high - u.low) / _range_xi(u.n) / math.sqrt(u.n)
    raise UncertaintyError(f"a {u.kind} carries no standard error")


def between_person_sd(u: Uncertainty) -> float:
    """The between-person SD implied by ``u``: what the n-th-root rule divides."""
    if u.kind == "sd":
        assert u.value is not None
        return u.value
    if u.kind == "range":
        assert u.low is not None and u.high is not None
        return (u.high - u.low) / _range_xi(u.n)
    return standard_error(u) * math.sqrt(u.n)


def _widened_se(u: Uncertainty) -> float:
    """The standard error, widened by ``sqrt(n / N_CAP)`` for an n past the cap."""
    return standard_error(u) * math.sqrt(max(1.0, u.n / N_CAP))


def curve_point_sigma(u: Uncertainty) -> float:
    """The ``CurvePoint.sigma`` of a point: its (widened) standard error."""
    _require(StatisticKind.CURVE, u)
    return _widened_se(u)


def _require(kind: StatisticKind, u: Uncertainty) -> None:
    problems = u.problems()
    if problems:
        raise UncertaintyError("; ".join(problems))
    if u.kind not in ALLOWED_KINDS[kind]:
        raise UncertaintyError(
            f"{kind.value} takes an uncertainty of kind {', '.join(sorted(ALLOWED_KINDS[kind]))}, "
            f"not {u.kind}",
        )


def to_sigma(kind: StatisticKind, target: float, u: Uncertainty) -> tuple[float, int]:
    """``(sigma, n_arm)`` for a spec of ``kind`` with ``target`` (in the marker's unit)."""
    _require(kind, u)
    n_eff = min(u.n, N_CAP)
    if kind is StatisticKind.CURVE:
        raise UncertaintyError("a curve's sigma is per point: use curve_point_sigma")
    if kind in (
        StatisticKind.MEAN_IN_WINDOW, StatisticKind.PEAK_VALUE, StatisticKind.TIME_TO_PEAK,
        StatisticKind.DELTA_MEANS, StatisticKind.DELTA_PEAKS, StatisticKind.RATIO_MEANS,
    ):
        sigma = between_person_sd(u)
        return _positive(sigma), n_eff
    if kind is StatisticKind.RATIO_INCREMENTS:
        return _positive(_widened_se(u)), 1
    if kind is StatisticKind.SPREAD:
        if u.kind == "n_only":
            if u.n < 2:
                raise UncertaintyError("a spread from n alone needs n of at least 2")
            return _positive(abs(target) / math.sqrt(2.0 * (n_eff - 1))), 1
        return _positive(_widened_se(u)), 1
    if kind is StatisticKind.CORRELATION:
        if not -1.0 < target < 1.0:
            raise UncertaintyError(f"a correlation of {target} is not in (-1, 1)")
        if u.kind == "n_only":
            if u.n < 3:
                raise UncertaintyError("a correlation from n alone needs n of at least 3")
            se = (1.0 - target**2) / math.sqrt(n_eff - 2)
        else:
            se = _widened_se(u)
        return max(CORRELATION_SIGMA_FLOOR, _positive(se)), 1
    raise UncertaintyError(f"no uncertainty rule for {kind.value}")


def _positive(x: float) -> float:
    if not x > 0.0 or math.isinf(x):
        raise UncertaintyError(f"the uncertainty gives a sigma of {x}")
    return x
