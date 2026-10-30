"""A prior over named person factors, drawn fresh into the 32-wide person code.

The code's first coordinates are the factors, in ``FACTORS`` order. Every
other coordinate is 0. The zero code is the median person: age 40, BMI 25,
sex-averaged, and every latent at its median. A published arm names the
population it was measured in and the draw is restricted to those people.
"""

from __future__ import annotations

import numpy as np
import torch

from ..knowledge.cohort_types import FACTORS, Population, factor_index
from ..types import EMBEDDING_DIM

# Code units. Zero is the reference median.
AGE_CENTER_YEARS = 40.0
AGE_SCALE_YEARS = 15.0
AGE_MIN_YEARS = 18.0
AGE_MAX_YEARS = 90.0
BMI_CENTER = 25.0
BMI_SCALE = 5.0
BMI_MIN = 16.0
BMI_MAX = 45.0

# A standard fed day the student spins through before a cohort arm.
FED_DAY_MINUTES = 24 * 60
# (clock hour, carbohydrate g, fat g, protein g). An ordinary three-meal day.
FED_DAY_MEALS: tuple[tuple[float, float, float, float], ...] = (
    (8.0, 70.0, 20.0, 25.0),
    (13.0, 90.0, 30.0, 35.0),
    (19.0, 80.0, 25.0, 30.0),
)
# Asleep from 23:00 to 07:00 on that day. The pre-fast after it is awake.
SLEEP_START_HOUR = 23.0
SLEEP_END_HOUR = 7.0


def teacher_tape_weight(epoch: int, total_epochs: int, base: float) -> float:
    """Linear decay of the teacher-tape term, from ``base`` to zero.

    Epoch 0 carries ``base``. The last epoch of the run carries 0, which is
    also the whole of a one-epoch run. A non-positive base stays off.
    """
    if base <= 0.0 or total_epochs <= 0:
        return 0.0
    if total_epochs == 1:
        return 0.0
    done = min(max(int(epoch), 0), total_epochs - 1)
    return float(base) * (1.0 - done / (total_epochs - 1))


def reference_prior_stats() -> tuple[list[float], list[float]]:
    """Diagonal Gaussian the calibration prior should use: N(0, 1) on each
    factor coordinate, and a point mass at 0 on every unnamed coordinate."""
    mean = [0.0] * EMBEDDING_DIM
    std = [0.0] * EMBEDDING_DIM
    for i in range(len(FACTORS)):
        std[i] = 1.0
    return mean, std


def calibration_prior_nll(
    code: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, weight: float,
) -> torch.Tensor:
    """Gaussian energy of the coordinates a person is actually drawn on.

    A width of 0 is a point mass. Those coordinates are not in the energy —
    dividing by the width made one stray step infinite — and the mean runs
    over the free coordinates, so the unnamed ones do not dilute the factors.
    ``project_point_mass`` is what keeps the point mass at its point.
    """
    free = std > 0
    if not bool(free.any()):
        return code.new_zeros(())
    z = (code[..., free] - mean[..., free]) / std[..., free]
    return weight * z.pow(2).mean()


def project_point_mass(code: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> None:
    """Put every point-mass coordinate back on its point. In-place."""
    fixed = std <= 0
    code[..., fixed] = mean[..., fixed]


def _truncnorm(
    rng: np.random.Generator, mean: float, sigma: float, lo: float, hi: float, n: int,
) -> np.ndarray:
    out = np.empty(n, dtype=np.float64)
    filled = 0
    while filled < n:
        draw = rng.normal(mean, sigma, size=max(n - filled, 1) * 4)
        keep = draw[(draw >= lo) & (draw <= hi)]
        take = min(int(keep.shape[0]), n - filled)
        if take:
            out[filled:filled + take] = keep[:take]
            filled += take
    return out


def _years(rng: np.random.Generator, n: int, population: Population | None) -> np.ndarray:
    if population is not None and population.age is not None:
        lo, hi = population.age
        if lo == hi:
            return np.full(n, float(lo))
        return rng.uniform(lo, hi, size=n)
    return _truncnorm(rng, AGE_CENTER_YEARS, AGE_SCALE_YEARS, AGE_MIN_YEARS, AGE_MAX_YEARS, n)


def _bmi(rng: np.random.Generator, n: int, population: Population | None) -> np.ndarray:
    if population is not None and population.bmi is not None:
        lo, hi = population.bmi
        if lo == hi:
            return np.full(n, float(lo))
        return rng.uniform(lo, hi, size=n)
    return _truncnorm(rng, BMI_CENTER, BMI_SCALE, BMI_MIN, BMI_MAX, n)


def _sex_code(rng: np.random.Generator, n: int, population: Population | None) -> np.ndarray:
    if population is not None and population.sex == "female":
        return np.full(n, -1.0)
    if population is not None and population.sex == "male":
        return np.full(n, 1.0)
    return rng.choice(np.array([-1.0, 1.0]), size=n)


def draw_factor_codes(
    n: int,
    rng: np.random.Generator,
    device: torch.device | str,
    population: Population | None = None,
) -> torch.Tensor:
    """``n`` fresh person codes, ``[n, EMBEDDING_DIM]``, from ``population``.

    The reference prior (``population is None``) is centred on the zero code.
    A restricted population shifts the axes it names and leaves the rest.
    """
    if n <= 0:
        return torch.zeros(0, EMBEDDING_DIM, device=device)
    codes = np.zeros((n, EMBEDDING_DIM), dtype=np.float64)
    latent = rng.normal(0.0, 1.0, size=(n, len(FACTORS)))
    if population is not None and population.healthy:
        # Sensitive side of the median, and beta-cell function that is at least
        # the median. The other latents stay a standard normal.
        ir = factor_index("insulin_resistance")
        beta = factor_index("beta_cell")
        latent[:, ir] = -np.abs(latent[:, ir])
        latent[:, beta] = np.abs(latent[:, beta])
    for name in (
        "insulin_resistance", "beta_cell", "fitness", "chronotype",
        "hpa_reactivity", "gastric_emptying",
    ):
        codes[:, factor_index(name)] = latent[:, factor_index(name)]
    codes[:, factor_index("age")] = (_years(rng, n, population) - AGE_CENTER_YEARS) / AGE_SCALE_YEARS
    codes[:, factor_index("adiposity")] = (_bmi(rng, n, population) - BMI_CENTER) / BMI_SCALE
    codes[:, factor_index("sex")] = _sex_code(rng, n, population)
    return torch.tensor(codes, dtype=torch.float32, device=device)


def years_of(codes: torch.Tensor) -> torch.Tensor:
    return codes[..., factor_index("age")] * AGE_SCALE_YEARS + AGE_CENTER_YEARS


def bmi_of(codes: torch.Tensor) -> torch.Tensor:
    return codes[..., factor_index("adiposity")] * BMI_SCALE + BMI_CENTER


__all__ = [
    "AGE_CENTER_YEARS",
    "AGE_SCALE_YEARS",
    "BMI_CENTER",
    "BMI_SCALE",
    "FED_DAY_MEALS",
    "FED_DAY_MINUTES",
    "bmi_of",
    "draw_factor_codes",
    "reference_prior_stats",
    "teacher_tape_weight",
    "years_of",
]
