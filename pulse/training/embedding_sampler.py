"""Who a training step rolls out.

Cohort evidence draws each person fresh from the factor prior
(``draw_people``). The zero code is the median person, appended when a
caller also wants the point the textbook benchmark queries. It is not a
draw, and it is not a row of a memorized table.

Signals that still pair a code with one teacher patient (the setpoint and
sweep supervision, while those weights are on) index the table they were
given. That table is not the training population.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..knowledge.cohort_types import Population
from ..types import EMBEDDING_DIM
from .factors import draw_factor_codes


def draw_people(
    n: int,
    rng: np.random.Generator,
    device: torch.device | str,
    population: Population | None = None,
) -> torch.Tensor:
    """``[n, EMBEDDING_DIM]`` codes drawn now, not looked up."""
    return draw_factor_codes(n, rng, device, population)


def select_supervised_embeddings(
    n: int,
    rng: np.random.Generator,
    device: torch.device | str,
    population: Population | None = None,
    include_default: bool = True,
) -> list[torch.Tensor]:
    """``n`` fresh draws, then the zero code when ``include_default``.

    ``n == 0`` is the median person alone, or nothing when the median person
    is excluded too. Draws do not carry gradient: they are inputs. The model
    that reads them does.
    """
    rows: list[torch.Tensor] = []
    if n > 0:
        codes = draw_people(n, rng, device, population)
        rows.extend(codes[i] for i in range(n))
    if include_default:
        rows.append(torch.zeros(EMBEDDING_DIM, device=device))
    return rows


def teacher_table_rows(
    embeddings: nn.Embedding,
    n_patients: int,
    sample_patients: int,
    rng: np.random.Generator,
    device: torch.device | str,
    include_default: bool = True,
) -> list[tuple[int | None, torch.Tensor]]:
    """Rows of a teacher-paired table, for a signal that still matches a
    patient id to a teacher target.

    The zero code is ``(None, zeros)`` and is appended last. ``None`` is the
    sentinel so patient 0 is not that row.
    """
    rows: list[tuple[int | None, torch.Tensor]] = []
    if n_patients > 0 and sample_patients > 0:
        k = min(sample_patients, n_patients)
        pids = rng.choice(n_patients, size=k, replace=False)
        for pid in pids:
            pid_i = int(pid)
            pid_t = torch.tensor(pid_i, dtype=torch.long, device=device)
            rows.append((pid_i, embeddings(pid_t)))
    if include_default:
        rows.append((None, torch.zeros(embeddings.embedding_dim, device=device)))
    return rows


__all__ = [
    "draw_people",
    "select_supervised_embeddings",
    "teacher_table_rows",
]
