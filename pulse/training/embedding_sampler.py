"""
Shared policy for selecting which embeddings receive cohort-style supervision
in a given epoch.

Cohort-statistic and dose-response signals both supervise the model's behavior
under specific physiological protocols. Both face the same train/eval
distribution-mismatch hazard: the textbook benchmark queries the model at the
zero ("default") embedding for every scenario, but per-epoch sampling of
patient embeddings alone never touches that point in embedding space. So both
signals want the same selection: a small random subset of patient embeddings
plus, optionally, the zero embedding.

Centralizing the policy keeps the two signals in sync and makes it obvious in
one place that "the textbook benchmark uses zero ⇒ the zero embedding must be
in the training distribution for these signals".
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def select_supervised_rows(
    embeddings: nn.Embedding,
    n_patients: int,
    sample_patients: int,
    rng: np.random.Generator,
    device: torch.device | str,
    include_default: bool = True,
) -> list[tuple[int | None, torch.Tensor]]:
    """Embeddings to supervise this epoch, each with the patient id it belongs to.

    Patient rows come from ``embeddings.forward`` so their gradients flow back
    into the table. The zero embedding is ``(None, zeros)`` and is appended
    last: it is the median person, not a draw, and it has no grad of its own.
    ``None`` is the sentinel so patient 0 is not confused with that row.
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


def select_supervised_embeddings(
    embeddings: nn.Embedding,
    n_patients: int,
    sample_patients: int,
    rng: np.random.Generator,
    device: torch.device | str,
    include_default: bool = True,
) -> list[torch.Tensor]:
    """Return embeddings to supervise this epoch.

    Always grad-enabled tensors, suitable for forward-then-backward through
    the model. See ``select_supervised_rows`` for which row is which patient.
    """
    return [
        emb for _pid, emb in select_supervised_rows(
            embeddings, n_patients, sample_patients, rng, device, include_default,
        )
    ]
