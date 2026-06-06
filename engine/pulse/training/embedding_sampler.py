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
    the model. Patient embeddings come from ``embeddings.forward`` so their
    gradients flow back into the embedding table; the zero embedding is a
    fresh tensor with no grad of its own (only model parameters update).
    """
    emb_list: list[torch.Tensor] = []
    if n_patients > 0 and sample_patients > 0:
        k = min(sample_patients, n_patients)
        pids = rng.choice(n_patients, size=k, replace=False)
        for pid in pids:
            pid_t = torch.tensor(int(pid), dtype=torch.long, device=device)
            emb_list.append(embeddings(pid_t))
    if include_default:
        emb_list.append(
            torch.zeros(embeddings.embedding_dim, device=device),
        )
    return emb_list
