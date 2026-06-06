"""Unit tests for the shared embedding selection policy used by both the
cohort-statistic and dose-response training signals.

The key invariant being protected: when ``include_default=True``, the zero
("default") embedding is in the returned list — the textbook benchmark calls
the model at zero for every scenario, so any signal that shapes physiology
must include zero in its training distribution.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn as nn

from pulse.training.embedding_sampler import select_supervised_embeddings


class TestSelectSupervisedEmbeddings(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cpu")
        self.embeddings = nn.Embedding(8, 16)
        nn.init.normal_(self.embeddings.weight, std=0.1)

    def test_includes_default_zero_embedding(self) -> None:
        rng = np.random.default_rng(0)
        out = select_supervised_embeddings(
            self.embeddings, n_patients=8, sample_patients=3,
            rng=rng, device=self.device, include_default=True,
        )
        self.assertEqual(len(out), 4)
        self.assertTrue(torch.equal(out[-1], torch.zeros(16)))

    def test_excludes_default_when_disabled(self) -> None:
        rng = np.random.default_rng(0)
        out = select_supervised_embeddings(
            self.embeddings, n_patients=8, sample_patients=3,
            rng=rng, device=self.device, include_default=False,
        )
        self.assertEqual(len(out), 3)
        self.assertFalse(any(torch.equal(t, torch.zeros(16)) for t in out))

    def test_no_patients_returns_only_default(self) -> None:
        rng = np.random.default_rng(0)
        out = select_supervised_embeddings(
            self.embeddings, n_patients=0, sample_patients=0,
            rng=rng, device=self.device, include_default=True,
        )
        self.assertEqual(len(out), 1)
        self.assertTrue(torch.equal(out[0], torch.zeros(16)))

    def test_no_patients_no_default_returns_empty(self) -> None:
        rng = np.random.default_rng(0)
        out = select_supervised_embeddings(
            self.embeddings, n_patients=0, sample_patients=0,
            rng=rng, device=self.device, include_default=False,
        )
        self.assertEqual(out, [])

    def test_clamps_sample_to_population(self) -> None:
        rng = np.random.default_rng(0)
        out = select_supervised_embeddings(
            self.embeddings, n_patients=2, sample_patients=10,
            rng=rng, device=self.device, include_default=False,
        )
        self.assertEqual(len(out), 2)

    def test_patient_embeddings_carry_grad(self) -> None:
        rng = np.random.default_rng(0)
        out = select_supervised_embeddings(
            self.embeddings, n_patients=4, sample_patients=2,
            rng=rng, device=self.device, include_default=True,
        )
        patient_embs = out[:-1]
        for emb in patient_embs:
            self.assertTrue(emb.requires_grad)
        self.assertFalse(out[-1].requires_grad)


if __name__ == "__main__":
    unittest.main()
