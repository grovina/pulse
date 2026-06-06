"""CohortArm series-length validation and end-to-end statistic loss smoke."""

from __future__ import annotations

import unittest

import torch

from pulse.cohort_loss import _optional_series_tensor, cohort_statistic_loss_one_spec
from pulse.knowledge.cohort_types import (
    CohortArmSpec,
    CohortStatisticSpec,
    StatisticKind,
    StatisticWindow,
)
from pulse.model import ModularPhysiologyNetwork
from pulse.types import EMBEDDING_DIM, NORM_CENTER


class TestOptionalSeriesTensor(unittest.TestCase):
    def test_none_returns_none(self) -> None:
        device = torch.device("cpu")
        self.assertIsNone(_optional_series_tensor("x", None, 10, device))

    def test_length_mismatch_raises(self) -> None:
        device = torch.device("cpu")
        with self.assertRaises(ValueError):
            _optional_series_tensor("x", (1.0, 2.0), 10, device)

    def test_ok_shape(self) -> None:
        device = torch.device("cpu")
        t = _optional_series_tensor("x", tuple([1.0] * 5), 5, device)
        self.assertEqual(t.shape, (5,))


class TestCohortStatisticSleepIntegration(unittest.TestCase):
    def test_sleep_delta_means_runs(self) -> None:
        """Smoke: sleep-delta spec rolls forward and produces a scalar finite loss."""
        spec = CohortStatisticSpec(
            name="test_sleep",
            source="test",
            description="test",
            arms=(
                CohortArmSpec(
                    label="control",
                    duration_min=120, start_hour=7.0, meals=(),
                    sleep_wake=tuple([1.0] * 120),
                ),
                CohortArmSpec(
                    label="restricted",
                    duration_min=120, start_hour=7.0, meals=(),
                    sleep_wake=tuple([0.0 if 60 <= i < 90 else 1.0 for i in range(120)]),
                ),
            ),
            marker_id="glucose",
            kind=StatisticKind.DELTA_MEANS,
            window=StatisticWindow(start_min=90, end_min=110),
            target=0.0,
            sigma=1.0,
        )
        device = torch.device("cpu")
        model = ModularPhysiologyNetwork(embedding_dim=EMBEDDING_DIM)
        emb_table = torch.nn.Embedding(4, EMBEDDING_DIM)
        initial = torch.tensor(NORM_CENTER, dtype=torch.float32, device=device)
        embs = [
            emb_table(torch.tensor(0)),
            emb_table(torch.tensor(1)),
            torch.zeros(EMBEDDING_DIM),
        ]
        loss, _pred, _z = cohort_statistic_loss_one_spec(model, embs, spec, initial)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
