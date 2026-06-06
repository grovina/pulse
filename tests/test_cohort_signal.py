"""End-to-end tests for ``CohortStatisticSignal`` focused on the embedding
selection contract: the zero embedding is supervised by default, model
parameters move under positive weight, and a zero weight is a no-op.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn as nn

from pulse.knowledge.cohort_types import (
    CohortArmSpec,
    CohortStatisticSpec,
    StatisticKind,
    StatisticWindow,
)
from pulse.model import ModularPhysiologyNetwork
from pulse.training import CohortStatisticSignal, SignalContext, WeightSchedule
from pulse.types import EMBEDDING_DIM, NORM_CENTER


def _tiny_model() -> ModularPhysiologyNetwork:
    return ModularPhysiologyNetwork(
        embedding_dim=EMBEDDING_DIM,
        metabolic_hidden=16, appetite_hidden=12, stress_hidden=12,
        cardiovascular_hidden=16, thermoreg_hidden=8, respiratory_hidden=8,
    )


def _toy_glucose_spec() -> CohortStatisticSpec:
    return CohortStatisticSpec(
        name="toy_glucose_meal",
        source="test",
        description="test",
        arms=(
            CohortArmSpec(label="fasted", duration_min=120, start_hour=8.0, meals=()),
            CohortArmSpec(
                label="fed", duration_min=120, start_hour=8.0,
                meals=((30.0, 50.0, 10.0, 15.0),),
            ),
        ),
        marker_id="glucose",
        kind=StatisticKind.DELTA_MEANS,
        window=StatisticWindow(start_min=60, end_min=110),
        target=20.0,
        sigma=10.0,
    )


class TestCohortStatisticSignalGating(unittest.TestCase):
    def test_zero_weight_is_noop(self) -> None:
        device = torch.device("cpu")
        model = _tiny_model()
        emb = nn.Embedding(2, EMBEDDING_DIM)
        params = list(model.parameters()) + list(emb.parameters())
        opt = torch.optim.Adam(params, lr=1e-3)
        sig = CohortStatisticSignal(
            specs=[_toy_glucose_spec()],
            n_patients=2, sample_patients=2,
            weight=WeightSchedule(0.0),
        )
        ctx = SignalContext(
            epoch=0, total_epochs=1, rng=np.random.default_rng(0),
            device=device, optimizer=opt, params=params, grad_clip=10.0,
        )
        result = sig.compute(model, emb, ctx)
        self.assertEqual(result.n_units, 0)

    def test_default_embedding_supervised_when_no_patients(self) -> None:
        """With n_patients=0 and include_default_embedding=True, the zero
        embedding still receives gradient — model parameters move."""
        device = torch.device("cpu")
        torch.manual_seed(3)
        model = _tiny_model()
        emb = nn.Embedding(1, EMBEDDING_DIM)
        params = list(model.parameters()) + list(emb.parameters())
        opt = torch.optim.Adam(params, lr=1e-2)
        sig = CohortStatisticSignal(
            specs=[_toy_glucose_spec()],
            n_patients=0, sample_patients=0,
            include_default_embedding=True,
            weight=WeightSchedule(1.0),
        )
        ctx = SignalContext(
            epoch=0, total_epochs=1, rng=np.random.default_rng(0),
            device=device, optimizer=opt, params=params, grad_clip=10.0,
        )
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        result = sig.compute(model, emb, ctx)
        moved = any(
            float((p.detach() - before[n]).abs().sum()) > 0
            for n, p in model.named_parameters()
        )
        self.assertEqual(result.n_units, 1)
        self.assertTrue(moved, "default embedding supervision did not move model params")

    def test_no_supervision_when_default_disabled_and_no_patients(self) -> None:
        device = torch.device("cpu")
        model = _tiny_model()
        emb = nn.Embedding(1, EMBEDDING_DIM)
        params = list(model.parameters()) + list(emb.parameters())
        opt = torch.optim.Adam(params, lr=1e-2)
        sig = CohortStatisticSignal(
            specs=[_toy_glucose_spec()],
            n_patients=0, sample_patients=0,
            include_default_embedding=False,
            weight=WeightSchedule(1.0),
        )
        ctx = SignalContext(
            epoch=0, total_epochs=1, rng=np.random.default_rng(0),
            device=device, optimizer=opt, params=params, grad_clip=10.0,
        )
        result = sig.compute(model, emb, ctx)
        self.assertEqual(result.n_units, 0)

    def test_cold_initial_state_runs_and_differs_from_norm_center(self) -> None:
        """With ``use_cold_initial_state=True``, per-spec init comes from the
        cold model (different from NORM_CENTER), and supervision still runs."""
        device = torch.device("cpu")
        torch.manual_seed(4)
        model = _tiny_model()
        emb = nn.Embedding(2, EMBEDDING_DIM)
        params = list(model.parameters()) + list(emb.parameters())
        opt = torch.optim.Adam(params, lr=1e-3)
        sig = CohortStatisticSignal(
            specs=[_toy_glucose_spec()],
            n_patients=2, sample_patients=2,
            use_cold_initial_state=True,
            weight=WeightSchedule(0.5),
        )
        ctx = SignalContext(
            epoch=0, total_epochs=1, rng=np.random.default_rng(0),
            device=device, optimizer=opt, params=params, grad_clip=10.0,
        )
        result = sig.compute(model, emb, ctx)
        self.assertEqual(result.n_units, 1)
        # Confirm the factory really produces a cold-derived state per spec.
        init_fn = sig._build_initial_state_fn(np.random.default_rng(0), device)
        cold_state = init_fn(_toy_glucose_spec())
        norm_state = torch.tensor(NORM_CENTER, dtype=torch.float32)
        self.assertEqual(cold_state.shape, norm_state.shape)
        self.assertFalse(torch.allclose(cold_state, norm_state))

    def test_no_cold_init_falls_back_to_norm_center(self) -> None:
        device = torch.device("cpu")
        sig = CohortStatisticSignal(
            specs=[_toy_glucose_spec()],
            use_cold_initial_state=False,
        )
        init_fn = sig._build_initial_state_fn(np.random.default_rng(0), device)
        self.assertTrue(
            torch.allclose(
                init_fn(_toy_glucose_spec()),
                torch.tensor(NORM_CENTER, dtype=torch.float32),
            )
        )

    def test_positive_weight_changes_embedding(self) -> None:
        device = torch.device("cpu")
        torch.manual_seed(2)
        model = _tiny_model()
        emb = nn.Embedding(2, EMBEDDING_DIM)
        nn.init.normal_(emb.weight, std=0.1)
        params = list(model.parameters()) + list(emb.parameters())
        opt = torch.optim.Adam(params, lr=1e-2)
        sig = CohortStatisticSignal(
            specs=[_toy_glucose_spec()],
            n_patients=2, sample_patients=2,
            weight=WeightSchedule(0.5),
        )
        ctx = SignalContext(
            epoch=0, total_epochs=1, rng=np.random.default_rng(0),
            device=device, optimizer=opt, params=params, grad_clip=10.0,
        )
        before = emb.weight.detach().clone()
        result = sig.compute(model, emb, ctx)
        after = emb.weight.detach().clone()

        self.assertEqual(result.n_units, 1)
        self.assertGreater(float((after - before).abs().sum()), 0.0)

    def test_multi_spec_per_spec_backward_does_not_double_traverse_graph(self) -> None:
        # Regression for iter 68 r5 crash: when ``compute`` does per-spec
        # backward, each spec must build its own embedding lookup so the
        # second spec's backward does not try to re-traverse the first's
        # already-freed saved tensors.
        device = torch.device("cpu")
        torch.manual_seed(3)
        model = _tiny_model()
        emb = nn.Embedding(2, EMBEDDING_DIM)
        nn.init.normal_(emb.weight, std=0.1)
        params = list(model.parameters()) + list(emb.parameters())
        opt = torch.optim.Adam(params, lr=1e-3)

        from dataclasses import replace
        spec_a = _toy_glucose_spec()
        spec_b = replace(_toy_glucose_spec(), name="toy_glucose_meal_b")

        sig = CohortStatisticSignal(
            specs=[spec_a, spec_b],
            n_patients=2, sample_patients=2,
            weight=WeightSchedule(0.5),
        )
        ctx = SignalContext(
            epoch=0, total_epochs=1, rng=np.random.default_rng(0),
            device=device, optimizer=opt, params=params, grad_clip=10.0,
        )
        result = sig.compute(model, emb, ctx)
        self.assertEqual(result.n_units, 1)
        self.assertEqual(len(result.sub_metrics), 2)


class TestCohortStatisticSignalAdaptive(unittest.TestCase):
    """Iter 74: violation-proportional reweighting between specs."""

    def _easy_spec(self) -> CohortStatisticSpec:
        # target ≈ model output → small z² loss.
        from dataclasses import replace
        return replace(_toy_glucose_spec(), name="easy", target=0.0, sigma=10.0)

    def _hard_spec(self) -> CohortStatisticSpec:
        # target wildly off → large z² loss regardless of model init.
        from dataclasses import replace
        return replace(_toy_glucose_spec(), name="hard", target=1000.0, sigma=10.0)

    def test_adaptive_populates_ema_and_emits_per_spec_weights(self) -> None:
        device = torch.device("cpu")
        torch.manual_seed(2)
        model = _tiny_model()
        emb = nn.Embedding(2, EMBEDDING_DIM)
        nn.init.normal_(emb.weight, std=0.1)
        params = list(model.parameters()) + list(emb.parameters())
        opt = torch.optim.Adam(params, lr=1e-2)
        sig = CohortStatisticSignal(
            specs=[self._easy_spec(), self._hard_spec()],
            n_patients=2, sample_patients=2,
            weight=WeightSchedule(0.5),
            adaptive=True,
        )
        ctx = SignalContext(
            epoch=0, total_epochs=1, rng=np.random.default_rng(0),
            device=device, optimizer=opt, params=params, grad_clip=10.0,
        )
        result = sig.compute(model, emb, ctx)
        # EMA initialised from the first epoch's per-spec losses.
        self.assertEqual(set(sig._violation_ema.keys()), {"easy", "hard"})
        # Adaptive mode emits the applied per-spec weights for logging.
        self.assertIn("w_easy", result.sub_metrics)
        self.assertIn("w_hard", result.sub_metrics)

    def test_adaptive_concentrates_budget_on_worse_spec(self) -> None:
        device = torch.device("cpu")
        torch.manual_seed(2)
        model = _tiny_model()
        emb = nn.Embedding(2, EMBEDDING_DIM)
        nn.init.normal_(emb.weight, std=0.1)
        params = list(model.parameters()) + list(emb.parameters())
        opt = torch.optim.Adam(params, lr=1e-3)
        sig = CohortStatisticSignal(
            specs=[self._easy_spec(), self._hard_spec()],
            n_patients=2, sample_patients=2,
            weight=WeightSchedule(0.5),
            adaptive=True,
        )
        ctx = SignalContext(
            epoch=0, total_epochs=1, rng=np.random.default_rng(0),
            device=device, optimizer=opt, params=params, grad_clip=10.0,
        )
        sig.compute(model, emb, ctx)
        weights = sig._adaptive_weights_from_ema()
        # The badly-missed spec must pull harder than the matched one...
        self.assertGreater(weights["hard"], weights["easy"])
        # ...and the total budget is preserved (only the distribution moved).
        self.assertAlmostEqual(
            sum(weights.values()), sig._base_weight_sum, places=5,
        )

    def test_non_adaptive_is_unchanged(self) -> None:
        # Without adaptive, no EMA is built and no per-spec weights are emitted
        # (back-compat with the fixed-budget weighted mean).
        device = torch.device("cpu")
        torch.manual_seed(2)
        model = _tiny_model()
        emb = nn.Embedding(2, EMBEDDING_DIM)
        nn.init.normal_(emb.weight, std=0.1)
        params = list(model.parameters()) + list(emb.parameters())
        opt = torch.optim.Adam(params, lr=1e-3)
        sig = CohortStatisticSignal(
            specs=[self._easy_spec(), self._hard_spec()],
            n_patients=2, sample_patients=2,
            weight=WeightSchedule(0.5),
            adaptive=False,
        )
        ctx = SignalContext(
            epoch=0, total_epochs=1, rng=np.random.default_rng(0),
            device=device, optimizer=opt, params=params, grad_clip=10.0,
        )
        result = sig.compute(model, emb, ctx)
        self.assertEqual(sig._violation_ema, {})
        self.assertFalse(any(k.startswith("w_") for k in result.sub_metrics))


if __name__ == "__main__":
    unittest.main()
