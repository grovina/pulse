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
from pulse.cohort_loss import cohort_statistic_loss_one_spec
from pulse.model import ModularPhysiologyNetwork
from pulse.training import (
    CohortStatisticSignal,
    SignalContext,
    WeightSchedule,
    joint_aux_step,
)
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
        # iter 78: aux signals accumulate; the trainer applies the joint step.
        joint_aux_step(ctx)
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
        # iter 78: aux signals accumulate; the trainer applies the joint step.
        joint_aux_step(ctx)
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


class TestCohortStatisticProtocolBatchingEquivalence(unittest.TestCase):
    """iter 79: cross-spec protocol batching must be numerically identical to
    the iter-68 per-spec path (same reported loss, same accumulated gradient
    on every parameter) — only faster. The scenario covers the cases the
    batching must get right: two specs SHARING an arm protocol but with
    DISTINCT per-spec cold-init states (so the rollout must stack states, not
    broadcast one), a singleton-protocol spec, and a 2-arm DELTA spec.
    """

    def test_protocol_batching_matches_per_spec_grad_and_loss(self) -> None:
        device = torch.device("cpu")
        torch.manual_seed(11)
        model = _tiny_model()
        emb = nn.Embedding(2, EMBEDDING_DIM)
        nn.init.normal_(emb.weight, std=0.3)
        params = list(model.parameters()) + list(emb.parameters())
        opt = torch.optim.SGD(params, lr=0.0)  # never stepped; grads compared directly

        meal = CohortArmSpec(
            label="meal", duration_min=120, start_hour=8.0,
            meals=((30.0, 60.0, 12.0, 18.0),),
        )
        fast = CohortArmSpec(label="fast", duration_min=120, start_hour=8.0, meals=())
        fasted = CohortArmSpec(label="fasted", duration_min=120, start_hour=8.0, meals=())
        fed = CohortArmSpec(
            label="fed", duration_min=120, start_hour=8.0,
            meals=((30.0, 60.0, 12.0, 18.0),),
        )
        win = StatisticWindow(start_min=60, end_min=110)

        # a & b share ``(meal,)`` → one group; targets far off so z² is large.
        spec_a = CohortStatisticSpec(
            name="a", source="t", description="t", arms=(meal,),
            marker_id="glucose", kind=StatisticKind.MEAN_IN_WINDOW,
            window=win, target=200.0, sigma=10.0, weight=1.0,
        )
        spec_b = CohortStatisticSpec(
            name="b", source="t", description="t", arms=(meal,),
            marker_id="hr", kind=StatisticKind.PEAK_VALUE,
            window=win, target=200.0, sigma=10.0, weight=2.0,
        )
        spec_c = CohortStatisticSpec(  # singleton protocol
            name="c", source="t", description="t", arms=(fast,),
            marker_id="glucose", kind=StatisticKind.MEAN_IN_WINDOW,
            window=win, target=50.0, sigma=10.0, weight=0.5,
        )
        spec_d = CohortStatisticSpec(  # 2-arm delta
            name="d", source="t", description="t", arms=(fasted, fed),
            marker_id="glucose", kind=StatisticKind.DELTA_MEANS,
            window=win, target=50.0, sigma=10.0, weight=1.5,
        )
        specs = [spec_a, spec_b, spec_c, spec_d]

        sig = CohortStatisticSignal(
            specs=specs, n_patients=2, sample_patients=2,
            include_default_embedding=True, weight=WeightSchedule(0.5),
        )
        w = sig.weight_at(0)
        total_weight = sum(s.weight for s in specs)
        init_fn = sig._build_initial_state_fn(np.random.default_rng(0), device)

        def fresh_emb_list() -> list[torch.Tensor]:
            return [
                emb(torch.tensor(0)), emb(torch.tensor(1)),
                torch.zeros(EMBEDDING_DIM),
            ]

        # a & b really do land in the same group (the dedup actually fires).
        groups: dict = {}
        for s in specs:
            groups.setdefault(s.arms, []).append(s)
        self.assertIn(2, [len(g) for g in groups.values()])

        # --- reference: iter-68 per-spec backward ---
        opt.zero_grad(set_to_none=True)
        ref_raw = 0.0
        for spec in specs:
            loss_t, _, _ = cohort_statistic_loss_one_spec(
                model, fresh_emb_list(), spec, init_fn(spec),
            )
            ((w * spec.weight / total_weight) * loss_t).backward()
            ref_raw += float(loss_t.detach()) * spec.weight
        ref_raw /= total_weight
        ref_grads = {
            id(p): (None if p.grad is None else p.grad.detach().clone())
            for p in params
        }
        ref_norm = sum(
            float(p.grad.pow(2).sum()) for p in params if p.grad is not None
        ) ** 0.5

        # --- new: iter-79 grouped compute() ---
        opt.zero_grad(set_to_none=True)
        ctx = SignalContext(
            epoch=0, total_epochs=1, rng=np.random.default_rng(0),
            device=device, optimizer=opt, params=params, grad_clip=10.0,
        )
        result = sig.compute(model, emb, ctx)

        self.assertGreater(ref_norm, 0.0, "test is trivial if no gradient flows")
        self.assertAlmostEqual(
            result.loss_sum, ref_raw, delta=abs(ref_raw) * 1e-5 + 1e-6,
        )
        for p in params:
            ref_g = ref_grads[id(p)]
            if ref_g is None:
                self.assertTrue(p.grad is None or float(p.grad.abs().max()) < 1e-9)
            else:
                self.assertTrue(
                    torch.allclose(p.grad, ref_g, atol=1e-6, rtol=1e-4),
                    msg=f"gradient mismatch on param shape {tuple(p.shape)}",
                )


if __name__ == "__main__":
    unittest.main()
