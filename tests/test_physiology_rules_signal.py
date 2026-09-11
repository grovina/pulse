"""Regression coverage for ``RolloutEvidenceSignal``.

iter 68 r6 OOM'd at phase 2 ep 50 because ``physiology_rules_epoch_loss``
summed all 60 rules' losses into one composite tensor before a single
backward, holding 60 autograd graphs in memory simultaneously. r7
refactored ``compute()`` to per-rule backward — these tests pin that
behavior and guard against the second-graph-traversal class of bug that
crashed r5.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn as nn

from pulse.knowledge.cohort_types import CohortArmSpec, InitMode
from pulse.knowledge.physiology_rules import (
    PhysiologyRule,
    hinge_min_drop,
    hinge_min_rise,
)
from pulse.model import ModularPhysiologyNetwork
from pulse.physiology_rules_loss import physiology_rule_loss_one_rule
from pulse.training import RolloutEvidenceSignal, SignalContext, WeightSchedule
from pulse.types import EMBEDDING_DIM


def _tiny_model() -> ModularPhysiologyNetwork:
    return ModularPhysiologyNetwork(
        embedding_dim=EMBEDDING_DIM,
        metabolic_hidden=16, appetite_hidden=12, stress_hidden=12,
        cardiovascular_hidden=16, thermoreg_hidden=8, respiratory_hidden=8,
    )


def _arm() -> CohortArmSpec:
    return CohortArmSpec(
        label="meal", duration_min=120, start_hour=8.0,
        meals=((30.0, 50.0, 10.0, 15.0),),
    )


def _glucose_rises_rule(name: str = "glucose_rises") -> PhysiologyRule:
    return PhysiologyRule(
        name=name,
        source="test",
        description="test",
        arms=(_arm(),),
        predicate=lambda traj, ctx: hinge_min_rise(
            traj, ctx.col("glucose"),
            pre=ctx.window(0.0, 30.0),
            post=ctx.window(30.0, 90.0),
            min_rise=5.0,
        ),
        scale=10.0,
        init_mode=InitMode.NORM_CENTER,
    )


def _glucagon_falls_rule(name: str = "glucagon_falls") -> PhysiologyRule:
    return PhysiologyRule(
        name=name,
        source="test",
        description="test",
        arms=(_arm(),),
        predicate=lambda traj, ctx: hinge_min_drop(
            traj, ctx.col("glucagon"),
            pre=ctx.window(0.0, 30.0),
            post=ctx.window(30.0, 90.0),
            min_drop=2.0,
        ),
        scale=5.0,
        init_mode=InitMode.NORM_CENTER,
    )


class TestRolloutEvidenceSignalPerRuleBackward(unittest.TestCase):
    def test_zero_weight_is_noop(self) -> None:
        device = torch.device("cpu")
        model = _tiny_model()
        emb = nn.Embedding(2, EMBEDDING_DIM)
        params = list(model.parameters()) + list(emb.parameters())
        opt = torch.optim.Adam(params, lr=1e-3)
        sig = RolloutEvidenceSignal(
            rules=[_glucose_rises_rule()],
            n_patients=2, sample_patients=2,
            weight=WeightSchedule(0.0),
        )
        ctx = SignalContext(
            epoch=0, total_epochs=1, rng=np.random.default_rng(0),
            device=device, optimizer=opt, params=params, grad_clip=10.0,
        )
        result = sig.compute(model, emb, ctx)
        self.assertEqual(result.n_units, 0)

    def test_multi_rule_per_rule_backward_does_not_double_traverse_graph(self) -> None:
        # The bug that crashed iter 68 r5 in cohort_statistic would have hit
        # physiology_rules too if r7's per-rule backward shared the embedding
        # lookup across rules. Two rules suffice to trigger the second
        # backward; if emb_list were built once outside the loop, the second
        # rule's backward would raise
        # "Trying to backward through the graph a second time".
        device = torch.device("cpu")
        torch.manual_seed(3)
        model = _tiny_model()
        emb = nn.Embedding(2, EMBEDDING_DIM)
        nn.init.normal_(emb.weight, std=0.1)
        params = list(model.parameters()) + list(emb.parameters())
        opt = torch.optim.Adam(params, lr=1e-3)

        sig = RolloutEvidenceSignal(
            rules=[_glucose_rises_rule("rule_a"), _glucagon_falls_rule("rule_b")],
            n_patients=2, sample_patients=2,
            weight=WeightSchedule(0.5),
        )
        ctx = SignalContext(
            epoch=0, total_epochs=1, rng=np.random.default_rng(0),
            device=device, optimizer=opt, params=params, grad_clip=10.0,
        )
        result = sig.compute(model, emb, ctx)
        self.assertEqual(result.n_units, 1)
        # both rules contribute diagnostics + weights
        self.assertIn("viol_rule_a", result.sub_metrics)
        self.assertIn("viol_rule_b", result.sub_metrics)
        self.assertIn("w_rule_a", result.sub_metrics)
        self.assertIn("w_rule_b", result.sub_metrics)

    def test_diagnostics_round_trip(self) -> None:
        # End-to-end: positive weight, two rules, fresh random model. We
        # don't assert params change (a hinge rule that happens to be
        # satisfied for this init produces zero gradient — that's correct
        # behavior, not a bug). Instead pin that compute() emits the
        # per-rule diagnostics needed for adaptive weighting and abort
        # forensics.
        device = torch.device("cpu")
        torch.manual_seed(2)
        model = _tiny_model()
        emb = nn.Embedding(2, EMBEDDING_DIM)
        nn.init.normal_(emb.weight, std=0.1)
        params = list(model.parameters()) + list(emb.parameters())
        opt = torch.optim.Adam(params, lr=1e-2)
        sig = RolloutEvidenceSignal(
            rules=[_glucose_rises_rule("ra"), _glucagon_falls_rule("rb")],
            n_patients=2, sample_patients=2,
            weight=WeightSchedule(0.5),
            adaptive=True,
        )
        ctx = SignalContext(
            epoch=0, total_epochs=1, rng=np.random.default_rng(0),
            device=device, optimizer=opt, params=params, grad_clip=10.0,
        )
        result = sig.compute(model, emb, ctx)
        for rname in ("ra", "rb"):
            self.assertIn(f"viol_{rname}", result.sub_metrics)
            self.assertIn(f"sat_{rname}", result.sub_metrics)
            self.assertIn(f"w_{rname}", result.sub_metrics)
        # Adaptive mode should have populated the EMA from the first epoch.
        self.assertEqual(set(sig._violation_ema.keys()), {"ra", "rb"})


class TestPhysiologyRulesArmMajorEquivalence(unittest.TestCase):
    """iter 79: the arm-major rollout-dedup compute() must be numerically
    identical to the iter-68 rule-major path (same reported loss, same
    accumulated gradient on every parameter) — only faster. This is the
    correctness guarantee that the honest baseline is unchanged.

    Scenario deliberately exercises the two cases dedup must get right:
    rules SHARING an arm (a, b, c all use ``meal``) and a MULTI-ARM rule
    (c spans ``meal`` + ``fast``, so its loss is assembled across two
    arm-groups). Cold init is used so the cold-rollout path is covered.
    """

    def test_arm_major_matches_per_rule_grad_and_loss(self) -> None:
        device = torch.device("cpu")
        torch.manual_seed(7)
        model = _tiny_model()
        emb = nn.Embedding(2, EMBEDDING_DIM)
        nn.init.normal_(emb.weight, std=0.3)
        params = list(model.parameters()) + list(emb.parameters())
        opt = torch.optim.SGD(params, lr=0.0)  # never stepped; grads compared directly

        arm_meal = CohortArmSpec(
            label="meal", duration_min=120, start_hour=8.0,
            meals=((30.0, 60.0, 12.0, 18.0),),
        )
        arm_fast = CohortArmSpec(
            label="fast", duration_min=120, start_hour=8.0, meals=(),
        )

        def rises(min_rise: float):
            return lambda traj, ctx: hinge_min_rise(
                traj, ctx.col("glucose"),
                pre=ctx.window(0.0, 30.0), post=ctx.window(30.0, 90.0),
                min_rise=min_rise,
            )

        def falls(min_drop: float):
            return lambda traj, ctx: hinge_min_drop(
                traj, ctx.col("glucagon"),
                pre=ctx.window(0.0, 30.0), post=ctx.window(30.0, 90.0),
                min_drop=min_drop,
            )

        # Aggressive thresholds so the hinges are violated for the random init
        # — guarantees non-zero gradient so the equivalence test is non-trivial.
        rule_a = PhysiologyRule(
            name="a", source="t", description="t", arms=(arm_meal,),
            predicate=rises(80.0), scale=10.0, init_mode=InitMode.COLD, weight=1.0,
        )
        rule_b = PhysiologyRule(
            name="b", source="t", description="t", arms=(arm_meal,),
            predicate=falls(40.0), scale=5.0, init_mode=InitMode.COLD, weight=2.0,
        )
        rule_c = PhysiologyRule(
            name="c", source="t", description="t", arms=(arm_meal, arm_fast),
            predicate=rises(80.0), scale=8.0, init_mode=InitMode.COLD, weight=0.5,
        )
        rules = [rule_a, rule_b, rule_c]

        sig = RolloutEvidenceSignal(
            rules=rules, n_patients=2, sample_patients=2,
            include_default_embedding=True, weight=WeightSchedule(0.5),
        )
        w = sig.weight_at(0)
        weight_sum = sum(r.weight for r in rules)
        init_fn = sig._hinge_initial_state_fn(device)

        def fresh_emb_list() -> list[torch.Tensor]:
            return [
                emb(torch.tensor(0)), emb(torch.tensor(1)),
                torch.zeros(EMBEDDING_DIM),
            ]

        # --- reference: the iter-68 rule-major per-rule backward ---
        opt.zero_grad(set_to_none=True)
        ref_raw = 0.0
        for rule in rules:
            init_for_arm = lambda arm, _r=rule: init_fn(_r, arm)
            loss_t, _, _ = physiology_rule_loss_one_rule(
                model, fresh_emb_list(), rule, init_for_arm,
            )
            ((w * rule.weight / weight_sum) * loss_t).backward()
            ref_raw += float(loss_t.detach()) * rule.weight
        ref_raw /= weight_sum
        ref_grads = {
            id(p): (None if p.grad is None else p.grad.detach().clone())
            for p in params
        }
        ref_norm = sum(
            float(p.grad.pow(2).sum()) for p in params if p.grad is not None
        ) ** 0.5

        # --- new: the iter-79 arm-major compute() ---
        opt.zero_grad(set_to_none=True)
        ctx = SignalContext(
            epoch=0, total_epochs=1, rng=np.random.default_rng(0),
            device=device, optimizer=opt, params=params, grad_clip=10.0,
        )
        result = sig.compute(model, emb, ctx)

        self.assertGreater(ref_norm, 0.0, "test is trivial if no gradient flows")
        # Relative tolerance: the two paths sum the same terms in a different
        # order, so float32 reduction rounding (~1e-8 relative) is expected.
        self.assertAlmostEqual(
            result.loss_sum, ref_raw, delta=abs(ref_raw) * 1e-5 + 1e-6,
        )
        for p in params:
            ref_g = ref_grads[id(p)]
            if ref_g is None:
                self.assertTrue(p.grad is None or float(p.grad.abs().max()) < 1e-9)
            else:
                # The two paths sum identical terms in different order, so the
                # divergence is pure float32 reduction-order noise (exact in real
                # arithmetic). Iter 83 floored the glucose setpoint gain Sg
                # (0.11 -> ~0.9), making the setpoint gradient ~8x larger, which
                # raised the absolute noise floor to ~3e-5 (max ~3e-3 relative on
                # small-magnitude elements, also amplified by physiology-rule
                # hinge kinks). Tolerances scaled accordingly; a real refactor
                # divergence would be ~1e-1+ relative, still caught with margin.
                self.assertTrue(
                    torch.allclose(p.grad, ref_g, atol=1e-4, rtol=2e-3),
                    msg=f"gradient mismatch on param shape {tuple(p.shape)}",
                )


if __name__ == "__main__":
    unittest.main()
