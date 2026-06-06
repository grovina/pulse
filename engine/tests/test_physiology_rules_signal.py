"""Regression coverage for ``PhysiologyRulesSignal``.

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
from pulse.training import PhysiologyRulesSignal, SignalContext, WeightSchedule
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


class TestPhysiologyRulesSignalPerRuleBackward(unittest.TestCase):
    def test_zero_weight_is_noop(self) -> None:
        device = torch.device("cpu")
        model = _tiny_model()
        emb = nn.Embedding(2, EMBEDDING_DIM)
        params = list(model.parameters()) + list(emb.parameters())
        opt = torch.optim.Adam(params, lr=1e-3)
        sig = PhysiologyRulesSignal(
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

        sig = PhysiologyRulesSignal(
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
        sig = PhysiologyRulesSignal(
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


if __name__ == "__main__":
    unittest.main()
