"""Unit tests for the insulin sweep training signal.

Covers, mirroring ``test_gut_dose_sweep_signal``:

* cold targets reproduce the Bergman GSIR / clearance shapes
  (monotone increasing in glucose above threshold; monotone
  decreasing in insulin)
* zero weight is a no-op; positive weight steps the metabolic module
* gradient lands on the metabolic module + its embedding projection,
  not on the gut, stress, or vital-sign modules
* a few SGD iterations strictly reduce the loss
"""

from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn as nn

from pulse.knowledge.full_body import PatientParams
from pulse.model import ModularPhysiologyNetwork
from pulse.training import (
    InsulinSweepProtocol,
    InsulinSweepSignal,
    SignalContext,
    SignalResult,
    WeightSchedule,
    joint_aux_step,
)
from pulse.training.insulin_sweep_signal import _cold_metabolic_rates
from pulse.types import EMBEDDING_DIM


def _tiny_model() -> ModularPhysiologyNetwork:
    return ModularPhysiologyNetwork(
        metabolic_hidden=16, appetite_hidden=12, stress_hidden=12,
        cardiovascular_hidden=16, thermoreg_hidden=8, respiratory_hidden=8,
    )


class TestColdTargets(unittest.TestCase):
    def test_dI_monotone_in_glucose_above_threshold(self) -> None:
        """GSIR: dI/dt strictly increases with G once G > h. Cold model uses
        ``params.gamma * max(G - h, 0) * incretin_factor`` as the secretion
        term, so any G > h gives a positive contribution that scales linearly."""
        params = PatientParams()
        di = [
            float(_cold_metabolic_rates(g, params.Ib, params)[1])
            for g in (60.0, 80.0, 95.0, 120.0, 150.0, 200.0, 250.0, 300.0)
        ]
        # First two are below or at threshold h=80 — clearance pulls insulin
        # toward effective_Ib < Ib, so dI is negative.
        self.assertLess(di[0], 0.0)
        # 95 mg/dL: just barely above h=80, slight positive secretion.
        # 200, 250, 300 should be strictly increasing in dI.
        for prev, nxt in zip(di[3:], di[4:]):
            self.assertLess(prev, nxt, f"dI not monotone in G: {di}")

    def test_dI_monotone_decreasing_in_insulin(self) -> None:
        """Clearance: dI/dt strictly decreases with I when GSIR contribution
        is fixed (G held at Gb). At G=Gb the gamma term = 0, so dI is
        purely ``-n * (I - Ib)``."""
        params = PatientParams()
        di = [
            float(_cold_metabolic_rates(params.Gb, i, params)[1])
            for i in (5.0, 10.0, 20.0, 40.0, 80.0)
        ]
        for prev, nxt in zip(di, di[1:]):
            self.assertGreater(prev, nxt, f"dI not monotone-decreasing in I: {di}")

    def test_hep_target_responds_to_insulin(self) -> None:
        """Hepatic glucose output target is suppressed by elevated insulin —
        the teacher's ``glucose_fluxes`` gates glycogenolysis and GNG on insulin
        above basal (``_glyc_ins_gate``). dHep should decrease as I goes from Ib up."""
        params = PatientParams()
        dhep = [
            float(_cold_metabolic_rates(params.Gb, i, params)[6])
            for i in (10.0, 20.0, 40.0, 80.0)
        ]
        for prev, nxt in zip(dhep, dhep[1:]):
            self.assertGreater(prev, nxt, f"dHep not suppressed by I: {dhep}")


class TestSignalGating(unittest.TestCase):
    def test_zero_weight_is_noop(self) -> None:
        device = torch.device("cpu")
        model = _tiny_model()
        emb = nn.Embedding(2, EMBEDDING_DIM)
        params = list(model.parameters()) + list(emb.parameters())
        opt = torch.optim.Adam(params, lr=1e-3)
        sig = InsulinSweepSignal(
            n_patients=2, sample_patients=2, weight=WeightSchedule(0.0),
        )
        ctx = SignalContext(
            epoch=0, total_epochs=1, rng=np.random.default_rng(0),
            device=device, optimizer=opt, params=params, grad_clip=10.0,
        )
        result = sig.compute(model, emb, ctx)
        self.assertEqual(result.n_units, 0)

    def test_positive_weight_takes_a_step(self) -> None:
        device = torch.device("cpu")
        torch.manual_seed(0)
        model = _tiny_model()
        emb = nn.Embedding(2, EMBEDDING_DIM)
        nn.init.normal_(emb.weight, std=0.1)
        params = list(model.parameters()) + list(emb.parameters())
        opt = torch.optim.Adam(params, lr=1e-2)
        sig = InsulinSweepSignal(
            n_patients=2, sample_patients=2, weight=WeightSchedule(1.0),
        )
        ctx = SignalContext(
            epoch=0, total_epochs=1, rng=np.random.default_rng(0),
            device=device, optimizer=opt, params=params, grad_clip=10.0,
        )
        before = {n: p.detach().clone() for n, p in model.metabolic.named_parameters()}
        result = sig.compute(model, emb, ctx)
        # iter 78: aux signals accumulate; the trainer applies the joint step.
        joint_aux_step(ctx)
        moved = any(
            float((p.detach() - before[n]).abs().sum()) > 0
            for n, p in model.metabolic.named_parameters()
        )
        self.assertEqual(result.n_units, 1)
        self.assertTrue(moved, "metabolic module parameters did not move")
        self.assertGreater(result.sub_metrics["n_grid_emb_pairs"], 0)


class TestGradientFlow(unittest.TestCase):
    def test_unrelated_modules_unchanged(self) -> None:
        device = torch.device("cpu")
        torch.manual_seed(0)
        model = _tiny_model()
        emb = nn.Embedding(1, EMBEDDING_DIM)
        params = list(model.parameters()) + list(emb.parameters())
        opt = torch.optim.SGD(params, lr=1e-2)
        sig = InsulinSweepSignal(
            n_patients=1, sample_patients=1, weight=WeightSchedule(1.0),
        )
        ctx = SignalContext(
            epoch=0, total_epochs=1, rng=np.random.default_rng(0),
            device=device, optimizer=opt, params=params, grad_clip=1e9,
        )
        before = {
            mod_name: {n: p.detach().clone() for n, p in mod.named_parameters()}
            for mod_name, mod in (
                ("gut", model.gut),
                ("respiratory", model.respiratory),
                ("cardiovascular", model.cardiovascular),
                ("thermoreg", model.thermoreg),
                ("stress", model.stress),
                ("appetite", model.appetite),
            )
        }
        sig.compute(model, emb, ctx)
        # iter 78: aux signals accumulate; the trainer applies the joint step.
        # Unrelated modules must still receive no gradient even when one is taken.
        if ctx.aux_accumulated:
            joint_aux_step(ctx)
        for mod_name, mod in (
            ("gut", model.gut),
            ("respiratory", model.respiratory),
            ("cardiovascular", model.cardiovascular),
            ("thermoreg", model.thermoreg),
            ("stress", model.stress),
            ("appetite", model.appetite),
        ):
            moved = any(
                float((p.detach() - before[mod_name][n]).abs().sum()) > 0
                for n, p in mod.named_parameters()
            )
            self.assertFalse(
                moved,
                f"{mod_name} module unexpectedly moved under insulin-sweep step",
            )

    def test_gradient_lands_on_metabolic_pipeline(self) -> None:
        """Manually replicate a sweep_rates → loss → backward to inspect grads."""
        torch.manual_seed(0)
        model = _tiny_model()
        sig = InsulinSweepSignal(protocol=InsulinSweepProtocol(
            glucose_sweep_mg_dL=(95.0, 200.0),
            insulin_sweep_uU_mL=(10.0, 40.0),
        ))
        emb_full = torch.zeros(1, EMBEDDING_DIM, requires_grad=False)
        g_states = sig._g_states
        g_pred = sig._sweep_rates(model, g_states, emb_full)
        loss = ((g_pred[..., 1] - sig._g_targets[:, 1].unsqueeze(1)) ** 2).mean()
        loss.backward()

        met_grad = sum(
            float(p.grad.abs().sum()) for p in model.metabolic.parameters()
            if p.grad is not None
        )
        proj_grad = sum(
            float(p.grad.abs().sum())
            for p in model.embedding_projections["metabolic"].parameters()
            if p.grad is not None
        )
        self.assertGreater(met_grad, 0.0)
        self.assertGreater(proj_grad, 0.0)

        for name, sub in (
            ("gut", model.gut),
            ("appetite", model.appetite),
            ("cardiovascular", model.cardiovascular),
            ("respiratory", model.respiratory),
        ):
            grad_total = sum(
                float(p.grad.abs().sum()) for p in sub.parameters()
                if p.grad is not None
            )
            self.assertEqual(
                grad_total, 0.0,
                f"{name} module unexpectedly received gradient ({grad_total})",
            )


class TestLearningSanity(unittest.TestCase):
    def test_loss_strictly_decreases_over_iterations(self) -> None:
        device = torch.device("cpu")
        torch.manual_seed(0)
        model = _tiny_model()
        emb = nn.Embedding(1, EMBEDDING_DIM)
        params = list(model.parameters()) + list(emb.parameters())
        opt = torch.optim.Adam(params, lr=5e-3)
        sig = InsulinSweepSignal(
            n_patients=1, sample_patients=1, weight=WeightSchedule(1.0),
        )
        rng = np.random.default_rng(0)
        losses: list[float] = []
        for ep in range(8):
            ctx = SignalContext(
                epoch=ep, total_epochs=8, rng=rng,
                device=device, optimizer=opt, params=params, grad_clip=10.0,
            )
            r = sig.compute(model, emb, ctx)
            # iter 78: aux signals accumulate; the trainer applies the joint step.
            joint_aux_step(ctx)
            losses.append(r.loss_sum)
        self.assertLess(losses[-1], losses[0],
                        f"loss did not decrease: {losses}")


class TestZeroEmbeddingOnlyByDefault(unittest.TestCase):
    """PLAN A6: the seven target rates are the DEFAULT patient's, so by default the zero
    embedding (the median person) is the only row scored against them.

    Until PLAN A6 ``sample_patients = 4`` also scored four sampled rows per step: from epoch 0,
    at the spec's 0.30 weight, a pull of each patient's metabolic rates toward the median person's.
    """

    _PROTO = InsulinSweepProtocol(
        glucose_sweep_mg_dL=(95.0, 200.0), insulin_sweep_uU_mL=(10.0, 40.0),
    )

    def _run(self, **kw: object) -> tuple[SignalResult, list[torch.Tensor], nn.Embedding]:
        torch.manual_seed(0)
        model = _tiny_model()
        emb = nn.Embedding(5, EMBEDDING_DIM)
        nn.init.normal_(emb.weight, std=0.5)
        params = list(model.parameters()) + list(emb.parameters())
        ctx = SignalContext(
            epoch=0, total_epochs=1, rng=np.random.default_rng(0),
            device=torch.device("cpu"), optimizer=torch.optim.SGD(params, lr=0.0),
            params=params, grad_clip=1e9,
        )
        sig = InsulinSweepSignal(n_patients=5, weight=WeightSchedule(1.0), protocol=self._PROTO, **kw)  # type: ignore[arg-type]
        seen: list[torch.Tensor] = []
        hook = model.embedding_projections["metabolic"].register_forward_hook(
            lambda _m, inp, _out: seen.append(inp[0].detach().clone()),
        )
        try:
            result = sig.compute(model, emb, ctx)
        finally:
            hook.remove()
        return result, seen, emb

    def test_defaults_supervise_exactly_one_embedding_and_it_is_zero(self) -> None:
        sig = InsulinSweepSignal()
        self.assertEqual(sig.sample_patients, 0)
        self.assertTrue(sig.include_default_embedding)
        result, seen, emb = self._run()
        self.assertEqual(len(seen), 2)  # the glucose sweep and the insulin sweep ...
        for rows in seen:
            self.assertEqual(tuple(rows.shape), (1, EMBEDDING_DIM))  # ... each over one row ...
            self.assertEqual(int(torch.count_nonzero(rows)), 0)  # ... the zero embedding
        self.assertEqual(result.sub_metrics["n_grid_emb_pairs"], 4.0)  # (2 + 2) grid points x 1 row
        # no table row is in the graph, so nothing is pulled toward the median person's rates
        self.assertIsNone(emb.weight.grad)

    def test_a_nonzero_sample_patients_is_still_honoured(self) -> None:
        result, seen, emb = self._run(sample_patients=2)
        for rows in seen:
            self.assertEqual(tuple(rows.shape), (3, EMBEDDING_DIM))
            self.assertGreater(float(rows[:2].abs().sum()), 0.0)  # the sampled rows ...
            self.assertEqual(int(torch.count_nonzero(rows[2])), 0)  # ... then zero, appended last
        self.assertEqual(result.sub_metrics["n_grid_emb_pairs"], 12.0)  # 4 grid points x 3 rows
        self.assertIsNotNone(emb.weight.grad)  # sampled rows are scored against the default patient

    def test_zero_embedding_off_and_no_patients_is_a_noop(self) -> None:
        result, seen, _ = self._run(include_default_embedding=False)
        self.assertEqual(result.n_units, 0)
        self.assertEqual(seen, [])


if __name__ == "__main__":
    unittest.main()
