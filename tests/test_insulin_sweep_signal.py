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

from pulse.knowledge.full_body import FullBody, PatientParams, resolve_derived_params
from pulse.model import ModularPhysiologyNetwork
from pulse.training import (
    InsulinSweepProtocol,
    InsulinSweepSignal,
    SignalContext,
    SignalResult,
    WeightSchedule,
    joint_aux_step,
)
from pulse.training.insulin_sweep_signal import _cold_metabolic_rates, patient_from_record
from pulse.types import EMBEDDING_DIM, MARKER_INDEX


def _tiny_model() -> ModularPhysiologyNetwork:
    return ModularPhysiologyNetwork(
        metabolic_hidden=16, appetite_hidden=12, stress_hidden=12,
        cardiovascular_hidden=16, thermoreg_hidden=8, respiratory_hidden=8,
    )


def _median_patients(n: int) -> dict[int, PatientParams]:
    """One median teacher per embedding row, so a sampled row has a target."""
    return {i: PatientParams() for i in range(n)}


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
            patient_params=_median_patients(2),
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
            patient_params=_median_patients(1),
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
            patient_params=_median_patients(1),
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
    """With ``sample_patients = 0`` the zero embedding is the only row, and it is scored
    against ``PatientParams()``. Sampling without a teacher patient for that row raises.
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

    def test_sampling_without_patient_params_fails(self) -> None:
        with self.assertRaises(RuntimeError) as caught:
            self._run(sample_patients=2)
        self.assertIn("no patient_params", str(caught.exception))

    def test_a_sampled_row_is_scored_and_zero_stays_last(self) -> None:
        result, seen, emb = self._run(
            sample_patients=2, patient_params=_median_patients(5),
        )
        for rows in seen:
            self.assertEqual(tuple(rows.shape), (3, EMBEDDING_DIM))
            self.assertGreater(float(rows[:2].abs().sum()), 0.0)  # the sampled rows ...
            self.assertEqual(int(torch.count_nonzero(rows[2])), 0)  # ... then zero, appended last
        self.assertEqual(result.sub_metrics["n_grid_emb_pairs"], 12.0)  # 4 grid points x 3 rows
        self.assertIsNotNone(emb.weight.grad)

    def test_zero_embedding_off_and_no_patients_is_a_noop(self) -> None:
        result, seen, _ = self._run(include_default_embedding=False)
        self.assertEqual(result.n_units, 0)
        self.assertEqual(seen, [])


class TestPerPatientTargets(unittest.TestCase):
    """PLAN B4: a sampled row is scored against its own teacher PatientParams.
    The zero row stays on PatientParams().
    """

    _PROTO = InsulinSweepProtocol(
        glucose_sweep_mg_dL=(80.0, 200.0), insulin_sweep_uU_mL=(10.0, 40.0),
    )

    def _own(self) -> PatientParams:
        return resolve_derived_params(PatientParams(
            Gb=120.0, Ib=18.0, gamma=0.2, n=0.4, body_mass_kg=110.0, glyc_ins_K=40.0,
        ))

    def test_the_sampled_row_uses_its_patient_and_the_zero_row_uses_the_median(self) -> None:
        own = self._own()
        sig = InsulinSweepSignal(patient_params={0: own}, protocol=self._PROTO)
        g_own, _i_own, _gs, i_states = sig._curves_for_row(0)
        g_zero, i_zero, _, _ = sig._curves_for_row(None)

        expected = np.stack([
            _cold_metabolic_rates(g, own.Ib, own) for g in self._PROTO.glucose_sweep_mg_dL
        ])
        self.assertTrue(np.allclose(g_own.numpy(), expected))
        median = np.stack([
            _cold_metabolic_rates(g, PatientParams().Ib, PatientParams())
            for g in self._PROTO.glucose_sweep_mg_dL
        ])
        self.assertTrue(np.allclose(g_zero.numpy(), median))
        self.assertFalse(np.allclose(g_own.numpy(), g_zero.numpy()))
        # The insulin sweep holds glucose at the patient's Gb, not at 95.
        self.assertTrue(np.allclose(i_states[:, MARKER_INDEX["glucose"]].numpy(), own.Gb))
        # Batch order is sampled rows, then the zero embedding.
        g_batch, i_batch, _, _ = sig._batch([0, None])
        self.assertTrue(torch.equal(g_batch[:, 0], g_own))
        self.assertTrue(torch.equal(g_batch[:, 1], g_zero))
        self.assertTrue(torch.equal(i_batch[:, 1], i_zero))

    def test_matching_each_row_to_its_own_target_is_zero_loss(self) -> None:
        own = self._own()
        sig = InsulinSweepSignal(patient_params={0: own}, protocol=self._PROTO)
        g_batch, i_batch, _, _ = sig._batch([0, None])
        for pred, target, species_w in (
            (g_batch, g_batch, sig._g_species_w),
            (i_batch, i_batch, sig._i_species_w),
        ):
            mse, rank, auc = sig._sweep_losses(pred, target, species_w)
            self.assertAlmostEqual(float(mse), 0.0, places=5)
            self.assertAlmostEqual(float(rank), 0.0, places=5)
            self.assertAlmostEqual(float(auc), 0.0, places=5)
        # Scoring both rows against the median leaves a residual on the sampled row.
        median = sig._curves_for_row(None)[0].unsqueeze(1).expand_as(g_batch)
        mse_median, _, _ = sig._sweep_losses(median, g_batch, sig._g_species_w)
        self.assertGreater(float(mse_median), 1e-6)

    def test_compute_loss_moves_when_the_sampled_patient_moves(self) -> None:
        torch.manual_seed(0)
        model = _tiny_model()
        model.eval()
        emb = nn.Embedding(1, EMBEDDING_DIM)
        nn.init.normal_(emb.weight, std=0.3)
        own = self._own()

        def loss(patients: dict[int, PatientParams]) -> float:
            sig = InsulinSweepSignal(
                n_patients=1, sample_patients=1, weight=WeightSchedule(1.0),
                protocol=self._PROTO, patient_params=patients,
            )
            params = list(model.parameters()) + list(emb.parameters())
            ctx = SignalContext(
                epoch=0, total_epochs=1, rng=np.random.default_rng(0),
                device=torch.device("cpu"), optimizer=torch.optim.SGD(params, lr=0.0),
                params=params, grad_clip=1e9,
            )
            model.zero_grad(set_to_none=True)
            return sig.compute(model, emb, ctx).loss_sum

        self.assertGreater(abs(loss({0: own}) - loss({0: PatientParams()})), 1e-4)

    def test_a_record_missing_a_field_does_not_fall_back_to_the_median(self) -> None:
        raw = {
            "body_mass_kg": 70.0, "si": 0.0004, "glyc_ins_k": 25.0,
            "gamma": 0.05, "k_ins": 0.15, "act_insulin_sens": 0.3,
        }
        got = patient_from_record(raw)
        median = PatientParams()
        self.assertAlmostEqual(got.gamma, median.gamma)
        self.assertAlmostEqual(got.Si, median.Si)
        broken = dict(raw)
        del broken["gamma"]
        with self.assertRaises(ValueError) as caught:
            patient_from_record(broken)
        self.assertIn("gamma", str(caught.exception))
        # A key this sweep does not read does not abort the conversion.
        patient_from_record({**raw, "k_second_phase": 0.2})

    def test_a_real_episode_record_becomes_that_patients_target(self) -> None:
        ep = FullBody(n_days=1).generate_episodes(1, np.random.default_rng(0))[0]
        assert ep.patient_params is not None
        got = patient_from_record(ep.patient_params)
        self.assertAlmostEqual(got.Si, ep.patient_params["si"])
        self.assertAlmostEqual(got.glyc_ins_K, ep.patient_params["glyc_ins_k"])
        self.assertAlmostEqual(got.gamma, ep.patient_params["gamma"])
        self.assertAlmostEqual(got.n, ep.patient_params["k_ins"])
        self.assertAlmostEqual(got.body_mass_kg, ep.patient_params["body_mass_kg"])
        self.assertAlmostEqual(got.act_insulin_sens, ep.patient_params["act_insulin_sens"])
        sig = InsulinSweepSignal(
            patient_params={3: got},
            protocol=InsulinSweepProtocol(glucose_sweep_mg_dL=(95.0,), insulin_sweep_uU_mL=(10.0,)),
        )
        g_target = sig._curves_for_row(3)[0]
        expected = _cold_metabolic_rates(95.0, got.Ib, got)
        self.assertTrue(np.allclose(g_target[0].numpy(), expected))


if __name__ == "__main__":
    unittest.main()
