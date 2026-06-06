"""Tests for the default-patient trajectory distillation in
``TrajectoryRolloutSignal``.

The textbook benchmark always queries the model at the zero ("default")
embedding. To keep zero-embedding rollouts inside the cold-model
distribution, the signal can supervise extra cold-model episodes through
that zero embedding directly. These tests pin down that contract.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn as nn

from pulse.model import ModularPhysiologyNetwork
from pulse.training import (
    SignalContext,
    TrajectoryRolloutSignal,
    WeightSchedule,
)
from pulse.training.trajectory_signal import generate_trajectory_dataset
from pulse.types import EMBEDDING_DIM


def _tiny_model() -> ModularPhysiologyNetwork:
    return ModularPhysiologyNetwork(
        metabolic_hidden=8,
        appetite_hidden=8,
        stress_hidden=8,
        cardiovascular_hidden=8,
        thermoreg_hidden=8,
        respiratory_hidden=8,
    )


class TestDefaultPatientDataset(unittest.TestCase):
    def test_default_entries_are_appended_with_flag(self) -> None:
        ds = generate_trajectory_dataset(
            n_patients=2, seed=0, n_days=1, weight_by_name=None,
            n_default_patients=2,
        )
        self.assertEqual(len(ds), 4)
        self.assertFalse(ds[0]["is_default"])
        self.assertFalse(ds[1]["is_default"])
        self.assertTrue(ds[2]["is_default"])
        self.assertTrue(ds[3]["is_default"])
        self.assertEqual(ds[2]["knowledge_source"], "full_body_default")
        self.assertLess(ds[2]["patient_id"], 0)
        self.assertNotEqual(ds[2]["patient_id"], ds[3]["patient_id"])

    def test_zero_default_patients_keeps_dataset_unchanged(self) -> None:
        ds = generate_trajectory_dataset(
            n_patients=3, seed=0, n_days=1, weight_by_name=None,
            n_default_patients=0,
        )
        self.assertEqual(len(ds), 3)
        for row in ds:
            self.assertFalse(row["is_default"])


class TestDefaultPatientSupervision(unittest.TestCase):
    def _signal(self, n_default: int) -> TrajectoryRolloutSignal:
        return TrajectoryRolloutSignal(
            n_patients=1,
            n_days=1,
            seed=0,
            contribution_weights={"full_body": 1.0},
            windows_per_patient=1,
            meal_window_bias=0.0,
            input_dropout=0.0,
            huber_delta=1.0,
            gut_loss_weight=0.0,
            coupling_weight=WeightSchedule(0.0),
            verifier_weight=WeightSchedule(0.0),
            n_default_patients=n_default,
        )

    def _run_one_epoch(self, sig: TrajectoryRolloutSignal) -> tuple[
        ModularPhysiologyNetwork, nn.Embedding,
    ]:
        torch.manual_seed(7)
        model = _tiny_model()
        emb = nn.Embedding(1, EMBEDDING_DIM)
        nn.init.normal_(emb.weight, std=0.1)
        params = list(model.parameters()) + list(emb.parameters())
        opt = torch.optim.Adam(params, lr=1e-3)
        ctx = SignalContext(
            epoch=0, total_epochs=1, rng=np.random.default_rng(0),
            device=torch.device("cpu"),
            optimizer=opt, params=params, grad_clip=10.0,
        )
        sig.compute(model, emb, ctx)
        return model, emb

    def test_default_patient_windows_increase_step_count(self) -> None:
        """Adding default-patient episodes adds the corresponding number of
        gradient steps per epoch."""
        sig0 = self._signal(n_default=0)
        sig2 = self._signal(n_default=2)
        torch.manual_seed(7)
        model = _tiny_model()
        emb = nn.Embedding(1, EMBEDDING_DIM)
        params = list(model.parameters()) + list(emb.parameters())
        opt = torch.optim.Adam(params, lr=1e-3)
        ctx = SignalContext(
            epoch=0, total_epochs=1, rng=np.random.default_rng(0),
            device=torch.device("cpu"),
            optimizer=opt, params=params, grad_clip=10.0,
        )
        r0 = sig0.compute(model, emb, ctx)

        torch.manual_seed(7)
        model2 = _tiny_model()
        emb2 = nn.Embedding(1, EMBEDDING_DIM)
        params2 = list(model2.parameters()) + list(emb2.parameters())
        opt2 = torch.optim.Adam(params2, lr=1e-3)
        ctx2 = SignalContext(
            epoch=0, total_epochs=1, rng=np.random.default_rng(0),
            device=torch.device("cpu"),
            optimizer=opt2, params=params2, grad_clip=10.0,
        )
        r2 = sig2.compute(model2, emb2, ctx2)

        # 1 patient + 2 default => 3 windows; 1 patient + 0 default => 1.
        self.assertEqual(r0.n_units, 1)
        self.assertEqual(r2.n_units, 3)

    def test_band_default_applies_only_to_default_episodes(self) -> None:
        """``trajectory_band_default`` controls the band on default-patient
        windows independently of ``trajectory_band``. With band=large for
        default and band=0 for patient: only the patient window contributes
        per-step distillation loss above the band; the default window's
        residuals are absorbed within the band, so its trajectory loss term
        is meaningfully smaller than the patient one (everything else equal).
        """
        torch.manual_seed(7)

        def run(band: float, band_default: float) -> float:
            torch.manual_seed(7)
            sig = TrajectoryRolloutSignal(
                n_patients=1, n_days=1, seed=0,
                contribution_weights={"full_body": 1.0},
                windows_per_patient=1,
                meal_window_bias=0.0, input_dropout=0.0,
                huber_delta=1.0, gut_loss_weight=0.0,
                coupling_weight=WeightSchedule(0.0),
                verifier_weight=WeightSchedule(0.0),
                trajectory_band=band,
                trajectory_band_default=band_default,
                n_default_patients=1,
            )
            model = _tiny_model()
            emb = nn.Embedding(1, EMBEDDING_DIM)
            nn.init.normal_(emb.weight, std=0.1)
            params = list(model.parameters()) + list(emb.parameters())
            opt = torch.optim.Adam(params, lr=1e-3)
            ctx = SignalContext(
                epoch=0, total_epochs=1, rng=np.random.default_rng(0),
                device=torch.device("cpu"),
                optimizer=opt, params=params, grad_clip=10.0,
            )
            return sig.compute(model, emb, ctx).avg_loss

        # Wide default band swallows residuals on the default window, so the
        # average loss must be smaller than with no default-band slack.
        loose = run(band=0.0, band_default=10.0)
        tight = run(band=0.0, band_default=0.0)
        self.assertLess(loose, tight,
                        f"loose band_default did not reduce avg loss "
                        f"(loose={loose}, tight={tight})")

    def test_default_window_only_updates_model_not_embedding(self) -> None:
        """Default-patient windows feed the zero embedding (a fresh tensor)
        through the model; only model parameters should move, not the
        learned embedding table."""
        sig = TrajectoryRolloutSignal(
            n_patients=0,
            n_days=1,
            seed=0,
            contribution_weights={"full_body": 1.0},
            windows_per_patient=1,
            meal_window_bias=0.0,
            input_dropout=0.0,
            huber_delta=1.0,
            gut_loss_weight=0.0,
            coupling_weight=WeightSchedule(0.0),
            verifier_weight=WeightSchedule(0.0),
            n_default_patients=2,
        )
        torch.manual_seed(7)
        model = _tiny_model()
        # No patient embeddings exist, but the table still must be a valid
        # parameter container; use 1 row.
        emb = nn.Embedding(1, EMBEDDING_DIM)
        nn.init.normal_(emb.weight, std=0.1)
        emb_before = emb.weight.detach().clone()
        params_before = [p.detach().clone() for p in model.parameters()]
        params = list(model.parameters()) + list(emb.parameters())
        opt = torch.optim.Adam(params, lr=1e-3)
        ctx = SignalContext(
            epoch=0, total_epochs=1, rng=np.random.default_rng(0),
            device=torch.device("cpu"),
            optimizer=opt, params=params, grad_clip=10.0,
        )
        sig.compute(model, emb, ctx)
        self.assertTrue(torch.allclose(emb.weight, emb_before))
        moved = sum(
            1 for p, before in zip(model.parameters(), params_before)
            if not torch.allclose(p, before)
        )
        self.assertGreater(moved, 0)


if __name__ == "__main__":
    unittest.main()
