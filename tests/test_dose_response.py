"""Unit tests for the dose-response training signal.

Covers:

* protocol validation (rejects bad parameter combinations)
* OLS slope identity on a perfect line
* slope gradient flows back through model parameters and the embedding
* zero loss at exactly the literature target slope
* end-to-end signal: zero weight → no-op; positive weight → step occurs and
  changes the embedding
"""

from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn as nn

from pulse.dose_response import (
    DoseResponseProtocol,
    _ols_slope,
    cold_initial_state,
    dose_response_epoch_loss,
    predicted_slopes_batched,
)
from pulse.model import ModularPhysiologyNetwork
from pulse.training import DoseResponseSignal, SignalContext, WeightSchedule
from pulse.types import EMBEDDING_DIM, NORM_CENTER


def _tiny_model() -> ModularPhysiologyNetwork:
    return ModularPhysiologyNetwork(
        metabolic_hidden=16, appetite_hidden=12, stress_hidden=12,
        cardiovascular_hidden=16, thermoreg_hidden=8, respiratory_hidden=8,
    )


def _initial_state(device: torch.device) -> torch.Tensor:
    return torch.tensor(NORM_CENTER, dtype=torch.float32, device=device)


class TestProtocolValidation(unittest.TestCase):
    def test_rejects_too_few_doses(self) -> None:
        with self.assertRaises(ValueError):
            DoseResponseProtocol(carb_doses_g=(50.0, 100.0))

    def test_rejects_bad_window(self) -> None:
        with self.assertRaises(ValueError):
            DoseResponseProtocol(meal_offset_min=10, pre_window=15)
        with self.assertRaises(ValueError):
            DoseResponseProtocol(meal_offset_min=30, post_window=300, duration_min=240)

    def test_default_protocol_matches_benchmark(self) -> None:
        """Defaults must mirror the meal_dose_response benchmark scenario."""
        proto = DoseResponseProtocol()
        self.assertEqual(proto.carb_doses_g, (30.0, 60.0, 90.0))
        self.assertEqual(proto.fats_g, 5.0)
        self.assertEqual(proto.proteins_g, 10.0)
        self.assertEqual(proto.meal_offset_min, 30)
        self.assertEqual(proto.duration_min, 240)
        self.assertEqual(proto.start_hour, 8.0)


class TestColdInitialState(unittest.TestCase):
    def test_returns_state_dim_tensor(self) -> None:
        from pulse.types import STATE_DIM
        state = cold_initial_state(DoseResponseProtocol(), device="cpu")
        self.assertEqual(state.shape, (STATE_DIM,))
        self.assertEqual(state.dtype, torch.float32)

    def test_rejects_zero_sigma(self) -> None:
        with self.assertRaises(ValueError):
            DoseResponseProtocol(sigma_slope=0.0)


class TestOLSSlopeIdentity(unittest.TestCase):
    def test_recovers_perfect_line(self) -> None:
        x = torch.tensor([25.0, 50.0, 75.0, 100.0])
        y = 0.7 * x + 5.0
        slope = _ols_slope(x, y)
        self.assertAlmostEqual(float(slope), 0.7, places=5)

    def test_constant_y_yields_zero_slope(self) -> None:
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        y = torch.ones_like(x) * 3.0
        self.assertAlmostEqual(float(_ols_slope(x, y)), 0.0, places=5)


class TestSlopeGradient(unittest.TestCase):
    def test_gradient_flows_to_model_and_embedding(self) -> None:
        device = torch.device("cpu")
        torch.manual_seed(0)
        model = _tiny_model()
        emb = nn.Embedding(1, EMBEDDING_DIM)
        proto = DoseResponseProtocol()
        slopes = predicted_slopes_batched(
            model, emb(torch.tensor([0])), proto, _initial_state(device),
        )
        self.assertEqual(slopes.shape, (1,))
        slopes.sum().backward()

        # At least one model parameter and the embedding row should have grad.
        any_model_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in model.parameters()
        )
        emb_grad = emb.weight.grad
        self.assertTrue(any_model_grad, "no gradient reached model parameters")
        self.assertIsNotNone(emb_grad)
        assert emb_grad is not None
        self.assertGreater(float(emb_grad.abs().sum()), 0.0)


class TestEpochLossAgainstTarget(unittest.TestCase):
    def test_zero_loss_when_predicted_equals_target(self) -> None:
        """If we shift target_slope to match the model's prediction, z² → 0."""
        device = torch.device("cpu")
        torch.manual_seed(0)
        model = _tiny_model()
        emb = nn.Embedding(1, EMBEDDING_DIM)
        proto = DoseResponseProtocol()

        with torch.no_grad():
            predicted = float(
                predicted_slopes_batched(
                    model, emb(torch.tensor([0])), proto, _initial_state(device),
                )[0]
            )
        matched = DoseResponseProtocol(target_slope=predicted, sigma_slope=proto.sigma_slope)
        loss, diagnostics = dose_response_epoch_loss(
            model, [emb(torch.tensor(0))], matched, _initial_state(device), device,
        )
        self.assertAlmostEqual(float(loss.item()), 0.0, places=5)
        self.assertAlmostEqual(diagnostics["predicted_slope"], predicted, places=5)

    def test_positive_loss_when_off_target(self) -> None:
        device = torch.device("cpu")
        torch.manual_seed(1)
        model = _tiny_model()
        emb = nn.Embedding(1, EMBEDDING_DIM)
        loss, _ = dose_response_epoch_loss(
            model, [emb(torch.tensor(0))], DoseResponseProtocol(), _initial_state(device), device,
        )
        self.assertGreater(float(loss.item()), 0.0)

    def test_empty_embedding_list_returns_zero(self) -> None:
        device = torch.device("cpu")
        model = _tiny_model()
        loss, diagnostics = dose_response_epoch_loss(
            model, [], DoseResponseProtocol(), _initial_state(device), device,
        )
        self.assertEqual(float(loss.item()), 0.0)
        self.assertEqual(diagnostics["predicted_slope"], 0.0)


class TestDoseResponseSignalGating(unittest.TestCase):
    def test_zero_weight_is_noop(self) -> None:
        device = torch.device("cpu")
        model = _tiny_model()
        emb = nn.Embedding(2, EMBEDDING_DIM)
        params = list(model.parameters()) + list(emb.parameters())
        opt = torch.optim.Adam(params, lr=1e-3)
        sig = DoseResponseSignal(
            n_patients=2, sample_patients=2, weight=WeightSchedule(0.0),
        )
        ctx = SignalContext(
            epoch=0, total_epochs=1, rng=np.random.default_rng(0),
            device=device, optimizer=opt, params=params, grad_clip=10.0,
        )
        result = sig.compute(model, emb, ctx)
        self.assertEqual(result.n_units, 0)

    def test_positive_weight_changes_embedding(self) -> None:
        device = torch.device("cpu")
        torch.manual_seed(2)
        model = _tiny_model()
        emb = nn.Embedding(2, EMBEDDING_DIM)
        nn.init.normal_(emb.weight, std=0.1)
        params = list(model.parameters()) + list(emb.parameters())
        opt = torch.optim.Adam(params, lr=1e-2)
        sig = DoseResponseSignal(
            n_patients=2, sample_patients=2, weight=WeightSchedule(0.5),
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
        self.assertIn("predicted_slope", result.sub_metrics)
        self.assertIn("glucose_loss", result.sub_metrics)

    def test_default_embedding_supervised_when_no_patients(self) -> None:
        """With n_patients=0 and include_default_embedding=True, the zero embedding
        still receives gradient — model parameters move."""
        device = torch.device("cpu")
        torch.manual_seed(3)
        model = _tiny_model()
        emb = nn.Embedding(1, EMBEDDING_DIM)
        params = list(model.parameters()) + list(emb.parameters())
        opt = torch.optim.Adam(params, lr=1e-2)
        sig = DoseResponseSignal(
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
        sig = DoseResponseSignal(
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


class TestMultiMarkerDoseResponse(unittest.TestCase):
    """Iter 69: dose-response axis extended to glucose + insulin + GLP-1.

    Validates the new ``MarkerDoseTarget`` plumbing, both modes (slope, rank),
    and the composite loss aggregation.
    """

    def test_marker_target_rejects_unknown_marker(self) -> None:
        from pulse.dose_response import MarkerDoseTarget
        with self.assertRaises(ValueError):
            MarkerDoseTarget(marker="bogus", mode="rank", rank_margin=0.5)

    def test_marker_target_rejects_bad_mode(self) -> None:
        from pulse.dose_response import MarkerDoseTarget
        with self.assertRaises(ValueError):
            MarkerDoseTarget(marker="glucose", mode="oops")

    def test_slope_mode_requires_target_and_sigma(self) -> None:
        from pulse.dose_response import MarkerDoseTarget
        with self.assertRaises(ValueError):
            MarkerDoseTarget(marker="glucose", mode="slope")

    def test_legacy_empty_targets_matches_glucose_slope(self) -> None:
        """Empty ``marker_targets`` must behave exactly like the iter-32-era
        single-glucose-slope protocol."""
        device = torch.device("cpu")
        torch.manual_seed(11)
        model = _tiny_model()
        emb = nn.Embedding(1, EMBEDDING_DIM)
        legacy = DoseResponseProtocol()
        loss_legacy, diag_legacy = dose_response_epoch_loss(
            model, [emb(torch.tensor(0))], legacy, _initial_state(device), device,
        )
        from pulse.dose_response import MarkerDoseTarget
        explicit = DoseResponseProtocol(
            marker_targets=(MarkerDoseTarget(
                marker="glucose", mode="slope",
                target_slope=legacy.target_slope, sigma_slope=legacy.sigma_slope,
            ),),
        )
        loss_explicit, diag_explicit = dose_response_epoch_loss(
            model, [emb(torch.tensor(0))], explicit, _initial_state(device), device,
        )
        self.assertAlmostEqual(float(loss_legacy.item()), float(loss_explicit.item()), places=5)
        self.assertAlmostEqual(
            diag_legacy["predicted_slope"], diag_explicit["predicted_slope"], places=5,
        )

    def test_multi_marker_composite_aggregates_all_legs(self) -> None:
        from pulse.dose_response import MarkerDoseTarget
        device = torch.device("cpu")
        torch.manual_seed(7)
        model = _tiny_model()
        emb = nn.Embedding(2, EMBEDDING_DIM)
        proto = DoseResponseProtocol(
            marker_targets=(
                MarkerDoseTarget(marker="glucose", mode="slope",
                                 target_slope=0.7, sigma_slope=0.25, weight=1.0),
                MarkerDoseTarget(marker="insulin", mode="rank", rank_margin=0.5, weight=1.0),
                MarkerDoseTarget(marker="glp1", mode="rank", rank_margin=0.3, weight=1.0),
            ),
        )
        loss, diag = dose_response_epoch_loss(
            model, [emb(torch.tensor(0)), emb(torch.tensor(1))],
            proto, _initial_state(device), device,
        )
        # All three markers must produce diagnostic loss entries.
        self.assertIn("glucose_loss", diag)
        self.assertIn("insulin_loss", diag)
        self.assertIn("glp1_loss", diag)
        # Composite is convex combination (sum of weights = 3, each scaled 1/3).
        composite = (diag["glucose_loss"] + diag["insulin_loss"] + diag["glp1_loss"]) / 3.0
        self.assertAlmostEqual(float(loss.item()), composite, places=4)
        # predicted_slope (legacy log key) must surface the glucose leg.
        self.assertIn("predicted_slope", diag)
        self.assertAlmostEqual(diag["predicted_slope"], diag["glucose_slope"], places=6)

    def test_rank_loss_zero_when_monotone_with_margin(self) -> None:
        from pulse.dose_response import _rank_loss
        # Peaks rise by exactly the margin → hinge slack is zero → loss zero.
        margin = 0.5
        peaks = torch.tensor([[1.0, 1.5, 2.0]])
        loss, _ = _rank_loss(peaks, margin)
        self.assertAlmostEqual(float(loss.item()), 0.0, places=6)

    def test_rank_loss_positive_when_inverted(self) -> None:
        from pulse.dose_response import _rank_loss
        # Inverted monotonicity → both adjacent diffs violate by 1+margin.
        margin = 0.5
        peaks = torch.tensor([[2.0, 1.5, 1.0]])
        loss, _ = _rank_loss(peaks, margin)
        # Per-pair shortfall: margin − (−0.5) = 1.0, squared = 1.0; mean over 2 pairs = 1.0.
        self.assertAlmostEqual(float(loss.item()), 1.0, places=6)

    def test_peak_loss_zero_on_literature_line(self) -> None:
        from pulse.dose_response import _peak_loss
        doses = torch.tensor([30.0, 60.0, 90.0])
        # Peaks exactly on target_per_g · dose with target 0.7 → zero loss.
        peaks = torch.tensor([[21.0, 42.0, 63.0]])
        loss, per_g = _peak_loss(peaks, doses, target_per_g=0.7, sigma_per_g=0.25)
        self.assertAlmostEqual(float(loss.item()), 0.0, places=6)
        self.assertAlmostEqual(float(per_g.mean().item()), 0.7, places=6)

    def test_peak_loss_penalises_uniformly_low_peaks_where_slope_would_not(self) -> None:
        """The crux of the iter-73 fix: peaks with the correct *slope* but a
        too-low absolute level pass _slope_loss yet must be penalised by
        _peak_loss."""
        from pulse.dose_response import _peak_loss, _slope_loss
        doses = torch.tensor([30.0, 60.0, 90.0])
        # Correct slope (0.7/g) but shifted down by 15 mg/dL everywhere.
        peaks = torch.tensor([[21.0 - 15.0, 42.0 - 15.0, 63.0 - 15.0]])
        slope_loss, slopes = _slope_loss(peaks, doses, target=0.7, sigma=0.25)
        peak_loss, _ = _peak_loss(peaks, doses, target_per_g=0.7, sigma_per_g=0.25)
        # Slope is still 0.7 → slope loss ≈ 0 (offset-invariant).
        self.assertAlmostEqual(float(slopes.mean().item()), 0.7, places=5)
        self.assertAlmostEqual(float(slope_loss.item()), 0.0, places=5)
        # Peak loss is clearly positive — the absolute shortfall is supervised.
        self.assertGreater(float(peak_loss.item()), 0.1)

    def test_peak_mode_requires_target_and_sigma(self) -> None:
        from pulse.dose_response import MarkerDoseTarget
        with self.assertRaises(ValueError):
            MarkerDoseTarget(marker="glucose", mode="peak")


class TestDoseResponseMarkerCLIParser(unittest.TestCase):
    def test_empty_returns_empty(self) -> None:
        from pulse.train import _parse_dose_response_markers
        self.assertEqual(_parse_dose_response_markers(""), ())
        self.assertEqual(_parse_dose_response_markers("   "), ())

    def test_parses_slope_and_rank_with_weights(self) -> None:
        from pulse.train import _parse_dose_response_markers
        targets = _parse_dose_response_markers(
            "glucose:slope:0.7:0.25;insulin:rank:0.5;glp1:rank:0.3:2.0",
        )
        self.assertEqual(len(targets), 3)
        self.assertEqual(targets[0].marker, "glucose")
        self.assertEqual(targets[0].mode, "slope")
        self.assertAlmostEqual(targets[0].target_slope, 0.7)
        self.assertAlmostEqual(targets[0].sigma_slope, 0.25)
        self.assertEqual(targets[1].marker, "insulin")
        self.assertEqual(targets[1].mode, "rank")
        self.assertAlmostEqual(targets[1].rank_margin, 0.5)
        self.assertAlmostEqual(targets[2].weight, 2.0)

    def test_parses_peak_mode(self) -> None:
        from pulse.train import _parse_dose_response_markers
        targets = _parse_dose_response_markers(
            "glucose:peak:0.7:0.25;insulin:peak:0.4:0.3:2.0;glp1:rank:0.3",
        )
        self.assertEqual(len(targets), 3)
        self.assertEqual(targets[0].mode, "peak")
        self.assertAlmostEqual(targets[0].target_slope, 0.7)
        self.assertAlmostEqual(targets[0].sigma_slope, 0.25)
        self.assertEqual(targets[1].mode, "peak")
        self.assertAlmostEqual(targets[1].weight, 2.0)
        self.assertEqual(targets[2].mode, "rank")

    def test_parser_rejects_unknown_mode(self) -> None:
        from pulse.train import _parse_dose_response_markers
        with self.assertRaises(ValueError):
            _parse_dose_response_markers("glucose:wrong:0.7:0.25")


if __name__ == "__main__":
    unittest.main()
