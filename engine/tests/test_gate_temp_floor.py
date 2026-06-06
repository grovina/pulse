"""Regression: sigmoid-gate temperatures are floored so the ``/ exp(log_temp)``
divisor cannot collapse to 0 and blow the gradient to NaN.

Iter 76 hit this: once glycogen became a real (non-constant) trajectory
target, the previously ~ungradiented GlycogenFluxHead gates got a strong signal,
drove their temperature toward 0, and NaN'd the gradient at phase-1 epoch 31
(finite loss, NaN grad on 105/151 params). The floor caps the gate gradient at
1/MIN_GATE_TEMP while leaving the normal regime (temps ~0.2-0.5) untouched.
"""
import unittest

import torch

from pulse.modules.base import MIN_GATE_TEMP, BasalPlusGatedPeakHead, gate_temp
from pulse.modules.metabolic import GlucoseGatedInsulinHead, GlycogenFluxHead


class TestGateTempFloor(unittest.TestCase):
    def test_floor_active_only_below_min(self) -> None:
        # Normal regime (init temps exp(-1.6)..exp(-0.7) = 0.20..0.50): no change.
        for lt in (-1.6, -0.7, 0.0):
            self.assertAlmostEqual(
                gate_temp(torch.tensor(lt)).item(), float(torch.exp(torch.tensor(lt))), places=6
            )
        # Collapse regime: floored.
        self.assertAlmostEqual(gate_temp(torch.tensor(-30.0)).item(), MIN_GATE_TEMP, places=6)

    def _assert_finite_grads_under_collapse(self, head: torch.nn.Module, input_dim: int) -> None:
        # Drive every gate temperature into the pathological collapse regime.
        with torch.no_grad():
            for name, p in head.named_parameters():
                if "log" in name and "temp" in name:
                    p.fill_(-30.0)
        x = torch.randn(8, input_dim, requires_grad=True)
        prod, cons = head(x, torch.zeros(8))
        loss = (prod.pow(2) + cons.pow(2)).mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(torch.isfinite(x.grad).all().item())
        for name, p in head.named_parameters():
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all().item(), f"{name} grad not finite")

    def test_glycogen_flux_head_finite_grad_under_collapse(self) -> None:
        head = GlycogenFluxHead(
            input_dim=12, hidden_dim=16, anabolic_idx=0, catabolic_idx=1, catabolic_dir=-1.0
        )
        self._assert_finite_grads_under_collapse(head, input_dim=12)

    def test_glucose_gated_insulin_head_finite_grad_under_collapse(self) -> None:
        head = GlucoseGatedInsulinHead(input_dim=12, hidden_dim=16)
        self._assert_finite_grads_under_collapse(head, input_dim=12)

    def test_basal_plus_gated_peak_head_finite_grad_under_collapse(self) -> None:
        head = BasalPlusGatedPeakHead(
            input_dim=12, hidden_dim=16, stimulus_idx=0, gate_dir=-1, init_log_temp=-0.7
        )
        self._assert_finite_grads_under_collapse(head, input_dim=12)


if __name__ == "__main__":
    unittest.main()
