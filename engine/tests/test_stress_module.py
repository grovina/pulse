"""StressModule (iter 64 compact cascade) — structural tests.

The cascade has three state species (cortisol, acth, crh). CRH is retained
as state (so the iter-69 23-D benchmark dataset works without regeneration)
but is mechanically inert this iter — its rate comes from the SetpointHead
only, no participation in the ACTH/cortisol mechanism. The iter-64 compact
mechanism is what these tests validate:

    cortisol_rate +=  α·relu(acth_excess)·prod_scale_cortisol
                  +   δ·(1+diurnal_carrier)·prod_scale_cortisol
    acth_rate     +=  γ·(1+diurnal_carrier)·prod_scale_acth
                  −   β·relu(cortisol_excess)·prod_scale_acth
"""

from __future__ import annotations

import unittest

import torch

from pulse.modules.stress import StressModule, _ACTH_IDX, _CORTISOL_IDX, _CRH_IDX
from pulse.types import EMBEDDING_DIM, MARKER_INDEX, MODULE_MARKER_INDICES


class TestStressModuleStructure(unittest.TestCase):
    def test_crh_is_internal_marker_in_stress_module(self) -> None:
        self.assertIn("crh", MARKER_INDEX)
        stress_indices = MODULE_MARKER_INDICES["stress"]
        self.assertEqual(len(stress_indices), 3)
        self.assertIn(MARKER_INDEX["cortisol"], stress_indices)
        self.assertIn(MARKER_INDEX["acth"], stress_indices)
        self.assertIn(MARKER_INDEX["crh"], stress_indices)

    def test_module_local_index_order(self) -> None:
        """Module local indices must match the order of MODULE_MARKER_INDICES['stress'].

        MODULE_MARKER_INDICES is built by enumerating MARKERS in order, so
        cortisol (10) and acth (11) come before crh (22) — local indices
        0, 1, 2 respectively.
        """
        self.assertEqual(_CORTISOL_IDX, 0)
        self.assertEqual(_ACTH_IDX, 1)
        self.assertEqual(_CRH_IDX, 2)

    def test_module_accepts_3_species_state(self) -> None:
        torch.manual_seed(0)
        module = StressModule(embedding_dim=EMBEDDING_DIM)
        state = torch.tensor([[12.0, 30.0, 100.0]])  # typical cortisol, acth, crh
        coupling = torch.zeros(1, 2)
        external = torch.zeros(1, 2)
        embedding = torch.zeros(1, EMBEDDING_DIM)
        # time_features = [linear_time, sin(2π·hr/24), cos(2π·hr/24)]
        time_features = torch.tensor([[8 / 24.0, 0.0, 1.0]])
        rate = module(state, coupling, external, embedding, time_features)
        self.assertEqual(rate.shape, (1, 3))


class TestCascadeMechanism(unittest.TestCase):
    """At typical baseline + zero embedding, α·acth_excess = 0 and β·cortisol_excess
    = 0, so the mechanism fires only when the relevant species exceeds typical.
    Drive a single stage at a time and confirm the downstream species *responds*
    (gets a non-zero, correct-sign rate adjustment on top of the SetpointHead)."""

    def _module(self) -> StressModule:
        torch.manual_seed(0)
        m = StressModule(embedding_dim=EMBEDDING_DIM)
        # Bias the mechanism parameters above the init floor (softplus(0)
        # ≈ 0.69 vs init softplus(-3) ≈ 0.05) so the mechanism signal is
        # measurable above the SetpointHead residual at random init.
        for p in (m._alpha_raw, m._delta_raw, m._gamma_raw, m._beta_raw):
            with torch.no_grad():
                p.fill_(0.0)
        return m

    def _zero_inputs(self):
        coupling = torch.zeros(1, 2)
        external = torch.zeros(1, 2)
        embedding = torch.zeros(1, EMBEDDING_DIM)
        # Pick a phase where diurnal carrier is balanced (sin=0, cos=1)
        # so we isolate the cascade contributions from the diurnal drive.
        time_features = torch.tensor([[8 / 24.0, 0.0, 1.0]])
        return coupling, external, embedding, time_features

    def test_acth_above_typical_raises_cortisol_rate(self) -> None:
        module = self._module()
        coupling, external, embedding, time_features = self._zero_inputs()
        elevated = torch.tensor([[12.0, 45.0, 100.0]])  # ACTH +50%
        baseline = torch.tensor([[12.0, 30.0, 100.0]])
        with torch.no_grad():
            rate_elevated = module(elevated, coupling, external, embedding, time_features)
            rate_baseline = module(baseline, coupling, external, embedding, time_features)
        self.assertGreater(
            float(rate_elevated[0, _CORTISOL_IDX].item()),
            float(rate_baseline[0, _CORTISOL_IDX].item()),
        )

    def test_cortisol_above_typical_lowers_acth_rate(self) -> None:
        """Iter-64 compact: β feedback hits ACTH directly (not CRH)."""
        module = self._module()
        coupling, external, embedding, time_features = self._zero_inputs()
        elevated = torch.tensor([[18.0, 30.0, 100.0]])  # cortisol +50%
        baseline = torch.tensor([[12.0, 30.0, 100.0]])
        with torch.no_grad():
            rate_elevated = module(elevated, coupling, external, embedding, time_features)
            rate_baseline = module(baseline, coupling, external, embedding, time_features)
        self.assertLess(
            float(rate_elevated[0, _ACTH_IDX].item()),
            float(rate_baseline[0, _ACTH_IDX].item()),
        )

    def test_diurnal_carrier_drives_acth(self) -> None:
        """Iter-64 compact: γ·(1+diurnal_carrier) drives ACTH directly.

        The diurnal carrier ranges over [0, 2] so morning (carrier high) vs
        evening (carrier low) must produce different ACTH rates.
        """
        module = self._module()
        coupling = torch.zeros(1, 2)
        external = torch.zeros(1, 2)
        embedding = torch.zeros(1, EMBEDDING_DIM)
        state = torch.tensor([[12.0, 30.0, 100.0]])
        # With phase_proj ≈ 0 at zero embedding, diurnal = sin(t).
        # t=06:00 → sin(π/2)=+1 → carrier=2; t=18:00 → sin(3π/2)=−1 → carrier=0.
        morning = torch.tensor([[6 / 24.0, 1.0, 0.0]])
        evening = torch.tensor([[18 / 24.0, -1.0, 0.0]])
        with torch.no_grad():
            rate_morning = module(state, coupling, external, embedding, morning)
            rate_evening = module(state, coupling, external, embedding, evening)
        self.assertGreater(
            float(rate_morning[0, _ACTH_IDX].item()),
            float(rate_evening[0, _ACTH_IDX].item()),
        )

    def test_crh_state_does_not_affect_acth_or_cortisol(self) -> None:
        """CRH is mechanically inert this iter — varying CRH state must not
        change ACTH or cortisol rates. Confirms the iter-69 cascade is fully
        decoupled, not just dampened."""
        module = self._module()
        coupling, external, embedding, time_features = self._zero_inputs()
        crh_low = torch.tensor([[12.0, 30.0, 50.0]])
        crh_high = torch.tensor([[12.0, 30.0, 200.0]])
        with torch.no_grad():
            rate_low = module(crh_low, coupling, external, embedding, time_features)
            rate_high = module(crh_high, coupling, external, embedding, time_features)
        self.assertAlmostEqual(
            float(rate_low[0, _CORTISOL_IDX].item()),
            float(rate_high[0, _CORTISOL_IDX].item()),
            places=5,
        )
        self.assertAlmostEqual(
            float(rate_low[0, _ACTH_IDX].item()),
            float(rate_high[0, _ACTH_IDX].item()),
            places=5,
        )


if __name__ == "__main__":
    unittest.main()
