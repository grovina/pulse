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

from pulse.modules.base import MassActionModule
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
    """Drive a single stage at a time and confirm the downstream species *responds*
    (gets a correct-sign rate adjustment on top of the mass-action base).

    Iter 96: the ACTH→cortisol drive is now PROPORTIONAL to ACTH rather than
    rectified at typical, and `_delta_raw` (a second, independent circadian on
    cortisol) is gone — cortisol follows its secretagogue and nothing else. The
    β feedback on ACTH is still rectified at typical."""

    def _module(self) -> StressModule:
        torch.manual_seed(0)
        m = StressModule(embedding_dim=EMBEDDING_DIM)
        # Bias the mechanism parameters above the init floor (softplus(0)
        # ≈ 0.69 vs init softplus(-3) ≈ 0.05) so the mechanism signal is
        # measurable above the SetpointHead residual at random init.
        for p in (m._alpha_raw, m._gamma_raw, m._beta_raw):
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

    def test_crh_carries_no_term_in_the_iter64_mechanism(self) -> None:
        """The iter-64 cascade mechanism contains no CRH term.

        Iter 95: this previously asserted that the module's TOTAL rate was
        CRH-independent. That was an initialization artifact, not a structural
        property — `SetpointHead` zero-initialized its final layer, so the head's
        output ignored its inputs at init and would have picked up a CRH dependence
        as soon as those weights moved. With `SetpointHead` removed (see
        modules/base.py) the head is a plain `SpeciesHead`, which reads the whole
        module state including CRH from step 0.

        That is not a regression: CRH -> ACTH -> cortisol is the real cascade, so a
        learned dependence there is physiologically correct. What the iter-64 design
        actually claims is narrower — that the explicit MECHANISM terms
        (alpha/beta/gamma/delta) route no CRH — and that is what is asserted here, by
        isolating the adjustment the subclass adds on top of the mass-action base.
        Unlike the old assertion, this one also holds after training.
        """
        module = self._module()
        coupling, external, embedding, time_features = self._zero_inputs()
        crh_low = torch.tensor([[12.0, 30.0, 50.0]])
        crh_high = torch.tensor([[12.0, 30.0, 200.0]])
        with torch.no_grad():
            args_low = (crh_low, coupling, external, embedding, time_features)
            args_high = (crh_high, coupling, external, embedding, time_features)
            mech_low = module(*args_low) - MassActionModule.forward(module, *args_low)
            mech_high = module(*args_high) - MassActionModule.forward(module, *args_high)
        for idx, name in ((_CORTISOL_IDX, "cortisol"), (_ACTH_IDX, "acth")):
            self.assertAlmostEqual(
                float(mech_low[0, idx].item()), float(mech_high[0, idx].item()),
                places=6, msg=f"iter-64 mechanism routes CRH into {name}",
            )


    def test_cortisol_mechanism_carries_no_independent_circadian(self) -> None:
        """Iter 96: cortisol is driven by ACTH alone — no second circadian.

        Through iter 95 the module added `δ·(1+diurnal)·prod_scale` to cortisol on
        top of the ACTH drive, duplicating a rhythm ACTH already carries and — since
        the carrier is in [0,2] and δ = softplus(·) ≥ 0 — laying a strictly
        non-negative production floor under it. That floor is why the student's
        overnight cortisol nadir sat at 10-12 µg/dL against the teacher's 4.4. The
        teacher stopped doing exactly this in iter 91; this asserts the student
        does not do it either, at ANY phase, holding ACTH fixed.
        """
        module = self._module()
        coupling = torch.zeros(1, 2)
        external = torch.zeros(1, 2)
        embedding = torch.zeros(1, EMBEDDING_DIM)
        state = torch.tensor([[12.0, 30.0, 100.0]])
        morning = torch.tensor([[6 / 24.0, 1.0, 0.0]])    # carrier = 2
        evening = torch.tensor([[18 / 24.0, -1.0, 0.0]])  # carrier = 0
        with torch.no_grad():
            args_m = (state, coupling, external, embedding, morning)
            args_e = (state, coupling, external, embedding, evening)
            mech_m = module(*args_m) - MassActionModule.forward(module, *args_m)
            mech_e = module(*args_e) - MassActionModule.forward(module, *args_e)
        self.assertAlmostEqual(
            float(mech_m[0, _CORTISOL_IDX].item()),
            float(mech_e[0, _CORTISOL_IDX].item()),
            places=6,
            msg="cortisol mechanism still carries a circadian term of its own",
        )
        # ...while ACTH's mechanism DOES move with phase (it owns the rhythm).
        self.assertGreater(
            float(mech_m[0, _ACTH_IDX].item()), float(mech_e[0, _ACTH_IDX].item()),
        )

    def test_cortisol_drive_is_proportional_below_typical_acth(self) -> None:
        """Iter 96: the ACTH→cortisol map is not flat below typical ACTH.

        The old `relu((ACTH − typical)/typical)` made cortisol unable to tell an
        ACTH of 5 from an ACTH of 12 — precisely the overnight range its nadir
        lives in, and a direct contributor to a nadir that never fell.
        """
        module = self._module()
        coupling, external, embedding, time_features = self._zero_inputs()
        low = torch.tensor([[12.0, 8.0, 100.0]])
        lower = torch.tensor([[12.0, 4.0, 100.0]])
        with torch.no_grad():
            args_l = (low, coupling, external, embedding, time_features)
            args_ll = (lower, coupling, external, embedding, time_features)
            mech_l = module(*args_l) - MassActionModule.forward(module, *args_l)
            mech_ll = module(*args_ll) - MassActionModule.forward(module, *args_ll)
        self.assertGreater(
            float(mech_l[0, _CORTISOL_IDX].item()),
            float(mech_ll[0, _CORTISOL_IDX].item()),
        )


if __name__ == "__main__":
    unittest.main()
