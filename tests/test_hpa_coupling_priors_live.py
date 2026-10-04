"""The HPA coupling priors point at edges the model actually has.

Iter 98 replaced the flat HPA block with the CRH -> ACTH -> cortisol cascade. In
that cascade ``dACTH = -k_acth*(ACTH - acth_per_crh*CRH)`` and
``dCort = -k_cort*(Cort - cort_target(ACTH))``, so neither reads cortisol or
glucose directly. Three registered priors --- ``cortisol -> acth``,
``glucose -> acth`` and ``glucose -> cortisol`` --- therefore had a sensitivity of
EXACTLY zero, which makes ``coupling_band_hinge`` a constant with no gradient:
declared knowledge that could never reach a parameter. PLAN.md reroutes them to
the two edges both models implement, cortisol -| CRH (negative feedback on
``crh_target``) and glucose -| CRH (the hypoglycaemia term in ``dCRH``).

These tests pin both halves: the old targets are dead, the new ones are live
where their mechanism is active and correctly zero where it is not. The second
half is why the rerouted bands carry ``lo = 0``: both edges are ONE-SIDED, so a
positive floor would score correct physiology as a violation at every sample
outside the active regime --- for the default patient, cortisol sits below its
basal for 14 of 24 hours.
"""

from __future__ import annotations

import unittest

import torch

from pulse.coupling_prior_loss import coupling_band_hinge, normalized_sensitivity
from pulse.knowledge.coupling_priors import ALL_COUPLING_PRIORS
from pulse.model import ModularPhysiologyNetwork
from pulse.types import EMBEDDING_DIM, MARKER_INDEX, NORM_CENTER

# Teacher sensitivities in the normalized frame the loss reads, measured on
# PatientParams() (scratchpad probe, analytic and confirmed numerically):
#   cortisol -> crh: -0.0466 just above basal, -0.0265 at 18, -0.0136 at 25 ug/dL
#   glucose  -> crh: -0.125 anywhere below the 70 mg/dL threshold, 0 above
_TEACHER_CORT_CRH_AT_18 = -0.0265
_TEACHER_GLUC_CRH_BELOW = -0.125


def _perturbed_model(seed: int = 0, scale: float = 0.02) -> ModularPhysiologyNetwork:
    """A model whose zero-init output layers have been nudged off zero.

    Without this every head emits exactly 0 and a dead edge is
    indistinguishable from a live one whose gain happens to start at zero.
    """
    torch.manual_seed(seed)
    m = ModularPhysiologyNetwork()
    for p in m.parameters():
        p.data.add_(torch.randn_like(p) * scale)
    return m


def _sensitivity(model, source: str, target: str, overrides: dict[str, float],
                 eps: float = 1e-2) -> float:
    """``d(rate_target)/d(state_source)`` in normalized units, by finite difference."""
    state = torch.tensor(NORM_CENTER, dtype=torch.float32).clone()
    for marker, value in overrides.items():
        state[MARKER_INDEX[marker]] = value
    si, ti = MARKER_INDEX[source], MARKER_INDEX[target]
    emb = torch.zeros(EMBEDDING_DIM)
    base = float(state[si])

    def rate(x: float) -> torch.Tensor:
        s = state.clone()
        s[si] = x
        return model(s, emb, torch.tensor(480.0), [], gut_clock_exempt=True)[ti]

    with torch.no_grad():
        raw = (rate(base + eps) - rate(base - eps)) / (2.0 * eps)
    return float(normalized_sensitivity(raw, source, target))


class TestRetiredEdgesWereDead(unittest.TestCase):
    """The three rerouted edges really were structurally zero, so the reroute is
    not a matter of taste."""

    def setUp(self) -> None:
        self.model = _perturbed_model()

    def test_cortisol_does_not_reach_acth(self) -> None:
        s = _sensitivity(self.model, "cortisol", "acth", {"cortisol": 18.0})
        self.assertEqual(s, 0.0, "the iter-98 cascade has no cortisol term in dACTH")

    def test_glucose_does_not_reach_acth_or_cortisol(self) -> None:
        for target in ("acth", "cortisol"):
            s = _sensitivity(self.model, "glucose", target, {"glucose": 55.0})
            self.assertEqual(s, 0.0, f"dynamics of {target} do not read glucose directly")

    def test_dead_edges_are_no_longer_registered(self) -> None:
        registered = {(p.source_marker, p.target_marker) for p in ALL_COUPLING_PRIORS}
        for edge in (("cortisol", "acth"), ("glucose", "acth"), ("glucose", "cortisol")):
            self.assertNotIn(edge, registered, f"{edge} has a zero sensitivity by construction")


class TestReroutedEdgesAreLive(unittest.TestCase):
    def setUp(self) -> None:
        self.model = _perturbed_model()
        self.priors = {(p.source_marker, p.target_marker): p for p in ALL_COUPLING_PRIORS}

    def test_both_rerouted_edges_are_registered(self) -> None:
        for edge in (("cortisol", "crh"), ("glucose", "crh")):
            self.assertIn(edge, self.priors)
            self.assertEqual(self.priors[edge].sign, -1)
            self.assertEqual(self.priors[edge].magnitude_range[0], 0.0,
                             "a one-sided edge must allow a zero sensitivity")

    def test_cortisol_feeds_back_on_crh_above_basal(self) -> None:
        """Live, correctly signed, and within a factor of ~2 of the teacher's own."""
        s = _sensitivity(self.model, "cortisol", "crh", {"cortisol": 18.0})
        self.assertLess(s, 0.0, "cortisol must SUPPRESS CRH (negative feedback)")
        self.assertGreater(abs(s), 0.5 * abs(_TEACHER_CORT_CRH_AT_18))
        self.assertLess(abs(s), 2.0 * abs(_TEACHER_CORT_CRH_AT_18))
        hinge = coupling_band_hinge(torch.tensor(s), self.priors[("cortisol", "crh")])
        self.assertEqual(float(hinge), 0.0, "the teacher's own value must sit inside the band")

    def test_hypoglycaemia_reaches_crh(self) -> None:
        s = _sensitivity(self.model, "glucose", "crh", {"glucose": 55.0})
        self.assertLess(s, 0.0, "falling glucose must RAISE CRH")
        self.assertGreater(abs(s), 0.5 * abs(_TEACHER_GLUC_CRH_BELOW))
        hinge = coupling_band_hinge(torch.tensor(s), self.priors[("glucose", "crh")])
        self.assertEqual(float(hinge), 0.0, "the teacher's own value must sit inside the band")

    def test_both_edges_are_one_sided_and_the_band_tolerates_it(self) -> None:
        """Outside the active regime the true sensitivity is zero, and ``lo = 0``
        means that costs nothing. With a positive floor it would not."""
        cases = (
            ("cortisol", "crh", {"cortisol": 8.0}),      # below Cort_b: feedback is relu'd off
            ("glucose", "crh", {"glucose": 100.0}),      # above the hypoglycaemia threshold
        )
        for source, target, overrides in cases:
            s = _sensitivity(self.model, source, target, overrides)
            self.assertEqual(s, 0.0, f"{source}->{target} must be inactive here")
            hinge = coupling_band_hinge(torch.tensor(s), self.priors[(source, target)])
            self.assertEqual(float(hinge), 0.0,
                             f"an inactive {source}->{target} must not be penalised")


if __name__ == "__main__":
    unittest.main()
