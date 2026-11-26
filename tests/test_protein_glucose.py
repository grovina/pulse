"""Absorbed protein releases glucose into plasma (gluconeogenic carbon), in proportion to its rate.

Before, 50 g of protein alone took glucose from 92 to 53 mg/dL in the model (insulin and glucagon up,
hepatic output flat) where Westphal 1990 saw no change; the carbon of absorbed amino acids is a
glucose source (Jungas, Halperin, Brosnan 1992, Physiol Rev 72:419)."""
from __future__ import annotations

import torch

from pulse.modules.metabolic import (
    _AA_GLUCOSE_FRAC, _AMINO_COUPLING_IDX, _GLUCOSE_IDX, _LIPID_UNITS_PER_G,
)
from pulse.types import MODULE_MARKER_INDICES
from pulse.modules.base import compute_time_features
from tests.test_insulin_potentiator import _metabolic


def _rate(met, amino_g: float) -> float:
    n = len(MODULE_MARKER_INDICES["metabolic"])
    state = torch.zeros(1, n)
    state[0, _GLUCOSE_IDX] = 0.0
    coupling = torch.zeros(1, met.n_coupling)
    coupling[0, _AMINO_COUPLING_IDX] = amino_g * _LIPID_UNITS_PER_G
    external = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    tf = compute_time_features(torch.tensor([8.0 * 60.0]))
    emb = torch.zeros(1, met.embedding_dim)
    with torch.no_grad():
        return float(met.fluxes(state, coupling, external, emb, tf)["glucose_rate"])


def test_protein_raises_the_glucose_rate_in_proportion():
    met = _metabolic()
    base, one, two = _rate(met, 0.0), _rate(met, 1.0), _rate(met, 2.0)
    assert one > base
    # ins/glucagon drive differ with protein, but the glucose rate's own protein term is linear
    # and the fluxes that read insulin and glucagon read the state, not the rate.
    assert abs((two - base) - 2 * (one - base)) < 1e-4 * abs(one - base) + 1e-6


def test_the_fraction_is_below_the_theoretical_ceiling():
    assert 0.0 < _AA_GLUCOSE_FRAC <= 0.57          # Jungas 1992: up to ~57 % of protein carbon
