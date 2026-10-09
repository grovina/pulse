"""Measurement models for observations that are not the ODE state.

A glucose meter reads interstitial glucose. Interstitial glucose follows plasma
with a delay of several minutes (Rebrin & Steil; the usual CGM lag is about
5–10 minutes). The delay is a fixed time constant, not a fitted network.
"""

from __future__ import annotations

import math

import torch

# Middle of the usual 5–10 minute interstitial delay.
INTERSTITIAL_LAG_MIN = 8.0


def interstitial_glucose(
    plasma: torch.Tensor,
    tau_min: float = INTERSTITIAL_LAG_MIN,
    dt: float = 1.0,
) -> torch.Tensor:
    """First-order lag of a plasma-glucose series.

    ``plasma`` is glucose at each minute, ``[T]``. The interstitial value starts
    equilibrated with plasma (a person who has been sitting at that glucose) and
    then relaxes toward plasma with time constant ``tau_min``:

        Gi(t) = a·Gi(t−1) + (1−a)·Gp(t),   a = exp(−dt / tau)

    ``tau_min <= 0`` returns plasma unchanged.
    """
    if tau_min <= 0.0 or plasma.shape[0] <= 1:
        return plasma
    a = math.exp(-float(dt) / float(tau_min))
    one_m_a = 1.0 - a
    gi = plasma[0]
    pieces = [gi]
    for t in range(1, int(plasma.shape[0])):
        gi = a * gi + one_m_a * plasma[t]
        pieces.append(gi)
    return torch.stack(pieces)
