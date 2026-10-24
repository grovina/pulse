"""A learned deformation of a default functional form.

PLAN section 2: mass action, a Hill curve and the one-compartment stomach are
defaults, not constraints. Each carries a correction that starts at zero, so a
wrong shape can still be reached. ``shrinkage_of`` is the penalty on those
corrections. A physiological rate is a different parameter and is not in it.
"""

from __future__ import annotations

import torch
from torch import nn

# exp(40) is finite in float32 and far past any shape the evidence can ask for.
_EXCESS = 40.0


class Correction(nn.Module):
    """One zero-initialised correction. ``forward`` is the term itself."""

    def __init__(self, n: int = 1) -> None:
        super().__init__()
        self.term = nn.Parameter(torch.zeros(n))

    def forward(self) -> torch.Tensor:
        return self.term

    def shrinkage(self) -> torch.Tensor:
        """Squared size of this term. Zero at init."""
        return self.term.pow(2).sum()


def shrinkage_of(module: nn.Module) -> torch.Tensor:
    """The shrinkage on every correction under ``module``.

    Physiological parameters are not terms here, so the penalty does not pull
    them back toward the value they were initialised at.
    """
    penalties = [m.shrinkage() for m in module.modules() if isinstance(m, Correction)]
    if not penalties:
        return torch.zeros(())
    return torch.stack([p.reshape(()) for p in penalties]).sum()


def correction_weights(correction: torch.Tensor | None, *, n: int, like: torch.Tensor) -> torch.Tensor:
    """Positive weights of length ``n``, exactly one when the correction is absent or zero."""
    if correction is None:
        return like.new_ones(n)
    return torch.exp(correction.clamp(-_EXCESS, _EXCESS))


def apply_correction(base: torch.Tensor, correction: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
    """``base`` times a factor that is exactly one at a zero correction and at a zero feature.

    ``feature`` is the shape coordinate. It is chosen to vanish where the
    physiological rate is defined, so a correction cannot move that rate.
    Clipping the exponent keeps the factor finite; neither identity sits on the clip.
    """
    excess = (correction * feature).clamp(-_EXCESS, _EXCESS)
    return base * torch.exp(excess)


def corrected_mass_action(
    rate: torch.Tensor,
    concentration: torch.Tensor,
    anchor: torch.Tensor | float,
    correction: torch.Tensor | None = None,
) -> torch.Tensor:
    """First-order consumption ``rate * concentration``.

    A correction changes the order about ``anchor``: the product at the anchor
    is the plain product for every correction, and a zero correction is the
    plain product at every concentration. The rate is not stored here.
    """
    base = rate * concentration
    if correction is None:
        return base
    if isinstance(anchor, torch.Tensor):
        anchor_t = anchor.to(dtype=concentration.dtype, device=concentration.device)
    else:
        anchor_t = torch.as_tensor(anchor, dtype=concentration.dtype, device=concentration.device)
    feature = torch.log(concentration.clamp(min=1e-12)) - torch.log(anchor_t.clamp(min=1e-12))
    return apply_correction(base, correction, feature)
