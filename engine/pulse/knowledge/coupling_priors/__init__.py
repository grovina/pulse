"""
First-class registry of coupling-edge priors harvested from medical literature.

Coupling priors used to live as a method on each ``KnowledgeContribution``
(``contrib.coupling_priors()``). That made sense when the only edges came
from the cold-model contributions themselves. Once we started harvesting
priors from physiology textbooks and review papers — sources that don't
own a trajectory generator — we needed a place to put them that didn't
require attaching a fake contribution.

Each domain file under this directory exposes
``COUPLING_PRIORS: list[CouplingPrior]`` with a header comment naming the
source and the page / section the magnitude range came from. The full
registry is assembled in ``ALL_COUPLING_PRIORS`` below; consumers should
prefer the merge helper in ``pulse.coupling_prior_loss`` which already
intersects magnitude ranges when multiple sources agree on sign.

Adding a new batch from a textbook means: drop a new file here, append it
to ``ALL_COUPLING_PRIORS``, run ``pytest -k coupling_prior``. No surgery
on contributions or training signals required.
"""

from __future__ import annotations

from ..base import CouplingPrior
from .endocrine import COUPLING_PRIORS as ENDOCRINE_PRIORS
from .metabolism import COUPLING_PRIORS as METABOLISM_PRIORS

ALL_COUPLING_PRIORS: list[CouplingPrior] = [
    *METABOLISM_PRIORS,
    *ENDOCRINE_PRIORS,
]

__all__ = [
    "ALL_COUPLING_PRIORS",
    "CouplingPrior",
    "ENDOCRINE_PRIORS",
    "METABOLISM_PRIORS",
]
