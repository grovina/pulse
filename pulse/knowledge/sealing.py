"""Which newly filed items are held out of training.

PLAN Phase 0: an item filed after the warm start is sealed at random, about one
in five, so nobody chooses which findings the scoreboard gets to see. The draw
is a hash of the item's name. It is reproducible, so re-filing an item gives the
same answer; it is independent across items, so every protocol family expects
the same share; and its author can move it only by renaming the item.

Items that were already in training when the plan was written stay in training:
the warm start has seen them, and sealing one would score the model on a finding
it was fitted to.
"""

from __future__ import annotations

import hashlib

__all__ = ["SEAL_SHARE", "sealed_at_random"]

SEAL_SHARE = 0.2


def sealed_at_random(name: str, share: float = SEAL_SHARE) -> bool:
    """The seal draw for the item called ``name``."""
    if not 0.0 <= share <= 1.0:
        raise ValueError(f"share must lie in [0, 1], got {share}")
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64) < share
