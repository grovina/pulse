"""Which findings are held out of training, and what they share with it.

PLAN Phase 0: the scoreboard estimates how well the model predicts findings it
has not seen, so the held-out set must be chosen by nobody. The unit of
holding out is the **paper**: every item of one paper (every point of its
curve, every arm of its cohort) is held out or trained together, because
findings of one paper share a cohort, a protocol and often a table, and a model
that has seen half of them has seen the rest.

``paper_is_sealed`` is the only decision point. It is a hash of the paper's
slug, reproducible and independent across papers, so its author can move it
only by renaming the paper, and every protocol family expects the same share.
Two things override the draw, both frozen lists pinned by tests so nobody
extends them quietly:

* ``GRANDFATHERED_PAPERS`` are never sealed: the warm start was fitted to them
  (every paper with a training item when sealing went by paper), or
  initialised from their numbers (Borgström 1985 sets ``bile_fat_floor``).
* ``HELD_OUT_ITEMS`` stay held out although their paper trains. They were
  sealed before sealing was by paper; unsealing them would hand the model
  findings it was never fitted to. The board reports them as held-out items,
  not as sealed papers, with what each shares with training
  (``SHARES_WITH_TRAINING``).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace
from typing import TypeVar

__all__ = [
    "GRANDFATHERED_PAPERS",
    "HELD_OUT_ITEMS",
    "SEAL_SHARE",
    "SHARES_WITH_TRAINING",
    "Overlap",
    "apply_seal",
    "paper_is_sealed",
]

SEAL_SHARE = 0.2

_SLUG = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")

GRANDFATHERED_PAPERS: tuple[str, ...] = (
    "acsm-guidelines",
    "ada-2024",
    "bergman-1979",
    "bergstrom-1967",
    "betts-2014",
    "boden-1996",
    "borgstrom-1985",
    "brooks-1985",
    "brooks-1986",
    "brunzell-1971",
    "burgess-1997",
    "cahill-1970",
    "cahill-2006",
    "cerasi-1967",
    "cherrington-1999",
    "cole-1999",
    "considine-1996",
    "coppack-1989",
    "coyle-1991",
    "cummings-2001",
    "czeisler-1999",
    "defronzo-1979",
    "diabetes-2017-66-1999",
    "digestion-1978",
    "donga-2010",
    "douglas-1982",
    "drucker-2006",
    "frayn-2003",
    "hill-2008",
    "holst-2007",
    "icp-reference-range-2022",
    "kario-2003",
    "krauchi-1994",
    "landau-1996",
    "liddle-1985",
    "lim-1996",
    "muller-1970",
    "nilsson-1973",
    "obrien-1988",
    "petersen-1999",
    "pmid-2424396",
    "pmid-3525269",
    "polonsky-1988",
    "pruessner-1997",
    "refinetti-1992",
    "rizza-1981",
    "robergs-2002",
    "romijn-1993",
    "rothman-1991",
    "schwartz-1987",
    "sinha-1996",
    "somers-1993",
    "task-force-1996",
    "taylor-1996",
    "textbook-cardiovascular-physiology",
    "textbook-pulse-oximetry",
    "trinder-2001",
    "unger-1971",
    "unger-1975",
    "unger-1981",
    "veldhuis-1990",
    "vilsboll-2003",
    "vollmer-2008",
    "wahren-1971",
    "weitzman-1971",
    "westphal-1990",
    "who-1999",
    "wolever-1991",
)

HELD_OUT_ITEMS: tuple[str, ...] = (
    "borgstrom_1985_fat_without_bile",
    "cryer_1987_epinephrine_baseline",
    "ogtt_75g_glucose_curve",
    "wahren_1971_resting_splanchnic_glucose",
    "wahren_1971_vo2_curve",
)


@dataclass(frozen=True)
class Overlap:
    """One way a held-out item is not independent of training.

    ``kind`` is ``"paper"`` (``with_`` is the paper's slug; training items
    carry it), ``"arm"`` (``with_`` names training items that run the same
    arm) or ``"init_constant"`` (``with_`` names the published numbers, as
    ``module.TABLE.key``, whose initial value the item's target equals).
    """

    kind: str
    with_: tuple[str, ...]


# Keyed by the authored item name (a curve's points share its entry). Arm
# overlaps name authored training items, a curve by its own name.
SHARES_WITH_TRAINING: dict[str, tuple[Overlap, ...]] = {
    "cryer_1987_epinephrine_baseline": (
        Overlap("paper", ("schwartz-1987",)),
        Overlap("arm", ("cryer_1987_fasting_glucose", "cryer_1987_fasting_glucose_spread")),
    ),
    "ogtt_75g_glucose_curve": (
        Overlap("paper", ("defronzo-1979",)),
    ),
    "borgstrom_1985_fat_without_bile": (
        Overlap("init_constant", ("pulse.bile.ABSORPTION.bile_fat_floor",)),
    ),
    "wahren_1971_resting_splanchnic_glucose": (
        Overlap("paper", ("wahren-1971",)),
        Overlap("arm", ("wahren_800_splanchnic_glucose", "wahren_800_arterial_glucose")),
    ),
    "wahren_1971_vo2_curve": (
        Overlap("paper", ("wahren-1971",)),
        Overlap("arm", (
            "wahren_800_splanchnic_glucose", "wahren_800_arterial_glucose", "wahren_1971_rq_curve",
        )),
        Overlap("init_constant", (
            "pulse.expenditure.GAS.vo2_rest", "pulse.expenditure.GAS.vo2_800",
        )),
    ),
}


def paper_is_sealed(paper: str) -> bool:
    """Whether every item of ``paper`` is held out of training."""
    if not _SLUG.fullmatch(paper):
        raise ValueError(f"a paper is a lowercase slug like 'schulla-2003', got {paper!r}")
    if paper in GRANDFATHERED_PAPERS:
        return False
    digest = hashlib.sha256(paper.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64) < SEAL_SHARE


_Item = TypeVar("_Item")


def apply_seal(item: _Item) -> _Item:
    """``item`` (a cohort spec or a rule) with ``sealed`` set by the registry.

    An item must name its paper, and an author may not seal it.
    """
    name, paper = item.name, item.paper  # type: ignore[attr-defined]
    if not paper:
        raise ValueError(f"{name}: names no paper")
    if item.sealed:  # type: ignore[attr-defined]
        raise ValueError(f"{name}: sealing is the registry's decision, not the author's")
    return replace(item, sealed=paper_is_sealed(paper) or name in HELD_OUT_ITEMS)  # type: ignore[type-var]
