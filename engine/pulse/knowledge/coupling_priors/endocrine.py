"""
Coupling priors for stress / appetite / autonomic axes that do not live in
fuel metabolism but cross-talk with it.

Sources:
  - Guyton & Hall, *Textbook of Medical Physiology*, 14th ed., chapters
    77 (adrenocortical hormones) and 56 (autonomic nervous system).
  - Boron & Boulpaep, *Medical Physiology*, 3rd ed., chapter 50 (the
    hypothalamus and pituitary) and chapter 58 (the adrenal medulla).
  - Cummings & Overduin (2007), *Gastrointestinal regulation of food
    intake*, J Clin Invest 117(1):13–23 — ghrelin and the incretin
    family on appetite.
"""

from __future__ import annotations

from ..base import CouplingPrior

def _p(src: str, tgt: str, sign: int, mag: tuple[float, float]) -> CouplingPrior:
    return CouplingPrior(source_marker=src, target_marker=tgt, sign=sign, magnitude_range=mag)


COUPLING_PRIORS: list[CouplingPrior] = [
    _p("acth", "cortisol", +1, (0.002, 0.02)),
    _p("cortisol", "acth", -1, (0.005, 0.05)),
    _p("cortisol", "hr", +1, (0.05, 0.5)),
    _p("cortisol", "sbp", +1, (0.05, 0.4)),
    _p("cortisol", "dbp", +1, (0.02, 0.2)),
    _p("cortisol", "hrv", -1, (0.05, 0.5)),
    _p("insulin", "ghrelin", -1, (0.005, 0.05)),
    _p("glp1", "ghrelin", -1, (0.001, 0.02)),
    _p("leptin", "ghrelin", -1, (0.0005, 0.01)),
    _p("lactate", "rr", +1, (0.05, 0.6)),
    _p("temp", "hr", +1, (0.05, 0.5)),
    _p("temp", "rr", +1, (0.02, 0.3)),
    _p("glucose", "cortisol", -1, (0.001, 0.015)),
    _p("glucose", "acth", -1, (0.001, 0.02)),
]
