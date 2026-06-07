"""
Coupling priors for fuel metabolism (glucose, insulin, glucagon, FFA, BHB,
hepatic glucose output).

Sign conventions and magnitude ranges are summarized from standard
endocrinology references. Magnitudes are loose order-of-magnitude bounds
used by the soft-hinge prior loss; intent is to declare *direction* with
a permissive band, not pin a specific rate constant.

Sources:
  - Guyton & Hall, *Textbook of Medical Physiology*, 14th ed., chapter 79
    (insulin, glucagon, and diabetes mellitus).
  - Boron & Boulpaep, *Medical Physiology*, 3rd ed., chapter 51
    (the endocrine pancreas) and chapter 58 (the adrenal medulla / HPA).
  - Frayn, *Metabolic Regulation*, 4th ed., chapters 6–8 (integrated
    fuel metabolism in the fed and fasted state).
"""

from __future__ import annotations

from ..base import CouplingPrior

def _p(src: str, tgt: str, sign: int, mag: tuple[float, float]) -> CouplingPrior:
    return CouplingPrior(source_marker=src, target_marker=tgt, sign=sign, magnitude_range=mag)


COUPLING_PRIORS: list[CouplingPrior] = [
    _p("glucose", "insulin", +1, (0.001, 0.02)),
    _p("insulin", "glucose", -1, (0.0001, 0.002)),
    _p("insulin", "hepatic_output", -1, (0.02, 0.15)),
    _p("glucose", "hepatic_output", -1, (0.005, 0.05)),
    _p("glucose", "glucagon", -1, (0.001, 0.02)),
    _p("insulin", "glucagon", -1, (0.001, 0.02)),
    _p("glucagon", "glucose", +1, (0.001, 0.02)),
    _p("glucagon", "hepatic_output", +1, (0.005, 0.05)),
    _p("insulin", "ffa", -1, (0.005, 0.06)),
    _p("glucagon", "ffa", +1, (0.001, 0.02)),
    _p("ffa", "bhb", +1, (0.001, 0.02)),
    _p("insulin", "bhb", -1, (0.001, 0.02)),
    _p("cortisol", "glucose", +1, (0.001, 0.012)),
    _p("cortisol", "hepatic_output", +1, (0.02, 0.15)),
    _p("cortisol", "ffa", +1, (0.002, 0.02)),
    _p("glp1", "insulin", +1, (0.001, 0.02)),
    # Holloszy (1967): trained mitochondrial / fat-oxidation capacity
    # raises the ceiling on FFA utilisation. Sign-only floor anchor for
    # the otherwise-unsupervised mitochondrial_capacity latent
    # (docs/physiology-coverage.md breadth floor).
    _p("mitochondrial_capacity", "ffa", +1, (0.0005, 0.01)),
]
