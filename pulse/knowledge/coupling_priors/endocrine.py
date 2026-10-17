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
    # 2026-10-04 (PLAN.md): cortisol feeds back on **CRH**, not on ACTH. Iter 98
    # replaced the flat HPA block with the CRH -> ACTH -> cortisol cascade, in which
    # `dACTH = -k_acth*(ACTH - acth_per_crh*CRH)` contains no cortisol term at all --
    # so `cortisol -> acth` has been a structurally ZERO sensitivity ever since, and
    # its band hinge a gradient-free constant. Rerouted to the edge both models
    # actually implement: cortisol suppresses `crh_target` through
    # `(1 - crh_fb_amp*relu(tanh(log(Cort/Cort_b))))`.
    #
    # The band is the TEACHER's own sensitivity, measured in the normalized frame
    # the loss reads: -0.0466 just above basal, -0.0391 at 14, -0.0265 at 18,
    # -0.0136 at 25 ug/dL. `lo = 0` because the feedback is ONE-SIDED by design (the
    # nocturnal nadir is sleep's, not feedback's -- see modules/stress.py), so below
    # Cort_b the true sensitivity is exactly 0 and a positive floor would penalise
    # correct physiology at every sample where cortisol sits under its basal, which
    # for the default patient is 14 of 24 hours.
    _p("cortisol", "crh", -1, (0.0, 0.06)),
    _p("cortisol", "hr", +1, (0.05, 0.5)),
    _p("cortisol", "sbp", +1, (0.05, 0.4)),
    _p("cortisol", "dbp", +1, (0.02, 0.2)),
    _p("cortisol", "hrv", -1, (0.05, 0.5)),
    _p("insulin", "ghrelin", -1, (0.005, 0.05)),
    _p("glp1", "ghrelin", -1, (0.001, 0.02)),
    _p("leptin", "ghrelin", -1, (0.0005, 0.01)),
    _p("temp", "hr", +1, (0.05, 0.5)),
    # Hypoglycaemia enters the axis at CRH (`dCRH += hypo_crh * relu(58 - G)`).
    # Neither ACTH nor cortisol reads glucose directly, so glucose→cortisol and
    # glucose→ACTH are structurally zero and the edge that exists is this one.
    # Sensitivity is zero above 58 mg/dL, which is the common case, hence `lo = 0`.
    # Cryer et al. 1987 (J Clin Invest 112884): cortisol's threshold is 58 ± 3;
    # epinephrine's is 69 ± 2 and lives on sns_tone, not on this prior.
    _p("glucose", "crh", -1, (0.0, 0.16)),
    _p("crh", "acth", +1, (0.002, 0.03)),
    _p("fat_mass", "leptin", +1, (0.001, 0.02)),
    _p("insulin_slow", "leptin", +1, (0.001, 0.02)),
]
