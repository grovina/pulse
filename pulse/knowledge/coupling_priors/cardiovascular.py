"""
Coupling priors for the cardiovascular / autonomic axis.

Until now the CV system carried only cortisol->{hr,sbp,dbp,hrv} and temp->hr
edges (see endocrine.py). The core autonomic relationships between the CV
markers themselves were unencoded, even though they are among the most
firmly established in the literature. These add the marker-to-marker edges
that the teacher full-body model already exhibits, so they REINFORCE the
trajectory signal rather than fight it (the teacher's HRV setpoint is
HRV0 * HR0 / HR, i.e. HRV falls as HR rises).

Sources:
  - Task Force of the ESC/NASPE (1996), *Heart rate variability: standards
    of measurement, physiological interpretation, and clinical use*,
    Circulation 93(5):1043-1065 — HRV is vagally mediated and falls as
    heart rate (sympathetic tone) rises; RR-interval variability and rate
    are inversely coupled both mathematically and physiologically.
  - Shaffer & Ginsberg (2017), *An Overview of Heart Rate Variability
    Metrics and Norms*, Front Public Health 5:258 — time-domain HRV
    (RMSSD/SDNN) tracks parasympathetic activity and is inversely related
    to mean heart rate.
  - Guyton & Hall, *Textbook of Medical Physiology*, 14th ed., ch. 18
    (nervous regulation of circulation) — the arterial baroreflex, an
    SBP->HR negative edge, is NOT yet added here: the teacher HR ODE has
    no BP term, so a coupling prior alone would train the student toward a
    reflex the trajectory signal cannot support. Adding it correctly means
    a coordinated change to full_body.py dHR first. Flagged, not encoded.
"""

from __future__ import annotations

from ..base import CouplingPrior


def _p(src: str, tgt: str, sign: int, mag: tuple[float, float]) -> CouplingPrior:
    return CouplingPrior(source_marker=src, target_marker=tgt, sign=sign, magnitude_range=mag)


COUPLING_PRIORS: list[CouplingPrior] = [
    # HR up -> HRV down. The dominant determinant of time-domain HRV is
    # mean heart rate itself (shorter RR intervals leave less room for
    # beat-to-beat variability) on top of the shared vagal drive. Stronger
    # than the cortisol->hrv edge (0.05, 0.5), so a slightly wider band.
    _p("hr", "hrv", -1, (0.05, 0.6)),
]
