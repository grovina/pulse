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
    (nervous regulation of circulation) — the arterial baroreflex.

NOT ADDED: the sbp->hr negative (baroreflex) edge. RESOLVED 2026-07-20 —
this was previously flagged here as "add a teacher dHR term first, then
this prior". That work was done as a measurement pass, and the answer is
that neither belongs in this model.

The teacher is a MINUTE-mean sim; the baroreflex is a beat-to-beat (~1-5 s)
mechanism that equilibrates within one integration step, so it is already
folded into the effective act_hr_gain/k_hr gains rather than being a
separate dynamic. Measured over 40 patients x 24 h, the activity-independent
SBP deviation a gated reflex could act on has sd 0.40 mmHg (vs 9.97 mmHg
total SBP sd) — a ~1 bpm effect. What residual exists is a first-order lag
artifact at exercise transitions (-17.7 mmHg at onset, +17.7 at offset), and
acting on it would corrupt the exercise HR rise and recovery, which are
currently correct against Cole 1999.

Decisive for THIS file specifically: the realized HR-SBP correlation in the
teacher trajectories is +0.964, because HR and SBP share the activity and
cortisol drives and central command resets the baroreflex operating point
upward during exercise. A -1 prior would be flatly contradicted by the
training signal — the same consistency failure that kept the resting-HRV
absolute anchor out of cohorts/cardiovascular.py. See the CARDIOVASCULAR
block in full_body.py for the full reasoning.
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
