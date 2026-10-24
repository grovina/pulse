"""Numbers of the sympathoadrenal arm, shared by the teacher and the student.

Cryer et al. 1987, J Clin Invest 112884, measured glycaemic thresholds
during an insulin clamp: epinephrine 69 ± 2 mg/dL, glucagon 68 ± 2,
growth hormone 66 ± 2, cortisol 58 ± 3, symptoms 53 ± 2. Growth hormone
and symptoms have no state here. Glucagon's response to glucose below a
person's own basal is a separate alpha-cell law; the absolute 68 is not
a second one.

Epinephrine clears in about two minutes. The neural tone that releases
it is faster. Resting plasma epinephrine is about 30 pg/mL.

The student learns the measured numbers from these values: the adrenal
side on the stress module (``ADRENAL``), epinephrine's metabolic action on
the metabolic module (``EPINEPHRINE_ACTION``). The level where that action
starts is its own number there, not the resting level: graded infusions
measure the two separately. The tone's resting value and its activity gain
are the units of a latent no study observes, and stay fixed. The teacher
keeps every value.
"""

from .published import Published

SNS_B = 0.20
EPI_B = 30.0
EPI_SPAN = 400.0          # pg/mL from rest to tone = 1
K_SNS = 1.0 / 1.5         # τ ≈ 1.5 min
K_EPI = 0.5               # τ = 2 min
SNS_ACT = 0.80            # activity 1 raises tone from rest to 1
SNS_HYPO = 0.025          # tone per mg/dL below the epinephrine threshold
SNS_SLEEP = 0.10
HYPO_EPI_MGDL = 69.0
HYPO_CORT_MGDL = 58.0
EPI_DRIVE_K = 3.0
# Glycogenolysis and lipolysis multipliers on epinephrine above basal.
# The student holds both as parameters, initialised here. At resting
# epinephrine the excess is zero, so the fed fixed point does not move.
# Wahren 1971, 800 kg-m/min, 40 min: splanchnic output rises 3.63
# mg/kg/min and arterial glucose rises 8.3 mg/dL. A glycogenolysis gain
# of 5 with the uptake below sits on both: the same window rises about
# 3.5 mg/kg/min and about 9 mg/dL, and free fatty acids still clear the
# exercise floor. The lipolysis gain is what that floor moves.
HEP_EPI_GAIN = 5.0
# Exercise blood-glucose uptake, mg/dL/min per unit activity per mg/dL
# of arterial glucose. Contraction-mediated, so it follows work and
# concentration, including below basal, and it is zero at rest.
EXERCISE_UPTAKE_K = 0.015
LIP_EPI_GAIN = 0.5
INS_EPI_GAIN = 0.5        # α2 inhibition of glucose-stimulated secretion
GN_EPI_SPAN = 20.0        # pg/mL added to glucagon's relaxation target

_CRYER = "Cryer 1987 stepped hypoglycaemic clamp"
ADRENAL: dict[str, Published] = {
    "epi_b": Published(EPI_B, "Plasma epinephrine at rest"),
    "epi_span": Published(EPI_SPAN, "Plasma epinephrine across graded exercise"),
    "k_sns": Published(K_SNS, "Plasma epinephrine's rise at the onset of exercise"),
    "k_epi": Published(K_EPI, "Plasma epinephrine's decay after an infusion stops"),
    "sns_hypo": Published(SNS_HYPO, f"{_CRYER}: epinephrine's rise below its threshold"),
    "sns_sleep": Published(SNS_SLEEP, "Plasma epinephrine asleep against awake, at rest"),
    "hypo_epi": Published(HYPO_EPI_MGDL, f"{_CRYER}: the glucose where epinephrine rises"),
    "hypo_cort": Published(HYPO_CORT_MGDL, f"{_CRYER}: the glucose where cortisol rises"),
}
_CLUTTER = "Clutter 1980 graded epinephrine infusions"
EPINEPHRINE_ACTION: dict[str, Published] = {
    "epi_threshold": Published(EPI_B, f"{_CLUTTER}: the level where glucose and glycerol rise"),
    "epi_drive_k": Published(EPI_DRIVE_K, f"{_CLUTTER}: the level at half the metabolic response"),
    "gn_epi_span": Published(GN_EPI_SPAN, "Plasma glucagon during an epinephrine infusion"),
    "ins_epi_gain": Published(
        INS_EPI_GAIN, "Insulin secretion during an epinephrine infusion, glucose clamped"),
}
