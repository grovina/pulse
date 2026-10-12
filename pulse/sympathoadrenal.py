"""Numbers of the sympathoadrenal arm, shared by the teacher and the student.

Cryer et al. 1987, J Clin Invest 112884, measured glycaemic thresholds
during an insulin clamp: epinephrine 69 ± 2 mg/dL, glucagon 68 ± 2,
growth hormone 66 ± 2, cortisol 58 ± 3, symptoms 53 ± 2. Growth hormone
and symptoms have no state here. Glucagon's response to glucose below a
person's own basal is a separate alpha-cell law; the absolute 68 is not
a second one.

Epinephrine clears in about two minutes. The neural tone that releases
it is faster. Resting plasma epinephrine is about 30 pg/mL.
"""

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
