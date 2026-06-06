"""
Cardiovascular module.

Heart rate, HRV, systolic BP, diastolic BP.
Fully learned dynamics. Receives cortisol from Stress, temperature from
Thermoregulation, and glucose + insulin from Metabolic so postprandial
sympathetic / autonomic effects can reach HR and BP.

Architectural constraint: SBP > DBP (physical — systolic is during
contraction, diastolic during relaxation).

Iter 32 note: glucose + insulin couplings added because hr_mape was
stuck at ~0.245 across iters 27-31. With cortisol+temperature only,
postprandial HR rise was structurally unreachable from the meal axis —
no amount of glucose-side weight tuning could move HR.

Iter 36 note (reverted): a per-user baseline head
(``state_centered = state - tanh_bounded_offset(embedding)``) was added
on the theory that calibration's shallow gradient through 12h of
rate-of-change integration could be bypassed with a unit-slope gradient
on baseline. Empirically (apps/pulse/docs/iter36-calibration-investigation.md):
hr_mape 0.193 → 0.264, sbp_mape 0.062 → 0.173, verifier_coupling
0.876 → 0.541. The head competed with the existing dynamics pathway
during training: the model learned to respond to glucose/insulin
couplings *in the centered frame*, but at calibration time
(zero embedding ⇒ offset=0) state arrives uncentered and the coupling
response misfires. Reverted so the codebase reflects iter 35 + Path 1
(the actual current best).
"""

from .base import LearnedDynamicsModule

# Coupling inputs: cortisol (1) + temperature (1) + glucose (1) + insulin (1) = 4
_N_COUPLING = 4

# External inputs: activity (1) + sleep_wake (1) = 2
_N_EXTERNAL = 2


class CardiovascularModule(LearnedDynamicsModule):
    def __init__(self, embedding_dim: int, hidden_dim: int = 48):
        super().__init__(
            n_state=4,
            n_coupling=_N_COUPLING,
            n_external=_N_EXTERNAL,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
        )
