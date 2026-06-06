"""
Respiratory module.

Respiratory rate and SpO2 with fully learned dynamics.
Receives lactate from Metabolic as a metabolic demand proxy.
"""

from .base import LearnedDynamicsModule

# Coupling inputs: lactate (1)
_N_COUPLING = 1

# External inputs: activity (1) + sleep_wake (1) = 2
_N_EXTERNAL = 2


class RespiratoryModule(LearnedDynamicsModule):
    def __init__(self, embedding_dim: int, hidden_dim: int = 24):
        super().__init__(
            n_state=2,
            n_coupling=_N_COUPLING,
            n_external=_N_EXTERNAL,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
        )
