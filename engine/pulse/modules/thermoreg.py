"""
Thermoregulation module.

Core temperature with fully learned dynamics. Receives metabolic rate
proxy (glucose) and cortisol from other modules.
"""

from .base import LearnedDynamicsModule

# Coupling inputs: glucose (metabolic rate proxy) (1) + cortisol (1) = 2
_N_COUPLING = 2

# External inputs: activity (1) + sleep_wake (1) = 2
_N_EXTERNAL = 2


class ThermoregModule(LearnedDynamicsModule):
    def __init__(self, embedding_dim: int, hidden_dim: int = 24):
        super().__init__(
            n_state=1,
            n_coupling=_N_COUPLING,
            n_external=_N_EXTERNAL,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
        )
