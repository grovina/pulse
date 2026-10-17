"""Phase offset and sleep debt. Sleep is the protocol; the update is the law."""

from __future__ import annotations

import math

import torch

from .base import PhysiologyModule
from ..circadian import step_clock_t
from ..types import MARKER_INDEX, MODULE_MARKER_IDS, NORM_CENTER, NORM_SCALE

_ORDER = ("circ_phase_offset", "sleep_debt")
if tuple(MODULE_MARKER_IDS["circadian"]) != _ORDER:
    raise RuntimeError(
        f"circadian marker order {MODULE_MARKER_IDS['circadian']} != {_ORDER}"
    )

_IDX = [MARKER_INDEX[name] for name in _ORDER]
_CENTER = torch.tensor([NORM_CENTER[i] for i in _IDX], dtype=torch.float32)
_SCALE = torch.tensor([NORM_SCALE[i] for i in _IDX], dtype=torch.float32)


def _hour_from_time_features(time_features: torch.Tensor) -> torch.Tensor:
    sin_t = time_features[..., 0]
    cos_t = time_features[..., 1]
    theta = torch.atan2(sin_t, cos_t)
    return torch.remainder(theta / (2.0 * math.pi) * 24.0, 24.0)


class CircadianModule(PhysiologyModule):
    """Two states. The wall hour and sleep are external; both rates are the shared step."""

    external_inputs = ("sleep_wake",)
    n_state = 2
    n_coupling = 0

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.register_buffer("_center", _CENTER.clone())
        self.register_buffer("_scale", _SCALE.clone())

    def drives(
        self,
        external: torch.Tensor,
        coupling: torch.Tensor,
        time_features: torch.Tensor,
        const: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return {
            "hour": _hour_from_time_features(time_features),
            "asleep": 1.0 - external[..., 0],
        }

    def step(
        self,
        state: torch.Tensor,
        coupling: torch.Tensor,
        const: dict[str, torch.Tensor],
        drv: dict[str, torch.Tensor],
        raw: dict[object, torch.Tensor],
    ) -> torch.Tensor:
        offset = self._center[0] + self._scale[0] * state[..., 0]
        debt = self._center[1] + self._scale[1] * state[..., 1]
        new_offset, new_debt = step_clock_t(offset, debt, drv["hour"], drv["asleep"])
        return torch.stack((new_offset - offset, new_debt - debt), dim=-1)
