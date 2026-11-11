"""The stomach as a state. Swallow is the protocol; the calorie rate is the law."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .base import PhysiologyModule
from ..correction import Correction, correction_weights
from ..gastric import EMPTY_KCAL_PER_MIN, KCAL_PER_G, leaving_grams
from ..types import MARKER_INDEX, MODULE_MARKER_IDS, NORM_CENTER, NORM_SCALE

_ORDER = ("gastric_carb", "gastric_protein", "gastric_fat")
if tuple(MODULE_MARKER_IDS["gastric"]) != _ORDER:
    raise RuntimeError(
        f"gastric marker order {MODULE_MARKER_IDS['gastric']} != {_ORDER}"
    )

_IDX = [MARKER_INDEX[name] for name in _ORDER]
_CENTER = torch.tensor([NORM_CENTER[i] for i in _IDX], dtype=torch.float32)
_SCALE = torch.tensor([NORM_SCALE[i] for i in _IDX], dtype=torch.float32)


def _inverse_softplus(y: float) -> float:
    return math.log(math.exp(y) - 1.0)


class GastricModule(PhysiologyModule):
    """Three pools. A minute's swallow is external; what leaves is the shared rate.

    The split across carbohydrate, protein and fat is the default shape.
    ``split_correction`` starts at zero and can move that split. The calorie
    rate stays ``empty_kcal``.
    """

    external_inputs = ("swallow_carb", "swallow_protein", "swallow_fat")
    n_state = 3
    n_coupling = 0

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.log_empty_kcal = nn.Parameter(
            torch.tensor(_inverse_softplus(EMPTY_KCAL_PER_MIN)))
        self.register_buffer("_center", _CENTER.clone())
        self.register_buffer("_scale", _SCALE.clone())
        self.split_correction = Correction(3)

    def empty_kcal(self) -> torch.Tensor:
        return nn.functional.softplus(self.log_empty_kcal)

    def constants(self, embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "empty_kcal": self.empty_kcal(),
            "kcal_per_g": self._center.new_tensor(KCAL_PER_G),
            "split_weight": correction_weights(
                self.split_correction(), n=self.n_state, like=self._center),
        }

    def drives(
        self,
        external: torch.Tensor,
        coupling: torch.Tensor,
        time_features: torch.Tensor,
        const: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return {"swallow": external}

    def step(
        self,
        state: torch.Tensor,
        coupling: torch.Tensor,
        const: dict[str, torch.Tensor],
        drv: dict[str, torch.Tensor],
        raw: dict[object, torch.Tensor],
    ) -> torch.Tensor:
        grams = (self._center + self._scale * state).clamp(min=0.0)
        swallow = drv["swallow"]
        filled = grams + swallow
        left = leaving_grams(
            filled, const["kcal_per_g"], const["empty_kcal"], const["split_weight"])
        return swallow - left
