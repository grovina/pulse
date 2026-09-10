"""Live student session for the lab viewer.

Now is the edge of time. Eat / Walk / Lie down write the input tape at now.
Play steps the student forward. Nothing past now is computed or shown.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .knowledge.full_body import PatientParams, resolve_derived_params
from .lab_export import (
    SAMPLE_EVERY_MIN,
    _protocol_env,
    _protocol_specs,
    pack_frame,
)
from .model import ModularPhysiologyNetwork, integrate
from .modules.gut import MealEvent
from .types import (
    DUODENAL_CHANNEL_IDS,
    EMBEDDING_DIM,
    GUT_CHANNEL_IDS,
    MARKERS,
    NORM_CENTER,
    STATE_DIM,
)

PLATES = {
    "snack": {"name": "Snack", "carbs": 20.0, "fats": 4.0, "proteins": 4.0},
    "plate": {"name": "Plate", "carbs": 50.0, "fats": 12.0, "proteins": 18.0},
    "rich": {"name": "Rich", "carbs": 40.0, "fats": 25.0, "proteins": 25.0},
}

_MODEL: ModularPhysiologyNetwork | None = None
_MODEL_META: dict[str, Any] | None = None


def load_lab_model(path: str | None = None) -> tuple[ModularPhysiologyNetwork, dict[str, Any]]:
    """Current student architecture. A checkpoint is optional; cold weights are honest."""
    raw = path or os.environ.get("PULSE_LAB_MODEL") or os.environ.get("MODEL_URI")
    meta: dict[str, Any] = {"path": None, "trained": False, "version": "untrained"}
    if raw and not str(raw).startswith("gs://"):
        ckpt_path = Path(raw)
        if ckpt_path.is_file():
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                model = ModularPhysiologyNetwork.from_checkpoint(ckpt, strict=True)
                model.eval()
                for p in model.parameters():
                    p.requires_grad_(False)
                version = str(ckpt.get("model_version", ckpt.get("trained_at", ckpt_path.stem)))
                return model, {"path": str(ckpt_path), "trained": True, "version": version}
            except Exception:
                pass
    model = ModularPhysiologyNetwork()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, meta


def get_lab_model() -> tuple[ModularPhysiologyNetwork, dict[str, Any]]:
    global _MODEL, _MODEL_META
    if _MODEL is None or _MODEL_META is None:
        _MODEL, _MODEL_META = load_lab_model()
    return _MODEL, _MODEL_META


class LabSession:
    def __init__(
        self,
        sample_every: int = SAMPLE_EVERY_MIN,
        model: ModularPhysiologyNetwork | None = None,
        model_meta: dict[str, Any] | None = None,
    ):
        if model is None:
            model, model_meta = get_lab_model()
        self.model = model
        self.model_meta = model_meta or {"path": None, "trained": False, "version": "untrained"}
        self.embedding = torch.zeros(EMBEDDING_DIM, dtype=torch.float32)
        self.params = resolve_derived_params(PatientParams())
        self.sample_every = sample_every
        self.protocol_id = "morning"
        self.t = 0
        self.env: dict[str, Any] = {}
        self.states: list[np.ndarray] = []
        self.run: dict[str, Any] = {}
        self.reset("morning")

    def protocols(self) -> list[dict[str, Any]]:
        return [
            {"id": pid, "title": spec["title"], "blurb": spec["blurb"]}
            for pid, spec in _protocol_specs().items()
        ]

    def reset(self, protocol_id: str) -> dict[str, Any]:
        if protocol_id not in _protocol_specs():
            raise KeyError(protocol_id)
        self.protocol_id = protocol_id
        self.env = _protocol_env(protocol_id)
        self.t = 0
        self.states = [np.asarray(NORM_CENTER, dtype=np.float64).copy()]
        self._rebuild_run()
        return self.snapshot()

    def eat(
        self, carbs: float, fats: float = 0.0, proteins: float = 0.0,
    ) -> dict[str, Any]:
        if self._asleep():
            raise ValueError("asleep")
        carbs = max(0.0, float(carbs))
        fats = max(0.0, float(fats))
        proteins = max(0.0, float(proteins))
        if carbs + fats + proteins <= 0.0:
            raise ValueError("empty meal")
        self.env["meals"].append((float(self.t), carbs, fats, proteins))
        self.env["meals"].sort(key=lambda m: m[0])
        self._rebuild_run()
        return self.snapshot()

    def eat_plate(self, plate: str = "plate") -> dict[str, Any]:
        meal = PLATES.get(plate)
        if meal is None:
            raise KeyError(plate)
        return self.eat(meal["carbs"], meal["fats"], meal["proteins"])

    def walk(self, minutes: int = 30, intensity: float = 0.45) -> dict[str, Any]:
        if self._asleep():
            raise ValueError("asleep")
        end = min(self.t + max(1, int(minutes)), int(self.env["duration_min"]))
        sw = self.env["sleep_wake"]
        act = self.env["activity"]
        act[self.t:end] = np.maximum(act[self.t:end], float(intensity) * sw[self.t:end])
        self._rebuild_run()
        return self.snapshot()

    def lie_down(self) -> dict[str, Any]:
        self.env["sleep_wake"][self.t:] = 0.0
        self.env["activity"][self.t:] = 0.0
        self._rebuild_run()
        return self.snapshot()

    def wake(self) -> dict[str, Any]:
        self.env["sleep_wake"][self.t:] = 1.0
        self._rebuild_run()
        return self.snapshot()

    def advance(self, minutes: int) -> dict[str, Any]:
        minutes = int(minutes)
        if minutes <= 0:
            return self.snapshot()
        last = int(self.env["duration_min"]) - 1
        n = min(minutes, last - self.t)
        if n <= 0:
            return self.snapshot()
        t0 = self.t
        n_steps = n + 1
        meals = self._window_meals(t0)
        sw = torch.as_tensor(self.env["sleep_wake"][t0:t0 + n_steps], dtype=torch.float32)
        act = torch.as_tensor(self.env["activity"][t0:t0 + n_steps], dtype=torch.float32)
        if sw.numel() < n_steps:
            pad = n_steps - int(sw.numel())
            sw = torch.nn.functional.pad(sw, (0, pad), value=float(sw[-1] if sw.numel() else 1.0))
            act = torch.nn.functional.pad(act, (0, pad), value=float(act[-1] if act.numel() else 0.0))
        state0 = torch.as_tensor(self.states[t0], dtype=torch.float32)
        clock = float(self.env["start_hour"]) * 60.0 + t0
        with torch.no_grad():
            traj = integrate(
                self.model,
                state0,
                self.embedding,
                n_steps=n_steps,
                dt=1.0,
                start_time_minutes=clock,
                meals=meals,
                sleep_wake=sw,
                activity=act,
            )
        chunk = traj.detach().cpu().numpy()
        for i in range(1, n_steps):
            self.states.append(np.asarray(chunk[i], dtype=np.float64))
        self.t = t0 + n
        self._rebuild_run()
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        return {
            "t": self.t,
            "protocol_id": self.protocol_id,
            "plates": PLATES,
            "asleep": self._asleep(),
            "model": self.model_meta,
            "run": self.run,
        }

    def _asleep(self) -> bool:
        return float(self.env["sleep_wake"][self.t]) < 0.5

    def _window_meals(self, t0: int) -> list[MealEvent]:
        return [
            MealEvent(time=float(mt) - float(t0), carbs=c, fats=f, proteins=p)
            for mt, c, f, p in self.env["meals"]
        ]

    def _rebuild_run(self) -> None:
        t_end = self.t
        times = torch.arange(t_end + 1, dtype=torch.float32)
        meals = self._window_meals(0)
        with torch.no_grad():
            emb_gut = self.model.embedding_projections["gut"](self.embedding)
            gut_all = self.model.gut.forward_window(times, meals, emb_gut).detach().cpu().numpy()
            duo_all = self.model.duodenal.forward_window(times, meals).detach().cpu().numpy()
        sample = list(range(0, t_end + 1, self.sample_every))
        if sample[-1] != t_end:
            sample.append(t_end)
        frames = []
        for tt in sample:
            prev_t = max(0, tt - self.sample_every)
            frames.append(pack_frame(
                tt, self.states[tt], gut_all[tt], duo_all[tt],
                float(self.env["sleep_wake"][tt]),
                float(self.env["activity"][tt]),
                float(self.env["start_hour"]),
                self.env["meals"],
                prev_state=None if tt == 0 else self.states[prev_t],
                dt=float(tt - prev_t) if tt else 1.0,
                params=self.params,
            ))
        spec = self.env["spec"]
        meals = self.env["meals"]
        self.run = {
            "schema": "pulse.lab.run.v2",
            "id": self.protocol_id,
            "title": spec["title"],
            "blurb": spec["blurb"],
            "engine": "student",
            "patient": "default",
            "duration_min": self.env["duration_min"],
            "start_hour": self.env["start_hour"],
            "sample_every_min": self.sample_every,
            "now": self.t,
            "marker_ids": [m.id for m in MARKERS],
            "gut_channel_ids": list(GUT_CHANNEL_IDS),
            "duodenal_channel_ids": list(DUODENAL_CHANNEL_IDS),
            "meals": [
                {"t": float(t), "carbs": float(c), "fats": float(f), "proteins": float(p)}
                for t, c, f, p in meals
            ],
            "phases": [],
            "frames": frames,
        }
        assert self.states[0].shape[0] == STATE_DIM
