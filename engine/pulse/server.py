from __future__ import annotations

import os
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI  # type: ignore[import-not-found]
from pydantic import BaseModel, Field

from .knowledge.textbook_scenarios.flow_story_protocol import dietary_carb_flow_phases_for_ui
from .model import ModularPhysiologyNetwork, integrate
from .modules.gut import MealEvent
from .types import (
    EMBEDDING_DIM, GUT_OUTPUT_DIM, MARKER_IDS, MARKER_INDEX, MARKERS,
    NORM_CENTER, NORM_SCALE,
)

app = FastAPI(title="Pulse Engine", version="0.3.0")

BASELINE_KEYS = [m.id for m in MARKERS]


class MealInput(BaseModel):
    time: float
    carbs: float
    fats: float = 0.0
    proteins: float = 0.0


class CheckInInput(BaseModel):
    time: float | None = None
    createdAt: str
    feelings: dict[str, Any] = Field(default_factory=dict)
    bodySignals: dict[str, Any] = Field(default_factory=dict)
    measurements: dict[str, Any] = Field(default_factory=dict)
    meal: dict[str, Any] = Field(default_factory=dict)
    waterIntakeMl: int | None = None
    sleepWake: float | None = None
    activity: float | None = None


class SimulateRequest(BaseModel):
    user_id: str
    duration_min: int = 720
    sample_interval: int = 5
    meals: list[MealInput] = Field(default_factory=list)
    check_ins: list[CheckInInput] = Field(default_factory=list)
    embedding: list[float] | None = None
    baseline: dict[str, float] | None = None
    calibrate: bool = False
    model_version: str | None = None


class LoadedModel(BaseModel):
    hidden_dim: int
    marker_ids: list[str]
    model_version: str


_MODEL: ModularPhysiologyNetwork | None = None
_MODEL_META: LoadedModel | None = None


@app.on_event("startup")
def load_model_on_startup():
    get_model()


@app.get("/health")
def health():
    return {
        "ok": True,
        "modelLoaded": _MODEL is not None,
        "markerCount": len(_MODEL_META.marker_ids) if _MODEL_META else 0,
        "modelVersion": _MODEL_META.model_version if _MODEL_META else None,
    }


@app.post("/simulate")
def simulate(body: SimulateRequest):
    model = get_model()
    duration_min = max(60, min(24 * 60, body.duration_min))
    sample_interval = max(1, min(30, body.sample_interval))

    meals = [MealEvent(
        time=float(m.time),
        carbs=max(0.0, float(m.carbs)),
        fats=max(0.0, float(m.fats)),
        proteins=max(0.0, float(m.proteins)),
    ) for m in body.meals]

    initial_baseline = normalize_baseline(body.baseline)
    initial_state = initial_state_from_baseline(initial_baseline)
    initial_embedding = get_initial_embedding(user_id=body.user_id, embedding=body.embedding)

    calibration = calibrate_embedding(
        model=model,
        user_id=body.user_id,
        initial_embedding=initial_embedding,
        initial_baseline=initial_baseline,
        check_ins=body.check_ins,
        duration_min=duration_min,
        meals=meals,
        initial_state=initial_state,
        enabled=body.calibrate,
    )

    predicted = predict_with_model(
        model=model,
        embedding=calibration.embedding,
        duration_min=duration_min,
        meals=meals,
        initial_state=initial_state,
    )

    sample_times = list(range(0, duration_min, sample_interval))
    gut_full = gut_profile_for_simulation(
        model, calibration.embedding, duration_min, meals, start_time_minutes=360.0,
    )
    meal_t = _first_carb_meal_time_min(meals)
    carb_flow = {
        "meal_time_min": meal_t,
        "phases": dietary_carb_flow_phases_for_ui(duration_min, meal_t),
        "times_min": sample_times,
        "glucose_appearance": [float(gut_full[t, 0]) for t in sample_times],
        "marker_ids": [
            "glucose",
            "insulin",
            "glucagon",
            "ffa",
            "ghrelin",
            "glp1",
            "temp",
        ],
        "series": {
            mid: [float(predicted[t, MARKER_INDEX[mid]]) for t in sample_times]
            for mid in [
                "glucose",
                "insulin",
                "glucagon",
                "ffa",
                "ghrelin",
                "glp1",
                "temp",
            ]
        },
    }

    return {
        "sample_interval": sample_interval,
        "marker_ids": _MODEL_META.marker_ids if _MODEL_META else MARKER_IDS,
        "model_version": _MODEL_META.model_version if _MODEL_META else None,
        "embedding": calibration.embedding.tolist(),
        "baseline": calibration.baseline,
        "calibration": {
            "train_loss": calibration.train_loss,
            "val_loss": calibration.val_loss,
            "quality": calibration.quality,
            "accepted": calibration.accepted,
            "steps": calibration.steps,
        },
        "times_min": sample_times,
        "series": predicted[sample_times].tolist(),
        "carb_flow": carb_flow,
    }


def seed_from_user_id(user_id: str) -> int:
    digest = sha256(user_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def get_model() -> ModularPhysiologyNetwork:
    global _MODEL, _MODEL_META
    if _MODEL is not None:
        return _MODEL

    model_uri = os.getenv("MODEL_URI")
    if not model_uri:
        raise RuntimeError("MODEL_URI is required for Pulse engine inference.")

    model_path = resolve_model_uri(model_uri)
    checkpoint = torch.load(model_path, map_location="cpu")
    hidden_dim = int(checkpoint.get("hidden_dim", 48))
    marker_ids = checkpoint.get("marker_ids", MARKER_IDS)
    model_version = str(checkpoint.get("model_version", checkpoint.get("trained_at", "unknown")))
    model_state = checkpoint.get("model_state", checkpoint)

    model = ModularPhysiologyNetwork(
        metabolic_hidden=hidden_dim,
        appetite_hidden=max(24, hidden_dim // 2),
        stress_hidden=max(24, hidden_dim // 2),
        cardiovascular_hidden=hidden_dim,
        thermoreg_hidden=max(16, hidden_dim // 3),
        respiratory_hidden=max(16, hidden_dim // 3),
    )
    model.load_state_dict(model_state)
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()

    _MODEL = model
    _MODEL_META = LoadedModel(hidden_dim=hidden_dim, marker_ids=marker_ids, model_version=model_version)
    return _MODEL


def resolve_model_uri(model_uri: str) -> str:
    if model_uri.startswith("gs://"):
        return download_from_gcs(model_uri)
    return model_uri


def download_from_gcs(uri: str) -> str:
    from google.cloud import storage  # type: ignore[import-untyped]

    without_scheme = uri[5:]
    bucket, _, blob_name = without_scheme.partition("/")
    if not bucket or not blob_name:
        raise ValueError(f"Invalid GCS model URI: {uri}")

    local_path = f"/tmp/{sha256(uri.encode('utf-8')).hexdigest()}-pulse-model.pt"
    client = storage.Client()
    client.bucket(bucket).blob(blob_name).download_to_filename(local_path)
    return local_path


def gut_profile_for_simulation(
    model: ModularPhysiologyNetwork,
    embedding: torch.Tensor,
    duration_min: int,
    meals: list[MealEvent],
    start_time_minutes: float = 360.0,
) -> np.ndarray:
    """Per-minute gut outputs (glucose/lipid/amino appearance + flag) for charting."""
    device = next(model.parameters()).device
    emb = embedding.to(device=device, dtype=torch.float32)
    emb_gut = model.embedding_projections["gut"](emb)
    out = np.zeros((duration_min, GUT_OUTPUT_DIM), dtype=np.float32)
    with torch.no_grad():
        for step in range(duration_min):
            t_abs = (start_time_minutes + float(step)) % 1440.0
            g = model.gut(t_abs, meals, emb_gut)
            out[step] = g.detach().cpu().numpy().astype(np.float32)
    return out


def predict_with_model(
    model: ModularPhysiologyNetwork,
    embedding: torch.Tensor,
    duration_min: int,
    meals: list[MealEvent],
    initial_state: np.ndarray,
) -> np.ndarray:
    initial_state_t = torch.tensor(initial_state, dtype=torch.float32)

    with torch.no_grad():
        predicted_t = integrate(
            model,
            initial_state_t,
            embedding,
            n_steps=duration_min,
            dt=1.0,
            start_time_minutes=360.0,
            meals=meals,
        )
    return predicted_t.numpy()


def _first_carb_meal_time_min(meals: list[MealEvent]) -> float | None:
    carb_meals = [m.time for m in meals if m.carbs > 1e-6]
    if not carb_meals:
        return None
    return float(min(carb_meals))


def seeded_embedding(user_id: str) -> torch.Tensor:
    seed = seed_from_user_id(f"embedding:{user_id}")
    rng = np.random.default_rng(seed)
    vector = rng.normal(0, 0.1, size=EMBEDDING_DIM).astype(np.float32)
    return torch.tensor(vector, dtype=torch.float32)


def get_initial_embedding(user_id: str, embedding: list[float] | None) -> torch.Tensor:
    if embedding and len(embedding) == EMBEDDING_DIM:
        return torch.tensor(np.array(embedding, dtype=np.float32), dtype=torch.float32)
    return seeded_embedding(user_id)


def initial_state_from_baseline(baseline: dict[str, float]) -> np.ndarray:
    state = np.array(NORM_CENTER, dtype=np.float32)
    for key, value in baseline.items():
        idx = MARKER_INDEX.get(key)
        if idx is not None:
            state[idx] = float(value)
    return state


@dataclass
class CalibrationResult:
    embedding: torch.Tensor
    baseline: dict[str, float]
    train_loss: float
    val_loss: float
    quality: float
    accepted: bool
    steps: int


def calibrate_embedding(
    model: ModularPhysiologyNetwork,
    user_id: str,
    initial_embedding: torch.Tensor,
    initial_baseline: dict[str, float],
    check_ins: list[CheckInInput],
    duration_min: int,
    meals: list[MealEvent],
    initial_state: np.ndarray,
    enabled: bool,
) -> CalibrationResult:
    updated_baseline = estimate_updated_baseline(initial_baseline, check_ins)
    observations = build_observations(check_ins, duration_min)
    soft_evidence = build_subjective_targets(check_ins, duration_min)

    if not enabled or (len(observations) + len(soft_evidence)) < 2:
        baseline_loss = float(compute_observation_loss(
            model=model,
            embedding=initial_embedding,
            observations=observations,
            subjective_targets=soft_evidence,
            duration_min=duration_min,
            meals=meals,
            initial_state=initial_state,
        ))
        return CalibrationResult(
            embedding=initial_embedding,
            baseline=updated_baseline,
            train_loss=baseline_loss,
            val_loss=baseline_loss,
            quality=to_quality_score(baseline_loss),
            accepted=False,
            steps=0,
        )

    all_items: list[tuple[int, str, int]] = \
        [(o[0], "obs", i) for i, o in enumerate(observations)] + \
        [(s.time, "sub", i) for i, s in enumerate(soft_evidence)]
    all_items.sort(key=lambda x: x[0])

    split_idx = max(1, int(len(all_items) * 0.8))
    train_items = all_items[:split_idx]
    val_items = all_items[split_idx:] if split_idx < len(all_items) else all_items[-1:]

    train_obs = [observations[e[2]] for e in train_items if e[1] == "obs"]
    train_sub = [soft_evidence[e[2]] for e in train_items if e[1] == "sub"]
    val_obs = [observations[e[2]] for e in val_items if e[1] == "obs"]
    val_sub = [soft_evidence[e[2]] for e in val_items if e[1] == "sub"]

    baseline_val = float(compute_observation_loss(
        model=model, embedding=initial_embedding,
        observations=val_obs, subjective_targets=val_sub,
        duration_min=duration_min, meals=meals, initial_state=initial_state,
    ))
    initial_train = float(compute_observation_loss(
        model=model, embedding=initial_embedding,
        observations=train_obs, subjective_targets=train_sub,
        duration_min=duration_min, meals=meals, initial_state=initial_state,
    ))

    candidate = torch.nn.Parameter(initial_embedding.clone())
    optimizer = torch.optim.Adam([candidate], lr=0.03)
    best_embedding = initial_embedding.clone()
    best_val = baseline_val
    best_train = initial_train
    patience = 8
    no_improvement = 0
    max_steps = 60
    ran_steps = 0

    for step in range(max_steps):
        ran_steps = step + 1
        optimizer.zero_grad()
        train_loss = compute_observation_loss(
            model=model, embedding=candidate,
            observations=train_obs, subjective_targets=train_sub,
            duration_min=duration_min, meals=meals, initial_state=initial_state,
            requires_grad=True,
        )
        prior_loss = (candidate - initial_embedding).pow(2).mean()
        norm_excess = torch.relu(torch.norm(candidate) - 2.5)
        objective = train_loss + 0.01 * prior_loss + 0.001 * norm_excess.pow(2)
        objective.backward()
        optimizer.step()

        with torch.no_grad():
            val_loss = float(compute_observation_loss(
                model=model, embedding=candidate,
                observations=val_obs, subjective_targets=val_sub,
                duration_min=duration_min, meals=meals, initial_state=initial_state,
            ))
            if val_loss < best_val:
                best_val = val_loss
                best_train = float(train_loss.item())
                best_embedding = candidate.detach().clone()
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement >= patience:
                    break

    accepted = bool(best_val <= baseline_val * 0.98)
    selected = best_embedding if accepted else initial_embedding
    final_train = best_train if accepted else initial_train
    final_val = best_val if accepted else baseline_val

    return CalibrationResult(
        embedding=selected.detach(),
        baseline=updated_baseline if accepted else initial_baseline,
        train_loss=float(final_train),
        val_loss=float(final_val),
        quality=to_quality_score(float(final_val)),
        accepted=accepted,
        steps=ran_steps,
    )


def build_observations(check_ins: list[CheckInInput], duration_min: int) -> list[tuple[int, list[int], list[float]]]:
    observations: list[tuple[int, list[int], list[float]]] = []
    measured_keys = [m.id for m in MARKERS]
    for check_in in check_ins:
        if check_in.time is None:
            continue
        t = int(round(check_in.time))
        if t < 0 or t >= duration_min:
            continue
        measurements = check_in.measurements or {}
        indices: list[int] = []
        values: list[float] = []
        for key in measured_keys:
            raw = as_float(measurements.get(key))
            if raw is None:
                continue
            indices.append(MARKER_INDEX[key])
            values.append(float(raw))
        if indices:
            observations.append((t, indices, values))
    observations.sort(key=lambda item: item[0])
    return observations


@dataclass(frozen=True)
class SoftEvidence:
    time: int
    marker_id: str
    direction: str
    threshold: float
    specificity: float
    scale: float


@dataclass(frozen=True)
class SoftEvidenceTemplate:
    marker_id: str
    direction: str
    threshold: float
    base_specificity: float
    scale: float


SUBJECTIVE_EVIDENCE: dict[str, list[SoftEvidenceTemplate]] = {
    "hungry": [
        SoftEvidenceTemplate("ghrelin", "above", 95.0, 0.3, 25.0),
        SoftEvidenceTemplate("glucose", "below", 95.0, 0.2, 20.0),
    ],
    "full": [
        SoftEvidenceTemplate("glp1", "above", 12.0, 0.3, 8.0),
    ],
    "stressed": [
        SoftEvidenceTemplate("cortisol", "above", 15.0, 0.3, 4.0),
        SoftEvidenceTemplate("acth", "above", 38.0, 0.25, 12.0),
        SoftEvidenceTemplate("hr", "above", 75.0, 0.25, 8.0),
    ],
    "tired": [
        SoftEvidenceTemplate("cortisol", "below", 8.0, 0.25, 4.0),
    ],
    "shaky": [
        SoftEvidenceTemplate("glucose", "below", 75.0, 0.4, 12.0),
    ],
}

BODY_SIGNAL_EVIDENCE: dict[str, list[SoftEvidenceTemplate]] = {
    "frequentUrination": [
        SoftEvidenceTemplate("glucose", "above", 140.0, 0.15, 30.0),
    ],
}


def _context_attenuation(
    signal_id: str,
    template: SoftEvidenceTemplate,
    check_in: CheckInInput,
) -> float:
    attenuation = 1.0
    if signal_id == "frequentUrination" and template.marker_id == "glucose":
        water = check_in.waterIntakeMl
        if water is not None and water > 400:
            attenuation *= 0.3
        elif water is not None and water > 250:
            attenuation *= 0.6
    if signal_id == "stressed" and template.marker_id == "hr":
        meal = check_in.meal or {}
        if as_float(meal.get("carbs")) is not None:
            attenuation *= 0.7
    return attenuation


def build_subjective_targets(
    check_ins: list[CheckInInput], duration_min: int,
) -> list[SoftEvidence]:
    targets: list[SoftEvidence] = []
    for check_in in check_ins:
        if check_in.time is None:
            continue
        t = int(round(check_in.time))
        if t < 0 or t >= duration_min:
            continue

        feelings = check_in.feelings or {}
        for signal_id, templates in SUBJECTIVE_EVIDENCE.items():
            if not bool(feelings.get(signal_id)):
                continue
            for tmpl in templates:
                atten = _context_attenuation(signal_id, tmpl, check_in)
                specificity = tmpl.base_specificity * atten
                if specificity < 0.01:
                    continue
                targets.append(SoftEvidence(
                    time=t, marker_id=tmpl.marker_id, direction=tmpl.direction,
                    threshold=tmpl.threshold, specificity=specificity, scale=tmpl.scale,
                ))

        body = check_in.bodySignals or {}
        for signal_id, templates in BODY_SIGNAL_EVIDENCE.items():
            if not bool(body.get(signal_id)):
                continue
            for tmpl in templates:
                atten = _context_attenuation(signal_id, tmpl, check_in)
                specificity = tmpl.base_specificity * atten
                if specificity < 0.01:
                    continue
                targets.append(SoftEvidence(
                    time=t, marker_id=tmpl.marker_id, direction=tmpl.direction,
                    threshold=tmpl.threshold, specificity=specificity, scale=tmpl.scale,
                ))

    targets.sort(key=lambda x: x.time)
    return targets


def compute_observation_loss(
    model: ModularPhysiologyNetwork,
    embedding: torch.Tensor,
    observations: list[tuple[int, list[int], list[float]]],
    subjective_targets: list[SoftEvidence],
    duration_min: int,
    meals: list[MealEvent],
    initial_state: np.ndarray,
    requires_grad: bool = False,
) -> float | torch.Tensor:
    has_obs = len(observations) > 0
    has_sub = len(subjective_targets) > 0
    if not has_obs and not has_sub:
        return 0.0 if not requires_grad else torch.tensor(0.0, dtype=torch.float32)

    initial_state_t = torch.tensor(initial_state, dtype=torch.float32)

    if requires_grad:
        predicted = integrate(
            model, initial_state_t, embedding,
            n_steps=duration_min, dt=1.0, start_time_minutes=360.0,
            meals=meals,
        )
    else:
        with torch.no_grad():
            predicted = integrate(
                model, initial_state_t, embedding,
                n_steps=duration_min, dt=1.0, start_time_minutes=360.0,
                meals=meals,
            )

    losses = []
    norm_scale = torch.tensor(NORM_SCALE, dtype=torch.float32)

    for time_idx, marker_indices, marker_values in observations:
        pred_values = predicted[time_idx, marker_indices]
        target = torch.tensor(marker_values, dtype=torch.float32)
        scale = norm_scale[marker_indices]
        diff = (pred_values - target) / scale
        losses.append(F.huber_loss(diff, torch.zeros_like(diff), delta=1.0, reduction="mean"))

    for evidence in subjective_targets:
        idx = MARKER_INDEX.get(evidence.marker_id)
        if idx is None:
            continue
        pred_value = predicted[evidence.time, idx]
        threshold_t = torch.tensor(evidence.threshold, dtype=torch.float32)
        if evidence.direction == "above":
            margin = (pred_value - threshold_t) / evidence.scale
        else:
            margin = (threshold_t - pred_value) / evidence.scale
        nll = -F.logsigmoid(margin)
        losses.append(evidence.specificity * nll)

    if not losses:
        return 0.0 if not requires_grad else torch.tensor(0.0, dtype=torch.float32)

    stacked = torch.stack(losses).mean()
    if requires_grad:
        return stacked
    return float(stacked.item())


def to_quality_score(loss_value: float) -> float:
    if loss_value <= 0:
        return 1.0
    return float(np.exp(-loss_value))


def normalize_baseline(raw: dict[str, float] | None) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    cleaned: dict[str, float] = {}
    for key in BASELINE_KEYS:
        value = as_float(raw.get(key))
        if value is None:
            continue
        cleaned[key] = float(value)
    return cleaned


def estimate_updated_baseline(
    current_baseline: dict[str, float],
    check_ins: list[CheckInInput],
    alpha: float = 0.35,
) -> dict[str, float]:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    measured: dict[str, list[float]] = {}
    measurable_keys = {"glucose", "hr", "sbp", "dbp", "temp"}
    for check_in in check_ins:
        measurements = check_in.measurements or {}
        for key in measurable_keys:
            value = as_float(measurements.get(key))
            if value is not None:
                measured.setdefault(key, []).append(float(value))

    updated = dict(current_baseline)
    for key, values in measured.items():
        if not values:
            continue
        robust = float(np.median(np.array(values, dtype=np.float32)))
        prev = current_baseline.get(key)
        updated[key] = robust if prev is None else (1 - alpha) * float(prev) + alpha * robust

    return updated


def as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None
