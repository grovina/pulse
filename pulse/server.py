from __future__ import annotations

import os
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI  # type: ignore[import-not-found]
from pydantic import BaseModel, Field

from .calibration import (
    CalibrationSettings,
    MeasurementPoint,
    SoftEvidence,
    calibrate_embedding as shared_calibrate,
    evaluate_data_loss,
)
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
    # Iter 97 (review 5.4): the real clock and the real masks. ``start_time_minutes``
    # is the minute-of-day at t=0 (the server used to hard-code 06:00);
    # ``sleep_wake`` / ``activity`` are per-minute arrays (1 = awake / 0 = rest ..
    # 1 = vigorous). When absent they are forward-filled from the check-ins'
    # ``sleepWake`` / ``activity`` fields; when those are absent too the model
    # falls back to its learned defaults (None), exactly as the benchmark does.
    start_time_minutes: float | None = None
    sleep_wake: list[float] | None = None
    activity: list[float] | None = None


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
    start_time_minutes = resolve_start_time_minutes(body.start_time_minutes)
    sleep_wake = build_minute_mask(body.sleep_wake, body.check_ins, "sleepWake", duration_min)
    activity = build_minute_mask(body.activity, body.check_ins, "activity", duration_min)

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
        start_time_minutes=start_time_minutes,
        sleep_wake=sleep_wake,
        activity=activity,
    )

    predicted = predict_with_model(
        model=model,
        embedding=calibration.embedding,
        duration_min=duration_min,
        meals=meals,
        initial_state=initial_state,
        start_time_minutes=start_time_minutes,
        sleep_wake=sleep_wake,
        activity=activity,
    )

    sample_times = list(range(0, duration_min, sample_interval))
    gut_full = gut_profile_for_simulation(
        model, calibration.embedding, duration_min, meals, start_time_minutes=start_time_minutes,
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
            "reason": calibration.reason,
            "baseline_val_loss": calibration.baseline_val_loss,
            "embedding_norm": calibration.embedding_norm,
        },
        "start_time_minutes": start_time_minutes,
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
    marker_ids = checkpoint.get("marker_ids", MARKER_IDS)
    model_version = str(checkpoint.get("model_version", checkpoint.get("trained_at", "unknown")))

    model = ModularPhysiologyNetwork.from_checkpoint(checkpoint)
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()

    hidden_dim = int(checkpoint.get("hidden_dim", model.constructor_kwargs.get("metabolic_hidden", 48)))
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
    start_time_minutes: float = 360.0,
    sleep_wake: torch.Tensor | None = None,
    activity: torch.Tensor | None = None,
) -> np.ndarray:
    initial_state_t = torch.tensor(initial_state, dtype=torch.float32)

    with torch.no_grad():
        predicted_t = integrate(
            model,
            initial_state_t,
            embedding,
            n_steps=duration_min,
            dt=1.0,
            start_time_minutes=start_time_minutes,
            meals=meals,
            sleep_wake=sleep_wake,
            activity=activity,
        )
    return predicted_t.numpy()


def resolve_start_time_minutes(raw: float | None) -> float:
    """Minute-of-day at t=0; 06:00 only when the client sent nothing."""
    if raw is None or isinstance(raw, bool):
        return 360.0
    return float(raw) % 1440.0


def build_minute_mask(
    explicit: list[float] | None,
    check_ins: list[CheckInInput],
    field: str,
    duration_min: int,
) -> torch.Tensor | None:
    """Per-minute external input for the whole window.

    An explicit per-minute array wins (padded/truncated to ``duration_min`` by
    holding the last value). Otherwise the check-ins' per-check-in value is
    forward-filled from each check-in time to the next (the first value also
    fills the minutes before the first check-in). No data -> None, so the
    model uses its learned default -- the same contract as the benchmark.
    """
    if explicit:
        vals = [float(v) for v in explicit[:duration_min]]
        if len(vals) < duration_min:
            vals = vals + [vals[-1]] * (duration_min - len(vals))
        return torch.tensor(vals, dtype=torch.float32)
    stamped: list[tuple[int, float]] = []
    for ci in check_ins:
        if ci.time is None:
            continue
        v = as_float(getattr(ci, field, None))
        if v is None:
            continue
        t = int(round(float(ci.time)))
        if 0 <= t < duration_min:
            stamped.append((t, float(v)))
    if not stamped:
        return None
    stamped.sort(key=lambda x: x[0])
    mask = np.empty(duration_min, dtype=np.float32)
    mask[: stamped[0][0]] = stamped[0][1]
    for k, (t, v) in enumerate(stamped):
        t_next = stamped[k + 1][0] if k + 1 < len(stamped) else duration_min
        mask[t:t_next] = v
    return torch.tensor(mask, dtype=torch.float32)


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
    reason: str = ""
    baseline_val_loss: float = float("nan")
    embedding_norm: float = float("nan")


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
    start_time_minutes: float = 360.0,
    sleep_wake: torch.Tensor | None = None,
    activity: torch.Tensor | None = None,
    settings: CalibrationSettings | None = None,
) -> CalibrationResult:
    """Personalize the embedding from check-ins -- the SAME algorithm as the gate.

    Iter 97 (review 5.4): this used to be a second, different calibration
    (60 steps, 80/20 split, whole-trajectory integration at 06:00, no masks).
    It now builds measured points + soft evidence from the check-ins and calls
    :func:`pulse.calibration.calibrate_embedding` with the real start time and
    per-minute masks; ``pulse.benchmark`` binds the very same function.
    """
    del user_id
    updated_baseline = estimate_updated_baseline(initial_baseline, check_ins)
    observations = build_measurement_points(check_ins, duration_min)
    soft_evidence = build_subjective_targets(check_ins, duration_min)
    initial_state_t = torch.tensor(initial_state, dtype=torch.float32)
    st = settings or CalibrationSettings.from_env()

    n_items = len(observations) + len(soft_evidence)
    if not enabled or n_items < 2:
        baseline_loss = evaluate_data_loss(
            model, initial_embedding, observations, initial_state_t, meals, duration_min,
            start_time_minutes=start_time_minutes, sleep_wake=sleep_wake, activity=activity,
            soft_evidence=soft_evidence, huber_delta=st.huber_delta,
        )
        return CalibrationResult(
            embedding=initial_embedding,
            baseline=updated_baseline,
            train_loss=baseline_loss,
            val_loss=baseline_loss,
            quality=to_quality_score(baseline_loss),
            accepted=False,
            steps=0,
            reason="disabled" if not enabled else "too_few_observations",
            baseline_val_loss=baseline_loss,
            embedding_norm=float(initial_embedding.norm()),
        )

    prior_mean = getattr(model, "_embedding_prior_mean", None)
    prior_std = getattr(model, "_embedding_prior_std", None)
    res = shared_calibrate(
        model, observations, initial_state_t, meals, duration_min,
        start_time_minutes=start_time_minutes, sleep_wake=sleep_wake, activity=activity,
        prior_mean=prior_mean, prior_std=prior_std,
        initial_embedding=initial_embedding, soft_evidence=soft_evidence, settings=st,
    )
    return CalibrationResult(
        embedding=res.embedding,
        baseline=updated_baseline if res.accepted else initial_baseline,
        train_loss=float(res.train_loss),
        val_loss=float(res.val_loss),
        quality=to_quality_score(float(res.val_loss)),
        accepted=bool(res.accepted),
        steps=int(res.n_steps),
        reason=res.reason,
        baseline_val_loss=float(res.baseline_val_loss),
        embedding_norm=res.embedding_norm,
    )


def build_measurement_points(check_ins: list[CheckInInput], duration_min: int) -> list[MeasurementPoint]:
    """One MeasurementPoint per measured marker per check-in (physical units)."""
    points: list[MeasurementPoint] = []
    for t, indices, values in build_observations(check_ins, duration_min):
        for idx, value in zip(indices, values):
            points.append(MeasurementPoint(time=int(t), marker_id=MARKER_IDS[idx], value=float(value)))
    return points


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
    start_time_minutes: float = 360.0,
    sleep_wake: torch.Tensor | None = None,
    activity: torch.Tensor | None = None,
) -> float:
    """Mean data loss of ``embedding`` on grouped observations (compat helper).

    Iter 97: delegates to :func:`pulse.calibration.evaluate_data_loss` so the
    number is the one the shared calibration optimizes. Always no-grad.
    """
    del requires_grad
    points = [
        MeasurementPoint(time=int(t), marker_id=MARKER_IDS[idx], value=float(v))
        for t, indices, values in observations for idx, v in zip(indices, values)
    ]
    return evaluate_data_loss(
        model, embedding, points, torch.tensor(initial_state, dtype=torch.float32), meals,
        duration_min, start_time_minutes=start_time_minutes, sleep_wake=sleep_wake,
        activity=activity, soft_evidence=subjective_targets,
    )


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
