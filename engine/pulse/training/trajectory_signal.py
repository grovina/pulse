"""
Cold-model trajectory distillation signal.

Owns the per-patient episode dataset and runs the per-window inner loop
(rollout, trajectory loss + soft range + optional coupling / verifier
surrogate / gut absorption / post-meal landmark losses, backward + step).
All gradient sources that share the same per-window rollout live here so we
don't pay for the forward pass twice.

Trajectory loss supports a ``trajectory_band``: per-step residuals within
±band (in normalized σ units) carry zero loss, only excursions outside the
band cost. This relaxes pure waveform imitation so the model can deviate
point-wise as long as the cold-model shape is broadly preserved. Landmark
distillation supplies the complementary "match the qualitative shape"
gradient via Δpeak / time-to-peak / AUC on glucose & insulin around each
carb meal.

In addition to per-patient episodes, the signal can include
``n_default_patients`` "default patient" episodes generated with the
cold-model defaults (``PatientParams()``) and supervised through the zero
("default") embedding. The textbook benchmark always queries the model at
the zero embedding, so it must be in the trajectory training distribution
or the zero-embedding rollouts diverge from cold-model physiology no matter
how well patient-conditioned losses do.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..coupling_prior_loss import coupling_prior_loss_on_window, merge_coupling_priors
from ..knowledge import ALL_CONTRIBUTIONS, FullBody
from ..knowledge.base import CouplingPrior, Episode
from ..knowledge.full_body import (
    PatientParams,
    generate_activity,
    generate_meal_plan,
    generate_sleep_wake,
    simulate_full_body,
)
from ..landmarks import post_meal_landmark_loss
from ..model import integrate, precompute_gut_outputs
from ..modules.gut import GUT_OUTPUT_SCALE, MealEvent
from ..training_verifier_loss import training_verifier_surrogate_loss
from ..types import EMBEDDING_DIM, MARKERS, NORM_CENTER, NORM_SCALE, STATE_DIM
from .safe_step import safe_step
from .signals import SignalContext, SignalResult, TrainingSignal, WeightSchedule

TRAIN_WINDOW = 240
SOFT_RANGE_REG = 0.001


def sample_window_start(
    n_steps: int,
    patient_meals: list[tuple[float, float, float, float]],
    rng: np.random.Generator,
    *,
    window: int = TRAIN_WINDOW,
    meal_bias_prob: float = 0.0,
) -> int:
    """Pick a training window start index, optionally biased to post-meal regions.

    With probability ``meal_bias_prob``, prefer windows where a carb meal sits
    in the verifier-friendly interior so post-meal dynamics receive gradient.
    """
    max_start = max(1, n_steps - window)
    if meal_bias_prob > 0.0 and patient_meals:
        carb_times = [
            int(round(float(t)))
            for t, c, _f, _p in patient_meals
            if float(c) > 0.0
        ]
        if carb_times and rng.random() < meal_bias_prob:
            rng.shuffle(carb_times)
            for m in carb_times:
                lo = max(0, m - window + 121)
                hi = min(max_start - 1, m - 15)
                if lo <= hi:
                    return int(rng.integers(lo, hi + 1))
    return int(rng.integers(0, max_start))


# Favor full-body episodes; override at construction.
DEFAULT_CONTRIBUTION_WEIGHTS: dict[str, float] = {
    "full_body": 0.45,
    "bergman_glucose_insulin": 0.22,
    "cortisol_circadian": 0.18,
    "cardiovascular_dynamics": 0.15,
}


def normalize_contribution_weights(
    weight_by_name: dict[str, float] | None,
) -> tuple[list[str], np.ndarray]:
    names = [c.name for c in ALL_CONTRIBUTIONS]
    if weight_by_name is None:
        w = np.array([float(DEFAULT_CONTRIBUTION_WEIGHTS.get(n, 1.0)) for n in names], dtype=np.float64)
    else:
        w = np.array([float(weight_by_name.get(n, 0.0)) for n in names], dtype=np.float64)
    if w.sum() <= 0:
        w = np.ones(len(ALL_CONTRIBUTIONS), dtype=np.float64)
    w = w / w.sum()
    return names, w


def _default_patient_episode(n_days: int, prng: np.random.Generator) -> Episode:
    """Cold-model rollout for the *default* ``PatientParams`` (no per-patient
    randomization). Meals / sleep / activity are still sampled randomly so
    different default episodes cover varied protocol contexts.

    These episodes drive the zero ("default") embedding's trajectory training,
    keeping it inside the cold-model distribution the textbook benchmark
    queries it against.
    """
    params = PatientParams()
    start_hour = 6.0
    duration_min = n_days * 1440
    meals = generate_meal_plan(n_days, prng, start_hour)
    sleep_wake = generate_sleep_wake(n_days, duration_min, start_hour, prng)
    activity = generate_activity(n_days, duration_min, start_hour, prng)
    trajectory, absorption_profile = simulate_full_body(
        params, meals, sleep_wake, activity,
        duration_min, start_hour, rng=prng,
    )
    return Episode(
        trajectory=trajectory,
        meals=meals,
        duration_min=duration_min,
        start_hour=start_hour,
        sleep_wake=sleep_wake,
        activity=activity,
        absorption_profile=absorption_profile,
        source="full_body_default",
    )


def generate_trajectory_dataset(
    n_patients: int,
    seed: int,
    n_days: int,
    weight_by_name: dict[str, float] | None,
    n_default_patients: int = 0,
) -> list[dict]:
    """One episode per virtual patient by sampling a knowledge contribution.

    When ``n_default_patients > 0``, append that many extra default-patient
    episodes (cold model with ``PatientParams()``) marked ``is_default=True``
    so the inner loop supervises them via the zero embedding instead of a
    learned patient embedding.
    """
    rng = np.random.default_rng(seed)
    _, probs = normalize_contribution_weights(weight_by_name)
    contribs = ALL_CONTRIBUTIONS
    dataset: list[dict] = []

    for pid in range(n_patients):
        prng = np.random.default_rng(rng.integers(0, 2**32))
        ci = int(rng.choice(len(contribs), p=probs))
        contrib = contribs[ci]
        if contrib.name == "full_body":
            episodes = FullBody(n_days=n_days).generate_episodes(1, prng)
        else:
            episodes = contrib.generate_episodes(1, prng)
        ep = episodes[0]

        dataset.append({
            "patient_id": pid,
            "is_default": False,
            "trajectory": ep.trajectory,
            "meals": ep.meals,
            "duration_min": ep.duration_min,
            "start_hour": ep.start_hour,
            "sleep_wake": ep.sleep_wake,
            "activity": ep.activity,
            "absorption_profile": ep.absorption_profile,
            "knowledge_source": contrib.name,
            "trajectory_loss_mode": contrib.trajectory_loss_mode(),
        })

    for di in range(n_default_patients):
        prng = np.random.default_rng(rng.integers(0, 2**32))
        ep = _default_patient_episode(n_days, prng)
        dataset.append({
            "patient_id": -(di + 1),
            "is_default": True,
            "trajectory": ep.trajectory,
            "meals": ep.meals,
            "duration_min": ep.duration_min,
            "start_hour": ep.start_hour,
            "sleep_wake": ep.sleep_wake,
            "activity": ep.activity,
            "absorption_profile": ep.absorption_profile,
            "knowledge_source": ep.source,
            "trajectory_loss_mode": "mse",
        })

    return dataset


@dataclass
class TrajectoryRolloutSignal(TrainingSignal):
    """Per-window distillation against cold-model trajectories.

    Bundles every loss that consumes the same forward rollout so the
    integrate() call is paid once per window.
    """

    n_patients: int
    n_days: int
    seed: int
    contribution_weights: dict[str, float] | None
    windows_per_patient: int
    meal_window_bias: float
    input_dropout: float
    huber_delta: float
    gut_loss_weight: float
    coupling_weight: WeightSchedule
    verifier_weight: WeightSchedule
    coupling_prior_samples: int = 3
    # Banded distillation: residuals within ±band (in normalized units) carry
    # zero loss; only excursions outside the band cost. 0 = pure imitation.
    # ``trajectory_band`` applies to per-patient (sampled embedding) episodes
    # where some divergence from the cold model is desirable so per-patient
    # variation isn't penalized. ``trajectory_band_default`` applies to
    # default-patient episodes (zero embedding) where the goal *is* exact
    # imitation — the zero embedding has no patient identity to preserve and
    # the band otherwise lets baseline drift accumulate (iter 11: +63 mg/dL
    # of unforced glucose drift in 4h fasted at zero).
    trajectory_band: float = 0.0
    trajectory_band_default: float = 0.0
    # Landmark distillation: per-meal Δpeak / time-to-peak / AUC supervision
    # on glucose+insulin around each carb meal in the window.
    landmark_weight: WeightSchedule = field(default_factory=lambda: WeightSchedule(0.0))
    landmark_pre_window: int = 15
    landmark_post_window: int = 120
    landmark_min_carbs: float = 5.0
    # Default-patient distillation: extra cold-model episodes supervised
    # through the zero ("default") embedding. Aligns the trajectory training
    # distribution with the textbook benchmark, which queries every scenario
    # at the zero embedding.
    n_default_patients: int = 0

    name: str = "trajectory_rollout"
    source: str = "cold_models"
    category: str = "trajectory"

    def __post_init__(self) -> None:
        self._dataset = generate_trajectory_dataset(
            n_patients=self.n_patients,
            seed=self.seed,
            n_days=self.n_days,
            weight_by_name=self.contribution_weights,
            n_default_patients=self.n_default_patients,
        )
        from ..knowledge.coupling_priors import ALL_COUPLING_PRIORS

        self._priors: list[CouplingPrior] = merge_coupling_priors(
            ALL_CONTRIBUTIONS, extra_priors=ALL_COUPLING_PRIORS,
        )
        self._typicals = torch.tensor(
            [m.typical for m in MARKERS], dtype=torch.float32,
        )
        self._norm_scales = torch.tensor(NORM_SCALE, dtype=torch.float32)
        self._abs_scale = torch.tensor(GUT_OUTPUT_SCALE, dtype=torch.float32)

    @property
    def dataset(self) -> list[dict]:
        return self._dataset

    @property
    def priors(self) -> list[CouplingPrior]:
        return self._priors

    def weight_at(self, epoch: int) -> float:
        # Trajectory loss is always on; sub-weights handled internally.
        return 1.0

    def compute(
        self,
        model: nn.Module,
        embeddings: nn.Embedding,
        ctx: SignalContext,
    ) -> SignalResult:
        device = ctx.device
        rng = ctx.rng
        norm_scales = self._norm_scales.to(device)
        typicals = self._typicals.to(device)
        abs_scale = self._abs_scale.to(device)
        cpl_w = self.coupling_weight.at(ctx.epoch)
        ver_w = self.verifier_weight.at(ctx.epoch)
        lm_w = self.landmark_weight.at(ctx.epoch)
        band_patient = float(self.trajectory_band)
        band_default = float(self.trajectory_band_default)

        loss_sum = 0.0
        gut_sum = 0.0
        coupling_sum = 0.0
        verifier_sum = 0.0
        landmark_sum = 0.0
        landmark_meals = 0
        n_windows = 0

        for patient_idx, patient_data in enumerate(self._dataset):
            is_default = bool(patient_data.get("is_default", False))
            trajectory_np = patient_data["trajectory"]
            patient_meals = patient_data["meals"]
            n_steps = patient_data["duration_min"]
            start_hour = patient_data["start_hour"]
            sleep_wake_np = patient_data["sleep_wake"]
            activity_np = patient_data["activity"]
            absorption_np = patient_data["absorption_profile"]
            traj_mode = patient_data.get("trajectory_loss_mode", "mse")
            if is_default:
                pid_tensor = None
                pid_for_abort = -1  # default-patient sentinel
            else:
                pid_tensor = torch.tensor(patient_data["patient_id"], device=device)
                pid_for_abort = int(patient_data["patient_id"])

            for window_idx in range(self.windows_per_patient):
                win_start = sample_window_start(
                    n_steps,
                    patient_meals,
                    rng,
                    window=TRAIN_WINDOW,
                    meal_bias_prob=self.meal_window_bias,
                )
                win_end = min(win_start + TRAIN_WINDOW, n_steps)
                win_steps = win_end - win_start

                init_row = trajectory_np[win_start].astype(np.float64)
                for mi in range(STATE_DIM):
                    if np.isnan(init_row[mi]):
                        init_row[mi] = NORM_CENTER[mi]
                initial_state = torch.tensor(init_row, dtype=torch.float32, device=device)
                win_time = (start_hour * 60 + win_start) % 1440
                if is_default:
                    embedding = torch.zeros(EMBEDDING_DIM, device=device)
                else:
                    assert pid_tensor is not None
                    embedding = embeddings(pid_tensor)

                win_meals = [
                    MealEvent(time=t - win_start, carbs=c, fats=f, proteins=p)
                    for t, c, f, p in patient_meals
                    if win_start - 120 <= t < win_end
                ]

                sw_tensor = None
                act_tensor = None
                if sleep_wake_np is not None and rng.random() > self.input_dropout:
                    sw_tensor = torch.tensor(
                        sleep_wake_np[win_start:win_end],
                        dtype=torch.float32, device=device,
                    )
                if activity_np is not None and rng.random() > self.input_dropout:
                    act_tensor = torch.tensor(
                        activity_np[win_start:win_end],
                        dtype=torch.float32, device=device,
                    )

                # Precompute gut over the whole window once — vectorized
                # forward_window across T meals, instead of one
                # GutModule.forward call per integrate step. The same tensor
                # also feeds the in-window gut_loss target below, so we
                # don't pay the gut kernel twice.
                gut_window = precompute_gut_outputs(
                    model, embedding, win_steps,
                    dt=1.0, start_time_minutes=win_time,
                    meals=win_meals,
                )
                pred_traj = integrate(
                    model, initial_state, embedding, win_steps,
                    dt=1.0, start_time_minutes=win_time,
                    meals=win_meals,
                    sleep_wake=sw_tensor,
                    activity=act_tensor,
                    gut_outputs=gut_window,
                )

                target = torch.tensor(
                    trajectory_np[win_start:win_end],
                    dtype=torch.float32, device=device,
                )
                mask = ~torch.isnan(target)
                diff = (pred_traj - target) / norm_scales
                diff = torch.where(mask, diff, torch.zeros_like(diff))
                denom = mask.float().sum().clamp(min=1.0)
                # Banded distillation: only excursions outside ±band cost.
                # band=0 reduces to pure imitation (identical to plain MSE/Huber).
                band = band_default if is_default else band_patient
                excess = F.relu(diff.abs() - band)
                if traj_mode == "huber":
                    d = torch.tensor(self.huber_delta, device=device, dtype=diff.dtype)
                    quad = 0.5 * excess.pow(2)
                    lin = d * (excess - 0.5 * d)
                    per = torch.where(excess < d, quad, lin)
                    loss = per.sum() / denom
                else:
                    loss = excess.pow(2).sum() / denom

                deviation = ((pred_traj - typicals) / norm_scales).pow(2).mean()
                loss = loss + SOFT_RANGE_REG * deviation

                cpl_component = 0.0
                if cpl_w > 0 and self._priors:
                    cpl = coupling_prior_loss_on_window(
                        model,
                        pred_traj,
                        embedding,
                        float(win_time),
                        win_meals,
                        sw_tensor,
                        act_tensor,
                        self._priors,
                        n_samples=self.coupling_prior_samples,
                    )
                    loss = loss + cpl_w * cpl
                    cpl_component = float(cpl.detach().item())
                    coupling_sum += cpl_component

                vloss_component = 0.0
                if ver_w > 0:
                    vloss = training_verifier_surrogate_loss(
                        pred_traj,
                        meals=win_meals,
                        start_hour=float(start_hour),
                        timeline_offset_min=float(win_start),
                    )
                    loss = loss + ver_w * vloss
                    vloss_component = float(vloss.detach().item())
                    verifier_sum += vloss_component

                gut_loss = torch.tensor(0.0, device=device)
                if absorption_np is not None and win_meals and self.gut_loss_weight > 0:
                    target_abs = torch.tensor(
                        absorption_np[win_start:win_end],
                        dtype=torch.float32, device=device,
                    )
                    # Reuse the precomputed gut window — same kernel call
                    # would otherwise run a second time here.
                    gut_loss = ((gut_window - target_abs) / abs_scale).pow(2).mean()
                    loss = loss + self.gut_loss_weight * gut_loss

                lm_component = 0.0
                lm_meals_window = 0
                if lm_w > 0.0 and win_meals:
                    lm_loss, lm_n = post_meal_landmark_loss(
                        pred_traj, target, win_meals,
                        pre_window=self.landmark_pre_window,
                        post_window=self.landmark_post_window,
                        min_carbs=self.landmark_min_carbs,
                    )
                    if lm_n > 0:
                        loss = loss + lm_w * lm_loss
                        lm_component = float(lm_loss.detach().item())
                        lm_meals_window = lm_n
                        landmark_sum += lm_component
                        landmark_meals += lm_n

                # Iter 25: strict abort. Per-window context goes into ``extra``
                # so the abort dump pinpoints the offending (patient, window,
                # win_start, sub-component breakdown) — iter 24's silent
                # per-window skip lost exactly this signal.
                safe_step(
                    loss,
                    ctx,
                    signal=f"{self.name}/window",
                    extra={
                        "patient_idx": float(patient_idx),
                        "patient_id": float(pid_for_abort),
                        "is_default": float(is_default),
                        "window_idx": float(window_idx),
                        "win_start": float(win_start),
                        "win_end": float(win_end),
                        "win_steps": float(win_steps),
                        "n_meals_in_window": float(len(win_meals)),
                        "traj_mode": 1.0 if traj_mode == "huber" else 0.0,
                        "band": float(band),
                        "gut_loss": float(gut_loss.detach().item()),
                        "coupling_loss": cpl_component,
                        "verifier_loss": vloss_component,
                        "landmark_loss": lm_component,
                        "landmark_meals_window": float(lm_meals_window),
                    },
                )
                loss_sum += float(loss.detach().item())
                gut_sum += float(gut_loss.detach().item())
                n_windows += 1

        sub: dict[str, float] = {
            "gut": gut_sum / max(n_windows, 1),
        }
        if cpl_w > 0 and self._priors:
            sub["coupling"] = coupling_sum / max(n_windows, 1)
        if ver_w > 0:
            sub["verifier_surrogate"] = verifier_sum / max(n_windows, 1)
        if lm_w > 0:
            sub["landmark"] = landmark_sum / max(n_windows, 1)
            sub["landmark_meals"] = float(landmark_meals)

        return SignalResult(
            loss_sum=loss_sum,
            n_units=n_windows,
            sub_metrics=sub,
        )
