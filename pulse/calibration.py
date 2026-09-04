"""
Embedding calibration -- ONE algorithm for the benchmark gate and the product.

Iter 97 (review 2026-09-04, 5.3 / 5.4). Until now the gate and ``server.py``
ran two different calibrations: 512 Adam steps vs 60, MSE vs Huber, no hold-out
vs an 80/20 split, a hard ``||emb|| <= 3`` clamp vs a soft prior, windowed vs
whole-trajectory integration; the server also hard-coded ``start_time_minutes
= 360`` and dropped ``sleepWake`` / ``activity`` from check-ins. The gate was
certifying a calibration the product never ran, and the gate's own number was
a function of where the 512 steps happened to stop (gabriel-night-03: held-out
hr MAPE across steps 1..40 read 0.034, 0.159, 0.214, 0.208, 0.058, 0.030, ...).

What the PRD asks for (Calibration; Epistemological humility):

* calibration adjusts ONLY the embedding, never the weights;
* every datapoint is evidence, not an assertion: a robust (Huber) residual,
  a prior toward the trained population, "no single observation should cause
  a large model update";
* accept an update only when held-out evidence improves.

So ``calibrate_embedding`` here:

1. integrates the model ONCE, continuously, from t=0 to the last check-in in
   the episode frame with the real start time and the per-minute sleep/activity
   masks -- the same forward map the scorer runs (review 1.3);
2. holds out the LAST ~20% of check-in times (chronological: the product
   predicts forward, so that is the honest split), scores the held-out
   residuals with no gradient at every step, stops early with patience, and
   returns the initial embedding unless the held-out loss improved;
3. shrinks toward the trained prior mean with a NON-ZERO default weight
   (0.25, ``PULSE_BENCHMARK_PRIOR_WEIGHT``; the sweep tunes it), replaces the
   hard 3.0 clamp with a soft norm penalty beyond a radius, and keeps only a
   much larger hard clamp as protection against integrator blow-up.

Every knob is explicit on :class:`CalibrationSettings` and every one has an
env override so the benchmark's process pool, the sweep and the server can be
pointed at the same numbers. ``settings.as_dict()`` goes into the report's
ruler fingerprint.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Sequence

import torch

from .model import ModularPhysiologyNetwork, integrate
from .modules.gut import MEAL_ACTIVE_WINDOW_MIN, MealEvent
from .types import EMBEDDING_DIM, MARKER_INDEX, NORM_SCALE


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v else default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v else default


@dataclass(frozen=True)
class MeasurementPoint:
    """A measured marker value at an episode minute (physical units)."""
    time: int
    marker_id: str
    value: float


@dataclass(frozen=True)
class SoftEvidence:
    """A one-sided, uncertain observation ("hungry" -> ghrelin probably above 95).

    ``specificity`` weights the term; ``scale`` is the softness of the margin
    in marker units. Both come from the server's evidence templates.
    """
    time: int
    marker_id: str
    direction: str  # "above" | "below"
    threshold: float
    specificity: float
    scale: float


@dataclass(frozen=True)
class CalibrationSettings:
    """Every eval-time calibration knob, in one place, env-overridable."""

    max_steps: int = 512
    lr: float = 0.05
    # Diagonal-Gaussian prior toward the trained table (mean_d ((e-mu)/sigma)^2),
    # used when the checkpoint carries embedding_prior_{mean,std}. The iter-91
    # sweeps that set this to 0 were run through the windowed forward map
    # (review 1.3) and with the hard clamp doing the prior's job; re-swept
    # after both are gone. 0.25 is the starting point, not a measurement.
    prior_weight: float = 0.25
    # Isotropic L2 toward the initial embedding when there is no trained prior.
    l2_weight: float = 0.003
    # Soft norm penalty  w * relu(||e|| - radius)^2  replaces the hard 3.0 clamp.
    soft_norm_weight: float = 0.1
    soft_norm_radius: float = 3.0
    # Hard clamp kept ONLY against integrator blow-up (iter 81 measured the
    # forward pass detonating around ||e|| ~ 10.8). <= 0 disables.
    max_norm: float = 8.0
    # Chronological hold-out of the last fraction of check-in TIMES. 0 disables
    # hold-out and acceptance (fixed-step legacy behaviour, used by the
    # Bayesian/Laplace path that needs a MAP point regardless).
    holdout_fraction: float = 0.2
    min_check_in_times: int = 2
    patience: int = 16
    # Huber transition in NORM_SCALE units (residual / NORM_SCALE). Quadratic
    # and equal to the legacy MSE inside +-delta, linear beyond. inf = MSE.
    huber_delta: float = 1.0
    # Accept the calibrated embedding only if held-out loss fell by this fraction.
    accept_rel_improvement: float = 0.01
    checkpoint_segments: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls, **overrides: Any) -> "CalibrationSettings":
        base = cls(
            max_steps=_env_int("PULSE_BENCHMARK_CALIBRATE_STEPS", cls.max_steps),
            lr=_env_float("PULSE_BENCHMARK_CALIBRATE_LR", cls.lr),
            prior_weight=_env_float("PULSE_BENCHMARK_PRIOR_WEIGHT", cls.prior_weight),
            l2_weight=_env_float("PULSE_BENCHMARK_CALIBRATE_L2", cls.l2_weight),
            soft_norm_weight=_env_float("PULSE_BENCHMARK_SOFT_NORM_WEIGHT", cls.soft_norm_weight),
            soft_norm_radius=_env_float("PULSE_BENCHMARK_SOFT_NORM_RADIUS", cls.soft_norm_radius),
            max_norm=_env_float("PULSE_BENCHMARK_CALIBRATE_MAX_NORM", cls.max_norm),
            holdout_fraction=_env_float("PULSE_BENCHMARK_HOLDOUT_FRACTION", cls.holdout_fraction),
            patience=_env_int("PULSE_BENCHMARK_CALIBRATE_PATIENCE", cls.patience),
            huber_delta=_env_float("PULSE_BENCHMARK_HUBER_DELTA", cls.huber_delta),
            accept_rel_improvement=_env_float(
                "PULSE_BENCHMARK_ACCEPT_REL_IMPROVEMENT", cls.accept_rel_improvement),
        )
        return replace(base, **overrides) if overrides else base

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["forward_map"] = "continuous [0, last_check_in] in the episode frame"
        d["loss"] = "huber(residual / NORM_SCALE) + prior + soft norm; hold-out on last check-in times"
        return d


@dataclass
class CalibrationResult:
    embedding: torch.Tensor
    accepted: bool
    reason: str
    n_steps: int
    best_step: int
    train_loss: float
    val_loss: float
    baseline_train_loss: float
    baseline_val_loss: float
    final_loss: float
    n_train_obs: int
    n_val_obs: int
    embedding_norm: float
    initial_norm: float
    settings: dict[str, Any]

    def as_report(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted, "reason": self.reason,
            "steps": self.n_steps, "best_step": self.best_step,
            "train_loss": self.train_loss, "val_loss": self.val_loss,
            "baseline_train_loss": self.baseline_train_loss,
            "baseline_val_loss": self.baseline_val_loss,
            "n_train_obs": self.n_train_obs, "n_val_obs": self.n_val_obs,
            "embedding_norm": self.embedding_norm, "initial_norm": self.initial_norm,
        }


# --------------------------------------------------------------------------- pieces


def active_meals(meals: Sequence[MealEvent], t_start: float, t_end: float) -> list[MealEvent]:
    """Meals whose absorption can touch ``[t_start, t_end)`` -- lookback is the gut
    kernel's own active window (review 1.4), not a hand-written constant."""
    return [m for m in meals if m.time < t_end and m.time + MEAL_ACTIVE_WINDOW_MIN >= t_start]


def huber(residual: torch.Tensor, delta: float) -> torch.Tensor:
    """Elementwise robust loss that EQUALS the squared residual inside +-delta
    (so the legacy MSE scale, and the prior weights swept against it, carry
    over) and grows linearly beyond: ``delta * (2|r| - delta)``."""
    if not math.isfinite(delta):
        return residual.pow(2)
    a = residual.abs()
    return torch.where(a <= delta, residual.pow(2), delta * (2.0 * a - delta))


def split_check_ins(
    times: Sequence[int], holdout_fraction: float, min_times: int = 2,
) -> tuple[set[int], set[int]]:
    """Chronological split of DISTINCT check-in times: the last ~fraction is held out.

    Returns ``(train_times, val_times)``. With too few distinct times, or a
    non-positive fraction, everything is train and ``val_times`` is empty.
    """
    distinct = sorted(set(int(t) for t in times))
    if holdout_fraction <= 0.0 or len(distinct) < max(2, min_times):
        return set(distinct), set()
    n_val = int(round(holdout_fraction * len(distinct)))
    n_val = max(1, min(n_val, len(distinct) - 1))
    return set(distinct[:-n_val]), set(distinct[-n_val:])


def _forward(
    embedding: torch.Tensor,
    *,
    model: ModularPhysiologyNetwork,
    n_steps: int,
    initial_state: torch.Tensor,
    meals: Sequence[MealEvent],
    start_time_minutes: float,
    sleep_wake: torch.Tensor | None,
    activity: torch.Tensor | None,
    checkpoint_segments: int,
) -> torch.Tensor:
    return integrate(
        model=model,
        initial_state=initial_state,
        embedding=embedding,
        n_steps=n_steps,
        dt=1.0,
        start_time_minutes=start_time_minutes,
        meals=active_meals(meals, 0.0, float(n_steps)),
        sleep_wake=sleep_wake[:n_steps] if sleep_wake is not None else None,
        activity=activity[:n_steps] if activity is not None else None,
        checkpoint_segments=checkpoint_segments,
    )


def _data_terms(
    predicted: torch.Tensor,
    observations: Sequence[MeasurementPoint],
    soft: Sequence[SoftEvidence],
    norm_scale: torch.Tensor,
    huber_delta: float,
) -> tuple[torch.Tensor, int]:
    """Sum of per-observation losses and the count (so callers take the mean)."""
    terms: list[torch.Tensor] = []
    if observations:
        times = torch.tensor([o.time for o in observations], dtype=torch.long)
        idxs = torch.tensor([MARKER_INDEX[o.marker_id] for o in observations], dtype=torch.long)
        targets = torch.tensor([o.value for o in observations], dtype=torch.float32)
        resid = (predicted[times, idxs] - targets) / norm_scale[idxs]
        terms.append(huber(resid, huber_delta).sum())
    for ev in soft:
        idx = MARKER_INDEX.get(ev.marker_id)
        if idx is None:
            continue
        pred = predicted[ev.time, idx]
        thr = torch.tensor(float(ev.threshold), dtype=torch.float32)
        margin = (pred - thr) / ev.scale if ev.direction == "above" else (thr - pred) / ev.scale
        terms.append(ev.specificity * (-torch.nn.functional.logsigmoid(margin)))
    n = len(observations) + sum(1 for ev in soft if ev.marker_id in MARKER_INDEX)
    if not terms:
        return torch.tensor(0.0), 0
    return torch.stack([t.reshape(()) for t in terms]).sum(), n


def _valid_observations(observations: Sequence[MeasurementPoint], duration_min: int) -> list[MeasurementPoint]:
    return [o for o in observations
            if o.marker_id in MARKER_INDEX and 0 <= int(o.time) < duration_min]


def _valid_soft(soft: Sequence[SoftEvidence], duration_min: int) -> list[SoftEvidence]:
    return [s for s in soft if s.marker_id in MARKER_INDEX and 0 <= int(s.time) < duration_min]


# --------------------------------------------------------------------------- the algorithm


def calibrate_embedding(
    model: ModularPhysiologyNetwork,
    observations: Sequence[MeasurementPoint],
    initial_state: torch.Tensor,
    meals: Sequence[MealEvent],
    duration_min: int,
    *,
    start_time_minutes: float = 360.0,
    sleep_wake: torch.Tensor | None = None,
    activity: torch.Tensor | None = None,
    prior_mean: torch.Tensor | None = None,
    prior_std: torch.Tensor | None = None,
    initial_embedding: torch.Tensor | None = None,
    soft_evidence: Sequence[SoftEvidence] = (),
    settings: CalibrationSettings | None = None,
) -> CalibrationResult:
    """Personalize the embedding from sparse check-ins; see the module docstring.

    ``initial_embedding`` defaults to ``prior_mean`` when a trained prior is
    supplied, else to zeros. ``observations`` accept anything with
    ``time / marker_id / value`` attributes (the benchmark's MeasurementPoint
    is structurally identical).
    """
    st = settings or CalibrationSettings.from_env()
    model.eval()

    if initial_embedding is not None:
        e0 = initial_embedding.detach().clone().float()
    elif prior_mean is not None:
        e0 = prior_mean.detach().clone().float()
    else:
        e0 = torch.zeros(EMBEDDING_DIM)
    use_prior = prior_mean is not None and prior_std is not None and st.prior_weight > 0.0
    if use_prior:
        prior_mean_t = prior_mean.detach().float()
        prior_std_t = prior_std.detach().float()

    obs = _valid_observations(list(observations), duration_min)
    soft = _valid_soft(list(soft_evidence), duration_min)
    norm_scale = torch.tensor(NORM_SCALE, dtype=torch.float32)

    def _result(emb: torch.Tensor, accepted: bool, reason: str, n_steps: int, best_step: int,
                tr: float, va: float, b_tr: float, b_va: float, final: float,
                n_tr: int, n_va: int) -> CalibrationResult:
        return CalibrationResult(
            embedding=emb.detach(), accepted=accepted, reason=reason, n_steps=n_steps,
            best_step=best_step, train_loss=tr, val_loss=va, baseline_train_loss=b_tr,
            baseline_val_loss=b_va, final_loss=final, n_train_obs=n_tr, n_val_obs=n_va,
            embedding_norm=float(emb.norm()), initial_norm=float(e0.norm()),
            settings=st.as_dict(),
        )

    all_times = [o.time for o in obs] + [s.time for s in soft]
    if not all_times:
        return _result(e0, False, "no_observations", 0, 0, float("nan"), float("nan"),
                       float("nan"), float("nan"), float("nan"), 0, 0)

    train_times, val_times = split_check_ins(all_times, st.holdout_fraction, st.min_check_in_times)
    if st.holdout_fraction > 0.0 and not val_times:
        # Too few check-in times to validate: no evidence-backed update possible.
        return _result(e0, False, "too_few_check_ins", 0, 0, float("nan"), float("nan"),
                       float("nan"), float("nan"), float("nan"), len(all_times), 0)

    train_obs = [o for o in obs if o.time in train_times]
    val_obs = [o for o in obs if o.time in val_times]
    train_soft = [s for s in soft if s.time in train_times]
    val_soft = [s for s in soft if s.time in val_times]
    n_train = len(train_obs) + len(train_soft)
    n_val = len(val_obs) + len(val_soft)
    n_steps_fwd = int(max(all_times)) + 1

    def regularizers(e: torch.Tensor) -> torch.Tensor:
        if use_prior:
            reg = st.prior_weight * ((e - prior_mean_t) / (prior_std_t + 1e-6)).pow(2).mean()
        else:
            reg = st.l2_weight * (e - e0).pow(2).mean()
        if st.soft_norm_weight > 0.0:
            reg = reg + st.soft_norm_weight * torch.relu(e.norm() - st.soft_norm_radius).pow(2)
        return reg

    emb = e0.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([emb], lr=st.lr)

    best_val = float("inf")
    best_train = float("nan")
    best_step = 0
    best_emb = e0.clone()
    baseline_train = float("nan")
    baseline_val = float("nan")
    final_loss = float("nan")
    no_improvement = 0
    steps_run = 0

    for step in range(st.max_steps + 1):
        # Iteration k evaluates e_k (train WITH grad, held-out WITHOUT), records
        # it, then takes the Adam step that produces e_{k+1}. k = 0 is the
        # baseline. The forward pass is shared: hold-out times are later than
        # train times, so one integration to the last check-in covers both.
        with torch.enable_grad():
            predicted = _forward(
                emb, model=model, n_steps=n_steps_fwd, initial_state=initial_state,
                meals=meals, start_time_minutes=start_time_minutes,
                sleep_wake=sleep_wake, activity=activity,
                checkpoint_segments=st.checkpoint_segments,
            )
            train_sum, n_tr = _data_terms(predicted, train_obs, train_soft, norm_scale, st.huber_delta)
            train_data = train_sum / max(n_tr, 1)
            objective = train_data + regularizers(emb)
        with torch.no_grad():
            if n_val > 0:
                val_sum, n_va = _data_terms(predicted.detach(), val_obs, val_soft, norm_scale, st.huber_delta)
                val_data = float(val_sum) / max(n_va, 1)
            else:
                val_data = float(train_data)  # no hold-out: track the train objective
        train_val = float(train_data.detach())
        if step == 0:
            baseline_train, baseline_val = train_val, val_data
        if val_data < best_val - 1e-12:
            best_val, best_train, best_step = val_data, train_val, step
            best_emb = emb.detach().clone()
            no_improvement = 0
        else:
            no_improvement += 1
        final_loss = float(objective)
        if step == st.max_steps:
            break
        if n_val > 0 and st.patience > 0 and no_improvement >= st.patience:
            break
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()
        steps_run = step + 1
        if st.max_norm > 0.0:
            with torch.no_grad():
                norm = float(emb.norm())
                if norm > st.max_norm:
                    emb.mul_(st.max_norm / norm)

    if n_val == 0:
        # Legacy / MAP mode: no hold-out, the final embedding is the answer.
        return _result(emb.detach(), True, "no_holdout", steps_run, steps_run, train_val,
                       val_data, baseline_train, baseline_val, final_loss, n_train, 0)

    improved = best_step > 0 and best_val <= baseline_val * (1.0 - st.accept_rel_improvement)
    if improved:
        return _result(best_emb, True, "accepted", steps_run, best_step, best_train, best_val,
                       baseline_train, baseline_val, final_loss, n_train, n_val)
    return _result(e0, False, "no_improvement", steps_run, 0, baseline_train, baseline_val,
                   baseline_train, baseline_val, final_loss, n_train, n_val)


def evaluate_data_loss(
    model: ModularPhysiologyNetwork,
    embedding: torch.Tensor,
    observations: Sequence[MeasurementPoint],
    initial_state: torch.Tensor,
    meals: Sequence[MealEvent],
    duration_min: int,
    *,
    start_time_minutes: float = 360.0,
    sleep_wake: torch.Tensor | None = None,
    activity: torch.Tensor | None = None,
    soft_evidence: Sequence[SoftEvidence] = (),
    huber_delta: float = 1.0,
) -> float:
    """Mean data loss of ``embedding`` on ``observations`` (no gradient, no prior)."""
    obs = _valid_observations(list(observations), duration_min)
    soft = _valid_soft(list(soft_evidence), duration_min)
    times = [o.time for o in obs] + [s.time for s in soft]
    if not times:
        return 0.0
    norm_scale = torch.tensor(NORM_SCALE, dtype=torch.float32)
    with torch.no_grad():
        predicted = _forward(
            embedding.detach().float(), model=model, n_steps=int(max(times)) + 1,
            initial_state=initial_state, meals=meals, start_time_minutes=start_time_minutes,
            sleep_wake=sleep_wake, activity=activity, checkpoint_segments=0,
        )
        total, n = _data_terms(predicted, obs, soft, norm_scale, huber_delta)
    return float(total) / max(n, 1)


__all__ = [
    "CalibrationResult",
    "CalibrationSettings",
    "MeasurementPoint",
    "SoftEvidence",
    "active_meals",
    "calibrate_embedding",
    "evaluate_data_loss",
    "huber",
    "split_check_ins",
]
