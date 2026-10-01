"""
Cold-model trajectory distillation signal.

Owns the per-patient episode dataset and runs the per-window inner loop
(rollout, trajectory loss + soft range + optional coupling / verifier
surrogate / gut absorption, backward + step).
All gradient sources that share the same per-window rollout live here so we
don't pay for the forward pass twice.

Trajectory loss supports a ``trajectory_band``: per-step residuals within
±band (in normalized σ units) carry zero loss, only excursions outside the
band cost. This relaxes pure waveform imitation so the model can deviate
point-wise as long as the cold-model shape is broadly preserved.

In addition to per-patient episodes, the signal can include
``n_default_patients`` "default patient" episodes generated with the
cold-model defaults (``PatientParams()``) and supervised through the zero
("default") embedding. The textbook benchmark always queries the model at
the zero embedding, so it must be in the trajectory training distribution
or the zero-embedding rollouts diverge from cold-model physiology no matter
how well patient-conditioned losses do.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..coupling_prior_loss import coupling_prior_loss_on_window, merge_coupling_priors
from ..knowledge import ALL_CONTRIBUTIONS, FullBody
from ..knowledge.base import CouplingPrior, Episode
from ..knowledge.evidence import TEACHER_TAPE_MARKERS, mask_trajectory
from ..knowledge.full_body import (
    PatientParams,
    generate_activity,
    generate_meal_plan,
    generate_sleep_wake,
    simulate_full_body,
)
from ..model import integrate, precompute_duodenal_outputs, precompute_gut_outputs
from ..modules.base import GutModuleBase
from ..modules.gut import GUT_OUTPUT_SCALE, MEAL_ACTIVE_WINDOW_MIN, MealEvent
from ..training_verifier_loss import training_verifier_surrogate_loss
from ..types import EMBEDDING_DIM, MARKERS, MARKER_INDEX, NORM_CENTER, NORM_SCALE, STATE_DIM

MARKER_INDEX_IDS = set(MARKER_INDEX)
from .safe_step import safe_step
from .signals import SignalContext, SignalResult, TrainingSignal, WeightSchedule

TRAIN_WINDOW = 240
# Iter 97 (review 4.2): the soft-range term is a CATASTROPHE FENCE, not a pull.
# It used to be ``0.001 * mean(((pred - typical)/scale)^2)`` at every step, i.e.
# every marker was pulled toward its population typical inside the band the
# trajectory loss had just declared free — a fasted BHB of 1.3 mmol/L (2.4 sigma
# off typical) paid for being right. Now it is a dead-zone hinge: nothing inside
# +/- SOFT_RANGE_DEADZONE normalized units of typical, quadratic beyond (the
# integrator's hard clamp is at 20). Every physiological excursion in the
# teacher's pool (48 h-fast BHB 3.5 = 6.8 sigma is the largest) sits inside 8.
SOFT_RANGE_REG = 0.001
SOFT_RANGE_DEADZONE = 8.0

# Iter 97 (review 4.2): per-marker trajectory bands, in NORM_SCALE units. The
# band is the teacher's uncertainty — the PRD's "fence": a residual inside it
# costs nothing. The observed vitals are what the ruler scores against real
# data and where the teacher is best (band 0.15 = 4.5 mg/dL glucose, 1.5 bpm);
# every other marker is an unobserved hormone or pool the teacher gets
# approximately right (0.30 = 2.4 ug/dL cortisol, 12 pg/mL ghrelin). The old
# single band 0.08 was below the CGM noise floor on glucose (2.4 mg/dL) and
# demanded 0.024 C on temperature.
# Wearable-frequency vitals the teacher tape is allowed to supervise.
# Same set as TEACHER_TAPE_MARKERS; imported name kept for the band map.
OBSERVED_VITALS: tuple[str, ...] = TEACHER_TAPE_MARKERS
DEFAULT_BAND_OBSERVED = 0.15
DEFAULT_BAND_UNOBSERVED = 0.30


def default_band_per_marker() -> dict[str, float]:
    return {
        m.id: (DEFAULT_BAND_OBSERVED if m.id in OBSERVED_VITALS else DEFAULT_BAND_UNOBSERVED)
        for m in MARKERS
    }


def parse_band_per_marker(raw: str | None) -> dict[str, float] | None:
    """``"glucose:0.15;hr:0.15;*:0.3"`` -> per-marker band map (``*`` = the rest).

    ``None`` / empty returns ``None`` (use the scalar bands). Unknown marker ids
    raise so a typo cannot silently leave a marker on the default.
    """
    if raw is None or not raw.strip():
        return None
    out = default_band_per_marker()
    star: float | None = None
    explicit: dict[str, float] = {}
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        name, _, val = item.partition(":")
        name = name.strip()
        band = float(val)
        if band < 0:
            raise ValueError(f"trajectory band for {name!r} must be >= 0")
        if name == "*":
            star = band
        elif name in MARKER_INDEX_IDS:
            explicit[name] = band
        else:
            raise ValueError(f"unknown marker id in --trajectory-band-per-marker: {name!r}")
    if star is not None:
        out = {k: star for k in out}
    out.update(explicit)
    return out


# Iter 97 (review 4.9): "meal logged, macros unknown". With probability
# ``meal_macro_dropout`` a window's meals keep their TIME but their macros are
# replaced by the population-typical meal (the mean of the teacher's meal
# generator: ~60 g carbohydrate, 20 g fat, 25 g protein), i.e. the student sees
# the nutrient flag and a default composition, not the true grams. The target
# trajectory still carries the true meal, so the band absorbs the difference
# and the student learns to be robust to unquantified meals — the PRD's
# "missing inputs are a normal condition". Dropping the meal entirely would
# ask the student to produce a meal response it cannot know about, which
# trains hallucination; a true "macros unknown" input flag is the student
# layer's to add.
DEFAULT_MEAL_MACROS: tuple[float, float, float] = (60.0, 20.0, 25.0)


def meals_with_default_macros(meals: list[MealEvent]) -> list[MealEvent]:
    c, f, p = DEFAULT_MEAL_MACROS
    return [MealEvent(time=m.time, carbs=c, fats=f, proteins=p) for m in meals]


def band_vector(band_map: dict[str, float]) -> torch.Tensor:
    return torch.tensor([float(band_map[m.id]) for m in MARKERS], dtype=torch.float32)


def shape_marker_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, band: torch.Tensor,
) -> torch.Tensor:
    """Window-mean + trend-sign supervision for markers with known teacher defects (review 4.2).

    ``pred`` / ``target`` are ``[T, K]`` normalized series for K such markers,
    ``mask`` marks observed target entries, ``band`` is ``[K]``. Two terms per
    marker: the window-MEAN residual (banded, squared) and a hinge on the
    TREND: the sign of (last-quarter mean - first-quarter mean) must agree with
    the teacher's, with the teacher's own trend magnitude as the margin. No
    pointwise term, so the teacher's mis-timed arc does not get copied.
    """
    T = pred.shape[0]
    m = mask.float()
    cnt = m.sum(dim=0).clamp(min=1.0)
    resid_mean = ((pred - target) * m).sum(dim=0) / cnt
    mean_term = F.relu(resid_mean.abs() - band).pow(2)
    q = max(1, T // 4)
    def _q(x: torch.Tensor, sl: slice) -> torch.Tensor:
        mm = m[sl]
        return (x[sl] * mm).sum(dim=0) / mm.sum(dim=0).clamp(min=1.0)
    t_pred = _q(pred, slice(T - q, T)) - _q(pred, slice(0, q))
    t_ref = _q(target, slice(T - q, T)) - _q(target, slice(0, q))
    sign = torch.sign(t_ref).detach()
    margin = t_ref.abs().detach()
    trend_term = F.relu(margin - t_pred * sign).pow(2)
    valid = (cnt > 1.0).float()
    return ((mean_term + trend_term) * valid).sum() / valid.sum().clamp(min=1.0)


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


def meals_in_window(
    patient_meals: list[tuple[float, float, float, float]],
    win_start: int,
    win_end: int,
    *,
    lookback_min: float = MEAL_ACTIVE_WINDOW_MIN,
) -> list[MealEvent]:
    """Meals the student can see in ``[win_start, win_end)``, on the window-offset clock.

    Iter 97 (review 1.4): a meal is visible for as long as the gut kernel is
    active (``MEAL_ACTIVE_WINDOW_MIN``, 720 min since iter 97), not 120 min. With the 120-min
    lookback, 67 % of sampled windows started from the teacher's FED state (the
    initial row carries the meal) while the inputs said "fasted"; those meals
    carried 47 % of the in-window nutrient flag. Same constant the benchmark and
    calibration use, so the three frames agree.
    """
    return [
        MealEvent(time=t - win_start, carbs=c, fats=f, proteins=p)
        for t, c, f, p in patient_meals
        if win_start - lookback_min <= t < win_end
    ]


@dataclass
class TrainingWindow:
    """One sampled training window: everything its rollout and loss read."""

    patient_idx: int
    patient_id: int  # -1 for a default patient (zero embedding)
    is_default: bool
    window_idx: int
    win_start: int
    win_end: int
    start_hour: float
    traj_mode: str
    initial_state: np.ndarray  # [STATE_DIM], NaN-free
    target: np.ndarray  # [T, STATE_DIM], NaN where the tape is unobserved
    meals: list[MealEvent]
    meals_defaulted: bool
    sleep_wake: np.ndarray | None  # [T], None when withheld (input dropout)
    activity: np.ndarray | None
    absorption: np.ndarray | None  # [T, GUT_OUTPUT_DIM] teacher appearance

    @property
    def n_steps(self) -> int:
        return self.win_end - self.win_start

    @property
    def win_time(self) -> float:
        return float((self.start_hour * 60 + self.win_start) % 1440)


# One generator. Views that used to dump dense hormone tapes are not
# training contributions.
DEFAULT_CONTRIBUTION_WEIGHTS: dict[str, float] = {
    "full_body": 1.0,
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
        trajectory=mask_trajectory(trajectory, TEACHER_TAPE_MARKERS),
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
            # Iter 90: ground-truth per-patient setpoints (None for contributions that
            # do not simulate a whole patient). Consumed by SetpointSupervisionSignal.
            "setpoints": ep.setpoints,
            # Iter 91: teacher's own standard-meal response for this patient. Consumed by
            # RolloutEvidenceSignal (family=meal) to unfreeze the per-patient meal gain Ra.
            "meal_response": ep.meal_response,
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
    # Iter 97 (review 4.2): per-marker bands (NORM_SCALE units). When set, this
    # map applies to EVERY episode — the band is the teacher's uncertainty, which
    # does not depend on whose embedding is being fitted — and the two scalar
    # bands above are ignored. ``shape_markers`` are supervised by window mean +
    # trend sign instead of pointwise (markers the teacher is known to get wrong
    # in shape: the review lists ghrelin's flat fast, the HPA rhythm, the
    # hepatic ledger).
    trajectory_band_per_marker: dict[str, float] | None = None
    shape_markers: tuple[str, ...] = ()
    # Iter 97 (review 4.9): probability that a window's meals are reduced to
    # "logged, macros unknown" (see DEFAULT_MEAL_MACROS). The trainer raises
    # this (and trajectory ``input_dropout``) in phase 3. Literature arm
    # rollouts do not read these fields.
    meal_macro_dropout: float = 0.0
    # Default-patient distillation: extra cold-model episodes supervised
    # through the zero ("default") embedding. Aligns the trajectory training
    # distribution with the textbook benchmark, which queries every scenario
    # at the zero embedding.
    n_default_patients: int = 0
    # Windows rolled out together per optimizer step (one batched ``integrate``,
    # loss = mean of the per-window losses). 1 = one step per window — the
    # historical cadence, and bit-identical to it. A rollout's cost is dominated by
    # per-step overhead, not by batch width (a B=8 rollout costs ~1.1x a B=1 one on
    # CPU), so B windows per step is ~B× the trajectory data per second; it is NOT
    # the same optimization (B× fewer, less noisy steps). See docs/training-efficiency.md.
    batch_windows: int = 1

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
        self._abs_scale = torch.tensor(
            GUT_OUTPUT_SCALE[:GutModuleBase.N_APPEARANCE], dtype=torch.float32,
        )
        self._band_vec: torch.Tensor | None = (
            band_vector(self.trajectory_band_per_marker)
            if self.trajectory_band_per_marker else None
        )
        for m in self.shape_markers:
            if m not in MARKER_INDEX:
                raise ValueError(f"unknown shape marker {m!r}")
        self._shape_idx = torch.tensor(
            [MARKER_INDEX[m] for m in self.shape_markers], dtype=torch.long,
        )
        self._pointwise_mask = torch.ones(STATE_DIM, dtype=torch.float32)
        if len(self.shape_markers):
            self._pointwise_mask[self._shape_idx] = 0.0

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
        """Run every window this epoch; see ``iter_windows`` for the interleaved form."""
        for _ in self.iter_windows(model, embeddings, ctx):
            pass
        return self.last_result

    @property
    def last_result(self) -> SignalResult:
        return getattr(self, "_last_result", SignalResult())

    def iter_windows(
        self,
        model: nn.Module,
        embeddings: nn.Embedding,
        ctx: SignalContext,
    ):
        """Generator: one optimizer step per ``yield``, which reports the cumulative
        number of windows stepped so far this epoch.

        Iter 97 (review 4.1): the trainer drives this generator and interleaves
        one joint auxiliary step every k windows, instead of running all ~96
        windows and then a single aux step per epoch (25 literature steps in a
        whole run vs ~5,300 imitation steps). ``last_result`` holds the epoch's
        aggregate once the generator is exhausted. With ``batch_windows`` B > 1
        each step covers B windows, so the yielded count advances by B.
        """
        sums: dict[str, float] = {
            "loss": 0.0, "gut": 0.0, "coupling": 0.0, "verifier": 0.0, "shape": 0.0,
        }
        n_windows = 0
        n_defaulted = 0
        cpl_w = self.coupling_weight.at(ctx.epoch)
        ver_w = self.verifier_weight.at(ctx.epoch)
        batch_size = max(1, int(self.batch_windows))
        pending: list[TrainingWindow] = []

        def _flush() -> None:
            nonlocal n_windows, n_defaulted
            parts = self._step_windows(model, embeddings, ctx, pending, cpl_w, ver_w)
            for key in sums:
                sums[key] += parts[key]
            n_windows += len(pending)
            n_defaulted += sum(int(w.meals_defaulted) for w in pending)
            pending.clear()

        for patient_idx, patient_data in enumerate(self._dataset):
            for window_idx in range(self.windows_per_patient):
                pending.append(self._sample_window(patient_idx, patient_data, window_idx, ctx.rng))
                if len(pending) == batch_size:
                    _flush()
                    yield n_windows
        if pending:
            _flush()
            yield n_windows

        sub: dict[str, float] = {
            "gut": sums["gut"] / max(n_windows, 1),
        }
        if cpl_w > 0 and self._priors:
            sub["coupling"] = sums["coupling"] / max(n_windows, 1)
        if ver_w > 0:
            sub["verifier_surrogate"] = sums["verifier"] / max(n_windows, 1)
        if len(self.shape_markers):
            sub["shape"] = sums["shape"] / max(n_windows, 1)
        sub["meals_defaulted"] = float(n_defaulted)

        self._last_result = SignalResult(
            loss_sum=sums["loss"],
            n_units=n_windows,
            sub_metrics=sub,
        )

    def _sample_window(
        self,
        patient_idx: int,
        patient_data: dict,
        window_idx: int,
        rng: np.random.Generator,
    ) -> TrainingWindow:
        """Draw one window (start, meal-macro dropout, input dropout) — the RNG
        draws, in order, that the per-window loop has always made."""
        trajectory_np = patient_data["trajectory"]
        patient_meals = patient_data["meals"]
        n_steps = patient_data["duration_min"]
        is_default = bool(patient_data.get("is_default", False))
        win_start = sample_window_start(
            n_steps,
            patient_meals,
            rng,
            window=TRAIN_WINDOW,
            meal_bias_prob=self.meal_window_bias,
        )
        win_end = min(win_start + TRAIN_WINDOW, n_steps)

        init_row = trajectory_np[win_start].astype(np.float64)
        for mi in range(STATE_DIM):
            if np.isnan(init_row[mi]):
                init_row[mi] = NORM_CENTER[mi]

        win_meals = meals_in_window(patient_meals, win_start, win_end)
        meals_defaulted = False
        if win_meals and self.meal_macro_dropout > 0.0 and rng.random() < self.meal_macro_dropout:
            win_meals = meals_with_default_macros(win_meals)
            meals_defaulted = True

        sleep_wake_np = patient_data["sleep_wake"]
        activity_np = patient_data["activity"]
        sw = None
        act = None
        if sleep_wake_np is not None and rng.random() > self.input_dropout:
            sw = np.asarray(sleep_wake_np[win_start:win_end], dtype=np.float32)
        if activity_np is not None and rng.random() > self.input_dropout:
            act = np.asarray(activity_np[win_start:win_end], dtype=np.float32)
        absorption_np = patient_data["absorption_profile"]
        return TrainingWindow(
            patient_idx=patient_idx,
            patient_id=-1 if is_default else int(patient_data["patient_id"]),
            is_default=is_default,
            window_idx=window_idx,
            win_start=win_start,
            win_end=win_end,
            start_hour=float(patient_data["start_hour"]),
            traj_mode=patient_data.get("trajectory_loss_mode", "mse"),
            initial_state=init_row,
            target=trajectory_np[win_start:win_end],
            meals=win_meals,
            meals_defaulted=meals_defaulted,
            sleep_wake=sw,
            activity=act,
            absorption=None if absorption_np is None else absorption_np[win_start:win_end],
        )

    def _rollout_windows(
        self,
        model: nn.Module,
        embeddings: nn.Embedding,
        windows: list[TrainingWindow],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """One batched rollout of ``windows`` → ``(pred[B, T, S], emb[B, E], gut[B, T, 4],
        duodenal[B, T, 3] or None when no window has a meal)``.

        Each window keeps its own embedding, clock, meals (its own gut and duodenal
        windows), and sleep/activity tape; a withheld tape is NaN, which the model
        reads as "missing" and replaces with its learned default — exactly what a
        ``None`` series did for a single window.
        """
        n_steps = windows[0].n_steps
        if any(w.n_steps != n_steps for w in windows):
            raise ValueError("windows in one batch must share a length")
        rows = []
        for w in windows:
            if w.is_default:
                rows.append(torch.zeros(EMBEDDING_DIM, device=device))
            else:
                rows.append(embeddings(torch.tensor(w.patient_id, device=device)))
        emb = torch.stack(rows, dim=0)
        initial = torch.tensor(
            np.stack([w.initial_state for w in windows]), dtype=torch.float32, device=device)
        start = torch.tensor([w.win_time for w in windows], dtype=torch.float64, device=device)
        # Precompute gut over each whole window once — vectorized forward_window
        # across T meals, instead of one GutModule.forward call per integrate
        # step. The same tensor also feeds the in-window gut loss.
        gut = torch.stack([
            precompute_gut_outputs(model, emb[i], n_steps, dt=1.0, meals=w.meals)
            for i, w in enumerate(windows)
        ], dim=0)
        duo = None
        if any(w.meals for w in windows):
            duo = torch.stack([
                precompute_duodenal_outputs(model, n_steps, dt=1.0, meals=w.meals)
                for w in windows
            ], dim=0)

        def _tape(key: str) -> torch.Tensor | None:
            series = [getattr(w, key) for w in windows]
            if all(x is None for x in series):
                return None
            filled = [np.full(n_steps, np.nan, dtype=np.float32) if x is None else x for x in series]
            return torch.tensor(np.stack(filled), dtype=torch.float32, device=device)

        pred = integrate(
            model, initial, emb, n_steps,
            dt=1.0, start_time_minutes=start,
            sleep_wake=_tape("sleep_wake"),
            activity=_tape("activity"),
            gut_outputs=gut,
            duodenal_outputs=duo,
        )
        return pred, emb, gut, duo

    def _step_windows(
        self,
        model: nn.Module,
        embeddings: nn.Embedding,
        ctx: SignalContext,
        windows: list[TrainingWindow],
        cpl_w: float,
        ver_w: float,
    ) -> dict[str, float]:
        """Roll ``windows`` out together, take ONE optimizer step on the mean of their
        losses, and return the summed per-window components for the epoch log."""
        device = ctx.device
        pred, emb, gut, duo = self._rollout_windows(model, embeddings, windows, device)
        losses: list[torch.Tensor] = []
        parts = {"loss": 0.0, "gut": 0.0, "coupling": 0.0, "verifier": 0.0, "shape": 0.0}
        bands: list[float] = []
        for i, w in enumerate(windows):
            loss_i, comp = self._window_loss(
                model, w, pred[i], emb[i], gut[i], None if duo is None else duo[i],
                cpl_w, ver_w, device)
            losses.append(loss_i)
            bands.append(comp["band"])
            parts["loss"] += float(loss_i.detach().item())
            for key in ("gut", "coupling", "verifier", "shape"):
                parts[key] += comp[key]
        loss = losses[0] if len(losses) == 1 else torch.stack(losses).mean()

        # Iter 25: strict abort. Per-window context goes into ``extra``
        # so the abort dump pinpoints the offending (patient, window,
        # win_start, sub-component breakdown) — iter 24's silent
        # per-window skip lost exactly this signal. A batch reports its first
        # window plus the batch's composition.
        w0 = windows[0]
        safe_step(
            loss,
            ctx,
            signal=f"{self.name}/window",
            extra={
                "patient_idx": float(w0.patient_idx),
                "patient_id": float(w0.patient_id),
                "is_default": float(w0.is_default),
                "window_idx": float(w0.window_idx),
                "win_start": float(w0.win_start),
                "win_end": float(w0.win_end),
                "win_steps": float(w0.n_steps),
                "n_meals_in_window": float(len(w0.meals)),
                "traj_mode": 1.0 if w0.traj_mode == "huber" else 0.0,
                "band": float(bands[0]),
                "gut_loss": parts["gut"],
                "coupling_loss": parts["coupling"],
                "verifier_loss": parts["verifier"],
                "shape_loss": parts["shape"],
                "meals_defaulted": float(w0.meals_defaulted),
                "batch_windows": float(len(windows)),
                "batch_patient_ids": [float(w.patient_id) for w in windows],
                "batch_win_starts": [float(w.win_start) for w in windows],
                "batch_window_losses": [float(x.detach().item()) for x in losses],
            },
        )
        return parts

    def _window_loss(
        self,
        model: nn.Module,
        w: TrainingWindow,
        pred_traj: torch.Tensor,
        embedding: torch.Tensor,
        gut_window: torch.Tensor,
        duodenal_window: torch.Tensor | None,
        cpl_w: float,
        ver_w: float,
        device: torch.device,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """The loss of one window from its rollout ``pred_traj[T, S]``."""
        norm_scales = self._norm_scales.to(device)
        typicals = self._typicals.to(device)
        band_vec = self._band_vec.to(device) if self._band_vec is not None else None
        pointwise_mask = self._pointwise_mask.to(device)
        shape_idx = self._shape_idx.to(device)

        target = torch.tensor(w.target, dtype=torch.float32, device=device)
        mask = ~torch.isnan(target)
        diff = (pred_traj - target) / norm_scales
        diff = torch.where(mask, diff, torch.zeros_like(diff))
        # Banded distillation: only excursions outside ±band cost.
        # band=0 reduces to pure imitation (identical to plain MSE/Huber).
        # Iter 97: per-marker bands when configured (review 4.2); the
        # shape markers drop out of the pointwise term entirely.
        if band_vec is not None:
            band_t: torch.Tensor | float = band_vec
            band = float(band_vec.mean())
        else:
            band = float(self.trajectory_band_default if w.is_default else self.trajectory_band)
            band_t = band
        pw_mask = mask.float() * pointwise_mask
        denom = pw_mask.sum().clamp(min=1.0)
        excess = F.relu(diff.abs() - band_t) * pw_mask
        if w.traj_mode == "huber":
            d = torch.tensor(self.huber_delta, device=device, dtype=diff.dtype)
            quad = 0.5 * excess.pow(2)
            lin = d * (excess - 0.5 * d)
            per = torch.where(excess < d, quad, lin)
            loss = per.sum() / denom
        else:
            loss = excess.pow(2).sum() / denom

        comp = {"band": band, "gut": 0.0, "coupling": 0.0, "verifier": 0.0, "shape": 0.0}
        if len(self.shape_markers):
            sh_pred = pred_traj[:, shape_idx] / norm_scales[shape_idx]
            sh_tgt = torch.where(
                mask[:, shape_idx], target[:, shape_idx], torch.zeros_like(target[:, shape_idx]),
            ) / norm_scales[shape_idx]
            sh_band = band_vec[shape_idx] if band_vec is not None else torch.full(
                (len(self.shape_markers),), float(band), device=device,
            )
            sh_loss = shape_marker_loss(sh_pred, sh_tgt, mask[:, shape_idx], sh_band)
            loss = loss + sh_loss
            comp["shape"] = float(sh_loss.detach().item())

        # Catastrophe fence, not a pull (see SOFT_RANGE_DEADZONE).
        z_typ = (pred_traj - typicals) / norm_scales
        deviation = F.relu(z_typ.abs() - SOFT_RANGE_DEADZONE).pow(2).mean()
        loss = loss + SOFT_RANGE_REG * deviation

        if cpl_w > 0 and self._priors:
            sw_t = None if w.sleep_wake is None else torch.tensor(w.sleep_wake, device=device)
            act_t = None if w.activity is None else torch.tensor(w.activity, device=device)
            cpl = coupling_prior_loss_on_window(
                model,
                pred_traj,
                embedding,
                w.win_time,
                w.meals,
                sw_t,
                act_t,
                self._priors,
                n_samples=self.coupling_prior_samples,
                gut_outputs=gut_window,
                duodenal_outputs=duodenal_window,
            )
            loss = loss + cpl_w * cpl
            comp["coupling"] = float(cpl.detach().item())

        if ver_w > 0:
            vloss = training_verifier_surrogate_loss(
                pred_traj,
                meals=w.meals,
                start_hour=float(w.start_hour),
                timeline_offset_min=float(w.win_start),
            )
            loss = loss + ver_w * vloss
            comp["verifier"] = float(vloss.detach().item())

        if w.absorption is not None and w.meals and self.gut_loss_weight > 0:
            target_abs = torch.tensor(w.absorption, dtype=torch.float32, device=device)
            # Appearance channels only. The teacher's nutrient_flag is a
            # binary appearance-rate gate; the student's is unabsorbed-mass
            # survival — matching them is a category error (iter 98).
            n_app = GutModuleBase.N_APPEARANCE
            gut_loss = (
                (gut_window[..., :n_app] - target_abs[..., :n_app]) / self._abs_scale.to(device)
            ).pow(2).mean()
            loss = loss + self.gut_loss_weight * gut_loss
            comp["gut"] = float(gut_loss.detach().item())
        return loss, comp
