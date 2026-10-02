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

from dataclasses import dataclass, field

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
    # Iter 108: windows per optimizer step. 1 is the historical per-window SGD
    # step; k > 1 rolls k windows (usually k different patients) as one batched
    # integrate and averages their losses — about k times the windows per second,
    # since a step costs nearly the same at batch 1 and batch 16, at the price of k
    # times fewer imitation steps per epoch.
    windows_per_step: int = 1
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

    def _draw_window(
        self,
        patient_idx: int,
        patient_data: dict,
        window_idx: int,
        embeddings: nn.Embedding,
        rng: np.random.Generator,
        device: torch.device,
    ) -> "_Window":
        """Sample one training window and its inputs. The rng draws (start, meal-macro
        dropout, sleep / activity dropout) happen here, in this order, per window."""
        is_default = bool(patient_data.get("is_default", False))
        trajectory_np = patient_data["trajectory"]
        patient_meals = patient_data["meals"]
        n_steps = patient_data["duration_min"]
        sleep_wake_np = patient_data["sleep_wake"]
        activity_np = patient_data["activity"]

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
        if is_default:
            embedding = torch.zeros(EMBEDDING_DIM, device=device)
            patient_id = -1  # default-patient sentinel
        else:
            patient_id = int(patient_data["patient_id"])
            embedding = embeddings(torch.tensor(patient_id, device=device))

        win_meals = meals_in_window(patient_meals, win_start, win_end)
        meals_defaulted = False
        if win_meals and self.meal_macro_dropout > 0.0 and rng.random() < self.meal_macro_dropout:
            win_meals = meals_with_default_macros(win_meals)
            meals_defaulted = True

        sw_tensor = None
        act_tensor = None
        if sleep_wake_np is not None and rng.random() > self.input_dropout:
            sw_tensor = torch.tensor(
                sleep_wake_np[win_start:win_end], dtype=torch.float32, device=device,
            )
        if activity_np is not None and rng.random() > self.input_dropout:
            act_tensor = torch.tensor(
                activity_np[win_start:win_end], dtype=torch.float32, device=device,
            )
        absorption_np = patient_data["absorption_profile"]
        return _Window(
            patient_idx=patient_idx,
            patient_id=patient_id,
            is_default=is_default,
            window_idx=window_idx,
            win_start=win_start,
            win_end=win_end,
            start_hour=float(patient_data["start_hour"]),
            win_time=(patient_data["start_hour"] * 60 + win_start) % 1440,
            initial_state=torch.tensor(init_row, dtype=torch.float32, device=device),
            embedding=embedding,
            meals=win_meals,
            meals_defaulted=meals_defaulted,
            sleep_wake=sw_tensor,
            activity=act_tensor,
            target=torch.tensor(
                trajectory_np[win_start:win_end], dtype=torch.float32, device=device,
            ),
            absorption=(
                torch.tensor(absorption_np[win_start:win_end], dtype=torch.float32, device=device)
                if absorption_np is not None else None
            ),
            traj_mode=patient_data.get("trajectory_loss_mode", "mse"),
        )

    def _rollout(
        self, model: nn.Module, windows: list["_Window"],
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """``(pred_traj, gut_window, duodenal_window)`` per window.

        One window rolls exactly as it always has. Several roll as ONE batched
        integrate, each row carrying its own clock, meals and (possibly dropped)
        sleep / activity — a dropped series is NaN, which the model fills with its
        learned default for that row only.
        """
        guts = []
        duos = []
        for w in windows:
            guts.append(precompute_gut_outputs(
                model, w.embedding, w.steps, dt=1.0, start_time_minutes=w.win_time, meals=w.meals,
            ))
            duos.append(precompute_duodenal_outputs(model, w.steps, meals=w.meals))
        if len(windows) == 1:
            w = windows[0]
            pred = integrate(
                model, w.initial_state, w.embedding, w.steps,
                dt=1.0, start_time_minutes=w.win_time,
                meals=w.meals,
                sleep_wake=w.sleep_wake,
                activity=w.activity,
                gut_outputs=guts[0],
                duodenal_outputs=duos[0],
            )
            return [(pred, guts[0], duos[0])]

        T = max(w.steps for w in windows)
        device = windows[0].initial_state.device

        def pad(x: torch.Tensor, fill: float) -> torch.Tensor:
            if x.shape[0] == T:
                return x
            tail = x.new_full((T - x.shape[0], *x.shape[1:]), fill)
            return torch.cat([x, tail], dim=0)

        def series(x: torch.Tensor | None) -> torch.Tensor:
            if x is None:
                return torch.full((T,), float("nan"), device=device)
            return pad(x, float("nan"))

        pred = integrate(
            model,
            torch.stack([w.initial_state for w in windows]),
            torch.stack([w.embedding for w in windows]),
            T,
            dt=1.0,
            start_time_minutes=torch.tensor(
                [float(w.win_time) for w in windows], dtype=torch.float64, device=device),
            meals=[w.meals for w in windows],
            sleep_wake=torch.stack([series(w.sleep_wake) for w in windows]),
            activity=torch.stack([series(w.activity) for w in windows]),
            gut_outputs=torch.stack([pad(g, 0.0) for g in guts]),
            duodenal_outputs=torch.stack([pad(d, 0.0) for d in duos]),
            member_steps=torch.tensor([w.steps for w in windows], device=device),
        )
        return [
            (pred[k, : w.steps], guts[k], duos[k]) for k, w in enumerate(windows)
        ]

    def iter_windows(
        self,
        model: nn.Module,
        embeddings: nn.Embedding,
        ctx: SignalContext,
    ):
        """Generator: one optimizer step per ``yield``, which yields the number of
        windows consumed so far this epoch.

        Iter 97 (review 4.1): the trainer drives this generator and interleaves
        one joint auxiliary step every k windows, instead of running all ~96
        windows and then a single aux step per epoch (25 literature steps in a
        whole run vs ~5,300 imitation steps). ``last_result`` holds the epoch's
        aggregate once the generator is exhausted.

        Iter 108: ``windows_per_step`` windows share one optimizer step — rolled as
        one batched integrate, their per-window losses averaged. 1 (the default) is
        the historical one-window SGD step, rng order included.
        """
        device = ctx.device
        rng = ctx.rng
        k = max(1, int(self.windows_per_step))

        totals = {"loss": 0.0, "gut": 0.0, "coupling": 0.0, "verifier": 0.0, "shape": 0.0}
        n_windows = 0
        n_defaulted = 0

        def draws():
            for patient_idx, patient_data in enumerate(self._dataset):
                for window_idx in range(self.windows_per_patient):
                    yield patient_idx, patient_data, window_idx

        pending = draws()
        while True:
            chunk = []
            for patient_idx, patient_data, window_idx in pending:
                chunk.append(self._draw_window(
                    patient_idx, patient_data, window_idx, embeddings, rng, device))
                if len(chunk) == k:
                    break
            if not chunk:
                break
            losses = []
            parts = []
            for w, (pred_traj, gut_window, duo_window) in zip(chunk, self._rollout(model, chunk)):
                loss, comp = self._window_loss(model, w, pred_traj, gut_window, duo_window, ctx)
                losses.append(loss)
                parts.append(comp)
            loss = losses[0] if len(losses) == 1 else torch.stack(losses).mean()
            # Iter 25: strict abort. Per-window context goes into ``extra``
            # so the abort dump pinpoints the offending (patient, window,
            # win_start, sub-component breakdown) — iter 24's silent
            # per-window skip lost exactly this signal.
            safe_step(
                loss,
                ctx,
                signal=f"{self.name}/window",
                extra=_abort_context(chunk, parts),
            )
            for w, lv, comp in zip(chunk, losses, parts):
                totals["loss"] += float(lv.detach().item())
                totals["gut"] += comp["gut_loss"]
                totals["coupling"] += comp["coupling_loss"]
                totals["verifier"] += comp["verifier_loss"]
                totals["shape"] += comp["shape_loss"]
                n_windows += 1
                n_defaulted += int(w.meals_defaulted)
            yield n_windows

        cpl_on = self.coupling_weight.at(ctx.epoch) > 0 and self._priors
        sub: dict[str, float] = {
            "gut": totals["gut"] / max(n_windows, 1),
        }
        if cpl_on:
            sub["coupling"] = totals["coupling"] / max(n_windows, 1)
        if self.verifier_weight.at(ctx.epoch) > 0:
            sub["verifier_surrogate"] = totals["verifier"] / max(n_windows, 1)
        if len(self.shape_markers):
            sub["shape"] = totals["shape"] / max(n_windows, 1)
        sub["meals_defaulted"] = float(n_defaulted)

        self._last_result = SignalResult(
            loss_sum=totals["loss"],
            n_units=n_windows,
            sub_metrics=sub,
        )

    def _window_loss(
        self,
        model: nn.Module,
        w: "_Window",
        pred_traj: torch.Tensor,
        gut_window: torch.Tensor,
        duo_window: torch.Tensor,
        ctx: SignalContext,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Every loss that reads this window's rollout, and its components (detached)."""
        device = ctx.device
        norm_scales = self._norm_scales.to(device)
        typicals = self._typicals.to(device)
        cpl_w = self.coupling_weight.at(ctx.epoch)
        ver_w = self.verifier_weight.at(ctx.epoch)
        band_vec = self._band_vec.to(device) if self._band_vec is not None else None

        target = w.target
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
            band = float(self.trajectory_band_default) if w.is_default else float(self.trajectory_band)
            band_t = band
        pw_mask = mask.float() * self._pointwise_mask.to(device)
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

        comp = {
            "band": band, "shape_loss": 0.0, "coupling_loss": 0.0,
            "verifier_loss": 0.0, "gut_loss": 0.0,
        }
        if len(self.shape_markers):
            shape_idx = self._shape_idx.to(device)
            sh_pred = pred_traj[:, shape_idx] / norm_scales[shape_idx]
            sh_tgt = torch.where(
                mask[:, shape_idx], target[:, shape_idx], torch.zeros_like(target[:, shape_idx]),
            ) / norm_scales[shape_idx]
            sh_band = band_vec[shape_idx] if band_vec is not None else torch.full(
                (len(self.shape_markers),), float(band), device=device,
            )
            sh_loss = shape_marker_loss(sh_pred, sh_tgt, mask[:, shape_idx], sh_band)
            loss = loss + sh_loss
            comp["shape_loss"] = float(sh_loss.detach().item())

        # Catastrophe fence, not a pull (see SOFT_RANGE_DEADZONE).
        z_typ = (pred_traj - typicals) / norm_scales
        deviation = F.relu(z_typ.abs() - SOFT_RANGE_DEADZONE).pow(2).mean()
        loss = loss + SOFT_RANGE_REG * deviation

        if cpl_w > 0 and self._priors:
            cpl = coupling_prior_loss_on_window(
                model,
                pred_traj,
                w.embedding,
                float(w.win_time),
                w.meals,
                w.sleep_wake,
                w.activity,
                self._priors,
                n_samples=self.coupling_prior_samples,
                gut_window=gut_window,
                duodenal_window=duo_window,
            )
            loss = loss + cpl_w * cpl
            comp["coupling_loss"] = float(cpl.detach().item())

        if ver_w > 0:
            vloss = training_verifier_surrogate_loss(
                pred_traj,
                meals=w.meals,
                start_hour=w.start_hour,
                timeline_offset_min=float(w.win_start),
            )
            loss = loss + ver_w * vloss
            comp["verifier_loss"] = float(vloss.detach().item())

        if w.absorption is not None and w.meals and self.gut_loss_weight > 0:
            # Appearance channels only. The teacher's nutrient_flag is a
            # binary appearance-rate gate; the student's is unabsorbed-mass
            # survival — matching them is a category error (iter 98).
            n_app = GutModuleBase.N_APPEARANCE
            gut_loss = (
                (gut_window[..., :n_app] - w.absorption[..., :n_app]) / self._abs_scale.to(device)
            ).pow(2).mean()
            loss = loss + self.gut_loss_weight * gut_loss
            comp["gut_loss"] = float(gut_loss.detach().item())
        return loss, comp


@dataclass
class _Window:
    """One sampled training window and everything its rollout and losses read."""

    patient_idx: int
    patient_id: int
    is_default: bool
    window_idx: int
    win_start: int
    win_end: int
    start_hour: float
    win_time: float
    initial_state: torch.Tensor
    embedding: torch.Tensor
    meals: list[MealEvent]
    meals_defaulted: bool
    sleep_wake: torch.Tensor | None
    activity: torch.Tensor | None
    target: torch.Tensor
    absorption: torch.Tensor | None
    traj_mode: str

    @property
    def steps(self) -> int:
        return self.win_end - self.win_start


def _abort_context(windows: list[_Window], parts: list[dict[str, float]]) -> dict[str, float]:
    """The NaN-abort dump's per-window context (the first window of a batched step,
    plus the batch size)."""
    w, comp = windows[0], parts[0]
    return {
        "patient_idx": float(w.patient_idx),
        "patient_id": float(w.patient_id),
        "is_default": float(w.is_default),
        "window_idx": float(w.window_idx),
        "win_start": float(w.win_start),
        "win_end": float(w.win_end),
        "win_steps": float(w.steps),
        "n_meals_in_window": float(len(w.meals)),
        "traj_mode": 1.0 if w.traj_mode == "huber" else 0.0,
        "band": float(comp["band"]),
        "gut_loss": comp["gut_loss"],
        "coupling_loss": comp["coupling_loss"],
        "verifier_loss": comp["verifier_loss"],
        "shape_loss": comp["shape_loss"],
        "meals_defaulted": float(w.meals_defaulted),
        "windows_in_step": float(len(windows)),
    }
