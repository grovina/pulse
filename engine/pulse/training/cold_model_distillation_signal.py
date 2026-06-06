"""
Cold-model distillation signal — teaches the learned model to reproduce the
knowledge model's *unobserved-marker* dynamics across a broad protocol
distribution.

Why this exists (the dead-pathway diagnosis, iters 38-46)
---------------------------------------------------------
Five markers — glucagon, ffa, ghrelin, leptin, acth (and to a lesser extent
cortisol, bhb) — have been byte-identical in the benchmark report across every
iteration since iter 38, regardless of weight tuning, multi-arm physiology
rules, or broader rule sampling. The root cause is structural, not parametric:

  1. The bench's targets for these markers come from ``simulate_full_body``
     (the cold knowledge model) — no real user self-measures FFA, so the only
     ground truth available is the textbook ODE.
  2. *No active training signal reaches the parameters that control these
     markers' level and shape.* Bench embedding calibration fits only the 5
     self-measured markers (glucose/hr/sbp/dbp/temp); trajectory MSE on
     real-user windows supervises only those 5; cohort-statistic weight split
     across ~19 specs is negligible per spec; the physiology rules produce
     ~1e-4 gradient onto glucagon/ghrelin (a correlation predicate on a flat
     trajectory has near-zero gradient) and FFA's larger rule gradient trains
     at sampled patient embeddings the cohort eval never visits; and
     ``marker_vitality`` (a range-floor band-aid) was never enabled and only
     constrained variance, not the curve.

  ⇒ The unobserved-marker heads sit at initialization, which emits a near-
     constant near-typical value. Nothing moves them.

The fix: distill the knowledge model's trajectories for these markers directly.
Each epoch, run the learned model at the zero embedding (= population-center
patient) on a sampled subset of a broad protocol pool — standard meal days,
OGTT, extended fast, high-fat meal, phase-shifted day, grazing — and MSE its
unobserved-marker trajectories against the cold-model reference for the same
protocol. The references are computed once at signal init (the cold model is
deterministic).

Two loss modes (``mode``)
-------------------------
- ``trajectory`` (iters 47-49): roll the learned model out from the cold
  initial state at a calibrated embedding and match the *integrated*
  trajectory of the unobserved markers (Huber on NORM_SCALE-normalized
  residuals). The 2026-05-13 diagnostic (`scripts/diagnose-dead-pathways.py`)
  showed this is the wrong loss shape: the gradient from the trajectory loss
  to the dead-marker rate heads is diluted ~T× through the integration chain
  (state[t+1] = state[t] + dt·rate[t]), arrives at ~1e-5, and the heads —
  which sit at the mass-action equilibrium pinning the marker at its
  ``typical`` value — never leave it. Byte-identical bench MAPE across every
  embedding/dose we tried.
- ``rate`` (iter 50+): **teacher-forced rate matching.** Feed the model the
  *cold* state at each timestep and require its instantaneous rate output for
  the unobserved markers to match the cold ODE's ``d/dt`` at that state
  (finite-differenced from the cold trajectory). One batched ``model.forward``
  over the whole protocol — no integration chain, gradient straight to the
  head, ~T× stronger. This is how you distil an ODE: matching rates ⊇ matching
  trajectories. The embedding is still calibrated (the rate reads a per-module
  embedding projection); the gut outputs are detached (the gut module has its
  own signals).

Patient axis: the cold references are ``simulate_full_body(PatientParams())``
— the population-mean patient — distilled at the *calibrated* embedding(s) the
bench lands near; ``pool`` controls how broad the protocol coverage is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..knowledge.full_body import (
    PatientParams,
    generate_activity,
    generate_meal_plan,
    generate_sleep_wake,
    simulate_full_body,
)
from ..model import integrate, precompute_gut_outputs
from ..modules.gut import MealEvent
from ..types import EMBEDDING_DIM, MARKER_INDEX, NORM_SCALE
from .safe_step import safe_step
from .signals import SignalContext, SignalResult, TrainingSignal, WeightSchedule

# Markers distilled from the cold model. These are the unobserved-but-load-
# bearing counter-regulatory + circadian markers — the ones with no real-user
# ground truth and no other supervision reaching them. (insulin / glp1 are
# deliberately excluded: they are already well-constrained by the glucose
# coupling, insulin-sweep, and gut-dose-sweep signals — re-distilling them here
# would just add a competing pull on already-calibrated dynamics.)
#
# Iter 76 (Move D): liver_glycogen + muscle_glycogen join the set. The cold
# ODE now SIMULATES them as flux integrators (full_body.py), so for the first
# time the slow pools have a real trajectory target — the measurement iters
# 55-57 lacked (those tried to drive a SetpointHead pool off `typical` via an
# indirect cohort window-mean delta, ~6700× weaker than the glucose spec on the
# same arms, and never moved it). Glycogen is a slow free-running LEVEL, so it
# is distilled in mode="anchored" (the rate term carries the synthesis/
# breakdown flux gradient through the strong gut-coupling path; the level
# anchor pins the integrated pool depth that teacher-forced rate matching is
# blind to). mito_capacity / crh stay out: the teacher still pads them (mito
# needs chronic-block protocols its ≤1-day pool can't provide; the cold ODE
# doesn't simulate crh).
_DEFAULT_DISTILL_MARKERS: tuple[str, ...] = (
    "glucagon", "ffa", "ghrelin", "leptin", "acth", "cortisol", "bhb",
    "liver_glycogen", "muscle_glycogen",
)

# Huber transition on the NORM_SCALE-normalized residual. The dead markers
# start grossly wrong (ghrelin ~1.9× off, glucagon ~0.4× off), so a plain
# MSE on the normalized residual would be dominated by a handful of large
# errors and could swamp the other training signals. Huber keeps the loss
# linear past one NORM_SCALE — the worst marker still leads, but bounded.
_HUBER_DELTA = 1.0

# Pool size and per-epoch sample count. The full pool is rebuilt once at
# signal init; each epoch draws ``protocols_per_epoch`` of them (minibatching
# the protocol set). Default 4-of-8 keeps the added epoch cost ~4 zero-
# embedding rollouts.
_DEFAULT_POOL_SIZE = 9
_DEFAULT_PROTOCOLS_PER_EPOCH = 4

# Build seed for the protocol pool + cold references. Fixed so the reference
# set is reproducible run-to-run.
_PROTOCOL_SEED = 20260511

# mode="rate": the cold ODE's d/dt is finite-differenced in marker-units per
# *minute* (dt=1). To make the normalized rate residual O(1) — so a Huber
# delta of 1.0 and a weight of ~0.3 are interpretable, the way the trajectory
# loss's residuals are — multiply by this many minutes (≈ a marker that swings
# by one NORM_SCALE over an hour ⇒ normalized rate ≈ 1).
_RATE_NORM_MINUTES = 60.0

# mode="anchored" (iter 75): reset-to-truth window for the short-horizon
# free-running level anchor. The protocol is tiled into non-overlapping
# windows of this many minutes; from the cold state at each window start the
# model rolls out FREELY and its absolute level is matched to the cold
# reference. W bounds the integration chain (gradient dilution ~W×, not the
# ~T× that killed full-trajectory mode) while still penalising the free-rollout
# drift teacher-forced rate matching is blind to. 60 min ≈ one meal-response
# arc; small enough that an untrained head can't diverge far, large enough to
# expose level drift the rate term ignores.
_DEFAULT_ANCHOR_WINDOW = 60

# Iter 67: L∞ clamp on the calibrated embedding during the Adam inner loop.
# Patient embeddings init with std=0.1; the bench's own calibration produces
# values well under this radius. Clamping defangs the Adam-divergence path
# that caused the iter-65 NaN (epoch 58, every cold-distill marker NaN at
# once because the cached emb went non-finite and tainted the next forward).
_CALIB_EMB_CLAMP = 5.0


def _finite_diff_rate(traj: np.ndarray) -> np.ndarray:
    """Cold ODE d/dt at each cold state: traj[t+1] - traj[t] (dt=1 min).

    Shape [T-1, n_markers]. (At least 1 row; degenerate 1-step protocols
    would give an empty array, which the rate loss skips.)
    """
    t = np.asarray(traj, dtype=np.float32)
    if t.shape[0] < 2:
        return np.zeros((0, t.shape[1]), dtype=np.float32)
    return (t[1:] - t[:-1]).astype(np.float32)


@dataclass(frozen=True)
class _Protocol:
    """One distillation protocol: a meal/sleep/activity schedule + the cold-
    model reference trajectory it produces at the population-mean patient."""

    name: str
    duration_min: int
    start_hour: float
    meal_events: tuple[MealEvent, ...]
    sleep_wake: np.ndarray  # [duration_min]
    activity: np.ndarray  # [duration_min]
    initial_state: np.ndarray  # [n_markers] — the cold model's t0 state
    reference: np.ndarray  # [duration_min, n_markers] — cold-model trajectory
    # Finite-difference of ``reference`` (= cold ODE's d/dt at each cold state,
    # marker-units per minute): cold_rate[t] = reference[t+1] - reference[t].
    # Shape [duration_min - 1, n_markers]. Used by mode="rate".
    cold_rate: np.ndarray
    # The cold ODE's gut-absorption coupling inputs at each step:
    # [glucose_appearance, lipid_appearance, amino_appearance, nutrient_flag],
    # shape [duration_min, 4] — same layout as the learned gut module's output.
    # mode="rate" feeds *these* (not the learned gut module's output) as the
    # gut_override so the model's rate is asked to match the cold rate under
    # the same coupling inputs the cold ODE saw.
    cold_absorption: np.ndarray
    # Explicit (time, marker_idx, value) observation points to calibrate the
    # embedding against. Empty for synthetic protocols (calibration falls back
    # to dense grid-sampling of ``reference`` on ``obs_markers``); populated
    # for bench-cohort protocols from the episode's calibration check-ins so
    # the calibration mirrors what the bench itself does.
    obs_points: tuple[tuple[int, int, float], ...] = ()


def _meal_events(meals_4tuple: Sequence[tuple[float, float, float, float]]) -> tuple[MealEvent, ...]:
    return tuple(
        MealEvent(time=float(t), carbs=float(c), fats=float(f), proteins=float(p))
        for t, c, f, p in meals_4tuple
        if t >= 0
    )


def _nocturnal_sleep(duration_min: int, start_hour: float) -> np.ndarray:
    """Awake during the day; asleep 23:00 → 07:00, smoothed."""
    sw = np.ones(duration_min, dtype=np.float32)
    bedtime = int((23.0 - start_hour) % 24 * 60)
    wake = int((31.0 - start_hour) % 24 * 60)  # 07:00 next day
    if wake <= bedtime:
        wake += 1440
    for i in range(duration_min):
        # asleep if i falls in any [bedtime + 1440k, wake + 1440k) window
        phase = i
        asleep = False
        for k in range(-1, 3):
            lo, hi = bedtime + 1440 * k, wake + 1440 * k
            if lo <= phase < hi:
                asleep = True
                break
        if asleep:
            sw[i] = 0.0
    kernel = np.ones(20, dtype=np.float32) / 20.0
    return np.clip(np.convolve(sw, kernel, mode="same").astype(np.float32), 0.0, 1.0)


def _build_protocol_pool(rng: np.random.Generator) -> list[dict[str, Any]]:
    """Spec the protocol pool (schedules only — cold rollout happens in init).

    Spans the regimes the bench's cohort episodes probe (long meal days,
    overnight fasting, postprandial spikes) plus circadian-phase variety,
    without copying the exact bench cohort meal grams / timing — the bench
    cohorts stay held out as an honest check of the distilled mechanism.
    """
    protocols: list[dict[str, Any]] = []

    # 1. Canonical 3-meal day, nocturnal sleep, no exercise (the old
    #    marker-vitality anchor protocol).
    protocols.append(dict(
        name="standard_3meal",
        duration_min=1440,
        start_hour=7.0,
        meals=[(60.0, 50.0, 15.0, 20.0), (360.0, 70.0, 20.0, 25.0), (720.0, 80.0, 25.0, 30.0)],
        sleep="nocturnal",
        activity_const=0.05,
    ))

    # 2-3. Two randomized realistic days (varied meal grams/timing, sleep
    #      bedtimes, an exercise bout) — the bulk-distribution protocols.
    for i, start_hour in enumerate((6.0, 8.0)):
        meals = generate_meal_plan(n_days=1, rng=rng, start_hour=start_hour)
        sw = generate_sleep_wake(n_days=1, duration_min=1440, start_hour=start_hour, rng=rng)
        act = generate_activity(n_days=1, duration_min=1440, start_hour=start_hour, rng=rng)
        protocols.append(dict(
            name=f"randomized_day_{i + 1}",
            duration_min=1440,
            start_hour=start_hour,
            meals=meals,
            sleep_wake=sw,
            activity=act,
        ))

    # 4. OGTT-style: single 75 g pure-glucose load at minute 60, fasted
    #    otherwise, awake throughout, light activity. Sharp insulin → glucagon
    #    suppression → FFA suppression signature.
    protocols.append(dict(
        name="ogtt_75g",
        duration_min=480,
        start_hour=8.0,
        meals=[(60.0, 75.0, 0.0, 0.0)],
        sleep="awake",
        activity_const=0.05,
    ))

    # 5. Extended 24 h fast: no meals, nocturnal sleep. Glucagon rise, FFA
    #    rise, ketogenesis (BHB) onset by hours 16-24.
    protocols.append(dict(
        name="fast_24h",
        duration_min=1440,
        start_hour=7.0,
        meals=[],
        sleep="nocturnal",
        activity_const=0.04,
    ))

    # 6. High-fat / low-carb meal: distinct FFA + glucagon trajectory vs the
    #    carb-heavy days.
    protocols.append(dict(
        name="high_fat_meal",
        duration_min=720,
        start_hour=8.0,
        meals=[(60.0, 20.0, 60.0, 25.0)],
        sleep="nocturnal",
        activity_const=0.05,
    ))

    # 7. Phase-shifted day: same standard meals but the rollout starts at
    #    14:00 — rotates which part of the circadian cycle (cortisol peak
    #    08:00, leptin peak 02:00) the rollout traverses, exercising the
    #    circadian time-features differently.
    protocols.append(dict(
        name="phase_shift_14h",
        duration_min=1440,
        start_hour=14.0,
        meals=[(180.0, 50.0, 15.0, 20.0), (480.0, 70.0, 20.0, 25.0), (840.0, 80.0, 25.0, 30.0)],
        sleep="nocturnal",
        activity_const=0.05,
    ))

    # 8. Grazing: six small meals every 2.5 h — sustained nutrient appearance
    #    keeps ghrelin suppressed for an extended window.
    protocols.append(dict(
        name="grazing_6meal",
        duration_min=1440,
        start_hour=7.0,
        meals=[(float(60 + 150 * k), 20.0, 8.0, 8.0) for k in range(6)],
        sleep="nocturnal",
        activity_const=0.05,
    ))

    # 9. Exercise day (iter 76 — Move D). Standard meals + two moderate
    #    (~Z2) bouts: a 45-min session at 10:00 and a 30-min session at
    #    17:00. This is the protocol that exercises the *muscle* glycogen
    #    depletion arc — muscle glycogen is rest-preserved (the other
    #    protocols only top it toward its cap via post-meal synthesis), so
    #    without a bona-fide bout the distillation would have no signal for
    #    its catabolic gate. Also drives the acute exercise responses
    #    (HR↑, glucose↓, FFA↑, cortisol↑, lactate↑) the cold ODE already
    #    models, broadening protocol coverage toward the acute_z2_bout work.
    ex_activity = np.full(1440, 0.05, dtype=np.float32)
    ex_activity[180:225] = 0.55  # 10:00, 45 min
    ex_activity[600:630] = 0.50  # 17:00, 30 min
    protocols.append(dict(
        name="exercise_day",
        duration_min=1440,
        start_hour=7.0,
        meals=[(60.0, 50.0, 15.0, 20.0), (360.0, 70.0, 20.0, 25.0), (720.0, 80.0, 25.0, 30.0)],
        sleep="nocturnal",
        activity=ex_activity,
    ))

    return protocols


@dataclass
class ColdModelDistillationSignal(TrainingSignal):
    """Distil the knowledge model's unobserved-marker trajectories into the
    learned model **at a calibrated embedding**, over a broad protocol pool.

    Iter-47 lesson (the reason this isn't "distil at the zero embedding")
    ------------------------------------------------------------------
    Iter 47 ran the distillation rollouts at the zero embedding (the assumed
    population centre). On the bench it did *nothing* for its targets — FFA /
    glucagon / ghrelin / leptin stayed byte-identical — while collaterally
    regressing acth / cortisol / hr and the textbook pass-rate. The cause:
    the bench evaluates each cohort at an embedding *calibrated* (512 steps,
    lr 0.05, l2 0.001) to that cohort's 5 self-measured markers, and that
    point is far enough from zero that (a) the distilled-at-zero unobserved
    dynamics don't surface there and (b) pulling shared params toward the
    zero-embedding optimum drags the *observed*-marker fit at the calibrated
    embedding off its iter-46 best. So the distillation must happen where the
    bench actually looks.

    What this does
    --------------
    Per epoch (active only once the additive losses turn on — phase 2 — when
    the observed-marker fit, and hence a meaningful embedding, exists):

      1. Sample ``protocols_per_epoch`` protocols from the pool.
      2. For each, fetch a per-protocol *calibrated embedding*: an embedding
         optimized to reproduce the protocol's cold observed-marker
         trajectory (glucose/hr/sbp/dbp/temp), mirroring the bench's
         calibration. Cold-start (``calib_steps``) the first time a protocol
         is seen; warm-restart (``calib_warm_steps`` from the cached value)
         when it has gone stale (``recalib_every`` epochs). The embedding is
         **detached** — it parametrizes *where* we distil, not *what*.
      3. Run the learned model at that embedding and add the NORM_SCALE-
         normalized Huber trajectory loss on ``markers`` (the unobserved
         counter-regulatory + circadian set) vs the cold reference.

    The protocol pool is *broad over protocols, centred on the population-
    mean patient* (cold references from ``simulate_full_body(PatientParams())``)
    — and it deliberately does not copy the bench cohorts' exact meals, so the
    bench cohorts stay a held-out check: if the dead markers move *there*, the
    distilled mechanism generalized across the protocol axis. (Patient-axis
    breadth — varied ``PatientParams`` — is still future work; the embedding
    table itself carries patient variation via the real-data trajectory loss.)

    References + protocol schedules are built once at construction (the cold
    model is deterministic). Forwards run with the model in eval mode so the
    distillation target — and the embedding it is calibrated against — see the
    same deterministic network the bench does.
    """

    weight: WeightSchedule = field(default_factory=lambda: WeightSchedule(0.0))
    markers: Sequence[str] = field(default_factory=lambda: _DEFAULT_DISTILL_MARKERS)
    # Which protocol pool to distil over:
    #   "synthetic"     — the 8-protocol broad pool (population-mean patient,
    #                     varied schedules; bench cohorts stay held out).
    #   "bench_cohorts" — exactly the two cohort episodes the bench scores
    #                     unobserved markers on (cohort_sleep_48h,
    #                     cohort_meal_postprandial), calibrated to their own
    #                     check-ins. Trades the held-out check for distilling
    #                     at the eval's actual operating point.
    #   "both"          — synthetic pool + the two bench cohorts.
    pool: str = "synthetic"
    pool_size: int = _DEFAULT_POOL_SIZE
    protocols_per_epoch: int = _DEFAULT_PROTOCOLS_PER_EPOCH
    # Loss shape: "trajectory" = match the integrated rollout (iters 47-49 —
    # gradient diluted ~T× through the integration chain, doesn't move the
    # dead heads); "rate" = teacher-forced rate matching (feed the cold state,
    # match the model's instantaneous d/dt to the cold ODE's — gradient
    # straight to the head); "anchored" (iter 75) = rate matching PLUS a
    # short-horizon free-running absolute-level anchor (see _level_terms). The
    # rate half keeps the strong shape gradient; the level half closes the
    # offset-invariance hole rate-only matching leaves open — the model can
    # reproduce the cold d/dt at every teacher-forced state while its OWN
    # free-rollout level drifts arbitrarily (the iter-74 acth/cortisol/glucagon
    # /ffa drift once adaptive cohort reweighting pulled their only level pin).
    # See the module docstring.
    mode: str = "trajectory"
    # mode="anchored": reset-to-truth window for the level anchor (minutes).
    anchor_window: int = _DEFAULT_ANCHOR_WINDOW
    # mode="anchored": max free-rollout windows evaluated per protocol per
    # epoch. Each window is an independent autograd graph but ALL are kept
    # alive until the signal's single backward(), so peak memory ~ samples×W
    # steps — bounding the count (rather than tiling the whole protocol) keeps
    # it well under the full-trajectory cost the iter-67 saga OOM'd on. The
    # windows are evenly spaced across the protocol (deterministic, low-
    # variance), so 8 over a 24 h day samples morning / each meal / overnight.
    anchor_samples: int = 8
    # Embedding calibration (mirrors pulse.benchmark.calibrate_embedding).
    obs_markers: Sequence[str] = field(
        default_factory=lambda: ("glucose", "hr", "sbp", "dbp", "temp"),
    )
    recalib_every: int = 10
    # Iter 68 round 2: dropped from 64/24 to 8/4. The originals were sized
    # for the iter-48 model (EMBEDDING_DIM=32, n_species=2); after iter 60
    # widened the embedding 32→64 and iter 66 went to n_species=4, the
    # per-step cost of integrate() quadrupled. Iter 68 (gradient checkpointing)
    # OOM'd at phase-2 ep 50 after 1h44m — checkpointing dropped per-step
    # memory but couldn't compensate for the work having quadrupled while
    # iteration count stayed constant. Adam converges within ~8-12 steps on
    # this loss surface; 64 was overkill even on the smaller model.
    calib_steps: int = 8
    calib_warm_steps: int = 4
    calib_lr: float = 0.05
    calib_l2: float = 0.001

    name: str = "cold_model_distillation"
    source: str = (
        "Cold knowledge model (simulate_full_body) — distillation of the "
        "unobserved counter-regulatory + circadian markers (glucagon, ffa, "
        "ghrelin, leptin, acth, cortisol, bhb) the bench targets but no other "
        "training signal reaches, evaluated at a per-protocol embedding "
        "calibrated to the cold observed markers (so the distilled dynamics "
        "land where the bench's calibrated eval looks). Replaces the iter-39 "
        "marker-vitality range-floor band-aid (matching the curve subsumes "
        "the variance floor)."
    )
    category: str = "mechanism"

    def __post_init__(self) -> None:
        if self.pool == "bench_cohorts":
            self._protocols: list[_Protocol] = self._build_bench_cohort_protocols()
        elif self.pool == "both":
            self._protocols = self._build_synthetic_protocols() + self._build_bench_cohort_protocols()
        else:  # "synthetic"
            self._protocols = self._build_synthetic_protocols()
        self._marker_idx: list[tuple[str, int]] = [
            (m, MARKER_INDEX[m]) for m in self.markers if m in MARKER_INDEX
        ]
        self._obs_idx: list[int] = [
            MARKER_INDEX[m] for m in self.obs_markers if m in MARKER_INDEX
        ]
        # pidx -> (embedding on CPU, epoch it was last (re)calibrated at).
        self._calib_cache: dict[int, tuple[torch.Tensor, int]] = {}
        self._n_recalibs = 0
        # mode="anchored": count windows skipped because the free rollout went
        # non-finite (surfaced in telemetry rather than aborting an 18 h run —
        # an early-training divergence on an untrained head is expected and
        # transient, unlike the teacher-forced rate path's hard NaN guard).
        self._n_anchor_skips = 0

    def _build_synthetic_protocols(self) -> list[_Protocol]:
        rng = np.random.default_rng(_PROTOCOL_SEED)
        specs = _build_protocol_pool(rng)
        if self.pool_size and self.pool_size < len(specs):
            specs = specs[: self.pool_size]
        return [self._materialize(s) for s in specs]

    def _build_bench_cohort_protocols(self) -> list[_Protocol]:
        """The exact cohort episodes the bench scores unobserved markers on.

        Pulled from ``benchmark_extras.all_cohort_benchmark_episodes()`` (the
        public entry the gate uses), so the schedules + observation check-ins
        track the bench by construction. The cold reference is re-run clean
        (``noise_scale=0``) — the bench's own targets carry 1e-3 noise, which
        is negligible against the marker scales.
        """
        from ..knowledge.benchmark_extras import all_cohort_benchmark_episodes

        out: list[_Protocol] = []
        for ep in all_cohort_benchmark_episodes():
            dur = int(ep.duration_min)
            start_hour = float(ep.start_time_minutes if ep.start_time_minutes is not None else 360.0) / 60.0
            meals_4t = [
                (float(m.time), float(m.carbs), float(m.fats), float(m.proteins)) for m in ep.meals
            ]
            sw = (
                np.asarray(ep.sleep_wake, dtype=np.float32)[:dur]
                if ep.sleep_wake is not None else np.ones(dur, dtype=np.float32)
            )
            act = (
                np.asarray(ep.activity, dtype=np.float32)[:dur]
                if ep.activity is not None else np.full(dur, 0.05, dtype=np.float32)
            )
            traj, absorption = simulate_full_body(
                PatientParams(), meals_4t, sw, act, dur, start_hour,
                noise_scale=0.0, rng=np.random.default_rng(_PROTOCOL_SEED),
            )
            traj = np.asarray(traj, dtype=np.float32)[:dur]
            absorption = np.asarray(absorption, dtype=np.float32)[:dur]
            obs_points: list[tuple[int, int, float]] = []
            for ci in ep.calibration_check_ins:
                t = int(ci["time"])
                for mname, val in ci["measurements"].items():
                    if mname in MARKER_INDEX and 0 <= t < dur:
                        obs_points.append((t, MARKER_INDEX[mname], float(val)))
            out.append(_Protocol(
                name=str(ep.user_id),
                duration_min=dur,
                start_hour=start_hour,
                meal_events=tuple(ep.meals),
                sleep_wake=sw,
                activity=act,
                initial_state=traj[0].copy(),
                reference=traj,
                cold_rate=_finite_diff_rate(traj),
                cold_absorption=absorption,
                obs_points=tuple(obs_points),
            ))
        return out

    @staticmethod
    def _resolve_sleep_activity(spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        dur = int(spec["duration_min"])
        start_hour = float(spec["start_hour"])
        if "sleep_wake" in spec:
            sw = np.asarray(spec["sleep_wake"], dtype=np.float32)[:dur]
        elif spec.get("sleep") == "awake":
            sw = np.ones(dur, dtype=np.float32)
        else:  # "nocturnal" (default)
            sw = _nocturnal_sleep(dur, start_hour)
        if "activity" in spec:
            act = np.asarray(spec["activity"], dtype=np.float32)[:dur]
        else:
            act = np.full(dur, float(spec.get("activity_const", 0.05)), dtype=np.float32)
        return sw, act

    def _materialize(self, spec: dict[str, Any]) -> _Protocol:
        dur = int(spec["duration_min"])
        start_hour = float(spec["start_hour"])
        meals_4t = [tuple(m) for m in spec["meals"]]
        sw, act = self._resolve_sleep_activity(spec)
        traj, absorption = simulate_full_body(
            PatientParams(), meals_4t, sw, act, dur, start_hour,
            noise_scale=0.0, rng=np.random.default_rng(_PROTOCOL_SEED),
        )
        traj = np.asarray(traj, dtype=np.float32)[:dur]
        absorption = np.asarray(absorption, dtype=np.float32)[:dur]
        return _Protocol(
            name=str(spec["name"]),
            duration_min=dur,
            start_hour=start_hour,
            meal_events=_meal_events(meals_4t),
            sleep_wake=sw,
            activity=act,
            initial_state=traj[0].copy(),
            reference=traj,
            cold_rate=_finite_diff_rate(traj),
            cold_absorption=absorption,
        )

    def weight_at(self, epoch: int) -> float:
        return self.weight.at(epoch)

    # -- embedding calibration ------------------------------------------------

    def _calibrate(
        self,
        model: nn.Module,
        proto: _Protocol,
        warm_start: torch.Tensor,
        n_steps: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Optimize an embedding to reproduce ``proto``'s cold observed-marker
        trajectory — the same objective ``pulse.benchmark.calibrate_embedding``
        minimizes (normalized obs MSE + L2 prior), full-trajectory rather than
        windowed (the protocols are <=1 day). Model is assumed already in eval
        mode. The gut outputs are precomputed once at the warm-start embedding
        and reused across steps — they are only weakly embedding-dependent and
        this halves the per-step cost.
        """
        scale = torch.tensor(NORM_SCALE, dtype=torch.float32, device=device)
        initial = torch.tensor(proto.initial_state, dtype=torch.float32, device=device)
        sw = torch.tensor(proto.sleep_wake, dtype=torch.float32, device=device)
        act = torch.tensor(proto.activity, dtype=torch.float32, device=device)
        ref = torch.tensor(proto.reference, dtype=torch.float32, device=device)
        start_min = proto.start_hour * 60.0
        T = ref.shape[0]

        # Calibration targets: either explicit (time, marker, value) check-ins
        # (bench-cohort protocols — mirrors what the bench calibrates against)
        # or a dense grid sample of the cold trajectory on ``obs_markers``.
        if proto.obs_points:
            pts = [(t, m, v) for (t, m, v) in proto.obs_points if 0 <= t < T]
            t_idx = torch.tensor([t for t, _, _ in pts], dtype=torch.long, device=device)
            m_idx = torch.tensor([m for _, m, _ in pts], dtype=torch.long, device=device)
            tgt = torch.tensor([v for _, _, v in pts], dtype=torch.float32, device=device)
        else:
            obs_idx = torch.tensor(self._obs_idx, dtype=torch.long, device=device)

        emb = warm_start.detach().to(device).clone().requires_grad_(True)
        opt = torch.optim.Adam([emb], lr=self.calib_lr)
        with torch.no_grad():
            gut = precompute_gut_outputs(
                model, emb.detach(), proto.duration_min,
                dt=1.0, start_time_minutes=start_min, meals=list(proto.meal_events),
            )
        # Iter 67 numerics fix: the iter-65 NaN crash (Phase 2 epoch 58, all
        # dist_* markers simultaneously NaN) traced to this loop. Adam can
        # take a divergent step that pushes emb to NaN/Inf — once cached and
        # used as the rate-matching embedding, every downstream forward
        # produces NaN. Two guards: (1) clamp emb each step to keep it in a
        # plausible neighbourhood of zero (the patient table itself initializes
        # with std=0.1, so an L∞ ≤ ``_CALIB_EMB_CLAMP`` ball comfortably
        # encloses the calibration target). (2) abort the calibration step on
        # non-finite loss/grad rather than letting NaN propagate into emb. The
        # caller validates emb finiteness post-return.
        # Iter 68: gradient checkpointing on the calibration integrate. The
        # iter-67 saga's "phase-2 OOM" turned out to be a single
        # ``loss.backward()`` through the unchunked ``proto.duration_min``-step
        # autograd graph hanging > 1 h on the wider iter-66+ model
        # (EMBEDDING_DIM=64, n_species=4). Chunking the integrate at
        # ~sqrt(n_steps) segments drops the held-activation memory from
        # O(T) to O(sqrt(T)) and the backward time correspondingly. Each
        # chunk's forward is re-executed once during backward — model is in
        # eval mode here (no dropout/BN), so this is mathematically
        # equivalent to the unchunked path.
        ckpt_segs = max(1, int(math.sqrt(max(1, int(proto.duration_min)))))
        for _ in range(max(0, int(n_steps))):
            opt.zero_grad()
            with torch.enable_grad():
                pred = integrate(
                    model, initial, emb, proto.duration_min,
                    dt=1.0, start_time_minutes=start_min, meals=list(proto.meal_events),
                    gut_outputs=gut, sleep_wake=sw, activity=act,
                    checkpoint_segments=ckpt_segs,
                )
                t = min(pred.shape[0], T)
                if proto.obs_points:
                    keep = t_idx < t
                    resid = (pred[t_idx[keep], m_idx[keep]] - tgt[keep]) / scale[m_idx[keep]]
                else:
                    resid = (pred[:t][:, obs_idx] - ref[:t][:, obs_idx]) / scale[obs_idx]
                loss = resid.pow(2).mean() + self.calib_l2 * emb.pow(2).mean()
            if not torch.isfinite(loss):
                # The Adam state already holds the prior emb; just don't step.
                # Returning the unmodified emb lets the next epoch's warm-start
                # try again with the same parameters.
                break
            loss.backward()
            if emb.grad is None or not torch.isfinite(emb.grad).all():
                opt.zero_grad()
                break
            opt.step()
            with torch.no_grad():
                emb.clamp_(-_CALIB_EMB_CLAMP, _CALIB_EMB_CLAMP)
        out = emb.detach()
        if not torch.isfinite(out).all():
            # Adam diverged before our per-step guards caught it (e.g., a NaN
            # in the very first backward). Fall back to the warm start (which
            # is itself validated finite — see _calibrated_embedding).
            return warm_start.detach().to(device).clone()
        return out

    def _calibrated_embedding(
        self,
        model: nn.Module,
        proto: _Protocol,
        pidx: int,
        ctx: SignalContext,
    ) -> torch.Tensor:
        cached = self._calib_cache.get(pidx)
        fresh = cached is not None and (ctx.epoch - cached[1]) < self.recalib_every
        if fresh:
            return cached[0].to(ctx.device)
        if cached is None:
            warm = torch.zeros(EMBEDDING_DIM)
            n_steps = self.calib_steps
        else:
            warm = cached[0]
            n_steps = self.calib_warm_steps
        emb = self._calibrate(model, proto, warm, n_steps, ctx.device)
        # Iter 67: never cache a non-finite emb. _calibrate already falls back
        # to warm on internal NaN; this is the belt-and-suspenders check that
        # nothing escapes to the rate-matching forward.
        if not torch.isfinite(emb).all():
            emb = warm.detach().to(ctx.device).clone()
        self._calib_cache[pidx] = (emb.detach().cpu(), ctx.epoch)
        self._n_recalibs += 1
        return emb.to(ctx.device)

    # -- per-protocol loss terms (one entry per distilled marker) ------------

    def _trajectory_terms(
        self,
        model: nn.Module,
        proto: _Protocol,
        emb: torch.Tensor,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Roll out from the cold initial state at ``emb`` and match the
        *integrated* unobserved-marker trajectories (Huber on NORM_SCALE-
        normalized residuals). The iter 47-49 loss; kept for back-compat —
        the 2026-05-13 diagnostic shows its gradient to the dead-marker heads
        is diluted ~T× through the integration chain and arrives ~1e-5.
        """
        scale = torch.tensor(NORM_SCALE, dtype=torch.float32, device=device)
        initial = torch.tensor(proto.initial_state, dtype=torch.float32, device=device)
        sw = torch.tensor(proto.sleep_wake, dtype=torch.float32, device=device)
        act = torch.tensor(proto.activity, dtype=torch.float32, device=device)
        ref = torch.tensor(proto.reference, dtype=torch.float32, device=device)
        start_min = proto.start_hour * 60.0
        gut = precompute_gut_outputs(
            model, emb, proto.duration_min,
            dt=1.0, start_time_minutes=start_min, meals=list(proto.meal_events),
        )
        pred = integrate(
            model, initial, emb, proto.duration_min,
            dt=1.0, start_time_minutes=start_min, meals=list(proto.meal_events),
            gut_outputs=gut, sleep_wake=sw, activity=act,
        )
        T = min(pred.shape[0], ref.shape[0])
        out: dict[str, torch.Tensor] = {}
        for marker, midx in self._marker_idx:
            pred_n = pred[:T, midx] / scale[midx]
            ref_n = ref[:T, midx] / scale[midx]
            out[marker] = F.huber_loss(pred_n, ref_n, delta=_HUBER_DELTA, reduction="mean")
        return out

    def _rate_terms(
        self,
        model: nn.Module,
        proto: _Protocol,
        emb: torch.Tensor,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Teacher-forced rate matching (iter 50): one batched
        ``model.forward`` over the whole protocol, feeding the cold state and
        the cold ODE's gut/absorption coupling at every step, then Huber on
        the normalized d/dt residual. Gradient straight to the rate heads —
        no integration chain — ~T× stronger than ``_trajectory_terms`` and
        the directly motivated fix for the diagnostic's vanishing-gradient
        finding. Matching rates ⊇ matching trajectories under Euler dt=1.
        """
        T = int(proto.cold_rate.shape[0])
        if T == 0:
            return {}
        scale = torch.tensor(NORM_SCALE, dtype=torch.float32, device=device)
        cold_state = torch.tensor(proto.reference[:T], dtype=torch.float32, device=device)
        cold_rate = torch.tensor(proto.cold_rate, dtype=torch.float32, device=device)
        gut_in = torch.tensor(proto.cold_absorption[:T], dtype=torch.float32, device=device)
        sw = torch.tensor(proto.sleep_wake[:T], dtype=torch.float32, device=device)
        act = torch.tensor(proto.activity[:T], dtype=torch.float32, device=device)
        start_min = proto.start_hour * 60.0
        t_min = torch.tensor(
            [(start_min + k) % 1440.0 for k in range(T)],
            dtype=torch.float32, device=device,
        )
        emb_b = emb.unsqueeze(0).expand(T, -1)
        rates = model(
            cold_state, emb_b, t_min, list(proto.meal_events),
            sleep_wake=sw, activity=act, gut_override=gut_in,
        )
        # Iter 67: loud assert on non-finite rates. If we reach here with NaN
        # output despite the calibration guards, something else in the model
        # has gone numerically wrong and silently emitting NaN losses would
        # paper over it — surface immediately so safe_step's abort dump
        # captures the protocol context (per memory: prefer loud errors over
        # graceful degradation).
        if not torch.isfinite(rates).all():
            raise RuntimeError(
                f"cold_model_distillation: model rates non-finite at protocol "
                f"{proto.name!r} (T={T}). emb finite={bool(torch.isfinite(emb).all().item())}, "
                f"cold_state finite={bool(torch.isfinite(cold_state).all().item())}, "
                f"gut_in finite={bool(torch.isfinite(gut_in).all().item())}.",
            )
        out: dict[str, torch.Tensor] = {}
        for marker, midx in self._marker_idx:
            pred_dr = rates[:, midx] * _RATE_NORM_MINUTES / scale[midx]
            cold_dr = cold_rate[:, midx] * _RATE_NORM_MINUTES / scale[midx]
            out[marker] = F.huber_loss(pred_dr, cold_dr, delta=_HUBER_DELTA, reduction="mean")
        return out

    def _level_terms(
        self,
        model: nn.Module,
        proto: _Protocol,
        emb: torch.Tensor,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Iter 75: short-horizon free-running *level* anchor.

        Teacher-forced rate matching (``_rate_terms``) is offset-invariant: it
        feeds the cold state at every step and matches d/dt only, so the model
        can score a perfect rate while its OWN integrated level drifts — the
        same disease iter 73 fixed for dose-response (slope→peak). This term
        closes that hole on the supervisor that owns the latent hormones.

        ``anchor_samples`` windows of ``anchor_window`` minutes are evenly
        spaced across the protocol. From the cold state at each window start
        the model rolls out *freely* (its own integrated state, not teacher-
        forced) under the cold ODE's gut/absorption + sleep/activity coupling,
        and the Huber on the NORM_SCALE-normalized *absolute* level over the
        window pins it to the cold reference. Resetting to truth every window
        bounds each integration chain to W steps — gradient dilution ~W×
        (60×), not the ~T× (1440×) that left full-trajectory mode's gradient at
        ~1e-5 — while still penalising free-rollout drift the rate term is
        blind to. The windows are independent graphs rooted at constant cold
        states but all are retained until the signal's single backward(), so
        capping the count (vs tiling all of [0, T)) keeps peak memory bounded
        (~samples×W steps).
        """
        T = int(proto.reference.shape[0])
        W = max(2, int(self.anchor_window))
        if T < 2:
            return {}
        scale = torch.tensor(NORM_SCALE, dtype=torch.float32, device=device)
        ref_all = torch.tensor(proto.reference, dtype=torch.float32, device=device)
        absorp = torch.tensor(proto.cold_absorption, dtype=torch.float32, device=device)
        sw_all = torch.tensor(proto.sleep_wake, dtype=torch.float32, device=device)
        act_all = torch.tensor(proto.activity, dtype=torch.float32, device=device)
        start_min = proto.start_hour * 60.0
        # Evenly-spaced window starts over [0, T-W], capped at anchor_samples.
        last_start = max(0, T - W)
        n = max(1, min(int(self.anchor_samples), last_start // W + 1))
        if n == 1:
            starts = [0]
        else:
            starts = sorted({(last_start * k) // (n - 1) for k in range(n)})
        per_marker: dict[str, list[torch.Tensor]] = {m: [] for m, _ in self._marker_idx}
        for t0 in starts:
            w = min(W, T - t0)
            if w < 2:
                continue
            # Roll out freely from the cold state at t0. gut_outputs carries the
            # cold ODE's absorption so the rollout sees the same meal coupling
            # the reference did (meals=[] — the gut appearance is already in
            # gut_outputs; the unobserved markers couple to meals only through
            # it). integrate returns out[0]=initial, out[k]≈ref[t0+k].
            pred = integrate(
                model, ref_all[t0], emb, w, dt=1.0,
                start_time_minutes=start_min + t0, meals=[],
                sleep_wake=sw_all[t0:t0 + w], activity=act_all[t0:t0 + w],
                gut_outputs=absorp[t0:t0 + w],
            )
            if not torch.isfinite(pred).all():
                self._n_anchor_skips += 1
                continue
            seg = ref_all[t0:t0 + w]
            L = min(pred.shape[0], seg.shape[0])
            for marker, midx in self._marker_idx:
                pred_n = pred[:L, midx] / scale[midx]
                ref_n = seg[:L, midx] / scale[midx]
                per_marker[marker].append(
                    F.huber_loss(pred_n, ref_n, delta=_HUBER_DELTA, reduction="mean")
                )
        return {m: torch.stack(v).mean() for m, v in per_marker.items() if v}

    # -- main signal ----------------------------------------------------------

    def compute(
        self,
        model: nn.Module,
        embeddings: nn.Embedding,
        ctx: SignalContext,
    ) -> SignalResult:
        w = self.weight_at(ctx.epoch)
        if w <= 0 or not self._protocols or not self._marker_idx:
            return SignalResult()

        device = ctx.device
        n = min(self.protocols_per_epoch, len(self._protocols))
        idxs = ctx.rng.choice(len(self._protocols), size=n, replace=False)

        per_marker_terms: dict[str, list[torch.Tensor]] = {m: [] for m, _ in self._marker_idx}
        protocol_losses: list[torch.Tensor] = []

        was_training = model.training
        model.eval()
        try:
            for i in idxs:
                proto = self._protocols[int(i)]
                emb = self._calibrated_embedding(model, proto, int(i), ctx)
                if self.mode == "anchored":
                    # Rate (shape) + short-horizon free-rollout level anchor.
                    # Average the two halves per marker so each contributes at
                    # the configured signal weight; a marker present in only
                    # one half (e.g. all level windows diverged) falls back to
                    # whichever half it has.
                    rate_t = self._rate_terms(model, proto, emb, device)
                    lvl_t = self._level_terms(model, proto, emb, device)
                    terms = {}
                    for marker, _ in self._marker_idx:
                        halves = [t for t in (rate_t.get(marker), lvl_t.get(marker)) if t is not None]
                        if halves:
                            terms[marker] = torch.stack(halves).mean()
                elif self.mode == "rate":
                    terms = self._rate_terms(model, proto, emb, device)
                else:
                    terms = self._trajectory_terms(model, proto, emb, device)
                if not terms:
                    continue
                for marker, term in terms.items():
                    per_marker_terms[marker].append(term.detach())
                protocol_losses.append(torch.stack(list(terms.values())).mean())
        finally:
            if was_training:
                model.train()

        if not protocol_losses:
            return SignalResult()
        loss = torch.stack(protocol_losses).mean()

        sub_metrics: dict[str, float] = {
            f"dist_{m}": float(torch.stack(v).mean().item())
            for m, v in per_marker_terms.items() if v
        }
        sub_metrics["n_protocols"] = float(len(protocol_losses))
        sub_metrics["n_recalibs"] = float(self._n_recalibs)
        sub_metrics["mode"] = {"trajectory": 0.0, "rate": 1.0, "anchored": 2.0}.get(self.mode, 0.0)
        if self.mode == "anchored":
            sub_metrics["anchor_skips"] = float(self._n_anchor_skips)

        safe_step(
            w * loss,
            ctx,
            signal=self.name,
            extra={
                "raw_loss": float(loss.detach().item()),
                "weight": float(w),
                **sub_metrics,
            },
        )

        return SignalResult(
            loss_sum=float(loss.detach().item()),
            n_units=len(protocol_losses),
            sub_metrics=sub_metrics,
        )
