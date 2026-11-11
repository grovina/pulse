"""
Cold initial states for cohort / rule arms, in ONE frame shared by training and
the teacher audits.

Iter 97 (review 4.5 / 4.9). Two things used to disagree silently:

* The cold initial state for an arm was ``cold_model_trajectory(...)[0]`` — the
  teacher's FED row 0 — even for arms whose label says "24 h fasted". The
  ``liver_glycogen_depleted_prolonged_fast`` rule then demanded < 30 g on what
  was really hours 0-24 of a fast (teacher: 100 g -> 55 g), and a cohort spec
  demanded 60 +/- 20 g on the same physiology.
* The teacher audit ran undeclared arms at activity 0 / awake while training
  ran them at the student's learned defaults.

``teacher_arm_trajectory`` is the single definition of "the teacher under this
arm": pre-run the declared fast (meal-free, at rest, awake unless the arm says
otherwise), then the arm itself. ``cold_initial_state_for_arm`` is its first
row. Both signals and both audit scripts call these.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from ..cohort_loss import ARM_DEFAULT_ACTIVITY, ARM_DEFAULT_SLEEP_WAKE
from ..knowledge.cohort_types import CohortArmSpec, InitMode
from ..knowledge.full_body import PatientParams, simulate_full_body
from ..model import integrate
from ..modules.gut import MealEvent
from ..types import NORM_CENTER, STATE_DIM
from .factors import FED_DAY_MEALS, FED_DAY_MINUTES, SLEEP_END_HOUR, SLEEP_START_HOUR

# The cold-init rollout keeps the tiny noise the textbook scenarios use (so the
# initial rows are the same the benchmark seeds from); the audit passes 0.
_COLD_INIT_NOISE = 0.001


def arm_input_series(arm: CohortArmSpec) -> tuple[np.ndarray, np.ndarray]:
    """The (sleep_wake, activity) series an arm runs under, explicit."""
    n = int(arm.duration_min)
    sw = (
        np.asarray(arm.sleep_wake, dtype=np.float32)
        if arm.sleep_wake is not None
        else np.full(n, ARM_DEFAULT_SLEEP_WAKE, dtype=np.float32)
    )
    act = (
        np.asarray(arm.activity, dtype=np.float32)
        if arm.activity is not None
        else np.full(n, ARM_DEFAULT_ACTIVITY, dtype=np.float32)
    )
    if sw.shape[0] != n or act.shape[0] != n:
        raise ValueError(
            f"arm {arm.label}: series length {sw.shape[0]}/{act.shape[0]} != duration {n}",
        )
    return sw, act


def teacher_arm_trajectory(
    arm: CohortArmSpec,
    params: PatientParams | None = None,
    *,
    rng: np.random.Generator | None = None,
    noise_scale: float = 0.0,
    include_prefast: bool = False,
) -> np.ndarray:
    """Teacher trajectory over the arm, ``[duration_min, STATE_DIM]``.

    With ``arm.prefast_hours > 0`` the teacher is first run meal-free (at rest,
    awake) for that long starting at ``start_hour - prefast_hours``, and the arm
    is simulated as a continuation; only the arm's rows are returned unless
    ``include_prefast`` is set. The rollout is ONE continuous simulation so every
    slow pool (glycogen, ketones, bile) carries over exactly.
    """
    params = params or PatientParams()
    n = int(arm.duration_min)
    sw, act = arm_input_series(arm)
    pre = int(round(float(arm.prefast_hours) * 60.0))
    if pre <= 0:
        traj, _ = simulate_full_body(
            params, [tuple(m) for m in arm.meals], sw, act, n, float(arm.start_hour),
            noise_scale=noise_scale, rng=rng,
        )
        return np.asarray(traj, dtype=np.float32)[:n]
    sw_full = np.concatenate([np.full(pre, ARM_DEFAULT_SLEEP_WAKE, dtype=np.float32), sw])
    act_full = np.concatenate([np.full(pre, ARM_DEFAULT_ACTIVITY, dtype=np.float32), act])
    meals = [(float(t) + pre, float(c), float(f), float(p)) for t, c, f, p in arm.meals]
    start_hour = (float(arm.start_hour) - pre / 60.0) % 24.0
    traj, _ = simulate_full_body(
        params, meals, sw_full, act_full, n + pre, start_hour,
        noise_scale=noise_scale, rng=rng,
    )
    traj = np.asarray(traj, dtype=np.float32)
    return traj if include_prefast else traj[pre:pre + n]


def cold_initial_state_for_arm(
    arm: CohortArmSpec,
    params: PatientParams | None = None,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """The row the student's rollout of ``arm`` starts from.

    Legacy arms (``prefast_hours == 0``) get the teacher's row 0, exactly as
    before (same noise as ``cold_model_trajectory``). Pre-fasted arms get the
    teacher's state at the end of the declared fast.
    """
    return teacher_arm_trajectory(arm, params, rng=rng, noise_scale=_COLD_INIT_NOISE)[0].copy()


def _asleep(clock_hour: torch.Tensor) -> torch.Tensor:
    return (clock_hour >= SLEEP_START_HOUR) | (clock_hour < SLEEP_END_HOUR)


def _spinup_protocol(
    arm: CohortArmSpec, fed: int, device: torch.device,
) -> tuple[int, float, list[MealEvent], torch.Tensor]:
    """The spin-up an arm starts from: ``(minutes, start clock in minutes, meals, sleep_wake)``.

    The arm begins at ``arm.start_hour``, so the spin-up begins that long earlier. It is
    ``fed`` minutes of the standard fed day (three meals, asleep 23:00-07:00) and then
    ``arm.prefast_hours`` of meal-free rest, awake.
    """
    pre = int(round(float(arm.prefast_hours) * 60.0))
    total = fed + pre
    start_hour = (float(arm.start_hour) - total / 60.0) % 24.0
    meals: list[MealEvent] = []
    for hour, carbs, fats, proteins in FED_DAY_MEALS:
        minute = (hour * 60.0 - start_hour * 60.0) % (24.0 * 60.0)
        if minute < fed:
            meals.append(MealEvent(time=float(minute), carbs=carbs, fats=fats, proteins=proteins))
    minutes = torch.arange(total, dtype=torch.float64, device=device)
    clock = (start_hour + minutes / 60.0) % 24.0
    fed_sleep = torch.where(_asleep(clock), 0.0, 1.0)
    sleep = torch.where(minutes < fed, fed_sleep, ARM_DEFAULT_SLEEP_WAKE).to(torch.float32)
    return total, start_hour * 60.0, meals, sleep


def student_initial_states_for_rows(
    model: torch.nn.Module,
    rows: Sequence[tuple[torch.Tensor, CohortArmSpec]],
    *,
    fed_day_minutes: int = FED_DAY_MINUTES,
) -> list[torch.Tensor]:
    """Where each ``(codes[B_i, EMB], arm)`` row group starts: the student's own fed
    day, then the arm's pre-fast. Each result is ``[B_i, STATE_DIM]`` and detached.

    The spin-up starts at the marker typicals and ends at the arm's ``start_hour``
    (see ``_spinup_protocol``). Every row group with the same spin-up length rolls
    together as ONE no-grad ``integrate`` call, each member on its own clock, meals
    and sleep: a step costs about the same at batch 1 and at batch 100, and the day
    is what the arms of a step would otherwise each redo. The protocol that follows is
    in the graph, the day that prepared it is not. A day of activations held for every
    item is what exhausted the trainer.
    """
    fed = max(int(fed_day_minutes), 0)
    out: list[torch.Tensor | None] = [None] * len(rows)
    by_length: dict[int, list[int]] = {}
    for i, (codes, arm) in enumerate(rows):
        basal = torch.tensor(NORM_CENTER, dtype=torch.float32, device=codes.device)
        out[i] = basal.unsqueeze(0).expand(int(codes.shape[0]), -1).contiguous()
        total = fed + int(round(float(arm.prefast_hours) * 60.0))
        if total > 0 and int(codes.shape[0]) > 0:
            by_length.setdefault(total, []).append(i)
    for total, members in by_length.items():
        device = rows[members[0]][0].device
        protocols = [_spinup_protocol(rows[i][1], fed, device) for i in members]
        sizes = [int(rows[i][0].shape[0]) for i in members]
        with torch.no_grad():
            traj = integrate(
                model, torch.cat([out[i] for i in members]),
                torch.cat([rows[i][0] for i in members]), total,
                dt=1.0,
                start_time_minutes=torch.cat([
                    torch.full((n,), start, dtype=torch.float64, device=device)
                    for n, (_t, start, _m, _s) in zip(sizes, protocols)
                ]),
                meals=[meals for n, (_t, _s, meals, _w) in zip(sizes, protocols) for _ in range(n)],
                sleep_wake=torch.cat([sleep.expand(n, total) for n, (_t, _s, _m, sleep) in zip(sizes, protocols)]),
                activity=torch.full((total,), ARM_DEFAULT_ACTIVITY, dtype=torch.float32, device=device),
            )
        final = traj[:, -1, :STATE_DIM].detach()
        row = 0
        for i, n in zip(members, sizes):
            out[i] = final[row:row + n]
            row += n
    return out  # type: ignore[return-value]


def student_initial_states(
    model: torch.nn.Module,
    codes: torch.Tensor,
    arm: CohortArmSpec,
    *,
    fed_day_minutes: int = FED_DAY_MINUTES,
) -> torch.Tensor:
    """``student_initial_states_for_rows`` for one arm: ``codes`` is ``[B, EMBEDDING_DIM]``
    and the result ``[B, STATE_DIM]``. ``fed_day_minutes`` is the length of the fed day;
    the training path uses ``FED_DAY_MINUTES``."""
    return student_initial_states_for_rows(
        model, [(codes, arm)], fed_day_minutes=fed_day_minutes,
    )[0]


def item_initial_state(init_mode: InitMode, arm: CohortArmSpec, seed: int) -> np.ndarray:
    """The row an evidence item's arm starts from, as training starts it."""
    if init_mode is InitMode.NORM_CENTER:
        return np.array(NORM_CENTER, dtype=np.float32)
    return cold_initial_state_for_arm(arm, PatientParams(), rng=np.random.default_rng(seed))


__all__ = [
    "arm_input_series",
    "cold_initial_state_for_arm",
    "item_initial_state",
    "student_initial_states",
    "student_initial_states_for_rows",
    "teacher_arm_trajectory",
]
