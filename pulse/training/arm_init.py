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

import numpy as np

from ..cohort_loss import ARM_DEFAULT_ACTIVITY, ARM_DEFAULT_SLEEP_WAKE
from ..knowledge.cohort_types import CohortArmSpec
from ..knowledge.full_body import PatientParams, simulate_full_body

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


__all__ = ["arm_input_series", "cold_initial_state_for_arm", "teacher_arm_trajectory"]
