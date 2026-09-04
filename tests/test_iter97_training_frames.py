"""Iter 97 (training): frame and convention fixes.

Review 1.2 (the rules' sleep arm was inverted), 1.4 (meal lookback 120 vs the
480-min gut kernel), 1.5 (argparse defaults that shadowed the function
defaults), 1.7 (undeclared cohort arms ran at the student's learned activity
instead of rest = 0), 4.8 (the verifier surrogate was a zero-gradient constant
on training windows) and the ruler hand-off (--frozen-ruler, dataset URI).
"""

from __future__ import annotations

import inspect
import os
import sys

import numpy as np
import pytest
import torch

import pulse

REPO = os.path.dirname(os.path.dirname(os.path.abspath(pulse.__file__)))
assert pulse.__file__.startswith(REPO), pulse.__file__

from pulse import cohort_loss  # noqa: E402
from pulse.cohort_loss import ARM_DEFAULT_ACTIVITY, ARM_DEFAULT_SLEEP_WAKE  # noqa: E402
from pulse.knowledge.cohort_types import CohortArmSpec  # noqa: E402
from pulse.knowledge.physiology_rules import _SLEEP_WAKE_24H_ARM  # noqa: E402
from pulse.model import ModularPhysiologyNetwork  # noqa: E402
from pulse.modules.gut import MEAL_ACTIVE_WINDOW_MIN  # noqa: E402
from pulse.train import build_arg_parser, train  # noqa: E402
from pulse.training.trajectory_signal import meals_in_window  # noqa: E402
from pulse.training_verifier_loss import training_verifier_surrogate_loss  # noqa: E402
from pulse.types import EMBEDDING_DIM, MARKER_INDEX, NORM_CENTER, STATE_DIM  # noqa: E402


# --- 1.2: sleep arm convention ---------------------------------------------

def test_sleep_arm_uses_engine_convention_zero_is_asleep() -> None:
    sw = np.asarray(_SLEEP_WAKE_24H_ARM.sleep_wake)
    assert sw[3 * 60] == 0.0, "03:00 must be asleep (0)"
    assert sw[12 * 60] == 1.0, "12:00 must be awake (1)"
    assert sw[22 * 60] == 0.0 and sw[6 * 60] == 1.0
    # The arm also declares rest explicitly (review 1.7 / 4.9).
    assert _SLEEP_WAKE_24H_ARM.activity is not None
    assert max(_SLEEP_WAKE_24H_ARM.activity) == 0.0


def test_sleep_arm_matches_the_cohort_files_convention() -> None:
    # The cohort files are the reference for the convention: their sleep arms
    # put 0 at night. If this ever disagrees with the rules arm, one of them is
    # inverted again.
    from pulse.knowledge.cohorts.sleep import SLEEP_RESTRICTION_NEXT_DAY_GLUCOSE
    arm = SLEEP_RESTRICTION_NEXT_DAY_GLUCOSE.arms[0]
    sw = np.asarray(arm.sleep_wake)
    night_idx = int(((3.0 - arm.start_hour) % 24) * 60)  # 03:00 of the first night
    assert sw[night_idx] < 0.5
    assert np.asarray(_SLEEP_WAKE_24H_ARM.sleep_wake)[3 * 60] < 0.5


# --- 1.4: meal lookback ------------------------------------------------------

def test_meal_lookback_equals_gut_active_window() -> None:
    # The lookback IS the kernel's active window (720 since the teacher's slow
    # carbohydrate component was made mass-conserving), never a separate literal.
    assert MEAL_ACTIVE_WINDOW_MIN >= 480.0
    meals = [(1000.0, 60.0, 10.0, 15.0)]  # 300 min before the window
    win = meals_in_window(meals, win_start=1300, win_end=1540)
    assert len(win) == 1 and win[0].time == pytest.approx(-300.0)
    # Just outside the kernel's active window: invisible, as it is to the kernel.
    assert meals_in_window([(1300 - MEAL_ACTIVE_WINDOW_MIN - 1.0, 60.0, 0.0, 0.0)], 1300, 1540) == []
    # Just inside it: visible.
    assert len(meals_in_window([(1300 - MEAL_ACTIVE_WINDOW_MIN + 1.0, 60.0, 0.0, 0.0)], 1300, 1540)) == 1
    # The legacy 120-min lookback would have dropped it (the review's 67 %).
    assert meals_in_window(meals, 1300, 1540, lookback_min=120.0) == []


# --- 1.5: argparse defaults defer to the function defaults -------------------

def test_argparse_defaults_do_not_shadow_function_defaults() -> None:
    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.cold_distill_markers is None
    assert args.verifier_loss_weight is None
    fn_defaults = inspect.signature(train).parameters
    # Every flag that maps 1:1 onto a train() kwarg and carries a concrete
    # argparse default must agree with the function default.
    skip = {"spec", "gcs_bucket", "gcs_object", "benchmark_dataset_uri",
            "benchmark_thresholds_uri", "benchmark_report_path", "deterministic",
            "benchmark_only", "frozen_ruler", "contribution_weights_json",
            "equal_contribution_weights", "dose_response_markers",
            "default_baseline_markers", "cohort_cold_init"}
    mismatched = []
    for name, val in vars(args).items():
        if name in skip or val is None or name not in fn_defaults:
            continue
        fd = fn_defaults[name].default
        if isinstance(fd, tuple):
            continue
        if fd is None and val == "":
            continue  # string flag parsed into a structured None
        if fd != val:
            mismatched.append((name, val, fd))
    assert not mismatched, f"argparse default != train() default: {mismatched}"


def test_frozen_ruler_flag_exists() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(["--frozen-ruler", "/tmp/x.json"])
    assert args.frozen_ruler == "/tmp/x.json"


# --- 1.7 / 4.9: undeclared arms run at rest, awake ---------------------------

def test_undeclared_arm_series_run_at_rest_and_awake(monkeypatch) -> None:
    assert ARM_DEFAULT_ACTIVITY == 0.0 and ARM_DEFAULT_SLEEP_WAKE == 1.0
    captured: dict[str, torch.Tensor | None] = {}

    def fake_integrate(model, state, emb, n_steps, **kw):
        captured["sleep_wake"] = kw.get("sleep_wake")
        captured["activity"] = kw.get("activity")
        return torch.zeros(state.shape[0], n_steps, STATE_DIM)

    monkeypatch.setattr(cohort_loss, "integrate", fake_integrate)
    model = ModularPhysiologyNetwork(
        metabolic_hidden=8, appetite_hidden=8, stress_hidden=8,
        cardiovascular_hidden=8, thermoreg_hidden=8, respiratory_hidden=8,
    )
    arm = CohortArmSpec(label="undeclared", duration_min=30, start_hour=8.0, meals=())
    emb = torch.zeros(2, EMBEDDING_DIM)
    state = torch.tensor(NORM_CENTER, dtype=torch.float32).unsqueeze(0).expand(2, -1)
    cohort_loss._rollout_arm_states(model, emb, arm, state)
    act = captured["activity"]
    sw = captured["sleep_wake"]
    assert act is not None and sw is not None, "series must be explicit, not None"
    assert float(act.abs().max()) == 0.0
    assert float(sw.min()) == 1.0 and act.shape == (30,)


# --- 4.8: verifier surrogate ---------------------------------------------------

def _flat_window(T: int) -> torch.Tensor:
    return torch.tensor(NORM_CENTER, dtype=torch.float32).unsqueeze(0).repeat(T, 1)


def test_verifier_surrogate_skips_phases_the_window_cannot_contain() -> None:
    # A 240-min window starting at 12:00 has neither a morning nor an evening
    # cortisol phase and no night: the circadian and sleep terms must be ABSENT
    # (no gradient on cortisol / temp / hr), not constants.
    x = _flat_window(240).requires_grad_(True)
    loss = training_verifier_surrogate_loss(x, meals=[], start_hour=12.0)
    loss.backward()
    g = x.grad
    for m in ("cortisol", "temp"):
        assert float(g[:, MARKER_INDEX[m]].abs().sum()) == 0.0, m
    # The review's measured constant floor on such a window was 0.1780; the
    # remaining coupling/sanity terms are strictly below it.
    assert float(loss.detach()) < 0.17


def test_verifier_surrogate_scores_sleep_dip_on_an_overnight_window() -> None:
    # 19:00 -> 07:00 (720 min): evening reference + night, exactly the cgm_real
    # shape. The sleep-dip term must exist (gradient on hr), which the old
    # T >= 1440 gate made impossible.
    T = 720
    x = _flat_window(T)
    hr = MARKER_INDEX["hr"]
    x[:, hr] = 70.0
    x = x.requires_grad_(True)
    loss = training_verifier_surrogate_loss(x, meals=[], start_hour=19.0)
    loss.backward()
    assert float(x.grad[:, hr].abs().sum()) > 0.0


def test_verifier_surrogate_matches_verifier_reference_policy() -> None:
    # Same window, numpy verifier: it must also report a sleep check with the
    # evening reference — the surrogate and the ruler agree on what is scored.
    from pulse.verifier import evaluate_weak_checks
    T = 720
    traj = np.tile(np.asarray(NORM_CENTER, dtype=np.float32), (T, 1))
    res = evaluate_weak_checks(traj, meals=[], start_hour=19.0)
    keys = {c["key"] for c in res["checks"]}
    assert "sleep_hr_dip" in keys
    # 19:00-07:00 also covers an evening (18-22) and a morning (06-09), so the
    # cortisol circadian check exists on both sides — and the surrogate then
    # carries a cortisol gradient on the same window.
    assert "circadian_cortisol_morning_peak" in keys
    x = _flat_window(T).requires_grad_(True)
    training_verifier_surrogate_loss(x, meals=[], start_hour=19.0).backward()
    assert float(x.grad[:, MARKER_INDEX["cortisol"]].abs().sum()) > 0.0
