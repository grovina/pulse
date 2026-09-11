"""Iter 97 (training): the training loop — aux cadence, phases, gradient balance, recipe pinning.

Review 4.1 (25 literature steps per run), 4.9 (phase 3 with dropout on all
inputs), 4.11 (per-signal gradient norms), 1.5 (spec == recorded config).
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch
import torch.nn as nn

import pulse

REPO = os.path.dirname(os.path.dirname(os.path.abspath(pulse.__file__)))
assert pulse.__file__.startswith(REPO), pulse.__file__

from pulse.model import ModularPhysiologyNetwork  # noqa: E402
from pulse.train import (  # noqa: E402
    _load_spec_train_args,
    _merge_results,
    _parse_aux_cadence,
    build_arg_parser,
    resolved_train_config,
    spec_config_divergence,
)
from pulse.training import (  # noqa: E402
    EmbeddingPriorSignal,
    SignalContext,
    SignalResult,
    TrajectoryRolloutSignal,
    WeightSchedule,
    accumulate_grad,
    joint_aux_step,
)
from pulse.types import EMBEDDING_DIM  # noqa: E402


def _tiny_model() -> ModularPhysiologyNetwork:
    return ModularPhysiologyNetwork(
        metabolic_hidden=8, appetite_hidden=8, stress_hidden=8,
        cardiovascular_hidden=8, thermoreg_hidden=8, respiratory_hidden=8,
    )


def _ctx(params, **kw) -> SignalContext:
    opt = torch.optim.SGD(params, lr=1e-3)
    base = dict(epoch=0, total_epochs=1, rng=np.random.default_rng(0), device=torch.device("cpu"),
                optimizer=opt, params=params, grad_clip=10.0)
    base.update(kw)
    return SignalContext(**base)


# --- 4.1: the trajectory signal yields per window so aux steps can interleave ---

def test_trajectory_signal_yields_per_window_and_keeps_its_result() -> None:
    sig = TrajectoryRolloutSignal(
        n_patients=1, n_days=1, seed=0, contribution_weights={"full_body": 1.0},
        windows_per_patient=2, meal_window_bias=0.0, input_dropout=0.0, huber_delta=1.0,
        gut_loss_weight=0.0, coupling_weight=WeightSchedule(0.0), verifier_weight=WeightSchedule(0.0),
        n_default_patients=1,
    )
    torch.manual_seed(0)
    model = _tiny_model()
    emb = nn.Embedding(1, EMBEDDING_DIM)
    params = list(model.parameters()) + list(emb.parameters())
    ctx = _ctx(params)
    yields = list(sig.iter_windows(model, emb, ctx))
    assert yields == [1, 2, 3, 4]              # (1 patient + 1 default) x 2 windows
    assert sig.last_result.n_units == 4
    assert len(ctx.traj_grad_norms) == 4       # 4.11: per-window pre-clip norms recorded


# --- 4.11: per-signal gradient norms and clips -----------------------------------

def test_aux_signal_norms_are_recorded_and_clipped_before_the_joint_step() -> None:
    p = nn.Parameter(torch.tensor([3.0]))
    ctx = _ctx([p], aux_signal_clip=1.0)
    accumulate_grad((p * p).sum() * 10.0, ctx, signal="loud")     # raw grad 60
    accumulate_grad((p * p).sum() * 0.01, ctx, signal="quiet")    # raw grad 0.06
    assert ctx.aux_grad_norms["loud"][0] == pytest.approx(60.0)
    assert ctx.aux_grad_norms["quiet"][0] == pytest.approx(0.06)
    # The loud one was rescaled to the per-signal clip; the quiet one untouched.
    assert float(p.grad) == pytest.approx(1.0 + 0.06)
    assert ctx.aux_steps_by_signal == {"loud": 1, "quiet": 1}
    stats = joint_aux_step(ctx)
    assert stats["joint_grad_norm_pre_clip"] == pytest.approx(1.06)
    assert ctx.aux_accumulated is False


def test_aux_steps_accumulate_across_epochs_through_a_shared_dict() -> None:
    p = nn.Parameter(torch.tensor([1.0]))
    shared: dict[str, int] = {}
    for epoch in range(3):
        ctx = _ctx([p], epoch=epoch, aux_steps_by_signal=shared)
        accumulate_grad((p * p).sum(), ctx, signal="s")
        joint_aux_step(ctx)
    assert shared == {"s": 3}


# --- 4.1: cadence helpers ------------------------------------------------------------

def test_parse_aux_cadence_and_merge_results() -> None:
    assert _parse_aux_cadence("cold_model_distillation:3;carb_mass_balance:4") == {
        "cold_model_distillation": 3, "carb_mass_balance": 4,
    }
    assert _parse_aux_cadence("") == {}
    with pytest.raises(ValueError):
        _parse_aux_cadence("x:0")
    merged = _merge_results([
        SignalResult(loss_sum=1.0, n_units=1, sub_metrics={"a": 1.0}),
        SignalResult(loss_sum=3.0, n_units=1, sub_metrics={"a": 3.0, "b": 2.0}),
    ])
    assert merged.loss_sum == 4.0 and merged.n_units == 2
    assert merged.sub_metrics == {"a": 2.0, "b": 2.0}


def test_interleaved_loop_runs_more_aux_steps_than_once_per_epoch() -> None:
    """Drive the generator the way train() does: k=2 over 4 windows -> 2 aux steps."""
    traj = TrajectoryRolloutSignal(
        n_patients=1, n_days=1, seed=0, contribution_weights={"full_body": 1.0},
        windows_per_patient=2, meal_window_bias=0.0, input_dropout=0.0, huber_delta=1.0,
        gut_loss_weight=0.0, coupling_weight=WeightSchedule(0.0), verifier_weight=WeightSchedule(0.0),
        n_default_patients=1,
    )
    aux = [
        EmbeddingPriorSignal(name="prior_a", weight=WeightSchedule(0.1)),
        EmbeddingPriorSignal(name="prior_b", weight=WeightSchedule(0.1)),
    ]
    torch.manual_seed(0)
    model = _tiny_model()
    emb = nn.Embedding(1, EMBEDDING_DIM)
    params = list(model.parameters()) + list(emb.parameters())
    shared: dict[str, int] = {}
    ctx = _ctx(params, aux_steps_by_signal=shared)
    k = 2
    n_aux = 0
    for n_win in traj.iter_windows(model, emb, ctx):
        if n_win % k == 0:
            for sig in aux:
                sig.compute(model, emb, ctx)
            if ctx.aux_accumulated:
                joint_aux_step(ctx)
                n_aux += 1
    assert n_aux == 2
    assert shared["prior_a"] == 2 and shared["prior_b"] == 2
    assert len(ctx.aux_grad_norms["prior_a"]) == 2


# --- 4.9: arm rollouts can drop sleep/activity (capability, not phase-3 policy) ------

def test_arm_rollout_drops_series_under_input_dropout(monkeypatch) -> None:
    from pulse import cohort_loss
    from pulse.knowledge.cohort_types import CohortArmSpec
    from pulse.types import NORM_CENTER, STATE_DIM
    seen: list[tuple[bool, bool]] = []

    def fake_integrate(model, state, emb, n_steps, **kw):
        seen.append((kw.get("sleep_wake") is None, kw.get("activity") is None))
        return torch.zeros(state.shape[0], n_steps, STATE_DIM)

    monkeypatch.setattr(cohort_loss, "integrate", fake_integrate)
    model = _tiny_model()
    arm = CohortArmSpec(label="a", duration_min=10, start_hour=8.0, meals=())
    state = torch.tensor(NORM_CENTER).unsqueeze(0)
    rng = np.random.default_rng(0)
    for _ in range(40):
        cohort_loss._rollout_arm_states(model, torch.zeros(1, EMBEDDING_DIM), arm, state,
                                        input_dropout=0.5, rng=rng)
    dropped_sw = sum(a for a, _ in seen)
    dropped_act = sum(b for _, b in seen)
    assert 8 < dropped_sw < 32 and 8 < dropped_act < 32
    seen.clear()
    cohort_loss._rollout_arm_states(model, torch.zeros(1, EMBEDDING_DIM), arm, state)
    assert seen == [(False, False)]


# --- 1.5: the recipe is the configuration -----------------------------------------------

_LOAD_BEARING = {
    "--coupling-prior-weight=0.03", "--input-dropout=0.3", "--huber-delta=1.0",
    "--carb-mass-balance-sample-patients=2", "--grad-clip=10.0", "--seed=42",
}


def test_spec_pins_every_load_bearing_flag_and_parses_without_divergence() -> None:
    spec_path = os.path.join(REPO, "train", "spec.json")
    spec_args = _load_spec_train_args(spec_path)
    missing = _LOAD_BEARING - set(spec_args)
    assert not missing, f"train/spec.json must set explicitly: {sorted(missing)}"
    parser = build_arg_parser()
    spec_only = parser.parse_args(spec_args + ["--spec", spec_path])
    same = parser.parse_args(spec_args + ["--spec", spec_path, "--gcs-bucket", "b", "--gcs-object", "o"])
    assert spec_config_divergence(spec_only, same) == []
    forked = parser.parse_args(spec_args + ["--spec", spec_path, "--lr", "0.1"])
    div = spec_config_divergence(spec_only, forked)
    assert [d[0] for d in div] == ["lr"]
    cfg = resolved_train_config(spec_only)
    assert "gcs_bucket" not in cfg and cfg["grad_clip"] == 10.0 and cfg["seed"] == 42
    json.dumps(cfg, default=str)  # must be recordable in the checkpoint
    # The iter-97 recipe's own numbers (review 4.1 / 4.9 / 4.12).
    assert spec_only.aux_every_k_windows > 0 and spec_only.phase2_lr_floor >= 1e-3
    assert spec_only.phase3_epochs > 0 and spec_only.phase3_input_dropout > 0.3
    assert "insulin" not in spec_only.cold_distill_markers.split(":")
    assert "glucagon" not in spec_only.cold_distill_markers.split(":")
    for marker in ("crh", "insulin_action", "fat_mass", "insulin_slow"):
        assert marker in spec_only.cold_distill_markers.split(":")
    assert spec_only.cold_distill_anchor_long_only.split(":") == [
        "mitochondrial_capacity", "fat_mass",
    ]
    assert spec_only.perturb_protocols and spec_only.trajectory_band_per_marker
    assert not any(
        any(s in a for s in ("landmark", "fasting-stability", "default-baseline", "postprandial-recovery"))
        for a in spec_args
    )


# --- student hand-off: the CVS setpoint frame ------------------------------------------

def test_setpoint_supervision_decodes_cvs_through_the_module() -> None:
    from pulse.training import SetpointSupervisionSignal
    torch.manual_seed(0)
    model = _tiny_model()
    emb = nn.Embedding(2, EMBEDDING_DIM)
    nn.init.normal_(emb.weight, std=0.3)
    params = list(model.parameters()) + list(emb.parameters())
    ctx = _ctx(params)
    targets = {0: {"glucose": 100.0, "hr": 60.0, "hrv": 50.0, "sbp": 125.0, "dbp": 80.0, "temp": 36.9},
               1: {"glucose": 90.0, "hr": 75.0, "hrv": 30.0, "sbp": 115.0, "dbp": 75.0, "temp": 37.1}}
    sig = SetpointSupervisionSignal(weight=WeightSchedule(0.5), targets=targets)
    res = sig.compute(model, emb, ctx)
    assert res.n_units == 1 and np.isfinite(res.loss_sum)
    e_cvs = model.embedding_projections["cardiovascular"](emb.weight)
    z = model.cardiovascular.setpoints_z(e_cvs)
    assert z.shape == (2, 4)
    raw = model.cardiovascular.setpoints_raw(e_cvs)
    assert bool((raw[:, 2] > raw[:, 3]).all())  # SBP > DBP by construction
    assert "sbp_mae" in res.sub_metrics and "hrv_mae" in res.sub_metrics


def test_setpoint_supervision_runs_on_the_current_head() -> None:
    from pulse.training import SetpointSupervisionSignal
    torch.manual_seed(0)
    model = _tiny_model()
    emb = nn.Embedding(1, EMBEDDING_DIM)
    params = list(model.parameters()) + list(emb.parameters())
    ctx = _ctx(params)
    sig = SetpointSupervisionSignal(weight=WeightSchedule(0.5), targets={0: {"glucose": 100.0, "hr": 60.0}})
    res = sig.compute(model, emb, ctx)
    assert res.n_units == 1 and np.isfinite(res.loss_sum) and "hr_mae" in res.sub_metrics


def test_checkpoint_records_model_config() -> None:
    import inspect
    from pulse import train as train_mod
    assert '"model_config": getattr(model, "constructor_kwargs", None)' in inspect.getsource(train_mod.train)
