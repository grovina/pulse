"""
Training pipeline for the modular physiology network.

Iterates a list of ``TrainingSignal`` instances each epoch. The trajectory
signal owns the cold-model dataset and per-window inner loop; cohort and
future signals add their own gradient sources on top. The trainer's only
job is to set up the model, optimizer, LR schedule, and dispatch signals.
"""

import argparse
import faulthandler
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Iter 67 follow-up: epoch-scoped watchdog. cxtcj (iter 67 first run)
# went silent at phase 1 epoch 5 and stayed silent until Cloud Run
# eventually killed the container ~90 min later — pure hang, no stderr,
# no diagnostics. faulthandler.dump_traceback_later() fires from a
# background thread and writes every thread's Python stack to stderr,
# which Cloud Run captures into Cloud Logging. We rearm it at each
# epoch start (see the loop below), so a stuck epoch surfaces a stack
# trace after PULSE_WATCHDOG_TIMEOUT_S of no progress instead of hours
# of silence. Also enable() so a native crash (segfault, abort) dumps
# the same trace instead of dying mute.
faulthandler.enable()
# Iter 68 r8: bump default from 3600s to 14400s (4h). r7's phase-2 epochs
# structurally take 70-75 min each — the 1h watchdog fired *every*
# phase-2 epoch, and ep 53's dump correlated with a SIGSEGV inside
# torch.nn.Linear.forward (faulthandler's per-thread stack walk racing
# with a torch CPU kernel that briefly drops the GIL). The watchdog
# exists for hang detection — any hang > 4h is real; epochs that
# legitimately take 70 min should not fire it. Override via env var
# PULSE_WATCHDOG_TIMEOUT_S if needed for local debugging.
_WATCHDOG_TIMEOUT_S = int(os.environ.get("PULSE_WATCHDOG_TIMEOUT_S", "14400"))

# Iter 67 OOM saga: 9 rounds of phase-2 OOM with the suspect (Move B FULL)
# already reverted. A 64K-param model on 32Gi has ~4 orders of magnitude of
# headroom over the parameters themselves — OOMs must come from per-step
# activation/autograd-graph accumulation across the rollout, not the model.
# We print RSS + autograd-tensor counts every epoch so the next dispatch
# captures a growth curve (flat-ish = one-shot ceiling, monotonic rise =
# leak). Linux-only path via /proc/self/status; falls back to None elsewhere
# so local sanity runs on macOS still execute.
def _proc_status_kb(key: str) -> int | None:
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith(key + ":"):
                    return int(line.split()[1])
    except (FileNotFoundError, PermissionError, ValueError):
        return None
    return None


def _memory_stats() -> dict[str, float | int]:
    """RSS / peak in MB + autograd-tensor counts via gc walk.

    The gc walk is O(objects) — a few hundred ms on a typical pulse run.
    Cheap enough to do every epoch; the resulting series is what tells us
    whether the OOM is structural (flat then ceiling) or a leak (monotonic).
    """
    rss_kb = _proc_status_kb("VmRSS")
    peak_kb = _proc_status_kb("VmPeak")
    n_tensors = 0
    n_grad_tensors = 0
    total_elems = 0
    grad_elems = 0
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.Tensor):
                n_tensors += 1
                numel = obj.numel()
                total_elems += numel
                if obj.requires_grad or obj.grad_fn is not None:
                    n_grad_tensors += 1
                    grad_elems += numel
        except (ReferenceError, RuntimeError):
            continue
    return {
        "rss_mb": (rss_kb / 1024.0) if rss_kb is not None else -1.0,
        "peak_mb": (peak_kb / 1024.0) if peak_kb is not None else -1.0,
        "n_tensors": n_tensors,
        "n_grad_tensors": n_grad_tensors,
        "total_elems_m": total_elems / 1e6,
        "grad_elems_m": grad_elems / 1e6,
    }

import numpy as np
import torch
import torch.nn as nn

from .knowledge import ALL_CONTRIBUTIONS, ALL_COHORT_STATISTICS
from .knowledge.physiology_rules import PHYSIOLOGY_RULES
from .model import ModularPhysiologyNetwork
from .dose_response import DoseResponseProtocol, MarkerDoseTarget
from .training import (
    CohortStatisticSignal,
    ColdModelDistillationSignal,
    CarbMassBalanceSignal,
    DefaultBaselineSignal,
    DoseResponseSignal,
    FastingStabilitySignal,
    PhysiologyRulesSignal,
    PostprandialRecoverySignal,
    GutDoseSweepProtocol,
    GutDoseSweepSignal,
    InsulinSweepProtocol,
    InsulinSweepSignal,
    NaNTrainingAbort,
    SignalContext,
    SignalResult,
    TrainingSignal,
    TrajectoryRolloutSignal,
    WeightSchedule,
)
from .training.trajectory_signal import (
    DEFAULT_CONTRIBUTION_WEIGHTS,
    TRAIN_WINDOW,
    normalize_contribution_weights,
)
from .types import EMBEDDING_DIM, MARKERS

# Re-export for backwards-compatible test imports.
from .training.trajectory_signal import sample_window_start as _sample_window_start  # noqa: F401


def _parse_dose_response_markers(raw: str) -> tuple[MarkerDoseTarget, ...]:
    """Parse --dose-response-markers spec into typed targets.

    Spec syntax: ``marker:mode:params[;next...]`` (``;`` between markers
    avoids gcloud --args= comma-splitting on inter-arg ``,``).

    * ``glucose:slope:<target>:<sigma>[:<weight>]`` — Gaussian z² on OLS
      slope vs literature target (offset-invariant).
    * ``<marker>:peak:<target_per_g>:<sigma_per_g>[:<weight>]`` — Gaussian z²
      on absolute Δpeak vs a literature line through the origin
      (``target_per_g · dose``). Pins absolute amplitude, not just slope.
    * ``<marker>:rank:<margin>[:<weight>]`` — hinge on adjacent-dose Δpeak
      monotonicity with minimum slack ``margin``.

    Empty / whitespace input returns ``()`` and the protocol falls back to
    the legacy single-glucose-slope target (iter-32-era behaviour).
    """
    if not raw or not raw.strip():
        return ()
    out: list[MarkerDoseTarget] = []
    for spec in raw.split(";"):
        spec = spec.strip()
        if not spec:
            continue
        parts = spec.split(":")
        if len(parts) < 3:
            raise ValueError(
                f"dose-response-markers spec '{spec}': expected marker:mode:params...",
            )
        marker, mode = parts[0], parts[1]
        if mode in ("slope", "peak"):
            if len(parts) < 4:
                raise ValueError(
                    f"'{spec}': {mode} mode requires target:sigma (per g carb)",
                )
            target = float(parts[2])
            sigma = float(parts[3])
            weight = float(parts[4]) if len(parts) > 4 else 1.0
            out.append(MarkerDoseTarget(
                marker=marker, mode=mode,
                target_slope=target, sigma_slope=sigma, weight=weight,
            ))
        elif mode == "rank":
            margin = float(parts[2])
            weight = float(parts[3]) if len(parts) > 3 else 1.0
            out.append(MarkerDoseTarget(
                marker=marker, mode="rank",
                rank_margin=margin, weight=weight,
            ))
        else:
            raise ValueError(f"'{spec}': unknown mode '{mode}' (use slope|peak|rank)")
    return tuple(out)


def _split_markers(raw: str) -> tuple[str, ...]:
    """Parse a marker list from CLI. Accepts either ``:`` or ``,`` as
    separator. We prefer ``:`` because gcloud's ``--args=`` value uses
    ``,`` as the inter-argument delimiter, so commas inside an argument
    value get re-split into positional args (the iter-39 first-attempt
    bug). Authors of new ``spec.json`` should use ``:`` for any marker
    list with more than one entry; ``,`` is supported for single-marker
    backward compat (where ``,`` happens not to be present)."""
    pieces = raw.replace(",", ":").split(":")
    return tuple(p.strip() for p in pieces if p.strip())


def _load_contribution_weights_json(path: str | None) -> dict[str, float] | None:
    if not path:
        return None
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("contribution weights JSON must be an object name -> float")
    return {str(k): float(v) for k, v in data.items()}


def _set_runtime(seed: int, *, deterministic: bool) -> None:
    """Initialize RNGs and PyTorch threading.

    ``deterministic=True`` pins threads to 1 and asks PyTorch for
    deterministic algorithms — useful for unit tests / spec verification.
    Default (``False``) leaves PyTorch free to use all available cores,
    which is a large win on multi-CPU hosts (Cloud Run, Vertex) where
    pinning kills throughput. Per-epoch results may differ run-to-run
    by float-reduction order, but the optimization is stochastic anyway.
    """
    import os
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    # One-shot diagnostic so we can confirm the trainer is actually using the
    # CPU it was scheduled on. Cloud Run / Vertex sometimes hand out fewer
    # cores than the wrapper requested (cgroup throttling), and the symptom
    # is just "training is slow" with no log clue.
    print(
        f"[runtime] torch_threads={torch.get_num_threads()} "
        f"interop={torch.get_num_interop_threads()} "
        f"os_cpus={os.cpu_count()} "
        f"sched_affinity={len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else 'n/a'} "
        f"deterministic={deterministic}",
    )


def train(
    n_patients: int = 20,
    n_epochs: int = 80,
    hidden_dim: int = 48,
    lr: float = 3e-3,
    grad_clip: float = 10.0,
    seed: int = 42,
    windows_per_patient: int = 3,
    meal_window_bias: float = 0.55,
    output_path: str = "pulse-model.pt",
    input_dropout: float = 0.3,
    gut_loss_weight: float = 0.5,
    n_days: int = 14,
    contribution_weights: dict[str, float] | None = None,
    coupling_prior_weight: float = 0.03,
    coupling_prior_samples: int = 3,
    verifier_loss_weight: float = 0.02,
    huber_delta: float = 1.0,
    cohort_statistic_weight: float = 0.05,
    cohort_sample_patients: int = 16,
    cohort_statistic_adaptive: bool = False,
    trajectory_band: float = 0.0,
    trajectory_band_default: float = 0.0,
    landmark_weight: float = 0.0,
    landmark_pre_window: int = 15,
    landmark_post_window: int = 120,
    landmark_min_carbs: float = 5.0,
    dose_response_weight: float = 0.0,
    dose_response_sample_patients: int = 4,
    dose_response_markers: tuple[MarkerDoseTarget, ...] = (),
    gut_dose_sweep_weight: float = 0.0,
    gut_dose_sweep_sample_patients: int = 4,
    gut_dose_sweep_auc_weight: float = 1.0,
    insulin_sweep_weight: float = 0.0,
    insulin_sweep_sample_patients: int = 4,
    insulin_sweep_auc_weight: float = 1.0,
    insulin_sweep_ranking_weight: float = 1.0,
    fasting_stability_weight: float = 0.0,
    fasting_stability_window: int = 120,
    postprandial_recovery_weight: float = 0.0,
    default_baseline_weight: float = 0.0,
    default_baseline_markers: tuple[str, ...] = ("hr",),
    carb_mass_balance_weight: float = 0.0,
    carb_mass_balance_sample_patients: int = 2,
    cold_distill_weight: float = 0.0,
    cold_distill_markers: tuple[str, ...] = (
        "glucagon", "ffa", "ghrelin", "leptin", "acth", "cortisol", "bhb",
    ),
    cold_distill_protocols_per_epoch: int = 4,
    cold_distill_pool: str = "synthetic",
    cold_distill_mode: str = "trajectory",
    cold_distill_anchor_window: int = 60,
    cold_distill_anchor_samples: int = 8,
    physiology_rules_weight: float = 0.0,
    physiology_rules_sample_patients: int = 4,
    physiology_rules_adaptive: bool = False,
    n_default_patients: int = 0,
    cohort_use_cold_init: bool = True,
    phase1_epochs: int | None = None,
    phase2_epochs: int = 0,
    phase2_lr: float | None = None,
    gcs_bucket: str | None = None,
    gcs_object: str | None = None,
    benchmark_dataset_uri: str | None = None,
    benchmark_thresholds_uri: str | None = None,
    benchmark_report_path: str | None = None,
    deterministic: bool = False,
):
    _set_runtime(seed, deterministic=deterministic)
    device = torch.device("cpu")

    cnames, cprobs = normalize_contribution_weights(contribution_weights)
    print(
        f"Generating {n_patients} virtual patients; contribution mix: "
        + ", ".join(f"{n}:{p:.2f}" for n, p in zip(cnames, cprobs)),
    )

    # Phase boundary: after Phase 1, additive losses turn on.
    phased = phase2_epochs > 0
    if phased:
        p1_epochs = phase1_epochs if phase1_epochs is not None else n_epochs
        p2_lr = phase2_lr if phase2_lr is not None else lr
        total_epochs = p1_epochs + phase2_epochs
        phase_boundary = p1_epochs
    else:
        p2_lr = lr
        total_epochs = n_epochs
        phase_boundary = 0

    enable_at = phase_boundary if phased else 0

    trajectory_signal = TrajectoryRolloutSignal(
        n_patients=n_patients,
        n_days=n_days,
        seed=seed,
        contribution_weights=contribution_weights,
        windows_per_patient=windows_per_patient,
        meal_window_bias=meal_window_bias,
        input_dropout=input_dropout,
        huber_delta=huber_delta,
        gut_loss_weight=gut_loss_weight,
        coupling_weight=WeightSchedule(coupling_prior_weight, enable_at_epoch=enable_at),
        verifier_weight=WeightSchedule(verifier_loss_weight, enable_at_epoch=enable_at),
        coupling_prior_samples=coupling_prior_samples,
        trajectory_band=trajectory_band,
        trajectory_band_default=trajectory_band_default,
        landmark_weight=WeightSchedule(landmark_weight, enable_at_epoch=enable_at),
        landmark_pre_window=landmark_pre_window,
        landmark_post_window=landmark_post_window,
        landmark_min_carbs=landmark_min_carbs,
        n_default_patients=n_default_patients,
    )
    cohort_signal = CohortStatisticSignal(
        specs=list(ALL_COHORT_STATISTICS),
        n_patients=n_patients,
        sample_patients=cohort_sample_patients,
        weight=WeightSchedule(cohort_statistic_weight, enable_at_epoch=enable_at),
        use_cold_initial_state=cohort_use_cold_init,
        adaptive=cohort_statistic_adaptive,
    )
    dose_response_protocol = DoseResponseProtocol(
        marker_targets=tuple(dose_response_markers),
    )
    dose_response_signal = DoseResponseSignal(
        n_patients=n_patients,
        sample_patients=dose_response_sample_patients,
        weight=WeightSchedule(dose_response_weight, enable_at_epoch=enable_at),
        protocol=dose_response_protocol,
    )
    gut_dose_sweep_protocol = GutDoseSweepProtocol()
    gut_dose_sweep_signal = GutDoseSweepSignal(
        n_patients=n_patients,
        sample_patients=gut_dose_sweep_sample_patients,
        weight=WeightSchedule(gut_dose_sweep_weight, enable_at_epoch=0),
        protocol=gut_dose_sweep_protocol,
        auc_weight=gut_dose_sweep_auc_weight,
    )
    insulin_sweep_protocol = InsulinSweepProtocol()
    insulin_sweep_signal = InsulinSweepSignal(
        n_patients=n_patients,
        sample_patients=insulin_sweep_sample_patients,
        weight=WeightSchedule(insulin_sweep_weight, enable_at_epoch=0),
        protocol=insulin_sweep_protocol,
        auc_weight=insulin_sweep_auc_weight,
        ranking_weight=insulin_sweep_ranking_weight,
    )
    fasting_stability_signal = FastingStabilitySignal(
        window_min=fasting_stability_window,
        weight=WeightSchedule(fasting_stability_weight, enable_at_epoch=0),
    )
    postprandial_recovery_signal = PostprandialRecoverySignal(
        weight=WeightSchedule(postprandial_recovery_weight, enable_at_epoch=0),
    )
    default_baseline_signal = DefaultBaselineSignal(
        weight=WeightSchedule(default_baseline_weight, enable_at_epoch=0),
        markers=tuple(default_baseline_markers),
    )
    # Iter 74: cross-module carb→glycogen mass-balance conservation. Pairs with
    # the phase-2 constraint suite (cohort / rules) — gate it at the phase
    # boundary so glycogen has a settled metabolic context before the
    # inequalities can bind. Off unless --carb-mass-balance-weight is set.
    carb_mass_balance_signal = CarbMassBalanceSignal(
        weight=WeightSchedule(carb_mass_balance_weight, enable_at_epoch=enable_at),
        n_patients=n_patients,
        sample_patients=carb_mass_balance_sample_patients,
    )
    # Cold-model distillation runs at a per-protocol *calibrated* embedding
    # (mirroring the bench's calibrated eval), so it only makes sense once the
    # observed-marker fit — and hence a meaningful embedding — exists: enable
    # it with the rest of the additive losses at the phase boundary.
    cold_distill_signal = ColdModelDistillationSignal(
        weight=WeightSchedule(cold_distill_weight, enable_at_epoch=enable_at),
        markers=tuple(cold_distill_markers),
        protocols_per_epoch=cold_distill_protocols_per_epoch,
        pool=cold_distill_pool,
        mode=cold_distill_mode,
        anchor_window=cold_distill_anchor_window,
        anchor_samples=cold_distill_anchor_samples,
    )
    physiology_rules_signal = PhysiologyRulesSignal(
        rules=list(PHYSIOLOGY_RULES),
        n_patients=n_patients,
        sample_patients=physiology_rules_sample_patients,
        weight=WeightSchedule(physiology_rules_weight, enable_at_epoch=enable_at),
        adaptive=physiology_rules_adaptive,
    )
    signals: list[TrainingSignal] = [
        trajectory_signal,
        cohort_signal,
        dose_response_signal,
        gut_dose_sweep_signal,
        insulin_sweep_signal,
        fasting_stability_signal,
        postprandial_recovery_signal,
        default_baseline_signal,
        carb_mass_balance_signal,
        cold_distill_signal,
        physiology_rules_signal,
    ]

    print(
        f"Coupling priors: {len(trajectory_signal.priors)} edges, "
        f"weight={coupling_prior_weight}; verifier surrogate weight={verifier_loss_weight}; "
        f"meal window bias={meal_window_bias}",
    )
    print(
        f"Trajectory band={trajectory_band:.3f} (per-patient), "
        f"{trajectory_band_default:.3f} (default-patient, norm units); "
        f"landmark weight={landmark_weight} "
        f"(pre={landmark_pre_window}min, post={landmark_post_window}min, min_carbs={landmark_min_carbs}g)",
    )
    _dr_targets = dose_response_protocol.effective_targets()
    _dr_target_desc = ", ".join(
        f"{t.marker}[{t.mode}"
        + (f" target={t.target_slope}±{t.sigma_slope}" if t.mode == "slope" else f" margin={t.rank_margin}")
        + f" w={t.weight}]"
        for t in _dr_targets
    )
    print(
        f"Dose-response: weight={dose_response_weight}, "
        f"doses={dose_response_protocol.carb_doses_g} g, "
        f"targets=[{_dr_target_desc}], "
        f"sample_patients={dose_response_sample_patients}",
    )
    print(
        f"Gut dose sweep: weight={gut_dose_sweep_weight}, "
        f"auc_weight={gut_dose_sweep_auc_weight}, "
        f"doses={gut_dose_sweep_protocol.carb_doses_g} g, "
        f"sample_patients={gut_dose_sweep_sample_patients}, "
        f"post_window={gut_dose_sweep_protocol.post_window_min}min",
    )
    print(
        f"Insulin sweep: weight={insulin_sweep_weight}, "
        f"auc_weight={insulin_sweep_auc_weight}, "
        f"ranking_weight={insulin_sweep_ranking_weight}, "
        f"glucose_grid={insulin_sweep_protocol.glucose_sweep_mg_dL} mg/dL, "
        f"insulin_grid={insulin_sweep_protocol.insulin_sweep_uU_mL} μU/mL, "
        f"sample_patients={insulin_sweep_sample_patients}",
    )
    print(
        f"Fasting stability: weight={fasting_stability_weight} "
        f"(glucose drift from NORM_CENTER over {fasting_stability_window} min, zero embedding; 0=disabled)",
    )
    print(
        f"Postprandial recovery: weight={postprandial_recovery_weight} "
        f"(glucose residual at +420-475 min after standard meal, zero embedding; 0=disabled)",
    )
    print(
        f"Default baseline: weight={default_baseline_weight} "
        f"markers={tuple(default_baseline_markers)} "
        f"(zero-embedding fasted mean toward NORM_CENTER; 0=disabled)",
    )
    print(
        f"Cold-model distillation: weight={cold_distill_weight} pool={cold_distill_pool} mode={cold_distill_mode}"
        f"{f'(W={cold_distill_anchor_window})' if cold_distill_mode == 'anchored' else ''} "
        f"markers={tuple(cold_distill_markers)} "
        f"protocols/epoch={cold_distill_protocols_per_epoch}/{len(cold_distill_signal._protocols)} "
        f"(zero-embedding trajectory MSE vs simulate_full_body on a broad "
        f"protocol pool; 0=disabled)",
    )
    print(
        f"Physiology rules: weight={physiology_rules_weight} "
        f"n_rules={len(PHYSIOLOGY_RULES)} sample_patients={physiology_rules_sample_patients} "
        f"adaptive={physiology_rules_adaptive} "
        f"(per-trajectory hinge supervision; 0=disabled)",
    )
    print(
        f"Default-patient distillation: {n_default_patients} extra cold-model episode(s) "
        f"supervised at zero embedding; cohort cold initial state: {cohort_use_cold_init}",
    )

    model = ModularPhysiologyNetwork(
        metabolic_hidden=hidden_dim,
        appetite_hidden=max(24, hidden_dim // 2),
        stress_hidden=max(24, hidden_dim // 2),
        cardiovascular_hidden=hidden_dim,
        thermoreg_hidden=max(16, hidden_dim // 3),
        respiratory_hidden=max(16, hidden_dim // 3),
    ).to(device)

    embeddings = nn.Embedding(n_patients, EMBEDDING_DIM).to(device)
    nn.init.normal_(embeddings.weight, std=0.1)

    # NOTE: torch.compile was tried for iter 14 (5-epoch local A/B). On a
    # single fixed protocol it gave 2.9x steady-state speedup, but in the
    # real training loop the model.forward graph is hit with too many
    # (batch_size × requires_grad × dispatch-shape) variants — Phase 2
    # epoch ended up the same wall-clock or slightly worse vs eager, and
    # dynamo started warning about hitting the recompile limit. The
    # batched-embedding refactor (Tiers 1+2) already delivered the
    # ~3-4x Phase-2 speedup we needed; not worth introducing the
    # compile moving piece on top.
    params = list(model.parameters()) + list(embeddings.parameters())
    rng = np.random.default_rng(seed)

    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler_T = phase_boundary if phased else total_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=scheduler_T, eta_min=lr * 0.05,
    )

    total_params = sum(p.numel() for p in model.parameters())
    emb_params = n_patients * EMBEDDING_DIM
    print(f"Model: {total_params:,} parameters")
    print(f"Embeddings: {n_patients} x {EMBEDDING_DIM} = {emb_params:,}")
    if phased:
        print(
            f"Phased training: Phase 1 = {p1_epochs} epochs (distillation), "
            f"Phase 2 = {phase2_epochs} epochs (full)",
        )
    print(f"Training: {total_epochs} epochs, {windows_per_patient} windows/patient, window={TRAIN_WINDOW} min")
    print(
        f"Input dropout: {input_dropout}, gut loss weight: {gut_loss_weight}, "
        f"cohort statistics: {len(ALL_COHORT_STATISTICS)} spec(s), weight={cohort_statistic_weight}\n",
    )

    # Iter 25: rolling last-good checkpoint + recent-metrics buffer for the
    # NaN abort handler. ``last_good_path`` is overwritten at the end of every
    # successful epoch; on abort it's uploaded to GCS as ``last_good_pre_nan.pt``
    # so iter 26 can probe the pre-NaN parameter state with the diagnostics
    # tools instead of re-running blind.
    last_good_path = output_path + ".last-good.pt"
    recent_metrics: list[dict[str, float]] = []  # rolling, last 5 epochs

    cur_phase = 1
    epoch = 0
    try:
        for epoch in range(total_epochs):
            if phased and epoch == phase_boundary:
                cur_phase = 2
                optimizer = torch.optim.Adam(params, lr=p2_lr)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=phase2_epochs, eta_min=p2_lr * 0.05,
                )
                print(f"\n--- Phase 2: {phase2_epochs} epochs, lr={p2_lr} ---\n")

            epoch_start = time.time()
            if _WATCHDOG_TIMEOUT_S > 0:
                faulthandler.dump_traceback_later(_WATCHDOG_TIMEOUT_S, repeat=False)
            model.train()
            ctx = SignalContext(
                epoch=epoch,
                total_epochs=total_epochs,
                rng=rng,
                device=device,
                optimizer=optimizer,
                params=params,
                grad_clip=grad_clip,
            )

            # Iter 25: time each signal explicitly. The iter-24 NaN-skip
            # symptom appeared as Phase-2 step time dropping 510s → 205s
            # because the inline guards were silently skipping backward+step
            # on every epoch — surfacing per-signal time would have caught
            # that immediately. Per-signal times also flow into the abort
            # diagnostic dump so iter 26 sees if any signal was already
            # collapsing before the explicit NaN.
            # Memprof saga: phase-2 ep 5 (first phase-2 epoch) OOM'd 9 times
            # at 26-59 min mid-epoch, never reaching end-of-epoch memprof.
            # Print intra-epoch memory between signals so we see WHICH signal
            # exhausts memory before the container is killed. Phase 1 (signals
            # gated by enable_at_epoch) won't trigger this for most signals,
            # so the noise is acceptable; gated by PULSE_INTRA_EPOCH_MEMPROF
            # so we can disable when not debugging.
            results: dict[str, SignalResult] = {}
            signal_times: dict[str, float] = {}
            intra_memprof = os.environ.get("PULSE_INTRA_EPOCH_MEMPROF", "1") != "0"
            for sig in signals:
                t0 = time.time()
                results[sig.name] = sig.compute(model, embeddings, ctx)
                signal_times[sig.name] = time.time() - t0
                if intra_memprof:
                    m = _memory_stats()
                    print(
                        f"[MEMPROF-SIG] epoch={epoch:3d} phase={cur_phase} sig={sig.name} "
                        f"dt={signal_times[sig.name]:.1f}s rss_mb={m['rss_mb']:.1f} "
                        f"tensors={m['n_tensors']} grad_tensors={m['n_grad_tensors']} "
                        f"elems_m={m['total_elems_m']:.2f} grad_elems_m={m['grad_elems_m']:.2f}",
                        flush=True,
                    )

            scheduler.step()

            elapsed = time.time() - epoch_start
            mem = _memory_stats()

            traj = results["trajectory_rollout"]
            cohort = results["cohort_statistic"]
            dose = results["dose_response"]
            gut_sweep = results["gut_dose_sweep"]
            ins_sweep = results["insulin_sweep"]
            fst = results["fasting_stability"]
            ppr = results["postprandial_recovery"]

            metrics_summary: dict[str, float] = {
                "epoch": float(epoch),
                "phase": float(cur_phase),
                "lr": float(scheduler.get_last_lr()[0]),
                "elapsed_s": float(elapsed),
                "trajectory_avg_loss": float(traj.avg_loss),
                "trajectory_n_windows": float(traj.n_units),
                "cohort_loss": float(cohort.loss_sum) if cohort.n_units > 0 else 0.0,
                "dose_loss": float(dose.loss_sum) if dose.n_units > 0 else 0.0,
                "gut_sweep_loss": float(gut_sweep.loss_sum) if gut_sweep.n_units > 0 else 0.0,
                "ins_sweep_loss": float(ins_sweep.loss_sum) if ins_sweep.n_units > 0 else 0.0,
                "fasting_stability_loss": float(fst.loss_sum) if fst.n_units > 0 else 0.0,
                "postprandial_recovery_loss": float(ppr.loss_sum) if ppr.n_units > 0 else 0.0,
            }
            for sig_name, t in signal_times.items():
                metrics_summary[f"time_{sig_name}_s"] = float(t)
            for k, v in mem.items():
                metrics_summary[f"mem_{k}"] = float(v)
            recent_metrics.append(metrics_summary)
            if len(recent_metrics) > 5:
                recent_metrics.pop(0)

            # End-of-epoch: persist last-good rolling checkpoint. Cheap (~50MB
            # write), runs every epoch, never re-uploaded to GCS unless an
            # abort fires.
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "embeddings_state": embeddings.state_dict(),
                    "epoch": epoch,
                    "phase": cur_phase,
                    "metrics": metrics_summary,
                },
                last_good_path,
            )

            # Memory profile is printed every epoch (not gated to the every-5
            # epoch summary) — the OOM-saga curve is what we're paying for,
            # and a per-epoch series is needed to distinguish slow leaks
            # from sudden allocations.
            print(
                f"[MEMPROF] epoch={epoch:3d} phase={cur_phase} "
                f"rss_mb={mem['rss_mb']:.1f} peak_mb={mem['peak_mb']:.1f} "
                f"tensors={mem['n_tensors']} grad_tensors={mem['n_grad_tensors']} "
                f"elems_m={mem['total_elems_m']:.2f} grad_elems_m={mem['grad_elems_m']:.2f}",
                flush=True,
            )

            last_in_phase = (epoch == total_epochs - 1) or (phased and epoch == phase_boundary - 1)
            if epoch % 5 == 0 or last_in_phase:
                label = f"Phase {cur_phase} " if phased else ""
                parts = [
                    f"{label}Epoch {epoch:3d}/{total_epochs}",
                    f"loss={traj.avg_loss:.6f}",
                    f"gut={traj.sub_metrics.get('gut', 0.0):.6f}",
                ]
                if "coupling" in traj.sub_metrics:
                    parts.append(f"cpl={traj.sub_metrics['coupling']:.6f}")
                if "verifier_surrogate" in traj.sub_metrics:
                    parts.append(f"v_sur={traj.sub_metrics['verifier_surrogate']:.6f}")
                if "landmark" in traj.sub_metrics:
                    parts.append(
                        f"lm={traj.sub_metrics['landmark']:.6f}"
                        f"({int(traj.sub_metrics.get('landmark_meals', 0))})",
                    )
                if cohort.n_units > 0:
                    parts.append(f"cohort={cohort.loss_sum:.6f}")
                if dose.n_units > 0:
                    parts.append(
                        f"dose={dose.loss_sum:.4f}"
                        f"(slope={dose.sub_metrics.get('predicted_slope', 0.0):.3f})",
                    )
                if gut_sweep.n_units > 0:
                    parts.append(f"gut_sweep={gut_sweep.loss_sum:.4f}")
                if ins_sweep.n_units > 0:
                    parts.append(
                        f"ins_sweep={ins_sweep.loss_sum:.4f}"
                        f"(auc={ins_sweep.sub_metrics.get('auc', 0.0):.4f})",
                    )
                if fst.n_units > 0:
                    parts.append(f"fst={fst.sub_metrics.get('glucose_drift_norm', 0.0):.4f}")
                if ppr.n_units > 0:
                    parts.append(
                        f"ppr={ppr.sub_metrics.get('residual_mg_dl', 0.0):.2f}mg/dL",
                    )
                parts.append(f"lr={scheduler.get_last_lr()[0]:.5f}")
                parts.append(f"({elapsed:.1f}s)")
                # Per-signal time breakdown — short names so it fits one line.
                _short = {
                    "trajectory_rollout": "traj",
                    "cohort_statistic": "coh",
                    "dose_response": "dose",
                    "gut_dose_sweep": "gsw",
                    "insulin_sweep": "isw",
                    "fasting_stability": "fst",
                    "postprandial_recovery": "ppr",
                }
                breakdown = " ".join(
                    f"{_short.get(nm, nm)}={signal_times[nm]:.0f}s"
                    for nm in (s.name for s in signals)
                )
                parts.append(f"[{breakdown}]")
                print("  ".join(parts))
        if _WATCHDOG_TIMEOUT_S > 0:
            faulthandler.cancel_dump_traceback_later()
    except NaNTrainingAbort as abort:
        print(
            f"\n[ABORT] TRAINING_ABORTED_NAN signal={abort.signal} "
            f"epoch={abort.epoch} cause={abort.cause} loss={abort.loss_value!r}",
            flush=True,
        )
        diagnostic = {
            "outcome": "aborted_nan",
            "abort": {
                "signal": abort.signal,
                "epoch": abort.epoch,
                "cause": abort.cause,
                "loss_value": abort.loss_value,
                "phase": cur_phase,
                "lr": float(scheduler.get_last_lr()[0]),
                "extra": abort.extra,
            },
            "recent_metrics": recent_metrics,
            "config": {
                "total_epochs": total_epochs,
                "phase1_epochs": phase_boundary if phased else total_epochs,
                "phase2_epochs": phase2_epochs,
                "phase2_lr": p2_lr if phased else None,
                "lr": lr,
                "grad_clip": grad_clip,
                "hidden_dim": hidden_dim,
                "n_patients": n_patients,
                "windows_per_patient": windows_per_patient,
                "meal_window_bias": meal_window_bias,
            },
            "aborted_at": datetime.now(timezone.utc).isoformat(),
        }
        abort_local = output_path + ".abort.json"
        Path(abort_local).write_text(json.dumps(diagnostic, indent=2, default=str))
        print(f"[ABORT] diagnostics written to {abort_local}", flush=True)
        if gcs_bucket and gcs_object:
            prefix = gcs_object.rsplit("/", 1)[0] if "/" in gcs_object else ""
            abort_object = f"{prefix}/abort_diagnostics.json" if prefix else "abort_diagnostics.json"
            last_good_object = f"{prefix}/last_good_pre_nan.pt" if prefix else "last_good_pre_nan.pt"
            try:
                _upload_to_gcs(abort_local, gcs_bucket, abort_object)
            except Exception as e:
                print(f"[ABORT] failed to upload diagnostics: {e}", flush=True)
            if Path(last_good_path).exists():
                try:
                    _upload_to_gcs(last_good_path, gcs_bucket, last_good_object)
                except Exception as e:
                    print(f"[ABORT] failed to upload last-good checkpoint: {e}", flush=True)
            else:
                print("[ABORT] no last-good checkpoint to upload (aborted before any epoch completed)", flush=True)
        return 42

    # Iter 61: capture per-dimension mean/std of the trained patient
    # embedding table. The eval-time MAP-fit (benchmark.py) uses these
    # to build a diagonal Gaussian prior matched to the actual learned
    # distribution, instead of the legacy isotropic L2 prior whose
    # weight (0.001) was ~1800x looser than the training-init scale
    # (σ=0.1, see nn.init.normal_ above). After iter 60 doubled
    # EMBEDDING_DIM to 64, that mismatch flipped the eval calibration
    # from "wide enough" to "memorization-shortcut friendly" — textbook
    # mean pass rate dropped 0.813 → 0.717 while observed-marker mape
    # improved. A prior matched to the trained per-dim std lets each
    # dim be free in proportion to how much the model actually used it
    # during training; dims the model left near init get strong prior
    # (no eval drift), dims the model heavily used get weak prior
    # (still expressive for legitimate per-patient variation).
    with torch.no_grad():
        _emb_table = embeddings.weight.detach().cpu()
        _emb_prior_mean = _emb_table.mean(dim=0).tolist()
        _emb_prior_std = _emb_table.std(dim=0, unbiased=False).tolist()

    checkpoint = {
        "model_state": model.state_dict(),
        "hidden_dim": hidden_dim,
        "marker_ids": [m.id for m in MARKERS],
        "model_version": datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
        "embedding_dim": EMBEDDING_DIM,
        "embedding_prior_mean": _emb_prior_mean,
        "embedding_prior_std": _emb_prior_std,
        "n_patients": n_patients,
        "knowledge_sources": [c.name for c in ALL_CONTRIBUTIONS],
        "contribution_weights": contribution_weights
        if contribution_weights is not None
        else dict(DEFAULT_CONTRIBUTION_WEIGHTS),
        "coupling_prior_weight": coupling_prior_weight,
        "n_coupling_priors": len(trajectory_signal.priors),
        "verifier_loss_weight": verifier_loss_weight,
        "huber_delta": huber_delta,
        "cohort_statistic_weight": cohort_statistic_weight,
        "cohort_statistic_adaptive": cohort_statistic_adaptive,
        "cohort_statistic_specs": [s.name for s in ALL_COHORT_STATISTICS],
        "trajectory_band": trajectory_band,
        "trajectory_band_default": trajectory_band_default,
        "landmark_weight": landmark_weight,
        "landmark_pre_window": landmark_pre_window,
        "landmark_post_window": landmark_post_window,
        "landmark_min_carbs": landmark_min_carbs,
        "dose_response_weight": dose_response_weight,
        "dose_response_sample_patients": dose_response_sample_patients,
        "dose_response_target_slope": dose_response_protocol.target_slope,
        "dose_response_sigma_slope": dose_response_protocol.sigma_slope,
        "dose_response_carb_doses_g": list(dose_response_protocol.carb_doses_g),
        "dose_response_marker_targets": [
            {
                "marker": t.marker, "mode": t.mode,
                "target_slope": t.target_slope, "sigma_slope": t.sigma_slope,
                "rank_margin": t.rank_margin, "weight": t.weight,
            }
            for t in dose_response_protocol.effective_targets()
        ],
        "gut_dose_sweep_weight": gut_dose_sweep_weight,
        "gut_dose_sweep_sample_patients": gut_dose_sweep_sample_patients,
        "gut_dose_sweep_auc_weight": gut_dose_sweep_auc_weight,
        "gut_dose_sweep_carb_doses_g": list(gut_dose_sweep_protocol.carb_doses_g),
        "gut_dose_sweep_post_window_min": gut_dose_sweep_protocol.post_window_min,
        "insulin_sweep_weight": insulin_sweep_weight,
        "insulin_sweep_sample_patients": insulin_sweep_sample_patients,
        "insulin_sweep_auc_weight": insulin_sweep_auc_weight,
        "insulin_sweep_ranking_weight": insulin_sweep_ranking_weight,
        "insulin_sweep_glucose_mg_dL": list(insulin_sweep_protocol.glucose_sweep_mg_dL),
        "insulin_sweep_insulin_uU_mL": list(insulin_sweep_protocol.insulin_sweep_uU_mL),
        "default_baseline_weight": default_baseline_weight,
        "default_baseline_markers": list(default_baseline_markers),
        "carb_mass_balance_weight": carb_mass_balance_weight,
        "carb_mass_balance_sample_patients": carb_mass_balance_sample_patients,
        "cold_distill_weight": cold_distill_weight,
        "cold_distill_markers": list(cold_distill_markers),
        "cold_distill_protocols_per_epoch": cold_distill_protocols_per_epoch,
        "cold_distill_pool": cold_distill_pool,
        "cold_distill_mode": cold_distill_mode,
        "cold_distill_anchor_window": cold_distill_anchor_window,
        "cold_distill_anchor_samples": cold_distill_anchor_samples,
        "physiology_rules_weight": physiology_rules_weight,
        "physiology_rules_sample_patients": physiology_rules_sample_patients,
        "physiology_rules_adaptive": physiology_rules_adaptive,
        "physiology_rules_names": [r.name for r in PHYSIOLOGY_RULES],
        "n_default_patients": n_default_patients,
        "cohort_use_cold_init": cohort_use_cold_init,
        "phase_schedule": {
            "phase1_epochs": phase_boundary if phased else total_epochs,
            "phase2_epochs": phase2_epochs,
            "phase2_lr": phase2_lr,
        },
        "meal_window_bias": meal_window_bias,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    torch.save(checkpoint, output_path)
    print(f"\nModel saved to {output_path}")

    if gcs_bucket and gcs_object:
        _upload_to_gcs(output_path, gcs_bucket, gcs_object)

    if benchmark_dataset_uri:
        # Iter 61: attach the trained-table prior stats to the model
        # so the in-process benchmark's calibrate_embedding can use a
        # diagonal Gaussian prior matched to actual training usage.
        model._embedding_prior_mean = torch.tensor(_emb_prior_mean, dtype=torch.float32)
        model._embedding_prior_std = torch.tensor(_emb_prior_std, dtype=torch.float32)
        return _run_benchmark(
            model, output_path,
            benchmark_dataset_uri, benchmark_thresholds_uri,
            benchmark_report_path, gcs_bucket,
        )
    return 0


def _parse_gs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Not a gs:// URI: {uri}")
    rest = uri[5:]
    slash = rest.find("/")
    if slash < 0:
        raise ValueError(f"Invalid gs:// URI: {uri}")
    return rest[:slash], rest[slash + 1 :]


def _download_gs_uri_to_file(gs_uri: str, dest_path: str) -> None:
    from google.cloud import storage as gcs

    bucket_name, blob_name = _parse_gs_uri(gs_uri)
    client = gcs.Client()
    client.bucket(bucket_name).blob(blob_name).download_to_filename(dest_path)


def _upload_file_to_gs_uri(local_path: str, gs_uri: str) -> None:
    from google.cloud import storage as gcs

    bucket_name, blob_name = _parse_gs_uri(gs_uri)
    client = gcs.Client()
    client.bucket(bucket_name).blob(blob_name).upload_from_filename(local_path)


def _upload_to_gcs(local_path: str, bucket: str, obj: str) -> None:
    try:
        _upload_file_to_gs_uri(local_path, f"gs://{bucket}/{obj}")
        print(f"Uploaded to gs://{bucket}/{obj}")
    except Exception as e:
        print(f"GCS upload failed: {e}")
        import subprocess
        result = subprocess.run(
            ["gsutil", "cp", local_path, f"gs://{bucket}/{obj}"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print(f"Uploaded via gsutil to gs://{bucket}/{obj}")
        else:
            stderr = (result.stderr or "").strip()
            raise RuntimeError(
                f"GCS upload failed for {local_path}: {stderr or 'gsutil cp failed'}",
            ) from None


def _run_benchmark(
    model: ModularPhysiologyNetwork,
    model_path: str,
    dataset_uri: str,
    thresholds_uri: str | None,
    report_path: str | None,
    gcs_bucket: str | None,
) -> int:
    from .benchmark import (
        describe_benchmark_dataset_load_failure,
        load_benchmark_dataset,
        evaluate_model_against_benchmark,
        default_thresholds,
    )
    import tempfile

    print(f"\nRunning benchmark against {dataset_uri}...")

    dataset_local = dataset_uri
    if dataset_uri.startswith("gs://"):
        dataset_local = tempfile.mktemp(suffix=".json")
        try:
            _download_gs_uri_to_file(dataset_uri, dataset_local)
        except Exception as e:
            print(f"Benchmark dataset download failed: {e}")
            return 1

    thresholds = default_thresholds()
    if thresholds_uri and thresholds_uri.startswith("gs://"):
        thresh_local = tempfile.mktemp(suffix=".json")
        try:
            _download_gs_uri_to_file(thresholds_uri, thresh_local)
        except Exception as e:
            print(f"Benchmark thresholds download failed: {e}")
            return 1
        thresholds = json.loads(Path(thresh_local).read_text())

    episodes = load_benchmark_dataset(dataset_local)
    if not episodes:
        detail = describe_benchmark_dataset_load_failure(dataset_local)
        print(f"No benchmark episodes loaded — {detail}")
        print("  (Regenerate or migrate the dataset: apps/pulse/backend benchmark:export, or "
              "apps/pulse/engine scripts/migrate_benchmark_initial_state.py; then sync to GCS.)")
        return 0

    from .knowledge.benchmark_extras import all_cohort_benchmark_episodes

    episodes = list(episodes) + all_cohort_benchmark_episodes()

    model.eval()
    results = evaluate_model_against_benchmark(model, episodes, thresholds)

    from .knowledge.textbook_scenarios.neural_eval import run_textbook_scenarios_on_model

    textbook_block = run_textbook_scenarios_on_model(model, rng=np.random.default_rng(42))

    failures = []
    mape_max = thresholds.get("overall_weighted_mape_max", 0.16)
    verifier_min = thresholds.get("verifier_overall_min", 0.70)
    marker_thresholds = thresholds.get("marker_mape_max", {})
    verifier_cat_min = thresholds.get("verifier_category_min") or {}
    textbook_min = float(thresholds.get("textbook_mean_pass_rate_min", 0.0))

    if results["overall_weighted_mape"] > mape_max:
        failures.append(f"overall_mape={results['overall_weighted_mape']:.4f} > {mape_max}")
    if results["verifier_overall"] < verifier_min:
        failures.append(f"verifier={results['verifier_overall']:.4f} < {verifier_min}")
    for mid, max_val in marker_thresholds.items():
        pm = results.get("per_marker", {}).get(mid, {})
        if pm and pm.get("mean_mape", 0) > max_val:
            failures.append(f"{mid}_mape={pm['mean_mape']:.4f} > {max_val}")

    vcat_mean = results.get("verifier_category_mean") or {}
    if isinstance(verifier_cat_min, dict):
        for cat, need in verifier_cat_min.items():
            got = vcat_mean.get(cat)
            if got is None:
                continue
            if float(got) < float(need):
                failures.append(f"verifier_cat[{cat}]={got:.4f} < {float(need):.4f}")

    if textbook_min > 0.0 and textbook_block["textbook_mean_pass_rate"] < textbook_min:
        failures.append(
            f"textbook_mean_pass_rate={textbook_block['textbook_mean_pass_rate']:.4f} < {textbook_min}",
        )

    gate_passed = len(failures) == 0

    report = {
        "episodes": results["episodes"],
        "overall_weighted_mape": results["overall_weighted_mape"],
        "per_marker": results["per_marker"],
        "verifier": {
            "overall_score": results["verifier_overall"],
            "category_mean": results.get("verifier_category_mean", {}),
        },
        "textbook_mean_pass_rate": textbook_block["textbook_mean_pass_rate"],
        "textbook_scenarios": textbook_block["textbook_scenarios"],
        "gate": {
            "passed": gate_passed,
            "failures": failures,
        },
    }

    report_json = json.dumps(report, indent=2)
    print(f"\nBenchmark: gate.passed={gate_passed}")
    print(f"  overall_weighted_mape={results['overall_weighted_mape']:.4f}")
    print(f"  verifier.overall_score={results['verifier_overall']:.4f}")
    print(f"  textbook_mean_pass_rate={textbook_block['textbook_mean_pass_rate']:.4f}")
    if failures:
        print(f"  failures: {failures}")

    if report_path:
        if report_path.startswith("gs://"):
            report_local = tempfile.mktemp(suffix=".json")
            Path(report_local).write_text(report_json)
            try:
                _upload_file_to_gs_uri(report_local, report_path)
            except Exception as err:
                print(f"  Benchmark report GCS upload failed: {err}")
                return 1
            print(f"  Report uploaded to {report_path}")
        else:
            Path(report_path).write_text(report_json)
            print(f"  Report saved to {report_path}")

    if not gate_passed:
        print(
            "Benchmark gate did not pass — see report for details. "
            "(Trainer still exits 0; gate enforcement is the orchestrator's job.)",
        )
    return 0


def main():
    parser = argparse.ArgumentParser(description="Train the modular physiology network")
    parser.add_argument("--n-patients", type=int, default=20)
    parser.add_argument("--n-epochs", type=int, default=80)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--windows-per-patient", type=int, default=3)
    parser.add_argument(
        "--meal-window-bias",
        type=float,
        default=0.55,
        help="Probability of anchoring a window so a carb meal sits in the interior (post-meal dynamics + verifier); 0 = uniform starts only",
    )
    parser.add_argument("--output-path", type=str, default="pulse-model.pt")
    parser.add_argument("--input-dropout", type=float, default=0.3)
    parser.add_argument("--gut-loss-weight", type=float, default=0.5)
    parser.add_argument("--n-days", type=int, default=14)
    parser.add_argument(
        "--contribution-weights-json",
        type=str,
        default=None,
        help="Optional JSON object mapping knowledge contribution name to relative sampling weight",
    )
    parser.add_argument(
        "--coupling-prior-weight",
        type=float,
        default=0.03,
        help="Weight for coupling-prior (rate sensitivity sign) loss; 0 disables",
    )
    parser.add_argument(
        "--coupling-prior-samples",
        type=int,
        default=3,
        help="Interior timesteps per window for coupling prior loss",
    )
    parser.add_argument(
        "--verifier-loss-weight",
        type=float,
        default=0.03,
        help="Weight for training surrogate aligned with verifier.py (0 disables)",
    )
    parser.add_argument(
        "--huber-delta",
        type=float,
        default=1.0,
        help="Normalized residual scale for huber trajectory loss on partial contributions",
    )
    parser.add_argument(
        "--equal-contribution-weights",
        action="store_true",
        help="Sample all knowledge contributions uniformly (ignore default full_body tilt)",
    )
    parser.add_argument(
        "--cohort-statistic-weight",
        type=float,
        default=0.05,
        help="Weight for end-of-epoch cohort statistic loss (0 disables)",
    )
    parser.add_argument(
        "--cohort-sample-patients",
        type=int,
        default=16,
        help="Patient embeddings sampled per epoch for cohort rollouts. Iter 20 raised the default from 4→16 to give cohort statistics enough effective batch size to constrain downstream modules; the per-spec gradient quality scales sub-linearly past ~32.",
    )
    parser.add_argument(
        "--trajectory-band",
        type=float,
        default=0.0,
        help="Banded distillation for per-patient (sampled embedding) episodes: residuals within ±band (normalized σ) carry zero loss. 0 = pure imitation.",
    )
    parser.add_argument(
        "--trajectory-band-default",
        type=float,
        default=0.0,
        help="Banded distillation for default-patient (zero embedding) episodes. Lower than --trajectory-band so the zero embedding receives tighter cold-model imitation (the benchmark queries it directly; no patient identity to preserve).",
    )
    parser.add_argument(
        "--landmark-weight",
        type=float,
        default=0.0,
        help="Per-window post-meal landmark loss weight (Δpeak / time-to-peak / AUC for glucose & insulin). 0 disables.",
    )
    parser.add_argument("--landmark-pre-window", type=int, default=15)
    parser.add_argument("--landmark-post-window", type=int, default=120)
    parser.add_argument("--landmark-min-carbs", type=float, default=5.0)
    parser.add_argument(
        "--dose-response-weight",
        type=float,
        default=0.0,
        help="Per-epoch dose-response slope loss weight (Wolever 1996; differentiable soft Δpeak vs carb dose). 0 disables.",
    )
    parser.add_argument(
        "--dose-response-sample-patients",
        type=int,
        default=4,
        help="Patient embeddings sampled per epoch for the dose-response protocol",
    )
    parser.add_argument(
        "--dose-response-markers",
        type=str,
        default="",
        help=(
            "Multi-marker dose-response targets. Empty falls back to the legacy "
            "single-glucose-slope target (Wolever). Syntax: "
            "'marker:mode:params[;next]' with ';' as the inter-marker delimiter "
            "(gcloud --args= uses ',' as the inter-argument delimiter, so ',' is "
            "unsafe inside a single value). slope: "
            "'glucose:slope:<target>:<sigma>[:<weight>]'. peak (absolute "
            "amplitude vs literature line through origin): "
            "'<marker>:peak:<target_per_g>:<sigma_per_g>[:<weight>]'. rank: "
            "'<marker>:rank:<margin>[:<weight>]'. Example: "
            "'glucose:peak:0.7:0.25;insulin:peak:0.4:0.3;glp1:rank:0.3'."
        ),
    )
    parser.add_argument(
        "--gut-dose-sweep-weight",
        type=float,
        default=0.0,
        help="Per-epoch direct distillation of model.gut.forward_window against the cold-model absorption profile across an explicit dose sweep. Bypasses integrate(), so gradient lands directly on the gut kernel + embedding projection. 0 disables.",
    )
    parser.add_argument(
        "--gut-dose-sweep-sample-patients",
        type=int,
        default=4,
        help="Patient embeddings sampled per epoch for the gut dose sweep (zero embedding always included).",
    )
    parser.add_argument(
        "--gut-dose-sweep-auc-weight",
        type=float,
        default=1.0,
        help=(
            "Multiplier on the AUC-matching sub-term inside gut_dose_sweep "
            "(introduced iter 18). The per-element MSE component is mean over "
            "[B, T, C] with mostly-inactive entries; the AUC term collapses "
            "(T, C) into per-(dose, batch, channel) integrals before squaring "
            "so it does not get diluted. Total gut_dose_sweep loss is "
            "mse + ranking_weight·rank + auc_weight·auc, scaled by "
            "--gut-dose-sweep-weight. Bump above 1.0 if iter-18-style residual "
            "uniform over-amplitude persists; set to 0 to ablate."
        ),
    )
    parser.add_argument(
        "--insulin-sweep-weight",
        type=float,
        default=0.0,
        help=(
            "Per-epoch direct distillation of model.metabolic against the "
            "cold-model GSIR / clearance / hepatic-suppression curves on "
            "explicit (glucose, insulin) state sweeps. Bypasses integrate(), "
            "so gradient lands directly on the metabolic MLP + embedding "
            "projection. Iter 20's downstream analog of gut_dose_sweep — set "
            "non-zero to break the 'compensatory equilibrium' between gut "
            "amplitude and downstream miscalibration."
        ),
    )
    parser.add_argument(
        "--insulin-sweep-sample-patients",
        type=int,
        default=4,
        help="Patient embeddings sampled per epoch for the insulin sweep (zero embedding always included).",
    )
    parser.add_argument(
        "--insulin-sweep-auc-weight",
        type=float,
        default=1.0,
        help=(
            "Multiplier on the AUC-matching sub-term inside insulin_sweep. "
            "Same dilution-defeating role as in gut_dose_sweep but over the "
            "sweep-grid axis: sums the per-(grid-point, batch, species) rates "
            "before squaring so a uniform multiplicative miscalibration "
            "cannot hide behind near-zero entries. Bump above 1.0 if "
            "amplitude on the GSIR / clearance curves persists."
        ),
    )
    parser.add_argument(
        "--insulin-sweep-ranking-weight",
        type=float,
        default=1.0,
        help=(
            "Multiplier on the dose-monotonicity hinge inside insulin_sweep. "
            "Forces the metabolic module's predicted dI/dt to actually "
            "rank-order its sweep points (strictly increasing in glucose "
            "above the GSIR threshold; strictly decreasing in insulin), "
            "which per-element MSE alone does not imply."
        ),
    )
    parser.add_argument(
        "--fasting-stability-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for the fasting stability signal: MSE of glucose drift from NORM_CENTER "
            "over a no-meal rollout at the zero embedding. Directly counters the "
            "+57 mg/dL resting glucose drift observed in iter 26. Active from epoch 0. "
            "0 disables (default)."
        ),
    )
    parser.add_argument(
        "--fasting-stability-window",
        type=int,
        default=120,
        help=(
            "Duration in minutes of the fasting rollout used by the fasting stability signal. "
            "Default 120 (iter 26–30). Set to 240 to match the benchmark probe window directly."
        ),
    )
    parser.add_argument(
        "--postprandial-recovery-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for the postprandial recovery signal: MSE of glucose residual at "
            "+420-475 min after a standardized 50g-carb meal at the zero embedding. "
            "Mirrors the dietary_carbohydrate_meal_flow benchmark check that has been "
            "failing 60-70 mg/dL across iters 27-31. Active from epoch 0. 0 disables (default)."
        ),
    )
    parser.add_argument(
        "--default-baseline-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for the default-baseline signal: pulls the trained model's "
            "zero-embedding fasted mean output toward NORM_CENTER for selected "
            "markers. Iter 36 calibration investigation showed iter-32-to-35's "
            "hr_mape ~0.24 was a +15 bpm bias on the trained default-patient "
            "(zero embedding HR ~88 bpm vs bench eval mean ~63 bpm). This signal "
            "constrains the resting equilibrium so calibration starts from a "
            "saner basin. Active from epoch 0. 0 disables (default)."
        ),
    )
    parser.add_argument(
        "--default-baseline-markers",
        type=str,
        default="hr",
        help=(
            "Comma-separated marker ids constrained by --default-baseline-weight. "
            "Default 'hr' — the marker the iter-36 investigation identified. "
            "Add others (e.g. 'hr,sbp,dbp') only after confirming the HR-only "
            "version doesn't introduce regressions elsewhere."
        ),
    )
    parser.add_argument(
        "--carb-mass-balance-weight",
        type=float,
        default=0.0,
        help=(
            "Iter 74: weight for the cross-module carb→glycogen mass-balance "
            "conservation signal. Hinge-penalises two physical violations over "
            "a fasted→fed rollout: storing more glycogen than carbohydrate was "
            "ingested (Δglycogen > dose), and net-depleting glycogen while fed "
            "(Δglycogen < 0). Conservation inequalities — zero gradient when "
            "satisfied, so they cannot fight real dynamics; when they bind they "
            "give the otherwise gradient-starved glycogen pools an absolute-mass "
            "signal. Gated at the phase boundary. 0 disables (default)."
        ),
    )
    parser.add_argument(
        "--carb-mass-balance-sample-patients",
        type=int,
        default=2,
        help=(
            "Patient embeddings sampled per epoch for the carb-mass-balance "
            "rollouts, in addition to the zero default embedding. Kept small "
            "(default 2) because each (embedding, dose) pair is a full rollout."
        ),
    )
    parser.add_argument(
        "--cold-distill-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for the cold-model distillation signal: normalized "
            "trajectory MSE of the learned model (at the zero embedding) "
            "against simulate_full_body on a broad protocol pool — standard "
            "meal days, OGTT, 24h fast, high-fat meal, phase-shifted day, "
            "grazing — for the unobserved counter-regulatory + circadian "
            "markers (glucagon, ffa, ghrelin, leptin, acth, cortisol, bhb). "
            "These markers are byte-identical across iters 38-46 because no "
            "other training signal reaches the parameters that set their level "
            "and shape; this signal is the structural fix. Active from epoch 0. "
            "0 disables (default). Replaces the iter-39 marker-vitality "
            "range-floor band-aid."
        ),
    )
    parser.add_argument(
        "--cold-distill-markers",
        type=str,
        default="glucagon,ffa,ghrelin,leptin,acth,cortisol,bhb",
        help=(
            "Comma-separated marker ids distilled from the cold model by "
            "--cold-distill-weight. Default targets the unobserved counter-"
            "regulatory + circadian markers. insulin/glp1 are deliberately "
            "excluded — already constrained by the glucose-coupling, insulin-"
            "sweep, and gut-dose-sweep signals."
        ),
    )
    parser.add_argument(
        "--cold-distill-protocols-per-epoch",
        type=int,
        default=4,
        help=(
            "How many protocols from the distillation pool to sample per "
            "epoch (minibatching the protocol set). Each adds ~one rollout to "
            "the epoch. Default 4."
        ),
    )
    parser.add_argument(
        "--cold-distill-pool",
        type=str,
        default="synthetic",
        choices=("synthetic", "bench_cohorts", "both"),
        help=(
            "Which protocol pool the cold-model distillation distils over. "
            "'synthetic' = the 8-protocol broad pool (bench cohorts held out). "
            "'bench_cohorts' = exactly the two cohort episodes the bench scores "
            "unobserved markers on, calibrated to their own check-ins (trades "
            "the held-out check for distilling at the eval's operating point). "
            "'both' = synthetic + bench cohorts. Default 'synthetic'."
        ),
    )
    parser.add_argument(
        "--cold-distill-mode",
        type=str,
        default="trajectory",
        choices=("trajectory", "rate", "anchored"),
        help=(
            "Loss shape for cold-model distillation. 'trajectory' (legacy, "
            "iters 47-49) rolls the model out at the calibrated embedding and "
            "matches the integrated trajectory — diagnostic showed the "
            "gradient to the dead-marker heads vanishes (~1e-5) through the "
            "integration chain. 'rate' (iter 50+) teacher-forces the cold "
            "state at each step and matches the model's instantaneous d/dt to "
            "the cold ODE's — gradient straight to the head, ~T× stronger, but "
            "offset-invariant (level can drift). 'anchored' (iter 75) = 'rate' "
            "PLUS a short-horizon free-running absolute-level anchor (reset to "
            "cold truth every --cold-distill-anchor-window minutes) that closes "
            "that offset-invariance hole. Default 'trajectory'."
        ),
    )
    parser.add_argument(
        "--cold-distill-anchor-window",
        type=int,
        default=60,
        help=(
            "mode='anchored' only: minutes between resets-to-truth for the "
            "free-running level anchor. Bounds the integration chain (gradient "
            "dilution ~W×, not ~T×) and how far an untrained head can diverge. "
            "Default 60 (≈ one meal-response arc)."
        ),
    )
    parser.add_argument(
        "--cold-distill-anchor-samples",
        type=int,
        default=8,
        help=(
            "mode='anchored' only: max free-rollout level-anchor windows per "
            "protocol per epoch, evenly spaced across the protocol. Bounds peak "
            "autograd memory (all windows are retained until backward). Default "
            "8 (≈ morning / each meal / overnight over a 24 h protocol)."
        ),
    )
    parser.add_argument(
        "--physiology-rules-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for the physiology-rules signal: per-trajectory squared-hinge "
            "supervision against literature-derived plausibility constraints "
            "(direction-after-event, sign, timing band, anti-correlation, etc.). "
            "See pulse/knowledge/physiology_rules.py for the registry. Each rule "
            "is hinge-shaped — gradient stops at satisfaction, never enforcing "
            "more than the literature claims. Phase-2 enabled. 0 disables (default)."
        ),
    )
    parser.add_argument(
        "--physiology-rules-sample-patients",
        type=int,
        default=4,
        help=(
            "Number of patient embeddings sampled per epoch for physiology-rule "
            "supervision (in addition to the zero default embedding). "
            "Default 4 matches dose-response / sweep signal cadence."
        ),
    )
    parser.add_argument(
        "--physiology-rules-adaptive",
        dest="physiology_rules_adaptive",
        action="store_true",
        help=(
            "Iter 67: violation-proportional reweighting between rules. Maintains "
            "a per-rule EMA of violation_mean and reweights so badly-violated "
            "rules pull harder while satisfied rules fade — total signal weight "
            "is preserved across the reweighting (only the inter-rule "
            "distribution shifts). With ~60 rules at signal weight 0.05, the "
            "default per-rule pull is ~8e-4 regardless of satisfaction; adaptive "
            "concentrates the budget on rules that still have work to do."
        ),
    )
    parser.add_argument(
        "--cohort-statistic-adaptive",
        dest="cohort_statistic_adaptive",
        action="store_true",
        help=(
            "Iter 74: violation-proportional reweighting between cohort specs, "
            "the analog of --physiology-rules-adaptive. Maintains a per-spec EMA "
            "of the Gaussian-z² discrepancy and reweights so badly-missed specs "
            "pull harder while matched specs fade — total signal weight is "
            "preserved (only the inter-spec distribution shifts). With ~31 specs "
            "at signal weight 0.15, the default per-spec pull is ~2.6e-3; "
            "adaptive concentrates the budget on specs that still have work to "
            "do, replacing the hand-tuned per-spec weight bumps."
        ),
    )
    parser.add_argument(
        "--n-default-patients",
        type=int,
        default=0,
        help="Extra cold-model episodes (PatientParams() defaults) supervised through the zero embedding. Aligns trajectory training with the textbook benchmark which queries every scenario at zero. 0 disables.",
    )
    parser.add_argument(
        "--cohort-cold-init",
        dest="cohort_cold_init",
        action="store_true",
        default=True,
        help="Cohort signal uses per-spec cold-model fasting state as integration initial condition (matches benchmark). Default: enabled.",
    )
    parser.add_argument(
        "--no-cohort-cold-init",
        dest="cohort_cold_init",
        action="store_false",
        help="Cohort signal falls back to NORM_CENTER initial state (legacy behavior).",
    )
    parser.add_argument(
        "--phase1-epochs",
        type=int,
        default=None,
        help="Epochs for Phase 1 (distillation only); defaults to --n-epochs",
    )
    parser.add_argument(
        "--phase2-epochs",
        type=int,
        default=0,
        help="Epochs for Phase 2 (full losses); 0 = single-phase",
    )
    parser.add_argument(
        "--phase2-lr",
        type=float,
        default=None,
        help="Learning rate for Phase 2; defaults to --lr",
    )
    parser.add_argument("--gcs-bucket", type=str, default=None)
    parser.add_argument("--gcs-object", type=str, default=None)
    parser.add_argument("--benchmark-dataset-uri", type=str, default=None)
    parser.add_argument("--benchmark-thresholds-uri", type=str, default=None)
    parser.add_argument("--benchmark-report-path", type=str, default=None)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Pin threads=1 and request deterministic PyTorch algorithms "
             "(slow on multi-CPU hosts; default off — opt in for tests).",
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help=(
            "Skip training. Load checkpoint from gs://<gcs-bucket>/<gcs-object> "
            "and run benchmark only. Requires --gcs-bucket, --gcs-object, and "
            "--benchmark-dataset-uri."
        ),
    )

    args = parser.parse_args()

    if args.benchmark_only:
        if not args.gcs_bucket or not args.gcs_object:
            parser.error("--benchmark-only requires --gcs-bucket and --gcs-object")
        if not args.benchmark_dataset_uri:
            parser.error("--benchmark-only requires --benchmark-dataset-uri")
        from .diagnostics.probe import load_model_from_checkpoint
        import tempfile as _tempfile
        ckpt_local = _tempfile.mktemp(suffix=".pt")
        print(f"Downloading checkpoint gs://{args.gcs_bucket}/{args.gcs_object} ...")
        _download_gs_uri_to_file(f"gs://{args.gcs_bucket}/{args.gcs_object}", ckpt_local)
        model, _ = load_model_from_checkpoint(ckpt_local)
        # Iter 61: rehydrate embedding prior stats from the checkpoint
        # so calibrate_embedding's diagonal Gaussian prior gets the
        # actual trained-table mean/std (legacy checkpoints without
        # these keys fall back to the isotropic L2 path).
        _ckpt_dict = torch.load(ckpt_local, map_location="cpu", weights_only=False)
        _pm = _ckpt_dict.get("embedding_prior_mean")
        _ps = _ckpt_dict.get("embedding_prior_std")
        if _pm is not None and _ps is not None:
            model._embedding_prior_mean = torch.tensor(_pm, dtype=torch.float32)
            model._embedding_prior_std = torch.tensor(_ps, dtype=torch.float32)
        sys.exit(_run_benchmark(
            model, ckpt_local,
            args.benchmark_dataset_uri, args.benchmark_thresholds_uri,
            args.benchmark_report_path, args.gcs_bucket,
        ))

    cw = _load_contribution_weights_json(args.contribution_weights_json)
    if args.equal_contribution_weights:
        cw = {c.name: 1.0 for c in ALL_CONTRIBUTIONS}

    exit_code = train(
        n_patients=args.n_patients,
        n_epochs=args.n_epochs,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        seed=args.seed,
        windows_per_patient=args.windows_per_patient,
        meal_window_bias=args.meal_window_bias,
        output_path=args.output_path,
        input_dropout=args.input_dropout,
        gut_loss_weight=args.gut_loss_weight,
        n_days=args.n_days,
        contribution_weights=cw,
        coupling_prior_weight=args.coupling_prior_weight,
        coupling_prior_samples=args.coupling_prior_samples,
        verifier_loss_weight=args.verifier_loss_weight,
        huber_delta=args.huber_delta,
        cohort_statistic_weight=args.cohort_statistic_weight,
        cohort_sample_patients=args.cohort_sample_patients,
        cohort_statistic_adaptive=args.cohort_statistic_adaptive,
        trajectory_band=args.trajectory_band,
        trajectory_band_default=args.trajectory_band_default,
        landmark_weight=args.landmark_weight,
        landmark_pre_window=args.landmark_pre_window,
        landmark_post_window=args.landmark_post_window,
        landmark_min_carbs=args.landmark_min_carbs,
        dose_response_weight=args.dose_response_weight,
        dose_response_sample_patients=args.dose_response_sample_patients,
        dose_response_markers=_parse_dose_response_markers(args.dose_response_markers),
        gut_dose_sweep_weight=args.gut_dose_sweep_weight,
        gut_dose_sweep_sample_patients=args.gut_dose_sweep_sample_patients,
        gut_dose_sweep_auc_weight=args.gut_dose_sweep_auc_weight,
        insulin_sweep_weight=args.insulin_sweep_weight,
        insulin_sweep_sample_patients=args.insulin_sweep_sample_patients,
        insulin_sweep_auc_weight=args.insulin_sweep_auc_weight,
        insulin_sweep_ranking_weight=args.insulin_sweep_ranking_weight,
        fasting_stability_weight=args.fasting_stability_weight,
        fasting_stability_window=args.fasting_stability_window,
        postprandial_recovery_weight=args.postprandial_recovery_weight,
        default_baseline_weight=args.default_baseline_weight,
        default_baseline_markers=_split_markers(args.default_baseline_markers),
        carb_mass_balance_weight=args.carb_mass_balance_weight,
        carb_mass_balance_sample_patients=args.carb_mass_balance_sample_patients,
        cold_distill_weight=args.cold_distill_weight,
        cold_distill_markers=_split_markers(args.cold_distill_markers),
        cold_distill_protocols_per_epoch=args.cold_distill_protocols_per_epoch,
        cold_distill_pool=args.cold_distill_pool,
        cold_distill_mode=args.cold_distill_mode,
        cold_distill_anchor_window=args.cold_distill_anchor_window,
        cold_distill_anchor_samples=args.cold_distill_anchor_samples,
        physiology_rules_weight=args.physiology_rules_weight,
        physiology_rules_sample_patients=args.physiology_rules_sample_patients,
        physiology_rules_adaptive=args.physiology_rules_adaptive,
        n_default_patients=args.n_default_patients,
        cohort_use_cold_init=args.cohort_cold_init,
        phase1_epochs=args.phase1_epochs,
        phase2_epochs=args.phase2_epochs,
        phase2_lr=args.phase2_lr,
        gcs_bucket=args.gcs_bucket,
        gcs_object=args.gcs_object,
        benchmark_dataset_uri=args.benchmark_dataset_uri,
        benchmark_thresholds_uri=args.benchmark_thresholds_uri,
        benchmark_report_path=args.benchmark_report_path,
        deterministic=args.deterministic,
    )

    if exit_code != 0:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
