#!/usr/bin/env python3
"""Iter 97 (review 4.11): the gradient balance among the auxiliary signals.

Runs every signal of the iter-97 recipe ONCE at its recipe weight and reports
the L2 norm of its own (weighted) gradient contribution — the number the joint
clip acts on — plus the raw loss. Balance among aux signals is set by gradient
scale, not by the weights (gut_dose_sweep at 0.10 out-pulled dose_response at
0.40 by 5x on the iter-96 artifact), so this is the table to read before
touching a weight. Also reports the per-window trajectory norm distribution.

Usage:
  uv run python scripts/iter97_signal_balance.py [--checkpoint model.pt] [--n-patients N] [--fast]

Untrained model by default (a fresh ModularPhysiologyNetwork at the recipe's
hidden_dim); ``--checkpoint`` loads an artifact instead. ``--fast`` skips the
long signals (cohort, distillation, rules) — the reviewer's probe under
scratchpad/training/probe_balance.py is the origin of this script.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import pulse  # noqa: E402

assert pulse.__file__.startswith(str(_ROOT)), pulse.__file__

from pulse.dose_response import DoseResponseProtocol  # noqa: E402
from pulse.knowledge import ALL_COHORT_STATISTICS  # noqa: E402
from pulse.knowledge.physiology_rules import PHYSIOLOGY_RULES  # noqa: E402
from pulse.model import ModularPhysiologyNetwork  # noqa: E402
from pulse.train import _load_spec_train_args, _parse_dose_response_markers, build_arg_parser  # noqa: E402
from pulse.training import (  # noqa: E402
    CarbMassBalanceSignal,
    CohortStatisticSignal,
    ColdModelDistillationSignal,
    DefaultBaselineSignal,
    DoseResponseSignal,
    EmbeddingPriorSignal,
    FastingStabilitySignal,
    GutDoseSweepSignal,
    InsulinSweepSignal,
    MealResponseSignal,
    PhysiologyRulesSignal,
    PostprandialRecoverySignal,
    SetpointSupervisionSignal,
    SignalContext,
    TrajectoryRolloutSignal,
    WeightSchedule,
)
from pulse.training import trajectory_signal as tsmod  # noqa: E402
from pulse.training.trajectory_signal import parse_band_per_marker  # noqa: E402
from pulse.types import EMBEDDING_DIM  # noqa: E402


class _Noop:
    def step(self) -> None: ...
    def zero_grad(self, *a, **k) -> None: ...


def _gnorm(ps) -> float:
    return math.sqrt(sum(float(p.grad.pow(2).sum()) for p in ps if p.grad is not None))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--spec", type=str, default=str(_ROOT / "train" / "spec.json"))
    ap.add_argument("--n-patients", type=int, default=8)
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()

    args = build_arg_parser().parse_args(_load_spec_train_args(a.spec) + ["--spec", a.spec])
    torch.manual_seed(0)
    if a.checkpoint:
        from pulse.diagnostics.probe import load_model_from_checkpoint
        model, _ = load_model_from_checkpoint(a.checkpoint)
    else:
        hd = int(args.hidden_dim)
        model = ModularPhysiologyNetwork(
            metabolic_hidden=hd, appetite_hidden=max(24, hd // 2), stress_hidden=max(24, hd // 2),
            cardiovascular_hidden=hd, thermoreg_hidden=max(16, hd // 3), respiratory_hidden=max(16, hd // 3),
        )
    model.train()
    N = int(a.n_patients)
    emb = nn.Embedding(N, EMBEDDING_DIM)
    nn.init.normal_(emb.weight, std=0.1)
    params = list(model.parameters()) + list(emb.parameters())

    def zero() -> None:
        for p in params:
            p.grad = None

    def modnorms() -> dict[str, float]:
        out = {}
        for name, mod in [("gut", model.gut), ("met", model.metabolic), ("cvs", model.cardiovascular),
                          ("stress", model.stress), ("app", model.appetite), ("thermo", model.thermoreg)]:
            out[name] = round(_gnorm(list(mod.parameters())), 3)
        out["emb"] = round(_gnorm([emb.weight]), 3)
        return out

    rng = np.random.default_rng(0)
    ctx = SignalContext(epoch=40, total_epochs=58, rng=rng, device=torch.device("cpu"),
                        optimizer=_Noop(), params=params, grad_clip=1e9, aux_signal_clip=0.0)
    ds = tsmod.generate_trajectory_dataset(n_patients=N, seed=1, n_days=2, weight_by_name={"full_body": 1.0})
    sp = {int(r["patient_id"]): r["setpoints"] for r in ds if r.get("setpoints")}
    mr = {int(r["patient_id"]): r["meal_response"] for r in ds if r.get("meal_response")}
    dr = DoseResponseProtocol(marker_targets=_parse_dose_response_markers(args.dose_response_markers))
    markers = tuple(args.cold_distill_markers.split(":")) if args.cold_distill_markers else None
    sigs = [
        DoseResponseSignal(n_patients=N, sample_patients=args.dose_response_sample_patients,
                           weight=WeightSchedule(args.dose_response_weight), protocol=dr, perturb_protocols=True),
        GutDoseSweepSignal(n_patients=N, sample_patients=args.gut_dose_sweep_sample_patients,
                           weight=WeightSchedule(args.gut_dose_sweep_weight), auc_weight=args.gut_dose_sweep_auc_weight),
        InsulinSweepSignal(n_patients=N, sample_patients=args.insulin_sweep_sample_patients,
                           weight=WeightSchedule(args.insulin_sweep_weight), auc_weight=args.insulin_sweep_auc_weight,
                           ranking_weight=args.insulin_sweep_ranking_weight),
        FastingStabilitySignal(window_min=args.fasting_stability_window, weight=WeightSchedule(args.fasting_stability_weight)),
        PostprandialRecoverySignal(weight=WeightSchedule(args.postprandial_recovery_weight), perturb_protocols=True),
        DefaultBaselineSignal(weight=WeightSchedule(args.default_baseline_weight),
                              markers=tuple(args.default_baseline_markers.split(":"))),
        SetpointSupervisionSignal(weight=WeightSchedule(args.setpoint_supervision_weight), targets=sp),
        MealResponseSignal(weight=WeightSchedule(args.meal_response_weight), n_patients=N,
                           sample_patients=args.meal_response_sample_patients, targets=mr),
        EmbeddingPriorSignal(weight=WeightSchedule(args.embedding_prior_weight)),
        CarbMassBalanceSignal(weight=WeightSchedule(args.carb_mass_balance_weight), n_patients=N,
                              sample_patients=args.carb_mass_balance_sample_patients),
    ]
    if not a.fast:
        sigs += [
            CohortStatisticSignal(specs=list(ALL_COHORT_STATISTICS), n_patients=N,
                                  sample_patients=args.cohort_sample_patients,
                                  weight=WeightSchedule(args.cohort_statistic_weight), adaptive=True,
                                  groups_per_step=args.cohort_groups_per_step, perturb_protocols=True),
            ColdModelDistillationSignal(weight=WeightSchedule(args.cold_distill_weight),
                                        **({"markers": markers} if markers else {}),
                                        protocols_per_epoch=args.cold_distill_protocols_per_epoch,
                                        pool=args.cold_distill_pool, mode=args.cold_distill_mode,
                                        anchor_window=args.cold_distill_anchor_window,
                                        anchor_samples=args.cold_distill_anchor_samples,
                                        anchor_local_scale_floor=args.cold_distill_anchor_local_scale_floor,
                                        anchor_long_window=args.cold_distill_anchor_long_window,
                                        anchor_long_samples=args.cold_distill_anchor_long_samples,
                                        anchor_level_band=args.cold_distill_level_band),
            PhysiologyRulesSignal(rules=list(PHYSIOLOGY_RULES), n_patients=N,
                                  sample_patients=args.physiology_rules_sample_patients,
                                  weight=WeightSchedule(args.physiology_rules_weight), adaptive=True,
                                  arms_per_step=args.rules_arms_per_step),
        ]
    print(f"{'signal':24} {'w':>6} {'raw_loss':>10} {'||g|| (weighted, one aux step)':>32} {'t':>7}  modules")
    for s in sigs:
        zero(); t0 = time.time()
        r = s.compute(model, emb, ctx)
        g = ctx.aux_grad_norms.get(s.name, [float("nan")])[-1]
        print(f"{s.name:24} {s.weight_at(40):>6} {r.loss_sum:10.4f} {g:32.4f} {time.time()-t0:7.1f}s  {modnorms()}")
        sys.stdout.flush()

    # Trajectory windows, per-window pre-clip norm (the joint clip is 10).
    zero()
    traj = TrajectoryRolloutSignal(
        n_patients=N, n_days=2, seed=1, contribution_weights={"full_body": 1.0}, windows_per_patient=1,
        meal_window_bias=args.meal_window_bias, input_dropout=args.input_dropout, huber_delta=args.huber_delta,
        gut_loss_weight=0.0, coupling_weight=WeightSchedule(args.coupling_prior_weight),
        verifier_weight=WeightSchedule(0.02), landmark_weight=WeightSchedule(args.landmark_weight),
        n_default_patients=1, trajectory_band_per_marker=parse_band_per_marker(args.trajectory_band_per_marker),
        shape_markers=tuple(args.trajectory_shape_markers.split(":")) if args.trajectory_shape_markers else (),
    )
    ctx2 = SignalContext(epoch=40, total_epochs=58, rng=rng, device=torch.device("cpu"),
                         optimizer=_Noop(), params=params, grad_clip=1e9)
    t0 = time.time()
    r = traj.compute(model, emb, ctx2)
    tn = np.asarray(ctx2.traj_grad_norms)
    print(f"\ntrajectory: {r.n_units} windows in {time.time()-t0:.1f}s; per-window ||g||: median {np.median(tn):.3f} "
          f"mean {tn.mean():.3f} max {tn.max():.3f} frac>10 {(tn > 10).mean():.2f}; loss {r.avg_loss:.4f} sub={r.sub_metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
