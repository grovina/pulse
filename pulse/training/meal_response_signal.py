"""
Per-patient meal-response supervision — unfreeze Ra.

Iter 91. The measured blocker coming out of iter 90.

WHAT WAS MEASURED. iter-90 fixed the embedding->physiology map: given a calibration window
with NO meal, calibration recovers a person's fasting glucose setpoint to within 0.96 mg/dL
(iter-89: 54 mg/dL, worse than predicting the population mean). But put a meal in that window
-- the realistic case -- and Gb recovery collapses to 17.84 mg/dL, skill -1.38.

The failure has a signature, and it is a BIAS, not scatter: calibration pushes Gb from the
prior mean (89.8) up to ~110 in every person, whether their true Gb is 100 or 85. The cause is
that the per-patient meal gain Ra is FROZEN -- trained std 0.01 across the population. So the
optimizer cannot fit an individual's postprandial amplitude, and compensates with the only
lever it has: it inflates their baseline.

That one bias explains BOTH of iter-90's failures. It breaks recovery, and it drives the gate
regression too: with Sg finally correct (tau 55 min, vs 515 min in iter-89) glucose actually
relaxes toward Gb inside the fasting eval window, so an inflated Gb now shows up directly as
an inflated prediction (glucose_mape 0.193 -> 0.210). In iter-89 the same bias was harmless
only because the broken Sg meant glucose never moved. Fixing the physics exposed a defect that
was always there -- exactly as the iter-87 meal-timing fix did.

WHY Ra WAS FROZEN. Nothing ever supervised per-patient meal amplitude:
  - SetpointSupervisionSignal (iter 90) supervises Gb / HR0 / HRV0 / SBP0 / DBP0 -- not Ra.
  - DoseResponseSignal supervises a POPULATION target (Wolever, 0.7 mg/dL/g) that is identical
    for every patient, so it carries no per-patient information at all.
So ra_baseline_net (added in iter 88) had no gradient that could distinguish one person from
another, and it learned nothing.

WHAT THIS DOES. The teacher knows each patient's true postprandial response exactly -- it is a
consequence of their sampled Si, Sg, absorption rates and insulin response -- and we were
discarding it, the same oversight that left the setpoints unused through iter 89. Episodes now
carry the teacher's own response to a standard 75 g meal (the OGTT dose), and this signal
supervises the student's realized response at that patient's embedding against it.

The per-patient spread is large and real: peak rise ranges 40.7-78.0 mg/dL (mean 58.5, sd 11.1,
CV 0.19) across teacher patients -- nearly 2x between individuals. That is the physiological
variation Ra has been unable to express.

Time-to-peak is supervised as well (teacher range 49-90 min). It carries the insulin-action
timing, and is the natural place for the lag constant p2 to get a gradient -- iter-90 drifted
p2 toward its floor (tau 83 min, teacher 33 min), which is plausibly the same compensation
showing up in a different parameter.

COST. One 240-minute rollout per sampled patient per epoch (the protocol is short and
meal-driven), mirroring how DoseResponseSignal already samples a handful of patients. Gated to
phase 2 like the other amplitude signals, so it shapes amplitude once a meaningful embedding
exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from ..knowledge.full_body import (
    STANDARD_MEAL_CARBS_G, STANDARD_MEAL_FATS_G, STANDARD_MEAL_PROTEINS_G,
)
from ..model import integrate, precompute_gut_outputs
from ..modules.gut import MealEvent
from ..modules import metabolic as _met
from ..types import MARKER_INDEX, NORM_CENTER, NORM_SCALE
from .safe_step import accumulate_grad
from .signals import SignalContext, SignalResult, TrainingSignal, WeightSchedule

_GLUCOSE = MARKER_INDEX["glucose"]
# Mirrors _standard_meal_response in knowledge/full_body.py — the student must be measured on
# exactly the protocol the teacher's reference was measured on.
_DURATION_MIN = 360
_MEAL_TIME_MIN = 120
_START_HOUR = 8.0

# Sigmas for the Gaussian discrepancy, in the units of each target. The peak sigma is the
# teacher population's own sd (11.1 mg/dL), so a patient at the population mean contributes
# ~0 and the signal only pushes on genuine per-patient deviation. Time-to-peak is softer
# (the student's peak timing is shaped by p2/Si and should not be over-constrained).
_SIGMA_PEAK_MG_DL = 11.0
_SIGMA_TPEAK_MIN = 20.0
# Iter 91: the settled pre-meal level is the OBSERVABLE fasting glucose -- what calibration
# actually sees. It is NOT params.Gb (the standing hepatic source puts the equilibrium at
# Gb + egp/Sg, measured +0.85..+5.09 mg/dL and varying per patient), so supervising the latent
# Gb alone injects a per-patient bias into exactly the quantity the benchmark scores.
_SIGMA_FASTING_MG_DL = 8.0
_FASTING_WEIGHT = 1.0
# Time-to-peak is worth less than amplitude: it is the secondary quantity here, and Ra is what
# we are unfreezing.
_TPEAK_WEIGHT = 0.3

# Softmax sharpness for the differentiable peak/argmax over the post-meal window, in raw
# mg/dL. Sharp enough to track the true peak (iter-90 sharpened the dose-response estimator
# for exactly this reason: a soft mean over a long window is not the peak the target names).
_SOFTARGMAX_BETA = 3.0


@dataclass
class MealResponseSignal(TrainingSignal):
    """Supervise each patient's realized meal response against the teacher's own."""

    weight: WeightSchedule = field(default_factory=lambda: WeightSchedule(0.0))
    n_patients: int = 0
    sample_patients: int = 4
    # pid -> {"glucose_peak_rise": mg/dL, "glucose_time_to_peak_min": min}
    targets: dict[int, dict[str, float]] = field(default_factory=dict)

    name: str = "meal_response"
    source: str = "Iter 91: Ra frozen (std 0.01) -> calibration inflates Gb to fit meal peaks"
    category: str = "mechanism"

    def weight_at(self, epoch: int) -> float:
        return self.weight.at(epoch)

    def compute(
        self,
        model: nn.Module,
        embeddings: nn.Embedding,
        ctx: SignalContext,
    ) -> SignalResult:
        w = self.weight_at(ctx.epoch)
        if w <= 0 or not self.targets:
            return SignalResult()

        device = ctx.device
        # Sample from the patients that actually HAVE a teacher meal-response target (only
        # full_body simulates a whole patient, ~45% of the mix). The shared
        # select_supervised_embeddings helper samples over all n_patients and prepends the zero
        # embedding, neither of which is right here: a patient without a target has nothing to
        # supervise, and the default patient's amplitude is already covered by dose-response.
        available = sorted(self.targets)
        if not available:
            return SignalResult()
        k = min(int(self.sample_patients), len(available))
        pids = [int(p) for p in ctx.rng.choice(available, size=k, replace=False)]

        pid_t = torch.tensor(pids, dtype=torch.long, device=device)
        emb = embeddings(pid_t)  # [P, EMB]
        P = emb.shape[0]

        meals = [MealEvent(
            time=float(_MEAL_TIME_MIN),
            carbs=STANDARD_MEAL_CARBS_G,
            fats=STANDARD_MEAL_FATS_G,
            proteins=STANDARD_MEAL_PROTEINS_G,
        )]
        # Start each patient at THEIR OWN fasting glucose, not the population centre.
        # The teacher's reference (_standard_meal_response) starts the simulation at that
        # patient's own params.Gb, so measuring the student from NORM_CENTER (95 mg/dL) would
        # compare responses launched from different baselines: a patient whose Gb is 70 would
        # be probed from 95 and spend the pre-meal window drifting, corrupting both the
        # pre-meal reference and the realized peak. Decode this patient's Gb from the same
        # head SetpointSupervisionSignal trains, and seed glucose there.
        initial = torch.tensor(NORM_CENTER, dtype=torch.float32, device=device)
        initial = initial.unsqueeze(0).repeat(P, 1)
        e_met = model.embedding_projections["metabolic"](emb)
        b_emb = _met._GLUCOSE_BASELINE_MAX_Z * torch.tanh(
            model.metabolic.glucose_baseline_net(e_met).squeeze(-1)
        )  # z-units
        gb_raw = NORM_CENTER[_GLUCOSE] + float(NORM_SCALE[_GLUCOSE]) * b_emb  # [P]
        initial = initial.clone()
        initial[:, _GLUCOSE] = gb_raw

        gut = precompute_gut_outputs(
            model, emb, _DURATION_MIN, dt=1.0,
            start_time_minutes=_START_HOUR * 60.0, meals=meals,
        )
        traj = integrate(
            model, initial, emb, _DURATION_MIN, dt=1.0,
            start_time_minutes=_START_HOUR * 60.0, meals=meals, gut_outputs=gut,
        )  # [P, T, STATE_DIM]

        glucose = traj[..., _GLUCOSE]                     # [P, T]
        pre = glucose[:, _MEAL_TIME_MIN - 1]              # [P] — this patient's own pre-meal level
        post = glucose[:, _MEAL_TIME_MIN:]                # [P, T-30]

        # Differentiable peak and time-to-peak (softargmax over the post-meal window).
        wts = torch.softmax(_SOFTARGMAX_BETA * post, dim=-1)     # [P, T-30]
        peak = (wts * post).sum(dim=-1)                          # [P]
        idx = torch.arange(post.shape[-1], device=device, dtype=post.dtype)
        tpeak = (wts * idx).sum(dim=-1)                          # [P]

        pred_rise = peak - pre

        tgt_rise = torch.tensor(
            [self.targets[p]["glucose_peak_rise"] for p in pids],
            dtype=torch.float32, device=device,
        )
        tgt_tpeak = torch.tensor(
            [self.targets[p]["glucose_time_to_peak_min"] for p in pids],
            dtype=torch.float32, device=device,
        )

        tgt_fasting = torch.tensor(
            [self.targets[p]["glucose_fasting"] for p in pids],
            dtype=torch.float32, device=device,
        )
        loss_fasting = ((pre - tgt_fasting) / _SIGMA_FASTING_MG_DL).pow(2).mean()
        loss_peak = ((pred_rise - tgt_rise) / _SIGMA_PEAK_MG_DL).pow(2).mean()
        loss_tpeak = ((tpeak - tgt_tpeak) / _SIGMA_TPEAK_MIN).pow(2).mean()
        loss = _FASTING_WEIGHT * loss_fasting + loss_peak + _TPEAK_WEIGHT * loss_tpeak

        sub = {
            "n_patients": float(P),
            "peak_rise_pred_mean": float(pred_rise.detach().mean()),
            "peak_rise_target_mean": float(tgt_rise.mean()),
            "peak_rise_mae": float((pred_rise - tgt_rise).abs().detach().mean()),
            # The number this signal exists to move: per-patient SPREAD. If the student's
            # realized peak-rise sd stays near zero while the target sd is ~11 mg/dL, Ra is
            # still frozen and the signal is not doing its job.
            "peak_rise_pred_sd": float(pred_rise.detach().std()) if P > 1 else 0.0,
            "peak_rise_target_sd": float(tgt_rise.std()) if P > 1 else 0.0,
            "tpeak_mae": float((tpeak - tgt_tpeak).abs().detach().mean()),
            # The OBSERVABLE fasting glucose error -- the quantity calibration must recover.
            "fasting_mae": float((pre - tgt_fasting).abs().detach().mean()),
        }

        accumulate_grad(
            w * loss, ctx, signal=self.name,
            extra={"raw_loss": float(loss.detach().item()), "weight": float(w), **sub},
        )
        return SignalResult(loss_sum=float(loss.detach().item()), n_units=1, sub_metrics=sub)
