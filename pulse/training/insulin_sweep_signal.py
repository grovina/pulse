"""
Insulin sweep distillation signal — the iter 20 downstream analog of
``GutDoseSweepSignal``.

The metabolic module is a mass-action ODE that computes seven rates at
once (glucose, insulin, glucagon, ffa, bhb, lactate, hepatic_output) from
its current state, gut output coupling, cortisol coupling, and embedding.
Iter 19 left it under-tuned: glucose_mape stayed at 0.578 even after the
gut amplitude collapsed, because trajectory MSE alone is too dilute to
pin down each species' rate response.

This signal directly supervises ``model.metabolic`` rates against the
analytical cold-model GSIR / clearance curves on two state sweeps:

* **Glucose sweep** — vary plasma glucose at fixed (insulin = Ib, all
  other state at typicals, no meal, no stress, no activity). Compares
  predicted dI/dt, dHep/dt, dGlucagon/dt, dG/dt against the cold model.
  This is the canonical glucose-stimulated insulin release curve, and
  the canonical hepatic suppression curve, in one batched forward pass.

* **Insulin sweep** — vary plasma insulin at fixed (G = Gb, all other
  state at typicals, no meal, no stress, no activity). Compares
  predicted dI/dt (clearance) and dHep/dt (insulin → hepatic
  suppression) against the cold model.

Loss formulation mirrors ``GutDoseSweepSignal``:

* ``mse``  — per-(grid-point, species) normalized squared error.
* ``rank`` — monotonicity hinges (dI/dt strictly increasing with G
  above the GSIR threshold; dI/dt strictly decreasing with I).
* ``auc``  — sum-over-grid undiluted amplitude term per species. Plays
  the same dilution-defeating role as the time-integral AUC term in
  the gut signal: sums first, squares once, so a uniform multiplicative
  miscalibration cannot hide behind near-zero entries.

Gradient lands on the metabolic MLP and its embedding projection only;
the rest of the network sees no perturbation from this signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import numpy as np
import torch
import torch.nn as nn

from ..knowledge.full_body import PatientParams, glucose_fluxes, resolve_derived_params
from ..model import ModularPhysiologyNetwork
from ..modules.base import compute_time_features
from ..types import (
    DUODENAL_DIM,
    EXTERNAL_INPUT_DIM,
    GUT_OUTPUT_DIM,
    MARKER_INDEX,
    MODULE_MARKER_INDICES,
    NORM_CENTER,
    NORM_SCALE,
)
from .embedding_sampler import select_supervised_rows
from .safe_step import accumulate_grad
from .signals import SignalContext, SignalResult, TrainingSignal, WeightSchedule


# Indices of the seven cold-modelled metabolic species inside the
# module's local state slice (which is what the module's ``forward``
# consumes — see ``model.ModularPhysiologyNetwork.forward``).
#
# Iter 55/56 extended the metabolic module to 10 species by adding three
# slow internal states (liver_glycogen, muscle_glycogen,
# mitochondrial_capacity at local indices 7, 8, 9). The cold ODE
# (`simulate_full_body`) does not model those; this signal restricts its
# rate-matching to the first 7 (observed/cold-modelled) species via
# `_N_COLD_METABOLIC`. The new heads are supervised by cohort-statistic +
# physiology-rule signals instead.
_MET_LOCAL_GLUCOSE = 0
_MET_LOCAL_INSULIN = 1
_MET_LOCAL_GLUCAGON = 2
_MET_LOCAL_FFA = 3
_MET_LOCAL_BHB = 4
_MET_LOCAL_LACTATE = 5
_MET_LOCAL_HEP = 6
_N_COLD_METABOLIC = 7


@dataclass(frozen=True)
class InsulinSweepProtocol:
    """Two state-axis sweeps for direct probe-style metabolic supervision.

    ``glucose_sweep_mg_dL`` spans a clinically meaningful glucose range
    from euglycemic-low through severe hyperglycemia: 60 (mild
    hypoglycemia floor for the GSIR threshold), 80 (sub-basal),
    95 (basal Gb), 120 (post-meal), 150, 200, 250, 300 (OGTT-peak
    range). Insulin held at the cold-model basal Ib=10.

    ``insulin_sweep_uU_mL`` spans basal through peak post-meal insulin:
    5 (sub-basal), 10 (basal), 20, 40 (typical post-meal peak), 80
    (large meal). Glucose held at Gb=95.

    ``species_weights`` controls how much each output species
    contributes to the per-element MSE. Insulin gets the dominant
    weight because it is the focus species (and the one most directly
    constrained by the GSIR cold model). Hepatic output is included
    because the cold model has a clean ``hep_target`` formula and HGO
    is one of the three "downstream" modules called out in the iter 20
    handoff. Other species get small weights — they constrain the
    metabolic MLP's overall shape without dominating the signal.

    ``rank_margin`` is the minimum required slope on monotone GSIR /
    clearance pairs, expressed as a fraction of the cold-target gap.
    Same role as in the gut sweep: forces the kernel to actually
    rank-order the sweep points, which the gut iters 13–15 showed
    is not implied by per-element MSE alone.
    """

    glucose_sweep_mg_dL: tuple[float, ...] = (60.0, 80.0, 95.0, 120.0, 150.0, 200.0, 250.0, 300.0)
    insulin_sweep_uU_mL: tuple[float, ...] = (5.0, 10.0, 20.0, 40.0, 80.0)
    # Per-species MSE weight on the (glucose-sweep) outputs.
    # Order: [glucose, insulin, glucagon, ffa, bhb, lactate, hepatic_output]
    glucose_sweep_species_weights: tuple[float, ...] = (
        0.5,   # dG: residual after Bergman + hepatic + cortisol; informative shape.
        2.0,   # dI: focus species — GSIR curve.
        0.3,   # dGn: counter-regulatory, opposes dI for low G.
        0.1,   # dFFA: insulin antilipolysis, slow.
        0.1,   # dBHB: ketogenesis, slow.
        0.0,   # dLac: not glucose-driven in this sweep.
        0.7,   # dHep: hepatic suppression by glucose (and insulin via I=Ib).
    )
    # Per-species MSE weight on the (insulin-sweep) outputs.
    insulin_sweep_species_weights: tuple[float, ...] = (
        0.0,   # dG: glucose pinned at Gb, residual ≈ 0 in cold model — uninformative.
        2.0,   # dI: clearance shape — focus species.
        0.2,   # dGn: insulin suppression of glucagon.
        0.3,   # dFFA: insulin antilipolysis.
        0.1,   # dBHB: indirect via FFA.
        0.0,   # dLac: not insulin-driven.
        1.0,   # dHep: insulin → hepatic suppression — strong target.
    )
    rank_margin: float = 0.3
    # The metabolic module receives a 3-d "time of day" feature. Pin
    # the sweep to a fixed mid-morning time so day/night circadian
    # state doesn't drift between sweep points.
    sweep_time_of_day_min: float = 480.0  # 08:00


def _cold_metabolic_rates(
    glucose: float,
    insulin: float,
    params: PatientParams,
) -> np.ndarray:
    """Cold-model dX/dt for the seven metabolic species at a given (G, I).

    All other inputs are pinned to a no-meal / no-stress / no-activity
    baseline:

    * other species at their basal/typical values (glycogen pools at ``_b``)
    * gut absorption Ra = 0 (no carbs, fats, or proteins)
    * cortisol at Cort_b (no HPA deviation)
    * GLP-1 at GLP1_b (no incretin from a meal)
    * activity = 0, X = 0 (no insulin-action transient), and the slow
      potentiation state at its equilibrium for this glucose (``Pot = (G-h)+``),
      so the target is the teacher's STEADY-STATE rate at this operating point.

    Iter 97 (teacher hand-off): the glucose and hepatic-output rates come from
    the teacher's own ``full_body.glucose_fluxes`` — the single place its carbon
    budget is written. Until now this function was a private copy of the
    pre-iter-97 teacher (``hep_to_glucose``, ``cort_gluco``, ``cort_hep``,
    ``gn_hep``, ``ins_hep`` — fields that no longer drive the teacher), so the
    sweep was distilling a metabolic model the trajectory signal had stopped
    teaching. The remaining species (I, Gn, FFA, BHB, Lac) mirror the per-step
    block of ``simulate_full_body`` line for line; ``test_insulin_sweep_signal``
    pins the fixed point and the monotonicities that block implies.

    Returns ``[7]`` in the order ``[dG, dI, dGn, dFFA, dBHB, dLac, dHep]``,
    matching the metabolic module's species ordering and ``MARKER_INDEX``
    indexing for the metabolic slice.
    """
    G, I = float(glucose), float(insulin)
    Gn = params.Gnb
    FFA = params.FFA_b
    BHB = params.BHB_b
    Lac = params.Lac_b
    Hep = params.Hep_b
    Cort = params.Cort_b
    LGly = params.LGly_b
    MGly = params.MGly_b
    Ra = 0.0
    act = 0.0
    X = 0.0

    # Glucose ledger + hepatic output target: the teacher's own function.
    fl = glucose_fluxes(params, G, I, X, Gn, Cort, FFA, LGly, MGly, Hep, Ra, act)
    dG = float(fl["dG"])
    dHep = -params.k_hep * (Hep - float(fl["hep_target"]))

    # Insulin: GLP-1 at basal -> incretin factor 1 (the teacher gates on GLP-1
    # ABOVE basal); potentiation at its equilibrium for this glucose.
    incretin_factor = 1.0
    glucose_ratio = min(G / max(params.Gb, 1.0), 1.0)
    effective_Ib = params.Ib * max(glucose_ratio ** params.fast_ins_exp, params.fast_ins_floor)
    g_above = max(G - params.h, 0.0)
    pot = g_above
    w2 = params.ins_phase2_frac
    dI = (-params.n * (I - effective_Ib)
          + params.gamma * ((1.0 - w2) * g_above + w2 * pot) * incretin_factor)

    glucagon_stim = params.alpha_gn * max(params.Gb - G, 0.0) / max(params.Gb, 1.0)
    glucagon_supp = 0.4 * max(I - params.Ib, 0.0) / (params.Ib + 10.0)
    dGn = -params.k_gn * (Gn - params.Gnb) + glucagon_stim - glucagon_supp

    lipolysis = params.lip_max / (1.0 + I / params.IC50_lip)
    dFFA = lipolysis - params.k_ffa * FFA

    # Liver pool at its basal level -> no glycogen-depletion boost on ketogenesis.
    ketogenesis = params.keto_max * FFA / (1.0 + I / params.IC50_keto)
    dBHB = ketogenesis - params.k_bhb * BHB

    dLac = -params.k_lac * (Lac - params.Lac_b)

    return np.array([dG, dI, dGn, dFFA, dBHB, dLac, dHep], dtype=np.float32)


def _build_baseline_state(params: PatientParams) -> np.ndarray:
    """Full STATE_DIM baseline state used as the sweep anchor.

    Markers outside the metabolic slice take ``NORM_CENTER`` (the
    typical values). Metabolic markers take cold-model basal values
    (which are essentially identical to the typicals but are the
    formal reference the cold-model rate function uses internally).
    """
    state = np.array(NORM_CENTER, dtype=np.float32)
    state[MARKER_INDEX["glucose"]] = params.Gb
    state[MARKER_INDEX["insulin"]] = params.Ib
    state[MARKER_INDEX["glucagon"]] = params.Gnb
    state[MARKER_INDEX["ffa"]] = params.FFA_b
    state[MARKER_INDEX["bhb"]] = params.BHB_b
    state[MARKER_INDEX["lactate"]] = params.Lac_b
    state[MARKER_INDEX["hepatic_output"]] = params.Hep_b
    state[MARKER_INDEX["cortisol"]] = params.Cort_b
    state[MARKER_INDEX["glp1"]] = params.GLP1_b
    return state


# Episode.patient_params key -> PatientParams field. These are the keys the
# trajectory record carries (the same dict setpoint supervision reads). A key
# this map requires and the record lacks is a hole in the teacher target, so
# it raises instead of leaving that field at the median person. Keys the map
# does not know are ignored: the teacher may record a parameter this sweep
# does not read.
_RECORD_FIELD: dict[str, str] = {
    "body_mass_kg": "body_mass_kg",
    "si": "Si",
    "glyc_ins_k": "glyc_ins_K",
    "gamma": "gamma",
    "k_ins": "n",
    "act_insulin_sens": "act_insulin_sens",
}


def patient_from_record(raw: dict[str, float]) -> PatientParams:
    """The teacher patient a trajectory record's ``patient_params`` names.

    Starts from ``PatientParams()`` and writes every recorded field, then
    resolves derived quantities the way the teacher does after it samples.
    """
    missing = [key for key in _RECORD_FIELD if key not in raw]
    if missing:
        raise ValueError(
            f"patient_params is missing {missing}; refusing to fill those "
            "fields from the median person"
        )
    params = PatientParams()
    for key, field_name in _RECORD_FIELD.items():
        setattr(params, field_name, float(raw[key]))
    return resolve_derived_params(params)


@dataclass
class InsulinSweepSignal(TrainingSignal):
    """Direct distillation of metabolic-module rates against cold-model curves.

    Pre-computes cold targets at construction (closed-form, fast). Per
    epoch: stack the supervised embeddings into one batched call to
    ``model.metabolic`` for each sweep point, compute mse / rank / auc losses,
    backward + step.

    The zero embedding is the median person and is scored against
    ``PatientParams()``. A sampled row is scored against that patient's own
    teacher ``PatientParams`` (PLAN B4). A6 had stopped sampling because every
    row, sampled or not, was scored against the median person's seven rates, so
    ``sample_patients = 4`` pulled each patient's secretion and clearance onto
    the median from epoch 0. The trajectory record's ``patient_params`` is the
    ground truth; a sampled id with no entry raises rather than falling back
    to the median target.
    """

    n_patients: int = 0
    # 0: zero embedding only, against PatientParams(). Above 0, each sampled
    # row is scored against patient_params[pid].
    sample_patients: int = 0
    # pid -> the teacher patient that row was simulated from. Empty is correct
    # when nothing is sampled. A sampled pid absent from this map is an error.
    patient_params: dict[int, PatientParams] = field(default_factory=dict)
    include_default_embedding: bool = True
    weight: WeightSchedule = field(default_factory=lambda: WeightSchedule(0.0))
    protocol: InsulinSweepProtocol = field(default_factory=InsulinSweepProtocol)
    ranking_weight: float = 1.0
    auc_weight: float = 1.0

    name: str = "insulin_sweep"
    source: str = "cold_model metabolic rates (Bergman GSIR + clearance)"
    category: str = "metabolic"

    def __post_init__(self) -> None:
        # The zero row. A sampled row builds the same curves from its own
        # teacher patient the first time it is drawn; Gb, Ib and the basal
        # state come from that patient, not from the median.
        g_t, i_t, g_s, i_s = self._curves_for_params(PatientParams())
        self._g_targets = g_t
        self._i_targets = i_t
        self._g_states = g_s
        self._i_states = i_s
        # Built on first use. A run that does not sample pays nothing, and a
        # sampled patient is built once.
        self._patient_curves: dict[
            int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}

        # Per-species weights and a normalization scale (NORM_SCALE for
        # the cold-modelled metabolic species, in the order the module
        # emits). Internal slow states (iter 55+) are sliced off — they
        # have no cold-model rate target.
        met_idx = MODULE_MARKER_INDICES["metabolic"][:_N_COLD_METABOLIC]
        norm_scale = np.array([NORM_SCALE[i] for i in met_idx], dtype=np.float32)
        self._met_norm_scale = torch.tensor(norm_scale, dtype=torch.float32)
        self._g_species_w = torch.tensor(
            self.protocol.glucose_sweep_species_weights, dtype=torch.float32,
        )
        self._i_species_w = torch.tensor(
            self.protocol.insulin_sweep_species_weights, dtype=torch.float32,
        )
        if self._g_species_w.shape != (7,) or self._i_species_w.shape != (7,):
            raise ValueError("species_weights must have length 7")

    def _curves_for_params(
        self, params: PatientParams,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cold targets and sweep states for one patient.

        Returns ``(g_targets [Ng, 7], i_targets [Ni, 7], g_states [Ng, STATE],
        i_states [Ni, STATE])``. The glucose sweep holds insulin at this
        patient's Ib; the insulin sweep holds glucose at this patient's Gb.
        """
        g_targets = np.stack([
            _cold_metabolic_rates(g, params.Ib, params) for g in self.protocol.glucose_sweep_mg_dL
        ])
        i_targets = np.stack([
            _cold_metabolic_rates(params.Gb, i, params) for i in self.protocol.insulin_sweep_uU_mL
        ])
        baseline = _build_baseline_state(params)
        gi = MARKER_INDEX["glucose"]
        ii = MARKER_INDEX["insulin"]
        g_states = np.tile(baseline, (len(self.protocol.glucose_sweep_mg_dL), 1))
        for k, g in enumerate(self.protocol.glucose_sweep_mg_dL):
            g_states[k, gi] = float(g)
        i_states = np.tile(baseline, (len(self.protocol.insulin_sweep_uU_mL), 1))
        for k, ival in enumerate(self.protocol.insulin_sweep_uU_mL):
            i_states[k, ii] = float(ival)
        return (
            torch.tensor(g_targets, dtype=torch.float32),
            torch.tensor(i_targets, dtype=torch.float32),
            torch.tensor(g_states, dtype=torch.float32),
            torch.tensor(i_states, dtype=torch.float32),
        )

    def _curves_for_row(
        self, pid: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Curves for one supervised row. ``None`` is the zero embedding."""
        if pid is None:
            return self._g_targets, self._i_targets, self._g_states, self._i_states
        params = self.patient_params.get(pid)
        if params is None:
            raise RuntimeError(
                f"insulin sweep sampled patient {pid} with no patient_params; "
                "refusing to score that row against the median person"
            )
        cached = self._patient_curves.get(pid)
        if cached is None:
            cached = self._curves_for_params(params)
            self._patient_curves[pid] = cached
        return cached

    def _batch(
        self, pids: list[int | None],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stack per-row curves on a new batch axis: ``[N, B, ...]``."""
        curves = [self._curves_for_row(pid) for pid in pids]
        return tuple(torch.stack([c[i] for c in curves], dim=1) for i in range(4))  # type: ignore[return-value]

    def weight_at(self, epoch: int) -> float:
        return self.weight.at(epoch)

    def _sweep_rates(
        self,
        net: ModularPhysiologyNetwork,
        sweep_states: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Run ``model.metabolic`` once over every (sweep point, embedding) pair.

        ``sweep_states`` is ``[N, STATE_DIM]`` (one operating point, broadcast
        across the batch) or ``[N, B, STATE_DIM]`` (each row its own patient).
        ``embeddings`` is ``[B, EMB]``. Returns ``[N, B, 7]``: rates for the
        7 metabolic species at each (sweep-point, embedding) pair, with no
        meal, no stress, no activity. Mirrors the input construction in
        ``ModularPhysiologyNetwork.forward`` so the supervision sees
        exactly what the integrator sees at this operating point.
        """
        device = sweep_states.device
        B = int(embeddings.shape[0])

        met_idx = list(MODULE_MARKER_INDICES["metabolic"])
        # Project embedding once per metabolic call — one [B, EMB_MET] tensor.
        emb_met = net.embedding_projections["metabolic"](embeddings)

        # No-meal gut output and no-activity / awake external inputs are
        # constant across the sweep; build them once.
        zero_gut = torch.zeros(B, GUT_OUTPUT_DIM, dtype=torch.float32, device=device)
        zero_external = torch.zeros(B, EXTERNAL_INPUT_DIM, dtype=torch.float32, device=device)
        # Resting / awake: external = [activity=0, sleep_wake=1] in the
        # order ``ModularPhysiologyNetwork.forward`` uses for metabolic.
        zero_external[:, 1] = 1.0

        # Time features at the sweep-fixed time of day.
        t_minutes = torch.tensor(
            [self.protocol.sweep_time_of_day_min], dtype=torch.float32, device=device,
        )
        time_feats = compute_time_features(t_minutes).expand(B, -1)

        # Every (sweep point, embedding) pair in one call: row n·B + b.
        N = int(sweep_states.shape[0])
        if sweep_states.dim() == 2:
            full_state = sweep_states.unsqueeze(1).expand(N, B, -1).reshape(N * B, -1)
        else:
            if int(sweep_states.shape[1]) != B:
                raise ValueError(
                    f"sweep states have batch {int(sweep_states.shape[1])}, embeddings have {B}"
                )
            full_state = sweep_states.reshape(N * B, -1)
        norm_state = (full_state - net.norm_center) / net.norm_scale
        coupling = net.coupling_for(
            "metabolic", norm_state, zero_gut.repeat(N, 1),
            torch.zeros(N * B, DUODENAL_DIM, dtype=torch.float32, device=device),
        )
        rates = net.metabolic(
            norm_state[:, met_idx], coupling, zero_external.repeat(N, 1),
            emb_met.repeat(N, 1), time_feats.repeat(N, 1),
        )
        # Drop internal slow-state rate outputs (iter 55+): no cold target.
        return rates.reshape(N, B, -1)[..., :_N_COLD_METABOLIC]  # [N, B, 7]

    def _sweep_losses(
        self,
        pred: torch.Tensor,           # [N, B, 7]
        target: torch.Tensor,         # [N, B, 7]
        species_w: torch.Tensor,      # [7]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """MSE / rank / AUC losses for one sweep axis.

        ``target`` is per row. Rank hinges are that row's own cold-model dI
        gaps, so a patient whose GSIR curve is steeper is not judged against
        the median person's gaps.

        * ``mse`` — per-element squared error normalized by NORM_SCALE,
          weighted per species, then averaged.
        * ``rank`` — pairwise hinge on dI: violations counted only on
          pairs where that row's cold-model dI gap is positive.
        * ``auc`` — sum-over-grid undiluted amplitude term per species,
          normalized by NORM_SCALE × N (the typical "AUC magnitude"
          under the integrated rate). Plays the same dilution-defeating
          role as the time-integral term in the gut sweep.
        """
        device = pred.device
        norm = self._met_norm_scale.to(device)            # [7]
        species_w = species_w.to(device)                   # [7]
        target = target.to(device)

        err = (pred - target) / norm                       # [N, B, 7]
        per_species = (err.pow(2)).mean(dim=(0, 1))        # [7]
        denom = species_w.sum().clamp_min(1e-8)
        mse_loss = (per_species * species_w).sum() / denom

        # Sum across grid for AUC term, [B, 7]
        pred_sums = pred.sum(dim=0)
        target_sums = target.sum(dim=0)                    # [B, 7]
        auc_norm = norm * float(int(pred.shape[0]))
        auc_err = (pred_sums - target_sums) / auc_norm     # [B, 7]
        per_species_auc = (auc_err.pow(2)).mean(dim=0)     # [7]
        auc_loss = (per_species_auc * species_w).sum() / denom

        # Rank hinge on dI only, against each row's own cold curve.
        pred_dI = pred[..., _MET_LOCAL_INSULIN]            # [N, B]
        tgt_dI = target[..., _MET_LOCAL_INSULIN]           # [N, B]
        pred_gap = pred_dI.unsqueeze(0) - pred_dI.unsqueeze(1)  # [N, N, B]
        rank_gap = tgt_dI.unsqueeze(0) - tgt_dI.unsqueeze(1)    # [N, N, B]
        rank_mask = (rank_gap > 0).to(torch.float32)
        margin_required = self.protocol.rank_margin * rank_gap
        norm_dI = norm[_MET_LOCAL_INSULIN]
        violation = torch.relu(margin_required - pred_gap) / norm_dI
        violation = violation * rank_mask
        n_active = rank_mask.sum()
        rank_loss = (
            violation.sum() / n_active if float(n_active) > 0
            else torch.zeros((), device=device)
        )

        return mse_loss, rank_loss, auc_loss

    def compute(
        self,
        model: nn.Module,
        embeddings: nn.Embedding,
        ctx: SignalContext,
    ) -> SignalResult:
        w = self.weight_at(ctx.epoch)
        if w <= 0:
            return SignalResult()

        rows = select_supervised_rows(
            embeddings=embeddings,
            n_patients=self.n_patients,
            sample_patients=self.sample_patients,
            rng=ctx.rng,
            device=ctx.device,
            include_default=self.include_default_embedding,
        )
        if not rows:
            return SignalResult()

        device = ctx.device
        net = cast(ModularPhysiologyNetwork, model)
        pids = [pid for pid, _emb in rows]
        embs = torch.stack([emb for _pid, emb in rows], dim=0)  # [B, EMB]
        g_target, i_target, g_states, i_states = self._batch(pids)

        g_pred = self._sweep_rates(net, g_states.to(device), embs)  # [Ng, B, 7]
        i_pred = self._sweep_rates(net, i_states.to(device), embs)  # [Ni, B, 7]

        g_mse, g_rank, g_auc = self._sweep_losses(
            g_pred, g_target.to(device), self._g_species_w,
        )
        i_mse, i_rank, i_auc = self._sweep_losses(
            i_pred, i_target.to(device), self._i_species_w,
        )

        mse_loss = 0.5 * (g_mse + i_mse)
        rank_loss = 0.5 * (g_rank + i_rank)
        auc_loss = 0.5 * (g_auc + i_auc)
        loss = mse_loss + self.ranking_weight * rank_loss + self.auc_weight * auc_loss

        n_pairs = int(embs.shape[0]) * (
            len(self.protocol.glucose_sweep_mg_dL) + len(self.protocol.insulin_sweep_uU_mL)
        )

        accumulate_grad(
            w * loss,
            ctx,
            signal=self.name,
            extra={
                "raw_loss": float(loss.detach().item()),
                "weight": float(w),
                "mse": float(mse_loss.detach().item()),
                "rank": float(rank_loss.detach().item()),
                "auc": float(auc_loss.detach().item()),
                "g_mse": float(g_mse.detach().item()),
                "g_auc": float(g_auc.detach().item()),
                "g_rank": float(g_rank.detach().item()),
                "i_mse": float(i_mse.detach().item()),
                "i_auc": float(i_auc.detach().item()),
                "i_rank": float(i_rank.detach().item()),
                "n_grid_emb_pairs": float(n_pairs),
            },
        )

        return SignalResult(
            loss_sum=float(loss.detach().item()),
            n_units=1,
            sub_metrics={
                "n_grid_emb_pairs": float(n_pairs),
                "mse": float(mse_loss.detach().item()),
                "rank": float(rank_loss.detach().item()),
                "auc": float(auc_loss.detach().item()),
                "g_mse": float(g_mse.detach().item()),
                "g_auc": float(g_auc.detach().item()),
                "g_rank": float(g_rank.detach().item()),
                "i_mse": float(i_mse.detach().item()),
                "i_auc": float(i_auc.detach().item()),
                "i_rank": float(i_rank.detach().item()),
            },
        )
