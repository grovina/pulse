"""
Hepatobiliary module — the enterohepatic circulation of bile acids.

Four states in series, mirroring the teacher (`knowledge/full_body.py`, iter 95):

    meal lipid/protein ──> [cck] ──gate──> [gallbladder_bile] ──> [intestinal_bile]
                                  ^                                       │
                          canalicular export                   ileal reabsorption ~95%
                                  │                                       v
                            hepatocyte <──────── portal return ───────────┘
                                  │
                         first-pass extraction ──> spillover ──> [bile_acids]

Sourced anchors and the design rationale: `docs/iter95-biliary-anchors.md`. In short:
the axis is entered at the MEDIATOR (bile acids) rather than at the damage readouts
(ALP/GGT/ALT/bilirubin), because those have no driver in this state vector and would
ship as constants — and the iter-94 ruler finding is that the gate cannot tell a
constant from a simulator. Cholestasis IS failure of canalicular export, so modelling
that step explicitly means the enzymes later hang off impairment of a step that
already exists.

WHICH STATES GET WHICH SHAPE. The iter-95 lesson (see `modules/base.py`) is that the
shape must match what the state physically IS:

  cck               a secreted, fast-cleared hormone — TRUE mass action (learned basal
                    production, learned clearance × concentration) plus a STRUCTURAL
                    meal-driven secretion term.
  gallbladder_bile  a POOL. Explicit flux balance with a headroom bracket, filled by
                    HEPATIC EXPORT (portal return + synthesis), emptied by a CCK gate.
  intestinal_bile   a transit compartment. Receives what the gallbladder ejects plus
                    the export that bypasses a full/contracting gallbladder; drains at
                    the ileal-uptake rate. ConstantFluxHead — pure transport.
  bile_acids        serum concentration. Spillover of unextracted portal return is the
                    source; clearance returns to hepatic secretion. ConstantFluxHead.
                    Lowering canalicular extraction raises serum (cholestasis).

Portal return is partitioned: extraction·portal is re-secreted, (1−extraction)·portal
spills to serum, serum clearance returns to bile. Conservation:
d(GB + INT) + dBA/spill_gain = synthesis − faecal loss.

ITER 97 — TWO FIXES FROM THE 2026-09-04 REVIEW (items 3.6 and 2.5):

  1. cck and bile_acids relaxed toward GLOBAL constants with a non-negative source
     (`− k·(x − typical) + source`), so their equilibrium was ≥ typical ALWAYS: cck
     min = 1.000 and bile_acids min = 6.000 exactly in every fast and fed run, while
     the fasting reference is 4.4-14.1 µmol/L. This was the iter-95 "absorbing floor
     at typical" reproduced in the module that was built to fix it. Both are now
     `prod·prod_scale + source − cons·cons_scale·raw`: the head's basal production is
     per patient (it reads the embedding), so a below-typical fasting level is
     representable, and the relaxation rate is the learned clearance.
  2. The loop was not closed: `gb_fill` came from nothing (`k_fill·capacity·headroom`)
     and the 95 % portal return fed only the serum readout. Hepatic export is now
     extracted portal + returned serum clearance + synthesis. Conservation of the
     mmol pool holds by construction:
     d(gallbladder + intestine)/dt + dBA/spill_gain = synthesis − faecal loss.

So the two heads whose outputs were unreachable by any loss (intestinal_bile and
bile_acids) are ConstantFluxHead.
"""

import math

import torch
import torch.nn as nn

from .base import ConstantFluxHead, MassActionModule, SpeciesHead
from ..types import DUODENAL_DIM, MODULE_COUPLING_CHANNELS, MODULE_MARKER_INDICES, NORM_CENTER, NORM_SCALE

# Species order matches MODULE_MARKER_INDICES["hepatobiliary"]:
#   0: cck, 1: gallbladder_bile, 2: intestinal_bile, 3: bile_acids
_CCK_IDX = 0
_GB_IDX = 1
_INT_IDX = 2
_BA_IDX = 3

_TYPICALS = [NORM_CENTER[i] for i in MODULE_MARKER_INDICES["hepatobiliary"]]
_NORM_SCALES = [NORM_SCALE[i] for i in MODULE_MARKER_INDICES["hepatobiliary"]]
_CONS_SCALES = [0.35, 0.02, 0.02, 0.03]

# Coupling: duodenal fat, protein (carbohydrate is delivered but I-cells ignore it).
_N_COUPLING = len(MODULE_COUPLING_CHANNELS["hepatobiliary"])
_FAT_DUO_IDX = 0
_PROT_DUO_IDX = 1

# External inputs: activity (1) + sleep_wake (1) = 2, matching every other module.
_N_EXTERNAL = 2

# --- structural constants, all mirroring the teacher's calibrated values -----------
# Gallbladder capacity as a multiple of typical. The store fills between meals — this
# is why the FIRST meal of the day gives the largest bile-acid excursion.
_GB_CAPACITY_FRAC = 1.5
# Bounded learned rates. Bands are wide enough to contain the teacher's value with
# margin; they exist because these are physical rate constants with known units, not
# to force any particular behaviour.
_K_EJECT_MIN, _K_EJECT_RANGE, _K_EJECT_INIT = 0.005, 0.075, 0.030   # /min, teacher 0.030
_K_ILEAL_MIN, _K_ILEAL_RANGE, _K_ILEAL_INIT = 0.004, 0.036, 0.013   # /min, teacher 0.013
# Half-maximal CCK excess (pmol/L above basal) for gallbladder contraction.
_K_CCK_GB = 2.5          # teacher K_cck_gb
_F_ILEAL = 0.95          # ileal reabsorption efficiency; the 5% remainder is faecal loss
# Hepatic de-novo synthesis scale (mmol/min per unit of the head's learned gain). At the
# init gain (softplus(0) ≈ 0.69) this is 0.7e-3 mmol/min, which at typical intestinal
# content (1 mmol, k_ileal 0.013) replaces the faecal loss (0.05·0.013·1 = 0.65e-3), so
# the loop is stationary at typical from the start.
_SYNTH_SCALE = 1.0e-3
_K_BASAL_MIN, _K_BASAL_RANGE, _K_BASAL_INIT = 0.0004, 0.008, 0.0026
_K_MMC_FED = 0.05
_HEP_EXTRACTION = 0.90
_CANALICULAR_OFFSET = 0.15
# µmol/L per mmol/min of unextracted portal return. Teacher derives this so BA_b
# is the mass-action fixed point; 145.7 is that derived default.
_BA_SPILL_GAIN_INIT = 145.7
_CCK_LOG_MAX = 0.7  # CCK_b = 1 · exp(±0.7 tanh) ∈ [0.50, 2.01]


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


class DuodenalDeliveryKernel(nn.Module):
    """Nutrient delivery INTO THE DUODENUM (g/min) — gastric emptying.

    This is NOT the gut module's appearance output. Those channels are SYSTEMIC
    appearance: fat reaches the blood via chylomicrons and peaks around 60 min,
    whereas duodenal I-cells see nutrient arriving within minutes of the meal.
    Driving CCK from systemic appearance put the TEACHER's modelled CCK peak at
    +68 min against a literature +10, and no gain could have fixed it — it was the
    wrong signal, not the wrong size (measured; see the iter-95 commit). The student
    would inherit exactly the same error from the same channel, so it gets the same
    separate kernel the teacher does.

    Two components, matching `full_body._duodenal_delivery` and the fast/slow split
    the gut module already uses for carbohydrate:

      fast  gamma-shaped, peaking at 1/fast_rate ≈ 10 min — rapid liquid-phase
            emptying, which produces the sharp early CCK peak.
      slow  exponential, τ ≈ 200 min — solid-phase emptying, which is what keeps CCK
            elevated for the observed 3-5 h.

    A single component reproduces the peak or the plateau but not both (measured on
    the teacher across a 16-point sweep).

    Deliberately NOT per-patient: it takes no embedding, so it is cheap, batch-
    independent, and needs no precompute path in `model.py`. Gastric emptying does
    vary between people, and that is a later question — making it per-patient now
    would add a third unsupervised embedding→physiology map, which is the thing
    iter 90 found goes wrong when nothing supervises it.
    """

    def __init__(self) -> None:
        super().__init__()
        # Bounded so the fast component's peak time stays in a physiological window:
        # fast_rate ∈ (0.02, 0.22) → peak at 1/rate ∈ (4.5, 50) min.
        self.log_fast_rate = nn.Parameter(torch.tensor(_logit((0.10 - 0.02) / 0.20)))
        self.log_slow_rate = nn.Parameter(torch.tensor(_logit((0.005 - 0.001) / 0.019)))
        self.logit_slow_frac = nn.Parameter(torch.tensor(_logit(0.35)))

    def rates(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fast = 0.02 + 0.20 * torch.sigmoid(self.log_fast_rate)
        slow = 0.001 + 0.019 * torch.sigmoid(self.log_slow_rate)
        frac = torch.sigmoid(self.logit_slow_frac)
        return fast, slow, frac

    def forward_window(self, times: torch.Tensor, meals: list) -> torch.Tensor:
        """Delivery at every time in ``times`` at once -> ``[T, 3]`` (fat, protein, carb).

        ``times`` are WINDOW-OFFSET minutes, the same frame as ``meal.time``.
        No hard age mask: both components decay smoothly (gamma-2 and exponential),
        so a late dinner cannot cliff CCK the way the old 480-min step did.
        """
        device = self.log_fast_rate.device
        T = int(times.shape[0])
        if not meals:
            return torch.zeros(T, DUODENAL_DIM, dtype=torch.float32, device=device)
        fast_rate, slow_rate, slow_frac = self.rates()
        meal_times = torch.tensor([m.time for m in meals], dtype=torch.float32,
                                  device=device)
        macros = torch.tensor(
            [(m.fats, m.proteins, m.carbs) for m in meals],
            dtype=torch.float32, device=device)
        dt = times.to(device).unsqueeze(1) - meal_times.unsqueeze(0)
        active = (dt >= 0.0).to(torch.float32)
        dt = dt.clamp(min=0.0)
        fast = fast_rate * fast_rate * dt * torch.exp(-fast_rate * dt)
        slow = slow_rate * torch.exp(-slow_rate * dt)
        shape = ((1.0 - slow_frac) * fast + slow_frac * slow) * active
        return shape @ macros

    def forward(self, t_minutes: torch.Tensor, meals: list) -> torch.Tensor:
        """Return ``[fat, protein, carb]`` delivery in g/min at ``t_minutes``."""
        device = self.log_fast_rate.device
        out = torch.zeros(DUODENAL_DIM, dtype=torch.float32, device=device)
        if not meals:
            return out
        fast_rate, slow_rate, slow_frac = self.rates()
        t = float(t_minutes) if not torch.is_tensor(t_minutes) else float(t_minutes.item())
        for m in meals:
            dt = t - m.time
            if dt < 0.0:
                continue
            dt_t = torch.as_tensor(dt, dtype=torch.float32, device=device)
            fast = fast_rate * fast_rate * dt_t * torch.exp(-fast_rate * dt_t)
            slow = slow_rate * torch.exp(-slow_rate * dt_t)
            shape = (1.0 - slow_frac) * fast + slow_frac * slow
            out = out + shape * torch.stack([
                torch.as_tensor(float(m.fats), dtype=torch.float32, device=device),
                torch.as_tensor(float(m.proteins), dtype=torch.float32, device=device),
                torch.as_tensor(float(m.carbs), dtype=torch.float32, device=device),
            ])
        return out


class HepatobiliaryModule(MassActionModule):
    """The enterohepatic loop. Rates are structural; heads supply learned gains.

    Head roles (the ``(prod, cons)`` protocol of ``MassActionModule``):

      cck               prod = basal secretion, cons = clearance         (mass action)
      gallbladder_bile  prod = ejection gain,   cons = hepatic synthesis gain
      intestinal_bile   no head (ConstantFluxHead) — pure transport
      bile_acids        no head (ConstantFluxHead) — spillover minus clearance
    """

    def __init__(self, embedding_dim: int, hidden_dim: int = 32):
        super().__init__(
            n_species=4,
            n_coupling=_N_COUPLING,
            n_external=_N_EXTERNAL,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            typicals=_TYPICALS,
            norm_scales=_NORM_SCALES,
            head_factories={
                _CCK_IDX: SpeciesHead,
                _GB_IDX: SpeciesHead,
                _INT_IDX: lambda inp, hd: ConstantFluxHead(),
                _BA_IDX: lambda inp, hd: ConstantFluxHead(),
            },
        )
        prod_scales = [c * t for c, t in zip(_CONS_SCALES, _TYPICALS)]
        self.prod_scale.copy_(torch.tensor(prod_scales, dtype=torch.float32))
        self.cons_scale.copy_(torch.tensor(_CONS_SCALES, dtype=torch.float32))

        # CCK secretion gains, per (g/min) of duodenal delivery. Structural rather than
        # buried in the head's MLP so meal→CCK amplitude stays reachable by gradient —
        # the same reason glucose carries an explicit Ra (PRD, mass-action relaxations).
        self.log_cck_fat_gain = nn.Parameter(torch.tensor(math.log(2.25)))
        self.log_cck_prot_gain = nn.Parameter(torch.tensor(math.log(0.79)))
        self.log_k_eject = nn.Parameter(
            torch.tensor(_logit((_K_EJECT_INIT - _K_EJECT_MIN) / _K_EJECT_RANGE)))
        self.log_k_ileal = nn.Parameter(
            torch.tensor(_logit((_K_ILEAL_INIT - _K_ILEAL_MIN) / _K_ILEAL_RANGE)))
        self.log_ba_spill_gain = nn.Parameter(
            torch.tensor(math.log(_BA_SPILL_GAIN_INIT)))
        self.log_k_ba = nn.Parameter(torch.tensor(math.log(0.03)))
        self.log_k_canalicular = nn.Parameter(torch.tensor(0.0))  # exp(0) = 1, healthy
        self.log_k_gb_basal = nn.Parameter(
            torch.tensor(_logit((_K_BASAL_INIT - _K_BASAL_MIN) / _K_BASAL_RANGE)))
        _bh = max(8, hidden_dim // 4)
        self.cck_baseline_net = nn.Sequential(
            nn.Linear(embedding_dim, _bh), nn.Tanh(), nn.Linear(_bh, 1),
        )
        with torch.no_grad():
            self.cck_baseline_net[-1].weight.zero_()
            self.cck_baseline_net[-1].bias.zero_()

    def cck_setpoint_raw(self, embedding: torch.Tensor) -> torch.Tensor:
        return _TYPICALS[_CCK_IDX] * torch.exp(
            _CCK_LOG_MAX * torch.tanh(self.cck_baseline_net(embedding).squeeze(-1)))

    def fluxes(
        self,
        state: torch.Tensor,
        coupling: torch.Tensor,
        external: torch.Tensor,
        embedding: torch.Tensor,
        time_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Every named flux of the loop (mmol/min unless noted). ``forward`` assembles
        the rates from these; tests and probes read them directly."""
        prod_raw, cons_raw = self.species_fluxes(
            state, coupling, external, embedding, time_features)
        raw = self.raw_state(state)
        relu = nn.functional.relu
        cck_raw = raw[..., _CCK_IDX]
        gb_raw = raw[..., _GB_IDX]
        int_raw = raw[..., _INT_IDX]
        ba_raw = raw[..., _BA_IDX]
        fat_duo = coupling[..., _FAT_DUO_IDX]
        prot_duo = coupling[..., _PROT_DUO_IDX]
        duo_total = fat_duo + prot_duo
        cck_b = self.cck_setpoint_raw(embedding)

        cck_secretion = (torch.exp(self.log_cck_fat_gain) * fat_duo
                         + torch.exp(self.log_cck_prot_gain) * prot_duo)
        cck_basal = prod_raw[..., _CCK_IDX] * self.prod_scale[_CCK_IDX]
        cck_clearance = cons_raw[..., _CCK_IDX] * self.cons_scale[_CCK_IDX] * cck_raw

        cck_excess = relu(cck_raw - cck_b)
        contraction = cck_excess / (cck_excess + _K_CCK_GB)
        k_eject = _K_EJECT_MIN + _K_EJECT_RANGE * torch.sigmoid(self.log_k_eject)
        k_basal = _K_BASAL_MIN + _K_BASAL_RANGE * torch.sigmoid(self.log_k_gb_basal)
        fed_gate = duo_total / (duo_total + _K_MMC_FED)
        gb_empty = (
            k_eject * contraction * gb_raw * prod_raw[..., _GB_IDX]
            + k_basal * (1.0 - fed_gate) * gb_raw
        )

        k_ileal = _K_ILEAL_MIN + _K_ILEAL_RANGE * torch.sigmoid(self.log_k_ileal)
        ileal_uptake = k_ileal * int_raw
        portal_return = _F_ILEAL * ileal_uptake
        fecal_loss = ileal_uptake - portal_return

        k_can = torch.exp(self.log_k_canalicular)
        extraction = (_HEP_EXTRACTION * (k_can / (k_can + _CANALICULAR_OFFSET))
                      / (1.0 / (1.0 + _CANALICULAR_OFFSET)))
        extraction = extraction.clamp(0.0, 0.995)
        spillover = (1.0 - extraction) * portal_return
        extracted = extraction * portal_return

        spill_gain = torch.exp(self.log_ba_spill_gain)
        ba_clearance = torch.exp(self.log_k_ba) * ba_raw
        serum_return = ba_clearance / spill_gain
        ba_spill = spill_gain * spillover
        synthesis = _SYNTH_SCALE * cons_raw[..., _GB_IDX]
        export = extracted + serum_return + synthesis
        capacity = _GB_CAPACITY_FRAC * _TYPICALS[_GB_IDX]
        headroom = relu(1.0 - gb_raw / capacity)
        to_gallbladder = export * headroom * (1.0 - contraction)
        to_intestine = export - to_gallbladder

        return {
            "cck_secretion": cck_secretion, "cck_basal": cck_basal,
            "cck_clearance": cck_clearance, "contraction": contraction,
            "cck_b": cck_b, "gb_empty": gb_empty, "ileal_uptake": ileal_uptake,
            "portal_return": portal_return, "fecal_loss": fecal_loss,
            "extraction": extraction, "spillover": spillover, "extracted": extracted,
            "synthesis": synthesis, "export": export, "serum_return": serum_return,
            "to_gallbladder": to_gallbladder, "to_intestine": to_intestine,
            "ba_spill": ba_spill, "ba_clearance": ba_clearance, "spill_gain": spill_gain,
        }

    def forward(
        self,
        state: torch.Tensor,
        coupling: torch.Tensor,
        external: torch.Tensor,
        embedding: torch.Tensor,
        time_features: torch.Tensor,
    ) -> torch.Tensor:
        f = self.fluxes(state, coupling, external, embedding, time_features)
        rates = torch.zeros_like(state)
        rates[..., _CCK_IDX] = f["cck_basal"] + f["cck_secretion"] - f["cck_clearance"]
        rates[..., _GB_IDX] = f["to_gallbladder"] - f["gb_empty"]
        rates[..., _INT_IDX] = f["gb_empty"] + f["to_intestine"] - f["ileal_uptake"]
        rates[..., _BA_IDX] = f["ba_spill"] - f["ba_clearance"]
        return rates
