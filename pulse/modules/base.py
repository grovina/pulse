"""
Base module classes for the modular physiology architecture.

MassActionModule enforces rate = production - consumption × concentration
for chemical species. This is fundamental chemistry.

LearnedDynamicsModule has fully learned rates for vital signs where
no single correct equation exists.

Both module types receive:
  - Own state (the markers this module owns)
  - Coupling inputs (named values from other modules)
  - External inputs (sleep/wake, activity — with learned defaults for missing)
  - Module-specific embedding (projected from the global person embedding)
  - Time features
"""

import math
from typing import Callable, Optional

import torch
import torch.nn as nn

from ..types import MG_DL_PER_G, TIME_FEATURES_DIM

# Floor on every sigmoid-gate temperature (the ``/ exp(log_temp)`` divisor).
# The gate gradient w.r.t. its inputs scales as 1/temp and the gradient w.r.t.
# log_temp scales as exp(-log_temp); if the optimizer drives the temperature
# toward 0 (a sharp gate) both blow up to Inf -> NaN. Iter 76 hit exactly this:
# once glycogen became a real (non-constant) trajectory target, the previously
# ~ungradiented GlycogenFluxHead gates got a strong signal and collapsed their
# temperature at phase-1 epoch 31 (finite loss, NaN grad on 105/151 params).
# All gate temperatures sit at exp(-1.6)..exp(-0.7) = 0.20..0.50 at init and in
# every stable iter, so a 0.05 floor is a pure safety rail — no behavioural
# change in the normal regime — that caps the gate gradient at 1/0.05 = 20.
MIN_GATE_TEMP = 0.05


def gate_temp(log_temp: torch.Tensor) -> torch.Tensor:
    """``exp(log_temp)`` floored at ``MIN_GATE_TEMP`` (see above)."""
    return torch.exp(log_temp).clamp(min=MIN_GATE_TEMP)


class SpeciesHead(nn.Module):
    """Per-species production/consumption head.

    Reads the full module input (state, coupling, external, embedding,
    time) and emits two scalars: ``raw_prod`` and ``raw_cons``. The
    parent ``MassActionModule`` applies the softplus + per-species
    scale and turns them into ``prod - cons * state_i``.

    Each species owns one of these — that is the iter-23 structural
    fix for representation collapse. With the iter-21/22 shared MLP,
    every species fed gradients into a single trunk, so the trunk
    became insulin-shaped (insulin had the dominant trajectory signal)
    and the small-magnitude species (GLP-1, glucagon, BHB, FFA, HGO,
    ghrelin) settled at near-zero outputs on a flat manifold where
    ∂pred/∂params ≈ 0 — gradient-budget interventions could not
    rescue them. Per-species heads give every species its own
    representation, so collapsing one no longer requires the others
    to also collapse, and gradient pressure on a small species lands
    on parameters that actually move its prediction.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor, state_self: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.network(x)
        prod = nn.functional.softplus(raw[..., 0:1]).squeeze(-1)
        cons = nn.functional.softplus(raw[..., 1:2]).squeeze(-1)
        return prod, cons


class ConstantFluxHead(nn.Module):
    """A parameter-free head that emits ``(1, 1)``.

    Iter 97. Several species have STRUCTURAL rates assembled in their module's
    ``forward`` (glucose, insulin_action, intestinal_bile) and never read their
    head's output. Through iter 96 those species still constructed a full
    ``SpeciesHead`` — 4,514 parameters each at hidden 48 — that no loss could
    reach: 12,555 of 75,914 parameters (16.5 %) had no gradient path (review
    2026-09-04, end of section 3). A parameter that cannot move is not
    "learned"; it is dead weight in every optimizer step and every checkpoint.
    This head takes the species' slot in the ``(prod, cons)`` protocol and costs
    nothing.
    """

    def forward(self, x: torch.Tensor, state_self: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ones = torch.ones_like(state_self)
        return ones, ones


# ITER 95 — `SetpointHead` REMOVED (was here, iters 51-94).
#
# It re-parameterised (prod, cons) as (target_z, k_factor) to escape a softplus
# saturation trap: in the old DEVIATION frame, sitting at `typical` required
# prod = 0, i.e. prod_raw -> -inf, where d(softplus)/d(prod_raw) vanishes and the
# head's gradient died. That trap was an artifact of the frame, not of physiology
# — see MassActionModule below. In the corrected concentration frame, sitting at
# typical requires prod == cons (two moderate positive values) and any positive
# equilibrium is reachable, so a plain SpeciesHead has live gradients everywhere
# and the workaround has nothing left to work around.
#
# It also had a failure mode of its own: `k_factor = softplus(raw)` could collapse
# to zero, which freezes the species AND zeroes the gradient onto target_z, since
# the rate is k_factor*(target_z - norm_state). Measured on the iter-94 artifact,
# bhb sat at k_factor = 1e-5 (tau ~ 265 days) with target_z = +2.64 — the head
# asking for 0.23 mmol/L with no authority to get there. Removing the head removes
# that mode; nothing replaces it.

class BasalPlusGatedPeakHead(nn.Module):
    """Basal + stimulus-gated peak head for stimulus-driven hormones.

    Generalises iter-23's ``GlucoseGatedInsulinHead`` into a reusable
    structural primitive. The motivation (iter 53) is that
    ``SetpointHead`` (iter 51) cleanly fits the *equilibrium near typical*
    but cannot represent a sharp transient peak triggered by a stimulus
    — and iter 52 made that gap visible by regressing insulin, glucagon,
    FFA, GLP-1 (all stimulus-driven peakers) when their heads either
    stayed on SpeciesHead (insulin) or were converted to SetpointHead
    (glucagon, FFA, GLP-1). The data wants: baseline equilibrium at
    typical, plus a sharp peak when a specific stimulus crosses a
    threshold.

    Emission shape:

        prod = softplus(raw_basal) + softplus(raw_peak) · σ(d · (s − θ) / τ)

    where ``s`` is the stimulus (a feature of the head's input vector,
    selected by ``stimulus_idx``), ``d ∈ {+1, −1}`` selects gate
    direction (peak when stimulus is high or low), and ``θ``, ``τ`` are
    learnable scalar parameters.

    Why this escapes the softplus-saturation trap that SpeciesHead
    suffered from: the basal output only needs to fit baseline
    production, not to *kill itself* to keep the marker at typical
    during a stimulus-driven excursion. The peak term is structurally
    separate and gated, so the optimizer can move the peak amplitude
    without disturbing the basal regime (or vice versa). Compared to
    SetpointHead, this head can represent SHARP onsets (the σ-gate is
    ~1 once the stimulus crosses θ + a few τ) which first-order
    relaxation can't.

    Init: ``g_thresh`` and ``log_g_temp`` are scalar parameters set at
    construction. Defaults pick a position that already differentiates
    common states from typical (e.g. for glucose-anti-gated glucagon,
    threshold at −0.5 z-score ≈ 80 mg/dL, so the gate fires when
    glucose drops noticeably below fasting). Final-layer bias init:
    ``raw_basal`` and ``raw_cons`` biases set so softplus(0)=log 2
    (matches SpeciesHead's converged behavior near typical); raw_peak
    bias = 0 (small initial peak that grows during training).

    ``stimulus_idx`` is the integer index into the head's input vector
    ``x`` (state-self || coupling || external || embedding || time)
    that selects the stimulus. The parent module's input layout is:
    state[0:n_species], coupling[n_species:n_species+n_coupling], ...

    Iter 97: ``forward`` also accepts an explicit ``stimulus`` tensor which
    overrides the ``x[..., stimulus_idx]`` read. A module that knows a
    per-patient reference (the metabolic module's Gb / Ib heads) passes the
    DEVIATION from that reference here, so the gate fires relative to the
    patient's own baseline instead of the population's (review 2026-09-04,
    item 3.4: population thresholds put the insulin gate at 116.5 mg/dL for
    everyone, so a Gb=120 patient fasted with a standing +34 % clearance and
    a Gb=75 patient fasted to 54 mg/dL).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        stimulus_idx: int,
        gate_dir: int = 1,
        init_thresh: float = 0.5,
        init_log_temp: float = -0.7,
    ):
        super().__init__()
        assert gate_dir in (-1, 1), "gate_dir must be +1 or -1"
        self.stimulus_idx = int(stimulus_idx)
        self.gate_dir = float(gate_dir)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3),
        )
        # Gate position + temperature, per-head learnable.
        self.g_thresh = nn.Parameter(torch.tensor(float(init_thresh)))
        self.log_g_temp = nn.Parameter(torch.tensor(float(init_log_temp)))

    def forward(
        self,
        x: torch.Tensor,
        state_self: torch.Tensor,
        stimulus: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if stimulus is None:
            stimulus = x[..., self.stimulus_idx]
        gate = torch.sigmoid(self.gate_dir * (stimulus - self.g_thresh) / gate_temp(self.log_g_temp))
        raw = self.network(x)
        basal = nn.functional.softplus(raw[..., 0])
        peak = nn.functional.softplus(raw[..., 1])
        cons = nn.functional.softplus(raw[..., 2])
        prod = basal + peak * gate
        return prod, cons


HeadFactory = Callable[[int, int], nn.Module]


class MassActionModule(nn.Module):
    """Module for chemical species with mass-action kinetics.

    rate_i = production_i - consumption_i × concentration_i

    Each species has its own ``SpeciesHead`` (or a subclass-supplied
    variant) that maps the module input to (prod, cons). Production
    and consumption are non-negative (softplus) and per-species scaled.
    This isolates each species' representation so insulin's dominant
    gradient pressure cannot collapse the small-magnitude species
    onto a flat manifold (iter-23 architectural surgery).

    ITER 95 — THE CONSUMPTION TERM NOW USES CONCENTRATION, AS THE LINE ABOVE
    ALWAYS CLAIMED. Through iter 94 it used the NORMALIZED DEVIATION instead:

        rate = prod·prod_scale − cons·cons_scale·norm_state
        norm_state = (raw − typical) / NORM_SCALE

    Modules receive the normalized state but their rates are applied to the RAW
    state (model.py), so this is a frame error of the same family as the iter-90
    Sg bug — and it had two consequences, both of which have been misdiagnosed
    for dozens of iterations.

    1. ``typical`` WAS AN ABSORBING FLOOR FOR EVERY SPECIES. At raw == typical
       the consumption term is exactly zero, so rate = prod ≥ 0; below typical,
       ``−cons·norm_state`` becomes a positive SOURCE. No species whose head
       emits non-negative production could ever fall below its typical value.
       Measured on the iter-94 artifact across fast / fed / big-meal / hard-bout
       protocols, all six metabolic species hit ``min − typical = 0.0000``
       exactly, while the teacher takes insulin to 2.81 and hepatic_output to
       1.12. iter 94 found this for the glycogen pools and read it as specific to
       ``GlycogenFluxHead``; it was never specific to anything.

    2. IT MANUFACTURED THE ITER-51 DEAD-PATHWAY TRAP. Sitting at typical required
       ``prod = 0``, i.e. ``prod_raw → −∞``, where ``d(softplus)/d(prod_raw) =
       sigmoid(prod_raw)`` vanishes and the head's gradient dies. Measured on the
       isolated assembly: asking the deviation frame for an equilibrium at 3.8
       (teacher's 24 h-fast insulin, typical 10) reaches 10.002 with prod_raw
       = −9.05 and 1.2e-04 of the gradient surviving. The concentration frame
       reaches 3.800 exactly with prod_raw = −0.23 and 4.4e-01 surviving.

    In the corrected frame the equilibrium is ``raw* = typical·prod/cons`` —
    reachable anywhere in (0, ∞) — and sitting at typical requires ``prod = cons``,
    two moderate positive values, so there is no saturation to escape. That is
    why ``SetpointHead`` no longer exists: it was invented in iter 51 to work
    around a trap that was an artifact of the frame, and iters 51-57's whole
    architecture of per-species workarounds was compensating for this one line.

    Note the effective relaxation rate changes from ``cons·cons_scale/NORM_SCALE``
    to ``cons·cons_scale`` — stiffer by NORM_SCALE for markers whose scale exceeds
    1 (insulin 10x, glucagon 20x). Trained ``cons`` values therefore do not
    transfer, and Euler headroom shrinks by the same factor; at dt=1 the limit is
    2/min, and insulin at cons≈2.4, cons_scale=0.1 sits at 0.24/min.
    """

    prod_scale: torch.Tensor
    cons_scale: torch.Tensor
    typical_val: torch.Tensor
    norm_scale_val: torch.Tensor

    def __init__(
        self,
        n_species: int,
        n_coupling: int,
        n_external: int,
        embedding_dim: int,
        hidden_dim: int = 48,
        typicals: list[float] | None = None,
        norm_scales: list[float] | None = None,
        head_factories: Optional[dict[int, HeadFactory]] = None,
    ):
        super().__init__()
        self.n_species = n_species
        self.n_coupling = n_coupling

        input_dim = n_species + n_coupling + n_external + embedding_dim + TIME_FEATURES_DIM
        self._input_dim = input_dim

        factories = head_factories or {}
        heads: list[nn.Module] = []
        for i in range(n_species):
            factory = factories.get(i)
            if factory is None:
                heads.append(SpeciesHead(input_dim, hidden_dim))
            else:
                heads.append(factory(input_dim, hidden_dim))
        self.heads = nn.ModuleList(heads)

        if typicals is None:
            typicals = [1.0] * n_species
        # Iter 95: NORM_SCALE per species, needed to rebuild the RAW concentration from
        # the normalized state the module is handed. Defaults to 1.0 (raw == normalized).
        if norm_scales is None:
            norm_scales = [1.0] * n_species
        cons_scales = [0.02] * n_species
        prod_scales = [c * t for c, t in zip(cons_scales, typicals)]

        self.register_buffer("prod_scale", torch.tensor(prod_scales, dtype=torch.float32))
        self.register_buffer("cons_scale", torch.tensor(cons_scales, dtype=torch.float32))
        self.register_buffer("typical_val", torch.tensor(typicals, dtype=torch.float32))
        self.register_buffer("norm_scale_val", torch.tensor(norm_scales, dtype=torch.float32))

    def forward(
        self,
        state: torch.Tensor,
        coupling: torch.Tensor,
        external: torch.Tensor,
        embedding: torch.Tensor,
        time_features: torch.Tensor,
    ) -> torch.Tensor:
        prod, cons = self.species_fluxes(
            state, coupling, external, embedding, time_features)
        return prod * self.prod_scale - cons * self.cons_scale * self.raw_state(state)

    def raw_state(self, state: torch.Tensor) -> torch.Tensor:
        """Rebuild the RAW concentration from the normalized state handed to the module.

        Iter 95: clamped at 0 because a concentration is not negative — the integrator's
        own physiological clamp already enforces that on the state, so this only guards
        a transient where the straight-through clamp lets a value dip fractionally below.
        """
        return torch.clamp(self.typical_val + self.norm_scale_val * state, min=0.0)

    def species_fluxes(
        self,
        state: torch.Tensor,
        coupling: torch.Tensor,
        external: torch.Tensor,
        embedding: torch.Tensor,
        time_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-species ``(prod, cons)`` before the mass-action scales are applied.

        Split out of ``forward`` (iter 94) so a subclass can build a species' rate
        from the raw head fluxes instead of from ``prod·prod_scale −
        cons·cons_scale·state``, without paying for a second pass over every head.
        The mass-action assembly stays in ``forward`` and is unchanged.
        """
        x = torch.cat([state, coupling, external, embedding, time_features], dim=-1)
        prods: list[torch.Tensor] = []
        conss: list[torch.Tensor] = []
        for i, head in enumerate(self.heads):
            prod_i, cons_i = head(x, state[..., i])
            prods.append(prod_i)
            conss.append(cons_i)
        return torch.stack(prods, dim=-1), torch.stack(conss, dim=-1)


class LearnedDynamicsModule(nn.Module):
    """Module for vital signs with fully learned dynamics.

    No imposed equation — the network directly outputs rates of change.
    The model discovers regulatory feedback from training data.
    """

    def __init__(
        self,
        n_state: int,
        n_coupling: int,
        n_external: int,
        embedding_dim: int,
        hidden_dim: int = 32,
    ):
        super().__init__()
        self.n_state = n_state
        self.n_coupling = n_coupling

        input_dim = n_state + n_coupling + n_external + embedding_dim + TIME_FEATURES_DIM

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_state),
        )
        # Iter 97: the output layer starts at zero, so a freshly built module rests at
        # its setpoint (cardiovascular / thermoreg) or holds level (respiratory) until
        # training gives the drivers authority. PyTorch's default init put an O(0.3)/min
        # driver on every vital — for temperature (k = 0.025/min) that is a +12 C
        # equilibrium offset: a fresh model integrated 24 h ran core temperature to
        # 41.4 C at the zero embedding. The weight gradient of a zero output layer is the
        # hidden activation, so nothing is starved; this is the same init the setpoint
        # heads already use.
        with torch.no_grad():
            self.network[-1].weight.zero_()
            self.network[-1].bias.zero_()

    def forward(
        self,
        state: torch.Tensor,
        coupling: torch.Tensor,
        external: torch.Tensor,
        embedding: torch.Tensor,
        time_features: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([state, coupling, external, embedding, time_features], dim=-1)
        return self.network(x)


class GutModuleBase(nn.Module):
    """Learned absorption kernel for the gut module.

    Iter 97 — THE KERNEL IS A NORMALIZED DENSITY TIMES A LEARNED MASS GAIN.

        K_ij(t; emb) = f_bio_ij(emb) · density_ij(t; emb)
        appearance_j(t) = Σ_i  macros_i · K_ij(t) with K diagonal (carbs→glucose, fat→lipid, protein→amino).

    where ``density_ij`` is a learned MIXTURE over a fixed bank of gamma
    shapes (``_KERNEL_BASIS``), every one of which is zero at t = 0, integrates
    to exactly 1 over [0, ∞), and decays smoothly. So, by construction and not
    by SGD:

    * ``K(0) = 0``                       — nothing appears at the instant of the meal;
    * ``∫ K_ij dt = f_bio_ij``           — the integral of appearance is the ingested
                                           mass times a learned per-patient gain;
    * ``K -> 0`` smoothly               — and every basis component has < 1 % of its
                                           mass beyond ``MEAL_ACTIVE_WINDOW_MIN``
                                           (asserted in tests), so the active-window
                                           mask in ``modules/gut.py`` is a numerical
                                           no-op rather than a cliff.

    Through iter 96 the kernel was an MLP on (t, emb) with softplus outputs.
    Nothing forced any of the three properties, and none held: for a 60/20/25 g
    meal the untrained kernel's appearance at dt = 0 was its MAXIMUM, its AUC
    was 3.5x the teacher's, 23 % of it lay beyond 240 min, and it stepped
    0.49 -> 0 at 480 min — which, for a 19:00 dinner, put an artificial
    -8 mg/dL/h glucose cliff at 03:00, inside the scored pre-dawn window
    (review 2026-09-04, item 2.3). A ``carb_mass_balance`` LOSS existed and
    still left the AUC 3.5x off; the only thing that makes conservation hold is
    to build it in.

    The two properties the previous kernel DID guarantee are kept: appearance
    is exactly zero at zero dose and exactly linear in dose (macros never enter
    the MLP), and gradients flow straight through to the embedding and the
    kernel parameters.

    Because the MLP sees ONLY the embedding — time enters through the analytic
    basis — it runs once per (meal, patient) rather than once per time step,
    and a freshly built kernel is already a plausible absorption curve: the
    output layer starts at zero with its bias set to the priors in
    ``_INIT_MIXTURE`` / ``_INIT_F_BIO`` (fitted to the teacher: carbohydrate
    peaking ~45 min with a slow tail, fat ~65 min, protein ~50 min), so training starts in
    the right regime and the embedding's authority over shape and gain grows
    from zero.

    ``f_bio`` is in APPEARANCE UNITS PER GRAM. Since iter 97 the teacher's
    carbohydrate kernel is MASS-CONSERVING: one gram integrates to
    ``MG_DL_PER_G`` (= 1000 / V_G = 7.72 mg/dL of glucose space,
    ``pulse.types``) and fat/protein appear as ``3.0 × g/min``. The student's
    kernel carries the same convention in ``APPEARANCE_UNITS_PER_G`` and
    initializes ``f_bio`` at it (so ``f_bio = 1`` means "all of it appears");
    the metabolic module divides by the same constant to recover grams for
    its carbon budget. When the teacher's units change, only this constant
    and the init move.

    The 4th output channel, ``nutrient_flag``, is no longer a separate MLP.
    It is ``1 − exp(−unabsorbed_mass / FLAG_GATE_SCALE_G)`` where
    ``unabsorbed_mass`` is the ingested mass whose kernel has NOT yet
    appeared (the mixture's survival function) — a physical "fed state" that
    is 0 at zero dose, ~1 just after a real meal, and decays smoothly to 0 as
    the meal is absorbed, with no cliff at the window edge.

    Outputs: [glucose_appearance, lipid_appearance, amino_appearance, nutrient_flag]
    """

    # Number of macro channels (carbs, fats, proteins) and appearance
    # rate channels (glucose, lipid, amino). Same number today; kept as
    # named constants in case we ever decouple them.
    N_MACROS: int = 3
    N_APPEARANCE: int = 3

    # Characteristic macro mass (grams) for the nutrient_flag presence
    # gate. The gate ``1 − exp(−unabsorbed / FLAG_GATE_SCALE_G)`` is 0
    # at zero dose, 0.63 at 10 g, 0.95 at 30 g, 0.9999 at 100 g. Set
    # small enough that even a snack pushes the flag toward saturation.
    FLAG_GATE_SCALE_G: float = 10.0

    # Appearance units per gram of macro, per channel — the teacher's convention
    # (see the class docstring). Order: (glucose, lipid, amino).
    APPEARANCE_UNITS_PER_G: tuple[float, float, float] = (MG_DL_PER_G, 3.0, 3.0)

    # The basis bank as (gamma shape, gamma rate /min). Each component is chosen by
    # its PEAK time ``(shape − 1) / rate`` and its shape is raised for the slow
    # components so that the mass beyond 480 min stays below 1 % for every one
    # of them (shape-2 at a 180-min peak would leave 31 % beyond 480). Peaks:
    # 15, 25, 40, 60, 90, 130, 180 min — spanning the teacher's fast carbohydrate
    # (0.04/min → 25 min) through its slow fat/carbohydrate tails (0.012-0.015/min
    # → 67-83 min) with room on both sides for the mixture to learn.
    _KERNEL_BASIS: tuple[tuple[float, float], ...] = (
        (2.0, 1.0 / 15.0),
        (2.0, 1.0 / 25.0),
        (2.0, 1.0 / 40.0),
        (2.0, 1.0 / 60.0),
        (3.0, 2.0 / 90.0),
        (4.0, 3.0 / 130.0),
        (6.0, 5.0 / 180.0),
    )
    # Prior mixture weights per diagonal channel over the basis bank.
    #   carbs -> glucose : teacher 75 % fast (peak 25) + 25 % slow (peak 83)
    #   fats  -> lipid   : teacher rate 0.015 (peak 67)
    #   prot  -> amino   : teacher rate 0.02 (peak 50)
    _INIT_MIXTURE: tuple[tuple[float, ...], ...] = (
        (0.00, 0.00, 0.48, 0.27, 0.00, 0.00, 0.25),
        (0.00, 0.00, 0.18, 0.41, 0.28, 0.00, 0.13),
        (0.00, 0.00, 0.64, 0.01, 0.35, 0.00, 0.00),
    )

    def __init__(self, embedding_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.n_basis = len(self._KERNEL_BASIS)
        self.register_buffer(
            "basis_shape",
            torch.tensor([s for s, _ in self._KERNEL_BASIS], dtype=torch.float32),
        )
        self.register_buffer(
            "basis_rate",
            torch.tensor([r for _, r in self._KERNEL_BASIS], dtype=torch.float32),
        )
        # log-normalizer of each gamma density: shape·log(rate) − lgamma(shape)
        self.register_buffer(
            "basis_log_norm",
            self.basis_shape * torch.log(self.basis_rate) - torch.lgamma(self.basis_shape),
        )
        self.register_buffer(
            "appearance_units_per_g",
            torch.tensor(self.APPEARANCE_UNITS_PER_G, dtype=torch.float32),
        )

        # Input is ONLY the embedding. Output: mixture logits for each of the
        # three diagonal channels over the basis, plus one f_bio per channel.
        output_dim = self.N_MACROS * self.n_basis + self.N_MACROS
        self.kernel = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )
        with torch.no_grad():
            nn.init.normal_(self.kernel[-1].weight, std=0.01)
            bias = torch.zeros(output_dim)
            logits = torch.zeros(self.N_MACROS, self.n_basis)
            f_raw = torch.zeros(self.N_MACROS)
            for i in range(self.N_MACROS):
                w = torch.tensor(self._INIT_MIXTURE[i], dtype=torch.float32)
                logits[i] = torch.log(w + 1e-3)
                f_raw[i] = _inverse_softplus(self.APPEARANCE_UNITS_PER_G[i])
            n_logit = self.N_MACROS * self.n_basis
            bias[:n_logit] = logits.reshape(-1)
            bias[n_logit:] = f_raw
            self.kernel[-1].bias.copy_(bias)

    # ---- the analytic basis ---------------------------------------------------

    def basis_density(self, dt: torch.Tensor) -> torch.Tensor:
        """Gamma densities of the bank at ``dt`` minutes → ``[..., n_basis]``.

        Each integrates to 1 over [0, ∞) and is exactly 0 at dt = 0 (all shapes
        > 1). Negative ``dt`` is treated as 0 (the caller masks it anyway).
        """
        t = dt.clamp(min=0.0).unsqueeze(-1)
        log_t = torch.log(t)  # -inf at t = 0 → density exactly 0 there
        log_d = self.basis_log_norm + (self.basis_shape - 1.0) * log_t - self.basis_rate * t
        return torch.exp(log_d)

    def basis_survival(self, dt: torch.Tensor) -> torch.Tensor:
        """Fraction of each basis component's mass NOT yet appeared by ``dt``.

        The regularized upper incomplete gamma for integer shape k:
        ``Q(k, x) = e^{-x} Σ_{j<k} x^j / j!`` with ``x = rate · dt``.
        """
        x = self.basis_rate * dt.clamp(min=0.0).unsqueeze(-1)
        k_max = int(self.basis_shape.max().item())
        term = torch.ones_like(x)
        total = torch.zeros_like(x)
        for j in range(k_max):
            total = total + term * (self.basis_shape > j).to(x.dtype)
            term = term * x / float(j + 1)
        return torch.exp(-x) * total

    def kernel_tail_mass(self, t_minutes: float) -> torch.Tensor:
        """Mass beyond ``t_minutes`` for every basis component (for the window test)."""
        return self.basis_survival(torch.tensor(float(t_minutes))).squeeze(0)

    # ---- the learned part -----------------------------------------------------

    def mixture(self, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(weights[..., 3, n_basis], f_bio[..., 3])`` — one mixture per macro."""
        raw = self.kernel(embedding)
        n_logit = self.N_MACROS * self.n_basis
        logits = raw[..., :n_logit].reshape(*raw.shape[:-1], self.N_MACROS, self.n_basis)
        weights = torch.softmax(logits, dim=-1)
        f_bio = nn.functional.softplus(raw[..., n_logit:]).reshape(
            *raw.shape[:-1], self.N_MACROS)
        return weights, f_bio

    def response_and_flag(
        self,
        macros: torch.Tensor,
        weights: torch.Tensor,
        f_bio: torch.Tensor,
        density: torch.Tensor,
        survival: torch.Tensor,
    ) -> torch.Tensor:
        """Assemble ``[..., 4]`` from broadcast-compatible pieces.

        macros ``[..., 3]``; weights ``[..., 3, K]``; f_bio ``[..., 3]``;
        density / survival ``[..., K]``. Appearance is diagonal: carbs feed
        glucose, fat lipid, protein amino.
        """
        dens = (weights * density.unsqueeze(-2)).sum(dim=-1)
        appearance = macros * f_bio * dens
        surv_i = (weights * survival.unsqueeze(-2)).sum(dim=-1)
        unabsorbed = (macros * surv_i).sum(dim=-1)
        nutrient_flag = 1.0 - torch.exp(-unabsorbed / self.FLAG_GATE_SCALE_G)
        return torch.cat([appearance, nutrient_flag.unsqueeze(-1)], dim=-1)

    def forward_single_meal(
        self,
        macros: torch.Tensor,
        time_since_meal: torch.Tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Compute appearance rates for a single meal.

        macros: ``[..., 3]`` (carbs, fats, proteins) in grams
        time_since_meal: ``[...]`` minutes
        embedding: ``[..., embedding_dim]``
        Returns: ``[..., 4]`` = (glucose, lipid, amino, nutrient_flag)
        """
        weights, f_bio = self.mixture(embedding)
        density = self.basis_density(time_since_meal)
        survival = self.basis_survival(time_since_meal)
        return self.response_and_flag(macros, weights, f_bio, density, survival)


def _inverse_softplus(y: float) -> float:
    return math.log(math.expm1(y))


def compute_time_features(t_minutes: torch.Tensor) -> torch.Tensor:
    """Cyclic time encoding: first and second harmonic of the 24 h clock.

    ``[sin θ, cos θ, sin 2θ, cos 2θ]`` with ``θ = 2π·hour/24``. Iter 97 dropped
    the leading ``(t_hours − 12)/12`` ramp: a sawtooth that jumped +1 → −1 at
    midnight and gave every module a vector-field discontinuity there (see
    ``types.TIME_FEATURES_DIM``). Continuous and periodic by construction; the
    second harmonic is what makes a dawn asymmetry expressible.
    """
    theta = 2 * math.pi * (t_minutes / 60.0) / 24.0
    return torch.stack([
        torch.sin(theta),
        torch.cos(theta),
        torch.sin(2 * theta),
        torch.cos(2 * theta),
    ], dim=-1)
