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

    def forward(self, x: torch.Tensor, state_self: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

        input_dim = n_species + n_coupling + n_external + embedding_dim + 3  # 3 = time features
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

        input_dim = n_state + n_coupling + n_external + embedding_dim + 3

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_state),
        )

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

    Factorized form: appearance rates are linear in meal macros, with a
    learned per-gram time-and-embedding-conditional response matrix. The
    kernel emits

        K(t, emb) ∈ R^{3 macros × 3 channels}_{≥0}

    and appearance is computed as ``macros @ K``. Two physical
    properties hold by construction (not by SGD):

    * ``appearance(macros=0) == 0``  — no food, no nutrient appearance.
    * ``appearance(α·macros) == α·appearance(macros)`` — dose-linearity,
      which also implies dose-monotonicity.

    These are the two pathologies that broke iters 13–15 (chronic
    nonzero baseline at zero dose; tanh-induced saturation past ~30 g).
    Solving them structurally costs zero parameters (kernel hypothesis
    space actually shrinks: macros no longer enter the MLP).

    The 4th output channel, ``nutrient_flag``, is a binary "fed-state"
    indicator that is *not* per-gram. It uses a separate sigmoid head
    on (t, emb), gated by an analytic ``1 − exp(−total_macro / 10g)``
    presence factor so it also vanishes at zero dose.

    Outputs: [glucose_appearance, lipid_appearance, amino_appearance, nutrient_flag]
    """

    # Number of macro channels (carbs, fats, proteins) and appearance
    # rate channels (glucose, lipid, amino). Same number today; kept as
    # named constants in case we ever decouple them.
    N_MACROS: int = 3
    N_APPEARANCE: int = 3

    # Characteristic macro mass (grams) for the nutrient_flag presence
    # gate. The gate ``1 − exp(−total_macro / FLAG_GATE_SCALE_G)`` is 0
    # at zero dose, 0.63 at 10 g, 0.95 at 30 g, 0.9999 at 100 g. Set
    # small enough that even a snack pushes the flag toward saturation.
    FLAG_GATE_SCALE_G: float = 10.0

    def __init__(self, embedding_dim: int, hidden_dim: int = 32):
        super().__init__()
        # Input is now ONLY (t/60, embedding) — macros are factored out
        # of the MLP and applied as a multiplicative outer factor.
        input_dim = 1 + embedding_dim
        # Output: N_MACROS × N_APPEARANCE response matrix entries, plus
        # one scalar logit for the nutrient_flag head.
        n_response = self.N_MACROS * self.N_APPEARANCE
        output_dim = n_response + 1

        self.kernel = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward_single_meal(
        self,
        macros: torch.Tensor,
        time_since_meal: torch.Tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Compute appearance rates for a single meal.

        macros: ``[..., 3]`` (carbs, fats, proteins) in grams
        time_since_meal: ``[...]`` minutes (will be normalized internally)
        embedding: ``[..., embedding_dim]``
        Returns: ``[..., 4]`` = (glucose, lipid, amino, nutrient_flag)
        """
        t_norm = time_since_meal / 60.0
        x = torch.cat([t_norm.unsqueeze(-1), embedding], dim=-1)
        raw = self.kernel(x)

        n_response = self.N_MACROS * self.N_APPEARANCE
        # Per-gram response matrix K ∈ R_{≥0}^{N_MACROS × N_APPEARANCE}.
        # softplus keeps every entry non-negative so appearance rates
        # cannot flip sign, and provides smooth gradients near zero.
        K = nn.functional.softplus(
            raw[..., :n_response].reshape(*raw.shape[:-1], self.N_MACROS, self.N_APPEARANCE)
        )
        # appearance[..., j] = Σ_i macros[..., i] · K[..., i, j]
        appearance = torch.einsum("...m,...mn->...n", macros, K)

        # nutrient_flag head: sigmoid on (t, emb), analytically gated by
        # meal-presence so it vanishes at zero dose.
        flag_logit = raw[..., n_response]
        total_macro = macros.sum(dim=-1)
        presence = 1.0 - torch.exp(-total_macro / self.FLAG_GATE_SCALE_G)
        nutrient_flag = (torch.sigmoid(flag_logit) * presence).unsqueeze(-1)

        return torch.cat([appearance, nutrient_flag], dim=-1)


def compute_time_features(t_minutes: torch.Tensor) -> torch.Tensor:
    """Cyclic time encoding: linear hour + sin/cos with 24h period."""
    t_hours = t_minutes / 60.0
    return torch.stack([
        (t_hours - 12.0) / 12.0,
        torch.sin(2 * math.pi * t_hours / 24.0),
        torch.cos(2 * math.pi * t_hours / 24.0),
    ], dim=-1)
