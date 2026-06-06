"""
Dedicated dose-response training signal — multi-marker amplitude axis.

Carbohydrate dose response is one of the few intra-subject quantitative
findings that's well-pinned in the literature (Wolever 1991/1996;
Brand-Miller 2003): peak postprandial glucose rises ≈0.6–1.0 mg/dL per gram
of carbohydrate in the 25–100 g range. The earlier cohort spec encoded this
finding but ran through the cohort signal's hard ``series.max()`` peak
extractor — gradient flowed only through the single argmax timestep, which
flips between adjacent steps and gives noisy / sparse signal. Combined with
sharing weight across six other cohort specs, the dose-response gradient was
effectively muted.

This module provides a *dedicated* signal, now multi-marker:

* its own per-epoch protocol — K patient embeddings × M carb doses, fasting
  baseline → 4 h rollout per arm;
* differentiable soft Δpeak per supervised marker via the softmax-weighted
  estimator used in ``landmarks.py`` (gradient distributed across the
  post-meal window);
* per-marker loss mode — ``slope`` (OLS slope across doses vs literature
  target as Gaussian z²) for the well-pinned glucose curve, ``rank`` (hinge
  on adjacent-dose Δpeak monotonicity) for insulin / GLP-1 where the
  literature slope is too noisy for a slope target but the ordering
  (90 g > 60 g > 30 g) is solid;
* its own weight, independent of the multi-spec cohort signal.

This complements the cohort signal: cohort still handles population means;
dose-response gets the focused cross-marker amplitude gradient that the
``meal_dose_response`` benchmark probes. Iter 69 (sha 66db1896→TBD) opened
this from glucose-only (iter 32 era) to glucose + insulin + GLP-1 after
iter 68 r8 landed meal_dose_response 0/3 (glucose 0.015 / 5, insulin 1.64 /
2, glp1 −0.015 / 0 — every check is an amplitude/scaling failure).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .knowledge.full_body import PatientParams
from .knowledge.textbook_scenarios.base import cold_model_trajectory
from .landmarks import LandmarkDirection, post_meal_landmarks
from .model import integrate, precompute_gut_outputs
from .modules.gut import MealEvent
from .types import MARKER_INDEX


@dataclass(frozen=True)
class MarkerDoseTarget:
    """Per-marker dose-response supervision spec.

    Three modes:

    * ``slope`` — OLS slope (Δpeak vs carb dose) compared to a literature
      target via Gaussian z². Pins the *gradient* of the dose-response line
      but is invariant to its intercept: a model whose peaks are uniformly
      too low still satisfies a correct slope. Use only when the absolute
      level is already well-anchored elsewhere.
    * ``peak`` — Gaussian z² on the *absolute* Δpeak at each dose against a
      literature line through the origin (``target_slope · dose``), with a
      dose-proportional σ (``sigma_slope · dose`` = constant CV). This pins
      both the slope AND the absolute amplitude (zero carbs → zero rise is
      the physically correct intercept). This is the iter-73 fix for the
      standing meal-amplitude gate failure: ``slope``/``rank`` left the
      absolute peak unsupervised, so the model learned the right *shape* at
      systematically too-low magnitude (glucose 0.29, insulin 1.09,
      verifier_meal 0.60 — all amplitude, not shape). glucose: Wolever
      0.7 mg/dL/g, σ 0.25/g.
    * ``rank`` — hinge loss on adjacent-dose Δpeak monotonicity
      (peak(j+1) − peak(j) > margin). Use when literature pins the
      *ordering* (90 g > 60 g > 30 g) but neither slope nor absolute level
      is reliable (e.g. GLP-1).

    ``weight`` scales this marker's contribution to the composite loss (the
    aggregate is then scaled by the top-level ``--dose-response-weight``).
    ``direction`` selects PEAK (default; postprandial markers) — kept as a
    field for completeness in case a future spec wants NADIR.
    """

    marker: str
    mode: str  # "slope" | "peak" | "rank"
    target_slope: float | None = None   # slope/peak — literature amplitude per g carb
    sigma_slope: float | None = None    # slope/peak — Gaussian std per g carb
    rank_margin: float = 0.5            # mode=rank — min Δpeak between adjacent doses
    weight: float = 1.0
    direction: LandmarkDirection = LandmarkDirection.PEAK

    def __post_init__(self) -> None:
        if self.marker not in MARKER_INDEX:
            raise ValueError(f"unknown marker '{self.marker}' for dose-response target")
        if self.mode not in ("slope", "peak", "rank"):
            raise ValueError(f"mode must be 'slope', 'peak' or 'rank', got '{self.mode}'")
        if self.mode in ("slope", "peak"):
            if self.target_slope is None or self.sigma_slope is None:
                raise ValueError(f"{self.mode} mode requires target_slope + sigma_slope")
            if self.sigma_slope <= 0:
                raise ValueError("sigma_slope must be positive")
        if self.mode == "rank" and self.rank_margin < 0:
            raise ValueError("rank_margin must be ≥ 0")
        if self.weight < 0:
            raise ValueError("weight must be ≥ 0")


@dataclass(frozen=True)
class DoseResponseProtocol:
    """Protocol describing the dose-response challenge.

    Defaults align with the ``meal_dose_response`` benchmark scenario so the
    training distribution matches what we measure: carb doses (30/60/90 g)
    bracket the benchmark's 30 g vs 90 g comparison, with the same macros and
    the same cold-model fasting initial state.

    Backward-compat fields: ``target_slope`` + ``sigma_slope`` describe the
    glucose slope target (Wolever 1991/1996: ≈ 0.7 mg/dL/g, σ ~0.25). When
    ``marker_targets`` is empty these legacy fields define a single
    glucose-slope supervision; when ``marker_targets`` is non-empty those
    explicit per-marker specs take over and the legacy fields are ignored.
    """

    carb_doses_g: tuple[float, ...] = (30.0, 60.0, 90.0)
    fats_g: float = 5.0
    proteins_g: float = 10.0
    meal_offset_min: int = 30
    duration_min: int = 240
    start_hour: float = 8.0
    pre_window: int = 15
    post_window: int = 180
    softargmax_beta: float = 0.10
    target_slope: float = 0.7
    sigma_slope: float = 0.25
    marker_targets: tuple[MarkerDoseTarget, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if len(self.carb_doses_g) < 3:
            raise ValueError("dose-response slope needs ≥3 doses")
        if self.meal_offset_min < self.pre_window:
            raise ValueError("meal_offset_min must leave room for pre_window baseline")
        if self.meal_offset_min + self.post_window > self.duration_min:
            raise ValueError("post_window extends past simulated duration")
        if self.sigma_slope <= 0:
            raise ValueError("sigma_slope must be positive")

    def effective_targets(self) -> tuple[MarkerDoseTarget, ...]:
        """Return the active per-marker targets.

        Legacy path: ``marker_targets`` empty → synthesise the glucose-slope
        target from the top-level ``target_slope`` / ``sigma_slope`` fields
        (the iter-32-era single-target behaviour preserved exactly).
        """
        if self.marker_targets:
            return self.marker_targets
        return (
            MarkerDoseTarget(
                marker="glucose",
                mode="slope",
                target_slope=self.target_slope,
                sigma_slope=self.sigma_slope,
                weight=1.0,
            ),
        )


def cold_initial_state(
    protocol: DoseResponseProtocol,
    rng: "np.random.Generator | None" = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Sample a realistic fasting initial state from the analytical cold model.

    Mirrors the benchmark's setup (``meal_dose_response_neural_on_model``), which
    uses ``cold_model_trajectory(...)[0]`` as the integration starting state.
    Per-epoch stochasticity (via ``rng``) prevents the model from overfitting to
    a single initial condition.
    """
    import numpy as np  # local import keeps top of module tensor-only

    if rng is None:
        rng = np.random.default_rng()
    meals = [(
        float(protocol.meal_offset_min),
        float(protocol.carb_doses_g[0]),
        float(protocol.fats_g),
        float(protocol.proteins_g),
    )]
    traj = cold_model_trajectory(
        PatientParams(),
        meals,
        protocol.duration_min,
        protocol.start_hour,
        rng=rng,
    )
    return torch.tensor(traj[0], dtype=torch.float32, device=device)


def _ols_slope(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Closed-form OLS slope of y on x (1-D, equal length)."""
    xm = x.mean()
    ym = y.mean()
    num = ((x - xm) * (y - ym)).sum()
    den = ((x - xm).pow(2)).sum().clamp(min=1e-8)
    return num / den


def predicted_peaks_batched(
    model: nn.Module,
    embeddings: torch.Tensor,
    protocol: DoseResponseProtocol,
    initial_state: torch.Tensor,
    markers: tuple[str, ...],
    directions: tuple[LandmarkDirection, ...],
) -> torch.Tensor:
    """Soft Δpeak per (marker, dose, embedding).

    Returns ``[B, M, D]`` — B embeddings × M markers × D doses. Runs one
    batched ``integrate`` per dose (same dose for every embedding in the
    batch), shared across all supervised markers — the marker dimension
    just re-extracts peaks from the same trajectory, no extra forward
    passes.
    """
    if len(markers) != len(directions):
        raise ValueError("markers / directions length mismatch")
    marker_indices = [MARKER_INDEX[m] for m in markers]
    n_steps = protocol.duration_min
    t0 = protocol.start_hour * 60.0
    B = int(embeddings.shape[0])
    state_b = initial_state.unsqueeze(0).expand(B, -1)
    M = len(markers)
    D = len(protocol.carb_doses_g)

    peaks_per_dose: list[torch.Tensor] = []
    for dose in protocol.carb_doses_g:
        meal = MealEvent(
            time=float(protocol.meal_offset_min),
            carbs=float(dose),
            fats=float(protocol.fats_g),
            proteins=float(protocol.proteins_g),
        )
        gut = precompute_gut_outputs(
            model, embeddings, n_steps,
            dt=1.0, start_time_minutes=t0, meals=[meal],
        )
        traj = integrate(
            model, state_b, embeddings, n_steps,
            dt=1.0, start_time_minutes=t0, meals=[meal],
            gut_outputs=gut,
        )  # [B, T, STATE]
        # Extract Δpeak per (marker, batch). post_meal_landmarks is 1-D
        # over the trajectory window; the inner loops are cheap relative
        # to the integrate above.
        per_marker: list[torch.Tensor] = []
        for m_i, idx in enumerate(marker_indices):
            direction = directions[m_i]
            per_emb: list[torch.Tensor] = []
            for b in range(B):
                delta_peak, _, _ = post_meal_landmarks(
                    traj[b, :, idx],
                    meal_step=protocol.meal_offset_min,
                    pre_window=protocol.pre_window,
                    post_window=protocol.post_window,
                    softargmax_beta=protocol.softargmax_beta,
                    direction=direction,
                )
                per_emb.append(delta_peak)
            per_marker.append(torch.stack(per_emb))  # [B]
        peaks_per_dose.append(torch.stack(per_marker, dim=0))  # [M, B]

    # peaks_per_dose: list of [M, B], length D → stack on dose axis → [D, M, B]
    # then permute to [B, M, D] for downstream slope / rank loss.
    peaks = torch.stack(peaks_per_dose, dim=0)  # [D, M, B]
    peaks = peaks.permute(2, 1, 0).contiguous()  # [B, M, D]
    assert peaks.shape == (B, M, D)
    return peaks


def predicted_slopes_batched(
    model: nn.Module,
    embeddings: torch.Tensor,
    protocol: DoseResponseProtocol,
    initial_state: torch.Tensor,
) -> torch.Tensor:
    """Legacy glucose-only slope path — preserved for tests + back-compat.

    Returns ``[B]`` glucose slopes (mg/dL per g carb). Equivalent to
    ``predicted_peaks_batched(... markers=("glucose",) ...)`` followed by
    OLS-slope reduction over the dose axis.
    """
    peaks = predicted_peaks_batched(
        model, embeddings, protocol, initial_state,
        markers=("glucose",), directions=(LandmarkDirection.PEAK,),
    )  # [B, 1, D]
    peaks_bd = peaks.squeeze(1)  # [B, D]
    doses = torch.tensor(
        protocol.carb_doses_g, dtype=torch.float32, device=embeddings.device,
    )
    xm = doses.mean()
    ym = peaks_bd.mean(dim=1, keepdim=True)
    num = ((doses - xm) * (peaks_bd - ym)).sum(dim=1)
    den = ((doses - xm).pow(2)).sum().clamp(min=1e-8)
    return num / den


def _slope_loss(peaks: torch.Tensor, doses: torch.Tensor, target: float, sigma: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-embedding Gaussian z² on OLS slope. peaks: [B, D]; returns (loss, slopes)."""
    xm = doses.mean()
    ym = peaks.mean(dim=1, keepdim=True)
    num = ((doses - xm) * (peaks - ym)).sum(dim=1)
    den = ((doses - xm).pow(2)).sum().clamp(min=1e-8)
    slopes = num / den  # [B]
    z = (slopes - target) / sigma
    return z.pow(2).mean(), slopes


def _peak_loss(
    peaks: torch.Tensor, doses: torch.Tensor, target_per_g: float, sigma_per_g: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gaussian z² on absolute Δpeak vs a literature line through the origin.

    ``peaks``: [B, D]. Target at dose d is ``target_per_g · d`` with std
    ``sigma_per_g · d`` (constant CV — larger doses get proportionally
    looser absolute tolerance, and the dose=0 intercept is pinned at 0 rise
    by construction). Unlike ``_slope_loss`` this is NOT offset-invariant:
    uniformly-too-low peaks are penalised. Returns (loss, mean Δpeak/dose).
    """
    target = target_per_g * doses                       # [D]
    sigma = (sigma_per_g * doses).clamp(min=1e-6)        # [D]
    z = (peaks - target) / sigma                         # [B, D] broadcast
    # Mean per-dose Δpeak-per-gram for logging (diagnostic, detached).
    per_g = (peaks.detach() / doses.clamp(min=1e-6)).mean(dim=0)  # [D]
    return z.pow(2).mean(), per_g


def _rank_loss(peaks: torch.Tensor, margin: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Hinge on adjacent-dose monotonicity. peaks: [B, D]; returns (loss, mean adjacent diff).

    For each adjacent pair (peaks[:, j], peaks[:, j+1]) we want
    peaks[:, j+1] − peaks[:, j] ≥ margin. The hinge ``relu(margin − diff)``
    penalises non-monotonicity and shortfall below the margin; gradient is
    zero once the ordering is established with at least ``margin`` slack
    (no over-fitting to magnitude past what the ranking demands).
    """
    diffs = peaks[:, 1:] - peaks[:, :-1]  # [B, D-1]
    hinge = torch.relu(margin - diffs)    # [B, D-1]
    return hinge.pow(2).mean(), diffs.detach().mean(dim=0)


def dose_response_epoch_loss(
    model: nn.Module,
    embeddings_to_supervise: list[torch.Tensor],
    protocol: DoseResponseProtocol,
    initial_state: torch.Tensor,
    device: torch.device | str,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Composite multi-marker dose-response loss.

    Returns ``(total_loss, diagnostics)``. ``diagnostics`` carries per-marker
    predicted slopes (slope mode) or mean adjacent-dose diffs (rank mode)
    plus the mean glucose slope under the key ``predicted_slope`` for
    back-compat with the iter-32-era logging.
    """
    if not embeddings_to_supervise:
        return torch.tensor(0.0, device=device), {"predicted_slope": 0.0}

    targets = protocol.effective_targets()
    markers = tuple(t.marker for t in targets)
    directions = tuple(t.direction for t in targets)

    embs = torch.stack(embeddings_to_supervise, dim=0)  # [B, EMB]
    peaks = predicted_peaks_batched(  # [B, M, D]
        model, embs, protocol, initial_state, markers=markers, directions=directions,
    )
    doses = torch.tensor(
        protocol.carb_doses_g, dtype=torch.float32, device=embs.device,
    )

    total_weight = sum(t.weight for t in targets)
    if total_weight <= 0:
        return torch.tensor(0.0, device=device), {"predicted_slope": 0.0}

    total_loss = torch.tensor(0.0, device=device)
    diagnostics: dict[str, float] = {}
    for m_i, target in enumerate(targets):
        peaks_bd = peaks[:, m_i, :]  # [B, D]
        if target.mode == "slope":
            loss, slopes = _slope_loss(
                peaks_bd, doses, target.target_slope, target.sigma_slope,
            )
            diagnostics[f"{target.marker}_slope"] = float(slopes.detach().mean().item())
            if target.marker == "glucose":
                diagnostics["predicted_slope"] = diagnostics[f"{target.marker}_slope"]
        elif target.mode == "peak":
            loss, per_g = _peak_loss(
                peaks_bd, doses, target.target_slope, target.sigma_slope,
            )
            diagnostics[f"{target.marker}_peak_per_g"] = float(per_g.mean().item())
            if target.marker == "glucose":
                # Back-compat logging key — peak mode implies a slope too.
                diagnostics["predicted_slope"] = diagnostics[f"{target.marker}_peak_per_g"]
        else:  # rank
            loss, diffs = _rank_loss(peaks_bd, target.rank_margin)
            for j in range(diffs.shape[0]):
                diagnostics[f"{target.marker}_diff_{j}"] = float(diffs[j].item())
        total_loss = total_loss + (target.weight / total_weight) * loss
        diagnostics[f"{target.marker}_loss"] = float(loss.detach().item())

    if "predicted_slope" not in diagnostics:
        # Composite has no glucose-slope leg; surface NaN to make absence
        # explicit in the training log rather than silently reporting 0.
        diagnostics["predicted_slope"] = float("nan")
    return total_loss, diagnostics
