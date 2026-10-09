"""Can a glucose window tell fasting glucose (Gb) apart from insulin sensitivity (Si)?

Si and Gb are per-person decodes in the metabolic module. Both move post-meal
glucose; Gb also sets the fasting level. Calibration freezes the weights and
fits the embedding, so the question is whether those two observations pull the
embedding in different directions.

At construction every person-head output layer is zero, so every embedding
decodes the same person and the embedding gradient through Si and Gb is
identically zero. That is the init, not the ODE. This script builds a fresh
ModularPhysiologyNetwork, leaves the rate laws at that init, and gives the two
heads orthogonal rank-1 reads of the embedding — the smallest opening at which
two people can differ in only one decode. Every other embedding path into the
rollout is silenced, so a glucose gradient has nowhere to go except those heads.

A meal reaches the Si head. A fast reaches the Gb head. Both embedding
gradients are still mostly the fasting-glucose direction: absolute glucose
sits on Gb in either window, so their cosine stays high. What separates them
is the Si axis, which a meal carries and a fast does not.
"""
from __future__ import annotations

import math

import torch

from pulse.model import ModularPhysiologyNetwork, integrate
from pulse.modules.gut import MealEvent
from pulse.types import EMBEDDING_DIM, MARKER_INDEX, NORM_CENTER

_GI = MARKER_INDEX["glucose"]
_GB_AXIS = 0
_SI_AXIS = 1
_FAST_STEPS = 180
_MEAL_STEPS = 220
_MEAL_AT = 40
_POST = 90
_MEAL = [MealEvent(time=float(_MEAL_AT), carbs=75.0, fats=5.0, proteins=10.0)]
_CLOCK = 8 * 60.0


def _fresh() -> ModularPhysiologyNetwork:
    torch.manual_seed(0)
    model = ModularPhysiologyNetwork(
        metabolic_hidden=16, appetite_hidden=8, stress_hidden=8,
        cardiovascular_hidden=16, thermoreg_hidden=8, respiratory_hidden=8,
        gut_hidden=8, hepatobiliary_hidden=8,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def _open_si_and_gb(model: ModularPhysiologyNetwork) -> None:
    """Gb reads embedding axis 0, Si reads axis 1. Nothing else reads either."""
    metabolic = model.metabolic
    with torch.no_grad():
        projection = model.embedding_projections["metabolic"]
        projection.weight.zero_()
        projection.bias.zero_()
        projection.weight[_GB_AXIS, _GB_AXIS] = 1.0
        projection.weight[_SI_AXIS, _SI_AXIS] = 1.0
        for head, axis in (
            (metabolic.glucose_baseline_net, _GB_AXIS),
            (metabolic.insulin_sens_net, _SI_AXIS),
        ):
            head[0].weight.zero_()
            head[0].bias.zero_()
            head[0].weight[0, axis] = 1.0
            head[-1].weight.zero_()
            head[-1].bias.zero_()
            head[-1].weight[0, 0] = 1.0
        columns = [
            i for i, (kind, _) in enumerate(metabolic.head_input_sources())
            if kind == "embedding"
        ]
        for network in metabolic.mlp_heads().values():
            network[0].weight[:, columns] = 0.0
        for name, layer in model.embedding_projections.items():
            if name != "metabolic":
                layer.weight.zero_()
        # The population meal is the kernel bias (a zero embedding ignores the
        # kernel weights). Zeroing the weights keeps that meal for every person.
        model.gut.kernel.kernel[-1].weight.zero_()
        model.default_inputs_net[0].weight[:, : model.embedding_dim] = 0.0


def _person(gb: float, si: float) -> torch.Tensor:
    embedding = torch.zeros(EMBEDDING_DIM)
    embedding[_GB_AXIS] = gb
    embedding[_SI_AXIS] = si
    return embedding


def _decode(model: ModularPhysiologyNetwork, embedding: torch.Tensor) -> dict[str, float]:
    projected = model.embedding_projections["metabolic"](embedding)
    with torch.no_grad():
        decoded = {key: float(value) for key, value in model.metabolic.person_params(projected).items()}
        decoded["gb"] = float(model.metabolic.glucose_setpoint_raw(projected))
    return decoded


def _roll(model: ModularPhysiologyNetwork, embedding: torch.Tensor, steps: int, meals: list) -> torch.Tensor:
    initial = torch.tensor(NORM_CENTER, dtype=torch.float32)
    return integrate(
        model, initial, embedding, steps,
        start_time_minutes=_CLOCK, meals=meals,
        sleep_wake=torch.ones(steps), activity=torch.zeros(steps),
    )


def _pair_gaps(model: ModularPhysiologyNetwork, low: torch.Tensor, high: torch.Tensor) -> tuple[float, float]:
    """(|Δ fasting glucose|, |Δ glucose increment 90 min after the meal|)."""
    with torch.no_grad():
        fast_low = _roll(model, low, _FAST_STEPS, [])
        fast_high = _roll(model, high, _FAST_STEPS, [])
        meal_low = _roll(model, low, _MEAL_STEPS, _MEAL)
        meal_high = _roll(model, high, _MEAL_STEPS, _MEAL)
    fasting = abs(float(fast_high[-1, _GI]) - float(fast_low[-1, _GI]))
    inc_low = float(meal_low[_MEAL_AT + _POST, _GI] - meal_low[_MEAL_AT - 1, _GI])
    inc_high = float(meal_high[_MEAL_AT + _POST, _GI] - meal_high[_MEAL_AT - 1, _GI])
    return fasting, abs(inc_high - inc_low)


def _grad_norm(grads: tuple[torch.Tensor, ...]) -> float:
    return math.sqrt(sum(float(grad.detach().pow(2).sum()) for grad in grads))


def _glucose_grads(
    model: ModularPhysiologyNetwork,
    embedding: torch.Tensor,
    steps: int,
    meals: list,
    index: int,
    heads: list[list[torch.nn.Parameter]],
) -> tuple[torch.Tensor, list[float]]:
    trajectory = _roll(model, embedding, steps, meals)
    flat = [param for head in heads for param in head]
    grads = torch.autograd.grad(trajectory[index, _GI], [embedding, *flat])
    norms: list[float] = []
    cursor = 1
    for head in heads:
        norms.append(_grad_norm(grads[cursor:cursor + len(head)]))
        cursor += len(head)
    return grads[0].detach(), norms


def _off_axis(grad: torch.Tensor) -> float:
    kept = grad.clone()
    kept[_GB_AXIS] = 0.0
    kept[_SI_AXIS] = 0.0
    return float(kept.norm())


def measure() -> dict[str, float]:
    """Trajectory gaps and the fasting-vs-meal embedding gradients.

    ``cold_*`` is the network as constructed (Si and Gb heads still closed).
    The other fields are the same laws with the two heads given orthogonal reads.
    """
    cold = _fresh()
    for param in (
        *cold.metabolic.insulin_sens_net.parameters(),
        *cold.metabolic.glucose_baseline_net.parameters(),
    ):
        param.requires_grad_(True)
    closed = torch.zeros(EMBEDDING_DIM, requires_grad=True)
    si_params = list(cold.metabolic.insulin_sens_net.parameters())
    gb_params = list(cold.metabolic.glucose_baseline_net.parameters())
    cold_fast_emb, (cold_fast_gb,) = _glucose_grads(
        cold, closed, _FAST_STEPS, [], _FAST_STEPS - 1, [gb_params])
    cold_meal_emb, (cold_meal_si,) = _glucose_grads(
        cold, closed, _MEAL_STEPS, _MEAL, _MEAL_AT + _POST, [si_params])

    model = _fresh()
    _open_si_and_gb(model)
    for param in (
        *model.metabolic.insulin_sens_net.parameters(),
        *model.metabolic.glucose_baseline_net.parameters(),
    ):
        param.requires_grad_(True)
    si_low, si_high = _person(0.0, -1.0), _person(0.0, 1.0)
    gb_low, gb_high = _person(-1.0, 0.0), _person(1.0, 0.0)
    si_lo, si_hi = _decode(model, si_low), _decode(model, si_high)
    gb_lo, gb_hi = _decode(model, gb_low), _decode(model, gb_high)
    si_fast, si_inc = _pair_gaps(model, si_low, si_high)
    gb_fast, gb_inc = _pair_gaps(model, gb_low, gb_high)

    probe = _person(0.4, 0.4).requires_grad_(True)
    si_params = list(model.metabolic.insulin_sens_net.parameters())
    gb_params = list(model.metabolic.glucose_baseline_net.parameters())
    grad_fast, (gb_head_fast, si_head_fast) = _glucose_grads(
        model, probe, _FAST_STEPS, [], _FAST_STEPS - 1, [gb_params, si_params])
    grad_meal, (gb_head_meal, si_head_meal) = _glucose_grads(
        model, probe, _MEAL_STEPS, _MEAL, _MEAL_AT + _POST, [gb_params, si_params])
    cosine = float(torch.nn.functional.cosine_similarity(grad_fast, grad_meal, dim=0))
    return {
        "si_pair_fasting_gap": si_fast,
        "si_pair_increment_gap": si_inc,
        "gb_pair_fasting_gap": gb_fast,
        "gb_pair_increment_gap": gb_inc,
        "cosine": cosine,
        "grad_fast_norm": float(grad_fast.norm()),
        "grad_meal_norm": float(grad_meal.norm()),
        "grad_fast_gb": float(grad_fast[_GB_AXIS]),
        "grad_fast_si": float(grad_fast[_SI_AXIS]),
        "grad_meal_gb": float(grad_meal[_GB_AXIS]),
        "grad_meal_si": float(grad_meal[_SI_AXIS]),
        "grad_fast_off_axis": _off_axis(grad_fast),
        "grad_meal_off_axis": _off_axis(grad_meal),
        "gb_head_grad_fast": gb_head_fast,
        "si_head_grad_fast": si_head_fast,
        "gb_head_grad_meal": gb_head_meal,
        "si_head_grad_meal": si_head_meal,
        "cold_grad_fast_norm": float(cold_fast_emb.norm()),
        "cold_grad_meal_norm": float(cold_meal_emb.norm()),
        "cold_gb_head_grad_fast": cold_fast_gb,
        "cold_si_head_grad_meal": cold_meal_si,
        "si_ratio": si_hi["si"] / si_lo["si"],
        "gb_span": gb_hi["gb"] - gb_lo["gb"],
        "si_pair_decode_leak": max(abs(si_lo[k] - si_hi[k]) for k in si_lo if k != "si"),
        "gb_pair_decode_leak": max(abs(gb_lo[k] - gb_hi[k]) for k in gb_lo if k != "gb"),
    }


def check(report: dict[str, float]) -> None:
    """The claims a recovery probe can treat as the structural result."""
    if report["si_pair_decode_leak"] > 1e-4 or report["gb_pair_decode_leak"] > 1e-4:
        raise AssertionError(
            "the two people do not differ in only one decode "
            f"(Si leak {report['si_pair_decode_leak']}, Gb leak {report['gb_pair_decode_leak']})"
        )
    if report["si_ratio"] < 3.0 or report["gb_span"] < 40.0:
        raise AssertionError(
            f"heads did not separate the people (Si ratio {report['si_ratio']}, Gb span {report['gb_span']})"
        )
    if report["cold_si_head_grad_meal"] < 1.0:
        raise AssertionError(
            "Si head is invisible to post-meal glucose "
            f"(parameter gradient {report['cold_si_head_grad_meal']})"
        )
    if report["si_head_grad_meal"] < 1.0 or report["grad_meal_si"] > -1.0:
        raise AssertionError(
            "a meal does not reach the Si head "
            f"(head {report['si_head_grad_meal']}, axis {report['grad_meal_si']})"
        )
    if report["gb_head_grad_fast"] < 1.0 or report["grad_fast_gb"] < 1.0:
        raise AssertionError(
            "a fast does not reach the Gb head "
            f"(head {report['gb_head_grad_fast']}, axis {report['grad_fast_gb']})"
        )
    if not report["si_pair_increment_gap"] > 10.0 * report["si_pair_fasting_gap"]:
        raise AssertionError(
            "Si did not move the meal increment more than the fasting level "
            f"(increment {report['si_pair_increment_gap']}, fast {report['si_pair_fasting_gap']})"
        )
    if not report["gb_pair_fasting_gap"] > 2.0 * report["gb_pair_increment_gap"]:
        raise AssertionError(
            "Gb did not move the fasting level more than the meal increment "
            f"(fast {report['gb_pair_fasting_gap']}, increment {report['gb_pair_increment_gap']})"
        )
    if abs(report["grad_meal_gb"]) < 2.0 * abs(report["grad_meal_si"]):
        raise AssertionError(
            "a meal gradient is not still dominated by Gb "
            f"(Gb axis {report['grad_meal_gb']}, Si axis {report['grad_meal_si']}, "
            f"cosine {report['cosine']})"
        )


def main() -> None:
    report = measure()
    visible_si = report["si_head_grad_meal"] > 1.0 and report["grad_meal_si"] < -1.0
    visible_gb = report["gb_head_grad_fast"] > 1.0 and report["grad_fast_gb"] > 1.0
    print(f"si_visible_to_meal={'yes' if visible_si else 'no'}")
    print(f"gb_visible_to_fast={'yes' if visible_gb else 'no'}")
    for key in (
        "cosine",
        "grad_fast_norm", "grad_meal_norm",
        "grad_fast_gb", "grad_fast_si",
        "grad_meal_gb", "grad_meal_si",
        "gb_head_grad_fast", "si_head_grad_meal",
        "gb_head_grad_meal", "si_head_grad_fast",
        "si_pair_fasting_gap", "si_pair_increment_gap",
        "gb_pair_fasting_gap", "gb_pair_increment_gap",
        "cold_si_head_grad_meal", "cold_gb_head_grad_fast",
        "cold_grad_fast_norm", "cold_grad_meal_norm",
    ):
        print(f"{key}={report[key]:.6e}")
    check(report)


if __name__ == "__main__":
    main()
