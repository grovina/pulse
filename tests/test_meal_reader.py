"""The gut reads what a meal is made of (``pulse/modules/food.py``).

Off, or on with meals that state nothing it reads, the model is the model without it,
bit for bit. On and freshly built it changes nothing (to float rounding, since the
stomach is then stepped meal by meal). Once its last layer moves, two meals with equal
grams and different details give different curves, and equal details equal curves; the
gradient of a cohort loss reaches it through the batched evidence step.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pulse.cohort_loss import _meals_from_spec, rollout_arms
from pulse.gastric import simulate, simulate_by_meal, swallow_by_meal, swallow_from_meals
from pulse.knowledge.cohort_statistics import ALL_COHORT_STATISTICS
from pulse.knowledge.cohort_types import (
    CohortArmSpec, CohortStatisticSpec, MealDetails, StatisticKind, StatisticWindow,
)
from pulse.model import ModularPhysiologyNetwork, integrate
from pulse.modules.food import FEATURE_DIM, FEATURE_NAMES, LOG_BOUND, MealReader, meal_features
from pulse.modules.gut import MealEvent
from pulse.train import (
    _load_spec_train_args, build_arg_parser, resolved_train_config, spec_config_divergence,
)
from pulse.training import SignalContext
from pulse.training.evidence_step import EvidenceTerm, accumulate_evidence_step
from pulse.types import MARKER_INDEX, NORM_CENTER

ROOT = Path(__file__).resolve().parents[1]
G = MARKER_INDEX["glucose"]
FED = 30
BUTTER = MealDetails(fat_g_by_type=(("saturated", 31.0), ("monounsaturated", 13.0), ("polyunsaturated", 2.0)),
                     fat_source="butter", form="solid", protein_source="dairy")
OIL = MealDetails(fat_g_by_type=(("saturated", 7.0), ("monounsaturated", 30.0), ("polyunsaturated", 9.0)),
                  fat_source="olive oil", form="solid", protein_source="dairy")
MEAL = ((10.0, 60.0, 50.0, 20.0),)


def _model(meal_details: bool, seed: int = 3) -> ModularPhysiologyNetwork:
    torch.manual_seed(seed)
    return ModularPhysiologyNetwork(
        metabolic_hidden=12, appetite_hidden=8, stress_hidden=8,
        cardiovascular_hidden=12, thermoreg_hidden=8, respiratory_hidden=8,
        gut_hidden=8, hepatobiliary_hidden=8, meal_details=meal_details,
    )


def _wake(model: ModularPhysiologyNetwork, scale: float = 1.0) -> ModularPhysiologyNetwork:
    g = torch.Generator().manual_seed(11)
    with torch.no_grad():
        last = model.meal_reader.net[-1]
        last.weight.copy_(scale * torch.randn(last.weight.shape, generator=g))
        last.bias.copy_(0.3 * scale * torch.randn(last.bias.shape, generator=g))
    return model


def _arm(label: str, details=(), minutes: int = 120, meals=MEAL, **kw) -> CohortArmSpec:
    return CohortArmSpec(label=label, duration_min=minutes, start_hour=8.0, meals=meals,
                         meal_details=tuple(details), **kw)


def _rows(model, arms, n: int = 3, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    emb = 0.3 * torch.randn(n, model.embedding_dim, generator=g)
    state = torch.tensor(NORM_CENTER).expand(n, -1).contiguous()
    return [(a, emb, state) for a in arms]


def _roll(model, arms, **kw):
    with torch.no_grad():
        return rollout_arms(model, _rows(model, arms, **kw))


def _corpus_arm(paper: str, label: str, minutes: int = 150) -> CohortArmSpec:
    arm = next(a for s in ALL_COHORT_STATISTICS if s.paper == paper for a in s.arms if a.label == label)
    return dataclasses.replace(arm, duration_min=minutes)


# --- features --------------------------------------------------------------------

def _event(details, carbs=60.0, fats=50.0, proteins=20.0):
    return MealEvent(time=0.0, carbs=carbs, fats=fats, proteins=proteins, details=details)


def test_features_have_a_fixed_layout_and_a_mask_for_every_field():
    assert FEATURE_DIM == len(FEATURE_NAMES) == 25
    full = MealDetails(fibre_g=15.0, sugars_g=30.0, fat_g_by_type=BUTTER.fat_g_by_type,
                       protein_source="whey", form="liquid", food="x", fat_source="butter")
    f = dict(zip(FEATURE_NAMES, meal_features(_event(full))))
    assert f["fibre_share"] == pytest.approx(15.0 / 75.0) and f["sugars_share"] == pytest.approx(0.5)
    assert f["saturated_share"] == pytest.approx(31.0 / 50.0)
    assert f["protein_whey"] == 1.0 and sum(v for k, v in f.items() if k.startswith("protein_")) == 1.0
    assert f["form_liquid"] == 1.0
    assert all(f[k] == 1.0 for k in FEATURE_NAMES if k.startswith("stated_"))


def test_not_stated_is_not_zero():
    zero = dict(zip(FEATURE_NAMES, meal_features(_event(MealDetails(fibre_g=0.0, form="solid")))))
    unstated = dict(zip(FEATURE_NAMES, meal_features(_event(MealDetails(form="solid")))))
    assert zero["fibre_share"] == unstated["fibre_share"] == 0.0
    assert zero["stated_fibre"] == 1.0 and unstated["stated_fibre"] == 0.0


def test_free_text_alone_is_no_feature():
    """``fat_source`` and ``food`` are names, not read: a meal stating only them is a
    meal with no details to the reader."""
    assert meal_features(_event(None)) is None
    assert meal_features(_event(MealDetails())) is None
    assert meal_features(_event(MealDetails(fat_source="coconut oil", food="toast"))) is None


def test_an_arms_details_ride_on_its_meals():
    meals = _meals_from_spec(((0.0, 60.0, 50.0, 20.0), (300.0, 50.0, 10.0, 10.0)), (BUTTER, MealDetails()))
    assert meals[0].details == BUTTER and meals[1].details is None
    assert all(m.details is None for m in _meals_from_spec(MEAL))


# --- the reader ------------------------------------------------------------------

def test_a_fresh_reader_is_the_identity_and_its_size_is_small():
    reader = MealReader()
    assert sum(p.numel() for p in reader.parameters()) == 262
    speed, shift = reader(torch.rand(5, FEATURE_DIM))
    assert torch.equal(speed, torch.ones(5, 3)) and torch.equal(shift, torch.zeros(5, 3))


def test_the_modifiers_are_bounded():
    reader = MealReader()
    with torch.no_grad():
        for p in reader.parameters():
            p.normal_(0.0, 100.0)
    with torch.no_grad():
        speed, shift = reader(torch.rand(200, FEATURE_DIM))
    assert float(speed.min()) >= 1 / 3 - 1e-6 and float(speed.max()) <= 3 + 1e-5
    assert float(shift.abs().max()) <= LOG_BOUND + 1e-6


# --- the stomach meal by meal -----------------------------------------------------

def _two_meals():
    return [MealEvent(5.0, 60.0, 30.0, 20.0), MealEvent(40.0, 25.0, 5.0, 30.0)]


def test_at_speed_one_the_stomach_by_meal_is_the_shared_stomach():
    meals = _two_meals()
    left, filled = simulate(swallow_from_meals(400, meals, torch.device("cpu")), 2.13)
    by_meal, filled_k = simulate_by_meal(
        swallow_by_meal(400, [meals], 2, torch.device("cpu")), 2.13, torch.ones(1, 2, 3))
    assert torch.allclose(by_meal.sum(dim=1)[0], left, atol=1e-4)
    assert torch.allclose(filled_k[0], filled, atol=1e-4)


def test_every_meal_leaves_whole_at_any_speed_and_a_faster_one_leaves_sooner():
    meals = _two_meals()
    speed = torch.tensor([[[2.5, 0.4, 1.0], [0.5, 3.0, 1.0 / 3]]])
    out, _ = simulate_by_meal(swallow_by_meal(1500, [meals], 2, torch.device("cpu")), 2.13, speed)
    eaten = torch.tensor([[m.carbs, m.proteins, m.fats] for m in meals])
    assert torch.allclose(out[0].sum(dim=1), eaten, rtol=1e-4)
    slow, _ = simulate_by_meal(swallow_by_meal(1500, [meals[:1]], 1, torch.device("cpu")), 2.13,
                               torch.full((1, 1, 3), 0.5))
    fast, _ = simulate_by_meal(swallow_by_meal(1500, [meals[:1]], 1, torch.device("cpu")), 2.13,
                               torch.full((1, 1, 3), 2.0))
    half = 0.5 * eaten[0].sum()
    t_slow = int((slow[0, 0].sum(-1).cumsum(0) < half).sum())
    t_fast = int((fast[0, 0].sum(-1).cumsum(0) < half).sum())
    assert t_fast < t_slow


# --- the switch --------------------------------------------------------------------

def test_off_is_the_model_without_a_reader():
    off = _model(False)
    assert off.meal_reader is None and "meal_details" not in off.constructor_kwargs
    on = _model(True)
    assert on.constructor_kwargs["meal_details"] is True
    # The reader is built last, so the same seed gives every other weight unchanged.
    off_state, on_state = off.state_dict(), on.state_dict()
    assert set(on_state) - set(off_state) == {k for k in on_state if k.startswith("meal_reader.")}
    assert all(torch.equal(off_state[k], on_state[k]) for k in off_state)
    assert sum(p.numel() for p in on.parameters()) - sum(p.numel() for p in off.parameters()) == 262


def test_a_checkpoint_rebuilds_the_reader():
    on = _wake(_model(True))
    again = ModularPhysiologyNetwork.from_checkpoint(
        {"model_state": on.state_dict(), "model_config": on.constructor_kwargs})
    assert again.meal_reader is not None
    assert torch.equal(again.meal_reader.net[-1].weight, on.meal_reader.net[-1].weight)
    plain = ModularPhysiologyNetwork.from_checkpoint(
        {"model_state": _model(False).state_dict(), "model_config": _model(False).constructor_kwargs})
    assert plain.meal_reader is None


PROTOCOLS = {
    "mixed meal": lambda: _arm("mixed", (BUTTER,)),
    "two meals": lambda: _arm("two", (BUTTER, OIL), meals=((10.0, 60.0, 50.0, 20.0), (70.0, 30.0, 5.0, 10.0))),
    "almoraie-2026 control meal": lambda: _corpus_arm("almoraie-2026", "control_meal"),
    "lecerf-2015 reduced dessert": lambda: _corpus_arm("lecerf-2015", "dessert_reduced", 126),
}


@pytest.mark.parametrize("name", PROTOCOLS)
def test_off_and_on_without_details_are_bit_for_bit_the_same(name):
    arm = PROTOCOLS[name]()
    bare = dataclasses.replace(arm, meal_details=())
    (off_full,) = _roll(_model(False), [arm])
    (off_bare,) = _roll(_model(False), [bare])
    (on_bare,) = _roll(_wake(_model(True)), [bare])
    assert torch.equal(off_full, off_bare)   # off does not read the details
    assert torch.equal(on_bare, off_bare)    # on, a meal that states nothing is unchanged


@pytest.mark.parametrize("name", PROTOCOLS)
def test_a_fresh_reader_changes_nothing(name):
    arm = PROTOCOLS[name]()
    (off,) = _roll(_model(False), [arm])
    (on,) = _roll(_model(True), [arm])
    assert torch.allclose(on, off, rtol=1e-5, atol=1e-4)


def test_integrate_reads_the_details_shared_and_per_member():
    """Without precomputed gut outputs ``integrate`` builds them itself
    (``model._meal_outputs``); a member with no details keeps the shared stomach exactly."""
    model = _wake(_model(True))
    emb = 0.3 * torch.randn(2, model.embedding_dim, generator=torch.Generator().manual_seed(1))
    state = torch.tensor(NORM_CENTER).expand(2, -1).contiguous()
    butter = _meals_from_spec(MEAL, (BUTTER,))
    oil = _meals_from_spec(MEAL, (OIL,))
    bare = _meals_from_spec(MEAL)
    off = _model(False)
    with torch.no_grad():
        mixed = integrate(model, state, emb, 90, meals=[butter, bare], start_time_minutes=480.0)
        plain = integrate(off, state, emb, 90, meals=[bare, bare], start_time_minutes=480.0)
        alone = integrate(model, state[:1], emb[:1], 90, meals=butter, start_time_minutes=480.0)
        other = integrate(model, state[:1], emb[:1], 90, meals=oil, start_time_minutes=480.0)
        pointwise = integrate(model, state[0], emb[0], 90, meals=butter, start_time_minutes=480.0,
                              planned=False)
    assert torch.equal(mixed[1], plain[1])
    assert torch.allclose(mixed[0], alone[0], rtol=1e-5, atol=1e-4)
    assert torch.allclose(pointwise, alone[0], rtol=1e-4, atol=1e-3)
    assert float((alone[0, :, G] - other[0, :, G]).abs().max()) > 0.1


# --- what it can tell apart --------------------------------------------------------

def test_equal_grams_and_other_details_give_another_curve_and_equal_details_the_same():
    model = _wake(_model(True))
    butter, oil, butter_again = _arm("b", (BUTTER,)), _arm("o", (OIL,)), _arm("b2", (BUTTER,))
    a, b = _roll(model, [butter, oil])
    assert float((a[..., G] - b[..., G]).abs().max()) > 0.1
    # Equality is asked of rollouts on the same rows (rows of one batch may differ in the
    # last bit of a matmul).
    (c,) = _roll(model, [butter_again])
    (a_alone,) = _roll(model, [butter])
    assert torch.equal(a_alone, c)
    off = _model(False)
    assert torch.equal(_roll(off, [butter])[0], _roll(off, [oil])[0])


def test_two_corpus_arms_that_differ_only_in_fat_types_part():
    """``almoraie-2026``: the same grams, part of the fat as pomegranate-seed oil."""
    control, pso = _corpus_arm("almoraie-2026", "control_meal"), _corpus_arm("almoraie-2026", "pso_meal")
    assert control.meals == pso.meals
    a, b = _roll(_wake(_model(True)), [control, pso])
    assert float((a[..., G] - b[..., G]).abs().max()) > 1e-3
    off = _model(False)
    assert torch.equal(_roll(off, [control])[0], _roll(off, [pso])[0])


def test_a_fat_source_alone_is_not_read():
    """``wong-2025``'s butter, coconut and vegetable-oil days state the fat only as a
    name, so the reader cannot tell them apart."""
    butter, coconut = _corpus_arm("wong-2025", "butter_day", 60), _corpus_arm("wong-2025", "coconut_day", 60)
    assert butter.meals == coconut.meals and butter.meal_details != coconut.meal_details
    model = _wake(_model(True))
    assert torch.equal(_roll(model, [butter])[0], _roll(model, [coconut])[0])


# --- the gradient ------------------------------------------------------------------

def _spec(name, arms, **kw):
    return CohortStatisticSpec(
        name=name, source="t", description="", arms=tuple(arms), marker_id="glucose",
        kind=kw.pop("kind", StatisticKind.MEAN_IN_WINDOW), window=StatisticWindow(10, arms[0].duration_min),
        target=kw.pop("target", 120.0), sigma=5.0, **kw,
    )


@pytest.mark.parametrize("woken", [False, True], ids=["fresh", "woken"])
def test_a_cohort_loss_reaches_the_reader_through_the_batched_evidence_step(woken):
    model = _model(True)
    if woken:
        _wake(model, 0.3)
    terms = [
        EvidenceTerm(_spec("one", [_arm("butter", (BUTTER,), minutes=60)]), 1.0, 2),
        EvidenceTerm(_spec("diff", [_arm("butter", (BUTTER,), minutes=60), _arm("oil", (OIL,), minutes=60)],
                           kind=StatisticKind.DELTA_MEANS, target=3.0), 1.0, 2),
    ]
    params = list(model.parameters())
    ctx = SignalContext(
        epoch=0, total_epochs=1, rng=np.random.default_rng(0), device=torch.device("cpu"),
        optimizer=torch.optim.SGD(params, lr=1e-3), params=params, grad_clip=10.0,
    )
    model.zero_grad()
    accumulate_evidence_step(model, ctx, terms, rng=np.random.default_rng(0), device="cpu", fed_day_minutes=FED)
    last = model.meal_reader.net[-1].weight.grad
    assert last is not None and torch.isfinite(last).all() and float(last.abs().sum()) > 0.0
    first = model.meal_reader.net[0].weight.grad
    if woken:
        assert first is not None and float(first.abs().sum()) > 0.0


# --- the trainer flag --------------------------------------------------------------

def test_the_trainer_flag_parses_and_is_recorded():
    parser = build_arg_parser()
    assert parser.parse_args([]).meal_details is False
    on = parser.parse_args(["--meal-details"])
    assert on.meal_details is True and resolved_train_config(on)["meal_details"] is True


def test_the_recipe_loads_without_divergence_and_without_the_reader():
    spec_path = str(ROOT / "train" / "spec.json")
    spec_args = _load_spec_train_args(spec_path)
    assert not any(a.startswith("--meal-details") for a in spec_args)
    parser = build_arg_parser()
    spec_only = parser.parse_args(spec_args + ["--spec", spec_path])
    assert spec_only.meal_details is False
    assert spec_config_divergence(spec_only, parser.parse_args(spec_args + ["--spec", spec_path])) == []
    forked = parser.parse_args(spec_args + ["--spec", spec_path, "--meal-details"])
    assert [d[0] for d in spec_config_divergence(spec_only, forked)] == ["meal_details"]
    json.dumps(resolved_train_config(forked), default=str)


def test_train_defaults_to_off():
    import inspect

    from pulse.train import train

    assert inspect.signature(train).parameters["meal_details"].default is False
