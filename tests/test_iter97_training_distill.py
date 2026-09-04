"""Iter 97 (training): the distillation's level anchors rolled out with no meal (review 4.7)."""

from __future__ import annotations

import inspect
import os

import numpy as np
import pytest
import torch

import pulse

REPO = os.path.dirname(os.path.dirname(os.path.abspath(pulse.__file__)))
assert pulse.__file__.startswith(REPO), pulse.__file__

from pulse.model import ModularPhysiologyNetwork, integrate  # noqa: E402
from pulse.training import ColdModelDistillationSignal, SignalContext, WeightSchedule  # noqa: E402
from pulse.types import EMBEDDING_DIM  # noqa: E402

MARKERS = ("cck", "gallbladder_bile", "intestinal_bile", "bile_acids", "cortisol")


def _sig(**kw) -> ColdModelDistillationSignal:
    base = dict(weight=WeightSchedule(0.3), markers=MARKERS, pool="synthetic", pool_size=1,
                mode="anchored", anchor_window=60, anchor_samples=6, calib_steps=1, calib_warm_steps=1)
    base.update(kw)
    return ColdModelDistillationSignal(**base)


def test_level_anchor_rollouts_carry_the_duodenal_stimulus() -> None:
    sig = _sig()
    proto = sig._protocols[0]  # standard_3meal
    model = ModularPhysiologyNetwork().eval()
    seen: list[float] = []
    h = model.hepatobiliary.register_forward_hook(
        lambda m, a, o: seen.append(float(a[1].detach().abs().max())),
    )
    try:
        with torch.no_grad():
            sig._level_terms(model, proto, torch.zeros(EMBEDDING_DIM), torch.device("cpu"),
                             rng=np.random.default_rng(0))
    finally:
        h.remove()
    # Review 4.7 measured max |duodenal| = 0.0000 across every level window.
    assert max(seen) > 0.0


@pytest.mark.skipif(
    "duodenal_outputs" not in inspect.signature(integrate).parameters,
    reason="student layer's integrate(duodenal_outputs=) not merged yet",
)
def test_level_anchor_passes_precomputed_duodenal_outputs(monkeypatch) -> None:
    import pulse.training.cold_model_distillation_signal as mod
    captured: list[dict] = []
    real = mod.integrate

    def spy(*a, **kw):
        captured.append(kw)
        return real(*a, **kw)

    monkeypatch.setattr(mod, "integrate", spy)
    monkeypatch.setattr(mod, "_INTEGRATE_HAS_DUODENAL", True)
    sig = _sig()
    model = ModularPhysiologyNetwork().eval()
    with torch.no_grad():
        sig._level_terms(model, sig._protocols[0], torch.zeros(EMBEDDING_DIM), torch.device("cpu"))
    assert captured and all(kw.get("duodenal_outputs") is not None for kw in captured)
    assert all(kw["duodenal_outputs"].shape[0] == kw["duodenal_outputs"].shape[0] for kw in captured)


def test_level_window_starts_are_jittered_per_epoch() -> None:
    sig = _sig()
    proto = sig._protocols[0]
    model = ModularPhysiologyNetwork().eval()
    starts_seen: list[tuple[int, ...]] = []
    import pulse.training.cold_model_distillation_signal as mod
    real = mod.integrate

    def spy(model_, init, emb, w, **kw):
        starts_seen.append(int(kw["start_time_minutes"]))
        return real(model_, init, emb, w, **kw)

    mod.integrate = spy
    try:
        with torch.no_grad():
            sig._level_terms(model, proto, torch.zeros(EMBEDDING_DIM), torch.device("cpu"),
                             rng=np.random.default_rng(1))
            a = tuple(starts_seen); starts_seen.clear()
            sig._level_terms(model, proto, torch.zeros(EMBEDDING_DIM), torch.device("cpu"),
                             rng=np.random.default_rng(2))
            b = tuple(starts_seen); starts_seen.clear()
            sig._level_terms(model, proto, torch.zeros(EMBEDDING_DIM), torch.device("cpu"), rng=None)
            c = tuple(starts_seen)
    finally:
        mod.integrate = real
    assert a != b, "starts must vary with the epoch rng"
    assert len(c) == 6 and c == tuple(sorted(c))  # deterministic without an rng


def test_level_band_is_a_dead_zone() -> None:
    sig0 = _sig(anchor_level_band=0.0)
    sig1 = _sig(anchor_level_band=0.5)
    torch.manual_seed(0)
    model = ModularPhysiologyNetwork().eval()
    emb = torch.zeros(EMBEDDING_DIM)
    with torch.no_grad():
        t0 = sig0._level_terms(model, sig0._protocols[0], emb, torch.device("cpu"))
        t1 = sig1._level_terms(model, sig1._protocols[0], emb, torch.device("cpu"))
    for m in t0:
        assert float(t1[m]) <= float(t0[m]) + 1e-9
