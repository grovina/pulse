"""Iter 97 (training): the distillation's level anchors rolled out with no meal (review 4.7)."""

from __future__ import annotations

import os

import numpy as np
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
    step = model.hepatobiliary.step

    # Every rate evaluation — planned rollout or pointwise forward — goes through
    # the module's ``step``; its coupling carries the duodenal (fat, protein) drive.
    def spy(state, coupling, *args, **kwargs):
        seen.append(float(coupling.detach().abs().max()))
        return step(state, coupling, *args, **kwargs)

    model.hepatobiliary.step = spy
    with torch.no_grad():
        sig._level_terms(model, proto, torch.zeros(EMBEDDING_DIM), torch.device("cpu"),
                         rng=np.random.default_rng(0))
    # Review 4.7 measured max |duodenal| = 0.0000 across every level window.
    assert max(seen) > 0.0


def test_level_anchor_passes_precomputed_duodenal_outputs(monkeypatch) -> None:
    import pulse.training.cold_model_distillation_signal as mod
    captured: list[dict] = []
    real = mod.integrate

    def spy(*a, **kw):
        captured.append(kw)
        return real(*a, **kw)

    monkeypatch.setattr(mod, "integrate", spy)
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
        # Iter 108: the windows of one length roll as one batched call, one start each.
        starts = torch.as_tensor(kw["start_time_minutes"]).reshape(-1)
        starts_seen.extend(int(s) for s in starts.tolist())
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
