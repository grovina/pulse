"""PLAN A1 — Gb decodes in log space (the metabolic half of item A1).

Companion to ``test_a1_log_setpoints.py``, which covers the vitals. The teacher
draws the fasting glucose setpoint LOGNORMALLY and then clips it
(``knowledge/full_body.py``: ``clip(vary(p.Gb, 0.25, ir=0.50), 70, 130)``), so
``PatientParams()``'s 95 mg/dL is the population MEDIAN and the population mean
sits above it. Through iter 109 the student decoded

    Gb = 95 + 30 · 2.2 · tanh(head)          (iter 90, additive)

which gives E[Gb] = 95 EXACTLY for any zero-mean embedding population — the mean
of a right-skewed population equal to its median. PLAN §2 fixes zero as the
median person, so that decode cannot also be right in the mean; PLAN §1 names the
remedy, "the decoder family must match the generative family". The decode is now

    Gb = 95 · exp(0.45 · tanh(head))         ∈ [60.6, 148.9] mg/dL

which gets both by Jensen with no loss term. These tests pin:

  * the reachable span, and that it covers the teacher's clipped [70, 130] with
    margin at both ends (and the benchmark's 60–120);
  * Gb(0) = 95, so zero is still the median person and every textbook scenario,
    fixed-point test and default patient keeps its meaning;
  * the point of the change — averaging a symmetric code population puts the mean
    strictly ABOVE Gb(0), which the additive decode cannot do;
  * ``glucose_setpoint_z`` is the module's own raw→z conversion, so
    ``SetpointSupervisionSignal`` no longer needs to know the decode's shape;
  * Gb is still an exact fixed point of the glucose balance at the decoded value,
    for setpoints the additive bound reached and for the new span's extremes.
"""

from __future__ import annotations

import math
import unittest

import torch

from pulse.model import ModularPhysiologyNetwork
from pulse.modules import metabolic as M
from pulse.modules.base import compute_time_features
from pulse.types import (
    EMBEDDING_DIM, MARKER_INDEX as MI, MODULE_MARKER_INDICES, NORM_CENTER, NORM_SCALE,
)

_CENTER = float(NORM_CENTER[MI["glucose"]])     # 95 mg/dL
_SCALE = float(NORM_SCALE[MI["glucose"]])       # 30 mg/dL
# The teacher's clip, and the benchmark span iter 90 widened the bound for.
_TEACHER_CLIP = (70.0, 130.0)
_BENCHMARK = (60.0, 120.0)


def _model(seed: int = 0) -> ModularPhysiologyNetwork:
    torch.manual_seed(seed)
    m = ModularPhysiologyNetwork(
        metabolic_hidden=16, appetite_hidden=16, stress_hidden=16, cardiovascular_hidden=16,
        thermoreg_hidden=16, respiratory_hidden=16, gut_hidden=16, hepatobiliary_hidden=16)
    m.eval()
    return m


def _head_input_dim(m: ModularPhysiologyNetwork) -> int:
    return int(m.metabolic.glucose_baseline_net[0].in_features)


def _odd_head(m: ModularPhysiologyNetwork, seed: int, std: float = 1.5) -> None:
    """Give the Gb head real authority AND make it an ODD function of its input.

    Both Linear biases zeroed and ``tanh`` odd ⇒ ``head(−e) = −head(e)`` exactly,
    so a code population closed under negation is exactly symmetric in the head's
    output and the only asymmetry left in Gb is the decode's own convexity. That
    is the quantity under test; a random bias would hide it behind a shift.
    """
    g = torch.Generator().manual_seed(seed)
    net = m.metabolic.glucose_baseline_net
    with torch.no_grad():
        for layer in (net[0], net[-1]):
            layer.weight.copy_(std * torch.randn(layer.weight.shape, generator=g))
            layer.bias.zero_()


class TestReachableSpan(unittest.TestCase):
    def test_the_span_is_the_one_the_comment_states(self) -> None:
        lo, hi = _CENTER * math.exp(-M._GB_LOG_MAX), _CENTER * math.exp(M._GB_LOG_MAX)
        self.assertAlmostEqual(lo, 60.57, places=2)
        self.assertAlmostEqual(hi, 148.99, places=2)
        m = _model(0)
        d = _head_input_dim(m)
        net = m.metabolic.glucose_baseline_net
        with torch.no_grad():
            net[-1].weight.zero_()
            for sign, edge in ((-1.0, lo), (+1.0, hi)):
                # tanh saturates: a large bias drives the decode to the bound.
                net[-1].bias.fill_(sign * 40.0)
                got = float(m.metabolic.glucose_setpoint_raw(torch.zeros(1, d)))
                self.assertAlmostEqual(got, edge, places=4)

    def test_the_span_covers_the_teachers_clipped_range_with_margin(self) -> None:
        """[60.6, 148.9] against the teacher's clipped [70, 130]: 9.4 mg/dL of
        headroom below and 19.0 above, so no sampled patient's Gb is unreachable
        and the decode is not saturated at either end of the population it fits.

        The floor is 0.6 mg/dL ABOVE the 60 mg/dL low end of the benchmark's
        observed fasting glucose (iter 90 widened the additive bound for that
        range). It is not a gap: Gb is the SETPOINT, and a trajectory sits below
        it whenever the liver pool is drawn down — the long-fast floor is
        ``gng/k_ii``, the same absolute number for every Gb (module docstring,
        review item 3.3). 60 mg/dL is reachable as an observation at any Gb in
        this span; what the span has to contain is the teacher's draw.
        """
        lo, hi = _CENTER * math.exp(-M._GB_LOG_MAX), _CENTER * math.exp(M._GB_LOG_MAX)
        self.assertLess(lo, _TEACHER_CLIP[0] - 5.0)
        self.assertGreater(hi, _TEACHER_CLIP[1] + 5.0)
        self.assertAlmostEqual(_TEACHER_CLIP[0] - lo, 9.43, places=2)
        self.assertAlmostEqual(hi - _TEACHER_CLIP[1], 18.99, places=2)
        # and most of the benchmark's own observed span, bar the last 0.6 mg/dL
        self.assertLess(lo, _BENCHMARK[0] + 1.0)
        self.assertGreaterEqual(hi, _BENCHMARK[1])

    def test_every_embedding_decodes_to_a_positive_survivable_glucose(self) -> None:
        """Log space means Gb > 0 by construction. The additive ±2.2 z bound reached
        29 mg/dL, which is not a phenotype — iter 96 had already had to clip the
        TEACHER at 70 for exactly that reason, and the student's decoder kept the
        unclipped span."""
        m = _model(1)
        d = _head_input_dim(m)
        _odd_head(m, seed=1, std=4.0)
        g = torch.Generator().manual_seed(2)
        for scale in (0.0, 1.0, 3.0, 8.0, 100.0):
            with torch.no_grad():
                gb = m.metabolic.glucose_setpoint_raw(scale * torch.randn(256, d, generator=g))
            self.assertTrue(bool((gb >= _CENTER * math.exp(-M._GB_LOG_MAX) - 1e-4).all()))
            self.assertTrue(bool((gb <= _CENTER * math.exp(M._GB_LOG_MAX) + 1e-4).all()))


class TestZeroIsTheMedianAndTheMeanIsAbove(unittest.TestCase):
    def test_zero_decodes_to_the_default_patient(self) -> None:
        for seed in range(3):
            m = _model(seed)
            d = _head_input_dim(m)
            with torch.no_grad():
                gb = m.metabolic.glucose_setpoint_raw(torch.zeros(1, d))
                z = m.metabolic.glucose_setpoint_z(torch.zeros(1, d))
            self.assertAlmostEqual(float(gb), _CENTER, places=5)
            self.assertAlmostEqual(float(z), 0.0, places=6)

    def test_the_decode_is_convex_so_a_symmetric_code_has_mean_above_median(self) -> None:
        """The statement without a model: ``d(u) = 95·exp(L·tanh u)`` is strictly
        convex about u = 0, so ``(d(u) + d(−u))/2 = 95·cosh(L·tanh u) > d(0)`` for
        every u ≠ 0. The additive decode gives exactly ``d(0)`` there, which is why
        it could not be right at the median and in the mean at once (PLAN §1)."""
        for u in (0.05, 0.2, 0.5, 1.0, 2.0, 5.0):
            t = math.tanh(u)
            avg = _CENTER * (math.exp(M._GB_LOG_MAX * t) + math.exp(-M._GB_LOG_MAX * t)) / 2.0
            self.assertGreater(avg, _CENTER)
            additive = _CENTER + _SCALE * 2.2 * (t + -t) / 2.0
            self.assertAlmostEqual(additive, _CENTER, places=9)

    def test_a_symmetric_embedding_population_has_mean_gb_above_median_gb(self) -> None:
        """The same thing through the real head: with an odd head and a code set
        closed under negation, the median decodes to 95 and the mean is strictly
        above it — E[Gb] > Gb(0) by Jensen, with no loss term asking for it."""
        for seed in (3, 4, 5):
            m = _model(seed)
            d = _head_input_dim(m)
            _odd_head(m, seed=seed, std=1.5)
            e = torch.randn(4096, d, generator=torch.Generator().manual_seed(seed))
            with torch.no_grad():
                gb = m.metabolic.glucose_setpoint_raw(torch.cat([e, -e], dim=0))
            mean, median = float(gb.mean()), float(gb.median())
            self.assertAlmostEqual(median, _CENTER, delta=0.5, msg=f"seed {seed}")
            self.assertGreater(mean, _CENTER, msg=f"seed {seed}: mean {mean:.3f}")
            # the teacher's own mean/median for Gb is ~1.03 at σ 0.25 pre-clip; the
            # student's is set by the code spread, so pin only the SIGN and a floor
            self.assertGreater(mean / median, 1.001, msg=f"seed {seed}")


class TestTheModuleOwnsTheDecode(unittest.TestCase):
    def test_glucose_setpoint_z_is_the_raw_decode_in_z_units(self) -> None:
        m = _model(6)
        d = _head_input_dim(m)
        _odd_head(m, seed=6, std=2.0)
        e = torch.randn(64, d, generator=torch.Generator().manual_seed(7))
        with torch.no_grad():
            raw = m.metabolic.glucose_setpoint_raw(e)
            z = m.metabolic.glucose_setpoint_z(e)
        torch.testing.assert_close(z, (raw - _CENTER) / _SCALE, atol=1e-6, rtol=1e-6)

    def test_the_setpoint_signal_no_longer_rebuilds_the_decode(self) -> None:
        """A1: ``SetpointSupervisionSignal`` used to compute
        ``_met._GLUCOSE_BASELINE_MAX_Z · tanh(head)`` itself, which is the z-score
        only while the decode is additive — a silent wrong answer the moment the
        family changed. The constant is gone and the signal goes through the
        module, the same hand-off ``setpoints_z`` made for CVS in iter 97."""
        import inspect

        from pulse.training import setpoint_supervision_signal as sig

        self.assertFalse(hasattr(M, "_GLUCOSE_BASELINE_MAX_Z"))
        src = inspect.getsource(sig.SetpointSupervisionSignal.compute)
        self.assertIn("glucose_setpoint_z", src)
        self.assertNotIn("_GLUCOSE_BASELINE_MAX_Z * torch.tanh", src)

    def test_the_signal_fits_gb_targets_across_the_whole_span(self) -> None:
        """End to end through the real signal: it drives the head to a low, a
        median and a high patient's Gb. The z it compares in is the module's, so
        every target inside the span is reachable — and 128 was NOT reachable in
        practice before, not because the additive bound could not express it, but
        because the signal and the module disagreed about what the z meant."""
        import numpy as np

        from pulse.training.setpoint_supervision_signal import SetpointSupervisionSignal
        from pulse.training.signals import SignalContext, WeightSchedule

        m = _model(8)
        targets = {0: {"glucose": 72.0}, 1: {"glucose": 95.0}, 2: {"glucose": 128.0}}
        emb = torch.nn.Embedding(len(targets), EMBEDDING_DIM)
        torch.nn.init.normal_(emb.weight, std=0.3)
        params = list(m.parameters()) + list(emb.parameters())
        sig = SetpointSupervisionSignal(weight=WeightSchedule(1.0), targets=targets)
        opt = torch.optim.Adam(params, lr=0.03)
        for _ in range(250):
            opt.zero_grad()
            ctx = SignalContext(
                epoch=0, total_epochs=1, rng=np.random.default_rng(0),
                device=torch.device("cpu"), optimizer=opt, params=params, grad_clip=10.0)
            sig.compute(m, emb, ctx)
            opt.step()
        with torch.no_grad():
            got = m.metabolic.glucose_setpoint_raw(
                m.embedding_projections["metabolic"](emb.weight))
        for pid, t in targets.items():
            self.assertAlmostEqual(float(got[pid]), t["glucose"], delta=1.5, msg=f"gb[{pid}]")


def _set_gb(m: ModularPhysiologyNetwork, gb: float) -> None:
    """Force the Gb head to decode exactly ``gb``: bias = atanh(log(gb/95)/L)."""
    with torch.no_grad():
        m.metabolic.glucose_baseline_net[-1].weight.zero_()
        m.metabolic.glucose_baseline_net[-1].bias.fill_(
            math.atanh(math.log(gb / _CENTER) / M._GB_LOG_MAX))


def _fasted_reference(m: ModularPhysiologyNetwork, gb: float):
    """The patient's fasted reference: every species at typical except glucose at
    Gb and insulin/FFA/glucagon at their decoded basals; no appearance, awake, rest.
    Kept local rather than imported from ``test_iter97_student_metabolic`` so this
    file stands alone (the repo has no cross-test imports)."""
    met = m.metabolic
    emb = m.embedding_projections["metabolic"](torch.zeros(1, EMBEDDING_DIM))
    n_species = len(MODULE_MARKER_INDICES["metabolic"])
    state = torch.zeros(1, n_species)
    with torch.no_grad():
        ib = float(met.insulin_setpoint_raw(emb))
        ffa_b = float(met.ffa_setpoint_raw(emb))
        gn_b = float(met.gn_setpoint_raw(emb))
    state[0, M._GLUCOSE_IDX] = (gb - 95.0) / 30.0
    state[0, M._INSULIN_IDX] = (ib - 10.0) / 10.0
    state[0, M._FFA_IDX] = (ffa_b - 0.5) / 0.2
    state[0, M._GLUCAGON_IDX] = (gn_b - 70.0) / 20.0
    coupling = torch.zeros(1, M._N_COUPLING)
    external = torch.tensor([[0.0, 1.0]])
    tf = compute_time_features(torch.tensor([600.0]))
    return state, coupling, external, emb, tf


class TestGbIsStillAnExactFixedPoint(unittest.TestCase):
    """The log decode must not cost the iter-97 invariant: at the fasted reference
    ``dG = EGP_b − k_ii·Gb = 0`` exactly, for any decoded Gb including the new
    span's edges (the old bound reached 29 and 161, where it was also exact but
    where no patient lives)."""

    def test_dg_is_zero_at_the_decoded_setpoint(self) -> None:
        m = _model(9)
        with torch.no_grad():
            for p in m.metabolic.parameters():
                p.add_(torch.randn_like(p))      # every parameter perturbed
        for gb in (60.6, 70.0, 95.0, 130.0, 148.9):
            _set_gb(m, gb)
            args = _fasted_reference(m, gb)
            with torch.no_grad():
                f = m.metabolic.fluxes(*args)
                rate = m.metabolic(*args)[0, M._GLUCOSE_IDX]
            self.assertAlmostEqual(float(f["gb"]), gb, places=3, msg=f"Gb={gb}")
            self.assertAlmostEqual(float(rate), 0.0, places=5, msg=f"Gb={gb}")
            self.assertAlmostEqual(float(f["egp_b"]), float(f["k_ii"]) * gb, places=6)


if __name__ == "__main__":
    unittest.main()
