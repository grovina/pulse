"""The clock's numbers and the sleep-debt gains are learnable population parameters.

The phase pull, the sleep need and the reference hour are the circadian module's own
numbers and live there. The three sleep-debt
gains (hepatic retention, lipolysis, peripheral disposal; Donga 2010) only ever scale how
the metabolic module answers the debt it receives by coupling, so they live with that
consumer. Each is log-space, initialised at its constant in ``pulse.circadian`` with no
prior. The gains cancel exactly at zero debt, so they bite only under sleep restriction —
the protocol that identifies them.
"""

from __future__ import annotations

import io
import math
import unittest

import torch
from torch import nn

from pulse.circadian import (
    CLOCK, DEBT_OF_RESTRICTED_NIGHT_H, HEPATIC_K, LIPOLYSIS_K, PERIPHERAL_K, PHASE_PULL_PER_H,
    step_clock_t,
)
from pulse.model import ModularPhysiologyNetwork
from pulse.modules import metabolic as M
from pulse.modules.base import compute_time_features
from pulse.modules.circadian import CircadianModule
from pulse.published import PublishedNumbers, initial_values
from pulse.types import EMBEDDING_DIM, MODULE_MARKER_INDICES

_GAIN_NAMES = ("log_hepatic_k", "log_lipolysis_k", "log_peripheral_k")
_GAIN_CONSTANTS = (HEPATIC_K, LIPOLYSIS_K, PERIPHERAL_K)


def _step_inputs():
    state = torch.tensor([[0.1, -0.2], [0.0, 0.3], [-0.4, 0.5]])
    hour = torch.tensor([2.0, 5.0, 23.0])
    asleep = torch.tensor([1.0, 1.0, 0.5])
    drv = {"hour": hour, "asleep": asleep}
    return state, drv


class TestLearnableCircadianPhase(unittest.TestCase):
    def test_the_clock_numbers_are_the_only_parameters(self) -> None:
        m = CircadianModule(0)
        self.assertEqual(
            set(dict(m.named_parameters())),
            {"log_phase_pull", "clock.log_need_h", "clock.log_phase_ref_h"},
        )
        self.assertIsInstance(m.log_phase_pull, nn.Parameter)

    def test_it_decodes_to_the_constant_at_init(self) -> None:
        m = CircadianModule(0)
        self.assertAlmostEqual(float(m.log_phase_pull.detach().exp()) / PHASE_PULL_PER_H, 1.0, delta=1e-6)

    def test_init_step_matches_the_fixed_constant(self) -> None:
        m = CircadianModule(0)
        state, drv = _step_inputs()
        got = m.step(state, torch.zeros(3, 0), m.constants(torch.zeros(3, 0)), drv, {})
        offset = m._center[0] + m._scale[0] * state[..., 0]
        debt = m._center[1] + m._scale[1] * state[..., 1]
        clock = {**initial_values(CLOCK), "phase_pull": torch.tensor(PHASE_PULL_PER_H)}
        new_offset, new_debt = step_clock_t(offset, debt, drv["hour"], drv["asleep"], clock)
        want = torch.stack((new_offset - offset, new_debt - debt), dim=-1)
        torch.testing.assert_close(got, want, atol=1e-6, rtol=1e-5)

    def test_gradient_reaches_the_phase_pull(self) -> None:
        m = CircadianModule(0)
        state, drv = _step_inputs()
        m.step(state, torch.zeros(3, 0), m.constants(torch.zeros(3, 0)), drv, {})[..., 0].sum().backward()
        self.assertIsNotNone(m.log_phase_pull.grad)
        self.assertNotEqual(float(m.log_phase_pull.grad), 0.0)
        self.assertNotEqual(float(m.clock.log_phase_ref_h.grad), 0.0)

    def test_state_dict_round_trips(self) -> None:
        src = CircadianModule(0)
        with torch.no_grad():
            src.log_phase_pull.add_(0.3)
        buf = io.BytesIO()
        torch.save(src.state_dict(), buf)
        buf.seek(0)
        dst = CircadianModule(0)
        dst.load_state_dict(torch.load(buf))
        self.assertTrue(torch.equal(dst.log_phase_pull, src.log_phase_pull))


def _metabolic_inputs(m: ModularPhysiologyNetwork, debt_h: float):
    """One hyperinsulinaemic time point with the sleep debt forced through coupling.

    The three gains are identities at basal insulin — the gates sit at 1 and
    ``insulin_dev`` is 0 — so they bite only when insulin is raised, which is exactly
    Donga 2010's clamp. Insulin is lifted above its setpoint and only the debt channel
    differs between the debt and no-debt cases, so the gains are the one thing that moves.
    """
    emb = m.embedding_projections["metabolic"](torch.zeros(1, EMBEDDING_DIM))
    state = torch.zeros(1, len(MODULE_MARKER_INDICES["metabolic"]))
    state[0, M._INSULIN_IDX] = 3.0  # supra-basal insulin, the clamp condition
    coupling = torch.zeros(1, M._N_COUPLING)
    coupling[0, M._DEBT_COUPLING_IDX] = (debt_h - M._DEBT_CENTER) / M._DEBT_SCALE
    external = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])  # awake, no drug, infusion, clamp, or fistula
    tf = compute_time_features(torch.tensor([600.0]))
    return state, coupling, external, emb, tf


class TestLearnableDebtGains(unittest.TestCase):
    def test_the_gains_are_metabolic_parameters(self) -> None:
        met = ModularPhysiologyNetwork().metabolic
        named = dict(met.named_parameters())
        for name, constant in zip(_GAIN_NAMES, _GAIN_CONSTANTS):
            self.assertIsInstance(named[name], nn.Parameter)
            self.assertAlmostEqual(float(named[name].detach().exp()) / constant, 1.0, delta=1e-6)

    def test_the_model_param_count_is_the_network_plus_the_published_numbers(self) -> None:
        # The network, then one scalar for every number a hub publishes.
        model = ModularPhysiologyNetwork()
        published = sum(
            p.numel() for mod in model.modules() if isinstance(mod, PublishedNumbers)
            for p in mod.parameters()
        )
        self.assertEqual(published, 43)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 37244 + published)

    def test_gradient_reaches_every_gain_under_sleep_debt(self) -> None:
        m = ModularPhysiologyNetwork()
        rate = m.metabolic(*_metabolic_inputs(m, DEBT_OF_RESTRICTED_NIGHT_H))
        rate.sum().backward()
        for name in _GAIN_NAMES:
            grad = getattr(m.metabolic, name).grad
            self.assertIsNotNone(grad, name)
            self.assertNotEqual(float(grad), 0.0, name)

    def test_the_gains_cancel_at_zero_debt_and_bite_under_debt(self) -> None:
        m = ModularPhysiologyNetwork()
        with torch.no_grad():
            rested = m.metabolic(*_metabolic_inputs(m, 0.0))
            short = m.metabolic(*_metabolic_inputs(m, DEBT_OF_RESTRICTED_NIGHT_H))
            for name in _GAIN_NAMES:
                getattr(m.metabolic, name).add_(0.5)
            rested_moved = m.metabolic(*_metabolic_inputs(m, 0.0))
            short_moved = m.metabolic(*_metabolic_inputs(m, DEBT_OF_RESTRICTED_NIGHT_H))
        self.assertTrue(torch.equal(rested, rested_moved))
        self.assertFalse(torch.allclose(short, short_moved))


if __name__ == "__main__":
    unittest.main()
