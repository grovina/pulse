"""Iter 97 — SBP > DBP and HRV > 0 by construction (review 2026-09-04, 3.5 / 3.7).

The reviewer's probe found an embedding at the calibration leash with setpoints
SBP 96.1 / DBP 97.1 and a 12 h rest rollout inverted for 716 of 720 min, and
HRV pinned at the 0-ms clamp for hundreds of minutes on 2 of 8 random
embeddings. These tests repeat both probes against the new parameterization
(pulse pressure and HRV in log space; multiplicative integrator step).
"""

from __future__ import annotations

import unittest

import torch

from pulse.model import ModularPhysiologyNetwork, euler_step, integrate
from pulse.modules import cardiovascular as C
from pulse.types import EMBEDDING_DIM, MARKER_INDEX as MI, NORM_CENTER

_HR, _HRV, _SBP, _DBP = MI["hr"], MI["hrv"], MI["sbp"], MI["dbp"]


def _small_model(seed: int = 0, perturb: float = 1.0) -> ModularPhysiologyNetwork:
    torch.manual_seed(seed)
    m = ModularPhysiologyNetwork(
        metabolic_hidden=16, appetite_hidden=16, stress_hidden=16, cardiovascular_hidden=16,
        thermoreg_hidden=16, respiratory_hidden=16, gut_hidden=16, hepatobiliary_hidden=16)
    # Zero-init heads make every embedding identical; give them authority.
    with torch.no_grad():
        for net in (m.cardiovascular.setpoint_net, m.cardiovascular.network):
            net[-1].weight.add_(perturb * torch.randn_like(net[-1].weight))
    m.eval()
    return m


class TestSetpointsAreOrdered(unittest.TestCase):
    def test_sbp_setpoint_exceeds_dbp_setpoint_for_random_embeddings(self) -> None:
        m = _small_model(0, perturb=3.0)
        cvs = m.cardiovascular
        emb = 3.0 * torch.randn(512, EMBEDDING_DIM)
        with torch.no_grad():
            sp = cvs.setpoints_raw(m.embedding_projections["cardiovascular"](emb))
        pp = sp[:, 2] - sp[:, 3]
        self.assertTrue(bool((pp > 0).all()))
        self.assertTrue(bool((pp >= 40.0 * torch.exp(torch.tensor(-C._PP_LOG_SP_MAX)) - 1e-4).all()))
        self.assertTrue(bool((sp[:, 1] > 0).all()))  # HRV setpoint

    def test_adversarial_embedding_cannot_invert_the_setpoints(self) -> None:
        """The reviewer's probe: gradient-descend the embedding to MINIMIZE SBP − DBP
        under the ||emb|| <= 3 leash. It used to reach −1 mmHg."""
        m = _small_model(1, perturb=3.0)
        cvs = m.cardiovascular
        proj = m.embedding_projections["cardiovascular"]
        e = torch.zeros(EMBEDDING_DIM, requires_grad=True)
        opt = torch.optim.Adam([e], lr=0.05)
        for _ in range(300):
            sp = cvs.setpoints_raw(proj(e))
            loss = sp[2] - sp[3]
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                n = e.norm()
                if n > 3:
                    e.mul_(3 / n)
        with torch.no_grad():
            sp = cvs.setpoints_raw(proj(e.detach()))
        self.assertGreater(float(sp[2] - sp[3]), 15.0)

    def test_zero_embedding_rests_at_norm_center(self) -> None:
        m = ModularPhysiologyNetwork(metabolic_hidden=16, cardiovascular_hidden=16)
        with torch.no_grad():
            z = m.cardiovascular.setpoints_z(m.embedding_projections["cardiovascular"](torch.zeros(EMBEDDING_DIM)))
        torch.testing.assert_close(z, torch.zeros(4), atol=1e-6, rtol=0)

    def test_log_form_reduces_to_the_iter89_additive_form_near_the_setpoint(self) -> None:
        """rate_hrv ≈ driver − k·(HRV − HRV_sp) to first order."""
        m = ModularPhysiologyNetwork(metabolic_hidden=16, cardiovascular_hidden=16)
        cvs = m.cardiovascular
        emb = torch.zeros(1, cvs.setpoint_net[0].in_features)
        state = torch.zeros(1, 4)
        state[0, 1] = 0.4 / 15.0   # HRV = 40.4 ms (setpoint 40)
        with torch.no_grad():
            rate = cvs(state, torch.zeros(1, 4), torch.zeros(1, 2), emb, torch.tensor([[0.0, 1.0, 0.0, 1.0]]))
            k = C._CVS_K_MIN + C._CVS_K_RANGE * torch.sigmoid(cvs.log_k)
        self.assertAlmostEqual(float(rate[0, 1]), float(-k[1] * 0.4), delta=0.01 * float(k[1] * 0.4))


class TestTrajectoriesStayOrderedAndPositive(unittest.TestCase):
    def _rollout(self, m, emb, n=720):
        typ = torch.tensor(NORM_CENTER)
        with torch.no_grad():
            return integrate(m, typ, emb, n, start_time_minutes=480, meals=[],
                             sleep_wake=torch.ones(n), activity=torch.zeros(n))

    def test_12h_rest_rollout_never_inverts_for_adversarial_embeddings(self) -> None:
        m = _small_model(2, perturb=3.0)
        for seed in range(4):
            torch.manual_seed(seed)
            e = torch.randn(EMBEDDING_DIM); e = 3.0 * e / e.norm()
            tr = self._rollout(m, e)
            pp = tr[:, _SBP] - tr[:, _DBP]
            self.assertTrue(bool((pp > 0).all()), msg=f"min pulse pressure {float(pp.min()):.2f}")
            self.assertTrue(bool((tr[:, _HRV] > 0).all()), msg=f"min HRV {float(tr[:, _HRV].min()):.2f}")
            self.assertFalse(bool(torch.isnan(tr).any()))

    def test_hrv_survives_a_pathological_negative_driver(self) -> None:
        """Force the HRV driver to −50 ms/min at typical (an off-manifold MLP output).
        Additive Euler would cross zero on step 1; the multiplicative step cannot."""
        m = _small_model(3, perturb=0.0)
        with torch.no_grad():
            m.cardiovascular.network[-1].bias[1] = -50.0
            m.cardiovascular.network[-1].bias[2] = -80.0  # and pulse pressure
        tr = self._rollout(m, torch.zeros(EMBEDDING_DIM), n=300)
        self.assertTrue(bool((tr[:, _HRV] > 0).all()))
        self.assertTrue(bool((tr[:, _SBP] > tr[:, _DBP]).all()))

    def test_euler_step_is_first_order_consistent(self) -> None:
        """For small rates the multiplicative step equals the additive one."""
        state = torch.tensor(NORM_CENTER).unsqueeze(0)
        rates = torch.zeros_like(state)
        rates[0, _HRV] = 0.3; rates[0, _SBP] = -0.2; rates[0, _DBP] = 0.1
        new = euler_step(state, rates, 1.0)
        torch.testing.assert_close(new, state + rates, atol=2e-3, rtol=0)


if __name__ == "__main__":
    unittest.main()
