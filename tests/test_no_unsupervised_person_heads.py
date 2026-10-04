"""No per-person head without a supervision path: the HPA phase and the CCK basal.

A per-person head that nothing supervises is free authority. Iter 109 measured the
cost: the glucagon-basal head took the prior person from 70 to 98 pg/mL while liver,
ketones and glucose were unchanged when it was zeroed, because every gate downstream
is normalized to the person's OWN basal. Two more heads were in that position, and in
both the teacher does not even vary the quantity, so no ground truth could ever
supervise them:

  stress.phase_proj              a per-embedding HPA phase shift of ±2 h
  hepatobiliary.cck_baseline_net a per-embedding CCK reference (×0.5 to ×2.0)

These tests pin the deletions (the heads are gone and the quantities are the same for
every embedding), pin the premise that justifies them (the teacher never draws either),
and guard the rest of the HPA cascade against the iter-97 failure, where `_beta_raw` was
bit-identical to its init after 21 h because the cascade read the wrong frame: deleting
a head must leave every remaining parameter reachable from a rollout.

If the teacher starts to draw a CCK basal or an HPA phase, the premise test fails on
purpose. The head comes back WITH a target in `SetpointSupervisionSignal`, not before.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from pulse.knowledge.full_body import PatientParams, randomize_params
from pulse.model import ModularPhysiologyNetwork, integrate
from pulse.modules import hepatobiliary as H
from pulse.modules import stress as S
from pulse.modules.base import compute_time_features
from pulse.types import EMBEDDING_DIM, MARKER_INDEX as MI, NORM_CENTER, NORM_SCALE

# The hard clamp on a calibrated embedding: CalibrationSettings.max_norm.
_MAX_NORM = 8.0
_N_EMBEDDINGS = 100


def _model(seed: int = 0, perturb: float = 1.0) -> ModularPhysiologyNetwork:
    """Small widths, every parameter of both modules and their embedding projections
    moved far from init. A surviving per-person head has a zero-init output layer, so
    at init it would be a no-op and these tests would pass for the wrong reason."""
    torch.manual_seed(seed)
    m = ModularPhysiologyNetwork(
        metabolic_hidden=12, appetite_hidden=8, stress_hidden=8, cardiovascular_hidden=12,
        thermoreg_hidden=8, respiratory_hidden=8, gut_hidden=8, hepatobiliary_hidden=8)
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for mod in (m.stress, m.hepatobiliary, m.embedding_projections):
            for p in mod.parameters():
                p.add_(perturb * torch.randn(p.shape, generator=g))
    m.eval()
    return m


def _embeddings(seed: int = 0) -> torch.Tensor:
    """100 codes in the model's embedding space: the init scale, the soft-norm radius,
    and the hard clamp (norm exactly 8), a third of each."""
    g = torch.Generator().manual_seed(seed)
    raw = torch.randn(_N_EMBEDDINGS, EMBEDDING_DIM, generator=g)
    unit = raw / raw.norm(dim=-1, keepdim=True)
    radius = torch.empty(_N_EMBEDDINGS)
    third = _N_EMBEDDINGS // 3
    radius[:third] = 0.1 * EMBEDDING_DIM ** 0.5
    radius[third:2 * third] = 3.0
    radius[2 * third:] = _MAX_NORM
    return unit * radius.unsqueeze(-1)


class TestTheTeacherDoesNotVaryThem(unittest.TestCase):
    """The premise of the deletions: no patient has a CCK basal or an HPA phase."""

    _FIELDS = ("CCK_b", "hpa_rise_start_h", "hpa_peak_h", "hpa_fall_tau_h")

    def test_randomize_params_never_draws_the_cck_basal_or_the_hpa_clock(self) -> None:
        ref = PatientParams()
        for seed in range(50):
            p = randomize_params(np.random.default_rng(seed))
            for name in self._FIELDS:
                self.assertEqual(
                    getattr(p, name), getattr(ref, name),
                    f"the teacher now varies {name}: the deletion's premise is gone. "
                    "Supervise a per-person head against it (SetpointSupervisionSignal) "
                    "before bringing one back.")

    def test_the_students_constants_are_the_teachers(self) -> None:
        self.assertEqual(H._TYPICALS[H._CCK_IDX], PatientParams().CCK_b)
        self.assertEqual(S._HPA_RISE_START_H, PatientParams().hpa_rise_start_h)
        self.assertEqual(S._HPA_PEAK_H, PatientParams().hpa_peak_h)
        self.assertEqual(S._HPA_FALL_TAU_H, PatientParams().hpa_fall_tau_h)


class TestStressClockIsNotAPerson(unittest.TestCase):
    def test_the_phase_head_is_gone(self) -> None:
        m = _model()
        self.assertFalse(hasattr(m.stress, "phase_proj"))
        self.assertEqual([n for n, _ in m.named_parameters() if "phase_proj" in n], [])
        emb = m.embedding_projections["stress"](_embeddings())
        const = m.stress.constants(emb)
        self.assertNotIn("phase_h", const)
        self.assertLessEqual(
            {"cort_b", "fb_amp", "k_crh", "k_acth", "k_cort", "hypo_gain", "act_gain"},
            set(const))

    def test_a_per_person_head_the_teacher_does_vary_is_visible_to_this_harness(self) -> None:
        """Control: cort_b (the teacher draws Cort_b per patient) does vary across these
        embeddings, so the identity below is not an artefact of a degenerate harness."""
        m = _model()
        with torch.no_grad():
            emb = m.embedding_projections["stress"](_embeddings())
            cort_b = m.stress.constants(emb)["cort_b"]
        self.assertGreater(float(cort_b.max() / cort_b.min()), 1.2)

    @staticmethod
    def _drive(m, emb: torch.Tensor, minutes: torch.Tensor, sleep_wake: float) -> torch.Tensor:
        """The CRH circadian target at every minute of the window, per embedding. The
        window goes to ``drives`` at once, as the planned integrator hands it over, with
        the embedding repeated along it; every embedding sees the same call shape, so
        equal inputs are bit-equal outputs."""
        n_t = int(minutes.shape[0])
        tf = compute_time_features(minutes)
        external = torch.stack(
            [torch.full_like(minutes, sleep_wake), torch.zeros_like(minutes)], -1)
        coupling = torch.zeros(n_t, 2)
        return torch.stack([
            m.stress.drives(
                external, coupling, tf, m.stress.constants(emb[i:i + 1].expand(n_t, -1))
            )["crh_target_open_loop"]
            for i in range(int(emb.shape[0]))
        ])

    def test_crh_circadian_drive_is_identical_across_embeddings(self) -> None:
        m = _model()
        emb = m.embedding_projections["stress"](_embeddings())
        minutes = torch.arange(0.0, 1440.0, 15.0)
        for sleep_wake in (0.0, 0.5, 1.0):
            with torch.no_grad():
                drive = self._drive(m, emb, minutes, sleep_wake)
            self.assertTrue(
                torch.equal(drive, drive[:1].expand_as(drive)),
                f"the CRH drive differs between embeddings at sleep_wake={sleep_wake}: "
                f"max spread {float((drive.max(0).values - drive.min(0).values).max()):.3e}")

    def test_the_pointwise_and_the_window_drive_agree(self) -> None:
        """``forward`` hands ``drives`` one minute per batch member; the planned
        integrator hands it the whole window, flattened to [B*T]. Same function."""
        m = _model()
        emb = m.embedding_projections["stress"](_embeddings())[:7]
        minutes = torch.tensor([0.0, 150.0, 390.0, 700.0, 1260.0, 1439.0])
        n_e, n_t = int(emb.shape[0]), int(minutes.shape[0])
        with torch.no_grad():
            window = self._drive(m, emb, minutes, 1.0)
            flat = m.stress.drives(
                torch.tensor([1.0, 0.0]).expand(n_e * n_t, -1), torch.zeros(n_e * n_t, 2),
                compute_time_features(minutes.repeat(n_e)),
                m.stress.constants(emb.repeat_interleave(n_t, dim=0)),
            )["crh_target_open_loop"].reshape(n_e, n_t)
            point = torch.stack([
                m.stress.drives(
                    torch.tensor([[1.0, 0.0]]), torch.zeros(1, 2),
                    compute_time_features(minutes[j:j + 1]), m.stress.constants(emb[i:i + 1]),
                )["crh_target_open_loop"][0]
                for i in range(n_e) for j in range(n_t)
            ]).reshape(n_e, n_t)
        torch.testing.assert_close(flat, window, atol=1e-4, rtol=1e-6)
        torch.testing.assert_close(point, window, atol=1e-4, rtol=1e-6)

    def test_the_drive_is_the_unshifted_clock(self) -> None:
        """Identical across embeddings would also hold for one shared shift; the clock
        itself is pinned too: awake, the drive crests at the teacher's peak hour."""
        m = _model()
        emb = m.embedding_projections["stress"](_embeddings())
        minutes = torch.arange(0.0, 1440.0, 15.0)
        with torch.no_grad():
            drive = self._drive(m, emb, minutes, 1.0)
        self.assertGreater(float(drive[0].max() - drive[0].min()), 1.0)
        crest = float(minutes[int(drive[0].argmax())]) / 60.0
        self.assertAlmostEqual(crest, S._HPA_PEAK_H, places=6)


class TestHepatobiliaryCckBasalIsTheTypical(unittest.TestCase):
    def test_the_cck_head_is_gone(self) -> None:
        m = _model()
        self.assertFalse(hasattr(m.hepatobiliary, "cck_baseline_net"))
        self.assertFalse(hasattr(H, "_CCK_LOG_MAX"))
        self.assertEqual([n for n, _ in m.named_parameters() if "cck_baseline" in n], [])

    def test_cck_basal_is_exactly_the_typical_for_every_embedding(self) -> None:
        m = _model()
        hb = m.hepatobiliary
        typical = NORM_CENTER[MI["cck"]]
        emb = m.embedding_projections["hepatobiliary"](_embeddings())
        with torch.no_grad():
            direct = hb.cck_setpoint_raw(emb)
            const = hb.constants(emb)["cck_b"]
        for got in (direct, const):
            self.assertTrue(torch.equal(got, torch.full_like(got, typical)))
        self.assertEqual(typical, PatientParams().CCK_b)

    def test_cck_basal_keeps_the_embeddings_batch_shape(self) -> None:
        """constants() contracts to the embedding's leading dims: the planned integrator
        calls it with [B, E] and with [B*T, E]."""
        m = _model()
        hb = m.hepatobiliary
        width = m.embedding_projections["hepatobiliary"].out_features
        for lead in ((), (5,), (3, 4)):
            got = hb.cck_setpoint_raw(torch.zeros(*lead, width))
            self.assertEqual(got.shape, torch.Size(lead))
            self.assertFalse(got.requires_grad)
            self.assertEqual(got.dtype, hb.log_cck_fat_gain.dtype)


class TestHpaCascadeStillTrains(unittest.TestCase):
    """Iter 97: `_beta_raw` was bit-identical to its init after 21 h. A gate that is
    exactly zero for the whole rollout gives its parameter exactly zero gradient."""

    _CASCADE = ("_fb_raw", "_hypo_raw", "_act_raw", "log_k_crh", "log_k_acth", "log_k_cort")

    def _rollout_grads(self, m: ModularPhysiologyNetwork) -> None:
        """Every gate live: cortisol above its basal (feedback), glucose under 70
        (hypoglycaemia drive), activity on. The loss reads the cascade's own states."""
        m.zero_grad()
        s0 = torch.tensor(NORM_CENTER).clone()
        s0[MI["cortisol"]] = 20.0
        s0[MI["glucose"]] = 55.0
        n = 120
        traj = integrate(
            m, s0, torch.zeros(EMBEDDING_DIM), n, start_time_minutes=360.0, meals=[],
            sleep_wake=torch.ones(n), activity=torch.full((n,), 0.5))
        cols = [MI["cortisol"], MI["acth"], MI["crh"]]
        scale = torch.tensor(NORM_SCALE)[cols]
        (traj[..., cols] / scale).pow(2).sum().backward()

    def test_every_remaining_stress_parameter_receives_gradient_from_a_rollout(self) -> None:
        m = _model(perturb=0.3)
        self._rollout_grads(m)
        names = dict(m.stress.named_parameters())
        for name in self._CASCADE:
            self.assertIn(name, names)
        dead = [
            n for n, p in names.items()
            if p.grad is None
            or not bool(torch.isfinite(p.grad).all())
            or float(p.grad.abs().max()) == 0.0
        ]
        self.assertEqual(dead, [], "stress parameters no rollout can move")

    def test_one_optimizer_step_moves_every_cascade_parameter(self) -> None:
        m = _model(perturb=0.3)
        self._rollout_grads(m)
        before = {n: getattr(m.stress, n).detach().clone() for n in self._CASCADE}
        torch.optim.Adam(m.stress.parameters(), lr=1e-2).step()
        for n in self._CASCADE:
            self.assertFalse(
                torch.equal(getattr(m.stress, n).detach(), before[n]),
                f"{n} is bit-identical to its init after a step")


if __name__ == "__main__":
    unittest.main()
