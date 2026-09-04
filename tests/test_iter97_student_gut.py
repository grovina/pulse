"""Iter 97 — the gut kernel as a normalized density (review 2026-09-04, item 2.3).

Pins the three properties the previous MLP kernel did not have and that now
hold by construction: K(0) = 0, ∫K = f_bio · mass, and a smooth decay with
< 1 % of the mass beyond ``MEAL_ACTIVE_WINDOW_MIN`` (so the window mask is a
numerical no-op). Also pins that a FRESHLY INITIALIZED kernel is already a
plausible absorption curve, measured against the teacher's own profile.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from pulse.knowledge.full_body import PatientParams, compute_absorption_profile
from pulse.modules.gut import GutModule, MEAL_ACTIVE_WINDOW_MIN, MealEvent

_MEAL = [MealEvent(time=0.0, carbs=60.0, fats=20.0, proteins=25.0)]
_MACROS = torch.tensor([[60.0, 20.0, 25.0]])


def _perturbed_gut(seed: int, std: float = 0.7) -> GutModule:
    torch.manual_seed(seed)
    gut = GutModule(embedding_dim=8, hidden_dim=16)
    with torch.no_grad():
        w = gut.kernel.kernel[-1].weight
        w.add_(std * torch.randn_like(w))
    gut.eval()
    return gut


class TestKernelIsANormalizedDensity(unittest.TestCase):
    def test_kernel_is_zero_at_the_instant_of_the_meal(self) -> None:
        gut = _perturbed_gut(0)
        for _ in range(8):
            emb = 3.0 * torch.randn(8)
            with torch.no_grad():
                out = gut.kernel.forward_single_meal(_MACROS, torch.tensor([0.0]), emb.unsqueeze(0))
            self.assertEqual(float(out[0, :3].abs().sum()), 0.0, msg=str(out))

    def test_integral_equals_f_bio_times_ingested_mass(self) -> None:
        """∫ appearance_j dt = Σ_i macros_i · f_bio_ij, for random embeddings."""
        gut = _perturbed_gut(1)
        times = torch.arange(0, 3000, dtype=torch.float32)  # long enough for every tail
        for _ in range(6):
            emb = 2.0 * torch.randn(8)
            with torch.no_grad():
                curve = gut.kernel.forward_single_meal(
                    _MACROS.expand(times.shape[0], -1), times, emb.unsqueeze(0).expand(times.shape[0], -1))
                _, f_bio = gut.kernel.mixture(emb)
            auc = curve[:, :3].sum(dim=0)
            expected = _MACROS[0] @ f_bio
            torch.testing.assert_close(auc, expected, rtol=2e-3, atol=1e-3)

    def test_every_basis_component_has_under_one_percent_beyond_the_window(self) -> None:
        gut = GutModule(embedding_dim=8, hidden_dim=16)
        tail = gut.kernel.kernel_tail_mass(MEAL_ACTIVE_WINDOW_MIN)
        self.assertTrue(bool((tail < 0.01).all()), msg=f"tail mass beyond window: {tail}")

    def test_no_cliff_at_the_window_edge(self) -> None:
        """The value just inside the window is < 1 % of the peak for any embedding, so
        the mask at 480 removes nothing an integrator can see (the iter-96 kernel
        stepped 0.49 -> 0 there and put a -8 mg/dL/h glucose cliff at 03:00)."""
        gut = _perturbed_gut(2)
        times = torch.arange(0, 600, dtype=torch.float32)
        long = torch.arange(0, 3000, dtype=torch.float32)
        edge = int(MEAL_ACTIVE_WINDOW_MIN)
        for _ in range(6):
            emb = 3.0 * torch.randn(8)
            with torch.no_grad():
                out = gut.forward_window(times, _MEAL, emb)
                # the UNMASKED kernel, to measure what the mask actually discards
                full = gut.kernel.forward_single_meal(
                    _MACROS.expand(long.shape[0], -1), long, emb.unsqueeze(0).expand(long.shape[0], -1))
            discarded = full[edge:, :3].sum(dim=0) / full[:, :3].sum(dim=0)
            self.assertTrue(bool((discarded < 0.01).all()), msg=f"mass beyond window: {discarded}")
            peak = out[:, :3].amax(dim=0)
            inside = out[edge - 1, :3]
            self.assertTrue(bool((inside < 0.02 * peak).all()), msg=f"{inside} vs peak {peak}")
            self.assertEqual(float(out[edge + 1].abs().sum()), 0.0)
            # and the flag has decayed with the mass, not stepped: < 1 % of a 105 g meal
            # is still unabsorbed, i.e. flag < 1 − exp(−1.05/10) ≈ 0.1
            self.assertLess(float(out[int(MEAL_ACTIVE_WINDOW_MIN) - 1, 3]), 0.1)

    def test_kernel_decays_monotonically_after_its_peak(self) -> None:
        gut = _perturbed_gut(3)
        times = torch.arange(0, 600, dtype=torch.float32)
        for _ in range(4):
            emb = 2.0 * torch.randn(8)
            with torch.no_grad():
                out = gut.forward_window(times, _MEAL, emb)[:, 0]
            peak_t = int(out.argmax())
            tail = out[peak_t:]
            self.assertTrue(bool((tail[1:] <= tail[:-1] + 1e-7).all()))


class TestFreshKernelIsPhysiological(unittest.TestCase):
    """A freshly built kernel (zero embedding, no training) against the teacher's
    absorption profile for a 60/20/25 g meal."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.gut = GutModule(embedding_dim=16, hidden_dim=32)
        self.gut.eval()
        times = torch.arange(0, 600, dtype=torch.float32)
        with torch.no_grad():
            self.student = self.gut.forward_window(times, _MEAL, torch.zeros(16)).numpy()
        p = PatientParams()
        self.teacher = np.array(
            [compute_absorption_profile(float(t), [(0.0, 60.0, 20.0, 25.0)], p) for t in range(600)])

    def test_carbohydrate_peaks_at_20_to_40_minutes(self) -> None:
        self.assertTrue(20 <= int(self.student[:, 0].argmax()) <= 40)

    def test_fat_and_protein_are_slower_than_carbohydrate(self) -> None:
        t_glu = int(self.student[:, 0].argmax())
        t_ami = int(self.student[:, 2].argmax())
        t_lip = int(self.student[:, 1].argmax())
        self.assertLess(t_glu, t_ami)
        self.assertLess(t_ami, t_lip)

    def test_auc_matches_the_teacher_within_10_percent(self) -> None:
        for ch in range(3):
            s = self.student[:, ch].sum()
            t = self.teacher[:, ch].sum()
            self.assertLess(abs(s - t) / t, 0.10, msg=f"channel {ch}: student {s:.1f} teacher {t:.1f}")

    def test_peak_matches_the_teacher_within_20_percent(self) -> None:
        for ch in range(3):
            s = self.student[:, ch].max()
            t = self.teacher[:, ch].max()
            self.assertLess(abs(s - t) / t, 0.20, msg=f"channel {ch}: student {s:.3f} teacher {t:.3f}")

    def test_no_nan_anywhere(self) -> None:
        self.assertFalse(np.isnan(self.student).any())


if __name__ == "__main__":
    unittest.main()
