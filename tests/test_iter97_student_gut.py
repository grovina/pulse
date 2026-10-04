"""Iter 97 — the gut kernel as a normalized density (review 2026-09-04, item 2.3).

Pins the three properties the previous MLP kernel did not have and that now
hold by construction: K(0) = 0, ∫K = f_bio · mass, and a smooth decay with
< 1 % of the mass beyond ``MEAL_ACTIVE_WINDOW_MIN`` (so the window mask is a
numerical no-op). Also pins that a FRESHLY INITIALIZED kernel is already a
plausible absorption curve, measured against the teacher's own profile.

PLAN A9/A10: ``f_bio`` is now ``APPEARANCE_UNITS_PER_G · σ(raw)``, so the
bioavailable FRACTION is bounded by 1 and absorbed ≤ ingested holds for every
embedding rather than only for the ones training happened to visit (softplus let
it past 1, which minted carbon, and below 1 it deleted carbon that no ledger
booked as malabsorbed). ``TestBioavailabilityIsBounded`` pins the bound itself;
``TestFreshKernelIsPhysiological`` now scores the fresh curve against
``F_BIO_INIT_FRACTION · teacher`` because the init fraction is deliberately 0.8
— the gain A10 moved out of ``MetabolicModule.log_ra``.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from pulse.knowledge.full_body import PatientParams, compute_absorption_profile
from pulse.modules.base import GutModuleBase
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
        """∫ appearance_j dt = macros_j · f_bio_j (diagonal: carbs do not appear as lipid)."""
        gut = _perturbed_gut(1)
        times = torch.arange(0, 3000, dtype=torch.float32)  # long enough for every tail
        for _ in range(6):
            emb = 2.0 * torch.randn(8)
            with torch.no_grad():
                curve = gut.kernel.forward_single_meal(
                    _MACROS.expand(times.shape[0], -1), times, emb.unsqueeze(0).expand(times.shape[0], -1))
                _, f_bio = gut.kernel.mixture(emb)
            auc = curve[:, :3].sum(dim=0)
            expected = _MACROS[0] * f_bio
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
        edge = int(MEAL_ACTIVE_WINDOW_MIN)
        times = torch.arange(0, edge + 120, dtype=torch.float32)
        long = torch.arange(0, 4 * edge, dtype=torch.float32)
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

    def test_kernel_is_diagonal(self) -> None:
        """Carbohydrate does not appear on the lipid or amino channel."""
        gut = GutModule(embedding_dim=8, hidden_dim=16)
        gut.eval()
        meal = [MealEvent(time=0.0, carbs=60.0, fats=0.0, proteins=0.0)]
        times = torch.arange(0, 400, dtype=torch.float32)
        with torch.no_grad():
            out = gut.forward_window(times, meal, torch.zeros(8))
        self.assertGreater(float(out[:, 0].max()), 0.1)
        self.assertLess(float(out[:, 1].abs().max()), 1e-6)
        self.assertLess(float(out[:, 2].abs().max()), 1e-6)


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

    def test_carbohydrate_peaks_at_30_to_60_minutes(self) -> None:
        # Oral glucose appearance peaks at 30-60 min (Dalla Man 2007; the teacher's
        # two-component kernel puts a 60 g load at 46 min).
        self.assertTrue(30 <= int(self.student[:, 0].argmax()) <= 60)

    def test_fat_and_protein_are_slower_than_carbohydrate(self) -> None:
        t_glu = int(self.student[:, 0].argmax())
        t_ami = int(self.student[:, 2].argmax())
        t_lip = int(self.student[:, 1].argmax())
        self.assertLess(t_glu, t_ami)
        self.assertLess(t_ami, t_lip)

    def test_auc_is_the_init_fraction_of_the_teacher_within_10_percent(self) -> None:
        """A10: the fresh fraction is ``F_BIO_INIT_FRACTION`` (0.8), not 1, so the
        AUC is deliberately 80 % of the teacher's absorbed mass — that is the gain
        moved out of the deleted ``Ra``, not a shape error. Measured ratios at this
        init: 0.806 / 0.801 / 0.803 for carb / fat / protein (1.007 / 1.000 / 1.004
        when the fraction was pinned at 1 and ``Ra`` carried the 0.8 downstream)."""
        frac = GutModuleBase.F_BIO_INIT_FRACTION
        for ch in range(3):
            s = self.student[:, ch].sum()
            t = frac * self.teacher[:, ch].sum()
            self.assertLess(abs(s - t) / t, 0.10, msg=f"channel {ch}: student {s:.1f} target {t:.1f}")

    def test_peak_is_the_init_fraction_of_the_teacher_within_20_percent(self) -> None:
        frac = GutModuleBase.F_BIO_INIT_FRACTION
        for ch in range(3):
            s = self.student[:, ch].max()
            t = frac * self.teacher[:, ch].max()
            self.assertLess(abs(s - t) / t, 0.20, msg=f"channel {ch}: student {s:.3f} target {t:.3f}")

    def test_the_shape_is_the_teacher_shape_up_to_that_one_scalar(self) -> None:
        """The fraction is a per-channel SCALE, so dividing it out must put the
        fresh curve back on the teacher's own curve pointwise — the AUC and peak
        tests above would also pass if the 0.8 had come from a reshaped kernel."""
        frac = GutModuleBase.F_BIO_INIT_FRACTION
        for ch in range(3):
            t = self.teacher[:, ch]
            s = self.student[:, ch] / frac
            peak = float(t.max())
            self.assertLess(float(np.abs(s - t).max()) / peak, 0.15, msg=f"channel {ch}")

    def test_no_nan_anywhere(self) -> None:
        self.assertFalse(np.isnan(self.student).any())


class TestBioavailabilityIsBounded(unittest.TestCase):
    """PLAN A9: ``f_bio = APPEARANCE_UNITS_PER_G · σ(raw)``, so ABSORBED ≤ INGESTED
    on every channel for every embedding — by the functional form, not by a loss.

    The ``softplus`` this replaces had no ceiling. Measured on the same perturbed
    kernels at ‖emb‖ ~ 3-8 (the calibration clamp is 8) it reaches a bioavailable
    fraction of 4.9 and exceeds 1 for 13-26 % of embeddings — a 60 g meal appearing
    as up to 290 g of glucose that nothing had eaten, with the metabolic carbon
    ledger closing on the invented mass because the gut is where carbon enters.
    Below 1 it was equally unbooked: the missing grams were not malabsorbed
    anywhere, they were simply gone.
    """

    def test_fraction_is_in_the_unit_interval_for_any_embedding(self) -> None:
        """≤ 1 is the conservation statement and it holds at every scale, including
        far past the ‖e‖ ≤ 8 calibration clamp. The STRICT inequality holds too, and
        not by luck: the kernel's hidden layers are ``tanh``, so the logit is
        bounded by the output layer's own weights and the fraction cannot reach the
        float32 ceiling where σ returns exactly 1 and the gradient dies (measured
        at ‖e‖ ~ 1e6 with std-2.0 weights: max 0.9999986, min 1.3e-3)."""
        units = torch.tensor(GutModuleBase.APPEARANCE_UNITS_PER_G)
        for seed in range(4):
            gut = _perturbed_gut(seed, std=2.0)
            g = torch.Generator().manual_seed(seed)
            for scale in (0.0, 1.0, 3.0, 8.0, 50.0, 1000.0, 1e6):
                emb = scale * torch.randn(16, 8, generator=g)
                with torch.no_grad():
                    frac = gut.kernel.bioavailable_fraction(emb)
                    _, f_bio = gut.kernel.mixture(emb)
                where = f"seed {seed} scale {scale:g}"
                self.assertEqual(frac.shape, (16, 3))
                self.assertTrue(bool((frac > 0.0).all()), msg=where)
                self.assertTrue(bool((frac <= 1.0).all()), msg=where)
                self.assertTrue(bool((frac < 1.0).all()), msg=where)
                # and f_bio is that fraction times the fixed unit conversion
                torch.testing.assert_close(f_bio, frac * units, atol=1e-6, rtol=1e-6)
                self.assertTrue(bool((f_bio <= units + 1e-6).all()), msg=where)

    def test_the_softplus_head_this_replaces_broke_the_bound_here(self) -> None:
        """The bound is not a formality. On the same perturbed kernels, reading the
        old ``softplus`` head's output as a fraction of ``APPEARANCE_UNITS_PER_G``
        gives up to 4.9 and exceeds 1 for 13-26 % of embeddings at ‖e‖ ~ 3-8 — a
        60 g meal appearing as 290 g of glucose, with the metabolic carbon ledger
        closing on the invented mass because the gut is where carbon enters."""
        units = torch.tensor(GutModuleBase.APPEARANCE_UNITS_PER_G)
        worst = 0.0
        for seed in range(3):
            gut = _perturbed_gut(seed, std=2.0)
            g = torch.Generator().manual_seed(seed)
            for scale in (3.0, 8.0):
                emb = scale * torch.randn(2048, 8, generator=g)
                with torch.no_grad():
                    raw = gut.kernel.kernel(emb)[..., -GutModuleBase.N_MACROS:]
                    old_frac = torch.nn.functional.softplus(raw) / units
                worst = max(worst, float(old_frac.max()))
                self.assertGreater(float((old_frac > 1.0).float().mean()), 0.05)
        self.assertGreater(worst, 2.0, msg=f"worst old fraction {worst:.2f}")

    def test_absorbed_mass_never_exceeds_ingested_mass(self) -> None:
        """The statement in grams: integrate the carbohydrate channel out to its
        tail, convert back with the population MG_DL_PER_G, and compare with the
        60 g that went in. The old softplus head could clear it."""
        from pulse.types import MG_DL_PER_G
        times = torch.arange(0, 3000, dtype=torch.float32)
        for seed in range(3):
            gut = _perturbed_gut(seed, std=2.0)
            for scale in (0.0, 3.0, 8.0, 40.0):
                emb = scale * torch.randn(8, generator=torch.Generator().manual_seed(seed))
                with torch.no_grad():
                    curve = gut.kernel.forward_single_meal(
                        _MACROS.expand(times.shape[0], -1), times,
                        emb.unsqueeze(0).expand(times.shape[0], -1))
                absorbed_g = float(curve[:, 0].sum()) / MG_DL_PER_G
                self.assertLessEqual(absorbed_g, float(_MACROS[0, 0]) + 1e-3,
                                     msg=f"seed {seed} scale {scale}: {absorbed_g:.2f} g absorbed")
                self.assertGreater(absorbed_g, 0.0)

    def test_integral_is_still_the_fraction_times_the_ingested_mass(self) -> None:
        """A9 must not cost the iter-97 identity: ∫K = f_bio·mass still holds, and
        now reads as ``fraction · mass · units_per_g`` — the same number, with the
        part that is a conservation statement separated from the unit."""
        gut = _perturbed_gut(7, std=1.5)
        times = torch.arange(0, 3000, dtype=torch.float32)
        units = torch.tensor(GutModuleBase.APPEARANCE_UNITS_PER_G)
        for scale in (1.0, 4.0, 8.0):
            emb = scale * torch.randn(8, generator=torch.Generator().manual_seed(int(scale)))
            with torch.no_grad():
                curve = gut.kernel.forward_single_meal(
                    _MACROS.expand(times.shape[0], -1), times,
                    emb.unsqueeze(0).expand(times.shape[0], -1))
                frac = gut.kernel.bioavailable_fraction(emb)
            auc = curve[:, :3].sum(dim=0)
            torch.testing.assert_close(auc, _MACROS[0] * frac * units, rtol=2e-3, atol=1e-3)

    def test_a_fresh_kernel_starts_at_the_init_fraction_on_every_channel(self) -> None:
        """The cold start A10 had to preserve: 0.8, the deleted ``Ra`` init."""
        gut = GutModule(embedding_dim=16, hidden_dim=32)
        gut.eval()
        with torch.no_grad():
            frac = gut.kernel.bioavailable_fraction(torch.zeros(16))
        for ch in range(3):
            self.assertAlmostEqual(float(frac[ch]), GutModuleBase.F_BIO_INIT_FRACTION, places=2)

    def test_the_fraction_is_learnable_in_both_directions(self) -> None:
        """A bounded init must still have gradient: 0.8 sits where σ' = 0.16, so the
        head can move the fraction up as well as down (at 0.98 it would be 0.02)."""
        gut = GutModule(embedding_dim=8, hidden_dim=16)
        emb = torch.zeros(8, requires_grad=True)
        frac = gut.kernel.bioavailable_fraction(emb)
        frac.sum().backward()
        bias = gut.kernel.kernel[-1].bias
        self.assertIsNotNone(bias.grad)
        tail = bias.grad[-GutModuleBase.N_MACROS:]
        self.assertTrue(bool((tail.abs() > 0.05).all()), msg=str(tail))


if __name__ == "__main__":
    unittest.main()
