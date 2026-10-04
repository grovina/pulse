"""Every synthetic profile's overrides must name a parameter the teacher reads.

``UserProfile.param_overrides`` is applied with ``setattr``, so a key that is not a
``PatientParams`` field creates an orphan attribute instead of raising: the profile
declares a phenotype and gets nothing. That is not hypothetical. ``shift_worker``
and ``anxious_stress`` both set ``cort_circ_amp``, which iter 98 stopped reading
when it replaced the flat HPA block with the CRH -> ACTH -> cortisol cascade, and
``shift_worker`` consequently had NO circadian change at all beyond its basal
cortisol for 18 iterations — while being the profile whose entire purpose is a
disrupted circadian rhythm.

Wave E deleted the three dead fields (``cort_circ_amp``, ``k_acth_to_cort``,
``cort_feedback_acth``), which turns the silent no-op into an orphan attribute.
These tests make it an error instead. The second test is the one that would have
caught the original bug: a name can exist on the dataclass and still be read by
nothing, so it also checks that overriding each key actually changes a simulated
trajectory.
"""

from __future__ import annotations

import unittest
from dataclasses import fields

import numpy as np

from pulse.knowledge.full_body import PatientParams, resolve_derived_params, simulate_full_body
from pulse.knowledge.synthetic_users import PROFILES
from pulse.types import MARKER_INDEX

# Derived in `resolve_derived_params`, so overriding one is legitimate but is
# overwritten before the ODE sees it. Declaring them here keeps the liveness test
# honest rather than silently excusing a key that does nothing.
_DERIVED = {
    "lip_max", "h", "k_bhb", "Hep_b", "Gng_b", "Sg", "acth_per_crh", "crh_circ_amp",
    "k_ba_synth", "ba_spill_gain", "LGly_max",
}


class TestOverridesNameRealParameters(unittest.TestCase):
    def test_every_override_key_is_a_patient_params_field(self) -> None:
        valid = {f.name for f in fields(PatientParams)}
        orphans = [
            (profile.name, key)
            for profile in PROFILES
            for key in profile.param_overrides
            if key not in valid
        ]
        self.assertEqual(
            orphans, [],
            "setattr would create an orphan attribute: the profile declares a "
            "phenotype the teacher never reads (iter 98's cort_circ_amp)",
        )

    def test_profiles_exist_and_are_named(self) -> None:
        self.assertGreater(len(PROFILES), 0)
        for profile in PROFILES:
            self.assertTrue(profile.name)
            self.assertTrue(profile.description)
            self.assertTrue(profile.param_overrides, f"{profile.name} overrides nothing")


class TestOverridesChangeTheSimulation(unittest.TestCase):
    """A field can exist and still be read by nothing — which is exactly how
    cort_circ_amp survived. Check that each override moves a trajectory."""

    DURATION = 1440
    MEALS = ((480.0, 60.0, 15.0, 20.0), (780.0, 70.0, 20.0, 30.0))

    @classmethod
    def _run(cls, params: PatientParams) -> np.ndarray:
        sleep_wake = np.ones(cls.DURATION, dtype=np.float32)
        sleep_wake[:120] = 0.0          # a sleep window, so sleep-gated terms are live
        activity = np.zeros(cls.DURATION, dtype=np.float32)
        activity[600:660] = 0.6        # a bout, so activity-gated terms are live
        traj, _ = simulate_full_body(
            resolve_derived_params(params), list(cls.MEALS), sleep_wake, activity,
            cls.DURATION, start_hour=6.0, noise_scale=0.0,
            rng=np.random.default_rng(0),
        )
        return traj

    def test_each_override_moves_some_marker(self) -> None:
        base = self._run(PatientParams())
        dead: list[tuple[str, str]] = []
        for profile in PROFILES:
            for key, value in profile.param_overrides.items():
                if key in _DERIVED:
                    continue
                params = PatientParams()
                if not hasattr(params, key):
                    continue  # the orphan test above owns this case
                if getattr(params, key) == value:
                    # The override restates the default, so it cannot move anything
                    # BY DEFINITION — a different problem from a parameter nothing
                    # reads, and not one this test can see. `athletic_lean` sets
                    # Si = 4e-4, which IS the default: see the finding recorded in
                    # synthetic_users.py's docstring, where the whole profile set
                    # turns out to sit at or below the population median.
                    continue
                setattr(params, key, value)
                moved = float(np.abs(self._run(params) - base).max())
                if moved == 0.0:
                    dead.append((profile.name, key))
        self.assertEqual(
            dead, [],
            "these overrides change nothing in a 24 h simulation, so the profile "
            "declares a phenotype it does not get",
        )

    def test_shift_worker_actually_shifts_the_cortisol_phase(self) -> None:
        """The regression that motivated this file: the profile exists to have a
        disrupted clock, and used to express it through a parameter nothing read."""
        shift = next(p for p in PROFILES if p.name == "shift_worker")
        params = PatientParams()
        for key, value in shift.param_overrides.items():
            setattr(params, key, value)
        cort = MARKER_INDEX["cortisol"]
        base_peak = int(np.argmax(self._run(PatientParams())[:, cort]))
        shifted_peak = int(np.argmax(self._run(params)[:, cort]))
        # Any real phase change; the direction depends on the declared shift.
        self.assertGreater(
            abs(shifted_peak - base_peak), 120,
            f"cortisol peak moved {abs(shifted_peak - base_peak)} min — a night-shift "
            f"profile must move it by hours, not minutes",
        )


if __name__ == "__main__":
    unittest.main()
