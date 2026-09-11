"""Iter 97 (training): the physiology-rule surface.

Review 4.5 (the teacher violated 22 of 61 rules; the 24-48 h fast arm started
fed; no rules audit), 4.6 (four value helpers still used an absolute soft-max
beta and read the window mean; adaptive weights ranked rules by raw units;
correlation eps in absolute units; two bands ran to 26 h), 4.12 (uncited
thresholds, hinges at the population mean) and 1.2 (the sleep arm).
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
import torch

import pulse

REPO = os.path.dirname(os.path.dirname(os.path.abspath(pulse.__file__)))
assert pulse.__file__.startswith(REPO), pulse.__file__

from pulse.knowledge.cohort_types import CohortArmSpec, InitMode  # noqa: E402
from pulse.knowledge.full_body import PatientParams  # noqa: E402
from pulse.knowledge.physiology_rules import (  # noqa: E402
    _FAST_24H_TO_48H_ARM,
    _SLEEP_WAKE_24H_ARM,
    ACTH_EVENING_TROUGH,
    CORTISOL_EVENING_TROUGH,
    HR_FALLS_DURING_SLEEP,
    HRV_HIGHER_DURING_SLEEP,
    PHYSIOLOGY_RULES,
    SBP_FALLS_DURING_SLEEP,
    SBP_MORNING_SURGE,
    PhysiologyRule,
    hinge_circadian_amplitude,
    hinge_max_correlation,
    hinge_max_drift,
    hinge_max_value,
    hinge_min_rise,
    hinge_min_value,
)
from pulse.physiology_rules_loss import rule_context_for_arm  # noqa: E402
from pulse.training import RolloutEvidenceSignal, WeightSchedule  # noqa: E402
from pulse.training.adaptive_weights import adaptive_multipliers  # noqa: E402
from pulse.training.arm_init import cold_initial_state_for_arm, teacher_arm_trajectory  # noqa: E402
from pulse.types import MARKER_INDEX, STATE_DIM  # noqa: E402


def _bump(T: int, base: float, ext: float, width: float = 10.0) -> torch.Tensor:
    idx = torch.arange(T, dtype=torch.float32)
    return (base + (ext - base) * torch.exp(-((idx - T / 2) / width) ** 2)).unsqueeze(1)


# --- 4.6: value helpers read the extremum, not the window mean ---------------

def test_soft_max_value_reads_the_peak() -> None:
    # Review 4.6: with the absolute beta 0.05 a 190 mg/dL peak on a 95 baseline
    # read 167 and glucose_postprandial_bounded (ceiling 180) never fired.
    v = hinge_max_value(_bump(180, 95.0, 190.0), 0, slice(0, 180), 180.0)
    # The soft max reads 187.5 of 190 (range-relative beta 20); the old
    # absolute beta read 167 and reported 0. Not the exact 10, but the peak.
    assert 6.0 < float(v) <= 10.0


def test_soft_min_value_reads_the_dip() -> None:
    # A 55 mg/dL dip read 69 against a 65 floor.
    v = hinge_min_value(_bump(420, 70.0, 55.0), 0, slice(0, 420), 65.0)
    assert 6.0 < float(v) <= 10.0


def test_circadian_amplitude_reads_the_swing() -> None:
    # A 1.0 C peak-to-trough sinusoid read 0.013 C; the 0.5-0.8 band must now
    # see the ~0.2 C overshoot.
    T = 1440
    idx = torch.arange(T, dtype=torch.float32)
    vals = (37.0 + 0.5 * torch.sin(2 * math.pi * idx / T)).unsqueeze(1)
    v = hinge_circadian_amplitude(vals, 0, slice(0, T), 0.5, max_amplitude=0.8)
    assert 0.1 < float(v) <= 0.2      # swing read as 0.95 C (was 0.013 C)
    d = hinge_max_drift(vals, 0, slice(0, T), 0.5)
    assert 0.4 < float(d) <= 0.5


def test_correlation_eps_is_unit_free() -> None:
    T = 180
    idx = torch.arange(T, dtype=torch.float32)
    a = torch.sin(idx / 20.0)
    b = -a + 0.3 * torch.cos(idx / 7.0)
    traj = torch.stack([a, b], dim=1)
    v1 = hinge_max_correlation(traj, 0, 1, slice(0, T), -0.3)
    v2 = hinge_max_correlation(traj * 1e-3, 0, 1, slice(0, T), -0.3)
    v3 = hinge_max_correlation(traj * 1e3, 0, 1, slice(0, T), -0.3)
    assert abs(float(v1) - float(v2)) < 1e-4 and abs(float(v1) - float(v3)) < 1e-4
    # A flat series against a moving one: finite value and finite gradient.
    flat = torch.zeros(T, requires_grad=True)
    tr = torch.stack([flat, b], dim=1)
    v = hinge_max_correlation(tr, 0, 1, slice(0, T), -0.3)
    v.backward()
    assert torch.isfinite(v) and torch.isfinite(flat.grad).all()


def test_evening_trough_bands_are_inside_their_windows() -> None:
    # Review 4.6: the bands ran to 26 h on 24 h arms. A trough at 23:00 must
    # satisfy both rules; a trough at 20:00 must violate them.
    T = 1440
    idx = np.arange(T, dtype=np.float32)
    for rule, marker in ((CORTISOL_EVENING_TROUGH, "cortisol"), (ACTH_EVENING_TROUGH, "acth")):
        ctx = rule_context_for_arm(rule.arms[0])
        for trough_h, expect_ok in ((23.0, True), (20.0, False)):
            traj = np.tile(np.zeros(STATE_DIM, dtype=np.float32), (T, 1))
            traj[:, MARKER_INDEX[marker]] = 10.0 + 8.0 * np.cos(2 * np.pi * (idx - trough_h * 60) / T + np.pi)
            v = float(rule.predicate(torch.tensor(traj), ctx))
            assert (v == 0.0) == expect_ok, (rule.name, trough_h, v)


# --- 4.5: pre-fasted arm + teacher audit ----------------------------------------

def test_fast_24_48_arm_starts_from_the_24h_fasted_row() -> None:
    assert _FAST_24H_TO_48H_ARM.prefast_hours == 24.0
    row = cold_initial_state_for_arm(_FAST_24H_TO_48H_ARM, PatientParams())
    assert row[MARKER_INDEX["liver_glycogen"]] < 70.0   # fed row is 100 g
    assert row[MARKER_INDEX["bhb"]] > 0.5              # ketosis has started
    assert row[MARKER_INDEX["insulin"]] < 8.0


def test_legacy_arm_cold_init_is_the_teacher_row_zero() -> None:
    arm = CohortArmSpec(label="x", duration_min=120, start_hour=8.0, meals=((60.0, 75.0, 0.0, 0.0),))
    row = cold_initial_state_for_arm(arm, PatientParams(), rng=np.random.default_rng(0))
    ref = teacher_arm_trajectory(arm, PatientParams(), noise_scale=0.0)[0]
    # Only the 1e-3 cold-init noise separates them.
    assert np.abs(row - ref).max() < 0.01 * max(1.0, float(np.abs(ref).max()))


def test_teacher_correction_flag_needs_a_note() -> None:
    with pytest.raises(ValueError):
        PhysiologyRule(
            name="x", source="t", description="t", arms=(_FAST_24H_TO_48H_ARM,),
            predicate=lambda traj, ctx: traj.new_tensor(0.0), scale=1.0,
            teacher_correction=True,
        )


@pytest.mark.slow
def test_rules_teacher_audit_has_no_unflagged_violations() -> None:
    """Every rule the current teacher violates is a declared correction.

    Measured at this commit (default patient, training frame): 2 of 61 rules
    violated, both flagged (ghrelin_rises_pre_meal, glucagon_rises_during_fast).
    Before iter 97: 22 of 61. Re-run scripts/rules_teacher_audit.py after any
    teacher change.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "rules_teacher_audit", os.path.join(REPO, "scripts", "rules_teacher_audit.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    rows, unflagged = mod.audit(list(PHYSIOLOGY_RULES), n_patients=0)
    assert unflagged == [], unflagged
    flagged = sorted(r["rule"] for r in rows if r["violated"])
    assert len(flagged) <= 4, flagged


# --- 1.2: the sleep arm, measured -------------------------------------------------

def test_sleep_rules_hold_on_the_teacher_under_the_corrected_arm() -> None:
    """With the inverted mask (pre-iter-97) the teacher violated
    sbp_falls_during_sleep by 24.6 mmHg (loss 6.04), hrv_higher_during_sleep by
    12.2 ms (1.50) and hr_falls_during_sleep by 7.4 bpm (0.54). Corrected: 0.
    """
    traj = torch.tensor(teacher_arm_trajectory(_SLEEP_WAKE_24H_ARM, PatientParams()))
    ctx = rule_context_for_arm(_SLEEP_WAKE_24H_ARM)
    for rule in (SBP_FALLS_DURING_SLEEP, HR_FALLS_DURING_SLEEP, HRV_HIGHER_DURING_SLEEP, SBP_MORNING_SURGE):
        assert float(rule.predicate(traj, ctx)) == 0.0, rule.name


def test_sbp_morning_surge_is_a_rise_from_the_sleep_trough() -> None:
    # Kario 2003: surge = post-wake mean minus night trough. A trajectory whose
    # 24 h maximum is at 12:00 but rises +15 from the night must SATISFY it.
    T = 1440
    traj = np.zeros((T, STATE_DIM), dtype=np.float32)
    sbp = np.full(T, 118.0, dtype=np.float32)
    sbp[120:300] = 104.0          # 02:00-05:00 trough
    sbp[390:570] = 119.0          # 06:30-09:30
    sbp[700:760] = 135.0          # midday maximum
    traj[:, MARKER_INDEX["sbp"]] = sbp
    ctx = rule_context_for_arm(_SLEEP_WAKE_24H_ARM)
    assert float(SBP_MORNING_SURGE.predicate(torch.tensor(traj), ctx)) == 0.0
    sbp[390:570] = 108.0          # no surge
    traj[:, MARKER_INDEX["sbp"]] = sbp
    assert float(SBP_MORNING_SURGE.predicate(torch.tensor(traj), ctx)) > 0.0


# --- 4.6 / 4.3: adaptive weights multiply the base weight, capped ----------------

def test_adaptive_multipliers_multiply_base_and_cap_share() -> None:
    base = {"a": 1.0, "b": 0.5, "c": 1.0, "d": 1.0}
    ema = {"a": 100.0, "b": 100.0, "c": 1.0, "d": 1.0}
    w = adaptive_multipliers(ema, base, cap_share=0.25)
    total = sum(base.values())
    assert abs(sum(w.values()) - total) < 1e-9
    assert max(w.values()) <= 0.25 * total + 1e-9
    # Same EMA, half the base weight -> half the adaptive weight (before the cap
    # binds on 'a'); the hand-set down-weight survives.
    w2 = adaptive_multipliers({"a": 10.0, "b": 10.0, "c": 1.0, "d": 1.0}, base, cap_share=1.0)
    assert abs(w2["b"] / w2["a"] - 0.5) < 1e-9
    # Members without an EMA sit at the population-mean multiplier (one), not zero.
    w3 = adaptive_multipliers({"a": 5.0, "c": 1.0}, base, cap_share=1.0)
    assert w3["d"] == pytest.approx(base["d"] * sum(base.values()) / sum(base.values()), rel=0.5)
    assert w3["a"] > w3["d"] > w3["c"] > 0.0


def test_rules_signal_ema_is_in_scale_units() -> None:
    arm = CohortArmSpec(label="m", duration_min=60, start_hour=8.0, meals=())

    def pred(min_rise: float):
        return lambda traj, ctx: hinge_min_rise(
            traj, ctx.col("glucose"), pre=ctx.window(0.0, 10.0), post=ctx.window(40.0, 60.0), min_rise=min_rise,
        )

    minutes = PhysiologyRule(name="minutes", source="t", description="t", arms=(arm,),
                             predicate=pred(120.0), scale=120.0, init_mode=InitMode.NORM_CENTER)
    mmol = PhysiologyRule(name="mmol", source="t", description="t", arms=(arm,),
                          predicate=pred(0.05), scale=0.05, init_mode=InitMode.NORM_CENTER)
    sig = RolloutEvidenceSignal(rules=[minutes, mmol], weight=WeightSchedule(0.1), adaptive=True)
    sig._update_rule_ema({
        "minutes": {"violation_mean": 120.0},   # one full scale
        "mmol": {"violation_mean": 0.05},       # one full scale
    })
    assert sig._violation_ema["minutes"] == pytest.approx(sig._violation_ema["mmol"])
    w = sig._adaptive_weights_from_ema()
    assert w["minutes"] == pytest.approx(w["mmol"])
