"""The clock and Donga's one-night clamp.

Donga et al., J Clin Endocrinol Metab 2010;95:2963, Table 1. A 4 h night
centered at 03:00 leaves 4 h of debt; an 8.5 h night centered there leaves
none. At that debt, insulin-dependent disposal, hepatic insulin suppression
and lipolysis match the clamp row. Basal insulin is unchanged.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from pulse.circadian import (
    CLAMP_INSULIN_FOLD,
    DEBT_OF_RESTRICTED_NIGHT_H,
    RD_NORMAL,
    RD_SHORT,
    peripheral_factor,
    step_clock,
    step_clock_t,
)
from pulse.knowledge.cohorts.sleep import DONGA_2010_BASAL_GLUCOSE
from pulse.knowledge.full_body import PatientParams, glucose_fluxes, simulate_full_body
from pulse.model import ModularPhysiologyNetwork, integrate
from pulse.types import MARKER_INDEX, NORM_CENTER


def _trace(bed_h: float, wake_h: float, hours: float = 36.0, start_h: float = 7.0):
    n = int(hours * 60)
    offset = debt = 0.0
    debts = []
    offsets = []
    for t in range(n):
        hour = (start_h + t / 60.0) % 24.0
        asleep = 1.0 if bed_h <= (start_h + t / 60.0) < wake_h else 0.0
        offset, debt = step_clock(offset, debt, hour, asleep)
        debts.append(debt)
        offsets.append(offset)
    return np.array(offsets), np.array(debts)


class TestTheNight(unittest.TestCase):
    def test_four_hours_commits_four_and_eight_and_a_half_commits_none(self) -> None:
        _, short = _trace(25.0, 29.0)
        _, full = _trace(23.0, 31.5)
        # 11:00–12:00 the next day, the table's basal window.
        window = slice(1680, 1740)
        self.assertAlmostEqual(float(short[window].mean()), 4.0, delta=0.02)
        self.assertAlmostEqual(float(full[window].mean()), 0.0, delta=0.02)

    def test_a_centered_night_does_not_move_the_phase_and_a_late_one_does(self) -> None:
        centered, _ = _trace(23.0, 31.5)
        late, _ = _trace(27.0, 35.0)  # 03:00–11:00, midpoint 07:00
        # 23:00–07:30 is centered at 03:15, a quarter-hour past the reference.
        self.assertLess(abs(float(centered[1740])), 0.12)
        self.assertGreater(float(late[1740]), 0.3)
        self.assertLess(float(late[1740]), 1.0)

    def test_the_tensor_step_matches_the_scalar_step(self) -> None:
        offset = debt = 0.0
        off_t = torch.zeros(())
        debt_t = torch.zeros(())
        for t in range(36 * 60):
            hour = (7.0 + t / 60.0) % 24.0
            clock = 7.0 + t / 60.0
            asleep = 1.0 if 25.0 <= clock < 29.0 else 0.0
            offset, debt = step_clock(offset, debt, hour, asleep)
            off_t, debt_t = step_clock_t(
                off_t, debt_t, torch.tensor(hour), torch.tensor(asleep))
        self.assertAlmostEqual(float(off_t), offset, places=4)
        self.assertAlmostEqual(float(debt_t), debt, places=4)

    def test_a_morning_before_any_bedtime_is_not_charged(self) -> None:
        _, debt = _trace(23.0, 31.5, hours=14.0)
        self.assertAlmostEqual(float(debt[:12 * 60].max()), 0.0, places=5)


class TestTheClampRow(unittest.TestCase):
    def test_peripheral_disposal_scales_with_the_published_ratio(self) -> None:
        self.assertAlmostEqual(peripheral_factor(0.0), 1.0, places=6)
        self.assertAlmostEqual(
            peripheral_factor(DEBT_OF_RESTRICTED_NIGHT_H), RD_SHORT / RD_NORMAL, places=4)

    def test_basal_insulin_is_unchanged_and_the_clamp_is_the_table(self) -> None:
        p = PatientParams()
        rest = dict(G=p.Gb, Gn=p.Gnb, Cort=p.Cort_b, FFA=p.FFA_b,
                    LGly=p.LGly_b, MGly=p.MGly_b, Hep=p.Hep_b)
        basal_0 = glucose_fluxes(p, I=p.Ib, X=0.0, Ra_carb=0.0, act=0.0, debt=0.0, **rest)
        basal_4 = glucose_fluxes(
            p, I=p.Ib, X=0.0, Ra_carb=0.0, act=0.0,
            debt=DEBT_OF_RESTRICTED_NIGHT_H, **rest)
        self.assertAlmostEqual(basal_4["hep_target"], basal_0["hep_target"], places=6)

        fold = CLAMP_INSULIN_FOLD * p.Ib
        clamp_0 = glucose_fluxes(p, I=fold, X=0.0, Ra_carb=0.0, act=0.0, debt=0.0, **rest)
        clamp_4 = glucose_fluxes(
            p, I=fold, X=0.0, Ra_carb=0.0, act=0.0,
            debt=DEBT_OF_RESTRICTED_NIGHT_H, **rest)
        self.assertAlmostEqual(
            clamp_4["hep_target"] / clamp_0["hep_target"], 4.4 / 3.6, places=3)

        from pulse.circadian import lipolysis_scale
        scale = lipolysis_scale(fold, p.Ib, p.IC50_lip, DEBT_OF_RESTRICTED_NIGHT_H)
        self.assertAlmostEqual(scale, 68.0 / 57.0, places=3)
        self.assertAlmostEqual(
            lipolysis_scale(p.Ib, p.Ib, p.IC50_lip, DEBT_OF_RESTRICTED_NIGHT_H), 1.0, places=6)


class TestTheRollout(unittest.TestCase):
    def test_the_two_nights_leave_basal_glucose_and_four_hours_of_debt(self) -> None:
        spec = DONGA_2010_BASAL_GLUCOSE
        p = PatientParams()
        window = slice(spec.window.start_min, spec.window.end_min)
        arms = []
        for arm in spec.arms:
            traj, _ = simulate_full_body(
                p, list(arm.meals), np.asarray(arm.sleep_wake, dtype=np.float32),
                np.zeros(arm.duration_min, dtype=np.float32),
                arm.duration_min, start_hour=arm.start_hour, noise_scale=0.0,
            )
            arms.append(traj)
        normal, short = arms
        self.assertAlmostEqual(
            float(normal[window, MARKER_INDEX["sleep_debt"]].mean()), 0.0, delta=0.05)
        self.assertAlmostEqual(
            float(short[window, MARKER_INDEX["sleep_debt"]].mean()), 4.0, delta=0.05)
        glucose_delta = float(
            short[window, MARKER_INDEX["glucose"]].mean()
            - normal[window, MARKER_INDEX["glucose"]].mean())
        self.assertLess(abs(glucose_delta), spec.sigma)
        self.assertLess(
            abs(float(short[window, MARKER_INDEX["circ_phase_offset"]].mean()
                      - normal[window, MARKER_INDEX["circ_phase_offset"]].mean())),
            0.15)

    def test_the_student_clock_is_the_teacher_clock(self) -> None:
        n = 26 * 60
        start_h = 7.0
        sw = torch.ones(n)
        sw[int((25.0 - start_h) * 60):int((29.0 - start_h) * 60)] = 0.0
        p = PatientParams()
        teacher, _ = simulate_full_body(
            p, [], sw.numpy(), np.zeros(n, dtype=np.float32),
            n, start_hour=start_h, noise_scale=0.0,
        )
        model = ModularPhysiologyNetwork()
        model.eval()
        with torch.no_grad():
            student = integrate(
                model, torch.tensor(NORM_CENTER), torch.zeros(model.embedding_dim), n,
                start_time_minutes=start_h * 60.0,
                sleep_wake=sw, activity=torch.zeros(n),
            )
        for name in ("circ_phase_offset", "sleep_debt"):
            torch.testing.assert_close(
                student[1:, MARKER_INDEX[name]],
                torch.tensor(teacher[:-1, MARKER_INDEX[name]], dtype=torch.float32),
                atol=1e-3, rtol=1e-3,
            )


if __name__ == "__main__":
    unittest.main()
