"""The literature queue stays internally honest, and the diastolic rule it landed holds."""

from __future__ import annotations

import unittest

import torch

from pulse.knowledge.physiology_rules import (
    PHYSIOLOGY_RULES,
    RuleContext,
    hinge_max_rise,
)
from pulse.knowledge.queue import HUBS, QUEUE, Status, blocked_items, ready_items
from pulse.types import MARKER_INDEX, STATE_DIM


def _registry_names() -> set[str]:
    """Training and held-out entries a landed queue row is allowed to name."""
    from pulse.knowledge.cohort_statistics import (
        ALL_COHORT_STATISTICS,
        SEALED_COHORT_STATISTICS,
    )
    from pulse.knowledge.physiology_rules import SEALED_PHYSIOLOGY_RULES
    from pulse.modules.metabolic import POPULATION_PRIOR_NAMES

    names = {rule.name for rule in PHYSIOLOGY_RULES}
    names |= {rule.name for rule in SEALED_PHYSIOLOGY_RULES}
    names |= {spec.name for spec in ALL_COHORT_STATISTICS}
    names |= {spec.name for spec in SEALED_COHORT_STATISTICS}
    names |= set(POPULATION_PRIOR_NAMES)
    return names


class TestQueue(unittest.TestCase):
    def test_ids_are_unique(self) -> None:
        ids = [item.id for item in QUEUE]
        self.assertEqual(len(ids), len(set(ids)))

    def test_blocked_items_name_a_hub(self) -> None:
        self.assertGreater(len(blocked_items()), 0)
        for item in blocked_items():
            self.assertIn(item.blocked_on, HUBS, item.id)

    def test_ready_items_name_a_real_marker(self) -> None:
        for item in ready_items():
            self.assertIn(item.marker_id, MARKER_INDEX, item.id)
            self.assertIsNone(item.blocked_on, item.id)

    def test_integrated_items_point_at_a_registry_entry(self) -> None:
        names = _registry_names()
        integrated = [item for item in QUEUE if item.status is Status.INTEGRATED]
        self.assertGreater(len(integrated), 0)
        for item in integrated:
            self.assertIn(item.spec_name, names, item.id)
            self.assertIsNone(item.blocked_on, item.id)

    def test_renal_specs_are_in_exactly_one_registry(self) -> None:
        from pulse.knowledge.cohort_statistics import (
            ALL_COHORT_STATISTICS,
            SEALED_COHORT_STATISTICS,
        )
        from pulse.knowledge.cohorts.renal import COHORT_STATISTICS
        from pulse.knowledge.sealing import paper_is_sealed

        train = {s.name for s in ALL_COHORT_STATISTICS}
        held = {s.name for s in SEALED_COHORT_STATISTICS}
        self.assertEqual(train & held, set())
        for spec in COHORT_STATISTICS:
            self.assertIn(spec.name, held if paper_is_sealed(spec.paper) else train, spec.name)

    def test_landed_rows_name_the_spec_and_blocked_rows_keep_the_real_gap(self) -> None:
        by_id = {item.id: item for item in QUEUE}
        names = _registry_names()
        threshold = by_id["glycosuria-threshold"]
        urine = by_id["empagliflozin-24h-uge"]
        cck_clock = by_id["liddle-1985-cck-peak-time"]
        self.assertIs(cck_clock.status, Status.BLOCKED)
        self.assertEqual(cck_clock.blocked_on, "m2.2")
        self.assertNotIn("cck_time_to_peak", names)
        self.assertNotIn("cck_postprandial_peak", names)
        self.assertIn("cck_fasting_basal", names)

        self.assertIs(threshold.status, Status.INTEGRATED)
        self.assertEqual(threshold.spec_name, "renal_glucose_at_190")
        self.assertIs(urine.status, Status.INTEGRATED)
        self.assertEqual(urine.spec_name, "empagliflozin_24h_urinary_glucose")
        for name in (
            "renal_glucose_at_250",
            "renal_glucose_at_300",
            "renal_glucose_at_84_empagliflozin",
            "renal_glucose_at_200_empagliflozin",
            "donga_2010_clamp_glucose_disposal",
            "donga_2010_clamp_endogenous_production",
            "donga_2010_clamp_nefa",
        ):
            self.assertIn(name, names, name)
        donga = by_id["donga-2010-sleep-clamp"]
        for name in (
            "donga_2010_clamp_glucose_disposal",
            "donga_2010_clamp_endogenous_production",
            "donga_2010_clamp_nefa",
        ):
            self.assertIn(name, donga.note, name)
        salehi = by_id["glp1-suppresses-glucagon"]
        self.assertIs(salehi.status, Status.BLOCKED)
        self.assertIn("antagonist", salehi.note)
        vilsboll = by_id["vilsboll-2003-equal-incretins"]
        self.assertIs(vilsboll.status, Status.BLOCKED)
        self.assertIsNone(vilsboll.spec_name)

    def test_a_backlog_item_is_not_pretending_to_be_ready(self) -> None:
        cole = next(item for item in QUEUE if item.id == "cole-1999-hrr1")
        self.assertIs(cole.status, Status.BACKLOG)
        self.assertIsNone(cole.spec_name)


class TestDiastolicCeiling(unittest.TestCase):
    def test_registered(self) -> None:
        names = {rule.name for rule in PHYSIOLOGY_RULES}
        self.assertIn("dbp_does_not_rise_with_exercise", names)

    def _violation(self, pre: float, post: float) -> float:
        rule = next(r for r in PHYSIOLOGY_RULES if r.name == "dbp_does_not_rise_with_exercise")
        traj = torch.zeros(120, STATE_DIM)
        col = MARKER_INDEX["dbp"]
        traj[:30, col] = pre
        traj[60:90, col] = post
        ctx = RuleContext(marker_index=MARKER_INDEX, step_min=1.0, arm=rule.arms[0])
        return float(rule.predicate(traj, ctx))

    def test_a_fall_and_a_small_rise_are_free(self) -> None:
        self.assertEqual(self._violation(80.0, 74.0), 0.0)
        self.assertEqual(self._violation(80.0, 89.0), 0.0)

    def test_a_systolic_sized_rise_is_a_violation(self) -> None:
        self.assertAlmostEqual(self._violation(80.0, 100.0), 10.0)

    def test_helper_matches_the_ceiling(self) -> None:
        traj = torch.zeros(10, STATE_DIM)
        traj[:4, 0] = 1.0
        traj[4:, 0] = 6.0
        self.assertAlmostEqual(float(hinge_max_rise(traj, 0, slice(0, 4), slice(4, 10), 3.0)), 2.0)
        self.assertEqual(float(hinge_max_rise(traj, 0, slice(0, 4), slice(4, 10), 5.0)), 0.0)
