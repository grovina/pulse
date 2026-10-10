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
        from pulse.knowledge.cohort_statistics import ALL_COHORT_STATISTICS
        from pulse.modules.metabolic import POPULATION_PRIOR_NAMES
        names = {rule.name for rule in PHYSIOLOGY_RULES}
        names |= {spec.name for spec in ALL_COHORT_STATISTICS}
        names |= set(POPULATION_PRIOR_NAMES)
        integrated = [item for item in QUEUE if item.status is Status.INTEGRATED]
        self.assertGreater(len(integrated), 0)
        for item in integrated:
            self.assertIn(item.spec_name, names, item.id)

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
