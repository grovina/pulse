"""dietary_carb_flow_phases_for_ui aligns with scenario constants."""

from __future__ import annotations

import unittest

from pulse.knowledge.textbook_scenarios.flow_story_protocol import (
    DIETARY_CARB_GLUCOSE_RECOVERY_START,
    DIETARY_CARB_PRE_MEAL_END,
    DIETARY_CARB_PROTOCOL_MEAL_REF_MIN,
    dietary_carb_flow_phases_for_ui,
)


class TestDietaryCarbFlowPhasesForUi(unittest.TestCase):
    def test_protocol_meal_matches_reference_windows(self) -> None:
        duration = 480
        meal = DIETARY_CARB_PROTOCOL_MEAL_REF_MIN
        phases = {p["id"]: p for p in dietary_carb_flow_phases_for_ui(duration, meal)}

        self.assertEqual(phases["baseline"]["start_min"], 0.0)
        self.assertEqual(phases["baseline"]["end_min"], float(DIETARY_CARB_PRE_MEAL_END))

        self.assertEqual(phases["appearance"]["start_min"], 30.0)
        self.assertEqual(phases["appearance"]["end_min"], 200.0)

        self.assertEqual(phases["recovery"]["start_min"], float(DIETARY_CARB_GLUCOSE_RECOVERY_START))
        self.assertEqual(phases["recovery"]["end_min"], 475.0)

    def test_later_meal_shifts_meal_relative_bands(self) -> None:
        duration = 480
        meal = 100.0
        phases = {p["id"]: p for p in dietary_carb_flow_phases_for_ui(duration, meal)}
        # appearance was 30–200 at ref meal 30 → 0–170 relative → 100–270 absolute
        self.assertEqual(phases["appearance"]["start_min"], 100.0)
        self.assertEqual(phases["appearance"]["end_min"], 270.0)
        # baseline unchanged
        self.assertEqual(phases["baseline"]["end_min"], float(DIETARY_CARB_PRE_MEAL_END))

    def test_no_meal_omits_meal_relative_phases(self) -> None:
        phases = dietary_carb_flow_phases_for_ui(480, None)
        ids = {p["id"] for p in phases}
        self.assertIn("baseline", ids)
        self.assertIn("recovery", ids)
        self.assertNotIn("appearance", ids)


if __name__ == "__main__":
    unittest.main()
