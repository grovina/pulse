"""Tests for the per-spec cohort ablation diagnostic."""
from __future__ import annotations

import unittest

from pulse.diagnostics.cohort_ablation import cohort_ablation, render
from pulse.knowledge import ALL_COHORT_STATISTICS
from pulse.model import ModularPhysiologyNetwork


def _tiny_model() -> ModularPhysiologyNetwork:
    return ModularPhysiologyNetwork(
        metabolic_hidden=16,
        appetite_hidden=24,
        stress_hidden=24,
        cardiovascular_hidden=16,
        thermoreg_hidden=16,
        respiratory_hidden=16,
    )


class TestCohortAblation(unittest.TestCase):
    def test_one_row_per_spec(self) -> None:
        model = _tiny_model()
        rows = cohort_ablation(model, sample_patients=2, specs=list(ALL_COHORT_STATISTICS[:3]))
        self.assertEqual(len(rows), 3)
        for r in rows:
            self.assertIn("gut", r.grad_norms)
            self.assertIn("metabolic", r.grad_norms)

    def test_at_least_one_module_receives_gradient(self) -> None:
        """Every spec should land *some* gradient — a fully-zero row would
        indicate the spec is structurally disconnected from the model."""
        model = _tiny_model()
        rows = cohort_ablation(model, sample_patients=2, specs=list(ALL_COHORT_STATISTICS[:5]))
        for r in rows:
            total = sum(r.grad_norms.values())
            self.assertGreater(total, 0.0, f"spec {r.name} landed no gradient on any module")

    def test_render_produces_table(self) -> None:
        model = _tiny_model()
        rows = cohort_ablation(model, sample_patients=2, specs=list(ALL_COHORT_STATISTICS[:2]))
        from pulse.diagnostics.cohort_ablation import CohortAblationReport
        report = CohortAblationReport(checkpoint="<test>", n_sample_patients=2, rows=rows)
        text = render(report)
        self.assertIn("cohort-ablation", text)
        self.assertIn("||g_met||", text)


if __name__ == "__main__":
    unittest.main()
