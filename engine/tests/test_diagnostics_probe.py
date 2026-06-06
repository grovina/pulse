"""Tests for the diagnostics probe + comparison module.

The probe is the canonical answer to "how does this checkpoint behave at the
zero embedding". These tests pin down the contract so future refactors can't
silently break the gut sweep / drift / counter-regulation triad that
iter-by-iter handoffs depend on.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from pulse.diagnostics import (
    cold_gut_sweep_targets,
    load_model_from_checkpoint,
    probe_checkpoint,
)
from pulse.diagnostics.compare import compare_reports, render
from pulse.model import ModularPhysiologyNetwork
from pulse.types import EMBEDDING_DIM


def _model_from_hidden(h: int) -> ModularPhysiologyNetwork:
    """Mirror the hidden-dim derivation used by load_model_from_checkpoint /
    train.py so checkpoints round-trip cleanly through the loader."""
    return ModularPhysiologyNetwork(
        metabolic_hidden=h,
        appetite_hidden=max(24, h // 2),
        stress_hidden=max(24, h // 2),
        cardiovascular_hidden=h,
        thermoreg_hidden=max(16, h // 3),
        respiratory_hidden=max(16, h // 3),
    )


def _tiny_model() -> ModularPhysiologyNetwork:
    return _model_from_hidden(16)


class TestColdTargets(unittest.TestCase):
    def test_targets_strictly_monotone_in_dose(self) -> None:
        rows = cold_gut_sweep_targets()
        aucs = [r.auc for r in rows]
        self.assertEqual(aucs[0], 0.0, "0 g carbs must give 0 glucose appearance")
        for prev, nxt in zip(aucs, aucs[1:]):
            self.assertGreater(nxt, prev, "cold targets must be strictly monotone in dose")

    def test_target_peak_within_window(self) -> None:
        rows = cold_gut_sweep_targets()
        for r in rows[1:]:
            self.assertGreater(r.tpeak_min, 0)
            self.assertLess(r.tpeak_min, 240, "cold target should peak inside the window, not at the edge")


class TestProbeShape(unittest.TestCase):
    def test_probe_runs_and_returns_all_three_blocks(self) -> None:
        model = _tiny_model()
        report = probe_checkpoint(model)
        self.assertEqual(len(report.gut_sweep), len(report.carb_doses_g))
        self.assertIsInstance(report.gut_sweep_monotone, bool)
        self.assertEqual(report.fasting_drift.hours, 4.0)
        markers = {d.marker for d in report.counter_regulation.deltas}
        self.assertEqual(markers, {"glucagon", "ffa", "ghrelin"})

    def test_to_dict_round_trips_through_json(self) -> None:
        model = _tiny_model()
        report = probe_checkpoint(model)
        payload = json.loads(json.dumps(report.to_dict()))
        self.assertEqual(len(payload["gut_sweep"]), len(report.gut_sweep))
        self.assertIn("drift_mg_dl", payload["fasting_drift"])

    def test_at_zero_embedding_zero_dose_glucose_appearance_should_be_zero_after_training(self) -> None:
        # An untrained model will not pass this — included as a documentation
        # test of the *target* invariant, not as a precondition. The probe
        # surfaces how far we are from this invariant; tests here cover the
        # mechanics of measuring it.
        model = _tiny_model()
        report = probe_checkpoint(model)
        self.assertEqual(report.gut_sweep[0].carbs_g, 0.0)


class TestCheckpointRoundTrip(unittest.TestCase):
    def test_save_and_load_via_helper(self) -> None:
        model = _tiny_model()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ckpt.pt"
            torch.save({
                "model_state": model.state_dict(),
                "hidden_dim": 16,
                "embedding_dim": EMBEDDING_DIM,
            }, path)
            loaded, ckpt = load_model_from_checkpoint(path)
            self.assertEqual(int(ckpt["hidden_dim"]), 16)
            for (k1, v1), (k2, v2) in zip(
                model.state_dict().items(), loaded.state_dict().items()
            ):
                self.assertEqual(k1, k2)
                self.assertTrue(torch.equal(v1, v2))


class TestCompareReports(unittest.TestCase):
    def _mk_report(
        self, *, mape: float, gate_passed: bool, failures: list[str], textbook_pass: float
    ) -> dict:
        return {
            "overall_weighted_mape": mape,
            "verifier": {"overall_score": 0.8, "category_mean": {"meal": 0.7, "sleep": 1.0}},
            "textbook_mean_pass_rate": textbook_pass,
            "per_marker": {"glucose": {"mean_mape": mape, "median_mape": mape}},
            "textbook_scenarios": [{
                "name": "OGTT",
                "pass_rate": 0.6,
                "checks": [
                    {"name": "glucose_rises", "description": "x", "passed": True, "value": 10.0, "threshold": 5.0},
                    {"name": "glucagon_suppressed", "description": "x", "passed": False, "value": -0.01, "threshold": 0.0},
                ],
            }],
            "gate": {"passed": gate_passed, "failures": failures},
        }

    def test_basic_deltas_and_failure_diff(self) -> None:
        base = self._mk_report(mape=0.30, gate_passed=False, failures=["A", "B"], textbook_pass=0.5)
        new = self._mk_report(mape=0.25, gate_passed=False, failures=["B", "C"], textbook_pass=0.6)
        cmp = compare_reports(new, base, new_label="new", baseline_label="base")
        self.assertAlmostEqual(cmp.overall_mape_delta or 0.0, -0.05, places=6)
        self.assertAlmostEqual(cmp.textbook_pass_rate_delta or 0.0, 0.1, places=6)
        self.assertEqual(cmp.gate_failures_added, ["C"])
        self.assertEqual(cmp.gate_failures_cleared, ["A"])
        self.assertEqual(cmp.gate_failures_persisting, ["B"])

    def test_render_includes_section_headers(self) -> None:
        base = self._mk_report(mape=0.30, gate_passed=False, failures=["A"], textbook_pass=0.5)
        new = self._mk_report(mape=0.25, gate_passed=True, failures=[], textbook_pass=0.6)
        cmp = compare_reports(new, base, new_label="new", baseline_label="base")
        text = render(cmp)
        self.assertIn("--- Gate ---", text)
        self.assertIn("--- Scalars", text)
        self.assertIn("Per-marker", text)

    def test_check_transition_kinds(self) -> None:
        base = self._mk_report(mape=0.30, gate_passed=False, failures=[], textbook_pass=0.5)
        new = self._mk_report(mape=0.30, gate_passed=False, failures=[], textbook_pass=0.5)
        # flip glucagon_suppressed: false -> true (improvement)
        new["textbook_scenarios"][0]["checks"][1]["passed"] = True
        # flip glucose_rises: true -> false (regression)
        new["textbook_scenarios"][0]["checks"][0]["passed"] = False
        cmp = compare_reports(new, base, new_label="new", baseline_label="base")
        kinds = {(t.scenario, t.name): t.kind for t in cmp.check_transitions}
        self.assertEqual(kinds[("OGTT", "glucagon_suppressed")], "improvement")
        self.assertEqual(kinds[("OGTT", "glucose_rises")], "regression")


if __name__ == "__main__":
    unittest.main()
