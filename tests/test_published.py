"""Measured numbers are learnable population parameters at their published values.

PLAN section 2: a number a physiologist could measure is a parameter, initialised
at today's best value with no pull back, and its declaration names the study whose
protocol identifies it.
"""

from __future__ import annotations

import math
import unittest

import torch

from pulse.diagnostics.identifiability import studies_by_parameter
from pulse.model import ModularPhysiologyNetwork, integrate
from pulse.published import Published, PublishedNumbers, initial_values
from pulse.types import EMBEDDING_DIM, MARKER_INDEX as MI, NORM_CENTER


def _groups(model: ModularPhysiologyNetwork):
    for name, mod in model._modules_by_name.items():
        for child_name, child in mod.named_children():
            if isinstance(child, PublishedNumbers):
                yield name, mod, child_name, child


class TestPublished(unittest.TestCase):
    def test_the_domain_is_checked(self) -> None:
        with self.assertRaises(ValueError):
            Published(0.0, "a study")
        with self.assertRaises(ValueError):
            Published(1.0, "a study", fraction=True)
        self.assertAlmostEqual(Published(0.25, "s", fraction=True).raw, math.log(1.0 / 3.0))
        self.assertAlmostEqual(Published(4.0, "s").raw, math.log(4.0))

    def test_init_is_the_value_and_no_step_leaves_the_domain(self) -> None:
        numbers = {"rate": Published(0.2, "s"), "share": Published(0.65, "s", fraction=True)}
        group = PublishedNumbers(numbers)
        for name, value in group().items():
            self.assertAlmostEqual(float(value), numbers[name].value, places=6)
        self.assertEqual(set(dict(group.named_parameters())), {"log_rate", "logit_share"})
        with torch.no_grad():
            for p in group.parameters():
                p.add_(50.0)
        out = group()
        self.assertGreater(float(out["rate"]), 0.0)
        self.assertLessEqual(float(out["share"]), 1.0)
        torch.testing.assert_close(
            initial_values(numbers), {k: torch.tensor(p.value) for k, p in numbers.items()})


class TestTheModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)
        cls.model = ModularPhysiologyNetwork()

    def test_every_number_names_its_study(self) -> None:
        studies = studies_by_parameter(self.model)
        params = dict(self.model.named_parameters())
        for dotted, study in studies.items():
            self.assertIn(dotted, params, f"{dotted} names a study but is not a parameter")
            self.assertTrue(study.strip(), dotted)
        for name, _, child_name, group in _groups(self.model):
            for local in dict(group.named_parameters()):
                self.assertIn(f"{name}.{child_name}.{local}", studies)

    def test_each_number_reaches_its_module_constants_under_its_own_name(self) -> None:
        """A key two groups share, or a group shares with the module's own constants,
        would be overwritten silently by the dict merge."""
        model = ModularPhysiologyNetwork()
        for name, mod, _, group in _groups(model):
            with torch.no_grad():
                for p in group.parameters():
                    p.add_(0.1)
            const = mod.constants(torch.zeros(1, mod.embedding_dim))
            for key, value in group().items():
                self.assertIn(key, const, f"{name}.{key}")
                torch.testing.assert_close(const[key], value, msg=f"{name}.{key}")
        names = [k for _, _, _, g in _groups(model) for k in g()]
        self.assertEqual(len(names), len(set(names)), "a number is published twice")

    def test_thermoreg_reads_the_metabolic_numbers_through_the_body(self) -> None:
        met = self.model.metabolic
        body = self.model.body_quantities(torch.zeros(3, EMBEDDING_DIM))
        for key, value in met.body_numbers().items():
            torch.testing.assert_close(body[key], value.expand(3))
        thermo = self.model.thermoreg
        with self.assertRaises(RuntimeError):
            thermo.bind_body(thermo.constants(torch.zeros(1, thermo.embedding_dim)), None)
        self.assertFalse(any(isinstance(c, PublishedNumbers) for c in thermo.children()))

    def test_a_rollout_moves_the_numbers_and_the_two_integrators_agree(self) -> None:
        """Planned and pointwise read the same body numbers, and a loss on the
        trajectory reaches the fat absorption floor and the thermic fractions."""
        model = ModularPhysiologyNetwork()
        state = torch.tensor(NORM_CENTER, dtype=torch.float32).clone()
        state[MI["intestinal_bile"]] = 0.0
        gut = torch.zeros(6, 4)
        gut[:, 0] = 2.0
        gut[:, 1] = 3.0
        gut[:, 2] = 1.0
        kw = dict(gut_outputs=gut, activity=torch.full((6,), 0.4),
                  sleep_wake=torch.ones(6), meals=[])
        emb = torch.zeros(EMBEDDING_DIM)
        with torch.no_grad():
            model.metabolic.thermic.logit_dit_fat.add_(1.0)
            pointwise = integrate(model, state, emb, 6, planned=False, **kw)
        planned = integrate(model, state, emb, 6, planned=True, **kw)
        torch.testing.assert_close(planned.detach(), pointwise, atol=1e-5, rtol=1e-5)
        planned[:, [MI["temp"], MI["glucose"], MI["fat_mass"]]].sum().backward()
        met = model.metabolic
        for p in (met.absorption.logit_bile_fat_floor, met.thermic.logit_dit_carb,
                  met.thermic.logit_dit_fat, met.thermic.logit_dit_prot):
            self.assertIsNotNone(p.grad)
            self.assertNotEqual(float(p.grad), 0.0)


if __name__ == "__main__":
    unittest.main()
