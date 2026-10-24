"""People are drawn from the factor prior. The zero code is the median person.

A memorized patient table is not the population. ``teacher_table_rows`` is the
lookup a teacher-paired signal still uses, and it is not what a training step
draws.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn as nn

from pulse.knowledge.cohort_types import FACTORS, Population, factor_index
from pulse.training.embedding_sampler import (
    draw_people,
    select_supervised_embeddings,
    teacher_table_rows,
)
from pulse.types import EMBEDDING_DIM


class TestFactorDraws(unittest.TestCase):
    def test_a_draw_is_a_fresh_code_of_the_module_width(self) -> None:
        rng = np.random.default_rng(0)
        codes = draw_people(5, rng, "cpu")
        self.assertEqual(tuple(codes.shape), (5, EMBEDDING_DIM))
        self.assertFalse(codes.requires_grad)
        self.assertTrue(torch.equal(codes[:, len(FACTORS):], torch.zeros(5, EMBEDDING_DIM - len(FACTORS))))

    def test_the_same_seed_draws_the_same_people(self) -> None:
        a = draw_people(4, np.random.default_rng(3), "cpu")
        b = draw_people(4, np.random.default_rng(3), "cpu")
        self.assertTrue(torch.equal(a, b))
        c = draw_people(4, np.random.default_rng(4), "cpu")
        self.assertFalse(torch.equal(a, c))

    def test_zero_draws_is_an_empty_code(self) -> None:
        codes = draw_people(0, np.random.default_rng(0), "cpu")
        self.assertEqual(tuple(codes.shape), (0, EMBEDDING_DIM))

    def test_a_healthy_young_cohort_is_restricted(self) -> None:
        pop = Population(name="healthy young", sex="female", age=(20, 35), bmi=(20, 25), healthy=True)
        codes = draw_people(40, np.random.default_rng(1), "cpu", pop)
        ir = codes[:, factor_index("insulin_resistance")]
        beta = codes[:, factor_index("beta_cell")]
        sex = codes[:, factor_index("sex")]
        age = codes[:, factor_index("age")]
        bmi = codes[:, factor_index("adiposity")]
        self.assertTrue(bool((ir <= 0).all()))
        self.assertTrue(bool((beta >= 0).all()))
        self.assertTrue(torch.equal(sex, torch.full((40,), -1.0)))
        # age code (years-40)/15, years in [20, 35] → code in [(20-40)/15, (35-40)/15]
        self.assertTrue(bool((age >= (20 - 40) / 15).all() and (age <= (35 - 40) / 15).all()))
        self.assertTrue(bool((bmi >= (20 - 25) / 5).all() and (bmi <= (25 - 25) / 5).all()))


class TestSelectSupervisedEmbeddings(unittest.TestCase):
    def test_appends_the_median_person(self) -> None:
        out = select_supervised_embeddings(3, np.random.default_rng(0), "cpu", include_default=True)
        self.assertEqual(len(out), 4)
        self.assertEqual(tuple(out[0].shape), (EMBEDDING_DIM,))
        self.assertTrue(torch.equal(out[-1], torch.zeros(EMBEDDING_DIM)))
        self.assertTrue(all(not t.requires_grad for t in out))

    def test_can_omit_the_median_person(self) -> None:
        out = select_supervised_embeddings(3, np.random.default_rng(0), "cpu", include_default=False)
        self.assertEqual(len(out), 3)
        self.assertFalse(any(torch.equal(t, torch.zeros(EMBEDDING_DIM)) for t in out))

    def test_no_draws_is_the_median_person_alone(self) -> None:
        out = select_supervised_embeddings(0, np.random.default_rng(0), "cpu", include_default=True)
        self.assertEqual(len(out), 1)
        self.assertTrue(torch.equal(out[0], torch.zeros(EMBEDDING_DIM)))

    def test_no_draws_and_no_median_is_empty(self) -> None:
        out = select_supervised_embeddings(0, np.random.default_rng(0), "cpu", include_default=False)
        self.assertEqual(out, [])


class TestTeacherTableRows(unittest.TestCase):
    def test_pairs_table_rows_with_patient_ids_and_appends_zero(self) -> None:
        table = nn.Embedding(8, EMBEDDING_DIM)
        nn.init.normal_(table.weight, std=0.1)
        rows = teacher_table_rows(table, n_patients=8, sample_patients=3, rng=np.random.default_rng(0), device="cpu")
        self.assertEqual(len(rows), 4)
        self.assertIsNone(rows[-1][0])
        self.assertTrue(torch.equal(rows[-1][1], torch.zeros(EMBEDDING_DIM)))
        ids = [pid for pid, _ in rows[:-1]]
        self.assertEqual(len(set(ids)), 3)
        for pid, emb in rows[:-1]:
            self.assertTrue(torch.equal(emb, table.weight[pid]))
            self.assertTrue(emb.requires_grad)


if __name__ == "__main__":
    unittest.main()
