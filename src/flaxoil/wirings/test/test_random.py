# Copyright 2024 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

import unittest

from ..random import PRNG


class TestPRNG(unittest.TestCase):
    def setUp(self) -> None:
        self.prng = PRNG(0)

    def test_choice_a_single(self) -> None:
        result = self.prng.choice([0])
        self.assertEqual(result, 0)

    def test_choice_b_list(self) -> None:
        choices = [0, 1, 2, 4]
        result = self.prng.choice(choices)
        self.assertIn(result, choices)

    def test_choice_c_multiple(self) -> None:
        n = 2
        choices = [0, 1, 2, 4]
        results = self.prng.choice(choices, n)
        for result in results:
            self.assertIn(result, choices)
        self.assertEqual(len(results), n)

    def test_choice_d_replace(self) -> None:
        choices = [0, 1, 2, 4]
        counters = {n: 0 for n in choices}
        results = self.prng.choice(choices, 16, replace=True)
        for result in results:
            self.assertIn(result, choices)
            counters[int(result)] += 1
        self.assertTrue(any(map(lambda n: n >= 2, counters.values())))


if __name__ == "__main__":
    unittest.main()
