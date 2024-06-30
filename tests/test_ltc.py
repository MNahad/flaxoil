# Copyright 2024 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

import unittest

from flax import linen as nn
from jax import numpy as jnp
import jax.random

from flaxoil import ltc, wirings


class TestLTC(unittest.TestCase):
    def setUp(self) -> None:
        self.t = jnp.linspace(0, 2 * jnp.pi, 10)
        self.x = jnp.sin(self.t).reshape(1, 10, 1)

    def test_ltc(self) -> None:
        output_size = 1
        ltc_cell = ltc.LTCCell(wirings.AutoNCP(6, output_size))
        rnn = nn.RNN(
            ltc_cell,
            variable_broadcast=["params", "wirings_constants"],
        )
        variables = rnn.init(jax.random.key(0), self.x)
        self.assertSetEqual(
            set(variables.keys()),
            {"params", "wirings_constants"},
        )
        y = rnn.apply(variables, self.x)
        self.assertTupleEqual(y.shape, (*self.x.shape[0:-1], output_size))


if __name__ == "__main__":
    unittest.main()
