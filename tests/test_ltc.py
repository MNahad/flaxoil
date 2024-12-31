# Copyright 2024 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

import unittest

from flax import linen as nn
from jax import numpy as jnp
import jax.random

from flaxoil import ltc


class TestLTC(unittest.TestCase):

    def setUp(self) -> None:
        points = 10
        self.t = jnp.linspace(0, 2 * jnp.pi, points)
        self.x = jnp.sin(self.t).reshape(1, points, 1)
        self.elapsed_time = jnp.arange(points).reshape(1, points, 1)

    def test_ltc(self) -> None:
        output_size = 1
        ltc_cell = ltc.LTCCell(
            {"ncp": {"units": 6, "output_size": output_size}}
        )
        rnn = nn.RNN(
            ltc_cell,
            variable_broadcast=["params", "wirings_constants"],
        )
        rngs = {"params": jax.random.key(0)}
        variables = rnn.init(rngs, self.x)
        self.assertSetEqual(
            set(variables.keys()),
            {"params", "wirings_constants"},
        )
        y = rnn.apply(variables, self.x, rngs=rngs)
        self.assertTupleEqual(y.shape, (*self.x.shape[:-1], output_size))

    def test_ltc_irregular_time(self) -> None:
        output_size = 1
        ltc_cell = ltc.LTCCell(
            {"ncp": {"units": 6, "output_size": output_size}},
            irregular_time_mode=True,
        )
        rnn = nn.RNN(
            ltc_cell,
            variable_broadcast=["params", "wirings_constants"],
        )
        rngs = {"params": jax.random.key(0)}
        inputs = jnp.dstack((self.elapsed_time, self.x))
        variables = rnn.init(rngs, inputs)
        y = rnn.apply(variables, inputs, rngs=rngs)
        self.assertTupleEqual(y.shape, (*inputs.shape[:-1], output_size))


if __name__ == "__main__":
    unittest.main()
