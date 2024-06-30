# Copyright 2024 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

import unittest

from jax import numpy as jnp

from ..wirings import AutoNCP, NCP


class TestNCP(unittest.TestCase):
    def test_autoncp(self) -> None:
        units = 6
        output_size = 1
        input_dim = 2
        auto_ncp = AutoNCP(units, output_size)
        self.assertEqual(auto_ncp.units, units)
        self.assertEqual(auto_ncp.output_dim, output_size)
        auto_ncp.build(input_dim)
        self.assertEqual(auto_ncp.input_dim, input_dim)

    def test_ncp(self) -> None:
        inter_neurons = 3
        command_neurons = 2
        motor_neurons = 1
        sensory_fanout = 1
        inter_fanout = 1
        recurrent_motor_synapses = 2
        motor_fanin = 1
        seed = 0
        input_dim = 2
        ncp = NCP(
            inter_neurons,
            command_neurons,
            motor_neurons,
            sensory_fanout,
            inter_fanout,
            recurrent_motor_synapses,
            motor_fanin,
            seed,
        )
        units = inter_neurons + command_neurons + motor_neurons
        self.assertEqual(ncp.units, units)
        self.assertEqual(ncp.output_dim, motor_neurons)
        ncp.build(input_dim)
        self.assertEqual(ncp.input_dim, input_dim)
        adjacency_matrix = ncp.erev_initializer()
        sensory_adjacency_matrix = ncp.sensory_erev_initializer()
        self.assertTupleEqual(adjacency_matrix.shape, (units, units))
        for polarity in jnp.unique(adjacency_matrix):
            self.assertIn(polarity, [-1, 0, 1])
        self.assertTupleEqual(
            sensory_adjacency_matrix.shape,
            (input_dim, units),
        )
        for polarity in jnp.unique(sensory_adjacency_matrix):
            self.assertIn(polarity, [-1, 0, 1])


if __name__ == "__main__":
    unittest.main()
