# Copyright 2024 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

import unittest

from jax import numpy as jnp, random

from ..wirings import AutoNCP, NCP


class TestNCP(unittest.TestCase):

    @staticmethod
    def get_submatrices(
        inter_neurons: int,
        command_neurons: int,
        motor_neurons: int,
    ) -> tuple[tuple[int, int], ...]:
        return (
            (0, motor_neurons),
            (motor_neurons, vertex := (motor_neurons + command_neurons)),
            (vertex, vertex + inter_neurons),
        )

    def test_autoncp(self) -> None:
        units = 6
        output_size = 1
        auto_ncp = AutoNCP(units, output_size)
        self.assertEqual(auto_ncp.units, units)
        self.assertEqual(auto_ncp.output_dim, output_size)

    def test_ncp(self) -> None:
        inter_neurons = 3
        command_neurons = 2
        motor_neurons = 1
        sensory_fanout = 1
        inter_fanout = 1
        recurrent_motor_synapses = 2
        motor_fanin = 1
        input_dim = 2
        ncp = NCP(
            inter_neurons,
            command_neurons,
            motor_neurons,
            sensory_fanout,
            inter_fanout,
            recurrent_motor_synapses,
            motor_fanin,
        )
        units = inter_neurons + command_neurons + motor_neurons
        self.assertEqual(ncp.units, units)
        self.assertEqual(ncp.output_dim, motor_neurons)
        key = random.key(0)
        adjacency_matrix = ncp.init_adjacency_matrix(key)
        self.assertTupleEqual(adjacency_matrix.shape, (units, units))
        motor_slice, command_slice, inter_slice = TestNCP.get_submatrices(
            inter_neurons,
            command_neurons,
            motor_neurons,
        )
        for polarity in jnp.unique(adjacency_matrix):
            self.assertIn(polarity, [-1, 0, 1])
        self.assertFalse(
            jnp.all(
                adjacency_matrix[
                    inter_slice[0] : inter_slice[1],
                    command_slice[0] : command_slice[1],
                ]
                == 0
            )
        )
        self.assertFalse(
            jnp.all(
                adjacency_matrix[
                    command_slice[0] : command_slice[1],
                    command_slice[0] : command_slice[1],
                ]
                == 0
            )
        )
        self.assertFalse(
            jnp.all(
                adjacency_matrix[
                    command_slice[0] : command_slice[1],
                    motor_slice[0] : motor_slice[1],
                ]
                == 0
            )
        )
        self.assertTrue(
            jnp.all(
                adjacency_matrix[
                    inter_slice[0] : inter_slice[1],
                    inter_slice[0] : inter_slice[1],
                ]
                == 0
            )
        )
        self.assertTrue(
            jnp.all(
                adjacency_matrix[
                    motor_slice[0] : motor_slice[1],
                    motor_slice[0] : motor_slice[1],
                ]
                == 0
            )
        )
        self.assertTrue(
            jnp.all(
                adjacency_matrix[
                    inter_slice[0] : inter_slice[1],
                    motor_slice[0] : motor_slice[1],
                ]
                == 0
            )
        )
        self.assertTrue(
            jnp.all(
                adjacency_matrix[
                    command_slice[0] : command_slice[1],
                    inter_slice[0] : inter_slice[1],
                ]
                == 0
            )
        )
        self.assertTrue(
            jnp.all(
                adjacency_matrix[
                    motor_slice[0] : motor_slice[1],
                    inter_slice[0] : inter_slice[1],
                ]
                == 0
            )
        )
        self.assertTrue(
            jnp.all(
                adjacency_matrix[
                    motor_slice[0] : motor_slice[1],
                    command_slice[0] : command_slice[1],
                ]
                == 0
            )
        )
        sensory_adjacency_matrix = ncp.init_sensory_adjacency_matrix(
            key,
            input_dim,
        )
        self.assertEqual(ncp.input_dim, input_dim)
        self.assertTupleEqual(
            sensory_adjacency_matrix.shape,
            (input_dim, units),
        )
        for polarity in jnp.unique(sensory_adjacency_matrix):
            self.assertIn(polarity, [-1, 0, 1])
        self.assertFalse(
            jnp.all(
                sensory_adjacency_matrix[
                    :input_dim, inter_slice[0] : inter_slice[1]
                ]
                == 0
            )
        )
        self.assertTrue(
            jnp.all(
                sensory_adjacency_matrix[
                    :input_dim, command_slice[0] : command_slice[1]
                ]
                == 0
            )
        )
        self.assertTrue(
            jnp.all(
                sensory_adjacency_matrix[
                    :input_dim, motor_slice[0] : motor_slice[1]
                ]
                == 0
            )
        )


if __name__ == "__main__":
    unittest.main()
