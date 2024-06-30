# Copyright 2020-2021 Mathias Lechner
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified
# Copyright 2024 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

import jax
from jax import numpy as jnp

from .random import PRNG


class Wiring:
    def __init__(self, units: int) -> None:
        self._units = units
        self._adjacency_matrix = jnp.zeros([units, units], dtype=jnp.int32)
        self._sensory_adjacency_matrix: jax.Array | None = None
        self._input_dim: int | None = None
        self._output_dim: int | None = None

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @input_dim.setter
    def input_dim(self, input_dim: int) -> None:
        self._input_dim = input_dim
        self._sensory_adjacency_matrix = jnp.zeros(
            [self._input_dim, self._units], dtype=jnp.int32
        )

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @output_dim.setter
    def output_dim(self, output_dim: int) -> None:
        self._output_dim = output_dim

    @property
    def units(self) -> int:
        return self._units

    def build(self, input_shape: int) -> None:
        raise NotImplementedError

    def erev_initializer(self, *, dtype=jnp.int32) -> jax.Array:
        return self._adjacency_matrix.astype(dtype)

    def sensory_erev_initializer(self, *, dtype=jnp.int32) -> jax.Array:
        return self._sensory_adjacency_matrix.astype(dtype)

    def _is_input_dim_initialized(self, input_dim: int) -> bool:
        if self.input_dim is not None and self.input_dim != input_dim:
            raise ValueError(
                "Conflicting input dimensions provided. set_input_dim() was called with {} but actual input has dimension {}".format(
                    self.input_dim, input_dim
                )
            )
        return self.input_dim is not None

    def _add_synapse(self, src: int, dest: int, polarity: int) -> None:
        if src < 0 or src >= self._units:
            raise ValueError(
                "Cannot add synapse originating in {} if cell has only {} units".format(
                    src, self._units
                )
            )
        if dest < 0 or dest >= self._units:
            raise ValueError(
                "Cannot add synapse feeding into {} if cell has only {} units".format(
                    dest, self._units
                )
            )
        if polarity not in [-1, 1]:
            raise ValueError(
                "Cannot add synapse with polarity {} (expected -1 or +1)".format(
                    polarity
                )
            )
        self._adjacency_matrix = self._adjacency_matrix.at[src, dest].set(
            polarity
        )

    def _add_sensory_synapse(self, src: int, dest: int, polarity: int) -> None:
        if self._input_dim is None:
            raise ValueError(
                "Cannot add sensory synapses before build() has been called!"
            )
        if src < 0 or src >= self._input_dim:
            raise ValueError(
                "Cannot add sensory synapse originating in {} if input has only {} features".format(
                    src, self._input_dim
                )
            )
        if dest < 0 or dest >= self._units:
            raise ValueError(
                "Cannot add synapse feeding into {} if cell has only {} units".format(
                    dest, self._units
                )
            )
        if polarity not in [-1, 1]:
            raise ValueError(
                "Cannot add synapse with polarity {} (expected -1 or +1)".format(
                    polarity
                )
            )
        self._sensory_adjacency_matrix = self._sensory_adjacency_matrix.at[
            src, dest
        ].set(polarity)


class NCP(Wiring):
    def __init__(
        self,
        inter_neurons: int,
        command_neurons: int,
        motor_neurons: int,
        sensory_fanout: int,
        inter_fanout: int,
        recurrent_command_synapses: int,
        motor_fanin: int,
        seed=22222,
    ) -> None:
        """
        Creates a Neural Circuit Policies wiring.
        The total number of neurons (= state size of the RNN) is given by the sum of inter, command, and motor neurons.
        For an easier way to generate a NCP wiring see the `AutoNCP` wiring class.

        :param inter_neurons: The number of inter neurons (layer 2)
        :param command_neurons: The number of command neurons (layer 3)
        :param motor_neurons: The number of motor neurons (layer 4 = number of outputs)
        :param sensory_fanout: The average number of outgoing synapses from the sensory to the inter neurons
        :param inter_fanout: The average number of outgoing synapses from the inter to the command neurons
        :param recurrent_command_synapses: The average number of recurrent connections in the command neuron layer
        :param motor_fanin: The average number of incoming synapses of the motor neurons from the command neurons
        :param seed: The random seed used to generate the wiring
        """
        super(NCP, self).__init__(
            inter_neurons + command_neurons + motor_neurons
        )
        self.output_dim = motor_neurons
        self._rng = PRNG(seed)
        self._num_inter_neurons = inter_neurons
        self._num_command_neurons = command_neurons
        self._num_motor_neurons = motor_neurons
        self._sensory_fanout = sensory_fanout
        self._inter_fanout = inter_fanout
        self._recurrent_command_synapses = recurrent_command_synapses
        self._motor_fanin = motor_fanin

        self._motor_neurons = list(range(0, self._num_motor_neurons))
        self._command_neurons = list(
            range(
                self._num_motor_neurons,
                self._num_motor_neurons + self._num_command_neurons,
            )
        )
        self._inter_neurons = list(
            range(
                self._num_motor_neurons + self._num_command_neurons,
                self._num_motor_neurons
                + self._num_command_neurons
                + self._num_inter_neurons,
            )
        )

        if self._motor_fanin > self._num_command_neurons:
            raise ValueError(
                "Error: Motor fanin parameter is {} but there are only {} command neurons".format(
                    self._motor_fanin, self._num_command_neurons
                )
            )
        if self._sensory_fanout > self._num_inter_neurons:
            raise ValueError(
                "Error: Sensory fanout parameter is {} but there are only {} inter neurons".format(
                    self._sensory_fanout, self._num_inter_neurons
                )
            )
        if self._inter_fanout > self._num_command_neurons:
            raise ValueError(
                "Error:: Inter fanout parameter is {} but there are only {} command neurons".format(
                    self._inter_fanout, self._num_command_neurons
                )
            )

    def build(self, input_shape: int) -> None:
        if super()._is_input_dim_initialized(input_shape):
            return
        self.input_dim = input_shape
        self._num_sensory_neurons = self.input_dim
        self._sensory_neurons = list(range(0, self._num_sensory_neurons))

        self._build_sensory_to_inter_layer()
        self._build_inter_to_command_layer()
        self._build_recurrent_command_layer()
        self._build_command__to_motor_layer()

    def _build_sensory_to_inter_layer(self) -> None:
        unreachable_inter_neurons = [n for n in self._inter_neurons]
        for src in self._sensory_neurons:
            for dest in self._rng.choice(
                self._inter_neurons, shape=self._sensory_fanout, replace=False
            ):
                if dest in unreachable_inter_neurons:
                    unreachable_inter_neurons.remove(dest)
                polarity = self._rng.choice([-1, 1])
                self._add_sensory_synapse(src, dest, polarity)

        mean_inter_neuron_fanin = int(
            self._num_sensory_neurons
            * self._sensory_fanout
            / self._num_inter_neurons
        )
        mean_inter_neuron_fanin = jnp.clip(
            mean_inter_neuron_fanin, 1, self._num_sensory_neurons
        )
        for dest in unreachable_inter_neurons:
            for src in self._rng.choice(
                self._sensory_neurons,
                shape=mean_inter_neuron_fanin,
                replace=False,
            ):
                polarity = self._rng.choice([-1, 1])
                self._add_sensory_synapse(src, dest, polarity)

    def _build_inter_to_command_layer(self) -> None:
        unreachable_command_neurons = [n for n in self._command_neurons]
        for src in self._inter_neurons:
            for dest in self._rng.choice(
                self._command_neurons, shape=self._inter_fanout, replace=False
            ):
                if dest in unreachable_command_neurons:
                    unreachable_command_neurons.remove(dest)
                polarity = self._rng.choice([-1, 1])
                self._add_synapse(src, dest, polarity)

        mean_command_neurons_fanin = int(
            self._num_inter_neurons
            * self._inter_fanout
            / self._num_command_neurons
        )
        mean_command_neurons_fanin = jnp.clip(
            mean_command_neurons_fanin, 1, self._num_command_neurons
        )
        for dest in unreachable_command_neurons:
            for src in self._rng.choice(
                self._inter_neurons,
                shape=mean_command_neurons_fanin,
                replace=False,
            ):
                polarity = self._rng.choice([-1, 1])
                self._add_synapse(src, dest, polarity)

    def _build_recurrent_command_layer(self) -> None:
        for _ in range(self._recurrent_command_synapses):
            src = self._rng.choice(self._command_neurons)
            dest = self._rng.choice(self._command_neurons)
            polarity = self._rng.choice([-1, 1])
            self._add_synapse(src, dest, polarity)

    def _build_command__to_motor_layer(self) -> None:
        unreachable_command_neurons = [n for n in self._command_neurons]
        for dest in self._motor_neurons:
            for src in self._rng.choice(
                self._command_neurons, shape=self._motor_fanin, replace=False
            ):
                if src in unreachable_command_neurons:
                    unreachable_command_neurons.remove(src)
                polarity = self._rng.choice([-1, 1])
                self._add_synapse(src, dest, polarity)

        mean_command_fanout = int(
            self._num_motor_neurons
            * self._motor_fanin
            / self._num_command_neurons
        )
        mean_command_fanout = jnp.clip(
            mean_command_fanout, 1, self._num_motor_neurons
        )
        for src in unreachable_command_neurons:
            for dest in self._rng.choice(
                self._motor_neurons, shape=mean_command_fanout, replace=False
            ):
                polarity = self._rng.choice([-1, 1])
                self._add_synapse(src, dest, polarity)


class AutoNCP(NCP):
    def __init__(
        self,
        units: int,
        output_size: int,
        sparsity_level=0.5,
        seed=22222,
    ) -> None:
        """Instantiate an NCP wiring with only needing to specify the number of units and the number of outputs

        :param units: The total number of neurons
        :param output_size: The number of motor neurons (=output size). This value must be less than units-2 (typically good choices are 0.3 times the total number of units)
        :param sparsity_level: A hyperparameter between 0.0 (very dense) and 0.9 (very sparse) NCP.
        :param seed: Random seed for generating the wiring
        """
        if output_size >= units - 2:
            raise ValueError(
                f"Output size must be less than the number of units-2 (given {units} units, {output_size} output size)"
            )
        if sparsity_level < 0.1 or sparsity_level > 1.0:
            raise ValueError(
                f"Sparsity level must be between 0.0 and 0.9 (given {sparsity_level})"
            )
        density_level = 1.0 - sparsity_level
        inter_and_command_neurons = units - output_size
        command_neurons = max(int(0.4 * inter_and_command_neurons), 1)
        inter_neurons = inter_and_command_neurons - command_neurons

        sensory_fanout = max(int(inter_neurons * density_level), 1)
        inter_fanout = max(int(command_neurons * density_level), 1)
        recurrent_command_synapses = max(
            int(command_neurons * density_level * 2), 1
        )
        motor_fanin = max(int(command_neurons * density_level), 1)
        super(AutoNCP, self).__init__(
            inter_neurons,
            command_neurons,
            output_size,
            sensory_fanout,
            inter_fanout,
            recurrent_command_synapses,
            motor_fanin,
            seed,
        )
