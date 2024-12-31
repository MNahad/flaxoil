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

from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

from jax import Array, numpy as jnp, random


type _WiringParams = dict[str, int]
type WiringParams = dict[str, _WiringParams]


class Wiring(Protocol):
    units: int
    input_dim: int
    output_dim: int

    def init_adjacency_matrix(self, key: Array) -> Array: ...

    def init_sensory_adjacency_matrix(
        self,
        key: Array,
        input_dim: int,
    ) -> Array: ...


@dataclass
class NCP:
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

    inter_neurons: int
    command_neurons: int
    motor_neurons: int
    sensory_fanout: int
    inter_fanout: int
    recurrent_command_synapses: int
    motor_fanin: int
    units: int = field(init=False)
    input_dim: int = field(init=False, default=0)
    output_dim: int = field(init=False)

    @staticmethod
    def _apply_on_submatrix(
        key: Array,
        matrix: Array,
        col_row_slices: tuple[tuple[int, int], tuple[int, int]],
        transform: Callable[[Array], Array],
        axis: int,
        filter_zeroed_only: bool = False,
    ) -> Array:
        row_slice = col_row_slices[1]
        col_slice = col_row_slices[0]
        axis_length = col_row_slices[axis][1] - col_row_slices[axis][0]
        keys = jnp.expand_dims(random.split(key, axis_length), axis)
        sub_matrix = (
            matrix[row_slice[0] : row_slice[1], col_slice[0] : col_slice[1]]
            * filter_zeroed_only
        )
        matrix = matrix.at[
            row_slice[0] : row_slice[1], col_slice[0] : col_slice[1]
        ].set(
            sub_matrix
            + (
                jnp.all(sub_matrix == 0, axis=axis, keepdims=True)
                * jnp.apply_along_axis(
                    lambda key_arr: transform(key_arr[0]),
                    axis,
                    keys,
                )
            )
        )
        return matrix

    @staticmethod
    def _clip(n: int, minimum: int, maximum: int) -> int:
        return min(max(n, minimum), maximum)

    @staticmethod
    def _get_wires(key: Array, amount: int, total: int) -> Array:
        wires = random.choice(
            key,
            jnp.array([-1, 1], dtype=jnp.int32),
            shape=(amount,),
        )
        wires = jnp.pad(
            wires,
            (0, total - amount),
            mode="constant",
            constant_values=(0,),
        )
        wires = random.permutation(key, wires)
        return wires

    def __post_init__(self) -> None:
        self.units = (
            self.inter_neurons + self.command_neurons + self.motor_neurons
        )
        self.output_dim = self.motor_neurons

    def init_adjacency_matrix(self, key: Array) -> Array:
        adj = jnp.zeros((self.units, self.units), dtype=jnp.int32)
        motor_range = (0, self.motor_neurons)
        command_range = (
            motor_range[1],
            motor_range[1] + self.command_neurons,
        )
        inter_range = (
            command_range[1],
            command_range[1] + self.inter_neurons,
        )
        mean_command_neuron_fanin = NCP._clip(
            int(self.inter_neurons * self.inter_fanout / self.command_neurons),
            1,
            self.command_neurons,
        )

        def inter_command_wire_fn(key: Array) -> Array:
            return NCP._get_wires(
                key,
                self.inter_fanout,
                self.command_neurons,
            )

        def inter_command_mean_wire_fn(key: Array) -> Array:
            return NCP._get_wires(
                key,
                mean_command_neuron_fanin,
                self.inter_neurons,
            )

        adj = NCP._apply_on_submatrix(
            key,
            adj,
            (
                (command_range[0], command_range[1]),
                (inter_range[0], inter_range[1]),
            ),
            inter_command_wire_fn,
            1,
        )
        adj = NCP._apply_on_submatrix(
            key,
            adj,
            (
                (command_range[0], command_range[1]),
                (inter_range[0], inter_range[1]),
            ),
            inter_command_mean_wire_fn,
            0,
            filter_zeroed_only=True,
        )
        recurrent_command_wires = NCP._get_wires(
            key,
            self.recurrent_command_synapses,
            self.command_neurons**2,
        )
        recurrent_command_wires = jnp.reshape(
            recurrent_command_wires,
            (self.command_neurons, self.command_neurons),
        )
        adj = adj.at[
            command_range[0] : command_range[1],
            command_range[0] : command_range[1],
        ].set(recurrent_command_wires)
        mean_command_neuron_fanout = NCP._clip(
            int(self.motor_neurons * self.motor_fanin / self.command_neurons),
            1,
            self.motor_neurons,
        )

        def command_motor_wire_fn(key: Array) -> Array:
            return NCP._get_wires(
                key,
                self.motor_fanin,
                self.command_neurons,
            )

        def command_motor_mean_wire_fn(key: Array) -> Array:
            return NCP._get_wires(
                key,
                mean_command_neuron_fanout,
                self.motor_neurons,
            )

        adj = NCP._apply_on_submatrix(
            key,
            adj,
            (
                (motor_range[0], motor_range[1]),
                (command_range[0], command_range[1]),
            ),
            command_motor_wire_fn,
            0,
        )
        adj = NCP._apply_on_submatrix(
            key,
            adj,
            (
                (motor_range[0], motor_range[1]),
                (command_range[0], command_range[1]),
            ),
            command_motor_mean_wire_fn,
            1,
            filter_zeroed_only=True,
        )
        return adj

    def init_sensory_adjacency_matrix(
        self,
        key: Array,
        input_dim: int,
    ) -> Array:
        self.input_dim = input_dim
        adj = jnp.zeros((self.input_dim, self.units), dtype=jnp.int32)
        inter_range = (
            offset := self.motor_neurons + self.command_neurons,
            offset + self.inter_neurons,
        )
        mean_inter_neuron_fanin = NCP._clip(
            int(self.input_dim * self.sensory_fanout / self.inter_neurons),
            1,
            self.input_dim,
        )

        def wire_fn(key: Array) -> Array:
            return NCP._get_wires(
                key,
                self.sensory_fanout,
                self.inter_neurons,
            )

        def mean_wire_fn(key: Array) -> Array:
            return NCP._get_wires(
                key,
                mean_inter_neuron_fanin,
                self.input_dim,
            )

        adj = NCP._apply_on_submatrix(
            key,
            adj,
            ((inter_range[0], inter_range[1]), (0, self.input_dim)),
            wire_fn,
            1,
        )
        adj = NCP._apply_on_submatrix(
            key,
            adj,
            ((inter_range[0], inter_range[1]), (0, self.input_dim)),
            mean_wire_fn,
            0,
            filter_zeroed_only=True,
        )
        return adj


def AutoNCP(units: int, output_size: int, sparsity_level=0.5) -> Wiring:
    """Instantiate an NCP wiring with only needing to specify the number of units and the number of outputs

    :param units: The total number of neurons
    :param output_size: The number of motor neurons (=output size). This value must be less than units-2 (typically good choices are 0.3 times the total number of units)
    :param sparsity_level: A hyperparameter between 0.0 (very dense) and 0.9 (very sparse) NCP.
    :param seed: Random seed for generating the wiring
    """
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
    return NCP(
        inter_neurons,
        command_neurons,
        output_size,
        sensory_fanout,
        inter_fanout,
        recurrent_command_synapses,
        motor_fanin,
    )


class WiringFactory:

    @staticmethod
    def new(wiring_params: WiringParams) -> Optional[Wiring]:
        """Create a new Wiring instance

        :param wiring_params: The params to use to construct the Wiring object
        :return object: A new Wiring instance
        """
        if "ncp" in wiring_params:
            ncp_params = wiring_params["ncp"]
            if "units" in ncp_params:
                return AutoNCP(
                    ncp_params["units"],
                    ncp_params["output_size"],
                    ncp_params.get("sparsity_level", 0.5),
                )
            else:
                return NCP(**ncp_params)
        else:
            return None

    @staticmethod
    def get_core_params(wiring_params: WiringParams) -> tuple[int, int]:
        """Get core wiring params from config

        :param wiring_params: The config
        :return tuple: (units, output_dim)
        """
        units, output_dim = 0, 0
        if "ncp" in wiring_params:
            ncp_params = wiring_params["ncp"]
            if "units" in ncp_params:
                units = ncp_params["units"]
            else:
                units = (
                    ncp_params["inter_neurons"]
                    + ncp_params["command_neurons"]
                    + ncp_params["motor_neurons"]
                )
            if "output_size" in ncp_params:
                output_dim = ncp_params["output_size"]
            else:
                output_dim = ncp_params["motor_neurons"]
        return units, output_dim
