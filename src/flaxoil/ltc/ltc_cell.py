# Copyright 2022 Mathias Lechner and Ramin Hasani
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
from flax import linen as nn
import flax.struct
import flax.core

from ..wirings import wirings


class LTCCell(nn.RNNCellBase):
    """A `Liquid time-constant (LTC) <https://doi.org/10.1609/aaai.v35i9.16936>`_ cell.

    .. Note::
        This is an RNNCell that processes single time-steps.
        To get a full RNN that can process sequences,
        wrap the cell in a `flax.linen.RNN`.

    :param wiring: The Wiring object to use with this instance
    :param epsilon: A small epsilon value to prevent dividing ODE terms by zero
    :param elapsed_time: The elapsed time between time-series data points
    :param init_ranges: Initialisation ranges for ODE terms
    """

    wiring: wirings.Wiring
    epsilon: float = 1e-8
    elapsed_time: float = 1.0
    init_ranges: dict[str, tuple[float, float]] = flax.struct.field(
        default_factory=lambda: {
            "gleak": (0.001, 1.0),
            "vleak": (-0.2, 0.2),
            "cm": (0.4, 0.6),
            "w": (0.001, 1.0),
            "sigma": (3.0, 8.0),
            "mu": (0.3, 0.8),
            "sensory_w": (0.001, 1.0),
            "sensory_sigma": (3.0, 8.0),
            "sensory_mu": (0.3, 0.8),
        }
    )

    @property
    def num_feature_axes(self) -> int:
        return 1

    def initialize_carry(
        self,
        rng: jax.Array,
        input_shape: tuple[int, ...],
    ) -> jax.Array:
        shape = input_shape[0:-1] + (self.wiring.units,)
        return nn.initializers.zeros_init()(rng, shape)

    @nn.compact
    def __call__(
        self,
        carry: jax.Array,
        inputs: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:

        input_dim = inputs.shape[-1]

        self.wiring.build(input_dim)

        params = self._get_params()

        inputs = self._map_inputs(params, inputs)

        next_state = self._ode_solver(
            params,
            inputs,
            carry,
            self.elapsed_time,
            self.epsilon,
        )

        outputs = self._map_outputs(
            params,
            next_state,
            self.wiring.output_dim,
            self.wiring.units,
        )

        return next_state, outputs

    def _get_params(self) -> flax.core.FrozenDict:
        def get_initializer(param_name):
            minval, maxval = self.init_ranges[param_name]
            if minval == maxval:
                return nn.initializers.constant(minval)
            else:
                return (
                    lambda key, shape, dtype: nn.initializers.uniform(
                        maxval - minval
                    )(key, shape, dtype)
                    + minval
                )

        def get_param(name, shape, dtype):
            return self.param(name, get_initializer(name), shape, dtype)

        params = {}
        for name, shape in (
            ("gleak", (self.wiring.units,)),
            ("vleak", (self.wiring.units,)),
            ("cm", (self.wiring.units,)),
            ("w", (self.wiring.units, self.wiring.units)),
            ("sigma", (self.wiring.units, self.wiring.units)),
            ("mu", (self.wiring.units, self.wiring.units)),
            ("sensory_w", (self.wiring.input_dim, self.wiring.units)),
            ("sensory_sigma", (self.wiring.input_dim, self.wiring.units)),
            ("sensory_mu", (self.wiring.input_dim, self.wiring.units)),
        ):
            params[name] = get_param(name, shape, jnp.float32)
        params["erev"] = self.param(
            "erev",
            lambda _: self.wiring.erev_initializer(dtype=jnp.float32),
        )
        params["sensory_erev"] = self.param(
            "sensory_erev",
            lambda _: self.wiring.sensory_erev_initializer(dtype=jnp.float32),
        )
        params["input_w"] = self.param(
            "input_w",
            nn.initializers.constant(1.0),
            (self.wiring.input_dim,),
            jnp.float32,
        )
        params["input_b"] = self.param(
            "input_b",
            nn.initializers.constant(0.0),
            (self.wiring.input_dim,),
            jnp.float32,
        )
        params["output_w"] = self.param(
            "output_w",
            nn.initializers.constant(1.0),
            (self.wiring.output_dim,),
            jnp.float32,
        )
        params["output_b"] = self.param(
            "output_b",
            nn.initializers.constant(0.0),
            (self.wiring.output_dim,),
            jnp.float32,
        )
        params["sparsity_mask"] = self.variable(
            "wirings_constants",
            "sparsity_matrix",
            lambda: jnp.abs(self.wiring.erev_initializer()),
        )
        params["sensory_sparsity_mask"] = self.variable(
            "wirings_constants",
            "sensory_sparsity_matrix",
            lambda: jnp.abs(self.wiring.sensory_erev_initializer()),
        )
        return flax.core.freeze(params)

    @nn.jit
    def _ode_solver(
        self,
        params: flax.core.FrozenDict,
        inputs: jax.Array,
        state: jax.Array,
        elapsed_time: float,
        epsilon: float,
    ) -> jax.Array:
        def sigmoid(v_pre, mu, sigma):
            v_pre = jnp.expand_dims(v_pre, axis=-1)
            mues = v_pre - mu
            x = sigma * mues
            return nn.sigmoid(x)

        ode_unfolds = 6

        v_pre = state

        sensory_w_activation = nn.softplus(params["sensory_w"]) * sigmoid(
            inputs, params["sensory_mu"], params["sensory_sigma"]
        )

        sensory_w_activation *= params["sensory_sparsity_mask"].value

        sensory_rev_activation = sensory_w_activation * params["sensory_erev"]

        w_numerator_sensory = jnp.sum(sensory_rev_activation, axis=1)
        w_denominator_sensory = jnp.sum(sensory_w_activation, axis=1)

        cm_t = nn.softplus(params["cm"]) / (elapsed_time / ode_unfolds)

        w = nn.softplus(params["w"])
        gleak = nn.softplus(params["gleak"])
        for _ in range(ode_unfolds):
            w_activation = w * sigmoid(v_pre, params["mu"], params["sigma"])

            w_activation *= params["sparsity_mask"].value

            rev_activation = w_activation * params["erev"]

            w_numerator = jnp.sum(rev_activation, axis=1) + w_numerator_sensory
            w_denominator = (
                jnp.sum(w_activation, axis=1) + w_denominator_sensory
            )

            numerator = cm_t * v_pre + gleak * params["vleak"] + w_numerator
            denominator = cm_t + gleak + w_denominator

            v_pre = numerator / (denominator + epsilon)

        return v_pre

    def _map_inputs(
        self,
        params: flax.core.FrozenDict,
        inputs: jax.Array,
    ) -> jax.Array:
        inputs = inputs * params["input_w"]
        inputs = inputs + params["input_b"]
        return inputs

    def _map_outputs(
        self,
        params: flax.core.FrozenDict,
        state: jax.Array,
        motor_size: int,
        state_size: int,
    ) -> jax.Array:
        output = state
        if motor_size < state_size:
            output = output[:, 0:motor_size]

        output = output * params["output_w"]
        output = output + params["output_b"]
        return output
