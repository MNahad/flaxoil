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

from flax import linen as nn
import flax.core
import flax.struct
import jax
from jax import numpy as jnp

from ..wirings import wirings


class LTCCell(nn.RNNCellBase):
    """A `Liquid time-constant (LTC) <https://doi.org/10.1609/aaai.v35i9.16936>`_ cell.

    .. Note::
        This is an RNNCell that processes single time-steps.
        To get a full RNN that can process sequences,
        wrap the cell in a `flax.linen.RNN`.

    :param wiring_params: The wiring params to use with this instance
    :param irregular_time_mode: Is elapsed time irregular?
        If True, then insert elapsed time as first feature
    :param epsilon: A small epsilon value to prevent dividing ODE terms by zero
    :param ode_unfolds: Number of ODE unfold iterations
    :param init_ranges: Initialisation ranges for ODE terms
    :num_feature_axes: Required by flax.linen.RNNCellBase
    """

    wiring_params: wirings.WiringParams
    irregular_time_mode: bool = False
    epsilon: float = 1e-8
    ode_unfolds: int = 6
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
    num_feature_axes: int = 1

    def initialize_carry(
        self,
        rng: jax.Array,
        input_shape: tuple[int, ...],
    ) -> jax.Array:
        shape = input_shape[:-1] + (
            wirings.WiringFactory.get_core_params(self.wiring_params)[0],
        )
        return nn.initializers.zeros_init()(rng, shape)

    @nn.compact
    def __call__(
        self,
        carry: jax.Array,
        inputs: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        wiring = wirings.WiringFactory.new(self.wiring_params)
        if wiring is None:
            raise RuntimeError(
                "Incorrect Wiring parameters. Please pass in the correct parameters."
            )
        if self.irregular_time_mode:
            elapsed_time, inputs = inputs[..., :1], inputs[..., 1:]
        else:
            elapsed_time = jnp.full((*inputs.shape[:-1], 1), 1.0)
        input_dim = inputs.shape[-1]
        params = self._get_params(wiring, input_dim)
        inputs = self._map_inputs(params, inputs)
        next_state = self._ode_solver(params, inputs, carry, elapsed_time)
        outputs = self._map_outputs(params, next_state)
        return next_state, outputs

    def _get_params(
        self,
        wiring: wirings.Wiring,
        input_dim: int,
    ) -> flax.core.FrozenDict:

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

        rng = self.make_rng()
        params = {}
        for name, shape in (
            ("gleak", (wiring.units,)),
            ("vleak", (wiring.units,)),
            ("cm", (wiring.units,)),
            ("w", (wiring.units, wiring.units)),
            ("sigma", (wiring.units, wiring.units)),
            ("mu", (wiring.units, wiring.units)),
            ("sensory_w", (input_dim, wiring.units)),
            ("sensory_sigma", (input_dim, wiring.units)),
            ("sensory_mu", (input_dim, wiring.units)),
        ):
            params[name] = get_param(name, shape, jnp.float32)
        params["erev"] = self.param(
            "erev",
            lambda _: wiring.init_adjacency_matrix(rng).astype(jnp.float32),
        )
        params["sensory_erev"] = self.param(
            "sensory_erev",
            lambda _: wiring.init_sensory_adjacency_matrix(
                rng,
                input_dim,
            ).astype(jnp.float32),
        )
        params["input_w"] = self.param(
            "input_w",
            nn.initializers.constant(1.0),
            (input_dim,),
            jnp.float32,
        )
        params["input_b"] = self.param(
            "input_b",
            nn.initializers.constant(0.0),
            (input_dim,),
            jnp.float32,
        )
        params["output_w"] = self.param(
            "output_w",
            nn.initializers.constant(1.0),
            (wiring.output_dim,),
            jnp.float32,
        )
        params["output_b"] = self.param(
            "output_b",
            nn.initializers.constant(0.0),
            (wiring.output_dim,),
            jnp.float32,
        )
        params["sparsity_mask"] = self.variable(
            "wirings_constants",
            "sparsity_matrix",
            lambda: jnp.abs(wiring.init_adjacency_matrix(rng)),
        )
        params["sensory_sparsity_mask"] = self.variable(
            "wirings_constants",
            "sensory_sparsity_matrix",
            lambda: jnp.abs(
                wiring.init_sensory_adjacency_matrix(rng, input_dim)
            ),
        )
        return flax.core.freeze(params)

    @nn.jit
    def _ode_solver(
        self,
        params: flax.core.FrozenDict,
        inputs: jax.Array,
        state: jax.Array,
        elapsed_time: jax.Array,
    ) -> jax.Array:

        def sigmoid(v_pre, mu, sigma):
            v_pre = jnp.expand_dims(v_pre, axis=-1)
            mues = v_pre - mu
            x = sigma * mues
            return nn.sigmoid(x)

        v_pre = state
        sensory_w_activation = nn.softplus(params["sensory_w"]) * sigmoid(
            inputs, params["sensory_mu"], params["sensory_sigma"]
        )
        sensory_w_activation *= params["sensory_sparsity_mask"].value
        sensory_rev_activation = sensory_w_activation * params["sensory_erev"]
        w_numerator_sensory = jnp.sum(sensory_rev_activation, axis=1)
        w_denominator_sensory = jnp.sum(sensory_w_activation, axis=1)
        cm_t = nn.softplus(params["cm"]) / (elapsed_time / self.ode_unfolds)
        w = nn.softplus(params["w"])
        gleak = nn.softplus(params["gleak"])
        for _ in range(self.ode_unfolds):
            w_activation = w * sigmoid(v_pre, params["mu"], params["sigma"])
            w_activation *= params["sparsity_mask"].value
            rev_activation = w_activation * params["erev"]
            w_numerator = jnp.sum(rev_activation, axis=1) + w_numerator_sensory
            w_denominator = (
                jnp.sum(w_activation, axis=1) + w_denominator_sensory
            )
            numerator = cm_t * v_pre + gleak * params["vleak"] + w_numerator
            denominator = cm_t + gleak + w_denominator
            v_pre = numerator / (denominator + self.epsilon)
        return v_pre

    @nn.jit
    def _map_inputs(
        self,
        params: flax.core.FrozenDict,
        inputs: jax.Array,
    ) -> jax.Array:
        inputs = inputs * params["input_w"]
        inputs = inputs + params["input_b"]
        return inputs

    @nn.jit
    def _map_outputs(
        self,
        params: flax.core.FrozenDict,
        state: jax.Array,
    ) -> jax.Array:
        units, motor_size = wirings.WiringFactory.get_core_params(
            self.wiring_params
        )
        output = state[..., : min(motor_size, units)]
        output = output * params["output_w"]
        output = output + params["output_b"]
        return output
