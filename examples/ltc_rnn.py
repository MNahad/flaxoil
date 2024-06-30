# Copyright 2024 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

import flax
from flax import linen as nn
from flax.training import train_state
from flaxoil import ltc, wirings
import jax
from jax import numpy as jnp
import optax


def get_dataset() -> tuple[jax.Array, jax.Array, jax.Array]:
    t = jnp.linspace(0, 2 * jnp.pi, 1000)
    x = jnp.sin(t).reshape(1, 1000, 1)
    y = jnp.cos(t).reshape(1, 1000, 1)
    return t, x, y


def create_train_state_and_constants(
    rnn: nn.RNN,
    rng: jax.Array,
    x: jax.Array,
    learning_rate: float,
) -> tuple[train_state.TrainState, flax.core.FrozenDict]:
    variables = rnn.init(rng, x)
    params = variables["params"]
    wirings_constants = variables["wirings_constants"]
    tx = optax.adam(learning_rate)
    return (
        train_state.TrainState.create(
            apply_fn=rnn.apply,
            params=params,
            tx=tx,
        ),
        wirings_constants,
    )


@jax.jit
def train(
    state: train_state.TrainState,
    x: jax.Array,
    y: jax.Array,
    wirings_constants: flax.core.FrozenDict,
) -> tuple[train_state.TrainState, jax.Array]:
    def loss_fn(params):
        y_predicted = state.apply_fn(
            {"params": params, "wirings_constants": wirings_constants},
            x,
        )
        loss = optax.squared_error(y_predicted, y).mean()
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def main() -> None:
    _, x, y = get_dataset()

    rnn = nn.RNN(
        ltc.LTCCell(wirings.AutoNCP(6, 1)),
        variable_broadcast=["params", "wirings_constants"],
    )

    state, wirings_constants = create_train_state_and_constants(
        rnn,
        jax.random.key(0),
        x,
        0.01,
    )

    for i in range(1, 1001):
        state, loss = train(state, x, y, wirings_constants)
        if not i % 10:
            print(f"i: {i}, loss: {loss}")


if __name__ == "__main__":
    main()
