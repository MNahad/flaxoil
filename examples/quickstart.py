# Copyright 2024 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from flax import linen as nn
from flaxoil import ltc, wirings
from jax import numpy as jnp
import jax.random


def main(units: int, output_size: int, timesteps_count: int) -> None:
    t = jnp.linspace(0, 2 * jnp.pi, timesteps_count)
    x = jnp.sin(t).reshape(1, timesteps_count, 1)

    wiring = wirings.AutoNCP(units, output_size)

    model = nn.RNN(
        ltc.LTCCell(wiring),
        variable_broadcast=["params", "wirings_constants"],
    )

    variables = model.init(jax.random.key(0), x)

    y_predicted = model.apply(variables, x)

    print("x.shape: ", x.shape)
    print("y_predicted.shape: ", y_predicted.shape)


if __name__ == "__main__":
    main(6, 1, 629)
