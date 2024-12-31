# Copyright 2024 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from flax import linen as nn
from flaxoil import ltc
from jax import numpy as jnp
import jax.random


def main(
    units: int = 6,
    output_size: int = 1,
    timesteps_count: int = 629,
) -> None:
    t = jnp.linspace(0, 2 * jnp.pi, timesteps_count)
    x = jnp.sin(t).reshape(1, timesteps_count, 1)
    rngs = {"params": jax.random.key(0)}

    model = nn.RNN(
        ltc.LTCCell({"ncp": {"units": units, "output_size": output_size}}),
        variable_broadcast=["params", "wirings_constants"],
    )

    variables = model.init(rngs, x)

    y_predicted = model.apply(variables, x, rngs=rngs)

    print("x.shape: ", x.shape)
    print("y_predicted.shape: ", y_predicted.shape)


if __name__ == "__main__":
    main()
