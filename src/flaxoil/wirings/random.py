# Copyright 2024 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

from jax import Array, numpy as jnp, random


class PRNG:
    def __init__(self, seed: int) -> None:
        self._key = random.key(seed)

    def choice(
        self,
        a: list[int],
        shape: tuple[int, ...] | None = None,
        replace=True,
    ) -> Array | int:
        arr = jnp.array(a)
        self._key, key = random.split(self._key)
        return random.choice(
            key, arr, (shape,) if shape is not None else (), replace
        )
