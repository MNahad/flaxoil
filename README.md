# Flaxoil

## Liquid Neural Networks on Flax

This package is based on [ncps](https://github.com/mlech26l/keras-ncp).

It runs on [JAX](https://github.com/google/jax) and [Flax](https://github.com/google/flax).

# Installation

```shell
# Warning: Make sure this is run inside a virtual environment
pip install --user -U git+https://github.com/mnahad/flaxoil.git@main
```

# Quickstart

```python
from flax import linen as nn
from flaxoil import ltc, wirings
from jax import numpy as jnp
import jax.random
```

```python
t = jnp.linspace(0, 2 * jnp.pi, 629)
x = jnp.sin(t).reshape(1, 629, 1)
```

```python
units = 6
output_size = 1

wiring = wirings.AutoNCP(units, output_size)

model = nn.RNN(
    ltc.LTCCell(wiring),
    variable_broadcast=["params", "wirings_constants"],
)
```

```python
variables = model.init(jax.random.key(0), x)
params = variables["params"]
wirings_constants = variables["wirings_constants"]
```

```python
y_predicted = model.apply(
    {"params": params, "wirings_constants": wirings_constants},
    x,
)
```

# Package Manifest

This package exports Neural Circuit Policies[1] wirings classes and Liquid Time-constant[2] RNN cells.

| Submodule | Class | Description |
| :---: | :---: | :--- |
| `wirings` | NCP | Creates an NCP wiring instance for use with Liquid neural networks |
| `wirings` | AutoNCP | NCP helper class |
| `ltc` | LTCCell | Create a Liquid Time-constant neural network cell to use with `flax.linen.RNN` |

# Examples

Refer to the [examples](/examples).

# References

1. M. Lechner, R. Hasani, A. Amini, T. A. Henzinger, D. Rus, and R. Grosu, "Neural circuit policies enabling auditable autonomy," Nature Machine Intelligence, vol. 2, no. 10, pp. 642-652, Oct 2020.
1. R. Hasani, M. Lechner, A. Amini, D. Rus, and R. Grosu, "Liquid Time-constant Networks", AAAI, vol. 35, no. 9, pp. 7657-7666, May 2021.
