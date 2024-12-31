# Flaxoil

## Liquid Neural Networks on Flax

This package is based on [ncps](https://github.com/mlech26l/keras-ncp).

It runs on [JAX](https://github.com/google/jax) and integrates with [Flax](https://github.com/google/flax).

# Installation

```shell
# Warning: Make sure this is run inside a virtual environment
pip install --user -U git+https://github.com/mnahad/flaxoil.git@main # or @{tag}
```

# Quickstart

```python
from flax import linen as nn
from flaxoil import ltc
from jax import numpy as jnp
import jax.random
```

```python
# Generate sample data
t = jnp.linspace(0, 2 * jnp.pi, 629)
x = jnp.sin(t).reshape(1, 629, 1)
```

```python
# NCP configuration
wiring = {
    "ncp": {
        "units": 6,
        "output_size": 1,
    },
}
```

```python
model = nn.RNN(
    ltc.LTCCell(wiring),
    variable_broadcast=["params", "wirings_constants"],
)
```

```python
rngs = {"params": jax.random.key(0)}
variables = model.init(rngs, x)
# Flax variables consist of parameters and NCP wiring constants
params = variables["params"]
wirings_constants = variables["wirings_constants"]
```

```python
y_predicted = model.apply(
    {"params": params, "wirings_constants": wirings_constants},
    x,
    rngs=rngs,
)
```

# Package Manifest

This package exports Neural Circuit Policies[1] wirings and Liquid Time-constant[2] RNN cells.

| Submodule | Export | Type | Description |
| :---: | :---: | :---: | :--- |
| `wirings` | `WiringParams` | type alias | Type for defining cell wiring |
| `ltc` | `LTCCell` | class | Liquid Time-constant neural network cell for use with `flax.linen.RNN` |

# API Specification

## Wirings

### WiringParams

`WiringParams` is a type alias which has the following structure:

```python
wiring_params: wirings.WiringParams = {
    "{wiring_type}": {
        "[wiring_param]": "..."
    }
}
```

Examples for NCP wiring parameters are:

```json
{
    "ncp": {
        "units": "{int}",
        "output_size": "{int}"
    }
}
```

```json
{
    "ncp": {
        "inter_neurons": "{int}",
        "command_neurons": "{int}",
        "motor_neurons": "{int}",
        "sensory_fanout": "{int}",
        "inter_fanout": "{int}",
        "recurrent_motor_synapses": "{int}",
        "motor_fanin": "{int}"
    }
}
```

## LTC

### LTCCell

`LTCCell` is a subclass of `flax.linen.RNNCellBase` and has the following API:

```python
ltc_cell = ltc.LTCCell(
    wiring_params, # The wiring params to use with this instance
    irregular_time_mode=False, # Is elapsed time irregular?
        # If True, then insert elapsed time as first feature
    epsilon=1e-8, # A small epsilon value to prevent dividing ODE terms by zero
    ode_unfolds=6, # Number of ODE unfold iterations
    init_ranges=init_ranges, # Initialisation ranges for ODE terms
    num_feature_axes=1, # Required by flax.linen.RNNCellBase
)
```

# Examples

Refer to the [examples](/examples).

# References

1. M. Lechner, R. Hasani, A. Amini, T. A. Henzinger, D. Rus, and R. Grosu, "Neural circuit policies enabling auditable autonomy," Nature Machine Intelligence, vol. 2, no. 10, pp. 642-652, Oct 2020.
1. R. Hasani, M. Lechner, A. Amini, D. Rus, and R. Grosu, "Liquid Time-constant Networks", AAAI, vol. 35, no. 9, pp. 7657-7666, May 2021.
