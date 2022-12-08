# Added to silence some warnings.
from jax.config import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import pennylane as qml
from jax import grad, jit, vmap

# import numpy as np
from pennylane import numpy as np

from typing import Any, Callable, Dict, List, Optional, Tuple


def x0(p: int, seed: int = 1):
    # np.random.seed(seed)
    gamma_init = np.array(2 * np.pi * np.random.rand(p))
    beta_init = np.array(np.pi * np.random.rand(p))
    x0 = np.concatenate((gamma_init, beta_init), axis=0)
    return x0


def x0_flex(p: int, N: int, seed: int = 1):
    np.random.rand(seed)
    gamma_init = np.array(2 * np.pi * np.random.rand(p))
    beta_init = np.array(np.pi * np.random.rand(N * p))
    x0 = np.concatenate((gamma_init, beta_init), axis=0)
    return x0


def x0_batches(batches: int, p: int, seed: int = 1):
    np.random.rand(seed)
    gamma_init = np.array(10 * np.pi * np.random.rand(batches, p))
    beta_init = np.array(10 * np.pi * np.random.rand(batches, p))
    x0 = jnp.concatenate((gamma_init, beta_init), axis=1)
    return x0


def x0_batches_flex(batches: int, p: int, N: int, seed: int = 1):
    np.random.rand(seed)
    gamma_init = np.array(2 * np.pi * np.random.rand(batches, p))
    beta_init = np.array(np.pi * np.random.rand(batches, N * p))
    x0 = jnp.concatenate((gamma_init, beta_init), axis=1)
    return x0


def get_initial_guess(p, mixer, N, **kwargs):
    # initial guess for the parameters
    if "batch_size" in kwargs.keys():
        if mixer == "row_swap_flex":
            params = x0_batches_flex(kwargs["batch_size"], p, N, kwargs["seed"])
        else:
            params = x0_batches(kwargs["batch_size"], p, kwargs["seed"])
        # self.kwargs.pop("batch_size")
    else:
        if mixer == "row_swap_flex":
            params = x0_flex(p, N, kwargs["seed"])
        else:
            params = x0(p, kwargs["seed"])
    return params


# def x0_jax(p:int):
#   seed = 1000
#   key = jax.random.PRNGKey(seed)
#   gamma_init = jnp.array(2 * jnp.pi * jax.random.uniform(key, shape=(p,)))
#   beta_init = jnp.array(jnp.pi * jax.random.uniform(key, shape=(p,)))
#   x0 = jnp.concatenate((gamma_init, beta_init), axis=0)
#   return x0

# def x0_batches_jax(batches:int, p:int):
#   seed = 1000
#   key = jax.random.PRNGKey(seed)
#   gamma_init = jnp.array(2 * jnp.pi * jax.random.uniform(key, shape=(batches,p)))
#   beta_init = jnp.array(jnp.pi * jax.random.uniform(key, shape=(batches,p)))
#   x0 = jnp.concatenate((gamma_init, beta_init), axis=1)
#   return x0
