import jax.numpy as jnp
import numpy as np
from typing import Dict

def build_crn_noise_grids(
    *,
    n_inner: int,
    n_steps: int,
    n_channels: int = 1,
    seed: int = 0,
) -> Dict[str, jnp.ndarray]:
    rng = np.random.default_rng(seed)
    wiener = jnp.asarray(
        rng.standard_normal((n_inner, n_steps, n_channels)),
        dtype=jnp.float64,
    )
    initial = jnp.asarray(
        rng.standard_normal((n_inner,)), dtype=jnp.float64,
    )
    return {'wiener': wiener, 'initial': initial}
