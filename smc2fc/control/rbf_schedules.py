from dataclasses import dataclass
import jax
import jax.numpy as jnp

@dataclass(frozen=True)
class RBFSchedule:
    n_steps: int
    dt: float
    n_anchors: int
    width_factor: float = 1.0
    output: str = 'identity'

    def design_matrix(self) -> jnp.ndarray:
        T_total = self.n_steps * self.dt
        centres = jnp.linspace(0.0, T_total, self.n_anchors)
        width = (T_total / max(self.n_anchors, 1)) * self.width_factor
        t_grid = jnp.arange(self.n_steps) * self.dt
        return jnp.exp(
            -0.5 * ((t_grid[:, None] - centres[None, :]) / width) ** 2
        )

    def from_theta(self, theta: jnp.ndarray, Phi: jnp.ndarray | None = None
                     ) -> jnp.ndarray:
        if Phi is None:
            Phi = self.design_matrix()
        raw = jnp.einsum('a,ta->t', theta, Phi)
        if self.output == 'identity':
            return raw
        elif self.output == 'softplus':
            return jax.nn.softplus(raw)
        elif self.output == 'sigmoid':
            return jax.nn.sigmoid(raw)
        else:
            raise ValueError(f"unknown output transform: {self.output!r}")
