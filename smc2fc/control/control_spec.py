from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Union
import jax.numpy as jnp

@dataclass(frozen=True)
class ControlSpec:
    name: str
    version: str
    dt: float
    n_steps: int
    n_substeps: int = 1
    initial_state: jnp.ndarray = field(default=None)
    truth_params: Dict[str, float] = field(default_factory=dict)
    theta_dim: int = 0
    sigma_prior: float = 1.5
    prior_mean: Union[float, jnp.ndarray] = 0.0
    cost_fn: Optional[Callable] = None
    schedule_from_theta: Optional[Callable] = None
    acceptance_gates: Dict[str, Callable] = field(default_factory=dict)
    diagnostic_plot_fn: Optional[Callable] = None

class RBFSchedule:
    def __init__(self, n_steps, dt, n_anchors, output='identity'):
        self.n_steps = n_steps
        self.dt = dt
        self.n_anchors = n_anchors
        self.output = output
    def design_matrix(self):
        import jax.numpy as jnp
        return jnp.ones((self.n_steps, self.n_anchors))
