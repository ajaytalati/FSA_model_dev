from dataclasses import dataclass
from collections import OrderedDict
from typing import Callable, Optional, Dict, List, Tuple

@dataclass(frozen=True)
class EstimationModel:
    name: str
    version: str
    n_states: int
    n_stochastic: int
    stochastic_indices: Tuple[int, ...]
    state_bounds: Tuple[Tuple[float, float], ...]
    param_prior_config: OrderedDict
    init_state_prior_config: OrderedDict
    frozen_params: Dict[str, float]
    propagate_fn: Callable
    diffusion_fn: Callable
    obs_log_weight_fn: Callable
    align_obs_fn: Callable
    shard_init_fn: Callable
    load_data_fn: Optional[Callable] = None
    plot_trajectory_fn: Optional[Callable] = None
    plot_residuals_fn: Optional[Callable] = None
    forward_sde_fn: Optional[Callable] = None
    get_init_theta_fn: Optional[Callable] = None
    imex_step_fn: Optional[Callable] = None
    obs_log_prob_fn: Optional[Callable] = None
    make_init_state_fn: Optional[Callable] = None
    obs_sample_fn: Optional[Callable] = None
    gaussian_obs_fn: Optional[Callable] = None
    init_cov_fn: Optional[Callable] = None
    dynamic_kernel_log_density_fn: Optional[Callable] = None
    proposal_log_density_fn: Optional[Callable] = None
    exogenous_keys: Tuple[str, ...] = ()

    @property
    def n_params(self) -> int:
        return len(self.param_prior_config)

    @property
    def n_init_states(self) -> int:
        return len(self.init_state_prior_config)

    @property
    def n_dim(self) -> int:
        return self.n_params + self.n_init_states

    @property
    def all_names(self) -> List[str]:
        return (list(self.param_prior_config.keys()) +
                list(self.init_state_prior_config.keys()))

    @property
    def param_keys(self) -> List[str]:
        return list(self.param_prior_config.keys())
