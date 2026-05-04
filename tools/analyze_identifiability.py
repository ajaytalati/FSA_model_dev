"""Fisher Information Matrix analysis for FSA-v2 identifiability.

Computes the FIM for FSA's 30 estimable parameters under a 1-day constant-Φ
trajectory. Reports:

- FIM rank (how many parameters are mathematically identifiable from the
  4-channel observation model under this excitation)
- Condition number (how ill-conditioned the inversion would be)
- Identifiable subset: parameter names whose FIM diagonal exceeds a
  threshold, separately for each channel and combined.

ALL FOUR observation channels are included in the sensitivity calculation
(HR, sleep-Bernoulli, stress, log-steps). This is deliberate — the SWAT
analogue tool initially omitted the sleep channel and silently reported
its sleep-cutoff parameters (`c_tilde`, `delta_c`) as unidentifiable; for
FSA we cover all channels so each parameter gets its full information.

The analysis is computed analytically via JAX's `jax.jacobian` of the
mean trajectory (Gaussian channels) and probability trajectory (Bernoulli
sleep channel) with respect to the estimable parameter vector. No Monte
Carlo sampling — the FIM is exact at the chosen reference point.
"""
import os
import sys
import math
from collections import OrderedDict
from pathlib import Path

# Force CPU + X64.
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'True'

import jax
import jax.numpy as jnp
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.fsa_high_res._dynamics import drift_jax
from models.fsa_high_res.simulation import (
    DEFAULT_PARAMS,
    DEFAULT_INIT,
    BINS_PER_DAY,
    DT_BIN_DAYS,
    circadian_jax,
)
from models.fsa_high_res.estimation import (
    PARAM_PRIOR_CONFIG,
    _PI,
    PHI_FROZEN,
)


# Estimable parameters in _PI order
_ESTIMABLE_NAMES = list(_PI.keys())


def _frozen_subset(p_dict: dict) -> dict:
    """Frozen params: anything in DEFAULT_PARAMS that isn't in PARAM_PRIOR_CONFIG."""
    return {k: jnp.float64(v) for k, v in p_dict.items()
            if isinstance(v, (int, float)) and k not in _PI}


def compute_fim(n_days: float = 1.0, Phi_const: float = 1.0):
    """Compute FIM at DEFAULT_PARAMS over a constant-Phi 1-day trajectory.

    Returns:
        fim:    (n_params, n_params) FIM matrix
        names:  list of parameter names matching FIM rows/cols
        per_channel_diag: dict {channel_name: (n_params,) FIM diag from that channel alone}
    """
    n_bins = int(round(n_days * BINS_PER_DAY))
    dt = DT_BIN_DAYS

    # Initial state
    y0 = jnp.array([DEFAULT_INIT['B_0'],
                     DEFAULT_INIT['F_0'],
                     DEFAULT_INIT['A_0']], dtype=jnp.float64)
    Phi = jnp.full(n_bins, Phi_const, dtype=jnp.float64)

    # Time grid + circadian C(t)
    t_grid = jnp.arange(n_bins, dtype=jnp.float64) * dt
    C_grid = circadian_jax(t_grid, phi=PHI_FROZEN)

    # Frozen params (incl. sigma_B, sigma_F, sigma_A, phi) — these do NOT
    # vary in the FIM analysis, only the 30 estimable params do.
    frozen = _frozen_subset(DEFAULT_PARAMS)
    p_truth_vec = jnp.array(
        [DEFAULT_PARAMS[name] for name in _ESTIMABLE_NAMES],
        dtype=jnp.float64)

    def _rollout_and_obs_means(theta_vec):
        """Forward-roll deterministic SDE under theta and return per-channel mean traj.

        Returns:
            traj:      (n_bins, 3) deterministic [B, F, A] trajectory
            hr_mean:   (n_bins,) HR predicted mean
            stress_mean: (n_bins,) stress predicted mean
            log_step_mean: (n_bins,) log-steps predicted mean
            sleep_p:   (n_bins,) Bernoulli sleep prob
        """
        # Build params dict (estimable + frozen)
        p_dict = {name: theta_vec[i] for i, name in enumerate(_ESTIMABLE_NAMES)}
        for fname, fval in frozen.items():
            p_dict[fname] = fval

        # Deterministic Euler rollout
        def step(y, k):
            d_y = drift_jax(y, p_dict, Phi[k])
            y_next = y + dt * d_y
            return y_next, y_next
        _, traj = jax.lax.scan(step, y0, jnp.arange(n_bins))

        B, F, A = traj[:, 0], traj[:, 1], traj[:, 2]

        hr_mean = (p_dict['HR_base']
                   - p_dict['kappa_B_HR'] * B
                   + p_dict['alpha_A_HR'] * A
                   + p_dict['beta_C_HR'] * C_grid)
        stress_mean = (p_dict['S_base']
                        + p_dict['k_F'] * F
                        - p_dict['k_A_S'] * A
                        + p_dict['beta_C_S'] * C_grid)
        log_step_mean = (p_dict['mu_step0']
                          + p_dict['beta_B_st'] * B
                          - p_dict['beta_F_st'] * F
                          + p_dict['beta_A_st'] * A
                          + p_dict['beta_C_st'] * C_grid)
        sleep_z = (p_dict['k_C'] * C_grid
                   + p_dict['k_A'] * A
                   - p_dict['c_tilde'])
        sleep_p = jax.nn.sigmoid(sleep_z)

        return traj, hr_mean, stress_mean, log_step_mean, sleep_p

    # Jacobians wrt theta — one per channel
    print(f"Computing per-channel sensitivity Jacobians (n_bins={n_bins}, "
          f"n_params={len(_ESTIMABLE_NAMES)})...")
    _rollout_means_only = lambda th: _rollout_and_obs_means(th)[1:]
    J_hr, J_stress, J_logstep, J_sleep = jax.jacobian(_rollout_means_only)(p_truth_vec)
    # Each J_X has shape (n_bins, n_params)

    # --- Build the FIM, channel by channel ---
    sigma_HR = float(DEFAULT_PARAMS['sigma_HR'])
    sigma_S  = float(DEFAULT_PARAMS['sigma_S'])
    sigma_st = float(DEFAULT_PARAMS['sigma_st'])

    # Gaussian: FIM_jk = sum_t (1/sigma^2) * J_jt * J_kt
    fim_hr     = (J_hr.T @ J_hr) / (sigma_HR ** 2)
    fim_stress = (J_stress.T @ J_stress) / (sigma_S ** 2)
    fim_steps  = (J_logstep.T @ J_logstep) / (sigma_st ** 2)

    # Bernoulli sleep: FIM_jk = sum_t [J_jt * J_kt] / [p_t (1 - p_t)]
    _, _, _, _, sleep_p = _rollout_and_obs_means(p_truth_vec)
    bern_var = sleep_p * (1.0 - sleep_p) + 1e-12
    weighted_J = J_sleep / jnp.sqrt(bern_var[:, None])
    fim_sleep = weighted_J.T @ weighted_J

    fim_total = np.array(fim_hr + fim_stress + fim_steps + fim_sleep)

    # Analytical FIM contribution for the noise-scale parameters
    # (sigma_HR, sigma_S, sigma_st). For a Gaussian observation
    # y ~ N(μ, σ²), the FIM diagonal entry for σ is 2·n_obs / σ².
    # Without this, the sensitivity-of-mean analysis would report sigma
    # params as unidentifiable, which is wrong — they ARE identifiable
    # from the variance of residuals at scale.
    n_obs = float(n_bins)
    for sigma_name, sigma_value in [('sigma_HR', sigma_HR),
                                       ('sigma_S',  sigma_S),
                                       ('sigma_st', sigma_st)]:
        idx = _PI[sigma_name]
        fim_total[idx, idx] += 2.0 * n_obs / (sigma_value ** 2)

    per_channel_diag = {
        'HR':     np.asarray(jnp.diag(fim_hr)),
        'stress': np.asarray(jnp.diag(fim_stress)),
        'steps':  np.asarray(jnp.diag(fim_steps)),
        'sleep':  np.asarray(jnp.diag(fim_sleep)),
    }

    return fim_total, _ESTIMABLE_NAMES, per_channel_diag


FIM_DIAG_THRESHOLD = 1e-3   # FIM diagonal entries below this are unidentifiable


def main():
    print("=== FSA-v2 FIM Identifiability Analysis ===")
    fim, names, per_channel = compute_fim(n_days=1.0, Phi_const=1.0)
    diag = np.diag(fim)

    print("\n--- Per-channel FIM diagonal contributions ---")
    print(f"{'param':<14} {'HR':>10} {'stress':>10} {'steps':>10} {'sleep':>10} {'TOTAL':>12}")
    for i, name in enumerate(names):
        print(f"{name:<14} "
              f"{per_channel['HR'][i]:10.2e} "
              f"{per_channel['stress'][i]:10.2e} "
              f"{per_channel['steps'][i]:10.2e} "
              f"{per_channel['sleep'][i]:10.2e} "
              f"{diag[i]:12.4e}")

    identifiable = [n for n, d in zip(names, diag) if d > FIM_DIAG_THRESHOLD]
    print(f"\n--- Summary ---")
    eigvals = np.linalg.eigvalsh(fim)
    rank = int(np.sum(eigvals > 1e-8))
    cond = float(eigvals[-1] / max(eigvals[0], 1e-18))
    print(f"  FIM rank:                        {rank} / {len(names)}")
    print(f"  Condition number:                {cond:.4e}")
    print(f"  Identifiable params (diag > {FIM_DIAG_THRESHOLD:.0e}): "
          f"{len(identifiable)} / {len(names)}")
    print(f"  Identifiable subset: {identifiable}")
    return {
        'fim':                fim,
        'names':              names,
        'rank':               rank,
        'cond':               cond,
        'identifiable_subset': identifiable,
        'per_channel_diag':   per_channel,
        'passed':             rank >= 8,
    }


if __name__ == "__main__":
    main()
