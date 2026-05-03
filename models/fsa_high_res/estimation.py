"""version_3/models/fsa_high_res/estimation.py — FSA-v3 EstimationModel.

Bimodal Training Extension:
  - 4D state space: [B, S, F, A]
  - 2D control input: [Phi_B, Phi_S]
  - 5 observation channels: HR, Sleep, Stress, Steps, VolumeLoad
"""

import math
import numpy as np
from collections import OrderedDict

import jax
import jax.numpy as jnp

from smc2fc.estimation_model import EstimationModel
from smc2fc._likelihood_constants import HALF_LOG_2PI

from models.fsa_high_res._dynamics import A_TYP, F_TYP


# =========================================================================
# FROZEN CONSTANTS
# =========================================================================

EPS_A_FROZEN   = 1.0e-4
EPS_B_FROZEN   = 1.0e-4
EPS_S_FROZEN   = 1.0e-4
SIGMA_B_FROZEN = 0.010
SIGMA_S_FROZEN = 0.008
SIGMA_F_FROZEN = 0.012
SIGMA_A_FROZEN = 0.020
PHI_FROZEN     = 0.0


# =========================================================================
# PRIORS — FSA-v3
# =========================================================================

PARAM_PRIOR_CONFIG = OrderedDict([
    # --- Dynamics ---
    ('tau_B',       ('lognormal', (math.log(42.0), 0.10))),
    ('kappa_B',     ('lognormal', (math.log(0.01248), 0.20))),
    ('epsilon_AB',  ('lognormal', (math.log(0.40), 0.05))),
    
    ('tau_S',       ('lognormal', (math.log(60.0), 0.10))),
    ('kappa_S',     ('lognormal', (math.log(0.00816), 0.20))),
    ('epsilon_AS',  ('lognormal', (math.log(0.20), 0.05))),

    ('tau_F',       ('lognormal', (math.log(6.3636), 0.15))),
    ('kappa_FB',    ('lognormal', (math.log(0.030), 0.20))),
    ('kappa_FS',    ('lognormal', (math.log(0.050), 0.20))),
    ('lambda_A',    ('lognormal', (math.log(1.00), 0.05))),

    ('mu_0',        ('lognormal', (math.log(0.036), 0.20))),
    ('mu_B',        ('lognormal', (math.log(0.30), 0.20))),
    ('mu_S',        ('lognormal', (math.log(0.15), 0.20))),
    ('mu_F',        ('lognormal', (math.log(0.26), 0.20))),
    ('mu_FF',       ('lognormal', (math.log(0.40), 0.05))),
    ('eta',         ('lognormal', (math.log(0.20), 0.15))),

    # --- Obs Coefficients ---
    ('HR_base',     ('normal',    (62.0, 2.0))),
    ('kappa_B_HR',  ('lognormal', (math.log(12.0), 0.15))),
    ('alpha_A_HR',  ('lognormal', (math.log(3.0),  0.20))),
    ('beta_C_HR',   ('normal',    (-2.5, 0.5))),
    ('sigma_HR',    ('lognormal', (math.log(2.0), 0.20))),

    ('k_C',         ('lognormal', (math.log(3.0), 0.15))),
    ('k_A',         ('lognormal', (math.log(2.0), 0.25))),
    ('c_tilde',     ('normal',    (0.5, 0.25))),

    ('S_base',      ('normal',    (30.0, 3.0))),
    ('k_F',         ('lognormal', (math.log(20.0), 0.20))),
    ('k_A_S',       ('lognormal', (math.log(8.0),  0.25))),
    ('beta_C_S',    ('normal',    (-4.0, 0.8))),
    ('sigma_S',     ('lognormal', (math.log(4.0), 0.20))),

    ('mu_step0',    ('normal',    (5.5, 0.3))),
    ('beta_B_st',   ('lognormal', (math.log(0.8), 0.20))),
    ('beta_F_st',   ('lognormal', (math.log(0.5), 0.25))),
    ('beta_A_st',   ('lognormal', (math.log(0.3), 0.25))),
    ('beta_C_st',   ('normal',    (-0.8, 0.2))),
    ('sigma_st',    ('lognormal', (math.log(0.5), 0.15))),

    ('beta_S_VL',   ('lognormal', (math.log(100.0), 0.15))),
    ('beta_F_VL',   ('lognormal', (math.log(20.0),  0.20))),
    ('sigma_VL',    ('lognormal', (math.log(10.0),  0.20))),
])

INIT_STATE_PRIOR_CONFIG = OrderedDict()
COLD_START_INIT = jnp.array([0.05, 0.10, 0.30, 0.10])

_PK = list(PARAM_PRIOR_CONFIG.keys())
_PI = {k: i for i, k in enumerate(_PK)}


# =========================================================================
# PROPAGATE_FN — 4D [B, S, F, A], 4 Gaussian channels fused
# =========================================================================

def propagate_fn(y, t, dt, params, grid_obs, k,
                 sigma_diag, noise, rng_key):
    del t, rng_key, sigma_diag

    # --- Dynamics params ---
    tau_B      = params[_PI['tau_B']]
    kappa_B    = params[_PI['kappa_B']]
    epsilon_AB = params[_PI['epsilon_AB']]
    tau_S      = params[_PI['tau_S']]
    kappa_S    = params[_PI['kappa_S']]
    epsilon_AS = params[_PI['epsilon_AS']]
    tau_F      = params[_PI['tau_F']]
    kappa_FB   = params[_PI['kappa_FB']]
    kappa_FS   = params[_PI['kappa_FS']]
    lambda_A   = params[_PI['lambda_A']]
    mu_0       = params[_PI['mu_0']]
    mu_B       = params[_PI['mu_B']]
    mu_S       = params[_PI['mu_S']]
    mu_F       = params[_PI['mu_F']]
    mu_FF      = params[_PI['mu_FF']]
    eta        = params[_PI['eta']]

    B, S, F, A = y[0], y[1], y[2], y[3]
    Phi_k = grid_obs['Phi'][k]
    Phi_B, Phi_S = Phi_k[0], Phi_k[1]
    C_k   = grid_obs['C'][k]

    # --- Drift predictions ---
    F_dev   = F - F_TYP
    mu_bif  = mu_0 + mu_B * B + mu_S * S - mu_F * F - mu_FF * F_dev * F_dev
    a_factor_B = (1.0 + epsilon_AB * A) / (1.0 + epsilon_AB * A_TYP)
    a_factor_S = (1.0 + epsilon_AS * A) / (1.0 + epsilon_AS * A_TYP)
    a_factor_F = (1.0 + lambda_A * A)  / (1.0 + lambda_A * A_TYP)
    
    drift_B = kappa_B * a_factor_B * Phi_B - B / tau_B
    drift_S = kappa_S * a_factor_S * Phi_S - S / tau_S
    drift_F = kappa_FB * Phi_B + kappa_FS * Phi_S - a_factor_F / tau_F * F
    drift_A = mu_bif * A - eta * A * A * A
    
    y_pred_det = y + dt * jnp.array([drift_B, drift_S, drift_F, drift_A])

    # --- Prior predictive covariance ---
    B_cl = jnp.clip(B, EPS_B_FROZEN, 1.0 - EPS_B_FROZEN)
    S_cl = jnp.clip(S, EPS_S_FROZEN, 1.0 - EPS_S_FROZEN)
    F_cl = jnp.maximum(F, 0.0)
    A_cl = jnp.maximum(A, 0.0)
    vars = jnp.array([
        SIGMA_B_FROZEN ** 2 * B_cl * (1.0 - B_cl) * dt,
        SIGMA_S_FROZEN ** 2 * S_cl * (1.0 - S_cl) * dt,
        SIGMA_F_FROZEN ** 2 * F_cl * dt,
        SIGMA_A_FROZEN ** 2 * (A_cl + EPS_A_FROZEN) * dt
    ])
    P_prior = jnp.diag(jnp.maximum(vars, 1e-12))

    # --- Obs params ---
    HR_base    = params[_PI['HR_base']]
    kappa_B_HR = params[_PI['kappa_B_HR']]
    alpha_A_HR = params[_PI['alpha_A_HR']]
    beta_C_HR  = params[_PI['beta_C_HR']]
    sigma_HR   = params[_PI['sigma_HR']]

    S_base     = params[_PI['S_base']]
    k_F        = params[_PI['k_F']]
    k_A_S      = params[_PI['k_A_S']]
    beta_C_S   = params[_PI['beta_C_S']]
    sigma_S    = params[_PI['sigma_S']]

    mu_step0   = params[_PI['mu_step0']]
    beta_B_st  = params[_PI['beta_B_st']]
    beta_F_st  = params[_PI['beta_F_st']]
    beta_A_st  = params[_PI['beta_A_st']]
    beta_C_st  = params[_PI['beta_C_st']]
    sigma_st   = params[_PI['sigma_st']]

    beta_S_VL  = params[_PI['beta_S_VL']]
    beta_F_VL  = params[_PI['beta_F_VL']]
    sigma_VL   = params[_PI['sigma_VL']]

    # --- Observation Jacobian ---
    # y_c = H_c @ [B, S, F, A] + bias_c
    H = jnp.array([
        [-kappa_B_HR, 0.0,       0.0,         alpha_A_HR], # HR
        [0.0,         0.0,       k_F,        -k_A_S],      # stress
        [beta_B_st,   0.0,      -beta_F_st,   beta_A_st],  # log_steps
        [0.0,         beta_S_VL, -beta_F_VL,  0.0],        # VL
    ])
    bias = jnp.array([
        HR_base  + beta_C_HR * C_k,
        S_base   + beta_C_S  * C_k,
        mu_step0 + beta_C_st * C_k,
        0.0,
    ])
    R_diag = jnp.array([sigma_HR ** 2, sigma_S ** 2, sigma_st ** 2, sigma_VL ** 2])

    obs_vals = jnp.array([
        grid_obs['hr_value'][k],
        grid_obs['stress_value'][k],
        grid_obs['log_steps_value'][k],
        grid_obs['vl_value'][k],
    ])
    obs_pres = jnp.array([
        grid_obs['hr_present'][k],
        grid_obs['stress_present'][k],
        grid_obs['steps_present'][k],
        grid_obs['vl_present'][k],
    ])

    # --- Kalman fusion ---
    def _kalman_step(carry, ch):
        mu, P, lp = carry
        h_i, b_i, r_i, y_i, pres_i = ch
        innov = y_i - (h_i @ mu + b_i)
        Ph    = P @ h_i
        S_i   = h_i @ Ph + r_i
        K_i   = Ph / S_i
        ll_i  = -0.5 * jnp.log(2.0 * jnp.pi * S_i) - 0.5 * innov ** 2 / S_i
        mu = mu + pres_i * K_i * innov
        P  = P  - pres_i * jnp.outer(K_i, Ph)
        lp = lp + pres_i * ll_i
        return (mu, P, lp), None

    (mu_fused, P_fused, log_pred_total), _ = jax.lax.scan(
        _kalman_step,
        (y_pred_det, P_prior, 0.0),
        (H, bias, R_diag, obs_vals, obs_pres),
    )

    P_safe = P_fused + 1e-10 * jnp.eye(4)
    L = jnp.linalg.cholesky(P_safe)
    x_new = mu_fused + L @ noise

    # Bounds
    B_new = jnp.clip(x_new[0], EPS_B_FROZEN, 1.0 - EPS_B_FROZEN)
    S_new = jnp.clip(x_new[1], EPS_S_FROZEN, 1.0 - EPS_S_FROZEN)
    F_new = jnp.maximum(x_new[2], 0.0)
    A_new = jnp.maximum(x_new[3], 0.0)
    y_new = jnp.array([B_new, S_new, F_new, A_new])

    # Weight correction
    preds_new = H @ y_new + bias
    resids_new = obs_vals - preds_new
    obs_ll_new = jnp.sum(obs_pres * (-0.5 * resids_new ** 2 / R_diag
                                      - 0.5 * jnp.log(R_diag) - HALF_LOG_2PI))
    pred_lw = log_pred_total - obs_ll_new

    return y_new, pred_lw


def diffusion_fn(params):
    del params
    return jnp.array([SIGMA_B_FROZEN, SIGMA_S_FROZEN, SIGMA_F_FROZEN, SIGMA_A_FROZEN])


# =========================================================================
# LIKELIHOOD
# =========================================================================

def _gaussian_obs_ll(y, grid_obs, k, params):
    B, S, F, A = y[0], y[1], y[2], y[3]
    C_k = grid_obs['C'][k]

    pred_HR  = params[_PI['HR_base']] - params[_PI['kappa_B_HR']] * B + params[_PI['alpha_A_HR']] * A + params[_PI['beta_C_HR']] * C_k
    pred_S   = params[_PI['S_base']] + params[_PI['k_F']] * F - params[_PI['k_A_S']] * A + params[_PI['beta_C_S']] * C_k
    pred_ST  = params[_PI['mu_step0']] + params[_PI['beta_B_st']] * B - params[_PI['beta_F_st']] * F + params[_PI['beta_A_st']] * A + params[_PI['beta_C_st']] * C_k
    pred_VL  = params[_PI['beta_S_VL']] * S - params[_PI['beta_F_VL']] * F

    def _ll(pred, obs_val, obs_pres, sigma):
        resid = obs_val - pred
        return obs_pres * (-0.5 * (resid / sigma) ** 2 - jnp.log(sigma) - HALF_LOG_2PI)

    lp  = _ll(pred_HR, grid_obs['hr_value'][k], grid_obs['hr_present'][k], params[_PI['sigma_HR']])
    lp += _ll(pred_S,  grid_obs['stress_value'][k], grid_obs['stress_present'][k], params[_PI['sigma_S']])
    lp += _ll(pred_ST, grid_obs['log_steps_value'][k], grid_obs['steps_present'][k], params[_PI['sigma_st']])
    lp += _ll(pred_VL, grid_obs['vl_value'][k], grid_obs['vl_present'][k], params[_PI['sigma_VL']])
    return lp


def obs_log_weight_fn(x_new, grid_obs, k, params):
    gauss_ll = _gaussian_obs_ll(x_new, grid_obs, k, params)
    C_k = grid_obs['C'][k]
    
    k_C     = params[_PI['k_C']]
    k_A     = params[_PI['k_A']]
    c_tilde = params[_PI['c_tilde']]
    z = k_C * C_k + k_A * x_new[3] - c_tilde
    p = jax.nn.sigmoid(z)
    p_safe = jnp.clip(p, 1e-8, 1.0 - 1e-8)
    s = grid_obs['sleep_label'][k].astype(p_safe.dtype)
    bern_ll = grid_obs['sleep_present'][k] * (s * jnp.log(p_safe) + (1.0 - s) * jnp.log(1.0 - p_safe))
    
    return gauss_ll + bern_ll


def obs_log_prob_fn(y, grid_obs, k, params):
    return obs_log_weight_fn(y, grid_obs, k, params)


# =========================================================================
# ALIGNMENT
# =========================================================================

def align_obs_fn(obs_data, t_steps, dt):
    T = t_steps
    def _get(name): return obs_data.get(name) if isinstance(obs_data, dict) else None

    # Standard channels
    hr_val, hr_pres = np.zeros(T, dtype=np.float32), np.zeros(T, dtype=np.float32)
    s_val, s_pres = np.zeros(T, dtype=np.float32), np.zeros(T, dtype=np.float32)
    log_st_val, st_pres = np.zeros(T, dtype=np.float32), np.zeros(T, dtype=np.float32)
    sl_label, sl_pres = np.zeros(T, dtype=np.int32), np.zeros(T, dtype=np.float32)
    
    for name, val, pres in [('obs_HR', hr_val, hr_pres), ('obs_stress', s_val, s_pres), 
                            ('obs_steps', log_st_val, st_pres), ('obs_sleep', sl_label, sl_pres)]:
        ch = _get(name)
        if ch and 't_idx' in ch:
            idx = np.asarray(ch['t_idx']).astype(int)
            mask = (idx >= 0) & (idx < T)
            if name == 'obs_steps':
                val[idx[mask]] = np.log(np.asarray(ch['obs_value'])[mask] + 1.0)
            elif name == 'obs_sleep':
                val[idx[mask]] = np.asarray(ch['sleep_label'])[mask]
            else:
                val[idx[mask]] = np.asarray(ch['obs_value'])[mask]
            pres[idx[mask]] = 1.0

    # VolumeLoad
    vl_val, vl_pres = np.zeros(T, dtype=np.float32), np.zeros(T, dtype=np.float32)
    vl_ch = _get('obs_volumeload')
    if vl_ch and 't_idx' in vl_ch:
        idx = np.asarray(vl_ch['t_idx']).astype(int)
        mask = (idx >= 0) & (idx < T)
        vl_val[idx[mask]] = np.asarray(vl_ch['obs_value'])[mask]
        vl_pres[idx[mask]] = 1.0

    # Phi (N, 2)
    p_ch = _get('Phi')
    Phi_val = np.zeros((T, 2), dtype=np.float32)
    if p_ch and 'Phi_value' in p_ch:
        raw = np.asarray(p_ch['Phi_value']).astype(np.float32)
        n = min(len(raw), T)
        Phi_val[:n] = raw[:n]

    C_val = np.zeros(T, dtype=np.float32)
    c_ch = _get('C')
    if c_ch and 'C_value' in c_ch:
        raw = np.asarray(c_ch['C_value']).astype(np.float32)
        n = min(len(raw), T)
        C_val[:n] = raw[:n]

    has_any = np.maximum.reduce([hr_pres, s_pres, st_pres, sl_pres, vl_pres])

    return {
        'hr_value': jnp.array(hr_val), 'hr_present': jnp.array(hr_pres),
        'stress_value': jnp.array(s_val), 'stress_present': jnp.array(s_pres),
        'log_steps_value': jnp.array(log_st_val), 'steps_present': jnp.array(st_pres),
        'sleep_label': jnp.array(sl_label), 'sleep_present': jnp.array(sl_pres),
        'vl_value': jnp.array(vl_val), 'vl_present': jnp.array(vl_pres),
        'Phi': jnp.array(Phi_val), 'C': jnp.array(C_val), 'has_any_obs': jnp.array(has_any),
    }


# =========================================================================
# FORWARD SDE
# =========================================================================

def forward_sde_stochastic(init_state, params, exogenous, dt, n_steps, rng_key=None):
    tau_B      = params[_PI['tau_B']]; kappa_B = params[_PI['kappa_B']]; epsilon_AB = params[_PI['epsilon_AB']]
    tau_S      = params[_PI['tau_S']]; kappa_S = params[_PI['kappa_S']]; epsilon_AS = params[_PI['epsilon_AS']]
    tau_F      = params[_PI['tau_F']]; kappa_FB = params[_PI['kappa_FB']]; kappa_FS = params[_PI['kappa_FS']]
    lambda_A   = params[_PI['lambda_A']]
    mu_0 = params[_PI['mu_0']]; mu_B = params[_PI['mu_B']]; mu_S = params[_PI['mu_S']]; mu_F = params[_PI['mu_F']]; mu_FF = params[_PI['mu_FF']]
    eta = params[_PI['eta']]
    
    sqrt_dt = jnp.sqrt(dt)
    Phi_arr = jnp.asarray(exogenous['Phi'])
    if rng_key is None: rng_key = jax.random.PRNGKey(0)

    def step(carry, i):
        y, key = carry
        key, nk = jax.random.split(key); noise = jax.random.normal(nk, (4,))
        B, S, F, A = y[0], y[1], y[2], y[3]
        Phi_B, Phi_S = Phi_arr[i][0], Phi_arr[i][1]
        
        F_dev = F - F_TYP
        mu = mu_0 + mu_B*B + mu_S*S - mu_F*F - mu_FF*F_dev*F_dev
        a_B = (1.0 + epsilon_AB*A) / (1.0 + epsilon_AB*A_TYP)
        a_S = (1.0 + epsilon_AS*A) / (1.0 + epsilon_AS*A_TYP)
        a_F = (1.0 + lambda_A*A) / (1.0 + lambda_A*A_TYP)
        
        dB = kappa_B * a_B * Phi_B - B / tau_B
        dS = kappa_S * a_S * Phi_S - S / tau_S
        dF = kappa_FB * Phi_B + kappa_FS * Phi_S - a_F / tau_F * F
        dA = mu * A - eta * A * A * A
        
        B_cl = jnp.clip(B, EPS_B_FROZEN, 1.0 - EPS_B_FROZEN)
        S_cl = jnp.clip(S, EPS_S_FROZEN, 1.0 - EPS_S_FROZEN)
        
        B_new = B + dt*dB + SIGMA_B_FROZEN*jnp.sqrt(B_cl*(1-B_cl))*sqrt_dt*noise[0]
        S_new = S + dt*dS + SIGMA_S_FROZEN*jnp.sqrt(S_cl*(1-S_cl))*sqrt_dt*noise[1]
        F_new = F + dt*dF + SIGMA_F_FROZEN*jnp.sqrt(jnp.maximum(F, 0.0))*sqrt_dt*noise[2]
        A_new = A + dt*dA + SIGMA_A_FROZEN*jnp.sqrt(jnp.maximum(A, 0.0) + EPS_A_FROZEN)*sqrt_dt*noise[3]
        
        y_next = jnp.array([jnp.clip(B_new, EPS_B_FROZEN, 1.0-EPS_B_FROZEN), 
                            jnp.clip(S_new, EPS_S_FROZEN, 1.0-EPS_S_FROZEN), 
                            jnp.maximum(F_new, 0.0), jnp.maximum(A_new, 0.0)])
        return (y_next, key), y_next

    (_, _), traj = jax.lax.scan(step, (init_state, rng_key), jnp.arange(n_steps))
    return traj


def imex_step_fn(y, t, dt, params, grid_obs):
    del t
    B, S, F, A = y[0], y[1], y[2], y[3]
    Phi_k = grid_obs.get('Phi_k', jnp.zeros(2))
    Phi_B, Phi_S = Phi_k[0], Phi_k[1]
    
    F_dev = F - F_TYP
    mu = params[_PI['mu_0']] + params[_PI['mu_B']]*B + params[_PI['mu_S']]*S - params[_PI['mu_F']]*F - params[_PI['mu_FF']]*F_dev*F_dev
    a_B = (1.0 + params[_PI['epsilon_AB']]*A) / (1.0 + params[_PI['epsilon_AB']]*A_TYP)
    a_S = (1.0 + params[_PI['epsilon_AS']]*A) / (1.0 + params[_PI['epsilon_AS']]*A_TYP)
    a_F = (1.0 + params[_PI['lambda_A']]*A) / (1.0 + params[_PI['lambda_A']]*A_TYP)
    
    dB = params[_PI['kappa_B']] * a_B * Phi_B - B / params[_PI['tau_B']]
    dS = params[_PI['kappa_S']] * a_S * Phi_S - S / params[_PI['tau_S']]
    dF = params[_PI['kappa_FB']] * Phi_B + params[_PI['kappa_FS']] * Phi_S - a_F / params[_PI['tau_F']] * F
    dA = mu * A - params[_PI['eta']] * A * A * A
    
    return jnp.array([B + dt*dB, S + dt*dS, F + dt*dF, A + dt*dA])


def shard_init_fn(time_offset, params, exogenous, global_init):
    del time_offset, params, exogenous
    return global_init


def make_init_state_fn(init_estimates, params):
    del params
    return init_estimates


def get_init_theta():
    all_config = OrderedDict()
    all_config.update(PARAM_PRIOR_CONFIG)
    all_config.update(INIT_STATE_PRIOR_CONFIG)
    def _m(pt, pa): return math.exp(pa[0] + pa[1]**2/2) if pt == 'lognormal' else pa[0]
    return np.array([_m(pt, pa) for _, (pt, pa) in all_config.items()], dtype=np.float32)


# =========================================================================
# ASSEMBLE
# =========================================================================

HIGH_RES_FSA_V3_ESTIMATION = EstimationModel(
    name="fsa_high_res_v3", version="3.0",
    n_states=4, n_stochastic=4, stochastic_indices=(0, 1, 2, 3),
    state_bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 10.0), (0.0, 5.0)),
    param_prior_config=PARAM_PRIOR_CONFIG,
    init_state_prior_config=INIT_STATE_PRIOR_CONFIG,
    frozen_params={
        'sigma_B': SIGMA_B_FROZEN, 'sigma_S': SIGMA_S_FROZEN, 
        'sigma_F': SIGMA_F_FROZEN, 'sigma_A': SIGMA_A_FROZEN, 'phi': PHI_FROZEN,
    },
    exogenous_keys=('Phi',),
    propagate_fn=propagate_fn,
    diffusion_fn=diffusion_fn,
    obs_log_weight_fn=obs_log_weight_fn,
    align_obs_fn=align_obs_fn,
    shard_init_fn=shard_init_fn,
    forward_sde_fn=forward_sde_stochastic,
    get_init_theta_fn=get_init_theta,
    imex_step_fn=imex_step_fn,
    obs_log_prob_fn=obs_log_prob_fn,
    make_init_state_fn=make_init_state_fn,
)
