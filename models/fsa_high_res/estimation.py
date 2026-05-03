"""version_4/models/fsa_high_res/estimation.py — FSA-v4 EstimationModel.

Variable Dose Extension:
  - 6D state space: [B, S, F, A, KFB, KFS]
  - 2D control input: [Phi_B, Phi_S]
  - Dynamic fatigue sensitivity (Busso 2003)
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
SIGMA_K_FROZEN = 0.005
PHI_FROZEN     = 0.0


# =========================================================================
# PRIORS — FSA-v4
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
    ('lambda_A',    ('lognormal', (math.log(1.00), 0.05))),

    ('KFB_0',       ('lognormal', (math.log(0.030), 0.20))),
    ('KFS_0',       ('lognormal', (math.log(0.050), 0.20))),
    ('tau_K',       ('lognormal', (math.log(21.0), 0.10))),
    ('mu_K',        ('lognormal', (math.log(0.005), 0.20))),

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
COLD_START_INIT = jnp.array([0.05, 0.10, 0.30, 0.10, 0.030, 0.050])

_PK = list(PARAM_PRIOR_CONFIG.keys())
_PI = {k: i for i, k in enumerate(_PK)}


# =========================================================================
# PROPAGATE_FN — 6D [B, S, F, A, KFB, KFS]
# =========================================================================

def propagate_fn(y, t, dt, params, grid_obs, k,
                 sigma_diag, noise, rng_key):
    del t, rng_key, sigma_diag

    # --- Dynamics params ---
    p = {name: params[_PI[name]] for name in _PK if name in _PI}

    B, S, F, A, KFB, KFS = y[0], y[1], y[2], y[3], y[4], y[5]
    Phi_k = grid_obs['Phi'][k]
    Phi_B, Phi_S = Phi_k[0], Phi_k[1]
    C_k   = grid_obs['C'][k]

    # --- Drift ---
    F_dev = F - F_TYP
    mu_bif = p['mu_0'] + p['mu_B']*B + p['mu_S']*S - p['mu_F']*F - p['mu_FF']*F_dev**2
    a_B = (1.0 + p['epsilon_AB']*A) / (1.0 + p['epsilon_AB']*A_TYP)
    a_S = (1.0 + p['epsilon_AS']*A) / (1.0 + p['epsilon_AS']*A_TYP)
    a_F = (1.0 + p['lambda_A']*A) / (1.0 + p['lambda_A']*A_TYP)
    
    dB = p['kappa_B']*a_B*Phi_B - B/p['tau_B']
    dS = p['kappa_S']*a_S*Phi_S - S/p['tau_S']
    dF = KFB*Phi_B + KFS*Phi_S - a_F/p['tau_F']*F
    dA = mu_bif*A - p['eta']*A**3
    dKFB = (p['KFB_0'] - KFB)/p['tau_K'] + p['mu_K']*Phi_B
    dKFS = (p['KFS_0'] - KFS)/p['tau_K'] + p['mu_K']*Phi_S
    
    y_pred_det = y + dt * jnp.array([dB, dS, dF, dA, dKFB, dKFS])

    # --- Covariance ---
    B_cl = jnp.clip(B, EPS_B_FROZEN, 1.0 - EPS_B_FROZEN)
    S_cl = jnp.clip(S, EPS_S_FROZEN, 1.0 - EPS_S_FROZEN)
    vars = jnp.array([
        SIGMA_B_FROZEN**2 * B_cl*(1-B_cl)*dt,
        SIGMA_S_FROZEN**2 * S_cl*(1-S_cl)*dt,
        SIGMA_F_FROZEN**2 * jnp.maximum(F,0)*dt,
        SIGMA_A_FROZEN**2 * (jnp.maximum(A,0)+EPS_A_FROZEN)*dt,
        SIGMA_K_FROZEN**2 * jnp.maximum(KFB,0)*dt,
        SIGMA_K_FROZEN**2 * jnp.maximum(KFS,0)*dt
    ])
    P_prior = jnp.diag(jnp.maximum(vars, 1e-12))

    # --- Obs Jacobian (6D state, but obs only see 4D) ---
    H = jnp.zeros((4, 6))
    H = H.at[0, 0].set(-p['kappa_B_HR'])
    H = H.at[0, 3].set(p['alpha_A_HR'])
    H = H.at[1, 2].set(p['k_F'])
    H = H.at[1, 3].set(-p['k_A_S'])
    H = H.at[2, 0].set(p['beta_B_st'])
    H = H.at[2, 2].set(-p['beta_F_st'])
    H = H.at[2, 3].set(p['beta_A_st'])
    H = H.at[3, 1].set(p['beta_S_VL'])
    H = H.at[3, 2].set(-p['beta_F_VL'])

    bias = jnp.array([
        p['HR_base'] + p['beta_C_HR']*C_k,
        p['S_base'] + p['beta_C_S']*C_k,
        p['mu_step0'] + p['beta_C_st']*C_k,
        0.0
    ])
    R_diag = jnp.array([p['sigma_HR']**2, p['sigma_S']**2, p['sigma_st']**2, p['sigma_VL']**2])

    obs_vals = jnp.array([grid_obs['hr_value'][k], grid_obs['stress_value'][k], grid_obs['log_steps_value'][k], grid_obs['vl_value'][k]])
    obs_pres = jnp.array([grid_obs['hr_present'][k], grid_obs['stress_present'][k], grid_obs['steps_present'][k], grid_obs['vl_present'][k]])

    # --- Kalman Fusion ---
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

    (mu_fused, P_fused, log_pred_total), _ = jax.lax.scan(_kalman_step, (y_pred_det, P_prior, 0.0), (H, bias, R_diag, obs_vals, obs_pres))

    P_safe = P_fused + 1e-10 * jnp.eye(6)
    L = jnp.linalg.cholesky(P_safe)
    x_new = mu_fused + L @ noise

    # Bounds
    B_n = jnp.clip(x_new[0], EPS_B_FROZEN, 1.0-EPS_B_FROZEN)
    S_n = jnp.clip(x_new[1], EPS_S_FROZEN, 1.0-EPS_S_FROZEN)
    F_n = jnp.maximum(x_new[2], 0.0)
    A_n = jnp.maximum(x_new[3], 0.0)
    KFB_n = jnp.maximum(x_new[4], 0.0)
    KFS_n = jnp.maximum(x_new[5], 0.0)
    y_new = jnp.array([B_n, S_n, F_n, A_n, KFB_n, KFS_n])

    # Weight correction
    preds_new = H @ y_new + bias
    resids_new = obs_vals - preds_new
    obs_ll_new = jnp.sum(obs_pres * (-0.5*resids_new**2/R_diag - 0.5*jnp.log(R_diag) - HALF_LOG_2PI))
    pred_lw = log_pred_total - obs_ll_new

    return y_new, pred_lw


def diffusion_fn(params):
    del params
    return jnp.array([SIGMA_B_FROZEN, SIGMA_S_FROZEN, SIGMA_F_FROZEN, SIGMA_A_FROZEN, SIGMA_K_FROZEN, SIGMA_K_FROZEN])


# =========================================================================
# LIKELIHOOD & ALIGNMENT — update for 6D
# =========================================================================

def _gaussian_obs_ll(y, grid_obs, k, params):
    B, S, F, A = y[0], y[1], y[2], y[3]
    C_k = grid_obs['C'][k]
    p = {name: params[_PI[name]] for name in _PK if name in _PI}

    pHR = p['HR_base'] - p['kappa_B_HR']*B + p['alpha_A_HR']*A + p['beta_C_HR']*C_k
    pS  = p['S_base'] + p['k_F']*F - p['k_A_S']*A + p['beta_C_S']*C_k
    pST = p['mu_step0'] + p['beta_B_st']*B - p['beta_F_st']*F + p['beta_A_st']*A + p['beta_C_st']*C_k
    pVL = p['beta_S_VL']*S - p['beta_F_VL']*F

    def _ll(pred, obs_val, obs_pres, sigma):
        resid = obs_val - pred
        return obs_pres * (-0.5*(resid/sigma)**2 - jnp.log(sigma) - HALF_LOG_2PI)

    lp =  _ll(pHR, grid_obs['hr_value'][k], grid_obs['hr_present'][k], p['sigma_HR'])
    lp += _ll(pS,  grid_obs['stress_value'][k], grid_obs['stress_present'][k], p['sigma_S'])
    lp += _ll(pST, grid_obs['log_steps_value'][k], grid_obs['steps_present'][k], p['sigma_st'])
    lp += _ll(pVL, grid_obs['vl_value'][k], grid_obs['vl_present'][k], p['sigma_VL'])
    return lp

def obs_log_weight_fn(x_new, grid_obs, k, params):
    gauss_ll = _gaussian_obs_ll(x_new, grid_obs, k, params)
    C_k = grid_obs['C'][k]
    p = {name: params[_PI[name]] for name in _PK if name in _PI}
    z = p['k_C']*C_k + p['k_A']*x_new[3] - p['c_tilde']
    prob = jax.nn.sigmoid(z)
    prob_safe = jnp.clip(prob, 1e-8, 1.0-1e-8)
    s = grid_obs['sleep_label'][k].astype(prob_safe.dtype)
    bern_ll = grid_obs['sleep_present'][k] * (s*jnp.log(prob_safe) + (1.0-s)*jnp.log(1.0-prob_safe))
    return gauss_ll + bern_ll

def obs_log_prob_fn(y, grid_obs, k, params):
    return obs_log_weight_fn(y, grid_obs, k, params)

def align_obs_fn(obs_data, t_steps, dt):
    # Same as v3 but returned dict must be consistent with propagate_fn logic
    from models.fsa_high_res.simulation import BINS_PER_DAY
    T = t_steps
    def _get(name): return obs_data.get(name) if isinstance(obs_data, dict) else None

    hr_val, hr_pres = np.zeros(T, dtype=np.float32), np.zeros(T, dtype=np.float32)
    s_val, s_pres = np.zeros(T, dtype=np.float32), np.zeros(T, dtype=np.float32)
    log_st_val, st_pres = np.zeros(T, dtype=np.float32), np.zeros(T, dtype=np.float32)
    sl_label, sl_pres = np.zeros(T, dtype=np.int32), np.zeros(T, dtype=np.float32)
    vl_val, vl_pres = np.zeros(T, dtype=np.float32), np.zeros(T, dtype=np.float32)

    for name, val, pres in [('obs_HR', hr_val, hr_pres), ('obs_stress', s_val, s_pres), 
                            ('obs_steps', log_st_val, st_pres), ('obs_sleep', sl_label, sl_pres),
                            ('obs_volumeload', vl_val, vl_pres)]:
        ch = _get(name)
        if ch and 't_idx' in ch:
            idx = np.asarray(ch['t_idx']).astype(int)
            mask = (idx >= 0) & (idx < T)
            if name == 'obs_steps': val[idx[mask]] = np.log(np.asarray(ch['obs_value'])[mask] + 1.0)
            elif name == 'obs_sleep': val[idx[mask]] = np.asarray(ch['sleep_label'])[mask]
            else: val[idx[mask]] = np.asarray(ch['obs_value'])[mask]
            pres[idx[mask]] = 1.0

    p_ch = _get('Phi'); Phi_val = np.zeros((T, 2), dtype=np.float32)
    if p_ch and 'Phi_value' in p_ch: Phi_val[:min(len(p_ch['Phi_value']), T)] = p_ch['Phi_value'][:T]

    c_ch = _get('C'); C_val = np.zeros(T, dtype=np.float32)
    if c_ch and 'C_value' in c_ch: C_val[:min(len(c_ch['C_value']), T)] = c_ch['C_value'][:T]

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
# FORWARD SDE & ASSEMBLE
# =========================================================================

def forward_sde_stochastic(init_state, params, exogenous, dt, n_steps, rng_key=None):
    p = {name: params[_PI[name]] for name in _PK if name in _PI}
    sqrt_dt = jnp.sqrt(dt)
    Phi_arr = jnp.asarray(exogenous['Phi'])
    if rng_key is None: rng_key = jax.random.PRNGKey(0)

    def step(carry, i):
        y, key = carry
        key, nk = jax.random.split(key); noise = jax.random.normal(nk, (6,))
        B, S, F, A, KFB, KFS = y[0], y[1], y[2], y[3], y[4], y[5]
        Phi_B, Phi_S = Phi_arr[i][0], Phi_arr[i][1]
        
        F_dev = F - F_TYP
        mu = p['mu_0'] + p['mu_B']*B + p['mu_S']*S - p['mu_F']*F - p['mu_FF']*F_dev**2
        a_B = (1.0 + p['epsilon_AB']*A) / (1.0 + p['epsilon_AB']*A_TYP)
        a_S = (1.0 + p['epsilon_AS']*A) / (1.0 + p['epsilon_AS']*A_TYP)
        a_F = (1.0 + p['lambda_A']*A) / (1.0 + p['lambda_A']*A_TYP)
        
        y_next = y + dt * jnp.array([p['kappa_B']*a_B*Phi_B - B/p['tau_B'],
                                     p['kappa_S']*a_S*Phi_S - S/p['tau_S'],
                                     KFB*Phi_B + KFS*Phi_S - a_F/p['tau_F']*F,
                                     mu*A - p['eta']*A**3,
                                     (p['KFB_0'] - KFB)/p['tau_K'] + p['mu_K']*Phi_B,
                                     (p['KFS_0'] - KFS)/p['tau_K'] + p['mu_K']*Phi_S])
        
        # State-dependent noise
        y_next += jnp.array([SIGMA_B_FROZEN*jnp.sqrt(jnp.clip(B,1e-4,0.999)*(1-B))*sqrt_dt*noise[0],
                             SIGMA_S_FROZEN*jnp.sqrt(jnp.clip(S,1e-4,0.999)*(1-S))*sqrt_dt*noise[1],
                             SIGMA_F_FROZEN*jnp.sqrt(jnp.maximum(F,0))*sqrt_dt*noise[2],
                             SIGMA_A_FROZEN*jnp.sqrt(jnp.maximum(A,0)+1e-4)*sqrt_dt*noise[3],
                             SIGMA_K_FROZEN*jnp.sqrt(jnp.maximum(KFB,0))*sqrt_dt*noise[4],
                             SIGMA_K_FROZEN*jnp.sqrt(jnp.maximum(KFS,0))*sqrt_dt*noise[5]])
        
        y_next = jnp.array([jnp.clip(y_next[0],1e-4,0.999), jnp.clip(y_next[1],1e-4,0.999), 
                            jnp.maximum(y_next[2],0), jnp.maximum(y_next[3],0),
                            jnp.maximum(y_next[4],0), jnp.maximum(y_next[5],0)])
        return (y_next, key), y_next

    (_, _), traj = jax.lax.scan(step, (init_state, rng_key), jnp.arange(n_steps))
    return traj


def imex_step_fn(y, t, dt, params, grid_obs):
    del t
    B, S, F, A, KFB, KFS = y[0], y[1], y[2], y[3], y[4], y[5]
    Phi_k = grid_obs.get('Phi_k', jnp.zeros(2))
    p = {name: params[_PI[name]] for name in _PK if name in _PI}
    
    F_dev = F - F_TYP
    mu = p['mu_0'] + p['mu_B']*B + p['mu_S']*S - p['mu_F']*F - p['mu_FF']*F_dev**2
    a_B = (1.0 + p['epsilon_AB']*A) / (1.0 + p['epsilon_AB']*A_TYP)
    a_S = (1.0 + p['epsilon_AS']*A) / (1.0 + p['epsilon_AS']*A_TYP)
    a_F = (1.0 + p['lambda_A']*A) / (1.0 + p['lambda_A']*A_TYP)
    
    y_next = y + dt * jnp.array([p['kappa_B']*a_B*Phi_k[0] - B/p['tau_B'],
                                 p['kappa_S']*a_S*Phi_k[1] - S/p['tau_S'],
                                 KFB*Phi_k[0] + KFS*Phi_k[1] - a_F/p['tau_F']*F,
                                 mu*A - p['eta']*A**3,
                                 (p['KFB_0'] - KFB)/p['tau_K'] + p['mu_K']*Phi_k[0],
                                 (p['KFS_0'] - KFS)/p['tau_K'] + p['mu_K']*Phi_k[1]])
    return y_next


def shard_init_fn(time_offset, params, exogenous, global_init):
    del time_offset, params, exogenous
    return global_init

def make_init_state_fn(init_estimates, params):
    del params
    return init_estimates

def get_init_theta():
    def _m(pt, pa): return math.exp(pa[0] + pa[1]**2/2) if pt == 'lognormal' else pa[0]
    return np.array([_m(pt, pa) for _, (pt, pa) in PARAM_PRIOR_CONFIG.items()], dtype=np.float32)


HIGH_RES_FSA_V4_ESTIMATION = EstimationModel(
    name="fsa_high_res_v4", version="4.0",
    n_states=6, n_stochastic=6, stochastic_indices=(0, 1, 2, 3, 4, 5),
    state_bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 10.0), (0.0, 5.0), (0.0, 1.0), (0.0, 1.0)),
    param_prior_config=PARAM_PRIOR_CONFIG,
    frozen_params={'sigma_B': SIGMA_B_FROZEN, 'sigma_S': SIGMA_S_FROZEN, 'sigma_F': SIGMA_F_FROZEN, 
                   'sigma_A': SIGMA_A_FROZEN, 'sigma_K': SIGMA_K_FROZEN, 'phi': PHI_FROZEN},
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
