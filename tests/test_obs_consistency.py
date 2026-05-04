"""Sim ↔ Estimator observation-channel consistency for FSA-v2.

Mirrors the SWAT_model_dev `test_obs_consistency.py` shape, adapted for
FSA's 3-state `[B, F, A]` system and 4 channels (HR / sleep / stress /
steps). The shape of the test is:

    For each obs channel:
      1. Compute the LaTeX-prescribed channel mean (or, for sleep, the
         Bernoulli probability) directly in the test from DEFAULT_PARAMS,
         using the formula in `LaTex_docs/main.tex`.
      2. Run the simulator's `gen_obs_*` with sigma → 0 — the returned
         observation must equal the LaTeX mean (within float tolerance).
      3. Run the estimator's `obs_log_weight_fn` with `obs = LaTeX_mean`
         and only that channel's `*_present` mask set to 1 — the
         resulting log-weight must equal the Gaussian-peak value
         `-log(sigma·sqrt(2π))` (or, for the Bernoulli sleep channel,
         `log p` where `p` is the LaTeX probability).

If either side drifts away from the LaTeX, exactly one of (2) or (3)
fails, pinpointing which side broke.

Why this matters
----------------
FSA's `_plant.py` (the simulator-as-plant for the closed-loop MPC
bench in `version_2/tools/bench_smc_*fsa*`) consumes the four
`gen_obs_*` samplers. The estimator (`HIGH_RES_FSA_V2_ESTIMATION`)
consumes `obs_log_weight_fn`. If the two sides drift apart, every
SMC² posterior is wrong, silently. The SWAT D1/D2 incident showed
this exact failure mode (sim missing `delta_HR` / `delta_s` while
the estimator included them).
"""
import math
import os
import sys
from pathlib import Path

# Force CPU + X64 BEFORE importing JAX so float comparisons are tight.
os.environ['JAX_ENABLE_X64'] = 'True'
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Anchor sys.path so the test works whether or not the user pip-installed.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.fsa_high_res.simulation import (
    DEFAULT_PARAMS,
    circadian,
    gen_obs_hr,
    gen_obs_sleep,
    gen_obs_steps,
    gen_obs_stress,
)
from models.fsa_high_res.estimation import (
    _PI,
    obs_log_weight_fn,
)


# ─────────────────────────────────────────────────────────────────────
# Test fixtures — a single non-trivial scenario
# ─────────────────────────────────────────────────────────────────────
#
# State picked so:
#  - (B, F, A) all non-zero so coupling coefficients matter.
#  - Single-bin t_grid simplifies the channel-mean comparison.
#  - C(t) has a known sign so the beta_C_* terms are exercised.

B_state = 0.65
F_state = 0.20
A_state = 0.40
T_DAYS = 0.20  # mid-morning — circadian C ≈ cos(2π·0.20) ≈ 0.31


def _build_params() -> dict:
    """Truth params dict (DEFAULT_PARAMS unchanged)."""
    return dict(DEFAULT_PARAMS)


def _build_estimator_params_vec(p_dict: dict) -> jnp.ndarray:
    """Pack a flat params vector indexed by `_PI` for `obs_log_weight_fn`."""
    return jnp.array([p_dict[name] for name in _PI], dtype=jnp.float64)


def _trajectory() -> np.ndarray:
    """Single-bin (B, F, A) trajectory at the test fixture state."""
    return np.array([[B_state, F_state, A_state]], dtype=np.float64)


def _t_grid() -> np.ndarray:
    return np.array([T_DAYS], dtype=np.float64)


def _C_at_t() -> float:
    """Circadian value at the test bin, using the same formula the estimator
    expects in `grid_obs['C'][k]`."""
    return float(circadian(_t_grid(), phi=DEFAULT_PARAMS.get('phi', 0.0))[0])


def _est_log_lik(channel: str, obs_value, x_state, p_dict, sleep_label=0) -> float:
    """Single-bin estimator log-weight, with only the named channel present.

    Builds a one-bin grid_obs that flips on exactly one channel's
    presence mask so the returned log-weight is the channel's likelihood
    in isolation.
    """
    C_k = _C_at_t()
    grid_obs = {
        'C':                jnp.array([C_k]),
        'Phi':              jnp.array([0.0]),
        'hr_value':         jnp.array([0.0]),  'hr_present':     jnp.array([0.0]),
        'stress_value':     jnp.array([0.0]),  'stress_present': jnp.array([0.0]),
        'log_steps_value':  jnp.array([0.0]),  'steps_present':  jnp.array([0.0]),
        'sleep_label':      jnp.array([int(sleep_label)]),
        'sleep_present':    jnp.array([0.0]),
    }
    if channel == 'hr':
        grid_obs['hr_value'] = jnp.array([float(obs_value)])
        grid_obs['hr_present'] = jnp.array([1.0])
    elif channel == 'stress':
        grid_obs['stress_value'] = jnp.array([float(obs_value)])
        grid_obs['stress_present'] = jnp.array([1.0])
    elif channel == 'steps':
        grid_obs['log_steps_value'] = jnp.array([float(obs_value)])
        grid_obs['steps_present'] = jnp.array([1.0])
    elif channel == 'sleep':
        grid_obs['sleep_label'] = jnp.array([int(obs_value)])
        grid_obs['sleep_present'] = jnp.array([1.0])
    else:
        raise ValueError(f"unknown channel {channel!r}")

    p_vec = _build_estimator_params_vec(p_dict)
    x_jax = jnp.asarray(x_state, dtype=jnp.float64)
    log_w = obs_log_weight_fn(x_jax, grid_obs, k=0, params=p_vec)
    return float(log_w)


def _max_gauss_log_density(sigma: float) -> float:
    """log N(x | x, sigma^2) — the value of the log-density at its peak.
    Equals -log(sigma·sqrt(2π))."""
    return -math.log(sigma * math.sqrt(2.0 * math.pi))


# ─────────────────────────────────────────────────────────────────────
# Channel tests
# ─────────────────────────────────────────────────────────────────────


def test_hr_channel_consistency():
    """HR channel: sim and estimator both produce the LaTeX mean.

    LaTeX (main.tex §inference): HR ~ N(HR_base - kappa_B_HR·B
    + alpha_A_HR·A + beta_C_HR·C, sigma_HR²); sleep-gated.
    """
    p = _build_params()
    state = np.array([B_state, F_state, A_state], dtype=np.float64)
    C_k = _C_at_t()

    # 1. LaTeX mean
    expected_mean = (p['HR_base']
                     - p['kappa_B_HR'] * B_state
                     + p['alpha_A_HR'] * A_state
                     + p['beta_C_HR'] * C_k)

    # 2. Simulator side — set sigma_HR ~ 0, and force the bin to be
    #    "during sleep" so the gated channel emits a sample.
    p_no_noise = dict(p, sigma_HR=0.0)
    prior = {'obs_sleep': {'sleep_label': np.array([1], dtype=np.int32)}}
    out = gen_obs_hr(_trajectory(), _t_grid(), p_no_noise,
                      aux=None, prior_channels=prior, seed=0)
    assert len(out['t_idx']) == 1, "sleep-gated HR should emit one sample at the asleep bin"
    sim_mean = float(out['obs_value'][0])
    assert math.isclose(sim_mean, expected_mean, abs_tol=1e-5), (
        f"sim HR mean {sim_mean} != LaTeX mean {expected_mean}; "
        f"check `gen_obs_hr` formula in models/fsa_high_res/simulation.py "
        f"against LaTeX (HR_base - kappa_B_HR·B + alpha_A_HR·A + beta_C_HR·C)")

    # 3. Estimator side — log-likelihood at obs = expected_mean must
    #    hit the Gaussian peak -log(sigma·sqrt(2π)).
    est_lp = _est_log_lik('hr', expected_mean, state, p)
    expected_peak = _max_gauss_log_density(p['sigma_HR'])
    assert math.isclose(est_lp, expected_peak, abs_tol=1e-8), (
        f"estimator HR mean does not match LaTeX mean: log-lik at obs={expected_mean:.4f} "
        f"is {est_lp:.6f}, peak should be {expected_peak:.6f}")


def test_stress_channel_consistency():
    """Stress channel: sim and estimator both produce the LaTeX mean.

    LaTeX: S ~ N(S_base + k_F·F - k_A_S·A + beta_C_S·C, sigma_S²);
    wake-gated.
    """
    p = _build_params()
    state = np.array([B_state, F_state, A_state], dtype=np.float64)
    C_k = _C_at_t()

    # 1. LaTeX mean
    expected_mean = (p['S_base']
                     + p['k_F'] * F_state
                     - p['k_A_S'] * A_state
                     + p['beta_C_S'] * C_k)

    # 2. Simulator side — sigma_S=0, force "awake" (sleep_label=0)
    #    so the wake-gated channel emits.
    p_no_noise = dict(p, sigma_S=0.0)
    prior = {'obs_sleep': {'sleep_label': np.array([0], dtype=np.int32)}}
    out = gen_obs_stress(_trajectory(), _t_grid(), p_no_noise,
                          aux=None, prior_channels=prior, seed=0)
    assert len(out['t_idx']) == 1, "wake-gated stress should emit at the awake bin"
    sim_mean = float(out['obs_value'][0])
    assert math.isclose(sim_mean, expected_mean, abs_tol=1e-5), (
        f"sim stress mean {sim_mean} != LaTeX mean {expected_mean}; "
        f"check `gen_obs_stress` formula against LaTeX")

    # 3. Estimator side
    est_lp = _est_log_lik('stress', expected_mean, state, p)
    expected_peak = _max_gauss_log_density(p['sigma_S'])
    assert math.isclose(est_lp, expected_peak, abs_tol=1e-8), (
        f"estimator stress mean does not match LaTeX mean: log-lik {est_lp:.6f} "
        f"!= peak {expected_peak:.6f}")


def test_steps_channel_consistency():
    """Steps channel: sim and estimator both produce the LaTeX log-mean.

    LaTeX: log(steps + 1) ~ N(mu_step0 + beta_B_st·B - beta_F_st·F
    + beta_A_st·A + beta_C_st·C, sigma_st²); wake-gated.

    The simulator returns RAW step counts (`exp(log_obs) - 1`); the
    estimator's `align_obs_fn` log-transforms them back. This test
    works directly in the log domain.
    """
    p = _build_params()
    state = np.array([B_state, F_state, A_state], dtype=np.float64)
    C_k = _C_at_t()

    # 1. LaTeX log-mean
    expected_log_mean = (p['mu_step0']
                         + p['beta_B_st'] * B_state
                         - p['beta_F_st'] * F_state
                         + p['beta_A_st'] * A_state
                         + p['beta_C_st'] * C_k)

    # 2. Simulator side — sigma_st=0, force awake.
    p_no_noise = dict(p, sigma_st=0.0)
    prior = {'obs_sleep': {'sleep_label': np.array([0], dtype=np.int32)}}
    out = gen_obs_steps(_trajectory(), _t_grid(), p_no_noise,
                         aux=None, prior_channels=prior, seed=0)
    assert len(out['t_idx']) == 1, "wake-gated steps should emit at the awake bin"
    # Sim returns raw counts (exp(log_obs) - 1). Reverse the transform.
    raw_steps = float(out['obs_value'][0])
    sim_log_mean = math.log(raw_steps + 1.0)
    assert math.isclose(sim_log_mean, expected_log_mean, abs_tol=1e-5), (
        f"sim steps log-mean {sim_log_mean} != LaTeX log-mean {expected_log_mean}; "
        f"check `gen_obs_steps` formula against LaTeX")

    # 3. Estimator side — obs is in log-space already (estimator only
    #    sees the log_steps_value field after alignment).
    est_lp = _est_log_lik('steps', expected_log_mean, state, p)
    expected_peak = _max_gauss_log_density(p['sigma_st'])
    assert math.isclose(est_lp, expected_peak, abs_tol=1e-8), (
        f"estimator steps log-mean does not match LaTeX: log-lik {est_lp:.6f} "
        f"!= peak {expected_peak:.6f}")


def test_sleep_channel_consistency():
    """Sleep channel: sim and estimator both use the LaTeX Bernoulli prob.

    LaTeX: sleep_label ~ Bernoulli(sigmoid(k_C·C + k_A·A - c_tilde)).

    Test both sides via the marginal probability:
      - estimator: log P(label=1 | A, C) must equal log(p_LaTeX);
                   log P(label=0 | A, C) must equal log(1 - p_LaTeX).
      - simulator: empirical frequency of label=1 over many bins
                   must converge to p_LaTeX.
    """
    p = _build_params()
    state = np.array([B_state, F_state, A_state], dtype=np.float64)
    C_k = _C_at_t()

    # 1. LaTeX probability of label=1 (sleep)
    z = p['k_C'] * C_k + p['k_A'] * A_state - p['c_tilde']
    p_sleep_LaTeX = 1.0 / (1.0 + math.exp(-z))

    # 2. Estimator side — for each of label=1 and label=0, the log-prob
    #    should equal log(p) and log(1-p).
    lp_label1 = _est_log_lik('sleep', 1, state, p)
    lp_label0 = _est_log_lik('sleep', 0, state, p)
    assert math.isclose(lp_label1, math.log(p_sleep_LaTeX), abs_tol=1e-10), (
        f"estimator P(label=1) disagrees with LaTeX: log-prob {lp_label1:.6f} "
        f"vs expected {math.log(p_sleep_LaTeX):.6f}")
    assert math.isclose(lp_label0, math.log(1.0 - p_sleep_LaTeX), abs_tol=1e-10), (
        f"estimator P(label=0) disagrees with LaTeX: log-prob {lp_label0:.6f} "
        f"vs expected {math.log(1.0 - p_sleep_LaTeX):.6f}")

    # 3. Simulator side — empirical frequency of label=1 over many
    #    independent bins (all at the same state + C(t)) must converge
    #    to p_LaTeX.
    n_bins = 30000
    trajectory = np.tile(_trajectory(), (n_bins, 1))
    t_grid = np.full(n_bins, T_DAYS, dtype=np.float64)
    out = gen_obs_sleep(trajectory, t_grid, p,
                          aux=None, prior_channels=None, seed=12345)
    p_emp = float(out['sleep_label'].mean())
    # Bernoulli 3σ tolerance: 3·sqrt(p(1-p)/n) at p~0.7, n=30 000 ≈ 0.008.
    assert abs(p_emp - p_sleep_LaTeX) < 0.012, (
        f"sim sleep marginal disagrees with LaTeX: empirical {p_emp:.4f} "
        f"vs expected {p_sleep_LaTeX:.4f}")


# ─────────────────────────────────────────────────────────────────────
# Belt-and-braces tests
# ─────────────────────────────────────────────────────────────────────


def test_hr_offset_actually_shifts_the_signal():
    """Sweep `HR_base` and check the simulated HR mean shifts 1:1.

    This is the FSA analogue of the SWAT D1 regression catcher — if any
    future change inadvertently drops `HR_base` from `gen_obs_hr`, the
    sim will produce a constant mean regardless of the truth value.
    """
    p_zero = dict(DEFAULT_PARAMS)
    p_zero['HR_base'] = 0.0    # baseline: no contribution from HR_base
    prior = {'obs_sleep': {'sleep_label': np.array([1], dtype=np.int32)}}

    for shift in (-10.0, -3.0, 5.0, 17.0):
        p_shifted = dict(p_zero, HR_base=shift, sigma_HR=0.0)
        out_shifted = gen_obs_hr(_trajectory(), _t_grid(), p_shifted,
                                   aux=None, prior_channels=prior, seed=0)
        out_baseline = gen_obs_hr(_trajectory(), _t_grid(),
                                    dict(p_zero, sigma_HR=0.0),
                                    aux=None, prior_channels=prior, seed=0)
        assert len(out_shifted['t_idx']) == 1
        delta = float(out_shifted['obs_value'][0]) - float(out_baseline['obs_value'][0])
        assert math.isclose(delta, shift, abs_tol=1e-5), (
            f"HR_base={shift} should shift HR mean by exactly {shift}; "
            f"got shift {delta} (regression of an FSA equivalent of SWAT bug D1)")


if __name__ == "__main__":
    # Allow running with `python tests/test_obs_consistency.py`.
    test_hr_channel_consistency()
    test_stress_channel_consistency()
    test_steps_channel_consistency()
    test_sleep_channel_consistency()
    test_hr_offset_actually_shifts_the_signal()
    print("All sim/est observation-channel consistency tests PASSED.")
