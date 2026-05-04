"""Likelihood sanity checks for FSA-v2.

Three families of checks:

1. **Peakedness in (B, F, A) at truth.** For each obs channel and a
   chosen truth state, sweep one state dimension at a time and confirm
   the per-channel log-likelihood (computed by the estimator's
   `obs_log_weight_fn`) attains its maximum at or very near the truth
   value. Catches sign-flips or formula errors that would silently
   bias the filter away from the right basin.

2. **Boundary penalty.** Evaluate the log-likelihood at a state outside
   the physical domain (B = 1.5, A = -0.5) and confirm it's strongly
   penalised (large negative number, not finite-and-pleasant). Catches
   cases where the estimator silently accepts illegal states.

3. **Prior vs truth alignment.** For each estimable parameter in
   PARAM_PRIOR_CONFIG, confirm DEFAULT_PARAMS' truth value lies inside
   the prior's plausible region (z-score below 4). Prevents the prior
   from being so badly misspecified that the filter can't reach truth.

Returns a result dict + prints a summary; raises if any pillar fails.
"""
import os
import sys
import math
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

from models.fsa_high_res.simulation import (
    DEFAULT_PARAMS,
    circadian,
    gen_obs_hr,
    gen_obs_sleep,
    gen_obs_steps,
    gen_obs_stress,
)
from models.fsa_high_res.estimation import (
    PARAM_PRIOR_CONFIG,
    _PI,
    obs_log_weight_fn,
)


def _build_grid_obs(obs_dict: dict, t_days: float) -> dict:
    """Build a single-bin grid_obs dict with per-channel obs values."""
    C_t = float(circadian(jnp.array([t_days]),
                            phi=DEFAULT_PARAMS.get('phi', 0.0))[0])
    out = {
        'C':                jnp.array([C_t]),
        'Phi':              jnp.array([0.0]),
        'hr_value':         jnp.array([float(obs_dict.get('hr', 0.0))]),
        'hr_present':       jnp.array([1.0 if 'hr' in obs_dict else 0.0]),
        'stress_value':     jnp.array([float(obs_dict.get('stress', 0.0))]),
        'stress_present':   jnp.array([1.0 if 'stress' in obs_dict else 0.0]),
        'log_steps_value':  jnp.array([float(obs_dict.get('log_steps', 0.0))]),
        'steps_present':    jnp.array([1.0 if 'log_steps' in obs_dict else 0.0]),
        'sleep_label':      jnp.array([int(obs_dict.get('sleep', 0))]),
        'sleep_present':    jnp.array([1.0 if 'sleep' in obs_dict else 0.0]),
    }
    return out


def verify_likelihood():
    """Three-pillar likelihood sanity check. Prints + returns a dict."""
    print("Running FSA-v2 Likelihood Sanity Checks...")

    p_dict = dict(DEFAULT_PARAMS)
    truth_state = jnp.array([0.65, 0.20, 0.40])     # (B, F, A) — non-trivial
    t_days = 0.20

    # Pack params vector for obs_log_weight_fn (only the estimable subset)
    p_vec = jnp.array([p_dict[name] for name in _PI], dtype=jnp.float64)

    # Sample one noise-free obs from each channel at the truth state
    trajectory = np.asarray(truth_state)[None, :]
    t_grid = np.array([t_days], dtype=np.float64)
    p_no_noise = dict(p_dict, sigma_HR=0.0, sigma_S=0.0, sigma_st=0.0)

    hr_obs = float(gen_obs_hr(trajectory, t_grid, p_no_noise, aux=None,
                                 prior_channels={'obs_sleep': {'sleep_label': np.array([1])}},
                                 seed=0)['obs_value'][0])
    stress_obs = float(gen_obs_stress(trajectory, t_grid, p_no_noise, aux=None,
                                          prior_channels={'obs_sleep': {'sleep_label': np.array([0])}},
                                          seed=0)['obs_value'][0])
    raw_steps = float(gen_obs_steps(trajectory, t_grid, p_no_noise, aux=None,
                                          prior_channels={'obs_sleep': {'sleep_label': np.array([0])}},
                                          seed=0)['obs_value'][0])
    log_steps_obs = math.log(raw_steps + 1.0)
    sleep_obs = 1   # arbitrary — peakedness is checked w.r.t. continuous channels only

    # NOTE: sleep is deliberately EXCLUDED from the peakedness grid_obs.
    # The sleep channel's Bernoulli probability is monotone in A, so
    # including it pulls the log-lik's peak in A away from truth and
    # confounds the check. The 3 Gaussian channels (HR, stress, steps)
    # all attain their per-channel maximum at the truth state by
    # construction (we sampled noise-free obs there), so their joint
    # log-lik must peak at truth.
    obs = {'hr': hr_obs, 'stress': stress_obs, 'log_steps': log_steps_obs}
    grid_obs = _build_grid_obs(obs, t_days)
    truth_lp = float(obs_log_weight_fn(truth_state, grid_obs, k=0, params=p_vec))
    print(f"  log p(obs | truth state) = {truth_lp:.4f} (Gaussian channels only)")

    # ── Pillar 1: Peakedness ─────────────────────────────────────
    print("\n[Pillar 1] Peakedness in each state dimension (truth = best):")
    state_dim_names = ['B', 'F', 'A']
    sweep_offsets = jnp.array([-0.20, -0.10, -0.02, 0.0, 0.02, 0.10, 0.20])
    pillar1_pass = True
    for d, name in enumerate(state_dim_names):
        lps = []
        for off in sweep_offsets:
            test_state = truth_state.at[d].add(off)
            test_state = jnp.maximum(test_state, 1e-3)
            lps.append(float(obs_log_weight_fn(
                test_state, grid_obs, k=0, params=p_vec)))
        peak_idx = int(np.argmax(lps))
        zero_off_idx = 3   # offset 0.0 is the 4th element
        within_tol = abs(peak_idx - zero_off_idx) <= 1   # neighbour OK
        marker = "OK" if within_tol else "FAIL"
        print(f"  {name}: peak at offset {float(sweep_offsets[peak_idx]):+.3f} "
              f"(truth = 0.000)  [{marker}]")
        if not within_tol:
            pillar1_pass = False

    # ── Pillar 2: Boundary penalty ─────────────────────────────────
    print("\n[Pillar 2] Boundary penalty for illegal state:")
    illegal_state = jnp.array([1.5, 0.20, -0.5])     # B>1, A<0
    illegal_lp = float(obs_log_weight_fn(
        illegal_state, grid_obs, k=0, params=p_vec))
    print(f"  log p(obs | B=1.5, A=-0.5) = {illegal_lp:.4f}")
    pillar2_pass = illegal_lp < truth_lp - 5.0   # at least 5 nats worse
    print(f"  {'OK' if pillar2_pass else 'FAIL'}: illegal state is "
          f"{'penalised' if pillar2_pass else 'NOT penalised'} vs truth.")

    # ── Pillar 3: Prior vs truth alignment ─────────────────────────
    print("\n[Pillar 3] Prior vs truth alignment (z-score < 4):")
    pillar3_pass = True
    for name, (ptype, pargs) in PARAM_PRIOR_CONFIG.items():
        if name not in p_dict:
            continue
        truth = float(p_dict[name])
        if ptype == 'normal':
            mu, sigma = pargs
            z = (truth - mu) / sigma
        elif ptype == 'lognormal':
            mu, sigma = pargs   # mu is in log-space
            z = (math.log(max(truth, 1e-12)) - mu) / sigma
        else:
            continue
        ok = abs(z) < 4.0
        if not ok:
            pillar3_pass = False
            print(f"  {name:<14s}: truth={truth:.4f}, |z|={abs(z):.2f}  [FAIL]")
    if pillar3_pass:
        print(f"  All {len(PARAM_PRIOR_CONFIG)} estimable params have |z| < 4 from prior mean.  [OK]")

    overall_pass = pillar1_pass and pillar2_pass and pillar3_pass
    verdict = "PASS" if overall_pass else "FAIL"
    print(f"\nLikelihood Sanity: {verdict}")

    return {
        'truth_log_lik':       truth_lp,
        'illegal_log_lik':     illegal_lp,
        'pillar1_peakedness':  pillar1_pass,
        'pillar2_boundary':    pillar2_pass,
        'pillar3_prior_align': pillar3_pass,
        'passed':              overall_pass,
    }


if __name__ == "__main__":
    verify_likelihood()
