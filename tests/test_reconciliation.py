"""Plant ↔ Estimator drift reconciliation (Mirror Test) for FSA-v2.

The plant (`StepwisePlant._plant_em_step`) imports `drift_jax` from
`models.fsa_high_res._dynamics`. The estimator's `propagate_fn`
inlines its own copy of the drift formula (lines ~150-160 of
`estimation.py`). If those two ever silently drift apart, every
closed-loop SMC²-MPC bench downstream produces wrong results — the
filter is propagating particles through different dynamics than the
plant being filtered.

This test pins them together: for a known state + Phi_t + dt, the
single-step Euler prediction from each side must be bit-equivalent.

Two layers of test
------------------
1. **Drift parity**: assert `drift_jax(y, params, Phi_t)` (used by
   the plant) equals the estimator's inline drift computation, recovered
   by stripping the noise + Kalman fusion out of `propagate_fn` (run it
   with all `*_present` masks = 0 so no fusion happens, and `noise = 0`
   so the sampled state equals `mu_prior`).

2. **Plant smoke**: run `StepwisePlant.advance(stride_bins=1)` and
   confirm it returns sensible state in the right shape, exercising
   the full plant pipeline end-to-end.
"""
import os
import sys
from pathlib import Path

# Force CPU + X64 BEFORE importing JAX.
os.environ['JAX_ENABLE_X64'] = 'True'
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import jax
import jax.numpy as jnp
import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.fsa_high_res._dynamics import drift_jax
from models.fsa_high_res._plant import StepwisePlant
from models.fsa_high_res.simulation import (
    DEFAULT_INIT,
    DEFAULT_PARAMS,
    BINS_PER_DAY,
    DT_BIN_DAYS,
    circadian,
)
from models.fsa_high_res.estimation import (
    HIGH_RES_FSA_V2_ESTIMATION,
    _PI,
    propagate_fn,
)


def test_plant_and_estimator_share_drift():
    """One-step Euler prediction must be bit-equivalent on both sides.

    Plant: y + dt · drift_jax(y, params_dict, Phi_t)
    Estimator: y + dt · (inline drift in propagate_fn)

    With `noise = 0` and all `*_present` masks = 0, the estimator's
    `propagate_fn` returns its `mu_prior` (the deterministic Euler
    prediction). Compare to the plant's drift_jax evaluation.
    """
    # Setup
    y0 = jnp.array([0.05, 0.30, 0.10], dtype=jnp.float64)   # COLD_START_INIT
    Phi_t = 1.2     # mid-range training-strain rate
    t_days = 0.20   # mid-morning bin
    dt = DT_BIN_DAYS

    # --- Plant's drift evaluation (the ground-truth side) ---
    drift_plant = drift_jax(y0, DEFAULT_PARAMS, Phi_t)
    y_plant_euler = y0 + dt * drift_plant

    # --- Estimator's drift via propagate_fn with no fusion + no noise ---
    p_vec = jnp.array(
        [DEFAULT_PARAMS[name] for name in _PI], dtype=jnp.float64)
    C_t = float(circadian(jnp.array([t_days]),
                            phi=DEFAULT_PARAMS.get('phi', 0.0))[0])
    grid_obs = {
        'C':                jnp.array([C_t]),
        'Phi':              jnp.array([Phi_t]),
        'hr_value':         jnp.array([0.0]),  'hr_present':     jnp.array([0.0]),
        'stress_value':     jnp.array([0.0]),  'stress_present': jnp.array([0.0]),
        'log_steps_value':  jnp.array([0.0]),  'steps_present':  jnp.array([0.0]),
        'sleep_label':      jnp.array([0]),    'sleep_present':  jnp.array([0.0]),
    }
    noise_zero = jnp.zeros(3, dtype=jnp.float64)
    y_est, pred_lw = propagate_fn(
        y0, t_days, dt, p_vec, grid_obs, k=0,
        sigma_diag=None, noise=noise_zero, rng_key=jax.random.PRNGKey(0))

    # With noise=0 and no fusion (*_present all 0), y_est should be
    # mu_prior + L @ 0 = mu_prior = y0 + dt · inline_drift.
    diff = np.abs(np.asarray(y_est) - np.asarray(y_plant_euler))
    assert float(diff.max()) < 1e-10, (
        f"Plant and estimator drift formulae disagree:\n"
        f"  plant Euler step:    {np.asarray(y_plant_euler)}\n"
        f"  estimator mu_prior:  {np.asarray(y_est)}\n"
        f"  abs diff:            {diff}\n"
        f"This means models/fsa_high_res/_dynamics.py:drift_jax has\n"
        f"drifted from the inline copy in estimation.py:propagate_fn.\n"
        f"Closed-loop MPC will silently produce wrong results.")


def test_plant_advance_smoke():
    """Drive `StepwisePlant.advance(1 bin)` with a constant Phi and
    verify the plant returns sensible state shapes + ranges.
    """
    plant = StepwisePlant(
        truth_params=dict(DEFAULT_PARAMS),
        state=np.array([
            DEFAULT_INIT['B_0'], DEFAULT_INIT['F_0'], DEFAULT_INIT['A_0']
        ], dtype=np.float64),
        seed_offset=42,
        dt=DT_BIN_DAYS,
    )

    # advance one 15-min bin under Phi=1.0 (one day's worth of Phi_daily
    # since stride_bins=1 < BINS_PER_DAY=96).
    out = plant.advance(stride_bins=1, Phi_daily=np.array([1.0]))

    # Trajectory shape
    assert 'trajectory' in out
    assert out['trajectory'].shape == (1, 3), (
        f"plant.advance(1) trajectory shape {out['trajectory'].shape} != (1, 3)")

    # State must remain in physical bounds: B in [0,1], F >= 0, A >= 0.
    B, F, A = out['trajectory'][0]
    assert 0.0 <= B <= 1.0, f"B={B} out of [0,1] after one bin"
    assert F >= 0.0, f"F={F} negative after one bin"
    assert A >= 0.0, f"A={A} negative after one bin"

    # All 4 obs channels + Phi + C must be in the output dict
    for key in ('obs_HR', 'obs_sleep', 'obs_stress', 'obs_steps', 'Phi', 'C'):
        assert key in out, f"plant output missing {key!r}"


if __name__ == "__main__":
    test_plant_and_estimator_share_drift()
    test_plant_advance_smoke()
    print("Reconciliation tests PASSED.")
