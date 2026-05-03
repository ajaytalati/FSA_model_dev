import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax
import jax.numpy as jnp
import numpy as np
from models.fsa_high_res.simulation import HIGH_RES_FSA_V4_MODEL, BINS_PER_DAY, DT_BIN_DAYS
from simulator.sde_solver_diffrax import solve_sde_jax

def test_v4_simulation_run():
    print("Running FSA-v4 simulation smoke test...")
    model = HIGH_RES_FSA_V4_MODEL
    
    from models.fsa_high_res.simulation import DEFAULT_PARAMS, DEFAULT_INIT
    
    t_total_days = 2.0
    n_steps = int(t_total_days * BINS_PER_DAY)
    t_grid = jnp.arange(n_steps) * DT_BIN_DAYS
    
    phi_schedule = jnp.zeros((n_steps, 2))
    phi_schedule = phi_schedule.at[:BINS_PER_DAY, 0].set(1.0)
    phi_schedule = phi_schedule.at[BINS_PER_DAY:, 1].set(1.5)
    
    from models.fsa_high_res.simulation import circadian_jax
    c_grid = jax.vmap(circadian_jax)(t_grid)
    exogenous = {'C': c_grid, 'Phi_arr': phi_schedule}
    
    print("  Solving 6D SDE...")
    trajectory = solve_sde_jax(
        model, DEFAULT_PARAMS, DEFAULT_INIT, t_grid, exogenous, seed=42
    )
    
    print(f"  Simulation complete. Trajectory shape: {trajectory.shape}")
    assert trajectory.shape == (n_steps, 6)
    
    B, S, F, A, KFB, KFS = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], trajectory[:, 3], trajectory[:, 4], trajectory[:, 5]
    assert jnp.all(B >= -1e-5) and jnp.all(B <= 1.00001)
    assert jnp.all(S >= -1e-5) and jnp.all(S <= 1.00001)
    assert jnp.all(F >= -1e-5)
    assert jnp.all(A >= -1e-5)
    assert jnp.all(KFB >= -1e-5)
    assert jnp.all(KFS >= -1e-5)
    
    # Check that KFB increases during day 1
    assert KFB[BINS_PER_DAY-1] > DEFAULT_INIT['KFB_0']
    # Check that KFS increases during day 2
    assert KFS[n_steps-1] > DEFAULT_INIT['KFS_0']

    print("  Physics checks passed.")

if __name__ == "__main__":
    test_v4_simulation_run()
