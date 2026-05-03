import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax
import jax.numpy as jnp
import numpy as np
from models.fsa_high_res.simulation import HIGH_RES_FSA_V3_MODEL, BINS_PER_DAY, DT_BIN_DAYS
from simulator.sde_solver_diffrax import solve_sde_jax

def test_v3_simulation_run():
    print("Running FSA-v3 simulation smoke test...")
    model = HIGH_RES_FSA_V3_MODEL
    
    from models.fsa_high_res.simulation import DEFAULT_PARAMS, DEFAULT_INIT
    
    t_total_days = 2.0
    n_steps = int(t_total_days * BINS_PER_DAY)
    t_grid = jnp.arange(n_steps) * DT_BIN_DAYS
    
    # 2D Phi schedule: [Phi_B, Phi_S]
    # Day 1: Aerobic=1.0, Strength=0.0
    # Day 2: Aerobic=0.0, Strength=1.5
    phi_schedule = jnp.zeros((n_steps, 2))
    phi_schedule = phi_schedule.at[:BINS_PER_DAY, 0].set(1.0)
    phi_schedule = phi_schedule.at[BINS_PER_DAY:, 1].set(1.5)
    
    from models.fsa_high_res.simulation import circadian_jax
    c_grid = jax.vmap(circadian_jax)(t_grid)
    exogenous = {'C': c_grid, 'Phi_arr': phi_schedule}
    
    print("  Solving 4D SDE...")
    trajectory = solve_sde_jax(
        model, DEFAULT_PARAMS, DEFAULT_INIT, t_grid, exogenous, seed=42
    )
    
    print(f"  Simulation complete. Trajectory shape: {trajectory.shape}")
    assert trajectory.shape == (n_steps, 4)
    
    # Basic physics checks
    B, S, F, A = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], trajectory[:, 3]
    assert jnp.all(B >= -1e-5) and jnp.all(B <= 1.00001)
    assert jnp.all(S >= -1e-5) and jnp.all(S <= 1.00001)
    assert jnp.all(F >= -1e-5)
    assert jnp.all(A >= -1e-5)
    
    # Stimulus response checks
    # B should increase in Day 1, S should stay flat
    # S should increase in Day 2, B should decay
    assert B[BINS_PER_DAY-1] > B[0]
    assert S[BINS_PER_DAY-1] < S[0] + 1e-4 # S might fluctuate slightly due to noise but shouldn't rise
    assert S[n_steps-1] > S[BINS_PER_DAY]
    assert B[n_steps-1] < B[BINS_PER_DAY-1]

    print("  Physics checks passed.")

if __name__ == "__main__":
    test_v3_simulation_run()
