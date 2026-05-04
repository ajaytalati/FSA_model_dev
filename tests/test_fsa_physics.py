import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax
import jax.numpy as jnp
import numpy as np
from models.fsa_high_res.simulation import HIGH_RES_FSA_V2_MODEL, BINS_PER_DAY, DT_BIN_DAYS
from smc2fc.simulator.sde_solver_diffrax import solve_sde_jax

def test_simulation_run():
    print("Running FSA-v2 simulation smoke test...")
    model = HIGH_RES_FSA_V2_MODEL
    
    # Healthy baseline params and init
    from models.fsa_high_res._dynamics import TRUTH_PARAMS
    from models.fsa_high_res.simulation import DEFAULT_INIT
    
    t_total_days = 2.0
    n_steps = int(t_total_days * BINS_PER_DAY)
    t_grid = jnp.arange(n_steps) * DT_BIN_DAYS
    
    # Simple constant Phi=1.0 schedule
    phi_schedule = jnp.ones(n_steps)
    
    # Prepare exogenous (circadian)
    from models.fsa_high_res.simulation import circadian_jax
    c_grid = jax.vmap(circadian_jax)(t_grid)
    exogenous = {'C': c_grid, 'Phi_arr': phi_schedule}
    
    print("  Solving SDE...")
    # init_state should be the dict that make_y0 expects
    trajectory = solve_sde_jax(
        model, TRUTH_PARAMS, DEFAULT_INIT, t_grid, exogenous, seed=42
    )
    
    print(f"  Simulation complete. Trajectory shape: {trajectory.shape}")
    assert trajectory.shape == (n_steps, 3)
    
    # Basic physics checks
    B, F, A = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
    assert jnp.all(B >= -1e-5)
    assert jnp.all(F >= -1e-5)
    assert jnp.all(A >= -1e-5)
    print("  Physics checks passed.")

if __name__ == "__main__":
    test_simulation_run()
