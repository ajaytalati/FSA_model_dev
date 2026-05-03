import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax
import jax.numpy as jnp
import numpy as np
from models.fsa_high_res.simulation import HIGH_RES_FSA_V3_MODEL, BINS_PER_DAY, DT_BIN_DAYS
from models.fsa_high_res._dynamics import TRUTH_PARAMS, drift_jax

def calculate_fim():
    """Simplified FIM rank check for FSA-v3."""
    print("Checking FSA-v3 Fisher Information Matrix (FIM) rank...")
    
    # We'll check the sensitivity of the state to a few key parameters:
    # kappa_B, kappa_S, kappa_FB, kappa_FS
    
    p_jax = {k: jnp.array(v) for k, v in TRUTH_PARAMS.items()}
    y0 = jnp.array([0.1, 0.1, 0.2, 0.1])
    
    # 2-day trajectory
    n_steps = 2 * BINS_PER_DAY
    dt = DT_BIN_DAYS
    
    # Bimodal stimulus: Cardio Day 1, Strength Day 2
    phi = jnp.zeros((n_steps, 2))
    phi = phi.at[:BINS_PER_DAY, 0].set(1.0)
    phi = phi.at[BINS_PER_DAY:, 1].set(1.0)
    
    def simulate(params_subset):
        # Update full params with subset
        p = p_jax.copy()
        p.update(params_subset)
        
        def step(y, i):
            y_next = y + dt * drift_jax(y, p, phi[i])
            return y_next, y_next
            
        _, traj = jax.lax.scan(step, y0, jnp.arange(n_steps))
        return traj

    # Parameters to check identifiability for
    target_params = {
        'kappa_B': p_jax['kappa_B'],
        'kappa_S': p_jax['kappa_S'],
        'kappa_FB': p_jax['kappa_FB'],
        'kappa_FS': p_jax['kappa_FS']
    }
    
    # Jacobian of trajectory w.r.t. parameters
    # shape: (n_steps, 4_states, n_params)
    jac_fn = jax.jacobian(simulate)
    jac = jac_fn(target_params)
    
    # Flatten across time and states for each parameter
    # list of (n_steps * 4,) arrays
    flat_jac = []
    for p_name in target_params.keys():
        flat_jac.append(jac[p_name].reshape(-1))
        
    J = jnp.stack(flat_jac, axis=1) # (N, 4)
    
    # FIM = J.T @ J
    FIM = J.T @ J
    
    print(f"FIM matrix:\n{FIM}")
    
    # Singular values
    s = jnp.linalg.svd(FIM, compute_uv=False)
    print(f"Singular values: {s}")
    
    rank = jnp.linalg.matrix_rank(FIM)
    print(f"FIM Rank: {rank} / {len(target_params)}")
    
    assert rank == len(target_params), "FIM is rank deficient! Parameters are collinear."
    print("  FIM Rank check passed.")

if __name__ == "__main__":
    calculate_fim()
