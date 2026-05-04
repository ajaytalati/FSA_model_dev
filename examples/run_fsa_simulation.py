import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from models.fsa_high_res.simulation import HIGH_RES_FSA_V2_MODEL, BINS_PER_DAY, DT_BIN_DAYS
from smc2fc.simulator.sde_solver_diffrax import solve_sde_jax
from models.fsa_high_res._dynamics import TRUTH_PARAMS, DEFAULT_INIT
from models.fsa_high_res.simulation import circadian_jax

def run_example():
    print("Running FSA-v2 Simulation Example...")
    
    t_total_days = 7.0
    n_steps = int(t_total_days * BINS_PER_DAY)
    t_grid = jnp.arange(n_steps) * DT_BIN_DAYS
    
    # Training schedule: high strain for 3 days, then rest
    phi_schedule = jnp.where(t_grid < 3.0, 2.0, 0.2)
    
    # Prepare exogenous
    c_grid = jax.vmap(circadian_jax)(t_grid)
    exogenous = {'C': c_grid, 'Phi_arr': phi_schedule}
    
    # Solve SDE
    print("  Solving SDE...")
    trajectory = solve_sde_jax(
        HIGH_RES_FSA_V2_MODEL, TRUTH_PARAMS, DEFAULT_INIT, t_grid, exogenous, seed=42
    )
    
    # Plotting (if matplotlib is available)
    try:
        fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
        
        axes[0].plot(t_grid, phi_schedule, 'k-', label='Training ($\Phi$)')
        axes[0].set_ylabel('Strain')
        axes[0].legend()
        
        axes[1].plot(t_grid, trajectory[:, 0], 'g-', label='Fitness (B)')
        axes[1].set_ylabel('B')
        axes[1].legend()
        
        axes[2].plot(t_grid, trajectory[:, 1], 'r-', label='Fatigue (F)')
        axes[2].set_ylabel('F')
        axes[2].legend()
        
        axes[3].plot(t_grid, trajectory[:, 2], 'b-', label='Autonomic (A)')
        axes[3].set_ylabel('A')
        axes[3].set_xlabel('Time (days)')
        axes[3].legend()
        
        plt.tight_layout()
        plt.savefig('fsa_simulation.png')
        print("  Plot saved to fsa_simulation.png")
    except Exception as e:
        print(f"  Could not plot: {e}")

if __name__ == "__main__":
    run_example()
