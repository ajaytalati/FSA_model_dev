import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from models.fsa_high_res.control import build_control_spec

# Enable 64-bit for stable optimization
from jax import config
config.update("jax_enable_x64", True)

def run_study_v4():
    print("\n--- Starting FSA-v4 Variable-Dose Study (20-week horizon) ---")
    T = 140.0 # 20 weeks
    n_anchors = int(T / 7)
    
    # Stricter fatigue budget for v4
    spec = build_control_spec(T_total_days=T, n_inner=64, n_anchors=n_anchors, 
                              lam_barrier=100.0, seed=123)
    
    theta = jnp.zeros(spec.theta_dim, dtype=jnp.float64)
    
    lr = 0.0005 
    l2_reg = 0.1 
    n_iters = 2000
    
    best_theta = theta
    min_cost = jnp.inf

    @jax.jit
    def step(theta_curr):
        def regularized_cost(t):
            return spec.cost_fn(t) + l2_reg * jnp.sum(t**2)
        val, grads = jax.value_and_grad(regularized_cost)(theta_curr)
        grads = jnp.clip(grads, -0.1, 0.1)
        theta_next = theta_curr - lr * grads
        return theta_next, val

    print(f"  Optimizing over {n_iters} iterations...")
    for i in range(n_iters):
        theta_next, cost = step(theta)
        if jnp.isnan(cost): break
        theta = theta_next
        if cost < min_cost:
            min_cost = cost
            best_theta = theta
        if i % 500 == 0:
            print(f"    Iter {i:4d}: Cost = {cost:10.4f}")

    # Rollout
    optimal_phi = spec.schedule_from_theta(best_theta)
    key = jax.random.PRNGKey(42)
    trajectory = spec._traj_sample_fn(best_theta, key)
    t_grid = jnp.arange(spec.n_steps) * spec.dt
    
    # Plotting (3 panels)
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    # Plot 1: Main States
    axes[0].plot(t_grid, trajectory[:, 0], 'b-', label='Fitness (B)')
    axes[0].plot(t_grid, trajectory[:, 1], 'g-', label='Strength (S)')
    axes[0].plot(t_grid, trajectory[:, 2], 'r--', label='Fatigue (F)', alpha=0.5)
    axes[0].plot(t_grid, trajectory[:, 3], 'k-', label='Autonomic (A)', linewidth=1.5)
    axes[0].axhline(0.40, color='r', linestyle=':', label='F_max')
    axes[0].set_ylabel('State Value')
    axes[0].set_title('FSA-v4 Variable Dose: 20-Week Transition')
    axes[0].legend(loc='upper left', fontsize='small', ncol=2)
    axes[0].grid(True, alpha=0.2)
    
    # Plot 2: Variable Dose (K gains)
    axes[1].plot(t_grid, trajectory[:, 4], 'b-', label='Aerobic Gain ($K_{FB}$)')
    axes[1].plot(t_grid, trajectory[:, 5], 'g-', label='Strength Gain ($K_{FS}$)')
    axes[1].set_ylabel('Fatigue Sensitivity')
    axes[1].legend(loc='upper left', fontsize='small')
    axes[1].grid(True, alpha=0.2)
    
    # Plot 3: Control Inputs
    axes[1].plot(t_grid, optimal_phi[:, 0], 'b-', alpha=0.2) # overlay for context
    axes[1].plot(t_grid, optimal_phi[:, 1], 'g-', alpha=0.2)
    
    axes[2].plot(t_grid, optimal_phi[:, 0], 'b-', alpha=0.7, label='$\Phi_B$')
    axes[2].plot(t_grid, optimal_phi[:, 1], 'g-', alpha=0.7, label='$\Phi_S$')
    axes[2].set_ylabel('Stimulus Rate')
    axes[2].set_xlabel('Days')
    axes[2].legend(loc='upper left', fontsize='small')
    axes[2].grid(True, alpha=0.2)
    
    plt.tight_layout()
    os.makedirs('LaTex_docs/figures', exist_ok=True)
    plt.savefig('LaTex_docs/figures/dynamic_transition_v4_20w.png', dpi=200)
    print("  Saved plot to LaTex_docs/figures/dynamic_transition_v4_20w.png")

if __name__ == "__main__":
    run_study_v4()
