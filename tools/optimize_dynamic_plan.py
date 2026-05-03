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

def run_study():
    horizons = [28.0, 56.0, 84.0, 112.0, 140.0]
    names = ["4w", "8w", "12w", "16w", "20w"]
    
    # Global plot directory
    os.makedirs('LaTex_docs/figures', exist_ok=True)

    for T, name in zip(horizons, names):
        print(f"\n--- Starting Horizon Study: {name} ({int(T)} days) ---")
        
        # Scale anchors: 1 per week (min 8)
        n_anchors = max(8, int(T / 7))
        # Stricter fatigue budget: lam_barrier=100.0
        spec = build_control_spec(T_total_days=T, n_inner=64, n_anchors=n_anchors, 
                                  lam_barrier=100.0, seed=123)
        
        theta = jnp.zeros(spec.theta_dim, dtype=jnp.float64)
        
        # Robust optimization settings
        lr = 0.0005 
        l2_reg = 0.1 # Stronger reg to keep plan stable under high barrier
        n_iters = 2000
        
        best_theta = theta
        min_cost = jnp.inf

        @jax.jit
        def step(theta_curr):
            def regularized_cost(t):
                return spec.cost_fn(t) + l2_reg * jnp.sum(t**2)
            val, grads = jax.value_and_grad(regularized_cost)(theta_curr)
            # Strict gradient clipping
            grads = jnp.clip(grads, -0.1, 0.1)
            theta_next = theta_curr - lr * grads
            return theta_next, val

        print(f"  Optimizing over {n_iters} iterations...")
        for i in range(n_iters):
            theta_next, cost = step(theta)
            
            if jnp.isnan(cost) or jnp.isinf(cost):
                print(f"    Warning: Numerical instability at iter {i}. Reverting to best.")
                break
            
            theta = theta_next
            # We want to minimize cost (maximize reward)
            if cost < min_cost:
                min_cost = cost
                best_theta = theta
            
            if i % 500 == 0:
                print(f"    Iter {i:4d}: Cost = {cost:10.4f}")

        print(f"  Optimization Finished. Best Cost: {min_cost:.4f}")
        
        # Final Rollout for Plotting
        theta_final = best_theta
        optimal_phi = spec.schedule_from_theta(theta_final)
        key = jax.random.PRNGKey(42)
        trajectory = spec._traj_sample_fn(theta_final, key)
        
        # Safety check for NaNs in trajectory
        if jnp.any(jnp.isnan(trajectory)):
            print(f"  CRITICAL: Rollout contains NaNs for {name}. Plot will be degraded.")
            # Fallback to zeros for plotting so we don't get blank axes
            trajectory = jnp.nan_to_num(trajectory)
            
        t_grid = jnp.arange(spec.n_steps) * spec.dt
        
        # Plotting
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Plot 1: Latent States
        axes[0].plot(t_grid, trajectory[:, 0], 'b-', label='B (Fitness)', alpha=0.9)
        axes[0].plot(t_grid, trajectory[:, 1], 'g-', label='S (Strength)', alpha=0.9)
        axes[0].plot(t_grid, trajectory[:, 2], 'r--', label='Fatigue (F)', alpha=0.5)
        axes[0].plot(t_grid, trajectory[:, 3], 'k-', label='Autonomic (A)', linewidth=1.5)
        axes[0].axhline(0.40, color='r', linestyle=':', label='F_max limit', alpha=0.8)
        axes[0].set_ylabel('State Value')
        axes[0].set_title(f'Optimal Bimodal Plan: {name} Horizon ({int(T)} days)')
        axes[0].legend(loc='upper left', fontsize='small', ncol=2)
        axes[0].grid(True, alpha=0.2, linestyle='--')
        
        # Plot 2: Stimulus Schedules
        axes[1].plot(t_grid, optimal_phi[:, 0], 'b-', alpha=0.7, label='Aerobic ($\Phi_B$)')
        axes[1].plot(t_grid, optimal_phi[:, 1], 'g-', alpha=0.7, label='Strength ($\Phi_S$)')
        axes[1].set_ylabel('Stimulus Rate')
        axes[1].set_xlabel('Time (days)')
        axes[1].legend(loc='upper left', fontsize='small')
        axes[1].grid(True, alpha=0.2, linestyle='--')
        
        # Final visual Polish
        plt.tight_layout()
        save_path = f'LaTex_docs/figures/dynamic_transition_{name}.png'
        plt.savefig(save_path, dpi=150) # Reduced DPI for faster compilation but still high quality
        plt.close(fig)
        print(f"  Successfully saved converged plot to {save_path}")

if __name__ == "__main__":
    run_study()
