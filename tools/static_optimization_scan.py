import numpy as np
import matplotlib.pyplot as plt
from models.fsa_high_res._dynamics import TRUTH_PARAMS, A_TYP, F_TYP

def static_optimization_scan():
    p = TRUTH_PARAMS
    
    # Range of stimuli (extend slightly higher to see collapse)
    phi_b_range = np.linspace(0.0, 5.0, 100)
    phi_s_range = np.linspace(0.0, 5.0, 100)
    Phi_B, Phi_S = np.meshgrid(phi_b_range, phi_s_range)
    
    # 1. Deterministic Equilibria
    B_star = np.clip(p['kappa_B'] * Phi_B * p['tau_B'], 0, 1.0)
    S_star = np.clip(p['kappa_S'] * Phi_S * p['tau_S'], 0, 1.0)
    F_star = (p['kappa_FB'] * Phi_B + p['kappa_FS'] * Phi_S) * p['tau_F']
    
    # 2. Stuart-Landau mu and A*
    F_dev = F_star - F_TYP
    mu = (p['mu_0'] 
          + p['mu_B'] * B_star 
          + p['mu_S'] * S_star 
          - p['mu_F'] * F_star 
          - p['mu_FF'] * F_dev**2)
    
    A_star = np.sqrt(np.maximum(mu, 0) / p['eta'])
    
    # 3. Concurrent Reward functional: R = A* + B* + S*
    # (Note: we use the steady state values directly)
    Reward = A_star + B_star + S_star
    
    # Find global optimum in this static landscape
    idx = np.unravel_index(np.argmax(Reward), Reward.shape)
    best_phi_b = phi_b_range[idx[1]]
    best_phi_s = phi_s_range[idx[0]]
    max_reward = Reward[idx]
    
    print(f"Static Optimization Results:")
    print(f"  Best Phi_B: {best_phi_b:.3f}")
    print(f"  Best Phi_S: {best_phi_s:.3f}")
    print(f"  Max Total Reward (A*+B*+S*): {max_reward:.3f}")
    print(f"  At this point: B*={B_star[idx]:.3f}, S*={S_star[idx]:.3f}, A*={A_star[idx]:.3f}, F*={F_star[idx]:.3f}")

    # Plotting the Reward Surface
    fig = plt.figure(figsize=(14, 6))
    
    # Left: Total Reward Surface
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(Phi_B, Phi_S, Reward, cmap='magma', edgecolor='none', alpha=0.8)
    ax1.scatter([best_phi_b], [best_phi_s], [max_reward], color='cyan', s=100, label='Optimum')
    ax1.set_xlabel('$\Phi_B$ (Aerobic)')
    ax1.set_ylabel('$\Phi_S$ (Strength)')
    ax1.set_zlabel('Reward ($A^*+B^*+S^*$)')
    ax1.set_title('Concurrent Training Reward Surface')
    
    # Right: Contour plot with optimum
    ax2 = fig.add_subplot(122)
    cp = ax2.contourf(Phi_B, Phi_S, Reward, levels=20, cmap='magma')
    fig.colorbar(cp, ax=ax2)
    ax2.plot(best_phi_b, best_phi_s, 'cx', markersize=10, markeredgewidth=2)
    ax2.set_xlabel('$\Phi_B$ (Aerobic)')
    ax2.set_ylabel('$\Phi_S$ (Strength)')
    ax2.set_title('Reward Contours')
    ax2.annotate(f'Optimum: ({best_phi_b:.2f}, {best_phi_s:.2f})', 
                 xy=(best_phi_b, best_phi_s), xytext=(best_phi_b+0.2, best_phi_s+0.2),
                 color='white', fontweight='bold')

    plt.tight_layout()
    plt.savefig('LaTex_docs/figures/concurrent_reward_surface.png', dpi=300)
    print("Saved reward surface plot to LaTex_docs/figures/concurrent_reward_surface.png")

    # Generate LaTeX Table
    print("\nReward Landscape Table for LaTeX:")
    print("Phi_B | Phi_S | B*    | S*    | A*    | Total Reward")
    print("------|-------|-------|-------|-------|-------------")
    # Sample points near the optimum and corners
    sample_indices = [
        (0,0), (20,0), (40,0), (60,0), # Aerobic only
        (0,20), (0,40), (0,60),        # Strength only
        (20,20), (40,40), (idx[1], idx[0]) # Mixed and Optimum
    ]
    # Sort by reward for clarity in the doc
    sample_points = []
    for j, i in sample_indices:
        sample_points.append((phi_b_range[j], phi_s_range[i], B_star[i,j], S_star[i,j], A_star[i,j], Reward[i,j]))
    
    for pb, ps, b, s, a, r in sorted(sample_points, key=lambda x: x[5], reverse=True):
        print(f"{pb:.1f}  | {ps:.1f}  | {b:.3f} | {s:.3f} | {a:.3f} | {r:.3f}")

if __name__ == "__main__":
    static_optimization_scan()
