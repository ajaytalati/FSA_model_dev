import numpy as np
import matplotlib.pyplot as plt
from models.fsa_high_res._dynamics import TRUTH_PARAMS, A_TYP, F_TYP

def compute_goldilocks_surface():
    p = TRUTH_PARAMS
    
    # Range of stimuli
    phi_b_range = np.linspace(0.0, 4.0, 50)
    phi_s_range = np.linspace(0.0, 4.0, 50)
    Phi_B, Phi_S = np.meshgrid(phi_b_range, phi_s_range)
    
    # 1. Compute deterministic equilibria for B, S, F
    # B* = kappa_B * Phi_B * tau_B / (1 + eps_AB * A_typ)
    # But in G1 form, kappa_B is effective gain at A_typ, so B* = kappa_B * Phi_B * tau_B
    # Actually, dB/dt = kappa_B * Gamma_B(A) * Phi_B - B/tau_B
    # At A=Atyp, Gamma=1, so B* = kappa_B * Phi_B * tau_B
    # We clip at 1.0
    B_star = np.clip(p['kappa_B'] * Phi_B * p['tau_B'], 0, 1.0)
    S_star = np.clip(p['kappa_S'] * Phi_S * p['tau_S'], 0, 1.0)
    
    # F* = (kappa_FB * Phi_B + kappa_FS * Phi_S) * tau_F
    F_star = (p['kappa_FB'] * Phi_B + p['kappa_FS'] * Phi_S) * p['tau_F']
    
    # 2. Compute Stuart-Landau growth rate mu(B*, S*, F*)
    F_dev = F_star - F_TYP
    mu = (p['mu_0'] 
          + p['mu_B'] * B_star 
          + p['mu_S'] * S_star 
          - p['mu_F'] * F_star 
          - p['mu_FF'] * F_dev**2)
    
    # 3. Oscillatory amplitude A* = sqrt(max(mu, 0) / eta)
    A_star = np.sqrt(np.maximum(mu, 0) / p['eta'])
    
    # Plotting
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(Phi_B, Phi_S, A_star, cmap='viridis', edgecolor='none')
    ax.set_xlabel('Aerobic Stimulus ($\Phi_B$)')
    ax.set_ylabel('Strength Stimulus ($\Phi_S$)')
    ax.set_zlabel('Autonomic Amplitude ($A^*$)')
    ax.set_title('FSA-v3 Goldilocks Surface')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    plt.savefig('LaTex_docs/figures/goldilocks_surface_v3.png', dpi=300)
    print("Saved Goldilocks surface plot to LaTex_docs/figures/goldilocks_surface_v3.png")

    # Generate a small table for LaTeX
    print("\nSelected Surface Values for LaTeX Table:")
    print("Phi_B | Phi_S | B*    | S*    | F*    | mu    | A*")
    print("------|-------|-------|-------|-------|-------|----")
    indices = [0, 12, 25, 37, 49]
    for i in indices:
        for j in indices:
            pb = phi_b_range[j]
            ps = phi_s_range[i]
            b = B_star[i, j]
            s = S_star[i, j]
            f = F_star[i, j]
            m = mu[i, j]
            a = A_star[i, j]
            print(f"{pb:.1f}  | {ps:.1f}  | {b:.3f} | {s:.3f} | {f:.3f} | {m:+.3f} | {a:.3f}")

if __name__ == "__main__":
    import os
    os.makedirs('LaTex_docs/figures', exist_ok=True)
    compute_goldilocks_surface()
