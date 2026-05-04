"""Numerical stability and stiffness audit for FSA-v2.

Computes the Jacobian of the drift `drift_jax(y, params, Phi_t)` across
a 4-dimensional grid of (B, F, A, Phi). The largest spectral radius
(absolute eigenvalue) sets the maximum stable step size for explicit
integration: `h_max = 2 / max_rho` (in /day units).

Returns a result dict with `h_max_mins`, `max_rho_per_day`,
`n_substeps_for_15min`, and a `passed` bool indicating whether the
model is stable for 15-minute bins.
"""
import os
import sys
from pathlib import Path

# Force CPU + X64 BEFORE importing JAX.
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'True'

import jax
import jax.numpy as jnp
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.fsa_high_res._dynamics import drift_jax
from models.fsa_high_res.simulation import DEFAULT_PARAMS


def audit_stiffness():
    """Vectorized stiffness audit. Prints + returns a result dict."""
    print("Running Vectorized Stiffness Audit (CPU)...")
    p_dict = {k: jnp.float64(v) for k, v in DEFAULT_PARAMS.items()
                if isinstance(v, (int, float))}

    # Jacobian of drift_jax wrt y, evaluated at (y, params, Phi_t).
    jac_fn = jax.jacobian(drift_jax, argnums=0)
    jac_vmap = jax.vmap(lambda y, phi: jac_fn(y, p_dict, phi))

    # Sample grid over the physiological range:
    #   B ∈ [0.05, 0.95]     fitness (Banister chronic)
    #   F ∈ [0.05, 1.0]      fatigue (Banister acute, F_TYP ≈ 0.30)
    #   A ∈ [0.05, 1.5]      autonomic amplitude (Stuart-Landau)
    #   Phi ∈ [0.0, 3.0]     training-strain rate (/day)
    n_samples = 6
    b_grid = jnp.linspace(0.05, 0.95, n_samples)
    f_grid = jnp.linspace(0.05, 1.00, n_samples)
    a_grid = jnp.linspace(0.05, 1.50, n_samples)
    phi_grid = jnp.linspace(0.0, 3.0, n_samples)

    B, F, A, Phi = jnp.meshgrid(
        b_grid, f_grid, a_grid, phi_grid, indexing='ij')
    y_flat = jnp.stack([B.flatten(), F.flatten(), A.flatten()], axis=1)
    phi_flat = Phi.flatten()
    n_pts = len(y_flat)

    print(f"Computing Jacobians for {n_pts} (B, F, A, Phi) grid points...")
    Js = jac_vmap(y_flat, phi_flat)             # (N, 3, 3)

    print("Computing spectral radii...")
    eigvals = jnp.linalg.eigvals(Js)             # (N, 3)
    rhos = jnp.max(jnp.abs(eigvals), axis=1)     # (N,)

    max_idx = int(jnp.argmax(rhos))
    max_rho = float(rhos[max_idx])
    worst_y = np.asarray(y_flat[max_idx])
    worst_phi = float(phi_flat[max_idx])

    h_max_days = 2.0 / max_rho
    h_max_mins = h_max_days * 24.0 * 60.0

    print(f"\nStiffness Audit Results:")
    print(f"  Max Spectral Radius (rho): {max_rho:.4f} /day")
    print(f"  Worst-case state: B={worst_y[0]:.3f}, F={worst_y[1]:.3f}, A={worst_y[2]:.3f}")
    print(f"  Worst-case control: Phi={worst_phi:.2f}")
    print(f"  Maximum stable step size (h_max): {h_max_mins:.2f} minutes")

    THRESHOLD_MINS = 15.0
    if h_max_mins >= THRESHOLD_MINS:
        print(f"  PASS: Model is stable for {THRESHOLD_MINS:.0f} min steps.")
        passed = True
        n_substeps = 1
    else:
        n_substeps = int(np.ceil(THRESHOLD_MINS / h_max_mins))
        print(f"  WARNING: Model is STIFF — needs at least {n_substeps} sub-steps "
              f"for {THRESHOLD_MINS:.0f} min bins.")
        passed = False

    return {
        'h_max_mins':              float(h_max_mins),
        'max_rho_per_day':         max_rho,
        'n_substeps_for_15min':    n_substeps,
        'worst_case_state':        {'B': float(worst_y[0]),
                                     'F': float(worst_y[1]),
                                     'A': float(worst_y[2]),
                                     'Phi': worst_phi},
        'passed':                  passed,
    }


if __name__ == "__main__":
    audit_stiffness()
