"""Pure-JAX dynamics for FSA-v4 (Variable-Dose Extension) — **G1-reparametrized**.

State [B, S, F, A, KFB, KFS]:
  B   aerobic fitness      (Banister chronic, Jacobi diffusion in [0, 1])
  S   strength             (Banister chronic, Jacobi diffusion in [0, 1])
  F   fatigue              (Unified pool,     CIR diffusion in [0, ∞))
  A   amplitude            (Stuart-Landau,    CIR diffusion in [0, ∞))
  KFB fatigue gain aerobic (Busso variable,   CIR diffusion in [0, ∞))
  KFS fatigue gain strength(Busso variable,   CIR diffusion in [0, ∞))

## Busso Variable-Dose Principle (Busso 2003)

The fatigue gains KFB and KFS are no longer static parameters but latent states.
Training stimulus concurrently builds fitness/strength and increases the 
fatigue sensitivity of the athlete.

  dK_{Fi}/dt = (K_{Fi}^0 - K_{Fi}) / tau_K + mu_K * Phi_i(t)

where K_{Fi}^0 is the baseline sensitivity and mu_K is the 'damage' rate.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


# ── Operating-point reference constants ───────────────────────────────
A_TYP   = 0.10
F_TYP   = 0.20
PHI_TYP = 1.0


# ── Truth parameters (FSA-v4, G1-reparametrized) ──────────────────────

TRUTH_PARAMS = dict(
    # Aerobic Fitness (B)
    tau_B=42.0,
    kappa_B=0.012 * (1.0 + 0.40 * A_TYP),  # κ_B^eff
    epsilon_AB=0.40,

    # Strength Adaptation (S)
    tau_S=60.0,
    kappa_S=0.008 * (1.0 + 0.20 * A_TYP),  # κ_S^eff
    epsilon_AS=0.20,

    # Unified Fatigue (F)
    tau_F=7.0 / (1.0 + 1.00 * A_TYP),     # τ_F^eff
    lambda_A=1.00,

    # Busso Variable-Dose (Fatigue Sensitivity K)
    KFB_0=0.030,                            # Baseline aerobic fatigue gain
    KFS_0=0.050,                            # Baseline strength fatigue gain
    tau_K=21.0,                             # Sensitivity recovery timescale (~3 weeks)
    mu_K=0.005,                             # Sensitivity 'damage' rate (Busso 2003)

    # Stuart-Landau bifurcation parameter
    mu_0=0.02 + 0.40 * (F_TYP ** 2),
    mu_B=0.30,
    mu_S=0.15,
    mu_F=0.10 + 2.0 * F_TYP * 0.40,
    mu_FF=0.40,
    eta=0.20,

    # State-dependent diffusion
    sigma_B=0.010,
    sigma_S=0.008,
    sigma_F=0.012,
    sigma_A=0.020,
    sigma_K=0.005,                          # Diffusion for sensitivity states
)


def drift_jax(y, params, Phi_t):
    """JAX drift for the FSA-v4 SDE.

    Args:
        y: state vector [B, S, F, A, KFB, KFS] of shape (6,).
        params: dict with keys from TRUTH_PARAMS.
        Phi_t: vector [Phi_B, Phi_S] — stimulus rates.

    Returns:
        d[B, S, F, A, KFB, KFS]/dt of shape (6,).
    """
    B, S, F, A, KFB, KFS = y[0], y[1], y[2], y[3], y[4], y[5]
    Phi_B, Phi_S = Phi_t[0], Phi_t[1]

    F_dev = F - F_TYP
    mu = (params['mu_0']
          + params['mu_B'] * B
          + params['mu_S'] * S
          - params['mu_F'] * F
          - params['mu_FF'] * F_dev * F_dev)

    # Capacity equations
    a_factor_B = (1.0 + params['epsilon_AB'] * A) / (1.0 + params['epsilon_AB'] * A_TYP)
    dB = params['kappa_B'] * a_factor_B * Phi_B - B / params['tau_B']

    a_factor_S = (1.0 + params['epsilon_AS'] * A) / (1.0 + params['epsilon_AS'] * A_TYP)
    dS = params['kappa_S'] * a_factor_S * Phi_S - S / params['tau_S']

    # Unified Fatigue Pool (using dynamic gains KFB, KFS)
    a_factor_F = (1.0 + params['lambda_A'] * A) / (1.0 + params['lambda_A'] * A_TYP)
    dF = (KFB * Phi_B + KFS * Phi_S - a_factor_F / params['tau_F'] * F)

    # Autonomic Amplitude
    dA = mu * A - params['eta'] * A * A * A

    # Busso Variable-Dose equations
    dKFB = (params['KFB_0'] - KFB) / params['tau_K'] + params['mu_K'] * Phi_B
    dKFS = (params['KFS_0'] - KFS) / params['tau_K'] + params['mu_K'] * Phi_S

    return jnp.array([dB, dS, dF, dA, dKFB, dKFS])


def diffusion_state_dep(y, params):
    """State-dependent diagonal diffusion vector (6D)."""
    B, S, F, A, KFB, KFS = y[0], y[1], y[2], y[3], y[4], y[5]
    return jnp.array([
        params['sigma_B'] * jnp.sqrt(jnp.maximum(B * (1.0 - B), 0.0)),
        params['sigma_S'] * jnp.sqrt(jnp.maximum(S * (1.0 - S), 0.0)),
        params['sigma_F'] * jnp.sqrt(jnp.maximum(F, 0.0)),
        params['sigma_A'] * jnp.sqrt(jnp.maximum(A, 0.0)),
        params['sigma_K'] * jnp.sqrt(jnp.maximum(KFB, 0.0)),
        params['sigma_K'] * jnp.sqrt(jnp.maximum(KFS, 0.0)),
    ])


def imex_step_substepped(y, params, noise, Phi_t, dt, n_substeps: int = 4):
    """Substepped Euler-Maruyama with state-dependent diffusion (6D)."""
    sub_dt = dt / float(n_substeps)

    def sub_body(y_inner, _):
        return y_inner + sub_dt * drift_jax(y_inner, params, Phi_t), None

    y_det, _ = jax.lax.scan(sub_body, y, jnp.arange(n_substeps))

    sigma_y = diffusion_state_dep(y_det, params)
    y_pred = y_det + sigma_y * jnp.sqrt(dt) * noise

    # Reflection boundaries
    B_p, S_p, F_p, A_p, KFB_p, KFS_p = y_pred[0], y_pred[1], y_pred[2], y_pred[3], y_pred[4], y_pred[5]
    
    B_next = jnp.where(B_p < 0.0, -B_p, jnp.where(B_p > 1.0, 2.0 - B_p, B_p))
    S_next = jnp.where(S_p < 0.0, -S_p, jnp.where(S_p > 1.0, 2.0 - S_p, S_p))
    F_next = jnp.abs(F_p)
    A_next = jnp.abs(A_p)
    KFB_next = jnp.abs(KFB_p)
    KFS_next = jnp.abs(KFS_p)

    return jnp.array([B_next, S_next, F_next, A_next, KFB_next, KFS_next])
