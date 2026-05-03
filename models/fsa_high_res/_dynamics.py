"""Pure-JAX dynamics for FSA-v3 (Bimodal Training Extension) — **G1-reparametrized**.

State [B, S, F, A]:
  B  aerobic fitness (Banister chronic, Jacobi diffusion in [0, 1])
  S  strength        (Banister chronic, Jacobi diffusion in [0, 1])
  F  fatigue         (Unified pool,     CIR diffusion in [0, ∞))
  A  amplitude       (Stuart-Landau,    CIR diffusion in [0, ∞))

## Reparametrization (Stage G1)

Maintains the G1 rotation from v2 for aerobic and fatigue parameters,
extending it to the strength compartment.

| param name   | NEW meaning                            |
|--------------|----------------------------------------|
| `kappa_B`    | κ_B^eff = κ_B·(1 + ε_{A,B}·A_typ)       |
| `kappa_S`    | κ_S^eff = κ_S·(1 + ε_{A,S}·A_typ)       |
| `tau_F`      | τ_F^eff = τ_F/(1 + λ_A·A_typ)          |
| `mu_F`       | slope at F_typ                         |

## Drift (4D, /day units)

  μ(B, S, F) = μ_0 + μ_B·B + μ_S·S − μ_F·F − μ_{FF}·(F − F_typ)²

  dB/dt = κ_B · (1 + ε_{A,B}·A) / (1 + ε_{A,B}·A_typ) · Φ_B(t) − B/τ_B
  dS/dt = κ_S · (1 + ε_{A,S}·A) / (1 + ε_{A,S}·A_typ) · Φ_S(t) − S/τ_S
  dF/dt = (κ_{F,B}·Φ_B(t) + κ_{F,S}·Φ_S(t)) − (1 + λ_A·A) / (1 + λ_A·A_typ) / τ_F · F
  dA/dt = μ·A − η·A³

## Diffusion (state-dependent Itô)

  σ_B · √(B (1 − B)) · dW_B   (Jacobi)
  σ_S · √(S (1 − S)) · dW_S   (Jacobi)
  σ_F · √F           · dW_F   (CIR)
  σ_A · √A           · dW_A   (CIR)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


# ── Operating-point reference constants ───────────────────────────────
A_TYP   = 0.10
F_TYP   = 0.20
PHI_TYP = 1.0


# ── Truth parameters (FSA-v3, G1-reparametrized) ──────────────────────

TRUTH_PARAMS = dict(
    # Aerobic Fitness (B)
    tau_B=42.0,
    kappa_B=0.012 * (1.0 + 0.40 * A_TYP),  # κ_B^eff
    epsilon_AB=0.40,                        # ε_{A,B}

    # Strength Adaptation (S)
    tau_S=60.0,
    kappa_S=0.008 * (1.0 + 0.20 * A_TYP),  # κ_S^eff
    epsilon_AS=0.20,                        # ε_{A,S}

    # Unified Fatigue (F)
    tau_F=7.0 / (1.0 + 1.00 * A_TYP),     # τ_F^eff
    kappa_FB=0.030,                         # κ_{F,B}
    kappa_FS=0.050,                         # κ_{F,S} (higher fatigue per unit stimulus)
    lambda_A=1.00,

    # Stuart-Landau bifurcation parameter
    mu_0=0.02 + 0.40 * (F_TYP ** 2),       # μ_0^eff
    mu_B=0.30,
    mu_S=0.15,                              # S contributes to autonomic robustness
    mu_F=0.10 + 2.0 * F_TYP * 0.40,        # μ_F^eff
    mu_FF=0.40,
    eta=0.20,

    # State-dependent diffusion
    sigma_B=0.010,
    sigma_S=0.008,
    sigma_F=0.012,
    sigma_A=0.020,
)


def drift_jax(y, params, Phi_t):
    """JAX drift for the FSA-v3 SDE.

    Args:
        y: state vector [B, S, F, A] of shape (4,).
        params: dict with keys from TRUTH_PARAMS.
        Phi_t: vector [Phi_B, Phi_S] — stimulus rates.

    Returns:
        d[B, S, F, A]/dt of shape (4,).
    """
    B = y[0]
    S = y[1]
    F = y[2]
    A = y[3]

    Phi_B = Phi_t[0]
    Phi_S = Phi_t[1]

    F_dev = F - F_TYP
    mu = (params['mu_0']
          + params['mu_B'] * B
          + params['mu_S'] * S
          - params['mu_F'] * F
          - params['mu_FF'] * F_dev * F_dev)

    # B equation
    a_factor_B = ((1.0 + params['epsilon_AB'] * A)
                   / (1.0 + params['epsilon_AB'] * A_TYP))
    dB = params['kappa_B'] * a_factor_B * Phi_B - B / params['tau_B']

    # S equation
    a_factor_S = ((1.0 + params['epsilon_AS'] * A)
                   / (1.0 + params['epsilon_AS'] * A_TYP))
    dS = params['kappa_S'] * a_factor_S * Phi_S - S / params['tau_S']

    # F equation (Unified Fatigue)
    a_factor_F = ((1.0 + params['lambda_A'] * A)
                   / (1.0 + params['lambda_A'] * A_TYP))
    dF = (params['kappa_FB'] * Phi_B 
          + params['kappa_FS'] * Phi_S
          - a_factor_F / params['tau_F'] * F)

    dA = mu * A - params['eta'] * A * A * A

    return jnp.array([dB, dS, dF, dA])


def diffusion_state_dep(y, params):
    """State-dependent diagonal diffusion vector (4D)."""
    B = y[0]
    S = y[1]
    F = y[2]
    A = y[3]
    return jnp.array([
        params['sigma_B'] * jnp.sqrt(jnp.maximum(B * (1.0 - B), 0.0)),
        params['sigma_S'] * jnp.sqrt(jnp.maximum(S * (1.0 - S), 0.0)),
        params['sigma_F'] * jnp.sqrt(jnp.maximum(F, 0.0)),
        params['sigma_A'] * jnp.sqrt(jnp.maximum(A, 0.0)),
    ])


def imex_step_substepped(y, params, noise, Phi_t, dt, n_substeps: int = 4):
    """Substepped Euler-Maruyama with state-dependent diffusion (4D)."""
    sub_dt = dt / float(n_substeps)

    def sub_body(y_inner, _):
        return y_inner + sub_dt * drift_jax(y_inner, params, Phi_t), None

    y_det, _ = jax.lax.scan(sub_body, y, jnp.arange(n_substeps))

    sigma_y = diffusion_state_dep(y_det, params)
    y_pred = y_det + sigma_y * jnp.sqrt(dt) * noise

    # Reflection boundaries
    B_pred, S_pred, F_pred, A_pred = y_pred[0], y_pred[1], y_pred[2], y_pred[3]
    
    B_next = jnp.where(B_pred < 0.0, -B_pred,
                        jnp.where(B_pred > 1.0, 2.0 - B_pred, B_pred))
    S_next = jnp.where(S_pred < 0.0, -S_pred,
                        jnp.where(S_pred > 1.0, 2.0 - S_pred, S_pred))
    F_next = jnp.abs(F_pred)
    A_next = jnp.abs(A_pred)

    return jnp.array([B_next, S_next, F_next, A_next])
