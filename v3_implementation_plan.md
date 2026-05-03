
# FSA-v3 Implementation Plan: Bimodal Training Extension

This document outlines the step-by-step implementation of the 4D FSA-v3 model, introducing Strength/Resistance Adaptation ($S$) and concurrent training support.

## Phase 1: Core Dynamics & Simulation
**Goal:** Extend the latent state space to 4D and update the physics engine.

1.  **Update `models/fsa_high_res/_dynamics.py`:**
    *   Expand `TRUTH_PARAMS` with strength-specific parameters ($\tau_S, \kappa_S, \kappa_{F,S}, \mu_S, \epsilon_{A,S}, \sigma_S$).
    *   Update `drift_jax` to handle the 4D state vector $[B, S, F, A]$ and 2D control input $[\Phi_B, \Phi_S]$.
    *   Implement the $dS$ equation (Jacobi diffusion).
    *   Update $dF$ to a unified pool: $\kappa_{F,B}\Phi_B + \kappa_{F,S}\Phi_S$.
    *   Update $dA$ (Stuart-Landau) to include $\mu_S S$ in the bifurcation parameter $\mu$.
    *   Update `diffusion_state_dep` and `imex_step_substepped` for 4D.

2.  **Update `models/fsa_high_res/simulation.py`:**
    *   Update `HIGH_RES_FSA_V2_MODEL` to `V3_MODEL`.
    *   Add `StateSpec` for "S".
    *   Update `drift`, `drift_jax`, `diffusion_diagonal`, `noise_scale_fn`, `make_aux`, `make_y0` to 4D.
    *   **New Channel:** Add `ChannelSpec` for "VolumeLoad" ($VL \sim \mathcal{N}(\beta_S S - \beta_F F, \sigma_{VL}^2)$).
    *   Update `verify_physics` to include $S$ checks.

3.  **Update `models/fsa_high_res/_phi_burst.py`:**
    *   Ensure stimulus expansion can handle dual inputs (Aerobic $\Phi_B$ and Strength $\Phi_S$).

## Phase 2: Estimation & Identifiability
**Goal:** Update the SMC² artifacts to support the 4D state and new observation channel.

1.  **Update `models/fsa_high_res/estimation.py`:**
    *   Expand `PARAM_PRIOR_CONFIG` with new parameters.
    *   Update `propagate_fn` (Guided Proposal):
        *   Extend the Kalman fusion to 4 channels (adding Volume Load).
        *   Update the state transition matrices ($F, H$) and covariance ($Q, R$).
    *   Update `obs_log_weight_fn` and `_gaussian_obs_ll` to include Volume Load likelihood.
    *   Update `align_obs_fn` to ingest "VolumeLoad" data and "Phi_S" stimulus.
    *   Update `forward_sde_stochastic` and `imex_step_fn` for 4D.

## Phase 3: Control & Optimization
**Goal:** Update the MPC to optimize 2D training schedules.

1.  **Update `models/fsa_high_res/control.py`:**
    *   Update the RBF schedule generator to output 2D control arrays.
    *   Update the cost function $J(\theta)$ to reward $S$ accrual and handle bimodal constraints.
    *   (Optional) Implement Risk-Sensitive MPC (CVaR) to handle early-stage uncertainty in $S$.

## Phase 4: Validation & Testing
**Goal:** Verify physics, identifiability, and control stability.

1.  **Update `tests/test_fsa_physics.py`:**
    *   Add 4D smoke tests.
    *   Test stimulus isolation (Aerobic-only vs. Strength-only).
2.  **New Test `tests/test_v3_identifiability.py`:**
    *   Verify FIM rank for the 4D system.
    *   Test parameter recovery with sparse Volume Load data.

## Phase 5: Documentation
1.  **Update LaTeX:** Add Section 07 for FSA-v3 Bimodal Dynamics.
2.  **Update `GEMINI.md`:** Document the 4D architecture and new control interface.
