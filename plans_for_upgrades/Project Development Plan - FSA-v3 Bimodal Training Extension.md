
# Project Development Plan: FSA-v3 Bimodal Training Extension

## 1. Executive Summary

This document outlines the architectural and mathematical extension of the FSA-v2 physiological model. The current model encapsulates Aerobic Fitness ($B$), Fatigue ($F$), and Autonomic Amplitude ($A$) responding to a single aerobic stimulus $\Phi(t)$.

The FSA-v3 extension introduces **Strength/Resistance Adaptation (**$S$**)** as a distinct state variable driven by a separate stimulus $\Phi_S(t)$. This upgrades the system to a 4D Stochastic Differential Equation (SDE) model capable of representing concurrent training paradigms, while maintaining a unified fatigue pool and autonomic interaction.

## 2. Mathematical Formulation (4D State Space)

The state vector expands to $y(t) = (B(t), S(t), F(t), A(t))^T$.

### 2.1 Control Inputs

The system now accepts a 2D control vector representing different physiological stressors:

- $\Phi_B(t)$: Aerobic/Endurance training load.
    
- $\Phi_S(t)$: Strength/Resistance training load.
    

### 2.2 The SDE System

The updated system of Itô SDEs is defined as follows:

**1. Aerobic Fitness (**$B$**)** - _Unchanged, but stimulus renamed_

$$dB = \left[\kappa_B \cdot \frac{1 + \epsilon_{A,B} A}{1 + \epsilon_{A,B} A_{typ}} \Phi_B(t) - \frac{B}{\tau_B}\right] dt + \sigma_B \sqrt{B(1-B)} dW_B$$

**2. Strength/Resistance Adaptation (**$S$**)** - _New Equation_

Assumes a Jacobi diffusion similar to $B$, constraining strength capacity to a normalized $[0, 1]$ interval.

$$dS = \left[\kappa_S \cdot \frac{1 + \epsilon_{A,S} A}{1 + \epsilon_{A,S} A_{typ}} \Phi_S(t) - \frac{S}{\tau_S}\right] dt + \sigma_S \sqrt{S(1-S)} dW_S$$

**3. Unified Fatigue (**$F$**)** - _Modified_

Both training modalities contribute to a single systemic fatigue pool, but with different gain coefficients reflecting their unique central/peripheral neurological tax.

$$dF = \left[\kappa_{F,B} \Phi_B(t) + \kappa_{F,S} \Phi_S(t) - \frac{1 + \lambda_A A}{1 + \lambda_A A_{typ}} \frac{F}{\tau_F}\right] dt + \sigma_F \sqrt{F} dW_F$$

**4. Autonomic Amplitude (**$A$**)** - _Modified_

The Stuart-Landau oscillator is maintained, but the bifurcation parameter $\mu$ is extended so that strength adaptation ($S$) also provides a positive chronotropic effect on autonomic robustness.

$$dA = [\mu(B,S,F)A - \eta A^3] dt + \sigma_A \sqrt{A} dW_A$$$$\mu(B,S,F) = \mu_0 + \mu_B B + \mu_S S - \mu_F F - \mu_{FF}(F - F_{typ})^2$$

## 3. Parameter Expansion & Identifiability

To support the $S$ compartment, the parameter space will increase. The G1-reparametrization philosophy must be maintained for structural identifiability.

### 3.1 New Dynamics Parameters

|   |   |   |
|---|---|---|
|**Symbol**|**Meaning**|**Expected Range/Behavior**|
|$\tau_S$|Strength chronic time constant|Likely longer than $\tau_B$ (neuromuscular adaptations decay slower than cardiovascular).|
|$\kappa_S$|Effective Strength gain|Modality specific gain at $A_{typ}$.|
|$\kappa_{F,S}$|Fatigue gain from strength|Expected to be high per unit of time compared to $\kappa_{F,B}$.|
|$\epsilon_{A,S}$|Residual A-boost on strength|May be different from $\epsilon_{A,B}$ (e.g., strength gains might be less sensitive to autonomic suppression than aerobic gains).|
|$\mu_S$|Strength effect on growth rate|Adds stability to the autonomic oscillator.|
|$\sigma_S$|Jacobi noise scale for S|Similar magnitude to $\sigma_B$.|

### 3.2 Observation Model Adjustments

To ensure the new parameters are identifiable, the SMC² observation model (`obs_log_weight_fn`) needs a channel that discriminates $S$ from $B$.

- **Primary Observation Channel (Volume Load):** Daily neuromuscular proxy channels (like grip strength or jump height) are often too noisy and impractical for regular collection. Instead, the model will utilize **Volume Load** (Sets $\times$ Reps $\times$ Load) logged during strength training sessions.
    
- **Observation Equation:** $VolumeLoad(t) \sim \mathcal{N}(\beta_S S - \beta_F F, \sigma_{VL}^2)$  
    
- **Identifiability and Sparse Sampling:** Strength capacity ($S$) is a slow-moving state variable driven by structural and neuromuscular adaptations. Its chronic time constant ($\tau_S$) is significantly longer than aerobic fitness ($\tau_B$). Therefore:
    
    - **Sampling Frequency:** High-frequency daily data is not required. Sparse sampling (e.g., logging Volume Load every other day) is perfectly sufficient to track the slow-moving $S$ trajectory.
        
    - **Data Horizon:** The model will likely require 60 to 90 days (2 to 3 months) of bi-daily Volume Load data for $S$ to become structurally and practically identifiable.
        
    - **Unified Fatigue Advantage:** Because the acute, fast-moving fatigue pool ($F$) is unified and continuously identified by daily aerobic channels (HR, sleep, stress), the algorithm can effectively decouple acute fatigue from chronic strength adaptation even with sparse Volume Load observations.
        

## 4. JAX/Diffrax Implementation Roadmap

### Phase 1: Core Dynamics (`models/fsa_high_res/_dynamics.py`)

1. **State Vector Refactoring:** Update named tuples or array indices to expect `shape=(4,)` instead of `(3,)`.
    
2. **SDE Drift Function:** Implement the new $dS$ equation and update the $dF$ and $dA$ drift calculations to ingest a `phi_control` array of `shape=(2,)`.
    
3. **SDE Diffusion Function:** Add the $\sigma_S \sqrt{S(1-S)}$ term to the diagonal diffusion matrix.
    

### Phase 2: Simulation & Physics Validation (`tests/test_fsa_physics.py`)

1. **Boundary Checks:** Verify that $S(t) \in [0, 1]$ under extreme stochastic noise.
    
2. **Stimulus Isolation Tests:**
    
    - _Test A:_ Apply only $\Phi_B(t)$. Verify $B$ and $F$ rise, $S$ remains at baseline, $A$ behaves as v2.
        
    - _Test B:_ Apply only $\Phi_S(t)$. Verify $S$ and $F$ rise, $B$ remains at baseline.
        

### Phase 3: SMC² Artifacts (`models/fsa_high_res/estimation.py`)

1. **Prior Definitions:** Add priors for $\tau_S, \kappa_S, \kappa_{F,S}, \mu_S, \epsilon_{A,S}, \sigma_S$ to `PARAM_PRIOR_CONFIG`.
    
2. **Observation Synthesis:** Update the log-likelihood function to map the 4D state to the observation channels.
    

### Phase 4: Control & Optimization (`models/fsa_high_res/control.py`)

1. **2D Control Schedule:** Update the RBF schedule basis to output two distinct schedules over the planning horizon.
    
2. **Cost Function** $J(\theta)$**:** Update the reward mechanisms. For example, penalizing concurrent training interference (if modeled) or setting target endpoints for both $B$ and $S$ independently while keeping $F$ below a threshold.
    

## 5. Potential Future Considerations (v3.1)

- **Interference Effect:** The current formulation assumes $B$ and $S$ accrue independently. If physiological "concurrent training interference" needs to be modeled, a negative interaction term could be added (e.g., high $B$ suppresses $\kappa_S$, or high $S$ suppresses $\kappa_B$).