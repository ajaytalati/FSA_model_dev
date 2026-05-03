# FSA_model_dev

Standalone sandbox for development and validation of the Functional Sub-unit Assembly (FSA) physiological SDE models.

## Model Version Roadmap & Branch Guide

This repository contains multiple generations of the FSA model. Each version is maintained in its own dedicated branch for research continuity.

| Version | Branch | State Space | Stimuli | Key Features |
| :--- | :--- | :--- | :--- | :--- |
| **FSA-v2** | `main` | 3D [B, F, A] | 1D [Phi] | G1-reparametrized Banister-coupled model. Stable baseline. |
| **FSA-v3** | `v3-bimodal-extension` | 4D [B, S, F, A] | 2D [Phi_B, Phi_S] | Bimodal training (Aerobic + Strength). Concurrent training cost functional ($A+B+S$). |
| **FSA-v4** | `v4-bimodal-variable-dose-extension` | 6D [B, S, F, A, KFB, KFS] | 2D [Phi_B, Phi_S] | **Variable-Dose (Busso 2003).** Dynamic fatigue sensitivities. Emergent deload periodization. |

---

## Current Version: FSA-v4 (Variable-Dose Extension)

- **6D State Space:** Models Fitness ($B$), Strength ($S$), Fatigue ($F$), Autonomic Amplitude ($A$), and **Dynamic Fatigue Sensitivities** ($K_{FB}, K_{FS}$).
- **Variable-Dose Dynamics:** Implements Busso (2003) principles where training increases future fatigue sensitivity, requiring explicit **deload phases**.
- **Bimodal Training:** Full concurrent support for Aerobic ($\Phi_B$) and Strength ($\Phi_S$) stimuli.
- **Pure JAX Dynamics:** Highly-optimised 6D SDE drift and diffusion implementation.
- **SMC² Artifacts:** Includes 6D-compatible `EstimationModel` and 2D-control `ControlSpec`.
- **Academic Documentation:** 18-page graduate-level lecture notes covering all versions from v2 to v4.
- **Stability & Control:** Built-in verification for the **Goldilocks Surface** and **Pareto Optimality** in concurrent training.

## Structure

- `models/fsa_high_res/`: The core model implementation (drift, diffusion, priors, cost functions).
- `simulator/`: A generic, JAX-native SDE simulation framework.
- `smc2fc/`: Lightweight stubs to allow the model to be developed independently.
- `LaTex_docs/`: Source for graduate lecture notes.
- `tests/`: Sanity tests for physics, identifiability, and artifact compatibility.
- `examples/`: Example scripts showing how to run simulations.

## Usage

1. Activate your environment (e.g., `conda activate comfyenv`).
2. Run a simulation example:
   ```bash
   PYTHONPATH=. python examples/run_fsa_simulation.py
   ```
3. Run tests:
   ```bash
   PYTHONPATH=. pytest tests/
   ```

## Documentation

The graduate lecture notes can be found in `LaTex_docs/main.pdf`. To recompile:
```bash
cd LaTex_docs
pdflatex main.tex
```

## Porting back to SMC2 Repo

The files in `models/fsa_high_res/` are designed to be dropped directly into the `version_2/models/fsa_high_res/` directory of the `python-smc2-filtering-control` repository once validated here.
