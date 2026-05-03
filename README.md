# FSA_model_dev

Standalone sandbox for the FSA-v3 (Bimodal Training Extension) physiological SDE model.

## Features

- **4D State Space:** Models Fitness ($B$), Strength ($S$), Fatigue ($F$), and Autonomic Amplitude ($A$).
- **Bimodal Training:** Concurrent support for Aerobic ($\Phi_B$) and Strength ($\Phi_S$) stimuli.
- **Unified Fatigue Pool:** A single systemic fatigue state driven by both modalities.
- **Pure JAX Dynamics:** Highly-optimised SDE drift and diffusion implementation.
- **SMC² Artifacts:** Includes 4D-compatible `EstimationModel` and 2D-control `ControlSpec`.
- **Academic Documentation:** Graduate-level lecture notes on the model's biology and math, now including FSA-v3.
- **Identifiability & Stability:** Built-in verification for FIM rank and Lyapunov stability for the 4D system.

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
