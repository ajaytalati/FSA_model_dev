# FSA_model_dev

Standalone sandbox for the FSA-v2 (Banister-coupled) physiological SDE model.

## Structure

- `models/fsa_high_res/`: The core model implementation (drift, diffusion, priors, cost functions).
- `simulator/`: A generic, JAX-native SDE simulation framework.
- `smc2fc/`: Lightweight stubs to allow the model to be developed independently of the main SMC² repository.
- `tests/`: Sanity tests for physics and SMC² artifact compatibility.
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

## Porting back to SMC2 Repo

The files in `models/fsa_high_res/` are designed to be dropped directly into the `version_2/models/fsa_high_res/` directory of the `python-smc2-filtering-control` repository once validated here.
