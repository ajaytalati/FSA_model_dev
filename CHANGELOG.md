# Changelog — FSA Model Dev Sandbox

All notable changes to this dev sandbox are recorded here. Newest entries at the top.

Each entry lists the local time it was made, the author, and the change with the reason behind it. The author is the person who decided / signed off the change; an "(assistant)" line names the tool that did the typing where relevant.

Dates are in `YYYY-MM-DD`; times are local (BST unless otherwise noted).

---

## 2026-05-04

### 14:01 BST — Phase 2 (tests): added obs-consistency + reconciliation tests; 10/10 green
- **Author:** Ajay Talati (per the approved plan)
- **Assistant:** Claude Code (Opus 4.7, 1M context)
- **Branch:** `claude/dev-sandbox-main`
- **Files added:**
  - [tests/test_obs_consistency.py](tests/test_obs_consistency.py) — 5 pytest tests asserting that the simulator's noise-free predicted mean (or, for sleep, Bernoulli probability) matches the estimator's likelihood prediction at the same `(B, F, A)` state, params, and circadian C(t). One test per Gaussian channel (HR, stress, steps), one for the Bernoulli sleep marginal (verified both in closed form against the estimator and via a 30 000-bin empirical frequency on the sim side), plus a belt-and-braces sweep over `HR_base` to catch any future regression of an FSA equivalent of SWAT's D1 bug.
  - [tests/test_reconciliation.py](tests/test_reconciliation.py) — 2 pytest tests:
    - `test_plant_and_estimator_share_drift` — single-step Euler prediction must be bit-equivalent on both sides (plant uses `_dynamics.drift_jax`; estimator inlines its own copy in `propagate_fn`). Diff < 1e-10.
    - `test_plant_advance_smoke` — drives `StepwisePlant.advance(1 bin)` end-to-end and confirms state stays in physical bounds (B ∈ [0,1], F ≥ 0, A ≥ 0) and the output dict contains all 4 obs channels + Phi + C.
- **Why:** These two test files are the structural protection layers identified in the parent CLAUDE.md's MPC-dependency-chain note. Without them, sim ↔ estimator divergence (the SWAT D1/D2 failure mode) and plant ↔ estimator drift divergence (the silent-MPC-corruption failure mode) both go undetected. Now any future change that breaks either invariant fails one of these tests with a clear message naming what's wrong.
- **Verified by:**
  - `cd /tmp && pytest /home/ajay/Repos/FSA_model_dev/tests/ -v` (no PYTHONPATH, not in repo dir) → **10/10 passed in 4.05 s**:
    - 3 pre-existing (`test_artifacts.py` × 2, `test_fsa_physics.py` × 1) — unchanged, still green.
    - 5 obs-consistency tests — all pass cleanly.
    - 2 reconciliation tests — drift parity to < 1e-10, plant smoke OK.
  - The sim/est formulas in FSA's current code are properly aligned (no D1/D2-equivalent asymmetries). Both sides implement HR / stress / steps / sleep using identical formulas referencing `(B, F, A, C)`; the test now pins this to LaTeX explicitly.

### 13:46 BST — Phase 1 complete: pyproject + LICENSE + .gitignore + CHANGELOG; bundled smc2fc/simulator stubs deleted; imports rewired to real smc2fc
- **Author:** Ajay Talati (per the approved plan at `claude_plans/FSA_dev_sandbox_*_2026-05-04_1346.md`)
- **Assistant:** Claude Code (Opus 4.7, 1M context)
- **Branch:** `claude/dev-sandbox-main` (branched from `main` at `fbedb98`); the existing `main`, `v3-bimodal-extension`, and `v4-bimodal-variable-dose-extension` branches are NOT touched.
- **Files added:**
  - [pyproject.toml](pyproject.toml) — declares `smc2fc` (pulled from `github.com/ajaytalati/python-smc2-filtering-control@master`) + `matplotlib>=3.8` as deps. Flat 4-package layout (`models`, `tools`, `tests`, `scenarios`). Pytest config under `[tool.pytest.ini_options]`.
  - [LICENSE](LICENSE) — MIT, attributed to Ajay Talati.
  - [CHANGELOG.md](CHANGELOG.md) — this file.
  - [claude_plans/FSA_dev_sandbox_..._2026-05-04_1346.md](claude_plans/) — archived copy of the implementation plan per the global CLAUDE.md rule.
- **Files updated:**
  - [.gitignore](.gitignore) — extended the existing 11-line gitignore to cover `*.egg-info/`, build/dist artefacts, `.mypy_cache/`, `.ruff_cache/`, `.cache/`, venv folders, JAX compilation cache, regenerable `outputs/` folder, additional LaTeX intermediates (`*.synctex.gz`, `*.fls`, `*.fdb_latexmk`, `auto/`), and editor / OS noise. The original entries (`__pycache__`, `*.pyc`, `.pytest_cache/`, `LaTex_docs/*.{aux,log,out,toc,pdf,b64}`, `fsa_simulation.png`) are kept.
  - [models/fsa_high_res/simulation.py:37](models/fsa_high_res/simulation.py#L37) — `from simulator.sde_model import ...` → `from smc2fc.simulator.sde_model import ...`. Same change `version_2/models/fsa_high_res/simulation.py` already had.
  - [tests/test_fsa_physics.py:9](tests/test_fsa_physics.py#L9) — `from simulator.sde_solver_diffrax` → `from smc2fc.simulator.sde_solver_diffrax`.
  - [examples/run_fsa_simulation.py:9](examples/run_fsa_simulation.py#L9) — same import-path update.
- **Files deleted (`git rm -r`):**
  - `simulator/` — 3 .py files (`sde_model.py`, `sde_observations.py`, `sde_solver_diffrax.py`). Verified that the corresponding files inside the real `smc2fc.simulator` package expose the exact same public API (`sde_model.py` is byte-identical; `sde_observations.py` and `sde_solver_diffrax.py` differ only by their internal `from simulator.sde_model` → `from smc2fc.simulator.sde_model` import-path lines, no functional difference).
  - `smc2fc/` — bundled stub package (4 files: `__init__.py`, `_likelihood_constants.py`, `estimation_model.py`, plus a `control/` subfolder with `__init__.py`, `calibration.py`, `control_spec.py`, `rbf_schedules.py`). Stubs were a ~144-line subset of the real `smc2fc` package; FSA's usage is forward-compatible with the real package's signatures.
- **Why:** The dev repo previously had no installable metadata, and the bundled `smc2fc/` and `simulator/` folders were duplicate copies of the real `smc2fc` package's submodules — destined to drift over time. `pip install -e .` now pulls real `smc2fc` from github, single source of truth, no vendored copies.
- **Verified by:**
  - `pip install -e ".[test]"` from inside the dev repo → pulled `smc2fc` from github, installed `fsa-model-dev-0.2.0`. No errors.
  - `cd /tmp && pytest /home/ajay/Repos/FSA_model_dev/tests/ -v` → **3/3 passed in 2.28 s** (`test_artifacts.py::test_estimation_artifact`, `test_artifacts.py::test_control_artifact`, `test_fsa_physics.py::test_simulation_run`). All resolved against real smc2fc.
  - Existing `models/fsa_high_res/control.py` and `estimation.py` were untouched — they already used absolute `from smc2fc.{control,estimation_model,_likelihood_constants}` imports that resolved against the bundled stubs and now resolve against the real package.

---

## How to add a new entry

1. New entries go at the **top** of today's date section. If today's date isn't there yet, add it as a new `## YYYY-MM-DD` heading at the top.
2. Format:
   ```
   ### HH:MM BST — Short title in active voice
   - **Author:** Person who signed off the change
   - **Assistant:** Tool that helped, if any (e.g. "Claude Code (Opus 4.7)")
   - **Branch:** Which working branch this happened on
   - **Files added / updated / deleted:** Bullet list with [markdown links](path)
   - **Why:** Reason behind the change.
   - **Verified by:** What was run to confirm the change is safe (tests, export pipeline, etc.)
   ```
3. Keep it readable. The audience is a future engineer (possibly future-you) trying to understand why something is the way it is.
