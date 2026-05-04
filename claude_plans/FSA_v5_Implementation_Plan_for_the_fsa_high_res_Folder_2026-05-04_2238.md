# FSA-v5 Implementation Plan for the `fsa_high_res/` Folder

> Archived from plan mode: 2026-05-04 22:38.

## Context

The user is preparing to import the FSA model into the separate `smc2fc` repository. The integration boundary is the directory `models/fsa_high_res/`. The v5 changes (Hill-function deconditioning, sedentary collapse, principled cost functional) are documented in §10–§12 of `LaTex_docs/main.tex` and partially implemented in `_dynamics.py`, but the rest of the folder is still v4 (or, in the case of `_plant.py`, v2 leftover). The folder needs a clean, well-commented v5 promotion before the import.

The user has explicitly asked for:
1. Going through every file in the folder and lifting it to v5.
2. Professional commenting throughout, suitable for verifiers/testers who didn't write the code.
3. Implementation of the chance-constraint cost (§9.6, equations 23–24) — flagged as the main novelty.
4. Pinning the difficult-to-estimate parameters per the FIM analysis (Section 11) so the first port-test can be done with simulated plant data without inference instability.

## Verified state of the folder (Phase 1 exploration)

| File | Status | v5 gap |
|---|---|---|
| `_dynamics.py` | v5-ready | None — Hill term in `drift_jax`, `TRUTH_PARAMS_V5` defined |
| `simulation.py` | v4 (drift inlined twice) | Lines 63–90 (numpy `drift`) and 93–121 (JAX `drift_jax`) need Hill term |
| `estimation.py` | v4 + broken constructor | (a) `EstimationModel(...)` line 369 missing `init_state_prior_config` (already defined at line 91); (b) drift inlined 3× at lines 115–116, 305–306, 341–342; (c) `PARAM_PRIOR_CONFIG` missing v5 deconditioning params |
| `control.py` | v4 — uses `drift_jax` ✓ | Cost is gradient-OT only; no chance-constraint cost |
| `_plant.py` | **v2 leftover** | 3D state `(B, F, A)` hard-coded (lines 66, 90, 95, 98–100); calls `drift_jax_v2` from `_dynamics.py` which is now 6D — runtime broken. Not in smc2fc import boundary. |
| `_phi_burst.py` | Generic | None |
| `__init__.py` | Empty | None |

**External integration boundary** (verified via grep): `smc2fc` and `tools/` import `TRUTH_PARAMS`, `TRUTH_PARAMS_V5`, `drift_jax`, `A_TYP`, `F_TYP`, `diffusion_state_dep` from `_dynamics.py`; `HIGH_RES_FSA_V4_MODEL`, `BINS_PER_DAY`, `DT_BIN_DAYS`, `circadian{,_jax}`, `DEFAULT_PARAMS`, `DEFAULT_INIT` from `simulation.py`; `build_control_spec` from `control.py`; `HIGH_RES_FSA_V4_ESTIMATION` from `estimation.py` (currently unimportable due to constructor bug). `StepwisePlant` from `_plant.py` is **not** in the smc2fc boundary.

## Approach

**Coexistence strategy.** Add v5 model objects (`HIGH_RES_FSA_V5_MODEL`, `HIGH_RES_FSA_V5_ESTIMATION`) alongside the existing v4 ones, so the v4 boundary symbols continue working for back-compat (tests, examples) while smc2fc imports the v5 ones. Internally, both v4 and v5 SDEModels share the same `drift` / `drift_jax` functions which now include the Hill term — but v4 calling code passes `mu_dec_B = mu_dec_S = 0` (already the default in `TRUTH_PARAMS`), so the Hill term is silent and v4 numerics are unchanged.

This avoids any duplication of the SDE drift across v4 and v5 — single source of truth in `_dynamics.py`, with `simulation.py`'s inlined copies kept in sync.

**Pinning strategy (per user instruction at plan-review):** maximum-pinning for first port-test, because simulated plant data will not have rich enough excitation to inform any of the v4/v5 extension parameters. Hard-freeze:
- `KFB_0 = 0.030`, `KFS_0 = 0.050` — structural (Section 11, slack direction #44).
- `n_dec = 4.0` — structural Hill shape parameter.
- `tau_K = 21.0` d — Busso-standard (Section 11.6 recommendation).
- `B_dec = 0.07`, `S_dec = 0.07` — pinned to the §10 Figure 7 (v5 island) values.
- `mu_dec_B = 0.10`, `mu_dec_S = 0.10` — same.

So **all 4 v5 deconditioning parameters are frozen**, plus the 3 structurally weak v4 params, plus `n_dec`. Net: PARAM_PRIOR_CONFIG goes from 40 (v4) → 40 − 3 (the v4 freezes) + 0 v5 entries (all 4 v5 deconditioning params and `n_dec` go straight into `frozen_params`) = **37 estimated parameters**. This is appropriate for the first end-to-end pipeline check on simple steady-state synthetic data; relaxing any of these for richer datasets later is a one-line move from `frozen_params` back into `PARAM_PRIOR_CONFIG`.

**Chance-constraint cost** (the v5 novelty). Implement as a reference function in a new file `control_v5.py`, with the signature
```python
def evaluate_chance_constrained_cost(
    theta_particles: jnp.ndarray,    # shape (n_particles, theta_dim)
    weights: jnp.ndarray,            # shape (n_particles,)
    Phi_schedule: jnp.ndarray,       # shape (n_steps, 2)
    spec: ControlSpecV5,
    alpha: float = 0.05,             # max allowed P[A_t < A_sep(Phi_t)]
    A_target: float = ...,           # min ∫A dt target
) -> dict:
    """Returns {effort, violation_rate, satisfies_constraint, ...}."""
```
- Per particle: forward-simulate the v5 SDE under Phi, with CRN noise.
- Compute analytical separatrix `A_sep(Phi_t)` via Brentq root-finding on `bar_mu(A; Phi_t) - eta * A^2 = 0` (the v5 version), reusing `find_A_separatrix` from `tools/stability_basins_v4.py`.
- Per-particle violation rate = fraction of bins where `A_i_t < A_sep(Phi_t)`.
- Effort = `∫ ||Phi||² dt` (deterministic in Phi).
- Returns aggregate metrics; the smc2fc outer SMC loop is responsible for using these to weight / reject particles.

**Doc-style commenting** target: every public function gets a docstring with (a) one-line summary, (b) what it does in plain words, (c) parameter shapes/units, (d) numerical assumptions, (e) where it sits in the §-of-LaTeX-doc concept map. Inline comments explain the Hill term, the slow-manifold reduction, the gating logic, and any unit conventions (days vs hours, ratio vs probability) that would otherwise look magic.

## File-by-file edit list

### 1. `models/fsa_high_res/_dynamics.py`
**Status:** v5-ready, no functional changes needed. **Action:** Re-comment the Hill block (lines 86–101 of `drift_jax`) with full prose explaining each line and its tie to LaTeX §10.2 equation \ref{eq:v5-mubar}. Also add a top-of-file map of which symbol goes to which v5 doc section.

### 2. `models/fsa_high_res/simulation.py`
- **Lines 73–74 (`drift`, numpy):** insert Hill term. Keep numpy idiom (`np.power`, `np.maximum`).
- **Lines 104–105 (`drift_jax`):** mirror Hill term using `jnp.power`. Keep matching comments.
- **Lines 348–377 (`HIGH_RES_FSA_V4_MODEL`):** rename to `_FSA_V4_LEGACY` (kept as an alias for back-compat, but with `mu_dec_B = mu_dec_S = 0` enforced). Add new `HIGH_RES_FSA_V5_MODEL` with v5 default params from `TRUTH_PARAMS_V5`.
- **`DEFAULT_PARAMS`:** add v5 keys with v4-recovering defaults (`mu_dec_* = 0`, `n_dec = 4`).
- **`DEFAULT_INIT`:** unchanged (6D already).

### 3. `models/fsa_high_res/estimation.py`
- **Line 91 (`INIT_STATE_PRIOR_CONFIG`):** unchanged — already an empty `OrderedDict`.
- **Line 41 (`PARAM_PRIOR_CONFIG`):** **do NOT add v5 deconditioning entries** (they are all frozen for the first port-test, see Pinning strategy above). Remove `KFB_0`, `KFS_0`, `tau_K` from the dict; everything else stays as v4.
- **Lines 115–116 (`propagate_fn`):** replace the inlined `mu_bif` line with a call to `drift_jax(y, params_dict, Phi)`. Eliminates one source of v4/v5 drift duplication.
- **Lines 305–306 (`forward_sde_stochastic`):** replace inlined drift with `drift_jax(y, p_jax, Phi_arr[i])`.
- **Lines 341–342 (`imex_step_fn`):** replace inlined drift with `drift_jax(y, p_jax, Phi_k)`.
- **Lines 369–387 (`HIGH_RES_FSA_V4_ESTIMATION` constructor):**
  - Add the missing `init_state_prior_config=INIT_STATE_PRIOR_CONFIG` arg.
  - Rename to `HIGH_RES_FSA_V5_ESTIMATION` (v5 is the new default).
  - Set `frozen_params` to the maximum-pinned set:
    ```python
    frozen_params={
        # FSA-v4 frozen (diffusion + structural):
        'sigma_B': SIGMA_B_FROZEN, 'sigma_S': SIGMA_S_FROZEN,
        'sigma_F': SIGMA_F_FROZEN, 'sigma_A': SIGMA_A_FROZEN,
        'sigma_K': SIGMA_K_FROZEN, 'phi': PHI_FROZEN,
        # Pinned per FIM §11.6 (structurally / weakly identifiable):
        'KFB_0': 0.030, 'KFS_0': 0.050, 'tau_K': 21.0,
        # Pinned for first port-test (v5 deconditioning, simple data):
        'n_dec':    4.0,
        'B_dec':    0.07,  'S_dec':    0.07,
        'mu_dec_B': 0.10,  'mu_dec_S': 0.10,
    }
    ```
  - Keep `HIGH_RES_FSA_V4_ESTIMATION` as a thin alias forcing `mu_dec_B = mu_dec_S = 0` (recovers v4 numerics exactly).
- **`get_init_theta`:** update for the pruned 37-entry param vector.

### 4. `models/fsa_high_res/control.py`
**Status:** uses `drift_jax` correctly, picks up v5 automatically once `_dynamics.py` Hill term is on. **Action:** add v5-aware `build_control_spec_v5(...)` factory that:
- Loads `TRUTH_PARAMS_V5` instead of `TRUTH_PARAMS`.
- Passes `mu_dec_*` and `B_dec`, `S_dec`, `n_dec` through.
- Same gradient-OT cost — **the chance-constraint cost lives in a separate file (item 6).**

### 5. `models/fsa_high_res/__init__.py`
**Action:** export the v5 boundary symbols explicitly:
```python
from models.fsa_high_res._dynamics import (
    TRUTH_PARAMS, TRUTH_PARAMS_V5, A_TYP, F_TYP,
    drift_jax, diffusion_state_dep,
)
from models.fsa_high_res.simulation import (
    HIGH_RES_FSA_V5_MODEL, BINS_PER_DAY, DT_BIN_DAYS,
    circadian, circadian_jax, DEFAULT_PARAMS, DEFAULT_INIT,
)
from models.fsa_high_res.estimation import HIGH_RES_FSA_V5_ESTIMATION
from models.fsa_high_res.control import build_control_spec, build_control_spec_v5
from models.fsa_high_res.control_v5 import evaluate_chance_constrained_cost
```

### 6. **NEW** `models/fsa_high_res/control_v5.py`
Reference implementation of the §9.6 chance-constraint cost. Self-contained file, ~150 lines.
- `find_A_sep_v5(Phi_B, Phi_S, params)`: Brent root-finder on `bar_mu(A; Phi) - eta * A^2 = 0`, returning the smaller positive root (the unstable separatrix). Re-uses logic from `tools/stability_basins_v4.py:find_A_separatrix`.
- `evaluate_chance_constrained_cost(theta_particles, weights, Phi_schedule, spec, alpha, A_target)`: vmapped over particles.
  - For each particle: forward-simulate via `drift_jax` with CRN noise.
  - Compute violation rate against `A_sep(Phi_t)`.
  - Return dict: `{mean_effort, mean_A_integral, violation_rate_per_particle, weighted_violation_rate, satisfies_chance_constraint, satisfies_target}`.
- Heavily commented mapping to LaTeX §9.6 equations 23–24. Includes a `__main__` block that runs a smoke test on `TRUTH_PARAMS_V5` with a tiny particle cloud (10 particles).

### 7. `models/fsa_high_res/_plant.py` — full v2→v5 rewrite (per user instruction)
The plant is the ground-truth simulator the closed-loop MPC runs against during testing. It must support 6D v5 dynamics so that simulated-feedback testing of the SMC$^2$ controller is possible. Specific edits:
- **`_plant_em_step` (lines 64–106):**
  - Change state width from 3 → 6 throughout: docstring, `noise = jax.random.normal(sub, (6,), ...)`, the `g` vector (add Jacobi-style diffusion for `S`, CIR-style for `KFB, KFS`), and the post-step clipping (`y_new.at[i].set(...)` for $i = 0$–$5$).
  - The drift call already uses `drift_jax` from `_dynamics.py` which is v5-ready, so no change there.
  - Mirror the diffusion structure of `_dynamics.diffusion_state_dep` (lines 116–127) — that is, `sigma_diag` becomes a 6-vector `(sigma_B, sigma_S, sigma_F, sigma_A, sigma_K, sigma_K)` and `g` becomes `(sqrt(B(1-B)), sqrt(S(1-S)), sqrt(F), sqrt(A+ε), sqrt(KFB), sqrt(KFS))`.
- **`StepwisePlant` dataclass (line 109+):**
  - Default `state` field: change initial 3D `(B_0, F_0, A_0)` to 6D `(B_0, S_0, F_0, A_0, KFB_0, KFS_0)`. Use `DEFAULT_INIT` from `simulation.py` which already has the 6 keys.
  - Default `truth_params`: should be `TRUTH_PARAMS_V5` (or `DEFAULT_PARAMS` once that includes v5 keys).
  - History buffers: ensure `'trajectory'` stores 6D rows.
  - `advance(...)` method: pass 6-vector through `_plant_em_step`. Update any obs-sampler calls so they read state by named index correctly (the obs samplers in `simulation.py` already expect 6D state since they were updated for v4).
- **Top-of-file docstring:** update from "FSA-v2 simulator" to "FSA-v5 (6D) simulator". Add an explicit reference to LaTeX §10 and §11 for the model spec.
- **Verification:** the new smoke test (item 8) instantiates `StepwisePlant`, runs it forward 14 days under a moderate-Phi schedule, and checks no NaN, all states in their bounded ranges, and that A approaches a healthy steady state under v5 truth params.

### 8. **NEW** `tests/test_fsa_v5_smoke.py`
End-to-end smoke test exercising the full v5 stack including the plant:

```python
def test_v5_plant_forward_pipeline():
    """StepwisePlant + propagate_fn end-to-end, 6D, no NaN, sane ranges."""
    # 1. Build a 14-day moderate Phi schedule Phi=(0.30, 0.30) (inside v5 island).
    # 2. Instantiate StepwisePlant with TRUTH_PARAMS_V5 and 6D DEFAULT_INIT.
    # 3. Advance the plant in 1-day strides; collect 6D state trajectory.
    # 4. Generate synthetic observations from the recorded state trajectory.
    # 5. Run estimation.propagate_fn over the synthetic data, with v5 frozen
    #    params, starting from a perturbed init.
    # 6. Assert: no NaN anywhere; B,S in [0,1]; F,A,KFB,KFS >= 0; A>0 by end of run.
    # 7. Assert: forward log-likelihood is finite.

def test_v5_chance_constrained_cost_smoke():
    """evaluate_chance_constrained_cost runs on a 10-particle cloud."""
    # 1. Build a 10-particle cloud with all theta = TRUTH_PARAMS_V5 + small jitter.
    # 2. Build a 28-day Phi schedule with one detraining week mid-block.
    # 3. Call evaluate_chance_constrained_cost with alpha=0.05.
    # 4. Assert: returned dict has keys {mean_effort, mean_A_integral,
    #    weighted_violation_rate, satisfies_chance_constraint, satisfies_target}.
    # 5. Assert: weighted_violation_rate in [0, 1]; mean_effort > 0;
    #    mean_A_integral > 0.
```

### 9. **NEW** `LaTex_docs/sections/appendix_v5_parameters.tex`
A new appendix section documenting every FSA-v5 parameter — symbol, units, role in the equations, prior or frozen value, and a one-line explanation. Wired into `main.tex` via `\input{sections/appendix_v5_parameters.tex}` after the existing `appendix.tex`. The structure:

```
\section{Appendix: Complete FSA-v5 Parameter Reference}

\subsection{Estimated parameters (37)}
\begin{longtable}{lllll}
Symbol & Units / type & Role / equation & Prior & Truth (TRUTH_PARAMS_V5) \\
...
\end{longtable}

\subsection{Frozen parameters (8 + 6 diffusion)}
\begin{tabular}{lllll}
Symbol & Units & Role / equation & Frozen value & Reason for freezing \\
...
\end{tabular}

\subsection{Initial state (no priors — set from data or DEFAULT_INIT)}
\begin{tabular}{lll}
State & Default & Bounds \\
...
\end{tabular}
```

Cross-references each parameter to where it first appears in §7, §10, or §11 of the doc. This appendix is the single canonical reference for anyone porting the codebase or doing inference setup.

## Critical files

- [models/fsa_high_res/_dynamics.py](models/fsa_high_res/_dynamics.py) — re-commenting only.
- [models/fsa_high_res/simulation.py](models/fsa_high_res/simulation.py) — drift edits + new V5 SDEModel.
- [models/fsa_high_res/estimation.py](models/fsa_high_res/estimation.py) — drift de-duplication via `drift_jax` calls + constructor fix + maximum-pinning frozen_params.
- [models/fsa_high_res/control.py](models/fsa_high_res/control.py) — add `build_control_spec_v5`.
- [models/fsa_high_res/_plant.py](models/fsa_high_res/_plant.py) — full 3D→6D rewrite for v5.
- **NEW** `models/fsa_high_res/control_v5.py` — chance-constraint cost reference.
- **NEW** `models/fsa_high_res/__init__.py` — re-exports.
- **NEW** `tests/test_fsa_v5_smoke.py` — end-to-end smoke test (plant + propagate_fn + chance-constraint cost).
- **NEW** `LaTex_docs/sections/appendix_v5_parameters.tex` — complete parameter reference.
- [LaTex_docs/main.tex](LaTex_docs/main.tex) — add the `\input` for the new appendix.

## Verification

1. `.fsa_venv/bin/python -c "from models.fsa_high_res import HIGH_RES_FSA_V5_ESTIMATION, HIGH_RES_FSA_V5_MODEL; print(HIGH_RES_FSA_V5_MODEL.name, len(HIGH_RES_FSA_V5_ESTIMATION.param_prior_config))"` — confirms the constructor bug is fixed and the pruned 37-entry param vector is in place.
2. `.fsa_venv/bin/python -m pytest tests/test_fsa_v5_smoke.py -v` — runs the full smoke test (plant + propagate_fn + chance-constraint cost).
3. `.fsa_venv/bin/python tools/stability_basins_v4.py` — regression sanity check that the v5 closed-island figure still reproduces (since `_dynamics.py` only gets re-commented, this should be byte-identical to the existing run).
4. `.fsa_venv/bin/python tools/fim_analysis_v5.py` — FIM sanity check with the new maximum-pinned param vector; the spectrum should now show only the directions for the 37 estimated params.
5. `cd LaTex_docs && latexmk -pdf -interaction=nonstopmode main.tex` — confirms the new appendix compiles into the doc cleanly.

## Out of scope (deliberate non-decisions)

- **Wiring the chance-constraint cost into a working Φ-optimiser.** That belongs in smc2fc, not here. This pass provides the reference cost function only.
- **A new SMC$^2$ outer-loop control engine.** Same reason — lives in smc2fc.
- **Re-running `tools/fim_analysis_v5.py` after pruning** to update the §11 numerical results in-text. The user can request this as a follow-up if they want the doc to track the post-pruning FIM eigenvalue spectrum. The new appendix will list the frozen vs estimated split, which is the more important output for the port.

## Decision points still to confirm at review time

1. **`HIGH_RES_FSA_V4_*` legacy aliases.** I keep them as thin aliases pointing at the v5 SDEModel with `mu_dec_*=0` frozen, so existing tests/examples and the older v4 doc references keep working without code edits. Alternative is to delete and let downstream code adapt. *Recommendation: keep the aliases — the cost is minimal (~5 lines) and the back-compat win is real.*
