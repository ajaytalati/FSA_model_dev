
# FSA-v5 Implementation Plan for the `fsa_high_res/` Folder

  

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

  

**Pinning strategy** (default — user can override at plan-review time): hard-freeze the structurally non-identifiable params per §11.6:

- `KFB_0`, `KFS_0` — structural (Section 11, slack direction #44).

- `n_dec` — structural shape parameter, no need to learn.

- `tau_K` — Busso-standard 21 d (Section 11.6 recommendation).

  

These move from `PARAM_PRIOR_CONFIG` into the EstimationModel `frozen_params` dict. PARAM_PRIOR_CONFIG goes from 40 (v4) → 40 + 4 v5 deconditioning − 3 freezes = **41 estimated params**. Easy to relax later by moving any back into PARAM_PRIOR_CONFIG.

  

**Chance-constraint cost** (the v5 novelty). Implement as a reference function in a new file `control_v5.py`, with the signature

```python

def evaluate_chance_constrained_cost(

theta_particles: jnp.ndarray, # shape (n_particles, theta_dim)

weights: jnp.ndarray, # shape (n_particles,)

Phi_schedule: jnp.ndarray, # shape (n_steps, 2)

spec: ControlSpecV5,

alpha: float = 0.05, # max allowed P[A_t < A_sep(Phi_t)]

A_target: float = ..., # min ∫A dt target

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

- **Line 41 (`PARAM_PRIOR_CONFIG`):** insert four v5 entries after `mu_FF`:

```python

('B_dec', ('lognormal', (math.log(0.07), 0.20))),

('S_dec', ('lognormal', (math.log(0.07), 0.20))),

('mu_dec_B', ('lognormal', (math.log(0.10), 0.30))),

('mu_dec_S', ('lognormal', (math.log(0.10), 0.30))),

```

Remove `KFB_0`, `KFS_0`, `tau_K` from `PARAM_PRIOR_CONFIG` (they will be frozen).

- **Lines 115–116 (`propagate_fn`):** replace the inlined `mu_bif` line with a call to `drift_jax(y, params_dict, Phi)`. This both (a) adds the Hill term automatically and (b) eliminates one source of v4/v5 drift duplication.

- **Lines 305–306 (`forward_sde_stochastic`):** replace inlined drift with `drift_jax(y, p_jax, Phi_arr[i])`.

- **Lines 341–342 (`imex_step_fn`):** replace inlined drift with `drift_jax(y, p_jax, Phi_k)`.

- **Lines 369–387 (`HIGH_RES_FSA_V4_ESTIMATION` constructor):**

- Add the missing `init_state_prior_config=INIT_STATE_PRIOR_CONFIG` arg.

- Rename to `HIGH_RES_FSA_V5_ESTIMATION` (v5 is the new default).

- Add v5 params to `frozen_params` dict: `KFB_0=0.030, KFS_0=0.050, tau_K=21.0, n_dec=4.0`.

- Keep `HIGH_RES_FSA_V4_ESTIMATION` as a thin alias for back-compat (`mu_dec_B = mu_dec_S = 0` frozen).

- **`get_init_theta`:** update for new param vector length.

  

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

  

### 7. `models/fsa_high_res/_plant.py`

**Action:** flag at top of file with a `# TODO(v5): rewrite for 6D state.` comment. Do not fix in this pass — it's not in the smc2fc boundary, and a full rewrite (3D → 6D, including state-dependent diffusion for KFB, KFS) is out of scope for this implementation. If `tests/test_v3_identifiability.py` or other downstream code breaks because of this, defer the fix.

  

### 8. **NEW** `tests/test_fsa_v5_smoke.py`

Single end-to-end smoke test:

```python

def test_v5_forward_pipeline_runs():

"""Plant→propagate_fn end-to-end, no NaN, all states stay in physical bounds."""

# 1. Build a 14-day Phi schedule (mix of moderate + over-training).

# 2. Forward-simulate the v5 SDE deterministically (via forward_sde_stochastic with σ=0).

# 3. Generate synthetic observations using the obs prediction equations.

# 4. Run estimation.propagate_fn over the synthetic data.

# 5. Assert: no NaN, all states in [0, ∞), A < 5, B,S in [0, 1], log-likelihood finite.

# 6. Run evaluate_chance_constrained_cost on a 10-particle cloud at TRUTH_PARAMS_V5.

# 7. Assert: returned dict has expected keys, violation_rate in [0, 1].

```

  

## Critical files

  

- [models/fsa_high_res/_dynamics.py](models/fsa_high_res/_dynamics.py) — re-commenting only.

- [models/fsa_high_res/simulation.py](models/fsa_high_res/simulation.py) — drift edits + new V5 SDEModel.

- [models/fsa_high_res/estimation.py](models/fsa_high_res/estimation.py) — drift de-duplication via `drift_jax` calls + constructor fix + frozen_params + v5 prior entries.

- [models/fsa_high_res/control.py](models/fsa_high_res/control.py) — add `build_control_spec_v5`.

- **NEW** `models/fsa_high_res/control_v5.py` — chance-constraint cost reference.

- **NEW** `models/fsa_high_res/__init__.py` — re-exports.

- **NEW** `tests/test_fsa_v5_smoke.py` — end-to-end smoke test.

  

## Verification

  

1. `python -c "from models.fsa_high_res import HIGH_RES_FSA_V5_ESTIMATION; print('ok')"` — confirms the constructor bug is fixed.

2. `python -c "from models.fsa_high_res import HIGH_RES_FSA_V5_MODEL; print(HIGH_RES_FSA_V5_MODEL.name)"` — confirms v5 SDEModel exposed.

3. `.fsa_venv/bin/python -m pytest tests/test_fsa_v5_smoke.py -v` — runs the full smoke test.

4. `.fsa_venv/bin/python tools/stability_basins_v4.py` — sanity check that the existing v5 stability tool still produces the same closed-island figure (regression on `_dynamics.py` re-commenting).

5. `.fsa_venv/bin/python tools/fim_analysis_v5.py` — sanity check on FIM with the new pruned PARAM_PRIOR_CONFIG; the eigenvalue spectrum should match the previous run minus the slack directions corresponding to the now-frozen params (`KFB_0`, `KFS_0`, `tau_K`).

6. `cd LaTex_docs && latexmk -pdf -interaction=nonstopmode main.tex` — confirms doc still compiles (it should, since the doc isn't being edited in this plan).

  

## Out of scope (deliberate non-decisions)

  

- **Rewriting `_plant.py` to v5**. It's v2-era and not on the smc2fc port path. Flagged as TODO; user can decide later.

- **Wiring the chance-constraint cost into a working Φ-optimiser.** That belongs in smc2fc, not here. We provide the reference cost function only.

- **A new SMC$^2$ outer-loop control engine**. Same reason.

- **Changes to `LaTex_docs/`**. The plan document text already covers everything; only the code is being touched.

- **Re-running `tools/fim_analysis_v5.py` after pruning** to update the §11 numerical results. The user can request this as a follow-up if they want the doc to track the post-pruning FIM.

  

## Decision points the user may want to override at review time

  

1. **Pinning approach:** I default to *hard freeze* (move to `frozen_params`) for `KFB_0, KFS_0, tau_K, n_dec`. Soft pin (keep with narrow priors) is reversible and may be preferred. *Recommendation: hard freeze for first port-test as the user said; relax later.*

  

2. **Chance-constraint cost — runnable vs spec.** I default to *runnable reference implementation* in `control_v5.py` because the user identified it as the main novelty. Alternative is a docstring-only specification.

  

3. **`_plant.py`** — I default to *flagging with TODO and not touching*. Alternative is to rewrite it to 6D in this same pass.

  

4. **`HIGH_RES_FSA_V4_*` legacy aliases.** I keep them as thin aliases pointing at the v5 SDEModel with `mu_dec_*=0` frozen. Alternative is to delete and let downstream code update.



---

Ok here are my replies to the decision points - 

1) Also pin the 4 deconditioning parameters to the same values as in figure 7, in addition to what you have stated you will hard freeze - basically i want you make the initial testing EASY - by NOT assuming much variation in the data, e.g. no tapering loading, detraining ect - so most of the parameters added by version 4 and 5 basically should be pinned as we should assume for testing that the data is very simple / no much excitation in training variations / assume basic steady state 

also write a new appendix to the Latex doc which explains ALL the parameters of the version 5 model and their priors and/ or frozen values

3) Yes the plant is very important for the first testing of the MPC closed look and model using simulated (feedback) data ??? So it needs to be re-written int he same pass


---

Great - so the current 54 page pdf document is more like a masters thesis than an implementation guide for a research engineer! There is a lot of outdated details of earlier models/verisons 

Can you write a NEW latex doc, FSA_version_5_technical_guide.tex  which contains the appendix you just wrote and all the information a research engineer which has to implement the model into the smc2fc code base should know.

1) All the model equations (both latent and observational) - what all the parameters mean / do in scientific/exercise and mathematical terms
2) Details of the probabilistic cost function (as that is the main difference between earlier versions)
3) The three stable basins A healthy through moderate training, A-> 0 via sedantary and overtraining 
4) Time step choices and potential stiffness
5) The choice of which parameters were frozen and why
6) Possible constant training load test scenarios, which can be used for testing - I think the plots (figure 7,8,9) and text and tables of section 10.3 -10.5 would be very helpful  
7) Any other information which you think will help an engineer which might run into bugs / common problems?
