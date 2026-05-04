
## Plan: `FSA_version_5_technical_guide.tex`

**Goal.** A standalone LaTeX doc, ~15-20 pages, that an engineer porting the model into `smc2fc` can read in one sitting. No v2/v3 history, no narrative arcs, no plain-language companion sections. Just the v5 model + what to do with it.

**Structure (8 sections + appendix):**

1. **Overview** — what this guide covers, what it deliberately omits, key file paths in the `fsa_high_res/` package, and how to read it.
2. **Latent dynamics (the SDE)** — 6D state, the cascade structure, all 6 drift equations, the v5 Hill deconditioning term, the state-dependent diffusion. Cross-references to `_dynamics.py:drift_jax` line ranges.
3. **Observation model** — 4 Gaussian channels (HR/Stress/Steps/VL), the Bernoulli sleep label, sleep/wake gating policy, the production 15-min grid. Cross-references to `estimation.py`.
4. **The three regimes** — closed-island basin geometry, transcritical + saddle-node curves, time-domain confirmation. Reuses the three existing figures (`v5_sedentary_collapse_island.png`, `v5_full_bifurcation.png`, `v5_trajectories_three_regimes.png`) and the closed-island sanity-check table.
5. **Probabilistic cost function (v5 novelty)** — chance-constraint formulation (eqs 23, 24), why gradient-OT can't represent it, the reference implementation in `control_v5.py`, the smc2fc plug-in API.
6. **Numerical integration choices** — 15-min bins (`FSA_STEP_MINUTES=15`), no sub-stepping, stiffness analysis (fastest mode ≈ τ_F, slowest ≈ τ_S), why daily sampling is non-viable for inference (refs §11 FIM result).
7. **Parameter set: 37 estimated + 14 frozen** — the maximum-pinning rationale per FIM, what each freeze costs and what would unlock it, how to relax in stages.
8. **Test scenarios for constant-Φ** — three deterministic-rollout tests (sedentary, moderate, overtraining) using `StepwisePlant`, plus detraining-from-trained-state and bistability-sensitivity (perturb A₀ across separatrix). Each has a code snippet + expected qualitative outcome.
9. **Common bugs and gotchas** — the engineering grit list, mostly from the implementation pass we just did:
    - `sigma_S` name collision (latent diffusion vs stress obs noise) — _I hit this_.
    - `JAX_ENABLE_X64=True` required — silent float32 truncation otherwise.
    - `EstimationModel` requires `init_state_prior_config` — _I hit this_.
    - Drift is duplicated in `simulation.py` and `_dynamics.py` — must stay in sync.
    - Plant truth_params must be `DEFAULT_PARAMS_V5` (not `TRUTH_PARAMS_V5` — the latter lacks obs coefs).
    - K baselines (`KFB_0, KFS_0`) are structurally non-identifiable — never try to estimate them.

**Appendix — Complete FSA-v5 parameter reference.** Lifted verbatim from `appendix_v5_parameters.tex`.

**Source reuse.** Most content lifts/adapts from the existing 54-page doc:

- §11.1 → Section 2/3 (full equations, gathered)
- §10.3-10.5 → Section 4 (basins + figures 7/8/9 + sanity-check table)
- §9.6 → Section 5 (chance-constraint formulation)
- §11.3 → Section 6 (sampling protocol)
- §11.6 → Section 7 (pinning rationale)
- New: Section 1 (overview), Section 8 (test scenarios), Section 9 (engineering pitfalls).

**Output.** New file `LaTex_docs/FSA_version_5_technical_guide.tex` — self-contained (its own preamble), inputs no other section files, references images at `LaTex_docs/figures/...`. Compiles independently of `main.tex`.

**Verification.** `latexmk -pdf FSA_version_5_technical_guide.tex` produces a clean PDF; check page count is in the 15-25 range; spot-check that the parameter appendix matches the implementation in `estimation.py:HIGH_RES_FSA_V5_ESTIMATION`.

Want me to proceed with this, or do you want to redirect any of the section choices first?