# FSA dev sandbox — v4/v5 branch port (Phase 6 of the original plan)

> Archived from plan mode: 2026-05-04 23:53.
> Updated: 2026-05-05 00:05 — Phases 1, 2, 4, 7 complete on `claude/dev-sandbox-v4`. Branch pushed to github (3 commits: a1f0638, 8feb8d2, 5f91785). 13/13 pytest green (1 v4 physics + 4 v5 smoke + 6 obs-consistency-v5 + 2 reconciliation-v5). Parent project CLAUDE.md gained an "FSA dev sandbox" subsection with the v5 dependency chain + sigma_S name-collision warning. Phase 3 (plant regression scenarios from v5 guide §8) and the export-to-framework orchestrator deferred to a follow-up — the v5 smoke + obs-consistency + reconciliation layer already protects the load-bearing path; scenarios + bundle exporter are nice-to-have on top.
> This is the v4-branch-specific execution of the original plan
> (whose master archive is the equivalent file already on
> `claude/dev-sandbox-main`). Phase 6 was deferred from the earlier
> session because another agent was working on `v4-bimodal-variable-dose-extension`;
> they have now committed `14b819d` (FSA-v5 sedentary-collapse extension +
> port-ready code) and the work resumes here on `claude/dev-sandbox-v4`.

---

## Context — what changed since the original plan was written

When the master plan was approved (2026-05-04 13:46), the v4 branch was
on `6c44f80` — a 6D Variable-Dose model (B, S, F, A, K_FB, K_FS) with
bimodal control (Phi_B, Phi_S) but no sedentary-collapse extension and
no chance-constrained cost. Since then, the other agent has shipped
commit `14b819d` which adds:

- **FSA-v5 Hill-deconditioning** in the `mu(B,S,F)` Stuart-Landau drive,
  closing the healthy basin into an island in (Phi_B, Phi_S) space.
- **Volume-Load** (5th observation channel) — Gaussian, gated to
  training-session bins.
- **`evaluate_chance_constrained_cost`** (the v5 main novelty) — a
  probabilistic cost evaluator that lets the SMC² controller handle
  basin-escape probability as a chance constraint.
- **`HIGH_RES_FSA_V5_*` symbols** in `models/fsa_high_res/__init__.py`
  as the new canonical export surface (v4 aliases preserved).
- **Their own validation suite** in:
  - `tests/test_fsa_v5_smoke.py` (4 tests: imports, plant pipeline,
    propagate_fn, chance-constrained cost)
  - `tools/fim_analysis_v5.py` (473-line FIM analysis with
    log-coordinate handling for lognormal-prior params + LaTeX/figure
    output — strictly richer than my Phase-2 `analyze_identifiability.py`
    on the main branch)
  - `tools/stability_basins_v4.py` (747 lines of basin-geometry tools)
  - Other v3/v4 optimization helpers
- **The 51 KB v5 technical guide** at `LaTex_docs/FSA_version_5_technical_guide.tex`
  — definitive reference for porting v5 into smc2fc.

## What this branch adds on top

What's still missing on v4 that the master plan called for:

1. **Install metadata** (pyproject.toml, LICENSE, .gitignore extension,
   CHANGELOG.md) — the dev repo had no `pip install -e .` story.
2. **Drop the bundled `smc2fc/` and `simulator/` stubs** — outdated
   subset of the real `smc2fc` package; FSA's signatures are
   forward-compatible.
3. **Sim ↔ estimator obs consistency** for the 5 v5 channels
   (HR sleep-gated / Sleep Bernoulli / Stress wake-gated /
   Steps wake-gated / VolumeLoad). The new VolumeLoad channel is
   structurally vulnerable to a SWAT-D1/D2 type bug; the structural
   protection layer is what the SWAT incident taught us to build.
4. **Plant ↔ estimator drift parity** (bit-equivalent Euler step
   between `_plant.py` and `estimation.py:propagate_fn_v5`'s
   inline-drift `mu_prior`).
5. **Master export gate** (`tools/export_to_framework.py`) that runs
   the other agent's existing v5 tools + my new tests + writes a
   verified bundle under `exports/fsa_v5_verified/`.
6. **Plant regression scenarios** for the three v5 regimes
   (sedentary collapse, healthy moderate, over-training collapse) —
   the test scenarios from §8 of the v5 technical guide.

## Branch + push hygiene

Working branch: `claude/dev-sandbox-v4`, branched from
`v4-bimodal-variable-dose-extension` at `14b819d`. The existing
`main`, `v3-bimodal-extension`, and `v4-bimodal-variable-dose-extension`
branches are not touched. Final push goes to `origin/claude/dev-sandbox-v4`
as a new branch only — no merge to v4 without the senior's explicit
sign-off.

## Phase status

| Phase | Status |
|---|---|
| 1 — install metadata + drop stubs | **In progress** (this commit) |
| 2 — obs consistency + drift parity tests + export orchestrator | Next |
| 3 — v5 plant regression scenarios | Next |
| 4 — push to github | Next |
| 7 — parent CLAUDE.md FSA dep-chain section | After v4 work pushed |

## Key v5 facts the plan must respect

From the v5 technical guide §8 and §9 (bugs/gotchas):

- **`sigma_S` name collision**: there are TWO `sigma_S` keys — one for
  latent-S Jacobi diffusion (~0.008), one for stress-channel obs noise
  (~4.0). Python dicts keep the last assignment (the obs noise). Plant
  hard-codes diffusion scales as `SIGMA_*_FROZEN` constants, bypassing
  the params dict. Any test code reading these from params will hit
  the bug; read them from `_dynamics.SIGMA_*_FROZEN` (or whatever
  the equivalent is in v4/v5) directly.
- **Particle dicts must include v5 Hill keys** (`B_dec, S_dec,
  mu_dec_B, mu_dec_S, n_dec`) before being passed to `drift_jax`. The
  estimator's PARAM_PRIOR_CONFIG only has the 37 estimable params; the
  Hill keys are frozen and live in `frozen_params`. Always merge before
  the drift call.
- **Plant `truth_params` must be `DEFAULT_PARAMS_V5` (not
  `TRUTH_PARAMS_V5`)** — the latter only has dynamics keys, the former
  has dynamics + observation coefficients. The obs samplers will
  KeyError otherwise.
- **15-min bins are non-negotiable**: switching to daily summaries
  inflates FIM condition number by 20 orders of magnitude (per §11 of
  the v5 guide).

## Verification

End-to-end after Phase 4:

```bash
cd /tmp
rm -rf FSA_v4_test_install
git clone -b claude/dev-sandbox-v4 https://github.com/ajaytalati/FSA_model_dev FSA_v4_test_install
cd FSA_v4_test_install
pip install -e ".[test]"
JAX_ENABLE_X64=True JAX_PLATFORMS=cpu pytest tests/ -v
JAX_ENABLE_X64=True JAX_PLATFORMS=cpu python tools/export_to_framework.py
```

Expected: every pytest test green, all gates pass, bundle written
under `exports/fsa_v5_verified/`.
