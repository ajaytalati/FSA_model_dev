# FSA dev sandbox — mirror the SWAT_model_dev pattern across main / v3 / v4

> Archived from plan mode: 2026-05-04 13:46.

> Plan for: `github.com/ajaytalati/FSA_model_dev` (local clone at `/home/ajay/Repos/FSA_model_dev/`).
> All work happens on **new `claude/...` branches**. Existing `main`, `v3-bimodal-extension`, `v4-bimodal-variable-dose-extension` are not touched until you approve a merge.

---

## Context — why this is being done

The FSA model files in `version_2/models/fsa_high_res/` of the parent `python-smc2-filtering-control` repo are the upstream production source. The matching dev repo at `github.com/ajaytalati/FSA_model_dev` exists but has **no in-isolation validation** — only 2 small smoke tests (`test_artifacts.py`, `test_fsa_physics.py`), no validation gates, no obs-channel consistency tests, no plant regression scenarios, and no installable package metadata (`pyproject.toml`, `LICENSE`, `.gitignore` all absent).

You just commissioned the same kind of dev sandbox for SWAT (now live at `github.com/ajaytalati/SWAT_model_dev` with 12 pytest tests + 5 validation gates + 6 plant regression scenarios). The job here is to mirror that pattern for FSA so:

1. **Main branch** continues to mirror version_2's FSA (currently in sync — only one cosmetic import-path difference); a fresh AI agent can `pip install -e .`, `pytest`, and confirm the model is ready to ship to smc2fc.
2. **`v3-bimodal-extension`** can be exercised by the same gates + tests, adapted to the bimodal-training changes, before any smc2fc port.
3. **`v4-bimodal-variable-dose-extension`** (the variable-dose Busso 2003 formulation on top of v3) gets the same treatment.

The model files themselves are NOT changed — only test/tool infrastructure is added around them.

---

## Critical findings from exploration (load-bearing assumptions)

1. **`FSA_model_dev/main` is in sync with `version_2/models/fsa_high_res/`**. All 7 model files are byte-identical except `simulation.py`, where the only difference is one import path (`from simulator.sde_model` vs `from smc2fc.simulator.sde_model`). No model logic differs.
2. **The bundled `smc2fc/` stubs in the dev repo are an outdated subset of the real `smc2fc` package.** Stubs cover only `EstimationModel`, `ControlSpec`, `RBFSchedule`, `_likelihood_constants` — total ~144 lines. Real package has all that plus `core/`, `filtering/`, `simulator/`, `transforms/`, etc. Field signatures are forward-compatible (FSA's usage is a subset of what the real package exposes).
3. **No psim involvement anywhere** — `grep -r psim` returns zero hits.
4. **Existing infrastructure on main**: `models/fsa_high_res/`, `simulator/`, `smc2fc/` (stubs), `tests/` (2 small files), `examples/`, `LaTex_docs/`, `README.md`. Missing: `pyproject.toml`, `LICENSE`, `.gitignore`, `CHANGELOG.md`, `tools/`, `scenarios/`, `bug_reports/`, `exports/`.
5. **State space differs from SWAT**: 3-state `[B, F, A]` (fitness / fatigue / autonomic amplitude) instead of 4-state `[W, Z, a, T]`. Single control `Φ` instead of three `(V_h, V_n, V_c)`. 4 obs channels, but **sleep is Bernoulli (not 3-level ordinal)**, **steps is log-Gaussian wake-gated**.
6. **22 estimated parameters + 7 frozen** (vs SWAT's 11 + 12). Frozen subset includes `σ_B`, `σ_F`, `σ_A`, `φ`.
7. **5 bench scripts in `version_2/tools/` import from `models.fsa_high_res.*`**: `bench_smc_filter_fsa.py`, `bench_smc_closed_loop_fsa.py`, `bench_smc_rolling_window_fsa.py`, `bench_smc_full_mpc_fsa.py`, `bench_lqg_baseline_fsa.py`. These are the downstream consumers — the dev sandbox's job is to ensure each scenario / test catches anything that would break them.

---

## Branch strategy

Three working branches, all named `claude/dev-sandbox-<base>`:

| Working branch | Branched from | Purpose |
|---|---|---|
| `claude/dev-sandbox-main` | `main` | Build the infrastructure first; this is the template branch. |
| `claude/dev-sandbox-v3` | `v3-bimodal-extension` | After Phase 4 succeeds, port the infrastructure to v3 with adapted scenario expectations. |
| `claude/dev-sandbox-v4` | `v4-bimodal-variable-dose-extension` | Same, on top of v4. |

All three get pushed to github so you can review the diffs as PRs against `main`/`v3`/`v4` before any merge happens. **No force-pushes. No rebases of existing branches.**

---

## Phases

### Phase 1 — Infrastructure on `claude/dev-sandbox-main`

Files to add at the FSA dev repo root:

| File | Purpose |
|---|---|
| `pyproject.toml` | Declare `smc2fc @ git+https://github.com/ajaytalati/python-smc2-filtering-control.git@master` + `matplotlib>=3.8` as deps. Same flat-package layout as SWAT (`models*`, `tools*`, `tests*`, `scenarios*` exposed). Pytest config. |
| `LICENSE` | MIT, attributed to Ajay Talati. |
| `.gitignore` | Same template as SWAT — Python noise, `.pytest_cache/`, `*.egg-info/`, `outputs/`, LaTeX intermediates, editor noise. |
| `CHANGELOG.md` | Initial entry recording this infrastructure rollout, with rationale + verified-by lines. |

Files to **delete** (replaced by the real `smc2fc` dep):

- `smc2fc/` (the bundled stub folder, ~144 lines across 4 files).
- `simulator/` (the generic SDE solver framework — moved into the real `smc2fc.simulator`).

One model-file edit:

- `models/fsa_high_res/simulation.py`: change `from simulator.sde_model import ...` → `from smc2fc.simulator.sde_model import ...` so the import resolves to the real package after the local `simulator/` folder is removed. This matches what version_2's copy already does.

### Phase 2 — Validation gates and test suite on `claude/dev-sandbox-main`

New files under `tools/` (5 validation pillars):

| Tool | What it does |
|---|---|
| `tools/analyze_identifiability.py` | FIM analysis for FSA's 22 estimated parameters across `bench_smc_filter_fsa.py`-style 1-day windows. Reports rank, condition number, and identifiable subset (parameters with FIM diagonal > threshold). Adapted from SWAT's tool but with the 3-state observation model (not 4-state) and FSA's 4 channels (HR, sleep_Bernoulli, stress, log-steps). |
| `tools/audit_stiffness.py` | Vmapped Jacobian audit over a `(B, F, A, Φ)` grid; reports max spectral radius and stable step size. |
| `tools/verify_likelihood.py` | Likelihood sanity: peakedness in `B`/`F`/`A` dimensions, boundary penalties, prior/truth alignment for the 22 free params. |
| `tools/verify_controller.py` | Build a `ControlSpec` on a pathological state, verify cost gradient is non-zero and a gradient step reduces cost. |
| `tools/export_to_framework.py` | Master gate: runs all 5 pillars, writes `exports/fsa_v2_verified/MANIFEST.json` + bundled model files. Anchored to `Path(__file__).resolve().parent.parent` so it works from any cwd (the bug we caught and fixed in SWAT). |

New files under `tests/` (3 pytest files):

| Test | What it asserts |
|---|---|
| `tests/test_reconciliation.py` | Mirror test: `StepwisePlant.advance(1 bin)` vs `EstimationModel.propagate_fn(1 bin)` produce equivalent latent state under deterministic control. Same shape as SWAT's. |
| `tests/test_obs_consistency.py` | For each of the 4 channels (HR, sleep, stress, steps), the simulator's noise-free predicted mean (or for sleep, the Bernoulli probability) matches the estimator's likelihood prediction at the same state/params/control. Uses non-zero subject-offset truth values to catch any D1/D2-class drift. |
| `tests/test_plant_regression_scenarios.py` | Parametrized over the 3 Banister-horizon scenarios (Phase 3 below) — drives `StepwisePlant` for the full horizon and asserts `B` lands in the expected basin. |

Pre-existing tests (`tests/test_artifacts.py`, `tests/test_fsa_physics.py`) are kept — they're useful smoke tests that complement the new layer.

### Phase 3 — Banister-horizon scenarios on `claude/dev-sandbox-main`

New folder `scenarios/` with the runner pattern:

- `scenarios/_common.py` — `run_fsa_scenario(scenario_key, ...) → int`. Drives `StepwisePlant.advance()` for the scenario's horizon under a daily Φ schedule, samples all 4 obs channels, applies dropout via the inlined `_apply_dropout` helper (same one as SWAT — no psim), saves `outputs/fsa/<scenario>/trajectory.npz` + per-channel `obs/*.npz`. Defines `EXPECTED_END_B` (basin classifier) — qualitative thresholds, not numerical targets.
- `scenarios/42d_horizon_max_sustainable.py` — T=42 d, daily Φ at max sustainable load (per the project memory: T=42 strategy = Banister overload, A inflects at ~day 20). Expected basin: `B` high, `A` slave to `μ(B,F)`.
- `scenarios/56d_horizon_periodised.py` — T=56 d, periodised training schedule.
- `scenarios/84d_horizon_long_block.py` — T=84 d, long aerobic block.

Truth Φ schedules + expected basins for each scenario are derived from `version_2`'s existing FSA bench-output reference numbers; if those aren't already documented anywhere, Phase 3 will read the recent `outputs/fsa_high_res/experiments/*/CHANGELOG.md` files in the parent repo and pin the basins from there.

### Phase 4 — Verify + push `claude/dev-sandbox-main`

End-to-end verification before pushing:

1. `pip install -e ".[test]"` from a fresh state — confirm the real `smc2fc` is pulled, the bundled stubs and `simulator/` folder are gone, all FSA imports still resolve.
2. `cd /tmp && pytest /home/ajay/Repos/FSA_model_dev/tests/ -v` from outside the repo — confirm all tests pass.
3. `cd /tmp && python /home/ajay/Repos/FSA_model_dev/tools/export_to_framework.py` — confirm all 5 gates pass and the bundle is written to `exports/fsa_v2_verified/`.
4. Push branch: `git push -u origin claude/dev-sandbox-main`. Pause and ask you to review before any merge to `main`.

### Phase 5 — Port to `claude/dev-sandbox-v3`

Branch from `v3-bimodal-extension`, then cherry-pick the infrastructure changes from `claude/dev-sandbox-main`. Adapt:

- The 22→? estimated-parameter list in `analyze_identifiability.py` (v3 adds bimodal-training params).
- The expected basin in each Banister-horizon scenario, since the bimodal extension changes the training-effect dynamics.
- `EXPECTED_END_B` thresholds in `scenarios/_common.py`.

Run the same end-to-end verification. Push as `claude/dev-sandbox-v3`. Pause for review.

### Phase 6 — Port to `claude/dev-sandbox-v4`

Same as Phase 5 but branched from `v4-bimodal-variable-dose-extension`. Adapt scenarios for the variable-dose changes. Push as `claude/dev-sandbox-v4`. Pause for review.

### Phase 7 — Documentation + parent CLAUDE.md

After all three working branches are pushed and you've reviewed:

- Add an "**FSA dev sandbox**" subsection to `/home/ajay/Repos/python-smc2-filtering-control/CLAUDE.md` mirroring the SWAT one. Include the FSA dependency tree:
  ```
  MPC bench (version_2/tools/bench_smc_*fsa*.py)
    ├── needs ControlSpec  → from control.py
    ├── needs the FILTER   → from estimation.py
    └── needs the PLANT    → from _plant.py
                                └── needs DEFAULT_PARAMS, DEFAULT_INIT,
                                    BINS_PER_DAY, drift, noise_scale_fn,
                                    generate_phi_sub_daily, gen_obs_*
                                    → from simulation.py
  ```
- Note that `_phi_burst.py` is also load-bearing for the plant (sub-daily Φ expansion).
- List the dev sandbox URL: `github.com/ajaytalati/FSA_model_dev`.

This step does NOT touch the dev repo — only the parent repo's CLAUDE.md.

---

## Critical files to modify (or add)

**Per branch under `/home/ajay/Repos/FSA_model_dev/`:**

- `pyproject.toml` (new)
- `LICENSE` (new, MIT)
- `.gitignore` (new)
- `CHANGELOG.md` (new)
- `tools/{analyze_identifiability,audit_stiffness,verify_likelihood,verify_controller,export_to_framework}.py` (new, 5 files)
- `tests/{test_reconciliation,test_obs_consistency,test_plant_regression_scenarios}.py` (new, 3 files)
- `scenarios/{_common,42d_horizon_max_sustainable,56d_horizon_periodised,84d_horizon_long_block}.py` (new, 4 files)
- `models/fsa_high_res/simulation.py` (one-line edit: import path change)
- `smc2fc/` folder (deleted)
- `simulator/` folder (deleted)
- `bug_reports/` folder (created, initially empty — placeholder for triage docs)
- `exports/fsa_v2_verified/` folder (auto-created by `export_to_framework.py`)

**In the parent repo `/home/ajay/Repos/python-smc2-filtering-control/`:**

- `CLAUDE.md` — new "FSA dev sandbox" subsection alongside the SWAT one.

---

## Existing functions / utilities being reused (reference paths)

These are the SWAT-side pieces being mirrored, NOT touched:

- `swat_model_factory/scenarios/_common.py:_apply_dropout` — copied verbatim into FSA's `_common.py`. Pure numpy, no deps.
- `swat_model_factory/tools/audit_stiffness.py` structure — vmapped Jacobian + spectral radius audit. Adapt for FSA's `(B, F, A, Φ)` 4-D grid.
- `swat_model_factory/tools/export_to_framework.py:_REPO_ROOT` pattern — anchor all paths to script location.
- `swat_model_factory/tests/test_obs_consistency.py:_est_log_lik_at` — channel-isolation pattern (set one channel's `*_present` mask to 1, others to 0). Adapt the channel set for FSA.
- `version_2/models/fsa_high_res/simulation.py` and friends — these are the model files being validated; the dev sandbox imports from them.

---

## Verification — how to test the changes end-to-end

After Phase 4 (main branch infrastructure), the success test is:

```bash
# Fresh-clone simulation
cd /tmp
rm -rf FSA_test_install
git clone -b claude/dev-sandbox-main https://github.com/ajaytalati/FSA_model_dev FSA_test_install
cd FSA_test_install

# Install + run gates
pip install -e ".[test]"
JAX_ENABLE_X64=True JAX_PLATFORMS=cpu pytest tests/ -v
JAX_ENABLE_X64=True JAX_PLATFORMS=cpu python tools/export_to_framework.py

# Negative test — re-introduce a known bug, prove the test catches it
# (e.g. drop subject-offset from gen_obs_hr); confirm pytest fails loudly.
```

Expected: all tests pass, all 5 export gates pass, bundle written to `exports/fsa_v2_verified/`. Negative test fails the obs-consistency test with a clear message.

After Phases 5 and 6, the same verification is run on the `claude/dev-sandbox-v3` and `claude/dev-sandbox-v4` branches.

---

## Open questions deferred to implementation time

1. **Truth Φ schedules and expected basins for the 3 Banister-horizon scenarios.** Phase 3 will read parent repo's `version_2/outputs/fsa_high_res/experiments/*/CHANGELOG.md` to pin numbers; if no usable reference exists, will return to you with a focused question.
2. **v3 / v4 expected-basin adaptations.** Phase 5/6 will read each branch's actual model edits + any `outputs/` reference data to set basin thresholds; if ambiguous, will return to you.

These are flagged as "ask later" rather than "guess now" per the verify-before-assert rule.
