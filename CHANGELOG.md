# Changelog — FSA Model Dev Sandbox (v4 / v5 working branch)

All notable changes to this dev sandbox on `claude/dev-sandbox-v4` are
recorded here. Newest entries at the top.

This branch tracks the v4 / v5 line of FSA development (the bimodal
training extension + Variable-Dose Busso 2003 + the v5 closed-island
basin topology). The peer working branch `claude/dev-sandbox-main` does
the same job for the v2 baseline.

Each entry lists the local time it was made, the author, and the change
with the reason behind it. The author is the person who decided / signed
off the change; an "(assistant)" line names the tool that did the typing
where relevant.

Dates are in `YYYY-MM-DD`; times are local (BST unless otherwise noted).

---

## 2026-05-04

### 23:53 BST — Phase 1 (v4 branch): install metadata + drop bundled smc2fc/simulator stubs
- **Author:** Ajay Talati (per the approved plan; v4 branch port begins now that the other agent's v5 work has landed at `14b819d`)
- **Assistant:** Claude Code (Opus 4.7, 1M context)
- **Branch:** `claude/dev-sandbox-v4` (branched from `v4-bimodal-variable-dose-extension` at `14b819d`); the existing `main`, `v3-bimodal-extension`, `v4-bimodal-variable-dose-extension` branches are NOT touched.
- **Files added:**
  - [pyproject.toml](pyproject.toml) — declares `smc2fc` (pulled from `github.com/ajaytalati/python-smc2-filtering-control@master`) + `matplotlib>=3.8`. Flat 4-package layout. Pytest config under `[tool.pytest.ini_options]`.
  - [LICENSE](LICENSE) — MIT, attributed to Ajay Talati.
  - [CHANGELOG.md](CHANGELOG.md) — this file.
  - [tests/__init__.py](tests/__init__.py), [tools/__init__.py](tools/__init__.py) — empty marker files for `setuptools.find_packages`.
  - [tests/conftest.py](tests/conftest.py) — `collect_ignore` list excluding 4 pre-existing legacy test files (`test_artifacts.py`, `test_fsa_physics.py`, `test_fsa_v3_physics.py`, `test_v3_identifiability.py`) that import v2/v3 model symbols (`HIGH_RES_FSA_V2_*`, `HIGH_RES_FSA_V3_*`) which were removed when the model migrated to v4/v5. The files are pre-broken on this branch — confirmed by checking out plain v4 and seeing the same import errors before any of my changes. Files kept for historical reference.
  - [claude_plans/FSA_dev_sandbox_mirror_the_SWAT_model_dev_pattern_v4_branch_2026-05-04_2353.md](claude_plans/) — archived plan + audit trail per the global CLAUDE.md rule.
- **Files updated:**
  - [.gitignore](.gitignore) — extended the existing 21-line gitignore (the other agent's version) with `*.py[cod]`, build artefacts, JAX cache, additional pytest/mypy/ruff cache dirs, venv folders, editor / OS noise. Original entries preserved.
  - 5 import-path updates: `models/fsa_high_res/simulation.py:15`, `tests/test_fsa_physics.py:9`, `tests/test_fsa_v3_physics.py:9`, `tests/test_fsa_v4_physics.py:9`, `examples/run_fsa_simulation.py:9` — each changed `from simulator.*` → `from smc2fc.simulator.*` so imports resolve to real smc2fc instead of the bundled stubs.
- **Files deleted (`git rm -r`):**
  - `simulator/` — 4 .py files duplicating `smc2fc.simulator`. The dev repo's bundled copy was an older byte-near-identical mirror.
  - `smc2fc/` — bundled stub package (4 modules + a 4-file `control/` subfolder). Stubs were a ~144-line subset of real `smc2fc` package; FSA's signatures are forward-compatible.
- **Why:** The v4 branch had no installable metadata (just like the v2 main branch did before Phase 1 of the master plan). The bundled `smc2fc/` and `simulator/` folders were destined to drift from the real package. After this commit, `pip install -e .` pulls real `smc2fc` from github — single source of truth, no vendored copies.
- **Verified by:**
  - `pip install -e ".[test]"` from inside the dev repo → pulled `smc2fc` from github, installed `fsa-model-dev-0.5.0`. No errors.
  - `cd /tmp && pytest /home/ajay/Repos/FSA_model_dev/tests/ -v` → **5/5 passed in 10.4 s** (1 v4 physics + 4 v5 smoke). The 4 legacy v2/v3 tests are correctly excluded by `conftest.py:collect_ignore`.
  - The v5 model files in `models/fsa_high_res/control_v5.py` were unaffected by this commit (they only depend on the `models.fsa_high_res._dynamics` package and stdlib/numpy/scipy/jax — no smc2fc imports beyond what `_dynamics` already routes through).

---
