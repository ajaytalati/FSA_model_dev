"""FSA-v2 Port-Package Exporter — runs all 5 validation gates + bundles.

Pillars (mirrors the SWAT_model_dev pattern):
    1. Identifiability (FIM)        — analyze_identifiability.compute_fim
    2. Stability (Stiffness)        — audit_stiffness.audit_stiffness
    3. Reconciliation (Mirror)      — tests/test_reconciliation.py
    4. Likelihood Sanity            — verify_likelihood.verify_likelihood
    5. Controller Sandbox           — verify_controller.verify_controller

If all 5 pass, copies the FSA model files into
`exports/fsa_v2_verified/model/` and writes a `MANIFEST.json` recording
the live values of each pillar's results (no hard-coded numbers — every
field comes from a real check that just ran).

Always uses absolute paths anchored to the repo root, so the script
works no matter what cwd it's invoked from. (This was the
`FileNotFoundError: 'models/...'` bug we caught and fixed in SWAT.)
"""
import os
import sys
import json
import shutil
from pathlib import Path

# Force CPU + X64 for deterministic gate output.
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'True'

import numpy as np

# Anchor every path to the dev-repo root (parent of `tools/`) so the
# script works no matter what cwd it's invoked from. Without this, an
# AI agent or CI runner calling `python tools/export_to_framework.py`
# from any other directory would FileNotFoundError on the model copy.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from tools.analyze_identifiability import compute_fim, FIM_DIAG_THRESHOLD
from tools.audit_stiffness import audit_stiffness
from tools.verify_likelihood import verify_likelihood
from tools.verify_controller import verify_controller
from tests.test_reconciliation import (
    test_plant_and_estimator_share_drift,
)


def export_package(output_name: str = "fsa_v2_verified"):
    print(f"=== FSA-v2 Port-Package Exporter: {output_name} ===\n")

    export_dir = _REPO_ROOT / "exports" / output_name
    export_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # ── Pillar 1: Identifiability ──────────────────────────────────
    print("Checking Pillar I: Identifiability...")
    fim, names, per_channel = compute_fim(n_days=1.0, Phi_const=1.0)
    diag = np.diag(fim)
    identifiable = [n for n, d in zip(names, diag) if d > FIM_DIAG_THRESHOLD]
    eigvals = np.linalg.eigvalsh(fim)
    rank = int(np.sum(eigvals > 1e-8))
    results['identifiability'] = {
        'rank':                rank,
        'cond':                float(eigvals[-1] / max(eigvals[0], 1e-18)),
        'identifiable_subset': identifiable,
        'n_identifiable':      len(identifiable),
        'note':                'FIM under 1-day constant-Phi=1.0 rollout, all 4 obs channels included',
        'passed':              rank >= 8,
    }
    print(f"  rank={rank}/{len(names)}, cond={results['identifiability']['cond']:.2e}")
    print(f"  identifiable: {len(identifiable)} of {len(names)} params")

    # ── Pillar 2: Stability ────────────────────────────────────────
    print("\nChecking Pillar II: Stability (Stiffness)...")
    stiff = audit_stiffness()
    results['stability'] = stiff

    # ── Pillar 3: Reconciliation ───────────────────────────────────
    print("\nChecking Pillar III: Plant↔Estimator Reconciliation...")
    try:
        test_plant_and_estimator_share_drift()
        results['reconciliation'] = {'passed': True}
        print("  Plant ↔ Estimator share drift: PASS")
    except AssertionError as e:
        print(f"  FAILED: {e}")
        results['reconciliation'] = {'passed': False, 'error': str(e)}

    # ── Pillar 4: Likelihood Sanity ────────────────────────────────
    print("\nChecking Pillar IV: Likelihood Sanity...")
    try:
        ll = verify_likelihood()
        results['likelihood'] = ll
    except AssertionError as e:
        print(f"  FAILED: {e}")
        results['likelihood'] = {'passed': False, 'error': str(e)}

    # ── Pillar 5: Controller Sandbox ───────────────────────────────
    print("\nChecking Pillar V: Controller Sandbox...")
    try:
        ctl = verify_controller()
        results['controller'] = ctl
    except AssertionError as e:
        print(f"  FAILED: {e}")
        results['controller'] = {'passed': False, 'error': str(e)}

    # ── Final gate ────────────────────────────────────────────────
    all_passed = all(v.get('passed', False) for v in results.values())

    if all_passed:
        print("\n=== ALL GATES PASSED. Bundling package... ===")

        # Bundle the model files
        model_src = _REPO_ROOT / "models" / "fsa_high_res"
        dest_src = export_dir / "model"
        dest_src.mkdir(parents=True, exist_ok=True)

        for f in ["__init__.py", "_dynamics.py", "_phi_burst.py",
                  "_plant.py", "control.py", "estimation.py",
                  "simulation.py"]:
            src_path = model_src / f
            if not src_path.exists():
                print(f"  WARNING: source file {src_path} not found, skipping")
                continue
            shutil.copy(src_path, dest_src / f)

        # Manifest — values pulled from the actual gates above
        manifest = {
            "model":      "FSA-v2 (Banister + autonomic amplitude)",
            "version":    "0.2.0-verified",
            "validation": _serialize(results),
            "export_config": {
                "h_max_mins":          results['stability']['h_max_mins'],
                "n_substeps_for_15min": results['stability']['n_substeps_for_15min'],
                "identifiable_subset": results['identifiability']['identifiable_subset'],
                "fim_rank":            results['identifiability']['rank'],
                "fim_cond":            results['identifiability']['cond'],
            },
        }
        with open(export_dir / "MANIFEST.json", "w") as f:
            json.dump(manifest, f, indent=4)

        print(f"\nPackage exported successfully to: {export_dir}")
    else:
        print("\n!!! EXPORT FAILED: One or more validation gates failed. !!!")
        print(json.dumps(_serialize(results), indent=4))
        sys.exit(1)


def _serialize(obj):
    """Recursively make `results` dict JSON-serialisable.

    Numpy scalars / arrays / tuples / sets are common offenders.
    """
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    if isinstance(obj, set):
        return sorted(_serialize(v) for v in obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


if __name__ == "__main__":
    export_package()
