"""Plant regression smoke tests — drive `StepwisePlant` for 42 / 56 /
84 days under constant daily Φ and assert the run completes with the
end-of-trial `(B, F, A)` finite and inside its physical bounds.

Why this test exists
--------------------
`_plant.py` (the simulator-as-plant) is what the closed-loop MPC drives
at each control-decision step. If `_plant.py`, `_dynamics.py`, or any of
the obs samplers in `simulation.py` quietly drift such that the long
horizon produces a non-finite trajectory or a B that escapes [0, 1] /
F < 0 / A < 0, the MPC will crash or — worse — silently produce wrong
results downstream.

The basin classifier in `scenarios/_common.py:_smoke_basin_check` is
currently SMOKE-ONLY (finite + physical bounds). Numerical thresholds
on end-of-trial B / F / A will be added in a follow-up commit once the
FSA team pins the canonical reference values per horizon.

Cost
----
Each scenario advances the plant for 42-84 days × 96 bins/day = 4032 to
8064 bins. On CPU each takes ~2 s; the full 3-scenario sweep adds ~6 s
to the pytest run. Acceptable for the regression net protecting the
plant.

The scenarios also write artefact files to `outputs/fsa/<scenario>/`
as a side effect — the artefacts let you visually inspect what the
plant produced if a smoke check fails.
"""
import os
import sys
from pathlib import Path

# Force CPU + X64 BEFORE importing JAX.
os.environ['JAX_ENABLE_X64'] = 'True'
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import pytest

# Make the dev-repo root importable so `from scenarios._common import …`
# works even when pytest is invoked from outside the repo.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scenarios._common import HORIZON_PHI, run_fsa_scenario


@pytest.mark.parametrize("scenario_key", sorted(HORIZON_PHI.keys()))
def test_scenario_smoke_basin(scenario_key):
    """Each Banister-horizon scenario must complete without crashing
    and end with the latent state finite + inside physical bounds.

    `run_fsa_scenario` returns 0 on smoke OK, 1 on smoke failure.
    """
    rc = run_fsa_scenario(scenario_key)
    assert rc == 0, (
        f"Plant regression: scenario {scenario_key!r} ended in a non-physical "
        f"state. Inspect outputs/fsa/{scenario_key}/trajectory.npz to see the "
        f"trajectory; check whether _plant.py, _dynamics.py, or the obs "
        f"samplers in simulation.py have drifted.")
