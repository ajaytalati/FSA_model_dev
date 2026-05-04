"""End-to-end Banister-horizon scenario runner — exercises `StepwisePlant`.

Each per-horizon script in this folder calls `run_fsa_scenario('42d')`
(or '56d', '84d'). The runner:

1. Reads the constant daily Φ for that horizon from `HORIZON_PHI`.
2. Builds a `StepwisePlant` with `DEFAULT_PARAMS` and the scenario's
   initial state (cold-start Stage-D: B_0=0.05, F_0=0.30, A_0=0.10).
3. Drives the plant for `n_days * BINS_PER_DAY` bins under the daily
   piecewise-constant Φ (expanded into morning-loaded sub-bins by
   `_phi_burst.py`).
4. Applies per-bin Bernoulli dropout to HR + stress (sleep + steps
   preserved) via the small inlined `_apply_dropout` helper.
5. Writes a packaged-style artefact to `outputs/fsa/<scenario>/`:
   - `trajectory.npz` — (n_bins, 3) latents [B, F, A] + per-bin Φ + C
   - `obs/obs_HR.npz`, `obs/obs_sleep.npz`, `obs/obs_stress.npz`,
     `obs/obs_steps.npz`
6. Prints a one-screen summary.

Smoke-only basin classifier
---------------------------
`EXPECTED_END_B` is currently smoke-only (pass = the run finished
without crashing and the latent trajectory stayed in physical bounds).
Numerical basin thresholds will be added in a follow-up commit once
the canonical Banister-horizon expected behaviour is pinned to live
reference data from the parent repo's bench outputs.

This intentionally bypasses any psim machinery — the dev repo stands
alone on `smc2fc` + JAX/numpy. The plant + obs samplers exercised here
are the same code path the closed-loop MPC bench uses in production.
A regression here means a regression in the closed-loop bench downstream.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


# Anchor every path to the dev-repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# Per-horizon scenario specs. Each entry: (n_days, daily_phi).
# Values are illustrative starting points; tune as the FSA team
# converges on the canonical max-sustainable-load Φ for each horizon.
HORIZON_PHI = {
    '42d': (42, 1.5),    # Banister chronic timescale; max-sustainable-load
    '56d': (56, 1.2),    # mid-horizon; periodised reduction
    '84d': (84, 1.0),    # long-horizon block; lower steady load
}


# Smoke-only "did it survive" basin checks. To be replaced with real
# numerical thresholds once the FSA team pins the canonical end-of-trial
# B/F/A values per horizon (currently flagged as an open question in
# the implementation plan at `claude_plans/`).
def _smoke_basin_check(end_state: np.ndarray) -> tuple[bool, str]:
    """Return (passed, message). Smoke-only — physical bounds + finite."""
    B, F, A = end_state
    if not np.all(np.isfinite(end_state)):
        return False, f"non-finite end state: B={B}, F={F}, A={A}"
    if B < 0.0 or B > 1.0:
        return False, f"B={B:.3f} outside [0,1]"
    if F < 0.0:
        return False, f"F={F:.3f} negative"
    if A < 0.0:
        return False, f"A={A:.3f} negative"
    return True, "smoke OK (finite, physical bounds respected)"


# Per-channel Bernoulli dropout — pure numpy, no external deps.
# Same helper as in SWAT_model_dev (formerly imported from psim;
# inlined so the dev repo is self-contained on smc2fc + JAX/numpy alone).
def _apply_dropout(obs_data: dict, channels, rate: float = 0.05, seed: int = 42):
    """Drop a fraction of observations per channel, in place."""
    rng = np.random.default_rng(seed)
    for ch in channels:
        if ch not in obs_data:
            continue
        d = obs_data[ch]
        if 't_idx' not in d or len(d['t_idx']) == 0:
            continue
        idx = d['t_idx']
        keep = rng.random(len(idx)) > rate
        d['t_idx'] = idx[keep]
        for key in list(d.keys()):
            if key == 't_idx':
                continue
            v = d[key]
            if hasattr(v, '__len__') and len(v) == len(keep):
                d[key] = np.asarray(v)[keep]
    return obs_data


def run_fsa_scenario(
    scenario_key: str,
    *,
    dropout_rate: float = 0.05,
    seed: int = 42,
) -> int:
    """Run one Banister-horizon FSA scenario end-to-end via StepwisePlant.

    Returns 0 on smoke-OK, 1 on smoke failure.
    """
    # Lazy imports so the module is itself import-clean.
    from models.fsa_high_res.simulation import (
        DEFAULT_PARAMS, DEFAULT_INIT, BINS_PER_DAY, DT_BIN_DAYS)
    from models.fsa_high_res._plant import StepwisePlant

    if scenario_key not in HORIZON_PHI:
        raise KeyError(f"Unknown scenario {scenario_key!r}; "
                        f"choices: {sorted(HORIZON_PHI.keys())}")
    n_days, daily_phi = HORIZON_PHI[scenario_key]
    stride_bins = n_days * BINS_PER_DAY

    print(f"=== FSA / {scenario_key} ===")
    print(f"  horizon: {n_days} days = {stride_bins} bins of {DT_BIN_DAYS * 24 * 60:.0f} min")
    print(f"  init:    B={DEFAULT_INIT['B_0']:.3f}, "
          f"F={DEFAULT_INIT['F_0']:.3f}, "
          f"A={DEFAULT_INIT['A_0']:.3f}  (Stage-D cold-start)")
    print(f"  control: constant daily Phi = {daily_phi}")

    # ── 1. Build plant + run forward sim ──────────────────────────
    print(f"\n[1/3] Forward-simulate plant + sample 4 channels via "
          f"StepwisePlant.advance() ...")
    plant = StepwisePlant(
        truth_params=dict(DEFAULT_PARAMS),
        state=np.array([DEFAULT_INIT['B_0'],
                         DEFAULT_INIT['F_0'],
                         DEFAULT_INIT['A_0']], dtype=np.float64),
        seed_offset=seed,
        dt=DT_BIN_DAYS,
    )
    Phi_daily = np.full(n_days, daily_phi, dtype=np.float64)
    out = plant.advance(stride_bins=stride_bins, Phi_daily=Phi_daily)
    traj = out['trajectory']
    print(f"   trajectory: shape {traj.shape}")
    print(f"   B: range [{traj[:, 0].min():.3f}, {traj[:, 0].max():.3f}], "
          f"start {traj[0, 0]:.3f} → end {traj[-1, 0]:.3f}")
    print(f"   F: range [{traj[:, 1].min():.3f}, {traj[:, 1].max():.3f}], "
          f"start {traj[0, 1]:.3f} → end {traj[-1, 1]:.3f}")
    print(f"   A: range [{traj[:, 2].min():.3f}, {traj[:, 2].max():.3f}], "
          f"start {traj[0, 2]:.3f} → end {traj[-1, 2]:.3f}")
    print(f"   obs samples: HR={len(out['obs_HR']['t_idx'])}, "
          f"sleep={len(out['obs_sleep']['t_idx'])}, "
          f"stress={len(out['obs_stress']['t_idx'])}, "
          f"steps={len(out['obs_steps']['t_idx'])}")

    # ── 2. Apply dropout to HR + stress ───────────────────────────
    print(f"\n[2/3] Apply {dropout_rate * 100:.0f}% dropout on hr/stress "
          f"(sleep + steps preserved)")
    _apply_dropout({'hr':     out['obs_HR'],
                     'stress': out['obs_stress']},
                    channels=['hr', 'stress'],
                    rate=dropout_rate,
                    seed=seed + 200)
    print(f"   after dropout: HR={len(out['obs_HR']['t_idx'])}, "
          f"stress={len(out['obs_stress']['t_idx'])}")

    # ── 3. Save artefact ──────────────────────────────────────────
    out_dir = _REPO_ROOT / 'outputs' / 'fsa' / scenario_key
    out_dir.mkdir(parents=True, exist_ok=True)
    obs_dir = out_dir / 'obs'
    obs_dir.mkdir(exist_ok=True)
    print(f"\n[3/3] Save artefact to {out_dir}")

    np.savez(
        out_dir / 'trajectory.npz',
        trajectory=traj,
        Phi_per_bin=out['Phi']['Phi_value'],
        C_per_bin=out['C']['C_value'],
    )
    np.savez(obs_dir / 'obs_HR.npz', **out['obs_HR'])
    np.savez(obs_dir / 'obs_sleep.npz', **out['obs_sleep'])
    np.savez(obs_dir / 'obs_stress.npz', **out['obs_stress'])
    np.savez(obs_dir / 'obs_steps.npz', **out['obs_steps'])

    # ── 4. Smoke-only basin verdict ───────────────────────────────
    end_state = traj[-1]
    ok, msg = _smoke_basin_check(end_state)
    verdict = "SMOKE OK" if ok else "SMOKE FAIL — investigate plant / dynamics"
    print(f"\nDone. Artefact at: {out_dir}")
    print(f"  end-of-trial state: B={end_state[0]:.3f}, F={end_state[1]:.3f}, "
          f"A={end_state[2]:.3f}  →  {verdict}")
    if not ok:
        print(f"  reason: {msg}")
    return 0 if ok else 1
