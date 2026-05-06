"""Python wrapper around the LEAN4 drift binary.

Spawns the long-lived `fsa_v5_cli` subprocess once per process and
exchanges line-delimited JSON. One drift call ≈ a few hundred
microseconds end-to-end (dominated by the JSON round-trip, not the
arithmetic), which is plenty fast for Hypothesis-driven differential
testing.

Usage:
    from lean.python_bridge import drift_lean
    deriv = drift_lean(state, phi, params)   # numpy arrays in, np array out
"""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import threading
from pathlib import Path

import numpy as np

# Canonical parameter key order matches the LEAN4 ``Params`` structure
# in ``lean/Fsa/V5/Types.lean``. Kept here as a list so callers can
# convert dict-shaped params reliably without reordering surprises.
PARAM_KEYS: tuple[str, ...] = (
    "tau_B", "kappa_B", "epsilon_AB",
    "tau_S", "kappa_S", "epsilon_AS",
    "tau_F", "lambda_A",
    "KFB_0", "KFS_0", "tau_K", "mu_K",
    "mu_0", "mu_B", "mu_S", "mu_F", "mu_FF", "eta",
    "sigma_B", "sigma_S", "sigma_F", "sigma_A", "sigma_K",
    "B_dec", "S_dec", "mu_dec_B", "mu_dec_S", "n_dec",
)

# Obs-channel parameter key order matches LEAN4's ``ObsParams`` in
# ``lean/Fsa/V5/Obs.lean``. NB: ``sigma_S_obs`` (stress obs noise) is
# distinct from PARAM_KEYS' ``sigma_S`` (state-noise) — that is the
# whole point of the Bug 1 structural fix.
OBS_PARAM_KEYS: tuple[str, ...] = (
    "HR_base", "kappa_B_HR", "alpha_A_HR", "beta_C_HR", "sigma_HR",
    "k_C", "k_A", "c_tilde",
    "S_base", "k_F", "k_A_S", "beta_C_S", "sigma_S_obs",
    "mu_step0", "beta_B_st", "beta_F_st", "beta_A_st", "beta_C_st", "sigma_st",
    "beta_S_VL", "beta_F_VL", "sigma_VL",
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LEAN_DIR = _REPO_ROOT / "lean"
_DEFAULT_BIN = _LEAN_DIR / ".lake" / "build" / "bin" / "fsa_v5_cli"


class LeanDriftClient:
    """Long-lived subprocess client for the LEAN4 drift binary.

    Thread-safe: a single lock serialises send/recv pairs since the
    subprocess can only handle one request/response cycle at a time.
    """

    def __init__(self, binary_path: Path | str | None = None) -> None:
        path = Path(binary_path) if binary_path is not None else _DEFAULT_BIN
        if not path.exists():
            raise FileNotFoundError(
                f"LEAN4 binary not found at {path}. Run `lake build` in "
                f"{_LEAN_DIR} first."
            )
        self._path = path
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._open()
        atexit.register(self.close)

    def _open(self) -> None:
        self._proc = subprocess.Popen(
            [str(self._path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
            cwd=str(_LEAN_DIR),
        )

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin and not self._proc.stdin.closed:
                self._proc.stdin.close()
            self._proc.wait(timeout=2.0)
        except Exception:
            try:
                self._proc.terminate()
            except Exception:
                pass
        finally:
            self._proc = None

    def _round_trip(self, request: dict) -> dict:
        """Send one request line and receive one response line."""
        if self._proc is None or self._proc.poll() is not None:
            raise RuntimeError("LEAN subprocess is not running")
        line = json.dumps(request, separators=(",", ":")) + "\n"
        with self._lock:
            assert self._proc is not None
            assert self._proc.stdin is not None
            assert self._proc.stdout is not None
            self._proc.stdin.write(line)
            self._proc.stdin.flush()
            response_line = self._proc.stdout.readline()
        if not response_line:
            stderr = ""
            try:
                if self._proc and self._proc.stderr:
                    stderr = self._proc.stderr.read() or ""
            except Exception:
                pass
            raise RuntimeError(
                f"LEAN subprocess produced no output. stderr: {stderr!r}"
            )
        out = json.loads(response_line)
        if "error" in out:
            raise RuntimeError(f"LEAN error: {out['error']}")
        return out

    def drift(
        self,
        state: np.ndarray,
        phi: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        """Call the LEAN drift function. Returns shape-(6,) ndarray."""
        if state.shape != (6,):
            raise ValueError(f"state must have shape (6,), got {state.shape}")
        if phi.shape != (2,):
            raise ValueError(f"phi must have shape (2,), got {phi.shape}")
        params_subset = {k: float(params[k]) for k in PARAM_KEYS}
        request = {
            "fn": "drift",
            "state": [float(x) for x in state],
            "phi": [float(x) for x in phi],
            "params": params_subset,
        }
        out = self._round_trip(request)
        deriv = out["deriv"]
        if len(deriv) != 6:
            raise RuntimeError(f"deriv must have length 6, got {len(deriv)}")
        return np.asarray(deriv, dtype=np.float64)

    def mu_bar(self, A: float, phi: np.ndarray, params: dict) -> float:
        """Call LEAN's `muBar` — the slow-manifold Stuart-Landau coefficient
        $\\bar\\mu(A; \\Phi)$. Returns a scalar."""
        if phi.shape != (2,):
            raise ValueError(f"phi must have shape (2,), got {phi.shape}")
        params_subset = {k: float(params[k]) for k in PARAM_KEYS}
        request = {
            "fn": "muBar",
            "A": float(A),
            "phi": [float(x) for x in phi],
            "params": params_subset,
        }
        out = self._round_trip(request)
        return float(out["muBar"])

    def schedule(
        self,
        theta: np.ndarray,
        phi_design: np.ndarray,
        c_phi: float,
        phi_max: float,
        n_anchors: int,
    ) -> np.ndarray:
        """Call LEAN's `scheduleFromTheta`. Returns shape (n_steps, 2)."""
        request = {
            "fn": "schedule",
            "theta": [float(x) for x in theta],
            "phi_design": [[float(x) for x in row] for row in phi_design],
            "c_phi": float(c_phi),
            "phi_max": float(phi_max),
            "n_anchors": int(n_anchors),
        }
        out = self._round_trip(request)
        sched = out["schedule"]
        return np.asarray(sched, dtype=np.float64)

    def em_step(
        self,
        state: np.ndarray,
        phi: np.ndarray,
        params: dict,
        sigma_diag: np.ndarray,
        dt: float,
        noise: np.ndarray,
    ) -> np.ndarray:
        """Call LEAN's `emStep`. Returns shape-(6,) next state."""
        if state.shape != (6,) or noise.shape != (6,) or sigma_diag.shape != (6,):
            raise ValueError("state, sigma_diag, noise must all be shape (6,)")
        params_subset = {k: float(params[k]) for k in PARAM_KEYS}
        request = {
            "fn": "emStep",
            "state": [float(x) for x in state],
            "phi": [float(x) for x in phi],
            "params": params_subset,
            "sigma_diag": [float(x) for x in sigma_diag],
            "dt": float(dt),
            "noise": [float(x) for x in noise],
        }
        out = self._round_trip(request)
        return np.asarray(out["next_state"], dtype=np.float64)

    def _obs_call(self, fn_name: str, key: str,
                   state: np.ndarray, C: float, obs_params: dict) -> float:
        op = {k: float(obs_params[k]) for k in OBS_PARAM_KEYS}
        request = {
            "fn": fn_name,
            "state": [float(x) for x in state],
            "C": float(C),
            "obs_params": op,
        }
        return float(self._round_trip(request)[key])

    def hr_mean(self, state: np.ndarray, C: float, obs_params: dict) -> float:
        return self._obs_call("hrMean", "hr_mean", state, C, obs_params)

    def sleep_prob(self, state: np.ndarray, C: float, obs_params: dict) -> float:
        return self._obs_call("sleepProb", "sleep_prob", state, C, obs_params)

    def stress_mean(self, state: np.ndarray, C: float, obs_params: dict) -> float:
        return self._obs_call("stressMean", "stress_mean", state, C, obs_params)

    def steps_log_mean(self, state: np.ndarray, C: float, obs_params: dict) -> float:
        return self._obs_call("stepsLogMean", "steps_log_mean", state, C, obs_params)

    def volume_load_mean(self, state: np.ndarray, obs_params: dict) -> float:
        op = {k: float(obs_params[k]) for k in OBS_PARAM_KEYS}
        request = {
            "fn": "volumeLoadMean",
            "state": [float(x) for x in state],
            "obs_params": op,
        }
        return float(self._round_trip(request)["vl_mean"])

    def find_a_sep(self, phi: np.ndarray, params: dict) -> float:
        """Call LEAN's `findASep` — the bistable separatrix root-finder.
        Returns -inf (mono-stable healthy), +inf (collapsed), or a finite
        scalar in (0, 2)."""
        if phi.shape != (2,):
            raise ValueError(f"phi must have shape (2,), got {phi.shape}")
        params_subset = {k: float(params[k]) for k in PARAM_KEYS}
        request = {
            "fn": "findASep",
            "phi": [float(x) for x in phi],
            "params": params_subset,
        }
        out = self._round_trip(request)
        return float(out["A_sep"])


# Module-level singleton client (lazy). Most diff tests use a single
# client; spawning multiple is wasteful.
_default_client: LeanDriftClient | None = None
_default_lock = threading.Lock()


def _get_default_client() -> LeanDriftClient:
    global _default_client
    if _default_client is None:
        with _default_lock:
            if _default_client is None:
                _default_client = LeanDriftClient()
    return _default_client


def drift_lean(
    state: np.ndarray, phi: np.ndarray, params: dict
) -> np.ndarray:
    """Call the LEAN reference drift through the default singleton client."""
    return _get_default_client().drift(state, phi, params)


def mu_bar_lean(A: float, phi: np.ndarray, params: dict) -> float:
    """Call the LEAN `muBar` reference through the default singleton client."""
    return _get_default_client().mu_bar(A, phi, params)


def find_a_sep_lean(phi: np.ndarray, params: dict) -> float:
    """Call the LEAN `findASep` reference through the default singleton client."""
    return _get_default_client().find_a_sep(phi, params)


def schedule_lean(theta, phi_design, c_phi, phi_max, n_anchors) -> np.ndarray:
    return _get_default_client().schedule(theta, phi_design, c_phi, phi_max, n_anchors)


def em_step_lean(state, phi, params, sigma_diag, dt, noise) -> np.ndarray:
    return _get_default_client().em_step(state, phi, params, sigma_diag, dt, noise)


def hr_mean_lean(state, C, obs_params) -> float:
    return _get_default_client().hr_mean(state, C, obs_params)


def sleep_prob_lean(state, C, obs_params) -> float:
    return _get_default_client().sleep_prob(state, C, obs_params)


def stress_mean_lean(state, C, obs_params) -> float:
    return _get_default_client().stress_mean(state, C, obs_params)


def steps_log_mean_lean(state, C, obs_params) -> float:
    return _get_default_client().steps_log_mean(state, C, obs_params)


def volume_load_mean_lean(state, obs_params) -> float:
    return _get_default_client().volume_load_mean(state, obs_params)
