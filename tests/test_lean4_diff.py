"""Differential test: LEAN4 reference vs Python implementation of FSA-v5 drift.

Per the LEAN4-first charter (LaTex_docs/lean4_first_charter.pdf §5 step 6):
both implementations are evaluated on a battery of inputs sampled from
the physiological-bound box. Any disagreement beyond the tolerance
threshold is by construction a Python bug.

Tolerance: 1e-6 absolute (single-step drift). The integrated-trajectory
1e-4 threshold from the charter applies to multi-step rollouts; this
file tests the single drift call only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Make sure JAX runs in fp64 to match the LEAN4 Float (IEEE-754 double).
jax.config.update("jax_enable_x64", True)

# Path setup so the test runs against the in-repo Python.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
sys.path.insert(0, str(_REPO_ROOT))

from hypothesis import given, settings, strategies as st  # noqa: E402

from lean.python_bridge import LeanDriftClient  # noqa: E402
from models.fsa_high_res._dynamics import (  # noqa: E402
    TRUTH_PARAMS_V5,
    drift_jax,
)
from models.fsa_high_res.control_v5 import (  # noqa: E402
    _jax_mu_bar,
    _jax_find_A_sep,
)


# Single LEAN client per pytest session — the subprocess is expensive to
# launch but cheap to reuse.
@pytest.fixture(scope="session")
def lean_client() -> LeanDriftClient:
    return LeanDriftClient()


# Physiological-bound box for hypothesis sampling. Values match the
# state-bounds the production estimation pipeline uses; everything inside
# this box is a "valid" point at which both implementations should agree.
def _state_strategy():
    return st.tuples(
        st.floats(min_value=0.001, max_value=1.0),  # B in (0, 1]
        st.floats(min_value=0.001, max_value=1.0),  # S in (0, 1]
        st.floats(min_value=0.0, max_value=10.0),   # F in [0, 10]
        st.floats(min_value=0.0, max_value=5.0),    # A in [0, 5]
        st.floats(min_value=0.0, max_value=1.0),    # KFB in [0, 1]
        st.floats(min_value=0.0, max_value=1.0),    # KFS in [0, 1]
    ).map(lambda t: np.asarray(t, dtype=np.float64))


def _phi_strategy():
    # Phi rates from sedentary to over-training (match charter §2.1
    # closed-island regime sweep at LaTeX §10.4 Table).
    return st.tuples(
        st.floats(min_value=0.0, max_value=2.0),
        st.floats(min_value=0.0, max_value=2.0),
    ).map(lambda t: np.asarray(t, dtype=np.float64))


# Tolerance: 1e-6 absolute. The charter's 1e-4 integrated-trajectory
# threshold doesn't apply to a single drift step.
SINGLE_STEP_TOL = 1e-6


def _drift_python_v5(state: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Wrap the JAX drift_jax to take/return numpy arrays under TRUTH_PARAMS_V5."""
    p_jax = {k: jnp.float64(float(v)) for k, v in TRUTH_PARAMS_V5.items()}
    y_jax = jnp.asarray(state, dtype=jnp.float64)
    phi_jax = jnp.asarray(phi, dtype=jnp.float64)
    out = drift_jax(y_jax, p_jax, phi_jax)
    return np.asarray(out, dtype=np.float64)


def test_drift_at_healthy_island_inside(lean_client):
    """Sanity: at the LaTeX §10.4 healthy-island reference point, both
    implementations agree to numerical precision."""
    state = np.array([0.50, 0.45, 0.20, 0.45, 0.06, 0.07], dtype=np.float64)
    phi = np.array([0.30, 0.30], dtype=np.float64)

    out_lean = lean_client.drift(state, phi, TRUTH_PARAMS_V5)
    out_python = _drift_python_v5(state, phi)

    np.testing.assert_allclose(
        out_lean, out_python, atol=SINGLE_STEP_TOL, rtol=0.0,
        err_msg=f"Drift disagrees at healthy-island reference point.\n"
                f"  LEAN:   {out_lean}\n  Python: {out_python}\n"
                f"  diff:   {out_lean - out_python}"
    )


def test_drift_at_sedentary_collapse(lean_client):
    """Sanity: at sedentary Phi=(0,0), both implementations agree."""
    state = np.array([0.10, 0.10, 0.05, 0.10, 0.04, 0.05], dtype=np.float64)
    phi = np.array([0.0, 0.0], dtype=np.float64)
    out_lean = lean_client.drift(state, phi, TRUTH_PARAMS_V5)
    out_python = _drift_python_v5(state, phi)
    np.testing.assert_allclose(
        out_lean, out_python, atol=SINGLE_STEP_TOL, rtol=0.0,
        err_msg=f"Drift disagrees at sedentary point.\n"
                f"  LEAN:   {out_lean}\n  Python: {out_python}"
    )


def test_drift_at_overtrain_runaway(lean_client):
    """Sanity: at over-training Phi=(1,1), both implementations agree."""
    state = np.array([0.50, 0.45, 1.50, 0.30, 0.10, 0.11], dtype=np.float64)
    phi = np.array([1.0, 1.0], dtype=np.float64)
    out_lean = lean_client.drift(state, phi, TRUTH_PARAMS_V5)
    out_python = _drift_python_v5(state, phi)
    np.testing.assert_allclose(
        out_lean, out_python, atol=SINGLE_STEP_TOL, rtol=0.0,
        err_msg=f"Drift disagrees at over-training point.\n"
                f"  LEAN:   {out_lean}\n  Python: {out_python}"
    )


@given(state=_state_strategy(), phi=_phi_strategy())
@settings(max_examples=200, deadline=None)
def test_drift_diff_random(state: np.ndarray, phi: np.ndarray):
    """Hypothesis-driven random sampling. Both implementations must agree
    on every input drawn from the physiological-bound box."""
    # Use a session-shared client (avoid spawning per example).
    from lean.python_bridge import drift_lean

    out_lean = drift_lean(state, phi, TRUTH_PARAMS_V5)
    out_python = _drift_python_v5(state, phi)

    np.testing.assert_allclose(
        out_lean, out_python, atol=SINGLE_STEP_TOL, rtol=0.0,
        err_msg=f"Drift disagrees at state={state}, phi={phi}.\n"
                f"  LEAN:   {out_lean}\n  Python: {out_python}\n"
                f"  diff:   {out_lean - out_python}"
    )


# ===========================================================================
# muBar — slow-manifold Stuart-Landau coefficient $\bar\mu(A; \Phi)$
# ===========================================================================

def _mu_bar_python(A: float, phi: np.ndarray) -> float:
    p_jax = {k: jnp.float64(float(v)) for k, v in TRUTH_PARAMS_V5.items()}
    return float(_jax_mu_bar(jnp.float64(A), jnp.float64(phi[0]), jnp.float64(phi[1]), p_jax))


def test_mubar_at_healthy_island(lean_client):
    """LaTeX §10.4 reference: at Phi=(0.30, 0.30) with A=0,
    mu_bar should be ≈ +0.011 (positive → healthy island)."""
    phi = np.array([0.30, 0.30], dtype=np.float64)
    out_lean = lean_client.mu_bar(0.0, phi, TRUTH_PARAMS_V5)
    out_python = _mu_bar_python(0.0, phi)
    assert abs(out_lean - out_python) < SINGLE_STEP_TOL, (
        f"muBar disagrees at healthy point. LEAN={out_lean}, Python={out_python}"
    )
    # Also check the actual value is approximately +0.011 (LaTeX Table)
    assert out_lean > 0.0, f"healthy regime should give positive muBar; got {out_lean}"


def test_mubar_at_sedentary(lean_client):
    """LaTeX §10.4: Phi=(0,0) with A=0 should give mu_bar ≈ -0.180
    (strong collapse)."""
    phi = np.array([0.0, 0.0], dtype=np.float64)
    out_lean = lean_client.mu_bar(0.0, phi, TRUTH_PARAMS_V5)
    out_python = _mu_bar_python(0.0, phi)
    assert abs(out_lean - out_python) < SINGLE_STEP_TOL
    assert out_lean < 0.0, f"sedentary regime should give negative muBar; got {out_lean}"


@given(
    A=st.floats(min_value=0.0, max_value=2.0),
    phi=_phi_strategy(),
)
@settings(max_examples=200, deadline=None)
def test_mubar_diff_random(A: float, phi: np.ndarray):
    """Random points (A, Phi); LEAN and Python `mu_bar` must agree."""
    from lean.python_bridge import mu_bar_lean

    out_lean = mu_bar_lean(A, phi, TRUTH_PARAMS_V5)
    out_python = _mu_bar_python(A, phi)
    np.testing.assert_allclose(
        out_lean, out_python, atol=SINGLE_STEP_TOL, rtol=0.0,
        err_msg=f"muBar disagrees at A={A}, phi={phi}.\n"
                f"  LEAN={out_lean}, Python={out_python}, diff={out_lean - out_python}"
    )


# ===========================================================================
# findASep — bistable separatrix root-finder
# ===========================================================================
#
# This is where Bug 2's structural prevention lives. The Python
# `_jax_find_A_sep` returns a scalar per (Phi, params) input; in the
# legacy `_compute_cost_internals`, this was vmapped over the schedule
# but NOT over the particle ensemble — collapsing the ensemble to
# particle-0's params. The LEAN4 type signature
# `findASep : BimodalPhi → Params → Float` and the array-of-arrays
# wrapper `aSepGrid : Array Params → Array BimodalPhi → Array (Array Float)`
# in `lean/Fsa/V5/Cost.lean` make the buggy collapse impossible to
# express.

def _find_a_sep_python(phi: np.ndarray) -> float:
    p_jax = {k: jnp.float64(float(v)) for k, v in TRUTH_PARAMS_V5.items()}
    return float(_jax_find_A_sep(jnp.float64(phi[0]), jnp.float64(phi[1]), p_jax))


def test_a_sep_closed_island_table_healthy(lean_client):
    """LaTeX §10.4 Table 1 row: Phi=(0.30, 0.30) → mono-stable healthy
    (A_sep = -inf)."""
    phi = np.array([0.30, 0.30], dtype=np.float64)
    out_lean = lean_client.find_a_sep(phi, TRUTH_PARAMS_V5)
    assert out_lean == float("-inf"), (
        f"healthy regime should give A_sep=-inf, got {out_lean}"
    )


def test_a_sep_closed_island_table_collapsed(lean_client):
    """LaTeX §10.4 Table 1 rows: sedentary, aerobic-only, strength-only,
    over-training all give A_sep=+inf (mono-stable collapsed)."""
    for phi_arr in [
        np.array([0.0, 0.0], dtype=np.float64),
        np.array([0.30, 0.0], dtype=np.float64),
        np.array([0.0, 0.30], dtype=np.float64),
        np.array([1.0, 1.0], dtype=np.float64),
    ]:
        out_lean = lean_client.find_a_sep(phi_arr, TRUTH_PARAMS_V5)
        assert out_lean == float("inf"), (
            f"collapsed regime should give A_sep=+inf at phi={phi_arr}, "
            f"got {out_lean}"
        )


@given(phi=_phi_strategy())
@settings(max_examples=100, deadline=None)
def test_a_sep_diff_random(phi: np.ndarray):
    """Random Phi; LEAN and Python A_sep must agree to 1e-6 in finite
    cases. ±inf cases must classify the same on both sides."""
    from lean.python_bridge import find_a_sep_lean

    out_lean = find_a_sep_lean(phi, TRUTH_PARAMS_V5)
    out_python = _find_a_sep_python(phi)

    # Classification must match (healthy/bistable/collapsed).
    if not np.isfinite(out_lean) or not np.isfinite(out_python):
        assert np.isneginf(out_lean) == np.isneginf(out_python), (
            f"A_sep classification disagrees at phi={phi}: "
            f"LEAN={out_lean}, Python={out_python}"
        )
        assert np.isposinf(out_lean) == np.isposinf(out_python), (
            f"A_sep classification disagrees at phi={phi}: "
            f"LEAN={out_lean}, Python={out_python}"
        )
        return
    # Both finite: must agree numerically.
    np.testing.assert_allclose(
        out_lean, out_python, atol=SINGLE_STEP_TOL, rtol=0.0,
        err_msg=f"A_sep disagrees at phi={phi}.\n"
                f"  LEAN={out_lean}, Python={out_python}, diff={out_lean - out_python}"
    )


# ===========================================================================
# Schedule decoder (RBF coefficients → bimodal Φ schedule)
# ===========================================================================

def test_schedule_at_default_theta(lean_client):
    """At θ=0, the schedule should equal Phi_default uniformly across
    all bins for both channels (sigmoid(c_Phi) * Phi_max = Phi_default
    by construction)."""
    n_anchors = 4
    n_steps = 24
    dt = 1.0 / 24.0
    Phi_max = 3.0
    Phi_default = 1.0
    c_phi = float(np.log((Phi_default / Phi_max) / (1.0 - Phi_default / Phi_max)))

    # Build the design matrix with the same Gaussian-RBF formula the
    # smc2fc framework uses, so both implementations agree on basis.
    T_total = n_steps * dt
    centres = np.linspace(0.0, T_total, n_anchors)
    width = (T_total / max(n_anchors, 1)) * 1.0
    t_grid = np.arange(n_steps) * dt
    Phi_design = np.exp(-0.5 * ((t_grid[:, None] - centres[None, :]) / width) ** 2)

    theta = np.zeros(2 * n_anchors, dtype=np.float64)
    out_lean = lean_client.schedule(theta, Phi_design, c_phi, Phi_max, n_anchors)
    assert out_lean.shape == (n_steps, 2)
    np.testing.assert_allclose(
        out_lean, Phi_default, atol=SINGLE_STEP_TOL, rtol=0.0,
        err_msg=f"At theta=0, schedule should be all Phi_default. Got {out_lean}"
    )


@given(
    theta=st.lists(
        st.floats(min_value=-3.0, max_value=3.0),
        min_size=8, max_size=8,
    ),
)
@settings(max_examples=50, deadline=None)
def test_schedule_diff_random(theta):
    """Random θ: LEAN and Python schedule decoders agree element-wise."""
    n_anchors = 4
    n_steps = 24
    dt = 1.0 / 24.0
    Phi_max = 3.0
    Phi_default = 1.0
    c_phi = float(np.log((Phi_default / Phi_max) / (1.0 - Phi_default / Phi_max)))

    T_total = n_steps * dt
    centres = np.linspace(0.0, T_total, n_anchors)
    width = (T_total / max(n_anchors, 1)) * 1.0
    t_grid = np.arange(n_steps) * dt
    Phi_design = np.exp(-0.5 * ((t_grid[:, None] - centres[None, :]) / width) ** 2)

    theta_arr = np.asarray(theta, dtype=np.float64)
    from lean.python_bridge import schedule_lean

    out_lean = schedule_lean(theta_arr, Phi_design, c_phi, Phi_max, n_anchors)

    # Python reference (mirrors control.py:60-68).
    theta_B = theta_arr[:n_anchors]
    theta_S = theta_arr[n_anchors:]
    raw_B = c_phi + np.einsum("a,ta->t", theta_B, Phi_design)
    raw_S = c_phi + np.einsum("a,ta->t", theta_S, Phi_design)
    out_B_py = Phi_max / (1.0 + np.exp(-raw_B))
    out_S_py = Phi_max / (1.0 + np.exp(-raw_S))
    out_python = np.stack([out_B_py, out_S_py], axis=1)

    np.testing.assert_allclose(
        out_lean, out_python, atol=SINGLE_STEP_TOL, rtol=0.0,
        err_msg=f"Schedule disagrees at theta={theta}"
    )


# ===========================================================================
# Plant — deterministic Euler-Maruyama step
# ===========================================================================

def _em_step_python(state, phi, params, sigma_diag, dt, noise):
    """Pure-python deterministic EM step matching `_plant.py:_plant_em_step`'s
    body but with explicit noise input (no JAX RNG). Used as the
    Python-side oracle for the Lean diff test.

    Mirrors lines 102-145 of _plant.py."""
    EPS_B = 1e-4
    EPS_S = 1e-4
    EPS_A = 1e-4
    sqrt_dt = np.sqrt(dt)
    # drift via JAX
    p_jax = {k: jnp.float64(float(v)) for k, v in params.items()}
    d_y = np.asarray(drift_jax(jnp.asarray(state, dtype=jnp.float64), p_jax,
                                 jnp.asarray(phi, dtype=jnp.float64)),
                     dtype=np.float64)
    B_cl = np.clip(state[0], EPS_B, 1.0 - EPS_B)
    S_cl = np.clip(state[1], EPS_S, 1.0 - EPS_S)
    F_cl = max(state[2], 0.0)
    A_cl = max(state[3], 0.0)
    KFB_cl = max(state[4], 0.0)
    KFS_cl = max(state[5], 0.0)
    g = np.array([
        np.sqrt(B_cl * (1.0 - B_cl)),
        np.sqrt(S_cl * (1.0 - S_cl)),
        np.sqrt(F_cl),
        np.sqrt(A_cl + EPS_A),
        np.sqrt(KFB_cl),
        np.sqrt(KFS_cl),
    ])
    y_next = state + dt * d_y + sigma_diag * g * sqrt_dt * noise
    y_next[0] = np.clip(y_next[0], EPS_B, 1.0 - EPS_B)
    y_next[1] = np.clip(y_next[1], EPS_S, 1.0 - EPS_S)
    y_next[2] = max(y_next[2], 0.0)
    y_next[3] = max(y_next[3], 0.0)
    y_next[4] = max(y_next[4], 0.0)
    y_next[5] = max(y_next[5], 0.0)
    return y_next


def test_em_step_at_healthy_zero_noise(lean_client):
    """Zero-noise EM step at the healthy regime: Lean and Python next
    state must agree."""
    state = np.array([0.50, 0.45, 0.20, 0.45, 0.06, 0.07], dtype=np.float64)
    phi = np.array([0.30, 0.30], dtype=np.float64)
    sigma_diag = np.array([0.010, 0.008, 0.012, 0.020, 0.005, 0.005], dtype=np.float64)
    noise = np.zeros(6, dtype=np.float64)
    dt = 1.0 / 96.0
    out_lean = lean_client.em_step(state, phi, TRUTH_PARAMS_V5, sigma_diag, dt, noise)
    out_python = _em_step_python(state, phi, TRUTH_PARAMS_V5, sigma_diag, dt, noise)
    np.testing.assert_allclose(
        out_lean, out_python, atol=SINGLE_STEP_TOL, rtol=0.0,
        err_msg=f"emStep disagrees at zero-noise healthy.\n"
                f"  LEAN:   {out_lean}\n  Python: {out_python}"
    )


@given(state=_state_strategy(), phi=_phi_strategy(),
        noise=st.lists(st.floats(min_value=-3.0, max_value=3.0),
                       min_size=6, max_size=6))
@settings(max_examples=100, deadline=None)
def test_em_step_diff_random(state, phi, noise):
    """Random (state, phi, noise): LEAN and Python emStep must agree."""
    sigma_diag = np.array([0.010, 0.008, 0.012, 0.020, 0.005, 0.005], dtype=np.float64)
    dt = 1.0 / 96.0
    noise_arr = np.asarray(noise, dtype=np.float64)
    from lean.python_bridge import em_step_lean
    out_lean = em_step_lean(state, phi, TRUTH_PARAMS_V5, sigma_diag, dt, noise_arr)
    out_python = _em_step_python(state, phi, TRUTH_PARAMS_V5, sigma_diag, dt, noise_arr)
    np.testing.assert_allclose(
        out_lean, out_python, atol=SINGLE_STEP_TOL, rtol=0.0,
        err_msg=f"emStep disagrees at state={state}, phi={phi}, noise={noise_arr}.\n"
                f"  LEAN:   {out_lean}\n  Python: {out_python}"
    )


# ===========================================================================
# Observation channels — deterministic mean / probability
# ===========================================================================
#
# Bug 1's structural prevention shows here: ObsParams is a separate
# Lean structure from Params. The Python DEFAULT_PARAMS_V5 dict already
# carries both `sigma_S` (state noise) and `sigma_S_obs` (stress obs
# noise) as distinct keys after the rename earlier this session; the
# Lean side mirrors this with two records.

from models.fsa_high_res.simulation import (  # noqa: E402
    DEFAULT_PARAMS_V5,
    _sleep_prob,
)


def test_hr_mean_at_healthy(lean_client):
    state = np.array([0.50, 0.45, 0.20, 0.45, 0.06, 0.07], dtype=np.float64)
    C = 0.5
    out_lean = lean_client.hr_mean(state, C, DEFAULT_PARAMS_V5)
    p = DEFAULT_PARAMS_V5
    out_python = (
        p["HR_base"] - p["kappa_B_HR"] * state[0] + p["alpha_A_HR"] * state[3]
        + p["beta_C_HR"] * C
    )
    assert abs(out_lean - out_python) < SINGLE_STEP_TOL, (
        f"hrMean disagrees: LEAN={out_lean}, Python={out_python}"
    )


@given(state=_state_strategy(), C=st.floats(min_value=-1.0, max_value=1.0))
@settings(max_examples=50, deadline=None)
def test_obs_means_diff_random(state, C):
    """All five obs channels must agree between LEAN and Python under
    random (state, C). Bug 1 (sigma_S collision) prevention is
    structural — `sigma_S` (state noise) and `sigma_S_obs` (stress obs
    noise) are different fields throughout."""
    from lean.python_bridge import (
        hr_mean_lean, sleep_prob_lean, stress_mean_lean,
        steps_log_mean_lean, volume_load_mean_lean,
    )
    p = DEFAULT_PARAMS_V5
    # HR
    out_lean = hr_mean_lean(state, C, p)
    out_python = (p["HR_base"] - p["kappa_B_HR"] * state[0]
                  + p["alpha_A_HR"] * state[3] + p["beta_C_HR"] * C)
    assert abs(out_lean - out_python) < SINGLE_STEP_TOL, f"hrMean: LEAN={out_lean}, Py={out_python}"
    # Sleep prob
    out_lean = sleep_prob_lean(state, C, p)
    out_python = _sleep_prob(state[3], C, p["k_C"], p["k_A"], p["c_tilde"])
    assert abs(out_lean - out_python) < SINGLE_STEP_TOL, f"sleepProb: LEAN={out_lean}, Py={out_python}"
    # Stress
    out_lean = stress_mean_lean(state, C, p)
    out_python = (p["S_base"] + p["k_F"] * state[2] - p["k_A_S"] * state[3]
                  + p["beta_C_S"] * C)
    assert abs(out_lean - out_python) < SINGLE_STEP_TOL, f"stressMean: LEAN={out_lean}, Py={out_python}"
    # Steps log-mean
    out_lean = steps_log_mean_lean(state, C, p)
    out_python = (p["mu_step0"] + p["beta_B_st"] * state[0] - p["beta_F_st"] * state[2]
                  + p["beta_A_st"] * state[3] + p["beta_C_st"] * C)
    assert abs(out_lean - out_python) < SINGLE_STEP_TOL, f"stepsLogMean: LEAN={out_lean}, Py={out_python}"
    # VolumeLoad
    out_lean = volume_load_mean_lean(state, p)
    out_python = p["beta_S_VL"] * state[1] - p["beta_F_VL"] * state[2]
    assert abs(out_lean - out_python) < SINGLE_STEP_TOL, f"volumeLoadMean: LEAN={out_lean}, Py={out_python}"
