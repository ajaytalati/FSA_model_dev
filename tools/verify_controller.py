"""Controller-sandbox verification for FSA-v2.

Sanity test for `models.fsa_high_res.control.build_control_spec`:

1. Build a `ControlSpec` for the canonical 42-day Banister horizon.
2. Confirm `cost_fn(theta=0)` returns a finite scalar.
3. Confirm `jax.grad(cost_fn)(theta=0)` is non-zero (the cost surface
   has non-trivial curvature in the schedule coefficients).
4. Take one gradient step in the negative-gradient direction and
   confirm the cost decreased — i.e. the controller's gradient signal
   is actually useful for optimisation.

This catches regressions where the controller's cost integrator silently
returns a constant or NaN, the schedule parameterisation degenerates,
or the gradient flips sign.
"""
import os
import sys
from pathlib import Path

# Force CPU + X64.
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'True'

import jax
import jax.numpy as jnp
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.fsa_high_res.control import build_control_spec


def verify_controller():
    """Build ControlSpec, evaluate cost + gradient, take a step. Returns dict."""
    print("Running FSA-v2 Controller Sandbox Verification...")

    # Build a 42-day horizon controller — Banister chronic timescale.
    spec = build_control_spec(T_total_days=42.0, n_inner=8, seed=0)
    theta_dim = int(spec.theta_dim)
    print(f"  Built ControlSpec: T=42 d, theta_dim={theta_dim}")

    # ── 1. Evaluate cost at theta=0 ──────────────────────────────────
    theta_zero = jnp.zeros(theta_dim, dtype=jnp.float64)
    initial_cost = float(spec.cost_fn(theta_zero))
    print(f"  Initial cost (theta=0): {initial_cost:.6f}")
    assert np.isfinite(initial_cost), (
        f"cost_fn returned non-finite value {initial_cost} at theta=0; "
        f"check the cost integrator in models/fsa_high_res/control.py")

    # ── 2. Evaluate gradient ─────────────────────────────────────────
    grad_fn = jax.grad(spec.cost_fn)
    grad = grad_fn(theta_zero)
    grad_arr = np.asarray(grad)
    grad_norm = float(np.linalg.norm(grad_arr))
    grad_min = float(grad_arr.min())
    grad_max = float(grad_arr.max())
    print(f"  Gradient |g|_2 = {grad_norm:.4e}")
    print(f"  Gradient range: [{grad_min:.4e}, {grad_max:.4e}]")
    assert np.all(np.isfinite(grad_arr)), (
        f"gradient has non-finite entries; cost_fn is non-differentiable "
        f"at theta=0 — check JAX-jit'd integrator for control flow")
    assert grad_norm > 1e-8, (
        f"gradient norm {grad_norm} ~= 0; the cost surface is flat at "
        f"theta=0, meaning the controller has no learning signal. "
        f"Check the cost integrator in models/fsa_high_res/control.py")

    # ── 3. One gradient step — cost must decrease ────────────────────
    learning_rate = 1.0   # safe small-step
    theta_step = theta_zero - learning_rate * grad
    new_cost = float(spec.cost_fn(theta_step))
    cost_improvement = initial_cost - new_cost
    print(f"  Cost after one gradient step (lr={learning_rate}): "
          f"{new_cost:.6f}  (delta = {cost_improvement:+.6f})")
    assert new_cost < initial_cost, (
        f"Gradient step INCREASED the cost ({initial_cost:.6f} → "
        f"{new_cost:.6f}). The gradient direction is wrong, or the "
        f"step size is far too large. Either way, the controller "
        f"can't be used for optimisation as-is.")

    print(f"\nController Sandbox: PASS")
    return {
        'theta_dim':         theta_dim,
        'initial_cost':      initial_cost,
        'gradient_norm':     grad_norm,
        'gradient_range':    (grad_min, grad_max),
        'new_cost':          new_cost,
        'cost_improvement':  cost_improvement,
        'passed':            True,
    }


if __name__ == "__main__":
    verify_controller()
