import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax
import jax.numpy as jnp
from models.fsa_high_res.estimation import HIGH_RES_FSA_V2_ESTIMATION
from models.fsa_high_res.control import build_control_spec

def test_estimation_artifact():
    print("Testing EstimationModel artifact...")
    em = HIGH_RES_FSA_V2_ESTIMATION
    print(f"  Model name: {em.name}, version: {em.version}")
    assert em.n_states == 3
    assert len(em.all_names) > 0
    print("  EstimationModel looks good.")

def test_control_artifact():
    print("Testing ControlSpec artifact...")
    # build_control_spec(n_steps, dt, ...)
    spec = build_control_spec(T_total_days=1.0, dt_days=1.0/96.0)
    print(f"  ControlSpec name: {spec.name}, theta_dim: {spec.theta_dim}")
    assert spec.theta_dim > 0
    
    # Evaluate cost_fn once
    print("  Evaluating cost_fn...")
    theta = jnp.zeros(spec.theta_dim)
    cost = spec.cost_fn(theta)
    print(f"  Cost at theta=0: {cost:.4f}")
    assert jnp.isfinite(cost)
    print("  ControlSpec looks good.")

if __name__ == "__main__":
    test_estimation_artifact()
    test_control_artifact()
