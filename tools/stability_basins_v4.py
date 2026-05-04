"""FSA-v4 stability analysis: closed-form equilibria, bifurcation curve, basins.

Generates the figures and tables consumed by ``LaTex_docs/sections/11_stability_v4.tex``.

Outline:

  1. Closed-form equilibria  K*(Phi), B*(A,Phi), S*(A,Phi), F*(A,Phi)  derived
     from the cascade structure of ``models/fsa_high_res/_dynamics.py``.
  2. Self-consistent A equation   mu_bar(A; Phi) = eta * A^2  solved by Brent.
  3. Block-triangular Jacobian — eigenvalues compared analytically vs jax.jacfwd.
  4. Drift-only RK4 rollouts (vmapped) classify (Phi, A0) basins numerically.
  5. Five PNG figures saved to LaTex_docs/figures/.
  6. Equilibrium values written to LaTex_docs/tables/v4_equilibrium_values.tex.

Run:  python tools/stability_basins_v4.py
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

from models.fsa_high_res._dynamics import (
    TRUTH_PARAMS, TRUTH_PARAMS_V5, A_TYP, F_TYP, drift_jax,
)


FIG_DIR = "LaTex_docs/figures"
TABLE_DIR = "LaTex_docs/tables"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

# ── Closed-form equilibria (numpy-friendly; broadcast over scalar or array) ──

def K_star(Phi_B, Phi_S, p):
    KFB = p['KFB_0'] + p['tau_K'] * p['mu_K'] * Phi_B
    KFS = p['KFS_0'] + p['tau_K'] * p['mu_K'] * Phi_S
    return KFB, KFS


def BS_star(A, Phi_B, Phi_S, p):
    a_B = (1.0 + p['epsilon_AB'] * A) / (1.0 + p['epsilon_AB'] * A_TYP)
    a_S = (1.0 + p['epsilon_AS'] * A) / (1.0 + p['epsilon_AS'] * A_TYP)
    B = p['tau_B'] * p['kappa_B'] * a_B * Phi_B
    S = p['tau_S'] * p['kappa_S'] * a_S * Phi_S
    return B, S


def F_star(A, Phi_B, Phi_S, p):
    KFB, KFS = K_star(Phi_B, Phi_S, p)
    a_F = (1.0 + p['lambda_A'] * A) / (1.0 + p['lambda_A'] * A_TYP)
    return p['tau_F'] * (KFB * Phi_B + KFS * Phi_S) / a_F


def mu_bar(A, Phi_B, Phi_S, p):
    """Effective Stuart-Landau coefficient at the slow manifold.

    Includes the v5 Hill-deconditioning term, which is silent (==0) when
    p['mu_dec_B'] = p['mu_dec_S'] = 0 (FSA-v4 defaults).
    """
    B, S = BS_star(A, Phi_B, Phi_S, p)
    F = F_star(A, Phi_B, Phi_S, p)
    F_dev = F - F_TYP
    n = p.get('n_dec', 4.0)
    B_dec = p.get('B_dec', 0.0)
    S_dec = p.get('S_dec', 0.0)
    mu_dec_B = p.get('mu_dec_B', 0.0)
    mu_dec_S = p.get('mu_dec_S', 0.0)
    Bn = np.maximum(B, 0.0) ** n
    Sn = np.maximum(S, 0.0) ** n
    Bdn = B_dec ** n
    Sdn = S_dec ** n
    dec_B = mu_dec_B * Bdn / (Bn + Bdn) if mu_dec_B > 0 else 0.0
    dec_S = mu_dec_S * Sdn / (Sn + Sdn) if mu_dec_S > 0 else 0.0
    return (p['mu_0'] + p['mu_B'] * B + p['mu_S'] * S
            - p['mu_F'] * F - p['mu_FF'] * F_dev * F_dev
            - dec_B - dec_S)


def find_A_roots(Phi_B, Phi_S, p, A_max=2.5, n_grid=4000):
    """All positive roots of g(A) = mu_bar(A; Phi) - eta*A^2, sorted ascending.

    Returns:
        list[float] of length 0, 1 or 2.
            - len 0: no positive roots → A=0 is the only equilibrium
            - len 1: single root A_s = stable interior equilibrium
                      (mu_bar(0)>0 case; A=0 unstable)
            - len 2: roots [A_low, A_high]
                      A_low = unstable separatrix, A_high = stable interior
                      (bistable case; A=0 also stable)
    """
    eta = p['eta']
    def g(a):
        return float(mu_bar(a, Phi_B, Phi_S, p) - eta * a * a)
    A_grid = np.linspace(1e-7, A_max, n_grid)
    g_vals = np.array([g(a) for a in A_grid])
    sign_flips = np.where(np.diff(np.sign(g_vals)) != 0)[0]
    roots = []
    for idx in sign_flips:
        try:
            roots.append(brentq(g, A_grid[idx], A_grid[idx+1]))
        except ValueError:
            pass
    return sorted(roots)


def find_A_star(Phi_B, Phi_S, p, **kwargs):
    """Largest positive root (the stable healthy attractor), or NaN if none."""
    roots = find_A_roots(Phi_B, Phi_S, p, **kwargs)
    if not roots:
        return float('nan')
    return roots[-1]


def find_A_separatrix(Phi_B, Phi_S, p, **kwargs):
    """Unstable lower root (basin boundary in bistable regime), or NaN."""
    roots = find_A_roots(Phi_B, Phi_S, p, **kwargs)
    if len(roots) < 2:
        return float('nan')
    return roots[0]


def initial_state_at_A(A0, Phi_B, Phi_S, p):
    """Slow-manifold initial state with autonomic value A0."""
    KFB, KFS = K_star(Phi_B, Phi_S, p)
    B, S = BS_star(A0, Phi_B, Phi_S, p)
    F = F_star(A0, Phi_B, Phi_S, p)
    return np.array([float(B), float(S), float(F),
                     float(A0), float(KFB), float(KFS)])

# ── Vectorised RK4 rollout via jax.vmap ───────────────────────────────────────

def make_rollout_fn(p, T, dt):
    p_j = {k: jnp.asarray(float(v)) for k, v in p.items()}
    n_steps = int(T / dt)

    def rk4_step(y, Phi):
        k1 = drift_jax(y, p_j, Phi)
        k2 = drift_jax(y + 0.5 * dt * k1, p_j, Phi)
        k3 = drift_jax(y + 0.5 * dt * k2, p_j, Phi)
        k4 = drift_jax(y + dt * k3, p_j, Phi)
        y_new = y + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        # Clip to physical domain (matches abs() reflection in _dynamics.py)
        y_new = y_new.at[0].set(jnp.clip(y_new[0], 0.0, 1.0))
        y_new = y_new.at[1].set(jnp.clip(y_new[1], 0.0, 1.0))
        y_new = y_new.at[2].set(jnp.maximum(y_new[2], 0.0))
        y_new = y_new.at[3].set(jnp.maximum(y_new[3], 0.0))
        y_new = y_new.at[4].set(jnp.maximum(y_new[4], 0.0))
        y_new = y_new.at[5].set(jnp.maximum(y_new[5], 0.0))
        return y_new

    @jax.jit
    def run_one(y0, Phi):
        def body(y, _):
            return rk4_step(y, Phi), None
        y_final, _ = jax.lax.scan(body, y0, None, length=n_steps)
        return y_final

    return jax.vmap(run_one)


def jacobian_fn(p):
    p_j = {k: jnp.asarray(float(v)) for k, v in p.items()}
    @jax.jit
    def J(y, Phi):
        return jax.jacfwd(lambda yy: drift_jax(yy, p_j, Phi))(y)
    return J


# ── Verify analytical equilibria against numerical drift ──────────────────────

def verify_equilibria(p, n=80, seed=42):
    rng = np.random.default_rng(seed)
    p_j = {k: jnp.asarray(float(v)) for k, v in p.items()}
    @jax.jit
    def drift_at(y, Phi):
        return drift_jax(y, p_j, Phi)
    max_err = 0.0
    for _ in range(n):
        pb = float(rng.uniform(0.05, 1.5))
        ps = float(rng.uniform(0.05, 1.5))
        # Boundary equilibrium
        y0 = initial_state_at_A(0.0, pb, ps, p)
        d = np.asarray(drift_at(jnp.asarray(y0), jnp.asarray((pb, ps))))
        max_err = max(max_err, float(np.max(np.abs(d))))
        # Interior equilibrium (if exists)
        A_st = find_A_star(pb, ps, p)
        if not np.isnan(A_st):
            y1 = initial_state_at_A(A_st, pb, ps, p)
            d1 = np.asarray(drift_at(jnp.asarray(y1), jnp.asarray((pb, ps))))
            max_err = max(max_err, float(np.max(np.abs(d1))))
    return max_err


# ── Figures ───────────────────────────────────────────────────────────────────

def figure_mu_vs_A(p):
    A_grid = np.linspace(0.0, 1.5, 300)
    eta = p['eta']
    examples = [
        ((0.20, 0.20), 'Mono-stable healthy: $\\Phi=(0.20,0.20)$'),
        ((0.50, 0.30), 'Bistable: $\\Phi=(0.50,0.30)$'),
        ((1.00, 1.00), 'Mono-stable collapsed: $\\Phi=(1.0,1.0)$'),
    ]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colours = ['#2ca02c', '#bcbd22', '#d62728']
    for ((pb, ps), lbl), c in zip(examples, colours):
        g_vals = np.array([float(mu_bar(a, pb, ps, p) - eta*a*a) for a in A_grid])
        ax.plot(A_grid, g_vals, color=c, lw=2, label=lbl)
        # Mark roots
        sf = np.where(np.diff(np.sign(g_vals)) != 0)[0]
        for idx in sf:
            try:
                r = brentq(lambda a: float(mu_bar(a, pb, ps, p) - eta*a*a),
                           A_grid[idx], A_grid[idx+1])
                ax.plot(r, 0, 'o', color=c, markersize=8)
            except ValueError:
                pass
    ax.axhline(0, color='k', linestyle=':', linewidth=0.8)
    ax.set_xlabel(r'$A$')
    ax.set_ylabel(r'$\bar\mu(A;\Phi) - \eta A^2$')
    ax.set_title('Self-consistency function for $A$ (zeros = equilibria)')
    ax.legend(loc='lower left')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'v4_mu_vs_A_three_regimes.png')
    plt.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  saved {out}")


def figure_bifurcation_curve(p, T=120.0, dt=0.5, grid=50):
    phi = np.linspace(0.05, 1.5, grid)
    Pb, Ps = np.meshgrid(phi, phi)

    # Analytical mu_bar(0; Phi)  — transcritical bifurcation contour
    mu0 = np.zeros_like(Pb)
    # Number of positive interior roots — saddle-node sits where this drops 2→0
    n_roots = np.zeros_like(Pb, dtype=int)
    for i in range(grid):
        for j in range(grid):
            mu0[i, j] = float(mu_bar(0.0, Pb[i, j], Ps[i, j], p))
            n_roots[i, j] = len(find_A_roots(Pb[i, j], Ps[i, j], p))

    # Numerical: A0 small to detect basin around A=0
    rollout_v = make_rollout_fn(p, T, dt)
    y0_list, Phi_list = [], []
    A0_seed = 0.05  # small perturbation; below typical separatrix
    for i in range(grid):
        for j in range(grid):
            y0_list.append(initial_state_at_A(A0_seed, Pb[i, j], Ps[i, j], p))
            Phi_list.append((Pb[i, j], Ps[i, j]))
    y0_arr = jnp.asarray(np.stack(y0_list))
    Phi_arr = jnp.asarray(np.stack(Phi_list))
    y_final = np.asarray(rollout_v(y0_arr, Phi_arr))
    A_final = y_final[:, 3].reshape(grid, grid)
    healthy = (A_final > 0.01)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    vmax = max(abs(mu0.min()), abs(mu0.max()))
    im = axes[0].pcolormesh(Pb, Ps, mu0, shading='auto', cmap='RdBu_r',
                             vmin=-vmax, vmax=vmax)
    # Transcritical: mu_bar(0) = 0
    axes[0].contour(Pb, Ps, mu0, levels=[0], colors='k', linewidths=2)
    # Saddle-node: boundary where n_roots drops below 2 (the bistable region)
    axes[0].contour(Pb, Ps, n_roots.astype(float), levels=[1.5],
                     colors='magenta', linewidths=2, linestyles='--')
    axes[0].set_xlabel(r'$\Phi_B$')
    axes[0].set_ylabel(r'$\Phi_S$')
    axes[0].set_title(r'Analytical: black = transcritical ($\bar\mu(0)=0$),'
                      r' magenta dashed = saddle-node')
    plt.colorbar(im, ax=axes[0])

    axes[1].pcolormesh(Pb, Ps, healthy.astype(float), shading='auto',
                        cmap='RdBu_r')
    axes[1].contour(Pb, Ps, mu0, levels=[0], colors='k',
                     linewidths=2, linestyles='-')
    axes[1].contour(Pb, Ps, n_roots.astype(float), levels=[1.5],
                     colors='magenta', linewidths=2, linestyles='--')
    axes[1].set_xlabel(r'$\Phi_B$')
    axes[1].set_ylabel(r'$\Phi_S$')
    axes[1].set_title(r'Numerical from $A_0=0.05$: red = healthy, blue = collapsed.'
                      '\nBetween black & magenta = bistable region')
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'v4_bifurcation_curve.png')
    plt.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  saved {out}")
    return mu0, healthy


def figure_basin_diagram(p, axis='B', other_phi=0.3, T=120.0, dt=0.5,
                          n_phi=40, n_A0=40):
    phi = np.linspace(0.05, 1.5, n_phi)
    A0 = np.linspace(0.001, 1.0, n_A0)
    rollout_v = make_rollout_fn(p, T, dt)
    y0_list = []
    Phi_list = []
    for i in range(n_A0):
        for j in range(n_phi):
            if axis == 'B':
                pb, ps = float(phi[j]), float(other_phi)
            else:
                pb, ps = float(other_phi), float(phi[j])
            y0_list.append(initial_state_at_A(float(A0[i]), pb, ps, p))
            Phi_list.append((pb, ps))
    y0_arr = jnp.asarray(np.stack(y0_list))
    Phi_arr = jnp.asarray(np.stack(Phi_list))
    y_final = np.asarray(rollout_v(y0_arr, Phi_arr))
    A_final = y_final[:, 3].reshape(n_A0, n_phi)
    healthy = (A_final > 0.01).astype(float)

    PHI, A0M = np.meshgrid(phi, A0)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.pcolormesh(PHI, A0M, healthy, shading='auto', cmap='RdBu_r')
    # Overlay analytical roots: stable upper (solid), unstable lower (dashed)
    A_upper = []
    A_lower = []
    for pj in phi:
        if axis == 'B':
            roots = find_A_roots(float(pj), float(other_phi), p)
        else:
            roots = find_A_roots(float(other_phi), float(pj), p)
        A_upper.append(roots[-1] if roots else np.nan)
        A_lower.append(roots[0] if len(roots) >= 2 else np.nan)
    A_upper = np.array(A_upper)
    A_lower = np.array(A_lower)
    valid_u = ~np.isnan(A_upper)
    valid_l = ~np.isnan(A_lower)
    if valid_u.any():
        ax.plot(phi[valid_u], A_upper[valid_u], 'k-', lw=2,
                 label=r'$A_{\rm stable}^*$ (analytical)')
    if valid_l.any():
        ax.plot(phi[valid_l], A_lower[valid_l], 'k--', lw=2,
                 label=r'$A_{\rm sep}$ (separatrix)')
    if valid_u.any() or valid_l.any():
        ax.legend(loc='upper right')
    ax.set_xlabel(fr'$\Phi_{axis}$')
    ax.set_ylabel(r'$A_0$ (initial autonomic amplitude)')
    other = 'S' if axis == 'B' else 'B'
    ax.set_title(fr'Basin diagram at $\Phi_{other}={other_phi}$ '
                 r'(red = healthy, blue = collapsed)')
    plt.tight_layout()
    out = os.path.join(FIG_DIR, f'v4_basin_diagram_phi{axis}.png')
    plt.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  saved {out}")


def figure_jacobian_eigenvalues(p, phi_s=0.3, n=80):
    pb_range = np.linspace(0.05, 1.5, n)
    J_fn = jacobian_fn(p)
    eigs_c = np.zeros((n, 6), dtype=complex)
    eigs_h = np.full((n, 6), np.nan, dtype=complex)
    A_st = np.zeros(n)
    for i, pb in enumerate(pb_range):
        sc = initial_state_at_A(0.0, float(pb), float(phi_s), p)
        Jc = np.asarray(J_fn(jnp.asarray(sc), jnp.asarray((pb, phi_s))))
        eigs_c[i] = np.linalg.eigvals(Jc)
        a = find_A_star(float(pb), float(phi_s), p)
        A_st[i] = a
        if not np.isnan(a):
            sh = initial_state_at_A(a, float(pb), float(phi_s), p)
            Jh = np.asarray(J_fn(jnp.asarray(sh), jnp.asarray((pb, phi_s))))
            eigs_h[i] = np.linalg.eigvals(Jh)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    axes[0].axhline(0, color='k', linestyle=':', linewidth=0.8)
    for k in range(6):
        axes[0].plot(pb_range, eigs_c[:, k].real, '.', markersize=4)
    axes[0].set_xlabel(r'$\Phi_B$')
    axes[0].set_ylabel(r'$\Re(\lambda)$')
    axes[0].set_title(rf'Eigenvalues at $A=0$ (collapsed, $\Phi_S={phi_s}$)')
    axes[0].grid(alpha=0.3)

    axes[1].axhline(0, color='k', linestyle=':', linewidth=0.8)
    valid = ~np.isnan(A_st)
    for k in range(6):
        axes[1].plot(pb_range[valid], eigs_h[valid, k].real, '.', markersize=4)
    axes[1].set_xlabel(r'$\Phi_B$')
    axes[1].set_title(rf'Eigenvalues at $A=A^*$ (healthy)')
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'v4_jacobian_eigenvalues.png')
    plt.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  saved {out}")


# ── Equilibrium-values LaTeX table ────────────────────────────────────────────

def write_equilibrium_table(p):
    grid = [0.20, 0.40, 0.55, 0.70, 1.00]
    out = os.path.join(TABLE_DIR, 'v4_equilibrium_values.tex')
    lines = []
    lines.append(r"\begin{tabular}{cc|cccccc}")
    lines.append(r"\toprule")
    lines.append(r"$\Phi_B$ & $\Phi_S$ & $K_{FB}^*$ & $K_{FS}^*$ & $B^*$ & $S^*$ & $F^*$ & $A^*$ \\")
    lines.append(r"\midrule")
    for pb in grid:
        for ps in grid:
            KFB, KFS = K_star(pb, ps, p)
            A_st = find_A_star(pb, ps, p)
            A_repr = 0.0 if np.isnan(A_st) else A_st
            B, S = BS_star(A_repr, pb, ps, p)
            F = F_star(A_repr, pb, ps, p)
            A_disp = "$0$" if np.isnan(A_st) else f"${A_st:.3f}$"
            lines.append(f"{pb:.2f} & {ps:.2f} & {float(KFB):.4f} & {float(KFS):.4f} & "
                          f"{float(B):.3f} & {float(S):.3f} & {float(F):.3f} & {A_disp} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    with open(out, 'w') as f:
        f.write("\n".join(lines))
    print(f"  saved {out}")


def write_jacobian_table(p):
    """Eigenvalues at representative equilibria across the three regimes."""
    out = os.path.join(TABLE_DIR, 'v4_jacobian_blocks.tex')
    p_j = {k: jnp.asarray(float(v)) for k, v in p.items()}
    @jax.jit
    def Jat(y, Phi):
        return jax.jacfwd(lambda yy: drift_jax(yy, p_j, Phi))(y)

    cases = [
        ("Mono-stable healthy", 0.20, 0.20),
        ("Bistable",            0.50, 0.30),
        ("Mono-stable collapsed", 1.00, 1.00),
    ]
    rows = []
    rows.append(r"\begin{tabular}{lccccccccc}")
    rows.append(r"\toprule")
    rows.append(r"Regime & $\Phi$ & Equilibrium & $\lambda_1$ & $\lambda_2$ & $\lambda_3$ & "
                r"$\lambda_4$ & $\lambda_5$ & $\lambda_6$ & verdict \\")
    rows.append(r"\midrule")
    for name, pb, ps in cases:
        roots = find_A_roots(pb, ps, p)
        # always evaluate at A=0
        for label, A_val in [(r"$A=0$", 0.0)] + (
            [(r"$A_{\rm sep}$", roots[0])] if len(roots) >= 2 else []
        ) + (
            [(r"$A^*$", roots[-1])] if roots else []
        ):
            y_eq = initial_state_at_A(A_val, pb, ps, p)
            J = np.asarray(Jat(jnp.asarray(y_eq), jnp.asarray((pb, ps))))
            eig = np.linalg.eigvals(J)
            sorted_eig = np.sort(eig.real)[::-1]  # descending
            eig_str = " & ".join(f"${e:+.4f}$" for e in sorted_eig)
            verdict = "stable" if (sorted_eig < 0).all() else "unstable"
            phi_str = f"$({pb:.2f},{ps:.2f})$"
            rows.append(f"{name} & {phi_str} & {label} & {eig_str} & {verdict} \\\\")
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    with open(out, 'w') as f:
        f.write("\n".join(rows))
    print(f"  saved {out}")


def write_lyapunov_summary_table():
    out = os.path.join(TABLE_DIR, 'v4_lyapunov_summary.tex')
    lines = [
        r"\begin{tabular}{lll}",
        r"\toprule",
        r"Block & Lyapunov component & Decay rate \\",
        r"\midrule",
        r"$K$-block      & $V_K = \tfrac{1}{2}\sum_i (K_{Fi}-K_{Fi}^*)^2$ & $1/\tau_K$ \\",
        r"$B$-block      & $V_B = \tfrac{1}{2}(B-B^*(A))^2$               & $1/\tau_B$ \\",
        r"$S$-block      & $V_S = \tfrac{1}{2}(S-S^*(A))^2$               & $1/\tau_S$ \\",
        r"$F$-block      & $V_F = \tfrac{1}{2}(F-F^*(A))^2$               & $a_F(A)/\tau_F$ \\",
        r"$A$-block (collapsed)  & $V_A = \tfrac{1}{2}A^2$                & $|\bar\mu(0;\Phi)|$ for $\bar\mu(0)<0$ \\",
        r"$A$-block (healthy)    & $V_A = \tfrac{\eta}{4}(A^2-(A^*)^2)^2$ & $2\eta(A^*)^2$ near $A^*$ \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    with open(out, 'w') as f:
        f.write("\n".join(lines))
    print(f"  saved {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def figure_v5_island(p_v4, p_v5, grid=80):
    """Side-by-side: v4 transcritical contour (open quarter-disc) vs v5
    transcritical contour (closed island). Same colour scale; black contour
    is mu_bar(0;Phi) = 0 in both panels.
    """
    phi = np.linspace(0.0, 2.0, grid)
    Pb, Ps = np.meshgrid(phi, phi)
    mu0_v4 = np.zeros_like(Pb)
    mu0_v5 = np.zeros_like(Pb)
    for i in range(grid):
        for j in range(grid):
            mu0_v4[i, j] = float(mu_bar(0.0, Pb[i, j], Ps[i, j], p_v4))
            mu0_v5[i, j] = float(mu_bar(0.0, Pb[i, j], Ps[i, j], p_v5))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    vmax = max(abs(mu0_v5).max(), abs(mu0_v4).max())
    im0 = axes[0].pcolormesh(Pb, Ps, mu0_v4, shading='auto', cmap='RdBu_r',
                              vmin=-vmax, vmax=vmax)
    axes[0].contour(Pb, Ps, mu0_v4, levels=[0], colors='k', linewidths=2)
    axes[0].set_xlabel(r'$\Phi_B$')
    axes[0].set_ylabel(r'$\Phi_S$')
    axes[0].set_title(r'FSA-v4: open quarter-disc' '\n'
                      r'(only over-training $\to A=0$)')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(Pb, Ps, mu0_v5, shading='auto', cmap='RdBu_r',
                              vmin=-vmax, vmax=vmax)
    axes[1].contour(Pb, Ps, mu0_v5, levels=[0], colors='k', linewidths=2)
    axes[1].set_xlabel(r'$\Phi_B$')
    axes[1].set_ylabel(r'$\Phi_S$')
    Bd = p_v5['B_dec']; Sd = p_v5['S_dec']
    mDB = p_v5['mu_dec_B']; mDS = p_v5['mu_dec_S']
    axes[1].set_title(
        rf'FSA-v5 (Hill, $B_{{\rm dec}}={Bd}, S_{{\rm dec}}={Sd},$'
        rf' $\mu_{{B-}}={mDB}, \mu_{{S-}}={mDS}$)'
        '\n closed island: sedentary AND over-training $\\to A=0$')
    plt.colorbar(im1, ax=axes[1])
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'v5_sedentary_collapse_island.png')
    plt.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  saved {out}")


def figure_v5_full_bifurcation(p, T=180.0, dt=0.5, grid=50):
    """v5 analogue of figure_bifurcation_curve. Shows transcritical (black) and
    saddle-node (magenta) curves on a heatmap of mu_bar(0;Phi), with numerical
    regime classification from a small initial A0=0.05.
    """
    phi = np.linspace(0.0, 2.0, grid)
    Pb, Ps = np.meshgrid(phi, phi)
    mu0 = np.zeros_like(Pb)
    n_roots = np.zeros_like(Pb, dtype=int)
    for i in range(grid):
        for j in range(grid):
            mu0[i, j] = float(mu_bar(0.0, Pb[i, j], Ps[i, j], p))
            n_roots[i, j] = len(find_A_roots(Pb[i, j], Ps[i, j], p))

    rollout_v = make_rollout_fn(p, T, dt)
    y0_list, Phi_list = [], []
    A0_seed = 0.05
    for i in range(grid):
        for j in range(grid):
            y0_list.append(initial_state_at_A(A0_seed, Pb[i, j], Ps[i, j], p))
            Phi_list.append((Pb[i, j], Ps[i, j]))
    y0_arr = jnp.asarray(np.stack(y0_list))
    Phi_arr = jnp.asarray(np.stack(Phi_list))
    y_final = np.asarray(rollout_v(y0_arr, Phi_arr))
    A_final = y_final[:, 3].reshape(grid, grid)
    healthy = (A_final > 0.01)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    vmax = max(abs(mu0).max(), 0.1)
    im = axes[0].pcolormesh(Pb, Ps, mu0, shading='auto', cmap='RdBu_r',
                             vmin=-vmax, vmax=vmax)
    axes[0].contour(Pb, Ps, mu0, levels=[0], colors='k', linewidths=2)
    axes[0].contour(Pb, Ps, n_roots.astype(float), levels=[1.5],
                     colors='magenta', linewidths=2, linestyles='--')
    axes[0].set_xlabel(r'$\Phi_B$')
    axes[0].set_ylabel(r'$\Phi_S$')
    axes[0].set_title(r'FSA-v5 analytical: black = transcritical island,'
                      '\nmagenta dashed = saddle-node (bistable annulus)')
    plt.colorbar(im, ax=axes[0])

    axes[1].pcolormesh(Pb, Ps, healthy.astype(float), shading='auto',
                        cmap='RdBu_r')
    axes[1].contour(Pb, Ps, mu0, levels=[0], colors='k', linewidths=2)
    axes[1].contour(Pb, Ps, n_roots.astype(float), levels=[1.5],
                     colors='magenta', linewidths=2, linestyles='--')
    axes[1].set_xlabel(r'$\Phi_B$')
    axes[1].set_ylabel(r'$\Phi_S$')
    axes[1].set_title(r'FSA-v5 numerical from $A_0=0.05$:'
                      '\nred = healthy, blue = collapsed')
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'v5_full_bifurcation.png')
    plt.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  saved {out}")


def figure_v5_trajectories(p, T=200.0, dt=0.5):
    """RK4 rollouts under v5 dynamics from a TRAINED initial state, under three
    constant-Phi scenarios:
      (a) sedentary    Phi = (0, 0)
      (b) moderate     Phi = (0.30, 0.20)   (inside the v5 healthy island)
      (c) over-training Phi = (1.0, 1.0)

    Demonstrates that BOTH (a) sedentary and (c) over-training drive A -> 0,
    while (b) moderate keeps A > 0.
    """
    # Trained initial state: high B, S, healthy A, low F, K at slow-manifold.
    KFB_init, KFS_init = K_star(0.5, 0.5, p)  # K equilibrated for moderate past load
    y0_trained = np.array([0.50, 0.45, 0.20, 0.45,
                            float(KFB_init), float(KFS_init)])
    scenarios = [
        ('Sedentary  $\\Phi=(0,0)$',          (0.00, 0.00)),
        ('Moderate  $\\Phi=(0.30, 0.30)$',    (0.30, 0.30)),
        ('Over-training  $\\Phi=(1.0, 1.0)$', (1.00, 1.00)),
    ]
    rollout_v = make_rollout_fn(p, T, dt)
    n_steps = int(T / dt)
    t_grid = np.arange(n_steps) * dt

    # Run all three scenarios in one vmap'd batch but also need the trajectory
    # through time, not just final. Easiest: re-implement step loop returning
    # full traj.
    p_j = {k: jnp.asarray(float(v)) for k, v in p.items()}
    def rk4_step(y, Phi):
        k1 = drift_jax(y, p_j, Phi)
        k2 = drift_jax(y + 0.5 * dt * k1, p_j, Phi)
        k3 = drift_jax(y + 0.5 * dt * k2, p_j, Phi)
        k4 = drift_jax(y + dt * k3, p_j, Phi)
        y_new = y + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        y_new = y_new.at[0].set(jnp.clip(y_new[0], 0.0, 1.0))
        y_new = y_new.at[1].set(jnp.clip(y_new[1], 0.0, 1.0))
        y_new = y_new.at[2].set(jnp.maximum(y_new[2], 0.0))
        y_new = y_new.at[3].set(jnp.maximum(y_new[3], 0.0))
        y_new = y_new.at[4].set(jnp.maximum(y_new[4], 0.0))
        y_new = y_new.at[5].set(jnp.maximum(y_new[5], 0.0))
        return y_new

    @jax.jit
    def full_traj(y0, Phi):
        def body(y, _):
            y_new = rk4_step(y, Phi)
            return y_new, y_new
        _, traj = jax.lax.scan(body, y0, None, length=n_steps)
        return traj  # (n_steps, 6)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    colours = ['#1f77b4', '#2ca02c', '#d62728']
    for col, ((title, Phi), col_c) in enumerate(zip(scenarios, colours)):
        traj = np.asarray(full_traj(jnp.asarray(y0_trained), jnp.asarray(Phi)))
        # Top row: capacities (B, S, A)
        axes[0, col].plot(t_grid, traj[:, 0], 'b-', lw=1.5, label=r'$B$ (aerobic)')
        axes[0, col].plot(t_grid, traj[:, 1], 'g-', lw=1.5, label=r'$S$ (strength)')
        axes[0, col].plot(t_grid, traj[:, 3], 'k-', lw=2.0, label=r'$A$ (autonomic)')
        axes[0, col].axhline(p['B_dec'], color='b', linestyle=':', alpha=0.5,
                              label=r'$B_{\rm dec}$')
        axes[0, col].set_ylim(-0.02, 0.7)
        axes[0, col].set_title(title)
        axes[0, col].grid(alpha=0.3)
        if col == 0:
            axes[0, col].set_ylabel('Capacity / amplitude')
            axes[0, col].legend(loc='upper right', fontsize=8)
        # Bottom row: fatigue + sensitivities
        axes[1, col].plot(t_grid, traj[:, 2], 'r-', lw=1.5, label=r'$F$ (fatigue)')
        axes[1, col].plot(t_grid, traj[:, 4], 'm--', lw=1.5,
                           label=r'$K_{FB}$')
        axes[1, col].plot(t_grid, traj[:, 5], 'c--', lw=1.5,
                           label=r'$K_{FS}$')
        axes[1, col].set_xlabel('Time (days)')
        axes[1, col].grid(alpha=0.3)
        if col == 0:
            axes[1, col].set_ylabel('Fatigue / sensitivity')
            axes[1, col].legend(loc='upper right', fontsize=8)

    fig.suptitle('FSA-v5 deterministic rollouts from a trained initial state '
                 r'$(B_0, S_0, A_0)=(0.50, 0.45, 0.45)$ — '
                 'sedentary AND over-training both drive $A \\to 0$',
                 fontsize=11)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'v5_trajectories_three_regimes.png')
    plt.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  saved {out}")


def write_v5_equilibrium_table(p):
    """Table of (Phi_B, Phi_S, A* at A=0 stability, A* interior)."""
    grid = [0.05, 0.15, 0.30, 0.45, 0.70, 1.00]
    out = os.path.join(TABLE_DIR, 'v5_equilibrium_values.tex')
    lines = []
    lines.append(r"\begin{tabular}{cc|ccccc}")
    lines.append(r"\toprule")
    lines.append(r"$\Phi_B$ & $\Phi_S$ & $\bar\mu(0;\Phi)$ & $A^*$ (stable) & $A_{\rm sep}$ & $B^*$ & Regime \\")
    lines.append(r"\midrule")
    for pb in grid:
        for ps in grid:
            mu0 = float(mu_bar(0.0, pb, ps, p))
            roots = find_A_roots(pb, ps, p)
            A_st = roots[-1] if roots else float('nan')
            A_sep = roots[0] if len(roots) >= 2 else float('nan')
            B_at = float(BS_star(A_st if not np.isnan(A_st) else 0.0,
                                  pb, ps, p)[0])
            if mu0 > 0 and len(roots) >= 1:
                regime = "healthy"
            elif mu0 < 0 and len(roots) >= 2:
                regime = "bistable"
            else:
                regime = "collapsed"
            astr = f"${A_st:.3f}$" if not np.isnan(A_st) else "---"
            sepstr = f"${A_sep:.3f}$" if not np.isnan(A_sep) else "---"
            lines.append(
                f"{pb:.2f} & {ps:.2f} & ${mu0:+.3f}$ & {astr} & {sepstr} & "
                f"{B_at:.3f} & {regime} \\\\"
            )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    with open(out, 'w') as f:
        f.write("\n".join(lines))
    print(f"  saved {out}")


def main():
    p = TRUTH_PARAMS
    print("=== FSA-v4 stability analysis ===")
    print("Verifying analytical equilibria against drift_jax ...")
    err = verify_equilibria(p)
    print(f"  max |drift| at analytical equilibria = {err:.2e}")
    assert err < 1e-9, f"Equilibrium check failed: max |drift| = {err}"

    print("Generating figures ...")
    figure_mu_vs_A(p)
    figure_jacobian_eigenvalues(p)
    figure_bifurcation_curve(p)
    figure_basin_diagram(p, axis='B', other_phi=0.3)
    figure_basin_diagram(p, axis='S', other_phi=0.3)

    print("v5 figures (closed-island topology) ...")
    p_v5 = TRUTH_PARAMS_V5
    figure_v5_island(p, p_v5)
    figure_v5_full_bifurcation(p_v5)
    figure_v5_trajectories(p_v5)

    print("Writing tables ...")
    write_equilibrium_table(p)
    write_jacobian_table(p)
    write_lyapunov_summary_table()
    write_v5_equilibrium_table(p_v5)
    print("Done.")


if __name__ == "__main__":
    main()
