"""FSA-v5 Fisher Information Matrix analysis.

Computes the *expected* observation FIM for the full FSA-v5 model — including
the v5 deconditioning extension — across all 4 Gaussian channels (HR, Stress,
Steps, VolumeLoad) plus the Bernoulli sleep channel, over a representative
informative Phi schedule that exercises both tails of the v5 island.

Outputs:
  1. LaTex_docs/figures/v5_fim_eigenvalues.png  — eigenvalue spectrum.
  2. LaTex_docs/tables/v5_fim_summary.tex       — condition number + rank.
  3. LaTex_docs/tables/v5_fim_slack_directions.tex — unidentifiable combos.
  4. Console: full eigenvalue dump + suggested reparameterisations.

FIM construction (log-parameter coordinates for the lognormal-prior params,
identity for the normal-prior ones):

  Gaussian channel c, time k:
    F^{c,k}_{ij} = (1/σ_c^2) ∂μ_{c,k}/∂η_i · ∂μ_{c,k}/∂η_j
                 + (2/σ_c^2) ∂σ_c/∂η_i  · ∂σ_c/∂η_j

  Bernoulli sleep channel, time k:
    F^{S,k}_{ij} = (1/[p_k(1-p_k)]) ∂p_k/∂η_i · ∂p_k/∂η_j

  Sum over (c, k) gives the total expected FIM in coordinates η,
  where η_i = log θ_i for lognormal-prior params, η_i = θ_i otherwise.
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

from collections import OrderedDict

from models.fsa_high_res._dynamics import (
    TRUTH_PARAMS_V5, A_TYP, F_TYP, drift_jax,
)

# Inlined from models/fsa_high_res/estimation.py PARAM_PRIOR_CONFIG to avoid
# triggering an unrelated EstimationModel constructor mismatch on import.
# Keep in sync with that file.
PARAM_PRIOR_CONFIG = OrderedDict([
    # --- Dynamics (FSA-v4) ---
    ('tau_B',       ('lognormal', (math.log(42.0), 0.10))),
    ('kappa_B',     ('lognormal', (math.log(0.01248), 0.20))),
    ('epsilon_AB',  ('lognormal', (math.log(0.40), 0.05))),
    ('tau_S',       ('lognormal', (math.log(60.0), 0.10))),
    ('kappa_S',     ('lognormal', (math.log(0.00816), 0.20))),
    ('epsilon_AS',  ('lognormal', (math.log(0.20), 0.05))),
    ('tau_F',       ('lognormal', (math.log(6.3636), 0.15))),
    ('lambda_A',    ('lognormal', (math.log(1.00), 0.05))),
    ('KFB_0',       ('lognormal', (math.log(0.030), 0.20))),
    ('KFS_0',       ('lognormal', (math.log(0.050), 0.20))),
    ('tau_K',       ('lognormal', (math.log(21.0), 0.10))),
    ('mu_K',        ('lognormal', (math.log(0.005), 0.20))),
    ('mu_0',        ('lognormal', (math.log(0.036), 0.20))),
    ('mu_B',        ('lognormal', (math.log(0.30), 0.20))),
    ('mu_S',        ('lognormal', (math.log(0.15), 0.20))),
    ('mu_F',        ('lognormal', (math.log(0.26), 0.20))),
    ('mu_FF',       ('lognormal', (math.log(0.40), 0.05))),
    ('eta',         ('lognormal', (math.log(0.20), 0.15))),
    # --- Obs Coefficients ---
    ('HR_base',     ('normal',    (62.0, 2.0))),
    ('kappa_B_HR',  ('lognormal', (math.log(12.0), 0.15))),
    ('alpha_A_HR',  ('lognormal', (math.log(3.0),  0.20))),
    ('beta_C_HR',   ('normal',    (-2.5, 0.5))),
    ('sigma_HR',    ('lognormal', (math.log(2.0), 0.20))),
    ('k_C',         ('lognormal', (math.log(3.0), 0.15))),
    ('k_A',         ('lognormal', (math.log(2.0), 0.25))),
    ('c_tilde',     ('normal',    (0.5, 0.25))),
    ('S_base',      ('normal',    (30.0, 3.0))),
    ('k_F',         ('lognormal', (math.log(20.0), 0.20))),
    ('k_A_S',       ('lognormal', (math.log(8.0),  0.25))),
    ('beta_C_S',    ('normal',    (-4.0, 0.8))),
    ('sigma_S_obs', ('lognormal', (math.log(4.0), 0.20))),
    ('mu_step0',    ('normal',    (5.5, 0.3))),
    ('beta_B_st',   ('lognormal', (math.log(0.8), 0.20))),
    ('beta_F_st',   ('lognormal', (math.log(0.5), 0.25))),
    ('beta_A_st',   ('lognormal', (math.log(0.3), 0.25))),
    ('beta_C_st',   ('normal',    (-0.8, 0.2))),
    ('sigma_st',    ('lognormal', (math.log(0.5), 0.15))),
    ('beta_S_VL',   ('lognormal', (math.log(100.0), 0.15))),
    ('beta_F_VL',   ('lognormal', (math.log(20.0),  0.20))),
    ('sigma_VL',    ('lognormal', (math.log(10.0),  0.20))),
])

FIG_DIR = "LaTex_docs/figures"
TABLE_DIR = "LaTex_docs/tables"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)


# ── Add v5 deconditioning params to the prior config ──────────────────────────
# (These don't yet appear in PARAM_PRIOR_CONFIG which is v4-era)
V5_EXTRA_PRIORS = [
    ('B_dec',     ('lognormal', (math.log(0.07), 0.20))),
    ('S_dec',     ('lognormal', (math.log(0.07), 0.20))),
    ('mu_dec_B',  ('lognormal', (math.log(0.10), 0.30))),
    ('mu_dec_S',  ('lognormal', (math.log(0.10), 0.30))),
    # n_dec held fixed at 4 (structural shape parameter, not estimated)
]
ALL_PRIORS = list(PARAM_PRIOR_CONFIG.items()) + V5_EXTRA_PRIORS
PARAM_NAMES = [k for k, _ in ALL_PRIORS]
PARAM_TYPES = {k: v[0] for k, v in ALL_PRIORS}      # 'lognormal' or 'normal'
PARAM_TRUTH = {}
for name, (ptype, pa) in ALL_PRIORS:
    if ptype == 'lognormal':
        PARAM_TRUTH[name] = math.exp(pa[0])         # mode of the lognormal
    else:
        PARAM_TRUTH[name] = pa[0]                   # mean of the normal
# Override truths from TRUTH_PARAMS_V5 for the dynamics (we know exact values)
for k in TRUTH_PARAMS_V5:
    if k in PARAM_TRUTH:
        PARAM_TRUTH[k] = float(TRUTH_PARAMS_V5[k])

N_PARAMS = len(PARAM_NAMES)
print(f"Total estimated parameters: N = {N_PARAMS}")


# ── Build truth parameter vector in η-coordinates ─────────────────────────────
# eta_i = log(theta_i) if lognormal-prior, else theta_i.

def theta_to_eta(theta_dict):
    return jnp.array([
        jnp.log(theta_dict[n]) if PARAM_TYPES[n] == 'lognormal' else theta_dict[n]
        for n in PARAM_NAMES
    ])

def eta_to_theta_dict(eta):
    out = {}
    for i, n in enumerate(PARAM_NAMES):
        out[n] = jnp.exp(eta[i]) if PARAM_TYPES[n] == 'lognormal' else eta[i]
    return out

ETA_TRUTH = theta_to_eta(PARAM_TRUTH)


# ── Representative informative Phi schedule (84 days, 15-min bin obs) ────────
# Matches production setup in models/fsa_high_res/simulation.py:
#   FSA_STEP_MINUTES default = 15  → BINS_PER_DAY = 96, DT_BIN_DAYS = 1/96 day.
# Each integration step is one bin, no sub-stepping (matches _plant_em_step).

T_TOTAL = 84.0                          # days
BINS_PER_DAY = 96                       # 15-min bins
DT = 1.0 / BINS_PER_DAY                 # integration step
N_STEPS = int(round(T_TOTAL * BINS_PER_DAY))   # 8064 bins
T_OBS_IDX = jnp.arange(N_STEPS)         # observe every bin (gating below)
N_OBS = N_STEPS

# ── Channel-level gating masks (sleep / wake / per-day) ──────────────────────
# Subject sleep window: 22:00 → 06:00 next day (8 hours). Bin index → hour:
#   hour_of_day = (k % BINS_PER_DAY) * (24 / BINS_PER_DAY)
def _build_gating():
    bin_in_day = np.arange(N_STEPS) % BINS_PER_DAY
    hour = bin_in_day * (24.0 / BINS_PER_DAY)
    sleep_mask = ((hour >= 22.0) | (hour < 6.0)).astype(np.float64)
    wake_mask  = 1.0 - sleep_mask
    # VolumeLoad: aggregate per training session, modelled as one obs/day at
    # 18:00 (bin index 72 in each day).
    vl_mask = (bin_in_day == 72).astype(np.float64)
    return sleep_mask, wake_mask, vl_mask
SLEEP_MASK_NP, WAKE_MASK_NP, VL_MASK_NP = _build_gating()
SLEEP_MASK = jnp.asarray(SLEEP_MASK_NP)   # HR observations (sleep-gated)
WAKE_MASK  = jnp.asarray(WAKE_MASK_NP)    # Stress + Steps (wake-gated)
VL_MASK    = jnp.asarray(VL_MASK_NP)      # VolumeLoad (1/day at 18:00)
# Effective channel obs counts (informational; not used in computation):
N_HR     = int(SLEEP_MASK_NP.sum())
N_STRESS = int(WAKE_MASK_NP.sum())
N_STEPS_OBS = int(WAKE_MASK_NP.sum())
N_VL     = int(VL_MASK_NP.sum())
N_SLEEP  = N_STEPS                          # sleep-label observed every bin
print(f"  bin grid: {N_STEPS} bins of {DT*1440:.1f} min over {T_TOTAL} days")
print(f"  per-channel observation counts: HR={N_HR} (sleep-gated), "
      f"Stress={N_STRESS} (wake-gated), Steps={N_STEPS_OBS} (wake-gated), "
      f"VL={N_VL} (1/day), Sleep={N_SLEEP} (every bin)")

def build_phi_schedule():
    """4-phase schedule covering both tails of the v5 island.

    Phi values are bin-step resolution (15 min); training sessions are the
    same daily total as before but spread across all bins to keep the FIM
    integration smooth — equivalent to a continuous low-grade stimulus,
    appropriate for steady-state behavioural FIM analysis. For event-based
    Phi (discrete sessions), the FIM script can be re-run with a
    session-aware schedule.
    """
    t = np.arange(N_STEPS) * DT
    phi_b = np.zeros(N_STEPS)
    phi_s = np.zeros(N_STEPS)
    # Phase 1 (0-21 d): build-up — ramp from 0 to 0.30 in both modalities
    m1 = (t < 21.0)
    phi_b[m1] = 0.30 * (t[m1] / 21.0)
    phi_s[m1] = 0.30 * (t[m1] / 21.0)
    # Phase 2 (21-49 d): balanced moderate, inside the v5 island
    m2 = (t >= 21.0) & (t < 49.0)
    phi_b[m2] = 0.30
    phi_s[m2] = 0.30
    # Phase 3 (49-63 d): detraining — Phi → 0
    m3 = (t >= 49.0) & (t < 63.0)
    phi_b[m3] = 0.0
    phi_s[m3] = 0.0
    # Phase 4 (63-84 d): over-training block
    m4 = (t >= 63.0)
    phi_b[m4] = 1.0
    phi_s[m4] = 1.0
    return jnp.stack([phi_b, phi_s], axis=1), t

PHI, T_GRID = build_phi_schedule()


# ── Forward integrate the deterministic model in η-coordinates ────────────────

def forward_state(eta):
    """Integrate the deterministic FSA-v5 ODE from a fixed initial state.
    Returns the state trajectory (n_steps, 6).
    """
    p = eta_to_theta_dict(eta)
    # Initial state: trained athlete
    y0 = jnp.array([0.50, 0.45, 0.20, 0.45,
                    p['KFB_0'] + p['tau_K']*p['mu_K']*0.5,
                    p['KFS_0'] + p['tau_K']*p['mu_K']*0.5])
    # We need the dynamics-only params subset for drift_jax
    dyn_p = {**p, 'n_dec': 4.0}      # n_dec held fixed
    dyn_p_jax = {k: jnp.asarray(v) for k, v in dyn_p.items()
                  if k in TRUTH_PARAMS_V5}
    def step(y, k):
        y_next = y + DT * drift_jax(y, dyn_p_jax, PHI[k])
        # Reflection / clamping
        y_next = y_next.at[0].set(jnp.clip(y_next[0], 1e-4, 1.0 - 1e-4))
        y_next = y_next.at[1].set(jnp.clip(y_next[1], 1e-4, 1.0 - 1e-4))
        y_next = y_next.at[2].set(jnp.maximum(y_next[2], 0.0))
        y_next = y_next.at[3].set(jnp.maximum(y_next[3], 0.0))
        y_next = y_next.at[4].set(jnp.maximum(y_next[4], 0.0))
        y_next = y_next.at[5].set(jnp.maximum(y_next[5], 0.0))
        return y_next, y_next
    _, traj = jax.lax.scan(step, y0, jnp.arange(N_STEPS))
    return traj


# ── Observation predictions (mean + noise) at observation times ───────────────

def observation_predictions(eta):
    """For each observation time, compute (mu_HR, sigma_HR, mu_S, sigma_S,
    mu_st, sigma_st, mu_VL, sigma_VL, p_sleep).

    Returns a single flat vector of length N_OBS * 9 so we can take its
    Jacobian wrt eta in one shot.
    """
    p = eta_to_theta_dict(eta)
    traj = forward_state(eta)
    y_obs = traj[T_OBS_IDX]                  # (N_OBS, 6)
    B = y_obs[:, 0]; S = y_obs[:, 1]
    F = y_obs[:, 2]; A = y_obs[:, 3]
    # Production circadian forcing: cos(2π t_days) — one cycle per day.
    # Matches simulation.py:circadian(t_days, phi=0).
    t_obs_days = jnp.arange(N_OBS, dtype=jnp.float64) * DT
    C = jnp.cos(2.0 * jnp.pi * t_obs_days)

    mu_HR = p['HR_base'] - p['kappa_B_HR']*B + p['alpha_A_HR']*A + p['beta_C_HR']*C
    mu_S  = p['S_base']  + p['k_F']*F        - p['k_A_S']*A      + p['beta_C_S']*C
    mu_st = (p['mu_step0'] + p['beta_B_st']*B - p['beta_F_st']*F
             + p['beta_A_st']*A + p['beta_C_st']*C)
    mu_VL = p['beta_S_VL']*S - p['beta_F_VL']*F
    z     = p['k_C']*C + p['k_A']*A - p['c_tilde']
    p_sleep = jax.nn.sigmoid(z)

    sig_HR = jnp.full(N_OBS, p['sigma_HR'])
    sig_S  = jnp.full(N_OBS, p['sigma_S_obs'])
    sig_st = jnp.full(N_OBS, p['sigma_st'])
    sig_VL = jnp.full(N_OBS, p['sigma_VL'])
    return jnp.concatenate([mu_HR, sig_HR, mu_S, sig_S, mu_st, sig_st,
                             mu_VL, sig_VL, p_sleep])


# ── Build FIM by chain-rule: dpred/d eta -> per-channel weighted outer prod ──

def compute_fim(eta):
    pred = observation_predictions(eta)
    J = jax.jacfwd(observation_predictions)(eta)        # (9*N_OBS, N_PARAMS)
    # Slice predictions
    K = N_OBS
    mu_HR  = pred[0*K:1*K];  sig_HR = pred[1*K:2*K]
    mu_S   = pred[2*K:3*K];  sig_S  = pred[3*K:4*K]
    mu_st  = pred[4*K:5*K];  sig_st = pred[5*K:6*K]
    mu_VL  = pred[6*K:7*K];  sig_VL = pred[7*K:8*K]
    p_sl   = pred[8*K:9*K]
    JmuHR  = J[0*K:1*K]; JsHR  = J[1*K:2*K]
    JmuS   = J[2*K:3*K]; JsS   = J[3*K:4*K]
    JmuST  = J[4*K:5*K]; JsST  = J[5*K:6*K]
    JmuVL  = J[6*K:7*K]; JsVL  = J[7*K:8*K]
    JpSL   = J[8*K:9*K]

    fim = jnp.zeros((N_PARAMS, N_PARAMS))
    # Each Gaussian channel has a per-bin presence mask m_k ∈ {0, 1}; the
    # FIM contribution at bin k is multiplied by m_k.
    for mu_c, sig_c, Jmu, Jsig, mask in [
        (mu_HR, sig_HR, JmuHR, JsHR, SLEEP_MASK),   # HR sleep-gated
        (mu_S,  sig_S,  JmuS,  JsS,  WAKE_MASK),    # Stress wake-gated
        (mu_st, sig_st, JmuST, JsST, WAKE_MASK),    # Steps wake-gated
        (mu_VL, sig_VL, JmuVL, JsVL, VL_MASK),      # VL once per day
    ]:
        weight = mask / (sig_c ** 2)
        fim = fim + (Jmu * weight[:, None]).T @ Jmu
        fim = fim + 2.0 * (Jsig * weight[:, None]).T @ Jsig
    # Bernoulli sleep — observed every bin (no gating)
    inv_var_sl = 1.0 / (p_sl * (1.0 - p_sl) + 1e-12)
    fim = fim + (JpSL * inv_var_sl[:, None]).T @ JpSL
    return fim


# ── Run analysis ──────────────────────────────────────────────────────────────

def main():
    print("Computing v5 observation FIM ...")
    F = np.asarray(compute_fim(ETA_TRUTH))
    print(f"  FIM shape: {F.shape}")
    F_sym = 0.5 * (F + F.T)         # numerical symmetrisation
    eigvals, eigvecs = np.linalg.eigh(F_sym)
    eigvals = eigvals[::-1]; eigvecs = eigvecs[:, ::-1]
    eigvals_pos = np.maximum(eigvals, 0.0)
    cond = eigvals_pos[0] / max(eigvals_pos[-1], 1e-30)
    rank_eps = 1e-8 * eigvals_pos[0]
    rank = int(np.sum(eigvals_pos > rank_eps))

    print(f"\nFIM eigenvalues (top 5):  {eigvals_pos[:5]}")
    print(f"FIM eigenvalues (bottom 5): {eigvals_pos[-5:]}")
    print(f"Condition number (max/min eig): {cond:.3e}")
    print(f"Numerical rank (eig > {rank_eps:.2e}): {rank} / {N_PARAMS}")

    # ── Save eigenvalue spectrum figure ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(np.arange(1, N_PARAMS+1), eigvals_pos + 1e-30, 'o-', markersize=4)
    ax.axhline(rank_eps, color='red', linestyle='--',
                label=f'rank threshold = {rank_eps:.2e}')
    ax.set_xlabel('Eigenvalue index (sorted descending)')
    ax.set_ylabel('FIM eigenvalue (log scale)')
    ax.set_title(f'FSA-v5 expected FIM spectrum (N={N_PARAMS}, rank={rank})')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'v5_fim_eigenvalues.png'), dpi=110)
    plt.close(fig)
    print(f"  saved figures/v5_fim_eigenvalues.png")

    # ── Slack directions (smallest eigenvectors) ─────────────────────────────
    print("\nSmallest 5 eigenvectors (largest projection onto these = "
          "least-identified parameter directions):")
    slack_lines = []
    slack_lines.append(r"\begin{tabular}{rlp{8cm}}")
    slack_lines.append(r"\toprule")
    slack_lines.append(r"\# & Eigenvalue & Top contributing parameters (|coef| > 0.30) \\")
    slack_lines.append(r"\midrule")
    for j in range(min(5, N_PARAMS)):
        idx = N_PARAMS - 1 - j
        ev = eigvals_pos[idx]
        v = eigvecs[:, idx]
        order = np.argsort(np.abs(v))[::-1]
        contribs = []
        for k in order:
            if abs(v[k]) >= 0.30:
                sign = '+' if v[k] > 0 else '-'
                contribs.append(f"{sign}{abs(v[k]):.2f}\\,{PARAM_NAMES[k].replace('_','\\_')}")
            if len(contribs) >= 4:
                break
        contrib_str = " ".join(contribs) if contribs else "(no single dominant component)"
        print(f"  Eig #{idx+1}: λ = {ev:.3e}, dominant: " + ", ".join(
            f"{PARAM_NAMES[k]} ({v[k]:+.2f})"
            for k in order[:4]))
        slack_lines.append(f"{idx+1} & ${ev:.2e}$ & {contrib_str} \\\\")
    slack_lines.append(r"\bottomrule")
    slack_lines.append(r"\end{tabular}")
    with open(os.path.join(TABLE_DIR, 'v5_fim_slack_directions.tex'), 'w') as f:
        f.write("\n".join(slack_lines))
    print(f"  saved tables/v5_fim_slack_directions.tex")

    # ── Summary table ────────────────────────────────────────────────────────
    summary = [
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Metric & Value \\",
        r"\midrule",
        f"Number of estimated parameters $N$ & {N_PARAMS} \\\\",
        f"Number of observation events $K$ & {N_OBS} \\\\",
        f"Number of observation channels & 5 (HR, Stress, Steps, VL, Sleep) \\\\",
        f"Largest FIM eigenvalue $\\lambda_{{\\max}}$ & ${eigvals_pos[0]:.3e}$ \\\\",
        f"Smallest FIM eigenvalue $\\lambda_{{\\min}}$ & ${eigvals_pos[-1]:.3e}$ \\\\",
        f"Condition number $\\kappa(F)$ & ${cond:.3e}$ \\\\",
        f"Numerical rank (threshold $10^{{-8}} \\lambda_{{\\max}}$) & {rank} / {N_PARAMS} \\\\",
        f"Effective rank (threshold $10^{{-4}} \\lambda_{{\\max}}$) & {int(np.sum(eigvals_pos > 1e-4 * eigvals_pos[0]))} / {N_PARAMS} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    with open(os.path.join(TABLE_DIR, 'v5_fim_summary.tex'), 'w') as f:
        f.write("\n".join(summary))
    print(f"  saved tables/v5_fim_summary.tex")

    # ── Top-5 most-identifiable directions (sanity check) ────────────────────
    print("\nTop 5 best-identified directions:")
    for j in range(min(5, N_PARAMS)):
        v = eigvecs[:, j]
        order = np.argsort(np.abs(v))[::-1]
        print(f"  Eig #{j+1}: λ = {eigvals_pos[j]:.3e}, dominant: "
              + ", ".join(f"{PARAM_NAMES[k]} ({v[k]:+.2f})"
                          for k in order[:3]))

    # ── Pairwise correlation diagnostic ──────────────────────────────────────
    # Approximate parameter covariance ≈ F^{-1} (Cramér-Rao). Restricted to
    # the well-conditioned subspace (Tikhonov-regularised inverse) so the
    # correlations are bounded in [-1, 1].
    reg = 1e-3 * eigvals_pos[0]    # 0.1% of largest eig — modest regularisation
    F_inv = np.linalg.inv(F_sym + reg * np.eye(N_PARAMS))
    diag = np.sqrt(np.maximum(np.diag(F_inv), 1e-30))
    corr = F_inv / np.outer(diag, diag)
    np.clip(corr, -1.0, 1.0, out=corr)
    pairs = []
    for i in range(N_PARAMS):
        for j in range(i+1, N_PARAMS):
            if abs(corr[i, j]) > 0.95:
                pairs.append((abs(corr[i, j]), i, j, corr[i, j]))
    pairs.sort(reverse=True)
    print(f"\nMost-correlated parameter pairs (|corr| > 0.95, reg = {reg:.2e}):")
    if not pairs:
        print("  none")
    for _, i, j, signed in pairs[:20]:
        print(f"  {PARAM_NAMES[i]:<14s} <-> {PARAM_NAMES[j]:<14s} : corr = {signed:+.4f}")

    # ── Reparameterisation recommendation table ──────────────────────────────
    print("\nReparameterisation candidates (slack directions with single dominant pair):")
    repar_lines = [
        r"\begin{tabular}{lp{6cm}}",
        r"\toprule",
        r"Slack direction & Suggested reparameterisation \\",
        r"\midrule",
    ]
    suggestions_added = 0
    for j in range(min(8, N_PARAMS)):
        idx = N_PARAMS - 1 - j
        ev = eigvals_pos[idx]
        if ev > 1e-4 * eigvals_pos[0]:
            continue   # Not really slack, skip
        v = eigvecs[:, idx]
        order = np.argsort(np.abs(v))[::-1]
        top1, top2 = order[0], order[1]
        n1, n2 = PARAM_NAMES[top1], PARAM_NAMES[top2]
        if abs(v[top1]) > 0.40 and abs(v[top2]) > 0.40:
            sign = '+' if np.sign(v[top1]) == np.sign(v[top2]) else '-'
            ratio_kind = "ratio" if sign == '-' else "product"
            comment = (f"$({n1.replace('_','\\_')}, {n2.replace('_','\\_')})$"
                        f" form a {ratio_kind} — fix one, infer the other")
            repar_lines.append(
                f"$\\lambda_{{{idx+1}}} = {ev:.2e}$ & {comment} \\\\"
            )
            suggestions_added += 1
    if suggestions_added == 0:
        repar_lines.append(r"(no clean two-parameter degeneracies — slack is multi-way) & --- \\")
    repar_lines.append(r"\bottomrule")
    repar_lines.append(r"\end{tabular}")
    with open(os.path.join(TABLE_DIR, 'v5_fim_reparam.tex'), 'w') as f:
        f.write("\n".join(repar_lines))
    print(f"  saved tables/v5_fim_reparam.tex")


if __name__ == "__main__":
    main()
