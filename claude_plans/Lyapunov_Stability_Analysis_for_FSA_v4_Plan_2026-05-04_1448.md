# Lyapunov Stability Analysis for FSA-v4 — Plan

> Archived from plan mode: 2026-05-04 14:48.

## Context

The user wants to understand the FSA-v4 (6D) model behaviour mathematically — specifically *what the basins of attraction are* — via a Lyapunov stability analysis. The existing `LaTex_docs/sections/05_stability.tex` only contains a Foster–Lyapunov drift inequality for the 3D FSA-v2 system and writes down no equilibria, no Jacobian, no explicit Lyapunov function, and no basins. So the analysis is from scratch on the 6D state $y = (B, S, F, A, K_{FB}, K_{FS})^T$ with constant exogenous input $(\Phi_B, \Phi_S)$.

**State verified on `v4-bimodal-variable-dose-extension` branch:**
- `LaTex_docs/main.tex` includes `09_error_data.tex` and `10_open_loop_limitation.tex` (the §5 errata and §6 open-loop work from earlier in this session is intact). Section numbering after their insertion: §1 physio, §2 math, §3 v3-bimodal, §4 v4-var-dose, §5 errata, §6 open-loop, §7 implementation, **§8 stability**, §9 inference, §10 sandbox. Note that `05_stability.tex` therefore renders as **section 8 in the PDF**, not section 5.
- A precursor stability tool exists: [`tools/generate_stability_data.py`](tools/generate_stability_data.py). It computes a v3-only "Goldilocks surface" $A^*(\Phi_B, \Phi_S)$ by **freezing** $A = A_{\text{TYP}}$ in the $B^*, S^*, F^*$ formulas (so it is not a self-consistent equilibrium), and uses the static $\kappa_{FB}, \kappa_{FS}$ from FSA-v3 — which do not exist as parameters in the v4 `TRUTH_PARAMS`. **Running this script as-is on v4 will fail with `KeyError`.** The new v4 script will reuse the structure but compute the self-consistent equilibrium with dynamic $K^*$ values.
- An archive copy of an earlier draft of this plan already exists at `claude_plans/Lyapunov Stability Analysis for FSA-v4 — Plan.md` (no date suffix, no `Archived from plan mode:` header). The post-approval first action will rename to the canonical `Lyapunov_Stability_Analysis_for_FSA_v4_Plan_<YYYY-MM-DD>_<HHMM>.md` and add the archive header per the project CLAUDE.md rule.

## Why this is mathematically tractable: the cascade structure

The 6D drift in [`models/fsa_high_res/_dynamics.py:74-113`](models/fsa_high_res/_dynamics.py#L74-L113) decomposes as a **feed-forward cascade** under constant $\Phi$:

$$
\underbrace{(K_{FB}, K_{FS})}_{\text{row 5,6}} \;\longrightarrow\; \underbrace{(B, S)}_{\text{rows 1,2}} \;\longrightarrow\; \underbrace{F}_{\text{row 3}} \;\longrightarrow\; \underbrace{A}_{\text{row 4}}
$$

with no feedback up the chain except through $A$ acting back on $(B, S, F)$ via `a_factor_*`. The $K$ block is linear and decouples completely. The $(B, S, F)$ block is linear in its own state for each fixed $A$. The only nonlinear, bifurcating element is the Stuart–Landau equation $\dot A = \mu A - \eta A^3$, where $\mu = \mu(B, S, F)$ is the well-known FSA bifurcation parameter.

This means the entire stability analysis collapses to a 1D fixed-point problem on $A$, with the rest of the state slaved to $A$ at the slow-manifold equilibrium. The basins are therefore determined by the geometry of one scalar potential — but the bifurcation surface lives in stimulus space $(\Phi_B, \Phi_S)$ and is the deliverable the user is asking for.

## Deliverables

1. **Closed-form equilibrium analysis** of the 6D autonomous ODE under constant $\Phi$.
2. **Bifurcation surface** $\mu(\Phi_B, \Phi_S) = 0$ in stimulus space — the boundary between the healthy ($A^* > 0$) and collapsed ($A^* = 0$) regimes.
3. **Composite Lyapunov function** $V = V_K + V_{BS} + V_F + V_A$ proving local exponential stability at each equilibrium and global descent toward the appropriate basin.
4. **Numerical basin map** — drift-only ODE rollouts on a $(\Phi_B, \Phi_S)$ × $A_0$ grid, classified by terminal $A$.
5. **LaTeX write-up** extending section 5 to FSA-v4.

## Plan

### Phase 1 — Symbolic equilibrium analysis (paper math)

1. **K equilibrium (trivial, closed-form):**
   $$K_{Fi}^* = K_{Fi}^0 + \tau_K \mu_K \Phi_i, \qquad i \in \{B, S\}.$$
   Globally exponentially stable; depends only on $\Phi$.

2. **$(B, S)$ equilibrium given $A$:**
   $$B^*(A) = \tau_B \kappa_B \frac{1 + \epsilon_{AB} A}{1 + \epsilon_{AB} A_{\text{TYP}}} \Phi_B, \qquad S^*(A) = \tau_S \kappa_S \frac{1 + \epsilon_{AS} A}{1 + \epsilon_{AS} A_{\text{TYP}}} \Phi_S.$$

3. **$F$ equilibrium given $(A, K^*)$:**
   $$F^*(A, \Phi) = \frac{\tau_F (K_{FB}^* \Phi_B + K_{FS}^* \Phi_S)}{1 + \lambda_A A} \cdot (1 + \lambda_A A_{\text{TYP}}).$$

4. **$A$ self-consistency:** define
   $$\bar\mu(A; \Phi) := \mu_0 + \mu_B B^*(A) + \mu_S S^*(A) - \mu_F F^*(A, \Phi) - \mu_{FF}(F^*(A, \Phi) - F_{\text{TYP}})^2.$$
   The interior equilibria of $A$ solve $\bar\mu(A; \Phi) = \eta A^2$; the boundary equilibrium is $A = 0$. This is a **single transcendental equation in one scalar**, yielding 0, 1, or 2 positive roots depending on $\Phi$. Plot $\bar\mu(A) - \eta A^2$ as a function of $A$ to enumerate roots graphically and analytically.

5. **Bifurcation surface:** the codimension-1 set $\{\Phi : \bar\mu(0; \Phi) = 0\}$ separates the regime where $A = 0$ is locally stable from where it is unstable. Compute it explicitly — under the cascade, $\bar\mu(0; \Phi)$ is a low-degree polynomial in $\Phi_B, \Phi_S$.

### Phase 2 — Local linear stability via the block-triangular Jacobian

Because of the cascade, the $6\times 6$ Jacobian $J = \partial f / \partial y$ at any equilibrium is **block lower-triangular** with diagonal blocks
- $K$-block: $-\frac{1}{\tau_K} I_2$ → eigenvalues $-1/\tau_K$ (×2).
- $(B,S)$-block: $\text{diag}(-1/\tau_B, -1/\tau_S)$ → eigenvalues $-1/\tau_B, -1/\tau_S$.
- $F$-block: $-a_F(A^*)/\tau_F$ → one negative eigenvalue.
- $A$-block: $\bar\mu(A^*; \Phi) - 3\eta (A^*)^2$ → sign determines stability.

So **the entire 6D stability is decided by the $A$-eigenvalue alone** — one scalar quantity. At $A^* = 0$: eigenvalue $= \bar\mu(0; \Phi)$. At $A^* > 0$: eigenvalue $= \bar\mu(A^*) - 3\eta (A^*)^2 = -2\eta (A^*)^2$ (negative ⇒ stable, after using the equilibrium condition $\bar\mu = \eta (A^*)^2$).

**Conclusion (to be proved formally):** whenever an interior equilibrium $A^* > 0$ exists, it is locally exponentially stable; the boundary equilibrium $A = 0$ is locally stable iff $\bar\mu(0; \Phi) < 0$.

### Phase 3 — Composite Lyapunov function

Construct $V(y; \Phi) = V_K + V_{BS} + V_F + V_A$ proving global exponential descent toward the basin of the appropriate equilibrium.

- $V_K = \tfrac{1}{2} \sum_{i \in \{B,S\}} (K_{Fi} - K_{Fi}^*)^2$ → $\dot V_K = -\frac{1}{\tau_K} \cdot 2 V_K$. Clean.
- $V_{BS}$ on centred $(B,S)$ given $A$: quadratic. Treat $A$-induced drift in $a_{factor}$ as a bounded perturbation; absorb via a Young's-inequality cross term.
- $V_F$ similar; rate $a_F(A)/\tau_F$ has a uniform positive lower bound for $A$ in any compact set.
- $V_A$: standard cubic-quartic potential
   - Collapsed regime ($\bar\mu(0;\Phi) < 0$): $V_A = \tfrac{1}{2} A^2$, $\dot V_A = (\bar\mu - \eta A^2) A^2 < 0$ ⇒ GES at $A = 0$.
   - Healthy regime ($\bar\mu(0;\Phi) > 0$, with $A^* > 0$): $V_A(A) = \tfrac{\eta}{4}(A^2 - (A^*)^2)^2$ — double-well potential. Acts as a Lyapunov function for the **interior** equilibrium on $A > 0$.

The composite $V$ proves: under constant $\Phi$, the deterministic ODE has **at most two attractors** ($A = 0$ and $A^* > 0$) and the basin geometry is governed entirely by the sign of $\bar\mu(0; \Phi)$ together with the value of $A_0$.

**Caveat:** the composite Lyapunov function is provisional. Cross-coupling terms (Young's-inequality estimates, choice of weights) need to be checked to ensure $\dot V \leq -c V$ globally. If the algebra doesn't close cleanly, fall back to a per-block argument plus singular-perturbation reasoning (the slow manifold $A$ vs fast dynamics $K, B, S, F$ — the cascade structure is precisely what singular perturbation theory needs).

### Phase 4 — Numerical basin map

Create `tools/stability_basins_v4.py` (a v4 generalisation of [`tools/generate_stability_data.py`](tools/generate_stability_data.py); reuse its meshgrid + `TRUTH_PARAMS` import pattern, but solve the *self-consistent* equilibrium rather than freezing $A = A_{\text{TYP}}$, and use the dynamic $K_{FB}^*, K_{FS}^*$ for v4):
- 4D-Runge-Kutta integration of the deterministic drift only (set diffusion to zero).
- Sweep $(\Phi_B, \Phi_S) \in [0, 3]^2$ on, say, a 60×60 grid.
- For each $\Phi$, sweep initial $A_0 \in [0, 1]$ on a 50-point grid, with $(B, S, F, K_{FB}, K_{FS})$ initialised at their slaved equilibria for that $A_0$ (or alternatively at a fixed physiological prior).
- Integrate to $T = 200$ days (well past slowest timescale $\tau_S = 60$).
- Classify terminal $A$: collapsed if $A < \epsilon$, healthy if $A > \epsilon$, with $\epsilon \approx 0.01$.
- Output two figures:
   1. **Bifurcation curve** in $(\Phi_B, \Phi_S)$: contour $\bar\mu(0; \Phi) = 0$, computed analytically and overlaid against numerical classification of $A_0 = 0^+$ rollouts.
   2. **Basin diagram** in $(\Phi_B, A_0)$ at fixed $\Phi_S$, and symmetric in $(\Phi_S, A_0)$ — colour-coded by terminal regime.

Save to `LaTex_docs/figures/v4_stability_*.png`. Reuse `_dynamics.py:drift_jax` for consistency — do not re-derive the drift.

### Phase 5 — Document (revised: whole new numbered section per user request)

Create a brand-new section file `LaTex_docs/sections/11_stability_v4.tex` and wire it into `main.tex` after `10_open_loop_limitation.tex` and before `03_implementation.tex`. With this insertion, the rendered numbering becomes:

> §1 physiology, §2 math, §3 v3-bimodal, §4 v4-var-dose, §5 errata, §6 open-loop, **§7 v4 stability (new)**, §8 implementation, §9 stability-v2 (existing `05_stability.tex`), §10 inference, §11 sandbox.

This gives v4 stability its own top-level chapter immediately after the open-loop critique, while leaving the v2 Foster–Lyapunov section intact as §9 for historical/comparison purposes.

The new §7 contents:

- **§7.1 Cascade reduction of the 6D system** — slow-manifold argument; closed-form $K^*, B^*(A), S^*(A), F^*(A)$.
- **§7.2 The bifurcation parameter $\bar\mu(A; \Phi)$** — explicit polynomial in $\Phi_B, \Phi_S$ at $A=0$; plot of $\bar\mu(A; \Phi)$ vs $A$ at three representative $\Phi$ values (collapsed / critical / healthy).
- **§7.3 Local stability via block-triangular Jacobian** — eigenvalues table; one row per equilibrium per regime.
- **§7.4 Composite Lyapunov function** — theorem statement, proof sketch with Young's-inequality estimates; full algebra deferred to appendix.
- **§7.5 Numerical basin diagrams** — figures from Phase 4 (bifurcation curve, basin maps).
- **§7.6 Equilibrium values table** — sample $(\Phi_B, \Phi_S)$ tabulated against $K^*, B^*, S^*, F^*, \bar\mu, A^*$ to anchor the math in numbers (mirrors the table at the end of `tools/generate_stability_data.py`, generalised to v4).
- **§7.7 Stochastic extension (provisional)** — Foster–Lyapunov extension; defer full proof.

Required figures (PNG, saved to `LaTex_docs/figures/`):
1. `v4_bifurcation_curve.png` — analytical $\bar\mu(0; \Phi) = 0$ contour overlaid on numerical regime classification in $(\Phi_B, \Phi_S)$ plane.
2. `v4_mu_vs_A_three_regimes.png` — $\bar\mu(A; \Phi) - \eta A^2$ vs $A$ at three $\Phi$ values, showing 0 / 1 / 2 positive roots.
3. `v4_basin_diagram_phiB.png` — basin classification in $(\Phi_B, A_0)$ at $\Phi_S$ fixed at $\Phi_S = 1.0$.
4. `v4_basin_diagram_phiS.png` — symmetric: $(\Phi_S, A_0)$ at $\Phi_B = 1.0$.
5. `v4_jacobian_eigenvalues.png` — eigenvalue magnitudes vs $\Phi_B$ at $\Phi_S = 1$, showing the bifurcation in the A-block while other blocks stay constant.

Required tables in the LaTeX:
- `tab:v4_equilibrium_values` — 5×5 grid of $(\Phi_B, \Phi_S)$ with columns $K_{FB}^*, K_{FS}^*, B^*, S^*, F^*, \bar\mu, A^*$.
- `tab:v4_jacobian_blocks` — eigenvalues per block at the healthy and collapsed equilibria.
- `tab:v4_lyapunov_summary` — composite $V$ components, decay rate, validity domain.

## Critical files

- [`models/fsa_high_res/_dynamics.py`](models/fsa_high_res/_dynamics.py) — drift function to be reused, *not* reimplemented.
- [`LaTex_docs/sections/05_stability.tex`](LaTex_docs/sections/05_stability.tex) — to be extended.
- New: `tools/stability_basins_v4.py` — basin sweep.
- New: `LaTex_docs/figures/v4_bifurcation_curve.png`, `v4_basin_diagram_*.png`.

## Verification

- **Math:** Every closed-form expression in Phase 1 verified by symbolic substitution back into the drift equations and confirming $f(y^*; \Phi) = 0$. Bifurcation polynomial cross-checked against the symbolic value of $\bar\mu(0; \Phi)$ from the code (just substitute $A=0, B=B^*, S=S^*, F=F^*$).
- **Local stability:** Jacobian computed via `jax.jacfwd(drift_jax)` at each numerical equilibrium; eigenvalues compared against the analytical block-triangular prediction.
- **Lyapunov:** numerically verify $\dot V < 0$ on a Monte Carlo cloud of states inside each basin (10⁵ samples, no noise).
- **Basins:** visual consistency check — the analytically-drawn bifurcation curve $\bar\mu(0; \Phi) = 0$ should coincide with the empirical regime boundary in the numerical sweep within grid resolution.
- **Document:** PDF compiles, references resolve, figures load, and the new claim "$A^* > 0$ exists iff $\bar\mu(0;\Phi) > 0$" is true on every grid point in the basin map.

## Open questions / scope

- **Deterministic vs stochastic.** This plan covers basins of the deterministic ODE (the natural setting for the question "what are the basins?"). The stochastic version — metastable lifetimes, quasi-stationary distributions — is a much harder follow-up and is *not* in scope here.
- **Constant vs time-varying $\Phi$.** Equilibria and basins are defined for autonomous systems, so this analysis is for constant exogenous input. A time-varying $\Phi(t)$ analysis (Lyapunov exponents, time-varying ROAs) would be a separate undertaking.
- **Section ordering.** Verified on the `v4-bimodal-variable-dose-extension` branch: `09_error_data.tex` and `10_open_loop_limitation.tex` are wired into `main.tex`, so the rendered §8 is `05_stability.tex`. I will append §8.3–§8.8 in place rather than insert a new top-level section.

## Post-approval first actions (per project CLAUDE.md "Plans get archived" rule)

Before doing any phase-1 work:
1. Rename `claude_plans/Lyapunov Stability Analysis for FSA-v4 — Plan.md` to `claude_plans/Lyapunov_Stability_Analysis_for_FSA_v4_Plan_<creation-date>_<HHMM>.md` per the slugified-title convention in CLAUDE.md.
2. Add `> Archived from plan mode: <YYYY-MM-DD HH:MM>.` directly under the title.
3. Sync the canonical plan content from `~/.claude/plans/i-want-you-to-adaptive-perlis.md` into the renamed archive.
4. From this point on, every meaningful update to the master plan is mirrored to the archive with a new `> Updated: <YYYY-MM-DD HH:MM> — <summary>.` line below the archive header.
