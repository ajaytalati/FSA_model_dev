import Lake
open Lake DSL

-- Per the LEAN4-first charter (LaTex_docs/lean4_first_charter.pdf),
-- this package is the canonical reference implementation of the FSA-v5
-- mathematical model. Python is differentially tested against the binary
-- produced here.
--
-- This is a Mathlib project so future theorems (Lipschitz bounds,
-- basin-of-attraction, monotonicity etc.) can be stated against
-- mathlib's `ℝ` / `MeasureTheory` / `ODE` infrastructure. The pre-built
-- mathlib cache is pulled with `lake exe cache get` — see Mathlib's
-- "Setting up a project that depends on Mathlib" docs. NEVER compile
-- mathlib from scratch (~2h on this hardware); always `cache get`.

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.30.0-rc2"

package Fsa where
  leanOptions := #[
    ⟨`pp.unicode.fun, true⟩,
    ⟨`autoImplicit, false⟩
  ]

@[default_target]
lean_lib Fsa where
  globs := #[.andSubmodules `Fsa]

-- CLI entry point used by the Python differential-test bridge.
-- Accepts JSON on stdin, prints JSON on stdout.
@[default_target]
lean_exe fsa_v5_cli where
  root := `Main
