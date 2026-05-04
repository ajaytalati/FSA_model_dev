#!/usr/bin/env python3
"""FSA-v2 — T=42 d Banister-horizon max-sustainable-load scenario.

Drives the plant for 42 days under constant daily Φ = 1.5/day. This is
the canonical Banister chronic timescale; per the project memory note
"T=42 strategy = Banister overload", expected end-of-trial behaviour is
B grows roughly linearly toward ~0.9 with A inflecting around day 20.

Smoke-only basin check for now (numerical thresholds pending the FSA
team's reference data — see scenarios/_common.py:EXPECTED_END_B).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import run_fsa_scenario   # noqa: E402


def main():
    return run_fsa_scenario('42d')


if __name__ == "__main__":
    sys.exit(main())
