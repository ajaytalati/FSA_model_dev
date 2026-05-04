#!/usr/bin/env python3
"""FSA-v2 — T=84 d Banister-horizon long-block scenario.

Drives the plant for 84 days under constant daily Φ = 1.0/day, the
"long aerobic block" operating point on the longest of the bench's
canonical Banister horizons.

Smoke-only basin check for now.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import run_fsa_scenario   # noqa: E402


def main():
    return run_fsa_scenario('84d')


if __name__ == "__main__":
    sys.exit(main())
