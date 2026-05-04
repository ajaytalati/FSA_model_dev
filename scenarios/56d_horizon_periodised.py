#!/usr/bin/env python3
"""FSA-v2 — T=56 d Banister-horizon mid-load scenario.

Drives the plant for 56 days under constant daily Φ = 1.2/day, a
moderately reduced training load on a longer-than-canonical horizon.

Smoke-only basin check for now.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import run_fsa_scenario   # noqa: E402


def main():
    return run_fsa_scenario('56d')


if __name__ == "__main__":
    sys.exit(main())
