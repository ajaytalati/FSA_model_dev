"""Python bridge to the LEAN4 reference binary for FSA-v5.

Per the LEAN4-first charter (LaTex_docs/lean4_first_charter.pdf §5):
the LEAN4 binary at `lean/.lake/build/bin/fsa_v5_cli` is the
authoritative reference. This package wraps it as a Python callable
for the differential-test harness.

The CLI binary uses line-delimited JSON over stdin/stdout for cheap
batching: one JSON request per line in, one JSON response per line
out. The wrapper keeps a single long-lived subprocess to amortise
process-start cost across hypothesis-many random samples.
"""

from .drift import (
    LeanDriftClient,
    drift_lean,
    mu_bar_lean,
    find_a_sep_lean,
    schedule_lean,
    em_step_lean,
    hr_mean_lean,
    sleep_prob_lean,
    stress_mean_lean,
    steps_log_mean_lean,
    volume_load_mean_lean,
    PARAM_KEYS,
    OBS_PARAM_KEYS,
)

__all__ = [
    "LeanDriftClient",
    "drift_lean",
    "mu_bar_lean",
    "find_a_sep_lean",
    "schedule_lean",
    "em_step_lean",
    "hr_mean_lean",
    "sleep_prob_lean",
    "stress_mean_lean",
    "steps_log_mean_lean",
    "volume_load_mean_lean",
    "PARAM_KEYS",
    "OBS_PARAM_KEYS",
]
