"""Pytest collection config — exclude pre-existing legacy v2/v3 tests.

Four test files in this directory import model symbols
(`HIGH_RES_FSA_V2_ESTIMATION`, `HIGH_RES_FSA_V2_MODEL`,
`HIGH_RES_FSA_V3_MODEL`) that no longer exist in the v4/v5 source. They
were authored against earlier versions of the model and were not
updated when the v3 → v4 → v5 migrations removed those exports. The
files are kept for historical reference but excluded from collection so
the rest of the suite can run.

If the v2/v3 model symbols are ever resurrected (or these test files
are rewritten against the v5 API), drop the corresponding entries from
``collect_ignore`` below.
"""
collect_ignore = [
    "test_artifacts.py",          # imports HIGH_RES_FSA_V2_ESTIMATION
    "test_fsa_physics.py",        # imports HIGH_RES_FSA_V2_MODEL
    "test_fsa_v3_physics.py",     # imports HIGH_RES_FSA_V3_MODEL
    "test_v3_identifiability.py", # imports HIGH_RES_FSA_V3_MODEL
]
