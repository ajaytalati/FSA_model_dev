# CLAUDE.md

Project-level guidance for Claude Code working in this repo.

## Behavioural rules

### Never assert unverified claims

Do not state facts you have not checked. This includes:

- Which script generated a particular file, plot, or artefact.
- Which function is called from where.
- Which version, branch, or configuration was used to produce something.
- What a piece of code does, beyond what you have read.

Pattern-matching on filenames, directory layout, or naming conventions is **a guess, not evidence**. If you have not verified, either:

1. Verify first (grep, read the file, run the check), or
2. State the uncertainty explicitly: "I think X based on Y, but I have not confirmed."

This rule matters most when the claim becomes the *premise* of further reasoning. A wrong premise propagates into a wrong analysis that the user has to refute. Specifically:

- Do not infer the generating script of a figure from the output filename.
- Do not infer "the optimiser used N iterations" from a sibling script with that value.
- Do not infer the active code path from "this is the obvious one".

When uncertain, ask, or check.
