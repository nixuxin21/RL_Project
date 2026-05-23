# cost_frontier_main_v1 Candidate Evidence Package

This package preserves the focused `cost_frontier_main_v1` paper-evidence draft.
It is a candidate evidence package, not part of the current paper-freeze boundary.

## Role

The package supports the narrow candidate claim that
`posterior_greedy_invitation_feedback` provides a low-protocol-cost
cost-quality frontier under limited IRS probing. It should not replace the
current frozen Coverage-Aware / invitation-mask paper line until the associated
source artifacts are promoted and audited.

## Local Source Artifacts

The referenced raw and analysis outputs are local generated artifacts:

- `results/main/cost_frontier_main_v1/`
- `results/main_analysis/cost_frontier_main_v1/`

These directories remain ignored by default. A clean clone should not depend on
them unless a later promotion explicitly tracks or regenerates the required
source artifacts.

## Documents

- `experiment_protocol.md`: run grid, staged execution and statistical analysis
  protocol.
- `cost_frontier_main_v1_summary.md`: evidence summary and safe interpretation.
- `main_claim_matrix.md`: supported, weak, unsupported and contradicted claims.
- `table_plan.md`: main/appendix table plan.
- `figure_plan.md`: main/appendix figure plan.
- `limitations.md`: limitations and unsafe wording to avoid.

## Promotion Checklist

Before this package becomes publication-facing freeze material:

1. Decide whether it supersedes or only supplements the current frozen mainline.
2. Track or regenerate the required `results/main*` source artifacts.
3. Update `.gitignore`, `results/README.md`, `docs/PAPER_FREEZE_MANIFEST.md`,
   `docs/PAPER_RESULT_PACKAGE.md` and `docs/PROJECT_MAP.md`.
4. Add explicit `tests/mainline_artifact_checks.py` checks for the new CSV,
   figure and text artifact chain.
5. Run `make check`, `make mainline-audit` and the relevant paper-suite dry run.
