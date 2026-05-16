# Paper Freeze Manifest

This file defines the paper-freeze artifact set for the current mainline. It is
not a new experiment result and should not be edited to change numerical claims.
Update the source CSVs or generators first, then regenerate the paper-facing
assets.

## Freeze Scope

The frozen mainline is:

- Problem: IRS-assisted MS-AirComp under temporal stale CSI and execution-channel mismatch.
- Main deployable method: `Mask-Corrected Coverage-Aware B=3 mc=1`.
- Reliable-feedback main result: direct mask correction at feedback-noise std `0`.
- High-noise robustness variant: `mc=1 clip=2` at feedback-noise std `0.1`.
- Main scenario grid: `channel_rho in {0.7, 0.9, 0.98}` and `csi_delay_slots in {1, 2, 3}`.
- Default system: `K=50`, `N=10`, `M=64`, `C=16`, `g_th=0.001`, `alpha_th=0.05`.

The source-of-truth paper documents are:

- `docs/PAPER_RESULT_PACKAGE.md`
- `docs/PAPER_STRUCTURE_MAP.md`
- `docs/PAPER_FIGURE_TABLE_SPECS.md`
- `docs/PAPER_APPENDIX_BOUNDARY.md`
- `docs/PAPER_TEXT_OUTLINE.md`
- `docs/PAPER_ASSET_GAP_CHECKLIST.md`

## Verification Commands

Run these before treating the freeze as valid:

```bash
make paper-table1
make paper-figures
make check
make mainline-audit
```

`make mainline-audit` is the artifact-chain gate. It does not rerun formal
experiments; it checks that the frozen CSV, Markdown and figure artifacts still
support the current claims.

## Frozen Generated Artifacts

These files are intentionally unignored in `.gitignore` because a clean clone
must be able to run `make mainline-audit` without relying on hidden local
result files.

Analysis-layer summary artifacts:

- `results/execution_mismatch/final_execution_baseline_summary.csv`
- `results/execution_mismatch/main_frontier_analysis.csv`
- `results/execution_mismatch/main_frontier_preview_gap.png`
- `results/execution_mismatch/main_frontier_failed_missed.png`
- `docs/EXECUTION_BASELINE_SUMMARY.md`
- `docs/MAIN_RESULTS_ANALYSIS.md`

Coverage-aware artifacts:

- `results/execution_mismatch/coverage_aware_ablation_analysis.csv`
- `results/execution_mismatch/coverage_aware_weight_ablation.png`
- `results/execution_mismatch/coverage_aware_power_ablation.png`
- `docs/COVERAGE_AWARE_ANALYSIS.md`
- `results/execution_mismatch/coverage_b3_failure_diagnosis_ep100_runs2_rho0p7-0p9-0p98_delay1-2-3_b3_sm4p1_tf0p75_cw0p5_cpw0_summary.csv`
- `results/execution_mismatch/coverage_b3_failure_diagnosis_ep100_runs2_rho0p7-0p9-0p98_delay1-2-3_b3_sm4p1_tf0p75_cw0p5_cpw0_trace.csv`
- `docs/COVERAGE_B3_FAILURE_DIAGNOSIS.md`

Invitation-mask correction artifacts:

- `results/execution_mismatch/invitation_mask_correction_formal_ep300_runs3_rho0p7-0p9-0p98_delay1-2-3_b3_sm4p1_tf0p75_cw0p5_cpw0_mc0-0p75-1.csv`
- `results/execution_mismatch/invitation_mask_correction_noise_ep300_runs3_rho0p7-0p9-0p98_delay1-2-3_b3_sm4p1_tf0p75_cw0p5_cpw0_mc0-0p75-1_fbn0-0p02-0p05-0p1.csv`
- `results/execution_mismatch/invitation_mask_correction_noise_aware_formal_ep300_runs3_rho0p7-0p9-0p98_delay1-2-3_b3_sm4p1_tf0p75_cw0p5_cpw0_mc0-0p75-1_clipinf-2_fbn0-0p02-0p05-0p1.csv`
- `results/execution_mismatch/final_invitation_mask_analysis.csv`
- `results/execution_mismatch/final_invitation_mask_gap_noise.png`
- `results/execution_mismatch/final_invitation_mask_failed_missed_noise.png`
- `docs/INVITATION_MASK_CORRECTION.md`
- `docs/INVITATION_MASK_CORRECTION_NOISE.md`
- `docs/INVITATION_MASK_CORRECTION_NOISE_AWARE.md`
- `docs/FINAL_INVITATION_MASK_ANALYSIS.md`

Paper-facing table and figure artifacts:

- `docs/PAPER_TABLE1_MAIN_RESULTS.md`
- `results/paper/table1_main_results.csv`
- `results/paper/figure2_figure3_points.csv`
- `results/paper/figure2_preview_gap_frontier.png`
- `results/paper/figure3_failed_missed_tradeoff.png`
- `results/paper/figure4_invitation_mask_noise_points.csv`
- `results/paper/figure4_invitation_mask_gap_noise.png`
- `results/paper/figure4_invitation_mask_failed_missed_noise.png`

## Non-Freeze Boundary

The rest of `results/` remains local by default. Historical SAC, imitation,
bandit, limited-CSI, learned-shortlist and other diagnostic artifacts are
indexed in `docs/RESULTS_INDEX.md` and `docs/DEPRECATED_DIRECTIONS.md`, but
they are not part of the paper-freeze commit unless explicitly needed for an
appendix or reviewer response.

Do not add new main-text methods to the freeze unless they improve
`Mask-Corrected Coverage-Aware B=3 mc=1` at the same or lower preview budget,
or directly explain its residual gap / feedback-noise boundary.
