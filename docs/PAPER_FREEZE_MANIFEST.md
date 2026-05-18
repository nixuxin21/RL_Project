# Paper Freeze Manifest

This file defines the paper-freeze artifact set for the current mainline. It is
not a new experiment result and should not be edited to change numerical claims.
Update the source CSVs or generators first, then regenerate the paper-facing
assets.

## Freeze Scope

The frozen mainline is:

- Problem: IRS-assisted MS-AirComp under temporal stale CSI and execution-channel mismatch.
- No-noise same-preview gap reference: `Coverage-Aware B=3 sm=4.1 cw=0.5 cpw=0`.
- Reliable-feedback trade-off method: `Mask-Corrected Coverage-Aware B=3 mc=1`.
- High-noise gap-best correction: direct `mc=1` at feedback-noise std `0.1`.
- High-noise failed-invitation diagnostic: `mc=1 clip=2` at feedback-noise std `0.1`.
- Main scenario grid: `channel_rho in {0.7, 0.9, 0.98}` and `csi_delay_slots in {1, 2, 3}`.
- Default system: `K=50`, `N=10`, `M=64`, `C=16`, `g_th=0.001`, `alpha_th=0.05`.

Model-version note: the temporal AR(1) evaluator now supports prehistory states
so early slots also use truly delayed CSI. The temporal formal sweeps in this
freeze were regenerated after that model fix; older pre-fix claims that direct
mask correction is the no-noise gap-best method should not be reused.

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
make paper-tables
make paper-figures
make check
make mainline-audit
```

`make mainline-audit` is the artifact-chain gate. It does not rerun formal
experiments; it checks that the frozen CSV, Markdown and figure artifacts still
support the current claims. It also verifies that paper-freeze artifacts and
`source_file` CSVs are tracked or staged in Git, so a clean clone can reproduce
the artifact audit.

## Frozen Generated Artifacts

These files are intentionally unignored in `.gitignore` because a clean clone
must be able to run `make mainline-audit` without relying on hidden local
result files.

Summary source CSVs referenced by `final_execution_baseline_summary.csv` or
`main_frontier_analysis.csv` are part of the freeze boundary too. They are
included so readers can trace paper-facing aggregate rows back to the exact
scenario-level result files used by the generators:

- `results/execution_mismatch/sparse_topk_frontier_ep300_runs3_rho0p7-0p9-0p98_delay1-2-3_b4-8_sm2-3_tf0p75.csv`
- `results/execution_mismatch/adaptive_sparse_topk_v2_pilot_ep100_runs2_rho0p7-0p9-0p98_delay1-2-3_b4_mt0p02-0p05_pc0-0p002-0p005-0p01.csv`
- `results/execution_mismatch/coverage_sparse_topk_frontier_ep300_runs3_rho0p7-0p9-0p98_delay1-2-3_b4_sm3_tf0p75_cw0p5_cpw0.csv`
- `results/execution_mismatch/coverage_budget_split_selected_ep300_runs3_rho0p7-0p9-0p98_delay1-2-3_b3_sm4p1_tf0p75_cw0p5_cpw0.csv`
- `results/execution_mismatch/invitation_mask_correction_formal_ep300_runs3_rho0p7-0p9-0p98_delay1-2-3_b3_sm4p1_tf0p75_cw0p5_cpw0_mc0-0p75-1.csv`
- `results/execution_mismatch/learned_sparse_shortlist_pilot_ep100_runs2_rho0p7-0p9-0p98_delay1-2-3_b4_ex1-2.csv`
- `results/execution_mismatch/learned_set_shortlist_pilot_ep100_runs2_rho0p7-0p9-0p98_delay1-2-3_b4_maxex1-2.csv`
- `results/execution_mismatch/learned_execution_value_shortlist_pilot_ep100_runs2_rho0p7-0p9-0p98_delay1-2-3_b4_maxex1-2_fw0p5_mw0p5.csv`
- `results/execution_mismatch/learned_pairwise_shortlist_pilot_ep100_runs2_rho0p7-0p9-0p98_delay1-2-3_b4_maxex1-2_fw0p5_mw0p5_pc0.csv`
- `results/execution_mismatch/learned_pairwise_shortlist_pilot_ep100_runs2_rho0p7-0p9-0p98_delay1-2-3_b4_maxex1-2_fw0p5_mw0p5_pc0p005.csv`

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
- `docs/PAPER_TABLE1_UNCERTAINTY.md`
- `results/paper/table1_main_results.csv`
- `results/paper/table1_scenario_uncertainty.csv`
- `results/paper/table1_paired_scenario_deltas.csv`
- `docs/PAPER_TABLE2_COVERAGE_AWARE_ABLATION.md`
- `results/paper/table2_coverage_aware_ablation.csv`
- `docs/PAPER_TABLE3_FAILURE_DIAGNOSIS.md`
- `results/paper/table3_failure_diagnosis.csv`
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
`Coverage-Aware B=3 sm=4.1 cw=0.5 cpw=0` on no-noise same-preview gap, improve
the `Mask-Corrected Coverage-Aware B=3 mc=1` trade-off at the same or lower
preview budget, or directly explain the feedback-noise boundary.
