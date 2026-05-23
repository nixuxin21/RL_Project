# Table Plan for cost_frontier_main_v1

This table plan is designed to support the narrow frontier claim without overstating unsupported results.

## Main Paper Tables

### Table 1: Main Deployable Performance and Cost

Source: `results/main_analysis/cost_frontier_main_v1/main_table.csv`

Rows:
- rotating_same_budget
- sparse_topk_same_budget
- coverage_aware
- count_only_mask_correction
- posterior_greedy_invitation_feedback
- posterior_invitation_feedback
- posterior_greedy_feedback

Columns:
- completion
- slots used
- oracle gap
- NMSE proxy
- total protocol cost
- current probe calls
- stale preview calls

Purpose:
- Establish that PGI has the best deployable oracle gap among the main methods while using much lower total protocol cost than count-only and coverage-aware.

Caption requirements:
- State that NMSE is `synthetic_unit_variance` proxy.
- State that lower oracle gap, NMSE proxy, and cost are better.
- Mark posterior_invitation and posterior_greedy as ablations.

### Table 2: Same-B Paired Deltas

Source: `results/main_analysis/cost_frontier_main_v1/same_B_paired_deltas.csv`

Rows:
- PGI - count_only_mask_correction
- PGI - coverage_aware
- PGI - sparse_topk_same_budget
- PGI - rotating_same_budget

Columns:
- delta completion
- delta oracle gap
- delta NMSE proxy
- delta total protocol cost
- oracle-gap win rate
- NMSE-proxy win rate

Purpose:
- Show that the same nominal probe-budget comparison is fair and that PGI improves oracle gap and cost, while same-B NMSE proxy vs count-only is essentially tied.

### Table 3: Closest-Cost Frontier Comparison

Source: `results/main_analysis/cost_frontier_main_v1/closest_cost_summary.csv`

Rows:
- PGI vs closest-cost count-only
- PGI vs closest-cost coverage-aware
- PGI vs closest-cost sparse-topK
- PGI vs closest-cost rotating

Columns:
- mean absolute cost distance
- nearest-neighbor delta oracle gap
- nearest-neighbor delta NMSE proxy
- oracle-gap win rate
- NMSE-proxy win rate
- interpolation-available rate
- interpolated delta oracle gap
- interpolated delta NMSE proxy

Purpose:
- This is the central evidence for the cost-quality frontier claim.

### Table 4: Clustered Bootstrap Confidence Intervals

Source: `results/main_analysis/cost_frontier_main_v1/cluster_bootstrap_ci.csv`

Rows:
- PGI - count_only
- PGI - coverage
- PGI - sparse_topK
- PGI - rotating

Columns:
- metric
- mean delta
- 95% CI low
- 95% CI high
- number of rows
- number of clusters
- cluster unit

Purpose:
- Provide statistical support without pretending scenario rows are fully independent.

## Appendix Tables

### Appendix Table A1: Probe-Budget Frontier by Method

Source: `results/main_analysis/cost_frontier_main_v1/cost_frontier.csv`

Columns:
- method
- B
- completion
- oracle gap
- NMSE proxy
- total protocol cost
- oracle-gap frontier flag
- NMSE frontier flag

Purpose:
- Show that PGI is on the frontier for B in {4, 6, 8, 12, 16}.

### Appendix Table A2: Subgroup Deltas

Source: `results/main_analysis/cost_frontier_main_v1/subgroup_table.csv`

Subgroups:
- C=64 vs C=128
- K=100 vs K=200
- slots=4 vs slots=6
- B=4/6/8/12/16
- rho=0.5/0.7/0.9
- delay=1 vs delay=3

Purpose:
- Support larger-codebook and larger-population trends.
- Show that "strongest under staler CSI" is not supported.

### Appendix Table A3: Saturation and Non-Saturated Robustness

Sources:
- `results/main_analysis/cost_frontier_main_v1/saturation_report.csv`
- `results/main_analysis/cost_frontier_main_v1/non_saturated_paired_deltas.csv`

Purpose:
- Report that 26/240 scenario-budget groups are saturated or weak evidence.
- Show that non-saturated-only oracle-gap and cost trends persist.

### Appendix Table A4: Component Ablation

Source: `results/main_analysis/cost_frontier_main_v1/component_ablation.csv`

Rows:
- posterior_invitation_feedback - count_only_mask_correction
- posterior_greedy_feedback - coverage_aware
- PGI - posterior_invitation_feedback
- PGI - posterior_greedy_feedback
- PGI - count_only_mask_correction
- PGI - coverage_aware

Purpose:
- Prevent overclaiming.
- Show that the combined policy is the meaningful contribution.

### Appendix Table A5: Oracle Diagnostic Boundary

Source: `results/main_analysis/cost_frontier_main_v1/oracle_diagnostics.csv`

Columns:
- method
- method role
- hidden current info used
- rows
- mean oracle gap
- min oracle gap
- negative oracle-gap rows
- achieved-exceeds-oracle rows

Purpose:
- Demonstrate no deployable method uses hidden current device-level CSI and that oracle diagnostics are clean.

## Internal Backup Only

### Backup Table B1: Full Raw Main Means With Standard Deviations

Source: `results/main_analysis/cost_frontier_main_v1/main_table.csv`

Reason:
- Useful for reviewer response, but too dense for the main paper.

### Backup Table B2: Full Closest-Cost Pair Rows

Source: `results/main_analysis/cost_frontier_main_v1/closest_cost_comparisons.csv`

Reason:
- Very useful for auditability, but too large for manuscript.

### Backup Table B3: Stage 1 Validation

Source: `results/main_analysis/cost_frontier_main_v1/stage1/`

Reason:
- Confirms that the staged execution protocol did not blindly run the full grid, but Stage 1 is not the final evidence table.

### Backup Table B4: Run Summary and Git Hygiene

Source: `results/main_analysis/cost_frontier_main_v1/run_summary.json`

Reason:
- Useful for reproducibility appendix or artifact review, not necessary in the main narrative.
