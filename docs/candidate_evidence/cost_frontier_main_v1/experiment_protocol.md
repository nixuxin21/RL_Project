# Experiment Protocol: cost_frontier_main_v1

This document records the protocol used for the focused main cost-frontier validation. It is written for reproducibility and paper-methods translation.

## Purpose

The experiment tests the revised claim:

> Posterior-greedy probing with count-conditioned invitation feedback provides a low-protocol-cost cost-quality frontier under limited IRS probing.

The protocol intentionally avoids claims that posterior invitation alone or posterior-greedy probing alone dominates existing baselines.

## Reproducibility State

- Checkpoint commit recorded in logs: `9b9e826b29af8bf421266fac5220c60d8bf61210`
- Git dirty value recorded in logs: `False`
- Raw output root: `results/main/cost_frontier_main_v1/cost_frontier_main_v1_full_f6ca5c9dc1`
- Analysis output root: `results/main_analysis/cost_frontier_main_v1`
- Analysis rows: 35,520
- Scenario logs: 8
- Seeds: 20
- Episodes per seed: 1

Episodes per seed equal to 1 is a limitation. It keeps the focused validation tractable but should be disclosed.

## Staged Execution

### Stage 0: Dry Run

Dry-run job count:
- Stage 1 dry run: 8 jobs
- Full dry run: 8 jobs

Purpose:
- Confirm that the job count matched the intended grid before executing experiments.
- Avoid accidentally running the older broad `main_easy`, `main_medium`, or `main_hard` grids.

### Stage 1: Reduced Validation

Grid:
- K: 100, 200
- slots: 4, 6
- codebook size C: 64, 128
- probe budget B: 4, 8, 16
- rho: 0.7, 0.9
- CSI delay: 1, 3
- feedback noise: none
- seeds: 10
- episodes per seed: 1

Outcome:
- Stage 1 confirmed the hard-cost trend.
- PGI vs closest-cost count-only: delta oracle gap = -4.4031, oracle-gap win rate = 0.8333.

### Stage 2: Full Focused Grid

Grid:
- K: 100, 200
- slots: 4, 6
- codebook size C: 64, 128
- IRS elements: 64
- probe budget B: 4, 6, 8, 12, 16
- rho: 0.5, 0.7, 0.9
- CSI delay: 1, 3
- feedback noise: none
- seeds: 20
- episodes per seed: 1

The optional `random_same_budget` method was not included in this run because runtime was already substantial and the method was optional in the request.

## Methods

Core deployable methods:
- `rotating_same_budget`
- `sparse_topk_same_budget`
- `coverage_aware`
- `count_only_mask_correction`
- `posterior_greedy_invitation_feedback`

Ablation methods:
- `posterior_invitation_feedback`
- `posterior_greedy_feedback`

Diagnostics:
- `full_stale_exhaustive`
- `full_current_oracle`

Diagnostic methods must not be mixed with deployable methods in claims of deployable performance.

## Metrics

Primary quality metrics:
- completion
- slots used
- oracle gap
- synthetic-unit-variance NMSE proxy
- failed invitations
- missed opportunities

Cost and overhead metrics:
- current probe calls
- stale preview calls
- total probe calls
- total protocol cost

The protocol cost model used equal unit costs in this run:

`total_protocol_cost = stale_preview_calls + current_probe_calls + execution_slots`

This cost model is simple and transparent but should be treated as a configurable accounting assumption, not as a universal physical-layer latency model.

## Statistical Analysis

The analysis includes:
- main method means
- method-budget frontier table
- same-B paired deltas
- closest-cost nearest-neighbor comparisons
- closest-cost local interpolation where available
- clustered bootstrap confidence intervals
- subgroup analysis
- saturation analysis
- non-saturated-only analysis
- component ablation
- oracle diagnostic checks

Clustered bootstrap unit:

`scenario_config_probe_budget_run_seed`

This avoids presenting every row as fully independent. However, because episodes per seed equal 1, the statistical conclusion is still limited by the number of independent seeds and scenario configurations.

## Analysis Files

Primary:
- `results/main_analysis/cost_frontier_main_v1/main_table.csv`
- `results/main_analysis/cost_frontier_main_v1/cost_frontier.csv`
- `results/main_analysis/cost_frontier_main_v1/closest_cost_summary.csv`
- `results/main_analysis/cost_frontier_main_v1/same_B_paired_deltas.csv`
- `results/main_analysis/cost_frontier_main_v1/cluster_bootstrap_ci.csv`

Diagnostics:
- `results/main_analysis/cost_frontier_main_v1/subgroup_table.csv`
- `results/main_analysis/cost_frontier_main_v1/saturation_report.csv`
- `results/main_analysis/cost_frontier_main_v1/non_saturated_analysis.csv`
- `results/main_analysis/cost_frontier_main_v1/non_saturated_paired_deltas.csv`
- `results/main_analysis/cost_frontier_main_v1/component_ablation.csv`
- `results/main_analysis/cost_frontier_main_v1/oracle_diagnostics.csv`

Figures:
- `results/main_analysis/cost_frontier_main_v1/figures/cost_vs_oracle_gap_frontier.png`
- `results/main_analysis/cost_frontier_main_v1/figures/cost_vs_oracle_gap_frontier.pdf`
- `results/main_analysis/cost_frontier_main_v1/figures/cost_vs_nmse_frontier.png`
- `results/main_analysis/cost_frontier_main_v1/figures/cost_vs_nmse_frontier.pdf`

## Paper-Ready Interpretation Rules

Use:
- "cost-quality frontier"
- "same-B oracle-gap improvement"
- "closest-cost improvement"
- "synthetic-unit-variance NMSE proxy"
- "non-deployable oracle diagnostic"

Avoid:
- "posterior invitation alone dominates"
- "posterior-greedy probing alone dominates"
- "true AirComp MSE improvement"
- "performance gains grow monotonically with CSI staleness"

## Reproduction Note

This evidence package was created from existing outputs only. No experiments were rerun and no source code was changed for this package.
