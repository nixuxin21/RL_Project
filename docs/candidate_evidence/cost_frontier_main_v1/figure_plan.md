# Figure Plan for cost_frontier_main_v1

This figure plan emphasizes frontier behavior and avoids unsupported component-dominance claims.

## Main Paper Figures

### Figure 1: Protocol Cost vs Oracle Gap Frontier

Source:
- `results/main_analysis/cost_frontier_main_v1/cost_frontier.csv`
- Existing figure: `results/main_analysis/cost_frontier_main_v1/figures/cost_vs_oracle_gap_frontier.png`
- Existing PDF: `results/main_analysis/cost_frontier_main_v1/figures/cost_vs_oracle_gap_frontier.pdf`

Design:
- x-axis: total protocol cost
- y-axis: oracle gap, lower is better
- points grouped by method and probe budget B
- label PGI points with B in {4, 6, 8, 12, 16}
- use a visually distinct line or marker for PGI frontier points
- include count-only and coverage-aware for comparison
- optionally show diagnostic oracles with hollow markers, not connected to deployable frontier

Message:
- PGI remains on the oracle-gap frontier at all tested B values.
- PGI is lower cost than count-only at comparable or better oracle gap.

Caption note:
- "Oracle diagnostics are non-deployable and shown only as references."

### Figure 2: Protocol Cost vs NMSE Proxy Frontier

Source:
- `results/main_analysis/cost_frontier_main_v1/cost_frontier.csv`
- Existing figure: `results/main_analysis/cost_frontier_main_v1/figures/cost_vs_nmse_frontier.png`
- Existing PDF: `results/main_analysis/cost_frontier_main_v1/figures/cost_vs_nmse_frontier.pdf`

Design:
- x-axis: total protocol cost
- y-axis: synthetic-unit-variance NMSE proxy, lower is better
- label PGI B values
- explicitly label NMSE as proxy in axis title or caption

Message:
- PGI is on the NMSE-proxy frontier across tested B values.
- Count-only at B=16 can slightly lower NMSE proxy than PGI at B=16, but only at substantially higher protocol cost.

Caption note:
- "This is a synthetic-unit-variance NMSE proxy, not waveform/source-signal AirComp MSE."

## Appendix Figures

### Figure A1: Subgroup Oracle-Gap Deltas

Source:
- `results/main_analysis/cost_frontier_main_v1/subgroup_table.csv`

Design:
- bar plot or dot-whisker plot
- panels:
  - PGI - count_only
  - PGI - coverage_aware
- x-axis: subgroup value
- y-axis: delta oracle gap, lower is better
- include subgroups C, K, slots, B, rho, delay

Message:
- Gains are stable and larger for C=128 and K=200.
- The stale-CSI monotonicity claim is not supported; rho=0.9 shows larger gain than rho=0.5 or rho=0.7 against count-only.

### Figure A2: Component Ablation Plot

Source:
- `results/main_analysis/cost_frontier_main_v1/component_ablation.csv`

Design:
- paired bar chart with delta oracle gap and delta total protocol cost
- comparisons:
  - posterior_invitation - count_only
  - posterior_greedy - coverage
  - PGI - posterior_invitation
  - PGI - posterior_greedy
  - PGI - count_only
  - PGI - coverage

Message:
- Standalone components do not dominate.
- The combined method is the meaningful protocol contribution.

### Figure A3: Saturation and Non-Saturated Robustness

Sources:
- `results/main_analysis/cost_frontier_main_v1/saturation_report.csv`
- `results/main_analysis/cost_frontier_main_v1/non_saturated_paired_deltas.csv`

Design:
- left panel: fraction of saturated scenario-budget groups
- right panel: non-saturated paired delta oracle gap and total cost

Message:
- Only 26/240 scenario-budget groups are flagged as saturated.
- Non-saturated-only evidence preserves the oracle-gap and cost advantages.

### Figure A4: Closest-Cost Win Rates

Source:
- `results/main_analysis/cost_frontier_main_v1/closest_cost_summary.csv`

Design:
- grouped bar chart
- y-axis: win rate
- bars: oracle-gap win rate and NMSE-proxy win rate
- x-axis: reference baseline

Message:
- PGI has high closest-cost win rates, especially vs count-only and coverage-aware.

## Internal Backup Figures

### Backup Figure B1: Stage 1 vs Full Consistency

Sources:
- `results/main_analysis/cost_frontier_main_v1/stage1/`
- `results/main_analysis/cost_frontier_main_v1/`

Purpose:
- Useful if reviewers ask whether the full run was launched after a cherry-picked preview.
- Not necessary in the main manuscript.

### Backup Figure B2: Full Method-Budget Scatter With All Diagnostics

Source:
- `results/main_analysis/cost_frontier_main_v1/cost_frontier.csv`

Purpose:
- Audit view containing all deployable and diagnostic methods.
- Keep as internal backup because too many markers can weaken the main narrative.

## Required Visual Labeling Rules

- Label `full_current_oracle` as non-deployable.
- Label `full_stale_exhaustive` as diagnostic.
- Label NMSE as `synthetic_unit_variance NMSE proxy`.
- Use "lower is better" in captions for oracle gap, NMSE proxy, and total protocol cost.
- Do not title any figure as "dominance" unless the plotted metric actually supports it.
