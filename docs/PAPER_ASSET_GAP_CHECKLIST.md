# Paper Asset Gap Checklist

本文件冻结投稿前 paper-facing assets 的完成度和缺口。它不是论文正文，也不是图表规格文件；它回答一个更实际的问题：哪些表图已经可以直接进入论文，哪些还只是分析层证据，哪些需要后续导出、生成或美化。

正文结果包见 `docs/PAPER_RESULT_PACKAGE.md`，图表规格见 `docs/PAPER_FIGURE_TABLE_SPECS.md`，正文骨架见 `docs/PAPER_TEXT_OUTLINE.md`。

## Current Asset Status

| Asset | Current Status | Existing Source / Artifact | Gap Before Submission | Priority |
|---|---|---|---|---|
| Table 1: main results | Ready as paper-facing CSV and Markdown, with scenario-level uncertainty companion. | `docs/PAPER_TABLE1_MAIN_RESULTS.md`, `docs/PAPER_TABLE1_UNCERTAINTY.md`, `results/paper/table1_main_results.csv`, `results/paper/table1_scenario_uncertainty.csv`, `results/paper/table1_paired_scenario_deltas.csv` | Journal formatting only; do not overstate companion as seed-level significance. | Done |
| Figure 1: system and feedback flow | SVG/PDF export is generated from the canonical Mermaid source. | `docs/figures/figure1_system_flow.mmd`, `results/paper/figure1_system_flow.svg`, `results/paper/figure1_system_flow.pdf` | Final journal sizing / font check. | Done |
| Figure 2: preview-gap frontier | Ready as paper-facing PNG and points CSV. | `results/paper/figure2_preview_gap_frontier.png`, `results/paper/figure2_figure3_points.csv` | Final journal sizing / font check. | P1 before submission |
| Figure 3: failed/missed tradeoff | Ready as paper-facing PNG and points CSV. | `results/paper/figure3_failed_missed_tradeoff.png`, `results/paper/figure2_figure3_points.csv` | Final journal sizing / font check. | P1 before submission |
| Figure 4: noise boundary | Ready as paper-facing PNGs and points CSV. | `results/paper/figure4_invitation_mask_noise_points.csv`, `results/paper/figure4_invitation_mask_gap_noise.png`, `results/paper/figure4_invitation_mask_failed_missed_noise.png` | Decide whether journal wants one combined panel or two separate files. | P1 before submission |
| Table 2: coverage-aware ablation | Ready as compact paper-facing CSV and Markdown. | `docs/PAPER_TABLE2_COVERAGE_AWARE_ABLATION.md`, `results/paper/table2_coverage_aware_ablation.csv`; source evidence: `docs/COVERAGE_AWARE_ANALYSIS.md`, `results/execution_mismatch/coverage_aware_ablation_analysis.csv` | Journal formatting only; full sweep details stay in `docs/COVERAGE_AWARE_ANALYSIS.md`. | Done |
| Table 3: failure diagnosis | Ready as compact paper-facing CSV and Markdown. | `docs/PAPER_TABLE3_FAILURE_DIAGNOSIS.md`, `results/paper/table3_failure_diagnosis.csv`; source evidence: `docs/COVERAGE_B3_FAILURE_DIAGNOSIS.md` | Journal formatting only; decide whether this remains main text or moves to appendix. | Done |
| Appendix A/B/C assets | Boundary ready, but no appendix tables generated. | `docs/PAPER_APPENDIX_BOUNDARY.md` | Generate appendix tables only after manuscript length and reviewer needs are known. | P2 |

## Missing Paper-Facing Artifacts

These assets should be generated only when the manuscript assembly begins. Proposed future filenames are intentionally not formatted as checked paths until they exist.

| Missing Asset | Proposed Future Filename | Source | Acceptance Criteria |
|---|---|---|---|
| Optional appendix tables | results/paper/appendix_* and docs/PAPER_APPENDIX_*.md | `docs/PAPER_APPENDIX_BOUNDARY.md`, `docs/RESULTS_INDEX.md` | Created only for the minimum Appendix A/B/C or reviewer-requested supplement. |

## Do Not Generate By Default

Do not create paper-facing assets for these branches unless the appendix boundary changes:

- SAC / Codebook-Aware SAC main figures.
- Imitation or learned probing figures.
- Full learned-shortlist variant tables.
- Adaptive v1/v3 or neighbor-coverage standalone figures.
- Action diagnostics figures.
- Broad historical leaderboard tables.

Those results remain diagnostics under `docs/PAPER_APPENDIX_BOUNDARY.md` and `docs/DEPRECATED_DIRECTIONS.md`.

## Readiness Gates

Before the manuscript is assembled:

1. Run `make paper-tables`.
2. Run `make paper-figures`.
3. Decide whether Table 2 and Table 3 stay in main text or Table 3 moves to appendix.
4. Re-export Figure 1 with `make paper-figure1` only if `docs/figures/figure1_system_flow.mmd` changes.
5. Inspect Figure 1-4 and Table 1-3 at target journal column width.
6. Run `make docs`, `make mainline-audit`, and `make check`.

## Current Decision

The project is not blocked on more experiments. The remaining paper-asset gaps are presentation and packaging gaps:

- Figure 1 SVG/PDF export is generated; it still needs final journal sizing / font inspection.
- Table 2 and Table 3 now have compact paper-facing artifacts; the remaining decision is placement and final formatting.
- Figure 2-4 need only final visual sizing / font checks unless the target journal requires vector formats.
- Appendix assets should stay deferred until manuscript length and reviewer needs are known.
