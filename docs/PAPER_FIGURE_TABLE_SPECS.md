# Paper Figure And Table Specs

本文件集中管理论文主图和主表的展示规格。它不是论文正文，也不是绘图源文件；它的作用是固定图表的科学信息边界、列名、方法顺序、caption 方向和禁止误画的内容。

图表的章节位置见 `docs/PAPER_STRUCTURE_MAP.md`，正文结果集合见 `docs/PAPER_RESULT_PACKAGE.md`。

## Global Rules

- 正文图表必须服务于 stale CSI、execution-channel mismatch、limited aggregate feedback 和 invitation-mask correction 这条主线。
- 不把 early SAC、Codebook-Aware SAC、imitation、learned probing 或 learned shortlist 放进正文主图主表。
- 不把 hidden current-channel oracle 画成可部署方法输入；`Temporal Deviation Oracle` 只能作为 temporal diagnostic reference，不能作为 global upper bound。
- 不声称方法使用 per-device current CSI；deployable feedback 是 aggregate current feedback count，device-level ordering comes only from stale-gain reranking under the confirmed IRS。
- 图表中的方法命名必须与 `docs/PAPER_RESULT_PACKAGE.md` 的 frozen main-text methods 保持一致。
- 数字精度默认与当前结果文档一致：slots / failed / missed / gap 保留 3 位小数，preview 保留 2 位小数，perfect rate 保留 2 位小数。
- Table 1 is a compact mean-only table. Use `docs/PAPER_TABLE1_UNCERTAINTY.md` as the scenario-level uncertainty companion, and do not describe it as a full seed-level significance test.

## Information-Role Taxonomy

Paper-facing tables and figures must use the same information-role boundary:

| Role label | Meaning | Paper handling |
|---|---|---|
| `deployable` | The method does not use hidden current-channel device-level CSI at decision time. It may use stale CSI, limited codebook preview, and aggregate current feedback counts. | Main-text method, baseline, or reference is allowed. |
| `diagnostic` | The method is useful for mechanism analysis or negative results but is not proposed as a deployable main method; learned diagnostics may use hidden current-channel supervision during training. | Appendix or supplement unless explicitly needed for a reviewer question. |
| `hidden-information temporal diagnostic` | The method uses hidden current-channel outcomes during temporal offset evaluation or selection. | Non-deployable headroom/reference only; never describe as a global upper bound. |

For Table 1 and Figures 2/3, `Rotating B=8`, `Sparse-TopK B=4 sm=3`,
`Coverage-Aware B=4 cw=0.5 cpw=0`, `Coverage-Aware B=3 sm=4.1
cw=0.5 cpw=0`, `Mask-Corrected Coverage-Aware B=3 mc=1`, and
`Stale-TopK B=4` are deployable under the stated stale-CSI /
aggregate-feedback assumptions. `Temporal Deviation Oracle B=4` is a
hidden-information temporal diagnostic reference, not a global upper bound.
Learned shortlist, learned temporal, RL, and
bandit-learning variants remain diagnostics and should not be mixed into the
main deployable-vs-oracle comparison.

## Naming Crosswalk

正文主表使用完整 frozen method labels；图中可以使用短标签，但短标签必须能唯一映射回完整方法或明确的 robustness variant。

| Context | Full / Source Label | Figure Label | Meaning |
|---|---|---|---|
| Table 1, Figure 2, Figure 3 | `Coverage-Aware B=3 sm=4.1 cw=0.5 cpw=0` | `Cov B3` | uncorrected same-preview B3 candidate-generation method |
| Table 1, Figure 2, Figure 3 | `Mask-Corrected Coverage-Aware B=3 mc=1` | `MaskCorr` | same-preview slot/failed trade-off; no-noise gap regression |
| Table 1, Figure 2, Figure 3 | `Temporal Deviation Oracle B=4` | `Oracle` | hidden-information temporal diagnostic reference, not deployable or globally bounding |
| Figure 4 source CSV | `Coverage-Aware B=3` | `B3` | abbreviation of `Coverage-Aware B=3 sm=4.1 cw=0.5 cpw=0` |
| Figure 4 source CSV | `Direct Mask Correction mc=1` | `Direct` | no-noise trade-off and high-noise gap-best direct target-count correction |
| Figure 4 source CSV | `Clipped Mask Correction mc=1 clip=2` | `Clip2` | failed-invitation control diagnostic, not a Table 1 main method |

## Main Figure And Table Plan

| Order | Item | Purpose | Source |
|---|---|---|---|
| Table 1 | Main result table | 展示 frozen main-text methods 在 9 个 temporal AR(1) 场景上的主结果。 | `docs/PAPER_TABLE1_MAIN_RESULTS.md`, `results/paper/table1_main_results.csv`, `results/execution_mismatch/final_execution_baseline_summary.csv` |
| Table 1 companion | Scenario uncertainty and paired deltas | 展示 Table 1 指标的 9-scenario variability 和 same-preview paired deltas。 | `docs/PAPER_TABLE1_UNCERTAINTY.md`, `results/paper/table1_scenario_uncertainty.csv`, `results/paper/table1_paired_scenario_deltas.csv` |
| Figure 1 | System and feedback flow | 展示 stale CSI、limited probes、aggregate count、confirmation 和 mask correction 的 deployable information flow。 | `docs/figures/figure1_system_flow.mmd` |
| Figure 2 | Preview-gap frontier | 展示 preview budget 与 oracle gap 的 cost-quality frontier。 | `results/paper/figure2_preview_gap_frontier.png`, `results/paper/figure2_figure3_points.csv` |
| Figure 3 | Failed/missed tradeoff | 展示 failed invitations 与 missed opportunities 的机制权衡。 | `results/paper/figure3_failed_missed_tradeoff.png`, `results/paper/figure2_figure3_points.csv` |
| Table 2 | Coverage-aware ablation | 展示 `cw=0.5 cpw=0` 和 `B=3 sm=4.1` 的选择依据。 | `docs/PAPER_TABLE2_COVERAGE_AWARE_ABLATION.md`, `results/paper/table2_coverage_aware_ablation.csv`, `docs/COVERAGE_AWARE_ANALYSIS.md` |
| Table 3 | Failure diagnosis | 展示 pool / selection / confirmation / invitation residual gap decomposition。 | `docs/PAPER_TABLE3_FAILURE_DIAGNOSIS.md`, `results/paper/table3_failure_diagnosis.csv`, `docs/COVERAGE_B3_FAILURE_DIAGNOSIS.md` |
| Figure 4 | Noise boundary | 展示 reliable-feedback trade-off、high-noise direct correction 和 clipped failed-invitation diagnostic。 | `results/paper/figure4_invitation_mask_gap_noise.png`, `results/paper/figure4_invitation_mask_failed_missed_noise.png`, `results/paper/figure4_invitation_mask_noise_points.csv` |

## Table 1: Main Result Table

Generated artifacts:

- `docs/PAPER_TABLE1_MAIN_RESULTS.md`
- `results/paper/table1_main_results.csv`
- `docs/PAPER_TABLE1_UNCERTAINTY.md`
- `results/paper/table1_scenario_uncertainty.csv`
- `results/paper/table1_paired_scenario_deltas.csv`

Regenerate them with:

```bash
make paper-tables
```

### Method Order

Table 1 固定使用以下方法顺序：

1. `Rotating B=8`
2. `Sparse-TopK B=4 sm=3`
3. `Coverage-Aware B=4 cw=0.5 cpw=0`
4. `Coverage-Aware B=3 sm=4.1 cw=0.5 cpw=0`
5. `Mask-Corrected Coverage-Aware B=3 mc=1`
6. `Stale-TopK B=4`
7. `Temporal Deviation Oracle B=4`

`Rotating B=4`、Adaptive V2 continuum 和 learned variants 不进入 Table 1；如需要，放 appendix / diagnostics。

All Table 1 rows except `Temporal Deviation Oracle B=4` are deployable
comparisons or references. The oracle row must be visually or textually marked
as a hidden-information temporal diagnostic reference, not as a method available
to the proposed pipeline and not as a global upper bound.

### Columns

Recommended columns:

| Column | Meaning | Precision |
|---|---|---|
| `Method` | Frozen method label. | exact label |
| `Role` | Baseline / reference / proposed method role. | short phrase |
| `Slots` | Mean completion slots. | 3 decimals |
| `Failed` | Mean failed invited devices. | 3 decimals |
| `Missed` | Mean missed opportunities. | 3 decimals |
| `Preview` | Mean decision preview calls per slot. | 2 decimals |
| `Gap` | Mean oracle tx gap. | 3 decimals |

`Perfect %` may be included if the journal format permits a wider table. If included, use 2 decimals. If space is tight, omit `Perfect %` because it is near-saturated and less explanatory than failed / missed / gap.

### Uncertainty Companion

The compact Table 1 is intentionally narrow and reports means only.
`docs/PAPER_TABLE1_UNCERTAINTY.md` is the companion generated by
`make paper-tables`. It reports scenario-level standard deviations / CI and
paired deltas for `Mask-Corrected Coverage-Aware B=3 mc=1` against the
same-preview baselines. The paired deltas currently support a slots/failed
trade-off claim versus `Coverage-Aware B=3`: slots and failed improve in 9/9
scenarios, missed improves in 6/9 scenarios, and gap improves in only 3/9
scenarios. Because the stored source CSVs do not retain per-seed values for
every Table 1 metric, describe this as scenario-level uncertainty, not as a
full seed-level significance test.

### Caption Direction

Candidate caption:

> Table 1. Main execution-mismatch results averaged over 9 temporal AR(1) stale-CSI scenarios. All non-oracle rows are deployable under the stated stale-CSI and aggregate-feedback information model. The mask-corrected coverage-aware method keeps the same preview budget as the uncorrected B3 baseline and reduces slots, failed invitations, and missed opportunities, but it increases the no-noise oracle gap under the corrected temporal prehistory model. The temporal-deviation oracle is a hidden-information temporal diagnostic reference, not a deployable method or a global upper bound. This compact table reports means only; use the companion scenario-level uncertainty table for variability and paired-delta evidence.

## Figure 1: System And Feedback Flow

Figure 1 should explain the information flow, not report numerical results.

The versioned source file is `docs/figures/figure1_system_flow.mmd`. The paper-facing exports are `results/paper/figure1_system_flow.svg` and `results/paper/figure1_system_flow.pdf`. The Mermaid source remains the canonical editable source; regenerate the SVG/PDF with `make paper-figure1` only after changing that source.

### Main Message

> The proposed pipeline separates IRS-state search from device-invitation correction under stale CSI and execution-channel mismatch.

The figure must show:

- stale CSI is available for candidate generation and initial invitation estimation;
- current channel is hidden during decision and appears only through limited aggregate probes;
- aggregate feedback count feeds both IRS confirmation and invitation-mask correction;
- corrected invitation mask is applied after confirmed IRS selection;
- hidden current-channel oracle is evaluation-only.

### Layout

Use two panels:

| Panel | Purpose | Required Content |
|---|---|---|
| (a) System and mismatch | Show physical IRS-assisted MS-AirComp and stale/current channel separation. | `K devices`, `IRS codebook C`, `AP / parameter server`, `Stale CSI at t-d`, `Current channel at t`, `Temporal AR(1) drift`. |
| (b) Deployable feedback pipeline | Show candidate generation, aggregate probing, confirmation and mask correction. | `Stale preview`, `Coverage-aware candidates`, `B aggregate probes`, `Aggregate count`, `Confirmed IRS`, `Stale invitation mask`, `Mask correction`, `Corrected invitation mask`, `Execution metrics`. |

### Key Arrows

| From | To | Label |
|---|---|---|
| `Stale preview` | `Coverage-aware candidates` | stale seed pool |
| `Coverage-aware candidates` | `B aggregate probes` | candidate IRS states |
| `B aggregate probes` | `Aggregate count` | current aggregate feedback |
| `Aggregate count` | `Confirmed IRS` | select confirmed IRS |
| `Confirmed IRS` | `Stale invitation mask` | stale invitation estimate |
| `Aggregate count` | `Mask correction` | target count |
| `Stale invitation mask` | `Mask correction` | stale mask cardinality and stale-gain reranking |
| `Mask correction` | `Corrected invitation mask` | corrected invited set |
| `Corrected invitation mask` | `Execution metrics` | current-channel execution |

The `Aggregate count -> Mask correction` arrow is the key contribution and should be visually emphasized.

### Visual Encoding

- Deployable flow: solid dark arrows.
- Hidden current-channel information: dashed gray arrows.
- Stale-to-current mismatch: muted red or orange dashed arrow.
- Aggregate feedback and correction block: one restrained accent color.
- Evaluation metrics: gray output block.

Avoid decorative gradients, 3D effects, dense math and method strings such as `Coverage-Aware B=3 sm=4.1 cw=0.5 cpw=0` inside the figure.

### Caption Direction

Candidate caption:

> Figure 1. Deployable information flow under stale CSI and execution-channel mismatch. Stale CSI generates a small coverage-aware IRS candidate set; limited aggregate current probes confirm the IRS state and provide a target count for correcting the stale invitation mask; device-level correction uses stale-gain reranking under the confirmed IRS before current-channel execution. Hidden current-channel information is used only for evaluation and oracle diagnostics.

### Source Maintenance

- Keep `docs/figures/figure1_system_flow.mmd` as the canonical editable source.
- Keep `results/paper/figure1_system_flow.svg` and `results/paper/figure1_system_flow.pdf` as the current paper-facing exports.
- Do not make `make check` depend on Mermaid rendering tools.
- Treat SVG/PDF exports as paper assets, not as replacements for the source.

## Figure 2 And Figure 3

Figures 2 and 3 should be treated as paired mechanism plots:

- Figure 2: preview calls per slot vs oracle gap.
- Figure 3: failed invitations vs missed opportunities.

Generated artifacts:

- `results/paper/figure2_figure3_points.csv`
- `results/paper/figure2_preview_gap_frontier.png`
- `results/paper/figure3_failed_missed_tradeoff.png`

Regenerate them with:

```bash
make paper-figures
```

Rules:

- Use the same method labels and colors across both figures.
- Mark `Mask-Corrected Coverage-Aware B=3 mc=1` as a same-preview trade-off, not as the no-noise gap-best method.
- Mark `Temporal Deviation Oracle B=4` as hidden-information temporal diagnostic, not as a global upper bound.
- Mark all non-oracle Table 1 methods in these figures as deployable
  comparisons under stale CSI plus aggregate feedback, even when they are
  high-cost references such as `Stale-TopK B=4`.
- Do not use Random IRS or early RL methods in these plots.
- Do not imply lower preview alone is always better; the point is cost-quality tradeoff.

Caption directions:

- Figure 2 should say `Coverage-Aware B=3` remains the no-noise same-preview gap reference and that the oracle is a hidden-information diagnostic.
- Figure 3 should say mask correction reduces failed invitations and slightly reduces missed opportunities versus B3, while increasing the no-noise oracle gap.

## Table 2: Coverage-Aware Ablation

Table 2 should justify the coverage-aware candidate-generation setting:

- `cw=0.5` is retained for interpretability, not because weight sensitivity is dramatic.
- `cpw=0` is the main setting because power penalty did not improve the main tradeoff.
- `B=3 sm=4.1` is selected as the same-preview refinement at preview 16.

Caption should emphasize budget split and missed-opportunity reduction.
Paper-facing artifacts are `docs/PAPER_TABLE2_COVERAGE_AWARE_ABLATION.md` and `results/paper/table2_coverage_aware_ablation.csv`.

## Table 3: Failure Diagnosis

Table 3 should report the residual gap decomposition for `Coverage-Aware B=3 sm=4.1 cw=0.5 cpw=0`.

Required message:

> The residual gap is dominated by invitation-mask mismatch, not by ordinary candidate-pool miss alone.

If page budget is tight, Table 3 can move to appendix, but the main text must retain the conclusion and cite the diagnosis.
Paper-facing artifacts are `docs/PAPER_TABLE3_FAILURE_DIAGNOSIS.md` and `results/paper/table3_failure_diagnosis.csv`.

## Figure 4: Noise Boundary

Figure 4 should show the boundary of aggregate-feedback correction:

- reliable feedback: direct `mc=1` is a no-noise trade-off, not a gap improvement;
- high-noise feedback std `0.1`: direct `mc=1` is the gap-best correction;
- clipped `mc=1 clip=2` is a failed-invitation control diagnostic because it reduces high-noise failed invitations relative to direct correction but has higher gap.

Generated artifacts:

- `results/paper/figure4_invitation_mask_noise_points.csv`
- `results/paper/figure4_invitation_mask_gap_noise.png`
- `results/paper/figure4_invitation_mask_failed_missed_noise.png`

Regenerate them with:

```bash
make paper-figures
```

Rules:

- Keep `Coverage-Aware B=3`, direct `mc=1`, and clipped `mc=1 clip=2` in the same color/label order across Figure 4 panels.
- Mark direct `mc=1` at feedback-noise std `0` as the no-noise correction trade-off in the source CSV.
- Mark direct `mc=1` at feedback-noise std `0.1` as the high-noise gap-best point in the source CSV.
- Keep clipped `mc=1 clip=2` at feedback-noise std `0.1` visible as the failed-invitation control diagnostic.
- Do not collapse the reliable-feedback and high-noise conclusions into a single winner statement.

Caption direction:

> Figure 4. Noise boundary of aggregate-feedback invitation-mask correction. Direct target-count correction is a no-noise trade-off and becomes the high-noise gap-best correction at feedback-noise std 0.1; clipping the correction to at most two devices is a failed-invitation control diagnostic rather than the gap-best variant. All correction variants keep the same coverage-aware B3 candidate generation and preview budget.

## Checklist

Before finalizing any paper figure or table:

1. Method order and labels match `docs/PAPER_RESULT_PACKAGE.md`.
2. Hidden current-channel information is never shown as an input to the deployable method.
3. Aggregate feedback is shown as count-level feedback, not per-device current CSI.
4. Table 1 does not include appendix-only methods.
5. Figure 1 makes `Aggregate count -> Mask correction` visually obvious.
6. Captions distinguish deployable methods from oracle diagnostics.
