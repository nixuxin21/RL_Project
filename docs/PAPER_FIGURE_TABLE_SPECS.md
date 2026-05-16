# Paper Figure And Table Specs

本文件集中管理论文主图和主表的展示规格。它不是论文正文，也不是绘图源文件；它的作用是固定图表的科学信息边界、列名、方法顺序、caption 方向和禁止误画的内容。

图表的章节位置见 `docs/PAPER_STRUCTURE_MAP.md`，正文结果集合见 `docs/PAPER_RESULT_PACKAGE.md`。

## Global Rules

- 正文图表必须服务于 stale CSI、execution-channel mismatch、limited aggregate feedback 和 invitation-mask correction 这条主线。
- 不把 early SAC、Codebook-Aware SAC、imitation、learned probing 或 learned shortlist 放进正文主图主表。
- 不把 hidden current-channel oracle 画成可部署方法输入；它只能作为 evaluation / diagnostic upper bound。
- 不声称方法使用 per-device current CSI；deployable feedback 是 aggregate current feedback count。
- 图表中的方法命名必须与 `docs/PAPER_RESULT_PACKAGE.md` 的 frozen main-text methods 保持一致。
- 数字精度默认与当前结果文档一致：slots / failed / missed / gap 保留 3 位小数，preview 保留 2 位小数，perfect rate 保留 2 位小数。

## Naming Crosswalk

正文主表使用完整 frozen method labels；图中可以使用短标签，但短标签必须能唯一映射回完整方法或明确的 robustness variant。

| Context | Full / Source Label | Figure Label | Meaning |
|---|---|---|---|
| Table 1, Figure 2, Figure 3 | `Coverage-Aware B=3 sm=4.1 cw=0.5 cpw=0` | `Cov B3` | uncorrected same-preview B3 candidate-generation method |
| Table 1, Figure 2, Figure 3 | `Mask-Corrected Coverage-Aware B=3 mc=1` | `MaskCorr` | proposed reliable-feedback main method |
| Table 1, Figure 2, Figure 3 | `Temporal Deviation Oracle B=4` | `Oracle` | hidden-current diagnostic upper bound, not deployable |
| Figure 4 source CSV | `Coverage-Aware B=3` | `B3` | abbreviation of `Coverage-Aware B=3 sm=4.1 cw=0.5 cpw=0` |
| Figure 4 source CSV | `Direct Mask Correction mc=1` | `Direct` | same method as `Mask-Corrected Coverage-Aware B=3 mc=1` under direct target-count correction |
| Figure 4 source CSV | `Clipped Mask Correction mc=1 clip=2` | `Clip2` | high-noise robustness variant, not a Table 1 main method |

## Main Figure And Table Plan

| Order | Item | Purpose | Source |
|---|---|---|---|
| Table 1 | Main result table | 展示 frozen main-text methods 在 9 个 temporal AR(1) 场景上的主结果。 | `docs/PAPER_TABLE1_MAIN_RESULTS.md`, `results/paper/table1_main_results.csv`, `results/execution_mismatch/final_execution_baseline_summary.csv` |
| Figure 1 | System and feedback flow | 展示 stale CSI、limited probes、aggregate count、confirmation 和 mask correction 的 deployable information flow。 | `docs/figures/figure1_system_flow.mmd` |
| Figure 2 | Preview-gap frontier | 展示 preview budget 与 oracle gap 的 cost-quality frontier。 | `results/paper/figure2_preview_gap_frontier.png`, `results/paper/figure2_figure3_points.csv` |
| Figure 3 | Failed/missed tradeoff | 展示 failed invitations 与 missed opportunities 的机制权衡。 | `results/paper/figure3_failed_missed_tradeoff.png`, `results/paper/figure2_figure3_points.csv` |
| Table 2 | Coverage-aware ablation | 展示 `cw=0.5 cpw=0` 和 `B=3 sm=4.1` 的选择依据。 | `docs/COVERAGE_AWARE_ANALYSIS.md`, `results/execution_mismatch/coverage_aware_ablation_analysis.csv` |
| Table 3 | Failure diagnosis | 展示 pool / selection / confirmation / invitation residual gap decomposition。 | `docs/COVERAGE_B3_FAILURE_DIAGNOSIS.md` |
| Figure 4 | Noise robustness | 展示 reliable-feedback direct correction 和 high-noise clipped variant。 | `results/paper/figure4_invitation_mask_gap_noise.png`, `results/paper/figure4_invitation_mask_failed_missed_noise.png`, `results/paper/figure4_invitation_mask_noise_points.csv` |

## Table 1: Main Result Table

Generated artifacts:

- `docs/PAPER_TABLE1_MAIN_RESULTS.md`
- `results/paper/table1_main_results.csv`

Regenerate them with:

```bash
make paper-table1
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

### Caption Direction

Candidate caption:

> Table 1. Main execution-mismatch results averaged over 9 temporal AR(1) stale-CSI scenarios. The proposed mask-corrected coverage-aware method keeps the same preview budget as the uncorrected coverage-aware baseline but reduces both failed invitations and missed opportunities. The temporal-deviation oracle is a hidden-information diagnostic upper bound, not a deployable method.

## Figure 1: System And Feedback Flow

Figure 1 should explain the information flow, not report numerical results.

The versioned source file is `docs/figures/figure1_system_flow.mmd`. This source is intentionally checked in before any polished SVG/PDF export so the scientific information flow can be reviewed in text.

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
| `Stale invitation mask` | `Mask correction` | stale mask cardinality |
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

> Figure 1. Deployable information flow under stale CSI and execution-channel mismatch. Stale CSI generates a small coverage-aware IRS candidate set; limited aggregate current probes confirm the IRS state and provide a target count for correcting the stale invitation mask before current-channel execution. Hidden current-channel information is used only for evaluation and oracle diagnostics.

### Source Maintenance

- Keep `docs/figures/figure1_system_flow.mmd` as the canonical editable source.
- Do not make `make check` depend on Mermaid rendering tools.
- If an SVG/PDF export is later added, it should be generated from this source and treated as a paper asset, not as a replacement for the source.

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
- Mark `Mask-Corrected Coverage-Aware B=3 mc=1` as the proposed method.
- Mark `Temporal Deviation Oracle B=4` as hidden-information oracle / diagnostic.
- Do not use Random IRS or early RL methods in these plots.
- Do not imply lower preview alone is always better; the point is cost-quality tradeoff.

Caption directions:

- Figure 2 should say the proposed mask-corrected B3 method is the best same-preview point and that the oracle is a hidden-information diagnostic.
- Figure 3 should say mask correction improves both failed invitations and missed opportunities, while candidate-generation variants mainly trade one against the other.

## Table 2: Coverage-Aware Ablation

Table 2 should justify the coverage-aware candidate-generation setting:

- `cw=0.5` is retained for interpretability, not because weight sensitivity is dramatic.
- `cpw=0` is the main setting because power penalty did not improve the main tradeoff.
- `B=3 sm=4.1` is selected as the same-preview refinement at preview 16.

Caption should emphasize budget split and missed-opportunity reduction.

## Table 3: Failure Diagnosis

Table 3 should report the residual gap decomposition for `Coverage-Aware B=3 sm=4.1 cw=0.5 cpw=0`.

Required message:

> The residual gap is dominated by invitation-mask mismatch, not by ordinary candidate-pool miss alone.

If page budget is tight, Table 3 can move to appendix, but the main text must retain the conclusion and cite the diagnosis.

## Figure 4: Noise Robustness

Figure 4 should show reliable-feedback direct correction and high-noise clipped correction:

- reliable feedback: direct `mc=1` is the main result;
- high-noise feedback std `0.1`: `mc=1 clip=2` is the conservative robustness variant;
- do not present clipped correction as replacing direct correction in low/no-noise settings.

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
- Mark direct `mc=1` at feedback-noise std `0` as the reliable-feedback main point in the source CSV.
- Mark clipped `mc=1 clip=2` at feedback-noise std `0.1` as the high-noise robustness point in the source CSV.
- Do not collapse the reliable-feedback and high-noise conclusions into a single winner statement.

Caption direction:

> Figure 4. Noise robustness of aggregate-feedback invitation-mask correction. Direct target-count correction is the reliable-feedback main method; clipping the correction to at most two devices is reported only as a conservative high-noise variant. All correction variants keep the same coverage-aware B3 candidate generation and preview budget.

## Checklist

Before finalizing any paper figure or table:

1. Method order and labels match `docs/PAPER_RESULT_PACKAGE.md`.
2. Hidden current-channel information is never shown as an input to the deployable method.
3. Aggregate feedback is shown as count-level feedback, not per-device current CSI.
4. Table 1 does not include appendix-only methods.
5. Figure 1 makes `Aggregate count -> Mask correction` visually obvious.
6. Captions distinguish deployable methods from oracle diagnostics.
