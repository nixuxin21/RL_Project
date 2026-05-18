# Paper Structure Map

本文件不是论文正文，也不是摘要草稿。它把 `docs/PAPER_RESULT_PACKAGE.md` 中已经冻结的结果映射到未来论文的章节、表图和附录位置，避免写作时重新把历史实验混入主线。主图主表规格见 `docs/PAPER_FIGURE_TABLE_SPECS.md`，附录最小集合见 `docs/PAPER_APPENDIX_BOUNDARY.md`，逐节写作骨架见 `docs/PAPER_TEXT_OUTLINE.md`，投稿前资产缺口见 `docs/PAPER_ASSET_GAP_CHECKLIST.md`。

## Paper Spine

建议论文主线保持为一句话：

> Under stale CSI and execution-channel mismatch, IRS-assisted MS-AirComp has two coupled decisions: which IRS states to probe and which devices to invite. Low-cost coverage-aware candidate generation addresses the IRS search side, while aggregate-feedback invitation-mask correction addresses the stale invitation side.

这条主线意味着论文不是 RL 论文，也不是单纯 baseline comparison。正文应围绕 limited preview、current aggregate feedback、execution mismatch 和 invitation-mask correction 展开。

## Section Map

| Section | Job | Required Content | Primary Evidence | Keep Out |
|---|---|---|---|---|
| Introduction | 说明 stale CSI / execution mismatch 下 full-current CSI oracle 不可用，低成本 feedback 是核心约束。 | 问题背景、两类错配、贡献列表、主结果一句话。 | `docs/PAPER_RESULT_PACKAGE.md`, `docs/MAIN_RESULTS_ANALYSIS.md` | 早期 SAC、imitation、learned probing 细节 |
| Related Work | 放置 AirComp、IRS-aided communication、limited CSI / probing、feedback-based scheduling 的文献位置。 | 只需要概念分组和本文差异点，具体引用后续再补。 | `docs/BASELINE_STRATEGY.md`, `docs/DEPRECATED_DIRECTIONS.md` | 结果表和内部实验历史 |
| System Model | 定义 IRS-assisted MS-AirComp、multi-slot execution、DFT codebook、stale CSI 和 aggregate feedback。 | `K=50`, `N=10`, `M=64`, `C=16` 默认设定；`g_th=0.001`, `alpha_th=0.05`；temporal AR(1) stale CSI。 | `test_env.py`, `ms_aircomp/channel_models.py`, `docs/PAPER_RESULT_PACKAGE.md` | 具体算法调参过程 |
| Problem Formulation | 明确 deployable method 不能访问 hidden current channel，只能做 limited IRS preview 和 aggregate feedback confirmation。 | preview budget、failed nodes、missed opportunities、oracle gap、perfect rate 等指标。 | `docs/EXECUTION_BASELINE_SUMMARY.md`, `docs/MAIN_RESULTS_ANALYSIS.md` | SAC reward 设计历史 |
| Method | 分成 candidate generation、current aggregate confirmation、invitation-mask correction 三段。 | `Sparse-TopK`, `Coverage-Aware`, `Mask-Corrected Coverage-Aware`, high-noise clipped variant。 | `ms_aircomp/probe_sets.py`, `ms_aircomp/confirmation.py`, `evaluate_invitation_mask_correction.py` | learned shortlist 训练细节 |
| Experiments | 用冻结主表、frontier 图、failure diagnosis、mask correction 和 noise robustness 组织证据。 | 9 个 `rho/delay` 场景、same-preview comparisons、hidden oracle reference。 | `docs/PAPER_RESULT_PACKAGE.md`, `docs/RESULTS_INDEX.md` | 早期 policy comparison 主表 |
| Discussion | 解释为何不是继续普通 heuristic 或 RL；说明 high-noise robustness 和 hidden oracle 的边界。 | negative/diagnostic branches、limitations、future work。 | `docs/DEPRECATED_DIRECTIONS.md`, `EXPERIMENT_REPORT.md` | 新增未验证 claim |

## Main Contributions

正文贡献建议控制为三条，不要扩展成历史实验清单：

1. A stale-CSI execution-mismatch evaluation framework for IRS-assisted multi-slot AirComp with limited preview and aggregate current feedback.
2. A low-cost coverage-aware sparse candidate-generation strategy that improves same-preview performance by reducing missed opportunities.
3. An aggregate-feedback invitation-mask correction mechanism that changes the slots/failed/missed/gap trade-off without changing IRS candidate generation.

可选第四条只在篇幅允许时加入：

4. A noise-boundary analysis showing that direct correction is gap-best under high aggregate-feedback noise while clipped target-count correction controls failed invitations.

## Main Table And Figure Plan

正文表图建议按机制顺序，而不是按实验时间顺序：

| Order | Item | Content | Source |
|---|---|---|---|
| Table 1 | Main result table | Frozen main-text methods across 9 temporal AR(1) scenarios, plus scenario-level uncertainty companion. | `docs/PAPER_TABLE1_MAIN_RESULTS.md`, `results/paper/table1_main_results.csv`, `docs/PAPER_TABLE1_UNCERTAINTY.md`, `results/paper/table1_scenario_uncertainty.csv`, `results/paper/table1_paired_scenario_deltas.csv`, `docs/PAPER_RESULT_PACKAGE.md` |
| Figure 1 | System and feedback flow diagram | Stale CSI, limited IRS preview, aggregate confirmation, invitation-mask correction. | `docs/PAPER_FIGURE_TABLE_SPECS.md`, `docs/figures/figure1_system_flow.mmd` |
| Figure 2 | Preview-gap frontier | Preview cost vs oracle gap for main methods. | `results/paper/figure2_preview_gap_frontier.png`, `results/paper/figure2_figure3_points.csv` |
| Figure 3 | Failed/missed tradeoff | Failed invitations and missed opportunities across main methods. | `results/paper/figure3_failed_missed_tradeoff.png`, `results/paper/figure2_figure3_points.csv` |
| Table 2 | Coverage-aware and budget split ablation | Why `cw=0.5 cpw=0` and why `B=3 sm=4.1`. | `docs/PAPER_TABLE2_COVERAGE_AWARE_ABLATION.md`, `results/paper/table2_coverage_aware_ablation.csv`, `docs/COVERAGE_AWARE_ANALYSIS.md` |
| Table 3 | Failure diagnosis | Pool/selection/confirmation/invitation residual gap decomposition. | `docs/PAPER_TABLE3_FAILURE_DIAGNOSIS.md`, `results/paper/table3_failure_diagnosis.csv`, `docs/COVERAGE_B3_FAILURE_DIAGNOSIS.md` |
| Figure 4 | Invitation-mask noise boundary | Reliable-feedback trade-off, high-noise direct correction, and clipped failed-invitation diagnostic. | `results/paper/figure4_invitation_mask_gap_noise.png`, `results/paper/figure4_invitation_mask_failed_missed_noise.png`, `results/paper/figure4_invitation_mask_noise_points.csv` |

如果期刊篇幅有限，Table 3 可以移到 appendix，但正文必须至少保留一句 diagnosis conclusion：residual gap is dominated by invitation-mask mismatch.

## Method Section Breakdown

方法部分建议使用下面的子结构：

### Baseline Feedback Policies

介绍 `Rotating B=8`、`Sparse-TopK B=4 sm=3`、`Stale-TopK B=4` 和 `Temporal Deviation Oracle B=4`。其中 `Temporal Deviation Oracle` 必须明确标注为 hidden-information temporal diagnostic reference。

### Coverage-Aware Candidate Generation

解释 stale seed pool、coverage diversity weight 和 current aggregate confirmation。重点不是说 `cw=0.5` 有巨大敏感性，而是说明 power penalty 被移除、coverage term 提供解释性，并且 budget split 把 same-preview setting 推到 `B=3 sm=4.1`。

### Invitation-Mask Correction

这是 trade-off mechanism 的核心。应说明它保持 confirmed IRS 不变，用 aggregate current feedback count 设定 target cardinality，并用 confirmed IRS 下的 stale-gain reranking 生成 corrected invitation mask，因此不访问 per-device current CSI。可靠反馈下它降低 slots、failed 和 missed，但在修正后的 temporal prehistory 模型下不降低 no-noise gap。

### Noise-Aware Boundary

作为 boundary subsection，而不是主方法替代。可靠反馈下 direct `mc=1` 是 no-noise trade-off；feedback-noise std `0.1` 时 direct `mc=1` 是 gap-best correction，`mc=1 clip=2` 是 failed-invitation control diagnostic。

Figure 4 中的 `Direct` 是 `Mask-Corrected Coverage-Aware B=3 mc=1` 的 noise-sweep 展示标签；`Clip2` 只表示 failed-invitation control diagnostic，不应被写成替代 Table 1 trade-off method 的新主方法。

## Experiment Section Breakdown

实验部分建议分成四组：

1. Main frontier: show frozen main methods and hidden-current headroom.
2. Candidate generation ablation: show coverage-aware weight / power penalty / budget split.
3. Residual gap diagnosis: show why invitation mismatch is the right next target.
4. Mask correction and noise boundary: show the no-noise trade-off, high-noise direct gap improvement, and clipped failed-invitation diagnostic.

不要把 early policy comparison、SAC、imitation 或 learned probing 放在实验正文主线中；它们可以在 appendix 中支撑“why not RL / learned selectors yet”。

## Appendix Map

最小附录边界以 `docs/PAPER_APPENDIX_BOUNDARY.md` 为准。正文写作时默认只保留 Appendix A/B/C 三组证据；supplement-only diagnostics 不主动进入论文，除非 reviewer 要求。

| Appendix Topic | Purpose | Source |
|---|---|---|
| Historical policy comparison | Background: Feature Argmax / Greedy / SAC context. | `results/policy_comparison/policy_comparison_summary_ep1000_runs5_seed2026_featargmax_powertie_cbsac.csv` |
| Runtime and probing cost | Explain preview-cost motivation. | `results/runtime/runtime_benchmark_ep200_seed2026.csv`, `results/probing_cost/probing_cost_tradeoff_slot0-0p05-0p1-0p2_preview0-0p0005-0p001-0p002-0p005-0p01-0p02-0p05_winners.csv` |
| Channel estimation and limited CSI | Background robustness and metric motivation. | `results/channel_estimation/channel_estimation_error_sweep_ep1000_runs5_seed2026_err0-0p02-0p05-0p1-0p2-0p3.csv`, `results/limited_csi/limited_csi_main_ep1000_runs5_seed2026.csv` |
| Bandit feedback diagnostics | Show simple bandit feedback is not enough. | `results/bandit_feedback/bandit_feedback_stress_formal_ep1000_runs5_seed2026.csv` |
| Learned shortlist diagnostics | Explain why learned variants are not main methods. | `docs/DEPRECATED_DIRECTIONS.md`, `docs/RESULTS_INDEX.md` |
| Full negative-direction index | Prevent repeating stopped directions. | `docs/DEPRECATED_DIRECTIONS.md`, `EXPERIMENT_REPORT.md` |

## Writing Guardrails

写作时保持以下边界：

- 不要把 `Rotating B=4` 当成主要 baseline；正文 baseline 是 `Rotating B=8`。
- 不要把 Adaptive V2 写成 proposed method；它是 continuum / appendix point。
- 不要把 learned shortlist 的负结果写得过重；一句 diagnosis + appendix 即可。
- 不要把 `Temporal Deviation Oracle B=4` 写成可部署方法。
- 不要声称 invitation-mask correction 使用 per-device current CSI；它使用 aggregate current feedback count。
- 不要把 high-noise clipped variant 写成 gap-best；它是 failed-invitation control diagnostic。

## Pre-Writing Checklist

真正开始写论文正文前，建议先完成下面几项：

1. 检查 Table 1、Figure 2、Figure 3 和 Figure 4 的 method naming 是否与 `docs/PAPER_FIGURE_TABLE_SPECS.md` 的 naming crosswalk 完全一致。
2. 以 `docs/figures/figure1_system_flow.mmd` 为 Figure 1 的 canonical editable source；如果导出 SVG/PDF，导出件必须仍保持 stale CSI、limited IRS probes、aggregate feedback count 和 mask correction 的信息边界。
3. 使用 `make paper-tables` 和 `make paper-figures` 生成正文 Table 1/2/3、Table 1 uncertainty companion、Figure 2、Figure 3 和 Figure 4，不手工改数值或绘图点表。
4. 按 `docs/PAPER_ASSET_GAP_CHECKLIST.md` 确认 Figure 1 export、Table 2/3 placement 和图表版式状态。
5. 从 `docs/RESULTS_INDEX.md` 抽出 appendix tables 的最小集合，避免附录过度膨胀。
6. 跑 `make mainline-audit` 和 `make check`，确保 artifact chain 和数值趋势仍成立。
