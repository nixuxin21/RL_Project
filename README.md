# IRS-Assisted MS-AirComp

本项目研究 IRS 辅助的 multi-slot AirComp 调度。早期工作验证了 IRS、规则式 IRS selection、SAC/Codebook-Aware SAC、partial probing、limited CSI 和 bandit feedback 等方向；当前主线已经收敛为：

> IRS-assisted multi-slot AirComp under stale/limited CSI and execution-channel mismatch, using low-cost IRS candidate generation plus current aggregate feedback.

也就是说，当前重点不是继续堆叠新的 RL baseline，而是在 stale/limited CSI 和执行阶段信道失配下，研究如何用较低 IRS preview / feedback 成本接近 hidden current-channel oracle。

## Quick Start

```bash
make test
make docs
make execution-baseline-summary
make main-results-analysis
make paper-table1
make paper-figures
```

常用入口：

- `make help`: 查看当前可复现实验入口。
- `make test`: 编译主要脚本并运行轻量 smoke checks。
- `make policy-comparison`: 复现基础 No IRS / Random IRS / Feature Argmax / PowerTie / Greedy 对比。
- `make sparse-topk-frontier`: 复现当前 Sparse-TopK frontier。
- `make coverage-sparse-topk-pilot`: 运行 coverage-aware Sparse-TopK pilot，检验 device coverage diversity 是否能降低 missed opportunities。
- `make coverage-sparse-topk-ablation`: 复现 coverage-aware weight ablation，用于检查 `cw` sensitivity。
- `make coverage-sparse-power-ablation`: 复现 coverage-aware power penalty ablation，用于确认主线 `cpw=0`。
- `make coverage-sparse-topk-frontier`: 复现 `Coverage-Aware B=4 cw=0.5 cpw=0` formal frontier。
- `make coverage-budget-split-selected`: 复现当前主线 `Coverage-Aware B=3 sm=4.1 cw=0.5 cpw=0` formal setting。
- `make coverage-budget-split-formal`: 复现 near-preview-16 budget split formal table。
- `make coverage-aware-analysis`: 生成 Coverage-Aware 消融分析文档、CSV 和图。
- `make coverage-b3-failure-diagnosis`: 诊断当前 B3 主线 residual oracle gap 的来源。
- `make invitation-mask-correction-formal`: 复现 invitation mask correction formal result。
- `make invitation-mask-correction-noise-sweep`: 复现 invitation mask correction 的 aggregate-feedback-noise robustness sweep。
- `make invitation-mask-correction-noise-aware-formal`: 复现 clipped target-count correction 的 high-noise robustness result。
- `make final-invitation-mask-analysis`: 生成 invitation-mask correction 的最终论文表、gap-noise 图和 failed/missed-noise 图。
- `make paper-table1`: 从 frozen mainline CSV 生成论文 Table 1 的 CSV 和 Markdown。
- `make paper-figures`: 从 frozen mainline CSV 生成论文版 Figure 2/3/4 PNG 和绘图点 CSV。
- `make adaptive-sparse-topk-v2-pilot`: 复现 Adaptive Sparse-TopK v2 continuum。
- `make execution-baseline-summary`: 生成 execution 主线汇总表。
- `make main-results-analysis`: 生成当前主结果机制分析、CSV 和图。

## Main Documents

| 文件 | 作用 |
|---|---|
| `docs/MAIN_STORY.md` | 当前主线、active research stack 和后续投入边界 |
| `docs/PAPER_RESULT_PACKAGE.md` | 论文撰写前冻结的正文结果包、主表/图和 appendix 边界 |
| `docs/PAPER_STRUCTURE_MAP.md` | 将冻结结果包映射到论文章节、主表主图和附录位置 |
| `docs/PAPER_FIGURE_TABLE_SPECS.md` | 论文主图主表的统一规格，包括 Figure 1、Table 1 和后续图表规则 |
| `docs/PAPER_APPENDIX_BOUNDARY.md` | 论文附录最小集合、supplement-only diagnostics 和禁入正文规则 |
| `docs/PAPER_TEXT_OUTLINE.md` | 论文正文最小骨架，固定每节 claim、证据和禁入内容 |
| `docs/PAPER_ASSET_GAP_CHECKLIST.md` | 投稿前表图资产缺口清单，固定 Figure 1、Table 2/3 和图表美化状态 |
| `docs/PAPER_FREEZE_MANIFEST.md` | 论文冻结结果包的 artifact 清单、验证命令和非冻结边界 |
| `docs/figures/figure1_system_flow.mmd` | Figure 1 系统与 feedback pipeline 的 Mermaid 可编辑源文件 |
| `docs/PAPER_TABLE1_MAIN_RESULTS.md` | 由 `make paper-table1` 生成的论文 Table 1 Markdown 主表 |
| `docs/MAIN_RESULTS_ANALYSIS.md` | 当前主线结果的机制分析和论文级解释 |
| `docs/COVERAGE_AWARE_ANALYSIS.md` | Coverage-Aware Sparse-TopK 的 weight/power ablation、budget split 和主设定选择 |
| `docs/COVERAGE_B3_FAILURE_DIAGNOSIS.md` | 当前 B3 主线 residual gap 的 pool/selection/confirmation/invitation 分解 |
| `docs/INVITATION_MASK_CORRECTION.md` | Mask-Corrected Coverage-Aware 的 formal result |
| `docs/INVITATION_MASK_CORRECTION_NOISE.md` | Mask-Corrected Coverage-Aware 的 feedback-noise robustness boundary |
| `docs/INVITATION_MASK_CORRECTION_NOISE_AWARE.md` | clipped target-count correction 的 high-noise robustness result |
| `docs/FINAL_INVITATION_MASK_ANALYSIS.md` | invitation-mask correction 的最终论文级结果包 |
| `docs/EXECUTION_BASELINE_SUMMARY.md` | execution mismatch 主线 baseline 与 learned diagnostics 汇总 |
| `docs/BASELINE_STRATEGY.md` | baseline 分层和主表/补充表边界 |
| `docs/PROJECT_MAP.md` | 仓库结构和脚本分层 |
| `docs/RESULTS_INDEX.md` | 关键结果文件索引 |
| `docs/DEPRECATED_DIRECTIONS.md` | 不再继续投入的负结果和诊断方向 |
| `EXPERIMENT_REPORT.md` | 完整历史实验记录和详细结论 |

README 只作为项目入口。论文前主线冻结以 `docs/PAPER_RESULT_PACKAGE.md` 为准，章节组织以 `docs/PAPER_STRUCTURE_MAP.md` 为准，图表规格以 `docs/PAPER_FIGURE_TABLE_SPECS.md` 为准；详细实验叙事以 `EXPERIMENT_REPORT.md` 和 `docs/RESULTS_INDEX.md` 为准。

## Current Main Result

当前主线结果由 `make main-results-analysis` 从已有 CSV 生成。关键输出：

```text
docs/MAIN_RESULTS_ANALYSIS.md
results/execution_mismatch/main_frontier_analysis.csv
results/execution_mismatch/main_frontier_preview_gap.png
results/execution_mismatch/main_frontier_failed_missed.png
```

写论文或整理最终表图时，优先使用 `docs/PAPER_RESULT_PACKAGE.md` 中冻结的正文方法集合，并按 `docs/PAPER_STRUCTURE_MAP.md` 组织章节和表图；Adaptive V2、Rotating B=4 和 learned variants 默认作为 continuum / appendix / diagnostics。

跨 9 个 temporal AR(1) `rho/delay` 场景的 equal-weight 平均结果：

| Method | Role | Slots | Perfect % | Failed | Missed | Preview | Gap |
|---|---|---:|---:|---:|---:|---:|---:|
| `Rotating B=4` | low-budget reference | 3.887 | 99.90 | 1.585 | 1.098 | 4.00 | 1.116 |
| `Rotating B=8` | low-cost deployment baseline | 3.429 | 100.00 | 1.360 | 0.620 | 8.00 | 0.726 |
| `Adaptive V2 pc=0.005` | adaptive continuum point | 3.511 | 99.44 | 0.497 | 1.524 | 14.58 | 0.636 |
| `Adaptive V2 pc=0.002` | adaptive high-quality point | 3.492 | 99.44 | 0.505 | 1.492 | 15.36 | 0.597 |
| `Sparse-TopK B=4 sm=3` | reportable medium-cost baseline | 3.376 | 99.79 | 0.477 | 1.295 | 16.00 | 0.534 |
| `Coverage-Aware B=4 cw=0.5 cpw=0` | same-cost B=4 coverage reference | 3.289 | 99.84 | 0.473 | 1.131 | 16.00 | 0.497 |
| `Coverage-Aware B=3 sm=4.1 cw=0.5 cpw=0` | current budget-split refinement | 3.189 | 99.93 | 0.546 | 0.864 | 16.00 | 0.432 |
| `Mask-Corrected Coverage-Aware B=3 mc=1` | current best same-preview method | 2.684 | 99.95 | 0.333 | 0.333 | 16.00 | 0.292 |
| `Stale-TopK B=4` | high-cost positive reference | 3.373 | 99.78 | 0.444 | 1.321 | 20.00 | 0.465 |
| `Temporal Deviation Oracle B=4` | hidden-info upper bound | 2.985 | 100.00 | 0.326 | 0.900 | 4.00 | 0.345 |

当前解释：

- `Rotating B=8` 是最强低成本部署 baseline。
- `Sparse-TopK B=4 sm=3` 是当前最适合报告的 medium-cost positive result。
- `Coverage-Aware B=4 cw=0.5 cpw=0` 是第一个 formal positive candidate-generation refinement；power penalty 消融支持移除 stale power penalty，weight 消融显示 `cw=0/0.25/0.5` 在主指标上基本打平，因此保留 `cw=0.5` 作为解释性主设定。
- `Coverage-Aware B=3 sm=4.1 cw=0.5 cpw=0` 是当前主线预算分配版本：同样 preview `16`，把预算从 `4` 个 current feedback probes + `12` 个 stale seeds 调整到 `3` 个 feedback probes + `13` 个 stale seeds，进一步降低 gap 和 missed opportunities，但 failed invitations 高于 B=4 版本。
- `Mask-Corrected Coverage-Aware B=3 mc=1` 是当前同 preview `16` 下最强结果：它不改 IRS candidate generation，只用 aggregate current feedback count 修正 confirmed IRS 的 stale invitation mask，把 gap 降到 `0.292`，failed/missed 同时降到 `0.333/0.333`。
- `make invitation-mask-correction-noise-aware-formal` 显示 clipped target-count correction 是高噪声下的保守变体：feedback-noise std `0.1` 时，`mc=1 clip=2` 把 direct `mc=1` 的 gap/failed/missed 从 `0.856/5.264/0.734` 降到 `0.818/3.275/0.599`；低/中噪声下仍保留 unclipped `mc=1` 作为最佳 gap 设置。
- `make final-invitation-mask-analysis` 已把最终结论整理为论文表和初始图；论文正文 Figure 4 资产由 `make paper-figures` 生成到 `results/paper/figure4_invitation_mask_gap_noise.png`、`results/paper/figure4_invitation_mask_failed_missed_noise.png` 和 `results/paper/figure4_invitation_mask_noise_points.csv`。
- `Coverage-Aware B=5/6/8` 的 near-preview-16 formal 对照没有超过 B=3：它们减少 failed invitations，但 missed opportunities 和 gap 明显退化。
- `Adaptive Sparse-TopK v2` 更适合作为 cost-quality continuum，而不是最终主方法。
- `Stale-TopK B=4` 是高成本正向 reference。
- `Temporal Deviation Oracle B=4` 证明同等 B=4 probe budget 下仍有 candidate-set headroom。
- 正向方法主要减少 failed invitations；Coverage-Aware 已部分缓解 missed opportunities，但距离 hidden-current oracle 仍有空间。

## Active Research Stack

后续默认围绕下面这条线推进：

```text
Rotating B=8
-> Adaptive Sparse-TopK v2
-> Sparse-TopK B=4 sm=3
-> Coverage-Aware Sparse-TopK B=4 cw=0.5 cpw=0
-> Coverage-Aware Sparse-TopK B=3 sm=4.1
-> Mask-Corrected Coverage-Aware B=3
-> Stale-TopK B=4
-> Temporal Deviation Oracle B=4
```

新增方法必须至少和这些点对比，尤其不能只和 Random IRS 或 Rotating B=4 对比。

## Deprecated Or Diagnostic Directions

以下方向保留结果和脚本，但不再作为新增研究的主线投入点：

- Full SAC / Codebook-Aware SAC 作为主方法。
- Fixed IRS 作为主 baseline。
- Greedy-index imitation / noisy supervised imitation。
- Learned low-dimensional probing selector。
- Basic bandit feedback learning、UCB、Thompson、feedback-conditioned MLP。
- Static / adaptive risk-aware invitation filters。
- Pure active diversity candidate set。
- Temporal offset regressor、DAgger、window/gated temporal deviation。
- Adaptive Sparse-TopK v3 的固定 history/local-neighbor heuristic。
- Neighbor-Coverage 的固定 local-neighbor reallocation。
- 当前线性 learned set-value、execution-value、pairwise/cost-aware shortlist。

这些内容的定位见 `docs/DEPRECATED_DIRECTIONS.md`。它们不是无效记录，而是 negative results / diagnostics，用于支撑论文讨论和避免重复探索。

## Repository Layout

| 路径 | 作用 |
|---|---|
| `test_env.py` | Gymnasium 环境 `MSAirCompEnv`，包含信道、IRS DFT codebook、动作解码、slot 执行和 preview API |
| `evaluate_policy_comparison.py` | 基础 No IRS / Random IRS / Feature Argmax / PowerTie / Greedy 对比 |
| `evaluate_execution_channel_mismatch.py` | 当前主线评估框架，覆盖 stale CSI、execution mismatch、Sparse-TopK、Coverage-Aware Sparse-TopK、Adaptive V2、Stale-TopK 和 oracle |
| `summarize_execution_baselines.py` | 生成 `docs/EXECUTION_BASELINE_SUMMARY.md` 和最终 baseline CSV |
| `analyze_main_frontier.py` | 生成 `docs/MAIN_RESULTS_ANALYSIS.md`、主线分析 CSV 和两张结果图 |
| `analyze_coverage_aware.py` | 生成 `docs/COVERAGE_AWARE_ANALYSIS.md`、coverage-aware weight/power 消融 CSV 和图 |
| `train_learned_sparse_shortlist.py` | learned sparse shortlist diagnostics |
| `ms_aircomp/experiment_utils.py` | 公共实验 helper |
| `tests/smoke_checks.py` | 轻量行为回归测试 |
| `experiments/archive/` | 早期探索脚本归档 |
| `results/` | 本地实验生成物，默认不版本化，关键结果由索引文档记录 |
| `rl_models/`, `rl_logs/` | 本地模型 checkpoint 和 TensorBoard 日志 |

完整结构说明见 `docs/PROJECT_MAP.md`。

## Environment

默认物理设定：

- 节点数 `K=50`
- 时隙数 `N=10`
- IRS 单元数 `M=64`
- IRS DFT codebook 大小 `C=16`
- 噪声方差 `noise_var=1e-9`
- 最大发射功率 `P_max=1.0 W`
- 默认固定门限 `g_th=0.001`, `alpha_th=0.05`

轻量依赖见 `requirements.txt`，完整锁定版本见 `requirements-lock.txt`。当前本地 `.venv` 关键版本：

- Python 3.13.2
- Stable-Baselines3 2.8.0
- PyTorch 2.11.0
- Gymnasium 1.2.3
- NumPy 2.4.4

注意：历史 SAC checkpoint 的元数据记录了不同训练环境版本。复现实验时应记录 Python / Gymnasium / NumPy 版本差异。

## Core Commands

基础验证：

```bash
make test
make smoke
make docs
make help
```

当前主线：

```bash
make sparse-topk-frontier
make coverage-sparse-topk-pilot
make coverage-sparse-topk-ablation
make coverage-sparse-power-ablation
make coverage-sparse-topk-frontier
make coverage-budget-split-selected
make coverage-budget-split-formal
make adaptive-sparse-topk-v2-pilot
make execution-baseline-summary
make main-results-analysis
make coverage-aware-analysis
```

基础 baseline：

```bash
make policy-comparison
make policy-comparison-static
make policy-comparison-learning
make runtime
make parameter-sweep
```

支撑实验：

```bash
make partial-probing-sweep
make limited-csi-sweep
make execution-mismatch-sweep
make active-probe-set-pilot
make adaptive-sparse-topk-pilot
```

诊断/归档实验：

```bash
make bandit-feedback-stress
make adaptive-sparse-topk-v3-pilot
make learned-sparse-shortlist-pilot
make learned-sparse-shortlist-marginal-pilot
make learned-set-shortlist-pilot
make learned-execution-value-shortlist-pilot
make learned-pairwise-shortlist-pilot
```

## Important Outputs

| 输出 | 说明 |
|---|---|
| `docs/EXECUTION_BASELINE_SUMMARY.md` | 当前主线 baseline 与 learned diagnostics 总表 |
| `results/execution_mismatch/final_execution_baseline_summary.csv` | 上述总表对应 CSV |
| `docs/MAIN_RESULTS_ANALYSIS.md` | 主线机制分析 |
| `results/execution_mismatch/main_frontier_analysis.csv` | 按 `rho/delay` 展开的主线结果 |
| `results/execution_mismatch/main_frontier_preview_gap.png` | preview cost vs oracle gap 图 |
| `results/execution_mismatch/main_frontier_failed_missed.png` | failed invitations vs missed opportunities 图 |

## Baseline Policy

基础 IRS 对比仍应保留：

- `No IRS`: 无 IRS 下界。
- `Random IRS`: 最基本 IRS baseline，比 Fixed IRS 更适合作为“未优化 IRS”对照。
- `Greedy IRS`: full-preview 高成本 reference。
- `Feature Argmax IRS` / `Feature Argmax PowerTie IRS`: 早期完整 CSI 下的强规则 baseline。

在当前 execution-mismatch 主线中，主表应优先报告：

- `Rotating B=8`
- `Adaptive Sparse-TopK v2`
- `Sparse-TopK B=4 sm=3`
- `Coverage-Aware B=4 cw=0.5 cpw=0`
- `Coverage-Aware B=3 sm=4.1 cw=0.5 cpw=0`
- `Stale-TopK B=4`
- `Temporal Deviation Oracle B=4`

`Fixed IRS`、SAC、imitation、bandit learning 和 learned shortlist variants 放入 supplement / diagnostics。

## Result Boundaries

当前结论的边界：

- 主线 execution mismatch 使用 temporal AR(1) stale CSI，当前汇总覆盖 `rho in {0.7, 0.9, 0.98}` 和 `delay in {1, 2, 3}`。
- `Temporal Deviation Oracle` 使用 hidden current channel，只能作为上界诊断，不能作为可部署策略。
- `Stale-TopK B=4` 是高成本 reference，因为它使用完整 stale ranking 加 current aggregate feedback。
- `Sparse-TopK` 和 `Adaptive V2` 的价值在于降低 stale ranking 成本，同时保留部分 current feedback 确认收益。
- 当前 learned shortlist 结果仍弱于 `Adaptive V2` 和 `Sparse-TopK sm=3`，因此只作为 diagnostic。
- `results/` 默认是本地生成物，不直接版本化；需要正式引用的结果通过 `docs/RESULTS_INDEX.md` 记录。

## Next Research Step

当前最值得继续做的是：

> 以 `Mask-Corrected Coverage-Aware B=3 mc=1` 作为可靠反馈下的主方法，以 `mc=1 clip=2` 作为 high-noise 保守变体；下一步应把这条 invitation-mask correction 线整理成论文主贡献和补充鲁棒性表。

`coverage_sparse_topk_feedback` 已接入主分析。它在 sparse stale pool 中保留 top stale anchors，再用 marginal device coverage gain 填充剩余 current-feedback probes；formal power ablation 选择 `cpw=0`，weight ablation 显示 `cw=0/0.25/0.5` 在主指标上基本打平，因此保留 `cw=0.5 cpw=0`。budget split formal 结果显示，在同样 preview `16` 下，`B=3 sm=4.1` 比 `B=4 sm=3` 进一步降低 gap 和 missed opportunities，代价是 failed invitations 增加；逐场景检查也显示 B=3 在全部 9 个 `rho/delay` 场景上都是最低 gap/slots 的 split，因此没有简单的 scenario-adaptive B 选择空间。

`make coverage-b3-failure-diagnosis` 显示，当前 B3 residual gap 的主要来源不是 sparse stale pool，也不是 final B=3 subset，而是 invitation mask mismatch：总体 gap share 为 invitation `0.530`、selection `0.251`、confirmation `0.116`、pool `0.104`。因此下一步算法应优先做 confirmed IRS 下的 stale invitation-mask correction，而不是继续增加固定邻域或普通 candidate-generation heuristic。

`make invitation-mask-correction-formal` 已验证该方向有效：`mc=1` 在同样 preview `16` 下将 slots/gap 从 `3.189/0.432` 降到 `2.684/0.292`，failed/missed 从 `0.546/0.864` 降到 `0.333/0.333`。

`make invitation-mask-correction-noise-sweep` 进一步完成 robustness boundary：feedback-noise std 为 `0.02/0.05` 时，`mc=1` 仍把 gap 分别从未修正的 `0.578/0.753` 降到 `0.503/0.668`；std 为 `0.1` 时，`mc=0.75` gap `0.842` 略好于 `mc=1` 的 `0.856`，且 failed invitations 更少。这个结果解释了为什么需要 target-count clipping。

`make invitation-mask-correction-noise-aware-formal` 已完成 clipping 版 target-count correction。deadband `z>0` 的 broad pilot 会压掉有用修正，因此没有进入 formal；formal 只保留 direct 与 `clip=2`。结果显示，`mc=1 clip=2` 在 feedback-noise std `0.1` 下把 gap 从 direct `mc=1` 的 `0.856` 降到 `0.818`，failed 从 `5.264` 降到 `3.275`，missed 从 `0.734` 降到 `0.599`。低/中噪声下 direct `mc=1` 仍是最低 gap，因此最终表述应是可靠反馈使用 unclipped `mc=1`，高噪声使用 clipped conservative variant。

最新 Neighbor-Coverage pilot 尝试把 `B=3 sm=4.1` 的部分 stale preview 从均匀 grid seed 改成 stale leader 的 local neighbors，但最佳 gap `0.452` 仍弱于当前主线 `0.432`，因此只保留为 negative diagnostic。

具体判断标准：

1. 在相近 preview 下降低 `Coverage-Aware B=3 sm=4.1` 的 oracle gap。
2. 不明显牺牲 slots / perfect rate。
3. 能解释 failed invitations 和 missed opportunities 的 tradeoff。
4. 至少对比 `Rotating B=8`、`Adaptive V2`、`Sparse-TopK sm=3`、`Stale-TopK` 和 `Temporal Deviation Oracle`。
