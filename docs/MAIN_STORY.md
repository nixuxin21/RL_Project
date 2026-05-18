# Main Story

本项目当前主线已经从“验证 IRS 是否能提升 multi-slot AirComp”收敛为：

> IRS-assisted multi-slot AirComp under stale/limited CSI and execution-channel mismatch, using low-cost IRS candidate generation plus current aggregate feedback.

主线目标不是继续堆叠所有历史策略，而是在有限 preview/feedback 成本下，解释哪些 IRS 候选生成机制能缩小与 hidden current-channel oracle 的差距。

## Active Research Stack

| 层级 | 方法 | 角色 |
|---|---|---|
| 基础下界 | `No IRS` | 证明 IRS 是否必要 |
| 基础 IRS baseline | `Random IRS` | 有 IRS 但不优化相位 |
| 高成本参照 | `Greedy IRS` / full preview | 说明完整 codebook 搜索能达到的效果 |
| 低成本部署 baseline | `Rotating B=8` | 当前最强低成本规则基线 |
| 中成本主候选 | `Sparse-TopK B=4 sm=3` | 当前最适合报告的 medium-cost positive result |
| 候选生成改进 | `Coverage-Aware B=4 cw=0.5 cpw=0` | 同等 preview 预算下缓解 missed opportunities 的 B=4 参照 |
| 主线预算分配 | `Coverage-Aware B=3 sm=4.1 cw=0.5 cpw=0` | 同等 preview 下把更多预算用于 stale candidate breadth 的设置 |
| Mask-correction trade-off | `Mask-Corrected Coverage-Aware B=3 mc=1` | 用 aggregate current feedback count 修正 stale invitation mask；降低 slots/failed/missed 但 no-noise gap 回升 |
| 自适应折中 | `Adaptive Sparse-TopK v2` | 展示 preview 成本和 execution gap 的连续折中 |
| 高成本正向参照 | `Stale-TopK B=4` | full stale ranking + current feedback 的稳定正信号 |
| 隐藏信息 temporal diagnostic reference | `Temporal Deviation Oracle B=4` | 证明同等 probe budget 下仍有 candidate-set headroom，但不是 global upper bound |

## Current Main Result

最终主表由 `make execution-baseline-summary` 生成，输出到：

- `docs/EXECUTION_BASELINE_SUMMARY.md`
- `results/execution_mismatch/final_execution_baseline_summary.csv`

主线机制分析由 `make main-results-analysis` 生成，输出到 `docs/MAIN_RESULTS_ANALYSIS.md`，并生成 preview-gap frontier 与 failed/missed tradeoff 图。

当前最重要结论：

| 方法 | Slots | Preview | Gap | 判断 |
|---|---:|---:|---:|---|
| `Rotating B=8` | 3.992 | 8.00 | 2.596 | 低成本部署强 baseline |
| `Adaptive V2 mt=0.05 pc=0.005` | 3.857 | 14.57 | 2.439 | 成本-质量中间点 |
| `Adaptive V2 mt=0.05 pc=0.002` | 3.814 | 15.27 | 2.396 | 更偏性能的 adaptive 点 |
| `Sparse-TopK B=4 sm=3` | 3.770 | 16.00 | 2.334 | 当前 reportable medium-cost baseline |
| `Coverage-Aware B=4 cw=0.5 cpw=0` | 3.649 | 16.00 | 2.246 | B=4 coverage reference |
| `Coverage-Aware B=3 sm=4.1 cw=0.5 cpw=0` | 3.604 | 16.00 | 2.207 | no-noise same-preview gap reference |
| `Mask-Corrected Coverage-Aware B=3 mc=1` | 3.232 | 16.00 | 2.382 | slots/failed trade-off; no-noise gap regression |
| `Stale-TopK B=4` | 3.803 | 20.00 | 2.370 | 高成本正向参照 |
| `Temporal Deviation Oracle B=4` | 3.359 | 4.00 | 2.162 | hidden-info temporal diagnostic |

## What To Continue

短期继续投入的方向：

1. 围绕 `Rotating B=8 -> Adaptive V2 -> Sparse-TopK sm=3 -> Coverage-Aware B=4 -> Coverage-Aware B=3 sm=4.1 -> Mask-Corrected Coverage-Aware B=3 -> Stale-TopK -> Temporal Deviation Oracle` 写清主线故事。
2. 做更干净的 cost-quality frontier，而不是继续增加学习式分支数量。
3. 以 `coverage_sparse_topk_feedback` 作为当前主线方法：formal power ablation 选择 `cpw=0`，weight ablation 显示 `cw=0/0.25/0.5` 在主指标上基本打平；budget split formal 选择 `Coverage-Aware B=3 sm=4.1 cw=0.5 cpw=0`，它在同样 preview `16` 下进一步降低 gap 和 missed opportunities。Neighbor-Coverage local reallocation pilot 没有超过它。
4. `make coverage-b3-failure-diagnosis` 显示当前 B3 residual gap share 为 invitation `0.687`、selection `0.260`、confirmation `0.024`、pool `0.029`，说明继续只改 seed pool 或 final subset 不是最高优先级。
5. `make invitation-mask-correction-formal` 验证了 mask correction 是同 preview trade-off：`mc=1` 在 preview `16` 下把 slots 从 `3.604` 降到 `3.232`，failed/missed 从 `5.804/5.438` 降到 `5.385/5.385`，但 no-noise gap 从 `2.207` 升到 `2.382`。
6. `make invitation-mask-correction-noise-aware-formal` 已完成 high-noise boundary：feedback-noise std `0.1` 时 direct `mc=1` 是 gap-best correction，gap `2.080`；`mc=1 clip=2` 把 failed invitations 从 direct 的 `8.803` 降到 `8.395`，但 gap 较 direct 更高。
7. `make final-invitation-mask-analysis` 已生成最终论文级结果包：no-noise direct `mc=1` trade-off，高噪声 direct gap-best，以及 `mc=1 clip=2` failed-invitation diagnostic。
8. 下一步不应再开普通 candidate-generation heuristic，而应把这条 invitation-mask correction 线写成主贡献与补充鲁棒性表。
9. 保留 learned shortlist 作为诊断表，不把当前线性 learned variants 当作主方法。

## What Not To Continue

以下方向不再作为新增研究投入点：

- 普通 SAC / Codebook-Aware SAC 作为主方法。
- Greedy-index imitation 或 noisy supervised imitation。
- 基础 bandit selector、UCB、Thompson、feedback-conditioned MLP。
- 纯几何 diversity candidate set。
- Adaptive Sparse-TopK v3 的固定 history/local-neighbor heuristic。
- Neighbor-Coverage 的固定 local-neighbor reallocation。
- 当前线性 learned hidden set-value、execution-value、pairwise/cost-aware shortlist。

这些方向的文件和结果不删除；它们作为 negative results 和复现实验归档。详细边界见 `docs/DEPRECATED_DIRECTIONS.md`。
