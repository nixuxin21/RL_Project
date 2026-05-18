# Baseline Strategy

当前主线研究问题是：

> IRS-assisted multi-slot AirComp 中，如何用低开销 IRS 配置提升节点聚合效率。

baseline 应该围绕这个问题分层，而不是把所有历史策略都放进主表。

execution mismatch 主线的最终聚合表由 `make execution-baseline-summary` 生成，输出为 `docs/EXECUTION_BASELINE_SUMMARY.md` 和 `results/execution_mismatch/final_execution_baseline_summary.csv`。

## 主线 Baselines

| 方法 | 角色 | 是否放主表 |
|---|---|---|
| `No IRS` | 无 IRS 下界，用来证明 IRS 是否必要 | 是 |
| `Random IRS` | 有 IRS 但不优化相位的基础 baseline | 是 |
| `Greedy IRS` | full-preview 高开销上界 | 是 |
| `Feature Argmax IRS` | 简单低复杂度规则方法；exact feature acquisition 需计入 full-codebook preview cost | 是 |
| `Feature Argmax PowerTie IRS` | exact feature + tie-break 诊断，不能把 feature acquisition 当作零成本 | 是 |

## 可选补充 Baselines

| 方法 | 角色 | 建议位置 |
|---|---|---|
| `Fixed IRS` | 单个静态 codebook 消融 | appendix 或归档 |
| `Best Fixed IRS (val-selected)` | validation-seed 选择的静态 codebook 消融，说明静态配置能力有限 | appendix |
| `SAC` / `SAC Fixed g/a` | 学习式 baseline，说明普通 RL 不自然优于规则方法 | supplement |
| `Codebook-Aware SAC` | 学习式 IRS selector，适合诊断而非主线核心 | supplement |

`Fixed IRS` 不再作为默认主线 baseline，因为它既不代表合理 IRS 优化，也不代表无需优化的 IRS 使用方式；`Random IRS` 更适合作为基础 IRS baseline。

## Partial / Active Probing Baselines

当研究问题转向有限 probing budget 时，主线 baseline 应改为：

| 方法 | 角色 |
|---|---|
| `No IRS` | 无 IRS 下界 |
| `Random IRS` | 无优化 IRS baseline |
| `Random Probe B` | budget-matched naive probing |
| `Rotating Grid Probe B` | 当前最强低成本规则 probing baseline |
| `Rotating Feedback Confirm B` | budget-matched 当前 aggregate feedback 确认基线 |
| `Active Diverse Feedback Grid B` | 当前新增低成本 active probe-set baseline |
| `Sparse-TopK Feedback Grid B` | sparse stale ranking + current feedback，默认成本约 `3B`，formal 中等成本点约 `4B` |
| `Coverage-Aware Sparse-TopK Feedback Grid B` | 在 sparse stale pool 中保留 top stale anchors，再按 marginal device coverage gain 填充剩余 feedback probes；用于直接检验 missed opportunities 是否来自候选覆盖集中 |
| `Adaptive Sparse-TopK Feedback Grid B` | v1 根据 stale margin 扩展 seed pool；v2 加入 rho/delay、历史稳定性、urgency 和 preview-cost gate；v3 测试 history/local-neighbor candidate generation |
| `Learned Sparse Shortlist Feedback Grid B` | 离线 hidden-label ranker 选择 nonuniform extra candidates；评估时仍只用 stale sparse preview + current feedback |
| `Learned Set Shortlist Feedback Grid B` | 离线 hidden-label set scorer 直接选择 final B-sized shortlist；评估时只 preview 最终选中的 extras |
| `Stale-TopK Feedback B` | 成本更高的 stale ranking + feedback confirmation 正向参照 |
| `Greedy Full Preview` | 高成本上界 |
| Proposed active probe-set method | 主方法 |

## 当前脚本约定

`evaluate_policy_comparison.py` 默认输出主线 baseline 组：

- `No IRS`
- `Random IRS`
- `Feature Argmax IRS`
- `Feature Argmax PowerTie IRS`
- `Greedy IRS`

如需复现历史静态消融，显式加入：

```bash
./.venv/bin/python evaluate_policy_comparison.py \
  --episodes 1000 \
  --num-seeds 5 \
  --skip-sac \
  --include-fixed-irs-baselines
```

如需学习式 baseline，显式加入 `--include-codebook-aware-sac`，并根据需要去掉 `--skip-sac`。

## 下一步 Active Probe-Set 入口

执行错配 / stale CSI 场景下，当前新增的第一版 active candidate-generation baseline 是：

```bash
./.venv/bin/python evaluate_execution_channel_mismatch.py \
  --mismatch-models temporal_ar1 \
  --channel-rho-values 0.7,0.9,0.98 \
  --csi-delay-slots 1,2,3 \
  --decision-error-std-values 0 \
  --execution-error-std-values 0 \
  --probe-budgets 4 \
  --policies execution_oracle,rotating,rotating_feedback_confirm,active_diverse_feedback,sparse_topk_feedback,stale_topk_feedback,temporal_deviation_oracle
```

`active_diverse_feedback` 先用一个低成本 rotating seed set 做 stale-CSI 预览，再围绕 stale winner 构造 codebook-diverse 候选集，最后只用候选集上的当前 aggregate feedback 确认 IRS。它的目标是在明显低于 `Stale-TopK Feedback` 的成本下，缩小和 `Temporal Deviation Oracle` 的差距。

`sparse_topk_feedback` 是第二版中间策略：默认 preview 约 `2B` 个 evenly spaced stale candidates，从中取约 `0.75B` 个 top 候选并保留 rotating coverage，再做 B 次当前 aggregate confirmation。它用于检验“少量 stale ranking 是否比纯几何多样性更有价值”。

当前 `make active-probe-set-pilot` 的正式结果支持第二版路线。跨 9 个 temporal AR(1) rho/delay 场景：

| 方法 | Slots | Failed | Missed | Preview | Oracle gap | 结论 |
|---|---:|---:|---:|---:|---:|---|
| `Estimated Rotating Grid B=4` | 3.887 | 1.585 | 1.098 | 4 | 1.116 | 低成本强基线 |
| `Active Diverse Feedback Grid B=4` | 4.148 | 0.579 | 2.436 | 9.35 | 1.035 | 降低 gap 但牺牲 slots/preview |
| `Sparse-TopK Feedback Grid B=4` | 3.751 | 0.518 | 1.818 | 12 | 0.773 | pilot 阶段主候选 |
| `Stale-TopK Feedback Grid B=4` | 3.373 | 0.444 | 1.321 | 20 | 0.465 | 高成本正向参照 |
| `Temporal Deviation Oracle B=4` | 2.985 | 0.326 | 0.900 | 4 | 0.345 | 隐藏 current-channel temporal diagnostic |

效用解释要谨慎：如果 preview 成本较高，普通 `Rotating B=4/B=8` 仍可能是最优；该 pilot 之后已继续 sweep `Sparse-TopK` 的 stale seed 数、top-k 比例和 probe budget，结论见下方 formal frontier。

后续 cost pilot 已完成，入口是 `make sparse-topk-cost-pilot`。主要更新：

| 配置 | Slots | Preview | Oracle gap | 判断 |
|---|---:|---:|---:|---|
| `Rotating B=8` | 3.465 | 8 | 0.760 | 成本敏感 utility 强基线 |
| `Sparse-TopK B=4 sm=2 tf=0.75` | 3.632 | 12 | 0.745 | 新默认候选，旧 `tf=0.5` 的同成本改进 |
| `Sparse-TopK B=4 sm=3 tf=0.75` | 3.423 | 16 | 0.556 | 接近 Stale-TopK，preview 更低 |
| `Stale-TopK B=4` | 3.472 | 20 | 0.492 | 高成本正向参照 |

正式 frontier 入口是 `make sparse-topk-frontier`，覆盖完整 9 个 rho/delay 场景。主要更新：

| 配置 | Slots | Preview | Oracle gap | 判断 |
|---|---:|---:|---:|---|
| `Rotating B=8` | 3.429 | 8 | 0.726 | 低成本部署强基线 |
| `Sparse-TopK B=4 sm=2 tf=0.75` | 3.635 | 12 | 0.736 | 不再优于 `Rotating B=8` |
| `Sparse-TopK B=4 sm=3 tf=0.75` | 3.376 | 16 | 0.535 | 当前可报告的中等成本候选 |
| `Stale-TopK B=4` | 3.373 | 20 | 0.465 | 高成本正向参照 |
| `Sparse-TopK B=8 sm=2 tf=0.75` | 3.322 | 24 | 0.454 | 高 preview，主要作为 frontier 端点 |

因此短期不应再把 `sm=2` 作为主推算法，而应把 `Rotating B=8` 作为最强低成本 baseline，报告 `Sparse-TopK B=4 sm=3` 作为用额外 8 次 preview 换取 gap 降低的中等成本方法。`sm=1` 退化明显，B=8 下 `sm=2` 已经打满 16 个 stale codebook，`sm=3` 没有额外意义。

Adaptive Sparse-TopK v1 已完成，入口是 `make adaptive-sparse-topk-pilot`。其作用是判断 adaptive seed pool 是否值得继续，而不是替换正式 baseline。跨完整 9 个 rho/delay 场景：

| 配置 | Slots | Preview | Oracle gap | Expansion | 判断 |
|---|---:|---:|---:|---:|---|
| `Sparse-TopK B=4 sm=2 tf=0.75` | 3.647 | 12.00 | 0.750 | - | 低成本起点 |
| `Adaptive Sparse-TopK B=4 mt=0.02` | 3.584 | 13.47 | 0.697 | 36.8% | 成本下降但不够稳 |
| `Adaptive Sparse-TopK B=4 mt=0.05` | 3.454 | 15.54 | 0.564 | 88.5% | 接近 `sm=3`，节省很小 |
| `Sparse-TopK B=4 sm=3 tf=0.75` | 3.447 | 16.00 | 0.561 | - | 中等成本参照 |

结论：adaptive margin gate 有信号，但当前单一 margin threshold 大多会扩展到 `3B`，不能把成本拉近 `Rotating B=8`。

Adaptive Sparse-TopK v2 已完成，入口是 `make adaptive-sparse-topk-v2-pilot`。它把 expansion gate 改成 margin + rho/delay uncertainty + deadline urgency - history stability - preview cost penalty。跨完整 9 个 rho/delay 场景：

| 配置 | Slots | Preview | Oracle gap | Expansion | 判断 |
|---|---:|---:|---:|---:|---|
| `V1 mt=0.02` | 3.584 | 13.47 | 0.697 | 36.8% | 低成本 adaptive 起点 |
| `V2 mt=0.02 pc=0` | 3.552 | 13.97 | 0.646 | 49.3% | 更好 gap，成本略高 |
| `V2 mt=0.05 pc=0.005` | 3.511 | 14.58 | 0.636 | 64.5% | 当前最有用中间点 |
| `V2 mt=0.05 pc=0.002` | 3.492 | 15.36 | 0.597 | 83.9% | 接近 v1 mt=0.05，但节省有限 |
| `V1 mt=0.05` | 3.454 | 15.54 | 0.564 | 88.5% | 更接近 `sm=3` |
| `Sparse-TopK B=4 sm=3` | 3.447 | 16.00 | 0.561 | - | 中等成本参照 |

结论：v2 适合作为 cost-quality continuum baseline，但仍没有把 `sm=3` 质量压到接近 `Rotating B=8` 成本。下一步不应继续只调 scalar gate，应改变候选生成本身，例如历史 best IRS、非均匀 stale shortlist 或 learned expansion utility。

Adaptive Sparse-TopK v3 已完成，入口是 `make adaptive-sparse-topk-v3-pilot`。它不再调 `2B -> 3B` expansion gate，而是在 `2B` stale sparse pool 上加入 recent confirmed IRS history prior 和 stale leader 的 local codebook neighbors。跨完整 9 个 rho/delay 场景：

| 配置 | Slots | Preview | Oracle gap | Extra | 判断 |
|---|---:|---:|---:|---:|---|
| `Sparse-TopK B=4 sm=2` | 3.647 | 12.00 | 0.750 | - | 低成本参照 |
| `Adaptive Sparse-TopK v3` | 3.583 | 12.92 | 0.754 | history prior 33.0%, selected extra 0.924 | slots 小幅好于 sm=2，但 gap 未改善 |
| `V2 mt=0.05 pc=0.002` | 3.492 | 15.36 | 0.597 | expansion 83.9% | 成本高但质量更好 |
| `Sparse-TopK B=4 sm=3` | 3.447 | 16.00 | 0.561 | - | 中等成本参照 |

结论：v3 是有用的负/中性诊断。它证明固定 history prior + local-neighbor heuristic 可以把成本压到约 `13` previews，但候选质量仍接近 `sm=2`，不能作为当前最优 baseline。下一步若继续推进，应做 learned/nonuniform shortlist scoring，而不是继续加固定邻居规则。

Learned Sparse Shortlist 已完成，入口是 `make learned-sparse-shortlist-pilot`。它先用 `train_learned_sparse_shortlist.py` 训练一个标准化线性 ranker，再在 evaluation 中选择 `ex=1/2` 个 nonuniform extra candidates。跨完整 9 个 rho/delay 场景：

| 配置 | Slots | Preview | Oracle gap | 判断 |
|---|---:|---:|---:|---|
| `Sparse-TopK B=4 sm=2` | 3.647 | 12.00 | 0.750 | 低成本参照 |
| `Adaptive Sparse-TopK v3` | 3.583 | 12.92 | 0.754 | 固定邻域诊断 |
| `Learned Sparse Shortlist ex=1` | 3.557 | 13.00 | 0.689 | 低成本 learned 改进 |
| `Learned Sparse Shortlist ex=2` | 3.570 | 14.00 | 0.653 | 当前 learned shortlist 最好 gap |
| `V2 mt=0.05 pc=0.002` | 3.492 | 15.36 | 0.597 | 成本更高，质量更好 |
| `Sparse-TopK B=4 sm=3` | 3.447 | 16.00 | 0.561 | 中等成本参照 |

结论：learned shortlist 是正向结果，但还不是最终主方法。它证明 nonuniform learned extras 比固定 local neighbors 更有价值；因此先测试了 marginal gain 标签，再决定是否需要 pairwise/listwise reranking。

Marginal-value Learned Sparse Shortlist 已完成，入口是 `make learned-sparse-shortlist-marginal-pilot`。它把训练标签改成“强制插入某个 extra candidate 后的 marginal gain”，但闭环没有优于 absolute-label 模型：

| 配置 | Slots | Preview | Oracle gap | 判断 |
|---|---:|---:|---:|---|
| `Absolute Learned ex=1` | 3.557 | 13.00 | 0.689 | 当前低成本 learned 点 |
| `Marginal Learned ex=1` | 3.607 | 13.00 | 0.689 | slots 退化 |
| `Absolute Learned ex=2` | 3.570 | 14.00 | 0.653 | 当前 learned shortlist 最好 |
| `Marginal Learned ex=2` | 3.603 | 14.00 | 0.662 | 没有超过 absolute |

结论：marginal scalar label 是有用诊断，但不应作为下一条主线。它的 offline top1 regret 较低，却没有转化为闭环收益，说明单候选 forced insertion 仍没有刻画最终 B 个 confirmation candidates 的交互。随后继续测试 set-level scorer。

Set-level Learned Shortlist 已完成，入口是 `make learned-set-shortlist-pilot`。它把训练样本改成 final B-sized shortlist variants，直接预测 hidden best-confirmation value；评估时从这些 variants 中选择一个最终确认集合。跨完整 9 个 rho/delay 场景：

| 配置 | Slots | Preview | Oracle gap | Selected extra | 判断 |
|---|---:|---:|---:|---:|---|
| `Absolute Learned ex=2` | 3.570 | 14.00 | 0.653 | 2.000 | 当前 learned shortlist 最好 |
| `Learned Set maxex=1` | 3.605 | 12.93 | 0.686 | 0.928 | 成本较低但质量退化 |
| `Learned Set maxex=2` | 3.659 | 13.68 | 0.703 | 1.681 | 选更多 extra 后更差 |
| `V2 mt=0.05 pc=0.002` | 3.492 | 15.36 | 0.597 | - | 成本更高，质量更好 |
| `Sparse-TopK B=4 sm=3` | 3.447 | 16.00 | 0.561 | - | 中等成本参照 |

结论：set-level hidden-value scorer 也是负/中性诊断。离线 top1 regret `0.0055` 很低，但闭环仍弱于 absolute-label `ex=2`，说明 hidden current set value 仍没有完全对齐 stale invitation mask 下的实际 execution outcome。随后继续测试 closed-loop execution-value label，显式把 confirmation 后的 failed invitations 和 missed opportunities 放进监督目标。

Closed-loop Execution-value Learned Shortlist 已完成，入口是 `make learned-execution-value-shortlist-pilot`。它仍在 set-level variants 上训练，但标签直接模拟当前 aggregate confirmation、stale invitation decision 和 execution-channel 成功结果，并对 failed invitations / missed opportunities 加惩罚。跨完整 9 个 rho/delay 场景：

| 配置 | Slots | Preview | Oracle gap | Selected extra | 判断 |
|---|---:|---:|---:|---:|---|
| `Absolute Learned ex=2` | 3.570 | 14.00 | 0.653 | 2.000 | 当前 learned shortlist 最好 |
| `Hidden Set maxex=2` | 3.659 | 13.68 | 0.703 | 1.681 | hidden set-value 参照 |
| `Execution-value maxex=1` | 3.589 | 12.95 | 0.702 | 0.947 | 成本较低，但 gap 未改善 |
| `Execution-value maxex=2` | 3.619 | 13.81 | 0.681 | 1.805 | 优于 hidden set-value，但仍弱于 absolute |
| `V2 mt=0.05 pc=0.002` | 3.492 | 15.36 | 0.597 | - | 成本更高，质量更好 |
| `Sparse-TopK B=4 sm=3` | 3.447 | 16.00 | 0.561 | - | 中等成本参照 |

结论：execution-value label 修正了一部分 hidden set-value 的目标错配，尤其 `maxex=2` 的 gap 从 `0.703` 降到 `0.681`；但它仍没有超过 absolute-label `ex=2` 的 `3.570/0.653/14.00`。因此随后测试 pairwise displacement / cost-aware variant selection：直接学习“替换哪个 stale candidate 才能用更少 preview 保持或降低 execution gap”。

Pairwise / Cost-aware Learned Shortlist 已完成，入口是 `make learned-pairwise-shortlist-pilot`。它把 set-level execution utility 转成 best-vs-rest pairwise differences 训练线性 ranker；`pc=0.005` 额外对每个 extra stale preview 加成本惩罚，另有 `pc=0` 诊断对照。跨完整 9 个 rho/delay 场景：

| 配置 | Slots | Preview | Oracle gap | Selected extra | 判断 |
|---|---:|---:|---:|---:|---|
| `Absolute Learned ex=2` | 3.570 | 14.00 | 0.653 | 2.000 | 当前 learned shortlist 最好 |
| `Execution-value maxex=1` | 3.589 | 12.95 | 0.702 | 0.947 | 标量 closed-loop 低成本点 |
| `Execution-value maxex=2` | 3.619 | 13.81 | 0.681 | 1.805 | 标量 closed-loop 高 extra 点 |
| `Pairwise pc=0 maxex=1/2` | 3.619 | 12.18 | 0.751 | 0.180 | 几乎不选 extra，质量退化 |
| `Pairwise pc=0.005 maxex=1/2` | 3.588 | 13.00 | 0.704 | 1.000 | 成本稳定，但 gap 未改善 |
| `V2 mt=0.05 pc=0.002` | 3.492 | 15.36 | 0.597 | - | 成本更高，质量更好 |
| `Sparse-TopK B=4 sm=3` | 3.447 | 16.00 | 0.561 | - | 中等成本参照 |

结论：当前线性 pairwise / cost-aware set scorer 也是负/中性诊断。`pc=0` 的离线 correlation 更高，但闭环几乎不选择 extra；`pc=0.005` 可以稳定选 1 个 extra，却只接近 execution-value `maxex=1`，仍弱于 absolute-label `ex=2`。短期不应继续在同一组线性 set features 上调标签或 cost，而应把 learned route 降级为诊断；主线报告应围绕 `Rotating B=8`、`Sparse-TopK sm=3`、v2 continuum、`Stale-TopK` 和 temporal deviation oracle。

Coverage-Aware Sparse-TopK 已作为 candidate-generation refinement 加入，pilot 入口是 `make coverage-sparse-topk-pilot`，weight ablation 入口是 `make coverage-sparse-topk-ablation`，power ablation 入口是 `make coverage-sparse-power-ablation`，B=4 formal 入口是 `make coverage-sparse-topk-frontier`，当前 budget split formal 入口是 `make coverage-budget-split-formal`。它不改变问题设定，也不是新主线；它针对 `Sparse-TopK sm=3` 的 missed opportunities 增加问题，在固定 sparse stale preview pool 内用 marginal device coverage gain 替代普通 rotating fill。formal power ablation 选择 `cpw=0`；weight ablation 显示 `cw=0/0.25/0.5` 在主指标上基本打平，因此保留 `cw=0.5 cpw=0`。修正 temporal prehistory 后，B=4 formal frontier 在 preview `16` 下为 slots/failed/missed/gap `3.649/5.271/6.133/2.246`；budget split formal 进一步选择 `B=3 sm=4.1`，同样 preview `16` 下为 `3.604/5.804/5.438/2.207`，它是当前 no-noise same-preview gap reference。Neighbor-Coverage local reallocation 在同样 preview `16` 下未超过当前 B3 formal，因此只保留为负诊断。`make coverage-b3-failure-diagnosis` 进一步显示，当前 B3 residual gap 的主导来源是 stale invitation-mask mismatch：invitation gap share `0.687`，高于 selection `0.260`、confirmation `0.024` 和 pool `0.029`。`make invitation-mask-correction-formal` 证明该诊断会转化为 trade-off：`Mask-Corrected Coverage-Aware B=3 mc=1` 在同样 preview `16` 下把 slots/failed/missed 降到 `3.232/5.385/5.385`，但 no-noise gap 升到 `2.382`。`make invitation-mask-correction-noise-aware-formal` 进一步显示，高噪声 std `0.1` 下 direct `mc=1` 是 gap-best correction，gap `2.080`；`mc=1 clip=2` 把 direct failed invitations 从 `8.803` 降到 `8.395`，但 gap 更高。`make final-invitation-mask-analysis` 已把这些结论整理成最终论文表和图。因此主线应报告 no-noise B3 gap reference、direct mask-correction trade-off、高噪声 direct gap-best 和 clipped failed-invitation diagnostic，而不是继续普通 candidate-generation heuristic。
