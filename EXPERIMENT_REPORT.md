# MS-AirComp IRS 实验报告

## 摘要结论

当前实验已经证明：在本项目默认设置下，`Feature Argmax IRS` 是最关键的强基线。它直接读取环境观测中的 16 维 `codebook quality features`，选择最大值对应的 IRS 码本索引，在成功覆盖率和完成时延上几乎等同于 `Greedy IRS`。

最新 `K/N/M/C` 参数扫描进一步说明：这个结论不是默认参数下的偶然结果。Feature Argmax 在节点数、时隙数和 IRS 单元数变化时都稳定贴近 Greedy；唯一明显退化来自码本数 `C=8`，说明主要瓶颈是 codebook 分辨率，而不是策略规则本身。

能耗改进实验也已经完成：在 Feature Argmax 的 max-count 并列候选内加入功率 tie-break 后，`Feature Argmax PowerTie IRS` 在所有扫描配置下复现 Greedy IRS 的覆盖率、时延和能耗，同时每个决策时隙只额外 preview 少量并列候选。

这意味着当前问题形态下，继续单纯训练 SAC 的收益不大。`Codebook-Aware SAC` 落后不是因为缺少有效特征，而是因为 RL 训练没有学到一个非常简单、稳定、可解释的规则：

```text
每个时隙选择预计可调度节点数最多的 IRS codebook index。
```

因此，后续论文或实验叙事应把 `Feature Argmax IRS` 作为核心 rule-based baseline，并重新定义 RL 的研究目标。

## 实验设置

默认环境：

- 节点数: `K=50`
- 时隙数: `N=10`
- IRS 单元数: `M=64`
- IRS DFT 码本数: `C=16`
- 噪声方差: `1e-9`
- 最大发射功率: `1.0 W`
- 固定 baseline 参数: `g_th=0.001`, `alpha_th=0.05`

主要命令：

```bash
./.venv/bin/python evaluate_policy_comparison.py \
  --episodes 1000 \
  --num-seeds 5 \
  --include-codebook-aware-sac
```

推荐引用结果文件：

```text
results/policy_comparison/policy_comparison_summary_ep1000_runs5_seed2026_featargmax_powertie_cbsac.csv
results/policy_comparison/policy_comparison_results_ep1000_runs5_seed2026_featargmax_powertie_cbsac.png
```

## 主结果

`episodes=1000, num_seeds=5, seed=2026`

| Policy | Success | Perfect % | Slots | Total Energy | Avg Power | Preview / Slot |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Feature Argmax PowerTie IRS | 49.9946 | 99.48 | 2.5496 | 15.9779 | 0.3639 | 2.3225 |
| Greedy IRS | 49.9946 | 99.48 | 2.5496 | 15.9779 | 0.3639 | 16.0000 |
| Feature Argmax IRS | 49.9946 | 99.48 | 2.5404 | 16.5734 | 0.4207 | 0.0000 |
| Random IRS | 49.9654 | 96.66 | 5.1460 | 16.5710 | 0.4475 | 0.0000 |
| Codebook-Aware SAC | 49.9058 | 91.50 | 5.4206 | 16.5266 | 0.4437 | 0.0000 |
| SAC Fixed g/a | 48.9152 | 32.44 | 9.0646 | 16.0169 | 0.4229 | 0.0000 |
| SAC | 48.4394 | 12.00 | 9.7784 | 16.6524 | 0.4362 | 0.0000 |
| Fixed IRS 7 | 40.9552 | 0.00 | 10.0000 | 12.2355 | 0.2988 | 0.0000 |
| No IRS | 38.9412 | 0.00 | 10.0000 | 13.0518 | 0.3351 | 0.0000 |

Interpretation:

- `Feature Argmax PowerTie IRS` 复现了 `Greedy IRS` 的成功率、时延、能耗和平均功率。
- PowerTie 每个决策时隙只额外 preview `2.3225` 个并列候选，而 Greedy 需要 preview 全部 `16` 个码本。
- `Feature Argmax IRS` 和 `Greedy IRS` 的成功率完全一致，时延甚至略低，但能耗更高。
- `Codebook-Aware SAC` 已经接近满覆盖，但时延明显更高，说明它没有学到快速清空节点的规则。
- `Random IRS` 覆盖率很高但时延明显较差，说明“每时隙变化 IRS”本身就很重要。
- 固定 IRS 和无 IRS 都明显较弱，证明 IRS 动态选择确实是有效机制。

## 运行时间复杂度

benchmark 命令：

```bash
./.venv/bin/python benchmark_policy_runtime.py \
  --episodes 200
```

推荐引用文件：

```text
results/runtime/runtime_benchmark_ep200_seed2026.csv
```

| Policy | Decision Mean ms | Decision P95 ms | Env Step Mean ms | Preview / Slot | Slots | Energy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Feature Argmax IRS | 0.0157 | 0.0213 | 0.4451 | 0.00 | 2.580 | 16.5360 |
| Feature Argmax PowerTie IRS | 0.0925 | 0.2478 | 0.4429 | 2.54 | 2.585 | 15.8917 |
| Greedy IRS | 0.4345 | 0.5041 | 0.0376 | 16.00 | 2.585 | 15.8917 |
| SAC | 0.0908 | 0.1184 | 0.0711 | 0.00 | 9.795 | 16.6677 |
| Codebook-Aware SAC | 0.1132 | 0.1648 | 0.5943 | 0.00 | 5.465 | 16.5704 |

Interpretation:

- PowerTie 的平均决策耗时约为 Greedy 的 `21.3%`，但覆盖率、时延和能耗与 Greedy 一致。
- Feature Argmax 决策最快，但能耗高于 PowerTie/Greedy。
- 对 feature-based 策略，`env_step_time` 包含下一观测 codebook features 的生成成本；`decision_preview_calls_per_slot` 只统计 features 已经可用后，策略决策阶段额外调用 preview 的次数。

## 动作诊断证据

诊断命令：

```bash
./.venv/bin/python diagnose_policy_actions.py --episodes 1000
```

推荐引用文件：

```text
results/action_diagnostics/action_diagnostics_ep1000_runs1_seed2026_summary.csv
results/action_diagnostics/action_diagnostics_ep1000_runs1_seed2026_slot_stats.csv
results/action_diagnostics/action_diagnostics_ep1000_runs1_seed2026_slot_curves.png
```

关键发现：

| Policy | Success | Perfect % | Slots | g_th mean | alpha_th mean | Oracle Match % | Oracle Tx Gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SAC | 48.467 | 12.90 | 9.754 | 0.01313 | 0.06479 | 3.55 | 2.024 |
| Codebook-Aware SAC | 49.899 | 91.00 | 5.446 | 0.00100 | 0.05000 | 7.16 | 1.831 |
| Greedy IRS | 49.997 | 99.70 | 2.536 | 0.00100 | 0.05000 | 100.00 | 0.000 |

逐时隙表现：

- `Greedy IRS` 第 1 槽平均调度 `44.86` 个节点。
- `Codebook-Aware SAC` 第 1 槽平均调度 `40.98` 个节点。
- `SAC` 第 1 槽平均调度 `40.16` 个节点。

Interpretation:

- 完整 SAC 的 `g_th` 和 `alpha_th` 明显偏高，破坏了覆盖率和完成时延。
- Codebook-Aware SAC 固定了传输参数，因此覆盖率大幅改善。
- Codebook-Aware SAC 的主要问题是前 1-3 个时隙没有足够激进地选择高覆盖 IRS 码本。

## Imitation 证据

训练命令：

```bash
./.venv/bin/python train_greedy_imitation_selector.py
```

推荐引用文件：

```text
results/imitation/greedy_imitation_train5000_eval1000_seed2026_eval_summary.csv
results/imitation/greedy_imitation_train5000_eval1000_seed2026_eval_slot_stats.csv
results/imitation/greedy_imitation_train5000_eval1000_seed2026_confusion.png
```

结果：

| Policy | Success | Perfect % | Slots | Oracle Match % | Oracle Tx Gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| Greedy Imitation | 49.996 | 99.60 | 2.537 | 57.43 | 0.0126 |
| Feature Argmax | 49.997 | 99.70 | 2.529 | 56.19 | 0.0000 |
| Greedy IRS | 49.997 | 99.70 | 2.536 | 100.00 | 0.0000 |

Interpretation:

- 监督 imitation selector 的标签准确率只有约 `58%`，但实际调度性能等同 Greedy。
- `Feature Argmax` 与 Greedy 的 index 匹配率也只有 `56.19%`，但调度节点数完全无差距。
- 这说明许多 IRS index 在“本时隙可调度节点数”上是等价的，具体 index 不必完全匹配 Greedy。
- Greedy 的主要额外价值是功率 tie-break，不是覆盖率或时延。

## 参数泛化证据

扫描命令：

```bash
./.venv/bin/python evaluate_parameter_sweep.py
```

推荐引用文件：

```text
results/parameter_sweep/parameter_sweep_ep300_seed2026_summary.csv
results/parameter_sweep/parameter_sweep_ep300_seed2026_success.png
results/parameter_sweep/parameter_sweep_ep300_seed2026_latency.png
results/parameter_sweep/parameter_sweep_ep300_seed2026_energy.png
```

默认扫描设置：`episodes=300, seed=2026`，一因子改变 `K in {30,50,80}`、`N in {5,10,15}`、`M in {32,64,128}`、`C in {8,16,32}`。

关键覆盖/时延结果：

| Sweep | Feature Argmax | Greedy IRS | Random IRS | 主要结论 |
| --- | ---: | ---: | ---: | --- |
| `K=30/50/80` success rate | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 0.999 / 0.999 / 0.999 | 规模增大时 Feature Argmax 仍贴近 Greedy |
| `K=30/50/80` slots | 2.11 / 2.53 / 3.06 | 2.11 / 2.53 / 3.08 | 4.34 / 5.14 / 6.05 | Feature Argmax/Greedy 时延约为 Random 的一半 |
| `N=5/10/15` perfect % | 99.3 / 100.0 / 99.7 | 99.3 / 100.0 / 99.7 | 64.7 / 95.3 / 100.0 | 短时隙下规则选择更稳健 |
| `M=32/64/128` perfect % | 99.0 / 99.3 / 99.7 | 99.0 / 99.3 / 99.7 | 95.7 / 97.0 / 97.0 | IRS 单元数变化不改变主结论 |
| `C=8/16/32` perfect % | 85.0 / 99.3 / 100.0 | 85.0 / 99.3 / 100.0 | 94.7 / 95.7 / 95.0 | `C=8` 暴露码本分辨率瓶颈 |
| `C=8/16/32` slots | 3.91 / 2.57 / 2.21 | 3.91 / 2.58 / 2.20 | 5.43 / 5.19 / 5.17 | 增大码本显著降低规则策略时延 |

能耗 tie-break 结果：

| Sweep | Feature Argmax Energy | PowerTie Energy | Greedy Energy | PowerTie Preview / Slot | Greedy Preview / Slot |
| --- | ---: | ---: | ---: | ---: | ---: |
| `K=30/50/80` | 9.949 / 16.515 / 26.567 | 9.422 / 15.953 / 25.803 | 9.422 / 15.953 / 25.803 | 3.13 / 2.24 / 2.22 | 16 / 16 / 16 |
| `N=5/10/15` | 16.633 / 16.533 / 16.547 | 15.948 / 15.860 / 16.022 | 15.948 / 15.860 / 16.022 | 2.55 / 2.24 / 2.26 | 16 / 16 / 16 |
| `M=32/64/128` | 16.716 / 16.695 / 16.512 | 16.126 / 16.071 / 15.860 | 16.126 / 16.071 / 15.860 | 2.25 / 2.23 / 2.44 | 16 / 16 / 16 |
| `C=8/16/32` | 16.353 / 16.707 / 16.329 | 15.980 / 16.084 / 15.716 | 15.980 / 16.084 / 15.716 | 2.15 / 2.43 / 3.47 | 8 / 16 / 32 |

Interpretation:

- Feature Argmax 的覆盖率和时延在参数扫描中基本等同 Greedy，说明它可以作为主算法而不是临时 baseline。
- `C=8` 时 Feature Argmax 和 Greedy 同时退化，说明问题在可选 IRS 码本本身，而不是 argmax 规则没有泛化。
- Feature Argmax PowerTie 在所有配置下把能耗降到 Greedy 水平，说明 Greedy 的主要额外价值已经可以由一个局部 tie-break 规则解释。
- PowerTie 的额外 preview 次数明显小于 Greedy；但这里的 preview 次数是在 codebook features 已经可用的前提下统计的决策阶段额外开销。
- No IRS 的 success rate 稳定在约 `0.775-0.784` 且完美覆盖率为 `0%`，动态 IRS 选择的价值在不同参数下都存在。

## 当前结论边界

当前结论成立于：

- 完整 codebook quality features 可观测。
- 固定传输参数 `g_th=0.001`、`alpha_th=0.05`。
- Rayleigh fading 信道和 DFT IRS codebook。
- 无显式 probing cost；preview 次数只作为决策阶段复杂度代理。

若加入 noisy/partial features、probing cost、信道估计误差或多目标约束，PowerTie 与学习策略的相对优势需要重新实验验证。

## 对当前论文叙事的影响

原始叙事如果是：

```text
SAC 学会 IRS 与传输参数联合优化，并优于传统 baseline。
```

当前结果不支持这个叙事。更准确的叙事应改为：

```text
在固定传输参数和可观测 codebook quality features 的 MS-AirComp 调度问题中，
一个简单的 Feature Argmax 规则即可达到近 Greedy-oracle 的覆盖率和时延。
SAC 未能自动学到该简单规则，说明 RL 在当前问题设定下不是必要工具。
```

如果论文必须保留 RL 方向，应把研究问题改为更难、更有必要使用学习的方法。

## 推荐下一阶段研究目标

### 方向 A: 规则算法论文

把 `Feature Argmax IRS` 作为主算法，强调：

- 简单
- 可解释
- 无训练成本
- 接近 Greedy-oracle
- 明显优于固定 IRS / 无 IRS / SAC

适合的论文贡献：

```text
提出一种基于 codebook quality feature 的低复杂度 IRS 选择规则，
在 MS-AirComp 多时隙调度中达到近 oracle 覆盖性能。
```

需要补充实验：

- 复杂度对比: Feature Argmax vs Greedy preview all codebooks vs SAC inference。
- 实际运行时间对比: Feature Argmax vs Feature Argmax PowerTie vs Greedy vs SAC inference。
- 码本分辨率分析: 重点解释 `C=8` 到 `C=32` 的性能变化。

### 方向 B: 让 RL 问题真正有价值

当前 RL 不占优，是因为观测已经直接给出了非常强的候选质量特征。若要保留 RL，需要让规则不再显然最优。

建议改造场景：

- 移除或噪声化 codebook quality features，只给低维历史状态。
- 加入 probing cost，使每时隙不能完整 preview 16 个 codebook。
- 让 `g_th/alpha_th` 不固定，优化覆盖率、能耗和 MSE 的多目标权衡。
- 引入分布迁移，例如训练时 `K=50, M=64`，测试时改变 `K/M/C/N`。
- 加入信道估计误差或延迟，使当前 feature 不再准确。
- 加入更强能耗约束，使 “max coverage first” 不一定最优。

适合的论文贡献：

```text
在部分观测、估计误差或多目标约束下，学习型 IRS selector 比简单 Feature Argmax 更稳健。
```

### 方向 C: 混合策略

把 Feature Argmax 作为 teacher 或 fallback：

- 训练策略默认模仿 Feature Argmax。
- 只有在功率/剩余时隙/失败风险触发时才偏离规则。
- 用 RL 学习 tie-break 或多目标权衡，而不是重新学习整个 IRS 选择问题。

适合的论文贡献：

```text
Rule-guided RL: 使用可解释规则提供强先验，再用 RL 学习能耗或鲁棒性改进。
```

## 建议的下一步实验

优先级从高到低：

1. Noisy feature 实验：给 codebook features 加噪声或延迟，测试规则和 RL 的鲁棒性差异。
2. Partial probing 实验：限制每时隙只能 preview 或观测部分 codebook，比较 PowerTie、Greedy 近似和学习策略。
3. Feature ablation：去掉 16 维 codebook features 后重新评估 Codebook-Aware SAC / no-feature selector。
4. Multi-objective reward：显式优化 coverage + latency + power，看 RL 是否能在能耗上超过 Feature Argmax PowerTie。
5. 整理论文图表：把主对比、runtime benchmark、参数扫描和动作诊断压缩成最终论文表格与图。

参数泛化、能耗 tie-break、正式主对比和 runtime benchmark 已经完成。下一步应构造 noisy/partial feature 场景，判断在信息不完美或 probing 成本存在时，RL 是否能比简单规则更稳健。
