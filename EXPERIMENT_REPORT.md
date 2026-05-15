# MS-AirComp IRS 实验报告

## 摘要结论

当前实验已经证明：在本项目默认设置下，`Feature Argmax IRS` 是最关键的强基线。它直接读取环境观测中的 16 维 `codebook quality features`，选择最大值对应的 IRS 码本索引，在成功覆盖率和完成时延上几乎等同于 `Greedy IRS`。

最新 `K/N/M/C` 参数扫描进一步说明：这个结论不是默认参数下的偶然结果。Feature Argmax 在节点数、时隙数和 IRS 单元数变化时都稳定贴近 Greedy；唯一明显退化来自码本数 `C=8`，说明主要瓶颈是 codebook 分辨率，而不是策略规则本身。

能耗改进实验也已经完成：在 Feature Argmax 的 max-count 并列候选内加入功率 tie-break 后，`Feature Argmax PowerTie IRS` 在所有扫描配置下复现 Greedy IRS 的覆盖率、时延和能耗，同时每个决策时隙只额外 preview 少量并列候选。

后续 noisy feature、partial probing、probing cost、channel estimation error、bandit feedback 和 execution mismatch 实验进一步收紧了结论边界：feature 噪声会显著拉长时延，preview 有成本时 Rotating Grid 通常优于 full Greedy；当前等效信道估计误差模型下，Estimated Greedy 与 Rotating Grid 仍保持接近满覆盖；把信道误差推进到执行阶段后，失败邀请、missed opportunities 和 oracle gap 开始明显增加，但静态 execution-risk-aware 保守邀请、机会成本过滤、AR1 mean prediction 和手工 temporal reliability ranking 都没有超过 rotating baseline。Temporal Deviation Oracle 诊断显示，若能更聪明地选择 B=4 probe IRS 集合，oracle gap 可以显著下降；但 Learned Temporal Deviation、DAgger 数据聚合和 Window Temporal Deviation scorer 都没有稳定超过 rotating，说明真正有空间的是 probe-set selection，而当前仅基于历史统计的 offset/window reranking 还不足以实现这个空间。

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

## Noisy Feature 鲁棒性证据

噪声扫描命令：

```bash
./.venv/bin/python experiments/archive/evaluate_noisy_feature_sweep.py \
  --episodes 1000 \
  --num-seeds 5 \
  --noise-std-values 0,0.02,0.05,0.1,0.15,0.2,0.3 \
  --include-codebook-aware-sac
```

推荐引用文件：

```text
results/noisy_features/noisy_feature_sweep_ep1000_runs5_seed2026_cbsac.csv
results/noisy_features/noisy_feature_sweep_ep1000_runs5_seed2026_cbsac.png
```

这里的噪声是在归一化后的 codebook quality features 上加入的高斯观测噪声。Greedy IRS 仍使用精确 preview，因此代表上界；Feature Argmax/PowerTie 和 Codebook-Aware SAC 只能看到 noisy features。

关键结果：

| Noise std | Policy | Success | Perfect % | Slots | Energy | Preview / Slot |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0.00 | Feature Argmax PowerTie IRS | 49.995 | 99.48 | 2.550 | 15.978 | 2.32 |
| 0.00 | Greedy IRS | 49.995 | 99.48 | 2.550 | 15.978 | 16.00 |
| 0.05 | Feature Argmax PowerTie IRS | 49.967 | 96.74 | 4.145 | 16.575 | 0.01 |
| 0.05 | Random IRS | 49.965 | 96.66 | 5.146 | 16.571 | 0.00 |
| 0.10 | Feature Argmax PowerTie IRS | 49.934 | 93.62 | 4.891 | 16.510 | 0.15 |
| 0.10 | Codebook-Aware SAC | 49.806 | 83.14 | 6.333 | 16.490 | 0.00 |
| 0.20 | Feature Argmax PowerTie IRS | 49.904 | 90.94 | 5.231 | 16.428 | 0.73 |
| 0.20 | Codebook-Aware SAC | 49.710 | 76.38 | 6.846 | 16.432 | 0.00 |
| 0.30 | Feature Argmax PowerTie IRS | 49.893 | 90.16 | 5.345 | 16.399 | 1.05 |
| 0.30 | Codebook-Aware SAC | 49.608 | 71.28 | 7.184 | 16.366 | 0.00 |
| 0.30 | Greedy IRS | 49.995 | 99.48 | 2.550 | 15.978 | 16.00 |

Interpretation:

- Noisy features 的主要影响不是让 success mean 立刻崩掉，而是显著拉低完美覆盖率并拉长完成时延。
- `noise_std=0.05` 时，Feature Argmax/PowerTie 的完美覆盖率已经从 `99.48%` 降到 `96.74%`，平均时隙从约 `2.55` 增加到约 `4.15`。
- `noise_std=0.10` 后，Feature Argmax/PowerTie 的时延接近 Random IRS；`noise_std=0.20/0.30` 时，PowerTie 的 slots 已经略高于 Random IRS，但完美覆盖率低于 Random IRS。
- PowerTie 在 noisy features 下仍能略降能耗和时延，但无法恢复 Greedy 的覆盖率和时延；它的优势仍是 tie-break，而不是纠正 noisy argmax。
- 现有 Codebook-Aware SAC 是 exact-feature 训练得到的零样本模型，在 noisy features 下不比规则策略更稳健；`noise_std=0.30` 时完美覆盖率降到 `71.28%`，slots 升到 `7.184`。
- 当前 noisy feature 实验已经证明 exact feature 是规则策略接近 Greedy 的关键条件之一，但也说明“直接复用现有 SAC”不是解决鲁棒性的有效路径。

## Noise-Aware Imitation 证据

在 noisy feature sweep 之后，进一步训练了一个 noise-aware supervised imitation selector：训练和验证时都在 codebook quality features 上加入 `noise_std=0.2` 的高斯噪声，但标签仍来自精确 Greedy IRS。评估时扫描 `noise_std in {0,0.02,0.05,0.1,0.15,0.2,0.3}`。

训练命令：

```bash
./.venv/bin/python train_greedy_imitation_selector.py \
  --train-episodes 5000 \
  --val-episodes 1000 \
  --eval-episodes 1000 \
  --codebook-feature-noise-std 0.2 \
  --eval-noise-std-values 0,0.02,0.05,0.1,0.15,0.2,0.3
```

推荐引用文件：

```text
results/imitation/greedy_imitation_train5000_eval1000_seed2026_featnoise0p2_train_history.csv
results/imitation/greedy_imitation_train5000_eval1000_seed2026_featnoise0p2_eval_summary.csv
results/imitation/greedy_imitation_train5000_eval1000_seed2026_featnoise0p2_eval_slot_stats.csv
results/imitation/greedy_imitation_train5000_eval1000_seed2026_featnoise0p2_confusion.png
results/imitation/greedy_imitation_train5000_eval1000_seed2026_featnoise0p2_eval_noise_sweep.png
```

关键结果：

| Eval noise | Policy | Success | Perfect % | Slots | Oracle match % | Oracle tx gap | Dominant IRS |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | Greedy Imitation | 49.356 | 55.30 | 7.034 | 6.65 | 1.473 | 13 |
| 0.00 | Feature Argmax | 49.997 | 99.70 | 2.529 | 56.19 | 0.000 | 0 |
| 0.00 | Greedy IRS | 49.997 | 99.70 | 2.536 | 100.00 | 0.000 | 7 |
| 0.10 | Greedy Imitation | 49.329 | 54.50 | 6.959 | 5.78 | 1.636 | 13 |
| 0.10 | Feature Argmax | 49.941 | 94.40 | 4.817 | 11.83 | 1.279 | 4 |
| 0.20 | Greedy Imitation | 49.389 | 58.40 | 6.777 | 5.12 | 1.750 | 13 |
| 0.20 | Feature Argmax | 49.905 | 91.40 | 5.409 | 8.41 | 1.598 | 0 |
| 0.30 | Greedy Imitation | 49.460 | 64.40 | 6.573 | 5.13 | 1.832 | 13 |
| 0.30 | Feature Argmax | 49.887 | 89.60 | 5.583 | 7.90 | 1.675 | 0 |
| 0.30 | Greedy IRS | 49.997 | 99.70 | 2.536 | 100.00 | 0.000 | 7 |

Interpretation:

- Noise-aware imitation 是负结果：它在所有评估噪声强度下都没有超过 Feature Argmax。
- 验证标签准确率峰值只有 `10.21%`，最终为 `8.54%`；相比 exact-feature imitation 的约 `58%`，noisy observations 难以恢复 Greedy index 标签。
- Imitation 策略学到了明显的偏置，dominant IRS index 长期为 `13`，dominant rate 约 `22%-25%`，更接近一个带状态微调的固定索引策略，而不是 robust Greedy selector。
- 在高噪声下 imitation 的平均时隙数略低于现有 Codebook-Aware SAC 零样本模型，但它的完美覆盖率远低于 Feature Argmax；这不足以支持继续投入 plain Greedy-index imitation。
- 下一步如果要做鲁棒学习，不应继续模仿 noisy observation 下的 Greedy index，而应转向 partial probing、probing cost、选择少量候选后再精确评估，或学习 tx-count/风险估计而不是直接预测 Greedy index。

## Partial Probing 证据

在 partial probing 场景中，不再假设每个时隙都能免费获得完整的 C 维 codebook quality features。策略每个决策时隙只能精确 preview `B` 个码本，并在已 preview 候选内按 Greedy 的调度节点数、平均功率、剩余增益排序选择 IRS。Full Greedy 仍作为 `B=C=16` 的 oracle 上界；`oracle_match_rate` 和 `oracle_tx_gap_mean` 只用于离线诊断，不计入策略 preview budget。

实验命令：

```bash
./.venv/bin/python evaluate_partial_probing_sweep.py \
  --episodes 1000 \
  --num-seeds 5 \
  --probe-budgets 1,2,4,8
```

推荐引用文件：

```text
results/partial_probing/partial_probing_sweep_ep1000_runs5_seed2026_b1-2-4-8.csv
results/partial_probing/partial_probing_sweep_ep1000_runs5_seed2026_b1-2-4-8.png
```

关键结果：

| Probe B | Policy | Success | Perfect % | Slots | Energy | Preview / Slot | Oracle tx gap |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Fixed Grid Probe | 41.019 | 0.02 | 9.998 | 12.283 | 1.00 | 7.302 |
| 1 | Random Probe | 49.857 | 87.22 | 5.943 | 16.484 | 1.00 | 1.840 |
| 1 | Rotating Grid Probe | 49.963 | 96.52 | 5.166 | 16.588 | 1.00 | 1.718 |
| 2 | Random Probe | 49.968 | 96.94 | 4.394 | 16.383 | 2.00 | 1.258 |
| 2 | Rotating Grid Probe | 49.993 | 99.30 | 3.873 | 16.377 | 2.00 | 1.212 |
| 4 | Random Probe | 49.991 | 99.12 | 3.402 | 16.226 | 4.00 | 0.781 |
| 4 | Rotating Grid Probe | 49.995 | 99.48 | 3.160 | 16.206 | 4.00 | 0.731 |
| 8 | Random Probe | 49.994 | 99.46 | 2.859 | 16.085 | 8.00 | 0.353 |
| 8 | Rotating Grid Probe | 49.995 | 99.48 | 2.769 | 16.078 | 8.00 | 0.316 |
| 16 | Greedy Full Preview | 49.995 | 99.48 | 2.550 | 15.978 | 16.00 | 0.000 |

Interpretation:

- Partial probing 是比 noisy Greedy-index imitation 更有价值的下一阶段问题：它保留了强 Greedy 上界，同时引入真实的 preview budget 约束。
- 固定探测少量码本非常弱：`B=1` 固定网格几乎退化到固定 IRS，完美覆盖率只有 `0.02%`。
- 单纯 local tracking 也不可靠；`B=2` Local Probe 只有 `49.18%` 完美覆盖率，说明上一槽最优 IRS 的邻域不一定适合下一槽剩余节点。
- Rotating Grid 是当前最强的非学习 probing baseline。`B=4` 已达到与 Greedy 相同的 `99.48%` 完美覆盖率，但 slots 仍为 `3.160`，高于 Greedy 的 `2.550`。
- `B=8` Rotating Grid 只用一半 preview 次数，slots 降到 `2.769`，已经接近 full Greedy；剩余差距主要是每槽 oracle tx gap 仍有 `0.316`。
- 如果继续做学习策略，目标不应是直接预测 Greedy index，而应学习 probe schedule / candidate selection，并且必须和 `B=2/4` Rotating Grid 这个强规则 baseline 对比。

## Learned Probing 证据

在 partial probing 之后，训练了一个低维状态 learned probing selector。模型不观察完整 codebook quality features；输入只包含基础 7 维观测和上一时隙 IRS index 编码，输出 C 个码本的预计可调度比例。评估时，策略只 preview 预测 top-B 候选，并在这些候选内按 Greedy 排序规则选 IRS。

训练和评估命令：

```bash
./.venv/bin/python experiments/archive/train_learned_probing_selector.py \
  --train-episodes 5000 \
  --val-episodes 1000 \
  --eval-episodes 1000 \
  --num-eval-seeds 5 \
  --probe-budgets 1,2,4,8
```

推荐引用文件：

```text
results/learned_probing/learned_probing_train5000_eval1000_seed2026_rotatinggrid_b4_evalb1-2-4-8_train_history.csv
results/learned_probing/learned_probing_train5000_eval1000_seed2026_rotatinggrid_b4_evalb1-2-4-8_val_topk.csv
results/learned_probing/learned_probing_train5000_eval1000_seed2026_rotatinggrid_b4_evalb1-2-4-8_eval_summary.csv
results/learned_probing/learned_probing_train5000_eval1000_seed2026_rotatinggrid_b4_evalb1-2-4-8_eval.png
```

验证集 top-k 诊断：

| Probe B | Top-k oracle hit % | Oracle tx gap |
| ---: | ---: | ---: |
| 1 | 22.15 | 2.123 |
| 2 | 36.98 | 1.390 |
| 4 | 55.57 | 0.793 |
| 8 | 78.27 | 0.342 |

评估关键结果：

| Probe B | Policy | Success | Perfect % | Slots | Energy | Oracle tx gap |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Learned Probe | 49.668 | 75.78 | 6.042 | 16.384 | 1.796 |
| 1 | Rotating Grid Probe | 49.963 | 96.52 | 5.166 | 16.588 | 1.718 |
| 2 | Learned Probe | 49.905 | 91.04 | 4.539 | 16.348 | 1.237 |
| 2 | Rotating Grid Probe | 49.993 | 99.30 | 3.873 | 16.377 | 1.212 |
| 4 | Learned Probe | 49.981 | 98.14 | 3.430 | 16.226 | 0.773 |
| 4 | Rotating Grid Probe | 49.995 | 99.48 | 3.160 | 16.206 | 0.731 |
| 8 | Learned Probe | 49.991 | 99.12 | 2.887 | 16.057 | 0.391 |
| 8 | Rotating Grid Probe | 49.995 | 99.48 | 2.769 | 16.078 | 0.316 |
| 16 | Greedy Full Preview | 49.995 | 99.48 | 2.550 | 15.978 | 0.000 |

Interpretation:

- Learned probing 是负结果：它在所有预算下都没有超过 Rotating Grid。
- `B=4` 时 learned probing 接近满覆盖，但 slots 为 `3.430`，仍明显高于 Rotating Grid 的 `3.160`。
- `B=8` 时 learned probing 的能耗略低，但覆盖率、时延和 oracle tx gap 仍落后于 Rotating Grid；这不是一个足够强的学习型改进。
- 验证集 top-k hit rate 表明，低维状态只能预测出平均意义上的好码本，不能稳定识别当前信道/剩余节点状态下的最优候选集合。
- 继续学习方向必须改变问题设定：增加更有信息量的观测、显式建模 probing cost、或让策略学习主动信息获取；继续调当前低维 MLP 不是优先路线。

## Probing Cost Tradeoff 证据

在 partial probing 与 learned probing 的 summary 结果上，进一步做了显式成本分析。这里不重新运行物理环境，而是把每个策略的覆盖、时延和 preview 次数转成 node-equivalent utility：

```text
utility = success_mean - slot_cost * slots_mean - preview_cost * total_preview_calls_mean
```

其中 `total_preview_calls_mean = decision_preview_calls_per_slot_mean * slots_mean`。这个定义用于回答：当一次 preview 有明确代价时，full Greedy 的低时延是否仍值得。

分析命令：

```bash
./.venv/bin/python experiments/archive/evaluate_probing_cost_tradeoff.py
```

推荐引用文件：

```text
results/probing_cost/probing_cost_tradeoff_slot0-0p05-0p1-0p2_preview0-0p0005-0p001-0p002-0p005-0p01-0p02-0p05_candidates.csv
results/probing_cost/probing_cost_tradeoff_slot0-0p05-0p1-0p2_preview0-0p0005-0p001-0p002-0p005-0p01-0p02-0p05_winners.csv
results/probing_cost/probing_cost_tradeoff_slot0-0p05-0p1-0p2_preview0-0p0005-0p001-0p002-0p005-0p01-0p02-0p05_frontier.png
results/probing_cost/probing_cost_tradeoff_slot0-0p05-0p1-0p2_preview0-0p0005-0p001-0p002-0p005-0p01-0p02-0p05_winners.png
```

代表性 winner：

| Slot cost | Preview cost | Winner | Utility | Success | Perfect % | Slots | Total previews |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0.0000 | Greedy Full Preview B=16 | 49.995 | 49.995 | 99.48 | 2.550 | 40.794 |
| 0.00 | 0.0005 | Rotating Grid Probe B=2 | 49.989 | 49.993 | 99.30 | 3.873 | 7.746 |
| 0.05 | 0.0010 | Rotating Grid Probe B=8 | 49.834 | 49.995 | 99.48 | 2.769 | 22.155 |
| 0.05 | 0.0050 | Rotating Grid Probe B=4 | 49.773 | 49.995 | 99.48 | 3.160 | 12.639 |
| 0.10 | 0.0010 | Greedy Full Preview B=16 | 49.699 | 49.995 | 99.48 | 2.550 | 40.794 |
| 0.10 | 0.0020 | Rotating Grid Probe B=8 | 49.673 | 49.995 | 99.48 | 2.769 | 22.155 |
| 0.10 | 0.0050 | Rotating Grid Probe B=4 | 49.615 | 49.995 | 99.48 | 3.160 | 12.639 |
| 0.10 | 0.0200 | Rotating Grid Probe B=2 | 49.451 | 49.993 | 99.30 | 3.873 | 7.746 |
| 0.20 | 0.0020 | Greedy Full Preview B=16 | 49.403 | 49.995 | 99.48 | 2.550 | 40.794 |
| 0.20 | 0.0050 | Rotating Grid Probe B=8 | 49.330 | 49.995 | 99.48 | 2.769 | 22.155 |
| 0.20 | 0.0100 | Rotating Grid Probe B=4 | 49.236 | 49.995 | 99.48 | 3.160 | 12.639 |

Interpretation:

- Full Greedy 只在 preview cost 为零或极低、且 slot cost 足够高时胜出。
- 只要 preview 有非零成本，Rotating Grid 的低预算版本通常成为最优；preview cost 越高，最优预算从 `B=8` 逐步降到 `B=4` 或 `B=2`。
- `slot_cost=0.05, preview_cost=0.001` 已经足以让 `Rotating Grid B=8` 超过 full Greedy，说明 full preview 的时延优势对 preview cost 非常敏感。
- Learned Probe 没有在默认成本网格中成为 winner；它仍不是当前 probing-cost 场景的主算法候选。
- 当前最稳的论文路线是：把 Rotating Grid / Feature Argmax PowerTie 作为低复杂度规则算法主线；学习策略只有在引入更丰富观测、估计误差或多目标约束后才值得继续尝试。

## Channel Estimation Error 证据

在信道估计误差实验中，环境执行仍使用真实等效信道；策略决策时的 preview 使用加入复高斯估计误差的等效信道。该实验回答的是：如果 IRS 选择只能基于估计信道，full preview / partial probing 规则会怎样退化。

实验命令：

```bash
./.venv/bin/python evaluate_channel_estimation_error_sweep.py \
  --episodes 1000 \
  --num-seeds 5 \
  --error-std-values 0,0.02,0.05,0.1,0.2,0.3
```

推荐引用文件：

```text
results/channel_estimation/channel_estimation_error_sweep_ep1000_runs5_seed2026_err0-0p02-0p05-0p1-0p2-0p3.csv
results/channel_estimation/channel_estimation_error_sweep_ep1000_runs5_seed2026_err0-0p02-0p05-0p1-0p2-0p3.png
```

关键结果：

| Error std | Policy | Success | Perfect % | Slots | Energy | Preview / Slot | Oracle match % | Oracle tx gap |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | Exact Greedy Full Preview | 49.995 | 99.48 | 2.550 | 15.978 | 16.00 | 100.00 | 0.000 |
| 0.00 | Estimated Greedy Full Preview | 49.995 | 99.48 | 2.550 | 15.978 | 16.00 | 99.94 | 0.000 |
| 0.10 | Estimated Count Argmax | 49.988 | 98.86 | 3.060 | 16.206 | 16.00 | 40.68 | 0.404 |
| 0.10 | Estimated Greedy Full Preview | 49.995 | 99.48 | 2.741 | 15.717 | 16.00 | 62.42 | 0.313 |
| 0.10 | Estimated Rotating Grid B=4 | 49.994 | 99.38 | 3.319 | 16.009 | 4.00 | 28.34 | 0.897 |
| 0.20 | Estimated Count Argmax | 49.949 | 95.00 | 3.801 | 16.044 | 16.00 | 28.62 | 0.746 |
| 0.20 | Estimated Greedy Full Preview | 49.994 | 99.44 | 3.012 | 15.663 | 16.00 | 45.90 | 0.622 |
| 0.20 | Estimated Rotating Grid B=8 | 49.993 | 99.34 | 3.219 | 15.766 | 8.00 | 34.22 | 0.810 |
| 0.30 | Estimated Count Argmax | 49.881 | 88.24 | 4.588 | 15.958 | 16.00 | 20.63 | 1.008 |
| 0.30 | Estimated Greedy Full Preview | 49.992 | 99.22 | 3.311 | 15.664 | 16.00 | 34.94 | 0.884 |
| 0.30 | Estimated Rotating Grid B=4 | 49.987 | 98.78 | 3.816 | 15.990 | 4.00 | 21.77 | 1.218 |
| 0.30 | Estimated Rotating Grid B=8 | 49.990 | 99.02 | 3.503 | 15.768 | 8.00 | 28.08 | 1.033 |

Interpretation:

- 等效信道估计误差首先表现为时延和 oracle match 退化，而不是 success mean 立刻崩溃。
- `Estimated Greedy Full Preview` 在 `error_std=0.3` 下仍有 `99.22%` 完美覆盖率，但 slots 从 `2.550` 增加到 `3.311`；说明估计误差会错过最优快速清空选择。
- `Estimated Count Argmax` 明显更脆弱，`error_std=0.3` 时完美覆盖率降到 `88.24%`，平均 slots 升到 `4.588`；只有 noisy count 而没有候选内真实排序时，误差影响更大。
- `Estimated Rotating Grid B=8` 在高误差下仍接近 full estimated greedy，`error_std=0.3` 时 `99.02%` 完美覆盖率、`3.503` slots；它仍是强 partial-probing baseline。
- 该误差模型没有改变“强规则优先”的主结论。若要让学习策略有价值，下一步需要更难的失配：执行阶段也使用估计动作后遭遇真实信道偏移、信道延迟/偏置、或者和能耗/MSE 约束耦合。

## Bandit Feedback Stress 证据

在 bandit feedback 场景中，策略不再读取完整 CSI、节点级可调度 mask 或完整 codebook features。每个时隙只能 probe 少量 IRS 码本，并且每次 probe 只返回 noisy aggregate feedback：可发送节点比例和平均功率。执行阶段仍使用真实信道，代表节点基于本地信道自选择是否参与 AirComp。

新增 stress sweep 用更困难的物理场景和 node-equivalent utility 检验有限反馈策略：

```text
utility = success_mean - 0.1 * slots_mean - 0.005 * total_probe_calls_mean
```

正式命令：

```bash
./.venv/bin/python evaluate_bandit_feedback_stress_sweep.py \
  --episodes 1000 \
  --num-seeds 5 \
  --seed 2026 \
  --scenarios short_slots,compound_hard \
  --feedback-noise-std-values 0,0.1,0.2,0.3,0.5 \
  --probe-budgets 1,2,4 \
  --output-prefix results/bandit_feedback/bandit_feedback_stress_formal_ep1000_runs5_seed2026
```

推荐引用文件：

```text
results/bandit_feedback/bandit_feedback_stress_formal_ep1000_runs5_seed2026.csv
results/bandit_feedback/bandit_feedback_stress_formal_ep1000_runs5_seed2026.png
```

关键结果：

| Scenario | Noise | Best non-oracle | Utility | Success | Perfect % | Slots | Total probes | Oracle success |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short_slots | 0.0 | Rotating Feedback Probe B=4 | 49.383 | 49.798/50 | 81.00 | 3.456 | 13.82 | 49.995/50 |
| short_slots | 0.1 | Rotating Feedback Probe B=1 | 49.096 | 49.550/50 | 64.66 | 4.327 | 4.33 | 49.995/50 |
| short_slots | 0.2 | Rotating Feedback Probe B=1 | 49.096 | 49.550/50 | 64.66 | 4.327 | 4.33 | 49.995/50 |
| short_slots | 0.3 | Rotating Feedback Probe B=1 | 49.096 | 49.550/50 | 64.66 | 4.327 | 4.33 | 49.995/50 |
| short_slots | 0.5 | Rotating Feedback Probe B=1 | 49.096 | 49.550/50 | 64.66 | 4.327 | 4.33 | 49.995/50 |
| compound_hard | 0.0 | Rotating Feedback Probe B=1 | 78.706 | 79.194/80 | 47.04 | 4.645 | 4.65 | 79.697/80 |
| compound_hard | 0.1 | Rotating Feedback Probe B=1 | 78.706 | 79.194/80 | 47.04 | 4.645 | 4.65 | 79.697/80 |
| compound_hard | 0.2 | Rotating Feedback Probe B=1 | 78.706 | 79.194/80 | 47.04 | 4.645 | 4.65 | 79.697/80 |
| compound_hard | 0.3 | Rotating Feedback Probe B=1 | 78.706 | 79.194/80 | 47.04 | 4.645 | 4.65 | 79.697/80 |
| compound_hard | 0.5 | Rotating Feedback Probe B=1 | 78.706 | 79.194/80 | 47.04 | 4.645 | 4.65 | 79.697/80 |

高噪声负例：

| Scenario | Noise | Policy | Success | Perfect % | Slots | Total probes | Oracle tx gap |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| short_slots | 0.5 | Full Noisy Feedback B=16 | 49.244/50 | 50.70 | 4.466 | 71.45 | 2.261 |
| short_slots | 0.5 | UCB Feedback Probe B=1 | 44.710/50 | 5.66 | 4.967 | 4.97 | 5.737 |
| short_slots | 0.5 | Thompson Feedback Probe B=1 | 47.089/50 | 15.02 | 4.898 | 4.90 | 4.324 |
| compound_hard | 0.5 | Full Noisy Feedback B=8 | 78.063/80 | 22.04 | 4.805 | 38.44 | 2.617 |
| compound_hard | 0.5 | UCB Feedback Probe B=1 | 71.183/80 | 1.64 | 4.993 | 4.99 | 7.814 |
| compound_hard | 0.5 | Thompson Feedback Probe B=1 | 73.848/80 | 3.74 | 4.978 | 4.98 | 6.276 |

Interpretation:

- 这个设定比 noisy features / limited CSI 更贴近“真实环境无法获得完整 CSI”的研究目标：策略只能使用 probe 得到的聚合反馈。
- 默认环境仍偏容易，success mean 不能充分区分策略；正式 sweep 因此聚焦 `short_slots` 和 `compound_hard` 两个压力场景。
- `Rotating Feedback Probe B=1` 在有反馈噪声时是最稳的非 oracle 策略；只有 `short_slots, noise=0` 下，`B=4` 因无噪声且可多 probe 而取得更高 utility。
- `Full Noisy Feedback` 即使 probe 全部码本也不等于 oracle，因为它只看 noisy aggregate feedback；在 `compound_hard, noise=0.5` 下为 `78.063/80`、`22.04%` perfect，明显低于 oracle 的 `79.697/80`。
- 朴素 `UCB/Thompson` 在短时隙和组合困难场景下退化明显，原因是每个 slot 的剩余节点集合不断改变，跨 slot 累积的 codebook mean 很容易变成 stale feedback。
- 后续若做学习策略，应把目标定义为“学习 probe schedule / feedback-conditioned selection”，并且必须超过 `Rotating Feedback Probe B=1` 这个强规则基线。

## Learned Feedback Probing 证据

在 bandit feedback stress 的基础上，新增 `train_bandit_feedback_selector.py` 训练一个 feedback-conditioned MLP selector。训练阶段允许用离线 full-oracle preview 生成监督标签；评估阶段严格限制为有限反馈观测，只能使用：

- 环境基础 7 维 observation。
- 每个 IRS 码本的历史 noisy aggregate feedback，包括 probe 次数、EWMA score、传输比例、功率和 recency。
- 上一次选择的 IRS one-hot 与上一次 aggregate feedback。

它在决策时不读取完整 CSI、不读取 node-level mask，也不读取完整 codebook quality feature。这个实验用于回答：在真实有限 CSI / noisy probe feedback 设定下，学习型 probe selector 是否能超过简单 rotating probe 规则。

pilot 命令：

```bash
./.venv/bin/python train_bandit_feedback_selector.py \
  --scenario short_slots \
  --train-episodes 3000 \
  --val-episodes 500 \
  --eval-episodes 300 \
  --num-eval-seeds 3 \
  --epochs 20 \
  --batch-size 256 \
  --feedback-noise-std-values 0.2 \
  --probe-budgets 1,2,4 \
  --output-prefix results/bandit_feedback/learned_feedback_probe_short_slots_pilot_train3000_eval300_runs3_noise0p2

./.venv/bin/python train_bandit_feedback_selector.py \
  --scenario compound_hard \
  --train-episodes 3000 \
  --val-episodes 500 \
  --eval-episodes 300 \
  --num-eval-seeds 3 \
  --epochs 20 \
  --batch-size 256 \
  --feedback-noise-std-values 0.5 \
  --probe-budgets 1,2,4 \
  --output-prefix results/bandit_feedback/learned_feedback_probe_compound_hard_pilot_train3000_eval300_runs3_noise0p5
```

推荐引用文件：

```text
results/bandit_feedback/learned_feedback_probe_short_slots_pilot_train3000_eval300_runs3_noise0p2.csv
results/bandit_feedback/learned_feedback_probe_short_slots_pilot_train3000_eval300_runs3_noise0p2.png
results/bandit_feedback/learned_feedback_probe_short_slots_pilot_train3000_eval300_runs3_noise0p2_val_topk.csv
results/bandit_feedback/learned_feedback_probe_compound_hard_pilot_train3000_eval300_runs3_noise0p5.csv
results/bandit_feedback/learned_feedback_probe_compound_hard_pilot_train3000_eval300_runs3_noise0p5.png
results/bandit_feedback/learned_feedback_probe_compound_hard_pilot_train3000_eval300_runs3_noise0p5_val_topk.csv
```

关键结果：

| Scenario | Noise | Policy | B | Utility | Success | Perfect % | Slots | Probes | Oracle tx gap |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| short_slots | 0.2 | Oracle Full Preview | 16 | 49.540 | 49.997/50 | 99.67 | 2.539 | 40.62 | 0.000 |
| short_slots | 0.2 | Rotating Feedback Probe | 1 | 49.076 | 49.533/50 | 64.33 | 4.359 | 4.36 | 1.909 |
| short_slots | 0.2 | Learned Feedback Probe | 1 | 48.922 | 49.374/50 | 55.33 | 4.312 | 4.31 | 1.876 |
| short_slots | 0.2 | Learned Feedback Probe | 2 | 48.595 | 49.083/50 | 41.89 | 4.438 | 8.88 | 1.942 |
| short_slots | 0.2 | Learned Feedback Probe | 4 | 48.496 | 49.037/50 | 41.78 | 4.506 | 18.02 | 1.939 |
| short_slots | 0.2 | UCB Feedback Probe | 4 | 48.377 | 48.932/50 | 38.11 | 4.626 | 18.50 | 2.293 |
| compound_hard | 0.5 | Oracle Full Preview | 8 | 79.149 | 79.683/80 | 73.78 | 3.816 | 30.52 | 0.000 |
| compound_hard | 0.5 | Rotating Feedback Probe | 1 | 78.677 | 79.168/80 | 44.67 | 4.678 | 4.68 | 1.806 |
| compound_hard | 0.5 | Random IRS | 0 | 77.499 | 77.982/80 | 20.22 | 4.831 | 0.00 | 2.588 |
| compound_hard | 0.5 | Learned Feedback Probe | 1 | 77.304 | 77.812/80 | 16.78 | 4.841 | 4.84 | 2.261 |
| compound_hard | 0.5 | Learned Feedback Probe | 4 | 77.101 | 77.684/80 | 16.44 | 4.862 | 19.45 | 2.666 |
| compound_hard | 0.5 | UCB Feedback Probe | 4 | 76.588 | 77.174/80 | 14.00 | 4.883 | 19.53 | 3.273 |

验证集 top-k 诊断：

| Scenario | B=1 top-k hit | B=2 top-k hit | B=4 top-k hit |
| --- | ---: | ---: | ---: |
| short_slots | 20.96% | 34.70% | 54.07% |
| compound_hard | 25.46% | 43.49% | 68.03% |

Interpretation:

- 这是一个有价值的负结果：`Learned Feedback Probe` 在两个 stress 场景下都没有超过 `Rotating Feedback Probe B=1`，因此目前不能作为主算法。
- 学习策略不是完全无效。它在 `short_slots` 下明显好于朴素 `UCB/Thompson`，说明 feedback-conditioned history 确实包含可学习信号。
- 失败的直接原因是 one-shot supervised MLP 的 top-k 命中率太低。即使放宽到 `B=4`，验证集命中率也只有 `54.07%` 和 `68.03%`；执行阶段再叠加 noisy aggregate feedback，更多 probe 反而可能把坏候选引入选择集合。
- `Rotating Feedback Probe B=1` 强的原因是它天然避免 stale history 过拟合：剩余节点集合每个 slot 都在变，固定历史均值或离线监督标签很容易失效。
- 后续若继续学习方向，应转向 sequence model、policy-gradient / contextual bandit、imitate rotating + learned deviation，或 DAgger-like on-policy 数据收集；不应继续只增加一个离线 one-shot regressor。

## Adaptive Feedback Probing 证据

在 learned feedback probing 负结果之后，新增 `evaluate_adaptive_feedback_probing.py` 做非学习版主动 probing 诊断。它不从零替代 `Rotating Feedback Probe B=1`，而是把 rotating 作为默认主 probe：

```text
先 probe 当前 rotating codebook。
如果 noisy aggregate feedback 低于 gate_ratio * remaining_nodes / remaining_slots，
则额外 probe 一个 backup codebook。
最后在已 probe 的候选中选择 noisy feedback 更好的码本执行。
```

backup 策略包括：

- `next`: probe 下一个 codebook。
- `least_recent`: probe 最久没有被 probe 的 codebook。
- `best_history`: probe 历史 aggregate score 最高的 codebook。
- `hybrid`: 优先补未 probe 码本，全部见过后再选历史最高分。

pilot 命令：

```bash
./.venv/bin/python evaluate_adaptive_feedback_probing.py \
  --scenarios short_slots \
  --episodes 300 \
  --num-seeds 3 \
  --feedback-noise-std-values 0.2 \
  --gate-ratios 0.7,0.9,1.1 \
  --backup-strategies next,least_recent,best_history,hybrid \
  --probe-budgets 1,2 \
  --output-prefix results/bandit_feedback/adaptive_feedback_probe_short_slots_pilot_ep300_runs3_noise0p2

./.venv/bin/python evaluate_adaptive_feedback_probing.py \
  --scenarios compound_hard \
  --episodes 300 \
  --num-seeds 3 \
  --feedback-noise-std-values 0.5 \
  --gate-ratios 0.7,0.9,1.1 \
  --backup-strategies next,least_recent,best_history,hybrid \
  --probe-budgets 1,2 \
  --output-prefix results/bandit_feedback/adaptive_feedback_probe_compound_hard_pilot_ep300_runs3_noise0p5
```

推荐引用文件：

```text
results/bandit_feedback/adaptive_feedback_probe_short_slots_pilot_ep300_runs3_noise0p2.csv
results/bandit_feedback/adaptive_feedback_probe_short_slots_pilot_ep300_runs3_noise0p2.png
results/bandit_feedback/adaptive_feedback_probe_compound_hard_pilot_ep300_runs3_noise0p5.csv
results/bandit_feedback/adaptive_feedback_probe_compound_hard_pilot_ep300_runs3_noise0p5.png
results/bandit_feedback/adaptive_feedback_probe_short_slots_conservative_ep300_runs3_noise0p2.csv
results/bandit_feedback/adaptive_feedback_probe_compound_hard_conservative_ep300_runs3_noise0p5.csv
```

关键结果：

| Scenario | Noise | Policy | Utility | Success | Perfect % | Slots | Probes | Trigger % | Oracle tx gap |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| short_slots | 0.2 | Oracle Full Preview | 49.540 | 49.997/50 | 99.67 | 2.539 | 40.62 | 0.00 | 0.000 |
| short_slots | 0.2 | Rotating Feedback Probe B=1 | 49.076 | 49.533/50 | 64.33 | 4.359 | 4.36 | 0.00 | 1.909 |
| short_slots | 0.2 | Rotating Feedback Probe B=2 | 48.996 | 49.476/50 | 59.89 | 4.361 | 8.72 | 0.00 | 1.841 |
| short_slots | 0.2 | Best Adaptive: hybrid r=1.1 | 48.809 | 49.288/50 | 50.33 | 4.488 | 6.09 | 35.78 | 2.022 |
| short_slots | 0.2 | Conservative best: hybrid r=0.3 | 48.815 | 49.292/50 | 50.78 | 4.481 | 5.90 | 31.69 | 2.002 |
| compound_hard | 0.5 | Oracle Full Preview | 79.149 | 79.683/80 | 73.78 | 3.816 | 30.52 | 0.00 | 0.000 |
| compound_hard | 0.5 | Rotating Feedback Probe B=1 | 78.677 | 79.168/80 | 44.67 | 4.678 | 4.68 | 0.00 | 1.806 |
| compound_hard | 0.5 | Best Adaptive: next r=0.7 | 78.142 | 78.652/80 | 31.56 | 4.770 | 6.70 | 40.56 | 2.171 |
| compound_hard | 0.5 | Conservative best: least_recent r=0.1 | 78.130 | 78.640/80 | 31.89 | 4.772 | 6.55 | 37.22 | 2.120 |
| compound_hard | 0.5 | Rotating Feedback Probe B=2 | 77.870 | 78.391/80 | 23.67 | 4.737 | 9.47 | 0.00 | 1.961 |

Interpretation:

- 这也是一个负结果：非学习版 adaptive backup 没有超过 `Rotating Feedback Probe B=1`。
- Adaptive 在 `compound_hard` 中超过了 `Rotating B=2`，说明“条件式少量额外 probe”比固定每槽多 probe 更合理；但它仍不如单 probe rotating。
- `best_history` 明显退化，进一步证明跨 slot 历史均值在剩余节点集合变化后容易 stale。
- 即使把 gate 降到 `0.1/0.3/0.5`，触发率仍有约 `30%-39%`。原因是 high-noise aggregate feedback 被裁剪到 `[0,1]` 后会频繁出现假低反馈，单次低反馈不足以可靠判断当前 rotating 失败。
- 后续不能直接用单次 noisy feedback 做硬阈值 gate；若继续主动 probing，应改成带不确定性校正的 gate、重复/确认式 probe，或学习一个 on-policy gate 来权衡“额外 probe 成本 vs 可能补救收益”。

## Execution Channel Mismatch 证据

前面的 `Channel Estimation Error` 实验只在决策 preview 阶段加入估计误差，真正执行时仍用同一个真实信道验证成功。这低估了真实系统中的一个关键风险：调度器基于 stale/estimated CSI 邀请节点，但数据 slot 执行时信道已经漂移。

新增 `evaluate_execution_channel_mismatch.py` 把两者显式分开：

- decision channel: 策略可见的 stale/estimated CSI，用于选择 IRS 和邀请节点。
- execution channel: 对同一 IRS 码本再施加 policy-independent drift 后的信道，用于验证被邀请节点是否真的成功。
- `Execution Oracle Full CSI`: 只作为离线上界，表示如果调度器提前知道执行信道能达到的最好结果。

pilot 命令：

```bash
./.venv/bin/python evaluate_execution_channel_mismatch.py \
  --episodes 300 \
  --num-seeds 3 \
  --num-slots 5 \
  --decision-error-std-values 0 \
  --execution-error-std-values 0,0.1,0.2,0.3,0.5 \
  --probe-budgets 1,4 \
  --policies execution_oracle,exact_greedy,estimated_greedy,rotating,robust_rotating,risk_rotating,adaptive_risk_rotating \
  --robust-gain-margins 1.25 \
  --robust-power-margins 0.9 \
  --risk-weights 0.5 \
  --risk-power-weights 0.1 \
  --risk-invite-thresholds 0.5 \
  --adaptive-risk-base-weights 0.5 \
  --output-prefix results/execution_mismatch/execution_mismatch_short_slots_pilot_ep300_runs3_execerr0-0p1-0p2-0p3-0p5
```

推荐引用文件：

```text
results/execution_mismatch/execution_mismatch_short_slots_pilot_ep300_runs3_execerr0-0p1-0p2-0p3-0p5.csv
results/execution_mismatch/execution_mismatch_short_slots_pilot_ep300_runs3_execerr0-0p1-0p2-0p3-0p5_decerr0.png
```

关键结果：

| Exec error | Policy | Success | Perfect % | Slots | Failed invites | Missed opp. | Preview / Slot | Oracle tx gap |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | Execution Oracle Full CSI | 49.997/50 | 99.67 | 2.539 | 0.00 | 0.00 | 16.00 | 0.000 |
| 0.0 | Exact Greedy Full CSI | 49.997/50 | 99.67 | 2.539 | 0.00 | 0.00 | 16.00 | 0.000 |
| 0.0 | Estimated Rotating Grid B=1 | 49.533/50 | 64.33 | 4.359 | 0.00 | 0.00 | 1.00 | 1.909 |
| 0.0 | Estimated Rotating Grid B=4 | 49.986/50 | 98.56 | 3.144 | 0.00 | 0.00 | 4.00 | 0.714 |
| 0.2 | Execution Oracle Full CSI | 50.000/50 | 100.00 | 2.294 | 0.00 | 0.00 | 16.00 | 0.000 |
| 0.2 | Exact Greedy Full CSI | 49.989/50 | 98.89 | 3.070 | 2.82 | 1.55 | 16.00 | 1.168 |
| 0.2 | Estimated Rotating Grid B=1 | 49.216/50 | 46.22 | 4.660 | 3.03 | 3.92 | 1.00 | 2.848 |
| 0.2 | Estimated Rotating Grid B=4 | 49.919/50 | 92.67 | 3.657 | 2.87 | 2.22 | 4.00 | 1.700 |
| 0.5 | Execution Oracle Full CSI | 50.000/50 | 100.00 | 2.009 | 0.00 | 0.00 | 16.00 | 0.000 |
| 0.5 | Exact Greedy Full CSI | 49.972/50 | 97.56 | 3.362 | 5.39 | 3.87 | 16.00 | 2.403 |
| 0.5 | Estimated Rotating Grid B=1 | 49.012/50 | 38.56 | 4.771 | 5.57 | 10.24 | 1.00 | 3.975 |
| 0.5 | Estimated Rotating Grid B=4 | 49.849/50 | 86.22 | 3.934 | 5.43 | 5.69 | 4.00 | 2.841 |

Interpretation:

- 执行阶段失配比单纯 decision-preview 估计误差更贴近本项目原始研究希望：有限 CSI 下通过 IRS 改善 multi-slot AirComp，但调度器不能假设执行时仍掌握完整真实 CSI。
- `Execution Oracle Full CSI` 在高执行误差下仍可满覆盖，说明信道漂移不是让问题物理不可行，而是让 stale decision 产生失败邀请和错过机会。
- `Exact Greedy Full CSI` 的 success mean 看起来仍高，但 `execerr=0.5` 时失败邀请达到 `5.39`、oracle gap 达到 `2.403`；因此后续不能只看成功节点均值，必须同时报告失败邀请、missed opportunities、slot gap 和 energy。
- 有限 preview 的 `B=1` 明显更敏感，`execerr=0.5` 时完美覆盖率只有 `38.56%`；`B=4` 能恢复一部分性能，但仍比 execution oracle 慢约 `1.9` 个 slot。
- 当前 `Risk-Aware Rotating Grid` 和 `Adaptive Risk-Aware Rotating Grid` 仍只把 decision CSI error 当成风险来源。这个 pilot 中 `decision_error_std=0`、只有 execution drift，所以它们基本退化为 rotating。下一步真正有研究价值的是：让策略使用执行漂移统计量来估计“邀请后失败风险”，而不是只对 noisy decision preview 做保守化。

### Execution-Risk-Aware pilot

为验证“显式使用执行漂移统计量”是否足够，新增两个本地策略：

- `Execution-Risk Rotating Grid`: 仍使用 rotating probe 集合，但将 `sqrt(decision_error_std^2 + execution_error_std^2)` 转成节点成功可靠度，再用 risk-aware score 排序和邀请。
- `Adaptive Execution-Risk Rotating Grid`: 在上述基础上按剩余 deadline/backlog 调整 risk weight。

它们不读取 realized execution channel；只使用漂移方差统计量。

pilot 命令：

```bash
./.venv/bin/python evaluate_execution_channel_mismatch.py \
  --episodes 300 \
  --num-seeds 3 \
  --num-slots 5 \
  --decision-error-std-values 0 \
  --execution-error-std-values 0.2,0.5 \
  --probe-budgets 1,4 \
  --policies execution_oracle,rotating,execution_risk_rotating,adaptive_execution_risk_rotating \
  --risk-weights 0.1,0.5 \
  --risk-power-weights 0.1 \
  --risk-invite-thresholds 0.5,0.55 \
  --adaptive-risk-base-weights 0.1,0.5 \
  --output-prefix results/execution_mismatch/execution_risk_pilot_ep300_runs3_execerr0p2-0p5_rw0p1-0p5_rt0p5-0p55
```

推荐引用文件：

```text
results/execution_mismatch/execution_risk_pilot_ep300_runs3_execerr0p2-0p5_rw0p1-0p5_rt0p5-0p55.csv
results/execution_mismatch/execution_risk_pilot_ep300_runs3_execerr0p2-0p5_rw0p1-0p5_rt0p5-0p55_decerr0.png
```

关键结果：

| Exec error | Policy | Success | Perfect % | Slots | Failed invites | Missed opp. | Oracle tx gap |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.2 | Estimated Rotating Grid B=1 | 49.216/50 | 46.22 | 4.660 | 3.03 | 3.92 | 2.848 |
| 0.2 | Execution-Risk B=1 rw=0.1 rt=0.55 | 49.141/50 | 43.33 | 4.704 | 2.46 | 5.22 | 3.172 |
| 0.2 | Estimated Rotating Grid B=4 | 49.919/50 | 92.67 | 3.657 | 2.87 | 2.22 | 1.700 |
| 0.2 | Execution-Risk B=4 rw=0.1 rt=0.55 | 49.911/50 | 91.67 | 3.787 | 2.12 | 3.20 | 1.976 |
| 0.5 | Estimated Rotating Grid B=1 | 49.012/50 | 38.56 | 4.771 | 5.57 | 10.24 | 3.975 |
| 0.5 | Execution-Risk B=1 rw=0.1 rt=0.55 | 48.741/50 | 28.89 | 4.864 | 4.80 | 14.93 | 5.064 |
| 0.5 | Estimated Rotating Grid B=4 | 49.849/50 | 86.22 | 3.934 | 5.43 | 5.69 | 2.841 |
| 0.5 | Execution-Risk B=4 rw=0.1 rt=0.55 | 49.762/50 | 79.00 | 4.293 | 4.64 | 9.66 | 3.738 |
| 0.5 | Adaptive Execution-Risk B=4 rw=0.5 rt=0.5 | 49.699/50 | 74.78 | 4.316 | 4.69 | 8.08 | 3.331 |

Interpretation:

- 这是新的负结果：显式知道 execution drift 方差本身还不够；静态保守邀请阈值没有超过普通 rotating。
- `rt=0.5` 基本等同于 ordinary rotating，因为估计可行节点的可靠度通常不低于 `0.5`。
- 把阈值轻微提高到 `0.55` 会减少失败邀请，但会错过更多真实可执行节点；在 `execerr=0.5, B=4` 下，failed 从 `5.43` 降到 `4.64`，但 missed opportunities 从 `5.69` 升到 `9.66`，oracle gap 从 `2.841` 升到 `3.738`。
- 后续鲁棒调度不能只做 static conservative invitation；需要显式建模机会成本，例如 deadline-aware threshold、按剩余 backlog 调整的 expected utility、或把 false reject 与 false accept 的代价同时纳入目标。

### Opportunity-Cost Execution-Risk pilot

为检验“机会成本感知”是否能修复 static threshold 的 false reject 问题，新增 `Opportunity-Cost Execution-Risk Rotating Grid`：

- IRS 候选集合仍来自 rotating probe，不使用 realized execution channel。
- 每个估计可行节点根据执行漂移方差得到 reliability。
- 邀请决策比较 expected utility：邀请收益为可靠成功进度，失败项为 false accept cost；不邀请损失为 missed opportunity cost。
- 当前 slot 的 urgency 由剩余 deadline 和 backlog 共同决定，用于放大临近 deadline 或剩余节点较多时的 false reject 代价。

pilot 命令：

```bash
./.venv/bin/python evaluate_execution_channel_mismatch.py \
  --episodes 300 \
  --num-seeds 3 \
  --num-slots 5 \
  --decision-error-std-values 0 \
  --execution-error-std-values 0.2,0.5 \
  --probe-budgets 1,4 \
  --policies execution_oracle,rotating,execution_risk_rotating,opportunity_execution_risk_rotating \
  --risk-weights 0.1 \
  --risk-power-weights 0.1 \
  --risk-invite-thresholds 0.55 \
  --opportunity-failure-costs 0.5,1,2 \
  --opportunity-missed-costs 1,2 \
  --opportunity-deadline-gains 0.5 \
  --opportunity-backlog-gains 0.5 \
  --output-prefix results/execution_mismatch/opportunity_execution_risk_pilot_ep300_runs3_execerr0p2-0p5_fc0p5-1-2_mc1-2
```

推荐引用文件：

```text
results/execution_mismatch/opportunity_execution_risk_pilot_ep300_runs3_execerr0p2-0p5_fc0p5-1-2_mc1-2.csv
results/execution_mismatch/opportunity_execution_risk_pilot_ep300_runs3_execerr0p2-0p5_fc0p5-1-2_mc1-2_decerr0.png
```

关键结果：

| Exec error | Policy | Success | Perfect % | Slots | Failed invites | Missed opp. | Oracle tx gap |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.2 | Estimated Rotating Grid B=1 | 49.216/50 | 46.22 | 4.660 | 3.03 | 3.92 | 2.848 |
| 0.2 | Opportunity-Cost B=1 fc=0.5 mc=1 | 49.216/50 | 46.22 | 4.660 | 3.03 | 3.92 | 2.848 |
| 0.2 | Estimated Rotating Grid B=4 | 49.919/50 | 92.67 | 3.657 | 2.87 | 2.22 | 1.700 |
| 0.2 | Opportunity-Cost B=4 fc=0.5 mc=1 | 49.898/50 | 90.67 | 3.746 | 2.50 | 2.43 | 1.799 |
| 0.2 | Opportunity-Cost B=4 fc=2 mc=1 | 49.727/50 | 76.67 | 4.210 | 1.87 | 3.58 | 2.325 |
| 0.5 | Estimated Rotating Grid B=1 | 49.012/50 | 38.56 | 4.771 | 5.57 | 10.24 | 3.975 |
| 0.5 | Opportunity-Cost B=1 fc=0.5 mc=1 | 49.012/50 | 38.56 | 4.771 | 5.57 | 10.24 | 3.975 |
| 0.5 | Estimated Rotating Grid B=4 | 49.849/50 | 86.22 | 3.934 | 5.43 | 5.69 | 2.841 |
| 0.5 | Opportunity-Cost B=4 fc=0.5 mc=1 | 49.823/50 | 84.22 | 4.023 | 5.07 | 6.37 | 2.949 |
| 0.5 | Opportunity-Cost B=4 fc=2 mc=1 | 48.988/50 | 39.56 | 4.762 | 4.12 | 12.38 | 4.404 |

Interpretation:

- 这是第二个负/中性结果：动态机会成本邀请过滤避免了 static threshold 的部分过度拒绝，但没有超过普通 rotating。
- 低失败代价时，机会成本规则基本退化为 rotating，因此不会修复 execution mismatch。
- 高失败代价时，failed invites 会下降，但代价是 success、perfect rate、slot 和 oracle gap 同时变差；这说明 false reject 仍然过贵。
- 当前证据表明，单纯在 rotating 候选内部做 reliability-based invitation filter 不足以产生研究增益。下一步更应考虑更真实的信道时间相关性、多目标约束、重复确认/置信反馈，或让策略改变 probe/IRS 选择本身，而不是只过滤已选 IRS 下的节点。

### Temporal AR(1) CSI delay pilot

为把 execution mismatch 从独立扰动推进到更真实的 stale CSI，新增 `temporal_ar1` mismatch model：

- 每个 episode 先生成一条 AR(1) 物理信道序列。
- 调度器在 slot `t` 只能看到 `t-delay` 的 delayed CSI。
- 执行 AirComp 时使用 slot `t` 的当前信道。
- `Execution Oracle Full CSI` 仍只作为离线上界。
- 新增 `AR1-Predict Rotating Grid`：用 `rho^delay` 缩放 delayed physical channels 作为当前信道均值预测，再做 rotating。

pilot 命令：

```bash
./.venv/bin/python evaluate_execution_channel_mismatch.py \
  --episodes 300 \
  --num-seeds 3 \
  --num-slots 5 \
  --mismatch-models temporal_ar1 \
  --channel-rho-values 0.7,0.9,0.98 \
  --csi-delay-slots 1,2,3 \
  --decision-error-std-values 0 \
  --execution-error-std-values 0 \
  --probe-budgets 1,4 \
  --policies execution_oracle,exact_greedy,rotating,ar1_predict_rotating \
  --output-prefix results/execution_mismatch/temporal_ar1_pilot_ep300_runs3_rho0p7-0p9-0p98_delay1-2-3 \
  --no-plots
```

推荐引用文件：

```text
results/execution_mismatch/temporal_ar1_pilot_ep300_runs3_rho0p7-0p9-0p98_delay1-2-3.csv
```

关键结果：

| Rho | Delay | Policy | Success | Perfect % | Slots | Failed invites | Missed opp. | Oracle tx gap |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.7 | 1 | Execution Oracle | 50.000/50 | 100.00 | 2.132 | 0.00 | 0.00 | 0.000 |
| 0.7 | 1 | Exact Greedy stale CSI | 49.989/50 | 98.89 | 3.139 | 1.38 | 0.51 | 0.533 |
| 0.7 | 1 | Rotating B=1 stale CSI | 49.560/50 | 64.33 | 4.451 | 2.38 | 4.23 | 2.525 |
| 0.7 | 1 | Rotating B=4 stale CSI | 49.941/50 | 94.22 | 3.673 | 1.87 | 1.35 | 1.267 |
| 0.7 | 3 | Rotating B=4 stale CSI | 49.910/50 | 91.44 | 3.901 | 1.89 | 1.72 | 1.299 |
| 0.7 | 3 | AR1-Predict B=4 | 13.298/50 | 0.00 | 5.000 | 0.87 | 160.50 | 35.724 |
| 0.9 | 3 | Rotating B=1 stale CSI | 49.032/50 | 39.56 | 4.739 | 2.46 | 4.43 | 2.642 |
| 0.9 | 3 | Rotating B=4 stale CSI | 49.817/50 | 84.11 | 3.924 | 1.89 | 1.25 | 1.215 |
| 0.9 | 3 | AR1-Predict B=4 | 46.480/50 | 2.78 | 4.992 | 2.14 | 21.71 | 6.194 |
| 0.98 | 3 | Rotating B=4 stale CSI | 49.871/50 | 88.56 | 3.651 | 1.04 | 0.60 | 0.967 |
| 0.98 | 3 | AR1-Predict B=4 | 49.774/50 | 81.00 | 4.027 | 1.04 | 2.15 | 1.460 |

Interpretation:

- Temporal CSI delay 是更贴近真实有限 CSI 的 mismatch setting：执行信道不是独立噪声，而是当前信道相对 stale CSI 的时间演化。
- `Execution Oracle` 在所有 rho/delay 下仍为 `50/50`，说明问题不是物理不可行，而是 stale/limited CSI 下调度信息不足。
- Full stale greedy 仍接近 oracle，说明只要能 full-preview 所有码本，即使用 delayed CSI 也很强；真正有研究价值的瓶颈仍在 partial probing。
- `Rotating B=4` 仍是强基线。即使 `rho=0.7, delay=3`，仍有 `49.910/50` 和 `91.44%` perfect rate；学习策略必须先超过这个基线。
- 朴素 AR1 mean prediction 是负结果。它减少了部分失败邀请，但因为按均值缩放信道幅度，产生大量 missed opportunities；后续如果做预测，应预测可调度概率、置信下界或分位数，而不是直接用复信道均值当作可执行信道。

### Temporal-Reliability Rotating pilot

为验证“预测可调度概率/分位数”是否比 AR1 复信道均值更合理，新增 `Temporal-Reliability Rotating Grid`：

- 仍只使用 rotating probe budget 内的候选，不访问真实当前 CSI。
- 根据 delayed CSI、`rho` 和 delay 得到 temporal error std。
- 对每个 stale candidate 估计当前 slot 的 node-level schedulability reliability。
- 用 `expected_success - risk_weight * risk_mass - risk_power_weight * power_avg` 排序候选，并把 quantile lower-bound valid count 作为 tie-break/诊断项。
- 不硬过滤邀请节点，避免 AR1 mean prediction 中过度 false reject 的问题。

pilot 命令：

```bash
./.venv/bin/python evaluate_execution_channel_mismatch.py \
  --episodes 300 \
  --num-seeds 3 \
  --num-slots 5 \
  --mismatch-models temporal_ar1 \
  --channel-rho-values 0.7,0.9,0.98 \
  --csi-delay-slots 1,2,3 \
  --decision-error-std-values 0 \
  --execution-error-std-values 0 \
  --probe-budgets 4 \
  --policies execution_oracle,exact_greedy,rotating,temporal_reliability_rotating \
  --risk-weights 0,0.5,1 \
  --risk-power-weights 0.1 \
  --temporal-reliability-z-values 0,1 \
  --output-prefix results/execution_mismatch/temporal_reliability_pilot_ep300_runs3_rho0p7-0p9-0p98_delay1-2-3_b4_rw0-0p5-1_qz0-1 \
  --no-plots
```

推荐引用文件：

```text
results/execution_mismatch/temporal_reliability_pilot_ep300_runs3_rho0p7-0p9-0p98_delay1-2-3_b4_rw0-0p5-1_qz0-1.csv
```

关键结果：

| Rho | Delay | Policy | Success | Perfect % | Slots | Failed invites | Missed opp. | Oracle tx gap |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.7 | 1 | Rotating B=4 stale CSI | 49.941/50 | 94.22 | 3.673 | 1.87 | 1.35 | 1.267 |
| 0.7 | 1 | Temporal-Reliability B=4 rw=0.5 | 49.947/50 | 94.67 | 3.721 | 1.86 | 1.53 | 1.373 |
| 0.7 | 3 | Rotating B=4 stale CSI | 49.910/50 | 91.44 | 3.901 | 1.89 | 1.72 | 1.299 |
| 0.7 | 3 | Temporal-Reliability B=4 rw=0 | 49.906/50 | 91.11 | 3.920 | 1.91 | 1.72 | 1.300 |
| 0.9 | 2 | Rotating B=4 stale CSI | 49.851/50 | 86.44 | 3.869 | 1.87 | 1.20 | 1.209 |
| 0.9 | 2 | Temporal-Reliability B=4 rw=0 | 49.856/50 | 86.67 | 3.870 | 1.86 | 1.19 | 1.224 |
| 0.9 | 3 | Rotating B=4 stale CSI | 49.817/50 | 84.11 | 3.924 | 1.89 | 1.25 | 1.215 |
| 0.9 | 3 | Temporal-Reliability B=4 rw=0 | 49.814/50 | 84.00 | 3.916 | 1.89 | 1.23 | 1.226 |
| 0.98 | 3 | Rotating B=4 stale CSI | 49.871/50 | 88.56 | 3.651 | 1.04 | 0.60 | 0.967 |
| 0.98 | 3 | Temporal-Reliability B=4 rw=0 | 49.862/50 | 87.78 | 3.669 | 1.05 | 0.61 | 1.002 |

Interpretation:

- 这是比 AR1 mean prediction 更温和的预测方式，但仍是负/中性结果。
- `rw=0` 基本退化为 stale rotating，只在少数场景有极小 success/perfect 波动，但 oracle gap 通常略差。
- `rw=0.5/1` 的风险惩罚会让选择更保守，常见结果是 slots、missed opportunities 和 oracle gap 上升。
- `qz=0/1` 在当前排序下几乎没有改变结果，因为 quantile lower-bound 只是 tie-break/诊断项，不是主评分项。
- 当前证据说明：手工 reliability ranking 仍不足以超过 `Rotating B=4`。若继续有限 CSI + temporal mismatch 方向，研究点应从“候选内部重排”转向学习何时偏离 rotating probe 顺序、on-policy feedback confirmation，或显式多目标约束。

### Temporal Deviation Oracle diagnostic

为判断“学习何时偏离 rotating probe 顺序”是否有足够上界空间，新增 `Temporal Deviation Oracle` 诊断：

- 它不可部署，使用隐藏 current execution channel 从全部 codebook 中选出当前 top-B IRS 候选。
- 它仍只把这 B 个 IRS 当作可 probe 集合，`Preview` 仍按 B 统计。
- 邀请节点仍由 stale/estimated decision CSI 决定，不直接使用 current CSI 的邀请 mask。
- 因此它回答的问题是：如果 B=4 probe set 选得更聪明，是否能明显超过 rotating B=4？

pilot 命令：

```bash
./.venv/bin/python evaluate_execution_channel_mismatch.py \
  --episodes 300 \
  --num-seeds 3 \
  --num-slots 5 \
  --mismatch-models temporal_ar1 \
  --channel-rho-values 0.7,0.9,0.98 \
  --csi-delay-slots 1,2,3 \
  --decision-error-std-values 0 \
  --execution-error-std-values 0 \
  --probe-budgets 4 \
  --policies execution_oracle,exact_greedy,rotating,temporal_deviation_oracle \
  --output-prefix results/execution_mismatch/temporal_deviation_oracle_pilot_ep300_runs3_rho0p7-0p9-0p98_delay1-2-3_b4 \
  --no-plots
```

推荐引用文件：

```text
results/execution_mismatch/temporal_deviation_oracle_pilot_ep300_runs3_rho0p7-0p9-0p98_delay1-2-3_b4.csv
```

关键结果：

| Rho | Delay | Policy | Success | Perfect % | Slots | Failed invites | Missed opp. | Oracle tx gap |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.7 | 1 | Rotating B=4 stale CSI | 49.941/50 | 94.22 | 3.673 | 1.87 | 1.35 | 1.267 |
| 0.7 | 1 | Temporal Deviation Oracle B=4 | 50.000/50 | 100.00 | 2.923 | 0.30 | 1.21 | 0.454 |
| 0.7 | 3 | Rotating B=4 stale CSI | 49.910/50 | 91.44 | 3.901 | 1.89 | 1.72 | 1.299 |
| 0.7 | 3 | Temporal Deviation Oracle B=4 | 49.990/50 | 99.00 | 3.230 | 0.30 | 1.62 | 0.493 |
| 0.9 | 2 | Rotating B=4 stale CSI | 49.851/50 | 86.44 | 3.869 | 1.87 | 1.20 | 1.209 |
| 0.9 | 2 | Temporal Deviation Oracle B=4 | 49.991/50 | 99.11 | 3.057 | 0.39 | 0.99 | 0.379 |
| 0.9 | 3 | Rotating B=4 stale CSI | 49.817/50 | 84.11 | 3.924 | 1.89 | 1.25 | 1.215 |
| 0.9 | 3 | Temporal Deviation Oracle B=4 | 49.988/50 | 98.78 | 3.117 | 0.38 | 1.06 | 0.383 |
| 0.98 | 3 | Rotating B=4 stale CSI | 49.871/50 | 88.56 | 3.651 | 1.04 | 0.60 | 0.967 |
| 0.98 | 3 | Temporal Deviation Oracle B=4 | 49.989/50 | 98.89 | 2.848 | 0.31 | 0.34 | 0.195 |

Interpretation:

- 这是一个正向诊断：与 hand-crafted reliability ranking 不同，oracle 级 probe-set selection 在所有 rho/delay 下都明显超过 `Rotating B=4`。
- 提升主要体现在 perfect rate、slots 和 oracle gap，而不仅是 success mean。例如 `rho=0.9, delay=3` 下 gap 从 `1.215` 降到 `0.383`。
- Deviation oracle 还经常优于 full stale greedy 的 oracle gap，因为它用 current hidden channel 选择 probe set，但它仍不是可部署策略。
- 当前证据表明，若继续学习方向，目标应放在 probe-set selection / deviation 上，而不是继续做候选内部 reliability/invitation filter；后续 learned 和 DAgger temporal deviation 用来检验低维 offset 表示是否足够。

### Learned Temporal Deviation pilot

为把上面的 oracle 诊断推进到可部署策略，新增 `train_temporal_deviation_selector.py`。它的动作不是直接选 IRS index，而是选相对 rotating window 的 offset；例如 `offset=+1` 表示把当前 `B=4` probe window 向后平移一个 codebook。训练阶段用隐藏 current execution channel 计算每个 offset 的监督 target score，评估阶段只能使用可观测 episode 状态、`rho/delay/budget`、历史 probe 统计和上一轮执行反馈，不访问当前完整 CSI。

pilot 命令：

```bash
./.venv/bin/python train_temporal_deviation_selector.py \
  --train-episodes 500 \
  --val-episodes 100 \
  --eval-episodes 100 \
  --num-eval-seeds 3 \
  --epochs 10 \
  --batch-size 128 \
  --channel-rho-values 0.7,0.9,0.98 \
  --csi-delay-slots 1,2,3 \
  --probe-budgets 4 \
  --offsets=-3,-2,-1,0,1,2,3 \
  --output-prefix results/execution_mismatch/learned_temporal_deviation_pilot_train500_val100_eval100_runs3_rho0p7-0p9-0p98_delay1-2-3_b4_offsets7 \
  --no-plots
```

推荐引用文件：

```text
results/execution_mismatch/learned_temporal_deviation_pilot_train500_val100_eval100_runs3_rho0p7-0p9-0p98_delay1-2-3_b4_offsets7.csv
results/execution_mismatch/learned_temporal_deviation_pilot_train500_val100_eval100_runs3_rho0p7-0p9-0p98_delay1-2-3_b4_offsets7_val_offsets.csv
```

validation 诊断：`samples=357`，offset hit rate 为 `27.17%`，target score gap mean 为 `0.0187`，target tx gap mean 为 `0.0169`。这说明很多 offset 的 target score 接近，但模型做 argmax 决策时仍不能稳定选到最优 offset。

关键结果：

| Rho | Delay | Policy | Success | Perfect % | Slots | Oracle tx gap |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 0.7 | 2 | Rotating B=4 stale CSI | 49.893/50 | 89.67 | 3.950 | 1.323 |
| 0.7 | 2 | Learned Temporal Deviation B=4 | 49.910/50 | 92.33 | 3.853 | 1.343 |
| 0.7 | 2 | Temporal Deviation Oracle B=4 | 49.993/50 | 99.33 | 3.160 | 0.501 |
| 0.9 | 1 | Rotating B=4 stale CSI | 49.913/50 | 91.67 | 3.707 | 1.192 |
| 0.9 | 1 | Learned Temporal Deviation B=4 | 49.933/50 | 93.33 | 3.750 | 1.230 |
| 0.9 | 1 | Temporal Deviation Oracle B=4 | 50.000/50 | 100.00 | 2.920 | 0.373 |
| 0.9 | 3 | Rotating B=4 stale CSI | 49.827/50 | 85.00 | 3.940 | 1.242 |
| 0.9 | 3 | Learned Temporal Deviation B=4 | 49.767/50 | 80.00 | 4.013 | 1.268 |
| 0.9 | 3 | Temporal Deviation Oracle B=4 | 49.987/50 | 98.67 | 3.157 | 0.406 |
| 0.98 | 3 | Rotating B=4 stale CSI | 49.887/50 | 90.00 | 3.683 | 1.004 |
| 0.98 | 3 | Learned Temporal Deviation B=4 | 49.820/50 | 84.67 | 3.793 | 0.983 |
| 0.98 | 3 | Temporal Deviation Oracle B=4 | 49.997/50 | 99.67 | 2.857 | 0.197 |

Interpretation:

- 这是一个负/中性学习结果：learned selector 偶尔改善 success 或 perfect rate，但整体没有稳定超过 `Rotating B=4`。
- 它没有吃到 deviation oracle 的主要收益。比如 `rho=0.9, delay=3` 下 oracle gap 没有下降，反而从 rotating 的 `1.242` 升到 learned 的 `1.268`，而 hidden oracle 可降到 `0.406`。
- 低维历史特征 + 离线 hidden-target regression 目前不足以学到可部署的 probe-set deviation，因此下一步先检验 DAgger 数据聚合是否能修复分布偏移。

### DAgger Temporal Deviation pilot

为检验分布偏移是否是离线 selector 失败的主要原因，`train_temporal_deviation_selector.py` 新增 DAgger 数据聚合模式：

- 先用原始离线数据训练一个 offset selector。
- 第 1 轮用 expert/model mixture rollout 收集当前策略会访问到的状态。
- 第 2 轮用当前模型 rollout 收集更接近闭环分布的状态。
- 每个访问到的状态仍用隐藏 current-channel target 打标签，但评估策略不访问隐藏 CSI。

pilot 命令：

```bash
./.venv/bin/python train_temporal_deviation_selector.py \
  --train-episodes 500 \
  --val-episodes 100 \
  --eval-episodes 100 \
  --num-eval-seeds 3 \
  --epochs 10 \
  --batch-size 128 \
  --dagger-iterations 2 \
  --dagger-episodes 250 \
  --dagger-beta-start 0.5 \
  --dagger-beta-end 0.0 \
  --channel-rho-values 0.7,0.9,0.98 \
  --csi-delay-slots 1,2,3 \
  --probe-budgets 4 \
  --offsets=-3,-2,-1,0,1,2,3 \
  --output-prefix results/execution_mismatch/dagger_temporal_deviation_pilot_train500_val100_dagger2x250_eval100_runs3_rho0p7-0p9-0p98_delay1-2-3_b4_offsets7_beta0p5-0 \
  --no-plots
```

推荐引用文件：

```text
results/execution_mismatch/dagger_temporal_deviation_pilot_train500_val100_dagger2x250_eval100_runs3_rho0p7-0p9-0p98_delay1-2-3_b4_offsets7_beta0p5-0.csv
results/execution_mismatch/dagger_temporal_deviation_pilot_train500_val100_dagger2x250_eval100_runs3_rho0p7-0p9-0p98_delay1-2-3_b4_offsets7_beta0p5-0_val_offsets.csv
```

validation 诊断：`samples=357`，offset hit rate 为 `25.49%`，target score gap mean 为 `0.0205`，target tx gap mean 为 `0.0183`。相比离线版 `27.17%` hit rate 和 `0.0169` target tx gap，DAgger 没有改善 offset argmax。

关键结果：

| Rho | Delay | Policy | Success | Perfect % | Slots | Oracle tx gap |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 0.7 | 2 | Rotating B=4 stale CSI | 49.893/50 | 89.67 | 3.950 | 1.323 |
| 0.7 | 2 | DAgger Temporal Deviation B=4 | 49.923/50 | 93.00 | 3.780 | 1.232 |
| 0.7 | 2 | Temporal Deviation Oracle B=4 | 49.993/50 | 99.33 | 3.160 | 0.501 |
| 0.9 | 3 | Rotating B=4 stale CSI | 49.827/50 | 85.00 | 3.940 | 1.242 |
| 0.9 | 3 | DAgger Temporal Deviation B=4 | 49.760/50 | 79.67 | 4.000 | 1.266 |
| 0.9 | 3 | Temporal Deviation Oracle B=4 | 49.987/50 | 98.67 | 3.157 | 0.406 |
| 0.98 | 3 | Rotating B=4 stale CSI | 49.887/50 | 90.00 | 3.683 | 1.004 |
| 0.98 | 3 | DAgger Temporal Deviation B=4 | 49.837/50 | 85.33 | 3.767 | 0.992 |
| 0.98 | 3 | Temporal Deviation Oracle B=4 | 49.997/50 | 99.67 | 2.857 | 0.197 |

Interpretation:

- 这是第三个负/中性学习结果：DAgger 可以在少数场景改善 gap 或 slots，但没有稳定超过 `Rotating B=4`。
- `rho=0.7, delay=2` 是少数正例，DAgger 把 gap 从 `1.323` 降到 `1.232`，slots 从 `3.950` 降到 `3.780`。
- 但在 `rho=0.9, delay=3` 和高 rho 场景，DAgger 经常牺牲 perfect rate 和 slots；它仍远离 deviation oracle。
- 结论更新为：失败不只是离线数据分布偏移；当前低维 offset action/feature 表示本身不够。因此下面先检验 richer candidate/window reranking；若仍不稳定，再转向 uncertainty-aware guardrail 或显式 feedback confirmation，而不是继续做同一 MLP 的 DAgger 变体。

### Window Temporal Deviation pilot

为检验“offset 表示太弱”是否可以通过候选窗口结构缓解，新增 `--feature-mode window`。它不再用一个全局 feature vector 直接输出所有 offset 分数，而是为每个 candidate offset window 构造显式历史统计：

- offset/window 位置特征。
- window 内历史 probe 次数和 recency。
- window 内历史 stale decision tx/power 统计。
- window 内历史 success/failed/missed 反馈统计。
- 共享 MLP scorer 对每个 offset window 打分。

该模式仍不访问当前完整 CSI，也不在选择 offset 前 preview 非选中 window；它只利用过去 probe/execution 历史。

pilot 命令：

```bash
./.venv/bin/python train_temporal_deviation_selector.py \
  --train-episodes 500 \
  --val-episodes 100 \
  --eval-episodes 100 \
  --num-eval-seeds 3 \
  --epochs 10 \
  --batch-size 128 \
  --feature-mode window \
  --channel-rho-values 0.7,0.9,0.98 \
  --csi-delay-slots 1,2,3 \
  --probe-budgets 4 \
  --offsets=-3,-2,-1,0,1,2,3 \
  --output-prefix results/execution_mismatch/window_temporal_deviation_pilot_train500_val100_eval100_runs3_rho0p7-0p9-0p98_delay1-2-3_b4_offsets7 \
  --no-plots
```

推荐引用文件：

```text
results/execution_mismatch/window_temporal_deviation_pilot_train500_val100_eval100_runs3_rho0p7-0p9-0p98_delay1-2-3_b4_offsets7.csv
results/execution_mismatch/window_temporal_deviation_pilot_train500_val100_eval100_runs3_rho0p7-0p9-0p98_delay1-2-3_b4_offsets7_val_offsets.csv
```

validation 诊断：`samples=357`，offset hit rate 为 `25.49%`，target score gap mean 为 `0.0200`，target tx gap mean 为 `0.0181`。hit rate 仍没有超过离线全局 MLP。

关键结果：

| Rho | Delay | Policy | Success | Perfect % | Slots | Oracle tx gap |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 0.7 | 1 | Rotating B=4 stale CSI | 49.940/50 | 94.00 | 3.690 | 1.281 |
| 0.7 | 1 | Window Temporal Deviation B=4 | 49.957/50 | 95.67 | 3.597 | 1.240 |
| 0.7 | 2 | Rotating B=4 stale CSI | 49.893/50 | 89.67 | 3.950 | 1.323 |
| 0.7 | 2 | Window Temporal Deviation B=4 | 49.913/50 | 91.67 | 3.833 | 1.281 |
| 0.9 | 3 | Rotating B=4 stale CSI | 49.827/50 | 85.00 | 3.940 | 1.242 |
| 0.9 | 3 | Window Temporal Deviation B=4 | 49.793/50 | 83.00 | 4.070 | 1.254 |
| 0.98 | 3 | Rotating B=4 stale CSI | 49.887/50 | 90.00 | 3.683 | 1.004 |
| 0.98 | 3 | Window Temporal Deviation B=4 | 49.830/50 | 86.33 | 3.733 | 0.966 |

Interpretation:

- Window scorer 有比全局 offset MLP 更清晰的局部信号：多数场景的 oracle gap 小幅下降，例如 `rho=0.7, delay=1` 从 `1.281` 降到 `1.240`，`rho=0.98, delay=3` 从 `1.004` 降到 `0.966`。
- 但这个改善不稳定，常常用 perfect rate 或 slots 换 gap。`rho=0.9, delay=3` 下 success、perfect、slots 和 gap 都弱于 rotating。
- 结论更新为：显式 window 表示方向比纯全局特征更合理，但仅靠历史统计 reranking 还不足以部署；下一步需要 uncertainty-aware guardrail 或 feedback confirmation，让模型只有在高置信时偏离 rotating。

## 当前结论边界

当前结论成立于：

- 默认主对比中，完整且精确的 codebook quality features 可观测。
- 固定传输参数 `g_th=0.001`、`alpha_th=0.05`。
- Rayleigh fading 信道和 DFT IRS codebook。
- 默认物理环境无显式 probing cost；probing cost tradeoff 是离线效用分析。
- Channel estimation error 实验只让决策 preview 使用估计信道，实际执行仍使用真实信道。
- Execution channel mismatch 实验把决策 CSI 和执行信道分开；execution oracle 只作为离线上界，不代表可部署策略。
- Temporal AR(1) mismatch 假设信道一阶时间相关；已测试已知 `rho` 的均值预测和手工 reliability/quantile 排序，二者都没有超过 stale `Rotating B=4`。
- Temporal Deviation Oracle 使用隐藏 current CSI 选择 top-B probe set，只是上界诊断，不是可部署策略；它证明 probe-set selection 有潜在提升空间。
- Learned/DAgger/Window Temporal Deviation 训练标签使用隐藏 current-channel outcome，但闭环评估不访问当前完整 CSI；当前 offset regression、DAgger 数据聚合和历史统计 window reranking 都没有稳定超过 `Rotating B=4`。
- Bandit feedback stress sweep 中策略只能看到 noisy aggregate probe feedback；oracle 只作为离线诊断上界。
- Learned Feedback Probing 中训练标签可来自离线 full-oracle preview，但评估策略不能访问完整 CSI 或 node-level 调度结果，只能看历史 aggregate probe feedback。
- Adaptive Feedback Probing 中 gate 只使用单次 noisy aggregate feedback 和剩余 deadline，不使用隐藏真实调度结果。

加入 noisy features 后，Feature Argmax/PowerTie 会出现明显时延退化；但现有 Codebook-Aware SAC 零样本并不占优。加入 partial probing 和 probing cost 后，Rotating Grid 这类简单非学习 probing 规则已经很强；当前低维 learned probing 也没有超过它。加入当前等效信道估计误差后，Estimated Greedy/Rotating Grid 仍保持接近满覆盖。加入 bandit feedback stress 后，问题更贴近有限 CSI 的研究目标，但简单 Rotating Probe 仍是非常强的规则基线。新增 Learned Feedback Probing 进一步表明，低维历史特征 + 离线监督 MLP 仍不足以超过 rotating baseline。Adaptive Feedback Probing 又表明，单次 noisy feedback 的硬阈值 backup gate 也不足以超过 rotating baseline。Execution Channel Mismatch 则证明执行阶段漂移会显著增加失败邀请和 oracle gap，是比单纯 decision-preview 误差更有研究价值的鲁棒性方向；但当前 static execution-risk-aware conservative invitation、opportunity-cost invitation filter、AR1 mean prediction 和 hand-crafted temporal reliability ranking 都没有超过 rotating。Temporal AR(1) delay 进一步确认：更真实的 stale CSI setting 可以保留；Temporal Deviation Oracle 显示 B=4 probe set 的选择/偏离确实有上界空间，但 Learned/DAgger/Window Temporal Deviation 说明仅靠历史统计的 offset/window reranking 还不能稳定实现这个空间。

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

1. Confidence-gated window deviation：在现有 window scorer 外加 guardrail，只有当 best-vs-rotating score margin 或 uncertainty 指标足够高时才偏离 rotating，否则回退到 `Rotating B=4`；目标是保住 perfect rate/slots，同时只在高置信场景吃到 oracle gap。
2. On-policy temporal/feedback probing：加入重复确认、置信区间或信息增益，让策略知道何时值得用额外 probe 验证 stale CSI，而不是只用单槽 aggregate feedback、手工可靠度重排、DAgger 化低维 offset regressor 或历史统计 window reranking。
3. Multi-objective reward / constraint：显式优化 coverage + latency + power/MSE，看学习策略是否能在能耗或风险约束下超过 Feature Argmax PowerTie / Rotating Grid。
4. Feature ablation：去掉 16 维 codebook features 后重新评估 Codebook-Aware SAC / no-feature selector。
5. 整理论文图表：把主对比、runtime benchmark、参数扫描、noisy feature、noisy imitation、partial probing、learned probing、probing cost、channel estimation、execution mismatch、temporal AR(1) mismatch、temporal reliability、temporal deviation oracle、learned temporal deviation、DAgger temporal deviation、window temporal deviation、bandit feedback stress、learned feedback probing 和 adaptive feedback probing 结果压缩成最终论文表格与图。

参数泛化、能耗 tie-break、正式主对比、runtime benchmark、noisy feature sweep、noise-aware imitation、partial probing sweep、learned probing、probing cost tradeoff、channel estimation error sweep、execution channel mismatch pilot、static execution-risk-aware pilot、opportunity-cost execution-risk pilot、temporal AR(1) mismatch pilot、temporal reliability pilot、temporal deviation oracle diagnostic、learned temporal deviation pilot、DAgger temporal deviation pilot、window temporal deviation pilot、bandit feedback stress、learned feedback probing 和 adaptive feedback probing pilot 已经完成。下一步若继续保留学习方向，应优先做 confidence-gated window deviation 或 on-policy feedback confirmation，而不是 plain noisy Greedy-index imitation、当前低维离线 MLP、DAgger 化的低维 offset regressor、历史统计 window reranking、单次 noisy feedback hard gate、static conservative invitation、opportunity-cost invitation filter、AR1 mean prediction、hand-crafted temporal reliability ranking，或仅决策 preview 误差。
