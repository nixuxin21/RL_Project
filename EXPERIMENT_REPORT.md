# MS-AirComp IRS 实验报告

## 摘要结论

当前实验已经证明：在本项目默认设置下，`Feature Argmax IRS` 是最关键的强基线。它直接读取环境观测中的 16 维 `codebook quality features`，选择最大值对应的 IRS 码本索引，在成功覆盖率和完成时延上几乎等同于 `Greedy IRS`。

最新 `K/N/M/C` 参数扫描进一步说明：这个结论不是默认参数下的偶然结果。Feature Argmax 在节点数、时隙数和 IRS 单元数变化时都稳定贴近 Greedy；唯一明显退化来自码本数 `C=8`，说明主要瓶颈是 codebook 分辨率，而不是策略规则本身。

能耗改进实验也已经完成：在 Feature Argmax 的 max-count 并列候选内加入功率 tie-break 后，`Feature Argmax PowerTie IRS` 在所有扫描配置下复现 Greedy IRS 的覆盖率、时延和能耗，同时每个决策时隙只额外 preview 少量并列候选。

后续 noisy feature、partial probing、probing cost 和 channel estimation error 实验进一步收紧了结论边界：feature 噪声会显著拉长时延，preview 有成本时 Rotating Grid 通常优于 full Greedy；但在当前等效信道估计误差模型下，Estimated Greedy 与 Rotating Grid 仍保持接近满覆盖，尚未给学习策略创造明确优势。

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

## 当前结论边界

当前结论成立于：

- 默认主对比中，完整且精确的 codebook quality features 可观测。
- 固定传输参数 `g_th=0.001`、`alpha_th=0.05`。
- Rayleigh fading 信道和 DFT IRS codebook。
- 默认物理环境无显式 probing cost；probing cost tradeoff 是离线效用分析。
- Channel estimation error 实验只让决策 preview 使用估计信道，实际执行仍使用真实信道。
- Bandit feedback stress sweep 中策略只能看到 noisy aggregate probe feedback；oracle 只作为离线诊断上界。
- Learned Feedback Probing 中训练标签可来自离线 full-oracle preview，但评估策略不能访问完整 CSI 或 node-level 调度结果，只能看历史 aggregate probe feedback。

加入 noisy features 后，Feature Argmax/PowerTie 会出现明显时延退化；但现有 Codebook-Aware SAC 零样本并不占优。加入 partial probing 和 probing cost 后，Rotating Grid 这类简单非学习 probing 规则已经很强；当前低维 learned probing 也没有超过它。加入当前等效信道估计误差后，Estimated Greedy/Rotating Grid 仍保持接近满覆盖。加入 bandit feedback stress 后，问题更贴近有限 CSI 的研究目标，但简单 Rotating Probe 仍是非常强的规则基线。新增 Learned Feedback Probing 进一步表明，低维历史特征 + 离线监督 MLP 仍不足以超过 rotating baseline。若继续引入更复杂学习、多目标约束或更真实的信道失配，学习策略必须超过这些新规则基线才有研究价值。

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

1. Multi-objective reward / constraint：显式优化 coverage + latency + power/MSE，看学习策略是否能在能耗或风险约束下超过 Feature Argmax PowerTie / Rotating Grid。
2. Bandit-feedback learned probing：在 `short_slots` / `compound_hard` stress 场景下，学习 probe schedule 或 feedback-conditioned IRS 选择，目标是超过 `Rotating Feedback Probe B=1`。
3. 更真实的信道失配：把估计误差推进到执行阶段，或加入延迟、偏置、相关误差，而不是只在决策 preview 中加独立噪声。
4. More informative learned probing：若继续学习，需要加入 probe history、少量实时测量或不确定性/信息增益目标，而不是只用 7 维基础状态。
5. Feature ablation：去掉 16 维 codebook features 后重新评估 Codebook-Aware SAC / no-feature selector。
6. 整理论文图表：把主对比、runtime benchmark、参数扫描、noisy feature、noisy imitation、partial probing、learned probing、probing cost、channel estimation 和 bandit feedback stress 结果压缩成最终论文表格与图。

参数泛化、能耗 tie-break、正式主对比、runtime benchmark、noisy feature sweep、noise-aware imitation、partial probing sweep、learned probing、probing cost tradeoff、channel estimation error sweep 和 bandit feedback stress pilot 已经完成。下一步若继续保留学习方向，应优先转向 bandit-feedback learned probing、多目标约束、更真实的执行阶段信道失配或更高信息量的主动 probing，而不是 plain noisy Greedy-index imitation、当前低维 MLP 或仅决策 preview 误差。
