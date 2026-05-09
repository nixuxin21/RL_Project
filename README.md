# IRS-Assisted MS-AirComp RL

本项目研究 IRS 辅助的多时隙空中计算调度。核心环境使用 Gymnasium 实现，训练侧使用 Stable-Baselines3 SAC。默认任务是在 Rayleigh 衰落信道下，让 50 个节点尽量在 10 个时隙内完成调度，同时控制发射功率和 AirComp 理论 MSE。

完整实验结论和后续研究路线见 `EXPERIMENT_REPORT.md`。

## 项目结构

- `EXPERIMENT_REPORT.md`: 当前主实验、动作诊断、imitation、参数扫描结果和下一阶段研究路线。
- `test_env.py`: 自定义环境 `MSAirCompEnv`，包含信道生成、IRS DFT 码本、动作解码、调度判定、奖励函数和评估用 preview API。
- `train_agent.py`: 训练完整 SAC，动作是 `[g_th, alpha_th, irs_codebook]`。
- `train_codebook_aware_agent.py`: 训练 IRS-only SAC selector，固定 `g_th/alpha_th`，只学习 IRS 码本选择，可使用 16 维 codebook quality features。
- `evaluate_agent.py`: 单回合详细评估完整 SAC。
- `evaluate_batch.py`: 多回合 Monte Carlo 评估完整 SAC。
- `evaluate_random_irs_baseline.py`: 随机 IRS 相位 baseline。
- `evaluate_policy_comparison.py`: 共享随机种子的策略对比脚本，覆盖 SAC、SAC Fixed g/a、Codebook-Aware SAC、Feature Argmax IRS、Feature Argmax PowerTie IRS、Greedy IRS、Random IRS、Fixed IRS、No IRS 等策略。
- `evaluate_parameter_sweep.py`: 对 `K/N/M/C` 做一因子参数扫描，比较 Feature Argmax、Feature Argmax PowerTie、Greedy、Random 和 No IRS。
- `benchmark_policy_runtime.py`: 运行时间 benchmark，统计 Feature Argmax、Feature Argmax PowerTie、Greedy、SAC 和 Codebook-Aware SAC 的决策耗时、环境 step 耗时和 preview 次数。
- `Makefile`: 常用 smoke test、主对比、runtime、参数扫描和动作诊断命令入口。
- `tests/smoke_checks.py`: 默认场景的轻量正确性测试，覆盖 preview 无副作用、codebook features 一致性、PowerTie 与 Greedy tie-break 一致性。
- `results/`: 已生成实验结果。关键 summary CSV 和图可以版本化，逐 episode/step 明细默认只本地保留。
- `rl_models/`: 模型 checkpoint 和 VecNormalize 统计文件。
- `rl_logs/`: TensorBoard 训练日志。

## 环境设定

默认环境参数：

- 节点数 `K=50`
- 时隙数 `N=10`
- IRS 单元数 `M=64`
- IRS DFT 码本数 `C=16`
- 噪声方差 `noise_var=1e-9`
- 节点最大发射功率 `P_max=1.0 W`

基础观测维度是 7。开启 codebook features 后，观测维度是 `7 + C = 23`。完整 SAC 动作维度是 3；IRS-only selector 通过 wrapper 将动作压缩为 1 维。

## 依赖

轻量依赖见 `requirements.txt`。当前本机虚拟环境的完整版本锁定见 `requirements-lock.txt`。

当前 `.venv` 关键版本：

- Python 3.13.2
- Stable-Baselines3 2.8.0
- PyTorch 2.11.0
- Gymnasium 1.3.0
- NumPy 2.4.4

注意：`rl_models/sac_final_model_v3.zip` 的模型元数据记录训练环境为 Python 3.10.20、Gymnasium 1.2.3、NumPy 2.2.6。复现实验时应记录版本差异。

## 训练命令

训练完整 SAC：

```bash
./.venv/bin/python train_agent.py
```

训练 codebook-aware IRS selector：

```bash
./.venv/bin/python train_codebook_aware_agent.py \
  --total-timesteps 200000 \
  --num-envs 8
```

训练不使用 codebook quality features 的 IRS selector：

```bash
./.venv/bin/python train_codebook_aware_agent.py \
  --disable-codebook-features \
  --total-timesteps 200000 \
  --num-envs 8
```

## 评估命令

单回合完整 SAC 评估：

```bash
./.venv/bin/python evaluate_agent.py
```

完整策略对比，推荐使用多 seed。当前脚本默认包含 `Feature Argmax PowerTie IRS`：

```bash
./.venv/bin/python evaluate_policy_comparison.py \
  --episodes 1000 \
  --num-seeds 5 \
  --include-codebook-aware-sac
```

如果不传 `--output` 和 `--csv-output`，脚本会按参数自动生成文件名，例如：

```text
results/policy_comparison/policy_comparison_results_ep1000_runs5_seed2026_featargmax_powertie_cbsac.png
results/policy_comparison/policy_comparison_summary_ep1000_runs5_seed2026_featargmax_powertie_cbsac.csv
```

快速 smoke test：

```bash
./.venv/bin/python evaluate_policy_comparison.py \
  --episodes 2 \
  --num-seeds 2 \
  --include-codebook-aware-sac \
  --output /tmp/policy_comparison_smoke.png \
  --csv-output /tmp/policy_comparison_smoke.csv
```

## 动作诊断

用于解释完整 SAC、Codebook-Aware SAC 和 Greedy IRS 的差距：

```bash
./.venv/bin/python diagnose_policy_actions.py --episodes 1000
```

默认输出会带参数后缀，例如：

```text
results/action_diagnostics/action_diagnostics_ep1000_runs1_seed2026_summary.csv
results/action_diagnostics/action_diagnostics_ep1000_runs1_seed2026_slot_stats.csv
results/action_diagnostics/action_diagnostics_ep1000_runs1_seed2026_episodes.csv
results/action_diagnostics/action_diagnostics_ep1000_runs1_seed2026_steps.csv
results/action_diagnostics/action_diagnostics_ep1000_runs1_seed2026_irs_hist.png
results/action_diagnostics/action_diagnostics_ep1000_runs1_seed2026_latency.png
results/action_diagnostics/action_diagnostics_ep1000_runs1_seed2026_slot_curves.png
```

## Greedy Imitation

生成 greedy IRS 标签数据，训练监督式 IRS selector，并同时评估一个直接选择最大 codebook feature 的规则基线：

```bash
./.venv/bin/python train_greedy_imitation_selector.py
```

默认输出会带训练和评估参数后缀，例如：

```text
results/imitation/greedy_imitation_train5000_eval1000_seed2026_classifier.pt
results/imitation/greedy_imitation_train5000_eval1000_seed2026_train_history.csv
results/imitation/greedy_imitation_train5000_eval1000_seed2026_eval_summary.csv
results/imitation/greedy_imitation_train5000_eval1000_seed2026_eval_slot_stats.csv
results/imitation/greedy_imitation_train5000_eval1000_seed2026_eval_episodes.csv
results/imitation/greedy_imitation_train5000_eval1000_seed2026_eval_steps.csv
results/imitation/greedy_imitation_train5000_eval1000_seed2026_confusion.png
results/imitation/greedy_imitation_train5000_eval1000_seed2026_latency.png
results/imitation/greedy_imitation_train5000_eval1000_seed2026_slot_curves.png
```

## 参数扫描

对节点数、时隙数、IRS 单元数和码本大小做一因子扫描：

```bash
./.venv/bin/python evaluate_parameter_sweep.py
```

默认设置为 `episodes=300, seed=2026`，扫描：

```text
K in {30, 50, 80}
N in {5, 10, 15}
M in {32, 64, 128}
C in {8, 16, 32}
```

默认输出：

```text
results/parameter_sweep/parameter_sweep_ep300_seed2026_summary.csv
results/parameter_sweep/parameter_sweep_ep300_seed2026_episodes.csv
results/parameter_sweep/parameter_sweep_ep300_seed2026_slot_stats.csv
results/parameter_sweep/parameter_sweep_ep300_seed2026_success.png
results/parameter_sweep/parameter_sweep_ep300_seed2026_latency.png
results/parameter_sweep/parameter_sweep_ep300_seed2026_energy.png
```

summary CSV 会额外记录 `decision_preview_calls_per_slot_mean` 和 `tie_candidates_mean`。这里的 preview 次数表示在已获得 codebook features 后，策略决策阶段额外调用 `preview_codebook_index` 的次数；Greedy IRS 则需要每个决策时隙 preview 全部 `C` 个候选。

## 运行时间 Benchmark

统计规则策略和学习策略的平均决策耗时、P50/P95、环境 step 耗时、episode wall time 和 preview 次数：

```bash
./.venv/bin/python benchmark_policy_runtime.py \
  --episodes 200
```

快速 smoke test：

```bash
./.venv/bin/python benchmark_policy_runtime.py \
  --episodes 2 \
  --skip-sac \
  --skip-codebook-aware-sac \
  --output /tmp/runtime_benchmark_smoke.csv
```

注意：`decision_time_*` 只统计策略从当前观测选择动作的时间；`env_step_time_*` 包含环境执行动作并生成下一观测的时间。对 feature-based 策略，当前实现的 `env.step()` 会生成下一时隙的 codebook features，因此 step time 会反映这部分实现成本。`decision_preview_calls_per_slot_mean` 仍按项目约定，只统计 features 已经可用后，决策阶段额外调用 preview 的次数。

## 复现入口

常用命令已固化到 `Makefile`：

```bash
make test
make smoke
make policy-comparison
make runtime
make parameter-sweep
make action-diagnostics
```

`make test` 会运行轻量正确性检查；`make smoke` 会额外跑极小规模策略对比和 runtime benchmark，并把临时文件写到 `/tmp`。

## 当前结论边界

当前结论成立于以下默认场景：

- 完整 codebook quality features 可观测。
- `g_th=0.001`、`alpha_th=0.05` 固定。
- Rayleigh fading 信道、DFT IRS codebook。
- 没有显式 probing cost；preview 次数只作为决策阶段复杂度代理。

如果加入 noisy/partial features、probing cost、信道估计误差或多目标约束，结论需要重新评估。

## 当前基线结果

当前推荐引用 `results/policy_comparison/policy_comparison_summary_ep1000_runs5_seed2026_featargmax_powertie_cbsac.csv`，对应 `episodes=1000, seed=2026, num_seeds=5, --include-codebook-aware-sac`。主要结论：

- Feature Argmax PowerTie IRS: `49.9946/50` 平均成功节点，`99.48%` 完美覆盖率，平均 `2.5496` 个时隙，总能耗 `15.9779`，决策阶段额外 preview `2.3225` 次/时隙。
- Greedy IRS: `49.9946/50` 平均成功节点，`99.48%` 完美覆盖率，平均 `2.5496` 个时隙，总能耗 `15.9779`，需要 preview `16` 次/时隙。
- Feature Argmax IRS: `49.9946/50` 平均成功节点，`99.48%` 完美覆盖率，平均 `2.5404` 个时隙，总能耗 `16.5734`。
- Codebook-Aware SAC: `49.9058/50` 平均成功节点，`91.5%` 完美覆盖率，平均 `5.4206` 个时隙。
- Random IRS: `49.9654/50` 平均成功节点，`96.66%` 完美覆盖率，平均 `5.1460` 个时隙。
- SAC: `48.4394/50` 平均成功节点，`12.0%` 完美覆盖率，平均 `9.7784` 个时隙。
- SAC Fixed g/a: `48.9152/50` 平均成功节点，`32.44%` 完美覆盖率，优于完整 SAC。
- Fixed IRS 和 No IRS 明显较弱，约 `39-41/50`。

## 当前诊断结论

当前推荐引用 `results/action_diagnostics/action_diagnostics_ep1000_runs1_seed2026_summary.csv` 和 `results/action_diagnostics/action_diagnostics_ep1000_runs1_seed2026_slot_stats.csv`。主要结论：

- 完整 SAC 的 `g_th` 均值为 `0.01313`，`alpha_th` 均值为 `0.06479`，明显高于固定实验参数 `g_th=0.001, alpha_th=0.05`，且 IRS 选择集中在 index 7/8。
- 完整 SAC 与当前状态局部 greedy IRS 的 index 匹配率只有 `3.55%`，平均每步少调度 `2.02` 个节点。
- Codebook-Aware SAC 固定了 `g_th/alpha_th`，平均成功节点达到 `49.899/50`，但和局部 greedy IRS 的 index 匹配率只有 `7.16%`，平均每步少调度 `1.83` 个节点。
- Greedy IRS 第 1 槽平均调度 `44.86` 个节点，Codebook-Aware SAC 第 1 槽平均调度 `40.98` 个节点。Codebook-Aware SAC 的主要差距是前几个时隙没有像 greedy 那样快速清空剩余节点。

## 当前 Imitation 结论

当前推荐引用 `results/imitation/greedy_imitation_train5000_eval1000_seed2026_eval_summary.csv` 和 `results/imitation/greedy_imitation_train5000_eval1000_seed2026_eval_slot_stats.csv`。主要结论：

- 监督 imitation selector 的验证标签准确率约 `58%`，但评估表现达到 `49.996/50`，完美覆盖率 `99.6%`，平均 `2.537` 个时隙。
- `Feature Argmax` 直接选择 16 维 codebook quality features 中最大的索引，评估表现达到 `49.997/50`，完美覆盖率 `99.7%`，平均 `2.529` 个时隙。
- `Feature Argmax` 与 Greedy IRS 的 index 匹配率只有 `56.19%`，但 `oracle_tx_gap_mean=0.0`，说明很多不同码本索引在调度节点数上是等价的；Greedy 的功率 tie-break 对覆盖率影响很小。
- 当前最强的非学习规则已经几乎等于 Greedy IRS，因此 Codebook-Aware SAC 落后的直接原因不是特征不足，而是 RL 训练没有学到“优先选择当前 codebook feature 最大值”的简单策略。

## 当前参数扫描结论

当前推荐引用 `results/parameter_sweep/parameter_sweep_ep300_seed2026_summary.csv`，对应默认 `K/N/M/C` 一因子扫描。主要结论：

- 在 `K=30/50/80`、`N=5/10/15`、`M=32/64/128` 的扫描中，Feature Argmax IRS 与 Greedy IRS 的成功率和时延几乎一致，均明显优于 Random IRS 和 No IRS。
- `C=8` 时 Feature Argmax 和 Greedy 的完美覆盖率都降到 `85.0%`，而 `C=16/32` 恢复到约 `99.3%-100%`，说明码本分辨率是主要限制。
- Feature Argmax PowerTie 在所有扫描配置下复现 Greedy IRS 的覆盖率、时延和能耗，说明 Greedy 的额外价值主要就是 max-count 候选内的功率 tie-break。
- Feature Argmax PowerTie 每个决策时隙通常只额外 preview `2.1-3.5` 个并列候选；Greedy IRS 需要 preview 全部 `C` 个候选。
- Random IRS 有时也接近满成功率，但平均需要约 `5` 个时隙；Feature Argmax/Greedy 通常只需约 `2-3` 个时隙。

运行时间结果推荐引用 `results/runtime/runtime_benchmark_ep200_seed2026.csv`。其中 PowerTie 的平均决策耗时约 `0.0925 ms`，Greedy 约 `0.4345 ms`；PowerTie 与 Greedy 能耗相同，但额外 preview 次数约 `2.54` vs `16.00`。

下一步研究重点应转向 noisy/partial feature 或多目标约束场景，让 RL 问题真正有学习价值。
