# IRS-Assisted MS-AirComp RL

本项目研究 IRS 辅助的多时隙空中计算调度。核心环境使用 Gymnasium 实现，训练侧使用 Stable-Baselines3 SAC。默认任务是在 Rayleigh 衰落信道下，让 50 个节点尽量在 10 个时隙内完成调度，同时控制发射功率和 AirComp 理论 MSE。

完整实验结论和后续研究路线见 `EXPERIMENT_REPORT.md`。

## 项目结构

- `EXPERIMENT_REPORT.md`: 当前主实验、动作诊断、imitation、参数扫描结果和下一阶段研究路线。
- `test_env.py`: 自定义环境 `MSAirCompEnv`，包含信道生成、IRS DFT 码本、动作解码、调度判定、奖励函数和评估用 preview API。
- `train_agent.py`: 训练完整 SAC，动作是 `[g_th, alpha_th, irs_codebook]`。
- `train_codebook_aware_agent.py`: 训练 IRS-only SAC selector，固定 `g_th/alpha_th`，只学习 IRS 码本选择，可使用 16 维 codebook quality features。
- `train_bandit_feedback_selector.py`: 训练 feedback-conditioned IRS probing selector；训练可使用离线 oracle 标签，评估时只能看基础状态、历史 noisy aggregate feedback 和自身 probing 历史。
- `train_temporal_deviation_selector.py`: 训练 temporal AR(1) stale-CSI 下的 learned temporal deviation selector；学习相对 rotating probe window 的 offset，评估时只使用可观测状态和历史执行反馈。
- `evaluate_agent.py`: 单回合详细评估完整 SAC。
- `evaluate_batch.py`: 多回合 Monte Carlo 评估完整 SAC。
- `evaluate_random_irs_baseline.py`: 随机 IRS 相位 baseline。
- `evaluate_policy_comparison.py`: 共享随机种子的策略对比脚本，覆盖 SAC、SAC Fixed g/a、Codebook-Aware SAC、Feature Argmax IRS、Feature Argmax PowerTie IRS、Greedy IRS、Random IRS、Fixed IRS、No IRS 等策略。
- `evaluate_parameter_sweep.py`: 对 `K/N/M/C` 做一因子参数扫描，比较 Feature Argmax、Feature Argmax PowerTie、Greedy、Random 和 No IRS。
- `evaluate_partial_probing_sweep.py`: 在每时隙只能 preview 少量 IRS 码本的设定下，比较随机 probing、固定网格、轮换网格、局部邻域和混合 probing。
- `evaluate_channel_estimation_error_sweep.py`: 在决策 preview 使用带误差的等效信道、执行仍使用真实信道的设定下，扫描 IRS 选择规则对信道估计误差的鲁棒性。
- `evaluate_limited_csi_ms_aircomp.py`: 有限 CSI 评估框架，将策略基于估计/partial probing 的节点邀请集合与真实信道执行成功集合分离。
- `evaluate_execution_channel_mismatch.py`: 执行阶段信道失配评估框架，策略先基于 stale/estimated CSI 邀请节点，实际 AirComp slot 再用漂移后的执行信道验证成功。
- `evaluate_bandit_feedback_ms_aircomp.py`: 更严格的有限观测评估框架，策略不能读取每个码本的完整 CSI 或节点级可调度 mask，只能通过少量 IRS probe 得到 noisy aggregate feedback，再在线选择码本。
- `evaluate_bandit_feedback_stress_sweep.py`: 在 bandit feedback 框架上扫描更困难的物理场景，并加入 slot/probe cost 的 node-equivalent utility。
- `evaluate_adaptive_feedback_probing.py`: 评估非学习版 Adaptive Rotating Backup；默认先执行 `Rotating B=1`，仅当单次 noisy feedback 低于剩余 deadline 所需速度时额外 probe 一个 backup 码本。
- `benchmark_policy_runtime.py`: 运行时间 benchmark，统计 Feature Argmax、Feature Argmax PowerTie、Greedy、SAC 和 Codebook-Aware SAC 的决策耗时、环境 step 耗时和 preview 次数。
- `experiments/archive/`: 早期探索实验归档，包括 noisy feature sweep、learned probing selector 和 probing cost tradeoff；这些脚本仍可通过 Makefile 运行，但不是当前主线。
- `Makefile`: 常用 smoke test、主对比、runtime、参数扫描、noisy feature sweep、partial probing sweep、learned/adaptive probing、probing cost tradeoff、channel estimation sweep、limited CSI sweep、execution mismatch sweep 和动作诊断命令入口。
- `tests/smoke_checks.py`: 默认场景的轻量正确性测试，覆盖 preview 无副作用、codebook features 一致性、PowerTie 与 Greedy tie-break 一致性、limited CSI 零误差一致性和“未邀请节点不自动成功”。
- `results/`: 已生成实验结果，默认按本地生成物处理；需要发布的关键 CSV/图可手动添加。
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

也可以训练 noise-aware imitation selector。下面命令在训练/验证观测的 codebook features 上加入 `noise_std=0.2`，再扫描多个评估噪声强度：

```bash
./.venv/bin/python train_greedy_imitation_selector.py \
  --train-episodes 5000 \
  --val-episodes 1000 \
  --eval-episodes 1000 \
  --codebook-feature-noise-std 0.2 \
  --eval-noise-std-values 0,0.02,0.05,0.1,0.15,0.2,0.3
```

对应输出前缀会追加 feature noise 后缀，例如：

```text
results/imitation/greedy_imitation_train5000_eval1000_seed2026_featnoise0p2_train_history.csv
results/imitation/greedy_imitation_train5000_eval1000_seed2026_featnoise0p2_eval_summary.csv
results/imitation/greedy_imitation_train5000_eval1000_seed2026_featnoise0p2_eval_slot_stats.csv
results/imitation/greedy_imitation_train5000_eval1000_seed2026_featnoise0p2_confusion.png
results/imitation/greedy_imitation_train5000_eval1000_seed2026_featnoise0p2_eval_noise_sweep.png
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

## Noisy Feature Sweep

给观测中的 C 维 codebook quality features 加高斯噪声，测试规则策略在 feature 不再精确时的退化速度：

```bash
./.venv/bin/python experiments/archive/evaluate_noisy_feature_sweep.py \
  --episodes 300 \
  --noise-std-values 0,0.05,0.1,0.2,0.3
```

如需同时评估已有 Codebook-Aware SAC 在 noisy features 下的零样本鲁棒性：

```bash
./.venv/bin/python experiments/archive/evaluate_noisy_feature_sweep.py \
  --episodes 300 \
  --include-codebook-aware-sac
```

默认输出：

```text
results/noisy_features/noisy_feature_sweep_ep300_runs1_seed2026.csv
results/noisy_features/noisy_feature_sweep_ep300_runs1_seed2026.png
```

也可以直接在主对比脚本中指定单个噪声强度：

```bash
./.venv/bin/python evaluate_policy_comparison.py \
  --episodes 1000 \
  --num-seeds 5 \
  --include-codebook-aware-sac \
  --codebook-feature-noise-std 0.1
```

## Partial Probing Sweep

限制每个决策时隙只能精确 preview `B` 个 IRS 码本，然后在已 preview 候选中按 Greedy 的调度节点数、平均功率和剩余增益排序选择 IRS：

```bash
./.venv/bin/python evaluate_partial_probing_sweep.py \
  --episodes 300 \
  --probe-budgets 1,2,4,8
```

默认比较：

```text
Random Probe
Fixed Grid Probe
Rotating Grid Probe
Local Probe
Hybrid Local+Grid Probe
Greedy Full Preview
```

默认输出：

```text
results/partial_probing/partial_probing_sweep_ep300_runs1_seed2026_b1-2-4-8.csv
results/partial_probing/partial_probing_sweep_ep300_runs1_seed2026_b1-2-4-8.png
```

CSV 中的 `decision_preview_calls_per_slot_mean` 是策略实际使用的 probe budget；`oracle_match_rate` 和 `oracle_tx_gap_mean` 使用 full Greedy 做离线诊断，不计入策略 preview budget。

## Learned Probing

训练一个低维状态的 learned probing selector。模型不观察完整 codebook quality features，只输入基础 7 维状态和上一 IRS index 编码，预测每个码本的可调度比例；评估时只 preview 预测 top-B 候选：

```bash
./.venv/bin/python experiments/archive/train_learned_probing_selector.py \
  --train-episodes 5000 \
  --val-episodes 1000 \
  --eval-episodes 1000 \
  --num-eval-seeds 5 \
  --probe-budgets 1,2,4,8
```

默认输出：

```text
results/learned_probing/learned_probing_train5000_eval1000_seed2026_rotatinggrid_b4_evalb1-2-4-8_train_history.csv
results/learned_probing/learned_probing_train5000_eval1000_seed2026_rotatinggrid_b4_evalb1-2-4-8_val_topk.csv
results/learned_probing/learned_probing_train5000_eval1000_seed2026_rotatinggrid_b4_evalb1-2-4-8_eval_summary.csv
results/learned_probing/learned_probing_train5000_eval1000_seed2026_rotatinggrid_b4_evalb1-2-4-8_eval.png
```

模型 checkpoint `*_regressor.pt` 默认只本地保留。

## Probing Cost Tradeoff

把每个 episode 的总 preview 次数纳入 node-equivalent utility：

```text
utility = success_mean - slot_cost * slots_mean - preview_cost * total_preview_calls_mean
```

运行成本网格分析：

```bash
./.venv/bin/python experiments/archive/evaluate_probing_cost_tradeoff.py
```

默认读取正式 partial probing 和 learned probing summary，输出：

```text
results/probing_cost/probing_cost_tradeoff_slot0-0p05-0p1-0p2_preview0-0p0005-0p001-0p002-0p005-0p01-0p02-0p05_candidates.csv
results/probing_cost/probing_cost_tradeoff_slot0-0p05-0p1-0p2_preview0-0p0005-0p001-0p002-0p005-0p01-0p02-0p05_utilities.csv
results/probing_cost/probing_cost_tradeoff_slot0-0p05-0p1-0p2_preview0-0p0005-0p001-0p002-0p005-0p01-0p02-0p05_winners.csv
results/probing_cost/probing_cost_tradeoff_slot0-0p05-0p1-0p2_preview0-0p0005-0p001-0p002-0p005-0p01-0p02-0p05_frontier.png
results/probing_cost/probing_cost_tradeoff_slot0-0p05-0p1-0p2_preview0-0p0005-0p001-0p002-0p005-0p01-0p02-0p05_winners.png
```

## Channel Estimation Error Sweep

在该实验中，环境执行仍使用真实等效信道；策略决策时的 preview 使用加入复高斯估计误差的等效信道：

```bash
./.venv/bin/python evaluate_channel_estimation_error_sweep.py \
  --episodes 1000 \
  --num-seeds 5 \
  --error-std-values 0,0.02,0.05,0.1,0.2,0.3
```

默认输出：

```text
results/channel_estimation/channel_estimation_error_sweep_ep1000_runs5_seed2026_err0-0p02-0p05-0p1-0p2-0p3.csv
results/channel_estimation/channel_estimation_error_sweep_ep1000_runs5_seed2026_err0-0p02-0p05-0p1-0p2-0p3.png
```

`Exact Greedy Full Preview` 作为真实信道 oracle；`Estimated Greedy Full Preview`、`Estimated Rotating Grid B=4/B=8` 和 `Estimated Count Argmax` 只在决策 preview 阶段看到估计信道。`oracle_match_rate` 与 `oracle_tx_gap_mean` 是离线诊断指标，不改变策略预算。

## Limited CSI MS-AirComp

该实验不再让环境自动调度所有真实可行节点。策略只能基于有限 probing/估计信道选择 IRS，并邀请估计可发送的节点；执行阶段再用真实信道验证这些被邀请节点是否真的成功：

```bash
./.venv/bin/python evaluate_limited_csi_ms_aircomp.py \
  --episodes 300 \
  --probe-budgets 2,4,8 \
  --error-std-values 0,0.05,0.1,0.2,0.3 \
  --robust-gain-margins 1.25 \
  --robust-power-margins 0.9 \
  --risk-weights 0.5 \
  --risk-power-weights 0.1 \
  --risk-invite-thresholds 0.5 \
  --adaptive-risk-base-weights 0.5
```

默认比较 `No IRS`、`Fixed IRS`、`Exact Greedy Full CSI`、`Estimated Greedy Full Preview`、`Estimated Random Probe`、`Estimated Rotating Grid`、`Robust Rotating Grid`、`Risk-Aware Rotating Grid` 和 `Adaptive Risk-Aware Rotating Grid`。`Risk-Aware` 策略会根据估计信道距离可行阈值的距离生成节点成功可靠度，优先选择期望成功数高、边界风险低、平均功率低的 IRS 候选；默认 `risk-invite-threshold=0.5` 主要体现风险感知 IRS 选择，调高该门限则会更保守地过滤边界节点。`Adaptive Risk-Aware` 会在 CSI error 较高时提高风险权重，并在接近 deadline 或剩余节点 backlog 较高时降低风险权重。新增指标包括 `scheduled_nodes_mean`、`failed_nodes_mean`、`execution_failure_rate`、`missed_opportunity_rate`、`decision_preview_calls_per_slot_mean`、`effective_risk_weight_mean` 和 `oracle_tx_gap_mean`。

## Execution Channel Mismatch

该实验把“有限 CSI”推进到更真实的执行阶段失配：策略决策时只能看到 stale/estimated CSI，并据此选择 IRS 和邀请节点；真正执行 AirComp slot 时，等效信道会再发生一次 policy-independent drift。只有“被邀请且在执行信道下仍满足门限”的节点才算成功。`Execution Oracle Full CSI` 只作为离线上界，表示如果调度器能提前知道执行信道可达到的最好结果。

短时隙 pilot：

```bash
./.venv/bin/python evaluate_execution_channel_mismatch.py \
  --episodes 300 \
  --num-seeds 3 \
  --num-slots 5 \
  --decision-error-std-values 0 \
  --execution-error-std-values 0,0.1,0.2,0.3,0.5 \
  --probe-budgets 1,4 \
  --policies execution_oracle,exact_greedy,estimated_greedy,rotating,robust_rotating,risk_rotating,adaptive_risk_rotating \
  --output-prefix results/execution_mismatch/execution_mismatch_short_slots_pilot_ep300_runs3_execerr0-0p1-0p2-0p3-0p5
```

默认输出：

```text
results/execution_mismatch/execution_mismatch_*.csv
results/execution_mismatch/execution_mismatch_*_decerr*.png
```

当前 pilot 结论：执行阶段失配主要表现为失败邀请、missed opportunities 和 oracle gap 增大，而不是 success mean 立刻崩溃。`execerr=0.5` 时，`Exact Greedy Full CSI` 仍有 `49.972/50` 成功节点，但失败邀请均值升到 `5.39`、oracle gap 升到 `2.403`；`Estimated Rotating Grid B=1` 降到 `49.012/50`、完美覆盖率 `38.56%`、平均 `4.771` slots。该结果说明“只在决策 preview 加估计误差”低估了真实系统风险，后续鲁棒策略应直接建模执行漂移统计量。

脚本也包含 `execution_risk_rotating` 和 `adaptive_execution_risk_rotating`，它们不读取真实执行信道，只把 `sqrt(decision_error_std^2 + execution_error_std^2)` 用作邀请可靠度估计：

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
  --risk-invite-thresholds 0.5,0.55 \
  --adaptive-risk-base-weights 0.1,0.5 \
  --output-prefix results/execution_mismatch/execution_risk_pilot_ep300_runs3_execerr0p2-0p5_rw0p1-0p5_rt0p5-0p55
```

当前结果是负的：`rt=0.55` 能减少失败邀请，但会显著增加 missed opportunities 和 oracle gap。例如 `execerr=0.5, B=4` 下，普通 rotating 为 `49.849/50`、`5.43` failed、`5.69` missed、gap `2.841`；`Execution-Risk B=4 rw=0.1 rt=0.55` 为 `49.762/50`、`4.64` failed、`9.66` missed、gap `3.738`。这说明固定保守邀请阈值不是足够好的鲁棒调度方法。

随后新增 `opportunity_execution_risk_rotating`，把邀请决策改成 expected utility：false accept 对应失败邀请代价，false reject 对应 missed opportunity 代价，并用 deadline/backlog 生成当前 slot 的 urgency：

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

结果仍是中性到负面：低失败代价配置基本退化为普通 rotating；高失败代价能减少 failed invites，但会牺牲 success、perfect rate 和 oracle gap。例如 `execerr=0.5, B=4` 下，最好成功率的 opportunity 配置为 `49.823/50`、`5.07` failed、`6.37` missed、gap `2.949`，仍弱于 rotating 的 `49.849/50`、`5.43` failed、`5.69` missed、gap `2.841`。这说明“只用可靠度加动态机会成本过滤邀请”仍不足以超过 rotating baseline，下一步应转向更真实的信道失配、多目标约束或带置信/重复确认的 feedback probing。

脚本还支持时间相关 CSI delay：`temporal_ar1` 先生成 AR(1) 物理信道序列，调度器只能看到 delay slots 之前的 stale CSI，执行则使用当前信道。`ar1_predict_rotating` 是一个简单均值预测 baseline：用 `rho^delay` 缩放 delayed physical channels 后再做 rotating。

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

当前 temporal pilot 说明：时间相关 stale CSI 能形成更真实的鲁棒性评估，但普通 rotating 仍是很强基线。`rho=0.9, delay=3` 下，`Execution Oracle` 为 `50/50`，full stale greedy 为 `49.936/50`，`Rotating B=1` 为 `49.032/50`、gap `2.642`，`Rotating B=4` 为 `49.817/50`、gap `1.215`。朴素 `AR1-Predict B=4` 只有 `46.480/50`、gap `6.194`，说明直接按均值缩放信道会过度保守；后续若做预测，需要预测可调度概率/分位数，而不是只用复信道均值。

随后新增 `temporal_reliability_rotating`：仍只在 rotating probe budget 内选候选，但用 delayed CSI 和 AR(1) error std 估计当前 slot 的可调度概率，并用 expected success、risk mass 和 quantile lower-bound count 对候选排序；它不使用真实当前 CSI，也不硬过滤邀请节点。

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

该 pilot 也是负/中性结果：`rw=0` 基本退化为 stale rotating，`rw=0.5/1` 会更保守但通常增加 slots 和 oracle gap。比如 `rho=0.9, delay=3, B=4` 下，普通 rotating 为 `49.817/50`、perfect `84.11%`、slots `3.924`、gap `1.215`；最佳 temporal-reliability 配置为 `49.814/50`、perfect `84.00%`、slots `3.916`、gap `1.226`。`rho=0.98, delay=3` 下 rotating 为 `49.871/50`、gap `0.967`，temporal-reliability 最佳为 `49.862/50`、gap `1.002`。这说明手工可靠度排序还不足以超过 strong rotating baseline；若继续该方向，应让策略学习何时偏离 rotating probe 顺序，或引入真正的多目标约束/反馈确认。

为确认“学习偏离 rotating probe 顺序”是否有上界空间，新增 `temporal_deviation_oracle` 诊断。它不可部署：每个 slot 用隐藏 current channel 从全部 codebook 中选出 top-B IRS 候选，但仍只按 stale/estimated CSI 生成邀请 mask，并且 `Preview` 仍按 B 统计。它用于回答：如果 B=4 的 probe 集合选得更聪明，最多能比 rotating 好多少。

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

该诊断给出正向信号：同样 `B=4` 下，deviation oracle 在所有 rho/delay 上都明显优于 rotating，并接近 full stale greedy / execution oracle。例如 `rho=0.9, delay=3` 下，`Rotating B=4` 为 `49.817/50`、perfect `84.11%`、slots `3.924`、gap `1.215`；`Temporal Deviation Oracle B=4` 为 `49.988/50`、perfect `98.78%`、slots `3.117`、gap `0.383`。`rho=0.98, delay=3` 下，rotating gap `0.967`，deviation oracle gap `0.195`。这说明真正值得学的不是候选内部 reliability 重排，而是如何在有限 CSI 下选择/偏离 probe 的 IRS 集合。

随后新增第一版 learned temporal deviation selector。它不直接预测 IRS index，而是预测相对 rotating probe window 的 offset；训练标签来自隐藏 current-channel outcome，评估时只能使用 slot/backlog/rho/delay、历史 probe 统计和上一轮执行反馈：

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

当前结果是中性到负面：validation offset hit rate 只有 `27.17%`，虽然平均 target tx gap 较小 `0.0169`，但闭环评估没有稳定超过 `Rotating B=4`，更没有接近 deviation oracle。例如 `rho=0.9, delay=3` 下，rotating 为 `49.827/50`、perfect `85.00%`、slots `3.940`、gap `1.242`；learned temporal deviation 为 `49.767/50`、perfect `80.00%`、slots `4.013`、gap `1.268`；deviation oracle gap 为 `0.406`。这说明低维历史特征 + 离线 hidden-target regression 还不足以学到可部署的 probe-set deviation，下一步不应继续简单调这个 MLP，而应转向 on-policy/DAgger、带置信反馈的确认机制，或更丰富的候选重排表示。

## Bandit Feedback MS-AirComp

该实验进一步去掉“probe 后可得到节点级 CSI/mask”的假设。每个 probe 只返回该 IRS 码本的 noisy aggregate feedback：预计可发送节点比例和平均功率；策略不能知道具体哪些节点可发送，也不能读取完整 codebook feature。执行阶段仍在真实信道上发生，代表节点基于本地信道自选择是否参与 AirComp。

```bash
./.venv/bin/python evaluate_bandit_feedback_ms_aircomp.py \
  --episodes 300 \
  --num-seeds 3 \
  --probe-budgets 1,2,4,8 \
  --feedback-noise-std-values 0,0.05,0.1,0.2,0.3
```

默认比较 `No IRS`、`Fixed IRS`、`Random IRS`、`Oracle Full Preview`、`Full Noisy Feedback`、`Random Feedback Probe`、`Rotating Feedback Probe`、`UCB Feedback Probe` 和 `Thompson Feedback Probe`。`Oracle Full Preview` 只作为离线上界；`Full Noisy Feedback` 表示探测全部码本但只看 noisy aggregate feedback；`UCB/Thompson` 是在线 bandit probing 基线。主要指标包括 `probe_calls_per_slot_mean`、`oracle_match_rate`、`oracle_tx_gap_mean`、`observed_tx_fraction_mean`、成功节点数、完美覆盖率、时隙数和能耗。

为了避免默认环境过容易，还可以运行 stress sweep。该脚本扫描短时隙、小码本、小 IRS 和组合困难场景，并报告：

```text
utility = success_mean - slot_cost * slots_mean - probe_cost * total_probe_calls_mean
```

推荐先跑中等规模 pilot：

```bash
./.venv/bin/python evaluate_bandit_feedback_stress_sweep.py \
  --episodes 200 \
  --num-seeds 3 \
  --scenarios default,short_slots,small_codebook,compound_hard \
  --feedback-noise-std-values 0.2 \
  --probe-budgets 1,2,4
```

默认输出：

```text
results/bandit_feedback/bandit_feedback_stress_*.csv
results/bandit_feedback/bandit_feedback_stress_*.png
```

## Learned Feedback Probing

在 bandit feedback stress 结果上继续推进学习策略。该脚本训练一个 MLP selector，输入仅包含基础 7 维状态、每个码本的历史 aggregate feedback 均值/次数/recency、上一 probe 的反馈和上一 IRS index。训练阶段可以用 full oracle preview 生成监督标签；评估阶段仍严格限制为 noisy aggregate probe feedback：

```bash
./.venv/bin/python train_bandit_feedback_selector.py \
  --scenario short_slots \
  --train-episodes 5000 \
  --val-episodes 1000 \
  --eval-episodes 1000 \
  --num-eval-seeds 5 \
  --feedback-noise-std-values 0.2 \
  --probe-budgets 1,2,4
```

也可以切到组合困难场景：

```bash
./.venv/bin/python train_bandit_feedback_selector.py \
  --scenario compound_hard \
  --train-episodes 5000 \
  --val-episodes 1000 \
  --eval-episodes 1000 \
  --num-eval-seeds 5 \
  --feedback-noise-std-values 0.5 \
  --probe-budgets 1,2,4
```

默认会和 `Rotating Feedback Probe`、`UCB Feedback Probe`、`Thompson Feedback Probe`、`Full Noisy Feedback` 和 `Oracle Full Preview` 对比。当前判断标准是：学习策略必须在 `short_slots` 或 `compound_hard` 下超过 `Rotating Feedback Probe B=1`，否则只能作为负结果。

## Adaptive Feedback Probing

非学习版主动 probing 诊断。该脚本把 `Rotating Feedback Probe B=1` 作为默认行为，只有当当前 noisy aggregate feedback 低于剩余节点/剩余时隙所要求的完成速度时，才额外 probe 一个 backup codebook。backup 可选 `next`、`least_recent`、`best_history` 或 `hybrid`：

```bash
./.venv/bin/python evaluate_adaptive_feedback_probing.py \
  --scenarios short_slots \
  --episodes 300 \
  --num-seeds 3 \
  --feedback-noise-std-values 0.2 \
  --gate-ratios 0.7,0.9,1.1 \
  --backup-strategies next,least_recent,best_history,hybrid \
  --probe-budgets 1,2
```

组合困难场景：

```bash
./.venv/bin/python evaluate_adaptive_feedback_probing.py \
  --scenarios compound_hard \
  --episodes 300 \
  --num-seeds 3 \
  --feedback-noise-std-values 0.5 \
  --gate-ratios 0.7,0.9,1.1 \
  --backup-strategies next,least_recent,best_history,hybrid \
  --probe-budgets 1,2
```

当前 pilot 结论仍是负结果：adaptive backup 可以超过 `Rotating B=2` 等更高成本 noisy-feedback 策略，但没有超过 `Rotating B=1`。单次 noisy feedback 很容易产生假低反馈，从而过度触发 backup probe。

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
make noisy-feature-sweep
make partial-probing-sweep
make learned-probing
make learned-feedback-probing
make adaptive-feedback-probing
make probing-cost-tradeoff
make channel-estimation-sweep
make limited-csi-sweep
make execution-mismatch-sweep
make bandit-feedback-sweep
make bandit-feedback-stress
make action-diagnostics
```

`make test` 会运行轻量正确性检查；`make smoke` 会额外跑极小规模策略对比和 runtime benchmark，并把临时文件写到 `/tmp`。

## 当前结论边界

当前结论成立于以下默认场景：

- 默认主对比假设完整且精确的 codebook quality features 可观测。
- `g_th=0.001`、`alpha_th=0.05` 固定。
- Rayleigh fading 信道、DFT IRS codebook。
- 默认物理环境没有显式 probing cost；probing cost tradeoff 是基于 summary 的离线效用分析。
- Channel estimation sweep 中只有决策 preview 使用估计信道，实际执行仍使用真实信道。
- Execution channel mismatch sweep 中决策 CSI 和执行信道被显式分开，execution oracle 只作为离线上界。
- Bandit feedback sweep 中策略只能看 noisy aggregate probe feedback；full oracle 指标仅用于离线诊断。
- Bandit feedback stress sweep 使用额外的 node-equivalent utility 来比较低延迟和低 probing 开销，并不改变底层物理执行模型。
- Learned/adaptive feedback probing 都不能访问完整 CSI 或 node-level mask；oracle 只用于训练标签或离线诊断。

Noisy feature sweep、noise-aware imitation、partial probing sweep、learned probing、probing cost tradeoff、channel estimation error sweep、bandit feedback stress、learned feedback probing 和 adaptive feedback probing 都已经完成：feature 噪声会显著拉低完美覆盖率并拉长时延，直接训练 noisy Greedy-index imitation 仍没有超过 Feature Argmax；partial probing 下，简单的 Rotating Grid Probe 已经是很强的新基线；低维状态 learned probing 也没有超过 Rotating Grid；显式 preview cost 下 Rotating Grid 通常比 full Greedy 更优；等效信道估计误差会降低 oracle match 并拉长时延，但在当前误差模型下 Estimated Greedy/Rotating Grid 仍维持接近满覆盖；bandit feedback 下 `Rotating B=1` 很强，离线 MLP 和单次 noisy feedback hard gate 都没有超过它。下一阶段更应转向多目标约束、执行阶段信道失配，或带置信/重复确认的 on-policy feedback probing。

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

Noise-aware imitation 推荐引用 `results/imitation/greedy_imitation_train5000_eval1000_seed2026_featnoise0p2_eval_summary.csv`。主要结论：

- 在 `noise_std=0.2` 的 noisy features 上训练后，验证标签准确率峰值只有 `10.21%`，最终为 `8.54%`。
- 该 selector 在所有评估噪声强度下都没有超过 Feature Argmax；例如 `eval_noise=0.2` 时，Greedy Imitation 为 `49.389/50`、完美覆盖率 `58.4%`、平均 `6.777` 个时隙，而 Feature Argmax 为 `49.905/50`、`91.4%`、`5.409` 个时隙。
- 该模型长期偏向 dominant IRS index `13`，更像带状态微调的固定索引策略。下一步不应继续 plain Greedy-index imitation，而应转向 partial probing 或 learned probing policy。

## 当前 Partial Probing 结论

当前推荐引用 `results/partial_probing/partial_probing_sweep_ep1000_runs5_seed2026_b1-2-4-8.csv`，对应 `episodes=1000, seed=2026, num_seeds=5, --probe-budgets 1,2,4,8`。主要结论：

- `Rotating Grid Probe` 是当前最强非学习 probing baseline。`B=4` 时达到 `49.9946/50`、`99.48%` 完美覆盖率、平均 `3.1598` 个时隙，只用 `4` 次 preview/slot。
- `B=8` 的 Rotating Grid 达到 `2.7694` 个时隙，接近 full Greedy 的 `2.5496`，但 preview 次数只有一半。
- `B=1` 固定网格退化明显，只有 `41.0186/50`、`0.02%` 完美覆盖率；说明固定少量码本不是有效 probing。
- `Local Probe` 也不够稳，`B=2` 只有 `49.18%` 完美覆盖率；上一时隙最优 IRS 的邻域并不可靠。
- 下一步若做学习策略，应学习 probe schedule / candidate selection，并且必须在 `B=2/4` 下超过 Rotating Grid，而不是继续直接预测 Greedy index。

## 当前 Learned Probing 结论

当前推荐引用 `results/learned_probing/learned_probing_train5000_eval1000_seed2026_rotatinggrid_b4_evalb1-2-4-8_eval_summary.csv`。该模型在 rotating-grid `B=4` 轨迹上收集训练状态，用 full preview 的 C 维 tx-count fractions 作为监督目标。主要结论：

- Learned Probe 没有在任何预算下超过 Rotating Grid。`B=4` 时 Learned Probe 为 `49.981/50`、`98.14%` 完美覆盖率、`3.430` slots；Rotating Grid 为 `49.995/50`、`99.48%`、`3.160` slots。
- `B=8` 时 Learned Probe 为 `2.887` slots，仍落后于 Rotating Grid 的 `2.769`，也略落后于 Random Probe 的 `2.859`。
- 验证集 top-k oracle hit rate 随预算上升，但不足以支撑策略胜出：`B=1/2/4/8` 分别为 `22.15%/36.98%/55.57%/78.27%`。
- 结论是低维状态加上一时隙 IRS index 不足以学习出比 deterministic rotating schedule 更强的 probing policy。继续学习方向需要更丰富的观测、显式 probing cost 目标，或把问题改成主动探索/信息增益，而不是再调当前 MLP。

## 当前 Probing Cost 结论

当前推荐引用 `results/probing_cost/probing_cost_tradeoff_slot0-0p05-0p1-0p2_preview0-0p0005-0p001-0p002-0p005-0p01-0p02-0p05_winners.csv`。主要结论：

- 当 `preview_cost=0` 时，full Greedy 因 `2.550` slots 的最低时延胜出。
- 当 preview 有非零成本时，winner 基本切换到 Rotating Grid：例如 `slot_cost=0.1, preview_cost=0.002` 时选 `B=8`，`preview_cost=0.005/0.01` 时选 `B=4`，`preview_cost=0.02/0.05` 时选 `B=2`。
- `slot_cost=0.05, preview_cost=0.001` 已经从 Greedy 切到 `Rotating Grid B=8`，说明 full preview 的额外 18 个左右 preview/episode 只有在 preview 几乎免费时才合理。
- Learned Probe 没有在默认成本网格中成为 winner。
- 论文叙事上应把 Rotating Grid 作为 probing-cost 场景的强规则算法，而不是把 learned probing 作为当前主贡献。

## 当前 Channel Estimation 结论

当前推荐引用 `results/channel_estimation/channel_estimation_error_sweep_ep1000_runs5_seed2026_err0-0p02-0p05-0p1-0p2-0p3.csv`。主要结论：

- `error_std=0.3` 时，`Estimated Greedy Full Preview` 仍达到 `49.9918/50`、`99.22%` 完美覆盖率、`3.3108` slots，但 oracle match 从 `99.94%` 降到 `34.94%`，说明很多估计错误仍落在近等价候选上。
- `Estimated Rotating Grid B=4` 在 `error_std=0.3` 下为 `49.9874/50`、`98.78%`、`3.8162` slots；`B=8` 为 `49.9900/50`、`99.02%`、`3.5034` slots。
- `Estimated Count Argmax` 对误差更敏感，`error_std=0.3` 时降到 `49.8812/50`、`88.24%` 完美覆盖率、`4.5880` slots。
- 当前等效信道误差模型还没有让学习策略自然变得必要；它主要确认 full/partial probing 规则有较强鲁棒性。下一步若继续做鲁棒性，应加入执行阶段信道失配、延迟/偏置估计或多目标约束。

## 当前 Execution Mismatch 结论

当前推荐引用 `results/execution_mismatch/execution_mismatch_short_slots_pilot_ep300_runs3_execerr0-0p1-0p2-0p3-0p5.csv`、`results/execution_mismatch/execution_risk_pilot_ep300_runs3_execerr0p2-0p5_rw0p1-0p5_rt0p5-0p55.csv`、`results/execution_mismatch/opportunity_execution_risk_pilot_ep300_runs3_execerr0p2-0p5_fc0p5-1-2_mc1-2.csv`、`results/execution_mismatch/temporal_ar1_pilot_ep300_runs3_rho0p7-0p9-0p98_delay1-2-3.csv`、`results/execution_mismatch/temporal_reliability_pilot_ep300_runs3_rho0p7-0p9-0p98_delay1-2-3_b4_rw0-0p5-1_qz0-1.csv`、`results/execution_mismatch/temporal_deviation_oracle_pilot_ep300_runs3_rho0p7-0p9-0p98_delay1-2-3_b4.csv` 和 `results/execution_mismatch/learned_temporal_deviation_pilot_train500_val100_eval100_runs3_rho0p7-0p9-0p98_delay1-2-3_b4_offsets7.csv`。主要结论：

- 当执行误差从 `0` 增加到 `0.5` 时，`Execution Oracle Full CSI` 仍可达到 `50/50`，说明物理上仍有可调度机会；问题在于决策 CSI 与执行信道不一致。
- `Exact Greedy Full CSI` 在 `execerr=0.5` 下仍有 `49.972/50` 成功节点，但平均失败邀请升到 `5.39`、missed opportunities 升到 `3.87`、oracle gap 升到 `2.403`，比只做决策 preview 误差更能暴露执行风险。
- 有限 preview 更敏感：`Estimated Rotating Grid B=1` 在 `execerr=0.5` 下为 `49.012/50`、`38.56%` 完美覆盖率、`4.771` slots；`B=4` 为 `49.849/50`、`86.22%`、`3.934` slots。
- 新增 execution-risk-aware 规则显式使用执行漂移统计量，但固定保守阈值没有超过 rotating：`execerr=0.5, B=4` 下，`Execution-Risk rw=0.1 rt=0.55` 虽把 failed 从 `5.43` 降到 `4.64`，但 missed opportunities 从 `5.69` 升到 `9.66`，oracle gap 从 `2.841` 升到 `3.738`。
- 机会成本版本进一步把 false reject / false accept 代价和 deadline/backlog urgency 纳入邀请决策，但仍没有超过 rotating：`execerr=0.5, B=4` 下，最好成功率配置为 `49.823/50`、`5.07` failed、`6.37` missed、gap `2.949`，弱于 rotating 的 `49.849/50`、gap `2.841`。这说明仅靠 invitation filter 不够，下一步应转向更真实的信道失配、多目标约束或带置信/重复确认的 feedback probing。
- 新增 temporal AR(1) CSI delay 后，execution oracle 仍满覆盖，full stale greedy 也接近 oracle；有限 preview 的主要差距体现在 slots、perfect rate 和 oracle gap。`rho=0.9, delay=3` 下，`Rotating B=1` 为 `49.032/50`、gap `2.642`，`B=4` 为 `49.817/50`、gap `1.215`。朴素 AR1 mean prediction 不如 stale rotating，说明下一步应学习可靠度/分位数或多目标策略，而不是直接预测复信道均值。
- Temporal reliability / quantile 排序进一步确认：只在 rotating 候选内部用手工可调度概率重排仍不足以超过 `Rotating B=4`。`rho=0.9, delay=3` 下普通 rotating 为 `49.817/50`、gap `1.215`，最佳 temporal-reliability 为 `49.814/50`、gap `1.226`；更保守的 `rw=1` 会明显牺牲 perfect rate 和 oracle gap。
- Temporal deviation oracle 是正向诊断：如果 hidden oracle 能在每个 slot 只选 B=4 个更好的 probe IRS，性能会接近 oracle。`rho=0.9, delay=3` 下 deviation oracle 为 `49.988/50`、perfect `98.78%`、gap `0.383`，明显好于 rotating 的 `49.817/50`、gap `1.215`。这说明下一步应学习 probe-set deviation，而不是继续堆手工 invitation filter。
- 第一版 learned temporal deviation selector 尚未有效吃到这个 oracle gap：validation offset hit rate 为 `27.17%`，闭环结果与 rotating 接近但不稳定；`rho=0.9, delay=3` 下 learned 为 `49.767/50`、perfect `80.00%`、gap `1.268`，弱于 rotating 的 `49.827/50`、perfect `85.00%`、gap `1.242`。这说明后续应换成 on-policy/DAgger 或更丰富反馈，而不是继续做低维离线 offset regression。

## 当前参数扫描结论

当前推荐引用 `results/parameter_sweep/parameter_sweep_ep300_seed2026_summary.csv`，对应默认 `K/N/M/C` 一因子扫描。主要结论：

- 在 `K=30/50/80`、`N=5/10/15`、`M=32/64/128` 的扫描中，Feature Argmax IRS 与 Greedy IRS 的成功率和时延几乎一致，均明显优于 Random IRS 和 No IRS。
- `C=8` 时 Feature Argmax 和 Greedy 的完美覆盖率都降到 `85.0%`，而 `C=16/32` 恢复到约 `99.3%-100%`，说明码本分辨率是主要限制。
- Feature Argmax PowerTie 在所有扫描配置下复现 Greedy IRS 的覆盖率、时延和能耗，说明 Greedy 的额外价值主要就是 max-count 候选内的功率 tie-break。
- Feature Argmax PowerTie 每个决策时隙通常只额外 preview `2.1-3.5` 个并列候选；Greedy IRS 需要 preview 全部 `C` 个候选。
- Random IRS 有时也接近满成功率，但平均需要约 `5` 个时隙；Feature Argmax/Greedy 通常只需约 `2-3` 个时隙。

运行时间结果推荐引用 `results/runtime/runtime_benchmark_ep200_seed2026.csv`。其中 PowerTie 的平均决策耗时约 `0.0925 ms`，Greedy 约 `0.4345 ms`；PowerTie 与 Greedy 能耗相同，但额外 preview 次数约 `2.54` vs `16.00`。

下一步研究重点应转向信道估计误差或多目标约束场景，让 RL 问题真正有学习价值。
