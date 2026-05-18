# 代码中文详解

本文面向第一次接触本仓库的读者，目标是把“代码在做什么、为什么这样分层、数据如何流动、每个脚本应如何理解”讲清楚。它不是论文草稿，也不是逐行翻译源码；如果逐行解释会把维护成本推到源码之外，并且很快过期。本文采用更适合维护的方式：先解释核心概念和运行链路，再按文件说明职责、关键函数、输入输出和常见误解。

当前项目主线可以概括为：

```text
IRS-assisted multi-slot AirComp
under stale/limited CSI and execution-channel mismatch
using low-cost IRS candidate generation plus current aggregate feedback
```

用中文说，就是研究一个 IRS 辅助的多时隙 AirComp 系统：决策时拿到的是过时或有限的信道信息，真正执行时信道可能已经变化；系统不能每个 slot 都完整扫描所有 IRS 相位，只能用少量候选 IRS 和当前 aggregate feedback 来尽量接近隐藏的 current-channel oracle。

## 建议阅读顺序

如果只想理解“项目如何跑起来”，按下面顺序读：

1. `test_env.py`: 先理解环境、动作、一次 slot 如何执行。
2. `ms_aircomp/channel_models.py`: 理解决策信道和执行信道为什么会不同。
3. `ms_aircomp/execution_candidates.py`: 理解一个 IRS candidate 是怎样被构造和比较的。
4. `ms_aircomp/feedback.py` 与 `ms_aircomp/confirmation.py`: 理解 current aggregate feedback 如何确认 IRS。
5. `ms_aircomp/probe_sets.py` 与 `ms_aircomp/execution_policies.py`: 理解 sparse / coverage-aware 候选集合如何产生。
6. `evaluate_execution_channel_mismatch.py`: 理解当前主线实验如何编排。
7. `evaluate_invitation_mask_correction.py`: 理解论文侧最新的 mask correction 方法。
8. `generate_paper_tables.py`、`generate_paper_figures.py` 和分析脚本：理解已有 CSV 如何变成表、图和文档。
9. `tests/`: 理解项目用哪些规则防止结果和边界漂移。

如果只想快速判断某个文件是否还重要，先看本文的“文件总览”和 `docs/PROJECT_STATUS.md`。

## 核心概念

| 概念 | 含义 |
|---|---|
| AirComp | Over-the-air computation。多个节点同时上传，系统关心聚合计算质量，而不是逐个解码每个用户的信息。 |
| IRS | Intelligent Reflecting Surface。这里用一组离散 codebook 相位来改变等效信道。 |
| DFT codebook | `test_env.py` 中生成的 IRS 相位候选集合。每个 index 对应一种 IRS 相位向量。 |
| slot | 一个传输时隙。每个 episode 有多个 slot，系统在每个 slot 邀请一部分节点传输。 |
| action | 环境动作，包含 gain threshold、alignment threshold 和 IRS codebook index。 |
| preview | 不真正执行，只估算某个 IRS index 在当前环境状态下会邀请多少节点、需要多少功率、可能带来多少成功节点。 |
| stale CSI | 决策时看到的旧信道状态。例如当前 slot 使用前几个 slot 的信道来选 IRS。 |
| current channel | 真正执行时的当前信道。它对部署策略通常不可完全见。 |
| execution mismatch | 决策候选和执行候选使用的信道不同。 |
| aggregate feedback | 只告诉策略某个 IRS probe 的聚合结果，例如邀请数、功率、tx count，不暴露每个节点的真实当前 CSI。 |
| candidate | 对一个 IRS index 的候选评估结果，通常包括 invited mask、tx count、power、MSE、preview cost 等。 |
| confirmed index | 用 aggregate feedback 在少量 probe 中确认出来的 IRS index。 |
| oracle | 使用隐藏真实信息的上界或诊断方法。它用于分析 headroom，不代表可部署方法。 |
| failed invitations | 被 stale decision 邀请但在执行信道下失败的节点数。 |
| missed opportunities | 执行信道下本可以成功但没有被邀请的节点数。 |
| oracle gap | 当前策略和 hidden current oracle 之间的指标差距。 |
| invitation mask correction | 已确认 IRS 后，用 current aggregate feedback count 修正 stale invitation mask 的节点数量和排序。 |

## 总体数据流

主线代码的数据流可以这样理解：

```text
1. MSAirCompEnv 生成物理信道、IRS codebook 和 episode 状态
2. temporal AR(1) 或 drift helper 构造 stale decision channel 与 current execution channel
3. 策略只用 stale / limited information 生成一小组 IRS probe candidates
4. 对 probe candidates 获取 current aggregate feedback
5. feedback confirmation 选出 confirmed IRS index
6. 用 confirmed IRS 构造最终执行候选，统计 tx、failed、missed、oracle gap
7. evaluator 按 seed / scenario / policy 聚合结果，写 CSV 和图
8. analysis / paper scripts 从 frozen CSV 生成 Markdown 表、PNG 图和论文资产
9. tests 审计代码边界、数值趋势、artifact 链接和结果字段
```

这里最重要的设计原则是“决策可见信息”和“诊断隐藏信息”分离。部署方法不能直接使用 hidden current CSI；oracle 和 learned diagnostics 可以用隐藏标签训练或诊断，但必须在 metadata 中标出，避免被误当成主方法。

## 核心环境：`test_env.py`

`test_env.py` 定义 `MSAirCompEnv`，它是所有实验的物理环境基础。

### `MSAirCompEnv` 的职责

- 生成节点、IRS、基站之间的复数信道。
- 构造 IRS DFT codebook。
- 定义 Gymnasium action space 和 observation space。
- 把连续 action 解码成物理参数：
  - `g_th`: 邀请节点所需的等效信道增益门限。
  - `alpha_th`: 相位对齐或聚合质量相关门限。
  - `irs_codebook`: 离散 IRS 相位 index。
- 在 `step()` 中执行一个 slot，并返回 observation、reward、done、truncated、info。
- 在 `preview_codebook_index()` 中无副作用地预览某个 IRS index 的效果。

### 环境状态如何更新

一次 episode 开始时，`reset()` 会生成或重置：

- 物理信道。
- 当前 slot index。
- 尚未成功上传的节点集合。
- 累计成功节点、能耗、MSE 等统计量。

每次 `step(action)`：

1. `_sanitize_action()` 先把 action 裁剪到合法范围，避免越界动作破坏环境。
2. `_decode_action()` 把神经网络或规则策略输出的归一化 action 转成物理参数。
3. 根据 IRS codebook index 得到 IRS 相位。
4. 计算每个节点的等效信道、是否被邀请、所需功率、是否成功。
5. 更新剩余节点和累计指标。
6. 在 `info` 中写出调试和统计字段，例如本 slot 的 tx count、power、MSE、selected IRS index。

### `preview_codebook_index()` 为什么重要

很多 baseline 需要比较多个 IRS index。如果直接调用 `step()`，环境会被推进，episode 会被改变。`preview_codebook_index()` 的意义是：在不改变当前环境状态的前提下，评估某个 IRS index 的候选效果。测试里有专门的 no-side-effect 检查来保护这一点。

## `ms_aircomp/` 可复用模块层

`ms_aircomp/` 是当前项目的 reusable experiment layer。新实验应尽量从这里导入 helper，而不是从 evaluator 顶层脚本复制或导入内部函数。这个边界由 `tests/dependency_boundary_checks.py` 守住。

### `ms_aircomp/experiment_utils.py`

公共无状态工具。

- `ensure_parent_dir(path)`: 写文件前确保父目录存在。
- `physical_to_action()`、`codebook_index_to_action()`、`action_to_alpha()`: 在物理量和 Gym action 表示之间转换。
- `make_run_seeds()`、`make_episode_seeds()`、`make_episode_seed_list()`: 统一 seed 生成，保证实验可复现。
- `validate_common_experiment_args()` 和一组 `validate_*_values()`: 检查 CLI 参数是否合法。
- `update_energy()`: 从环境 `info` 中累加 episode 能耗。

这类函数本身不理解主线策略，只负责让不同脚本共享相同的参数、seed 和输出规则。

### `ms_aircomp/channel_models.py`

负责“信道为什么会变”的建模。

- `capture_channel_state(env)`: 从环境里取出当前物理信道快照。
- `apply_channel_state(env, state)`: 把某个信道快照写回环境。
- `drift_channels(h_total, execution_error_std, rng)`: 对等效信道加扰动，模拟执行阶段信道误差。
- `build_temporal_channel_trace()` / `build_temporal_channel_states()`: 构造 temporal AR(1) 信道序列。
- `delayed_channel_state()`: 给定 slot 和 delay，取出 stale CSI。
- `ar1_predict_channel_state()`: 用 AR(1) 模型把 delayed CSI 预测到当前。
- `temporal_uncertainty_std()`: 根据 `rho` 和 delay 估计 stale CSI 的不确定性。

主线实验中，决策时先 `apply_channel_state(stale_state)`，执行时再切回 current state。这样代码层面强制区分“策略看见的信道”和“真实执行信道”。

### `ms_aircomp/execution_candidates.py`

负责构造执行阶段候选。

- `execution_candidates()`: 对一组 IRS indices 生成候选列表。
- `execution_candidate_for_decision()`: 对一个 decision candidate 计算其执行信道下的真实效果。
- `execution_oracle_candidate()` / `choose_execution_oracle()`: 扫描所有 IRS index，得到隐藏 current-channel oracle。

candidate 一般包含：

- `index`: IRS codebook index。
- `action`: 可传给环境的动作。
- `tx_count`: 预计或真实成功传输节点数。
- `power`: 发射功率或聚合功率。
- `mse`: 聚合误差。
- `invited_mask`: 被邀请节点的布尔 mask。
- `preview_count` 或相关 cost 字段。

### `ms_aircomp/feedback.py`

负责把候选转换为 aggregate feedback。

- `confirmation_feedback(candidate, args, feedback_noise_std, feedback_power_weight, rng)`: 生成某个 probe 的反馈分数。反馈可以带噪声，也可以把功率惩罚合进分数。
- `confirmed_index_from_feedback(selected_indices, feedbacks)`: 从一组 probe feedback 中选出最终确认的 IRS index。

这部分体现了项目的部署约束：策略看到的是聚合反馈，不是完整节点级 current CSI。

### `ms_aircomp/confirmation.py`

负责“用当前反馈确认 IRS”的流程封装。

- `confirm_index_with_current_feedback()`: 输入一组候选 IRS index，计算每个 index 的 current aggregate feedback，再返回 confirmed index 和相关候选。

很多策略都需要这个步骤，所以它被抽到独立模块，避免每个策略重复写 confirmation loop。

### `ms_aircomp/probe_sets.py`

负责候选 IRS index 集合的构造和补齐。

- `ordered_unique_prefix()`: 保留顺序、去重，并截断到预算。
- `circular_codebook_distance()`: 计算 codebook index 在环上的距离。
- `fill_diverse_codebook_indices()`: 当优先候选不足时，用分散的 index 补齐。
- `coverage_increment_stats()`: 统计某个新 index 能覆盖多少尚未覆盖的节点。
- `coverage_aware_sparse_indices()`: 生成 coverage-aware sparse candidate set。

Coverage-aware 方法的核心思想是：不要只选 stale gain 最大的 IRS，也要考虑不同 IRS 候选覆盖的节点是否重复。这样可以降低 missed opportunities。

### `ms_aircomp/execution_policies.py`

主线非学习策略集合。

- `choose_no_irs_fallback()`: 不使用 IRS 的 fallback。
- `choose_rotating_feedback_confirm_decision()`: 固定轮换 probe，再用 current feedback 确认。
- `choose_stale_topk_feedback_decision()`: 用 stale full preview 选 top-k，再用 current feedback 确认。
- `choose_active_diverse_feedback_decision()`: 用多样性选 probe set。
- `choose_sparse_topk_feedback_decision()`: 先用较少 stale preview 生成 seed pool，再选 top-k feedback probes。
- `choose_coverage_sparse_topk_feedback_decision()`: 在 sparse-topk 基础上加入 coverage diversity。
- `choose_neighbor_coverage_sparse_topk_feedback_decision()`: 把部分预算分配给局部邻居，用于诊断邻域扩展是否有收益。

这些函数通常返回一个 decision dict，里面包含选中的 candidate、probe indices、preview cost、confirmed index、metadata 等。Evaluator 不需要知道每种策略内部怎么选，只负责调用 dispatch 后执行。

### `ms_aircomp/adaptive_sparse_policies.py`

Adaptive Sparse-TopK 系列。

- `sparse_topk_count_margin()`: 判断 stale top candidates 之间的 tx count margin 是否足够大。
- `choose_adaptive_sparse_topk_feedback_decision()`: v1，根据 margin 决定是否扩展候选。
- `adaptive_sparse_history_signal()`: 从历史 confirmed feedback 中提取信号。
- `adaptive_sparse_deadline_urgency()`: 越接近 deadline、剩余节点越多，扩展候选的动机越强。
- `adaptive_sparse_stale_uncertainty()`: 根据 stale CSI 不确定性调节风险。
- `choose_adaptive_sparse_topk_v2_feedback_decision()`: v2，综合 cost、history 和 uncertainty，是当前 continuum diagnostic 中较有价值的版本。
- `choose_adaptive_sparse_topk_v3_feedback_decision()`: v3，加入 history 和 local-neighbor heuristic，但当前不作为主线。

Adaptive 系列的目标不是固定一个预算，而是在不同场景下自适应增加或减少 preview。

### `ms_aircomp/execution_risk_policies.py`

早期风险感知执行策略。

- `execution_risk_error_std()`: 合并 decision error 和 execution error。
- `candidate_with_execution_reliability()`: 为候选估计可靠性。
- `execution_risk_candidate_set()`: 生成带可靠性字段的候选。
- `execution_urgency()`: 根据 deadline 和 backlog 计算紧迫度。
- `opportunity_cost_candidate()`: 把机会成本纳入候选评价。
- `choose_opportunity_execution_risk_decision()`、`choose_execution_risk_decision()`: 返回风险感知决策。

这些方法保留为 diagnostic / historical baseline，不是当前最终主线。

### `ms_aircomp/temporal_policies.py`

时间相关策略和 oracle 诊断。

- `temporal_reliability_candidate()`: 用 stale CSI 不确定性估计候选可靠性。
- `choose_temporal_reliability_decision()`: 根据 temporal reliability 选择 IRS。
- `choose_temporal_deviation_oracle_decision()`: hidden-info temporal diagnostic，用当前真实信息判断应偏移到哪个邻域。

`Temporal Deviation Oracle` 不是可部署方法，它用来说明同等 probe budget 下仍有 candidate-set headroom。

### `ms_aircomp/invitation_mask_correction.py`

Invitation mask correction 的纯函数核心。

- `validate_mask_correction_rerank_mode()`: 检查 rerank mode 是否支持。
- `rank_remaining_by_stale_gain()`: 在 confirmed IRS 下按 stale gain 对剩余节点排序。
- `corrected_target_count()`: 根据 aggregate feedback count、deadband、clip 等规则计算目标邀请数量。
- `apply_invitation_mask_correction()`: 对原始 stale invitation mask 做修正。

这个模块故意只做纯函数，不直接运行 episode。这样 evaluator 可以清楚地区分“选 IRS”和“修正 invitation mask”。

### `ms_aircomp/learned_shortlist.py`

学习式 shortlist 诊断的部署侧 helper。

- `load_learned_shortlist_model()` / `load_learned_set_shortlist_model()`: 读取线性模型。
- `learned_shortlist_context()`: 构建 slot 上下文，例如 stale candidates、feedback history、budget 等。
- `learned_shortlist_feature_vector()` / `learned_shortlist_feature_matrix()`: 为单个 IRS index 构建特征。
- `score_learned_shortlist_candidates()`: 用模型给候选打分。
- `learned_set_shortlist_variants()` 和 set feature helpers: 把“选择一组 indices”作为候选集合学习问题。
- `choose_learned_sparse_shortlist_feedback_decision()` / `choose_learned_set_shortlist_feedback_decision()`: 在闭环评估中使用模型选 probe set。

注意：训练这些模型时可能用 hidden current labels，所以新生成结果必须带 diagnostic metadata。闭环推理时不应访问 hidden current CSI。

### `ms_aircomp/limited_csi.py`

有限 CSI 研究线的可复用 helper。

- `parse_int_list()`、`parse_float_list()`: CLI 列表解析。
- `grid_indices()`、`unique_fill()`: probe index 选择基础工具。
- `make_env()`: 用 limited-CSI evaluator 参数创建环境。
- `effective_channels()`: 得到某些 IRS index 下的等效信道。
- `success_gain_threshold()`、`estimate_success_reliability()`: 估计成功概率和门限。
- `build_candidate()`、`true_preview_candidates()`、`estimated_preview_candidates()`: 构造真实或估计 candidate。
- `risk_aware_candidate()` 和相关 key 函数: 风险感知候选。
- `select_indices()`、`choose_policy_candidate()`: 根据策略名选候选。
- `execute_limited_csi_slot()`: 决策用估计 CSI，执行只统计真实可成功节点。
- `oracle_candidate()`、`true_candidate_for_decision()`: 上界和诊断 helper。

这条线支撑了当前主线之前的 limited-CSI 结论，部分思想被 execution-mismatch 主线吸收。

### `ms_aircomp/execution_policy_registry.py`

策略和场景注册表。

- `policy_label()`: 把策略配置转成人类可读 label。
- `policy_configs(args)`: 根据 CLI 参数展开所有要跑的 policy configuration。
- `mismatch_scenarios(args)`: 根据 CLI 参数展开 mismatch scenarios。

这个文件减少了 evaluator 中的 if/else，使新策略可以通过 registry 统一进入实验矩阵。

### `ms_aircomp/execution_decision_dispatch.py`

策略分发层。

- `choose_decision()`: 根据 policy config 调用对应 reusable helper。
- `choose_execution_mismatch_decision()`: 保留 legacy 兼容入口。

Evaluator 只需要传入环境、参数、policy 名称、slot 信息和 history；具体策略实现由 dispatch 隔离。

### `ms_aircomp/execution_output.py`

输出相关 helper。

- `resolve_output_prefix()`: 处理 CSV / PNG 输出前缀。
- `print_progress()`: 统一进度打印。
- `print_summary()`: 控制台输出汇总表。
- `plot_results()`: 生成 execution mismatch 图。

这类代码和策略逻辑无关，独立出来可以降低 evaluator 复杂度。

### `ms_aircomp/execution_result_summary.py`

结果聚合和 CSV 写出。

- `seed_summary()`: 单个 seed 的 summary。
- `aggregate_seed_results()`: 多 seed 结果聚合。
- `metric_mean_ci()`: 计算均值和置信区间。
- `summarize_results()`: 把 evaluator 原始结果变成 summary rows。
- `write_csv()`: 写出统一字段 CSV。

这个模块还定义了主线 CSV 字段和 metadata 字段，保证 paper / analysis / tests 读到的格式稳定。

### `ms_aircomp/__init__.py`

包入口。它通常只 re-export 当前项目希望稳定使用的 helper。不要把实验脚本里的临时函数随意放进这里。

## 当前主线 evaluator：`evaluate_execution_channel_mismatch.py`

这是当前最重要的实验脚本。

### 它解决什么问题

传统 evaluation 往往假设决策时和执行时看到的是同一个信道。当前主线认为这不现实：实际部署时，策略可能只能看到 stale CSI，真正执行时信道已经变化。因此脚本要比较：

- 不用 IRS 的 fallback。
- 低成本 rotating feedback。
- Sparse-TopK。
- Coverage-Aware Sparse-TopK。
- Adaptive Sparse-TopK。
- Stale-TopK 高成本 reference。
- Temporal Deviation Oracle 等 hidden-info diagnostic。

### 主要函数

- `parse_args()`: 定义所有 CLI 参数，例如 episodes、seeds、probe budgets、rho、delay、policy list、noise std 等。
- `validate_args(args)`: 检查参数范围，避免无意义实验进入结果目录。
- `evaluate_policy(...)`: 核心 episode loop。
- `main()`: 解析参数、展开 policy/scenario、调用 evaluator、写 CSV/图和 summary。

### `evaluate_policy()` 内部逻辑

可以把它理解成四层循环：

```text
for seed:
  for episode:
    reset env
    build temporal channel states if needed
    for slot:
      apply stale decision state
      call policy dispatch to choose decision
      apply current execution state
      evaluate selected decision under execution channel
      compare with hidden oracle
      update metrics and history
```

每个 slot 会同时维护：

- 策略实际可见的 stale candidate 信息。
- current aggregate feedback 确认的 IRS index。
- 执行信道下真实成功节点。
- hidden oracle 的结果。
- failed / missed / gap 等诊断指标。

### 为什么 evaluator 还保留一些 re-export

历史脚本曾经从 `evaluate_execution_channel_mismatch.py` 直接导入 helper。现在这些 helper 已逐步搬到 `ms_aircomp/`，但为了不立刻破坏旧脚本，保留了少量 legacy compatibility surface。新增代码不应继续从 evaluator 导入 reusable helper。

## Mask correction evaluator：`evaluate_invitation_mask_correction.py`

这个脚本研究的是：确认 IRS index 之后，原来的 stale invitation mask 是否还应该原样使用。

### 背景

Coverage-Aware B=3 可以用同样 preview cost 获得较低 no-noise gap，但仍存在 invitation-mask mismatch：IRS index 选得不错，不代表 stale CSI 下邀请的节点集合在 current channel 下仍然合适。

### 主要流程

1. `choose_coverage_b3_decision()` 复用主线 Coverage-Aware B3 决策。
2. 用 current aggregate feedback 得到 confirmed IRS index 和 aggregate count。
3. `apply_invitation_mask_correction()` 计算 corrected invitation mask。
4. 在 current execution channel 下执行 corrected mask。
5. 统计 slots、failed、missed、gap，并和 uncorrected B3 比较。

### 主要函数

- `parse_args()`、`normalize_args()`、`validate_args()`: CLI 和参数校验。
- `make_env()`: 创建环境。
- `choose_coverage_b3_decision()`: 生成基础 B3 决策。
- `run_episode()`: 单个 episode 的完整执行。
- `correction_configs()`: 展开 correction strength、deadband、clip、rerank mode 等配置。
- `run_evaluation()`: 多 seed / 多场景实验。
- `aggregate_by_policy()`: 聚合结果。
- `write_csv()`、`write_markdown()`: 写出结果。

### 结果如何理解

`Mask-Corrected Coverage-Aware B=3 mc=1` 当前是 trade-off method：它降低 slots、failed 和 missed，但 no-noise gap 不一定优于未修正的 B3。因此论文或报告中不能简单写成“全面优于”，而应写成“在同 preview 下改善 invitation-level execution reliability，但存在 gap trade-off”。

## 分析和论文资产脚本

这些脚本通常不重新跑环境，只读取已有 CSV，生成 Markdown、CSV companion 和图。

### `summarize_execution_baselines.py`

把多个 execution mismatch CSV 汇总成 baseline 表。它的核心是：

- 读取 source CSV。
- 按 policy 选出需要报告的行。
- 计算跨 scenario 平均指标。
- 写 `docs/EXECUTION_BASELINE_SUMMARY.md` 和 summary CSV。

### `analyze_main_frontier.py`

生成当前主线 frontier 分析。

- `load_method_rows()` 读取多个主线结果源。
- `aggregate_by_label()` 按方法 label 聚合。
- `plot_preview_gap()` 画 preview cost 和 oracle gap 的关系。
- `plot_failed_missed()` 画 failed / missed trade-off。
- `write_markdown()` 输出 `docs/MAIN_RESULTS_ANALYSIS.md`。

### `analyze_coverage_aware.py`

分析 Coverage-Aware 的 weight、power penalty、budget split 和 neighbor variants。

- `load_analysis_rows()`、`load_power_summaries()`、`load_budget_split_summaries()` 读取各类消融结果。
- `aggregate_sparse()`、`aggregate_coverage()` 做方法聚合。
- `plot_weight_ablation()`、`plot_power_ablation()` 生成消融图。
- `write_markdown()` 输出 `docs/COVERAGE_AWARE_ANALYSIS.md`。

### `diagnose_coverage_b3_failures.py`

把 B3 residual gap 拆成多个来源。

- `trace_one_slot()` 是核心诊断：同一个 slot 下分别检查 pool gap、selection gap、confirmation gap、invitation gap。
- `primary_gap_type()` 给 gap 归因。
- `run_trace()` 跑完整诊断。
- `summarize_rows()` 和 `write_markdown()` 输出 CSV/Markdown。

这个脚本不只是算结果，而是在回答“为什么 B3 还没有达到 oracle”。

### `analyze_invitation_mask_final.py`

把 invitation-mask correction formal 和 noise-aware 结果整理成最终表和图。

- `aggregate_method()` 聚合同一方法同一 noise level 的指标。
- `add_deltas()` 加上相对 B3 的变化。
- `plot_gap_vs_noise()` 和 `plot_failed_missed_vs_noise()` 生成 Figure 4 相关图。
- `write_markdown()` 输出最终分析文档。

### `generate_paper_tables.py`

从 frozen source CSV 生成论文 Table 1/2/3。

- `build_table1()` 生成主结果表。
- `build_table2_rows()` 生成 coverage-aware ablation / budget-split 表。
- `build_table3_rows()` 生成 failure diagnosis 表。
- `build_uncertainty_rows()` 和 `build_paired_delta_rows()` 生成 uncertainty companion 和 paired deltas。
- `write_*_markdown()` 写论文侧 Markdown 表。

### `generate_paper_figures.py`

从 frozen CSV 生成论文 Figure 2/3/4。

- `build_points()` 为 Figure 2/3 选点。
- `plot_preview_gap()` 画 preview-gap frontier。
- `plot_failed_missed()` 画 failed/missed trade-off。
- `build_figure4_points()` 和 Figure 4 plot 函数处理 mask correction noise sweep。

## 基础评估脚本

这些脚本是历史和背景实验的重要组成部分，但不是当前最终主线。

### `evaluate_policy_comparison.py`

基础 baseline 表入口。

包含：

- No IRS。
- Random IRS。
- Feature Argmax。
- Feature Argmax PowerTie。
- Greedy IRS。
- 可选 SAC / Codebook-Aware SAC。
- 可选 Fixed IRS / Best Fixed IRS。

核心函数包括：

- `evaluate_fixed_action_policy()`: 固定动作策略。
- `evaluate_sac_policy()`: SAC 模型评估。
- `evaluate_codebook_aware_sac_policy()`: IRS-only SAC selector 评估。
- `evaluate_greedy_irs_policy()`: 完整 preview greedy 上界。
- `evaluate_feature_argmax_irs_policy()` 和 PowerTie variant。
- `summarize()`、`plot_results()`。

### `evaluate_limited_csi_ms_aircomp.py`

有限 CSI 评估入口。它现在依赖 `ms_aircomp.limited_csi` 中的 reusable helper。

它的重点是：决策时只能看到估计 CSI，执行时用真实信道统计成功节点。因此它会记录估计候选和真实执行之间的偏差。

### `evaluate_channel_estimation_error_sweep.py`

研究 noisy equivalent channel 对策略的影响。

- `estimated_preview_candidates()` 加入估计误差。
- `true_preview_candidates()` 或 exact preview 作为对照。
- `choose_policy_candidate()` 根据 policy 选择候选。

### `evaluate_partial_probing_sweep.py`

研究每个 slot 只 preview 少数 codebook index 的情况。

- grid、local、random、rotating 等 probe selector。
- full greedy upper bound。
- 每个预算下的 success / slot / energy 统计。

### `evaluate_bandit_feedback_ms_aircomp.py`

更严格的 aggregate feedback 研究线。

- 策略只能 probe 少量 IRS index。
- 每个 probe 只获得 noisy aggregate feedback。
- 不暴露节点级 current CSI。

它为当前 confirmation 设计提供了早期依据。

### `evaluate_bandit_feedback_stress_sweep.py`

在更困难 scenario 中压力测试 bandit feedback 策略，例如短 slot、高噪声等。

### `evaluate_adaptive_feedback_probing.py`

非学习 adaptive rotating backup。它根据 primary probe 的反馈情况决定是否启用 backup probe。

### `evaluate_parameter_sweep.py`

对 `K/N/M/C` 等环境规模做一因子扫描，用于背景敏感性分析。

### `benchmark_policy_runtime.py`

测量不同策略的决策时间、环境 step 时间和 preview 调用数量。它回答的是“策略是否只在指标上好，还是运行成本也可接受”。

### `diagnose_policy_actions.py`

诊断 SAC / Codebook-Aware SAC / Greedy 的动作分布、IRS index 使用、slot latency 和 oracle gap。它主要用于解释为什么早期 RL 方向没有成为当前主线。

### `evaluate_random_irs_baseline.py`

早期随机 IRS baseline 脚本。当前更完整的基础比较应优先看 `evaluate_policy_comparison.py`。

### `evaluate_agent.py` 与 `evaluate_batch.py`

早期 SAC agent 评估脚本。保留用于复现旧模型行为，不是当前主线入口。

## 训练脚本

这些脚本负责训练历史或诊断模型。当前项目结论显示，多数学习式分支不作为最终主方法，但代码仍保留以支持负结果和诊断。

### `train_agent.py`

训练完整 SAC。动作包含传输门限和 IRS index。`TrackSuccessCallback` 用于训练过程中追踪成功节点数。

### `train_codebook_aware_agent.py`

训练 IRS-only SAC selector。

- `FixedTransmissionActionWrapper` 固定传输参数，只让 agent 选择 IRS codebook。
- 目标是把学习问题聚焦到 IRS 选择。

### `train_greedy_imitation_selector.py`

监督学习模仿 Greedy IRS index。

- `collect_greedy_dataset()` 用 Greedy 生成训练标签。
- `ImitationSelector` 是分类模型。
- `evaluate_policy()` 检查模型闭环表现。

现有结论表明，简单 imitation 不足以成为主方法。

### `train_bandit_feedback_selector.py`

训练 feedback-conditioned probe selector。

- 从 aggregate feedback history 构造特征。
- 训练 `FeedbackSelector` 预测候选分数。
- 闭环评估 learned probing 是否优于规则策略。

### `train_temporal_deviation_selector.py`

学习 temporal AR(1) stale CSI 下的 offset/window selection。

- 构造历史特征和 offset target scores。
- 支持 DAgger。
- `OffsetSelector` / `WindowOffsetSelector` 输出 offset 评分。

它使用 hidden current outcomes 作为训练信号，因此属于 diagnostic。

### `train_learned_sparse_shortlist.py`

训练 sparse / set shortlist 线性模型。

- 支持 absolute-score、marginal-gain、set-value、execution-value、pairwise difference 等标签。
- `build_dataset()` 采集训练样本。
- `fit_ridge_linear()` 或 pairwise variant 训练线性模型。
- 输出 checkpoint 和 diagnostics。

当前 learned shortlist 结果弱于主线规则策略，保留为 diagnostic。

## 归档脚本：`experiments/archive/`

归档脚本可以复现历史探索，但不应作为新增研究默认入口。

- `evaluate_noisy_feature_sweep.py`: 对 exact codebook features 加噪，检查 feature baseline 鲁棒性。
- `evaluate_probing_cost_tradeoff.py`: 对早期 probing 结果做 cost trade-off 后处理。
- `train_learned_probing_selector.py`: 早期 learned partial probing selector。

归档的意义不是“无用”，而是记录哪些方向已经探索过、为什么不继续投入。

## 测试代码

测试不是简单检查能否运行，而是在保护项目结论和代码边界。

### `tests/smoke_checks.py`

轻量行为回归测试。覆盖内容包括：

- preview 不应改变环境状态。
- codebook features 应和 preview 一致。
- noisy feature 必须可复现且被裁剪。
- 非法环境维度和非法 CLI 参数应被拒绝。
- 越界 action 应被裁剪。
- seed 生成应稳定。
- temporal trace 支持 prehistory。
- partial probing / bandit / active diverse / sparse top-k 必须遵守预算。
- limited CSI 执行只统计 invited nodes。
- execution drift 应保持 index 稳定。
- invitation mask correction 的 count rules 和 noop 行为。
- policy registry 能正确展开 label 和 scenarios。
- diagnostic metadata 能标记 hidden-label 边界。
- learned shortlist 决策不能超预算。

### `tests/dependency_boundary_checks.py`

检查依赖方向：

```text
experiment scripts -> ms_aircomp reusable modules
```

它防止新增代码重新从 evaluator 顶层脚本导入 reusable helper，也防止已经抽出的 policy/output/result helper 回流到 evaluator。

### `tests/mainline_regression_checks.py`

固定 seed 的数值趋势检查。它不会追求每个浮点完全一样，而是检查主线趋势是否仍成立，例如 coverage-aware gap 是否在合理范围、mask correction trade-off 是否保持。

### `tests/mainline_artifact_checks.py`

审计主线 artifact：

- 关键 CSV / Markdown / PNG 是否存在并被 git 跟踪。
- `source_file` 链接是否存在。
- 论文表图是否能追溯到 frozen source。
- final invitation-mask 结果之间的关系是否符合文档 claim。

### `tests/validation_checks.py`

专门检查 CLI validation。它用错误参数调用脚本，确认脚本失败，并输出期望的错误片段。

### `tests/test_project_checks.py`

标准 pytest wrapper。它让现有脚本式 checks 可以通过 `python -m pytest` 或 CI 统一运行。

## 文件总览

| 路径 | 当前定位 | 读者应如何理解 |
|---|---|---|
| `test_env.py` | active core | 物理环境和 slot 执行逻辑。 |
| `ms_aircomp/*.py` | active reusable layer | 新策略和新实验优先复用这里的 helper。 |
| `evaluate_execution_channel_mismatch.py` | active mainline | 当前主线实验总入口。 |
| `evaluate_invitation_mask_correction.py` | active mainline | mask correction 实验入口。 |
| `diagnose_coverage_b3_failures.py` | active diagnostic | 解释 B3 gap 来源。 |
| `summarize_execution_baselines.py` | active summary | 生成 baseline 汇总文档和 CSV。 |
| `analyze_main_frontier.py` | active summary | 生成主线 frontier 分析。 |
| `analyze_coverage_aware.py` | active summary | 生成 coverage-aware 消融分析。 |
| `analyze_invitation_mask_final.py` | active summary | 生成 final mask correction 分析。 |
| `generate_paper_tables.py` | paper artifact generator | 从 frozen CSV 生成论文表。 |
| `generate_paper_figures.py` | paper artifact generator | 从 frozen CSV 生成论文图。 |
| `evaluate_policy_comparison.py` | background baseline | 基础规则、SAC 和 fixed IRS 对比。 |
| `evaluate_limited_csi_ms_aircomp.py` | historical / support | 有限 CSI 主评估脚本。 |
| `evaluate_channel_estimation_error_sweep.py` | historical / support | noisy channel estimation 扫描。 |
| `evaluate_partial_probing_sweep.py` | historical / support | partial codebook probing 扫描。 |
| `evaluate_bandit_feedback_ms_aircomp.py` | historical / support | aggregate feedback 早期严格模型。 |
| `evaluate_bandit_feedback_stress_sweep.py` | diagnostic | bandit feedback 压力测试。 |
| `evaluate_adaptive_feedback_probing.py` | diagnostic | adaptive rotating backup。 |
| `evaluate_parameter_sweep.py` | background | 环境规模敏感性分析。 |
| `benchmark_policy_runtime.py` | support | 运行时和 preview cost 诊断。 |
| `diagnose_policy_actions.py` | diagnostic | 解释 SAC / Greedy 动作行为。 |
| `evaluate_random_irs_baseline.py` | legacy | 早期随机 IRS baseline。 |
| `evaluate_agent.py`, `evaluate_batch.py` | legacy | 早期 agent 评估。 |
| `train_agent.py` | historical training | 完整 SAC 训练。 |
| `train_codebook_aware_agent.py` | historical training | IRS-only SAC 训练。 |
| `train_greedy_imitation_selector.py` | diagnostic training | Greedy imitation。 |
| `train_bandit_feedback_selector.py` | diagnostic training | feedback-conditioned learned selector。 |
| `train_temporal_deviation_selector.py` | diagnostic training | hidden-label temporal offset 学习。 |
| `train_learned_sparse_shortlist.py` | diagnostic training | learned sparse/set shortlist。 |
| `experiments/archive/*.py` | archive | 早期探索和负结果复现。 |
| `tests/*.py` | active guardrail | 保护行为、边界、结果和 artifact。 |

## 如何新增一个策略

建议流程：

1. 如果策略逻辑可复用，先在 `ms_aircomp/` 中新建或扩展 helper。
2. 在 `ms_aircomp/execution_policy_registry.py` 的 `policy_configs()` 中加入 CLI 展开规则和 label。
3. 在 `ms_aircomp/execution_decision_dispatch.py` 中加入分发分支。
4. 如果需要新的 probe-set 选择，优先扩展 `probe_sets.py` 或 `execution_policies.py`。
5. 在 `tests/smoke_checks.py` 增加预算、不变性或 metadata 检查。
6. 跑 `make check`。
7. 如果影响主线结果或 artifact，跑 `make mainline-audit`。

不要把新 helper 直接塞进 `evaluate_execution_channel_mismatch.py`，除非它只服务于 CLI 编排且不会被其他脚本复用。

## 如何新增一个指标

建议流程：

1. 在执行 slot 的地方记录原始 per-slot 信息。
2. 在 `ms_aircomp/execution_result_summary.py` 增加聚合字段。
3. 在相关 analysis script 中读取并展示。
4. 在 artifact checks 中加入字段或关系审计。
5. 文档中说明该指标是否为 deployable visible、diagnostic hidden，还是 paper-facing metric。

指标命名应尽量明确，例如 `failed_invitations` 比 `fail` 更清楚。

## 如何新增一个实验脚本

新增脚本应回答一个明确问题，例如：

- 是否降低 same-preview oracle gap？
- 是否减少 failed invitations？
- 是否改善 high-noise feedback robustness？
- 是否解释某个 residual gap 来源？

脚本结构建议：

```text
parse_args
validate_args
make_env
run one episode / one policy
aggregate results
write CSV
optional plot / markdown
main
```

已有脚本多数遵循这个形状，读 `evaluate_invitation_mask_correction.py` 可以看到相对现代的写法。

## 常见误解

1. `preview` 不是免费信息。主线结果会记录 preview cost，很多方法的价值就在于用较低 preview 接近 oracle。
2. `oracle` 不是部署方法。它使用 hidden current information，只能作为上界或诊断。
3. `aggregate feedback` 不等于完整 CSI。它只提供聚合结果，不能反推出每个节点的真实当前信道。
4. `Coverage-Aware` 不是简单增加 probe 数。它在相近预算下改变 candidate set 的覆盖结构。
5. `Mask correction` 不只是改邀请数量。当前有效版本包括 aggregate target-count correction 和 confirmed IRS 下的 stale-gain reranking。
6. `learned` 结果不能自动当主方法。凡是训练用 hidden current labels 的模型，都必须被视为 diagnostic，除非闭环部署约束和 metadata 清楚说明。
7. `results/` 中很多文件是生成物。是否 paper-facing 要看 `docs/PAPER_FREEZE_MANIFEST.md`、`docs/RESULTS_INDEX.md` 和 artifact tests。
8. 当前项目不是“除了论文都完成就永远不用改”。非论文侧已具备可维护状态，但后续若新增方法、改指标或换环境，仍要补测试和文档。

## 给不同读者的入口

| 读者 | 优先阅读 |
|---|---|
| 只想运行项目 | `README.md`, `docs/ENVIRONMENT.md`, `Makefile` 的 `help` 和 `docs` target。 |
| 想理解代码 | 本文、`test_env.py`, `ms_aircomp/README.md`, `ms_aircomp/*.py`。 |
| 想理解当前研究主线 | `docs/MAIN_STORY.md`, `docs/PROJECT_STATUS.md`, `docs/MAIN_RESULTS_ANALYSIS.md`。 |
| 想写论文 | `docs/PAPER_RESULT_PACKAGE.md`, `docs/PAPER_STRUCTURE_MAP.md`, `docs/PAPER_FIGURE_TABLE_SPECS.md`。 |
| 想验证结果 | `tests/mainline_artifact_checks.py`, `tests/mainline_regression_checks.py`, `docs/PAPER_FREEZE_MANIFEST.md`。 |
| 想继续开发 | 本文的“如何新增策略/指标/实验脚本”、`tests/dependency_boundary_checks.py`。 |

## 当前代码完整度判断

从非论文侧看，当前代码已经具备以下完整度：

- 有清晰主线 evaluator。
- 有 reusable helper 模块层。
- 有 active / diagnostic / archive 边界。
- 有本地检查、pytest wrapper、artifact audit 和 CI workflow。
- 有 frozen result package 和生成论文表图的脚本。
- 有中文项目状态、项目地图和现在这份代码读本。

仍需注意的边界：

- 源码本身没有逐行中文注释；当前选择是用集中式中文读本维护解释，避免把研究代码改成大段叙述文本。
- 若后续新增策略、指标、论文图表或训练模型，应同步更新本文、`docs/PROJECT_STATUS.md` 和测试。
- 论文侧正文、符号系统和最终叙事仍需另行完成；本文只解释代码，不替代论文写作。
