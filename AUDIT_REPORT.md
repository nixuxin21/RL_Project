# Project Audit Report

## 0. Resolution Status

本报告最初记录的是审阅时的风险清单。当前仓库已经完成一轮工程收口，状态如下：

| ID | Current Status | Evidence |
|---|---|---|
| AUD-001 | Resolved for implementation consistency. | `ms_aircomp/channel_models.py` 的 `execution_rng(..., candidate_index=...)` 让同一 IRS index 的 execution drift 不再依赖候选 batch 形状；`ms_aircomp/execution_candidates.py` 逐 index 构造 drifted candidate。 |
| AUD-002 | Resolved for temporal AR(1) mainline. | `build_temporal_channel_trace(..., prehistory_slots=...)` 和 `delayed_channel_state(..., history_states=...)` 支持 early slots 使用真实 prehistory stale CSI。 |
| AUD-003 | Resolved for clean-clone artifact tracing. | `.gitignore` 已 unignore freeze/source CSV；`tests/mainline_artifact_checks.py` 会检查 freeze artifacts 和 source CSV 是否 tracked/staged。 |
| AUD-006 | Mitigated. | `docs/PAPER_TABLE1_UNCERTAINTY.md`、`results/paper/table1_scenario_uncertainty.csv` 和 `results/paper/table1_paired_scenario_deltas.csv` 提供 scenario-level variability；仍不声称完整 seed-level significance。 |
| AUD-007 | Resolved for project setup. | `.python-version`、`pyproject.toml`、`requirements.txt`、`requirements-lock.txt` 和 `docs/ENVIRONMENT.md` 已固定环境；2026-05-18 fresh clone + independent venv passed `make quick-audit`。 |
| AUD-008 | Resolved for release boundary. | `results/README.md` 和 `docs/PAPER_FREEZE_MANIFEST.md` 明确哪些结果需要版本化；`make mainline-audit` 检查 clean-clone 边界。 |
| AUD-009 | Resolved for current entry points. | `ms_aircomp/experiment_utils.py` 提供共享 validation helpers；`tests/validation_checks.py` 覆盖关键 CLI validation failure paths。 |
| AUD-010 | Resolved for public env API. | `test_env.py` 中 `_sanitize_action()` 检查 action shape/finite 并 clip 到 action space；`tests/smoke_checks.py` 覆盖非法 action。 |
| AUD-011 | Resolved in method description. | 文档和 `ms_aircomp/invitation_mask_correction.py` 均明确 mask correction 是 aggregate target-count correction + stale-gain reranking。 |
| AUD-012 | Mitigated by documentation. | `docs/RESULTS_INDEX.md` 说明 limited-CSI 与 execution-mismatch failure-slot 指标口径差异。 |
| AUD-013 | Mitigated by boundary docs/tests. | `docs/PAPER_APPENDIX_BOUNDARY.md`、`docs/DEPRECATED_DIRECTIONS.md` 和 dependency-boundary checks 将 learned/RL branches 固定为 diagnostics/archive。 |

当前工程状态以 `docs/PROJECT_STATUS.md`、`docs/ENVIRONMENT.md`、`results/README.md` 和 CI 工作流为准；下面保留原始审阅记录，便于追溯问题来源。

## 1. Project Understanding

本项目研究 IRS-assisted multi-slot AirComp 在有限/过期 CSI 下的调度与 IRS 候选选择问题。当前主线不是通用 SAC 强化学习结果，而是面向论文复现实验的低预算候选预览策略：在 `K=50, N=10, M=64, C=16` 的仿真环境中，用过期 CSI 生成少量 IRS 候选，再用当前聚合反馈确认候选，并进一步用聚合反馈修正邀请掩码。

核心方法链路如下：

- 物理环境：`test_env.py` 中的 `MSAirCompEnv` 生成 Rayleigh 信道、DFT IRS 码本、节点发送状态、功率约束和奖励。
- 有限 CSI 候选：`evaluate_limited_csi_ms_aircomp.py` 提供等效信道、候选构造、候选排序、有限 CSI 执行逻辑。
- 执行失配主线：`evaluate_execution_channel_mismatch.py` 评估 temporal AR(1) 或 independent mismatch 下的策略。
- 当前主推策略：`Coverage-Aware Sparse-TopK B=3` 加 `apply_invitation_mask_correction`，在相同 preview budget 下减少 missed opportunities 与 failed invitations。
- 论文产物：`summarize_execution_baselines.py`、`analyze_main_frontier.py`、`analyze_invitation_mask_final.py`、`generate_paper_tables.py`、`generate_paper_figures.py` 汇总 CSV、Markdown 和图片。

主要流程：

- 训练流程：历史 SAC/learned shortlist 训练入口仍存在，例如 `train_agent.py`、`train_codebook_aware_agent.py`、`train_learned_sparse_shortlist.py`、`train_temporal_deviation_selector.py`。当前文档把 learned variants 定位为 diagnostics/appendix，不是主线方法。
- 测试流程：`make test` 先 `py_compile`，再运行 `tests/smoke_checks.py`；`make boundary-test` 检查模块边界；`make regression-test` 做固定种子的主线数值回归；`make mainline-audit` 检查冻结论文产物。
- 评估流程：主线入口是 `evaluate_execution_channel_mismatch.py` 和 `evaluate_invitation_mask_correction.py`；历史/附录入口包括 `evaluate_policy_comparison.py`、`evaluate_limited_csi_ms_aircomp.py` 等。
- 画图流程：`generate_paper_figures.py` 从已有 summary CSV 生成 paper figures；`analyze_main_frontier.py`、`analyze_coverage_aware.py` 等会生成中间分析 CSV/PNG/MD。
- 结果保存：论文冻结产物在 `results/execution_mismatch/`、`results/paper/` 和 `docs/` 中；大量 pilot/source CSV 在本机存在但多数被 `.gitignore` 忽略。

最可能存在风险的模块：

- `ms_aircomp/execution_candidates.py`：执行阶段 drift 的 RNG 与候选集合形状耦合，可能影响 independent/nonzero execution-error 结论。
- `ms_aircomp/channel_models.py`：过期 CSI 的 early-slot clipping 会让 episode 前几个 slot 的 stale CSI 过于有利。
- `ms_aircomp/invitation_mask_correction.py`：掩码修正不是单纯 cardinality correction，而是基于 stale gain 重新挑节点，论文表述需要更精确。
- `evaluate_policy_comparison.py`：旧 baseline 有 preview-cost undercount 和 test-tuned baseline 风险。
- 结果链路：部分 summary 指向未跟踪 source CSV，本地 audit 能过，但 clean clone 复现存在风险。

## 2. Run Entry Points

推荐只读检查顺序：

1. `make docs`：查看项目主线文档和论文产物路径。
2. `make test`：编译核心脚本并运行 smoke checks。
3. `make boundary-test`：检查主线模块边界。
4. `make regression-test`：检查固定种子数值趋势。
5. `make mainline-audit`：检查冻结论文结果、图表、Markdown 和路径一致性。

推荐主线复现顺序：

1. 用 `evaluate_execution_channel_mismatch.py` 生成 execution mismatch frontier/source CSV。
2. 用 `evaluate_invitation_mask_correction.py` 生成 mask-correction CSV。
3. 用 `analyze_main_frontier.py`、`analyze_coverage_aware.py`、`analyze_invitation_mask_final.py` 生成分析产物。
4. 用 `summarize_execution_baselines.py` 汇总 final execution baseline summary。
5. 用 `generate_paper_tables.py` 和 `generate_paper_figures.py` 生成 paper-facing table/figures。
6. 最后运行 `make mainline-audit` 验证冻结产物。

不建议优先使用的旧入口：

- `evaluate_policy_comparison.py` 中的 SAC/Feature Argmax/Best Fixed IRS 更像历史 baseline 或 diagnostics；除非论文明确说明成本与调参边界，否则不应作为主线公平比较证据。

## 3. Critical Issues

未发现已确认的 P0 问题。本地编译、smoke、boundary、mainline regression、artifact audit、表格/图表生成和小规模 dry-run 均通过。

### AUD-001

- 严重程度：P1
- 涉及文件和位置：`ms_aircomp/execution_candidates.py:16-30`, `ms_aircomp/channel_models.py:19-42`, `ms_aircomp/confirmation.py:26-35`, `evaluate_execution_channel_mismatch.py:607-660`
- 问题描述：execution drift RNG 只由 episode seed、execution error、slot 和 no-IRS 标志决定，但实际噪声矩阵由传入候选集合形状决定。oracle 用 `range(C)` 一次性构造全码本候选，实际执行某个已选 IRS 时只构造 `[irs_index]`。同一个 IRS 在 oracle 和 selected execution 中可能对应不同 drift realization。confirmation 阶段还会对每个 index 单独调用 `execution_candidates(indices=[index])`，导致不同 IRS 候选可能复用同一个 RNG 首行噪声模式。
- 为什么这是问题：independent/nonzero `execution_error_std` 场景下，oracle、反馈确认和实际执行不是同一个执行信道样本，比较对象不再严格共享隐藏真实世界。
- 可能造成的后果：`oracle_tx_gap_mean`、feedback-confirmed choice 和 failed/missed 指标可能偏乐观或偏悲观；execution-error sweeps 的结论可信度下降。当前 frozen mainline temporal AR(1) 且 `execution_error_std=0`，因此主表主要结论不直接受这个问题影响。
- 建议修复方式：每个 episode/slot 预生成完整 execution equivalent channel table 或 drift table，再按 IRS index slice；或者将 RNG seed 纳入 `irs_index`，保证同一 index 在不同调用形状下得到一致 drift。修复后需要重跑所有 nonzero execution-error 相关实验。
- 是否需要我确认后再修改：是。该修复会改变非零 execution-error 实验结果。

### AUD-002

- 严重程度：P1
- 涉及文件和位置：`ms_aircomp/channel_models.py:76-101`, `evaluate_execution_channel_mismatch.py:582-600`, `evaluate_invitation_mask_correction.py:316-327`
- 问题描述：`delayed_channel_state` 用 `max(0, slot_idx - delay)` 处理 CSI delay。对于 delay=1/2/3，episode 前 `delay` 个 slot 只能退回 state 0，其中 slot 0 的 stale CSI 与 current execution state 完全相同。
- 为什么这是问题：论文主线用 delay 1/2/3 做 temporal stale CSI，但 early slots 没有完整过期历史。这会系统性降低前几个 slot 的 stale difficulty。
- 可能造成的后果：slots、failed/missed、oracle gap 在高 delay 场景下可能偏乐观；不同 delay 之间差异被稀释。由于 AirComp 前几个 slot 往往发送大量节点，这个建模细节可能影响主结论强度。
- 建议修复方式：在每个 episode 生成 pre-history states，从 `t=-max_delay` 开始滚动到 `N-1`；或者保留现实现但在论文中明确说明“episode starts with synchronized CSI”，并补充 no-warmup/with-warmup ablation。
- 是否需要我确认后再修改：是。属于实验模型定义问题。

### AUD-003

- 严重程度：P1
- 涉及文件和位置：`.gitignore:28-58`, `results/execution_mismatch/final_execution_baseline_summary.csv`, `analyze_main_frontier.py:39-58`, `tests/mainline_artifact_checks.py:447-452`
- 问题描述：`final_execution_baseline_summary.csv` 和 `main_frontier_analysis.csv` 的 `source_file` 字段指向多个 source CSV，例如 `sparse_topk_frontier_ep300...csv`、`coverage_budget_split_selected_ep300...csv`、`adaptive_sparse_topk_v2_pilot...csv`。这些文件在当前本机存在，但没有被 `git ls-files` 跟踪，且 `.gitignore` 默认忽略 `results/execution_mismatch/*` 后只 unignore 一小部分 freeze artifacts。
- 为什么这是问题：`make mainline-audit` 会检查 `source_file` 是否存在，但不会检查是否被版本化。当前 audit 通过依赖本机 ignored 文件；clean clone 可能缺少 source CSV，导致 audit 或深度复现失败。
- 可能造成的后果：导师/审稿人从仓库 clean clone 后无法追溯 final summary 到 raw/source CSV；论文结果可复现性被削弱。
- 建议修复方式：二选一：跟踪所有 `source_file` 引用的 source CSV；或把 freeze manifest 和 artifact check 改为只要求 frozen aggregate，并在报告中说明 raw source 不随仓库发布。更推荐跟踪主表所需 raw/source CSV 或提供下载/生成命令。
- 是否需要我确认后再修改：是。需要决定仓库体积和发布策略。

### AUD-004

- 严重程度：P1
- 涉及文件和位置：`test_env.py:203-228`, `evaluate_policy_comparison.py:719-758`, `evaluate_policy_comparison.py:761-780`
- 问题描述：Feature Argmax 的 C 维 codebook features 由 `preview_codebook_index` 对全码本逐个计算得到，但 `evaluate_feature_argmax_irs_policy` 语义上把其描述为“不额外 preview”；PowerTie 只统计 tie-break preview calls。
- 为什么这是问题：如果 `decision_preview_calls_per_slot` 被解释为 sensing/preview 成本，那么 Feature Argmax 使用了全码本精确信息却显示为低成本。
- 可能造成的后果：历史 policy comparison、runtime 或低成本 baseline 结论可能不公平。当前 paper-facing 主表未纳入 Feature Argmax，因此不直接影响主线 Table 1。
- 建议修复方式：把 codebook feature acquisition 计入 preview cost，或将指标拆成 `feature_acquisition_cost` 与 `decision_preview_calls`；文档中把 Feature Argmax 标成 exact-feature diagnostic。
- 是否需要我确认后再修改：是。涉及 baseline 公平性定义。

### AUD-005

- 严重程度：P1
- 涉及文件和位置：`evaluate_policy_comparison.py` 中 Best Fixed IRS 相关评估函数
- 问题描述：Best Fixed IRS 在 evaluation seeds 上搜索最佳固定 IRS index 后，再在同一批 seeds 上报告结果。
- 为什么这是问题：这等价于 test-set tuned baseline，不能作为普通 deployable baseline。
- 可能造成的后果：Best Fixed IRS 指标偏乐观；如果拿来和学习策略或低预算策略比较，会构成数据泄漏/不公平评估。
- 建议修复方式：用独立 validation seeds 选择 fixed IRS，再用 held-out test seeds 报告；或明确命名为 `test-tuned fixed IRS oracle`。
- 是否需要我确认后再修改：是。需要先确认该 baseline 是否仍会写入论文/展示。

### AUD-006

- 严重程度：P1
- 涉及文件和位置：`ms_aircomp/execution_result_summary.py:231-241`, `results/paper/table1_main_results.csv`, `docs/PAPER_TABLE1_MAIN_RESULTS.md`, `docs/PAPER_RESULT_PACKAGE.md`
- 问题描述：底层 summary CSV 计算了部分 run-seed CI，但 paper-facing Table 1 只展示均值；主线正式实验为 9 个 temporal scenarios、每个 300 episodes、3 run seeds，没有在 paper-facing 表中展示 CI/误差条/配对显著性。
- 为什么这是问题：Mask-Corrected B3 的优势方向很清楚，但论文表述如果使用“strongest/best”等结论，应让读者看到 seed/scenario 不确定性。
- 可能造成的后果：实验结论显得过强，特别是当差异来自少数 scenario 或 run seed 时。
- 建议修复方式：在 Table 1 或 appendix 增加 scenario-level/seed-level CI；补充 paired bootstrap 或 paired scenario table；主文措辞避免超过统计证据。
- 是否需要我确认后再修改：是。涉及论文呈现和结论强度。

## 4. Reproducibility Issues

### AUD-007

- 严重程度：P2
- 涉及文件和位置：`requirements.txt:1-4`, `requirements-lock.txt:1-42`, `README.md`
- 问题描述：`requirements.txt` 是宽松依赖，`requirements-lock.txt` 是全量 pin，但没有工具元数据、hash、Python marker 或明确安装命令；README 指向 Python 3.13.2，但环境约束没有机器可验证地编码。`stable-baselines3[extra]` 会间接带入大量训练依赖。
- 为什么这是问题：clean environment 复现时，pip resolver、平台和 PyTorch wheel 差异可能导致不可复现或安装失败。
- 可能造成的后果：他人无法稳定构建与本机一致的 `.venv`；不同 torch/numpy/gymnasium 组合可能改变训练或运行行为。
- 建议修复方式：增加 `pyproject.toml`/`environment.yml`/`uv.lock` 中至少一种；明确 Python 版本、安装命令、CPU/GPU PyTorch 选择；把 paper reproduction 所需最小依赖和 training extras 分开。
- 是否需要我确认后再修改：是。属于工程配置变更。

### AUD-008

- 严重程度：P1
- 涉及文件和位置：`.gitignore:28-58`, `docs/PAPER_FREEZE_MANIFEST.md`, `results/README.md`, `git ls-files results`
- 问题描述：`results/README.md` 说实验结果默认不版本化，但仓库实际跟踪了部分历史 diagnostics/action/imitation/policy/runtime 结果；同时若干主线 source CSV 未跟踪。
- 为什么这是问题：版本化策略不一致。读者不清楚哪些结果是发布必需、哪些只是本机历史残留。
- 可能造成的后果：clean clone 与本地状态不一致；复现审查时容易误判哪些文件是权威结果。
- 建议修复方式：重新定义 `results/` 发布策略：只跟踪 freeze manifest 内文件，或扩展 manifest 覆盖全部 tracked results；对非主线历史结果移到 archive 或明确标注。
- 是否需要我确认后再修改：是。涉及删除/取消跟踪文件，必须确认。

### AUD-009

- 严重程度：P2
- 涉及文件和位置：`evaluate_invitation_mask_correction.py:686-715`, `evaluate_execution_channel_mismatch.py` 参数解析与校验
- 问题描述：`evaluate_invitation_mask_correction.py` 只校验 episodes、num_seeds、mask strength/deadband/max_delta/noise，未校验 `num_nodes`、`num_slots`、`num_codebook_states`、`probe_budget`、`rho` 范围、delay 非负、`g_th/alpha_th` 正值等。
- 为什么这是问题：错误参数可能在深层函数中以不清晰方式失败，或更糟糕地产生无意义结果。
- 可能造成的后果：实验脚本被误调用时生成不可解释 CSV；结果文件名看似正常但配置非法。
- 建议修复方式：提取共享 CLI validation helper，覆盖物理参数、预算参数、噪声参数和 list 非空约束；非法配置直接 `ValueError`。
- 是否需要我确认后再修改：是，但这是低风险工程修复。

### AUD-010

- 严重程度：P2
- 涉及文件和位置：`test_env.py:126-138`, `test_env.py:245-270`
- 问题描述：环境定义了 `action_space=[-1,1]^3`，但 `_decode_action` 不主动 clip action。如果外部代码绕过 SB3/Gym action-space 约束直接传入越界 action，`g_th` 和 `alpha_th` 会超出预期范围。
- 为什么这是问题：研究脚本当前多用固定合法 action 或模型输出，风险较低；但环境作为公共 API 时不够稳健。
- 可能造成的后果：dry-run 或新 baseline 写错 action 后产生异常结果而不报错。
- 建议修复方式：在 `step` 或 `_decode_action` 中显式 `np.clip(action, -1, 1)`，并可在 info 中记录是否 clipped。
- 是否需要我确认后再修改：是。虽然不改变现有合法输入结果，但会改变非法输入处理。

## 5. Experimental Validity

### AUD-011

- 严重程度：P2
- 涉及文件和位置：`ms_aircomp/invitation_mask_correction.py:12-25`, `ms_aircomp/invitation_mask_correction.py:51-111`, `evaluate_invitation_mask_correction.py:1-9`
- 问题描述：邀请掩码修正不是只按 aggregate feedback count 调整 cardinality。只要 target count 和 stale count 不同，就丢弃原 stale mask，按 confirmed IRS 下的 stale gain 对所有 remaining nodes 排序，取 top target nodes。
- 为什么这是问题：该机制利用了 stale per-node gain ranking 重新选节点，可能邀请原 stale-valid mask 没有选中的节点，其中包括 stale 视角下原本不满足门限/功率的节点。这是可部署的 stale-CSI reranking，但不是单纯“修正数量”。
- 可能造成的后果：论文若描述为“仅用聚合反馈修复邀请掩码数量”，会低估方法实际使用的信息和机制强度；结果解释可能不够准确。
- 建议修复方式：在方法描述中明确“aggregate count + stale-gain reranking”；补充 ablation：只 prune/add near-boundary nodes、只在 stale-valid set 内调 cardinality、或保留原 mask 并局部替换。
- 是否需要我确认后再修改：是。涉及方法命名、论文描述和可能的补充实验。

### AUD-012

- 严重程度：P2
- 涉及文件和位置：`evaluate_limited_csi_ms_aircomp.py:771-795`, `evaluate_execution_channel_mismatch.py:657-718`, `ms_aircomp/execution_result_summary.py:310`
- 问题描述：limited CSI evaluator 中 `failure_slots` 保存 raw count，execution mismatch evaluator 保存 failure-slot fraction，summary 又乘以 100 得到 rate。两个评估脚本同名指标语义不完全一致。
- 为什么这是问题：如果把两个脚本输出横向比较，`failure_slot_rate` 可能被误读。
- 可能造成的后果：附录/历史结果混入主线表时出现指标单位混乱。
- 建议修复方式：统一字段语义，例如 `failure_slot_count` 和 `failure_slot_rate` 分离；在 CSV field 文档中说明单位。
- 是否需要我确认后再修改：是。涉及历史输出兼容性。

### AUD-013

- 严重程度：P2
- 涉及文件和位置：`train_learned_sparse_shortlist.py`, `train_temporal_deviation_selector.py`, `docs/PAPER_APPENDIX_BOUNDARY.md`
- 问题描述：learned shortlist/temporal deviation 系列使用隐藏标签或 oracle-style 监督信号训练，当前文档把它们放到 diagnostics/appendix 是合理的。但这些脚本仍在主目录入口中，容易被误当作 deployable mainline。
- 为什么这是问题：如果 learned/oracle-supervised 结果进入主文，必须明确训练信息边界，否则会构成不公平比较。
- 可能造成的后果：论文审查时被质疑数据泄漏或 hidden-information baseline。
- 建议修复方式：保持 appendix/diagnostic 标签；在每个相关输出 CSV/MD 增加 `role=diagnostic` 或 `uses_hidden_labels=true` 字段；README 入口列表中更明确区分 mainline/deprecated/diagnostic。
- 是否需要我确认后再修改：是。主要是文档和 metadata 修复。

## 6. Code Quality

### AUD-014

- 严重程度：P2
- 涉及文件和位置：`ms_aircomp/execution_candidates.py:5`, 多个 `ms_aircomp/*` 模块
- 问题描述：可复用包 `ms_aircomp` 仍然 import 顶层脚本 `evaluate_limited_csi_ms_aircomp.py` 作为 `limited` helper。文档中的理想边界是 top-level scripts 调用 `ms_aircomp`，但实际存在反向依赖。
- 为什么这是问题：包边界不清，复用 `ms_aircomp` 时会隐式导入庞大的 evaluator script；未来拆分、测试和安装都更困难。
- 可能造成的后果：小改动容易跨脚本破坏；dependency boundary test 目前只禁止依赖 execution evaluator，没有覆盖 limited evaluator。
- 建议修复方式：把 `effective_channels`、`build_candidate`、`best_candidate`、`grid_indices`、`stable_rng` 等 helper 迁入 `ms_aircomp/limited_csi.py` 或 `ms_aircomp/experiment_utils.py`，顶层脚本反向调用包。
- 是否需要我确认后再修改：是。属于结构性重构，需要分步做。

### AUD-015

- 严重程度：P2
- 涉及文件和位置：`evaluate_execution_channel_mismatch.py`, `evaluate_limited_csi_ms_aircomp.py`, `evaluate_policy_comparison.py`, `train_temporal_deviation_selector.py`
- 问题描述：多个顶层脚本超过千行，CSV/Markdown/plotting/argparse/aggregation/helper 逻辑重复。
- 为什么这是问题：项目已经有 `ms_aircomp` 包，但历史脚本仍然承担大量公共逻辑，增加维护成本。
- 可能造成的后果：修复一个指标或字段时容易漏掉另一个 evaluator；baseline 命名和指标定义逐渐漂移。
- 建议修复方式：按风险分阶段抽公共模块：参数解析/validation、CSV schema、metric aggregation、paper artifact generation。避免一次性重构算法逻辑。
- 是否需要我确认后再修改：是。

### AUD-016

- 严重程度：P3
- 涉及文件和位置：`tests/*.py`, `Makefile`
- 问题描述：测试是脚本式 asserts，文件名不是标准 `test_*.py`，没有 `pytest.ini`、coverage、lint 或 type-check 配置。
- 为什么这是问题：Makefile 能跑，但外部开发者默认运行 `pytest` 可能发现不了测试；静态错误只能靠人工。
- 可能造成的后果：小型回归难以及时发现，IDE/CI 集成不标准。
- 建议修复方式：保留 Makefile，同时添加 pytest wrapper 或重命名测试文件；增加最小 `ruff`/`pyright` 或 `mypy` 检查，先从非算法文件开始。
- 是否需要我确认后再修改：是。

### AUD-017

- 严重程度：P3
- 涉及文件和位置：`.gitignore:6-8`, tracked `.vscode/settings.json`
- 问题描述：`.gitignore` 已忽略 `.vscode/`，但 `.vscode/settings.json` 已被 git 跟踪。
- 为什么这是问题：IDE 本地状态不应混入研究复现仓库，且当前 ignore 策略和 tracked 文件不一致。
- 可能造成的后果：不同开发者的 IDE 设置互相污染；审计时产生 repo hygiene 噪音。
- 建议修复方式：确认后 `git rm --cached .vscode/settings.json`，如确需共享设置则改为 `.vscode/extensions.json` 或文档说明。
- 是否需要我确认后再修改：是。涉及取消跟踪文件。

## 7. Documentation

### AUD-018

- 严重程度：P2
- 涉及文件和位置：`README.md`, `docs/PAPER_FREEZE_MANIFEST.md`, `docs/RESULTS_INDEX.md`
- 问题描述：文档总体非常完整，但没有清楚区分“clean clone 可直接验证的冻结产物”和“当前本机存在的 ignored source artifacts”。`make mainline-audit` 目前依赖本机 source files，这一点未显式说明。
- 为什么这是问题：复现审查者会自然期望 repo clone 后直接 audit，而不是依赖未跟踪本地文件。
- 可能造成的后果：对外展示/提交时出现“我这里能过、别人那里不能过”的问题。
- 建议修复方式：在 freeze manifest 中加一节 `Tracked artifacts` vs `Local source artifacts`；或者把 source artifacts 纳入 tracked freeze set。
- 是否需要我确认后再修改：是。

### AUD-019

- 严重程度：P2
- 涉及文件和位置：`docs/PAPER_TABLE1_MAIN_RESULTS.md`, `docs/PAPER_FIGURE_TABLE_SPECS.md`
- 问题描述：paper-facing table 省略 `Perfect %` 是有说明的，但也省略 CI/seed variability。figure/table specs 没有明确指出 Temporal Deviation Oracle 和 learned variants 的 hidden-information 边界之外的所有 baseline 均为 deployable。
- 为什么这是问题：审稿视角下，oracle/diagnostic/deployable 的界限越清楚越好。
- 可能造成的后果：读者可能误读 Temporal Deviation Oracle 为可部署方法，或低估主方法相对 oracle 的差距。
- 建议修复方式：在 table caption/notes 中增加 `deployable`、`diagnostic`、`hidden-information upper bound` 的统一标签；补充 CI 或 appendix uncertainty table。
- 是否需要我确认后再修改：是。

### AUD-020

- 严重程度：P3
- 涉及文件和位置：`README.md`, `Makefile`
- 问题描述：README 提供了主命令和文档地图，但缺少一个“最小 clean-room 复现命令块”：从环境创建、安装依赖、运行主线 smoke、生成 `/tmp` dry-run、验证 paper artifacts 的完整顺序。
- 为什么这是问题：项目文档多而细，第一次接手者需要自己判断入口。
- 可能造成的后果：复现者可能运行昂贵或会覆盖结果的 target。
- 建议修复方式：增加 `Quick Reproduction Audit` 小节，明确哪些命令只读、哪些会重写 `results/`。
- 是否需要我确认后再修改：是。

## 8. Recommended Fix Plan

必须修：

- AUD-001：修复 execution drift RNG 的 index-stable 采样，并重跑 nonzero execution-error 相关实验。
- AUD-002：确认 stale CSI early-slot 模型；若论文要强调 delay robustness，补充 warmup/prehistory ablation 或明确披露假设。
- AUD-003/AUD-008/AUD-018：统一 frozen artifact 与 source CSV 的版本化策略，确保 clean clone 能复现 audit 或清楚说明 source 不随仓库发布。
- AUD-006/AUD-019：给主表/附录补充 CI、scenario-level variability 或 paired significance，避免结论过强。

强烈建议修：

- AUD-011：把 invitation mask correction 描述为 `aggregate count + stale-gain reranking`，并补一组受限 reranking ablation。
- AUD-004/AUD-005：如果旧 policy comparison 仍会展示，修正 Feature Argmax preview cost 和 Best Fixed IRS test tuning。
- AUD-007：把 Python/依赖环境机器可验证地固定下来。
- AUD-009：补齐 CLI 参数校验。
- AUD-014/AUD-015：逐步把 `evaluate_limited_csi_ms_aircomp.py` 中的公共 helper 迁入 `ms_aircomp` 包，减少反向依赖和重复代码。

可以后续优化：

- AUD-010：环境 action 显式 clip/校验。
- AUD-012：统一历史 evaluator 指标字段单位。
- AUD-013：给 hidden-label learned diagnostics 增加 metadata 标签。
- AUD-016：标准化 pytest/lint/type check。
- AUD-017：清理 tracked IDE 设置。
- AUD-020：增加 clean-room quick reproduction audit 文档。

## 9. Commands Tried

已运行命令和结果：

- `./.venv/bin/python --version`：通过，Python `3.13.2`。
- `./.venv/bin/python -m pip show numpy gymnasium stable-baselines3 torch matplotlib pandas`：通过，关键版本为 NumPy `2.4.4`、Gymnasium `1.2.3`、Stable-Baselines3 `2.8.0`、Torch `2.11.0`、Matplotlib `3.10.9`、Pandas `3.0.2`。
- `make docs`：通过，打印主线 docs、paper table/figure、results index 等路径。
- `make test`：通过，`py_compile` 全部目标成功，`smoke checks passed`。
- `make boundary-test`：通过，`dependency boundary checks passed`。
- `make regression-test`：通过，输出 `coverage frontier: rotating_b4_gap=1.611, coverage_b3_gap=0.672, temporal_oracle_b4_gap=0.521`；`mask correction: gap 0.672 -> 0.417, missed 1.250 -> 0.438`。
- `make mainline-audit`：通过，`mainline artifact checks passed`，检查 artifacts `40`、documented paths `81`、CSV rows `469`。
- `./.venv/bin/python generate_paper_tables.py --source results/execution_mismatch/final_execution_baseline_summary.csv --table1-csv /tmp/audit_table1_main_results.csv --table1-md /tmp/audit_table1_main_results.md`：通过，写入 `/tmp`。
- `./.venv/bin/python generate_paper_figures.py ... --figure* /tmp/...`：通过，写入 `/tmp` 中 Figure 2/3/4 points 与 PNG。
- `./.venv/bin/python evaluate_execution_channel_mismatch.py --episodes 2 --seed 7 --num-seeds 1 --probe-budgets 2 --mismatch-models temporal_ar1 --channel-rho-values 0.9 --csi-delay-slots 1 --decision-error-std-values 0 --execution-error-std-values 0 --policies rotating_feedback_confirm --output-prefix /tmp/audit_execution_mismatch --no-plots`：通过，生成 `/tmp/audit_execution_mismatch.csv`；dry-run summary 为 success `50.000`、perfect `100%`、slots `3.500`、fail `0.00`、miss `2.00`、preview `4.00`、gap `1.667`。
- `./.venv/bin/python evaluate_invitation_mask_correction.py --episodes 2 --num-seeds 1 --seed 7 --channel-rho-values 0.9 --csi-delay-slots 1 --probe-budget 3 --sparse-topk-seed-multiplier 2 --mask-correction-strengths 0,1 --confirmation-feedback-noise-std-values 0 --mask-correction-noise-deadband-z-values 0 --mask-correction-max-delta-values -1 --output-prefix /tmp/audit_invitation_mask --doc-output /tmp/audit_invitation_mask.md`：通过；dry-run 中 uncorrected B3 gap `1.000`，mask-corrected gap `0.667`。
- `git status --short`：通过，无输出；除本报告外，验证阶段没有修改仓库文件。

没有失败的验证命令。辅助搜索时本机没有 `rg`，已改用 `find`/`grep`/`sed`/`nl` 做只读审计。

## 10. Final Assessment

项目已经具备较强的复现实验骨架：主线文档完整，Makefile 检查可用，冻结产物和 paper-facing table/figures 本地一致，核心 smoke/regression/audit 均通过。当前距离“可以交给导师检查/组会展示”已经比较近。

距离“可以严肃写进论文或对外发布复现包”还需要先处理 P1 风险。最关键的是：execution drift RNG 的非零误差实验公平性、stale CSI early-slot 假设、clean clone 对 source artifacts 的可追溯性，以及主表统计不确定性展示。主线 Mask-Corrected B3 结果本身没有在本次审计中发现会直接推翻结论的 P0 问题，但方法描述和 artifact 复现边界需要更严格。

建议先修“复现与结论可信度”问题，再做结构重构。不要先大规模重构算法脚本，否则会增加重新验证成本。
