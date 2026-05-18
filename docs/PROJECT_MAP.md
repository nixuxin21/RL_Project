# Project Map

本文件把当前仓库按“主线代码、研究分支、归档分支、生成物”分层，避免所有入口都堆在 README 里。

## 核心模型

- `test_env.py`: 物理环境 `MSAirCompEnv`。包含信道生成、IRS DFT codebook、动作解码、slot 执行、奖励和 `preview_codebook_index`。
- `train_agent.py`: 完整 SAC，动作是 `[g_th, alpha_th, irs_codebook]`。
- `train_codebook_aware_agent.py`: 固定传输参数的 IRS-only SAC selector。
- `rl_models/`: 本地模型和 VecNormalize 统计。目录默认忽略，已有关键 checkpoint 可作为本机复现实验输入。

## 当前主线

- `docs/MAIN_STORY.md`: 当前研究主线和 active research stack。后续新增实验默认应服务于 stale/limited CSI + execution mismatch + low-cost IRS candidate generation。
- `docs/PAPER_RESULT_PACKAGE.md`: 论文撰写前冻结的正文结果包，明确 main-text methods、主表/图、evidence chain 和 appendix / diagnostics 边界。
- `docs/PAPER_STRUCTURE_MAP.md`: 将冻结结果包映射到未来论文的章节功能、贡献、主表主图和 appendix 位置。
- `docs/PAPER_FIGURE_TABLE_SPECS.md`: 论文主图主表的统一规格，固定 Figure 1、Table 1 和后续图表的展示边界。
- `docs/PAPER_APPENDIX_BOUNDARY.md`: 论文附录最小集合和 supplement-only diagnostics 边界，防止历史实验重新进入正文主线。
- `docs/PAPER_TEXT_OUTLINE.md`: 论文正文最小骨架，逐节固定 claim、证据、禁入内容和图表引用顺序。
- `docs/PAPER_ASSET_GAP_CHECKLIST.md`: 投稿前资产缺口清单，固定哪些表图 ready、source-only、analysis-layer 或 deferred。
- `docs/PAPER_FREEZE_MANIFEST.md`: 论文冻结结果包的 artifact 清单、验证命令和非冻结边界。
- `docs/figures/figure1_system_flow.mmd`: Figure 1 的 Mermaid 可编辑源文件，展示 system mismatch 和 deployable feedback pipeline；当前导出件为 `results/paper/figure1_system_flow.svg` 和 `results/paper/figure1_system_flow.pdf`。
- `generate_paper_tables.py`: 从 frozen mainline CSV、coverage-aware analysis CSV 和 failure-diagnosis CSV 生成论文 Table 1/2/3 artifact；入口是 `make paper-tables`。
- `generate_paper_figures.py`: 从 frozen mainline CSV 生成论文版 Figure 2/3/4 artifact；入口是 `make paper-figures`。
- `summarize_execution_baselines.py`: 从已有 execution mismatch CSV 生成最终主线 baseline 和 learned 诊断汇总表；入口是 `make execution-baseline-summary`。
- `analyze_main_frontier.py`: 从已有 execution mismatch CSV 生成主线机制分析、preview-gap frontier 图和 failed/missed tradeoff 图；入口是 `make main-results-analysis`。
- `analyze_coverage_aware.py`: 从 Coverage-Aware formal weight/power ablation 和 budget split result 生成分析文档、CSV 和图；入口是 `make coverage-aware-analysis`。
- `analyze_invitation_mask_final.py`: 从 invitation-mask formal/noise-aware CSV 生成最终论文表、gap-noise 图和 failed/missed-noise 图；入口是 `make final-invitation-mask-analysis`。
- `evaluate_execution_channel_mismatch.py`: 当前最重要评估框架。覆盖 temporal AR(1) stale CSI、Rotating、Sparse-TopK、Coverage-Aware Sparse-TopK、Adaptive Sparse-TopK v2、Stale-TopK 和 Temporal Deviation Oracle。

## 基础评估

- `evaluate_policy_comparison.py`: 默认基础比较入口。主表聚焦 No IRS、Random IRS、Feature Argmax、Feature Argmax PowerTie 和 Greedy；Feature baselines 的 preview cost 包含 exact codebook-feature acquisition，学习式策略与 validation-selected Fixed IRS 静态消融显式开启。
- `benchmark_policy_runtime.py`: 决策耗时、环境 step 耗时和 preview 调用数 benchmark。
- `diagnose_policy_actions.py`: SAC / Codebook-Aware SAC / Greedy 的动作和 oracle gap 诊断。
- `evaluate_parameter_sweep.py`: `K/N/M/C` 一因子参数扫描。

## 有限观测和错配研究线

- `evaluate_partial_probing_sweep.py`: 每个 slot 只能 preview 少量 codebook 的 partial probing。
- `evaluate_channel_estimation_error_sweep.py`: 决策 preview 用 noisy equivalent channel，执行用真实信道。
- `evaluate_limited_csi_ms_aircomp.py`: 策略基于有限/估计 CSI 邀请节点，真实执行只统计实际成功节点。
- `evaluate_bandit_feedback_ms_aircomp.py`: 严格 aggregate feedback。策略只能 probe 少量 codebook 并看到 noisy aggregate tx/power。
- `evaluate_bandit_feedback_stress_sweep.py`: bandit feedback 的困难场景和 utility 扫描。
- `evaluate_adaptive_feedback_probing.py`: 非学习 Adaptive Rotating Backup。
- `evaluate_execution_channel_mismatch.py`: 决策信道和执行信道分离，覆盖 independent drift、temporal AR(1) stale CSI、feedback confirmation、Sparse-TopK 和 Adaptive Sparse-TopK 等。
- `active_diverse_feedback` / `sparse_topk_feedback` / `coverage_sparse_topk_feedback` / `adaptive_sparse_topk_feedback` / `adaptive_sparse_topk_v2_feedback` / `adaptive_sparse_topk_v3_feedback` / `learned_sparse_shortlist_feedback` / `learned_set_shortlist_feedback`: `evaluate_execution_channel_mismatch.py` 中的低成本 active probe-set baselines，用 sparse stale preview、candidate generation 和 current aggregate feedback 替代 full stale ranking。
- `diagnose_coverage_b3_failures.py`: 当前 `Coverage-Aware B=3 sm=4.1` 主线的 residual oracle gap 诊断，把 gap 分解为 pool、selection、confirmation 和 stale invitation-mask mismatch。
- `evaluate_invitation_mask_correction.py`: 在 confirmed IRS 后用 aggregate current feedback count 设定 target cardinality，并用 confirmed IRS 下的 stale-gain reranking 生成 corrected invitation mask；当前 formal `mc=1` 是同 preview `16` 下的 no-noise trade-off，并支持 high-noise direct correction、clipped failed-invitation diagnostic sweep 和 `global_stale_gain` vs `prune_only` rerank ablation。rerank ablation 显示 `prune_only` 降 failed 但显著增加 missed/gap，因此方法应写成 target-count correction + stale-gain replacement。
- `train_learned_sparse_shortlist.py`: 训练 `learned_sparse_shortlist_feedback` / `learned_set_shortlist_feedback` 使用的线性 ranker，支持 absolute-score、marginal-gain、set-value hidden labels、closed-loop execution-value labels 和 pairwise execution differences；闭环评估不访问 hidden current labels。新生成模型和 diagnostics CSV 写入 `result_role=diagnostic`、`uses_hidden_training_labels=true` 等 metadata。

## 学习式探索和诊断

- `train_greedy_imitation_selector.py`: 监督学习模仿 Greedy IRS index。
- `train_bandit_feedback_selector.py`: feedback-conditioned probing selector。
- `train_temporal_deviation_selector.py`: temporal AR(1) stale-CSI 下学习 probe-window offset；训练目标来自 hidden current-channel outcomes，新生成 checkpoint、诊断 CSV 和 evaluation CSV 写入 diagnostic metadata。
- `train_learned_sparse_shortlist.py`: learned sparse shortlist 诊断。当前 best learned point 仍弱于 Adaptive V2 和 Sparse-TopK sm=3，不作为主方法；新生成 artifacts 必须保留 hidden-label metadata。

这些脚本目前更多承担“验证学习式方法是否有增益”的研究作用；根据现有报告，多个学习式分支是负结果或诊断结果。具体停止投入边界见 `docs/DEPRECATED_DIRECTIONS.md`。

## 归档实验

- `experiments/archive/README.md`: 归档说明。
- `experiments/archive/evaluate_noisy_feature_sweep.py`: exact codebook feature 加噪后的鲁棒性扫描。
- `experiments/archive/train_learned_probing_selector.py`: 早期低维 learned partial probing selector。
- `experiments/archive/evaluate_probing_cost_tradeoff.py`: 早期 partial/learned probing CSV 的 post-hoc cost tradeoff 分析。

归档脚本可复现历史结论，但不应作为新增研究的默认入口。

## 支撑文件

- `README.md`: 面向日常使用的入口说明。
- `docs/PROJECT_STATUS.md`: 当前 active/diagnostic/archive 边界、维护命令和重构前检查入口。
- `docs/MAIN_STORY.md`: 当前主线和后续投入边界。
- `docs/PAPER_RESULT_PACKAGE.md`: 论文前结果包冻结层，作为写论文和判断新实验是否进入正文的默认依据。
- `docs/PAPER_STRUCTURE_MAP.md`: 论文前结构映射层，作为组织 Introduction/Method/Experiments/Discussion 和附录的默认依据。
- `docs/PAPER_FIGURE_TABLE_SPECS.md`: 论文图表规格层，作为 Figure 1、Table 1 和后续主图主表的默认依据。
- `docs/PAPER_APPENDIX_BOUNDARY.md`: 论文附录边界层，作为选择 minimal appendix 与 supplement-only diagnostics 的默认依据。
- `docs/PAPER_TEXT_OUTLINE.md`: 论文正文骨架层，作为正式写作前检查每节 claim 和 evidence traceability 的默认依据。
- `docs/PAPER_ASSET_GAP_CHECKLIST.md`: 论文资产缺口层，作为 manuscript assembly 前检查 Figure 1 导出件、Table 2/3 paper artifacts 和图表版式状态的默认依据。
- `docs/PAPER_FREEZE_MANIFEST.md`: 论文冻结清单层，作为提交 frozen CSV/PNG/MD artifact 前的默认依据。
- `docs/figures/figure1_system_flow.mmd`: Figure 1 系统图源文件；`results/paper/figure1_system_flow.svg` 和 `results/paper/figure1_system_flow.pdf` 是从该源导出的当前 paper-facing 版本，重新导出可用 `make paper-figure1`。
- `docs/PAPER_TABLE1_MAIN_RESULTS.md`: 由 `make paper-tables` 生成的论文 Table 1 Markdown 主表。
- `docs/PAPER_TABLE2_COVERAGE_AWARE_ABLATION.md`: 由 `make paper-tables` 生成的 compact coverage-aware ablation / budget-split 表。
- `docs/PAPER_TABLE3_FAILURE_DIAGNOSIS.md`: 由 `make paper-tables` 生成的 compact B3 failure diagnosis 表。
- `results/paper/figure2_preview_gap_frontier.png`, `results/paper/figure3_failed_missed_tradeoff.png`: 由 `make paper-figures` 生成的论文版 Figure 2/3。
- `results/paper/figure4_invitation_mask_noise_points.csv`, `results/paper/figure4_invitation_mask_gap_noise.png`, `results/paper/figure4_invitation_mask_failed_missed_noise.png`: 由 `make paper-figures` 生成的论文版 Figure 4。
- `EXPERIMENT_REPORT.md`: 当前研究结论和下一阶段路线。
- `docs/BASELINE_STRATEGY.md`: 当前 baseline 分层和主线/补充 baseline 边界。
- `docs/RESULTS_INDEX.md`: 已有关键结果文件索引。
- `docs/MAIN_RESULTS_ANALYSIS.md`: 当前主线结果的论文级分析说明。
- `docs/COVERAGE_AWARE_ANALYSIS.md`: Coverage-Aware Sparse-TopK 的 weight/power ablation、budget split 和主设定选择。
- `docs/INVITATION_MASK_CORRECTION_NOISE.md`: Mask-Corrected Coverage-Aware 的 aggregate-feedback-noise 鲁棒性边界。
- `docs/INVITATION_MASK_CORRECTION_NOISE_AWARE.md`: clipped target-count correction 的 high-noise robustness result。
- `docs/FINAL_INVITATION_MASK_ANALYSIS.md`: invitation-mask correction 的最终论文级表格、图和贡献表述。
- `docs/INVITATION_MASK_RERANK_ABLATION.md`: invitation-mask correction 的 reranking-mode 诊断协议，用于区分 aggregate target-count correction 与 stale-gain replacement 的贡献。
- `docs/DEPRECATED_DIRECTIONS.md`: 负结果和不再继续投入方向的归档索引。
- `Makefile`: 测试、smoke、主实验 target。
- `tests/smoke_checks.py`: 轻量行为回归测试。
- `tests/dependency_boundary_checks.py`: evaluator dependency-boundary 守门测试，禁止重新从 `evaluate_execution_channel_mismatch.py` 导入 reusable helper，并防止已抽出的 policy/output/result helper 回流进 evaluator。
- `tests/mainline_artifact_checks.py`: 主线 artifact 审计，检查关键 CSV/PNG/MD、CSV 字段、`source_file` 链接和 final invitation-mask 结果关系。
- `ms_aircomp/README.md`: reusable experiment layer 的模块边界和 import 规则。
- `ms_aircomp/adaptive_sparse_policies.py`: Adaptive Sparse-TopK v1/v2/v3 的 expansion/history/uncertainty/local-neighbor policy helper。
- `ms_aircomp/channel_models.py`: execution drift、channel snapshot/apply 和 temporal AR(1) stale CSI helper。
- `ms_aircomp/confirmation.py`: current aggregate-feedback IRS confirmation flow。
- `ms_aircomp/execution_candidates.py`: drifted execution candidate 和 hidden execution oracle helper。
- `ms_aircomp/execution_decision_dispatch.py`: execution-mismatch policy 名称到 reusable decision helper 的统一分发层。
- `ms_aircomp/execution_output.py`: execution-mismatch output prefix、progress/summary print 和 plotting helper。
- `ms_aircomp/execution_policies.py`: rotating/stale-topK/sparse/coverage feedback policy decision functions。
- `ms_aircomp/execution_policy_registry.py`: execution-mismatch policy/mismatch 名称、label 和默认配置 registry。
- `ms_aircomp/execution_risk_policies.py`: execution-risk reliability re-scoring、adaptive execution-risk rotating 和 opportunity-cost policy helper。
- `ms_aircomp/execution_result_summary.py`: execution-mismatch per-seed aggregation、CSV 字段、CI summary 和 CSV writer。
- `ms_aircomp/feedback.py`: aggregate feedback 打分和 confirmed index 选择 helper。
- `ms_aircomp/invitation_mask_correction.py`: invitation-mask correction 的 target count、remaining rank 和 mask update 纯函数。
- `ms_aircomp/learned_shortlist.py`: learned sparse/set shortlist 的特征、模型加载、variant scoring 和 deployable feedback policy helper。
- `ms_aircomp/limited_csi.py`: limited-CSI policy 常量、grid 采样、candidate 构造和 slot execution helper。
- `ms_aircomp/probe_sets.py`: ordered/diverse probe set 和 coverage-aware sparse candidate selection helper。
- `ms_aircomp/temporal_policies.py`: temporal-reliability rotating policy 和 temporal-deviation oracle probe-set diagnostic。
- `ms_aircomp/experiment_utils.py`: 新增的无状态公共实验工具函数。

## 生成物边界

- `results/`: 实验 CSV/PNG。默认视为生成物，只保留支撑正式结论的关键文件。
- `rl_logs/`: TensorBoard 日志。默认生成物。
- `.matplotlib/`, `__pycache__/`, `.pytest_cache/`: 本地缓存。
