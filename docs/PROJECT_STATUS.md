# Project Status

本文件面向项目维护，不是论文草稿。它记录当前仓库哪些部分是 active mainline、哪些是 diagnostics/archive，以及继续实验前应跑哪些检查。

## Current Focus

当前主线是：

> IRS-assisted multi-slot AirComp under stale/limited CSI and execution-channel mismatch, using low-cost IRS candidate generation plus current aggregate feedback.

当前 no-noise same-preview gap reference 是 `Coverage-Aware B=3 sm=4.1 cw=0.5 cpw=0`。`Mask-Corrected Coverage-Aware B=3 mc=1` 是同 preview trade-off method：它保持 B3 的 IRS candidate generation 和 current aggregate confirmation 不变，只在 confirmed IRS 后用 aggregate current feedback count 设定 target cardinality，并用 confirmed IRS 下的 stale-gain reranking 生成 corrected invitation mask；修正后的 temporal prehistory 结果显示它降低 slots/failed/missed，但不降低 no-noise gap。

当前最重要的维护目标不是继续扩展普通 heuristic 数量，而是让这条主线更可靠、更可复现、更容易重构。

论文撰写前的主线结果包已冻结在 `docs/PAPER_RESULT_PACKAGE.md`。章节、表图和附录位置已映射到 `docs/PAPER_STRUCTURE_MAP.md`，主图主表规格已统一固定在 `docs/PAPER_FIGURE_TABLE_SPECS.md`，附录最小集合已固定在 `docs/PAPER_APPENDIX_BOUNDARY.md`，正文最小写作骨架已固定在 `docs/PAPER_TEXT_OUTLINE.md`，投稿前资产缺口已固定在 `docs/PAPER_ASSET_GAP_CHECKLIST.md`，冻结 artifact 清单已固定在 `docs/PAPER_FREEZE_MANIFEST.md`。后续写论文、补图表或判断新实验是否进入正文时，优先以这些文件为准。

## Repository Readiness

非论文侧当前维护状态：

- 标准本地门槛是 `make check`、`make mainline-audit` 和 `make quick-audit`。
- 2026-05-18 已用 `/tmp` fresh clone + 独立 Python `3.13.2` venv 验证 `requirements.txt` 可安装并通过 `make quick-audit`。
- `.github/workflows/checks.yml` 已加入 CI，在 push 和 pull request 上运行同一套 `make quick-audit PYTHON=python`。
- 结果发布边界由 `.gitignore`、`docs/PAPER_FREEZE_MANIFEST.md`、`results/README.md` 和 `tests/mainline_artifact_checks.py` 共同约束；clean clone 不应依赖本机 ignored result 文件。
- `docs/CODE_EXPLANATION_CN.md` 已加入作为非论文侧中文代码读本，覆盖核心概念、模块职责、脚本分层、测试边界和后续扩展流程。

## Cleanup Policy

当前整理策略是先做入口和索引瘦身，而不是大规模物理移动文件。顶层仍保留若干历史训练、评估和诊断脚本，是为了兼容 Makefile、smoke checks、历史报告引用和旧复现实验命令。

默认入口分层如下：

- `README.md`: 只放最短日常入口、当前主线、paper-freeze 边界和复现审计。
- `make help`: 展示可运行 target，但把 current mainline、supporting baseline 和 diagnostic/archive 分开。
- `make docs`: 展示文档入口和 paper-facing artifact，不再充当完整结果索引。
- `docs/PROJECT_MAP.md`: 维护完整结构说明和脚本分层。
- `docs/RESULTS_INDEX.md`: 维护历史结果和 artifact 证据链。

后续若要物理移动顶层脚本，必须先同步更新 `Makefile`、`README.md`、`docs/PROJECT_MAP.md`、`tests/smoke_checks.py` 和相关历史引用，并运行 `make check` 与 `make mainline-audit`。本地证据草稿目录，例如 `docs/paper_evidence/`，默认由 `.gitignore` 忽略；只有在同步 source artifacts、freeze manifest、results index 和审计检查后，才应显式 promote 到发布边界。

已解决的历史审计报告归档在 `docs/archive/AUDIT_REPORT.md`。它保留问题来源和修复记录，但不再作为当前维护入口；当前状态以本文件、`docs/PROJECT_MAP.md`、`docs/PAPER_FREEZE_MANIFEST.md` 和 `results/README.md` 为准。

## Active Code

| 文件 | 状态 | 用途 |
|---|---|---|
| `test_env.py` | active core | `MSAirCompEnv` 物理环境、DFT codebook、slot 执行和 preview API |
| `ms_aircomp/adaptive_sparse_policies.py` | active utility | Adaptive Sparse-TopK v1/v2/v3 gates、history/uncertainty helper 和 local-neighbor generation |
| `ms_aircomp/channel_models.py` | active utility | execution drift、physical channel snapshot、temporal AR(1) stale CSI helpers |
| `ms_aircomp/confirmation.py` | active utility | current aggregate-feedback IRS confirmation flow |
| `ms_aircomp/experiment_utils.py` | active utility | seed、action mapping、energy 等无状态工具 |
| `ms_aircomp/execution_candidates.py` | active utility | drifted execution candidates and hidden execution oracle helpers |
| `ms_aircomp/execution_policies.py` | active utility | rotating/stale-topK/sparse/coverage feedback policy decision functions |
| `ms_aircomp/execution_risk_policies.py` | active utility | execution-risk reliability re-scoring、adaptive execution-risk rotating 和 opportunity-cost policy helpers |
| `ms_aircomp/feedback.py` | active utility | aggregate feedback scoring and confirmed-index selection helpers |
| `ms_aircomp/learned_shortlist.py` | active utility | learned sparse/set shortlist features、model loading 和 feedback policy helpers |
| `ms_aircomp/probe_sets.py` | active utility | ordered/diverse probe sets and coverage-aware sparse candidate selection |
| `ms_aircomp/temporal_policies.py` | active utility | temporal-reliability rotating policy 和 temporal-deviation oracle diagnostic |
| `evaluate_execution_channel_mismatch.py` | active mainline | temporal AR(1) stale CSI、execution mismatch、Sparse-TopK、Coverage-Aware、Adaptive V2、Stale-TopK、Temporal Deviation Oracle |
| `evaluate_invitation_mask_correction.py` | active mainline | aggregate-feedback invitation-mask correction 与 noise-aware clipped correction |
| `diagnose_coverage_b3_failures.py` | active diagnostic | 把 `Coverage-Aware B=3` residual gap 分解到 pool/selection/confirmation/invitation |
| `summarize_execution_baselines.py` | active summary | 汇总当前 execution mismatch baseline |
| `analyze_main_frontier.py` | active summary | 生成主线 frontier 分析和图 |
| `analyze_coverage_aware.py` | active summary | 汇总 coverage-aware ablation 和 budget split |
| `analyze_invitation_mask_final.py` | active summary | 汇总 final invitation-mask results |
| `tests/smoke_checks.py` | active test | 行为不变量、预算、no-side-effect、seed stability |
| `tests/dependency_boundary_checks.py` | active test | 防止从 evaluator 重新导入 reusable helper，固化 `ms_aircomp` 边界 |
| `tests/mainline_artifact_checks.py` | active test | 不重跑实验，审计主线 CSV/PNG/MD artifact 证据链 |
| `tests/mainline_regression_checks.py` | active test | 固定 seed 的主线数值趋势回归检查 |
| `tests/test_project_checks.py` | active test | 标准 pytest wrapper，调用现有脚本式 checks |

## Diagnostic Or Archive Code

以下方向保留用于复现历史结论和解释负结果，但不应作为默认新增实验入口：

- Full SAC / Codebook-Aware SAC: `train_agent.py`, `train_codebook_aware_agent.py`
- Greedy imitation: `train_greedy_imitation_selector.py`
- Basic bandit feedback and learned feedback selector: `evaluate_bandit_feedback_ms_aircomp.py`, `train_bandit_feedback_selector.py`
- Temporal learned offset/window: `train_temporal_deviation_selector.py`
- Learned sparse/set shortlist: `train_learned_sparse_shortlist.py`
- Early learned probing and noisy feature sweeps: `experiments/archive/`

更完整的停止投入边界见 `docs/DEPRECATED_DIRECTIONS.md`。

## Module Boundary

`ms_aircomp/` 是逐步稳定的 reusable experiment layer。新代码应优先从这些模块导入 channel、execution candidate、feedback、confirmation、probe-set、active policy、learned-shortlist、execution policy registry、result summary 和 plotting/output helpers。每个模块用 `__all__` 标明当前项目内希望稳定使用的函数。

`evaluate_execution_channel_mismatch.py` 现在主要承担 CLI parsing/validation、temporal scenario orchestration、`evaluate_policy()` episode loop 和 legacy compatibility re-export surface。CSV 写出、plot/summary 输出、policy registry、metric summary、policy decision dispatch 和 invitation-mask correction 已移入 `ms_aircomp` helper 模块。新增代码不应使用 `from evaluate_execution_channel_mismatch import ...` 导入 helper。整体 import evaluator 只允许少数编排入口，例如 `train_temporal_deviation_selector.py` 和 `tests/mainline_regression_checks.py`。

`evaluate_limited_csi_ms_aircomp.py` 现在也只作为 limited-CSI evaluation runner 和 reporting script。Reusable limited-CSI constants、grid selection、candidate construction、risk helpers、environment factory 和 slot execution helper 都应从 `ms_aircomp.limited_csi` 导入；除测试历史 summary/reporting surface 外，不应把顶层 limited evaluator 当 helper module import。

这个边界由 `tests/dependency_boundary_checks.py` 固化；如果后续有人重新从 evaluator 导入 helper、在 evaluator 中新增顶层 reusable helper、或让已拆出的 policy/output/result helper 回流进 evaluator，`make check` 会失败。

后续继续拆其余 orchestration logic 时，应保持依赖方向为：

```text
experiment scripts -> ms_aircomp reusable modules
```

不要让 `ms_aircomp` 模块 import orchestration scripts。

## Safe Maintenance Commands

日常改代码后优先运行：

```bash
make test
make lint
make pytest-test
```

涉及 current mainline、execution mismatch、coverage-aware 或 mask correction 的改动后运行：

```bash
make check
```

其中 `make check` 包含：

- `py_compile`: major scripts and reusable modules
- `make lint`: high-signal Ruff rules for syntax/undefined-name failures
- `make pytest-test`: standard pytest wrappers for smoke, boundary, regression, and mainline artifact checks

传统单项入口仍保留：

- `make test`: py_compile + smoke checks
- `make boundary-test`: evaluator dependency-boundary checks
- `make regression-test`: fixed-seed mainline numerical checks

需要更完整的一次性 smoke 时运行：

```bash
make smoke
```

`make smoke` 会跑多个小规模实验，比 `make check` 更慢。

整理主线结果、论文表格或提交前运行：

```bash
make mainline-audit
```

`make mainline-audit` 不重跑实验，只检查当前主线结果文件、CSV 字段、`source_file` 链接、final invitation-mask 关系和 `docs/RESULTS_INDEX.md` 中引用的路径是否一致。

## Main Result Entrypoints

当前主线结果和分析入口：

| Target | 输出 | 用途 |
|---|---|---|
| `docs/PAPER_RESULT_PACKAGE.md` | paper-facing frozen package | 冻结正文结果包、主表/图和 appendix 边界 |
| `docs/PAPER_STRUCTURE_MAP.md` | paper-facing structure map | 映射论文章节、贡献、主表主图和附录位置 |
| `docs/PAPER_FIGURE_TABLE_SPECS.md` | paper-facing figure/table specs | 固定 Figure 1、Table 1 和后续主图主表规格 |
| `docs/PAPER_APPENDIX_BOUNDARY.md` | paper-facing appendix boundary | 固定 minimal appendix、supplement-only diagnostics 和禁入正文规则 |
| `docs/PAPER_TEXT_OUTLINE.md` | paper-facing text outline | 固定每节 claim、证据、禁入内容和图表引用顺序 |
| `docs/PAPER_ASSET_GAP_CHECKLIST.md` | paper-facing asset checklist | 固定投稿前表图资产完成度、缺口和后续包装任务 |
| `docs/PAPER_FREEZE_MANIFEST.md` | paper-facing freeze manifest | 固定论文冻结 artifact 清单、验证命令和非冻结边界 |
| `make paper-tables` | `docs/PAPER_TABLE1_MAIN_RESULTS.md`, `docs/PAPER_TABLE1_UNCERTAINTY.md`, `docs/PAPER_TABLE2_COVERAGE_AWARE_ABLATION.md`, `docs/PAPER_TABLE3_FAILURE_DIAGNOSIS.md`, `results/paper/table1_main_results.csv`, `results/paper/table1_scenario_uncertainty.csv`, `results/paper/table1_paired_scenario_deltas.csv`, `results/paper/table2_coverage_aware_ablation.csv`, `results/paper/table3_failure_diagnosis.csv` | 从 frozen mainline CSV、coverage-aware analysis CSV 和 failure-diagnosis CSV 生成论文 Table 1/2/3、scenario-level uncertainty 和 paired deltas |
| `make paper-figures` | `results/paper/figure2_figure3_points.csv`, `results/paper/figure2_preview_gap_frontier.png`, `results/paper/figure3_failed_missed_tradeoff.png`, `results/paper/figure4_invitation_mask_noise_points.csv`, `results/paper/figure4_invitation_mask_gap_noise.png`, `results/paper/figure4_invitation_mask_failed_missed_noise.png` | 从 frozen mainline CSV 生成论文版 Figure 2/3/4 |
| `make execution-baseline-summary` | `docs/EXECUTION_BASELINE_SUMMARY.md`, `results/execution_mismatch/final_execution_baseline_summary.csv` | 当前 baseline 主表 |
| `make main-results-analysis` | `docs/MAIN_RESULTS_ANALYSIS.md`, frontier CSV/PNG | cost-quality frontier 和 failed/missed tradeoff |
| `make coverage-aware-analysis` | `docs/COVERAGE_AWARE_ANALYSIS.md` | coverage-aware weight/power/budget split |
| `make coverage-b3-failure-diagnosis` | diagnosis CSV/MD | B3 residual gap 分解 |
| `make invitation-mask-correction-formal` | formal correction CSV/MD | reliable feedback 下的 no-noise trade-off result |
| `make invitation-mask-correction-noise-aware-formal` | noise-aware CSV/MD | high-noise direct correction 和 clipped failed-invitation diagnostic |
| `make final-invitation-mask-analysis` | final CSV/PNG/MD | final invitation-mask result package |

## Results Boundaries

`results/` 是生成物目录，不按物理移动方式区分 active/archive，因为文档和 Makefile target 仍直接引用历史路径。维护时按以下规则理解：

- `results/execution_mismatch/`: 当前主线和主要 diagnostics。
- `results/policy_comparison/`, `results/runtime/`, `results/parameter_sweep/`, `results/partial_probing/`, `results/channel_estimation/`, `results/limited_csi/`: 支撑背景结论。
- `results/action_diagnostics/`, `results/imitation/`, `results/noisy_features/`, `results/learned_probing/`, `results/bandit_feedback/`, `results/probing_cost/`: diagnostic/archive。

关键结果索引见 `docs/RESULTS_INDEX.md`。

## Before Adding New Experiments

新增实验应先满足下面至少一项：

1. 直接解释或降低 invitation-mask mismatch。
2. 在相同或更低 preview 下改善 `Coverage-Aware B=3 sm=4.1` 的 no-noise gap，或改善 `Mask-Corrected Coverage-Aware B=3` 的 slots/failed/missed/gap trade-off。
3. 提供对 failed/missed/oracle gap 的新诊断，而不只是新增一个策略名称。
4. 改善 high-noise aggregate feedback robustness。

新增实验默认应对比：

- `Rotating B=8`
- `Sparse-TopK B=4 sm=3`
- `Coverage-Aware B=3 sm=4.1 cw=0.5 cpw=0`
- `Mask-Corrected Coverage-Aware B=3 mc=1`
- `Stale-TopK B=4`
- `Temporal Deviation Oracle B=4`

## Refactor Readiness

`evaluate_execution_channel_mismatch.py` 已从大型混合脚本收敛为较薄的主线 evaluator shell。当前已抽出的稳定边界包括：

- `ms_aircomp/execution_policy_registry.py`: policy/mismatch 名称、label 和默认配置。
- `ms_aircomp/execution_result_summary.py`: per-seed aggregation、CSV 字段和 summary statistics。
- `ms_aircomp/execution_decision_dispatch.py`: policy name 到 reusable decision helper 的分发。
- `ms_aircomp/execution_output.py`: output prefix、progress/summary print 和 plotting。
- `ms_aircomp/invitation_mask_correction.py`: invitation-mask correction 的纯函数核心。

后续不建议继续为了“变小”而拆分。只有在实际修改触碰对应区域时，再考虑把 CLI validation/schema 或 episode metric accumulator 抽成 helper。每次拆分后先跑 `make check`，再考虑扩大实验。
