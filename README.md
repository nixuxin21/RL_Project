# IRS-Assisted MS-AirComp

本项目研究 IRS 辅助的 multi-slot AirComp 调度。早期工作验证了 IRS、规则式 IRS selection、SAC / Codebook-Aware SAC、partial probing、limited CSI 和 bandit feedback 等方向；当前主线已经收敛为：

> IRS-assisted multi-slot AirComp under stale/limited CSI and execution-channel mismatch, using low-cost IRS candidate generation plus current aggregate feedback.

当前重点不是继续堆叠新的 RL baseline，而是在 stale/limited CSI 和执行阶段信道失配下，研究如何用较低 IRS preview / feedback 成本接近 hidden current-channel oracle。README 只保留日常入口；完整分层见 `docs/PROJECT_MAP.md`，维护状态见 `docs/PROJECT_STATUS.md`，停止投入方向见 `docs/DEPRECATED_DIRECTIONS.md`。

## Quick Start

```bash
make quick-audit
make docs
make help
```

常用入口分三层：

- 验证仓库健康：`make check`, `make mainline-audit`, `make quick-audit`
- 生成论文表图：`make paper-tables`, `make paper-figures`, `make paper-figure1`
- 复现当前主线分析：`make execution-baseline-summary`, `make main-results-analysis`, `make coverage-aware-analysis`, `make final-invitation-mask-analysis`

实验大网格默认不自动运行。正式跑 paper-grade suite 前先用：

```bash
make paper-suite-main-hard-dry-run
```

## Current Mainline

当前主线结果由 `make main-results-analysis` 和 `make execution-baseline-summary` 从已有 CSV 生成。写论文或整理最终表图时，优先使用 `docs/PAPER_RESULT_PACKAGE.md` 中冻结的正文方法集合，并按 `docs/PAPER_STRUCTURE_MAP.md` 组织章节和表图。

跨 9 个 temporal AR(1) `rho/delay` 场景的核心判断：

- `Rotating B=8` 是当前低成本部署 baseline。
- `Sparse-TopK B=4 sm=3` 是 reportable medium-cost baseline。
- `Coverage-Aware B=3 sm=4.1 cw=0.5 cpw=0` 是当前 no-noise same-preview gap reference。
- `Mask-Corrected Coverage-Aware B=3 mc=1` 是同 preview `16` 下的 trade-off result：降低 slots/failed/missed，但 no-noise gap 回升。
- `mc=1 clip=2` 是 high-noise failed-invitation control diagnostic，不替代 high-noise gap-best direct correction。
- `Temporal Deviation Oracle B=4` 使用 hidden current channel，只能作为 temporal diagnostic reference。

当前 active research stack 是：

```text
Rotating B=8
-> Adaptive Sparse-TopK v2
-> Sparse-TopK B=4 sm=3
-> Coverage-Aware Sparse-TopK B=4 cw=0.5 cpw=0
-> Coverage-Aware Sparse-TopK B=3 sm=4.1
-> Mask-Corrected Coverage-Aware B=3
-> Posterior-Guided Count-Refined Coverage-Aware B=3
-> Stale-TopK B=4
-> Temporal Deviation Oracle B=4
```

新增方法必须至少解释或改善 invitation-mask mismatch、same-preview gap、slots/failed/missed trade-off，或 high-noise aggregate feedback robustness；不要只增加一个普通 heuristic 名称。

## Paper Freeze

论文冻结边界以这些文件为准：

| 文件 | 作用 |
|---|---|
| `docs/PAPER_RESULT_PACKAGE.md` | 冻结正文结果包、主表/图和 appendix 边界 |
| `docs/PAPER_STRUCTURE_MAP.md` | 论文章节、贡献、主表主图和附录映射 |
| `docs/PAPER_FIGURE_TABLE_SPECS.md` | Figure 1、Table 1 和后续主图主表规格 |
| `docs/PAPER_APPENDIX_BOUNDARY.md` | 最小附录和 supplement-only diagnostics 边界 |
| `docs/PAPER_TEXT_OUTLINE.md` | 正文最小骨架、claim、证据和禁入内容 |
| `docs/PAPER_ASSET_GAP_CHECKLIST.md` | 投稿前表图资产缺口 |
| `docs/PAPER_FREEZE_MANIFEST.md` | 冻结 artifact 清单和验证命令 |

关键 paper-facing artifacts：

| 输出 | 说明 |
|---|---|
| `docs/PAPER_TABLE1_MAIN_RESULTS.md` | Table 1 Markdown 主表 |
| `results/paper/table1_main_results.csv` | Table 1 CSV |
| `results/paper/table1_scenario_uncertainty.csv` | scenario-level uncertainty |
| `results/paper/table1_paired_scenario_deltas.csv` | paired scenario deltas |
| `results/paper/table2_coverage_aware_ablation.csv` | coverage-aware compact ablation |
| `results/paper/table3_failure_diagnosis.csv` | B3 failure diagnosis compact table |
| `docs/figures/figure1_system_flow.mmd` | Figure 1 Mermaid 源文件 |
| `results/paper/figure1_system_flow.svg` | Figure 1 SVG |
| `results/paper/figure1_system_flow.pdf` | Figure 1 PDF |
| `results/paper/figure2_preview_gap_frontier.png` | preview-gap frontier |
| `results/paper/figure3_failed_missed_tradeoff.png` | failed/missed trade-off |
| `results/paper/figure4_invitation_mask_gap_noise.png` | Figure 4 gap-noise |
| `results/paper/figure4_invitation_mask_failed_missed_noise.png` | Figure 4 failed/missed-noise |

`make mainline-audit` 是 artifact-chain gate。它不重跑 formal experiments，只检查 freeze artifact、CSV 字段、`source_file` 链接、paper-facing 文本和文档引用路径。

## Commands

基础验证：

```bash
make test
make lint
make pytest-test
make check
make mainline-audit
make quick-audit
```

当前主线和论文资产：

```bash
make execution-baseline-summary
make main-results-analysis
make coverage-aware-analysis
make coverage-b3-failure-diagnosis
make invitation-mask-correction-formal
make invitation-mask-correction-noise-aware-formal
make final-invitation-mask-analysis
make paper-tables
make paper-figures
```

Paper experiment suite：

```bash
make paper-suite-smoke
make paper-suite-analyze-smoke
make paper-suite-main-hard-dry-run
```

历史 baseline、learned variants、bandit feedback、SAC / imitation 和其他负结果默认属于 diagnostics / archive。它们保留用于论文讨论、答辩和避免重复探索；入口见 `make help` 的 diagnostic section、`docs/DEPRECATED_DIRECTIONS.md` 和 `docs/RESULTS_INDEX.md`。

## Repository Map

| 路径 | 当前角色 |
|---|---|
| `test_env.py` | `MSAirCompEnv` 物理环境、DFT codebook、slot 执行和 preview API |
| `ms_aircomp/` | reusable experiment layer；新 helper 优先放这里 |
| `evaluate_execution_channel_mismatch.py` | 当前主线 execution-mismatch evaluator shell |
| `evaluate_invitation_mask_correction.py` | invitation-mask correction evaluator |
| `summarize_execution_baselines.py`, `analyze_*.py`, `generate_paper_*.py` | paper / mainline artifact layer |
| `experiments/` | paper-grade suite 和物理归档脚本 |
| `results/` | 实验生成物；默认本地，paper-freeze artifact 例外 |
| `tests/` | smoke、dependency boundary、regression、artifact audit 和 validation checks |

顶层仍保留若干历史训练/评估脚本，是为了保持旧命令、报告引用和 smoke checks 兼容。逻辑归档边界见 `docs/PROJECT_STATUS.md` 和 `docs/DEPRECATED_DIRECTIONS.md`。

## Environment

环境约束见 `.python-version`、`pyproject.toml`、`requirements.txt`、`requirements-lock.txt` 和 `docs/ENVIRONMENT.md`。复现优先安装完整锁定版本：

```bash
python3.13 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements-lock.txt
make quick-audit
```

默认物理设定：

- 节点数 `K=50`
- 时隙数 `N=10`
- IRS 单元数 `M=64`
- IRS DFT codebook 大小 `C=16`
- 噪声方差 `noise_var=1e-9`
- 最大发射功率 `P_max=1.0 W`
- 默认固定门限 `g_th=0.001`, `alpha_th=0.05`

## Quick Reproduction Audit

Clean-room reviewer check from a fresh clone:

```bash
python3.13 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements-lock.txt
make quick-audit
```

`make quick-audit` runs:

| Step | Command | Writes |
|---|---|---|
| Compile, lint and pytest wrappers | `make check` | Python cache files only |
| Frozen artifact/index audit | `make mainline-audit` | Python cache files only |
| Tiny execution-mismatch dry-run | `make quick-audit-dry-run` | `/tmp/rl_project_quick_execution_mismatch.csv` only |

This is the recommended first-pass reproducibility audit. It does not rerun the formal experiments and does not rewrite `results/`. Use the larger experiment targets only when intentionally regenerating specific result families.

## Result Boundaries

- `results/` 默认是本地生成物；paper-freeze artifacts 和 summary `source_file` CSV 必须随仓库版本化，具体边界见 `docs/PAPER_FREEZE_MANIFEST.md` 和 `results/README.md`。
- `results/execution_mismatch/` 包含当前主线和主要 diagnostics，不能按“生成物”一概清理。
- `results/policy_comparison/`, `results/runtime/`, `results/parameter_sweep/`, `results/partial_probing/`, `results/channel_estimation/`, `results/limited_csi/` 支撑背景和 appendix。
- `results/action_diagnostics/`, `results/imitation/`, `results/noisy_features/`, `results/learned_probing/`, `results/bandit_feedback/`, `results/probing_cost/` 默认是 diagnostics / archive。

## Continuous Checks

GitHub Actions runs the same lightweight gate on push and pull request:

```text
.github/workflows/checks.yml -> make quick-audit PYTHON=python
```

The workflow installs `requirements.txt` on the Python version pinned by `.python-version`, then runs compile/lint/pytest, artifact-chain audit and the tiny `/tmp` execution-mismatch dry-run. It intentionally does not rerun formal experiment sweeps or regenerate tracked result artifacts.
