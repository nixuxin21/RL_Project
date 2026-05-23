# Results Directory

`results/` 保存本地实验生成物。当前项目不按物理移动结果文件区分主线和归档，因为大量文档、Makefile target 和汇总脚本直接引用这些路径。

## Versioning Policy

默认情况下，新实验结果仍然留在本地，不自动进入仓库。例外是 `docs/PAPER_FREEZE_MANIFEST.md` 中列出的 paper-freeze artifact，以及 `final_execution_baseline_summary.csv` / `main_frontier_analysis.csv` 通过 `source_file` 字段引用的 source CSV。这些 source CSV 必须随 freeze 一起版本化，否则 clean clone 无法追溯主表和主图数字。

`make mainline-audit` 会检查 freeze artifact 和 `source_file` CSV 是否已经被 Git 跟踪或 staged。新增 paper-facing 结果时，需要同时更新 `.gitignore` 的 unignore 规则、`docs/PAPER_FREEZE_MANIFEST.md` 和 audit 检查。

`results/main/` 和 `results/main_analysis/` 可保存 paper-suite 或候选证据包输出。它们仍按本地生成物处理；例如 `docs/candidate_evidence/cost_frontier_main_v1/` 引用的 `cost_frontier_main_v1` 输出，只有在完成 source-artifact promotion 后才应进入 paper-freeze。

当前仓库中还保留了一些历史 diagnostic 结果，例如 `results/action_diagnostics/`、`results/imitation/`、`results/policy_comparison/` 和 `results/runtime/` 下的文件。它们不是当前 paper-freeze 的权威结果；若后续要清理，应先确认对应文档不再引用，再单独取消跟踪。

## Main Results

| 目录 | 用途 |
|---|---|
| `results/execution_mismatch/` | 当前主线：stale/limited CSI、execution mismatch、Sparse-TopK、Adaptive V2、Stale-TopK、Temporal Deviation Oracle |
| `results/main/`, `results/main_analysis/` | paper-suite / candidate evidence 输出，默认本地保留 |
| `results/policy_comparison/` | 早期主 baseline：No IRS、Random IRS、Feature Argmax、PowerTie、Greedy、SAC diagnostics |
| `results/runtime/` | runtime / preview 成本 |
| `results/parameter_sweep/` | 默认环境参数稳定性 |
| `results/partial_probing/` | Rotating probing baseline 证据 |
| `results/channel_estimation/` | noisy decision preview 证据 |
| `results/limited_csi/` | limited CSI framework 证据 |

## Diagnostic Or Archived Results

| 目录 | 状态 |
|---|---|
| `results/action_diagnostics/` | SAC / Codebook-Aware SAC 诊断 |
| `results/imitation/` | Greedy imitation 负结果 |
| `results/noisy_features/` | noisy feature 归档 |
| `results/learned_probing/` | learned probing 归档 |
| `results/bandit_feedback/` | bandit feedback 诊断 |
| `results/probing_cost/` | early post-hoc utility analysis |

主线结论索引见 `docs/RESULTS_INDEX.md`，不再继续投入的方向见 `docs/DEPRECATED_DIRECTIONS.md`。
