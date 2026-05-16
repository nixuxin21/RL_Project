# Results Directory

`results/` 保存本地实验生成物。当前项目不按物理移动结果文件区分主线和归档，因为大量文档、Makefile target 和汇总脚本直接引用这些路径。

## Main Results

| 目录 | 用途 |
|---|---|
| `results/execution_mismatch/` | 当前主线：stale/limited CSI、execution mismatch、Sparse-TopK、Adaptive V2、Stale-TopK、Temporal Deviation Oracle |
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
