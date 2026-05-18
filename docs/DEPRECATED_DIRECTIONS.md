# Deprecated Directions

本文件记录当前不建议继续投入的研究方向。这里的 deprecated 不是“错误”或“删除”，而是指它们不再进入主线实验、主表和下一步开发计划。

## Policy

- 不删除已有负结果；它们用于论文讨论、答辩和避免重复探索。
- 不在 README 和 Makefile help 中把这些方向列为默认主线。
- 如需复现，保留脚本、结果和报告引用。
- 新方法必须至少对比 `Rotating B=8`、`Sparse-TopK B=4 sm=3`、`Adaptive V2`、`Stale-TopK B=4` 和 `Temporal Deviation Oracle B=4`。

## Deprecated Or Diagnostic Branches

| 方向 | 代表脚本/结果 | 当前结论 | 保留位置 |
|---|---|---|---|
| Full SAC / Codebook-Aware SAC | `train_agent.py`, `train_codebook_aware_agent.py`, `results/policy_comparison/` | 没有稳定超过规则方法；适合说明普通 RL 不是自然优势 | supplement / diagnostics |
| Fixed IRS | `evaluate_policy_comparison.py --include-fixed-irs-baselines` | 静态 IRS 只能做消融，不适合作主 baseline | appendix |
| Greedy-index imitation | `train_greedy_imitation_selector.py`, `results/imitation/` | 直接模仿 Greedy index 没有超过 Feature Argmax / PowerTie | archive |
| Noisy feature sweep | `experiments/archive/evaluate_noisy_feature_sweep.py`, `results/noisy_features/` | 说明 exact feature 受噪声影响，但不是主贡献 | archive |
| Learned probing selector | `experiments/archive/train_learned_probing_selector.py`, `results/learned_probing/` | 没有稳定超过 Rotating probing | archive |
| Basic bandit feedback | `evaluate_bandit_feedback_ms_aircomp.py`, `evaluate_bandit_feedback_stress_sweep.py` | Rotating feedback probe 仍是强 baseline；UCB/Thompson/MLP 不够强 | diagnostics |
| Adaptive rotating backup | `evaluate_adaptive_feedback_probing.py` | 单次 noisy feedback gate 不能稳定优于 rotating | diagnostics |
| Risk-aware execution heuristics | `evaluate_execution_channel_mismatch.py` risk policies | 降低 failed 往往换来更多 missed opportunities | diagnostics |
| Pure active diversity | `active_diverse_feedback` | gap 小幅下降但 slots/preview 不理想 | negative result |
| Temporal learned offset / DAgger / window | `train_temporal_deviation_selector.py` | 历史 offset/window reranking 不稳定，未超过 rotating | diagnostics |
| Adaptive Sparse-TopK v3 | `adaptive_sparse_topk_v3_feedback` | 固定 history/local-neighbor 降成本但没有改善 gap | negative result |
| Neighbor-Coverage local reallocation | `neighbor_coverage_sparse_topk_feedback` | 同样 preview `16` 下把部分 uniform stale seeds 换成 stale leader local neighbors；修正 temporal prehistory 后仍弱于当前 B3 no-noise gap reference | negative result |
| Learned set-value shortlist | `learned_set_shortlist_feedback` | 离线 regret 低，但闭环不如 absolute learned diagnostic | diagnostics |
| Learned execution-value shortlist | `learned_set_shortlist_feedback` with execution labels | 修正部分目标错配，但仍弱于 absolute learned diagnostic | diagnostics |
| Learned pairwise/cost-aware shortlist | pairwise execution labels | 成本可控但 gap 退化；不应继续在线性 set features 上调标签 | diagnostics |

## Active Replacement

这些 deprecated branches 的主线替代关系如下：

| 不再主推 | 当前替代 |
|---|---|
| Fixed IRS | `Random IRS` 作为基础 IRS baseline |
| SAC / Codebook-Aware SAC | `Feature Argmax PowerTie` 和 execution-mismatch frontier |
| Greedy imitation | `Feature Argmax` / `PowerTie` 规则诊断 |
| Learned probing | `Rotating B=4/B=8` |
| Basic bandit learning | `Rotating Feedback` only as diagnostic; main line uses `Sparse-TopK` |
| Pure diversity candidates | `Sparse-TopK B=4 sm=3` |
| Scalar gate tuning only | `Adaptive Sparse-TopK v2` as continuum, plus candidate-generation analysis |
| Fixed local-neighbor reallocation | `Coverage-Aware B=3 sm=4.1` |
| Linear learned set labels | learned results retained as diagnostics, not main method |

## Revisit Criteria

只有满足下面至少一项时，才值得重新打开 deprecated branch：

1. 新方法在 equal preview 或 lower preview 下超过 `Rotating B=8`。
2. 新方法在相近 preview 下超过 `Sparse-TopK B=4 sm=3` 的 gap。
3. 新方法能解释或显著降低 missed opportunities，而不是只降低 failed invitations。
4. 新方法给出论文中必须使用的机制解释，而不只是多一个数值点。
