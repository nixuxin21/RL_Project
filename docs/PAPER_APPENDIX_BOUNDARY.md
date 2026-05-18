# Paper Appendix Boundary

本文件冻结论文附录和 diagnostics 的最小使用边界。它不是附录正文，也不是历史结果索引；它的作用是防止写作阶段把历史实验重新混入 main text。

正文主线、主表和主图以 `docs/PAPER_RESULT_PACKAGE.md`、`docs/PAPER_STRUCTURE_MAP.md` 和 `docs/PAPER_FIGURE_TABLE_SPECS.md` 为准。完整历史结果索引仍见 `docs/RESULTS_INDEX.md`，停止投入方向见 `docs/DEPRECATED_DIRECTIONS.md`。

## Policy

- Main text 只报告 frozen main-text methods、Table 1、Figure 1-4、coverage-aware ablation、failure diagnosis 和 invitation-mask correction。
- Appendix 只服务于三件事：说明 baseline 背景、解释 preview / CSI / feedback 约束、交代为什么不继续 RL / learned / ordinary heuristic 分支。
- Appendix 不再引入新的主方法排名，不重新开启历史 policy-comparison 主表。
- 除非 reviewer 明确要求，appendix tables 控制在 3 张以内。
- 所有 appendix 结论必须回到当前主线：stale CSI、execution-channel mismatch、low-cost IRS candidate generation、aggregate feedback 和 invitation-mask correction。

## Default Appendix Minimum

默认附录只保留下面三组证据。它们足够支撑背景和负结果边界，不会把论文改写成历史实验报告。

| Appendix Item | Purpose | Primary Artifacts | Default Form |
|---|---|---|---|
| Appendix A: Historical baseline context | 说明早期 full-preview / rule-based / SAC baselines 的背景，避免 reviewer 误以为没有比较学习式方法。 | `results/policy_comparison/policy_comparison_summary_ep1000_runs5_seed2026_featargmax_powertie_cbsac.csv`, `results/policy_comparison/policy_comparison_results_ep1000_runs5_seed2026_featargmax_powertie_cbsac.png`, `docs/BASELINE_STRATEGY.md` | One compact table or paragraph. Do not move these methods into Table 1. |
| Appendix B: Preview-cost and CSI motivation | 说明为什么本文从 full preview 转向 limited preview、stale CSI 和 execution mismatch。 | `results/runtime/runtime_benchmark_ep200_seed2026.csv`, `results/probing_cost/probing_cost_tradeoff_slot0-0p05-0p1-0p2_preview0-0p0005-0p001-0p002-0p005-0p01-0p02-0p05_winners.csv`, `results/channel_estimation/channel_estimation_error_sweep_ep1000_runs5_seed2026_err0-0p02-0p05-0p1-0p2-0p3.csv`, `results/limited_csi/limited_csi_main_ep1000_runs5_seed2026.csv` | One compact table plus short text. |
| Appendix C: Diagnostic branches not used as proposed methods | 说明 aggregate-feedback learning、learned shortlist 和 ordinary heuristic extensions 没有超过 frozen mainline。 | `results/bandit_feedback/bandit_feedback_stress_formal_ep1000_runs5_seed2026.csv`, `results/execution_mismatch/learned_sparse_shortlist_pilot_ep100_runs2_rho0p7-0p9-0p98_delay1-2-3_b4_ex1-2.csv`, `results/execution_mismatch/learned_pairwise_shortlist_pilot_ep100_runs2_rho0p7-0p9-0p98_delay1-2-3_b4_maxex1-2_fw0p5_mw0p5_pc0p005.csv`, `docs/DEPRECATED_DIRECTIONS.md` | One summary table. Keep detailed per-variant results in `docs/RESULTS_INDEX.md`. |

## Supplement Only On Demand

下面这些结果默认不进入正文，也不进入最小附录。只有当 reviewer specifically asks for the branch 时，才作为 supplement 或 response material 使用。

| Branch | Representative Artifacts | Reason To Keep Out By Default |
|---|---|---|
| Action diagnostics | `results/action_diagnostics/action_diagnostics_ep1000_runs1_seed2026_summary.csv`, `results/action_diagnostics/action_diagnostics_ep1000_runs1_seed2026_steps.csv` | 解释 SAC action behavior，但不服务于当前 mainline figure/table。 |
| Greedy imitation | `results/imitation/greedy_imitation_train5000_eval1000_seed2026_eval_summary.csv`, `results/imitation/greedy_imitation_train5000_eval1000_seed2026_train_history.csv` | 早期 imitation 诊断，和 current aggregate-feedback method 无直接关系。 |
| Noisy feature sweep | `results/noisy_features/` | 早期 exact-feature robustness 背景，不是 current stale-CSI execution mismatch evidence。 |
| Learned probing selector | `results/learned_probing/` | 没有稳定超过 rotating probing，已归档。 |
| Full learned-shortlist label variants | `docs/RESULTS_INDEX.md` | 可作为 detailed diagnostics，但正文只需要一行说明 learned shortlist 没超过 frozen mainline。 |
| Adaptive v1/v3 and neighbor-coverage details | `docs/RESULTS_INDEX.md`, `docs/DEPRECATED_DIRECTIONS.md` | 支撑“ordinary candidate-generation heuristic is saturated”，不应扩展成新主线。 |

## Main-Text Exclusion Rules

以下内容不得进入正文主表或作为 proposed method 叙述：

1. `Rotating B=4` 不能作为主要 baseline；正文 low-cost baseline 是 `Rotating B=8`。
2. `Adaptive Sparse-TopK v2` 只能作为 cost-quality continuum / appendix point，不能写成 proposed method。
3. SAC、Codebook-Aware SAC、imitation 和 learned probing 只能作为 background / supplement diagnostics。
4. Learned shortlist variants 不能作为正文主方法；当前结论是 positive diagnostic but below frozen mainline。
5. Bandit feedback learning 不能替代 current aggregate-feedback confirmation；当前结果只支持它作为 diagnostic。
6. `Temporal Deviation Oracle B=4` 可进入正文作为 hidden-information temporal diagnostic reference，但必须明确不可部署。
7. `mc=1 clip=2` 是 failed-invitation control diagnostic，不替代 high-noise gap-best direct correction，也不替代 Table 1 的 `Mask-Corrected Coverage-Aware B=3 mc=1` trade-off row。

新生成的 learned-shortlist、learned temporal-deviation 和 oracle-style diagnostic CSV/model artifacts 必须带机器可读边界字段：`result_role`、`uses_hidden_training_labels`、`inference_uses_hidden_current_csi` 和 `supervision_signal`。`uses_hidden_training_labels=true` 表示训练目标来自 hidden current-channel outcome；`inference_uses_hidden_current_csi=true` 表示该结果只能作为 hidden-information temporal diagnostic reference。

## Reopen Criteria

只有满足至少一项，历史分支才值得从 appendix / diagnostics 回到 active research：

1. 在相同或更低 preview 下超过 no-noise gap reference `Coverage-Aware B=3 sm=4.1 cw=0.5 cpw=0`，或在 slots/failed/missed/gap trade-off 上明确优于 `Mask-Corrected Coverage-Aware B=3 mc=1`。
2. 直接解释或降低 invitation-mask mismatch，而不是只增加一个候选生成 heuristic。
3. 在 high-noise aggregate feedback 下明显优于 direct `mc=1` 的 gap，或优于 `mc=1 clip=2` 的 failed-invitation control。
4. 提供论文主张必须依赖的新机制，而不是只增加一个数值点。

## Writing Checklist

写作或投稿前按下面顺序使用 appendix 资料：

1. 先完成正文 Table 1 和 Figure 1-4。
2. 再写 Appendix A/B/C 的最小三组证据。
3. 最后才按 reviewer 需求从 `docs/RESULTS_INDEX.md` 抽取 supplement-only diagnostics。
4. 不把 appendix 结果重新加入正文方法排序。
5. 跑 `make mainline-audit`，确认 appendix boundary 和引用路径仍能被审计。
