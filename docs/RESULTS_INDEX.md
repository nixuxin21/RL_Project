# Results Index

本文件记录当前仓库中最重要的结果文件及其支撑的结论。完整论证仍以 `EXPERIMENT_REPORT.md` 为准。

当前主线故事见 `docs/MAIN_STORY.md`。不再继续投入的负结果和诊断分支见 `docs/DEPRECATED_DIRECTIONS.md`。
当前主结果机制分析见 `docs/MAIN_RESULTS_ANALYSIS.md`。
论文冻结 artifact 清单见 `docs/PAPER_FREEZE_MANIFEST.md`。
该 freeze manifest 同时列出 summary `source_file` 引用的 source CSV；这些文件属于 clean-clone 复现边界，不应只存在于本机 ignored results 中。
注意：temporal AR(1) 代码已修正为支持 prehistory states，避免 early slots 使用 current CSI 作为 stale CSI。该修复会改变 temporal formal sweep 数值；旧冻结 CSV 在重跑前只能说明 artifact-chain 一致性，不能作为最终论文数字。
论文撰写前的冻结结果包见 `docs/PAPER_RESULT_PACKAGE.md`，章节和表图组织见 `docs/PAPER_STRUCTURE_MAP.md`，主图主表规格见 `docs/PAPER_FIGURE_TABLE_SPECS.md`，附录最小集合见 `docs/PAPER_APPENDIX_BOUNDARY.md`，正文最小骨架见 `docs/PAPER_TEXT_OUTLINE.md`，投稿前资产缺口见 `docs/PAPER_ASSET_GAP_CHECKLIST.md`；正文主表、主图和 appendix 边界以这些文件为准。

## 主策略比较

- `results/policy_comparison/policy_comparison_summary_ep1000_runs5_seed2026_featargmax_powertie_cbsac.csv`
- `results/policy_comparison/policy_comparison_results_ep1000_runs5_seed2026_featargmax_powertie_cbsac.png`

支撑结论：`Feature Argmax PowerTie IRS` 在默认设定下匹配 `Greedy IRS` 的 coverage、latency 和 energy。注意该历史 baseline 依赖 C 维 exact codebook quality features；公平 preview 成本应计入 feature acquisition 的 full-codebook cost，再加 tied candidates 的额外功率 preview。

Baseline 叙事：正式主表建议聚焦 `No IRS`、`Random IRS`、`Greedy IRS`、`Feature Argmax IRS` 和 `Feature Argmax PowerTie IRS`。该历史 CSV 还包含 SAC、Codebook-Aware SAC、Fixed IRS 等补充/消融结果。

## Runtime

- `results/runtime/runtime_benchmark_ep200_seed2026.csv`

支撑结论：`Feature Argmax IRS` 和 `Feature Argmax PowerTie IRS` 的神经/规则决策开销很低，但 exact codebook features 本身需要 full-codebook acquisition。runtime 表只能用于说明计算开销，不应单独作为 sensing/preview 成本优势证据。

## Diagnostics

- `results/action_diagnostics/action_diagnostics_ep1000_runs1_seed2026_summary.csv`
- `results/action_diagnostics/action_diagnostics_ep1000_runs1_seed2026_steps.csv`

支撑结论：完整 SAC 的传输参数选择偏保守或不稳定；Codebook-Aware SAC 覆盖率高于完整 SAC，但 IRS 选择仍没有 Greedy/Feature Argmax 激进。

## Imitation

- `results/imitation/greedy_imitation_train5000_eval1000_seed2026_eval_summary.csv`
- `results/imitation/greedy_imitation_train5000_eval1000_seed2026_train_history.csv`

支撑结论：低标签精确匹配率不必然导致差性能，因为很多 codebook 在 tx count 上并列；Feature Argmax 本身已经接近 Greedy。

## Parameter Sweep

- `results/parameter_sweep/parameter_sweep_ep300_seed2026_summary.csv`
- `results/parameter_sweep/parameter_sweep_ep300_seed2026_slot_stats.csv`

支撑结论：Feature Argmax/PowerTie/Greedy 在多个 `K/N/M/C` 变体上保持稳定；小 codebook 会暴露 resolution bottleneck。

## Partial Probing

- `results/partial_probing/partial_probing_sweep_ep1000_runs5_seed2026_b1-2-4-8.csv`

支撑结论：`Rotating Grid Probe` 是强 non-learning baseline；`B=4` 已有高 perfect rate，但 latency 明显高于 full Greedy；`B=8` 更接近 Greedy。

## Probing Cost

- `results/probing_cost/probing_cost_tradeoff_slot0-0p05-0p1-0p2_preview0-0p0005-0p001-0p002-0p005-0p01-0p02-0p05_winners.csv`

支撑结论：一旦 preview 有显式成本，低预算 Rotating Grid 往往优于 full Greedy，除非 preview 成本接近 0 且 slot 成本很高。

## Channel Estimation

- `results/channel_estimation/channel_estimation_error_sweep_ep1000_runs5_seed2026_err0-0p02-0p05-0p1-0p2-0p3.csv`

支撑结论：noisy decision preview 主要增加 latency 和 oracle gap，不会立即让 coverage 崩溃。

## Limited CSI

- `results/limited_csi/limited_csi_main_ep1000_runs5_seed2026.csv`

支撑结论：将“决策邀请”和“真实执行成功”分离后，scheduled/failed/missed/opportunity gap 比单纯 success 更重要；risk-aware 系列目前没有稳定优势。

指标口径：新生成的 limited-CSI CSV 同时写出 `failure_slot_count_mean` 和 `failure_slot_rate`；前者是每 episode 出现失败邀请的 slot 数，后者是按实际使用 slot 归一化后的百分比。历史 limited-CSI CSV 可能只有 `failure_slot_rate`。Execution-mismatch CSV 的 `failure_slot_rate` 也是百分比，但代码内部保存的是 per-episode fraction，不再与 limited-CSI 的 raw count 共用同名内部字段。

## Bandit Feedback

- `results/bandit_feedback/bandit_feedback_stress_formal_ep1000_runs5_seed2026.csv`

支撑结论：严格 aggregate feedback 下，Rotating Feedback Probe 是强基线；UCB/Thompson 容易受 stale history 限制。

## Execution Mismatch

### Evidence Chain

| Claim | Target / Script | Primary Artifacts | Notes |
|---|---|---|---|
| 论文前冻结的正文结果包和 appendix 边界 | `docs/PAPER_RESULT_PACKAGE.md` | `docs/PAPER_RESULT_PACKAGE.md` | 固定正文方法集合、主表/图和新增实验进入正文的条件 |
| 论文前章节、主表主图和附录映射 | `docs/PAPER_STRUCTURE_MAP.md` | `docs/PAPER_STRUCTURE_MAP.md` | 固定 Introduction/Method/Experiments/Discussion 的证据放置方式 |
| 论文主图主表规格 | `docs/PAPER_FIGURE_TABLE_SPECS.md` | `docs/PAPER_FIGURE_TABLE_SPECS.md` | 固定 Figure 1、Table 1、frontier/noise 图和诊断表的展示边界 |
| 论文附录最小集合 | `docs/PAPER_APPENDIX_BOUNDARY.md` | `docs/PAPER_APPENDIX_BOUNDARY.md` | 固定 Appendix A/B/C、supplement-only diagnostics 和禁入正文规则 |
| 论文正文最小骨架 | `docs/PAPER_TEXT_OUTLINE.md` | `docs/PAPER_TEXT_OUTLINE.md` | 固定 Abstract/Introduction/Method/Experiments/Discussion 的 claim、证据和禁入内容 |
| 投稿前资产缺口清单 | `docs/PAPER_ASSET_GAP_CHECKLIST.md` | `docs/PAPER_ASSET_GAP_CHECKLIST.md` | 固定 Figure 1 export、Table 2/3 paper artifacts 和图表美化状态 |
| Figure 1 可编辑源文件 | `docs/figures/figure1_system_flow.mmd` | `docs/figures/figure1_system_flow.mmd` | 固定 system mismatch 和 deployable feedback pipeline 的 Mermaid 源 |
| Table 1/2/3 可复现主表和诊断表 | `make paper-tables` | `docs/PAPER_TABLE1_MAIN_RESULTS.md`, `results/paper/table1_main_results.csv`, `docs/PAPER_TABLE1_UNCERTAINTY.md`, `results/paper/table1_scenario_uncertainty.csv`, `results/paper/table1_paired_scenario_deltas.csv`, `docs/PAPER_TABLE2_COVERAGE_AWARE_ABLATION.md`, `results/paper/table2_coverage_aware_ablation.csv`, `docs/PAPER_TABLE3_FAILURE_DIAGNOSIS.md`, `results/paper/table3_failure_diagnosis.csv` | 从 frozen mainline CSV、coverage-aware analysis CSV 和 failure-diagnosis CSV 自动生成论文主表、9-scenario variability、same-preview paired deltas、coverage-aware compact table 和 failure-diagnosis compact table |
| Figure 2/3/4 论文版机制图 | `make paper-figures` | `results/paper/figure2_figure3_points.csv`, `results/paper/figure2_preview_gap_frontier.png`, `results/paper/figure3_failed_missed_tradeoff.png`, `results/paper/figure4_invitation_mask_noise_points.csv`, `results/paper/figure4_invitation_mask_gap_noise.png`, `results/paper/figure4_invitation_mask_failed_missed_noise.png` | 从 frozen mainline CSV 自动生成正文方法集合的 preview-gap、failed/missed 和 noise-robustness 图 |
| 主线 baseline frontier 和 hidden-current headroom | `make execution-baseline-summary`, `make main-results-analysis` | `results/execution_mismatch/final_execution_baseline_summary.csv`, `results/execution_mismatch/main_frontier_analysis.csv`, `docs/EXECUTION_BASELINE_SUMMARY.md`, `docs/MAIN_RESULTS_ANALYSIS.md` | 连接 Rotating、Sparse-TopK、Coverage-Aware、Stale-TopK 和 Temporal Deviation Oracle |
| Coverage-Aware B=3 residual gap 来源 | `make coverage-b3-failure-diagnosis` | `results/execution_mismatch/coverage_b3_failure_diagnosis_ep100_runs2_rho0p7-0p9-0p98_delay1-2-3_b3_sm4p1_tf0p75_cw0p5_cpw0_summary.csv`, `results/execution_mismatch/coverage_b3_failure_diagnosis_ep100_runs2_rho0p7-0p9-0p98_delay1-2-3_b3_sm4p1_tf0p75_cw0p5_cpw0_trace.csv`, `docs/COVERAGE_B3_FAILURE_DIAGNOSIS.md` | 证明 residual gap 主要来自 invitation-mask mismatch |
| Coverage-Aware 参数和 budget split | `make coverage-aware-analysis`, `make coverage-budget-split-formal` | `docs/COVERAGE_AWARE_ANALYSIS.md`, `results/execution_mismatch/coverage_aware_ablation_analysis.csv`, `results/execution_mismatch/coverage_aware_weight_ablation.png`, `results/execution_mismatch/coverage_aware_power_ablation.png` | 固定 `cw=0.5 cpw=0`，并选择 `B=3 sm=4.1` 作为 same-preview refinement |
| Reliable-feedback invitation-mask correction trade-off | `make invitation-mask-correction-formal` | `results/execution_mismatch/invitation_mask_correction_formal_ep300_runs3_rho0p7-0p9-0p98_delay1-2-3_b3_sm4p1_tf0p75_cw0p5_cpw0_mc0-0p75-1.csv`, `docs/INVITATION_MASK_CORRECTION.md` | `mc=1` 在同 preview `16` 下降低 slots/failed/missed，但 no-noise gap 回升 |
| Feedback-noise boundary and clipped failed-control diagnostic | `make invitation-mask-correction-noise-sweep`, `make invitation-mask-correction-noise-aware-formal`, `make final-invitation-mask-analysis` | `results/execution_mismatch/invitation_mask_correction_noise_ep300_runs3_rho0p7-0p9-0p98_delay1-2-3_b3_sm4p1_tf0p75_cw0p5_cpw0_mc0-0p75-1_fbn0-0p02-0p05-0p1.csv`, `results/execution_mismatch/invitation_mask_correction_noise_aware_formal_ep300_runs3_rho0p7-0p9-0p98_delay1-2-3_b3_sm4p1_tf0p75_cw0p5_cpw0_mc0-0p75-1_clipinf-2_fbn0-0p02-0p05-0p1.csv`, `results/execution_mismatch/final_invitation_mask_analysis.csv`, `docs/FINAL_INVITATION_MASK_ANALYSIS.md` | high-noise `std=0.1` 下 direct `mc=1` 是 gap-best，`clip=2` 是 failed-invitation control diagnostic |

`make mainline-audit` 会检查上述关键 artifact 是否存在、核心 CSV 字段是否完整、summary 中的 `source_file` 是否能解析到真实结果文件、以及 final invitation-mask 的主要数值关系是否仍成立。它不重跑实验。

- `results/execution_mismatch/final_execution_baseline_summary.csv`
- `results/execution_mismatch/main_frontier_analysis.csv`
- `results/execution_mismatch/main_frontier_preview_gap.png`
- `results/execution_mismatch/main_frontier_failed_missed.png`
- `docs/EXECUTION_BASELINE_SUMMARY.md`
- `results/execution_mismatch/coverage_b3_failure_diagnosis_ep100_runs2_rho0p7-0p9-0p98_delay1-2-3_b3_sm4p1_tf0p75_cw0p5_cpw0_trace.csv`
- `results/execution_mismatch/coverage_b3_failure_diagnosis_ep100_runs2_rho0p7-0p9-0p98_delay1-2-3_b3_sm4p1_tf0p75_cw0p5_cpw0_summary.csv`
- `docs/COVERAGE_B3_FAILURE_DIAGNOSIS.md`
- `results/execution_mismatch/invitation_mask_correction_formal_ep300_runs3_rho0p7-0p9-0p98_delay1-2-3_b3_sm4p1_tf0p75_cw0p5_cpw0_mc0-0p75-1.csv`
- `docs/INVITATION_MASK_CORRECTION.md`
- `results/execution_mismatch/invitation_mask_correction_noise_ep300_runs3_rho0p7-0p9-0p98_delay1-2-3_b3_sm4p1_tf0p75_cw0p5_cpw0_mc0-0p75-1_fbn0-0p02-0p05-0p1.csv`
- `docs/INVITATION_MASK_CORRECTION_NOISE.md`
- `results/execution_mismatch/invitation_mask_correction_noise_aware_formal_ep300_runs3_rho0p7-0p9-0p98_delay1-2-3_b3_sm4p1_tf0p75_cw0p5_cpw0_mc0-0p75-1_clipinf-2_fbn0-0p02-0p05-0p1.csv`
- `docs/INVITATION_MASK_CORRECTION_NOISE_AWARE.md`
- `results/execution_mismatch/final_invitation_mask_analysis.csv`
- `results/execution_mismatch/final_invitation_mask_gap_noise.png`
- `results/execution_mismatch/final_invitation_mask_failed_missed_noise.png`
- `docs/PAPER_TABLE1_UNCERTAINTY.md`
- `results/paper/table1_scenario_uncertainty.csv`
- `results/paper/table1_paired_scenario_deltas.csv`
- `results/paper/figure4_invitation_mask_noise_points.csv`
- `results/paper/figure4_invitation_mask_gap_noise.png`
- `results/paper/figure4_invitation_mask_failed_missed_noise.png`
- `docs/FINAL_INVITATION_MASK_ANALYSIS.md`

Coverage B3 failure diagnosis 结论：当前 `Coverage-Aware B=3 sm=4.1 cw=0.5 cpw=0` 的 residual gap 主要来自 confirmed IRS 下 stale invitation mask 和 current execution opportunity 的错配。100 episodes × 2 seeds × 9 temporal AR(1) 场景的 trace 中，总体 gap share 为 invitation `0.687`、selection `0.260`、confirmation `0.024`、pool `0.029`；exact oracle index 不在 seed 的比例为 `0.175`，但 pool gap share 只有 `0.029`，说明多个 IRS index 常有相近 current tx count。下一步应优先研究 invitation-mask correction，而不是继续固定 local-neighbor 或普通 seed-pool heuristic。

Invitation Mask Correction formal 结论：保持 `Coverage-Aware B=3 sm=4.1` 的 candidate generation 和 current aggregate confirmation 不变，用 confirmed IRS 的 aggregate feedback count 设定 target cardinality，并用 confirmed IRS 下的 stale-gain reranking 生成 corrected invitation mask。`mc=1` 在同样 preview `16` 下把 slots 从 `3.604` 降到 `3.232`，failed/missed 从 `5.804/5.438` 降到 `5.385/5.385`，但 no-noise gap 从 `2.207` 升到 `2.382`。这是 reliable-feedback trade-off，不是 no-noise gap-best result。

Invitation Mask Correction noise sweep 结论：feedback-noise std `0.05/0.1` 下，direct `mc=1` 开始改善 gap：std `0.05` 把 B3 gap 从 `2.380` 降到 `2.184`，std `0.1` 把 B3 gap 从 `2.519` 降到 `2.080`。这说明 invitation-mask correction 的收益依赖 feedback-noise boundary，而不能把 no-noise 结果写成 gap improvement。

Noise-aware clipped correction formal 结论：broad pilot 显示 deadband `z>0` 会压掉有用修正，因此 formal 只保留 direct 与 `clip=2`。feedback-noise std `0.1` 时，direct `mc=1` 是 gap-best correction，gap `2.080`；`mc=1 clip=2` 把 direct `mc=1` 的 failed invitations 从 `8.803` 降到 `8.395`，但 gap 升到 `2.291`。最终报告应采用 no-noise trade-off + high-noise direct gap-best + clipped failed-control diagnostic，而不是继续普通 candidate-generation heuristic。

Final Invitation Mask Analysis 结论：`make final-invitation-mask-analysis` 已把最终主张整理成论文级表格和两张图。可靠反馈 no-noise direct `mc=1` 是 trade-off：slots/failed/missed 降低但 gap 上升；高噪声 std `0.1` 下 direct `mc=1` 是 gap-best correction，`mc=1 clip=2` 是 failed-invitation control diagnostic。两者都保持 preview `16`，不改变 IRS candidate generation。

Invitation Mask Rerank Ablation 结论：`make invitation-mask-rerank-ablation` 生成 `results/execution_mismatch/invitation_mask_rerank_ablation_ep100_runs2_rho0p7-0p9-0p98_delay1-2-3_b3_sm4p1_tf0p75_cw0p5_cpw0_mc0-1_modesglobal-prune.csv` 和 `docs/INVITATION_MASK_RERANK_ABLATION.md`。这是 diagnostic pilot，不是 frozen paper result。`prune_only` 只能 prune 原 stale-valid mask，不能添加 stale-invalid devices；它把 failed 降到 `5.238`，但 slots/missed/gap 退化到 `3.755/6.891/2.538`。原始 `global_stale_gain` correction 为 `3.267/5.392/5.392/2.384`，说明 balanced trade-off 依赖 stale-gain replacement，不是纯 target-count correction。

Paper tables 结论：`make paper-tables` 现在同时生成 `docs/PAPER_TABLE1_UNCERTAINTY.md`、`results/paper/table1_scenario_uncertainty.csv`、`results/paper/table1_paired_scenario_deltas.csv`、`docs/PAPER_TABLE2_COVERAGE_AWARE_ABLATION.md`、`results/paper/table2_coverage_aware_ablation.csv`、`docs/PAPER_TABLE3_FAILURE_DIAGNOSIS.md` 和 `results/paper/table3_failure_diagnosis.csv`。Table 1 companion 是 scenario-level，不是完整 seed-level significance test；它显示 `Mask-Corrected Coverage-Aware B=3 mc=1` 相对 `Coverage-Aware B=3` 的 slots/failed 为 9/9 场景改善，missed 为 6/9 改善，但 gap 只在 3/9 场景改善。Table 2 固化 coverage-aware / budget-split 选择证据，Table 3 固化 invitation-mask mismatch 主导 residual gap 的诊断证据。
- `docs/MAIN_RESULTS_ANALYSIS.md`
- `results/execution_mismatch/active_probe_set_pilot_ep300_runs3_rho0p7-0p9-0p98_delay1-2-3_b4.csv`
- `results/execution_mismatch/sparse_topk_cost_pilot_ep100_runs2_rho0p7-0p9-0p98_delay1-3_b2-4-8_sm1-2-3_tf0p25-0p5-0p75.csv`
- `results/execution_mismatch/sparse_topk_frontier_ep300_runs3_rho0p7-0p9-0p98_delay1-2-3_b4-8_sm2-3_tf0p75.csv`
- `results/execution_mismatch/adaptive_sparse_topk_pilot_ep100_runs2_rho0p7-0p9-0p98_delay1-2-3_b4_mt0-0p02-0p05-0p1-0p2.csv`
- `results/execution_mismatch/adaptive_sparse_topk_v2_pilot_ep100_runs2_rho0p7-0p9-0p98_delay1-2-3_b4_mt0p02-0p05_pc0-0p002-0p005-0p01.csv`
- `results/execution_mismatch/adaptive_sparse_topk_v3_pilot_ep100_runs2_rho0p7-0p9-0p98_delay1-2-3_b4_nr1_nc2_hc1.csv`
- `results/execution_mismatch/learned_sparse_shortlist_pilot_train500_val100_rho0p7-0p9-0p98_delay1-2-3_b4_bm2_tf0p75_diagnostics.csv`
- `results/execution_mismatch/learned_sparse_shortlist_pilot_ep100_runs2_rho0p7-0p9-0p98_delay1-2-3_b4_ex1-2.csv`
- `results/execution_mismatch/learned_sparse_shortlist_marginal_pilot_train500_val100_rho0p7-0p9-0p98_delay1-2-3_b4_bm2_tf0p75_tex2_diagnostics.csv`
- `results/execution_mismatch/learned_sparse_shortlist_marginal_pilot_ep100_runs2_rho0p7-0p9-0p98_delay1-2-3_b4_ex1-2.csv`
- `results/execution_mismatch/learned_set_shortlist_pilot_train500_val100_rho0p7-0p9-0p98_delay1-2-3_b4_bm2_tf0p75_maxex2_diagnostics.csv`
- `results/execution_mismatch/learned_set_shortlist_pilot_ep100_runs2_rho0p7-0p9-0p98_delay1-2-3_b4_maxex1-2.csv`
- `results/execution_mismatch/learned_execution_value_shortlist_pilot_train500_val100_rho0p7-0p9-0p98_delay1-2-3_b4_bm2_tf0p75_maxex2_fw0p5_mw0p5_diagnostics.csv`
- `results/execution_mismatch/learned_execution_value_shortlist_pilot_ep100_runs2_rho0p7-0p9-0p98_delay1-2-3_b4_maxex1-2_fw0p5_mw0p5.csv`
- `results/execution_mismatch/learned_pairwise_shortlist_pilot_train500_val100_rho0p7-0p9-0p98_delay1-2-3_b4_bm2_tf0p75_maxex2_fw0p5_mw0p5_pc0_diagnostics.csv`
- `results/execution_mismatch/learned_pairwise_shortlist_pilot_ep100_runs2_rho0p7-0p9-0p98_delay1-2-3_b4_maxex1-2_fw0p5_mw0p5_pc0.csv`
- `results/execution_mismatch/learned_pairwise_shortlist_pilot_train500_val100_rho0p7-0p9-0p98_delay1-2-3_b4_bm2_tf0p75_maxex2_fw0p5_mw0p5_pc0p005_diagnostics.csv`
- `results/execution_mismatch/learned_pairwise_shortlist_pilot_ep100_runs2_rho0p7-0p9-0p98_delay1-2-3_b4_maxex1-2_fw0p5_mw0p5_pc0p005.csv`
- `results/execution_mismatch/stale_topk_feedback_pilot_ep100_runs3_rho0p7-0p9-0p98_delay1-2-3_b4.csv`
- `results/execution_mismatch/temporal_deviation_oracle_pilot_ep300_runs3_rho0p7-0p9-0p98_delay1-2-3_b4.csv`
- `results/execution_mismatch/rotating_feedback_confirm_pilot_ep100_runs3_rho0p7-0p9-0p98_delay1-2-3_b4.csv`

最终主线汇总由 `make execution-baseline-summary` 生成，论文级机制分析由 `make main-results-analysis` 生成。论文前冻结的正文主表应报告 `Rotating B=8`、`Sparse-TopK B=4 sm=3`、`Coverage-Aware B=4 cw=0.5 cpw=0`、`Coverage-Aware B=3 sm=4.1 cw=0.5 cpw=0`、`Mask-Corrected Coverage-Aware B=3 mc=1`、`Stale-TopK B=4` 和 `Temporal Deviation Oracle B=4`；adaptive v2 continuum 和 learned shortlist 放入 appendix / diagnostics。支撑结论：hidden-current candidate set 有明显 headroom；`Stale-TopK Feedback` 是高成本正向参照；low-cost coverage-aware candidate generation + aggregate-feedback invitation-mask correction 是当前最值得报告的主线。

Active probe-set pilot 结论：`Active Diverse Feedback Grid B=4` 只小幅降低 gap，但 slots 和 preview 成本都不理想；`Sparse-TopK Feedback Grid B=4` 在 pilot 阶段更值得继续。跨 9 个 rho/delay 场景，`Sparse-TopK` 平均 slots `3.751`、gap `0.773`、preview `12`，优于普通 `Rotating B=4` 的 slots `3.887`、gap `1.116`、preview `4`，但仍弱于高成本 `Stale-TopK` 的 slots `3.373`、gap `0.465`、preview `20`。因此后续正式 sweep 转向 sparse stale ranking 的成本-收益曲线，而不是继续优化纯几何多样性候选集。

Sparse-TopK cost pilot 结论：`topk_fraction=0.75` 在同等 preview 下优于旧的 `0.5`，因此已作为新的默认值。正式 frontier 进一步修正了策略定位：跨完整 9 个 rho/delay 场景，`Rotating B=8` 为 slots `3.429`、gap `0.726`、preview `8`；`Sparse-TopK B=4 sm=2 tf=0.75` 为 slots `3.635`、gap `0.736`、preview `12`，不再优于 Rotating B=8；`Sparse-TopK B=4 sm=3 tf=0.75` 为 slots `3.376`、gap `0.535`、preview `16`，接近 `Stale-TopK B=4` 的 slots `3.373`、gap `0.465`、preview `20`。因此当前应把 `Rotating B=8` 作为低成本部署强基线，把 `Sparse-TopK B=4 sm=3` 作为中等成本候选，把 `Stale-TopK B=4` 作为高成本正向参照，把 `Temporal Deviation Oracle` 作为隐藏信息 temporal diagnostic 参照。

Coverage-Aware Sparse-TopK weight ablation 输出为 `results/execution_mismatch/coverage_sparse_topk_ablation_ep300_runs3_rho0p7-0p9-0p98_delay1-2-3_b4_sm3_tf0p75_cw0-0p25-0p5-1-2_cpw0.csv`，入口是 `make coverage-sparse-topk-ablation`，分析文档是 `docs/COVERAGE_AWARE_ANALYSIS.md`。power ablation 输出为 `results/execution_mismatch/coverage_sparse_power_ablation_ep300_runs3_rho0p7-0p9-0p98_delay1-2-3_b4_sm3_tf0p75_cw0p5_cpw0-0p02-0p05-0p1-0p2.csv`，显示 `cpw=0` 是当前主设置。B=4 formal frontier 输出为 `results/execution_mismatch/coverage_sparse_topk_frontier_ep300_runs3_rho0p7-0p9-0p98_delay1-2-3_b4_sm3_tf0p75_cw0p5_cpw0.csv`，入口是 `make coverage-sparse-topk-frontier`。budget split formal 入口是 `make coverage-budget-split-formal`，覆盖 `B=3 sm=4.1`、`B=4 sm=3`、`B=5 sm=2.2`、`B=6 sm=1.6` 和 `B=8 sm=1`。修正 temporal prehistory 后，B=4 formal frontier 在 preview `16` 下为 slots/failed/missed/gap `3.649/5.271/6.133/2.246`；`B=3 sm=4.1` 为 `3.604/5.804/5.438/2.207`，是当前 no-noise same-preview gap reference。Neighbor-Coverage pilot 输出为 `results/execution_mismatch/neighbor_coverage_pilot_ep100_runs2_rho0p7-0p9-0p98_delay1-2-3_b3_sm4p1_tf0p75_cw0p5_cpw0_nr1_nc1.csv`、`..._nr1_nc2.csv`、`..._nr1_nc3.csv` 和 `..._nr2_nc3.csv`；修正后未超过当前 B3 formal。该方法不是另开新主线，而是在 `Sparse-TopK sm=3` 的同一 sparse preview budget 内改进 candidate generation 和预算分配；当前 no-noise gap reference 是更宽 stale pool、更少 current confirmation 的 `B=3 sm=4.1`。

Adaptive Sparse-TopK pilot 结论：`mt=0.02` 平均 slots `3.584`、gap `0.697`、preview `13.47`，比固定 `sm=2` 有提升但在高 rho 长 delay 下不稳定；`mt=0.05` 平均 slots `3.454`、gap `0.564`、preview `15.54`，几乎复现固定 `sm=3` 的效果但只节省约 `0.46` preview。该结果支持 adaptive shortlist 作为下一步方向，但不支持把一维 stale margin threshold 当作最终低成本算法。

Adaptive Sparse-TopK v2 pilot 结论：加入 rho/delay、历史稳定性、deadline urgency 和 preview-cost penalty 后，策略形成更平滑的成本-质量曲线。当前最有用中间点是 `mt=0.05, pc=0.005`，平均 slots `3.511`、gap `0.636`、preview `14.58`、expansion `64.5%`；它优于 v1 `mt=0.02` 的 slots/gap，但仍弱于 v1 `mt=0.05` 和固定 `sm=3`。因此 v2 可作为 continuum baseline，但下一步应改变候选生成本身，而不是继续调 scalar expansion gate。

Adaptive Sparse-TopK v3 pilot 结论：history prior + local-neighbor candidate generation 把平均 preview 压到 `12.92`，并把 slots 从 fixed `sm=2` 的 `3.647` 降到 `3.583`；但 oracle gap 为 `0.754`，没有优于 `sm=2` 的 `0.750`，也明显弱于 `sm=3` 的 slots/gap `3.447/0.561` 和 v2 `mt=0.05, pc=0.002` 的 `3.492/0.597`。因此 v3 不应作为当前主 baseline，只作为说明固定局部邻域启发式不足的诊断结果；下一步应转向 learned/nonuniform shortlist scoring。

Learned Sparse Shortlist pilot 结论：线性 ranker 的 validation correlation 为 `0.996`，闭环中 absolute-label `ex=1` 平均 slots/gap/preview 为 `3.557/0.689/13.00`，`ex=2` 为 `3.570/0.653/14.00`。这明显优于 v3 的 gap `0.754` 和 fixed `sm=2` 的 gap `0.750`，说明 nonuniform learned extras 有价值；但仍弱于 v2 `mt=0.05, pc=0.002` 的 `3.492/0.597/15.36` 和 `sm=3` 的 `3.447/0.561/16.00`。marginal-label pilot 的 validation top1 regret 降到 `0.0064`，但闭环 `ex=2` 只有 `3.603/0.662/14.00`，没有超过 absolute-label `ex=2`。set-level hidden-value pilot 的 validation top1 regret 为 `0.0055`，但闭环 `maxex=1` 为 `3.605/0.686/12.93`，`maxex=2` 为 `3.659/0.703/13.68`，也没有超过 absolute-label `ex=2`。closed-loop execution-value pilot 的 validation top1 regret 为 `0.0099`，闭环 `maxex=1` 为 `3.589/0.702/12.95`，`maxex=2` 为 `3.619/0.681/13.81`；它修正了 hidden set-value `maxex=2` 的一部分目标错配，但仍没有超过 absolute-label `ex=2`。pairwise execution pilot 中，`pc=0` 的 validation top1 regret 为 `0.0111`，闭环 `maxex=1/2` 为 `3.619/0.751/12.18`，几乎不选 extra；`pc=0.005` 的 top1 regret 为 `0.0131`，闭环 `maxex=1/2` 为 `3.588/0.704/13.00`，稳定选 1 个 extra 但仍弱于 absolute-label `ex=2`。因此当前学习路线的结论是：线性 learned shortlist 可作为诊断和低成本候选，但不是主方法；短期主线应报告 `Sparse-TopK sm=3` 和 v2 continuum，而不是继续调 learned set-label。

新生成 learned shortlist / learned temporal-deviation artifacts 会写入 metadata：`result_role=diagnostic`、`uses_hidden_training_labels=true`、`inference_uses_hidden_current_csi=false`。Temporal/Execution Oracle 类结果写入 `result_role=diagnostic_upper_bound` 和 `inference_uses_hidden_current_csi=true`。历史 CSV 可能缺少这些列，写论文时以本节文字边界和 `docs/PAPER_APPENDIX_BOUNDARY.md` 为准。
