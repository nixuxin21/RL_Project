# Paper Result Package

本文件不是论文草稿，而是论文撰写前的结果包冻结层。它回答三个问题：

1. 正文主线应报告哪些方法、表和图。
2. 哪些结果只作为 appendix / diagnostics。
3. 后续新增实验必须满足什么条件，才值得进入正文主线。

章节、主表主图顺序和 appendix 位置见 `docs/PAPER_STRUCTURE_MAP.md`。Figure 1、Table 1 和后续主图主表规格见 `docs/PAPER_FIGURE_TABLE_SPECS.md`。附录最小集合和 diagnostics 边界见 `docs/PAPER_APPENDIX_BOUNDARY.md`。逐节写作 claim、证据和禁入内容见 `docs/PAPER_TEXT_OUTLINE.md`。投稿前表图资产缺口见 `docs/PAPER_ASSET_GAP_CHECKLIST.md`。本文档负责冻结结果包；结构映射文档负责把结果包放进未来论文骨架。

## Scope

当前论文级问题设定为：

> IRS-assisted multi-slot AirComp under temporal stale CSI and execution-channel mismatch, using low-cost IRS candidate generation plus current aggregate feedback.

默认主评估覆盖 9 个 temporal AR(1) stale-CSI 场景：

- `channel_rho in {0.7, 0.9, 0.98}`
- `csi_delay_slots in {1, 2, 3}`
- 默认环境为 `K=50`, `N=10`, `M=64`, `C=16`
- 默认传输门限为 `g_th=0.001`, `alpha_th=0.05`

当前主张应表述为：

> Low-cost coverage-aware candidate generation is the no-noise same-preview gap reference. Aggregate-feedback invitation-mask correction is a same-preview slot/failed/missed trade-off under reliable feedback, and direct correction becomes gap-improving under higher aggregate-feedback noise.

Uncertainty status: the compact paper Table 1 is mean-only, with a companion
scenario-level uncertainty artifact at `docs/PAPER_TABLE1_UNCERTAINTY.md`. The
companion reports 9-scenario variability and paired same-preview deltas; it is
not a full seed-level significance test.

其中 `Temporal Deviation Oracle B=4` 只作为 hidden-information temporal diagnostic reference，不是可部署方法，也不是 global upper bound。

## Frozen Main-Text Methods

正文主表冻结为以下方法。除非新实验直接改善 invitation-mask mismatch、same-preview performance 或 high-noise robustness，否则不要再把新 heuristic 加入正文主表。

| Method | Role | Main Text Status |
|---|---|---|
| `Rotating B=8` | low-cost deployment baseline | main baseline |
| `Sparse-TopK B=4 sm=3` | reportable medium-cost sparse candidate baseline | main baseline |
| `Coverage-Aware B=4 cw=0.5 cpw=0` | same-cost B=4 coverage reference | main ablation/reference |
| `Coverage-Aware B=3 sm=4.1 cw=0.5 cpw=0` | budget-split refinement at preview 16 | main pre-correction method |
| `Mask-Corrected Coverage-Aware B=3 mc=1` | slot/failed trade-off; no-noise gap regression | main trade-off / mechanism result |
| `Stale-TopK B=4` | high-cost positive reference | main reference |
| `Temporal Deviation Oracle B=4` | hidden current-channel temporal diagnostic | main diagnostic reference |

`Rotating B=4` 只作为 low-budget historical reference；Adaptive Sparse-TopK v2 可作为 cost-quality continuum；Adaptive v1/v3、neighbor coverage、learned shortlist、learned temporal、bandit feedback、limited CSI risk-aware、SAC / Codebook-Aware SAC、imitation、learned probing、noisy feature、partial probing、probing cost、channel estimation、runtime/action diagnostics 等均进入 appendix / diagnostics，不作为正文主方法。

## Frozen Main Result Table

跨 9 个 temporal AR(1) `rho/delay` 场景的 equal-weight 平均结果如下。正文主表应优先使用这个视角，而不是重新混入早期 RL 或 partial-probing 主表。

| Method | Role | Slots | Perfect % | Failed | Missed | Preview | Gap |
|---|---|---:|---:|---:|---:|---:|---:|
| `Rotating B=8` | low-cost deployment baseline | 3.992 | 99.98 | 8.878 | 4.580 | 8.00 | 2.596 |
| `Sparse-TopK B=4 sm=3` | reportable medium-cost baseline | 3.770 | 99.77 | 5.272 | 6.643 | 16.00 | 2.334 |
| `Coverage-Aware B=4 cw=0.5 cpw=0` | same-cost B=4 coverage reference | 3.649 | 99.77 | 5.271 | 6.133 | 16.00 | 2.246 |
| `Coverage-Aware B=3 sm=4.1 cw=0.5 cpw=0` | budget-split refinement | 3.604 | 99.94 | 5.804 | 5.438 | 16.00 | 2.207 |
| `Mask-Corrected Coverage-Aware B=3 mc=1` | slot/failed trade-off; no-noise gap regression | 3.232 | 99.95 | 5.385 | 5.385 | 16.00 | 2.382 |
| `Stale-TopK B=4` | high-cost positive reference | 3.803 | 99.69 | 5.337 | 6.841 | 20.00 | 2.370 |
| `Temporal Deviation Oracle B=4` | hidden-info temporal diagnostic | 3.359 | 100.00 | 5.129 | 5.642 | 4.00 | 2.162 |

Supporting artifacts:

- `docs/PAPER_TABLE1_MAIN_RESULTS.md`
- `results/paper/table1_main_results.csv`
- `docs/PAPER_TABLE1_UNCERTAINTY.md`
- `results/paper/table1_scenario_uncertainty.csv`
- `results/paper/table1_paired_scenario_deltas.csv`
- `docs/PAPER_TABLE2_COVERAGE_AWARE_ABLATION.md`
- `results/paper/table2_coverage_aware_ablation.csv`
- `docs/PAPER_TABLE3_FAILURE_DIAGNOSIS.md`
- `results/paper/table3_failure_diagnosis.csv`
- `docs/EXECUTION_BASELINE_SUMMARY.md`
- `results/execution_mismatch/final_execution_baseline_summary.csv`
- `docs/MAIN_RESULTS_ANALYSIS.md`
- `results/execution_mismatch/main_frontier_analysis.csv`
- `results/execution_mismatch/main_frontier_preview_gap.png`
- `results/execution_mismatch/main_frontier_failed_missed.png`
- `results/paper/figure2_figure3_points.csv`
- `results/paper/figure2_preview_gap_frontier.png`
- `results/paper/figure3_failed_missed_tradeoff.png`
- `results/paper/figure4_invitation_mask_noise_points.csv`
- `results/paper/figure4_invitation_mask_gap_noise.png`
- `results/paper/figure4_invitation_mask_failed_missed_noise.png`

Paper-facing generated artifacts are separated from analysis-layer figures:

- `make paper-tables` owns Table 1, Table 2, Table 3 and the Table 1 scenario-level uncertainty companion (`docs/PAPER_TABLE1_MAIN_RESULTS.md`, `docs/PAPER_TABLE1_UNCERTAINTY.md`, `docs/PAPER_TABLE2_COVERAGE_AWARE_ABLATION.md`, `docs/PAPER_TABLE3_FAILURE_DIAGNOSIS.md`, `results/paper/table1_main_results.csv`, `results/paper/table1_scenario_uncertainty.csv`, `results/paper/table1_paired_scenario_deltas.csv`, `results/paper/table2_coverage_aware_ablation.csv`, `results/paper/table3_failure_diagnosis.csv`).
- `make paper-figures` owns Figure 2, Figure 3 and Figure 4 (`results/paper/...`).
- `make main-results-analysis` and `make final-invitation-mask-analysis` own analysis-layer CSV/PNG/MD files under `results/execution_mismatch/` and `docs/`.

When drafting, cite the `results/paper/` assets for main-text figures and use the analysis-layer files as traceability evidence.

## Main Figures And Tables

正文建议组织为三组证据，而不是按实验历史顺序堆叠：

| Evidence | Primary Artifacts | Intended Use |
|---|---|---|
| Main frontier and hidden-current headroom | `docs/EXECUTION_BASELINE_SUMMARY.md`, `docs/MAIN_RESULTS_ANALYSIS.md`, `results/execution_mismatch/final_execution_baseline_summary.csv`, `results/execution_mismatch/main_frontier_analysis.csv`, `results/execution_mismatch/main_frontier_preview_gap.png`, `results/execution_mismatch/main_frontier_failed_missed.png` | 证明 low-cost feedback candidate generation 有明确 cost-quality frontier，并保留 hidden-current headroom |
| Coverage-aware candidate generation and budget split | `docs/PAPER_TABLE2_COVERAGE_AWARE_ABLATION.md`, `results/paper/table2_coverage_aware_ablation.csv`, `docs/COVERAGE_AWARE_ANALYSIS.md`, `results/execution_mismatch/coverage_aware_ablation_analysis.csv`, `results/execution_mismatch/coverage_aware_weight_ablation.png`, `results/execution_mismatch/coverage_aware_power_ablation.png` | 解释为何采用 `cw=0.5 cpw=0`，以及为何 `B=3 sm=4.1` 是 same-preview refinement |
| Coverage-Aware B=3 failure diagnosis | `docs/PAPER_TABLE3_FAILURE_DIAGNOSIS.md`, `results/paper/table3_failure_diagnosis.csv`, `docs/COVERAGE_B3_FAILURE_DIAGNOSIS.md`, `results/execution_mismatch/coverage_b3_failure_diagnosis_ep100_runs2_rho0p7-0p9-0p98_delay1-2-3_b3_sm4p1_tf0p75_cw0p5_cpw0_summary.csv` | 证明 residual gap 主要来自 invitation-mask mismatch |
| Invitation-mask correction final result | `docs/FINAL_INVITATION_MASK_ANALYSIS.md`, `results/execution_mismatch/final_invitation_mask_analysis.csv`, `results/paper/figure4_invitation_mask_noise_points.csv`, `results/paper/figure4_invitation_mask_gap_noise.png`, `results/paper/figure4_invitation_mask_failed_missed_noise.png` | 作为 reliable-feedback trade-off 和 feedback-noise boundary 证据 |

## Evidence Chain

主线论证链冻结如下：

| Claim | Evidence |
|---|---|
| `Rotating B=8` 是强 low-cost deployment baseline，不能只和 `Rotating B=4` 或 Random IRS 比较。 | `docs/MAIN_RESULTS_ANALYSIS.md`, `results/execution_mismatch/main_frontier_analysis.csv` |
| Sparse stale preview + current aggregate confirmation 能在 preview `16` 附近接近高成本 `Stale-TopK B=4`。 | `docs/EXECUTION_BASELINE_SUMMARY.md`, `results/execution_mismatch/final_execution_baseline_summary.csv` |
| Coverage-aware candidate generation 的主要价值在降低 missed opportunities；`B=3 sm=4.1` 是当前 same-preview budget split。 | `docs/COVERAGE_AWARE_ANALYSIS.md`, `results/execution_mismatch/coverage_aware_ablation_analysis.csv` |
| `Coverage-Aware B=3` 的 residual gap 主要来自 stale invitation mask 与 current opportunity 的错配。 | `docs/COVERAGE_B3_FAILURE_DIAGNOSIS.md`, `results/execution_mismatch/coverage_b3_failure_diagnosis_ep100_runs2_rho0p7-0p9-0p98_delay1-2-3_b3_sm4p1_tf0p75_cw0p5_cpw0_summary.csv` |
| Aggregate-feedback invitation-mask correction 不改变 IRS candidate generation；它用 aggregate current feedback count 设定 target cardinality，并用 confirmed IRS 下的 stale-gain reranking 选择 corrected invited set，在同 preview `16` 下相对 B3 降低 observed mean slots、failed 和 missed，但 no-noise oracle gap 上升。 | `docs/INVITATION_MASK_CORRECTION.md`, `docs/FINAL_INVITATION_MASK_ANALYSIS.md`, `results/execution_mismatch/final_invitation_mask_analysis.csv` |
| 该 trade-off 不是单个场景造成的：相对 `Coverage-Aware B=3`，slots 和 failed 为 9/9 场景改善，missed 为 6/9 改善，但 gap 只在 3/9 场景改善。 | `docs/PAPER_TABLE1_UNCERTAINTY.md`, `results/paper/table1_paired_scenario_deltas.csv` |
| 高噪声下 direct `mc=1` 是 gap-best correction；`mc=1 clip=2` 降低 failed invitations，但不是 high-noise gap-best method。 | `docs/INVITATION_MASK_CORRECTION_NOISE.md`, `docs/INVITATION_MASK_CORRECTION_NOISE_AWARE.md`, `docs/FINAL_INVITATION_MASK_ANALYSIS.md` |

Coverage B3 diagnosis 的 overall gap share 为 invitation `0.687`、selection `0.260`、confirmation `0.024`、pool `0.029`。因此下一阶段若继续做实验，应优先围绕 invitation mask / aggregate feedback robustness，而不是普通 candidate-pool heuristic。

## Appendix And Diagnostics Boundary

以下内容保留，但默认不进入正文主方法集合：

- `Rotating B=4`: low-budget reference。
- Adaptive Sparse-TopK v2: cost-quality continuum point，而不是最终方法。
- Adaptive Sparse-TopK v1/v3: scalar expansion gate 与 local-neighbor heuristic 诊断。
- Neighbor-Coverage: local-neighbor reallocation 诊断，未超过当前 B3 formal。
- Learned sparse / set / execution-value / pairwise shortlist: nonuniform learned extras 诊断，未超过 frozen mainline。
- Learned temporal deviation / window / DAgger: temporal learned selector 诊断。
- Bandit feedback UCB / Thompson / feedback-conditioned MLP: aggregate-feedback learning 诊断。
- Limited CSI risk-aware filters: invitation reliability 诊断。
- SAC / Codebook-Aware SAC / imitation / learned probing / noisy feature / partial probing / probing cost / channel estimation / parameter sweep / runtime / action diagnostics: background and appendix only。

Appendix 默认只保留 `docs/PAPER_APPENDIX_BOUNDARY.md` 中的 three-item minimum：historical baseline context、preview-cost / CSI motivation、diagnostic branches summary。其余 diagnostics 只有 reviewer 要求时才从 `docs/RESULTS_INDEX.md` 抽取。

Appendix 和 diagnostics 的索引以 `docs/PAPER_APPENDIX_BOUNDARY.md`、`docs/DEPRECATED_DIRECTIONS.md`、`docs/RESULTS_INDEX.md` 和 `EXPERIMENT_REPORT.md` 为准。

## Reproduction And Audit

论文前结果包默认不重跑所有实验；先检查 artifact chain，再按需重跑生成分析文档。

```bash
make mainline-audit
make check
```

需要重建主线分析文档时运行：

```bash
make execution-baseline-summary
make main-results-analysis
make coverage-aware-analysis
make coverage-b3-failure-diagnosis
make invitation-mask-correction-formal
make invitation-mask-correction-noise-aware-formal
make final-invitation-mask-analysis
```

`make mainline-audit` 必须覆盖本文件，确保结果包不会从主线审计中脱钩。

## Freeze Boundary

在真正开始写论文前，默认冻结以下边界：

- 正文主表不再加入普通 heuristic，除非它改善 `Coverage-Aware B=3 sm=4.1 cw=0.5 cpw=0` 的 no-noise same-preview gap，或解释 mask-correction trade-off / feedback-noise boundary。
- 新方法必须在相同或更低 preview 下对比 frozen main-text methods。
- 新诊断必须服务于 invitation mismatch、same-preview performance 或 feedback-noise robustness。
- 学习式方法若不能超过 frozen mainline，应作为 appendix / diagnostics，而不是重新开启主线。
