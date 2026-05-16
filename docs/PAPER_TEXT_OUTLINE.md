# Paper Text Outline

本文件是论文正文写作前的最小骨架，不是论文草稿。它把每一节必须表达的 claim、对应证据、必须避免的误写和图表引用顺序固定下来，避免正式写作时重新把历史实验混入正文。

结果包见 `docs/PAPER_RESULT_PACKAGE.md`，章节和图表放置见 `docs/PAPER_STRUCTURE_MAP.md`，图表规格见 `docs/PAPER_FIGURE_TABLE_SPECS.md`，附录边界见 `docs/PAPER_APPENDIX_BOUNDARY.md`，投稿前资产缺口见 `docs/PAPER_ASSET_GAP_CHECKLIST.md`。

## Paper Spine

正文主线只围绕下面一句话展开：

> Under stale CSI and execution-channel mismatch, IRS-assisted MS-AirComp has two coupled decisions: which IRS states to probe and which devices to invite. Low-cost coverage-aware candidate generation addresses the IRS-state search side, while aggregate-feedback invitation-mask correction addresses the stale invitation side.

Do not frame the paper as:

- a generic reinforcement-learning paper;
- a broad baseline leaderboard;
- a full-current-CSI optimization paper;
- a learned shortlist paper;
- a pure IRS codebook search paper without invitation-mask mismatch.

## Abstract Skeleton

Must say:

1. The deployment setting has stale CSI and execution-channel mismatch.
2. Full current-channel CSI and exhaustive preview are not deployable assumptions.
3. The method uses low-cost coverage-aware IRS candidate generation plus aggregate current feedback.
4. Aggregate feedback is used twice: IRS confirmation and invitation-mask correction.
5. The main method is `Mask-Corrected Coverage-Aware B=3 mc=1`; high-noise feedback uses clipped correction only as a robustness variant.

Evidence to cite later in the paper:

- Table 1: same-preview main result.
- Figure 2: preview-gap frontier.
- Figure 3: failed/missed mechanism.
- Figure 4: noise robustness.

Must not say:

- that the method observes per-device current CSI;
- that `Temporal Deviation Oracle B=4` is deployable;
- that `mc=1 clip=2` replaces the reliable-feedback main method.

## Introduction

### Required Claims

| Claim | Evidence | Placement |
|---|---|---|
| Stale CSI and execution-channel mismatch make full-current-CSI scheduling unrealistic. | `docs/PAPER_RESULT_PACKAGE.md`, `docs/MAIN_RESULTS_ANALYSIS.md` | Opening motivation |
| IRS-assisted MS-AirComp under limited feedback has two coupled decisions: IRS state and invited devices. | Figure 1 source: `docs/figures/figure1_system_flow.mmd` | Problem paragraph |
| Low-cost candidate generation alone improves the frontier but leaves invitation-mask mismatch. | Figure 2, Figure 3, Table 2, Table 3 | Gap paragraph |
| Aggregate-feedback invitation-mask correction improves both failed and missed opportunities at the same preview budget. | Table 1, Figure 3 | Main result paragraph |
| High-noise aggregate feedback needs a conservative clipped variant. | Figure 4 | Robustness paragraph |

### Contribution Bullets

Use exactly these contribution types unless new evidence is added:

1. A stale-CSI execution-mismatch evaluation framework for IRS-assisted multi-slot AirComp with limited preview and aggregate current feedback.
2. A low-cost coverage-aware sparse candidate-generation strategy that improves same-preview performance by reducing missed opportunities.
3. An aggregate-feedback invitation-mask correction mechanism that improves both failed invitations and missed opportunities without changing IRS candidate generation.
4. Optional, if space allows: a robustness analysis showing clipped target-count correction is preferable under high aggregate-feedback noise.

### Keep Out

- SAC / Codebook-Aware SAC details.
- Imitation-learning details.
- Learned shortlist training details.
- A long historical timeline of all pilots.
- Claims about journal novelty beyond what the current artifacts support.

## Related Work

### Required Buckets

| Bucket | Role In Paper | Local Evidence |
|---|---|---|
| IRS-assisted AirComp | Establish physical communication context. | `test_env.py`, `docs/PAPER_RESULT_PACKAGE.md` |
| Limited CSI / stale CSI scheduling | Motivate execution-channel mismatch. | `ms_aircomp/channel_models.py`, `docs/PAPER_RESULT_PACKAGE.md` |
| Limited probing / feedback-based selection | Position aggregate feedback and preview budget. | `ms_aircomp/confirmation.py`, `ms_aircomp/feedback.py` |
| Learning-based IRS or scheduling methods | Explain why this paper does not rely on RL as the main method. | `docs/PAPER_APPENDIX_BOUNDARY.md`, `docs/DEPRECATED_DIRECTIONS.md` |

### Keep Out

- No result tables.
- No internal pilot chronology.
- No claim that learning methods are generally bad; only say they are not the strongest route in this artifact set.

## System Model

### Required Content

| Topic | Required Detail | Evidence |
|---|---|---|
| Physical setup | IRS-assisted multi-slot AirComp with `K=50`, `N=10`, `M=64`, `C=16`. | `test_env.py`, `docs/PAPER_RESULT_PACKAGE.md` |
| IRS codebook | DFT codebook, finite IRS state set. | `test_env.py` |
| Stale CSI | Decisions use stale CSI snapshot, not hidden current channel. | `ms_aircomp/channel_models.py` |
| Temporal AR(1) scenarios | `channel_rho in {0.7, 0.9, 0.98}`, `csi_delay_slots in {1, 2, 3}`. | `docs/PAPER_RESULT_PACKAGE.md` |
| Feedback | Deployable feedback is aggregate count-level current feedback. | `ms_aircomp/feedback.py`, `ms_aircomp/confirmation.py` |
| Metrics | slots, failed invitations, missed opportunities, preview calls, oracle tx gap. | `docs/PAPER_TABLE1_MAIN_RESULTS.md`, `docs/MAIN_RESULTS_ANALYSIS.md` |

### Keep Out

- SAC reward design.
- Training hyperparameters for learned variants.
- Hidden current-channel oracle as an algorithmic input.

## Problem Formulation

### Required Claims

1. Deployable policies can use stale CSI, limited IRS previews, and aggregate current feedback.
2. Deployable policies cannot use per-device current CSI.
3. The hidden current-channel oracle is an evaluation / diagnostic upper bound.
4. The main budget comparison is same-preview `16` for Sparse-TopK / Coverage-Aware / Mask-Corrected B3.
5. Failed invitations and missed opportunities are both needed because a method can reduce one while increasing the other.

### Suggested Notation Boundary

Keep notation limited to:

- IRS codebook and selected IRS state;
- stale vs current channel snapshot;
- candidate IRS set;
- aggregate feedback count;
- stale invitation mask and corrected invitation mask;
- metrics.

Do not introduce extra RL policy notation, learned scorer notation, or appendix-only variants into the main formulation.

## Method

### Section Order

| Subsection | Must Explain | Primary Evidence |
|---|---|---|
| Candidate generation | Rotating, Sparse-TopK, Coverage-Aware and why B3 budget split is retained. | `docs/COVERAGE_AWARE_ANALYSIS.md`, `ms_aircomp/probe_sets.py` |
| Aggregate confirmation | Candidate IRS states are confirmed using aggregate current feedback only. | `ms_aircomp/confirmation.py`, `ms_aircomp/feedback.py` |
| Invitation-mask correction | Confirmed IRS remains fixed; aggregate feedback count corrects stale invitation mask cardinality. | `evaluate_invitation_mask_correction.py`, `docs/INVITATION_MASK_CORRECTION.md` |
| High-noise clipped variant | Clip target-count correction under noisy aggregate feedback. | `docs/INVITATION_MASK_CORRECTION_NOISE_AWARE.md`, Figure 4 |

### Method Naming

Use `docs/PAPER_FIGURE_TABLE_SPECS.md` naming crosswalk:

- Main method: `Mask-Corrected Coverage-Aware B=3 mc=1`.
- Figure 4 `Direct`: same method under direct target-count correction.
- Figure 4 `Clip2`: high-noise robustness variant only.
- Oracle: `Temporal Deviation Oracle B=4`, hidden-information diagnostic only.

### Keep Out

- Do not describe learned shortlist internals in the main method.
- Do not present `Adaptive Sparse-TopK v2` as the proposed method.
- Do not state or imply that mask correction changes IRS candidate generation.

## Experiments

### Setup Paragraph

Must include:

- 9 temporal AR(1) stale-CSI scenarios.
- Equal-weight averages over `rho/delay` combinations.
- Main environment constants from `docs/PAPER_RESULT_PACKAGE.md`.
- Frozen main-text method set.
- Preview budget and metrics.
- Hidden oracle boundary.

### Result Flow

| Order | Evidence | Required Message |
|---|---|---|
| Table 1 | `docs/PAPER_TABLE1_MAIN_RESULTS.md`, `results/paper/table1_main_results.csv` | `Mask-Corrected Coverage-Aware B=3 mc=1` is the strongest same-preview method and improves failed/missed simultaneously. |
| Figure 2 | `results/paper/figure2_preview_gap_frontier.png`, `results/paper/figure2_figure3_points.csv` | Same-preview quality improves from Sparse-TopK to Coverage-Aware to Mask-Corrected B3; oracle remains diagnostic headroom. |
| Figure 3 | `results/paper/figure3_failed_missed_tradeoff.png`, `results/paper/figure2_figure3_points.csv` | Candidate generation trades failed against missed; mask correction improves both. |
| Table 2 | `docs/COVERAGE_AWARE_ANALYSIS.md`, `results/execution_mismatch/coverage_aware_ablation_analysis.csv` | `cw=0.5 cpw=0` and `B=3 sm=4.1` are retained as interpretable same-preview choices. |
| Table 3 | `docs/COVERAGE_B3_FAILURE_DIAGNOSIS.md` | Residual gap is dominated by invitation-mask mismatch. |
| Figure 4 | `results/paper/figure4_invitation_mask_gap_noise.png`, `results/paper/figure4_invitation_mask_failed_missed_noise.png`, `results/paper/figure4_invitation_mask_noise_points.csv` | Direct correction is the reliable-feedback main method; clipping is a high-noise robustness variant. |

### Keep Out

- Early policy comparison as a main result.
- `Rotating B=4` as the main low-cost baseline.
- Detailed learned shortlist variants.
- Any result not listed in `docs/PAPER_RESULT_PACKAGE.md` or `docs/PAPER_APPENDIX_BOUNDARY.md`.

## Discussion

### Required Points

1. The method is deployable under aggregate current feedback, not per-device current CSI.
2. The remaining gap to `Temporal Deviation Oracle B=4` is a diagnostic headroom, not a failure of the deployable method.
3. High-noise aggregate feedback changes the correction rule: clipped target-count correction is safer at std `0.1`.
4. Learning-based and ordinary heuristic branches are retained as diagnostics, not discarded.
5. Future work should target invitation-mask mismatch and aggregate-feedback robustness, not broad heuristic proliferation.

### Keep Out

- Overclaiming journal impact.
- Calling negative branches useless.
- Promising unrun experiments.

## Appendix Minimum

Use `docs/PAPER_APPENDIX_BOUNDARY.md`.

Default appendix:

1. Appendix A: historical baseline context.
2. Appendix B: preview-cost and CSI motivation.
3. Appendix C: diagnostic branches not used as proposed methods.

Do not add supplement-only diagnostics unless reviewer requirements force them.

## Claim Traceability Checklist

Before drafting each paragraph, verify:

1. Every numerical claim points to Table 1, Figure 2/3/4, Table 2/3, or an analysis document listed above.
2. Every method name matches `docs/PAPER_FIGURE_TABLE_SPECS.md`.
3. Every appendix reference is allowed by `docs/PAPER_APPENDIX_BOUNDARY.md`.
4. Hidden current-channel information is described only as oracle / diagnostic.
5. `Mask-Corrected Coverage-Aware B=3 mc=1` remains the reliable-feedback main method.
6. `mc=1 clip=2` remains high-noise robustness only.
7. No learned or RL branch is reintroduced as a main method.

## Pre-Draft Commands

Run these before writing the actual manuscript:

```bash
make paper-table1
make paper-figures
make docs
make mainline-audit
make check
```

Also check `docs/PAPER_ASSET_GAP_CHECKLIST.md` before manuscript assembly. Figure 1 export and compact Table 2/3 paper artifacts are presentation tasks, not new experiments.
