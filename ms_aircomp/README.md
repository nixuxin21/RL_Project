# ms_aircomp Package Boundary

`ms_aircomp` is the reusable experiment layer for the MS-AirComp IRS project. It should contain stateless helpers and policy components that can be reused by evaluation, diagnosis, and training scripts.

The top-level experiment scripts remain responsible for CLI parsing, scenario orchestration, and legacy compatibility. Script-specific output, summary, and dispatch logic can be promoted here once it becomes reusable or large enough to guard. In particular, `evaluate_execution_channel_mismatch.py` is an experiment runner, not the default home for reusable helpers.

## Modules

| Module | Responsibility |
|---|---|
| `channel_models.py` | Physical channel snapshots, execution-channel drift, temporal AR(1) stale CSI, delayed CSI, AR(1) prediction, temporal uncertainty |
| `execution_candidates.py` | Drifted execution-stage candidates and hidden execution oracle helpers |
| `execution_decision_dispatch.py` | Execution-mismatch policy dispatch that maps policy names and config parameters to concrete decisions |
| `execution_output.py` | Execution-mismatch output prefix naming, progress logging, console summaries, and plots |
| `feedback.py` | Aggregate feedback record construction and feedback-based confirmed-index selection |
| `confirmation.py` | Current aggregate-feedback confirmation flow over a selected IRS candidate set |
| `invitation_mask_correction.py` | Aggregate-feedback invitation-mask target-count correction and reranking-mode ablation helpers |
| `probe_sets.py` | Ordered/diverse probe sets and coverage-aware sparse candidate selection |
| `adaptive_sparse_policies.py` | Adaptive Sparse-TopK v1/v2/v3 policy gates, history/uncertainty helpers, and local-neighbor candidate generation |
| `limited_csi.py` | Reusable limited-CSI policy constants, codebook grid selection, candidate construction, environment factory, progress logging, and limited-CSI slot execution |
| `execution_policies.py` | Rotating, stale-TopK, sparse-TopK, coverage-aware sparse, and neighbor-coverage feedback policies |
| `execution_policy_registry.py` | Execution-mismatch policy aliases, display labels, parameter-grid expansion, and mismatch scenario expansion |
| `execution_result_summary.py` | Execution-mismatch result schema, seed aggregation, confidence intervals, CSV rows, and CSV writing |
| `execution_risk_policies.py` | Execution-risk reliability re-scoring, adaptive execution-risk rotating policy, and opportunity-cost execution-risk policy |
| `learned_shortlist.py` | Learned sparse-shortlist feature construction, model loading, set-level variant scoring, and deployable learned-shortlist feedback policies |
| `temporal_policies.py` | Temporal-reliability rotating policy and temporal-deviation oracle probe-set diagnostic |
| `experiment_utils.py` | General experiment utilities: output directories, action mapping, seed generation, filename suffixes, energy accounting |

## Import Policy

Prefer direct module imports:

```python
from ms_aircomp.confirmation import confirm_index_with_current_feedback
from ms_aircomp.execution_policies import choose_coverage_sparse_topk_feedback_decision
```

Do not import helper symbols from top-level experiment runners such as
`evaluate_execution_channel_mismatch.py` or `evaluate_limited_csi_ms_aircomp.py`:

```python
from evaluate_execution_channel_mismatch import choose_sparse_topk_feedback_decision  # do not add
import evaluate_limited_csi_ms_aircomp as limited  # do not add inside ms_aircomp
```

Whole-module imports of the execution-mismatch evaluator are allowed only for orchestration/test runners that call its CLI/parser/evaluation functions. Limited-CSI reusable helpers should come from `ms_aircomp.limited_csi`; top-level limited evaluator imports are reserved for tests that exercise its reporting/summary surface. This boundary is checked by `tests/dependency_boundary_checks.py` and `make check`.

## API Stability

Each module declares `__all__` for the functions that are intended to be stable within this project. Underscore-free helpers not listed in `__all__` should be treated as implementation details until promoted.

## Refactor Rule

New reusable logic should flow into `ms_aircomp`, while experiment scripts should call into it. Avoid reverse imports from `ms_aircomp` back into orchestration scripts; that creates circular dependencies and makes later extraction harder.
