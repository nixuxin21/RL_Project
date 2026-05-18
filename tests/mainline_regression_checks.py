"""模块 `tests/mainline_regression_checks.py`：封装本项目实验、分析或测试所需的代码逻辑。"""

from argparse import Namespace
import contextlib
from pathlib import Path
import sys
from unittest import mock

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import evaluate_execution_channel_mismatch as execution_runner
import evaluate_invitation_mask_correction as mask_correction
import ms_aircomp.execution_decision_dispatch as execution_decision_dispatch
import ms_aircomp.execution_output as execution_output
import ms_aircomp.execution_policy_registry as execution_policy_registry
import ms_aircomp.execution_result_summary as execution_result_summary
from ms_aircomp.experiment_utils import make_episode_seeds


EPISODES = 16
BASE_SEED = 2026
CHANNEL_RHO = 0.7
CSI_DELAY = 1

MISMATCH_TEMPORAL_AR1 = "temporal_ar1"
POLICY_ROTATING_GRID = "Estimated Rotating Grid"
POLICY_COVERAGE_SPARSE_TOPK = "Coverage-Aware Sparse-TopK Feedback Grid"
POLICY_TEMPORAL_DEVIATION_ORACLE = "Temporal Deviation Oracle"

COVERAGE_GAP_CEILING = 3.3
ROTATING_OR_ORACLE_PREVIEW_BUDGET = 4.0
COVERAGE_PREVIEW_BUDGET = 16.0
MASK_CORRECTED_GAP_CEILING = 3.0
MASK_CORRECTED_MISSED_CEILING = 7.0
MASK_CORRECTED_FAILED_TOLERANCE = 0.25


def _assert_less(name, observed, threshold):
    if not observed < threshold:
        raise AssertionError(f"{name}: expected {observed:.6g} < {threshold:.6g}")


def _assert_less_equal(name, observed, threshold):
    if not observed <= threshold:
        raise AssertionError(f"{name}: expected {observed:.6g} <= {threshold:.6g}")


def _assert_close(name, observed, expected, atol=1e-9):
    if not np.isclose(observed, expected, atol=atol):
        raise AssertionError(f"{name}: expected {expected:.6g}, got {observed:.6g}")


def _assert_metric_below(results, left_key, right_key, label):
    _assert_less(label, results[left_key], results[right_key])


def _assert_preview_budget(result, expected, label):
    _assert_close(label, _mean(result, "decision_preview_calls_per_slot"), expected)


def _mean(result, key):
    return float(np.mean(result[key]))


def make_execution_args():
    argv = [
        "mainline_regression_checks.py",
        "--episodes",
        str(EPISODES),
        "--num-seeds",
        "1",
        "--seed",
        str(BASE_SEED),
        "--mismatch-models",
        "temporal_ar1",
        "--channel-rho-values",
        str(CHANNEL_RHO),
        "--csi-delay-slots",
        str(CSI_DELAY),
        "--decision-error-std-values",
        "0",
        "--execution-error-std-values",
        "0",
        "--probe-budgets",
        "3,4",
        "--policies",
        "rotating,coverage_sparse_topk_feedback,temporal_deviation_oracle",
        "--sparse-topk-seed-multipliers",
        "4.1",
        "--sparse-topk-fractions",
        "0.75",
        "--coverage-sparse-weights",
        "0.5",
        "--coverage-sparse-power-weights",
        "0",
        "--no-plots",
    ]
    with mock.patch.object(sys, "argv", argv):
        args = execution_runner.parse_args()
    execution_runner.validate_args(args)
    return args


def evaluate_mismatch_policy(args, policy_name, budget, **kwargs):
    episode_seeds = make_episode_seeds(args, BASE_SEED)
    with open("/dev/null", "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull):
            return execution_runner.evaluate_policy(
                episode_seeds,
                args,
                0.0,
                0.0,
                MISMATCH_TEMPORAL_AR1,
                CHANNEL_RHO,
                CSI_DELAY,
                policy_name,
                budget=budget,
                **kwargs,
            )


def assert_coverage_frontier_trend():
    args = make_execution_args()
    rotating_b4 = evaluate_mismatch_policy(
        args,
        POLICY_ROTATING_GRID,
        budget=4,
    )
    coverage_b3 = evaluate_mismatch_policy(
        args,
        POLICY_COVERAGE_SPARSE_TOPK,
        budget=3,
        sparse_topk_seed_multiplier=4.1,
        sparse_topk_fraction=0.75,
        coverage_sparse_weight=0.5,
        coverage_sparse_power_weight=0.0,
    )
    temporal_oracle_b4 = evaluate_mismatch_policy(
        args,
        POLICY_TEMPORAL_DEVIATION_ORACLE,
        budget=4,
    )

    frontier = {
        "rotating_b4_gap": _mean(rotating_b4, "oracle_tx_gap_mean"),
        "coverage_b3_gap": _mean(coverage_b3, "oracle_tx_gap_mean"),
        "temporal_oracle_b4_gap": _mean(temporal_oracle_b4, "oracle_tx_gap_mean"),
    }

    _assert_metric_below(
        frontier,
        "coverage_b3_gap",
        "rotating_b4_gap",
        "Coverage-Aware B3 gap should stay below Rotating B4",
    )
    _assert_less("Coverage-Aware B3 fixed-seed gap ceiling", frontier["coverage_b3_gap"], COVERAGE_GAP_CEILING)
    _assert_metric_below(
        frontier,
        "temporal_oracle_b4_gap",
        "rotating_b4_gap",
        "Temporal Deviation Oracle should stay below Rotating B4",
    )
    _assert_less_equal(
        "Temporal Deviation Oracle should not trail Coverage-Aware B3",
        frontier["temporal_oracle_b4_gap"],
        frontier["coverage_b3_gap"],
    )
    _assert_preview_budget(rotating_b4, ROTATING_OR_ORACLE_PREVIEW_BUDGET, "Rotating B4 preview budget")
    _assert_preview_budget(coverage_b3, COVERAGE_PREVIEW_BUDGET, "Coverage-Aware B3 preview budget")
    _assert_preview_budget(
        temporal_oracle_b4,
        ROTATING_OR_ORACLE_PREVIEW_BUDGET,
        "Temporal Deviation Oracle B4 preview budget",
    )
    return {
        "rotating_b4_gap": frontier["rotating_b4_gap"],
        "coverage_b3_gap": frontier["coverage_b3_gap"],
        "temporal_oracle_b4_gap": frontier["temporal_oracle_b4_gap"],
    }


def make_mask_correction_args():
    return Namespace(
        episodes=EPISODES,
        num_seeds=1,
        seed=BASE_SEED,
        seed_stride=1000,
        num_nodes=50,
        num_slots=10,
        num_irs_elements=64,
        num_codebook_states=16,
        g_th=0.001,
        alpha_th=0.05,
        channel_rho_values=[CHANNEL_RHO],
        csi_delay_slots=[CSI_DELAY],
        decision_error_std=0.0,
        execution_error_std=0.0,
        probe_budget=3,
        sparse_topk_seed_multiplier=4.1,
        sparse_topk_fraction=0.75,
        coverage_sparse_weight=0.5,
        coverage_sparse_power_weight=0.0,
        confirmation_feedback_noise_std=0.0,
        confirmation_feedback_noise_std_values=[0.0],
        confirmation_feedback_power_weight=0.05,
        mask_correction_strengths=[0.0, 1.0],
        mask_correction_noise_deadband_z_values=[0.0],
        mask_correction_max_delta_values=[-1.0],
        mask_correction_rerank_modes=["global_stale_gain"],
    )


def evaluate_mask_correction(args, strength):
    episode_seeds = make_episode_seeds(args, BASE_SEED)
    rows = [
        mask_correction.run_episode(
            args,
            episode_seed,
            CHANNEL_RHO,
            CSI_DELAY,
            strength,
            0.0,
            -1.0,
            "global_stale_gain",
        )
        for episode_seed in episode_seeds
    ]
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }


def assert_mask_correction_trend():
    args = make_mask_correction_args()
    uncorrected = evaluate_mask_correction(args, 0.0)
    corrected = evaluate_mask_correction(args, 1.0)

    _assert_less(
        "Mask correction should reduce fixed-seed oracle gap",
        corrected["oracle_gap"],
        uncorrected["oracle_gap"],
    )
    _assert_less(
        "Mask correction should reduce fixed-seed missed opportunities",
        corrected["missed_opportunities"],
        uncorrected["missed_opportunities"],
    )
    _assert_less("Mask-corrected fixed-seed gap ceiling", corrected["oracle_gap"], MASK_CORRECTED_GAP_CEILING)
    _assert_less(
        "Mask-corrected missed-opportunity ceiling",
        corrected["missed_opportunities"],
        MASK_CORRECTED_MISSED_CEILING,
    )
    _assert_less_equal(
        "Mask correction failed invitations should not grow materially",
        corrected["failed_nodes"],
        uncorrected["failed_nodes"] + MASK_CORRECTED_FAILED_TOLERANCE,
    )
    _assert_close("Uncorrected Coverage-Aware preview budget", uncorrected["preview_calls"], COVERAGE_PREVIEW_BUDGET)
    _assert_close("Mask-Corrected Coverage-Aware preview budget", corrected["preview_calls"], COVERAGE_PREVIEW_BUDGET)
    _assert_less("Mask correction should be exercised by this fixed seed", 0.0, corrected["mask_applied"])
    return {
        "uncorrected_gap": uncorrected["oracle_gap"],
        "corrected_gap": corrected["oracle_gap"],
        "uncorrected_missed": uncorrected["missed_opportunities"],
        "corrected_missed": corrected["missed_opportunities"],
    }


def assert_external_result_aggregation_defaults():
    minimal_result = {
        "name": "External Temporal Diagnostic",
        "decision_error_std": 0.0,
        "execution_error_std": 0.0,
        "mismatch_model": MISMATCH_TEMPORAL_AR1,
        "channel_rho": CHANNEL_RHO,
        "csi_delay_slots": CSI_DELAY,
        "probe_budget": 4,
        "success_nodes": np.asarray([49.0, 50.0], dtype=float),
    }
    aggregated = execution_runner.aggregate_seed_results([[minimal_result]])
    if len(aggregated) != 1:
        raise AssertionError("external diagnostic aggregation should return one result")
    result = aggregated[0]
    _assert_close("default sparse seed multiplier", result["sparse_topk_seed_multiplier"], 2.0)
    _assert_close(
        "default adaptive expansion diagnostics",
        float(np.mean(result["adaptive_sparse_expanded"])),
        0.0,
    )
    summary_args = Namespace(
        num_seeds=1,
        num_nodes=50,
        num_slots=10,
        num_irs_elements=64,
        num_codebook_states=16,
        g_th=0.001,
        alpha_th=0.05,
    )
    rows = execution_runner.summarize_results(summary_args, aggregated)
    if len(rows) != 1:
        raise AssertionError("external diagnostic summary should return one CSV row")
    row = rows[0]
    _assert_close("external diagnostic success mean", row["success_mean"], 49.5)
    _assert_close("external diagnostic perfect rate", row["perfect_rate"], 50.0)
    _assert_close("external diagnostic default avg power", row["avg_power"], 0.0)
    _assert_close("external diagnostic default preview mean", row["decision_preview_calls_per_slot_mean"], 0.0)
    _assert_close("external diagnostic default sparse seed multiplier row", row["sparse_topk_seed_multiplier"], 2.0)


def assert_execution_policy_registry_legacy_exports():
    if execution_runner.POLICY_CHOICES is not execution_policy_registry.POLICY_CHOICES:
        raise AssertionError("execution runner should re-export POLICY_CHOICES from the registry")
    if execution_runner.policy_configs is not execution_policy_registry.policy_configs:
        raise AssertionError("execution runner should re-export policy_configs from the registry")
    if execution_runner.policy_label is not execution_policy_registry.policy_label:
        raise AssertionError("execution runner should re-export policy_label from the registry")
    if execution_runner.mismatch_scenarios is not execution_policy_registry.mismatch_scenarios:
        raise AssertionError("execution runner should re-export mismatch_scenarios from the registry")


def assert_execution_result_summary_legacy_exports():
    if execution_runner.NUMERIC_RESULT_KEYS is not execution_result_summary.NUMERIC_RESULT_KEYS:
        raise AssertionError("execution runner should re-export NUMERIC_RESULT_KEYS from result summary")
    if execution_runner.OPTIONAL_RESULT_DEFAULTS is not execution_result_summary.OPTIONAL_RESULT_DEFAULTS:
        raise AssertionError("execution runner should re-export OPTIONAL_RESULT_DEFAULTS from result summary")
    if execution_runner.CSV_FIELDS is not execution_result_summary.CSV_FIELDS:
        raise AssertionError("execution runner should re-export CSV_FIELDS from result summary")
    if execution_runner.aggregate_seed_results is not execution_result_summary.aggregate_seed_results:
        raise AssertionError("execution runner should re-export aggregate_seed_results from result summary")
    if execution_runner.summarize_results is not execution_result_summary.summarize_results:
        raise AssertionError("execution runner should re-export summarize_results from result summary")
    if execution_runner.write_csv is not execution_result_summary.write_csv:
        raise AssertionError("execution runner should re-export write_csv from result summary")


def assert_execution_decision_dispatch_legacy_exports():
    if execution_runner.choose_decision is not execution_decision_dispatch.choose_decision:
        raise AssertionError("execution runner should re-export choose_decision from dispatch")
    if execution_runner.choose_execution_mismatch_decision is not execution_decision_dispatch.choose_execution_mismatch_decision:
        raise AssertionError(
            "execution runner should re-export choose_execution_mismatch_decision from dispatch"
        )


def assert_execution_output_legacy_exports():
    if execution_runner.resolve_output_prefix is not execution_output.resolve_output_prefix:
        raise AssertionError("execution runner should re-export resolve_output_prefix from output helpers")
    if execution_runner.print_progress is not execution_output.print_progress:
        raise AssertionError("execution runner should re-export print_progress from output helpers")
    if execution_runner.print_summary is not execution_output.print_summary:
        raise AssertionError("execution runner should re-export print_summary from output helpers")
    if execution_runner.plot_results is not execution_output.plot_results:
        raise AssertionError("execution runner should re-export plot_results from output helpers")


def main():
    coverage_summary = assert_coverage_frontier_trend()
    correction_summary = assert_mask_correction_trend()
    assert_external_result_aggregation_defaults()
    assert_execution_policy_registry_legacy_exports()
    assert_execution_result_summary_legacy_exports()
    assert_execution_decision_dispatch_legacy_exports()
    assert_execution_output_legacy_exports()
    print("mainline regression checks passed")
    print(
        "  coverage frontier: "
        f"rotating_b4_gap={coverage_summary['rotating_b4_gap']:.3f}, "
        f"coverage_b3_gap={coverage_summary['coverage_b3_gap']:.3f}, "
        f"temporal_oracle_b4_gap={coverage_summary['temporal_oracle_b4_gap']:.3f}"
    )
    print(
        "  mask correction: "
        f"gap {correction_summary['uncorrected_gap']:.3f}"
        f" -> {correction_summary['corrected_gap']:.3f}, "
        f"missed {correction_summary['uncorrected_missed']:.3f}"
        f" -> {correction_summary['corrected_missed']:.3f}"
    )


if __name__ == "__main__":
    main()
