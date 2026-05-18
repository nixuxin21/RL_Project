"""评估邀请掩码修正：在确认 IRS 后用聚合反馈数量修正 stale invitation mask。"""

import argparse
import csv
import os

import numpy as np

import ms_aircomp.limited_csi as limited
from ms_aircomp.channel_models import (
    apply_channel_state,
    build_temporal_channel_trace,
    delayed_channel_state,
)
from ms_aircomp.execution_candidates import (
    execution_candidate_for_decision,
    execution_oracle_candidate,
)
from ms_aircomp.confirmation import confirm_index_with_current_feedback
from ms_aircomp.experiment_utils import (
    ensure_parent_dir,
    format_float_for_suffix,
    make_episode_seeds,
    make_run_seeds,
    validate_nonempty_values,
    validate_common_experiment_args,
    validate_nonnegative_values,
    validate_positive_values,
    validate_probability_values,
    validate_probe_budget_values,
)
from ms_aircomp.invitation_mask_correction import (
    MASK_CORRECTION_MODE_GLOBAL_STALE_GAIN,
    apply_invitation_mask_correction,
    corrected_target_count,
    rank_remaining_by_stale_gain,
    validate_mask_correction_rerank_mode,
)
from ms_aircomp.probe_sets import coverage_aware_sparse_indices
from test_env import MSAirCompEnv


DEFAULT_RESULTS_DIR = os.path.join("results", "execution_mismatch")
DEFAULT_DOC_OUTPUT = os.path.join("docs", "INVITATION_MASK_CORRECTION.md")

CSV_FIELDS = [
    "policy",
    "channel_rho",
    "csi_delay_slots",
    "episodes",
    "num_seeds",
    "mask_correction_strength",
    "mask_correction_noise_deadband_z",
    "mask_correction_max_delta",
    "mask_correction_rerank_mode",
    "confirmation_feedback_noise_std",
    "success_mean",
    "perfect_rate",
    "slots_mean",
    "failed_nodes_mean",
    "missed_opportunities_mean",
    "true_opportunities_mean",
    "decision_preview_calls_per_slot_mean",
    "oracle_tx_gap_mean",
    "mask_correction_added_mean",
    "mask_correction_pruned_mean",
    "mask_correction_requested_target_delta_mean",
    "mask_correction_target_delta_mean",
    "mask_correction_unmet_additions_mean",
    "mask_correction_applied_rate",
]


def parse_args():
    """解析命令行参数，集中声明实验规模、策略配置、输入输出路径和绘图开关。"""
    parser = argparse.ArgumentParser(
        description="Evaluate aggregate-feedback invitation mask correction."
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--num-seeds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--seed-stride", type=int, default=1000)
    parser.add_argument("--num-nodes", type=int, default=50)
    parser.add_argument("--num-slots", type=int, default=10)
    parser.add_argument("--num-irs-elements", type=int, default=64)
    parser.add_argument("--num-codebook-states", type=int, default=16)
    parser.add_argument("--g-th", type=float, default=0.001)
    parser.add_argument("--alpha-th", type=float, default=0.05)
    parser.add_argument("--channel-rho-values", default="0.7,0.9,0.98")
    parser.add_argument("--csi-delay-slots", default="1,2,3")
    parser.add_argument("--decision-error-std", type=float, default=0.0)
    parser.add_argument("--execution-error-std", type=float, default=0.0)
    parser.add_argument("--probe-budget", type=int, default=3)
    parser.add_argument("--sparse-topk-seed-multiplier", type=float, default=4.1)
    parser.add_argument("--sparse-topk-fraction", type=float, default=0.75)
    parser.add_argument("--coverage-sparse-weight", type=float, default=0.5)
    parser.add_argument("--coverage-sparse-power-weight", type=float, default=0.0)
    parser.add_argument("--confirmation-feedback-noise-std", type=float, default=0.0)
    parser.add_argument(
        "--confirmation-feedback-noise-std-values",
        default=None,
        help=(
            "Optional comma-separated noise std sweep. If omitted, the scalar "
            "--confirmation-feedback-noise-std value is used."
        ),
    )
    parser.add_argument("--confirmation-feedback-power-weight", type=float, default=0.05)
    parser.add_argument(
        "--mask-correction-strengths",
        default="0,0.5,1",
        help="Comma-separated interpolation strengths; 0 reproduces uncorrected B3.",
    )
    parser.add_argument(
        "--mask-correction-noise-deadband-z-values",
        default="0",
        help=(
            "Comma-separated z multipliers for the feedback-count noise deadband. "
            "The corrected delta is shrunk by z * noise_std * num_nodes."
        ),
    )
    parser.add_argument(
        "--mask-correction-max-delta-values",
        default="-1",
        help="Comma-separated max absolute target-count deltas; -1 disables clipping.",
    )
    parser.add_argument(
        "--mask-correction-rerank-modes",
        default=MASK_CORRECTION_MODE_GLOBAL_STALE_GAIN,
        help=(
            "Comma-separated invitation-mask rerank modes. "
            "global_stale_gain preserves the existing method; prune_only is "
            "an ablation that never adds stale-invalid nodes."
        ),
    )
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--doc-output", default=DEFAULT_DOC_OUTPUT)
    return parser.parse_args()


def parse_float_list(value):
    """解析浮点数、列表参数，通常把逗号分隔的命令行字符串转换成类型明确的 Python 列表。"""
    return [float(item) for item in str(value).split(",") if str(item).strip()]


def parse_int_list(value):
    """解析整数、列表参数，通常把逗号分隔的命令行字符串转换成类型明确的 Python 列表。"""
    return [int(item) for item in str(value).split(",") if str(item).strip()]


def parse_string_list(value):
    """解析字符串、列表参数，通常把逗号分隔的命令行字符串转换成类型明确的 Python 列表。"""
    return [str(item).strip() for item in str(value).split(",") if str(item).strip()]


def normalize_args(args):
    """处理normalize、参数相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    args.channel_rho_values = parse_float_list(args.channel_rho_values)
    args.csi_delay_slots = parse_int_list(args.csi_delay_slots)
    args.mask_correction_strengths = parse_float_list(args.mask_correction_strengths)
    args.mask_correction_noise_deadband_z_values = parse_float_list(
        args.mask_correction_noise_deadband_z_values
    )
    args.mask_correction_max_delta_values = parse_float_list(
        args.mask_correction_max_delta_values
    )
    args.mask_correction_rerank_modes = parse_string_list(
        args.mask_correction_rerank_modes
    )
    if args.confirmation_feedback_noise_std_values is None:
        args.confirmation_feedback_noise_std_values = [float(args.confirmation_feedback_noise_std)]
    else:
        args.confirmation_feedback_noise_std_values = parse_float_list(
            args.confirmation_feedback_noise_std_values
        )
    return args


def validate_args(args):
    """校验解析后的命令行参数，尽早拒绝非法规模、预算或概率配置。"""
    validate_common_experiment_args(args)
    args.probe_budget = validate_probe_budget_values(
        [args.probe_budget],
        args.num_codebook_states,
        "--probe-budget",
    )[0]

    validate_probability_values(args.channel_rho_values, "--channel-rho-values")
    validate_nonnegative_values(args.csi_delay_slots, "--csi-delay-slots")
    validate_nonnegative_values([args.decision_error_std], "--decision-error-std")
    validate_nonnegative_values([args.execution_error_std], "--execution-error-std")
    validate_positive_values([args.sparse_topk_seed_multiplier], "--sparse-topk-seed-multiplier")
    validate_positive_values([args.sparse_topk_fraction], "--sparse-topk-fraction")
    if args.sparse_topk_fraction > 1.0:
        raise ValueError("--sparse-topk-fraction must be in (0, 1]")
    validate_nonnegative_values([args.coverage_sparse_weight], "--coverage-sparse-weight")
    validate_nonnegative_values([args.coverage_sparse_power_weight], "--coverage-sparse-power-weight")
    validate_nonnegative_values(
        [args.confirmation_feedback_noise_std],
        "--confirmation-feedback-noise-std",
    )
    validate_nonnegative_values(
        args.confirmation_feedback_noise_std_values,
        "--confirmation-feedback-noise-std-values",
    )
    validate_nonnegative_values(
        [args.confirmation_feedback_power_weight],
        "--confirmation-feedback-power-weight",
    )
    validate_probability_values(args.mask_correction_strengths, "--mask-correction-strengths")
    validate_nonnegative_values(
        args.mask_correction_noise_deadband_z_values,
        "--mask-correction-noise-deadband-z-values",
    )
    validate_nonempty_values(
        args.mask_correction_max_delta_values,
        "--mask-correction-max-delta-values",
    )
    if any(not np.isfinite(float(max_delta)) for max_delta in args.mask_correction_max_delta_values):
        raise ValueError("--mask-correction-max-delta-values must contain finite values")
    if any(max_delta < -1.0 for max_delta in args.mask_correction_max_delta_values):
        raise ValueError("--mask-correction-max-delta-values must be >= -1")
    validate_nonempty_values(
        args.mask_correction_rerank_modes,
        "--mask-correction-rerank-modes",
    )
    args.mask_correction_rerank_modes = [
        validate_mask_correction_rerank_mode(mode)
        for mode in args.mask_correction_rerank_modes
    ]


def resolve_output_prefix(args):
    """处理resolve、输出、前缀相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    if args.output_prefix:
        ensure_parent_dir(args.output_prefix)
        return args.output_prefix
    rho_label = "-".join(format_float_for_suffix(value) for value in args.channel_rho_values)
    delay_label = "-".join(str(value) for value in args.csi_delay_slots)
    strength_label = "-".join(format_float_for_suffix(value) for value in args.mask_correction_strengths)
    deadband_label = "-".join(
        format_float_for_suffix(value) for value in args.mask_correction_noise_deadband_z_values
    )
    max_delta_label = "-".join(
        format_float_for_suffix(value) for value in args.mask_correction_max_delta_values
    )
    noise_label = "-".join(
        format_float_for_suffix(value) for value in args.confirmation_feedback_noise_std_values
    )
    prefix = (
        "invitation_mask_correction_pilot_"
        f"ep{args.episodes}_runs{args.num_seeds}_"
        f"rho{rho_label}_delay{delay_label}_"
        f"b{args.probe_budget}_"
        f"sm{format_float_for_suffix(args.sparse_topk_seed_multiplier)}_"
        f"tf{format_float_for_suffix(args.sparse_topk_fraction)}_"
        f"cw{format_float_for_suffix(args.coverage_sparse_weight)}_"
        f"cpw{format_float_for_suffix(args.coverage_sparse_power_weight)}_"
        f"mc{strength_label}"
    )
    if (
        len(args.mask_correction_noise_deadband_z_values) != 1
        or abs(args.mask_correction_noise_deadband_z_values[0]) > 1e-12
    ):
        prefix = f"{prefix}_dbz{deadband_label}"
    if (
        len(args.mask_correction_max_delta_values) != 1
        or args.mask_correction_max_delta_values[0] >= 0.0
    ):
        prefix = f"{prefix}_clip{max_delta_label}"
    if len(args.confirmation_feedback_noise_std_values) != 1 or abs(args.confirmation_feedback_noise_std_values[0]) > 1e-12:
        prefix = f"{prefix}_fbn{noise_label}"
    if args.mask_correction_rerank_modes != [MASK_CORRECTION_MODE_GLOBAL_STALE_GAIN]:
        mode_label = "-".join(args.mask_correction_rerank_modes)
        prefix = f"{prefix}_mode{mode_label}"
    return os.path.join(DEFAULT_RESULTS_DIR, prefix)


def make_env(args):
    """构建env所需的数据结构，供评估循环、训练流程或报告生成继续使用。"""
    return MSAirCompEnv(
        num_nodes=args.num_nodes,
        num_slots=args.num_slots,
        num_irs_elements=args.num_irs_elements,
        num_codebook_states=args.num_codebook_states,
    )


def choose_coverage_b3_decision(
    args,
    env,
    episode_seed,
    slot_idx,
    execution_state,
    strength,
    deadband_z,
    max_delta,
    rerank_mode,
):
    """按照覆盖感知、b3、决策规则选择候选或索引，并返回后续执行、确认或聚合需要的信息。"""
    budget = min(int(args.probe_budget), int(args.num_codebook_states))
    seed_budget = min(
        int(args.num_codebook_states),
        max(budget, int(np.ceil(float(args.sparse_topk_seed_multiplier) * budget))),
    )
    seed_indices = limited.grid_indices(
        args.num_codebook_states,
        seed_budget,
        offset=slot_idx,
    )
    error_rng = limited.stable_rng(
        episode_seed,
        args.decision_error_std,
        limited.POLICY_ROTATING_GRID,
        seed_budget,
        salt=191 + slot_idx,
    )
    seed_candidates = limited.estimated_preview_candidates(
        env,
        args,
        indices=seed_indices,
        error_std=args.decision_error_std,
        rng=error_rng,
    )
    selected_indices, _anchor_count, _marginal_mean, _overlap_mean = coverage_aware_sparse_indices(
        seed_candidates,
        args,
        budget,
        args.sparse_topk_fraction,
        args.coverage_sparse_weight,
        args.coverage_sparse_power_weight,
    )
    candidate_by_index = {int(candidate["irs_index"]): candidate for candidate in seed_candidates}
    if len(selected_indices) < budget:
        ranked_indices = [
            int(candidate["irs_index"])
            for candidate in sorted(seed_candidates, key=limited.candidate_key, reverse=True)
        ]
        selected_indices = []
        seen = set()
        for index in ranked_indices:
            if int(index) in seen:
                continue
            selected_indices.append(int(index))
            seen.add(int(index))
            if len(selected_indices) >= budget:
                break

    confirmed_index, feedbacks = confirm_index_with_current_feedback(
        env,
        args,
        selected_indices,
        args.execution_error_std,
        slot_idx,
        episode_seed,
        execution_state=execution_state,
        feedback_salt=191,
    )
    decision = dict(candidate_by_index[int(confirmed_index)])
    feedback_by_index = {int(feedback["irs_index"]): feedback for feedback in feedbacks}
    decision["confirmed_irs_index"] = int(confirmed_index)
    decision["confirmation_feedback_count"] = int(len(feedbacks))
    decision["mask_correction_strength"] = float(strength)
    decision["mask_correction_noise_deadband_z"] = float(deadband_z)
    decision["mask_correction_max_delta"] = float(max_delta)
    decision["mask_correction_rerank_mode"] = "none"
    decision["mask_correction_stale_count"] = int(decision["tx_this_slot"])
    decision["mask_correction_feedback_count"] = int(
        round(float(feedback_by_index[int(confirmed_index)]["observed_tx_fraction"]) * float(args.num_nodes))
    )
    decision["mask_correction_requested_target_count"] = int(decision["tx_this_slot"])
    decision["mask_correction_target_count"] = int(decision["tx_this_slot"])
    decision["mask_correction_added"] = 0
    decision["mask_correction_pruned"] = 0
    decision["mask_correction_requested_delta"] = 0
    decision["mask_correction_target_delta"] = 0
    decision["mask_correction_unmet_additions"] = 0
    decision["mask_correction_applied"] = 0
    if float(strength) > 0.0:
        decision = apply_invitation_mask_correction(
            decision,
            args,
            env,
            feedback_by_index[int(confirmed_index)],
            strength,
            deadband_z,
            max_delta,
            rerank_mode,
        )
    preview_calls = len(seed_indices) + len(selected_indices)
    return decision, preview_calls


def run_episode(
    args,
    episode_seed,
    channel_rho,
    csi_delay_slots,
    strength,
    deadband_z,
    max_delta,
    rerank_mode,
):
    """运行回合流程，串联参数解析、实验执行、结果聚合和文件输出。"""
    env = make_env(args)
    env.reset(seed=episode_seed)
    env._last_seed = episode_seed
    temporal_history, temporal_states = build_temporal_channel_trace(
        env,
        args,
        episode_seed,
        channel_rho,
        prehistory_slots=csi_delay_slots,
    )
    episode_failed = 0
    episode_missed = 0
    episode_true_opportunities = 0
    episode_preview_calls = []
    episode_oracle_gaps = []
    episode_added = []
    episode_pruned = []
    episode_requested_deltas = []
    episode_target_deltas = []
    episode_unmet_additions = []
    episode_applied = []
    episode_slots = args.num_slots
    total_tx = 0

    for slot_idx in range(args.num_slots):
        execution_state = temporal_states[min(slot_idx, len(temporal_states) - 1)]
        stale_state = delayed_channel_state(
            temporal_states,
            slot_idx,
            csi_delay_slots,
            history_states=temporal_history,
        )

        apply_channel_state(env, execution_state)
        execution_oracle = execution_oracle_candidate(
            env,
            args,
            args.execution_error_std,
            slot_idx,
        )
        apply_channel_state(env, stale_state)
        decision, preview_calls = choose_coverage_b3_decision(
            args,
            env,
            episode_seed,
            slot_idx,
            execution_state,
            strength,
            deadband_z,
            max_delta,
            rerank_mode,
        )

        apply_channel_state(env, execution_state)
        true_selected = execution_candidate_for_decision(
            env,
            args,
            decision,
            args.execution_error_std,
            slot_idx,
        )
        info, done = limited.execute_limited_csi_slot(env, args, decision, true_selected)
        total_tx = int(info["total_tx"])
        episode_slots = int(info.get("slots_used", slot_idx + 1))
        episode_failed += int(info["failed_this_slot"])
        episode_missed += int(info["missed_opportunity_this_slot"])
        episode_true_opportunities += int(info["true_opportunity_this_slot"])
        episode_preview_calls.append(int(preview_calls))
        episode_oracle_gaps.append(
            max(0.0, float(execution_oracle["tx_this_slot"]) - float(info["tx_this_slot"]))
        )
        episode_added.append(float(decision.get("mask_correction_added", 0.0)))
        episode_pruned.append(float(decision.get("mask_correction_pruned", 0.0)))
        episode_requested_deltas.append(
            float(decision.get("mask_correction_requested_delta", 0.0))
        )
        episode_target_deltas.append(float(decision.get("mask_correction_target_delta", 0.0)))
        episode_unmet_additions.append(
            float(decision.get("mask_correction_unmet_additions", 0.0))
        )
        episode_applied.append(float(decision.get("mask_correction_applied", 0.0)))
        if done:
            break

    return {
        "success_nodes": float(total_tx),
        "perfect": float(total_tx >= args.num_nodes),
        "slots_used": float(episode_slots),
        "failed_nodes": float(episode_failed),
        "missed_opportunities": float(episode_missed),
        "true_opportunities": float(episode_true_opportunities),
        "preview_calls": float(np.mean(episode_preview_calls)) if episode_preview_calls else 0.0,
        "oracle_gap": float(np.mean(episode_oracle_gaps)) if episode_oracle_gaps else 0.0,
        "mask_added": float(np.mean(episode_added)) if episode_added else 0.0,
        "mask_pruned": float(np.mean(episode_pruned)) if episode_pruned else 0.0,
        "mask_requested_delta": (
            float(np.mean(episode_requested_deltas))
            if episode_requested_deltas
            else 0.0
        ),
        "mask_target_delta": float(np.mean(episode_target_deltas)) if episode_target_deltas else 0.0,
        "mask_unmet_additions": (
            float(np.mean(episode_unmet_additions))
            if episode_unmet_additions
            else 0.0
        ),
        "mask_applied": float(np.mean(episode_applied)) if episode_applied else 0.0,
    }


def mean(items, field):
    """处理均值相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    if not items:
        return 0.0
    return float(sum(float(item[field]) for item in items) / len(items))


def policy_label(strength, deadband_z, max_delta, rerank_mode):
    """处理策略、标签相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    if abs(float(strength)) < 1e-12:
        return "Coverage-Aware B=3 sm=4.1"
    mode_prefix = ""
    if rerank_mode != MASK_CORRECTION_MODE_GLOBAL_STALE_GAIN:
        mode_prefix = f"{rerank_mode} "
    if abs(float(deadband_z)) < 1e-12 and float(max_delta) < 0.0:
        return (
            f"{mode_prefix}Mask-Corrected Coverage-Aware B=3 "
            f"mc={float(strength):g}"
        )
    suffix = f"mc={float(strength):g}"
    if abs(float(deadband_z)) >= 1e-12:
        suffix = f"{suffix} z={float(deadband_z):g}"
    if float(max_delta) >= 0.0:
        suffix = f"{suffix} clip={float(max_delta):g}"
    return f"{mode_prefix}Noise-Aware Mask-Corrected Coverage-Aware B=3 {suffix}"


def correction_configs(args):
    """处理correction、configs相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    yielded_uncorrected = False
    for strength in args.mask_correction_strengths:
        if abs(float(strength)) < 1e-12:
            if not yielded_uncorrected:
                yielded_uncorrected = True
                yield float(strength), 0.0, -1.0, "none"
            continue
        for rerank_mode in args.mask_correction_rerank_modes:
            for deadband_z in args.mask_correction_noise_deadband_z_values:
                for max_delta in args.mask_correction_max_delta_values:
                    yield (
                        float(strength),
                        float(deadband_z),
                        float(max_delta),
                        rerank_mode,
                    )


def run_evaluation(args):
    """运行evaluation流程，串联参数解析、实验执行、结果聚合和文件输出。"""
    rows = []
    run_seeds = make_run_seeds(args)
    episode_seed_sets = [make_episode_seeds(args, run_seed) for run_seed in run_seeds]
    original_feedback_noise_std = float(args.confirmation_feedback_noise_std)
    try:
        for feedback_noise_std in args.confirmation_feedback_noise_std_values:
            args.confirmation_feedback_noise_std = float(feedback_noise_std)
            for channel_rho in args.channel_rho_values:
                for csi_delay_slots in args.csi_delay_slots:
                    for (
                        strength,
                        deadband_z,
                        max_delta,
                        rerank_mode,
                    ) in correction_configs(args):
                        episode_results = []
                        for episode_seeds in episode_seed_sets:
                            for episode_seed in episode_seeds:
                                episode_results.append(
                                    run_episode(
                                        args,
                                        episode_seed,
                                        channel_rho,
                                        csi_delay_slots,
                                        strength,
                                        deadband_z,
                                        max_delta,
                                        rerank_mode,
                                    )
                                )
                        rows.append(
                            {
                                "policy": policy_label(
                                    strength,
                                    deadband_z,
                                    max_delta,
                                    rerank_mode,
                                ),
                                "channel_rho": float(channel_rho),
                                "csi_delay_slots": int(csi_delay_slots),
                                "episodes": int(args.episodes * args.num_seeds),
                                "num_seeds": int(args.num_seeds),
                                "mask_correction_strength": float(strength),
                                "mask_correction_noise_deadband_z": float(deadband_z),
                                "mask_correction_max_delta": float(max_delta),
                                "mask_correction_rerank_mode": rerank_mode,
                                "confirmation_feedback_noise_std": float(feedback_noise_std),
                                "success_mean": mean(episode_results, "success_nodes"),
                                "perfect_rate": 100.0 * mean(episode_results, "perfect"),
                                "slots_mean": mean(episode_results, "slots_used"),
                                "failed_nodes_mean": mean(episode_results, "failed_nodes"),
                                "missed_opportunities_mean": mean(episode_results, "missed_opportunities"),
                                "true_opportunities_mean": mean(episode_results, "true_opportunities"),
                                "decision_preview_calls_per_slot_mean": mean(episode_results, "preview_calls"),
                                "oracle_tx_gap_mean": mean(episode_results, "oracle_gap"),
                                "mask_correction_added_mean": mean(episode_results, "mask_added"),
                                "mask_correction_pruned_mean": mean(episode_results, "mask_pruned"),
                                "mask_correction_requested_target_delta_mean": mean(
                                    episode_results,
                                    "mask_requested_delta",
                                ),
                                "mask_correction_target_delta_mean": mean(episode_results, "mask_target_delta"),
                                "mask_correction_unmet_additions_mean": mean(
                                    episode_results,
                                    "mask_unmet_additions",
                                ),
                                "mask_correction_applied_rate": mean(episode_results, "mask_applied"),
                            }
                        )
    finally:
        args.confirmation_feedback_noise_std = original_feedback_noise_std
    return rows


def aggregate_by_policy(rows):
    """聚合by、策略结果，把逐时隙、逐回合或逐场景数据压缩为可比较的摘要。"""
    groups = {}
    for row in rows:
        key = (
            row["policy"],
            float(row["confirmation_feedback_noise_std"]),
            float(row["mask_correction_strength"]),
            float(row["mask_correction_noise_deadband_z"]),
            float(row["mask_correction_max_delta"]),
            row["mask_correction_rerank_mode"],
        )
        groups.setdefault(key, []).append(row)
    summaries = []
    for (
        policy,
        feedback_noise_std,
        strength,
        deadband_z,
        max_delta,
        rerank_mode,
    ), group in groups.items():
        item = {
            "policy": policy,
            "confirmation_feedback_noise_std": float(feedback_noise_std),
            "mask_correction_strength": float(strength),
            "mask_correction_noise_deadband_z": float(deadband_z),
            "mask_correction_max_delta": float(max_delta),
            "mask_correction_rerank_mode": rerank_mode,
            "scenario_count": len(group),
        }
        for field in CSV_FIELDS:
            if field in {
                "policy",
                "channel_rho",
                "csi_delay_slots",
                "episodes",
                "num_seeds",
                "confirmation_feedback_noise_std",
                "mask_correction_strength",
                "mask_correction_noise_deadband_z",
                "mask_correction_max_delta",
                "mask_correction_rerank_mode",
            }:
                continue
            item[field] = mean(group, field)
        summaries.append(item)
    return sorted(
        summaries,
        key=lambda item: (
            float(item["confirmation_feedback_noise_std"]),
            float(item["mask_correction_strength"]),
            float(item["mask_correction_noise_deadband_z"]),
            float(item["mask_correction_max_delta"]),
            item["mask_correction_rerank_mode"],
        ),
    )


def write_csv(path, rows):
    """写出CSV结果，并统一字段顺序、目录创建和后续文档读取口径。"""
    ensure_parent_dir(path)
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def format_float(value, digits=3):
    """格式化浮点数显示文本，保证控制台、CSV 和 Markdown 中的数值表达一致。"""
    return f"{float(value):.{digits}f}"


def format_clip(value):
    """格式化clip显示文本，保证控制台、CSV 和 Markdown 中的数值表达一致。"""
    if float(value) < 0.0:
        return "none"
    return f"{float(value):g}"


def markdown_table(headers, rows):
    """渲染 Markdown 表格，把表头和结果行转换成文档可直接引用的表格文本。"""
    output = ["| " + " | ".join(headers) + " |"]
    output.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        output.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(output)


def find_summary_by_mode(summaries, mode):
    """处理find、摘要、by、mode相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    for item in summaries:
        if item.get("mask_correction_rerank_mode") == mode:
            return item
    return None


def write_markdown(path, csv_path, summaries):
    """写出markdown结果，并统一字段顺序、目录创建和后续文档读取口径。"""
    ensure_parent_dir(path)
    table_rows = []
    for item in summaries:
        table_rows.append(
            [
                item["policy"],
                item["mask_correction_rerank_mode"],
                format_float(item["confirmation_feedback_noise_std"]),
                format_float(item["mask_correction_noise_deadband_z"]),
                format_clip(item["mask_correction_max_delta"]),
                format_float(item["slots_mean"]),
                format_float(item["perfect_rate"], 2),
                format_float(item["failed_nodes_mean"]),
                format_float(item["missed_opportunities_mean"]),
                format_float(item["decision_preview_calls_per_slot_mean"], 2),
                format_float(item["oracle_tx_gap_mean"]),
                format_float(item["mask_correction_added_mean"]),
                format_float(item["mask_correction_pruned_mean"]),
                format_float(item["mask_correction_unmet_additions_mean"]),
                format_float(item["mask_correction_applied_rate"]),
            ]
        )
    noise_levels = sorted({float(item["confirmation_feedback_noise_std"]) for item in summaries})
    best_by_noise_rows = []
    for noise_level in noise_levels:
        noise_items = [
            item
            for item in summaries
            if abs(float(item["confirmation_feedback_noise_std"]) - noise_level) < 1e-12
        ]
        best_for_noise = min(
            noise_items,
            key=lambda item: (
                float(item["oracle_tx_gap_mean"]),
                float(item["slots_mean"]),
                float(item["failed_nodes_mean"]),
            ),
        )
        best_by_noise_rows.append(
            [
                format_float(noise_level),
                best_for_noise["policy"],
                format_float(best_for_noise["slots_mean"]),
                format_float(best_for_noise["failed_nodes_mean"]),
                format_float(best_for_noise["missed_opportunities_mean"]),
                format_float(best_for_noise["oracle_tx_gap_mean"]),
            ]
        )
    best = min(
        summaries,
        key=lambda item: (
            float(item["oracle_tx_gap_mean"]),
            float(item["slots_mean"]),
            float(item["failed_nodes_mean"]),
        ),
    )
    has_clipped_variant = any(float(item["mask_correction_max_delta"]) >= 0.0 for item in summaries)
    baseline = find_summary_by_mode(summaries, "none")
    global_stale_gain = find_summary_by_mode(summaries, MASK_CORRECTION_MODE_GLOBAL_STALE_GAIN)
    prune_only = find_summary_by_mode(summaries, "prune_only")
    if len(noise_levels) > 1 and has_clipped_variant:
        interpretation = (
            "The feedback-noise sweep should be read as a reliability boundary. "
            "Mask correction reduces missed opportunities by trusting the aggregate "
            "current feedback count, but larger count noise can turn that trust into "
            "extra failed invitations. The clipped variants test whether limiting "
            "per-slot target-count changes can recover high-noise robustness without "
            "changing the IRS candidate-generation stage."
        )
    elif len(noise_levels) > 1:
        interpretation = (
            "The feedback-noise sweep should be read as a reliability boundary. "
            "Mask correction reduces missed opportunities by trusting the aggregate "
            "current feedback count, but larger count noise can turn that trust into "
            "extra failed invitations. The follow-up should test clipped target-count "
            "correction before treating exact count matching as the deployment rule."
        )
    else:
        interpretation = (
            "The no-noise setting isolates the invitation-mask mismatch diagnosed "
            "for Coverage-Aware B=3 and tests whether aggregate current feedback "
            "count can repair that mismatch at the same preview cost."
        )
    if baseline is not None and global_stale_gain is not None and prune_only is not None:
        interpretation = (
            f"{interpretation}\n\n"
            "The rerank-mode ablation shows that target-count correction alone is "
            "not the mechanism behind the balanced trade-off. The `prune_only` "
            f"variant lowers failed invitations to `{format_float(prune_only['failed_nodes_mean'])}` "
            f"versus `{format_float(global_stale_gain['failed_nodes_mean'])}` for "
            "`global_stale_gain`, but it increases missed opportunities "
            f"to `{format_float(prune_only['missed_opportunities_mean'])}` and "
            f"gap to `{format_float(prune_only['oracle_tx_gap_mean'])}`. The "
            "`global_stale_gain` variant keeps the same preview budget while "
            f"achieving lower slots `{format_float(global_stale_gain['slots_mean'])}`, "
            f"lower missed opportunities `{format_float(global_stale_gain['missed_opportunities_mean'])}`, "
            f"and lower gap `{format_float(global_stale_gain['oracle_tx_gap_mean'])}` "
            "than `prune_only`. The method should therefore be described as "
            "aggregate target-count correction plus stale-gain replacement, not "
            "as a pure cardinality-only correction."
        )
    content = f"""# Invitation Mask Correction Analysis

Generated by `evaluate_invitation_mask_correction.py`.

Scenario CSV: `{csv_path}`

This method keeps the current `Coverage-Aware B=3 sm=4.1` IRS candidate generation and aggregate confirmation unchanged. After an IRS index is confirmed, it uses the aggregate current feedback count as a target cardinality and reranks remaining devices by stale gain under the confirmed IRS. It does not use per-device current CSI.

`global_stale_gain` is the original correction rule. `prune_only` is an ablation that can only remove nodes from the stale-valid invitation mask; it never adds nodes that stale CSI marked invalid. This separates target-count correction from stale-gain replacement outside the stale-valid set.

{markdown_table(
        [
            "Policy",
            "Mode",
            "Noise std",
            "Deadband z",
            "Clip",
            "Slots",
            "Perfect %",
            "Failed",
            "Missed",
            "Preview",
            "Gap",
            "Added",
            "Pruned",
            "Unmet adds",
            "Applied",
        ],
        table_rows,
    )}

Best by feedback-noise level:

{markdown_table(
        [
            "Noise std",
            "Best policy",
            "Slots",
            "Failed",
            "Missed",
            "Gap",
        ],
        best_by_noise_rows,
    )}

Best setting by gap: `{best["policy"]}` at feedback-noise std `{format_float(best["confirmation_feedback_noise_std"])}` with gap `{format_float(best["oracle_tx_gap_mean"])}`.

## Interpretation

{interpretation}
"""
    with open(path, "w", encoding="utf-8") as markdown_file:
        markdown_file.write(content)


def main():
    """脚本入口：串联参数解析、实验执行、结果聚合和文件输出。"""
    args = parse_args()
    normalize_args(args)
    validate_args(args)

    output_prefix = resolve_output_prefix(args)
    csv_path = f"{output_prefix}.csv"
    rows = run_evaluation(args)
    summaries = aggregate_by_policy(rows)
    write_csv(csv_path, rows)
    write_markdown(args.doc_output, csv_path, summaries)
    print(f"Wrote {csv_path}")
    print(f"Wrote {args.doc_output}")
    for item in summaries:
        print(
            f"{item['policy']} noise={item['confirmation_feedback_noise_std']:.3f}: "
            f"mode={item['mask_correction_rerank_mode']}, "
            f"z={item['mask_correction_noise_deadband_z']:.3f}, "
            f"clip={item['mask_correction_max_delta']:.3f}, "
            f"slots={item['slots_mean']:.3f}, "
            f"failed={item['failed_nodes_mean']:.3f}, "
            f"missed={item['missed_opportunities_mean']:.3f}, "
            f"gap={item['oracle_tx_gap_mean']:.3f}"
        )


if __name__ == "__main__":
    main()
