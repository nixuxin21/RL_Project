"""诊断覆盖感知 B=3 的残余 oracle 差距，把误差分解到候选池、选择、确认和邀请掩码。"""

import argparse
import csv
import json
import os

import numpy as np

import ms_aircomp.limited_csi as limited
from ms_aircomp.channel_models import (
    apply_channel_state,
    build_temporal_channel_trace,
    capture_channel_state,
    delayed_channel_state,
)
from ms_aircomp.execution_candidates import (
    execution_candidate_for_decision,
    execution_candidates,
    execution_oracle_candidate,
)
from ms_aircomp.confirmation import confirm_index_with_current_feedback
from ms_aircomp.experiment_utils import (
    ensure_parent_dir,
    format_float_for_suffix,
    make_episode_seeds,
    make_run_seeds,
)
from ms_aircomp.probe_sets import coverage_aware_sparse_indices
from test_env import MSAirCompEnv


DEFAULT_RESULTS_DIR = os.path.join("results", "execution_mismatch")
DEFAULT_DOC_OUTPUT = os.path.join("docs", "COVERAGE_B3_FAILURE_DIAGNOSIS.md")

TRACE_FIELDS = [
    "run_seed",
    "episode_index",
    "episode_seed",
    "channel_rho",
    "csi_delay_slots",
    "slot_idx",
    "remaining_before",
    "seed_indices",
    "selected_indices",
    "feedbacks",
    "preview_calls",
    "oracle_index",
    "seed_best_index",
    "selected_best_index",
    "confirmed_index",
    "oracle_tx",
    "seed_best_current_tx",
    "selected_best_current_tx",
    "confirmed_current_tx",
    "actual_success",
    "scheduled",
    "failed",
    "missed",
    "true_opportunities",
    "pool_gap",
    "selection_gap",
    "confirmation_gap",
    "invitation_gap",
    "total_gap",
    "primary_gap_type",
    "oracle_in_seed",
    "oracle_in_selected",
    "confirmed_is_oracle",
    "seed_best_in_selected",
    "selected_best_confirmed",
    "stale_confirmed_tx",
    "stale_confirmed_power",
    "coverage_marginal_fraction",
    "coverage_overlap_fraction",
    "episode_done",
    "total_tx_after_slot",
]

SUMMARY_FIELDS = [
    "label",
    "trace_slots",
    "slots_with_gap",
    "avg_total_gap",
    "avg_pool_gap",
    "avg_selection_gap",
    "avg_confirmation_gap",
    "avg_invitation_gap",
    "pool_gap_share",
    "selection_gap_share",
    "confirmation_gap_share",
    "invitation_gap_share",
    "primary_pool_rate",
    "primary_selection_rate",
    "primary_confirmation_rate",
    "primary_invitation_rate",
    "oracle_not_in_seed_rate",
    "oracle_in_seed_not_selected_rate",
    "oracle_selected_not_confirmed_rate",
    "avg_actual_success",
    "avg_scheduled",
    "avg_failed",
    "avg_missed",
    "avg_true_opportunities",
]


def parse_args():
    """解析命令行参数，集中声明实验规模、策略配置、输入输出路径和绘图开关。"""
    parser = argparse.ArgumentParser(
        description="Trace residual gap sources for Coverage-Aware B=3 sm=4.1."
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
    parser.add_argument("--confirmation-feedback-power-weight", type=float, default=0.05)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--doc-output", default=DEFAULT_DOC_OUTPUT)
    return parser.parse_args()


def parse_float_list(value):
    """解析浮点数、列表参数，通常把逗号分隔的命令行字符串转换成类型明确的 Python 列表。"""
    return [float(item) for item in str(value).split(",") if str(item).strip()]


def parse_int_list(value):
    """解析整数、列表参数，通常把逗号分隔的命令行字符串转换成类型明确的 Python 列表。"""
    return [int(item) for item in str(value).split(",") if str(item).strip()]


def json_compact(value):
    """处理json、compact相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    return json.dumps(value, separators=(",", ":"))


def resolve_output_prefix(args):
    """处理resolve、输出、前缀相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    if args.output_prefix:
        ensure_parent_dir(args.output_prefix)
        return args.output_prefix

    rho_label = "-".join(format_float_for_suffix(value) for value in args.channel_rho_values)
    delay_label = "-".join(str(value) for value in args.csi_delay_slots)
    prefix = (
        "coverage_b3_failure_diagnosis_"
        f"ep{args.episodes}_runs{args.num_seeds}_"
        f"rho{rho_label}_delay{delay_label}_"
        f"b{args.probe_budget}_"
        f"sm{format_float_for_suffix(args.sparse_topk_seed_multiplier)}_"
        f"tf{format_float_for_suffix(args.sparse_topk_fraction)}_"
        f"cw{format_float_for_suffix(args.coverage_sparse_weight)}_"
        f"cpw{format_float_for_suffix(args.coverage_sparse_power_weight)}"
    )
    return os.path.join(DEFAULT_RESULTS_DIR, prefix)


def make_env(args):
    """构建env所需的数据结构，供评估循环、训练流程或报告生成继续使用。"""
    return MSAirCompEnv(
        num_nodes=args.num_nodes,
        num_slots=args.num_slots,
        num_irs_elements=args.num_irs_elements,
        num_codebook_states=args.num_codebook_states,
    )


def best_current_candidate(current_by_index, indices):
    """处理best、当前、候选相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    return max(
        (current_by_index[int(index)] for index in indices),
        key=limited.candidate_key,
    )


def primary_gap_type(pool_gap, selection_gap, confirmation_gap, invitation_gap, total_gap):
    """处理primary、差距、type相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    if int(total_gap) <= 0:
        return "none"
    components = {
        "pool": int(pool_gap),
        "selection": int(selection_gap),
        "confirmation": int(confirmation_gap),
        "invitation": int(invitation_gap),
    }
    label, value = max(
        components.items(),
        key=lambda item: (item[1], item[0] == "invitation", item[0]),
    )
    return label if value > 0 else "none"


def trace_one_slot(
    args,
    env,
    episode_seed,
    channel_rho,
    csi_delay_slots,
    temporal_history,
    temporal_states,
    slot_idx,
):
    """处理轨迹、one、时隙相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    execution_state = temporal_states[min(slot_idx, len(temporal_states) - 1)]
    stale_state = delayed_channel_state(
        temporal_states,
        slot_idx,
        csi_delay_slots,
        history_states=temporal_history,
    )
    remaining_before = int(np.sum(~env.transmitted_flags))

    apply_channel_state(env, execution_state)
    oracle = execution_oracle_candidate(env, args, args.execution_error_std, slot_idx)
    current_candidates = execution_candidates(
        env,
        args,
        indices=range(args.num_codebook_states),
        execution_error_std=args.execution_error_std,
        slot_idx=slot_idx,
    )
    current_by_index = {int(candidate["irs_index"]): candidate for candidate in current_candidates}

    apply_channel_state(env, stale_state)
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
    selected_indices, anchor_count, marginal_mean, overlap_mean = coverage_aware_sparse_indices(
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

    seed_best = best_current_candidate(current_by_index, seed_indices)
    selected_best = best_current_candidate(current_by_index, selected_indices)
    confirmed_current = current_by_index[int(confirmed_index)]

    apply_channel_state(env, execution_state)
    true_selected = execution_candidate_for_decision(
        env,
        args,
        decision,
        args.execution_error_std,
        slot_idx,
    )
    info, done = limited.execute_limited_csi_slot(env, args, decision, true_selected)

    oracle_tx = int(oracle["tx_this_slot"])
    seed_best_tx = int(seed_best["tx_this_slot"])
    selected_best_tx = int(selected_best["tx_this_slot"])
    confirmed_current_tx = int(confirmed_current["tx_this_slot"])
    actual_success = int(info["tx_this_slot"])
    pool_gap = max(0, oracle_tx - seed_best_tx)
    selection_gap = max(0, seed_best_tx - selected_best_tx)
    confirmation_gap = max(0, selected_best_tx - confirmed_current_tx)
    invitation_gap = max(0, confirmed_current_tx - actual_success)
    total_gap = max(0, oracle_tx - actual_success)

    feedback_rows = [
        {
            "irs": int(feedback["irs_index"]),
            "tx": round(float(feedback["observed_tx_fraction"]) * float(args.num_nodes), 6),
            "score": round(float(feedback["observed_score"]), 9),
            "power": round(float(feedback["observed_power"]), 9),
        }
        for feedback in feedbacks
    ]
    oracle_index = int(oracle["irs_index"])
    row = {
        "run_seed": "",
        "episode_index": "",
        "episode_seed": int(episode_seed),
        "channel_rho": float(channel_rho),
        "csi_delay_slots": int(csi_delay_slots),
        "slot_idx": int(slot_idx),
        "remaining_before": remaining_before,
        "seed_indices": json_compact([int(index) for index in seed_indices]),
        "selected_indices": json_compact([int(index) for index in selected_indices]),
        "feedbacks": json_compact(feedback_rows),
        "preview_calls": int(len(seed_indices) + len(selected_indices)),
        "oracle_index": oracle_index,
        "seed_best_index": int(seed_best["irs_index"]),
        "selected_best_index": int(selected_best["irs_index"]),
        "confirmed_index": int(confirmed_index),
        "oracle_tx": oracle_tx,
        "seed_best_current_tx": seed_best_tx,
        "selected_best_current_tx": selected_best_tx,
        "confirmed_current_tx": confirmed_current_tx,
        "actual_success": actual_success,
        "scheduled": int(info["scheduled_this_slot"]),
        "failed": int(info["failed_this_slot"]),
        "missed": int(info["missed_opportunity_this_slot"]),
        "true_opportunities": int(info["true_opportunity_this_slot"]),
        "pool_gap": pool_gap,
        "selection_gap": selection_gap,
        "confirmation_gap": confirmation_gap,
        "invitation_gap": invitation_gap,
        "total_gap": total_gap,
        "primary_gap_type": primary_gap_type(
            pool_gap,
            selection_gap,
            confirmation_gap,
            invitation_gap,
            total_gap,
        ),
        "oracle_in_seed": int(oracle_index in {int(index) for index in seed_indices}),
        "oracle_in_selected": int(oracle_index in {int(index) for index in selected_indices}),
        "confirmed_is_oracle": int(int(confirmed_index) == oracle_index),
        "seed_best_in_selected": int(
            int(seed_best["irs_index"]) in {int(index) for index in selected_indices}
        ),
        "selected_best_confirmed": int(int(selected_best["irs_index"]) == int(confirmed_index)),
        "stale_confirmed_tx": int(decision["tx_this_slot"]),
        "stale_confirmed_power": float(decision["power_avg"]),
        "coverage_marginal_fraction": float(marginal_mean),
        "coverage_overlap_fraction": float(overlap_mean),
        "episode_done": int(done),
        "total_tx_after_slot": int(info["total_tx"]),
    }
    return row, done


def run_trace(args):
    """运行轨迹流程，串联参数解析、实验执行、结果聚合和文件输出。"""
    rows = []
    run_seeds = make_run_seeds(args)
    episode_seed_sets = [make_episode_seeds(args, run_seed) for run_seed in run_seeds]
    for channel_rho in args.channel_rho_values:
        for csi_delay_slots in args.csi_delay_slots:
            for run_seed, episode_seeds in zip(run_seeds, episode_seed_sets):
                for episode_index, episode_seed in enumerate(episode_seeds):
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
                    for slot_idx in range(args.num_slots):
                        row, done = trace_one_slot(
                            args,
                            env,
                            episode_seed,
                            channel_rho,
                            csi_delay_slots,
                            temporal_history,
                            temporal_states,
                            slot_idx,
                        )
                        row["run_seed"] = "" if run_seed is None else int(run_seed)
                        row["episode_index"] = int(episode_index)
                        rows.append(row)
                        if done:
                            break
    return rows


def mean(rows, field):
    """处理均值相关的局部逻辑，封装重复步骤并让调用处保持清晰。"""
    if not rows:
        return 0.0
    return float(sum(float(row[field]) for row in rows) / len(rows))


def summarize_group(label, rows):
    """聚合group结果，把逐时隙、逐回合或逐场景数据压缩为可比较的摘要。"""
    total_gap_sum = sum(float(row["total_gap"]) for row in rows)
    slot_count = len(rows)
    slots_with_gap = sum(int(float(row["total_gap"]) > 0.0) for row in rows)

    def component_share(field):
        if total_gap_sum <= 0.0:
            return 0.0
        return float(sum(float(row[field]) for row in rows) / total_gap_sum)

    def primary_rate(label_name):
        if slot_count <= 0:
            return 0.0
        return float(
            sum(1 for row in rows if row["primary_gap_type"] == label_name) / slot_count
        )

    oracle_in_seed_not_selected = [
        row
        for row in rows
        if int(row["oracle_in_seed"]) == 1 and int(row["oracle_in_selected"]) == 0
    ]
    oracle_selected_not_confirmed = [
        row
        for row in rows
        if int(row["oracle_in_selected"]) == 1 and int(row["confirmed_is_oracle"]) == 0
    ]
    return {
        "label": label,
        "trace_slots": slot_count,
        "slots_with_gap": slots_with_gap,
        "avg_total_gap": mean(rows, "total_gap"),
        "avg_pool_gap": mean(rows, "pool_gap"),
        "avg_selection_gap": mean(rows, "selection_gap"),
        "avg_confirmation_gap": mean(rows, "confirmation_gap"),
        "avg_invitation_gap": mean(rows, "invitation_gap"),
        "pool_gap_share": component_share("pool_gap"),
        "selection_gap_share": component_share("selection_gap"),
        "confirmation_gap_share": component_share("confirmation_gap"),
        "invitation_gap_share": component_share("invitation_gap"),
        "primary_pool_rate": primary_rate("pool"),
        "primary_selection_rate": primary_rate("selection"),
        "primary_confirmation_rate": primary_rate("confirmation"),
        "primary_invitation_rate": primary_rate("invitation"),
        "oracle_not_in_seed_rate": (
            1.0 - mean(rows, "oracle_in_seed") if rows else 0.0
        ),
        "oracle_in_seed_not_selected_rate": (
            float(len(oracle_in_seed_not_selected) / slot_count) if slot_count else 0.0
        ),
        "oracle_selected_not_confirmed_rate": (
            float(len(oracle_selected_not_confirmed) / slot_count) if slot_count else 0.0
        ),
        "avg_actual_success": mean(rows, "actual_success"),
        "avg_scheduled": mean(rows, "scheduled"),
        "avg_failed": mean(rows, "failed"),
        "avg_missed": mean(rows, "missed"),
        "avg_true_opportunities": mean(rows, "true_opportunities"),
    }


def summarize_rows(rows):
    """聚合结果行结果，把逐时隙、逐回合或逐场景数据压缩为可比较的摘要。"""
    summaries = [summarize_group("overall", rows)]
    groups = {}
    for row in rows:
        label = f"rho={float(row['channel_rho']):g}, delay={int(row['csi_delay_slots'])}"
        groups.setdefault(label, []).append(row)
    for label in sorted(groups):
        summaries.append(summarize_group(label, groups[label]))
    return summaries


def write_csv(path, rows, fields):
    """按固定字段顺序写出 CSV，保证后续报告和测试读取稳定。"""
    ensure_parent_dir(path)
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def format_float(value, digits=3):
    """格式化浮点数显示文本，保证控制台、CSV 和 Markdown 中的数值表达一致。"""
    return f"{float(value):.{digits}f}"


def markdown_table(headers, rows):
    """渲染 Markdown 表格，把表头和结果行转换成文档可直接引用的表格文本。"""
    output = ["| " + " | ".join(headers) + " |"]
    output.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        output.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(output)


def write_markdown(path, args, trace_path, summary_path, summaries):
    """写出markdown结果，并统一字段顺序、目录创建和后续文档读取口径。"""
    ensure_parent_dir(path)
    overall = summaries[0]
    component_shares = {
        "pool": overall["pool_gap_share"],
        "selection": overall["selection_gap_share"],
        "confirmation": overall["confirmation_gap_share"],
        "invitation": overall["invitation_gap_share"],
    }
    dominant = max(component_shares, key=component_shares.get)

    overall_rows = [
        [
            "overall",
            overall["trace_slots"],
            overall["slots_with_gap"],
            format_float(overall["avg_total_gap"]),
            format_float(overall["avg_pool_gap"]),
            format_float(overall["avg_selection_gap"]),
            format_float(overall["avg_confirmation_gap"]),
            format_float(overall["avg_invitation_gap"]),
            format_float(overall["pool_gap_share"]),
            format_float(overall["selection_gap_share"]),
            format_float(overall["confirmation_gap_share"]),
            format_float(overall["invitation_gap_share"]),
        ]
    ]
    scenario_rows = []
    for item in summaries[1:]:
        scenario_rows.append(
            [
                item["label"],
                item["trace_slots"],
                format_float(item["avg_total_gap"]),
                format_float(item["pool_gap_share"]),
                format_float(item["selection_gap_share"]),
                format_float(item["confirmation_gap_share"]),
                format_float(item["invitation_gap_share"]),
                format_float(item["avg_failed"]),
                format_float(item["avg_missed"]),
            ]
        )

    exact_rows = [
        [
            "overall",
            format_float(overall["oracle_not_in_seed_rate"]),
            format_float(overall["oracle_in_seed_not_selected_rate"]),
            format_float(overall["oracle_selected_not_confirmed_rate"]),
            format_float(overall["primary_pool_rate"]),
            format_float(overall["primary_selection_rate"]),
            format_float(overall["primary_confirmation_rate"]),
            format_float(overall["primary_invitation_rate"]),
        ]
    ]

    content = f"""# Coverage B3 Failure Diagnosis

Generated by `diagnose_coverage_b3_failures.py`.

Setting: `Coverage-Aware Sparse-TopK B={args.probe_budget} sm={args.sparse_topk_seed_multiplier:g} tf={args.sparse_topk_fraction:g} cw={args.coverage_sparse_weight:g} cpw={args.coverage_sparse_power_weight:g}` under temporal AR(1) stale CSI.

Trace CSV: `{trace_path}`

Summary CSV: `{summary_path}`

## Gap Decomposition

The diagnostic decomposes each slot's hidden oracle gap into four terms:

- `pool`: best current IRS quality is not present in the sparse stale pool.
- `selection`: the pool has a better current candidate, but final B-candidate selection drops it.
- `confirmation`: final B contains a better current candidate, but aggregate feedback confirms another one.
- `invitation`: the confirmed IRS index has current potential, but the stale invitation mask fails or misses devices.

{markdown_table(
        [
            "Scope",
            "Trace slots",
            "Gap slots",
            "Avg gap",
            "Pool",
            "Selection",
            "Confirmation",
            "Invitation",
            "Pool share",
            "Selection share",
            "Confirmation share",
            "Invitation share",
        ],
        overall_rows,
    )}

Dominant residual component: `{dominant}`.

## Per-Scenario Breakdown

{markdown_table(
        [
            "Scenario",
            "Trace slots",
            "Avg gap",
            "Pool share",
            "Selection share",
            "Confirmation share",
            "Invitation share",
            "Failed",
            "Missed",
        ],
        scenario_rows,
    )}

## Exact Oracle-Index Diagnostics

These exact-index rates are stricter than the gap decomposition because multiple IRS indices can have the same current tx count.

{markdown_table(
        [
            "Scope",
            "Oracle not in seed",
            "Oracle in seed not selected",
            "Oracle selected not confirmed",
            "Primary pool",
            "Primary selection",
            "Primary confirmation",
            "Primary invitation",
        ],
        exact_rows,
    )}

## Interpretation

The next algorithmic step should target the dominant component above rather than adding another broad heuristic. If `pool` dominates, improve stale seed generation; if `selection` dominates, change the final B-candidate subset objective; if `confirmation` dominates, revise aggregate feedback scoring; if `invitation` dominates, address stale invitation masks for the confirmed IRS index.
"""
    with open(path, "w", encoding="utf-8") as markdown_file:
        markdown_file.write(content)


def main():
    """脚本入口：串联参数解析、实验执行、结果聚合和文件输出。"""
    args = parse_args()
    args.channel_rho_values = parse_float_list(args.channel_rho_values)
    args.csi_delay_slots = parse_int_list(args.csi_delay_slots)
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.num_seeds <= 0:
        raise ValueError("--num-seeds must be positive")
    if args.probe_budget <= 0:
        raise ValueError("--probe-budget must be positive")

    output_prefix = resolve_output_prefix(args)
    trace_path = f"{output_prefix}_trace.csv"
    summary_path = f"{output_prefix}_summary.csv"

    rows = run_trace(args)
    summaries = summarize_rows(rows)
    write_csv(trace_path, rows, TRACE_FIELDS)
    write_csv(summary_path, summaries, SUMMARY_FIELDS)
    write_markdown(args.doc_output, args, trace_path, summary_path, summaries)

    overall = summaries[0]
    print(f"Wrote {trace_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {args.doc_output}")
    print(
        "Overall avg gap="
        f"{overall['avg_total_gap']:.3f}, shares: "
        f"pool={overall['pool_gap_share']:.3f}, "
        f"selection={overall['selection_gap_share']:.3f}, "
        f"confirmation={overall['confirmation_gap_share']:.3f}, "
        f"invitation={overall['invitation_gap_share']:.3f}"
    )


if __name__ == "__main__":
    main()
